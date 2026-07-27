// BLACKSITE — Renderer + lighting rig.
// Owner: the Renderer agent. This file owns:
//   * the live THREE.WebGLRenderer (main.js drives setAnimationLoop through `this.renderer`)
//   * colour management / tone mapping / exposure
//   * the shadow-map configuration and the per-frame fitted, texel-snapped sun shadow cascade
//   * the whole lighting rig (sun, hemisphere, cool fill, warm ground bounce)
//   * adaptive resolution scaling
//
// Art direction it implements: late-afternoon golden hour (~16:20), warm low key from the west,
// cool skylight fill in shadow, warm sand bounce from below, filmic AgX rolloff.
//
// Public API consumed by collaborators:
//   game.renderer.renderer            -> THREE.WebGLRenderer (live)
//   game.renderer.sun                 -> THREE.DirectionalLight  (Sky drives its direction)
//   game.renderer.sunTarget           -> THREE.Object3D          (already parented to the scene)
//   game.renderer.hemi / .fill / .bounce
//   game.renderer.sunDir              -> THREE.Vector3, unit, points FROM the world TOWARD the sun
//   game.renderer.setSunDirection(x, y, z) | (Vector3)
//   game.renderer.setSunAngles(elevationDeg, azimuthDeg)
//   game.renderer.setSunColor(hexOrColor) / setSunIntensity(f)
//   game.renderer.setAmbientIntensity(scale)   // Sky can dial the rig back once an IBL exists
//   game.renderer.setExposure(f) / setToneMapping('agx'|'aces'|'neutral'|'none'|...)
//   game.renderer.setToneMappingEnabled(bool)  // PostFX calls this if it grades in its own pass
//   game.renderer.setQuality('low'|'high'|'ultra')
//   game.renderer.setShadowsEnabled(bool) / setShadowDistance(metres)
//   game.renderer.setAdaptive(bool)
//   game.renderer.getStats() -> {fps, frameMs, drawCalls, triangles, programs, pixelRatio, renderScale}
//   game.registry.lights               <- every light in the rig is registered here

import * as THREE from 'three';

const DEG = Math.PI / 180;

// Per-quality overrides. `0` means "inherit the Config value".
const QUALITY_PRESETS = {
  low: {
    shadowMapSize: 1024,
    shadowFit: 30,
    pixelRatioCap: 1.0,
    shadowType: 'pcf',
    ambientScale: 1.0,
    casterExtent: 60,
  },
  high: {
    shadowMapSize: 2048,
    shadowFit: 42,
    pixelRatioCap: 1.5,
    shadowType: 'pcfsoft',
    ambientScale: 1.0,
    casterExtent: 80,
  },
  ultra: {
    shadowMapSize: 0,          // -> Config.render.shadowMapSize (4096)
    shadowFit: 52,
    pixelRatioCap: 0,          // -> Config.render.pixelRatioCap
    shadowType: 'pcfsoft',
    ambientScale: 1.0,
    casterExtent: 95,
  },
};

// Resolution ladder used by the adaptive scaler. Index 0 is native.
const RENDER_SCALES = [1.0, 0.88, 0.78, 0.68, 0.58];

// Art-direction gain applied on top of Config.sky.sunIntensity.
//
// Config declares 3.2, which is a sane scene-linear key for a sun overhead. At the
// configured time of day the sun sits at ~18 degrees, so every horizontal surface in the
// compound is lit at NdotL ~= 0.31 — the direct term collapses to a third of its nominal
// value while the sky IBL, which integrates over the whole hemisphere, does not. Measured
// on the boot scene the un-gained rig put the ambient ABOVE the key on the courtyard
// floor: sunlit and shadowed sand differed by under 20%, there was no light-to-dark
// composition and the "warm raking key" of the art direction simply did not exist on
// screen. This restores the ratio a low-sun exterior needs (roughly 3:1 lit vs shade on
// the ground, 6:1 on a sun-facing wall). Config remains the single tunable; this is the
// documented multiplier the renderer applies to it.
const KEY_GAIN = 2.25;

export class Renderer {

  constructor(game) {
    this.g = game;
    const cfg = game.config;

    // ---------------------------------------------------------------- renderer
    const canvas = document.createElement('canvas');
    canvas.style.outline = 'none';
    canvas.style.touchAction = 'none';

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: false,
      depth: true,
      stencil: false,                       // PostFX does not need a stencil buffer
      antialias: false,                     // PostFX owns AA (TAA + SMAA)
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
      logarithmicDepthBuffer: false,        // kills early-Z and hurts precision on the viewmodel
      failIfMajorPerformanceCaveat: false,  // headless capture runs on a software rasteriser
    });

    this.renderer = renderer;

    // Colour management: linear working space, sRGB display transform on present.
    THREE.ColorManagement.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Tone mapping — AgX in r180, ACES as the fallback on older builds.
    this._toneMapEnabled = true;
    this._toneMapName = cfg.render.toneMapping || 'agx';
    renderer.toneMapping = this._resolveToneMapping(this._toneMapName);
    renderer.toneMappingExposure = cfg.render.exposure;

    // Shadows.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.autoUpdate = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this._shadowTypeName = 'pcfsoft';

    renderer.sortObjects = true;
    renderer.localClippingEnabled = false;
    // Manual reset: PostFX renders the scene through several passes per frame, and
    // auto-reset would leave info describing only the last fullscreen quad. We reset once
    // per frame in update() so the numbers describe the whole frame, shadow pass included.
    renderer.info.autoReset = false;
    renderer.debug.checkShaderErrors = true;

    // Hardware anisotropy ceiling — clamp the shared Config value so Materials (built after us)
    // never asks for more than the GPU can give.
    this.maxAnisotropy = renderer.capabilities.getMaxAnisotropy?.() ?? 1;
    cfg.render.maxAnisotropy = Math.max(1, Math.min(cfg.render.maxAnisotropy, this.maxAnisotropy));

    // Resolution.
    this._baseDpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1;
    this._pixelRatioCap = cfg.render.pixelRatioCap;
    this._scaleIndex = 0;
    this.renderScale = RENDER_SCALES[0];
    this._viewW = Math.max(1, innerWidth);
    this._viewH = Math.max(1, innerHeight);
    this._applyPixelRatio(false);
    renderer.setSize(this._viewW, this._viewH, false); // CSS already stretches the canvas to 100%

    (document.getElementById('app') || document.body).appendChild(canvas);

    // ------------------------------------------------------- context loss guard
    this._onCtxLost = (e) => { e.preventDefault(); this._contextLost = true; console.warn('[Renderer] WebGL context lost — waiting for restore'); };
    this._onCtxRestored = () => { this._contextLost = false; this._invalidateShadow(); console.warn('[Renderer] WebGL context restored'); };
    this._contextLost = false;
    canvas.addEventListener('webglcontextlost', this._onCtxLost, false);
    canvas.addEventListener('webglcontextrestored', this._onCtxRestored, false);

    // -------------------------------------------------------------- scratch math
    // Everything update() touches is preallocated. Zero garbage per frame.
    this._vCenter = new THREE.Vector3();
    this._vSnap = new THREE.Vector3();
    this._vTmp = new THREE.Vector3();
    this._vUp = new THREE.Vector3(0, 1, 0);
    this._vNegDir = new THREE.Vector3();
    this._vZero = new THREE.Vector3();
    this._mLight = new THREE.Matrix4();
    this._mLightInv = new THREE.Matrix4();

    // Cached sphere fit (only recomputed when fov / aspect / fit distance change).
    this._fitFov = -1;
    this._fitAspect = -1;
    this._fitNear = -1;
    this._fitDist = -1;
    this._sphereZ = 0;      // distance in front of the camera to the sphere centre
    this._sphereR = 1;      // sphere radius, quantised for stability

    // ------------------------------------------------------------- lighting rig
    this.sunDir = new THREE.Vector3();
    this.sunElevation = 15.4;   // degrees above the horizon; re-derived from timeOfDay below
    this.sunAzimuth = 238;      // degrees, 0 = +Z, 90 = +X; 238 = west-south-west
    this._biasSinE = -1;
    this.ambientAutoTrim = true;  // scale the analytic fills back if Sky provides an IBL
    this.ambientScale = 1.0;

    this._buildLights(cfg);

    // --------------------------------------------------------------- quality
    this.shadowBiasScale = 1.0;
    this.quality = 'ultra';
    this.shadowsEnabled = true;
    this._shadowFit = QUALITY_PRESETS.ultra.shadowFit;
    this._casterExtent = QUALITY_PRESETS.ultra.casterExtent;
    this._shadowMapSize = cfg.render.shadowMapSize;
    this.setQuality(cfg.quality || 'ultra');

    // ---------------------------------------------------- adaptive resolution
    // Adaptive resolution targets real GPUs. The headless screenshot harness runs on a
    // software rasteriser at single-digit fps, where scaling down would only produce blurry
    // capture frames and tell us nothing — so detect that case and pin the resolution.
    this.softwareRasterizer = this._detectSoftwareRasterizer();
    this.adaptiveEnabled = !this.softwareRasterizer;
    this.frameBudgetMs = 1000 / 60 * 1.28;   // ~21.3 ms — drop below this and we scale down
    this.frameGoodMs = 1000 / 60 * 0.82;     // ~13.7 ms — sustained headroom lets us scale back up
    this._overFrames = 0;
    this._underFrames = 0;
    this._warmupFrames = 0;
    this._lastNow = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this.frameMs = 16.7;
    this.frameMsSmooth = 16.7;
    this.fps = 60;
    this.drawCalls = 0;
    this.triangles = 0;
    this.programs = 0;

    // Applied a few frames in, and only if Sky never booted — Sky owns the atmosphere and
    // deliberately leaves scene.background null in favour of a dome mesh, so we must not
    // pre-empt it. This only exists so a missing Sky is a warm haze, never a black void.
    this._atmosphereChecked = false;
  }

  async init() {
    // Install the render-time shadow fit and warm the shadow pipeline so the first frames
    // after boot do not hitch on a shadow-map allocation + program compile.
    this._hookAttempts = 0;
    this._fitFrame = -1;
    this._lastFitFrame = undefined;
    this._ensureSceneHook();
    this._fitSunShadow(true);
  }

  // ==========================================================================
  //  LIGHTING RIG
  // ==========================================================================

  _buildLights(cfg) {
    const scene = this.g.scene;
    const lights = this.g.registry?.lights;

    // --- KEY: the sun. Low, warm, physically-plausible for ~15 deg elevation through
    //     6.2 turbidity of dust — the disc reads amber, not white.
    const sun = new THREE.DirectionalLight(0xffd9a0, cfg.sky.sunIntensity * KEY_GAIN);
    sun.name = 'sun';
    sun.castShadow = true;
    sun.matrixAutoUpdate = true;
    sun.shadow.mapSize.set(cfg.render.shadowMapSize, cfg.render.shadowMapSize);
    sun.shadow.autoUpdate = true;
    sun.shadow.intensity = 1.0;
    sun.shadow.bias = -0.00006;
    sun.shadow.normalBias = 0.045;
    sun.shadow.radius = 1.0;          // ignored by PCFSoft, used by plain PCF at 'low'
    sun.shadow.blurSamples = 8;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 260;
    sun.shadow.camera.left = -60;
    sun.shadow.camera.right = 60;
    sun.shadow.camera.top = 60;
    sun.shadow.camera.bottom = -60;
    sun.shadow.camera.up.set(0, 1, 0);
    sun.shadow.camera.updateProjectionMatrix();

    const sunTarget = new THREE.Object3D();
    sunTarget.name = 'sunTarget';
    sun.target = sunTarget;
    scene.add(sunTarget);
    scene.add(sun);

    this.sun = sun;
    this.sunTarget = sunTarget;

    // --- FILL: cool skylight + ground bounce. Hemisphere gives the vertical gradient
    //     that makes shadowed concrete read blue-grey on top and warm sand-tinted below.
    const hemi = new THREE.HemisphereLight(0x9cc2ef, 0x6f5334, 0.44);
    hemi.name = 'skyFill';
    hemi.position.set(0, 60, 0);
    scene.add(hemi);
    this.hemi = hemi;
    this._hemiBase = 0.44;

    // --- RIM/FILL: a cool directional from the opposite side of the sun. This is the
    //     single most important cheat for CoD-looking shadow interiors — it keeps the
    //     dark side of every object readable and tinted, never crushed to black.
    const fill = new THREE.DirectionalLight(0x7fa6d9, 0.34);
    fill.name = 'coolFill';
    fill.castShadow = false;
    const fillTarget = new THREE.Object3D();
    scene.add(fillTarget);
    fill.target = fillTarget;
    scene.add(fill);
    this.fill = fill;
    this.fillTarget = fillTarget;
    this._fillBase = 0.34;

    // --- BOUNCE: warm light thrown back up off hot sand/asphalt. Subtle, but it is what
    //     separates "3D render" from "photograph" on undersides, ledges and weapon bodies.
    const bounce = new THREE.DirectionalLight(0xd8a066, 0.20);
    bounce.name = 'groundBounce';
    bounce.castShadow = false;
    const bounceTarget = new THREE.Object3D();
    scene.add(bounceTarget);
    bounce.target = bounceTarget;
    bounce.position.set(6, -30, -10);
    scene.add(bounce);
    this.bounce = bounce;
    this.bounceTarget = bounceTarget;
    this._bounceBase = 0.20;

    if (Array.isArray(lights)) lights.push(sun, hemi, fill, bounce);

    // Seed the sun direction from the configured time of day so the rig is correct even if
    // Sky never boots. Done last: it positions the fill lights, which must already exist.
    this._sunDirFromTimeOfDay(cfg.sky.timeOfDay);
    this.setSunAngles(this.sunElevation, this.sunAzimuth);
    this.sun.position.copy(this.sunDir).multiplyScalar(120);
    this.sunTarget.position.set(0, 0, 0);
  }

  // Rough solar position for a mid-latitude late afternoon. Sunrise 06:12, sunset 18:36.
  // The 1.35 exponent flattens the curve toward the ends of the day so 16:24 lands at
  // ~15 degrees of elevation — the low raking key the art direction asks for — instead
  // of the ~22 degrees a naive linear ramp would give. Only used to seed the rig; Sky
  // owns the truth once it boots.
  _sunDirFromTimeOfDay(hours) {
    const h = typeof hours === 'number' ? hours : 16.4;
    const sunrise = 6.2, sunset = 18.6;
    const noon = (sunrise + sunset) * 0.5;
    const halfDay = (sunset - sunrise) * 0.5;
    const x = THREE.MathUtils.clamp(1 - Math.abs(h - noon) / halfDay, 0, 1);
    this.sunElevation = Math.max(1.2, Math.pow(x, 1.35) * 62);
    this.sunAzimuth = 90 + THREE.MathUtils.clamp((h - sunrise) / (sunset - sunrise), 0, 1) * 180;
  }

  /**
   * @param {number} elevationDeg  degrees above the horizon
   * @param {number} azimuthDeg    degrees; 0 = +Z, 90 = +X, 180 = -Z, 270 = -X
   */
  setSunAngles(elevationDeg, azimuthDeg) {
    this.sunElevation = elevationDeg;
    this.sunAzimuth = azimuthDeg;
    const e = elevationDeg * DEG, a = azimuthDeg * DEG;
    const c = Math.cos(e);
    this.setSunDirection(Math.sin(a) * c, Math.sin(e), Math.cos(a) * c);
    return this;
  }

  /** Direction points FROM the world TOWARD the sun. Accepts (Vector3) or (x, y, z). */
  setSunDirection(x, y, z) {
    if (x && typeof x === 'object') { this.sunDir.set(x.x, x.y, x.z); }
    else { this.sunDir.set(x, y, z); }
    if (this.sunDir.lengthSq() < 1e-8) this.sunDir.set(-0.62, 0.28, -0.73);
    this.sunDir.normalize();
    this._syncFillLights();
    return this;
  }

  /** Sky retints the key as the sun drops. The fill deliberately does NOT follow — it
   *  represents skylight, which stays cool while the direct beam reddens; that widening
   *  split between warm key and cool fill is what sells golden hour. */
  setSunColor(c) { this.sun.color.set(c); return this; }

  setSunIntensity(v) { this.sun.intensity = v; return this; }

  /** Sky calls this once it has an IBL up, so the analytic ambient does not double-count. */
  setAmbientIntensity(scale) {
    this.ambientScale = Math.max(0, scale);
    this.hemi.intensity = this._hemiBase * this.ambientScale;
    this.fill.intensity = this._fillBase * this.ambientScale;
    this.bounce.intensity = this._bounceBase * this.ambientScale;
    return this;
  }

  // Keep the non-shadowing lights anchored opposite / below the key.
  _syncFillLights() {
    if (!this.fill || !this.bounce) return;   // called during rig construction
    const d = this.sunDir;
    // Cool fill: mirrored horizontally, lifted so it never rakes from directly behind.
    this.fill.position.set(-d.x * 70, Math.max(18, 40 - d.y * 22), -d.z * 70);
    this.fillTarget.position.set(0, 0, 0);
    // Warm bounce: from below, offset slightly toward the sun so the bounce direction
    // reads as light kicking off the ground in front of the key.
    this.bounce.position.set(d.x * 18, -34, d.z * 18);
    this.bounceTarget.position.set(0, 0, 0);
  }

  // ==========================================================================
  //  SHADOW CASCADE FIT
  // ==========================================================================
  //
  //  One shadow map, refit every frame to the visible frustum instead of a static
  //  200 m box. The near cascade (Config.render.shadowDistance / cascades ~= 35 m,
  //  widened to ~52 m of camera depth at ultra) is enclosed by the minimal bounding
  //  sphere of that frustum slice. The sphere is rotation-invariant, so panning the
  //  camera never changes the shadow texel density, and its centre is snapped to whole
  //  shadow-map texels in light space, so the shadow edges never crawl as the player
  //  walks. That snap is the difference between AAA and hobby.
  //
  //  Because the enclosing sphere of a 68-degree-vertical frustum slice bulges well past
  //  the slice itself, the effective shadowed depth along the view axis is roughly
  //  2.4x the fit distance (~125 m at ultra) — the whole compound stays shadowed.

  _recomputeFitSphere(cam) {
    const near = Math.max(cam.near, 0.05);
    const far = this._shadowFit;
    const tanY = Math.tan(cam.fov * 0.5 * DEG);
    const tanX = tanY * cam.aspect;
    const t2 = tanX * tanX + tanY * tanY;

    // Optimal sphere centre distance for a frustum slice (equal radius to near+far corners).
    let zc = (near + far) * (t2 + 1) * 0.5;
    let r;
    if (zc >= far) {
      // Degenerate case for wide FOV: the far rectangle circumscribes everything.
      zc = far;
      r = far * Math.sqrt(t2);
    } else {
      const dx = far - zc;
      r = Math.sqrt(far * far * t2 + dx * dx);
    }

    this._sphereZ = zc;
    this._sphereR = Math.ceil(r * 4) / 4;   // quantise: kills the fov-transition shimmer
    this._fitFov = cam.fov;
    this._fitAspect = cam.aspect;
    this._fitNear = cam.near;
    this._fitDist = far;

    const mapSize = this.sun.shadow.mapSize.x || 1024;
    this._texelWorld = (this._sphereR * 2) / mapSize;
    this._updateShadowBias();
  }

  // --- BIAS, derived from the real world size of one shadow texel and the real sun
  //     elevation rather than guessed at.
  //
  // Adjacent shadow texels covering a plane lit at incidence theta differ in depth by
  // texel * tan(theta). The worst case is the ground: at a 11-degree sun that is
  // cot(11deg) ~= 5.1 texels, i.e. ~18 cm of depth spread across one 3.5 cm texel. The
  // bias has to clear about half of that:
  //   * normalBias pushes the lookup along the surface normal, which on a grazing surface
  //     only buys b * sin(elevation) of depth — it cannot carry the correction alone, and
  //     raising it far enough would visibly shrink every contact shadow.
  //   * the constant bias carries the rest, slope-scaled by cot(elevation) so it tightens
  //     automatically as Sky lifts the sun through the day, and hard-capped at 7.5 cm along
  //     the light axis (~15 cm of shadow slide on the ground at this sun angle) so contact
  //     shadows never visibly detach.
  // Everything scales with texel size, so every quality level and every fov stays tuned.
  _updateShadowBias() {
    const texel = this._texelWorld || 0.05;
    const sinE = THREE.MathUtils.clamp(Math.abs(this.sunDir.y), 0.06, 1);
    const cotE = Math.sqrt(Math.max(0, 1 - sinE * sinE)) / sinE;
    const depthRange = Math.max(1, (this._sphereR + this._casterExtent) + this._sphereR + 25);
    const k = this.shadowBiasScale;

    const worldBias = THREE.MathUtils.clamp(0.6 * texel * cotE * k, texel * 0.5, 0.075);
    this.sun.shadow.bias = -(worldBias / depthRange);
    this.sun.shadow.normalBias = THREE.MathUtils.clamp(texel * 2.0 * k, 0.02, 0.11);
    this._biasSinE = sinE;
  }

  _fitSunShadow(force) {
    const cam = this.g.camera;
    if (!cam || !this.sun.castShadow) return;

    cam.updateMatrixWorld();

    if (force || cam.fov !== this._fitFov || cam.aspect !== this._fitAspect ||
        cam.near !== this._fitNear || this._shadowFit !== this._fitDist) {
      this._recomputeFitSphere(cam);
    } else if (Math.abs(Math.abs(this.sunDir.y) - this._biasSinE) > 0.008) {
      // Sky moved the sun far enough that the slope-scaled bias needs re-deriving.
      this._updateShadowBias();
    }

    const r = this._sphereR;

    // Sphere centre in world space: straight down the camera's forward axis.
    this._vCenter.set(0, 0, -this._sphereZ).applyMatrix4(cam.matrixWorld);

    // Build the light-space rotation (translation-free, so the texel grid is anchored to
    // the world origin and therefore stable frame to frame).
    const dir = this.sunDir;
    this._vUp.set(0, 1, 0);
    if (Math.abs(dir.y) > 0.998) this._vUp.set(0, 0, 1);
    this.sun.shadow.camera.up.copy(this._vUp);

    this._vNegDir.copy(dir).negate();
    this._mLight.lookAt(this._vZero.set(0, 0, 0), this._vNegDir, this._vUp);
    this._mLightInv.copy(this._mLight).transpose();  // pure rotation -> inverse == transpose

    // Snap the centre to whole texels along the light's own axes.
    const texel = (r * 2) / (this.sun.shadow.mapSize.x || 1024);
    this._vSnap.copy(this._vCenter).applyMatrix4(this._mLightInv);
    this._vSnap.x = Math.floor(this._vSnap.x / texel) * texel;
    this._vSnap.y = Math.floor(this._vSnap.y / texel) * texel;
    this._vSnap.applyMatrix4(this._mLight);

    // Pull the light back far enough up-sun that tall casters outside the view still
    // reach the near plane. In light space "up-sun" is +Z, so this costs depth range only.
    const back = r + this._casterExtent;
    this.sun.position.set(
      this._vSnap.x + dir.x * back,
      this._vSnap.y + dir.y * back,
      this._vSnap.z + dir.z * back,
    );
    this.sunTarget.position.copy(this._vSnap);

    const sc = this.sun.shadow.camera;
    const nearP = 1;
    const farP = back + r + 25;
    if (sc.left !== -r || sc.right !== r || sc.top !== r || sc.bottom !== -r ||
        sc.near !== nearP || sc.far !== farP) {
      sc.left = -r; sc.right = r; sc.top = r; sc.bottom = -r;
      sc.near = nearP; sc.far = farP;
      sc.updateProjectionMatrix();
    }

    // The shadow camera is derived from these two transforms during render; update them
    // now so a mid-frame shadow render (PostFX prepasses) sees the fitted state.
    this.sun.updateMatrixWorld();
    this.sunTarget.updateMatrixWorld();
  }

  // --------------------------------------------------------------------------
  //  Sky is constructed after us and drives the sun by writing sun.position each
  //  frame — which would stomp the fitted shadow position if we only fitted inside
  //  update(). So the authoritative fit runs from scene.onBeforeRender, which three
  //  invokes at the top of WebGLRenderer.render, before shadowMap.render. That is
  //  after every module's update() has run, so whatever Sky wrote is respected as the
  //  sun *direction*, and the fitted position always wins for the shadow camera.
  //  The hook chains any handler already installed and re-installs itself if another
  //  module later replaces it.
  // --------------------------------------------------------------------------
  _ensureSceneHook() {
    const scene = this.g.scene;
    if (!scene || scene.onBeforeRender === this._sceneHook) return;
    if (this._hookAttempts >= 6) return;
    this._hookAttempts = (this._hookAttempts || 0) + 1;

    const prev = (typeof scene.onBeforeRender === 'function') ? scene.onBeforeRender : null;
    this._prevSceneHook = prev;
    const self = this;
    this._sceneHook = function (renderer, sc, camera, renderTarget) {
      if (self._prevSceneHook) {
        try { self._prevSceneHook.call(this, renderer, sc, camera, renderTarget); }
        catch (e) { console.error('[Renderer] chained scene.onBeforeRender threw', e); }
      }
      self._onSceneRender(camera);
    };
    this._sceneHook.__blacksiteRenderer = true;
    scene.onBeforeRender = this._sceneHook;
  }

  _onSceneRender(camera) {
    // PostFX may render the scene more than once per frame (prepasses); fit exactly once,
    // and only for the real player camera.
    if (camera !== this.g.camera) return;
    const frame = this.g.frame;
    if (this._fitFrame === frame) return;
    this._fitFrame = frame;
    this._lastFitFrame = frame;
    if (this._contextLost) return;
    this._readSunDirectionFromLight();
    if (this.shadowsEnabled && this.sun.castShadow) this._fitSunShadow(false);
  }

  // Sky drives the rig by moving the light; recover the unit direction from it so the
  // fitted position we write back never destroys the direction Sky asked for.
  _readSunDirectionFromLight() {
    this._vTmp.copy(this.sun.position).sub(this.sunTarget.position);
    if (this._vTmp.lengthSq() <= 1e-6) return;
    this._vTmp.normalize();
    if (this._vTmp.dot(this.sunDir) < 0.999995) {
      this.sunDir.copy(this._vTmp);
      this._syncFillLights();
    }
  }

  _invalidateShadow() {
    const sh = this.sun?.shadow;
    if (!sh) return;
    if (sh.map) { sh.map.dispose(); sh.map = null; }
    if (sh.mapPass) { sh.mapPass.dispose(); sh.mapPass = null; }
    sh.needsUpdate = true;
  }

  // ==========================================================================
  //  QUALITY
  // ==========================================================================

  setQuality(level) {
    const preset = QUALITY_PRESETS[level] || QUALITY_PRESETS.ultra;
    const cfg = this.g.config;
    this.quality = QUALITY_PRESETS[level] ? level : 'ultra';
    cfg.quality = this.quality;

    // Shadow map resolution.
    const size = preset.shadowMapSize || cfg.render.shadowMapSize;
    if (size !== this._shadowMapSize || this.sun.shadow.mapSize.x !== size) {
      this._shadowMapSize = size;
      this.sun.shadow.mapSize.set(size, size);
      this._invalidateShadow();
    }

    // Shadow filtering.
    this.setShadowType(preset.shadowType);

    // Cascade extents.
    this._shadowFit = Math.min(preset.shadowFit, cfg.render.shadowDistance);
    this._casterExtent = preset.casterExtent;
    this._fitDist = -1;                       // force a sphere recompute next fit

    // Resolution.
    this._pixelRatioCap = preset.pixelRatioCap || cfg.render.pixelRatioCap;
    this._scaleIndex = 0;
    this._applyPixelRatio(true);

    // Ambient trim.
    this.setAmbientIntensity(this.ambientScale * preset.ambientScale);

    this._fitSunShadow(true);
    return this;
  }

  setShadowType(name) {
    const map = {
      basic: THREE.BasicShadowMap,
      pcf: THREE.PCFShadowMap,
      pcfsoft: THREE.PCFSoftShadowMap,
      vsm: THREE.VSMShadowMap,
    };
    const type = map[name] ?? THREE.PCFSoftShadowMap;
    if (this.renderer.shadowMap.type === type && this._shadowTypeName === name) return this;
    this._shadowTypeName = name;
    this.renderer.shadowMap.type = type;
    if (type === THREE.VSMShadowMap) {
      this.sun.shadow.radius = 2.5;
      this.sun.shadow.blurSamples = 10;
    } else if (type === THREE.PCFShadowMap) {
      this.sun.shadow.radius = 1.6;
    }
    this._invalidateShadow();
    this._recompileMaterials();
    return this;
  }

  setShadowsEnabled(on) {
    this.shadowsEnabled = !!on;
    this.renderer.shadowMap.enabled = this.shadowsEnabled;
    this.sun.castShadow = this.shadowsEnabled;
    this._invalidateShadow();
    this._recompileMaterials();
    return this;
  }

  /** Scales the auto-derived bias pair. 1 = tuned default. Raise if a collaborator's
   *  geometry shows acne, lower if contact shadows detach. */
  setShadowBiasScale(k) {
    this.shadowBiasScale = Math.max(0.05, k);
    this._fitDist = -1;
    this._fitSunShadow(true);
    return this;
  }

  setShadowDistance(metres) {
    this._shadowFit = Math.max(8, Math.min(metres, this.g.config.render.shadowDistance));
    this._fitDist = -1;
    this._fitSunShadow(true);
    return this;
  }

  // Shadow-type / shadow-enable changes alter the shader defines of every lit material.
  // Only ever called from explicit quality changes, never per frame.
  _recompileMaterials() {
    const seen = this._recompileSet || (this._recompileSet = new Set());
    seen.clear();
    this.g.scene?.traverse((o) => {
      const m = o.material;
      if (!m) return;
      if (Array.isArray(m)) { for (let i = 0; i < m.length; i++) if (m[i] && !seen.has(m[i])) { seen.add(m[i]); m[i].needsUpdate = true; } }
      else if (!seen.has(m)) { seen.add(m); m.needsUpdate = true; }
    });
    seen.clear();
  }

  // ==========================================================================
  //  TONE MAPPING / EXPOSURE
  // ==========================================================================

  _resolveToneMapping(name) {
    switch (String(name).toLowerCase()) {
      case 'none': return THREE.NoToneMapping;
      case 'linear': return THREE.LinearToneMapping;
      case 'reinhard': return THREE.ReinhardToneMapping;
      case 'cineon': return THREE.CineonToneMapping;
      case 'aces': return THREE.ACESFilmicToneMapping;
      case 'neutral': return THREE.NeutralToneMapping ?? THREE.ACESFilmicToneMapping;
      case 'agx':
      default: return THREE.AgXToneMapping ?? THREE.ACESFilmicToneMapping;
    }
  }

  setToneMapping(name) {
    this._toneMapName = name;
    if (this._toneMapEnabled) this.renderer.toneMapping = this._resolveToneMapping(name);
    return this;
  }

  /** PostFX calls setToneMappingEnabled(false) if it applies the display transform itself. */
  setToneMappingEnabled(on) {
    this._toneMapEnabled = !!on;
    this.renderer.toneMapping = this._toneMapEnabled
      ? this._resolveToneMapping(this._toneMapName)
      : THREE.NoToneMapping;
    return this;
  }

  setExposure(v) {
    this.g.config.render.exposure = v;
    this.renderer.toneMappingExposure = v;
    return this;
  }

  // ==========================================================================
  //  RESOLUTION / ADAPTIVE SCALER
  // ==========================================================================

  _applyPixelRatio(notify) {
    const pr = Math.max(0.5, Math.min(this._baseDpr, this._pixelRatioCap) * this.renderScale);
    if (Math.abs(pr - (this._appliedPixelRatio ?? -1)) < 1e-4) return;
    this._appliedPixelRatio = pr;
    this.renderer.setPixelRatio(pr);
    if (notify) {
      const post = this.g.post;
      post?.setPixelRatio?.(pr);
      post?.composer?.setPixelRatio?.(pr);
      post?.resize?.(this._viewW, this._viewH);
    }
  }

  _detectSoftwareRasterizer() {
    try {
      const gl = this.renderer.getContext();
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const name = String(
        (ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) || gl.getParameter(gl.RENDERER) || ''
      );
      this.gpuName = name;
      return /swiftshader|llvmpipe|software|basic render|microsoft basic|mesa offscreen/i.test(name);
    } catch (e) {
      this.gpuName = 'unknown';
      return false;
    }
  }

  setAdaptive(on) { this.adaptiveEnabled = !!on; this._overFrames = 0; this._underFrames = 0; return this; }

  setRenderScaleIndex(i) {
    const idx = THREE.MathUtils.clamp(i | 0, 0, RENDER_SCALES.length - 1);
    if (idx === this._scaleIndex) return false;
    this._scaleIndex = idx;
    this.renderScale = RENDER_SCALES[idx];
    this._applyPixelRatio(true);
    return true;
  }

  _adapt(frameMs) {
    if (!this.adaptiveEnabled) return;
    // Ignore the first second: shader compiles and texture uploads dominate.
    if (this._warmupFrames < 90) { this._warmupFrames++; return; }
    // A single 100 ms hitch (GC, decal atlas rebuild) is not a resolution problem.
    if (frameMs > 100) { this._overFrames = 0; this._underFrames = 0; return; }

    if (frameMs > this.frameBudgetMs) { this._overFrames++; this._underFrames = 0; }
    else if (frameMs < this.frameGoodMs) { this._underFrames++; this._overFrames = 0; }
    else { this._overFrames = 0; this._underFrames = 0; }

    if (this._overFrames >= 60) {
      this._overFrames = 0; this._underFrames = 0;
      if (this.setRenderScaleIndex(this._scaleIndex + 1)) this._warmupFrames = 60;
    } else if (this._underFrames >= 360) {
      this._overFrames = 0; this._underFrames = 0;
      if (this.setRenderScaleIndex(this._scaleIndex - 1)) this._warmupFrames = 60;
    }
  }

  // ==========================================================================
  //  FRAME
  // ==========================================================================

  update(dt) {
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const frameMs = now - this._lastNow;
    this._lastNow = now;
    if (frameMs > 0 && frameMs < 2000) {
      this.frameMs = frameMs;
      this.frameMsSmooth += (frameMs - this.frameMsSmooth) * 0.08;
      this.fps = 1000 / Math.max(0.5, this.frameMsSmooth);
    }

    // Stats accumulated over every pass of the frame we just presented.
    const info = this.renderer.info;
    this.drawCalls = info.render.calls;
    this.triangles = info.render.triangles;
    this.programs = info.programs ? info.programs.length : 0;
    this.geometries = info.memory.geometries;
    this.textures = info.memory.textures;
    if (info.autoReset === false) info.reset();

    if (this._contextLost) return;

    // Keep the render-time hook installed (cheap identity compare; re-chains at most a
    // handful of times if another module replaces scene.onBeforeRender).
    if ((this.g.frame & 31) === 0) this._ensureSceneHook();

    if (!this._atmosphereChecked && this.g.frame > 3) {
      this._atmosphereChecked = true;
      const scene = this.g.scene;
      if (!this.g.sky) {
        if (!scene.background) scene.background = new THREE.Color(0x9db2c4);
        if (!scene.fog) scene.fog = new THREE.FogExp2(0xc0aa8a, this.g.config.sky.fogDensity);
      }
      // If Sky put a real IBL up, the hemisphere + cool fill are describing the same sky
      // energy a second time. Measured on the boot scene, the analytic rig was adding a
      // flat ~6% of the total ground irradiance on top of the probe — pure contrast loss.
      // Trim the analytic rig back to a complement rather than a duplicate. Full strength
      // is kept when there is no IBL, so a missing Sky still lights the level correctly.
      if (this.ambientAutoTrim && scene.environment) this.setAmbientIntensity(0.45);
    }

    // Fallback path: if the hook has not fired for a few frames (a collaborator renders
    // the scene some other way), fit here instead so shadows never go stale.
    if (this._lastFitFrame === undefined || this.g.frame - this._lastFitFrame > 2) {
      this._readSunDirectionFromLight();
      if (this.shadowsEnabled && this.sun.castShadow) this._fitSunShadow(false);
    }

    this._adapt(frameMs);
  }

  resize(w, h) {
    this._viewW = Math.max(1, w);
    this._viewH = Math.max(1, h);
    this._baseDpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1;
    this._applyPixelRatio(false);
    this.renderer.setSize(this._viewW, this._viewH, false);
    this._fitDist = -1;               // aspect changed -> refit the cascade sphere
    this._fitSunShadow(true);
  }

  getStats() {
    return {
      fps: this.fps,
      frameMs: this.frameMs,
      drawCalls: this.drawCalls,
      triangles: this.triangles,
      programs: this.programs,
      pixelRatio: this._appliedPixelRatio,
      renderScale: this.renderScale,
      quality: this.quality,
      shadowTexel: this._texelWorld,
      shadowBias: this.sun.shadow.bias,
      shadowNormalBias: this.sun.shadow.normalBias,
      gpu: this.gpuName,
      software: this.softwareRasterizer,
      adaptive: this.adaptiveEnabled,
    };
  }

  dispose() {
    const c = this.renderer.domElement;
    c.removeEventListener('webglcontextlost', this._onCtxLost);
    c.removeEventListener('webglcontextrestored', this._onCtxRestored);
    this._invalidateShadow();
    this.renderer.dispose();
  }
}

export default Renderer;
