// BLACKSITE — PostFX.js
// Owner: the post-processing agent. This module owns the ENTIRE image chain after the
// scene is submitted, and owns the FINAL PRESENT to the default framebuffer.
//
// main.js calls `post.render(dt)` once per frame instead of renderer.render(). If this
// module is missing or has failed, main.js falls back to a direct render — so every code
// path here is defensive: any throw permanently degrades to a direct render rather than
// black-screening the game.
//
// ---------------------------------------------------------------------------------------
// PIPELINE (in execution order)
// ---------------------------------------------------------------------------------------
//   0. Sky god-ray prepass          game.sky.renderGodrays(renderer)   (zero latency)
//   1. Scene -> sceneRT             HDR RGBA16F + DEPTH24, camera projection sub-pixel
//                                   jittered on a 16-sample Halton (2,3) sequence
//   2. Ground-truth-ish AO          half-res horizon/normal AO reconstructed from depth.
//                                   Small world radius (contact only), interleaved
//                                   rotation, bilateral cross blur. No grey halo.
//   3. TAA resolve                  reprojection through the previous frame's jittered
//                                   view-projection, 3x3 variance clipping, Karis
//                                   luminance weighting, disocclusion rejection. AO is
//                                   folded in here so it gets temporally denoised too.
//   4. Bloom                        Jimenez/CoD-style progressive down/up pyramid, 6 mips,
//                                   13-tap downsample + 9-tap tent upsample, firefly clamp
//                                   and a high soft-knee threshold. Procedural lens dirt.
//   5. Viewmodel                    game.weapon.viewmodelScene rendered to its own HDR
//                                   RGBA target with alpha, its own depth, NOT jittered
//                                   and NOT temporally filtered (stays razor sharp).
//   6. Composite                    one big pass: DOF + camera motion blur as a single
//                                   velocity-elongated bokeh gather, bloom add, god-ray
//                                   add, viewmodel over-blend, linear vignette, AgX tone
//                                   map, FILM GRADE, sRGB encode -> LDR.
//   7. SMAA                         three's addon, driven manually (LDR gamma-space in).
//   8. Present                      radial chromatic aberration + contrast-adaptive
//                                   sharpen + animated luminance-weighted grain + ordered
//                                   dither -> default framebuffer.
//
// NOTE ON TONE MAPPING: three only applies renderer.toneMapping when a material is drawn
// to the DEFAULT framebuffer. Everything here renders into render targets, so the scene
// arrives at the composite as untouched scene-linear HDR and this module owns the whole
// display transform. Renderer.setToneMappingEnabled() is deliberately left alone so the
// emergency direct-render fallback still looks sane.
//
// NOTE ON SSR: deliberately NOT enabled. A correct screen-space reflection needs a normal
// + roughness G-buffer; this pipeline is forward-rendered with a depth-only prepass, and a
// depth-reconstructed SSR on a dry, dusty, fully-rough golden-hour compound buys nothing
// but shimmer and cost. Config.post.ssr is honoured (read + reported) but maps to no pass.
//
// ---------------------------------------------------------------------------------------
// PUBLIC API
// ---------------------------------------------------------------------------------------
//   game.post.render(dt)                     called by main.js — final present
//   game.post.enabled                        master switch (false -> plain direct render)
//   game.post.params                         every live tunable (see _defaultParams)
//   game.post.set(name, value)               e.g. set('bloomStrength', 0.08)
//   game.post.setEnabled(flagName, bool)     flagName matches Config.post keys
//   game.post.setFocus(metres | null)        null = auto focus on the screen centre
//   game.post.setPixelRatio(pr)              Renderer calls this when it rescales
//   game.post.resize(w, h)                   resizes every render target
//   game.post.getStats()                     {passes, targets, mb, taaFrames}
//   game.post.sceneTexture / .depthTexture   HDR colour + depth of the current frame

import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

// =========================================================================================
// Halton (2,3) — the standard TAA sub-pixel sequence. 16 samples is enough to look like
// 16x supersampling on a static frame while staying short enough to reconverge fast.
// =========================================================================================
function halton(index, base) {
  let f = 1, r = 0, i = index;
  while (i > 0) { f /= base; r += f * (i % base); i = Math.floor(i / base); }
  return r;
}
const JITTER = (() => {
  const a = new Float32Array(32);
  for (let i = 0; i < 16; i++) {
    a[i * 2] = halton(i + 1, 2) - 0.5;
    a[i * 2 + 1] = halton(i + 1, 3) - 0.5;
  }
  return a;
})();

// =========================================================================================
// Shared GLSL
// =========================================================================================
const VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}`;

const GLSL_COMMON = /* glsl */`
float hash12( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}
vec2 hash22( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * vec3( 0.1031, 0.1030, 0.0973 ) );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.xx + p3.yz ) * p3.zy );
}
float luma( vec3 c ) { return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) ); }
float maxc( vec3 c ) { return max( c.r, max( c.g, c.b ) ); }
`;

// Self-contained display transform. Deliberately NOT `#include <tonemapping_pars_fragment>`:
// three already prepends the colour-space chunk (and, depending on the material's tone-map
// state, the tone-mapping chunk) to every non-raw ShaderMaterial, so including them again is
// a redefinition error. This is three's own AgX, verbatim in maths, driven by our own
// uniform so nothing depends on renderer.toneMapping.
const GLSL_DISPLAY = /* glsl */`
const mat3 AGX_REC2020_TO_SRGB = mat3(
  vec3(  1.6605, -0.1246, -0.0182 ),
  vec3( -0.5876,  1.1329, -0.1006 ),
  vec3( -0.0728, -0.0083,  1.1187 ) );
const mat3 AGX_SRGB_TO_REC2020 = mat3(
  vec3( 0.6274, 0.0691, 0.0164 ),
  vec3( 0.3293, 0.9195, 0.0880 ),
  vec3( 0.0433, 0.0113, 0.8956 ) );

vec3 agxContrast( vec3 x ) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return + 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
         - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agxToneMap( vec3 color, float exposure ) {
  const mat3 AgXInset = mat3(
    vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
    vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
    vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 ) );
  const mat3 AgXOutset = mat3(
    vec3( 1.1271005818144368, -0.1413297634984383, -0.14132976349843826 ),
    vec3( -0.11060664309660323, 1.157823702216272, -0.11060664309660294 ),
    vec3( -0.016493938717834573, -0.016493938717834257, 1.2519364065950405 ) );
  const float AgxMinEv = -12.47393;
  const float AgxMaxEv = 4.026069;
  color *= exposure;
  color = AGX_SRGB_TO_REC2020 * color;
  color = AgXInset * color;
  color = max( color, 1e-10 );
  color = log2( color );
  color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
  color = clamp( color, 0.0, 1.0 );
  color = agxContrast( color );
  color = AgXOutset * color;
  color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
  color = AGX_REC2020_TO_SRGB * color;
  return clamp( color, 0.0, 1.0 );
}

vec3 encodeSRGB( vec3 c ) {
  return mix( pow( max( c, vec3( 0.0 ) ), vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ),
              c * 12.92, vec3( lessThanEqual( c, vec3( 0.0031308 ) ) ) );
}
`;

// Positive metres in front of the camera from a [0,1] hardware depth sample.
const GLSL_DEPTH = /* glsl */`
float linearZ( float d, float n, float f ) {
  float z = d * 2.0 - 1.0;
  return ( 2.0 * n * f ) / ( f + n - z * ( f - n ) );
}
vec3 viewFromDepth( mat4 projInv, vec2 uv, float d ) {
  vec4 c = projInv * vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
  return c.xyz / c.w;
}
`;

// =========================================================================================
export class PostFX {

  constructor(game) {
    this.g = game;
    this.cfg = game.config;
    this.enabled = true;

    this._failed = false;
    this._time = 0;
    this._frame = 0;
    this._taaFrames = 0;
    this._jitterIndex = 0;
    this._historyValid = false;

    this.width = 1;
    this.height = 1;
    this._pixelRatio = 1;

    // ---- scratch (nothing in render() allocates) ---------------------------------------
    this._size = new THREE.Vector2();
    this._clearColor = new THREE.Color();
    this._curVP = new THREE.Matrix4();
    this._prevVP = new THREE.Matrix4();
    this._curVPInv = new THREE.Matrix4();
    this._projInv = new THREE.Matrix4();
    this._savedProjInv = new THREE.Matrix4();
    this._godrayTint = new THREE.Color(1.0, 0.86, 0.66);
    this._savedE8 = 0;
    this._savedE9 = 0;

    this.params = this._defaultParams();
    this.flags = this._readFlags();

    // 1x1 fallbacks so every sampler is always bound to something real.
    this._blackTex = this._makeSolidTexture(0, 0, 0, 0);
    this._whiteTex = this._makeSolidTexture(255, 255, 255, 255);
    this._dirtTex = this._makeLensDirt(512);

    this._quad = new FullScreenQuad(null);
    this._targets = [];
    this._materials = [];

    this._buildMaterials();
    this._smaa = new SMAAPass();
  }

  // =======================================================================================
  //  TUNING
  // =======================================================================================
  //
  // Every number below was walked in against the art direction: late-afternoon golden hour,
  // warm key / cool skylight, heavy haze, filmic grade with lifted cool blacks and a warm
  // highlight rolloff. The grade block is the single biggest lever in the whole renderer.
  _defaultParams() {
    return {
      // --- exposure ----------------------------------------------------------------
      // Multiplied by Config.render.exposure. The scene is lit in three's physical units
      // by a ~10.7-degree sun, so almost every surface is at a grazing incidence and the
      // scene-linear average sits around 0.04 — roughly three stops under a correctly
      // exposed plate. This is the camera's exposure compensation, not a fudge: it puts
      // sunlit plaster near 0.75 display and keeps the sky off the clip point.
      exposure: 1.35,

      // --- ambient occlusion (contact only; anything wider reads as a dirty grey halo)
      aoRadius: 0.62,              // metres
      aoIntensity: 0.95,
      aoBias: 0.030,
      aoMaxRadiusPx: 42,           // screen-space clamp so near-field AO can't smear
      aoStrength: 0.85,            // how much of the AO term reaches the beauty buffer
      aoDepthSigma: 3.2,           // bilateral blur depth rejection (1/metres)

      // --- temporal AA -------------------------------------------------------------
      taaBlend: 0.085,             // weight of the CURRENT frame when fully converged
      taaBlendMoving: 0.26,        // weight while the camera is moving fast
      taaClipGamma: 1.20,          // neighbourhood variance clipping width
      taaMotionScale: 34.0,        // uv/s -> blend ramp

      // --- bloom -------------------------------------------------------------------
      bloomThreshold: 1.02,        // display-referred; only genuinely hot pixels bloom
      bloomKnee: 0.55,
      bloomClamp: 9.0,             // firefly / sun-disc clamp in scene-linear units
      bloomStrength: 0.058,
      bloomScatter: 0.80,          // upsample tent weight -> how wide the veil spreads
      lensDirt: 0.14,              // barely-there veil modulation; see _makeLensDirt

      // --- god rays ----------------------------------------------------------------
      godrayStrength: 0.62,

      // --- depth of field ----------------------------------------------------------
      dofStrength: 0.72,           // CoC gain; deliberately shallow
      dofMaxCoC: 4.2,              // px at 1080p
      dofNearScale: 0.85,          // foreground defocuses a touch harder than the far field
      dofFarScale: 0.14,           // background barely defocuses — the skyline must stay crisp
      dofFocusBias: 0.0,

      // --- motion blur -------------------------------------------------------------
      motionScale: 0.52,           // fraction of a frame's screen motion
      motionMax: 0.016,            // uv, hard cap (~31 px at 1080p)

      // --- lens ---------------------------------------------------------------------
      chromatic: 1.35,             // px of R/B separation at the frame corner
      chromaPow: 2.6,              // how hard it falls off toward the centre
      vignette: 0.34,
      vignetteStart: 0.30,
      vignettePow: 1.55,

      // --- grain / sharpen ----------------------------------------------------------
      grain: 0.021,
      grainSize: 1.0,
      sharpen: 0.34,

      // --- FILM GRADE ----------------------------------------------------------------
      // Applied post tone-map, in display-referred linear.
      lift: [0.0125, 0.0155, 0.0270],   // milky blacks, cool-blue biased
      gain: [1.012, 1.000, 0.980],      // warm gain
      gamma: [1.000, 1.005, 1.020],     // pull the blue mids down a hair
      contrast: 1.115,
      contrastPivot: 0.415,
      softS: 0.20,                       // extra toe + shoulder on top of the linear contrast
      shadowTint: [-0.012, 0.004, 0.030],  // teal / cool-blue shadows
      highTint: [0.032, 0.013, -0.022],    // amber highlights
      splitAmount: 1.0,
      shadowSat: 0.70,                   // deep shadows desaturate
      highSat: 0.90,
      saturation: 1.07,
      highWarm: [1.020, 0.998, 0.958],   // highlight rolloff drifts warm
      blackFloor: [0.0115, 0.0140, 0.0230],
      whiteCeil: [0.988, 0.980, 0.960],
    };
  }

  _readFlags() {
    const p = this.cfg?.post || {};
    const q = this.cfg?.quality || 'ultra';
    return {
      taa: p.taa !== false,
      smaa: p.smaa !== false,
      ssao: p.ssao !== false && q !== 'low',
      ssr: false,                         // see the header note — intentionally inert
      bloom: p.bloom !== false,
      motionBlur: p.motionBlur !== false,
      dof: p.dof !== false,
      chromatic: p.chromatic !== false,
      grain: p.grain !== false,
      vignette: p.vignette !== false,
      sharpen: p.sharpen !== false,
      lensDirt: p.lensDirt !== false,
      godrays: this.cfg?.sky?.godRays !== false,
    };
  }

  // =======================================================================================
  //  PROCEDURAL TEXTURES
  // =======================================================================================
  _makeSolidTexture(r, g, b, a) {
    const d = new Uint8Array([r, g, b, a]);
    const t = new THREE.DataTexture(d, 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.needsUpdate = true;
    t.colorSpace = THREE.NoColorSpace;
    return t;
  }

  // Front-element grime: smeared blobs, radial scratches and dust specks. It only ever
  // modulates the bloom veil, so it reads as a real lens rather than an overlay.
  _makeLensDirt(N) {
    const c = document.createElement('canvas');
    c.width = c.height = N;
    const x = c.getContext('2d');
    if (!x) return this._whiteTex;

    x.fillStyle = '#000000';
    x.fillRect(0, 0, N, N);

    // Everything below is deliberately LOW CONTRAST. The previous pass painted
    // 0.6px hard-edged scratches into a 384px map that then got magnified ~3x
    // across a 1280px frame: the result was a permanent bright hairline arcing
    // over the viewmodel and a fixed orange bloom smudge that sat in exactly
    // the same screen spot in every single camera. A real front element only
    // shows up as a broad, soft, barely-there breathing of the bloom veil.

    // soft smudges — broad and faint
    for (let i = 0; i < 30; i++) {
      const px = Math.random() * N, py = Math.random() * N;
      const r = (0.08 + Math.pow(Math.random(), 1.6) * 0.22) * N;
      const g = x.createRadialGradient(px, py, 0, px, py, r);
      const a = 0.035 + Math.random() * 0.085;
      g.addColorStop(0, `rgba(255,255,255,${a})`);
      g.addColorStop(0.55, `rgba(255,255,255,${a * 0.40})`);
      g.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = g;
      x.beginPath();
      x.ellipse(px, py, r, r * (0.55 + Math.random() * 0.8), Math.random() * Math.PI, 0, Math.PI * 2);
      x.fill();
    }

    // wipe marks — wide and soft, never a hairline
    x.lineCap = 'round';
    for (let i = 0; i < 14; i++) {
      const px = Math.random() * N, py = Math.random() * N;
      const ang = Math.random() * Math.PI;
      const len = (0.10 + Math.random() * 0.30) * N;
      const grd = x.createLinearGradient(px, py, px + Math.cos(ang) * len, py + Math.sin(ang) * len);
      const a = 0.018 + Math.random() * 0.038;
      grd.addColorStop(0, 'rgba(255,255,255,0)');
      grd.addColorStop(0.5, `rgba(255,255,255,${a})`);
      grd.addColorStop(1, 'rgba(255,255,255,0)');
      x.strokeStyle = grd;
      x.lineWidth = N * (0.012 + Math.random() * 0.022);
      x.beginPath();
      x.moveTo(px, py);
      x.quadraticCurveTo(
        px + Math.cos(ang + 0.35) * len * 0.5, py + Math.sin(ang + 0.35) * len * 0.5,
        px + Math.cos(ang) * len, py + Math.sin(ang) * len,
      );
      x.stroke();
    }

    // dust specks — sub-pixel once magnified, so they only add grain
    for (let i = 0; i < 700; i++) {
      const px = Math.random() * N, py = Math.random() * N;
      const r = 0.4 + Math.pow(Math.random(), 3.0) * 2.0;
      x.fillStyle = `rgba(255,255,255,${0.02 + Math.random() * 0.10})`;
      x.beginPath();
      x.arc(px, py, r, 0, Math.PI * 2);
      x.fill();
    }

    // Blur the whole plate so nothing survives magnification as an edge.
    // Wrapped: canvas filters are universally supported in the browsers this
    // ships to, but a failure here must never take the boot down.
    try {
      const b = document.createElement('canvas');
      b.width = b.height = N;
      const bx = b.getContext('2d');
      if (bx && 'filter' in bx) {
        bx.filter = `blur(${(N * 0.016).toFixed(2)}px)`;
        bx.drawImage(c, 0, 0);
        bx.filter = 'none';
        x.clearRect(0, 0, N, N);
        x.fillStyle = '#000000';
        x.fillRect(0, 0, N, N);
        x.drawImage(b, 0, 0);
      }
    } catch (e) { /* keep the unblurred plate */ }

    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = true;
    t.colorSpace = THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
  }

  // =======================================================================================
  //  RENDER TARGETS
  // =======================================================================================
  _rt(w, h, opts) {
    const t = new THREE.WebGLRenderTarget(Math.max(1, w | 0), Math.max(1, h | 0), Object.assign({
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    }, opts));
    t.texture.colorSpace = THREE.NoColorSpace;
    this._targets.push(t);
    return t;
  }

  _allocTargets(w, h) {
    const W = Math.max(2, w | 0), H = Math.max(2, h | 0);
    if (this.width === W && this.height === H && this.sceneRT) return;
    this.width = W;
    this.height = H;

    this._disposeTargets();

    // --- HDR scene + hardware depth ------------------------------------------------------
    const depth = new THREE.DepthTexture(W, H);
    depth.type = THREE.UnsignedIntType;
    depth.format = THREE.DepthFormat;
    depth.minFilter = THREE.NearestFilter;
    depth.magFilter = THREE.NearestFilter;
    depth.generateMipmaps = false;
    this._depthTex = depth;

    this.sceneRT = this._rt(W, H, { depthBuffer: true, depthTexture: depth });
    this.sceneRT.texture.name = 'post.scene';

    // --- TAA history ping-pong ------------------------------------------------------------
    this.taaA = this._rt(W, H);
    this.taaB = this._rt(W, H);
    this.taaA.texture.name = 'post.taaA';
    this.taaB.texture.name = 'post.taaB';
    this._taaRead = this.taaB;
    this._taaWrite = this.taaA;

    // --- AO (half res) ----------------------------------------------------------------
    const aw = Math.max(2, W >> 1), ah = Math.max(2, H >> 1);
    this.aoRT = this._rt(aw, ah, { type: THREE.UnsignedByteType });
    this.aoBlurRT = this._rt(aw, ah, { type: THREE.UnsignedByteType });
    this.aoRT.texture.name = 'post.ao';

    // --- bloom pyramid -------------------------------------------------------------------
    this.bloomMips = [];
    let bw = Math.max(2, W >> 1), bh = Math.max(2, H >> 1);
    const levels = Math.min(6, Math.max(3, Math.floor(Math.log2(Math.min(W, H))) - 3));
    for (let i = 0; i < levels; i++) {
      const rt = this._rt(bw, bh);
      rt.texture.name = 'post.bloom' + i;
      this.bloomMips.push(rt);
      bw = Math.max(2, bw >> 1);
      bh = Math.max(2, bh >> 1);
    }

    // --- viewmodel ------------------------------------------------------------------------
    this.vmRT = this._rt(W, H, { depthBuffer: true });
    this.vmRT.texture.name = 'post.viewmodel';

    // --- LDR chain -------------------------------------------------------------------------
    this.ldrRT = this._rt(W, H, { type: THREE.UnsignedByteType });
    this.aaRT = this._rt(W, H, { type: THREE.UnsignedByteType });
    this.ldrRT.texture.name = 'post.ldr';
    this.aaRT.texture.name = 'post.aa';

    this._smaa?.setSize(W, H);
    this._historyValid = false;
    this._taaFrames = 0;
    this._syncStaticUniforms();
  }

  _disposeTargets() {
    for (const t of this._targets) t.dispose();
    this._targets.length = 0;
    this._depthTex?.dispose?.();
    this._depthTex = null;
    this.sceneRT = this.taaA = this.taaB = this.aoRT = this.aoBlurRT = null;
    this.vmRT = this.ldrRT = this.aaRT = null;
    this.bloomMips = null;
  }

  // =======================================================================================
  //  MATERIALS
  // =======================================================================================
  _mat(name, fragment, uniforms, defines) {
    const m = new THREE.ShaderMaterial({
      name,
      defines: defines || {},
      uniforms,
      vertexShader: VERT,
      fragmentShader: fragment,
      depthTest: false,
      depthWrite: false,
      transparent: false,
      fog: false,
      toneMapped: false,
      blending: THREE.NoBlending,
    });
    this._materials.push(m);
    return m;
  }

  _buildMaterials() {
    const q = this.cfg?.quality || 'ultra';
    const aoDirs = q === 'low' ? 4 : q === 'high' ? 5 : 6;
    const aoSteps = q === 'low' ? 2 : 3;
    const dofTaps = q === 'low' ? 6 : q === 'high' ? 10 : 14;

    // ------------------------------------------------------------------ AO
    this.aoMat = this._mat('PostFX.AO', /* glsl */`
      uniform sampler2D tDepth;
      uniform mat4 uProjInv;
      uniform vec2 uAoTexel;
      uniform vec2 uFullTexel;
      uniform float uRadius, uIntensity, uBias, uMaxRadius, uProjScale, uAspect, uSeed;
      varying vec2 vUv;
      ${GLSL_COMMON}
      ${GLSL_DEPTH}

      void main() {
        float d = texture2D( tDepth, vUv ).x;
        if ( d >= 0.999995 ) { gl_FragColor = vec4( 1.0 ); return; }

        vec3 p = viewFromDepth( uProjInv, vUv, d );

        // normal from depth: pick the closer neighbour on each axis so silhouettes stay sharp
        vec2 ex = vec2( uFullTexel.x, 0.0 );
        vec2 ey = vec2( 0.0, uFullTexel.y );
        vec3 pr = viewFromDepth( uProjInv, vUv + ex, texture2D( tDepth, vUv + ex ).x );
        vec3 pl = viewFromDepth( uProjInv, vUv - ex, texture2D( tDepth, vUv - ex ).x );
        vec3 pu = viewFromDepth( uProjInv, vUv + ey, texture2D( tDepth, vUv + ey ).x );
        vec3 pd = viewFromDepth( uProjInv, vUv - ey, texture2D( tDepth, vUv - ey ).x );
        vec3 dx = ( abs( pr.z - p.z ) < abs( pl.z - p.z ) ) ? ( pr - p ) : ( p - pl );
        vec3 dy = ( abs( pu.z - p.z ) < abs( pd.z - p.z ) ) ? ( pu - p ) : ( p - pd );
        vec3 n = cross( dx, dy );
        float nl = length( n );
        if ( nl < 1e-8 ) { gl_FragColor = vec4( 1.0 ); return; }
        n /= nl;
        if ( n.z < 0.0 ) n = -n;

        // interleaved rotation — the TAA resolve turns this noise into extra samples
        vec2 rnd = hash22( gl_FragCoord.xy + uSeed );
        float rot = rnd.x * 6.2831853;
        float jit = rnd.y;

        float uvR = uProjScale * uRadius / max( 0.05, -p.z );
        uvR = min( uvR, uMaxRadius * uAoTexel.y );

        float occ = 0.0;
        float r2 = uRadius * uRadius;

        for ( int i = 0; i < DIRS; i ++ ) {
          float a = rot + float( i ) * ( 6.2831853 / float( DIRS ) );
          vec2 dir = vec2( cos( a ) / uAspect, sin( a ) );
          for ( int s = 1; s <= STEPS; s ++ ) {
            float t = ( float( s ) - 0.5 + jit * 0.9 ) / float( STEPS );
            vec2 suv = vUv + dir * uvR * t;
            float sd = texture2D( tDepth, suv ).x;
            if ( sd >= 0.999995 ) continue;
            vec3 sp = viewFromDepth( uProjInv, suv, sd );
            vec3 v = sp - p;
            float l2 = dot( v, v );
            float l = sqrt( max( l2, 1e-8 ) );
            float nd = dot( n, v / l );
            float att = 1.0 / ( 1.0 + l2 / r2 );
            occ += max( 0.0, nd - uBias ) * att;
          }
        }

        float ao = 1.0 - occ / float( DIRS * STEPS ) * uIntensity;
        ao = clamp( ao, 0.0, 1.0 );
        // slight gamma keeps the contact darkening tight instead of a broad wash
        ao = pow( ao, 1.35 );
        gl_FragColor = vec4( ao, ao, ao, 1.0 );
      }`, {
      tDepth: { value: null },
      uProjInv: { value: new THREE.Matrix4() },
      uAoTexel: { value: new THREE.Vector2() },
      uFullTexel: { value: new THREE.Vector2() },
      uRadius: { value: 0.62 },
      uIntensity: { value: 0.95 },
      uBias: { value: 0.03 },
      uMaxRadius: { value: 42 },
      uProjScale: { value: 1 },
      uAspect: { value: 1.777 },
      uSeed: { value: 0 },
    }, { DIRS: aoDirs, STEPS: aoSteps });

    // ------------------------------------------------------------- AO bilateral blur
    this.aoBlurMat = this._mat('PostFX.AOBlur', /* glsl */`
      uniform sampler2D tAO, tDepth;
      uniform vec2 uAoTexel;
      uniform float uNear, uFar, uSigma;
      varying vec2 vUv;
      ${GLSL_DEPTH}
      void main() {
        float refZ = linearZ( texture2D( tDepth, vUv ).x, uNear, uFar );
        float sum = 0.0, wsum = 0.0;
        for ( int y = -2; y <= 2; y ++ ) {
          for ( int x = -2; x <= 2; x ++ ) {
            vec2 o = vec2( float( x ), float( y ) ) * uAoTexel;
            float z = linearZ( texture2D( tDepth, vUv + o ).x, uNear, uFar );
            float wz = exp( -abs( z - refZ ) * uSigma );
            float ws = exp( -( float( x * x + y * y ) ) * 0.22 );
            float w = wz * ws;
            sum += texture2D( tAO, vUv + o ).r * w;
            wsum += w;
          }
        }
        float ao = wsum > 0.0 ? sum / wsum : 1.0;
        gl_FragColor = vec4( ao, ao, ao, 1.0 );
      }`, {
      tAO: { value: null },
      tDepth: { value: null },
      uAoTexel: { value: new THREE.Vector2() },
      uNear: { value: 0.05 },
      uFar: { value: 900 },
      uSigma: { value: 3.2 },
    });

    // ------------------------------------------------------------------ TAA resolve
    this.taaMat = this._mat('PostFX.TAA', /* glsl */`
      uniform sampler2D tCurrent, tHistory, tDepth, tAO;
      uniform mat4 uInvVP, uPrevVP;
      uniform vec2 uTexel;
      uniform float uBlend, uClipGamma, uAOStrength, uHistoryValid;
      varying vec2 vUv;
      ${GLSL_COMMON}

      vec3 fetch( vec2 uv ) {
        return min( max( texture2D( tCurrent, uv ).rgb, 0.0 ), 4096.0 );
      }

      void main() {
        vec3 c = fetch( vUv );

        // ---- ambient occlusion, folded in so it is temporally denoised for free ---------
        float ao = texture2D( tAO, vUv ).r;
        float aoF = mix( 1.0, ao, uAOStrength );
        c *= aoF;

        // ---- reprojection ---------------------------------------------------------------
        float d = texture2D( tDepth, vUv ).x;
        vec4 wp = uInvVP * vec4( vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
        wp /= wp.w;
        vec4 pp = uPrevVP * vec4( wp.xyz, 1.0 );
        vec2 prevUV = ( pp.xy / max( abs( pp.w ), 1e-6 ) * sign( pp.w ) ) * 0.5 + 0.5;

        bool valid = uHistoryValid > 0.5 && pp.w > 0.0 &&
          prevUV.x > 0.0 && prevUV.x < 1.0 && prevUV.y > 0.0 && prevUV.y < 1.0;

        if ( ! valid ) { gl_FragColor = vec4( c, 1.0 ); return; }

        vec3 h = min( max( texture2D( tHistory, prevUV ).rgb, 0.0 ), 4096.0 );

        // ---- 3x3 variance clipping -------------------------------------------------------
        vec3 m1 = vec3( 0.0 ), m2 = vec3( 0.0 );
        for ( int y = -1; y <= 1; y ++ ) {
          for ( int x = -1; x <= 1; x ++ ) {
            vec3 s = fetch( vUv + vec2( float( x ), float( y ) ) * uTexel ) * aoF;
            m1 += s;
            m2 += s * s;
          }
        }
        m1 /= 9.0;
        m2 /= 9.0;
        vec3 sigma = sqrt( max( m2 - m1 * m1, 0.0 ) );
        vec3 lo = m1 - sigma * uClipGamma;
        vec3 hi = m1 + sigma * uClipGamma;
        h = clamp( h, lo, hi );

        // ---- Karis luminance weighting: kills the fizz around the sun and specular ------
        float wc = 1.0 / ( 1.0 + luma( c ) );
        float wh = 1.0 / ( 1.0 + luma( h ) );
        float a = uBlend;
        float wsum = a * wc + ( 1.0 - a ) * wh;
        vec3 outc = ( c * a * wc + h * ( 1.0 - a ) * wh ) / max( wsum, 1e-5 );

        gl_FragColor = vec4( max( outc, 0.0 ), 1.0 );
      }`, {
      tCurrent: { value: null },
      tHistory: { value: null },
      tDepth: { value: null },
      tAO: { value: null },
      uInvVP: { value: new THREE.Matrix4() },
      uPrevVP: { value: new THREE.Matrix4() },
      uTexel: { value: new THREE.Vector2() },
      uBlend: { value: 0.085 },
      uClipGamma: { value: 1.2 },
      uAOStrength: { value: 0.85 },
      uHistoryValid: { value: 0 },
    });

    // ------------------------------------------------------------- bloom prefilter
    this.bloomPreMat = this._mat('PostFX.BloomPrefilter', /* glsl */`
      uniform sampler2D tSrc;
      uniform vec2 uTexel;
      uniform float uThreshold, uKnee, uClampMax, uExposure;
      varying vec2 vUv;
      ${GLSL_COMMON}

      vec3 tap( vec2 uv ) { return min( max( texture2D( tSrc, uv ).rgb, 0.0 ), uClampMax ); }

      void main() {
        // 4-tap Karis-averaged box: this is where fireflies die
        vec3 a = tap( vUv + vec2( -1.0, -1.0 ) * uTexel );
        vec3 b = tap( vUv + vec2(  1.0, -1.0 ) * uTexel );
        vec3 c = tap( vUv + vec2( -1.0,  1.0 ) * uTexel );
        vec3 d = tap( vUv + vec2(  1.0,  1.0 ) * uTexel );
        float wa = 1.0 / ( 1.0 + luma( a ) * uExposure );
        float wb = 1.0 / ( 1.0 + luma( b ) * uExposure );
        float wc = 1.0 / ( 1.0 + luma( c ) * uExposure );
        float wd = 1.0 / ( 1.0 + luma( d ) * uExposure );
        vec3 col = ( a * wa + b * wb + c * wc + d * wd ) / max( wa + wb + wc + wd, 1e-5 );

        // soft-knee threshold, evaluated in display-referred terms
        float l = maxc( col ) * uExposure;
        float knee = max( uThreshold * uKnee, 1e-4 );
        float soft = clamp( l - uThreshold + knee, 0.0, 2.0 * knee );
        soft = soft * soft / ( 4.0 * knee );
        float w = max( soft, l - uThreshold ) / max( l, 1e-4 );
        gl_FragColor = vec4( col * clamp( w, 0.0, 1.0 ), 1.0 );
      }`, {
      tSrc: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uThreshold: { value: 1.02 },
      uKnee: { value: 0.55 },
      uClampMax: { value: 9.0 },
      uExposure: { value: 1.0 },
    });

    // ------------------------------------------------------------- bloom downsample
    this.bloomDownMat = this._mat('PostFX.BloomDown', /* glsl */`
      uniform sampler2D tSrc;
      uniform vec2 uTexel;
      varying vec2 vUv;
      void main() {
        vec2 t = uTexel;
        vec3 a = texture2D( tSrc, vUv + vec2( -2.0,  2.0 ) * t ).rgb;
        vec3 b = texture2D( tSrc, vUv + vec2(  0.0,  2.0 ) * t ).rgb;
        vec3 c = texture2D( tSrc, vUv + vec2(  2.0,  2.0 ) * t ).rgb;
        vec3 d = texture2D( tSrc, vUv + vec2( -2.0,  0.0 ) * t ).rgb;
        vec3 e = texture2D( tSrc, vUv ).rgb;
        vec3 f = texture2D( tSrc, vUv + vec2(  2.0,  0.0 ) * t ).rgb;
        vec3 g = texture2D( tSrc, vUv + vec2( -2.0, -2.0 ) * t ).rgb;
        vec3 h = texture2D( tSrc, vUv + vec2(  0.0, -2.0 ) * t ).rgb;
        vec3 i = texture2D( tSrc, vUv + vec2(  2.0, -2.0 ) * t ).rgb;
        vec3 j = texture2D( tSrc, vUv + vec2( -1.0,  1.0 ) * t ).rgb;
        vec3 k = texture2D( tSrc, vUv + vec2(  1.0,  1.0 ) * t ).rgb;
        vec3 l = texture2D( tSrc, vUv + vec2( -1.0, -1.0 ) * t ).rgb;
        vec3 m = texture2D( tSrc, vUv + vec2(  1.0, -1.0 ) * t ).rgb;
        vec3 col = e * 0.125;
        col += ( a + c + g + i ) * 0.03125;
        col += ( b + d + f + h ) * 0.0625;
        col += ( j + k + l + m ) * 0.125;
        gl_FragColor = vec4( max( col, 0.0 ), 1.0 );
      }`, {
      tSrc: { value: null },
      uTexel: { value: new THREE.Vector2() },
    });

    // --------------------------------------------------------------- bloom upsample
    // Additive: the destination already holds the mip below, so we only add the tent.
    this.bloomUpMat = this._mat('PostFX.BloomUp', /* glsl */`
      uniform sampler2D tSrc;
      uniform vec2 uTexel;
      uniform float uScatter;
      varying vec2 vUv;
      void main() {
        vec2 t = uTexel;
        vec3 col = texture2D( tSrc, vUv + vec2( -1.0,  1.0 ) * t ).rgb * 1.0;
        col += texture2D( tSrc, vUv + vec2(  0.0,  1.0 ) * t ).rgb * 2.0;
        col += texture2D( tSrc, vUv + vec2(  1.0,  1.0 ) * t ).rgb * 1.0;
        col += texture2D( tSrc, vUv + vec2( -1.0,  0.0 ) * t ).rgb * 2.0;
        col += texture2D( tSrc, vUv ).rgb * 4.0;
        col += texture2D( tSrc, vUv + vec2(  1.0,  0.0 ) * t ).rgb * 2.0;
        col += texture2D( tSrc, vUv + vec2( -1.0, -1.0 ) * t ).rgb * 1.0;
        col += texture2D( tSrc, vUv + vec2(  0.0, -1.0 ) * t ).rgb * 2.0;
        col += texture2D( tSrc, vUv + vec2(  1.0, -1.0 ) * t ).rgb * 1.0;
        gl_FragColor = vec4( max( col, 0.0 ) * ( 1.0 / 16.0 ) * uScatter, 1.0 );
      }`, {
      tSrc: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uScatter: { value: 0.8 },
    });
    this.bloomUpMat.blending = THREE.AdditiveBlending;
    this.bloomUpMat.transparent = true;

    // ==================================================================== COMPOSITE
    this.compositeMat = this._mat('PostFX.Composite', /* glsl */`
      uniform sampler2D tColor, tDepth, tBloom, tGodray, tVM, tDirt;
      uniform mat4 uInvVP, uPrevVP;
      uniform vec2 uTexel;
      uniform float uAspect, uNear, uFar;
      uniform float uBloomStrength, uDirtAmount;
      uniform vec3 uGodrayColor;
      uniform float uGodrayStrength;
      uniform float uDofStrength, uDofMaxCoC, uDofNear, uDofFar, uFocusOverride, uFocusBias;
      uniform float uMotionScale, uMotionMax;
      uniform float uVignette, uVignetteStart, uVignettePow;
      uniform float uVMEnabled;
      uniform float uExposure;

      // --- film grade -----------------------------------------------------------------
      uniform vec3 uLift, uGain, uGamma;
      uniform float uContrast, uPivot, uSoftS;
      uniform vec3 uShadowTint, uHighTint;
      uniform float uSplit, uShadowSat, uHighSat, uSaturation;
      uniform vec3 uHighWarm, uBlackFloor, uWhiteCeil;

      varying vec2 vUv;
      ${GLSL_COMMON}
      ${GLSL_DEPTH}
      ${GLSL_DISPLAY}

      // Golden-angle bokeh disc, elongated along the per-pixel screen velocity. One gather
      // buys both a shallow depth of field and camera motion blur.
      vec3 gather( vec2 uv, float radiusPx, vec2 vel ) {
        vec3 acc = texture2D( tColor, uv ).rgb;
        float wsum = 1.0;
        for ( int i = 0; i < TAPS; i ++ ) {
          float fi = float( i );
          float a = fi * 2.39996323;
          float r = sqrt( ( fi + 0.5 ) / float( TAPS ) );
          vec2 disc = vec2( cos( a ), sin( a ) ) * r * radiusPx * uTexel;
          vec2 line = vel * ( ( fi + 0.5 ) / float( TAPS ) - 0.5 );
          vec3 s = texture2D( tColor, uv + disc + line ).rgb;
          // weight bright samples down slightly so a hot pixel cannot smear a comet
          float w = 1.0 / ( 1.0 + luma( s ) * 0.25 );
          acc += s * w;
          wsum += w;
        }
        return acc / wsum;
      }

      vec3 filmGrade( vec3 c ) {
        c = clamp( c, 0.0, 1.0 );

        // lift / gain / gamma — the lift fades out as the pixel gets brighter so only the
        // shadows go milky and cool, exactly like a print stock's toe.
        c = c * uGain + uLift * ( 1.0 - c );
        c = pow( max( c, 0.0 ), uGamma );

        // linear contrast about a pivot, then a gentle S on top for the toe + shoulder
        c = ( c - uPivot ) * uContrast + uPivot;
        c = clamp( c, 0.0, 1.0 );
        c = mix( c, c * c * ( 3.0 - 2.0 * c ), uSoftS );

        float l = luma( c );

        // split tone: teal shadows, amber highlights
        float sh = 1.0 - smoothstep( 0.0, 0.44, l );
        float hi = smoothstep( 0.42, 1.0, l );
        c += uShadowTint * sh * uSplit;
        c += uHighTint * hi * uSplit;

        // warm highlight rolloff
        c = mix( c, c * uHighWarm, hi );

        // shadows desaturate, highlights desaturate a little less
        float sat = mix( uShadowSat, 1.0, smoothstep( 0.0, 0.34, l ) );
        sat *= mix( 1.0, uHighSat, hi );
        float gl = luma( c );
        c = mix( vec3( gl ), c, sat * uSaturation );

        // never a pure black, never a clipped white
        c = clamp( c, 0.0, 1.0 );
        c = uBlackFloor + c * ( uWhiteCeil - uBlackFloor );
        return c;
      }

      void main() {
        float d = texture2D( tDepth, vUv ).x;
        float vz = linearZ( d, uNear, uFar );

        // ---- focus: nearest of a small cross at the screen centre (the aim point) ------
        float fz = uFocusOverride;
        if ( fz <= 0.0 ) {
          float c0 = texture2D( tDepth, vec2( 0.5, 0.5 ) ).x;
          float c1 = texture2D( tDepth, vec2( 0.5, 0.5 ) + vec2( 6.0, 0.0 ) * uTexel ).x;
          float c2 = texture2D( tDepth, vec2( 0.5, 0.5 ) - vec2( 6.0, 0.0 ) * uTexel ).x;
          float c3 = texture2D( tDepth, vec2( 0.5, 0.5 ) + vec2( 0.0, 6.0 ) * uTexel ).x;
          float c4 = texture2D( tDepth, vec2( 0.5, 0.5 ) - vec2( 0.0, 6.0 ) * uTexel ).x;
          float cd = min( min( min( c0, c1 ), min( c2, c3 ) ), c4 );
          fz = linearZ( cd, uNear, uFar );
        }
        fz = max( fz + uFocusBias, 0.30 );

        // ---- circle of confusion: shallow, and the foreground defocuses harder ---------
        // The BACKGROUND term is deliberately tiny. A hip-fire FPS focuses on the aim
        // point ~20 m out; if the far field carried the same CoC as the foreground the
        // entire skyline would be defocused and no building edge would ever resolve.
        // Far blur only exists to take the aliasing edge off the deep distance.
        float coc = uDofStrength * ( 1.0 - fz / max( vz, 0.05 ) );
        coc *= ( coc < 0.0 ) ? uDofNear : uDofFar;
        float radiusPx = min( abs( coc ), 1.0 ) * uDofMaxCoC;
        if ( d >= 0.999995 ) radiusPx = uDofStrength > 0.0 ? uDofMaxCoC * uDofFar : 0.0;

        // ---- camera motion vector from depth reprojection ------------------------------
        vec2 vel = vec2( 0.0 );
        if ( uMotionScale > 0.0 ) {
          vec4 wp = uInvVP * vec4( vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
          wp /= wp.w;
          vec4 pp = uPrevVP * vec4( wp.xyz, 1.0 );
          if ( pp.w > 0.0 ) {
            vec2 prevUV = ( pp.xy / pp.w ) * 0.5 + 0.5;
            vel = ( vUv - prevUV ) * uMotionScale;
            float vl = length( vel );
            if ( vl > uMotionMax ) vel *= uMotionMax / vl;
          }
        }

        vec3 col = gather( vUv, radiusPx, vel );

        // ---- bloom + lens dirt ----------------------------------------------------------
        vec3 bloom = texture2D( tBloom, vUv ).rgb;
        float dirt = texture2D( tDirt, vUv * vec2( uAspect, 1.0 ) * 0.44 ).r;
        col += bloom * uBloomStrength * mix( 1.0, 0.78 + dirt * 1.15, uDirtAmount );

        // ---- god rays (Sky's radial scattering buffer) ----------------------------------
        col += texture2D( tGodray, vUv ).rgb * uGodrayColor * uGodrayStrength;

        // ---- viewmodel over the world, still in HDR so it takes the same grade ----------
        vec4 vm = texture2D( tVM, vUv );
        col = mix( col, max( vm.rgb, 0.0 ), clamp( vm.a, 0.0, 1.0 ) * uVMEnabled );

        // ---- vignette, applied in linear so it reads as exposure falloff ----------------
        vec2 vd = ( vUv - 0.5 ) * vec2( uAspect, 1.0 );
        float vr = length( vd ) / length( vec2( uAspect, 1.0 ) * 0.5 );
        float vig = 1.0 - uVignette * pow( smoothstep( uVignetteStart, 1.0, vr ), uVignettePow );
        col *= vig;

        // ---- display transform ----------------------------------------------------------
        // agxToneMap() returns DISPLAY-LINEAR sRGB. The film grade below is a print
        // emulation authored in gamma-encoded display space (pivot 0.415, split-tone
        // thresholds at 0.42/0.44, a black floor in code values) — feeding it linear
        // data crushes everything under the pivot to near-zero and turns every mid-dark
        // subject into a flat silhouette. Encode first, grade second.
        col = max( col, 0.0 );
        vec3 tm = encodeSRGB( agxToneMap( col, uExposure ) );
        gl_FragColor = vec4( filmGrade( tm ), 1.0 );
      }`, {
      tColor: { value: null },
      tDepth: { value: null },
      tBloom: { value: null },
      tGodray: { value: null },
      tVM: { value: null },
      tDirt: { value: null },
      uInvVP: { value: new THREE.Matrix4() },
      uPrevVP: { value: new THREE.Matrix4() },
      uTexel: { value: new THREE.Vector2() },
      uAspect: { value: 1.777 },
      uNear: { value: 0.05 },
      uFar: { value: 900 },
      uBloomStrength: { value: 0.058 },
      uDirtAmount: { value: 0.14 },
      uGodrayColor: { value: new THREE.Color(1.0, 0.86, 0.66) },
      uGodrayStrength: { value: 0.0 },
      uDofStrength: { value: 1.05 },
      uDofMaxCoC: { value: 5.2 },
      uDofNear: { value: 1.35 },
      uDofFar: { value: 0.14 },
      uFocusOverride: { value: -1 },
      uFocusBias: { value: 0 },
      uMotionScale: { value: 0.52 },
      uMotionMax: { value: 0.016 },
      uVignette: { value: 0.34 },
      uVignetteStart: { value: 0.30 },
      uVignettePow: { value: 1.55 },
      uVMEnabled: { value: 0 },
      uExposure: { value: 1.0 },
      uLift: { value: new THREE.Vector3() },
      uGain: { value: new THREE.Vector3(1, 1, 1) },
      uGamma: { value: new THREE.Vector3(1, 1, 1) },
      uContrast: { value: 1.115 },
      uPivot: { value: 0.415 },
      uSoftS: { value: 0.20 },
      uShadowTint: { value: new THREE.Vector3() },
      uHighTint: { value: new THREE.Vector3() },
      uSplit: { value: 1 },
      uShadowSat: { value: 0.70 },
      uHighSat: { value: 0.90 },
      uSaturation: { value: 1.07 },
      uHighWarm: { value: new THREE.Vector3(1, 1, 1) },
      uBlackFloor: { value: new THREE.Vector3() },
      uWhiteCeil: { value: new THREE.Vector3(1, 1, 1) },
    }, { TAPS: dofTaps });

    // ====================================================================== PRESENT
    this.presentMat = this._mat('PostFX.Present', /* glsl */`
      uniform sampler2D tDiffuse;
      uniform vec2 uTexel, uResolution;
      uniform float uChromatic, uChromaPow, uSharpen, uGrain, uGrainSize, uTime, uAspect;
      varying vec2 vUv;
      ${GLSL_COMMON}

      void main() {
        vec2 d = ( vUv - 0.5 ) * vec2( uAspect, 1.0 );
        float r = length( d ) / length( vec2( uAspect, 1.0 ) * 0.5 );
        float caAmt = pow( clamp( r, 0.0, 1.0 ), uChromaPow ) * uChromatic;
        vec2 caDir = ( r > 1e-5 ) ? normalize( vUv - 0.5 ) : vec2( 0.0 );
        vec2 caOff = caDir * caAmt * uTexel;

        // radial chromatic aberration — zero in the centre, only bites at the corners
        vec3 c;
        c.r = texture2D( tDiffuse, vUv + caOff ).r;
        c.g = texture2D( tDiffuse, vUv ).g;
        c.b = texture2D( tDiffuse, vUv - caOff ).b;

        // contrast-adaptive sharpen with a ringing guard
        vec3 n0 = texture2D( tDiffuse, vUv + vec2( 0.0, uTexel.y ) ).rgb;
        vec3 n1 = texture2D( tDiffuse, vUv - vec2( 0.0, uTexel.y ) ).rgb;
        vec3 n2 = texture2D( tDiffuse, vUv + vec2( uTexel.x, 0.0 ) ).rgb;
        vec3 n3 = texture2D( tDiffuse, vUv - vec2( uTexel.x, 0.0 ) ).rgb;
        vec3 avg = ( n0 + n1 + n2 + n3 ) * 0.25;
        vec3 mn = min( min( n0, n1 ), min( n2, n3 ) );
        vec3 mx = max( max( n0, n1 ), max( n2, n3 ) );
        vec3 sharp = c + ( c - avg ) * uSharpen;
        c = clamp( sharp, mn - 0.14, mx + 0.14 );

        // animated, luminance-weighted, fine film grain (triangular PDF -> no banding)
        float l = luma( c );
        vec2 gp = floor( gl_FragCoord.xy / max( uGrainSize, 0.5 ) );
        float n = hash12( gp + fract( uTime * 71.13 ) * 1024.0 );
        float m = hash12( gp.yx + 17.0 + fract( uTime * 43.71 ) * 1024.0 );
        float g = ( n + m - 1.0 );
        // strongest through the mids, backs off in the blacks and off the highlights
        float w = 1.0 - abs( l * 2.0 - 1.0 );
        c += g * uGrain * ( 0.22 + 0.78 * w * w );

        // 8-bit ordered dither so the graded sky never bands
        c += ( hash12( gl_FragCoord.xy * 1.7 ) - 0.5 ) * ( 1.0 / 255.0 );

        gl_FragColor = vec4( clamp( c, 0.0, 1.0 ), 1.0 );
      }`, {
      tDiffuse: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uResolution: { value: new THREE.Vector2() },
      uChromatic: { value: 1.35 },
      uChromaPow: { value: 2.6 },
      uSharpen: { value: 0.34 },
      uGrain: { value: 0.021 },
      uGrainSize: { value: 1.0 },
      uTime: { value: 0 },
      uAspect: { value: 1.777 },
    });

    // Straight blit, used when a stage is disabled.
    this.copyMat = this._mat('PostFX.Copy', /* glsl */`
      uniform sampler2D tDiffuse;
      varying vec2 vUv;
      void main() { gl_FragColor = texture2D( tDiffuse, vUv ); }`, {
      tDiffuse: { value: null },
    });
  }

  // =======================================================================================
  //  INIT
  // =======================================================================================
  async init() {
    const r = this.g.renderer?.renderer;
    if (!r) { this._failed = true; console.warn('[PostFX] no renderer — post disabled'); return; }

    r.getDrawingBufferSize(this._size);
    this._allocTargets(this._size.x, this._size.y);
    this._applyParams();

    // Warm every program so the first presented frame does not hitch on a compile.
    try { this._warm(r); } catch (e) { /* non-fatal */ }
  }

  _warm(r) {
    const prev = r.getRenderTarget();
    const mats = [
      [this.aoMat, this.aoRT], [this.aoBlurMat, this.aoBlurRT],
      [this.taaMat, this.taaA], [this.bloomPreMat, this.bloomMips[0]],
      [this.bloomDownMat, this.bloomMips[1] || this.bloomMips[0]],
      [this.compositeMat, this.ldrRT], [this.presentMat, this.aaRT],
      [this.copyMat, this.aaRT],
    ];
    this.aoMat.uniforms.tDepth.value = this._depthTex;
    this.aoBlurMat.uniforms.tAO.value = this.aoRT.texture;
    this.aoBlurMat.uniforms.tDepth.value = this._depthTex;
    this.taaMat.uniforms.tCurrent.value = this.sceneRT.texture;
    this.taaMat.uniforms.tHistory.value = this.taaB.texture;
    this.taaMat.uniforms.tDepth.value = this._depthTex;
    this.taaMat.uniforms.tAO.value = this._whiteTex;
    this.bloomPreMat.uniforms.tSrc.value = this.taaA.texture;
    this.bloomDownMat.uniforms.tSrc.value = this.bloomMips[0].texture;
    this.compositeMat.uniforms.tColor.value = this.taaA.texture;
    this.compositeMat.uniforms.tDepth.value = this._depthTex;
    this.compositeMat.uniforms.tBloom.value = this.bloomMips[0].texture;
    this.compositeMat.uniforms.tGodray.value = this._blackTex;
    this.compositeMat.uniforms.tVM.value = this._blackTex;
    this.compositeMat.uniforms.tDirt.value = this._dirtTex;
    this.presentMat.uniforms.tDiffuse.value = this.ldrRT.texture;
    this.copyMat.uniforms.tDiffuse.value = this.ldrRT.texture;

    for (const [m, rt] of mats) {
      if (!m || !rt) continue;
      this._quad.material = m;
      r.setRenderTarget(rt);
      this._quad.render(r);
    }
    r.setRenderTarget(prev);
  }

  // =======================================================================================
  //  PARAMETERS
  // =======================================================================================
  set(name, value) {
    if (!(name in this.params)) return this;
    this.params[name] = value;
    this._applyParams();
    return this;
  }

  setEnabled(flag, on) {
    if (flag in this.flags) this.flags[flag] = !!on;
    if (this.cfg?.post && flag in this.cfg.post) this.cfg.post[flag] = !!on;
    this._applyParams();
    return this;
  }

  /** metres, or null/<=0 for automatic focus on the screen centre. */
  setFocus(metres) {
    this._focusOverride = (typeof metres === 'number' && metres > 0) ? metres : -1;
    if (this.compositeMat) this.compositeMat.uniforms.uFocusOverride.value = this._focusOverride;
    return this;
  }

  _v3(u, arr) { u.set(arr[0], arr[1], arr[2]); }

  _applyParams() {
    const p = this.params;
    const f = this.flags;
    if (!this.aoMat) return;

    const ao = this.aoMat.uniforms;
    ao.uRadius.value = p.aoRadius;
    ao.uIntensity.value = p.aoIntensity;
    ao.uBias.value = p.aoBias;
    ao.uMaxRadius.value = p.aoMaxRadiusPx;

    this.aoBlurMat.uniforms.uSigma.value = p.aoDepthSigma;

    const t = this.taaMat.uniforms;
    t.uClipGamma.value = p.taaClipGamma;
    t.uAOStrength.value = f.ssao ? p.aoStrength : 0.0;

    const bp = this.bloomPreMat.uniforms;
    bp.uThreshold.value = p.bloomThreshold;
    bp.uKnee.value = p.bloomKnee;
    bp.uClampMax.value = p.bloomClamp;
    this.bloomUpMat.uniforms.uScatter.value = p.bloomScatter;

    const c = this.compositeMat.uniforms;
    c.uBloomStrength.value = f.bloom ? p.bloomStrength : 0.0;
    c.uDirtAmount.value = f.lensDirt ? p.lensDirt : 0.0;
    c.uDofStrength.value = f.dof ? p.dofStrength : 0.0;
    c.uDofMaxCoC.value = p.dofMaxCoC;
    c.uDofNear.value = p.dofNearScale;
    c.uDofFar.value = p.dofFarScale;
    c.uFocusBias.value = p.dofFocusBias;
    c.uMotionScale.value = f.motionBlur ? p.motionScale : 0.0;
    c.uMotionMax.value = p.motionMax;
    c.uVignette.value = f.vignette ? p.vignette : 0.0;
    c.uVignetteStart.value = p.vignetteStart;
    c.uVignettePow.value = p.vignettePow;
    c.uContrast.value = p.contrast;
    c.uPivot.value = p.contrastPivot;
    c.uSoftS.value = p.softS;
    c.uSplit.value = p.splitAmount;
    c.uShadowSat.value = p.shadowSat;
    c.uHighSat.value = p.highSat;
    c.uSaturation.value = p.saturation;
    this._v3(c.uLift.value, p.lift);
    this._v3(c.uGain.value, p.gain);
    this._v3(c.uGamma.value, p.gamma);
    this._v3(c.uShadowTint.value, p.shadowTint);
    this._v3(c.uHighTint.value, p.highTint);
    this._v3(c.uHighWarm.value, p.highWarm);
    this._v3(c.uBlackFloor.value, p.blackFloor);
    this._v3(c.uWhiteCeil.value, p.whiteCeil);

    const pr = this.presentMat.uniforms;
    pr.uChromatic.value = f.chromatic ? p.chromatic : 0.0;
    pr.uChromaPow.value = p.chromaPow;
    pr.uSharpen.value = f.sharpen ? p.sharpen : 0.0;
    pr.uGrain.value = f.grain ? p.grain : 0.0;
    pr.uGrainSize.value = p.grainSize;

    this._syncStaticUniforms();
  }

  _syncStaticUniforms() {
    if (!this.sceneRT || !this.aoMat) return;
    const W = this.width, H = this.height;
    const aspect = W / Math.max(1, H);

    this.aoMat.uniforms.uAoTexel.value.set(2 / W, 2 / H);
    this.aoMat.uniforms.uFullTexel.value.set(1 / W, 1 / H);
    this.aoMat.uniforms.uAspect.value = aspect;
    this.aoBlurMat.uniforms.uAoTexel.value.set(2 / W, 2 / H);
    this.taaMat.uniforms.uTexel.value.set(1 / W, 1 / H);
    this.bloomPreMat.uniforms.uTexel.value.set(1 / W, 1 / H);
    this.compositeMat.uniforms.uTexel.value.set(1 / W, 1 / H);
    this.compositeMat.uniforms.uAspect.value = aspect;
    this.presentMat.uniforms.uTexel.value.set(1 / W, 1 / H);
    this.presentMat.uniforms.uResolution.value.set(W, H);
    this.presentMat.uniforms.uAspect.value = aspect;

    // DOF / motion-blur radii are authored at 1080p; scale so the look is resolution-independent.
    const s = H / 1080;
    this.compositeMat.uniforms.uDofMaxCoC.value = this.params.dofMaxCoC * Math.max(0.5, s);
    this.presentMat.uniforms.uChromatic.value =
      (this.flags.chromatic ? this.params.chromatic : 0) * Math.max(0.5, s);
    this.presentMat.uniforms.uGrainSize.value = this.params.grainSize * Math.max(1, Math.round(s));
  }

  // =======================================================================================
  //  FRAME UPDATE (runs before render, with every other module)
  // =======================================================================================
  update(dt, t) {
    this._time = t;
    if (this._failed) return;

    // Collaborators built after us — lazily, guarded, never in the constructor.
    const weapon = this.g.weapon;
    const ads = !!(weapon?.ads ?? weapon?.isAds ?? weapon?.adsActive);
    if (ads !== this._ads) {
      this._ads = ads;
      // Aiming tightens the depth of field and calms the lens a touch.
      const c = this.compositeMat?.uniforms;
      if (c) {
        this._dofTarget = this.flags.dof ? (ads ? this.params.dofStrength * 1.25 : this.params.dofStrength) : 0;
      }
    }
    const c = this.compositeMat?.uniforms;
    if (c && this._dofTarget !== undefined) {
      c.uDofStrength.value += (this._dofTarget - c.uDofStrength.value) * Math.min(1, dt * 7.5);
    }

    if (this.presentMat) this.presentMat.uniforms.uTime.value = t;
  }

  // =======================================================================================
  //  RENDER — the final present
  // =======================================================================================
  render(dt) {
    const r = this.g.renderer?.renderer;
    if (!r) return;

    if (this._failed || !this.enabled) {
      // Hand the viewmodel overlay back to Weapon — it self-renders from
      // scene.onAfterRender when nobody else is driving the pass.
      if (this.g.weapon && this.g.weapon.autoRender === false) this.g.weapon.autoRender = true;
      r.setRenderTarget(null);
      r.render(this.g.scene, this.g.camera);
      return;
    }

    // We own the viewmodel pass (step 5 renders it into vmRT and composites it in HDR).
    // Tell Weapon so, or it also draws itself from scene.onAfterRender — an extra full
    // viewmodel render every frame, into whichever offscreen target happens to be bound.
    const wpn = this.g.weapon;
    if (wpn?.viewmodelScene && wpn.autoRender !== false) wpn.autoRender = false;

    try {
      this._render(r, dt);
    } catch (e) {
      this._failed = true;
      console.error('[PostFX] pipeline failed — falling back to a direct render', e);
      if (wpn) wpn.autoRender = true;
      try {
        r.setRenderTarget(null);
        r.render(this.g.scene, this.g.camera);
      } catch (e2) { /* nothing left to do */ }
    }
  }

  _render(r, dt) {
    const scene = this.g.scene;
    const cam = this.g.camera;
    if (!scene || !cam) return;

    // ---- size ---------------------------------------------------------------------------
    r.getDrawingBufferSize(this._size);
    if (this._size.x !== this.width || this._size.y !== this.height) {
      this._allocTargets(this._size.x, this._size.y);
      this._applyParams();
    }
    if (!this.sceneRT) return;

    this._frame++;
    const W = this.width, H = this.height;

    const prevAutoClear = r.autoClear;
    r.autoClear = true;

    // ---- 0. Sky god-ray prepass (zero latency: same camera, same frame) ------------------
    let godTex = this._blackTex;
    let godStrength = 0;
    const sky = this.g.sky;
    if (this.flags.godrays && sky?.renderGodrays) {
      try {
        const tex = sky.renderGodrays(r);
        if (tex) {
          godTex = tex;
          godStrength = (sky.godrayIntensity || 0) * this.params.godrayStrength;
          if (sky.godrayColor) this._godrayTint.copy(sky.godrayColor);
        }
      } catch (e) { this.flags.godrays = false; }
    }

    // ---- 1. Scene render, sub-pixel jittered --------------------------------------------
    const useTAA = this.flags.taa;
    let jx = 0, jy = 0;
    if (useTAA) {
      this._jitterIndex = (this._jitterIndex + 1) % 16;
      jx = JITTER[this._jitterIndex * 2];
      jy = JITTER[this._jitterIndex * 2 + 1];
    }

    const pe = cam.projectionMatrix.elements;
    this._savedE8 = pe[8];
    this._savedE9 = pe[9];
    this._savedProjInv.copy(cam.projectionMatrixInverse);
    if (useTAA) {
      pe[8] = this._savedE8 + (jx * 2) / W;
      pe[9] = this._savedE9 + (jy * 2) / H;
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    }
    this._projInv.copy(cam.projectionMatrixInverse);

    cam.updateMatrixWorld();
    this._curVP.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this._curVPInv.copy(this._curVP).invert();

    r.setRenderTarget(this.sceneRT);
    r.render(scene, cam);

    // restore the camera immediately — nobody outside this function ever sees the jitter
    pe[8] = this._savedE8;
    pe[9] = this._savedE9;
    cam.projectionMatrixInverse.copy(this._savedProjInv);

    // ---- 2. Ambient occlusion ------------------------------------------------------------
    let aoTex = this._whiteTex;
    if (this.flags.ssao) {
      const au = this.aoMat.uniforms;
      au.tDepth.value = this._depthTex;
      au.uProjInv.value.copy(this._projInv);
      au.uProjScale.value = 0.5 * cam.projectionMatrix.elements[5];
      au.uSeed.value = (this._frame % 64) * 1.618;
      this._blit(r, this.aoMat, this.aoRT);

      const bu = this.aoBlurMat.uniforms;
      bu.tAO.value = this.aoRT.texture;
      bu.tDepth.value = this._depthTex;
      bu.uNear.value = cam.near;
      bu.uFar.value = cam.far;
      this._blit(r, this.aoBlurMat, this.aoBlurRT);
      aoTex = this.aoBlurRT.texture;
    }

    // ---- 3. TAA resolve (also applies AO) ------------------------------------------------
    const tu = this.taaMat.uniforms;
    tu.tCurrent.value = this.sceneRT.texture;
    tu.tHistory.value = this._taaRead.texture;
    tu.tDepth.value = this._depthTex;
    tu.tAO.value = aoTex;
    tu.uInvVP.value.copy(this._curVPInv);
    tu.uPrevVP.value.copy(this._prevVP);

    // Converge fast while still, and hand authority back to the current frame the moment
    // the camera whips — a long history on a fast pan is exactly what ghosting looks like.
    const motion = this._cameraMotion(dt);
    const blend = useTAA
      ? THREE.MathUtils.clamp(
          this.params.taaBlend + (this.params.taaBlendMoving - this.params.taaBlend) *
          THREE.MathUtils.clamp(motion * this.params.taaMotionScale, 0, 1), 0.02, 1)
      : 1.0;
    tu.uBlend.value = blend;
    tu.uHistoryValid.value = (useTAA && this._historyValid) ? 1 : 0;
    this._blit(r, this.taaMat, this._taaWrite);

    const resolved = this._taaWrite;
    // swap history
    const tmp = this._taaRead;
    this._taaRead = this._taaWrite;
    this._taaWrite = tmp;
    this._historyValid = true;
    this._taaFrames++;

    // ---- 4. Bloom pyramid -----------------------------------------------------------------
    let bloomTex = this._blackTex;
    if (this.flags.bloom && this.bloomMips && this.bloomMips.length >= 2) {
      const exposure = (this.cfg.render?.exposure ?? 1) * this.params.exposure;
      const pu = this.bloomPreMat.uniforms;
      pu.tSrc.value = resolved.texture;
      pu.uTexel.value.set(1 / W, 1 / H);
      pu.uExposure.value = exposure;
      this._blit(r, this.bloomPreMat, this.bloomMips[0]);

      for (let i = 1; i < this.bloomMips.length; i++) {
        const src = this.bloomMips[i - 1];
        this.bloomDownMat.uniforms.tSrc.value = src.texture;
        this.bloomDownMat.uniforms.uTexel.value.set(1 / src.width, 1 / src.height);
        this._blit(r, this.bloomDownMat, this.bloomMips[i]);
      }

      // additive tent upsample back down the chain
      r.autoClear = false;
      for (let i = this.bloomMips.length - 1; i >= 1; i--) {
        const src = this.bloomMips[i];
        this.bloomUpMat.uniforms.tSrc.value = src.texture;
        this.bloomUpMat.uniforms.uTexel.value.set(1 / src.width, 1 / src.height);
        this._quad.material = this.bloomUpMat;
        r.setRenderTarget(this.bloomMips[i - 1]);
        this._quad.render(r);
      }
      r.autoClear = true;
      bloomTex = this.bloomMips[0].texture;
    }

    // ---- 5. Viewmodel ----------------------------------------------------------------------
    let vmTex = this._blackTex;
    let vmOn = 0;
    const weapon = this.g.weapon;
    const vmScene = weapon?.viewmodelScene;
    if (vmScene && this.vmRT) {
      let vmCam = weapon.viewmodelCamera;
      if (!vmCam) {
        if (!this._vmCam) {
          this._vmCam = new THREE.PerspectiveCamera(
            this.cfg.weapon?.viewmodelFov ?? 55, W / H, 0.01, 12);
        }
        vmCam = this._vmCam;
        vmCam.position.set(0, 0, 0);
        vmCam.quaternion.identity();
      }
      if (vmCam.isPerspectiveCamera) {
        const a = W / Math.max(1, H);
        if (Math.abs(vmCam.aspect - a) > 1e-4) { vmCam.aspect = a; vmCam.updateProjectionMatrix(); }
      }
      const cc = r.getClearColor(this._clearColor);
      const ca = r.getClearAlpha();
      r.setClearColor(0x000000, 0);
      r.setRenderTarget(this.vmRT);
      r.render(vmScene, vmCam);
      r.setClearColor(cc, ca);
      vmTex = this.vmRT.texture;
      vmOn = 1;
    }

    // ---- 6. Composite ---------------------------------------------------------------------
    const cu = this.compositeMat.uniforms;
    cu.tColor.value = resolved.texture;
    cu.tDepth.value = this._depthTex;
    cu.tBloom.value = bloomTex;
    cu.tGodray.value = godTex;
    cu.tVM.value = vmTex;
    cu.tDirt.value = this._dirtTex;
    cu.uVMEnabled.value = vmOn;
    cu.uNear.value = cam.near;
    cu.uFar.value = cam.far;
    cu.uInvVP.value.copy(this._curVPInv);
    cu.uPrevVP.value.copy(this._prevVP);
    cu.uGodrayColor.value.copy(this._godrayTint);
    cu.uGodrayStrength.value = godStrength;
    cu.uExposure.value = (this.cfg.render?.exposure ?? 1) * this.params.exposure;
    this._blit(r, this.compositeMat, this.ldrRT);

    // ---- 7. SMAA ----------------------------------------------------------------------------
    let aaSource = this.ldrRT;
    if (this.flags.smaa && this._smaa) {
      this._smaa.renderToScreen = false;
      this._smaa.render(r, this.aaRT, this.ldrRT, dt, false);
      aaSource = this.aaRT;
    }

    // ---- 8. Present -------------------------------------------------------------------------
    this.presentMat.uniforms.tDiffuse.value = aaSource.texture;
    this.presentMat.uniforms.uTime.value = this._time;
    this._quad.material = this.presentMat;
    r.setRenderTarget(null);
    this._quad.render(r);

    // ---- bookkeeping -------------------------------------------------------------------------
    this._prevVP.copy(this._curVP);
    r.autoClear = prevAutoClear;
  }

  _blit(r, material, target) {
    this._quad.material = material;
    r.setRenderTarget(target);
    this._quad.render(r);
  }

  // Rough screen-space camera motion in uv/second — used to bias the TAA blend.
  _cameraMotion(dt) {
    const cam = this.g.camera;
    if (!cam) return 0;
    if (!this._prevCamPos) {
      this._prevCamPos = new THREE.Vector3().copy(cam.position);
      this._prevCamQuat = new THREE.Quaternion().copy(cam.quaternion);
      return 0;
    }
    const dp = this._prevCamPos.distanceTo(cam.position);
    const dq = 1 - Math.min(1, Math.abs(this._prevCamQuat.dot(cam.quaternion)));
    this._prevCamPos.copy(cam.position);
    this._prevCamQuat.copy(cam.quaternion);
    const idt = 1 / Math.max(1e-3, dt || 0.016);
    // rotation dominates apparent screen motion in an FPS
    return (dq * 26.0 + dp * 0.06) * idt;
  }

  // =======================================================================================
  //  RESIZE / LIFECYCLE
  // =======================================================================================
  setPixelRatio(pr) {
    this._pixelRatio = pr;
    const r = this.g.renderer?.renderer;
    if (!r) return;
    r.getDrawingBufferSize(this._size);
    this._allocTargets(this._size.x, this._size.y);
    this._applyParams();
  }

  resize(w, h) {
    const r = this.g.renderer?.renderer;
    if (!r || this._failed) return;
    r.getDrawingBufferSize(this._size);
    this._allocTargets(this._size.x, this._size.y);
    this._applyParams();
  }

  getStats() {
    let bytes = 0;
    for (const t of this._targets) {
      const bpp = t.texture.type === THREE.HalfFloatType ? 8 : 4;
      bytes += t.width * t.height * bpp;
    }
    return {
      width: this.width,
      height: this.height,
      targets: this._targets.length,
      mb: +(bytes / 1048576).toFixed(1),
      taaFrames: this._taaFrames,
      bloomMips: this.bloomMips ? this.bloomMips.length : 0,
      failed: this._failed,
      flags: this.flags,
    };
  }

  get sceneTexture() { return this.sceneRT?.texture || null; }
  get depthTexture() { return this._depthTex || null; }

  dispose() {
    this._disposeTargets();
    for (const m of this._materials) m.dispose();
    this._materials.length = 0;
    this._quad?.dispose?.();
    this._smaa?.dispose?.();
    this._blackTex?.dispose();
    this._whiteTex?.dispose();
    this._dirtTex?.dispose();
  }
}

export default PostFX;
