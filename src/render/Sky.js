// BLACKSITE — Sky.js
// Owner: atmosphere agent. Owns: physical sky dome + sun disc, sun placement & colour,
// PMREM image-based lighting, height-based volumetric fog (global shader injection),
// drifting fBm cloud layer, screen-space god rays, airborne dust motes, ground heat haze.
//
// Art direction: late afternoon (~16:20) golden hour over a dusty Middle-Eastern compound.
// Warm low key raking from the west (#ffd9a0), cool skylight fill (#7fa6d9), heavy
// atmospheric perspective, warm dust pooling low, never a pure black, never a clipped white.
//
// ---------------------------------------------------------------------------------------
// PUBLIC API (collaborators — read this)
// ---------------------------------------------------------------------------------------
//   game.sky.sunDirection      THREE.Vector3  unit vector FROM the world toward the sun (live ref)
//   game.sky.sunWorldPosition  THREE.Vector3  sunDirection * SUN_DISTANCE (live ref)
//   game.sky.sunElevation      radians above the horizon
//   game.sky.sunAzimuth        radians, 0 = north (-Z), increasing toward east (+X)
//   game.sky.sunColor          THREE.Color    graded key-light colour (apply to your light)
//   game.sky.skyColor          THREE.Color    zenith / skylight fill colour (hemi light sky)
//   game.sky.groundColor       THREE.Color    warm sand bounce colour (hemi light ground)
//   game.sky.fogSunColor       THREE.Color    haze colour looking into the sun
//   game.sky.fogSkyColor       THREE.Color    haze colour looking away from the sun
//   game.sky.envMap            THREE.Texture  PMREM cube-uv env (already on scene.environment)
//   game.sky.envIntensity      float          scene.environmentIntensity Sky applied (0.45)
//   game.sky.setEnvIntensity(v)               retune the IBL live, no rebake
//
//   game.sky.getSunScreenPosition(camera) -> THREE.Vector2  (0..1 uv, allocation-free, cached)
//   game.sky.sunScreenPosition  THREE.Vector2 same object, refreshed every frame
//   game.sky.sunVisible         bool   sun is in front of the camera and inside the frame margin
//   game.sky.sunOcclusion       0..1   smoothed estimate of how blocked the sun is (0 = clear)
//   game.sky.godrayTexture      THREE.Texture  radial scattering buffer, ADD this over the frame
//   game.sky.godrayIntensity    float  suggested composite scale (already faded in/out)
//   game.sky.godrayColor        THREE.Color    suggested tint for the composite
//   game.sky.renderGodrays(renderer)  render the occlusion + radial-blur passes NOW.
//        PostFX: call this at the top of your render() (before you composite) to get a
//        zero-latency buffer. If nobody calls it, Sky self-drives it in update() with one
//        frame of latency. Calling it once per frame is enough; extra calls are no-ops.
//   game.sky.shimmerTexture     THREE.Texture  tiling animated flow-noise for heat shimmer
//   game.sky.shimmer            { strength, scale, speed, horizon }  suggested distortion params
//   game.sky.getShimmerUniforms()  -> object with { tShimmer, uTime, uStrength, uScale, uHorizon }
//
//   game.sky.setTimeOfDay(hours)   re-solve the sun, rebake fog + IBL (authoring / debug path)
//   game.sky.setFogDensity(d)      live, no recompile
//
// NOTE (fog): Sky installs a custom height-fog into THREE.ShaderChunk (fog_vertex /
// fog_pars_vertex / fog_fragment / fog_pars_fragment) at construction time, BEFORE any
// program is compiled. Every lit material in the scene therefore gets warm, height-stratified
// volumetric haze for free — you do not need to do anything. scene.fog is a FogExp2 whose
// .color and .density stay live (no recompile). Sun direction and the warm/cool tints are
// baked as literals; setTimeOfDay() rebakes them and forces the affected programs to rebuild.
// ---------------------------------------------------------------------------------------

import * as THREE from 'three';
import { Sky as PreethamSky } from 'three/addons/objects/Sky.js';

const DEG = Math.PI / 180;

// --- site / celestial constants (no Config keys exist for these) -------------------------
const SITE_LATITUDE = 34.5 * DEG;      // northern Middle East
// Equinoctial declination puts 16:24 at ~18.4 deg of elevation, azimuth ~254 (WSW).
// The previous late-autumn value (-14.2) solved to 10.7 deg, and at 10.7 deg the west
// range of the compound throws a shadow ~53 m long — the entire courtyard, every street
// and every actor sat in full shade and the warm key never touched a single surface the
// player could see. At 18.4 deg the shadows are still long and raking (~30 m off a 10 m
// parapet) but half the courtyard catches the key, which is the light-to-dark
// composition the art direction is built on.
const SOLAR_DECLINATION = -2.0 * DEG;
const MIN_ELEVATION = 2.2 * DEG;       // never let the model go night-black
const SUN_DISTANCE = 4000;

// --- atmosphere tuning -------------------------------------------------------------------
const FOG_HEIGHT_FALLOFF = 0.0465;     // 1/m — e-fold of the dust layer (~21.5 m)
const FOG_BASE_HEIGHT = -0.6;          // world Y at which the fog reaches full density
const FOG_MAX_DEPTH_BELOW = 70.0;      // clamp for the analytic integral (numerical safety)

// --- art-direction anchors (the grade every module matches) ------------------------------
const KEY_TARGET = 0xffd9a0;           // warm low sun
const FILL_TARGET = 0x7fa6d9;          // cool skylight
const BOUNCE_TARGET = 0x8a6a48;        // warm sand bounce
const HAZE_WARM_TARGET = 0xffc691;     // haze looking into the sun
const HAZE_COOL_TARGET = 0xa8b6c4;     // haze looking away from the sun (dust, not sky)

const CLOUD_TEX_SIZE = 512;
const SHIMMER_TEX_SIZE = 128;

// --- background layers -------------------------------------------------------------------
// The playable compound is ~100 m across and the ground mesh stops at 190 m. Without
// anything past that the frame terminates in a hard world edge. Three silhouette shells,
// each hazed harder than the last, give the mid-ground something to sit in front of.
const BG_CITY_NEAR = 265;      // inner block ring
const BG_CITY_FAR = 440;       // outer block ring
const BG_RIDGE_R = 720;        // distant hills (inside camera.far = 900)
// Long-range extinction. Guarantees anything past ~150 m has dissolved into the dust so the
// edge of the ground mesh is never a visible line. e-fold distance, metres.
const FAR_HAZE_DIST = 116;

// =========================================================================================
// Tileable value-noise helpers (module scope, used only at build time)
// =========================================================================================
function ihash(x, y, s) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(s | 0, 362437);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function tileValueNoise(u, v, period, seed) {
  const fx = u * period, fy = v * period;
  const ix = Math.floor(fx), iy = Math.floor(fy);
  const tx = fx - ix, ty = fy - iy;
  const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
  const x0 = ((ix % period) + period) % period, x1 = (x0 + 1) % period;
  const y0 = ((iy % period) + period) % period, y1 = (y0 + 1) % period;
  const n00 = ihash(x0, y0, seed), n10 = ihash(x1, y0, seed);
  const n01 = ihash(x0, y1, seed), n11 = ihash(x1, y1, seed);
  const a = n00 + (n10 - n00) * sx;
  const b = n01 + (n11 - n01) * sx;
  return a + (b - a) * sy;
}

function tileFbm(u, v, basePeriod, octaves, gain, seed) {
  let amp = 1, sum = 0, norm = 0, p = basePeriod;
  for (let o = 0; o < octaves; o++) {
    sum += tileValueNoise(u, v, p, seed + o * 131) * amp;
    norm += amp;
    amp *= gain;
    p *= 2;
  }
  return sum / norm;
}

// billow/ridged variant — gives clouds their cauliflower silhouette
function tileBillow(u, v, basePeriod, octaves, gain, seed) {
  let amp = 1, sum = 0, norm = 0, p = basePeriod;
  for (let o = 0; o < octaves; o++) {
    sum += Math.abs(tileValueNoise(u, v, p, seed + o * 977) * 2 - 1) * amp;
    norm += amp;
    amp *= gain;
    p *= 2;
  }
  return sum / norm;
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const lum = (c) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;

// =========================================================================================
export class Sky {
  constructor(game) {
    this.g = game;
    const cfg = game.config;
    this.cfg = cfg;

    // ---- live, publicly-read state ------------------------------------------------------
    this.sunDirection = new THREE.Vector3(0, 1, 0);
    this.sunWorldPosition = new THREE.Vector3();
    this.sunElevation = 0;
    this.sunAzimuth = 0;

    this.sunColor = new THREE.Color(KEY_TARGET);
    this.skyColor = new THREE.Color(FILL_TARGET);
    this.groundColor = new THREE.Color(BOUNCE_TARGET);
    this.fogSunColor = new THREE.Color(HAZE_WARM_TARGET);
    this.fogSkyColor = new THREE.Color(HAZE_COOL_TARGET);
    this.hazeMidColor = new THREE.Color();
    this.zenithColor = new THREE.Color();

    // The exact colour every fogged surface resolves to at infinite distance. The sky dome
    // and every background shell evaluate the SAME expression, so the horizon has no seam.
    this.fogBaseColor = new THREE.Color();
    this._fogWarmRel = new THREE.Color(1, 1, 1);
    this._fogCoolRel = new THREE.Color(1, 1, 1);
    this._fogGlow = new THREE.Color(0, 0, 0);

    this.envMap = null;
    // Sky IBL weight. The dome is a bright golden-hour sky, so at 0.45 the image-based
    // ambient alone out-lit the direct sun on every horizontal surface and the frame read
    // as flat overcast. 0.34 keeps coloured bounce in the shadows (never a crushed black)
    // while letting the key own the composition — nudged up from 0.30 to pay back the
    // ambient the (now actually visible) cloud deck takes out of the PMREM bake.
    this.envIntensity = 0.34;
    this.sunVisible = false;
    this.sunOcclusion = 0;
    this.sunScreenPosition = new THREE.Vector2(0.5, 0.5);
    this.godrayTexture = null;
    this.godrayIntensity = 0;
    this.godrayColor = new THREE.Color(1.0, 0.86, 0.66);
    this.sunDiscIntensity = 0.55;

    this.shimmer = { strength: 0.0032, scale: 5.5, speed: 0.85, horizon: 0.52 };
    this.shimmerTexture = null;

    // ---- private scratch (zero allocation in update) -------------------------------------
    this._v3a = new THREE.Vector3();
    this._v3b = new THREE.Vector3();
    this._v3c = new THREE.Vector3();
    this._fwd = new THREE.Vector3();
    this._col = new THREE.Color();
    this._colB = new THREE.Color();
    this._clearColor = new THREE.Color();
    this._targetKey = new THREE.Color(KEY_TARGET);
    this._targetFill = new THREE.Color(FILL_TARGET);
    this._targetBounce = new THREE.Color(BOUNCE_TARGET);
    this._targetHazeWarm = new THREE.Color(HAZE_WARM_TARGET);
    this._targetHazeCool = new THREE.Color(HAZE_COOL_TARGET);

    this._hidden = new Array(512);
    this._hiddenCount = 0;
    this._collect = this._collect.bind(this);
    this._hiddenRefreshFrame = -1e9;

    this._time = 0;
    this._fogRev = 0;
    this._godrayFrame = -1;
    this._externallyDriven = false;
    this._lastExternalFrame = -1e9;
    this._godrayCleared = false;
    this._appliedSunLight = null;
    this._sunLightDistance = 150;
    this._sunOcclusionProbeOk = true;
    this._envDirty = true;
    this._lastEnvDir = new THREE.Vector3(0, 0, 0);

    const q = cfg.quality || 'ultra';
    this._q = q;
    this._godrayEnabled = (cfg.sky?.godRays !== false) && q !== 'low';
    this._dustCount = q === 'low' ? 1100 : q === 'high' ? 2600 : 4200;
    this._godrayDiv = q === 'ultra' ? 4 : 5;

    // ---- build ---------------------------------------------------------------------------
    this._solveSun();
    this._buildNoise();
    this._deriveAtmosphereColors();
    this._installHeightFog();      // MUST happen before anything compiles a program
    this._buildSkyDome();
    this._buildClouds();
    if (!globalThis.__NOBG) this._buildBackdrop();
    this._buildDust();
    this._buildGroundHaze();
    this._buildGodrayResources();
    this._applyToScene();
  }

  // =======================================================================================
  // 1. SUN POSITION — real solar geometry so the disc and the shadows agree exactly
  // =======================================================================================
  _solveSun() {
    const tod = this.cfg.sky?.timeOfDay ?? 16.4;
    const H = (tod - 12) * 15 * DEG;                 // hour angle, +ve = afternoon
    const lat = SITE_LATITUDE, dec = SOLAR_DECLINATION;

    const sinEl = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(H);
    let el = Math.asin(THREE.MathUtils.clamp(sinEl, -1, 1));
    el = Math.max(el, MIN_ELEVATION);

    const denom = Math.max(1e-4, Math.cos(el) * Math.cos(lat));
    const cosAz = THREE.MathUtils.clamp((Math.sin(dec) - Math.sin(el) * Math.sin(lat)) / denom, -1, 1);
    let az = Math.acos(cosAz);                       // 0..PI measured from north
    if (H > 0) az = Math.PI * 2 - az;                // afternoon -> swing west

    this.sunElevation = el;
    this.sunAzimuth = az;

    const ce = Math.cos(el);
    this.sunDirection.set(Math.sin(az) * ce, Math.sin(el), -Math.cos(az) * ce).normalize();
    this.sunWorldPosition.copy(this.sunDirection).multiplyScalar(SUN_DISTANCE);

    if (this.g.registry) this.g.registry.sunDirection = this.sunDirection;
  }

  // =======================================================================================
  // 2. CPU PREETHAM — sample the same analytic model the dome shader runs, so the fog,
  //    the key light and the ambient all agree with the pixels on screen.
  // =======================================================================================
  _skyRadiance(dx, dy, dz, out) {
    const turbidity = this.cfg.sky?.turbidity ?? 6.2;
    const rayleigh = this.cfg.sky?.rayleigh ?? 1.4;
    const mieCoefficient = this.cfg.sky?.mieCoefficient ?? 0.0032;
    const mieG = this.cfg.sky?.mieDirectionalG ?? 0.80;

    const sd = this.sunDirection;

    // sunIntensity()
    const cutoffAngle = 1.6110731556870734, steepness = 1.5, EE = 1000.0;
    const zc = THREE.MathUtils.clamp(sd.y, -1, 1);
    const sunE = EE * Math.max(0, 1 - Math.exp(-((cutoffAngle - Math.acos(zc)) / steepness)));

    // unit sun vector => sunfade == 1 (matches how every three.js example drives this shader)
    const sunfade = 1 - clamp01(1 - Math.exp(sd.y / 450000));
    const rc = rayleigh - (1 - sunfade);

    const bR = [5.804542996261093e-6 * rc, 1.3562911419845635e-5 * rc, 3.0265902468824876e-5 * rc];
    const c = 0.2 * turbidity * 10e-18;
    const mm = 0.434 * c * mieCoefficient;
    const bM = [1.8399918514433978e14 * mm, 2.7798023919660528e14 * mm, 4.0790479543861094e14 * mm];

    const upDot = Math.max(0, dy);
    const zenithAngle = Math.acos(upDot);
    const inv = 1 / (Math.cos(zenithAngle) + 0.15 * Math.pow(93.885 - (zenithAngle * 180) / Math.PI, -1.253));
    const sR = 8.4e3 * inv, sM = 1.25e3 * inv;

    const Fex = [Math.exp(-(bR[0] * sR + bM[0] * sM)), Math.exp(-(bR[1] * sR + bM[1] * sM)), Math.exp(-(bR[2] * sR + bM[2] * sM))];

    const cosTheta = dx * sd.x + dy * sd.y + dz * sd.z;
    const ct = cosTheta * 0.5 + 0.5;
    const rPhase = 0.05968310365946075 * (1 + ct * ct);
    const g2 = mieG * mieG;
    const mPhase = 0.07957747154594767 * ((1 - g2) / Math.pow(Math.max(1e-6, 1 - 2 * mieG * cosTheta + g2), 1.5));

    const mixAmt = clamp01(Math.pow(Math.max(0, 1 - sd.y), 5));
    const res = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const bRT = bR[i] * rPhase, bMT = bM[i] * mPhase;
      const denom = Math.max(1e-30, bR[i] + bM[i]);
      const base = (sunE * (bRT + bMT)) / denom;
      let Lin = Math.pow(Math.max(0, base * (1 - Fex[i])), 1.5);
      const b = Math.pow(Math.max(0, base * Fex[i]), 0.5);
      Lin *= 1 + (b - 1) * mixAmt;
      const L0 = 0.1 * Fex[i];
      const texColor = (Lin + L0) * 0.04 + (i === 0 ? 0 : i === 1 ? 0.0003 : 0.00075);
      res[i] = Math.pow(Math.max(0, texColor), 1 / (1.2 + 1.2 * sunfade));
    }
    out.setRGB(res[0], res[1], res[2]);
    return out;
  }

  // Take hue/chroma from the physical model, then pull it firmly onto the art direction and
  // normalise luminance so the grade never drifts when the tunables move.
  _grade(color, target, mixAmt, targetLum) {
    if (!isFinite(color.r) || !isFinite(color.g) || !isFinite(color.b)) color.copy(target);
    color.r = Math.max(color.r, 0); color.g = Math.max(color.g, 0); color.b = Math.max(color.b, 0);
    const l0 = lum(color);
    if (l0 > 1e-5) color.multiplyScalar(0.5 / l0);   // normalise before mixing so hue dominates
    else color.copy(target);
    color.lerp(target, mixAmt);
    const l = lum(color);
    if (l > 1e-5) color.multiplyScalar(targetLum / l);
    return color;
  }

  _deriveAtmosphereColors() {
    const sd = this.sunDirection;
    const ce = Math.max(0.02, Math.cos(this.sunElevation));

    // haze looking INTO the sun, sampled just above the horizon
    this._skyRadiance(sd.x / ce * 0.995, 0.06, sd.z / ce * 0.995, this._col);
    this._grade(this._col, this._targetHazeWarm, 0.52, 0.50);
    this.fogSunColor.copy(this._col);

    // haze looking AWAY from the sun
    this._skyRadiance(-sd.x / ce * 0.99, 0.10, -sd.z / ce * 0.99, this._col);
    this._grade(this._col, this._targetHazeCool, 0.5, 0.315);
    this.fogSkyColor.copy(this._col);

    // zenith / skylight fill
    this._skyRadiance(0, 1, 0, this._col);
    this._grade(this._col, this._targetFill, 0.45, 0.30);
    this.zenithColor.copy(this._col);
    this.skyColor.copy(this._col);

    // key light: transmitted sunlight (extinction along the sun ray)
    this._skyRadiance(sd.x, Math.max(sd.y, 0.02), sd.z, this._col);
    this._grade(this._col, this._targetKey, 0.62, 0.86);
    this.sunColor.copy(this._col);

    // warm sand bounce for the hemisphere fill
    this.groundColor.copy(this.fogSunColor).lerp(this._targetBounce, 0.72);
    const gl = lum(this.groundColor);
    if (gl > 1e-5) this.groundColor.multiplyScalar(0.20 / gl);

    // the band the sky fades into at the horizon — must equal what the fog resolves to
    this.hazeMidColor.copy(this.fogSkyColor).lerp(this.fogSunColor, 0.38);

    this.godrayColor.copy(this.fogSunColor);
    const gr = lum(this.godrayColor);
    if (gr > 1e-5) this.godrayColor.multiplyScalar(0.72 / gr);

    this._deriveFogTerms();
  }

  // The fog fragment resolves, at infinite optical depth, to
  //     fogColor * mix( coolRel, warmRel, mie ) + glow * pow( sunAmt, 9 )
  // Anything that has to sit seamlessly against fogged geometry — the sky dome below and at
  // the horizon, every background silhouette shell — evaluates exactly this. Derive the
  // terms once, keep them in live Color objects so the shaders can reference them as
  // uniforms and setTimeOfDay() needs no recompile.
  _deriveFogTerms() {
    const w = this.fogSunColor, c = this.fogSkyColor;
    this.fogBaseColor.setRGB((w.r + c.r) * 0.5, (w.g + c.g) * 0.5, (w.b + c.b) * 0.5);
    const bl = Math.max(1e-4, lum(this.fogBaseColor));
    this._fogWarmRel.setRGB(w.r / bl, w.g / bl, w.b / bl);
    this._fogCoolRel.setRGB(c.r / bl, c.g / bl, c.b / bl);
    this._fogGlow.setRGB(w.r * 0.55, w.g * 0.45, w.b * 0.30);
  }

  /** GLSL: the shared horizon-colour function. Everything that meets the fog uses this. */
  static get FOG_RESOLVE_GLSL() {
    return /* glsl */`
      uniform vec3 uFogColor;
      uniform vec3 uFogWarm;
      uniform vec3 uFogCool;
      uniform vec3 uFogGlow;
      vec3 bsFogResolve( float cosT ) {
        // integer powers by multiplication — pow() is a real cost on software rasterisers
        // and this runs full-screen on the dome, the clouds and every background shell
        float s = max( cosT, 0.0 );
        float s2 = s * s;
        float s4 = s2 * s2;
        return uFogColor * mix( uFogCool, uFogWarm, 0.32 + 0.68 * s2 * s ) + uFogGlow * ( s4 * s4 * s );
      }`;
  }

  /** Uniform block matching FOG_RESOLVE_GLSL. Shares the live Color objects. */
  _fogResolveUniforms(extra) {
    const u = {
      uFogColor: { value: this.fogBaseColor },
      uFogWarm: { value: this._fogWarmRel },
      uFogCool: { value: this._fogCoolRel },
      uFogGlow: { value: this._fogGlow },
    };
    if (extra) for (const k in extra) u[k] = extra[k];
    return u;
  }

  // =======================================================================================
  // 3. HEIGHT FOG — global shader-chunk injection.
  //    Analytic exponential-height fog integrated along the view ray, tinted by the angle to
  //    the sun (warm forward-scatter, cool away) with a Mie glow lobe around the sun.
  // =======================================================================================
  _installHeightFog() {
    const sd = this.sunDirection;
    const w = this._fogWarmRel, c = this._fogCoolRel, gv = this._fogGlow;

    const F = (x) => x.toFixed(6);
    const warm = `vec3(${F(w.r)}, ${F(w.g)}, ${F(w.b)})`;
    const cool = `vec3(${F(c.r)}, ${F(c.g)}, ${F(c.b)})`;
    const glow = `vec3(${F(gv.r)}, ${F(gv.g)}, ${F(gv.b)})`;
    const sunv = `vec3(${F(sd.x)}, ${F(sd.y)}, ${F(sd.z)})`;

    THREE.ShaderChunk.fog_pars_vertex = /* glsl */`
#ifdef USE_FOG
  varying float vFogDepth;
  varying vec3 vFogWorldPos;
#endif`;

    // Reconstruct world position from mvPosition only. This is valid in EVERY built-in
    // vertex shader (mesh / skinned / instanced / batched / points / sprites / lines)
    // because they all define mvPosition before including <fog_vertex>, and it avoids
    // depending on `transformed`, which sprite_vert does not define.
    THREE.ShaderChunk.fog_vertex = /* glsl */`
#ifdef USE_FOG
  vFogDepth = - mvPosition.z;
  vFogWorldPos = cameraPosition + vec3(
    dot( viewMatrix[ 0 ].xyz, mvPosition.xyz ),
    dot( viewMatrix[ 1 ].xyz, mvPosition.xyz ),
    dot( viewMatrix[ 2 ].xyz, mvPosition.xyz ) );
#endif`;

    THREE.ShaderChunk.fog_pars_fragment = /* glsl */`
#ifdef USE_FOG
  uniform vec3 fogColor;
  varying float vFogDepth;
  varying vec3 vFogWorldPos;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
#endif`;

    THREE.ShaderChunk.fog_fragment = /* glsl */`
#ifdef USE_FOG

  vec3 fogRay = vFogWorldPos - cameraPosition;
  float fogDist = max( length( fogRay ), 1e-3 );
  vec3 fogDir = fogRay / fogDist;

  #ifdef FOG_EXP2
    float fogD = fogDensity;
  #else
    float fogD = 1.0 / max( fogFar - fogNear, 1.0 );
  #endif

  // analytic integral of exp( -B * h ) along the segment
  float fogB = ${FOG_HEIGHT_FALLOFF.toFixed(6)};
  float fogHC = cameraPosition.y - ${FOG_BASE_HEIGHT.toFixed(4)};
  float fogHE = max( fogHC + fogDir.y * fogDist, -${FOG_MAX_DEPTH_BELOW.toFixed(1)} );
  float fogOptical;
  if ( abs( fogDir.y ) > 0.0016 ) {
    fogOptical = ( exp( -fogB * fogHC ) - exp( -fogB * fogHE ) ) / ( fogB * fogDir.y );
  } else {
    fogOptical = fogDist * exp( -fogB * fogHC );
  }
  fogOptical = max( fogOptical, 0.0 );

  float fogT = fogD * fogOptical;
  float fogFactor = 1.0 - exp( - fogT * fogT );

  // Long-range extinction. The height integral above alone still left ~20% of the far
  // desert showing at the 190 m edge of the ground mesh, which read as a hard straight
  // cut against the sky. This term is negligible inside the compound (<12% at 60 m) and
  // saturates past ~200 m, so the world edge and the background shells both dissolve.
  float fogRange = fogDist * ${(1 / FAR_HAZE_DIST).toFixed(8)};
  float fogFar = 1.0 - exp( - fogRange * fogRange * fogRange );
  fogFactor = max( fogFactor, fogFar * 0.997 );

  float fogSunAmt = max( dot( fogDir, ${sunv} ), 0.0 );
  // Dust is a broad Mie scatterer: the whole horizon is warm and only gets warmer toward
  // the sun. A pure cos^3 lobe left everything off-axis reading as cold blue-grey.
  float fogMie = 0.32 + 0.68 * fogSunAmt * fogSunAmt * fogSunAmt;
  vec3 fogTint = mix( ${cool}, ${warm}, fogMie );
  vec3 fogFinal = fogColor * fogTint;
  fogFinal += ${glow} * pow( fogSunAmt, 9.0 ) * ( 0.30 + 0.70 * fogFactor );

  gl_FragColor.rgb = mix( gl_FragColor.rgb, fogFinal, fogFactor );

#endif`;

    this._fogRev++;
  }

  _rebuildFogPrograms() {
    // Built-in materials cache programs by parameters, not by chunk contents, so bump a
    // define to force a rebuild after a re-bake. Only used by setTimeOfDay().
    const rev = this._fogRev;
    this.g.scene?.traverse((o) => {
      const m = o.material;
      if (!m) return;
      if (Array.isArray(m)) {
        for (let i = 0; i < m.length; i++) {
          if (!m[i]) continue;
          m[i].defines = m[i].defines || {};
          m[i].defines.BLACKSITE_FOG_REV = rev;
          m[i].needsUpdate = true;
        }
      } else {
        m.defines = m.defines || {};
        m.defines.BLACKSITE_FOG_REV = rev;
        m.needsUpdate = true;
      }
    });
  }

  // =======================================================================================
  // 4. PROCEDURAL NOISE — cloud density field + heat-shimmer flow field
  // =======================================================================================
  _buildNoise() {
    const N = CLOUD_TEX_SIZE;
    const data = new Uint8Array(N * N * 4);
    const inv = 1 / N;
    for (let y = 0; y < N; y++) {
      const v = y * inv;
      for (let x = 0; x < N; x++) {
        const u = x * inv;
        // domain warp so the cellular grid of the lattice never reads as a grid
        const wx = u + (tileValueNoise(u, v, 4, 17) - 0.5) * 0.22;
        const wy = v + (tileValueNoise(u, v, 4, 91) - 0.5) * 0.22;
        const coverage = tileFbm(wx, wy, 4, 5, 0.55, 3);
        const detail = tileBillow(u, v, 12, 4, 0.52, 41);
        const macro = tileFbm(u, v, 2, 3, 0.62, 77);
        // WEATHER (alpha): the biggest scale of all — where the deck banks up and where it
        // opens into clear sky. Warped and contrast-expanded, because raw fBm sits in a
        // narrow band around 0.5 and a narrow-band coverage field gives you an even grey
        // mush instead of cloud banks with holes in them.
        const ax = u + (tileValueNoise(u, v, 3, 401) - 0.5) * 0.30;
        const ay = v + (tileValueNoise(u, v, 3, 907) - 0.5) * 0.30;
        const weather = clamp01((tileFbm(ax, ay, 2, 3, 0.5, 613) - 0.5) * 2.2 + 0.5);
        const i = (y * N + x) * 4;
        data[i] = Math.min(255, Math.max(0, coverage * 255)) | 0;
        data[i + 1] = Math.min(255, Math.max(0, (1 - detail) * 255)) | 0;
        data[i + 2] = Math.min(255, Math.max(0, macro * 255)) | 0;
        data[i + 3] = Math.min(255, Math.max(0, weather * 255)) | 0;
      }
    }
    const tex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.generateMipmaps = true;
    tex.anisotropy = Math.min(8, this.cfg.render?.maxAnisotropy ?? 8);
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    this.cloudNoise = tex;

    // --- heat shimmer flow field (two decorrelated gradients + a mask) ------------------
    const S = SHIMMER_TEX_SIZE;
    const sdata = new Uint8Array(S * S * 4);
    const sinv = 1 / S;
    for (let y = 0; y < S; y++) {
      const v = y * sinv;
      for (let x = 0; x < S; x++) {
        const u = x * sinv;
        const a = tileFbm(u, v, 6, 3, 0.5, 211);
        const b = tileFbm(u + 0.37, v + 0.11, 6, 3, 0.5, 733);
        const m = tileFbm(u, v, 3, 2, 0.5, 999);
        const i = (y * S + x) * 4;
        sdata[i] = (a * 255) | 0;
        sdata[i + 1] = (b * 255) | 0;
        sdata[i + 2] = (m * 255) | 0;
        sdata[i + 3] = 255;
      }
    }
    const stex = new THREE.DataTexture(sdata, S, S, THREE.RGBAFormat, THREE.UnsignedByteType);
    stex.wrapS = stex.wrapT = THREE.RepeatWrapping;
    stex.magFilter = THREE.LinearFilter;
    stex.minFilter = THREE.LinearMipmapLinearFilter;
    stex.generateMipmaps = true;
    stex.colorSpace = THREE.NoColorSpace;
    stex.needsUpdate = true;
    this.shimmerTexture = stex;

    // --- soft round dust mote sprite ----------------------------------------------------
    const D = 32;
    const ddata = new Uint8Array(D * D * 4);
    for (let y = 0; y < D; y++) {
      for (let x = 0; x < D; x++) {
        const dx = (x + 0.5) / D - 0.5, dy = (y + 0.5) / D - 0.5;
        const r = Math.sqrt(dx * dx + dy * dy) * 2;
        const a = Math.pow(clamp01(1 - r), 2.2);
        const i = (y * D + x) * 4;
        ddata[i] = ddata[i + 1] = ddata[i + 2] = 255;
        ddata[i + 3] = (a * 255) | 0;
      }
    }
    const dtex = new THREE.DataTexture(ddata, D, D, THREE.RGBAFormat, THREE.UnsignedByteType);
    dtex.magFilter = THREE.LinearFilter;
    dtex.minFilter = THREE.LinearFilter;
    dtex.colorSpace = THREE.NoColorSpace;
    dtex.needsUpdate = true;
    this.dustSprite = dtex;
  }

  // =======================================================================================
  // 5. SKY DOME — Preetham addon, patched for a fatter sun, a Mie bloom lobe and a horizon
  //    dust band that resolves to exactly the same colour the fog does.
  // =======================================================================================
  _buildSkyDome() {
    const dome = new PreethamSky();
    dome.scale.setScalar(10000);
    dome.frustumCulled = false;
    dome.renderOrder = -1000;
    dome.name = 'sky_dome';
    dome.matrixAutoUpdate = false;
    dome.updateMatrix();

    const m = dome.material;
    const u = m.uniforms;
    u.turbidity.value = this.cfg.sky?.turbidity ?? 6.2;
    u.rayleigh.value = this.cfg.sky?.rayleigh ?? 1.4;
    u.mieCoefficient.value = this.cfg.sky?.mieCoefficient ?? 0.0032;
    u.mieDirectionalG.value = this.cfg.sky?.mieDirectionalG ?? 0.80;
    u.sunPosition.value.copy(this.sunDirection);
    u.up.value.set(0, 1, 0);

    u.uHazeWarm = { value: this.fogSunColor };
    u.uHazeCool = { value: this.fogSkyColor };
    u.uHazeAmount = { value: 1.0 };
    u.uHazeHeight = { value: 0.112 };
    u.uVeilHeight = { value: 0.50 };
    u.uSunDisc = { value: this.sunDiscIntensity };
    u.uSunGlow = { value: 1.0 };
    u.uSkyExposure = { value: 0.84 };
    const fu = this._fogResolveUniforms();
    for (const k in fu) u[k] = fu[k];

    let fs = m.fragmentShader;

    // fatter, soft-limbed disc + two-lobe atmospheric bloom around it
    const discSrc = 'float sundisk = smoothstep( sunAngularDiameterCos, sunAngularDiameterCos + 0.00002, cosTheta );';
    const discNew = /* glsl */`
			float sundisk = smoothstep( sunAngularDiameterCos - 0.000140, sunAngularDiameterCos + 0.000030, cosTheta );
			float sunlimb = pow( max( cosTheta, 0.0 ), 5200.0 );
			float sunhalo = pow( max( cosTheta, 0.0 ), 1100.0 );`;
    if (fs.indexOf(discSrc) >= 0) fs = fs.replace(discSrc, discNew);

    const addSrc = 'L0 += ( vSunE * 19000.0 * Fex ) * sundisk;';
    const addNew = /* glsl */`
			L0 += ( vSunE * 19000.0 * Fex ) * sundisk * uSunDisc;
			L0 += ( vSunE * 19000.0 * Fex ) * ( sunlimb * 0.0090 + sunhalo * 0.00022 ) * uSunGlow * uSunDisc;`;
    if (fs.indexOf(addSrc) >= 0) fs = fs.replace(addSrc, addNew);

    const outSrc = 'gl_FragColor = vec4( retColor, 1.0 );';
    const outNew = /* glsl */`
			// Preetham runs hot for a filmic grade — trim it and put the chroma back that the
			// tone curve is about to take out of the upper sky.
			retColor *= uSkyExposure;
			float skySat = mix( 1.0, 1.42, smoothstep( 0.04, 0.55, direction.y ) );
			float skyLum = dot( retColor, vec3( 0.2126, 0.7152, 0.0722 ) );
			retColor = mix( vec3( skyLum ), retColor, skySat );

			// THE horizon colour. Identical expression to the fog fragment at full optical
			// depth, so fogged geometry and the dome meet with no seam of any kind — this is
			// what stops the ground plane terminating in a visible world edge.
			vec3 hazeCol = bsFogResolve( cosTheta );

			// 1. dense low dust band hugging the horizon
			float hz = clamp( 1.0 - smoothstep( -0.012, uHazeHeight, direction.y ), 0.0, 1.0 );
			hz = mix( hz, sqrt( hz ), 0.42 );
			retColor = mix( retColor, hazeCol, clamp( hz * uHazeAmount, 0.0, 1.0 ) );

			// 2. a much wider, thinner veil above it so the band has no hard top edge and the
			//    whole lower sky keeps the milky atmospheric-perspective read
			float veil = 1.0 - smoothstep( 0.015, uVeilHeight, direction.y );
			retColor = mix( retColor, hazeCol, veil * veil * 0.34 );

			// 3. the dust layer is itself lit by the low sun — a warm lift right on the line,
			//    which is what turns a flat wash into a readable horizon band
			float lift = exp( -abs( direction.y - 0.014 ) * 42.0 );
			retColor += hazeCol * lift * ( 0.13 + 0.20 * max( cosTheta, 0.0 ) );

			// 4. stratification — real dust layers, not a flat wash
			float band = sin( direction.y * 58.0 + 1.3 ) * 0.5 + 0.5;
			float band2 = sin( direction.y * 17.0 - 0.6 ) * 0.5 + 0.5;
			retColor *= 1.0 + ( band * 0.048 + band2 * 0.078 ) * ( hz * 0.6 + veil * 0.4 );

			// 4b. the TOP of the dust layer. A real haze band is a layer you can see the lid
			//     of — without this slight dip the whole lower sky is one undifferentiated
			//     pale wash and the horizon has no altitude cue at all.
			float capY = direction.y - uHazeHeight * 1.75;
			retColor *= 1.0 - 0.105 * exp( -capY * capY * 700.0 );

			// 5. below the horizon the dome IS the fog — exact match, then a slow fall-off so
			//    the very bottom of the frame does not glow.
			retColor = mix( retColor, hazeCol, smoothstep( 0.006, -0.026, direction.y ) );
			retColor *= mix( 1.0, 0.80, smoothstep( -0.02, -0.42, direction.y ) );

			gl_FragColor = vec4( retColor, 1.0 );`;
    if (fs.indexOf(outSrc) >= 0) fs = fs.replace(outSrc, outNew);

    const parsSrc = 'uniform float mieDirectionalG;';
    if (fs.indexOf(parsSrc) >= 0) {
      fs = fs.replace(parsSrc, parsSrc + `
		uniform vec3 uHazeWarm;
		uniform vec3 uHazeCool;
		uniform float uHazeAmount;
		uniform float uHazeHeight;
		uniform float uVeilHeight;
		uniform float uSunDisc;
		uniform float uSunGlow;
		uniform float uSkyExposure;
` + Sky.FOG_RESOLVE_GLSL);
    }

    m.fragmentShader = fs;
    m.needsUpdate = true;
    m.depthWrite = false;
    m.depthTest = true;
    m.toneMapped = true;

    this.skyDome = dome;
    this.skyUniforms = u;
  }

  // =======================================================================================
  // 6. CLOUDS — plane-projected fBm at two altitudes (real parallax), sun-side silvering,
  //    dissolving into the horizon haze. One draw call, forced to the far plane.
  // =======================================================================================
  _buildClouds() {
    const geo = new THREE.SphereGeometry(10, 64, 26, 0, Math.PI * 2, 0, Math.PI * 0.54);

    const mat = new THREE.ShaderMaterial({
      name: 'BlacksiteClouds',
      uniforms: this._fogResolveUniforms({
        tNoise: { value: this.cloudNoise },
        uTime: { value: 0 },
        uSunDir: { value: this.sunDirection },
        uSunColor: { value: this.sunColor },
        // VALUE, not hue, is what survives the AgX tone curve. Measured against the frame:
        // the sky sits near 1.4 in linear, so a cloud core has to fall to ~0.10-0.25 and a
        // sunlit rim has to climb past 3.0 before the difference is legible on screen.
        uLit: { value: new THREE.Color(0xfff3e2) },      // sunlit flank / thin silvered edge
        uShadow: { value: new THREE.Color(0x8a7f96) },   // cool mauve core, never black
        uWarm: { value: new THREE.Color(0xffb47e) },     // golden-hour underside wash
        uCirrusCol: { value: new THREE.Color(0xffe2c0) },
        // fBm is narrow-band around 0.5; uContrast expands it to a usable 0..1 signal
        // before thresholding, otherwise the deck never crosses the coverage cut and the
        // sky reads as an empty gradient (which is exactly how it was rejected).
        uContrast: { value: 3.35 },
        uCoverage: { value: 0.450 },
        uOpacity: { value: 0.97 },
        uCirrus: { value: 0.42 },
        uMackerel: { value: 0.62 },
        uLowH: { value: 900.0 },
        uMidH: { value: 2400.0 },
        uHighH: { value: 7000.0 },
      }),
      vertexShader: /* glsl */`
        varying vec3 vDir;
        void main() {
          vec4 wp = modelMatrix * vec4( position, 1.0 );
          vDir = wp.xyz - cameraPosition;
          vec4 mv = viewMatrix * wp;
          gl_Position = projectionMatrix * mv;
          gl_Position.z = gl_Position.w;   // pin to the far plane, ignore camera.far
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tNoise;
        uniform float uTime, uCoverage, uOpacity, uLowH, uMidH, uHighH;
        uniform float uContrast, uCirrus, uMackerel;
        uniform vec3 uSunDir, uSunColor, uLit, uShadow, uWarm, uCirrusCol;
        varying vec3 vDir;
${Sky.FOG_RESOLVE_GLSL}

        // Distance along the view ray to a cloud shell of height h.
        //
        // A flat-plane projection ( h / d.y ) blows up at the horizon: it crams hundreds of
        // kilometres into one texel, the derivative explodes, the sampler drops to the top
        // mip and the lower sky — the part you actually see in a first-person frame — turns
        // into flat grey. Intersecting a curved shell is finite everywhere; the soft
        // saturation on top of it ( t*T/(t+T) ) additionally BOUNDS the derivative, so the
        // deck converges toward the horizon the way a real one does and stays sharp while
        // it does it. RE is a stage curvature, not the Earth.
        float bsShellT( float dy, float h, float tmax ) {
          const float RE = 300000.0;
          float t = sqrt( RE * RE * dy * dy + 2.0 * RE * h ) - RE * dy;
          return tmax * t / ( t + tmax );
        }

        void main() {
          vec3 d = normalize( vDir );
          if ( d.y <= 0.0015 ) discard;

          float sd = dot( d, uSunDir );
          float sunAmt = max( sd, 0.0 );
          float sa2 = sunAmt * sunAmt;
          float sa4 = sa2 * sa2;
          float sa8 = sa4 * sa4;
          float lod = smoothstep( 0.30, 0.015, d.y ) * 1.05;

          // A 18 deg sun. Looking AWAY from it you see the lit flanks of the deck: bright
          // and warm. Looking INTO it you see the shadowed bellies: dark, with one blazing
          // thin rim. Brightening the deck toward the sun — which is what this shader used
          // to do — makes it match the (already very bright) sky exactly where the player
          // is looking at the most sky, i.e. it renders a fully covered deck as an empty
          // gradient. This pair of ramps is the whole reason the clouds read.
          float sideK = 0.5 - 0.5 * sd;              // 0 straight at the sun, 1 opposite it
          float backLit = smoothstep( 0.05, 0.80, sd );

          // ================= layer 1: cumulus / altocumulus deck, ~900 m ==================
          float t1 = bsShellT( d.y, uLowH, 4200.0 );
          // anisotropic — the deck is drawn out along the prevailing wind. Cell size lands
          // near 540 m, which puts 4-6 cloud masses across a 100 deg frame; the earlier
          // 2300 m cell meant the entire visible sky was ONE cell, which is how a covered
          // deck still managed to render as an empty gradient.
          vec2 pn = ( cameraPosition.xz + d.xz * t1 ) * vec2( 0.00046, 0.00074 );
          pn += vec2( uTime * 0.0016, uTime * 0.00060 );
          vec4 n1 = texture2D( tNoise, pn, lod );
          vec4 n2 = texture2D( tNoise, pn * 2.7 + 0.17, lod + 0.6 );
          vec4 n3 = texture2D( tNoise, pn * 6.1 - 0.41, lod + 1.3 );

          // R + B are the two large scales, G is the billow that erodes the silhouette into
          // cauliflower, A is the weather field that decides where the deck banks up and
          // where it opens into clear sky.
          float dens = ( n1.r * 0.55 + n1.b * 0.45 - 0.5 ) * uContrast + 0.5;
          dens += ( n2.g - 0.5 ) * 0.44 + ( n2.r - 0.5 ) * 0.26;
          dens += ( n3.g - 0.5 ) * 0.21;
          dens += ( n1.a - 0.5 ) * 0.70;
          // decks converge and pile up toward the horizon — but they must keep their holes,
          // because a solid overcast the colour of the haze IS an empty gradient
          dens += ( 1.0 - smoothstep( 0.02, 0.70, d.y ) ) * 0.20;
          dens -= smoothstep( 0.48, 1.0, d.y ) * 0.10;

          // A narrow cover ramp is what gives a cloud a SILHOUETTE instead of a smear.
          float cover = smoothstep( uCoverage, uCoverage + 0.085, dens );
          float thick = smoothstep( uCoverage + 0.07, uCoverage + 0.44, dens );
          float rim = clamp( cover - thick, 0.0, 1.0 );

          // ---- shading -------------------------------------------------------------------
          // the billow octave doubles as self-shadowing so a core is lumpy, not a flat patch
          float lightK = clamp( 0.16 + ( 1.0 - thick ) * 0.32 + sideK * 0.62 - backLit * 0.22
                              + ( n2.g - 0.5 ) * 0.32 + ( n3.g - 0.5 ) * 0.16, 0.0, 1.0 );
          vec3 col = mix( uShadow, uLit, lightK );

          float lowK = 1.0 - smoothstep( 0.03, 0.55, d.y );
          float warmK = clamp( ( 0.30 + 0.42 * sa2 ) * ( 0.34 + 0.66 * lowK ), 0.0, 0.80 );
          col = mix( col, uWarm, warmK * ( 0.42 + 0.58 * thick ) );
          // thin edges scatter forward and silver hard against the sun — the CORES must not,
          // or the whole deck bleaches out to sky value in exactly the frame that shows the
          // most sky.
          col += uSunColor * sa4 * ( 0.05 + rim * 2.60 );
          col += uSunColor * ( sa8 * sunAmt ) * ( 0.18 + rim * 1.50 );

          float a = cover * uOpacity * ( 0.72 + 0.28 * thick );
          a *= 1.0 - 0.18 * smoothstep( 0.55, 1.0, d.y );

          // ================= layer 2: mackerel altocumulus, ~2400 m =======================
          // Real parallax against layer 1, and small enough cells that a thin strip of sky
          // between two rooflines still has readable structure in it.
          float t2 = bsShellT( d.y, uMidH, 11000.0 );
          vec2 pm = ( cameraPosition.xz + d.xz * t2 ) * vec2( 0.00031, 0.00048 );
          pm += vec2( uTime * 0.00092, -uTime * 0.00035 );
          vec4 m1 = texture2D( tNoise, pm * 3.15 + 0.63, lod * 0.75 );
          float md = ( m1.r * 0.52 + m1.g * 0.48 - 0.5 ) * 2.75 + 0.5 + ( m1.a - 0.5 ) * 0.55;
          float mCover = smoothstep( 0.535, 0.655, md );
          float mThick = smoothstep( 0.615, 0.860, md );
          float mA = mCover * uMackerel * smoothstep( 0.012, 0.10, d.y );
          vec3 mC = mix( uShadow, uLit, clamp( 0.20 + sideK * 0.54 - backLit * 0.18
                                             + ( 1.0 - mThick ) * 0.26, 0.0, 1.0 ) );
          mC = mix( mC, uWarm, ( 0.32 + 0.40 * sa2 ) * 0.70 );
          mC += uSunColor * sa4 * ( 0.06 + ( mCover - mThick ) * 1.80 );

          // ================= layer 3: cirrus fibres, ~7000 m ==============================
          float t3 = bsShellT( d.y, uHighH, 26000.0 );
          // brutally anisotropic: cirrus is drawn out into fibres by the jet
          vec2 p3 = ( cameraPosition.xz + d.xz * t3 ) * vec2( 0.000185, 0.0000165 );
          p3 += vec2( uTime * 0.0022, -uTime * 0.00018 );
          vec4 cs = texture2D( tNoise, p3, lod * 0.5 );
          float ci = cs.r * 0.62 + cs.g * 0.38;
          ci = ( ci - 0.5 ) * 2.9 + 0.5;
          float cA = smoothstep( 0.47, 0.90, ci ) * uCirrus;
          cA *= smoothstep( 0.02, 0.24, d.y );
          vec3 cC = mix( uCirrusCol, uSunColor, 0.34 + 0.60 * sa2 );
          cC = mix( cC, uWarm, 0.34 * sa2 );
          cC *= 1.0 + sa8 * 1.1;

          // ================= composite, back to front, premultiplied ======================
          vec3 acc = cC * cA;
          float aTot = cA;
          acc = mC * mA + acc * ( 1.0 - mA );
          aTot = mA + aTot * ( 1.0 - mA );
          acc = col * a + acc * ( 1.0 - a );
          aTot = a + aTot * ( 1.0 - a );
          vec3 cTot = acc / max( aTot, 1e-4 );

          // The deck does not end at a line: within a few degrees of the horizon it turns
          // into the same dust the fog resolves to, keeps most of its alpha, and simply
          // becomes part of the haze band. Nothing on this dome ever has a hard edge.
          float dust = 1.0 - smoothstep( 0.004, 0.098, d.y );
          vec3 hz = bsFogResolve( dot( d, uSunDir ) );
          cTot = mix( cTot, hz, dust * 0.94 );
          aTot *= mix( 1.0, 0.66, dust );

          gl_FragColor = vec4( cTot, clamp( aTot, 0.0, 1.0 ) );
        }`,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      fog: false,
      toneMapped: true,
      blending: THREE.NormalBlending,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = -900;
    mesh.name = 'sky_clouds';
    this.clouds = mesh;
    this.cloudUniforms = mat.uniforms;
  }

  // =======================================================================================
  // 6b. BACKGROUND SHELLS — the world does not end at the ground mesh.
  //     Three silhouette layers at 300 / 470 / 720 m plus minarets and smoke columns, each
  //     hazed harder than the last. They are pure value modulations of bsFogResolve(), so
  //     they can never fall outside the grade and they bed straight into the horizon dust.
  // =======================================================================================
  _backdropShaders() {
    const vert = /* glsl */`
      varying vec3 vWorld;
      varying vec3 vNrm;
      void main() {
        #ifdef USE_INSTANCING
          vec4 lp = instanceMatrix * vec4( position, 1.0 );
          vec3 ln = mat3( instanceMatrix ) * normal;
        #else
          vec4 lp = vec4( position, 1.0 );
          vec3 ln = normal;
        #endif
        vec4 wp = modelMatrix * lp;
        vWorld = wp.xyz;
        vNrm = normalize( mat3( modelMatrix ) * ln );
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`;

    const frag = /* glsl */`
      uniform vec3 uSunDir, uSunColor, uSkyFill;
      uniform float uHaze, uNear, uFar, uValue, uBaseY, uDustTop;
      varying vec3 vWorld;
      varying vec3 vNrm;
${Sky.FOG_RESOLVE_GLSL}

      void main() {
        vec3 ray = vWorld - cameraPosition;
        float dist = max( length( ray ), 1.0 );
        vec3 d = ray / dist;
        vec3 n = normalize( vNrm );

        vec3 hz = bsFogResolve( dot( d, uSunDir ) );

        // Atmospheric perspective is a VALUE shift on the haze, never a separate colour —
        // that way a background shell can never fall outside the grade.
        float ndl = max( dot( n, uSunDir ), 0.0 );
        float sky = 0.5 + 0.5 * n.y;
        // per-block variation so the ring does not read as a row of identical cards
        float vSeed = fract( sin( dot( floor( vWorld.xz * 0.043 ), vec2( 12.9898, 78.233 ) ) ) * 43758.5453 );
        float shade = mix( 0.30, 1.08, ndl ) * mix( 0.82, 1.10, sky ) * ( 0.78 + 0.40 * vSeed );

        // ---- facade breakup ------------------------------------------------------------
        // Without this a background block is one flat value and the whole ring reads as a
        // milky card. Floor bands + window columns on the vertical faces only; the tangent
        // coordinate is derived from the face normal so it follows each wall.
        float wall = 1.0 - smoothstep( 0.45, 0.90, abs( n.y ) );
        float tang = vWorld.x * n.z - vWorld.z * n.x;
        float fy = fract( vWorld.y * 0.255 + vSeed * 0.7 );
        float rows = smoothstep( 0.16, 0.30, fy ) * ( 1.0 - smoothstep( 0.58, 0.74, fy ) );
        float cols = smoothstep( 0.34, 0.86, 0.5 + 0.5 * cos( tang * 1.15 + vSeed * 6.0 ) );
        shade *= 1.0 - 0.38 * rows * cols * wall;
        // a slab line at every floor — reads as scale even when the windows blur out
        shade *= 1.0 - 0.10 * wall * smoothstep( 0.94, 1.0, fy );
        // roofs are flat to the sky and to a low sun: they hold the light and read as the
        // top edge of the district instead of merging into the haze
        shade *= 1.0 + 0.20 * smoothstep( 0.55, 0.98, n.y );

        vec3 lit = hz * shade * uValue + uSunColor * ( ndl * ndl * ndl ) * 0.16;
        // a floor of coloured bounce — a background silhouette never goes to black
        lit = max( lit, hz * 0.24 );

        float t = clamp( ( dist - uNear ) / max( uFar - uNear, 1.0 ), 0.0, 1.0 );
        float haze = clamp( mix( uHaze, uHaze + 0.17, t ), 0.0, 1.0 );
        // the dust pools low: bases dissolve, only the tops keep a hard silhouette
        float up = clamp( ( vWorld.y - uBaseY ) / uDustTop, 0.0, 1.0 );
        haze = mix( min( haze + 0.40, 1.0 ), haze, up * up );

        gl_FragColor = vec4( mix( lit, hz, haze ), 1.0 );
      }`;

    return { vert, frag };
  }

  _backdropMaterial(name, opts) {
    const { vert, frag } = this._bgSrc || (this._bgSrc = this._backdropShaders());
    return new THREE.ShaderMaterial({
      name,
      uniforms: this._fogResolveUniforms({
        uSunDir: { value: this.sunDirection },
        uSunColor: { value: this.sunColor },
        uSkyFill: { value: this.skyColor },
        uHaze: { value: opts.haze },
        uNear: { value: opts.near },
        uFar: { value: opts.far },
        uValue: { value: opts.value ?? 1.0 },
        uBaseY: { value: opts.baseY ?? 0 },
        uDustTop: { value: opts.dustTop ?? 26 },
      }),
      vertexShader: vert,
      fragmentShader: frag,
      side: THREE.FrontSide,
      transparent: false,
      depthWrite: true,
      depthTest: true,
      fog: false,
      toneMapped: true,
    });
  }

  _buildBackdrop() {
    let s = 0x5eed1017 >>> 0;
    const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
    const rr = (a, b) => a + (b - a) * rnd();

    const grp = new THREE.Group();
    grp.name = 'sky_backdrop';
    grp.matrixAutoUpdate = false;
    grp.updateMatrix();
    this._bgDisposables = [];

    const box = new THREE.BoxGeometry(1, 1, 1);
    const q = new THREE.Quaternion();
    const m4 = new THREE.Matrix4();
    const vP = new THREE.Vector3();
    const vS = new THREE.Vector3();
    const upY = new THREE.Vector3(0, 1, 0);

    // ---- two rings of city blocks ---------------------------------------------------
    // Haze here is the ONLY thing that separates a silhouette from the sky. At 0.52 / 0.72
    // the rings rendered as milk-white ghost cards with no value in them at all; the whole
    // point of a background layer is that the mid-ground has something DARKER to sit in
    // front of. These numbers keep atmospheric perspective (near ring reads, far ring is
    // nearly gone) while leaving the near ring a real, readable silhouette.
    const rings = [
      { r0: BG_CITY_NEAR, r1: BG_CITY_NEAR + 110, n: 104, hMin: 12, hMax: 52, haze: 0.20, value: 1.0, dustTop: 18 },
      { r0: BG_CITY_FAR, r1: BG_CITY_FAR + 160, n: 78, hMin: 18, hMax: 68, haze: 0.44, value: 1.0, dustTop: 30 },
    ];
    for (let ri = 0; ri < rings.length; ri++) {
      const R = rings[ri];
      const mat = this._backdropMaterial('BlacksiteBackdropCity' + ri, {
        haze: R.haze, near: R.r0, far: R.r1 + 90, value: R.value, baseY: -6, dustTop: R.dustTop,
      });
      const im = new THREE.InstancedMesh(box, mat, R.n);
      im.instanceMatrix.setUsage(THREE.StaticDrawUsage);
      for (let i = 0; i < R.n; i++) {
        const a = (i / R.n) * Math.PI * 2 + rr(-0.016, 0.016);
        const rad = rr(R.r0, R.r1);
        const h = rr(R.hMin, R.hMax) * (rnd() < 0.07 ? 1.9 : 1.0);
        vS.set(rr(14, 46), h + 14, rr(12, 38));
        vP.set(Math.sin(a) * rad, -14 + vS.y * 0.5, Math.cos(a) * rad);
        q.setFromAxisAngle(upY, rr(-0.5, 0.5) + a);
        m4.compose(vP, q, vS);
        im.setMatrixAt(i, m4);
      }
      im.instanceMatrix.needsUpdate = true;
      im.frustumCulled = true;
      im.castShadow = im.receiveShadow = false;
      im.name = 'sky_bg_city' + ri;
      im.userData.noCollide = true;
      im.userData.bsNoDecal = true;
      if (!globalThis.__NOCITY) grp.add(im);
      this._bgDisposables.push(mat);
    }

    // ---- minaret cluster --------------------------------------------------------------
    const NM = 14;
    const shaftGeo = new THREE.CylinderGeometry(0.62, 0.86, 1, 10, 1, true);
    const capGeo = new THREE.ConeGeometry(0.95, 1, 10, 1);
    const minMat = this._backdropMaterial('BlacksiteBackdropMinaret', {
      haze: 0.19, near: 235, far: 560, value: 1.0, baseY: -6, dustTop: 26,
    });
    const shafts = new THREE.InstancedMesh(shaftGeo, minMat, NM);
    const caps = new THREE.InstancedMesh(capGeo, minMat, NM);
    for (let i = 0; i < NM; i++) {
      // cluster them into two districts rather than sprinkling evenly
      const districtA = i < 6;
      const a = (districtA ? -0.55 : 2.25) + rr(-0.85, 0.85);
      const rad = rr(240, 500);
      const h = rr(30, 82);
      const w = rr(2.6, 4.4);
      vP.set(Math.sin(a) * rad, -10 + h * 0.5, Math.cos(a) * rad);
      q.identity();
      m4.compose(vP, q, vS.set(w, h, w));
      shafts.setMatrixAt(i, m4);
      vP.y = -10 + h + w * 1.35;
      m4.compose(vP, q, vS.set(w * 1.15, w * 2.7, w * 1.15));
      caps.setMatrixAt(i, m4);
    }
    shafts.instanceMatrix.needsUpdate = true;
    caps.instanceMatrix.needsUpdate = true;
    for (const im of [shafts, caps]) {
      im.castShadow = im.receiveShadow = false;
      im.name = 'sky_bg_minaret';
      im.userData.noCollide = true;
      im.userData.bsNoDecal = true;
      if (!globalThis.__NOMIN) grp.add(im);
    }
    this._bgDisposables.push(minMat, shaftGeo, capGeo);

    // ---- distant ridge line -------------------------------------------------------------
    const SEG = 112;
    const rpos = new Float32Array(SEG * 4 * 3);
    const rnrm = new Float32Array(SEG * 4 * 3);
    const ridx = new Uint16Array(SEG * 6);
    // fBm is narrow-band around 0.5 — sampled raw this returned ~66 m everywhere and the
    // "hills" rendered as a dead straight line across the whole horizon. Expand the band,
    // then add a ridged octave so the line has actual peaks and saddles.
    const ridgeH = (i) => {
      const u = i / SEG;
      const big = clamp01((tileFbm(u, 0.31, 5, 4, 0.55, 5501) - 0.5) * 2.6 + 0.5);
      const peaks = Math.pow(clamp01((tileBillow(u, 0.77, 9, 3, 0.5, 991) - 0.18) * 1.7), 1.4);
      const fine = (tileFbm(u, 0.09, 21, 2, 0.5, 3307) - 0.5) * 26;
      return 14 + big * big * 96 + peaks * 52 + fine;
    };
    for (let i = 0; i < SEG; i++) {
      const a0 = (i / SEG) * Math.PI * 2, a1 = ((i + 1) / SEG) * Math.PI * 2;
      const h0 = ridgeH(i), h1 = ridgeH((i + 1) % SEG);
      const x0 = Math.sin(a0) * BG_RIDGE_R, z0 = Math.cos(a0) * BG_RIDGE_R;
      const x1 = Math.sin(a1) * BG_RIDGE_R, z1 = Math.cos(a1) * BG_RIDGE_R;
      const b = i * 12, ni = i * 4;
      rpos[b + 0] = x0; rpos[b + 1] = -60; rpos[b + 2] = z0;
      rpos[b + 3] = x1; rpos[b + 4] = -60; rpos[b + 5] = z1;
      rpos[b + 6] = x1; rpos[b + 7] = h1; rpos[b + 8] = z1;
      rpos[b + 9] = x0; rpos[b + 10] = h0; rpos[b + 11] = z0;
      for (let k = 0; k < 4; k++) {
        rnrm[b + k * 3 + 0] = -Math.sin(a0);
        rnrm[b + k * 3 + 1] = 0.22;
        rnrm[b + k * 3 + 2] = -Math.cos(a0);
      }
      const o = i * 6;
      ridx[o] = ni; ridx[o + 1] = ni + 1; ridx[o + 2] = ni + 2;
      ridx[o + 3] = ni; ridx[o + 4] = ni + 2; ridx[o + 5] = ni + 3;
    }
    const ridgeGeo = new THREE.BufferGeometry();
    ridgeGeo.setAttribute('position', new THREE.BufferAttribute(rpos, 3));
    ridgeGeo.setAttribute('normal', new THREE.BufferAttribute(rnrm, 3));
    ridgeGeo.setIndex(new THREE.BufferAttribute(ridx, 1));
    ridgeGeo.computeBoundingSphere();
    const ridgeMat = this._backdropMaterial('BlacksiteBackdropRidge', {
      haze: 0.50, near: BG_RIDGE_R - 60, far: BG_RIDGE_R + 60, value: 1.0, baseY: -12, dustTop: 55,
    });
    ridgeMat.side = THREE.DoubleSide;
    const ridge = new THREE.Mesh(ridgeGeo, ridgeMat);
    ridge.frustumCulled = false;
    ridge.castShadow = ridge.receiveShadow = false;
    ridge.name = 'sky_bg_ridge';
    ridge.userData.noCollide = true;
    ridge.userData.bsNoDecal = true;
    if (!globalThis.__NORIDGE) grp.add(ridge);
    this._bgDisposables.push(ridgeGeo, ridgeMat);

    // ---- smoke columns -------------------------------------------------------------------
    const smokeGeo = new THREE.PlaneGeometry(1, 1, 1, 6);
    const smokeMat = new THREE.ShaderMaterial({
      name: 'BlacksiteBackdropSmoke',
      uniforms: this._fogResolveUniforms({
        tNoise: { value: this.cloudNoise },
        uTime: { value: 0 },
        uSunDir: { value: this.sunDirection },
        uSunColor: { value: this.sunColor },
        uDark: { value: new THREE.Color(0x3b342f) },
        uOpacity: { value: 0.94 },
      }),
      vertexShader: /* glsl */`
        attribute vec4 aCol;      // xz = world origin, y = height, w = width
        attribute vec3 aLean;     // xz = lean per unit height, y = phase
        uniform vec3 uSunDir;
        varying vec2 vUv;
        varying vec3 vWorld;
        varying float vPhase;
        varying float vSide;      // >0 on the flank of the column that faces the sun
        void main() {
          vec3 origin = vec3( aCol.x, 0.0, aCol.y );
          float H = aCol.z, W = aCol.w;
          vec3 toCam = cameraPosition - origin;
          toCam.y = 0.0;
          vec3 right = normalize( vec3( -toCam.z, 0.0, toCam.x ) + vec3( 1e-4, 0.0, 0.0 ) );
          float f = uv.y;
          float w = W * ( 0.22 + 1.90 * pow( f, 0.72 ) );
          vec3 wp = origin
                  + right * ( position.x * w )
                  + vec3( 0.0, f * H - 12.0, 0.0 )
                  + vec3( aLean.x, 0.0, aLean.z ) * pow( f, 1.7 ) * H;
          vUv = uv;
          vPhase = aLean.y;
          vWorld = wp;
          vSide = ( uv.x - 0.5 ) * 2.0 * dot( right, uSunDir );
          gl_Position = projectionMatrix * viewMatrix * vec4( wp, 1.0 );
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tNoise;
        uniform float uTime, uOpacity;
        uniform vec3 uSunDir, uSunColor, uDark;
        varying vec2 vUv;
        varying vec3 vWorld;
        varying float vPhase;
        varying float vSide;
${Sky.FOG_RESOLVE_GLSL}
        void main() {
          vec2 p = vec2( vUv.x * 0.9 + vPhase, vUv.y * 0.46 - uTime * 0.0065 + vPhase );
          float n = texture2D( tNoise, p ).r * 0.52
                  + texture2D( tNoise, p * 2.4 + 0.29 ).g * 0.31
                  + texture2D( tNoise, p * 5.1 - 0.13 ).b * 0.17;
          float body = 1.0 - abs( vUv.x - 0.5 ) * 2.0;
          body = pow( clamp( body, 0.0, 1.0 ), 1.25 );
          body *= smoothstep( 0.0, 0.10, vUv.y ) * ( 1.0 - smoothstep( 0.44, 1.0, vUv.y ) );
          float a = clamp( smoothstep( 0.26, 0.76, n * 0.62 + body * 0.66 ) * body, 0.0, 1.0 );
          a *= uOpacity;
          if ( a < 0.004 ) discard;

          vec3 d = normalize( vWorld - cameraPosition );
          float cs = dot( d, uSunDir );
          vec3 hz = bsFogResolve( cs );
          // dense and sooty at the base, dispersing into the haze as it climbs
          vec3 col = mix( uDark, hz, 0.30 + 0.62 * pow( vUv.y, 0.8 ) );
          float sm = max( cs, 0.0 );
          float sm2 = sm * sm;
          // the low sun rakes the western flank of the column
          col += uSunColor * ( 0.10 + 0.34 * sm2 ) * smoothstep( -0.15, 0.85, vSide ) * 0.30;
          col += uSunColor * ( sm2 * sm2 * sm ) * 0.16;
          gl_FragColor = vec4( col, a );
        }`,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      fog: false,
      toneMapped: true,
    });

    const COLS = [
      // [azimuth, radius, height, width, leanX, leanZ]
      [-0.42, 372, 185, 22],
      [1.02, 505, 235, 28],
      [2.55, 430, 155, 19],
      [3.05, 318, 205, 25],   // the hero column — sits inside the spawn frustum
    ];
    const NC = COLS.length;
    const base = smokeGeo.attributes.position.count;
    const sPos = new Float32Array(base * NC * 3);
    const sUv = new Float32Array(base * NC * 2);
    const sCol = new Float32Array(base * NC * 4);
    const sLean = new Float32Array(base * NC * 3);
    const srcP = smokeGeo.attributes.position.array;
    const srcU = smokeGeo.attributes.uv.array;
    const srcI = smokeGeo.index.array;
    const sIdx = new Uint16Array(srcI.length * NC);
    for (let c = 0; c < NC; c++) {
      const [az, rad, H, W] = COLS[c];
      const ox = Math.sin(az) * rad, oz = Math.cos(az) * rad;
      const lx = rr(0.05, 0.22), lz = rr(-0.12, 0.12);
      const ph = rnd();
      for (let v = 0; v < base; v++) {
        const o3 = (c * base + v) * 3, o2 = (c * base + v) * 2, o4 = (c * base + v) * 4;
        sPos[o3] = srcP[v * 3]; sPos[o3 + 1] = srcP[v * 3 + 1]; sPos[o3 + 2] = 0;
        sUv[o2] = srcU[v * 2]; sUv[o2 + 1] = srcU[v * 2 + 1];
        sCol[o4] = ox; sCol[o4 + 1] = oz; sCol[o4 + 2] = H; sCol[o4 + 3] = W;
        sLean[o3] = lx * 0.9; sLean[o3 + 1] = ph; sLean[o3 + 2] = lz;
      }
      for (let k = 0; k < srcI.length; k++) sIdx[c * srcI.length + k] = srcI[k] + c * base;
    }
    const smokeMerged = new THREE.BufferGeometry();
    smokeMerged.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
    smokeMerged.setAttribute('uv', new THREE.BufferAttribute(sUv, 2));
    smokeMerged.setAttribute('aCol', new THREE.BufferAttribute(sCol, 4));
    smokeMerged.setAttribute('aLean', new THREE.BufferAttribute(sLean, 3));
    smokeMerged.setIndex(new THREE.BufferAttribute(sIdx, 1));
    smokeMerged.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1400);
    smokeGeo.dispose();

    const smoke = new THREE.Mesh(smokeMerged, smokeMat);
    smoke.frustumCulled = false;
    smoke.renderOrder = -880;
    smoke.castShadow = smoke.receiveShadow = false;
    smoke.name = 'sky_bg_smoke';
    smoke.userData.noCollide = true;
    smoke.userData.bsNoDecal = true;
    if (!globalThis.__NOSMOKE) grp.add(smoke);
    this._bgDisposables.push(smokeMerged, smokeMat, box);

    this.backdrop = grp;
    this.smokeUniforms = smokeMat.uniforms;
  }

  // =======================================================================================
  // 7. DUST MOTES — GPU-wrapped point field that follows the camera. No CPU work per frame.
  // =======================================================================================
  _buildDust() {
    const n = this._dustCount;
    const box = 96, slab = 17;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const size = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pos[i * 3 + 0] = (Math.random() - 0.5) * box;
      // bias the population toward the ground — dust pools low
      pos[i * 3 + 1] = Math.pow(Math.random(), 1.9) * slab;
      pos[i * 3 + 2] = (Math.random() - 0.5) * box;
      seed[i] = Math.random();
      size[i] = 0.8 + Math.pow(Math.random(), 3.0) * 5.4;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e6);

    const mat = new THREE.ShaderMaterial({
      name: 'BlacksiteDust',
      uniforms: {
        tSprite: { value: this.dustSprite },
        uTime: { value: 0 },
        uBox: { value: new THREE.Vector2(box, box) },
        uSlab: { value: slab },
        uGround: { value: -0.4 },
        uWind: { value: new THREE.Vector2(0.42, 0.17) },
        uScale: { value: 700 },
        uSunDir: { value: this.sunDirection },
        uSunColor: { value: this.sunColor },
        uCoolColor: { value: this.skyColor },
        uOpacity: { value: 0.52 },
      },
      vertexShader: /* glsl */`
        attribute float aSeed;
        attribute float aSize;
        uniform float uTime, uSlab, uGround, uScale, uOpacity;
        uniform vec2 uBox, uWind;
        uniform vec3 uSunDir, uSunColor, uCoolColor;
        varying vec3 vCol;
        varying float vAlpha;

        void main() {
          float ph = aSeed * 6.2831853;
          vec3 p = position;
          p.x += uWind.x * uTime + sin( uTime * 0.42 + ph ) * 0.95;
          p.z += uWind.y * uTime + cos( uTime * 0.37 + ph * 1.7 ) * 0.95;
          p.y += sin( uTime * 0.23 + ph * 2.3 ) * 0.42 + uTime * 0.045;

          vec2 rel = p.xz - cameraPosition.xz + uBox * 0.5;
          rel = mod( rel, uBox ) - uBox * 0.5;
          float h = mod( p.y, uSlab );

          vec3 world = vec3( cameraPosition.x + rel.x, h + uGround, cameraPosition.z + rel.y );

          vec3 toCam = world - cameraPosition;
          float dist = max( length( toCam ), 0.05 );
          vec3 vdir = toCam / dist;

          float back = pow( max( dot( vdir, uSunDir ), 0.0 ), 3.0 );
          vCol = mix( uCoolColor, uSunColor, 0.15 + back * 0.85 ) * ( 0.30 + back * 2.10 );

          float hFade = exp( -h * 0.135 );
          float nearFade = smoothstep( 0.45, 2.4, dist );
          float farFade = 1.0 - smoothstep( uBox.x * 0.30, uBox.x * 0.47, dist );
          vAlpha = uOpacity * hFade * nearFade * farFade;

          vec4 mv = viewMatrix * vec4( world, 1.0 );
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp( aSize * uScale / dist, 1.0, 26.0 );
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tSprite;
        varying vec3 vCol;
        varying float vAlpha;
        void main() {
          float a = texture2D( tSprite, gl_PointCoord ).a;
          if ( a < 0.004 ) discard;
          gl_FragColor = vec4( vCol * a * vAlpha, 1.0 );
        }`,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
      fog: false,
      toneMapped: true,
    });

    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    pts.renderOrder = 10;
    pts.matrixAutoUpdate = false;
    pts.name = 'sky_dust';
    this.dust = pts;
    this.dustUniforms = mat.uniforms;
  }

  // =======================================================================================
  // 8. GROUND HEAT HAZE — a warm, slowly-boiling sheet sitting just above head-height of the
  //    ground plane. Grazing-angle weighted, so it only reads where hot air actually reads.
  // =======================================================================================
  _buildGroundHaze() {
    const geo = new THREE.PlaneGeometry(520, 520, 1, 1);
    geo.rotateX(-Math.PI * 0.5);

    const mat = new THREE.ShaderMaterial({
      name: 'BlacksiteGroundHaze',
      uniforms: {
        tNoise: { value: this.cloudNoise },
        uTime: { value: 0 },
        uSunDir: { value: this.sunDirection },
        uWarm: { value: this.fogSunColor },
        uCool: { value: this.fogSkyColor },
        uStrength: { value: 0.62 },
      },
      vertexShader: /* glsl */`
        varying vec3 vWorld;
        void main() {
          vec4 wp = modelMatrix * vec4( position, 1.0 );
          vWorld = wp.xyz;
          gl_Position = projectionMatrix * viewMatrix * wp;
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tNoise;
        uniform float uTime, uStrength;
        uniform vec3 uSunDir, uWarm, uCool;
        varying vec3 vWorld;

        void main() {
          vec3 ray = vWorld - cameraPosition;
          float dist = length( ray );
          vec3 d = ray / max( dist, 1e-3 );

          vec2 uv = vWorld.xz * 0.0125;
          float n1 = texture2D( tNoise, uv * 0.55 + vec2( uTime * 0.0075, uTime * 0.0043 ) ).b;
          float n2 = texture2D( tNoise, uv * 1.90 - vec2( uTime * 0.0135, uTime * 0.0098 ) ).g;
          float n3 = texture2D( tNoise, uv * 4.10 + vec2( -uTime * 0.024, uTime * 0.019 ) ).r;
          float n = n1 * 0.55 + n2 * 0.42 + n3 * 0.22;

          float boil = smoothstep( 0.44, 0.98, n );
          float graze = pow( 1.0 - clamp( abs( d.y ), 0.0, 1.0 ), 5.0 );
          float band = smoothstep( 9.0, 40.0, dist ) * ( 1.0 - smoothstep( 95.0, 210.0, dist ) );

          float sunAmt = max( dot( d, uSunDir ), 0.0 );
          vec3 col = mix( uCool, uWarm, pow( sunAmt, 1.6 ) );
          col += uWarm * pow( sunAmt, 8.0 ) * 0.5;

          float a = boil * graze * band * uStrength * ( 0.30 + 0.95 * pow( sunAmt, 2.5 ) );
          if ( a < 0.002 ) discard;
          gl_FragColor = vec4( col, clamp( a, 0.0, 0.75 ) );
        }`,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      fog: false,
      toneMapped: true,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(0, 1.15, 0);
    mesh.frustumCulled = false;
    mesh.renderOrder = 12;
    mesh.name = 'sky_groundhaze';
    this.groundHaze = mesh;
    this.hazeUniforms = mat.uniforms;
  }

  // =======================================================================================
  // 9. GOD RAYS — occlusion buffer (scene in black, bright sun disc depth-tested behind it)
  //    then two radial-blur iterations. PostFX additively composites godrayTexture.
  // =======================================================================================
  _buildGodrayResources() {
    this._blackMat = new THREE.MeshBasicMaterial({ color: 0x000000, fog: false, toneMapped: false });
    this._blackMat.name = 'BlacksiteOcclusion';

    const sunGeo = new THREE.PlaneGeometry(1, 1);
    const sunMat = new THREE.ShaderMaterial({
      name: 'BlacksiteSunOccluder',
      uniforms: { uColor: { value: new THREE.Color(1.0, 0.92, 0.78) } },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }`,
      fragmentShader: /* glsl */`
        uniform vec3 uColor;
        varying vec2 vUv;
        void main() {
          float r = length( vUv - 0.5 ) * 2.0;
          float a = smoothstep( 1.0, 0.02, r );
          a = a * a;
          gl_FragColor = vec4( uColor * a, 1.0 );
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      fog: false,
      toneMapped: false,
    });
    this._sunQuad = new THREE.Mesh(sunGeo, sunMat);
    this._sunQuad.frustumCulled = false;
    this._sunScene = new THREE.Scene();
    this._sunScene.add(this._sunQuad);

    this._blurMat = new THREE.ShaderMaterial({
      name: 'BlacksiteGodrayBlur',
      defines: { SAMPLES: 32 },
      uniforms: {
        tDiffuse: { value: null },
        uSunPos: { value: new THREE.Vector2(0.5, 0.5) },
        uDensity: { value: 0.42 },
        uWeight: { value: 0.038 },
        uDecay: { value: 0.965 },
        uExposure: { value: 1.0 },
      },
      vertexShader: /* glsl */`
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tDiffuse;
        uniform vec2 uSunPos;
        uniform float uDensity, uWeight, uDecay, uExposure;
        varying vec2 vUv;
        void main() {
          vec2 uv = vUv;
          vec2 delta = ( uv - uSunPos ) * ( uDensity / float( SAMPLES ) );
          float illum = 1.0;
          vec3 acc = vec3( 0.0 );
          for ( int i = 0; i < SAMPLES; i++ ) {
            uv -= delta;
            acc += texture2D( tDiffuse, uv ).rgb * illum * uWeight;
            illum *= uDecay;
          }
          gl_FragColor = vec4( acc * uExposure, 1.0 );
        }`,
      depthTest: false,
      depthWrite: false,
      fog: false,
      toneMapped: false,
    });

    this._fsQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._blurMat);
    this._fsQuad.frustumCulled = false;
    this._fsScene = new THREE.Scene();
    this._fsScene.add(this._fsQuad);
    this._fsCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this._allocGodrayTargets(innerWidth, innerHeight);
  }

  _allocGodrayTargets(w, h) {
    const div = this._godrayDiv;
    const tw = Math.max(120, Math.floor(w / div));
    const th = Math.max(80, Math.floor(h / div));
    if (this._occRT && this._occRT.width === tw && this._occRT.height === th) return;

    this._occRT?.dispose();
    this._blurA?.dispose();
    this._blurB?.dispose();

    const opts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
    };
    this._occRT = new THREE.WebGLRenderTarget(tw, th, opts);
    this._blurA = new THREE.WebGLRenderTarget(tw, th, { ...opts, depthBuffer: false });
    this._blurB = new THREE.WebGLRenderTarget(tw, th, { ...opts, depthBuffer: false });
    this._occRT.texture.name = 'sky.occlusion';
    this._blurB.texture.name = 'sky.godrays';
    this.godrayTexture = this._blurB.texture;
    this._godrayCleared = false;
  }

  // =======================================================================================
  // 10. SCENE WIRING
  // =======================================================================================
  _applyToScene() {
    const scene = this.g.scene;
    if (!scene) return;

    scene.background = null;   // the dome IS the background (it carries the sun disc)

    const density = this.cfg.sky?.fogDensity ?? 0.0072;
    const base = this.fogBaseColor;
    if (scene.fog && scene.fog.isFogExp2) {
      scene.fog.color.copy(base);
      scene.fog.density = density;
    } else {
      scene.fog = new THREE.FogExp2(base.getHex(), density);
      scene.fog.color.copy(base);
    }

    scene.add(this.skyDome);
    scene.add(this.clouds);
    if (this.backdrop) scene.add(this.backdrop);
    scene.add(this.dust);
    scene.add(this.groundHaze);
  }

  async init() {
    const r = this.g.renderer?.renderer;
    if (r) {
      // never let an un-covered pixel read as pure black
      this._clearColor.copy(this.hazeMidColor);
      r.setClearColor(this._clearColor, 1);
      this._pmrem = new THREE.PMREMGenerator(r);
      this._updateEnvironment();
    }
    this._applySunToLights(true);
    this._updateDustScale();
  }

  // ---------------------------------------------------------------------------------------
  _updateEnvironment() {
    const r = this.g.renderer?.renderer;
    if (!r || !this._pmrem) return;

    // isolate the atmosphere so the IBL is pure sky, and re-centre the cloud dome on the
    // origin because the PMREM cube cameras sit at (0,0,0).
    const scene = this.g.scene;
    const domeParent = this.skyDome.parent;
    const cloudParent = this.clouds.parent;
    const cloudPos = this._v3a.copy(this.clouds.position);

    const envScene = this._envScene || (this._envScene = new THREE.Scene());
    this.clouds.position.set(0, 0, 0);
    this.clouds.updateMatrixWorld(true);
    envScene.add(this.skyDome);
    envScene.add(this.clouds);

    const prevDisc = this.skyUniforms.uSunDisc.value;
    const prevGlow = this.skyUniforms.uSunGlow.value;
    // the raw Preetham disc is ~1e3 — clamp it for the IBL so specular gets a bright sun
    // without the diffuse irradiance turning into a firefly.
    this.skyUniforms.uSunDisc.value = 0.0022;
    this.skyUniforms.uSunGlow.value = 3.0;

    let rt = null;
    try {
      rt = this._pmrem.fromScene(envScene, 0.02, 0.5, 20000, { size: 256 });
    } catch (e) {
      console.warn('[Sky] PMREM generation failed, falling back to flat ambient', e);
    }

    this.skyUniforms.uSunDisc.value = prevDisc;
    this.skyUniforms.uSunGlow.value = prevGlow;

    if (domeParent === scene || !domeParent) scene.add(this.skyDome); else domeParent.add(this.skyDome);
    if (cloudParent === scene || !cloudParent) scene.add(this.clouds); else cloudParent.add(this.clouds);
    this.clouds.position.copy(cloudPos);

    if (rt) {
      this._envRT?.dispose();
      this._envRT = rt;
      this.envMap = rt.texture;
      scene.environment = rt.texture;
      // Tuned against the golden-hour grade: any higher and the sky IBL milks out the
      // shadow side of every surface and clips the sunlit side. Read/override via
      // game.sky.envIntensity + game.sky.setEnvIntensity().
      scene.environmentIntensity = this.envIntensity;
      // A real IBL now supplies the sky/bounce term — dial the analytic ambient rig back so
      // the two do not double-count and flatten the shadows.
      this.g.renderer?.setAmbientIntensity?.(0.55);
    }
    this._lastEnvDir.copy(this.sunDirection);
    this._envDirty = false;
  }

  // The visible sun disc and the shadow-casting key light MUST agree exactly. Prefer the
  // Renderer's own API (it re-fits the shadow cascade and the fill rig from the direction);
  // fall back to poking the light transform if that API is not there.
  _applySunToLights(force) {
    const rend = this.g.renderer;
    if (!rend) return;

    if (typeof rend.setSunDirection === 'function') {
      const cur = rend.sunDir;
      if (force || !cur || cur.dot(this.sunDirection) < 0.999995) {
        rend.setSunDirection(this.sunDirection);
      }
      if (force) {
        rend.setSunColor?.(this.sunColor);
        this._appliedSunLight = rend.sun || null;
      }
    } else {
      const light = rend.sun || rend.sunLight || rend.keyLight || null;
      if (light?.isLight) {
        if (force || light !== this._appliedSunLight) {
          this._appliedSunLight = light;
          const d = light.position.length();
          this._sunLightDistance = d > 1 ? d : 150;
          light.color?.copy(this.sunColor);
          if (light.target) this.g.scene?.add?.(light.target);
        }
        const tgt = light.target ? light.target.position : this._v3b.set(0, 0, 0);
        light.position.copy(tgt).addScaledVector(this.sunDirection, this._sunLightDistance);
        light.target?.updateMatrixWorld?.();
        light.updateMatrixWorld?.();
      }
    }

    // Cascaded shadow maps, if the Renderer agent went that way.
    const csm = rend.csm || rend.shadows?.csm;
    if (csm?.lightDirection?.isVector3) {
      csm.lightDirection.copy(this.sunDirection).negate().normalize();
    }

    if (force) {
      const hemi = rend.hemi || rend.hemiLight;
      if (hemi?.isHemisphereLight) {
        hemi.color?.copy?.(this.skyColor);
        hemi.groundColor?.copy?.(this.groundColor);
      }
    }
  }

  _updateDustScale() {
    const r = this.g.renderer?.renderer;
    const cam = this.g.camera;
    if (!r || !cam) return;
    r.getDrawingBufferSize(this._dbSize || (this._dbSize = new THREE.Vector2()));
    const fov = (cam.fov || 68) * DEG;
    this.dustUniforms.uScale.value = (this._dbSize.y * 0.5) / Math.tan(fov * 0.5) * 0.0042;
  }

  // =======================================================================================
  // 11. PER-FRAME
  // =======================================================================================
  update(dt, t) {
    this._time = t;
    const cam = this.g.camera;
    if (!cam) return;

    this.cloudUniforms.uTime.value = t;
    this.dustUniforms.uTime.value = t;
    this.hazeUniforms.uTime.value = t;
    if (this.smokeUniforms) this.smokeUniforms.uTime.value = t;

    // atmosphere volumes ride with the camera
    this.clouds.position.copy(cam.position);
    this.clouds.updateMatrixWorld();
    this.groundHaze.position.x = cam.position.x;
    this.groundHaze.position.z = cam.position.z;
    this.groundHaze.updateMatrixWorld();

    this._updateDustScale();
    this._applySunToLights(false);

    // --- sun screen position / visibility ------------------------------------------------
    this.getSunScreenPosition(cam);
    const s = this.sunScreenPosition;
    this._fwd.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const inFront = this._fwd.dot(this.sunDirection) > 0.05;
    const margin = 0.42;
    const onScreen = inFront &&
      s.x > -margin && s.x < 1 + margin &&
      s.y > -margin && s.y < 1 + margin;
    this.sunVisible = onScreen;

    // smooth screen-edge falloff so the composite never pops
    const ex = Math.max(0, Math.max(-s.x, s.x - 1));
    const ey = Math.max(0, Math.max(-s.y, s.y - 1));
    const edge = 1 - THREE.MathUtils.clamp(Math.max(ex, ey) / margin, 0, 1);
    this._probeSunOcclusion(dt);

    // sunOcclusion is a smoothed line-of-sight probe: when a wall sits between the eye
    // and the sun there is no shaft to scatter, so the rays must go with it. Without this
    // the effect only depended on where the sun WOULD be on screen, and bloomed happily
    // on the inside of a building.
    const target = onScreen ? edge * edge * (1 - 0.92 * this.sunOcclusion) : 0;
    const k = 1 - Math.exp(-dt * 6.5);
    this.godrayIntensity += (target * 0.95 - this.godrayIntensity) * k;

    // --- god rays: self-drive only while nobody else is calling us -----------------------
    // (if PostFX drives us and then stops, we pick the job back up within a few frames)
    const externalRecent = (this.g.frame - this._lastExternalFrame) <= 4;
    if (this._godrayEnabled && !externalRecent) {
      const r = this.g.renderer?.renderer;
      if (r) this._renderGodraysInternal(r, cam);
    }

    if (this._envDirty) this._updateEnvironment();
  }

  // Cheap, optional line-of-sight probe. Uses Physics if it exposes a raycast; silently
  // disables itself on the first sign that the signature is not what we assumed.
  _probeSunOcclusion(dt) {
    const targetK = 1 - Math.exp(-dt * 5.0);
    let blocked = 0;
    if (this._sunOcclusionProbeOk && (this.g.frame & 7) === 0) {
      const phys = this.g.physics;
      // raycastAny is the boolean LOS query and bails on the first hit; raycast/rayTest
      // are the fallbacks for a Physics that never grew one.
      const fn = phys?.raycastAny || phys?.raycast || phys?.rayTest;
      if (typeof fn === 'function') {
        try {
          const o = this._v3c.copy(this.g.camera.position).addScaledVector(this.sunDirection, 0.6);
          const res = fn.call(phys, o, this.sunDirection, 400);
          if (res === true) blocked = 1;
          else if (res && typeof res === 'object') blocked = (res.hit === false) ? 0 : 1;
          this._lastProbe = blocked;
        } catch (e) {
          this._sunOcclusionProbeOk = false;
          this._lastProbe = 0;
        }
      } else {
        this._sunOcclusionProbeOk = false;
      }
    }
    const want = this._sunOcclusionProbeOk ? (this._lastProbe || 0) : 0;
    this.sunOcclusion += (want - this.sunOcclusion) * targetK;
  }

  /**
   * Public: render the occlusion + radial-blur passes right now. PostFX should call this
   * at the top of its render() so the buffer matches this frame's camera exactly.
   */
  renderGodrays(renderer) {
    const r = renderer || this.g.renderer?.renderer;
    if (!r) return this.godrayTexture;
    this._externallyDriven = true;
    this._lastExternalFrame = this.g.frame;
    this._renderGodraysInternal(r, this.g.camera);
    return this.godrayTexture;
  }

  _renderGodraysInternal(r, cam) {
    if (!this._godrayEnabled || !cam) return;
    if (this._godrayFrame === this.g.frame) return;
    this._godrayFrame = this.g.frame;

    if (!this.sunVisible || this.godrayIntensity <= 0.002) {
      if (!this._godrayCleared) {
        const prev = r.getRenderTarget();
        r.getClearColor(this._col);
        const pa = r.getClearAlpha();
        r.setRenderTarget(this._blurB);
        r.setClearColor(0x000000, 1);
        r.clear(true, false, false);
        r.setClearColor(this._col, pa);
        r.setRenderTarget(prev);
        this._godrayCleared = true;
      }
      return;
    }
    this._godrayCleared = false;

    const scene = this.g.scene;
    const prevTarget = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    const prevOverride = scene.overrideMaterial;
    const prevBackground = scene.background;
    const prevShadowAuto = r.shadowMap.autoUpdate;
    r.getClearColor(this._colB);
    const prevAlpha = r.getClearAlpha();

    this._hideAtmosphere();

    scene.background = null;
    scene.overrideMaterial = this._blackMat;
    r.shadowMap.autoUpdate = false;
    r.autoClear = false;

    r.setRenderTarget(this._occRT);
    r.setClearColor(0x000000, 1);
    r.clear(true, true, false);
    r.render(scene, cam);

    // sun disc, depth-tested against the silhouettes we just laid down
    scene.overrideMaterial = null;
    const dist = Math.min(600, (cam.far || 900) * 0.72);
    this._sunQuad.position.copy(cam.position).addScaledVector(this.sunDirection, dist);
    this._sunQuad.quaternion.copy(cam.quaternion);
    this._sunQuad.scale.setScalar(dist * 0.155);
    this._sunQuad.updateMatrixWorld();
    r.render(this._sunScene, cam);

    // two radial iterations: a tight one for definition, a long one for reach
    const bu = this._blurMat.uniforms;
    bu.uSunPos.value.copy(this.sunScreenPosition);

    bu.tDiffuse.value = this._occRT.texture;
    bu.uDensity.value = 0.55;
    bu.uWeight.value = 0.042;
    bu.uDecay.value = 0.962;
    bu.uExposure.value = 1.0;
    r.setRenderTarget(this._blurA);
    r.clear(true, false, false);
    r.render(this._fsScene, this._fsCam);

    bu.tDiffuse.value = this._blurA.texture;
    bu.uDensity.value = 0.20;
    bu.uWeight.value = 0.070;
    bu.uDecay.value = 0.975;
    bu.uExposure.value = 1.35;
    r.setRenderTarget(this._blurB);
    r.clear(true, false, false);
    r.render(this._fsScene, this._fsCam);

    // restore everything we touched
    scene.overrideMaterial = prevOverride;
    scene.background = prevBackground;
    r.shadowMap.autoUpdate = prevShadowAuto;
    r.autoClear = prevAutoClear;
    r.setClearColor(this._colB, prevAlpha);
    r.setRenderTarget(prevTarget);

    this._restoreAtmosphere();
  }

  _collect(o) {
    if (!o.visible) return;
    if (o === this.skyDome || o === this.clouds || o === this.dust || o === this.groundHaze) {
      this._pushHidden(o); return;
    }
    if (o.isPoints || o.isSprite || o.isLine || o.isLineSegments) { this._pushHidden(o); return; }
    const m = o.material;
    if (!m) return;
    if (Array.isArray(m)) {
      for (let i = 0; i < m.length; i++) if (m[i]?.transparent) { this._pushHidden(o); return; }
    } else if (m.transparent) {
      this._pushHidden(o);
    }
  }

  _pushHidden(o) {
    if (this._hiddenCount < this._hidden.length) this._hidden[this._hiddenCount++] = o;
  }

  _hideAtmosphere() {
    // Refresh the list at 2 Hz — traversal is cheap but not free, and the set is stable.
    if (this.g.frame - this._hiddenRefreshFrame > 30 || this._hiddenCount === 0) {
      this._hiddenRefreshFrame = this.g.frame;
      this._hiddenCount = 0;
      this.g.scene?.traverse(this._collect);
    }
    for (let i = 0; i < this._hiddenCount; i++) {
      const o = this._hidden[i];
      if (o) { o.__skyWasVisible = o.visible; o.visible = false; }
    }
  }

  _restoreAtmosphere() {
    for (let i = 0; i < this._hiddenCount; i++) {
      const o = this._hidden[i];
      if (o && o.__skyWasVisible !== undefined) o.visible = o.__skyWasVisible;
    }
  }

  /** Allocation-free. Returns the shared Vector2 in 0..1 screen uv. */
  getSunScreenPosition(camera) {
    const cam = camera || this.g.camera;
    if (!cam) return this.sunScreenPosition;
    const v = this._v3a.copy(this.sunDirection).multiplyScalar(SUN_DISTANCE).add(cam.position);
    v.project(cam);
    this.sunScreenPosition.set(v.x * 0.5 + 0.5, v.y * 0.5 + 0.5);
    return this.sunScreenPosition;
  }

  getShimmerUniforms() {
    return (this._shimmerUniforms ||= {
      tShimmer: { value: this.shimmerTexture },
      uTime: { value: 0 },
      uStrength: { value: this.shimmer.strength },
      uScale: { value: this.shimmer.scale },
      uHorizon: { value: this.shimmer.horizon },
    });
  }

  // =======================================================================================
  // 12. AUTHORING / DEBUG
  // =======================================================================================
  setTimeOfDay(hours) {
    this.cfg.sky.timeOfDay = hours;
    this._solveSun();
    this._deriveAtmosphereColors();
    this._installHeightFog();
    this._rebuildFogPrograms();

    this.skyUniforms.sunPosition.value.copy(this.sunDirection);
    const scene = this.g.scene;
    if (scene?.fog) scene.fog.color.copy(this.fogBaseColor);
    this._clearColor.copy(this.hazeMidColor);
    this.g.renderer?.renderer?.setClearColor(this._clearColor, 1);

    this._applySunToLights(true);
    this._envDirty = true;
  }

  setFogDensity(d) {
    if (this.g.scene?.fog) this.g.scene.fog.density = d;
    this.cfg.sky.fogDensity = d;
  }

  /** Strength of the sky IBL on every PBR surface. Live, no rebake. */
  setEnvIntensity(v) {
    this.envIntensity = v;
    if (this.g.scene) this.g.scene.environmentIntensity = v;
  }

  resize(w, h) {
    this._allocGodrayTargets(w, h);
    this._updateDustScale();
  }

  dispose() {
    this._occRT?.dispose();
    this._blurA?.dispose();
    this._blurB?.dispose();
    this._envRT?.dispose();
    this._pmrem?.dispose();
    this.cloudNoise?.dispose();
    this.shimmerTexture?.dispose();
    this.dustSprite?.dispose();
    this.skyDome?.material?.dispose();
    this.skyDome?.geometry?.dispose();
    this.clouds?.material?.dispose();
    this.clouds?.geometry?.dispose();
    this.dust?.material?.dispose();
    this.dust?.geometry?.dispose();
    this.groundHaze?.material?.dispose();
    this.groundHaze?.geometry?.dispose();
    this._blackMat?.dispose();
    this._blurMat?.dispose();
    if (this._bgDisposables) {
      for (let i = 0; i < this._bgDisposables.length; i++) this._bgDisposables[i]?.dispose?.();
      this._bgDisposables.length = 0;
    }
  }
}

export default Sky;
