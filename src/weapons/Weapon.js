// ============================================================================
// BLACKSITE — src/weapons/Weapon.js
// Owner: Weapon agent.  First-person viewmodel: procedural AR-pattern carbine,
// overlay render pass, ADS, recoil, sway, reload/inspect animation, muzzle
// flash, shell ejection and fire control.
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators — read this)
// ---------------------------------------------------------------------------
//   game.weapon.setAds(bool)              force ADS (debug critic calls this)
//   game.weapon.ads                       bool, requested ADS state
//   game.weapon.adsT                      0..1 eased ADS blend
//   game.weapon.mag / .reserve / .magSize ammo state
//   game.weapon.reloading / .firing       bool
//   game.weapon.spread                    current cone half-angle, radians
//   game.weapon.crosshairSpread           0..1 normalised, for the HUD reticle
//   game.weapon.name                      'M4 CARBINE'
//   game.weapon.muzzleWorld               THREE.Vector3, live, world-space muzzle
//   game.weapon.viewmodelScene            THREE.Scene   overlay scene
//   game.weapon.viewmodelCamera           THREE.PerspectiveCamera (origin, identity)
//   game.weapon.renderViewmodel(renderer) draw the overlay NOW (clears depth first)
//   game.weapon.autoRender                bool; set false if you drive the pass
//   game.weapon.reload() / .fire() / .inspect()
//
// PostFX: either call renderViewmodel(renderer) yourself after compositing the
// world (the module then stops self-driving within one frame), or set
// weapon.autoRender = false and render viewmodelScene/viewmodelCamera by hand.
// If nobody does anything, the module renders itself from scene.onAfterRender
// with autoClear disabled and a depth clear, so the gun is never occluded.
//
// Bus: emits 'shot' {origin,dir,weapon}, 'ammo' {mag,reserve},
//      'reload' {phase:'start'|'magout'|'magin'|'end'}, 'ads' {active}.
// ============================================================================

import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries, mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

/* ------------------------------------------------------------------ *
 *  math
 * ------------------------------------------------------------------ */

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const sstep = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0 || 1e-6)); return t * t * (3 - 2 * t); };
const smoother = t => t * t * t * (t * (t * 6 - 15) + 10);

// snappy arrival with a small overshoot — never a linear lerp
function easeOutBack(t, k = 0.95) {
  const p = t - 1;
  return 1 + (k + 1) * p * p * p + k * p * p;
}
function easeOutCubic(t) { const p = 1 - t; return 1 - p * p * p; }
function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) * 0.5; }

// cheap deterministic hash used only at build time
function fhash(x, y, z) {
  const n = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return n - Math.floor(n);
}

// exponential smoothing that is correct at any framerate
const approach = (cur, goal, rate, dt) => goal + (cur - goal) * Math.exp(-rate * dt);

/* ------------------------------------------------------------------ *
 *  Per-vertex SURFACE patch.
 *
 *  A single roughness value over a whole gun is what makes procedural
 *  hard-surface read as injection-moulded plastic: aluminium, phosphate,
 *  polymer and rubber all end up with the same highlight. This patch feeds
 *  the baked `aSurf` attribute (x = edge-wear 0..1, y = macro grime -0.5..0.5)
 *  into the standard/physical fragment shader:
 *
 *    - worn edges polish DOWN in roughness and UP in metalness, so every
 *      chamfer catches a tight bright specular line of bare metal;
 *    - the macro grime field pushes roughness both ways at low frequency, so
 *      a flat panel is never one uniform sheen.
 *
 *  uSurfK = (wearRough, wearMetal, grimeRough, unused)
 * ------------------------------------------------------------------ */

const SURF_VERT_PARS = /* glsl */`
attribute vec2 aSurf;
varying vec2 vSurf;
`;
const SURF_FRAG_PARS = /* glsl */`
varying vec2 vSurf;
uniform vec4 uSurfK;
uniform vec4 uAlbK;
`;
/**
 * Albedo LEVELS remap — the single most important thing in this file.
 *
 * Materials' procedural gun surfaces are authored dark: the polymer plate is
 * 0x2e3032 (~0.027 linear) and the phosphate is 0x35383b (~0.036). A hex tint
 * can only ever multiply that DOWN, so tinting 'polymer' with a coyote-tan hex
 * produces a near-black brown, and the whole weapon collapses into one
 * two-tone grey/black plastic no matter how the roughness is authored. That is
 * exactly what the critic saw.
 *
 * So: rescale. gain (uAlbK.w) stretches the map's own variation, the bias
 * (uAlbK.xyz) lifts the family to its real linear albedo, and the cap
 * (uSurfK.w) stops the bright rub-through blotches in the metal maps from
 * flaring — on a metal, albedo IS the specular colour.
 *
 * The bias is modulated by vColor so the baked AO / grime / edge-wear survives
 * the lift: a flat lift would have flattened every bake in this file.
 */
const SURF_ALBEDO = /* glsl */`
	diffuseColor.rgb = min( diffuseColor.rgb * uAlbK.w + uAlbK.xyz * vColor.rgb, vec3( uSurfK.w ) );
`;
const SURF_ROUGH = /* glsl */`
{
	float bsW = clamp( vSurf.x, 0.0, 1.0 );
	roughnessFactor = roughnessFactor * ( 1.0 - uSurfK.x * bsW ) + uSurfK.z * vSurf.y;
	roughnessFactor = clamp( roughnessFactor, 0.035, 1.0 );
}
`;
const SURF_METAL = /* glsl */`
{
	float bsW = clamp( vSurf.x, 0.0, 1.0 );
	metalnessFactor = clamp( metalnessFactor + uSurfK.y * bsW, 0.0, 1.0 );
}
`;

/* ------------------------------------------------------------------ *
 *  keyframe tracks:  [{ t, v:[...], e? }]  — smootherstep by default
 * ------------------------------------------------------------------ */

function sampleTrack(keys, t, out) {
  const n = keys.length;
  const first = keys[0], last = keys[n - 1];
  if (t <= first.t) { for (let i = 0; i < out.length; i++) out[i] = first.v[i]; return out; }
  if (t >= last.t) { for (let i = 0; i < out.length; i++) out[i] = last.v[i]; return out; }
  let k = 0;
  while (k < n - 2 && t > keys[k + 1].t) k++;
  const a = keys[k], b = keys[k + 1];
  let u = (t - a.t) / Math.max(1e-6, b.t - a.t);
  if (b.e === 'lin') { /* raw */ }
  else if (b.e === 'back') u = easeOutBack(u, 1.35);
  else if (b.e === 'out') u = easeOutCubic(u);
  else u = smoother(u);
  for (let i = 0; i < out.length; i++) out[i] = a.v[i] + (b.v[i] - a.v[i]) * u;
  return out;
}

/* ------------------------------------------------------------------ *
 *  Geometry rig — bakes wear/AO into vertex colours, merges by material
 * ------------------------------------------------------------------ */

const _m4 = new THREE.Matrix4();
const _q4 = new THREE.Quaternion();
const _e3 = new THREE.Euler();
const _v3a = new THREE.Vector3();
const _v3b = new THREE.Vector3();
const _v3c = new THREE.Vector3();

/**
 * Vertex-colour bake. Runs in the primitive's OWN local space, before the part
 * transform, so the axis-aligned normal test is a valid curvature proxy:
 *   flat face  -> max(|n|) == 1 -> edge 0
 *   45deg fillet -> max(|n|) ~ 0.71 -> edge 0.29
 * Worn metal shows through on the fillets of high-touch parts; undersides get
 * a contact-shadow darkening; upward faces catch a hint of dust.
 * Colours are float and may exceed 1 — that is deliberate.
 */
function bakeSurface(geo, o) {
  const pos = geo.attributes.position;
  const nor = geo.attributes.normal;
  const n = pos.count;
  const col = new Float32Array(n * 3);
  const srf = new Float32Array(n * 2);
  const wear = o.wear ?? 0;
  const aoK = o.ao ?? 1;
  const base = o.shade ?? 1;
  // Worn metal is a MULTIPLIER over the family's albedo floor, so when the
  // floor drops (a real anodised receiver is dark) the rub-through has to climb
  // to keep the chamfer reading as a bright machined line instead of dying into
  // the body value. This ratio is what makes a dark gun still look faceted.
  const wr = o.wearColor ? o.wearColor[0] : 2.06;
  const wg = o.wearColor ? o.wearColor[1] : 1.94;
  const wb = o.wearColor ? o.wearColor[2] : 1.76;
  const eLo = o.edgeLo ?? 0.06, eHi = o.edgeHi ?? 0.40;
  const gK = o.grime ?? 1;                       // macro breakup strength
  const dK = o.dust ?? 1;                        // warm dust settling on up-faces
  const tr = o.tintv ? o.tintv[0] : 1;
  const tg = o.tintv ? o.tintv[1] : 1;
  const tb = o.tintv ? o.tintv[2] : 1;
  for (let i = 0; i < n; i++) {
    const px = pos.getX(i), py = pos.getY(i), pz = pos.getZ(i);
    const nx = nor ? nor.getX(i) : 0, ny = nor ? nor.getY(i) : 1, nz = nor ? nor.getZ(i) : 0;
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    const mx = ax > ay ? (ax > az ? ax : az) : (ay > az ? ay : az);
    const e = sstep(eLo, eHi, 1 - mx);
    const h = fhash(px * 640, py * 611, pz * 593);
    const w = clamp01(wear * e * (0.18 + 1.25 * h));
    // coarse field: reads as blotchy handling grime / uneven finish, not noise
    const gl = fhash(px * 41.3 + 3.1, py * 37.7 - 1.7, pz * 33.1 + 5.9);
    const gl2 = fhash(px * 12.9 - 7.3, py * 11.1 + 2.9, pz * 9.7 - 4.1);
    const gm = (gl * 0.62 + gl2 * 0.38 - 0.5) * gK;
    const down = ny < 0 ? -ny : 0;
    const up = ny > 0 ? ny : 0;
    const sh = base
      * (1 - 0.27 * down * aoK + 0.050 * up)
      * (1 - 0.155 * gK * clamp01(gl2 * 1.45 - 0.40));
    const o3 = i * 3;
    // dust is warm and settles on horizontal faces — a cool-blue gun in a
    // golden-hour street should never be neutral on top
    const du = up * up * dK;
    col[o3] = lerp(sh, wr, w) * tr * (1 + 0.105 * du);
    col[o3 + 1] = lerp(sh, wg, w) * tg * (1 + 0.062 * du);
    col[o3 + 2] = lerp(sh, wb, w) * tb * (1 - 0.030 * du);
    const s2 = i * 2;
    srf[s2] = w;
    srf[s2 + 1] = gm;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSurf', new THREE.BufferAttribute(srf, 2));
}

function toMatrix(o) {
  const r = o.r || null;
  if (o.q) _q4.copy(o.q);
  else {
    if (r) _e3.set(r[0] || 0, r[1] || 0, r[2] || 0, 'XYZ'); else _e3.set(0, 0, 0, 'XYZ');
    _q4.setFromEuler(_e3);
  }
  const p = o.p || null;
  _v3a.set(p ? (p[0] || 0) : 0, p ? (p[1] || 0) : 0, p ? (p[2] || 0) : 0);
  const s = o.s;
  if (s === undefined) _v3b.set(1, 1, 1);
  else if (typeof s === 'number') _v3b.set(s, s, s);
  else _v3b.set(s[0], s[1], s[2]);
  return _m4.compose(_v3a, _q4, _v3b);
}

class Rig {
  constructor() { this.buckets = new Map(); this.tris = 0; }

  /** add(materialKey, geometry, { p, r, s, wear, ao, shade, wearColor }) */
  add(key, geo, o = {}) {
    let g = geo;
    if (!g.index) g = mergeVertices(g, 1e-5);
    for (const k of Object.keys(g.attributes)) {
      if (k !== 'position' && k !== 'normal' && k !== 'uv') g.deleteAttribute(k);
    }
    if (!g.attributes.normal) g.computeVertexNormals();
    if (!g.attributes.uv) {
      g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    }
    bakeSurface(g, o);
    g.applyMatrix4(toMatrix(o));
    g.setAttribute('uv1', new THREE.BufferAttribute(g.attributes.uv.array.slice(), 2));
    let arr = this.buckets.get(key);
    if (!arr) { arr = []; this.buckets.set(key, arr); }
    arr.push(g);
    this.tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
    return this;
  }

  /** Merge each bucket into one mesh and parent it. Returns the parent. */
  build(parent, matFor, name) {
    for (const [key, arr] of this.buckets) {
      let geo;
      if (arr.length === 1) geo = arr[0];
      else {
        geo = mergeGeometries(arr, false);
        if (!geo) { geo = arr[0]; console.warn('[Weapon] merge failed for', key); }
        else for (const g of arr) g.dispose();
      }
      geo.computeBoundingSphere();
      const m = new THREE.Mesh(geo, matFor(key));
      m.name = (name || 'part') + ':' + key;
      m.castShadow = true;
      m.receiveShadow = true;
      m.matrixAutoUpdate = false;
      m.updateMatrix();
      parent.add(m);
    }
    this.buckets.clear();
    return parent;
  }
}

/* ------------------------------------------------------------------ *
 *  primitive helpers
 * ------------------------------------------------------------------ */

// Every hard edge gets a real chamfer band. A 90-degree edge dies into shadow;
// a 0.6-2.6mm chamfer catches a specular line and is what separates a machined
// part from a primitive. seg=1 gives one chamfer facet, seg=2 a two-facet round.
const rbox = (w, h, d, r = 0.0026, seg = 1) =>
  new RoundedBoxGeometry(w, h, d, seg, Math.min(r, Math.min(w, Math.min(h, d)) * 0.46));
// hero parts — the big panels the eye lands on get a proper rolled edge
const hbox = (w, h, d, r = 0.0034) => rbox(w, h, d, r, 2);
const cyl = (rt, rb, h, seg = 14, open = false) =>
  new THREE.CylinderGeometry(rt, rb, h, seg, 1, open);
// a cylinder lying along Z (bore axis)
const tube = (rt, rb, len, seg = 14, open = false) => {
  const g = cyl(rt, rb, len, seg, open);
  g.rotateX(Math.PI * 0.5);
  return g;
};

/* ------------------------------------------------------------------ *
 *  Recoil pattern — a learned, repeatable spray with per-shot jitter.
 *  Horizontal: signed multiples of the base horizontal impulse.  The shape is
 *  the classic AR walk — dead straight for three, then a left-right-left
 *  pendulum that widens as the barrel heats.
 * ------------------------------------------------------------------ */

const H_PATTERN = [
  0.00, 0.04, -0.06, 0.18, 0.34, 0.30, 0.10, -0.16, -0.42, -0.55,
  -0.46, -0.20, 0.14, 0.44, 0.62, 0.58, 0.30, -0.08, -0.44, -0.68,
  -0.72, -0.50, -0.14, 0.28, 0.60, 0.76, 0.70, 0.42, 0.04, -0.34,
];
// vertical climb multiplier: violent for the first three, then a plateau
const V_PATTERN = [
  1.00, 1.02, 0.96, 0.84, 0.76, 0.71, 0.68, 0.66, 0.65, 0.64,
  0.64, 0.65, 0.66, 0.66, 0.67, 0.67, 0.68, 0.68, 0.69, 0.69,
  0.70, 0.70, 0.71, 0.71, 0.72, 0.72, 0.73, 0.73, 0.74, 0.74,
];

/* ------------------------------------------------------------------ *
 *  Poses.  All values are viewmodel-camera space (metres / radians).
 *  The camera sits at the origin looking down -Z, so these ARE the
 *  screen-space placement of the gun.
 * ------------------------------------------------------------------ */

// hip: low-right, muzzle toed in toward the screen centre, slight outboard cant.
//
// Round-3 reframe. The old pose failed the single most important test a
// viewmodel has: a viewer could not name the parts. Three specific reasons,
// all geometric, all fixed here:
//
//   1. y was -0.126 and z only -0.458, so the pistol grip, the firing hand and
//      the bottom half of the magazine were BELOW the frame edge. A carbine
//      with no visible grip, no visible trigger hand and a clipped magazine
//      reads as a slab with a box on it. Raising to -0.085 and pushing to
//      -0.500 brings the whole fire-control group and the full magazine into
//      frame, and shrinks the stock at the same time.
//   2. yaw was only 0.206, so the 130mm of exposed barrel and the muzzle brake
//      projected onto almost exactly the same pixels as the handguard mouth —
//      the lateral travel and the perspective shrink cancelled. At 0.290 the
//      brake clears the handguard by ~40px and there is a readable barrel.
//   3. x was 0.122, which parked the stock and butt pad in open frame at the
//      bottom-right where they read as a black box hanging off the weapon.
//      At 0.255 the butt runs off the right-hand edge the way it does in every
//      shipped shooter, and what is left in frame is receiver → magwell →
//      grip → handguard → muzzle, in that order, left to right.
const POSE_HIP = { p: [0.2450, -0.0850, -0.5000], r: [0.0200, 0.3800, 0.1060] };
// sprint: canted hard down and outboard, receiver swung across the body
const POSE_SPRINT = { p: [0.1500, -0.1720, -0.3600], r: [0.2600, 0.6200, -0.4400] };
// low-ready when the weapon is holstered-ish (unused hook, kept for collaborators)
const POSE_LOW = { p: [0.1300, -0.1900, -0.2600], r: [0.4200, 0.1400, 0.0600] };

/* ------------------------------------------------------------------ *
 *  Animation tracks.  [px,py,pz, rx,ry,rz] additive over the base pose.
 * ------------------------------------------------------------------ */

// tactical reload (rounds still in the chamber) — 1.95 s
const RELOAD_TAC = [
  { t: 0.00, v: [0, 0, 0, 0, 0, 0] },
  { t: 0.17, v: [0.014, -0.034, 0.030, 0.155, -0.300, 0.360], e: 'out' },
  { t: 0.44, v: [0.017, -0.046, 0.034, 0.190, -0.350, 0.430] },
  { t: 0.86, v: [0.013, -0.041, 0.028, 0.150, -0.290, 0.360] },
  { t: 1.06, v: [0.007, -0.030, 0.019, 0.095, -0.190, 0.230] },
  { t: 1.24, v: [0.004, -0.016, 0.010, 0.048, -0.100, 0.120], e: 'back' },
  { t: 1.44, v: [0.006, -0.026, 0.015, 0.078, -0.150, 0.180] },
  { t: 1.72, v: [0.002, -0.009, 0.005, 0.026, -0.050, 0.060] },
  { t: 1.95, v: [0, 0, 0, 0, 0, 0], e: 'back' },
];

// empty reload — mag change plus a charging-handle cycle. 2.62 s
const RELOAD_EMPTY = [
  { t: 0.00, v: [0, 0, 0, 0, 0, 0] },
  { t: 0.17, v: [0.014, -0.034, 0.030, 0.155, -0.300, 0.360], e: 'out' },
  { t: 0.46, v: [0.018, -0.049, 0.036, 0.200, -0.360, 0.445] },
  { t: 0.90, v: [0.013, -0.042, 0.029, 0.152, -0.292, 0.362] },
  { t: 1.10, v: [0.007, -0.031, 0.020, 0.098, -0.192, 0.234] },
  { t: 1.30, v: [0.004, -0.017, 0.011, 0.050, -0.104, 0.126], e: 'back' },
  { t: 1.52, v: [-0.004, -0.024, 0.008, 0.062, 0.235, 0.098] },  // roll left, expose the handle
  { t: 1.78, v: [-0.007, -0.028, 0.010, 0.070, 0.300, 0.110] },
  { t: 1.98, v: [-0.005, -0.022, 0.014, 0.058, 0.255, 0.090], e: 'lin' },
  { t: 2.10, v: [-0.002, -0.026, 0.020, 0.070, 0.190, 0.070], e: 'back' },
  { t: 2.36, v: [0.001, -0.010, 0.006, 0.024, 0.070, 0.026] },
  { t: 2.62, v: [0, 0, 0, 0, 0, 0], e: 'back' },
];

// support hand: off the handguard, down to the pouch for a fresh mag, and back
const HANDL_TAC = [
  { t: 0.00, v: [0, 0, 0, 0, 0, 0] },
  { t: 0.24, v: [-0.030, -0.120, 0.190, 0.34, 0.52, 0.30], e: 'out' },
  { t: 0.62, v: [-0.034, -0.135, 0.205, 0.38, 0.56, 0.33] },
  { t: 0.92, v: [-0.016, -0.058, 0.096, 0.16, 0.26, 0.15] },
  { t: 1.10, v: [0, 0, 0, 0, 0, 0], e: 'back' },
  { t: 1.95, v: [0, 0, 0, 0, 0, 0] },
];

// same, plus a trip up to the charging handle to run the bolt
const HANDL_EMPTY = [
  { t: 0.00, v: [0, 0, 0, 0, 0, 0] },
  { t: 0.26, v: [-0.030, -0.122, 0.192, 0.34, 0.52, 0.30], e: 'out' },
  { t: 0.66, v: [-0.034, -0.138, 0.208, 0.38, 0.56, 0.33] },
  { t: 0.98, v: [-0.016, -0.058, 0.096, 0.16, 0.26, 0.15] },
  { t: 1.18, v: [0, 0, 0, 0, 0, 0], e: 'back' },
  { t: 1.40, v: [-0.010, 0.020, 0.130, -0.10, 0.18, 0.08], e: 'out' },
  { t: 1.62, v: [-0.014, 0.052, 0.252, -0.20, 0.30, 0.14] },
  { t: 1.90, v: [-0.014, 0.056, 0.302, -0.22, 0.32, 0.15] },
  { t: 2.04, v: [-0.012, 0.046, 0.238, -0.18, 0.27, 0.12], e: 'lin' },
  { t: 2.30, v: [0, 0, 0, 0, 0, 0], e: 'back' },
  { t: 2.62, v: [0, 0, 0, 0, 0, 0] },
];

// inspect — bring it in, roll to show the left side, tip the muzzle, settle back
const INSPECT = [
  { t: 0.00, v: [0, 0, 0, 0, 0, 0] },
  { t: 0.30, v: [-0.036, 0.028, 0.062, 0.075, 0.620, 0.120], e: 'out' },
  { t: 0.85, v: [-0.042, 0.032, 0.070, 0.060, 0.690, 0.150] },
  { t: 1.35, v: [-0.010, 0.014, 0.052, -0.320, -0.180, -0.560] },
  { t: 1.95, v: [-0.014, 0.018, 0.056, -0.360, -0.230, -0.610] },
  { t: 2.35, v: [-0.006, 0.010, 0.030, -0.120, 0.120, -0.200] },
  { t: 2.70, v: [0, 0, 0, 0, 0, 0], e: 'back' },
];

/* ================================================================== *
 *  Weapon
 * ================================================================== */

export class Weapon {

  constructor(game) {
    this.g = game;
    const cfg = game.config.weapon;

    this.name = 'M4 CARBINE';
    this.magSize = cfg.magSize;
    this.mag = cfg.magSize;
    this.reserve = cfg.reserve;
    this.rpm = cfg.rpm;
    this.shotInterval = 60 / Math.max(1, cfg.rpm);

    // ---- state ------------------------------------------------------
    this.ads = false;
    this.adsT = 0;             // eased 0..1
    this._adsRaw = 0;          // linear 0..1 driver
    this.firing = false;
    this.triggerHeld = false;
    this.reloading = false;
    this.inspecting = false;
    this.sprintT = 0;
    this.enabled = true;
    this.autoRender = true;

    this._fireTimer = 0;
    this._shotIndex = 0;
    this._sinceShot = 99;
    this._animT = 0;
    this._animDur = 0;
    this._animTrack = null;
    this._animKind = '';        // 'reload' | 'inspect'
    this._emptyReload = false;
    this._phaseIdx = 0;
    this._phases = null;

    // spring-damper recoil channels: [value, velocity]
    this._rKick = [0, 0];       // rearward translation, m
    this._rRise = [0, 0];       // pitch, rad
    this._rYaw = [0, 0];       // yaw, rad
    this._rRoll = [0, 0];       // roll, rad
    this._rLift = [0, 0];       // vertical translation, m

    // sway / lag
    this._swayX = 0; this._swayY = 0;
    this._swayVX = 0; this._swayVY = 0;
    this._prevYaw = 0; this._prevPitch = 0;
    this._bobPhase = 0;
    this._speed = 0;
    this._prevPos = new THREE.Vector3();
    this._havePrev = false;

    // moving parts
    this._boltZ = 0;            // 0 = forward, 1 = fully back
    this._boltT = 99;
    this._chZ = 0;              // charging handle travel 0..1
    this._dust = 0;             // dust cover open 0..1
    this._triggerT = 0;
    this._magVis = 1;

    // muzzle flash
    this._flashT = 99;
    this._flashDur = 0.052;
    this._flashRoll = 0;
    this._flashScale = 1;

    this.spread = 0.012;
    this.crosshairSpread = 0.25;
    this.muzzleWorld = new THREE.Vector3();

    // ---- overlay scene ---------------------------------------------
    this.viewmodelScene = new THREE.Scene();
    this.viewmodelScene.name = 'viewmodel';
    this.viewmodelScene.matrixAutoUpdate = true;
    this.viewmodelCamera = new THREE.PerspectiveCamera(
      cfg.viewmodelFov, (innerWidth || 1920) / (innerHeight || 1080), 0.0032, 8,
    );
    this.viewmodelCamera.name = 'viewmodelCamera';

    this.root = new THREE.Group();       // the gun + hands
    this.root.name = 'weaponRoot';
    this.viewmodelScene.add(this.root);

    // ---- scratch (zero allocation in update) ------------------------
    this._sPos = new THREE.Vector3();
    this._sRot = new THREE.Euler(0, 0, 0, 'XYZ');
    this._anim = [0, 0, 0, 0, 0, 0];
    this._handTmp = [0, 0, 0, 0, 0, 0];
    this._handDirty = false;
    this._afterFrame = -1;
    this._tmpV = new THREE.Vector3();
    this._tmpV2 = new THREE.Vector3();
    this._retV = new THREE.Vector3();
    this._tmpQ = new THREE.Quaternion();
    this._tmpM = new THREE.Matrix4();
    this._sunView = new THREE.Vector3(0.4, 0.7, 0.4);

    // pooled vectors for the 'shot' event (bus consumers read them synchronously,
    // but a small ring keeps a listener that defers by a frame honest)
    this._shotRing = [];
    for (let i = 0; i < 8; i++) this._shotRing.push({ origin: new THREE.Vector3(), dir: new THREE.Vector3(), weapon: this });
    this._shotRingI = 0;

    // Live-tunable copy of the hip pose. Kept as an instance field so the pose
    // can be nudged from the console without a rebuild while framing is being
    // dialled in; update() reads this, never the module constant.
    this.poseHip = { p: POSE_HIP.p.slice(), r: POSE_HIP.r.slice() };

    this._extUntilFrame = -1;
    this._inHook = false;
    this._hookAttempts = 0;
    this._bound = false;
    this._camFov = game.config.weapon.fovHip;
  }

  // ==================================================================
  //  INIT
  // ==================================================================

  async init() {
    this._makeMaterials();
    this._makeFlashTextures();
    this._buildWeapon();
    this._buildHands();
    this._buildMuzzleFlash();
    this._buildWorldFx();
    this._buildLightRig();
    this._bindInput();
    this._hookRender();

    // Derive the exact ADS pose from the optic we actually built, so the sight
    // is perfectly centred no matter how the model is retuned.
    //
    // Eye relief was 288mm, which is not eye relief — it is arm's length. Two
    // things went wrong at that distance. The optic subtended only ~18% of the
    // half-frame, so a physically correct 2-MOA dot landed on two pixels and
    // the sight picture read as an empty hoop; and the butt pad ended up 28mm
    // in front of the lens, filling the bottom third of the aimed frame with a
    // defocused brown slab.  75mm is a real cheek weld: the housing now fills
    // ~29% of the half-frame, and everything behind the ejection port falls
    // BEHIND the camera exactly as it does when a stock is in your shoulder.
    this.ADS_RELIEF = 0.1880;
    this.POSE_ADS = {
      p: [-this.opticCentre.x, -this.opticCentre.y, -this.ADS_RELIEF - this.opticCentre.z],
      r: [0, 0, 0],
    };

    this.g.bus.emit('ammo', { mag: this.mag, reserve: this.reserve });
    this.resize(innerWidth, innerHeight);
    this.update(0, 0);
  }

  /* ---------------------------------------------------------------- *
   *  Materials — every one is a private variant so the viewmodel never
   *  shares a program-state with the fogged world scene.
   * ---------------------------------------------------------------- */

  _mk(key, opts) {
    const M = this.g.materials;
    let m;
    if (M?.make) { try { m = M.make(key, opts); } catch (e) { m = null; } }
    if (!m) {
      m = new THREE.MeshStandardMaterial({
        color: opts?.tint ?? 0x8a8a8a,
        roughness: opts?.roughness ?? 0.7,
        metalness: opts?.metalness ?? 0.5,
      });
    }
    m.vertexColors = true;
    m.dithering = true;
    return m;
  }

  /** Promote a standard material to physical so brushed metal gets a real
   *  stretched highlight. Falls back silently if anything is unavailable. */
  _aniso(std, strength, rot) {
    if (!THREE.MeshPhysicalMaterial) return std;
    try {
      const p = new THREE.MeshPhysicalMaterial({
        name: std.name + ':aniso',
        map: std.map, normalMap: std.normalMap,
        roughnessMap: std.roughnessMap, metalnessMap: std.metalnessMap, aoMap: std.aoMap,
        color: std.color.clone(),
        roughness: std.roughness, metalness: std.metalness,
        envMapIntensity: std.envMapIntensity,
        aoMapIntensity: std.aoMapIntensity,
        vertexColors: true, dithering: true,
        anisotropy: strength, anisotropyRotation: rot,
      });
      p.normalScale.copy(std.normalScale);
      if (std.onBeforeCompile) p.onBeforeCompile = std.onBeforeCompile;
      if (std.customProgramCacheKey) p.customProgramCacheKey = std.customProgramCacheKey;
      p.userData.bsSurface = std.userData.bsSurface;
      p.userData.bsDetail = std.userData.bsDetail;
      std.dispose();
      return p;
    } catch (e) { return std; }
  }

  /**
   * Wire the baked `aSurf` attribute into the shader so roughness and metalness
   * vary per-vertex. Chains any existing onBeforeCompile (Materials installs a
   * detail-normal patch) and takes its OWN program cache key — Materials uses a
   * single constant key for every material it builds, so a patched variant that
   * kept that key would be handed a program compiled without this code.
   */
  _patchSurf(mat, k) {
    if (!mat || mat.userData.bsSurfPatched) return mat;
    const prev = (typeof mat.onBeforeCompile === 'function') ? mat.onBeforeCompile : null;
    const K = new THREE.Vector4(k?.wearRough ?? 0.55, k?.wearMetal ?? 0.30, k?.grimeRough ?? 0.10, k?.albedoCap ?? 8.0);
    const b = k?.bias || [0, 0, 0];
    const A = new THREE.Vector4(b[0], b[1], b[2], k?.gain ?? 1.0);
    mat.userData.bsSurfK = K;
    mat.userData.bsAlbK = A;
    mat.userData.bsSurfPatched = true;
    const tag = 'bs-wpn-surf-2:' + (mat.userData.bsDetail ? 'd' : 'n');
    mat.onBeforeCompile = function (shader, renderer) {
      if (prev) { try { prev.call(this, shader, renderer); } catch (e) { /* keep the base material */ } }
      shader.uniforms.uSurfK = { value: K };
      shader.uniforms.uAlbK = { value: A };
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\n' + SURF_VERT_PARS)
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n\tvSurf = aSurf;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\n' + SURF_FRAG_PARS)
        .replace('#include <map_fragment>', '#include <map_fragment>\n' + SURF_ALBEDO)
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\n' + SURF_ROUGH)
        .replace('#include <metalnessmap_fragment>', '#include <metalnessmap_fragment>\n' + SURF_METAL);
    };
    mat.customProgramCacheKey = () => tag;
    return mat;
  }

  _makeMaterials() {
    // ---------------------------------------------------------------
    // Four substances that must never be mistaken for one another:
    //   1. hard-anodised aluminium  — dark, satin, TIGHT specular, low env
    //   2. phosphated steel         — darker, rougher, warmer, no gloss
    //   3. bare/bright steel        — polished, near-mirror, high env
    //   4. FDE polymer + rubber     — tan, matte, BROAD dull response, no env
    // The roughness numbers are multipliers over the Materials ORM map, and the
    // per-vertex wear term then polishes every chamfer back down toward bare
    // metal, so the two families never converge on one sheen.
    // ---------------------------------------------------------------

    // Hard-anodised aluminium upper/lower. TIGHT specular (roughness pulled
    // well under the polymer), anisotropic along the extrusion axis, and a real
    // environment response so the sky rakes a hard line down the flat-top.
    // envMapIntensity pulled back from 0.60. The world probe is a golden-hour
    // sky dome, i.e. mostly cool blue overhead, and at 0.60 a 96%-metal
    // receiver mirrored enough of it to read as cold blue-grey plastic rather
    // than warm-neutral hardcoat aluminium — the "flat blue-grey slab".
    const recv = this._mk('gunmetal', { repeat: [6, 6], tint: 0x9aa0a6, roughness: 1.06, metalness: 0.96, normalScale: 1.05, envMapIntensity: 0.44, detailScale: 3.2 });
    // phosphated barrel / gas block / brake — darkest part of the gun, matte
    const steel = this._mk('gunmetal', { repeat: [7, 7], tint: 0x7f858d, roughness: 1.14, metalness: 1.0, normalScale: 1.15, envMapIntensity: 0.86, detailScale: 2.8 });
    // bare bright steel: bolt carrier, charging-handle latch, pins, trigger
    const bright = this._mk('metal', { repeat: [6, 6], tint: 0xcfcabe, roughness: 0.26, metalness: 1.0, envMapIntensity: 1.52, detailScale: 4.0 });

    // Anisotropy pulled well back on the receiver. At 0.52 with the rotation
    // aligned to the extrusion, the stretched highlight ran the entire length
    // of the flat-top and the rail rendered as one solid white strip filling
    // the bottom third of the ADS frame. It wants a directional sheen, not a
    // mirror smeared along the bore.
    this.matRecv = this._aniso(recv, 0.28, 1.5708);
    this.matSteel = this._aniso(steel, 0.44, 1.5708);
    this.matBright = this._aniso(bright, 0.72, 0.0);

    // FDE furniture — handguard, stock, grip, magazine. Broad, dull, textured:
    // the roughness multiplier is deliberately pushed past 1 and the env
    // response starved, so next to the receiver these read as a completely
    // different substance rather than the same moulding in another colour.
    this.matFde = this._mk('polymer', { repeat: [6, 6], tint: 0xffd79a, roughness: 1.54, metalness: 0.0, normalScale: 1.32, envMapIntensity: 0.52, detailScale: 3.0 });
    // black polymer — optic body, dust cover, rail covers, small furniture
    this.matPoly = this._mk('polymer', { repeat: [7, 7], tint: 0x929aa4, roughness: 1.46, metalness: 0.0, normalScale: 1.20, envMapIntensity: 0.54, detailScale: 2.6 });
    // rubber — grip panels, butt pad, optic bumpers. Flattest thing on the gun.
    this.matRub = this._mk('rubber', { repeat: [5, 5], tint: 0x968d84, roughness: 1.66, metalness: 0.0, normalScale: 1.25, envMapIntensity: 0.34 });
    // gloves / webbing
    this.matGlove = this._mk('cloth_tan', { repeat: [4, 4], tint: 0xdcc396, roughness: 1.20, metalness: 0.0, normalScale: 1.2, envMapIntensity: 0.66 });
    // reinforcement panels: the SAME family as the glove shell, only a couple
    // of stops down. On a rubber base these went to near-black and the fingers
    // read as sausages capped with black thimbles.
    this.matGloveDark = this._mk('cloth_tan', { repeat: [7, 7], tint: 0x9c8965, roughness: 1.34, metalness: 0.0, normalScale: 1.35, envMapIntensity: 0.54 });
    this.matWeb = this._mk('fabric', { repeat: [3, 3], tint: 0x8b8270, roughness: 1.24, metalness: 0.0, envMapIntensity: 0.46 });
    // combat-shirt sleeve — a DIFFERENT garment from the glove. Without its own
    // olive value the forearm merged into the glove and the whole arm read as
    // one tan blob with no wrist.
    this.matSleeve = this._mk('cloth_tan', { repeat: [3, 3], tint: 0xa8a078, roughness: 1.30, metalness: 0.0, normalScale: 1.1, envMapIntensity: 0.52 });
    // painted white index marks — selector letters, roll marks, witness paint
    this.matMark = new THREE.MeshStandardMaterial({
      name: 'wpn:mark', color: 0xc9c2ae, roughness: 0.74, metalness: 0.0,
      envMapIntensity: 0.5, vertexColors: true, dithering: true,
    });

    // ---- optic glass ------------------------------------------------
    // Was a 17%-opacity tint with no map: functionally invisible, which is why
    // the sight read as an empty hoop you could see straight through. It now
    // carries a real coated-lens texture — a teal/violet AR bloom in the middle
    // going deep and near-opaque at the rim, which IS the tube vignette — and
    // it is metallic enough at low roughness that the sky lays a coating
    // reflection across the objective.
    this._makeOpticTextures();
    this.matLens = new THREE.MeshStandardMaterial({
      name: 'wpn:lens', map: this.texLens, color: 0xffffff,
      roughness: 0.055, metalness: 0.62, envMapIntensity: 1.85,
      transparent: true, opacity: 1.0, vertexColors: true,
      side: THREE.FrontSide, depthWrite: false, dithering: true,
    });

    // reticle + flash — unlit, additive, bloom food.
    // The colour deliberately exceeds 1.0 so the bloom pass has something above
    // threshold to work with: a red dot that is not blooming is a red pixel.
    this.matReticle = new THREE.MeshBasicMaterial({
      name: 'wpn:reticle', transparent: true, opacity: 1,
      blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true,
      side: THREE.DoubleSide, toneMapped: true,
    });
    this.matReticle.color.setRGB(7.2, 0.62, 0.16);

    // world-scene props
    this.matBrass = this._mk('metal', { repeat: [2, 2], tint: 0xffd57a, roughness: 0.24, metalness: 1.0, envMapIntensity: 2.1 });
    this.matDropMag = this._mk('polymer', { repeat: [3, 3], tint: 0xffdca6, roughness: 1.34, metalness: 0.0, envMapIntensity: 0.86 });

    // ---- per-substance surface response --------------------------------
    // (wearRough, wearMetal, grimeRough): how hard a worn chamfer polishes,
    // how far it drifts toward bare metal, and how much macro breakup the flats
    // carry. Metal polishes hard; polymer only scuffs; rubber barely changes.
    //
    // (gain, bias, albedoCap): the LEVELS remap described at SURF_ALBEDO. bias
    // is the family's real linear albedo floor; gain stretches the source map's
    // own machining / stipple variation on top of it; cap holds the metal maps'
    // rub-through blobs back from flaring.
    //   FDE polymer  ~ (0.30, 0.195, 0.098) linear = coyote tan
    //   anodised Al  ~ (0.077, 0.075, 0.077)       = DARK satin hardcoat
    //   phosphate    ~ (0.083, 0.078, 0.073)       = warm near-black metal
    //   black polymer~ 0.030                       = a real polymer black
    //
    // Round-1 note: these floors used to sit two and a half stops higher. On a
    // metal, albedo IS the specular colour, so an anodised receiver lifted to
    // 0.20 linear and then hit by a 3.1-intensity key renders as a pale cool
    // slab — "grey primitives", exactly what the critic saw. A real hardcoat
    // upper is DARK; what makes it read as machined aluminium rather than
    // plastic is the CONTRAST between that dark body and the bright rub-through
    // on every chamfer, which is why wearMetal / wearColor climb as the floors
    // drop. The polymer families move much less: they are already dark, and
    // their whole job is to answer the metal with a broad, dull, flat response.
    // Round-3: wearRough / wearMetal pulled back. At 0.80/0.26 every chamfer on
    // the flat-top polished to a near-mirror, and because the rail is nothing
    // BUT chamfered edges the whole top of the receiver rendered as one white
    // strip — the thing that ate the ADS frame. The dark body / bright edge
    // contrast still has to be there, just an order of magnitude less shouty.
    this._patchSurf(this.matRecv, {
      wearRough: 0.50, wearMetal: 0.14, grimeRough: 0.20,
      gain: 1.24, bias: [0.0300, 0.0276, 0.0248], albedoCap: 0.071,
    });
    this._patchSurf(this.matSteel, {
      wearRough: 0.66, wearMetal: 0.20, grimeRough: 0.15,
      gain: 1.90, bias: [0.0330, 0.0308, 0.0284], albedoCap: 0.118,
    });
    // Bright steel pulled down a stop and a half. It is correct that a charging
    // handle latch is the shiniest thing on a carbine, but at cap 0.330 with a
    // 0.48 polish it was the ONLY thing on the weapon anywhere near white, and
    // in the hip frame it read as a blown blob stuck to the rear of the upper
    // rather than as a machined part.
    this._patchSurf(this.matBright, {
      wearRough: 0.36, wearMetal: 0.05, grimeRough: 0.12,
      gain: 1.15, bias: [0.086, 0.083, 0.077], albedoCap: 0.232,
    });
    this._patchSurf(this.matFde, {
      wearRough: 0.30, wearMetal: 0.0, grimeRough: 0.26,
      gain: 2.25, bias: [0.1280, 0.0812, 0.0372], albedoCap: 0.306,
    });
    // Round-3 LIFT. These two families were authored at 0.016 and 0.015 linear.
    // Nothing man-made is that dark: a moulded "black" polymer measures about
    // 0.045 and a rubber butt pad about 0.038, and both of those are BEFORE the
    // golden-hour key. At the old values the butt pad — the single largest,
    // closest object in the hip frame — rendered as a featureless black
    // parallelogram, which is exactly why a blind viewer read it as a magazine
    // slab hanging off the weapon. It is also a straight "no pure blacks"
    // violation. Lifting them roughly doubles the value and lets the moulding
    // seams, ribs and the polymer insert survive into the image.
    this._patchSurf(this.matPoly, {
      wearRough: 0.34, wearMetal: 0.0, grimeRough: 0.24,
      gain: 1.24, bias: [0.0252, 0.0260, 0.0278], albedoCap: 0.112,
    });
    this._patchSurf(this.matRub, {
      wearRough: 0.18, wearMetal: 0.0, grimeRough: 0.18,
      gain: 1.45, bias: [0.0300, 0.0278, 0.0250], albedoCap: 0.142,
    });
    this._patchSurf(this.matGlove, {
      wearRough: 0.20, wearMetal: 0.0, grimeRough: 0.26,
      gain: 1.00, bias: [0.0176, 0.0124, 0.0055], albedoCap: 0.455,
    });
    this._patchSurf(this.matGloveDark, {
      wearRough: 0.22, wearMetal: 0.0, grimeRough: 0.24,
      gain: 1.00, bias: [0.0078, 0.0050, 0.0019], albedoCap: 0.212,
    });
    this._patchSurf(this.matWeb, {
      wearRough: 0.14, wearMetal: 0.0, grimeRough: 0.20,
      gain: 1.00, bias: [0.0146, 0.0132, 0.0090], albedoCap: 0.285,
    });
    // Sleeve pushed DOWN and toward olive. It was the brightest cloth on the
    // rig and the forearm is the single largest area in frame after the
    // receiver, so a pale tan tube next to a coyote-tan handguard made the
    // support arm read as part of the weapon rather than as an arm.
    this._patchSurf(this.matSleeve, {
      wearRough: 0.12, wearMetal: 0.0, grimeRough: 0.24,
      gain: 0.85, bias: [0.0176, 0.0190, 0.0116], albedoCap: 0.186,
    });
    this._patchSurf(this.matBrass, {
      wearRough: 0.40, wearMetal: 0.0, grimeRough: 0.10,
      gain: 1.50, bias: [0.196, 0.160, 0.070], albedoCap: 0.720,
    });
    this._patchSurf(this.matDropMag, {
      wearRough: 0.30, wearMetal: 0.0, grimeRough: 0.26,
      gain: 2.25, bias: [0.1280, 0.0812, 0.0372], albedoCap: 0.306,
    });
  }

  _matFor(key) {
    switch (key) {
      case 'recv': return this.matRecv;
      case 'steel': return this.matSteel;
      case 'bright': return this.matBright;
      case 'fde': return this.matFde;
      case 'poly': return this.matPoly;
      case 'rub': return this.matRub;
      case 'glove': return this.matGlove;
      case 'gloveDark': return this.matGloveDark;
      case 'web': return this.matWeb;
      case 'sleeve': return this.matSleeve;
      case 'mark': return this.matMark;
      case 'brass': return this.matBrass;
      default: return this.matRecv;
    }
  }

  // ==================================================================
  //  WEAPON GEOMETRY
  //  Local space: bore on Y=0/X=0, muzzle toward -Z, receiver rear at Z=0.
  // ==================================================================

  _buildWeapon() {
    const rig = new Rig();

    this.opticCentre = new THREE.Vector3(0, 0.0695, -0.1200);
    this.muzzleLocal = new THREE.Vector3(0, 0, -0.5480);
    this.portLocal = new THREE.Vector3(0.0235, 0.0125, -0.0620);

    this._bUpper(rig);
    this._bLower(rig);
    this._bHandguard(rig);
    this._bBarrel(rig);
    this._bStock(rig);
    this._bOptic(rig);
    this._bSights(rig);
    this._bSling(rig);
    this._bGreeble(rig);

    this.gun = new THREE.Group();
    this.gun.name = 'gun';
    rig.build(this.gun, k => this._matFor(k), 'gun');
    this.root.add(this.gun);
    this.staticTris = rig.tris | 0;

    this._bLens();
    this._bMagazine();
    this._bChargingHandle();
    this._bBolt();
    this._bDustCover();
    this._bTrigger();
  }

  /* ---- picatinny rail: base slab + machined cross-slots ------------ */
  _rail(rig, z0, z1, y, w, key, wear, opts) {
    const len = Math.abs(z1 - z0);
    const zc = (z0 + z1) * 0.5;
    const skipA = opts?.skip?.[0] ?? -1, skipB = opts?.skip?.[1] ?? -2;
    // The base slab is a long thin box, so its chamfer band covers most of its
    // vertices; at the wear the TEETH want it turns the whole rail into a white
    // strip down the middle of the ADS frame. The teeth carry the rub-through.
    // Round-3: the slab is also pushed two stops DOWN in shade. Between the
    // teeth you are looking into a machined slot, and a slot floor that renders
    // at the same value as the tooth crowns is what turned the flat-top into a
    // solid bright bar instead of a row of ribs.
    rig.add(key, rbox(w, 0.0048, len, 0.0009), {
      p: [0, y + 0.0024, zc], wear: wear * 0.16, ao: 1.35, shade: 0.62, grime: 1.2,
    });
    const pitch = 0.01016;
    const n = Math.max(1, Math.floor(len / pitch));
    const start = zc + len * 0.5 - pitch * 0.5;
    for (let i = 0; i < n; i++) {
      if (i >= skipA && i <= skipB) continue;            // covered by a rail panel
      const z = start - i * pitch;
      rig.add(key, rbox(w * 0.985, 0.0044, 0.0058, 0.0011), {
        p: [0, y + 0.0068, z], wear, ao: 1, shade: 0.88,
        // rail tops are rubbed, but only the crown: a narrow band, not the
        // whole tooth, or the rail turns back into a white ribbon
        edgeLo: 0.10, edgeHi: 0.30,
      });
      // shadowed slot floor between every pair of teeth — the read that makes
      // a Picatinny section legible at viewmodel distance
      rig.add(key, rbox(w * 0.90, 0.0022, 0.0044, 0.0006), {
        p: [0, y + 0.0058, z - pitch * 0.5], shade: 0.30, ao: 1.6, wear: 0.05, grime: 0.5,
      });
      // T-marked slot numbers every fifth slot, and a witness dot between
      if (opts?.numbers && i % 5 === 0) {
        rig.add('mark', rbox(0.0013, 0.0013, 0.0022, 0.0003), {
          p: [-w * 0.5 - 0.0006, y + 0.0058, z - 0.0050], shade: 0.95, wear: 0, dust: 0.3,
        });
      }
    }
  }

  /**
   * Laser-etched roll marks / serial block. Reads as engraving at any distance
   * a viewmodel is ever seen from, and is what tells the eye a panel is a real
   * machined part rather than a primitive.
   *   nx = character columns, rows = lines of "text"
   */
  _engrave(rig, x, y, z, sx, cols, rows, seed) {
    const chW = 0.0016, chH = 0.0022, gapZ = 0.0022, gapY = 0.0042;
    for (let r = 0; r < rows; r++) {
      const n = 3 + ((fhash(seed + r, 11, 3) * (cols - 2)) | 0);
      for (let c = 0; c < n; c++) {
        const h = fhash(seed + r * 7, c * 13, 5);
        if (h < 0.16) continue;                       // word breaks
        rig.add('poly', rbox(0.0011, chH * (0.7 + h * 0.4), chW, 0.0003), {
          p: [x, y - r * gapY, z - c * gapZ], r: [0, 0, 0],
          shade: 0.16, ao: 1.5, wear: 0.0, grime: 0.2, dust: 0,
        });
      }
    }
  }

  /* ---- upper receiver --------------------------------------------- */
  _bUpper(rig) {
    const W = 0.0400, H = 0.0448;
    // main flat-top body, rear at +0.012 (behind it is the buffer-tube boss)
    rig.add('recv', hbox(W, H, 0.2020, 0.0042), { p: [0, 0.0106, -0.0890], wear: 0.30, ao: 1 });
    // Machined relief along both flanks. Flush strips rather than a proud
    // cylinder: a raised half-round on the side of an AR upper reads as a
    // casting defect, but a shadowed inset band gives the slab a second plane.
    for (const sx of [-1, 1]) {
      rig.add('recv', rbox(0.0044, 0.0210, 0.0900, 0.0034, 2), {
        p: [sx * 0.0182, 0.0126, -0.0980], shade: 0.86, ao: 1.30, wear: 0.20,
      });
      rig.add('recv', rbox(0.0060, 0.0032, 0.0910, 0.0012), {
        p: [sx * 0.0192, 0.0238, -0.0980], wear: 0.34, edgeHi: 0.44,
      });
      rig.add('recv', rbox(0.0060, 0.0032, 0.0910, 0.0012), {
        p: [sx * 0.0192, 0.0014, -0.0980], wear: 0.30, edgeHi: 0.44,
      });
    }
    // charging-handle boss / rear of the upper
    rig.add('recv', rbox(0.0356, 0.0300, 0.0300, 0.0034, 2), { p: [0, 0.0160, 0.0250], wear: 0.34 });
    // takedown-pin bosses
    rig.add('recv', cyl(0.0068, 0.0068, 0.0424, 12), { p: [0, -0.0055, -0.1830], r: [0, 0, Math.PI * 0.5], wear: 0.40 });
    rig.add('recv', cyl(0.0068, 0.0068, 0.0424, 12), { p: [0, -0.0060, 0.0050], r: [0, 0, Math.PI * 0.5], wear: 0.40 });
    rig.add('bright', cyl(0.0032, 0.0032, 0.0442, 10), { p: [0, -0.0055, -0.1830], r: [0, 0, Math.PI * 0.5], wear: 0.85 });
    rig.add('bright', cyl(0.0032, 0.0032, 0.0442, 10), { p: [0, -0.0060, 0.0050], r: [0, 0, Math.PI * 0.5], wear: 0.85 });

    // top rail from the rear of the upper to the barrel nut
    // Wear on the receiver rail is deliberately low. This rail sits 6cm from
    // the lens in ADS, and at any meaningful rub-through its chamfers mirror
    // the sun straight into the bottom third of the aimed frame.
    this._rail(rig, 0.0100, -0.1880, 0.0330, 0.0212, 'recv', 0.17, { numbers: true });

    // ---- ejection port: a genuinely recessed pocket, not an indent ----
    // back wall of the cavity, deep and dark
    rig.add('poly', rbox(0.0038, 0.0250, 0.0570, 0.0010), { p: [0.0142, 0.0118, -0.0620], shade: 0.10, ao: 1.6, grime: 0.3 });
    // side walls of the cutout catch a sliver of light and give it depth
    rig.add('recv', rbox(0.0080, 0.0032, 0.0570, 0.0008), { p: [0.0180, 0.0244, -0.0620], shade: 0.55, wear: 0.30, ao: 1.4 });
    rig.add('recv', rbox(0.0080, 0.0032, 0.0570, 0.0008), { p: [0.0180, -0.0004, -0.0620], shade: 0.42, wear: 0.22, ao: 1.5 });
    rig.add('recv', rbox(0.0080, 0.0250, 0.0030, 0.0008), { p: [0.0180, 0.0118, -0.0348], shade: 0.50, wear: 0.35, ao: 1.4 });
    rig.add('recv', rbox(0.0080, 0.0250, 0.0030, 0.0008), { p: [0.0180, 0.0118, -0.0892], shade: 0.50, wear: 0.35, ao: 1.4 });
    // port lip / relief cut — the rolled outer edge of the cutout
    rig.add('recv', rbox(0.0046, 0.0044, 0.0624, 0.0014), { p: [0.0212, 0.0268, -0.0620], wear: 0.62, edgeHi: 0.46 });
    rig.add('recv', rbox(0.0046, 0.0042, 0.0624, 0.0014), { p: [0.0212, -0.0026, -0.0620], wear: 0.52, edgeHi: 0.46 });
    // brass deflector — the little wedge behind the port
    rig.add('recv', rbox(0.0110, 0.0150, 0.0170, 0.0030, 2), { p: [0.0182, 0.0170, -0.0250], r: [0, 0, -0.24], wear: 0.62 });
    // carbon wash blown out of the port and dragged rearward over the deflector
    // and the rear of the upper — the dirtiest 40mm on any gas-impingement gun
    for (let i = 0; i < 4; i++) {
      const t = i / 3;
      rig.add('recv', rbox(0.0026, 0.0074 - t * 0.0022, 0.0300 + t * 0.0140, 0.0006), {
        p: [0.0206, 0.0250 - i * 0.0072, -0.0240 - t * 0.0090],
        tintv: [0.34 + t * 0.22, 0.31 + t * 0.21, 0.29 + t * 0.20],
        shade: 1.0, wear: 0.04, grime: 1.9, dust: 0,
      });
    }
    // forward assist
    rig.add('recv', cyl(0.0062, 0.0074, 0.0130, 12), { p: [0.0214, 0.0038, -0.0210], r: [0, 0, Math.PI * 0.5], wear: 0.50 });
    rig.add('bright', cyl(0.0044, 0.0044, 0.0092, 10), { p: [0.0290, 0.0038, -0.0210], r: [0, 0, Math.PI * 0.5], wear: 0.95 });
    rig.add('recv', rbox(0.0044, 0.0100, 0.0100, 0.0014), { p: [0.0314, 0.0038, -0.0210], wear: 0.75 });

    // left-side wall: a raised serial panel with real engraving on it. This is
    // the face the player actually looks at, so it carries the roll marks.
    rig.add('recv', rbox(0.0034, 0.0190, 0.0500, 0.0016), { p: [-0.0200, 0.0060, -0.0660], shade: 0.96, wear: 0.24 });
    this._engrave(rig, -0.0222, 0.0122, -0.0470, -1, 9, 3, 17);
    // magnified index ring around the pivot pin, plus a maker's crest
    rig.add('recv', new THREE.TorusGeometry(0.0092, 0.0013, 6, 14), { p: [-0.0208, -0.0055, -0.1830], r: [0, Math.PI * 0.5, 0], wear: 0.42 });

    // barrel-nut / handguard interface
    rig.add('steel', cyl(0.0210, 0.0210, 0.0180, 18), { p: [0, 0, -0.1980], r: [Math.PI * 0.5, 0, 0], wear: 0.35 });
    rig.add('recv', cyl(0.0170, 0.0210, 0.0140, 18), { p: [0, 0, -0.1870], r: [Math.PI * 0.5, 0, 0], wear: 0.30 });
  }

  /* ---- lower receiver, magwell, grip, trigger guard ---------------- */
  _bLower(rig) {
    // body
    rig.add('recv', hbox(0.0344, 0.0400, 0.1560, 0.0044), { p: [0, -0.0322, -0.0820], wear: 0.26 });
    // magwell — flared funnel
    rig.add('recv', hbox(0.0372, 0.0520, 0.0560, 0.0042), { p: [0, -0.0402, -0.1300], r: [0.0900, 0, 0], wear: 0.34 });
    rig.add('recv', rbox(0.0404, 0.0130, 0.0596, 0.0038, 2), { p: [0, -0.0632, -0.1322], r: [0.0900, 0, 0], wear: 0.58, edgeHi: 0.48 });
    // magwell roll marks + a stamped lot block on the left flank
    rig.add('recv', rbox(0.0030, 0.0240, 0.0330, 0.0012), { p: [-0.0190, -0.0400, -0.1300], r: [0.0900, 0, 0], shade: 0.95, wear: 0.20 });
    this._engrave(rig, -0.0208, -0.0330, -0.1180, -1, 7, 2, 41);
    // mag catch fence + button (a very high-wear surface)
    rig.add('recv', rbox(0.0110, 0.0170, 0.0170, 0.0030, 2), { p: [0.0208, -0.0300, -0.1010], wear: 0.60 });
    rig.add('bright', cyl(0.0056, 0.0056, 0.0090, 12), { p: [0.0268, -0.0300, -0.1010], r: [0, 0, Math.PI * 0.5], wear: 1.0, edgeHi: 0.5 });
    // left-side extended mag release paddle — fence, paddle and checkered face
    rig.add('recv', rbox(0.0104, 0.0148, 0.0148, 0.0024, 2), { p: [-0.0212, -0.0300, -0.1010], wear: 0.72 });
    rig.add('bright', rbox(0.0044, 0.0112, 0.0112, 0.0016), { p: [-0.0278, -0.0300, -0.1010], wear: 0.70, edgeHi: 0.52 });
    for (let i = 0; i < 3; i++) {
      rig.add('bright', rbox(0.0022, 0.0016, 0.0104, 0.0005), { p: [-0.0300, -0.0334 + i * 0.0034, -0.1010], wear: 0.72 });
    }
    // bolt catch (left side) — paddle, upper shelf and pivot
    rig.add('recv', rbox(0.0062, 0.0132, 0.0300, 0.0020), { p: [-0.0204, -0.0180, -0.0800], r: [0, 0, 0.10], wear: 0.68 });
    rig.add('recv', rbox(0.0050, 0.0068, 0.0120, 0.0016), { p: [-0.0232, -0.0128, -0.0900], r: [0, 0, 0.18], wear: 0.86, edgeHi: 0.5 });
    rig.add('bright', cyl(0.0020, 0.0020, 0.0060, 8), { p: [-0.0228, -0.0206, -0.0700], r: [0, 0, Math.PI * 0.5], wear: 0.9 });
    // safety selector — detent barrel, both levers, and the SAFE/FIRE marks
    rig.add('recv', cyl(0.0070, 0.0070, 0.0392, 12), { p: [0, -0.0192, -0.0430], r: [0, 0, Math.PI * 0.5], wear: 0.44 });
    rig.add('recv', cyl(0.0092, 0.0092, 0.0044, 14), { p: [-0.0214, -0.0192, -0.0430], r: [0, 0, Math.PI * 0.5], wear: 0.55, shade: 0.96 });
    rig.add('poly', rbox(0.0060, 0.0084, 0.0250, 0.0018), { p: [-0.0232, -0.0210, -0.0350], r: [0.35, 0, 0], wear: 0.60 });
    rig.add('poly', rbox(0.0060, 0.0084, 0.0250, 0.0018), { p: [0.0232, -0.0210, -0.0350], r: [0.35, 0, 0], wear: 0.60 });
    // serrations along the top of the left lever
    for (let i = 0; i < 4; i++) {
      rig.add('poly', rbox(0.0064, 0.0018, 0.0028, 0.0006), { p: [-0.0232, -0.0176 - i * 0.0022, -0.0286 - i * 0.0060], r: [0.35, 0, 0], wear: 0.85 });
    }
    // white index pointer on the lever + SAFE / SEMI witness marks on the boss
    // Painted index pointer on the lever, and SAFE / SEMI / AUTO index bars
    // struck radially around the boss. Three 2.6mm dots were sub-pixel at any
    // distance the viewmodel is ever seen from; short radial bars survive the
    // downsample and are what tells the eye this is a fire-control group.
    rig.add('mark', rbox(0.0022, 0.0026, 0.0140, 0.0005), { p: [-0.0254, -0.0208, -0.0326], r: [0.35, 0, 0], shade: 1.0, wear: 0, dust: 0.4 });
    for (const a of [1.92, 2.64, 3.52]) {
      const rr = 0.0136;
      rig.add('mark', rbox(0.0018, 0.0044, 0.0044, 0.0005), {
        p: [-0.0238, -0.0192 + Math.sin(a) * rr, -0.0430 + Math.cos(a) * rr],
        r: [-a + Math.PI * 0.5, 0, 0], shade: 1.0, wear: 0, dust: 0.4,
      });
    }
    // witness paint on the takedown pin — a second white accent so the marks
    // read as a system rather than one stray dot
    rig.add('mark', rbox(0.0016, 0.0030, 0.0030, 0.0005), { p: [-0.0240, -0.0080, -0.1830], shade: 0.94, wear: 0, dust: 0.4 });

    // ---- trigger guard: an extruded U ------------------------------
    const s = new THREE.Shape();
    s.moveTo(-0.0290, -0.0060);
    s.lineTo(-0.0290, -0.0140);
    s.quadraticCurveTo(-0.0290, -0.0270, -0.0170, -0.0286);
    s.lineTo(0.0110, -0.0286);
    s.quadraticCurveTo(0.0210, -0.0280, 0.0220, -0.0180);
    s.lineTo(0.0220, -0.0060);
    s.lineTo(0.0166, -0.0060);
    s.lineTo(0.0166, -0.0166);
    s.quadraticCurveTo(0.0160, -0.0228, 0.0096, -0.0230);
    s.lineTo(-0.0158, -0.0230);
    s.quadraticCurveTo(-0.0236, -0.0226, -0.0236, -0.0140);
    s.lineTo(-0.0236, -0.0060);
    s.closePath();
    const guard = new THREE.ExtrudeGeometry(s, { depth: 0.0104, bevelEnabled: true, bevelThickness: 0.0009, bevelSize: 0.0009, bevelSegments: 1, curveSegments: 6 });
    guard.translate(0, 0, -0.0052);
    // shape is authored in (z,y); rotate it into the bore frame
    rig.add('recv', guard, { p: [0, -0.0480, -0.0640], r: [0, Math.PI * 0.5, 0], wear: 0.52 });

    // ---- pistol grip: raked column with finger swells ---------------
    const gripR = [-0.4300, 0, 0];   // rake back ~24.6 deg
    rig.add('fde', cyl(0.0196, 0.0166, 0.0940, 14), { p: [0, -0.0890, -0.0330], r: gripR, s: [0.86, 1, 1.06], wear: 0.30 });
    rig.add('rub', cyl(0.0186, 0.0180, 0.0700, 14, true), { p: [0, -0.0840, -0.0308], r: gripR, s: [0.90, 1, 1.10], wear: 0.55, shade: 0.9 });
    for (let i = 0; i < 3; i++) {
      const t = i * 0.0208;
      rig.add('rub', new THREE.TorusGeometry(0.0116, 0.0028, 6, 14, Math.PI * 1.15), {
        p: [0, -0.0640 - t * 0.92, -0.0472 - t * 0.42], r: [gripR[0] + Math.PI * 0.5, 0, Math.PI * 0.5],
        s: [1, 1, 0.72], wear: 0.42, shade: 0.86,
      });
    }
    // grip cap / trap door
    rig.add('poly', rbox(0.0300, 0.0074, 0.0290, 0.0026, 2), { p: [0, -0.1320, -0.0130], r: gripR, wear: 0.50 });
    // beavertail where the grip meets the receiver
    rig.add('fde', rbox(0.0300, 0.0170, 0.0220, 0.0050, 2), { p: [0, -0.0490, -0.0250], r: [-0.30, 0, 0], wear: 0.28 });
  }

  /* ---- free-float M-LOK handguard --------------------------------- */
  _bHandguard(rig) {
    const z0 = -0.1980, z1 = -0.4180;
    const len = z0 - z1, zc = (z0 + z1) * 0.5;
    const R = 0.0248, panelW = 0.0198, panelT = 0.0052;

    // eight radial panels; the top one is replaced by the rail platform
    for (let i = 1; i < 8; i++) {
      const a = i * Math.PI * 0.25;
      const isTopish = (i === 1 || i === 7);
      rig.add('fde', rbox(panelW, panelT, len, 0.0012), {
        p: [Math.sin(a) * R, Math.cos(a) * R, zc],
        r: [0, 0, -a], wear: isTopish ? 0.26 : 0.20,
      });
    }
    // rail platform on top of the handguard, height-matched to the receiver rail
    rig.add('fde', rbox(0.0248, 0.0180, len, 0.0020), { p: [0, 0.0245, zc], wear: 0.22 });
    // Slots 4-11 are hidden under a ladder rail cover — an unbroken run of forty
    // identical teeth is the single most "procedural" thing a carbine can have.
    this._rail(rig, z0 - 0.0020, z1 + 0.0020, 0.0330, 0.0212, 'fde', 0.26, { skip: [4, 11] });
    this._bRailCover(rig, -0.2440, -0.3320, 0.0330, 0.0224);

    // M-LOK slots — recessed cutouts on the diagonal, side and bottom panels.
    // Face 6 (the 9-o'clock flat) is added in round 3: it is the face the
    // player's eye actually lands on, and it was the ONLY panel with no
    // cutouts, which is why the handguard read as a smooth tan tube. Its slots
    // are shifted forward so they clear the laser module clamped to that flat.
    const slotFaces = [1, 3, 4, 5, 6, 7];
    for (const i of slotFaces) {
      const a = i * Math.PI * 0.25;
      const cx = Math.sin(a) * (R - 0.0016), cy = Math.cos(a) * (R - 0.0016);
      const nSlot = i === 6 ? 3 : 4;
      for (let s = 0; s < nSlot; s++) {
        const z = i === 6 ? (z0 - 0.1020 - s * 0.0450) : (z0 - 0.0300 - s * 0.0450);
        rig.add('poly', rbox(0.0112, 0.0032, 0.0330, 0.0011), {
          p: [cx, cy, z], r: [0, 0, -a], shade: 0.22, ao: 1.4, grime: 0.6,
        });
        rig.add('poly', rbox(0.0066, 0.0046, 0.0286, 0.0009), {
          p: [Math.sin(a) * (R - 0.0050), Math.cos(a) * (R - 0.0050), z], r: [0, 0, -a], shade: 0.07, ao: 1.6, grime: 0.4,
        });
        // the machined lip around each slot catches a bright line
        for (const sz of [-1, 1]) {
          rig.add('fde', rbox(0.0126, 0.0030, 0.0038, 0.0009), {
            p: [Math.sin(a) * (R + 0.0004), Math.cos(a) * (R + 0.0004), z + sz * 0.0180], r: [0, 0, -a],
            wear: 0.62, edgeHi: 0.44,
          });
        }
      }
    }
    // end cap ring + barrel-nut collar
    rig.add('fde', new THREE.TorusGeometry(R - 0.0044, 0.0044, 8, 16), { p: [0, 0, z1 + 0.0026], wear: 0.42 });
    rig.add('fde', cyl(R - 0.0010, R + 0.0016, 0.0130, 16), { p: [0, 0, z0 - 0.0050], r: [Math.PI * 0.5, 0, 0], wear: 0.30 });
    // anti-rotation tab + QD sling socket
    // QD socket moved to the FRONT of the handguard: at its old station it sat
    // directly under the support hand, so the sling left the weapon behind the
    // fingers and hung across them as a dark fin.
    rig.add('poly', cyl(0.0052, 0.0052, 0.0050, 12), { p: [-R + 0.0006, 0.0044, z1 + 0.0180], r: [0, 0, Math.PI * 0.5], shade: 0.5, wear: 0.6 });
    rig.add('steel', cyl(0.0026, 0.0026, 0.0060, 10), { p: [-R + 0.0000, 0.0044, z1 + 0.0180], r: [0, 0, Math.PI * 0.5], shade: 0.4, wear: 0.8 });

    // handstop / index panel on the underside
    rig.add('poly', rbox(0.0180, 0.0130, 0.0250, 0.0026, 2), { p: [0, -R - 0.0040, z1 + 0.0640], r: [-0.28, 0, 0], wear: 0.40 });

    // ---- M-LOK laser/illuminator module on the LEFT flank -------------
    // The player's eye sits outboard of the gun's left side, so this is the
    // face that carries the hardware. A hard black box against tan polymer with
    // its own glass and cable is the single strongest "issued weapon" cue.
    this._bLaserModule(rig, -R - 0.0080, 0.0068, -0.2470);
  }

  /* ---- polymer ladder rail cover ----------------------------------- */
  _bRailCover(rig, z0, z1, y, w) {
    const len = Math.abs(z1 - z0), zc = (z0 + z1) * 0.5;
    rig.add('poly', rbox(w, 0.0072, len, 0.0016), { p: [0, y + 0.0058, zc], wear: 0.34, shade: 0.94 });
    const n = Math.max(2, Math.floor(len / 0.0118));
    for (let i = 0; i < n; i++) {
      const z = zc + len * 0.5 - 0.0059 - i * 0.0118;
      rig.add('poly', rbox(w * 1.03, 0.0044, 0.0064, 0.0012), { p: [0, y + 0.0104, z], wear: 0.52, edgeHi: 0.42 });
      rig.add('poly', rbox(w * 1.02, 0.0026, 0.0034, 0.0008), { p: [0, y + 0.0090, z + 0.0059], shade: 0.42, ao: 1.3, wear: 0.2 });
    }
    // end lugs that clip over the rail
    rig.add('poly', rbox(w * 1.05, 0.0110, 0.0044, 0.0012), { p: [0, y + 0.0060, zc + len * 0.5 + 0.0018], wear: 0.5 });
    rig.add('poly', rbox(w * 1.05, 0.0110, 0.0044, 0.0012), { p: [0, y + 0.0060, zc - len * 0.5 - 0.0018], wear: 0.5 });
  }

  /* ---- side-mounted IR laser / white-light module ------------------ */
  _bLaserModule(rig, x, y, z) {
    // body
    rig.add('poly', rbox(0.0210, 0.0250, 0.0620, 0.0030, 2), { p: [x, y, z], wear: 0.42, shade: 0.92 });
    // machined top plate + hex screws
    rig.add('poly', rbox(0.0170, 0.0040, 0.0560, 0.0012), { p: [x, y + 0.0140, z], wear: 0.55, shade: 0.98 });
    for (let i = 0; i < 3; i++) {
      rig.add('bright', cyl(0.0021, 0.0021, 0.0026, 6), { p: [x, y + 0.0158, z - 0.0200 + i * 0.0200], wear: 0.8 });
    }
    // emitter bezels on the front face, one clear one dark
    rig.add('steel', tube(0.0062, 0.0062, 0.0060, 12), { p: [x - 0.0030, y + 0.0058, z - 0.0338], wear: 0.6 });
    rig.add('poly', tube(0.0044, 0.0044, 0.0040, 12), { p: [x - 0.0030, y + 0.0058, z - 0.0360], shade: 0.08, ao: 1.5 });
    rig.add('steel', tube(0.0056, 0.0056, 0.0060, 12), { p: [x - 0.0028, y - 0.0068, z - 0.0338], wear: 0.6 });
    rig.add('poly', tube(0.0038, 0.0038, 0.0040, 12), { p: [x - 0.0028, y - 0.0068, z - 0.0360], shade: 0.20, ao: 1.4 });
    // rotary mode selector on the rear face
    rig.add('rub', cyl(0.0060, 0.0064, 0.0060, 10), { p: [x - 0.0020, y + 0.0040, z + 0.0342], r: [Math.PI * 0.5, 0, 0], wear: 0.7, shade: 0.9 });
    rig.add('mark', rbox(0.0016, 0.0016, 0.0022, 0.0004), { p: [x - 0.0020, y + 0.0104, z + 0.0350], shade: 1.0, wear: 0, dust: 0.3 });
    // M-LOK clamp foot tucked back against the handguard
    rig.add('poly', rbox(0.0090, 0.0150, 0.0400, 0.0016), { p: [x + 0.0140, y - 0.0010, z], wear: 0.3, shade: 0.85, ao: 1.2 });
    // pressure-pad cable looping back toward the receiver
    const cx = x + 0.0010;
    for (let i = 0; i < 7; i++) {
      const u = i / 6;
      const cz = z + 0.0330 + u * 0.0520;
      const cyy = y + 0.0060 - Math.sin(u * Math.PI) * 0.0130 - u * 0.0060;
      rig.add('rub', cyl(0.0020, 0.0020, 0.0110, 6), {
        p: [cx - u * 0.0022, cyy, cz], r: [Math.PI * 0.5 - 0.5 + u * 0.9, 0, 0], wear: 0.15, shade: 0.85,
      });
    }
  }

  /* ---- barrel, gas block, muzzle brake ---------------------------- */
  _bBarrel(rig) {
    // barrel inside the handguard (thin, visible through the front opening)
    rig.add('steel', tube(0.0092, 0.0104, 0.2360, 16), { p: [0, 0, -0.3160], wear: 0.18, shade: 0.85 });
    // exposed section
    rig.add('steel', tube(0.0090, 0.0092, 0.0760, 16), { p: [0, 0, -0.4560], wear: 0.30 });
    rig.add('steel', tube(0.0104, 0.0104, 0.0090, 16), { p: [0, 0, -0.4230], wear: 0.42 });
    // low-profile gas block + gas tube (peeking under the rail at the front)
    rig.add('steel', rbox(0.0170, 0.0170, 0.0230, 0.0016), { p: [0, 0, -0.4030], wear: 0.32, shade: 0.9 });
    rig.add('bright', tube(0.0022, 0.0022, 0.1900, 8), { p: [0, 0.0112, -0.3050], wear: 0.5, shade: 0.8 });

    // shoulder + crush washer
    rig.add('steel', tube(0.0116, 0.0116, 0.0056, 16), { p: [0, 0, -0.4952], wear: 0.55 });
    // muzzle brake — lathed body with three port pairs
    const pts = [];
    pts.push(new THREE.Vector2(0.0000, 0.0000));
    pts.push(new THREE.Vector2(0.0112, 0.0000));
    pts.push(new THREE.Vector2(0.0126, 0.0034));
    pts.push(new THREE.Vector2(0.0130, 0.0120));
    pts.push(new THREE.Vector2(0.0124, 0.0300));
    pts.push(new THREE.Vector2(0.0132, 0.0348));
    pts.push(new THREE.Vector2(0.0132, 0.0430));
    pts.push(new THREE.Vector2(0.0092, 0.0470));
    pts.push(new THREE.Vector2(0.0092, 0.0480));
    pts.push(new THREE.Vector2(0.0000, 0.0480));
    const brake = new THREE.LatheGeometry(pts, 18);
    brake.rotateX(-Math.PI * 0.5);
    rig.add('steel', brake, { p: [0, 0, -0.4980], wear: 0.60, shade: 0.92 });
    // bore
    rig.add('poly', tube(0.0058, 0.0058, 0.0300, 14), { p: [0, 0, -0.5340], shade: 0.06, ao: 1.5 });

    // ---- carbon fouling -------------------------------------------------
    // Every round that leaves this barrel lays soot back over the brake and the
    // first few centimetres of shank. Without it the muzzle end is the same
    // clean phosphate value as the gas block and the weapon reads as a render
    // of a part rather than a weapon that has been fired.
    const SOOT = [0.30, 0.27, 0.25];
    rig.add('steel', tube(0.0134, 0.0134, 0.0300, 18, true), {
      p: [0, 0, -0.5060], tintv: SOOT, shade: 1.0, wear: 0.10, grime: 1.8, dust: 0,
    });
    rig.add('steel', tube(0.0120, 0.0120, 0.0230, 18, true), {
      p: [0, 0, -0.4830], tintv: [0.52, 0.48, 0.45], shade: 1.0, wear: 0.16, grime: 1.6, dust: 0,
    });
    // asymmetric blow-back streaks off the top ports, thinning rearward
    for (let i = 0; i < 5; i++) {
      const t = i / 4;
      rig.add('steel', rbox(0.0044 - t * 0.0016, 0.0016, 0.0230 + t * 0.0130, 0.0004), {
        p: [(-0.0090 + i * 0.0046), 0.0106 - t * 0.0016, -0.4780 - t * 0.0090],
        tintv: SOOT, shade: 0.9, wear: 0.05, grime: 1.8, dust: 0,
      });
    }
    // ports
    for (let i = 0; i < 3; i++) {
      const z = -0.5090 - i * 0.0112;
      for (const sx of [-1, 1]) {
        rig.add('poly', rbox(0.0060, 0.0092, 0.0058, 0.0010), { p: [sx * 0.0104, 0.0026, z], shade: 0.14, ao: 1.4 });
      }
      rig.add('poly', rbox(0.0074, 0.0060, 0.0058, 0.0010), { p: [0, 0.0104, z], shade: 0.14, ao: 1.4 });
    }
  }

  /* ---- buffer tube + collapsible stock ---------------------------- */
  _bStock(rig) {
    // Receiver extension. It must END INSIDE the stock: at its old length the
    // tube's flat end cap punched straight through the butt pad and presented a
    // black disc at the closest point in frame — the single worst read on the
    // whole viewmodel.
    rig.add('recv', tube(0.0154, 0.0154, 0.1130, 16), { p: [0, 0.0040, 0.0660], wear: 0.30 });
    rig.add('recv', tube(0.0196, 0.0196, 0.0140, 16), { p: [0, 0.0040, 0.0180], wear: 0.34 });
    // Length-of-pull detent notches. Moved back under the stock's WAIST (see
    // the profile below) — that is the one place the tube is actually exposed,
    // and a row of machined notches on a bare steel cylinder poking out from
    // under tan polymer is the cheapest hard-surface read on the whole stock.
    for (let i = 0; i < 3; i++) {
      const z = 0.0530 + i * 0.0148;
      rig.add('recv', rbox(0.0108, 0.0058, 0.0080, 0.0014), { p: [0, -0.0118, z], shade: 0.40, ao: 1.45, wear: 0.55 });
      rig.add('recv', rbox(0.0128, 0.0036, 0.0030, 0.0008), { p: [0, -0.0144, z - 0.0046], wear: 0.78, edgeHi: 0.46 });
    }
    // castle nut with its staking slots + the end plate behind it
    rig.add('bright', cyl(0.0182, 0.0182, 0.0090, 10), { p: [0, 0.0040, 0.0300], r: [Math.PI * 0.5, 0, 0], wear: 0.9 });
    for (let i = 0; i < 5; i++) {
      const a = i * Math.PI * 0.4 + 0.3;
      rig.add('bright', rbox(0.0040, 0.0034, 0.0094, 0.0007), {
        p: [Math.sin(a) * 0.0164, 0.0040 + Math.cos(a) * 0.0164, 0.0300], r: [0, 0, -a], shade: 0.6, ao: 1.3, wear: 0.8,
      });
    }
    rig.add('recv', rbox(0.0330, 0.0330, 0.0044, 0.0016), { p: [0, 0.0040, 0.0224], wear: 0.55 });
    // ================================================================
    //  STOCK BODY — extruded side profile, NOT a box.
    //
    //  The stock is the closest and largest object in frame and it was two
    //  axis-aligned tan cuboids with a couple of shallow panels stuck on the
    //  outside: a cardboard carton with a rubber brick on the end. It is now
    //  authored as a 2D side silhouette and extruded, which buys three things
    //  no amount of greeble on a box can:
    //    - a rising comb and a hooked toe, so the top and bottom lines are not
    //      parallel and the outline reads as a shape rather than a rectangle;
    //    - a WAIST between the collar and the toe where the profile lifts
    //      clear of the buffer tube, so the bare steel tube and its detent
    //      notches are visible passing underneath — real interior form;
    //    - a narrow core (34mm) with a proud perimeter FRAME on each flank, so
    //      the side pocket is genuinely 3.6mm recessed instead of being a
    //      lighter rectangle painted on a slab.
    // ================================================================
    const SP = [
      [0.0242, 0.0288], [0.0264, 0.0332], [0.0450, 0.0352], [0.0760, 0.0372],
      [0.1120, 0.0356], [0.1250, 0.0292], [0.1322, 0.0150], [0.1334, -0.0030],
      [0.1300, -0.0200], [0.1224, -0.0294], [0.1120, -0.0324], [0.0980, -0.0312],
      [0.0870, -0.0230], [0.0760, -0.0104], [0.0640, -0.0038], [0.0500, -0.0070],
      [0.0360, -0.0180], [0.0268, -0.0252], [0.0244, -0.0150],
    ];
    const sSh = new THREE.Shape();
    sSh.moveTo(-SP[0][0], SP[0][1]);
    for (let i = 1; i < SP.length; i++) sSh.lineTo(-SP[i][0], SP[i][1]);
    sSh.closePath();
    // Core is deliberately NARROW (28.6mm). The stock's real width comes from
    // the perimeter frame bolted to each flank, which is what leaves the pocket
    // between them genuinely 5-6mm deep instead of a shaded rectangle.
    const stockGeo = new THREE.ExtrudeGeometry(sSh, {
      depth: 0.0250, bevelEnabled: true, bevelThickness: 0.0018, bevelSize: 0.0018,
      bevelSegments: 2, curveSegments: 3, steps: 1,
    });
    stockGeo.translate(0, 0, -0.0125);
    // shape authored in (-z, y); Ry(90) rotates it into the bore frame and the
    // extrusion depth becomes the stock's width
    // grime damped: the extrusion has sparse vertices, so a full-strength macro
    // field interpolates across huge triangles and blotches the flat comb
    rig.add('fde', stockGeo, { r: [0, Math.PI * 0.5, 0], wear: 0.32, edgeLo: 0.05, edgeHi: 0.42, grime: 0.34 });

    // ---- flank frame + recessed pocket -----------------------------
    // The frame stands 5mm proud of a deliberately dark pocket floor. A 2mm
    // step read as a lighter rectangle painted on a slab; at 5mm with the floor
    // two stops down it reads as a machined cavity, which is the whole point.
    for (const sx of [-1, 1]) {
      const FX = sx * 0.0188;   // frame plane   — outer face 0.0220
      const PX = sx * 0.0150;   // pocket floor  — outer face 0.0163
      rig.add('fde', rbox(0.0026, 0.0426, 0.0876, 0.0014), {
        p: [PX, 0.0056, 0.0772], shade: 0.58, ao: 1.45, wear: 0.08, grime: 1.5,
      });
      // perimeter frame: comb rail, toe rail, front and rear posts
      rig.add('fde', rbox(0.0064, 0.0090, 0.0900, 0.0018), { p: [FX, 0.0318, 0.0770], wear: 0.46, edgeHi: 0.44 });
      rig.add('fde', rbox(0.0064, 0.0082, 0.0420, 0.0018), { p: [FX, -0.0278, 0.1050], r: [0.06, 0, 0], wear: 0.44, edgeHi: 0.44 });
      rig.add('fde', rbox(0.0064, 0.0530, 0.0104, 0.0018), { p: [FX, 0.0046, 0.0296], wear: 0.40, edgeHi: 0.44 });
      rig.add('fde', rbox(0.0064, 0.0540, 0.0110, 0.0018), { p: [FX, 0.0034, 0.1252], wear: 0.42, edgeHi: 0.44 });
      // two ribs, one raked and one square, so the pocket is three unequal
      // cells instead of one rectangle bisected down the middle
      rig.add('fde', rbox(0.0060, 0.0410, 0.0070, 0.0015), { p: [sx * 0.0180, 0.0060, 0.0700], r: [-0.34, 0, 0], wear: 0.40, edgeHi: 0.44 });
      rig.add('fde', rbox(0.0060, 0.0420, 0.0062, 0.0015), { p: [sx * 0.0180, 0.0052, 0.1010], wear: 0.40, edgeHi: 0.44 });
      // Black polymer battery door sunk into the forward cell. Two materials
      // on the flank is the difference between "moulded shape" and "tan slab":
      // the pocket alone was still one substance at one value.
      rig.add('poly', rbox(0.0040, 0.0320, 0.0286, 0.0022, 2), { p: [sx * 0.0164, 0.0066, 0.0470], wear: 0.30, shade: 0.66, ao: 1.4 });
      rig.add('poly', rbox(0.0030, 0.0088, 0.0074, 0.0012), { p: [sx * 0.0182, 0.0066, 0.0470], wear: 0.62, shade: 1.0, edgeHi: 0.5 });
      rig.add('bright', cyl(0.0019, 0.0019, 0.0030, 6), { p: [sx * 0.0190, 0.0066, 0.0470], r: [0, 0, Math.PI * 0.5], wear: 0.9 });
      // moulded cage number stamped on the pocket floor of the rear cell
      rig.add('poly', rbox(0.0024, 0.0050, 0.0180, 0.0007), { p: [sx * 0.0158, 0.0142, 0.1130], shade: 0.26, ao: 1.4, wear: 0.2 });
    }
    // rubber cheek strip along the comb
    rig.add('rub', rbox(0.0244, 0.0046, 0.0700, 0.0015), { p: [0, 0.0388, 0.0790], r: [-0.010, 0, 0], wear: 0.42, shade: 0.88 });
    for (let i = 0; i < 5; i++) {
      rig.add('rub', rbox(0.0252, 0.0022, 0.0034, 0.0008), { p: [0, 0.0400, 0.0530 + i * 0.0130], wear: 0.60, shade: 0.80, edgeHi: 0.5 });
    }
    // adjustment / release lever hanging under the toe, with a ribbed thumb pad
    rig.add('poly', rbox(0.0182, 0.0128, 0.0300, 0.0032, 2), { p: [0, -0.0352, 0.1070], r: [0.12, 0, 0], wear: 0.55 });
    for (let i = 0; i < 3; i++) {
      rig.add('poly', rbox(0.0156, 0.0028, 0.0040, 0.0008), { p: [0, -0.0412, 0.0966 + i * 0.0062], r: [0.12, 0, 0], wear: 0.8 });
    }
    // sling loop cast into the toe — a real open bar with daylight under it
    rig.add('recv', rbox(0.0132, 0.0058, 0.0170, 0.0018), { p: [0, -0.0308, 0.1216], wear: 0.62, shade: 0.90 });
    rig.add('poly', rbox(0.0086, 0.0028, 0.0104, 0.0009), { p: [0, -0.0320, 0.1216], shade: 0.22, ao: 1.6 });

    // ---- butt pad --------------------------------------------------
    // Sits at the very closest point in frame. It used to be a 40x55mm rubber
    // brick, dead flat to camera, with a five-bar checker moulded on it — the
    // "flat black cylinder placeholder" the critic called out. Now: narrower
    // than the stock so the frame rails step in to meet it, canted off both
    // screen axes, rolled at heel and toe so the outline curves, and carrying
    // only four traction ribs instead of a full checkerboard.
    const PADR = [-0.100, 0, 0.048];
    rig.add('rub', rbox(0.0348, 0.0512, 0.0146, 0.0064, 3), { p: [0, 0.0026, 0.1398], r: PADR, wear: 0.42, shade: 0.90 });
    rig.add('rub', cyl(0.0070, 0.0070, 0.0322, 12), { p: [0, 0.0268, 0.1402], r: [0, 0, Math.PI * 0.5], wear: 0.58, shade: 0.94, edgeHi: 0.5 });
    rig.add('rub', cyl(0.0062, 0.0062, 0.0310, 12), { p: [0, -0.0222, 0.1398], r: [0, 0, Math.PI * 0.5], wear: 0.60, shade: 0.84, edgeHi: 0.5 });
    // hard polymer insert down the middle — a second material at the nearest
    // point in frame is what stops the pad reading as one moulded slab
    rig.add('poly', rbox(0.0148, 0.0300, 0.0048, 0.0022, 2), { p: [0, 0.0026, 0.1458], r: PADR, wear: 0.34, shade: 0.92 });
    rig.add('poly', rbox(0.0100, 0.0248, 0.0026, 0.0012), { p: [0, 0.0026, 0.1478], r: PADR, shade: 0.30, ao: 1.5, wear: 0.2 });
    for (let i = 0; i < 4; i++) {
      const lg = (i & 1) === 0;
      for (const sx of [-1, 1]) {
        rig.add('rub', rbox(lg ? 0.0094 : 0.0066, 0.0036, 0.0046, 0.0012), {
          p: [sx * (lg ? 0.0112 : 0.0098), 0.0158 - i * 0.0092, 0.1464], r: PADR, wear: 0.5, shade: 0.76,
        });
      }
    }
    // perimeter bead — the bright rolled lip that separates pad from stock
    for (const sy of [-1, 1]) {
      rig.add('rub', rbox(0.0344, 0.0034, 0.0134, 0.0014), { p: [0, 0.0026 + sy * 0.0248, 0.1402], r: PADR, wear: 0.74, shade: 0.86, edgeHi: 0.48 });
    }
    for (const sx of [-1, 1]) {
      rig.add('rub', rbox(0.0032, 0.0500, 0.0134, 0.0014), { p: [sx * 0.0168, 0.0026, 0.1402], r: PADR, wear: 0.74, shade: 0.86, edgeHi: 0.48 });
    }
    // QD sling socket the sling actually terminates in (see _bSling). Steel,
    // not bare chrome: at 2.25 env intensity the old bright ring read as a
    // floating white paperclip with nothing attached to it.
    rig.add('poly', cyl(0.0030, 0.0030, 0.0068, 10), { p: [-0.0228, -0.0022, 0.0800], r: [0, 0, Math.PI * 0.5], shade: 0.30, ao: 1.5 });
    rig.add('recv', cyl(0.0054, 0.0054, 0.0044, 12), { p: [-0.0212, -0.0022, 0.0800], r: [0, 0, Math.PI * 0.5], wear: 0.55, shade: 0.86 });
  }

  /* ---- red dot on a riser ----------------------------------------- */
  _bOptic(rig) {
    const cy = 0.0695, cz = -0.1200;
    // riser + mount body clamped to the rail
    rig.add('poly', rbox(0.0230, 0.0130, 0.0620, 0.0026, 2), { p: [0, 0.0448, cz], wear: 0.34 });
    rig.add('poly', rbox(0.0300, 0.0110, 0.0250, 0.0026, 2), { p: [0, 0.0392, cz + 0.0180], wear: 0.40 });
    rig.add('bright', cyl(0.0034, 0.0034, 0.0330, 10), { p: [0, 0.0392, cz + 0.0180], r: [0, 0, Math.PI * 0.5], wear: 0.9 });
    rig.add('poly', rbox(0.0230, 0.0110, 0.0180, 0.0024, 2), { p: [0, 0.0530, cz], wear: 0.30 });

    // tube body — OPEN ended: a capped cylinder would put a black disc exactly
    // where the sight picture belongs
    rig.add('poly', tube(0.0176, 0.0176, 0.0600, 20, true), { p: [0, cy, cz], wear: 0.30 });
    rig.add('poly', tube(0.0192, 0.0176, 0.0090, 20, true), { p: [0, cy, cz - 0.0300], wear: 0.55 });  // objective bell
    rig.add('poly', tube(0.0176, 0.0192, 0.0090, 20, true), { p: [0, cy, cz + 0.0300], wear: 0.55 });  // ocular
    // knurled adjustment turrets
    rig.add('poly', cyl(0.0074, 0.0080, 0.0100, 12), { p: [0, cy + 0.0210, cz + 0.0080], wear: 0.55 });
    rig.add('poly', cyl(0.0074, 0.0080, 0.0100, 12), { p: [0.0210, cy, cz + 0.0080], r: [0, 0, Math.PI * 0.5], wear: 0.55 });
    // battery cap
    rig.add('poly', cyl(0.0086, 0.0092, 0.0074, 12), { p: [-0.0212, cy, cz - 0.0020], r: [0, 0, Math.PI * 0.5], wear: 0.60 });
    // brightness rotary
    rig.add('rub', cyl(0.0064, 0.0064, 0.0084, 10), { p: [-0.0264, cy, cz - 0.0020], r: [0, 0, Math.PI * 0.5], wear: 0.7, shade: 0.85 });
    // kill-flash shade + rubber bumpers
    rig.add('rub', new THREE.TorusGeometry(0.0182, 0.0028, 6, 20), { p: [0, cy, cz - 0.0348], wear: 0.5, shade: 0.85 });
    rig.add('rub', new THREE.TorusGeometry(0.0182, 0.0028, 6, 20), { p: [0, cy, cz + 0.0348], wear: 0.5, shade: 0.85 });
    // internal tube wall — dark, so the lens reads as a real cavity
    rig.add('poly', tube(0.0150, 0.0150, 0.0560, 18, true), { p: [0, cy, cz], shade: 0.10, ao: 1.4 });
  }

  /* ---- folding back-up iron sights --------------------------------- */
  _bSights(rig) {
    // rear: FOLDED flat behind the optic — a deployed aperture would sit right
    // in the middle of the sight picture, which is exactly what a shooter folds
    // it down to avoid. Keeps the ADS view clean and stays authentic.
    // Sat 5mm proud of where it should and rendered at sky value, which put a
    // pale block in the middle of the aimed sight picture 100mm from the lens.
    // Dropped onto the rail and given a fouled, low-value finish so it sinks
    // into the receiver line instead of competing with the dot.
    rig.add('poly', rbox(0.0206, 0.0100, 0.0290, 0.0020), { p: [0, 0.0378, -0.0300], wear: 0.34, shade: 0.72, tintv: [0.80, 0.78, 0.76] });
    rig.add('poly', rbox(0.0176, 0.0062, 0.0230, 0.0018), { p: [0, 0.0430, -0.0330], r: [-0.06, 0, 0], wear: 0.46, shade: 0.70, tintv: [0.80, 0.78, 0.76] });
    rig.add('poly', new THREE.TorusGeometry(0.0042, 0.0020, 6, 12), { p: [0, 0.0434, -0.0450], r: [Math.PI * 0.5, 0, 0], wear: 0.52, shade: 0.70 });
    rig.add('bright', cyl(0.0026, 0.0026, 0.0210, 8), { p: [0, 0.0410, -0.0206], r: [0, 0, Math.PI * 0.5], wear: 0.55, shade: 0.8 });
    // Front: hooded post, also FOLDED. It used to stand deployed at y=0.062,
    // which put it 7px under the optical axis — dead in the middle of the
    // aimed sight picture as a grey blob. A shooter running a red dot folds the
    // front BUIS for exactly that reason, and folding it also matches the rear.
    rig.add('poly', rbox(0.0206, 0.0100, 0.0230, 0.0020), { p: [0, 0.0392, -0.3960], wear: 0.42 });
    // blade assembly hinged forward, lying along the rail
    rig.add('poly', rbox(0.0176, 0.0056, 0.0244, 0.0016), { p: [0, 0.0466, -0.4110], r: [0.05, 0, 0], wear: 0.60 });
    for (const sx of [-1, 1]) {
      rig.add('poly', rbox(0.0040, 0.0042, 0.0212, 0.0011), { p: [sx * 0.0074, 0.0502, -0.4126], r: [0.05, 0, 0], wear: 0.62 });
    }
    rig.add('bright', rbox(0.0026, 0.0030, 0.0130, 0.0006), { p: [0, 0.0502, -0.4210], r: [0.05, 0, 0], wear: 0.8 });
    // detent plunger + hinge pin at the base, so the fold reads as a mechanism
    rig.add('bright', cyl(0.0026, 0.0026, 0.0212, 8), { p: [0, 0.0442, -0.3908], r: [0, 0, Math.PI * 0.5], wear: 0.85 });
    rig.add('poly', cyl(0.0038, 0.0038, 0.0050, 10), { p: [-0.0116, 0.0442, -0.3908], r: [0, 0, Math.PI * 0.5], wear: 0.7, shade: 0.9 });
  }

  /* ---- two-point padded sling -------------------------------------- */
  /**
   * Runs from the handguard QD forward-left, sags below the magwell, and
   * returns to the receiver end-plate QD. Two things this buys:
   *   - it breaks the lower silhouette, which is otherwise a straight
   *     receiver-to-magwell L that reads as a machined block;
   *   - it is the only soft, gravity-shaped thing on the weapon, so it sells
   *     everything around it as hard surface by contrast.
   */
  _bSling(rig) {
    const P0 = [-0.0256, 0.0026, -0.4000];   // front QD on the handguard
    const P1 = [-0.0432, -0.1520, -0.1560];  // control point — the sag
    const P2 = [-0.0246, -0.0022, 0.0798];   // rear QD on the stock socket
    const N = 16;
    const strapW = 0.0212, strapT = 0.0024;
    let px = P0[0], py = P0[1], pz = P0[2];
    for (let i = 1; i <= N; i++) {
      const t = i / N, u = 1 - t;
      const nx = u * u * P0[0] + 2 * u * t * P1[0] + t * t * P2[0];
      const ny = u * u * P0[1] + 2 * u * t * P1[1] + t * t * P2[1];
      const nz = u * u * P0[2] + 2 * u * t * P1[2] + t * t * P2[2];
      _v3a.set(nx - px, ny - py, nz - pz);
      const len = _v3a.length();
      if (len > 1e-5) {
        _v3a.multiplyScalar(1 / len);
        _v3b.set(0, 0, 1);
        _q4.setFromUnitVectors(_v3b, _v3a);
        // The strap twists flat where it is loaded and rolls on the bends. The
        // 1.30 base roll turns its 25mm face toward the eye: without it
        // setFromUnitVectors left the webbing edge-on for the whole run and a
        // two-point sling read as a bare wire strung under the gun.
        const tw = 1.30 + Math.sin(t * Math.PI) * 0.34;
        _e3.set(0, 0, tw, 'XYZ');
        // resolve the orientation ONCE: rig.add writes through the shared _q4
        // scratch, so re-deriving it per primitive would compound the twist
        const q = _q4.clone().multiply(new THREE.Quaternion().setFromEuler(_e3));
        const w = strapW * (1 - 0.14 * Math.sin(t * Math.PI * 2));
        const mx = (px + nx) * 0.5, my = (py + ny) * 0.5, mz = (pz + nz) * 0.5;
        rig.add('web', rbox(strapT, w, len * 1.10, 0.0009), {
          p: [mx, my, mz], q,
          wear: 0.12, shade: 0.90 - 0.10 * Math.sin(t * Math.PI), grime: 1.3,
        });
        // stitched edge binding — two darker beads, offset along the strap's
        // own width axis (not world Y — the strap is twisted)
        _v3c.set(0, 1, 0).applyQuaternion(q);
        for (const s of [-1, 1]) {
          rig.add('web', rbox(strapT * 1.25, 0.0028, len * 1.08, 0.0006), {
            p: [mx + _v3c.x * s * w * 0.42, my + _v3c.y * s * w * 0.42, mz + _v3c.z * s * w * 0.42], q,
            wear: 0.25, shade: 0.74, grime: 1.0,
          });
        }
      }
      px = nx; py = ny; pz = nz;
    }
    // QD swivels at both ends. Steel and small: at bright-chrome value and 7mm
    // radius these two rings read as floating white paperclips with no strap
    // attached, which is worse than having no sling at all.
    rig.add('steel', new THREE.TorusGeometry(0.0052, 0.0016, 5, 10), { p: [-0.0262, -0.0020, -0.4004], r: [0, Math.PI * 0.5, 0.5], wear: 0.85 });
    rig.add('steel', new THREE.TorusGeometry(0.0052, 0.0016, 5, 10), { p: [-0.0254, -0.0064, 0.0806], r: [0, Math.PI * 0.5, -0.3], wear: 0.85 });
    // tri-glide adjuster on the rear run, with the tail doubled back
    rig.add('poly', rbox(0.0044, 0.0280, 0.0088, 0.0012), { p: [-0.0306, -0.0930, 0.0060], r: [0.60, 0, 0.16], wear: 0.5, shade: 0.85 });
    rig.add('poly', rbox(0.0030, 0.0212, 0.0024, 0.0008), { p: [-0.0314, -0.0930, 0.0060], r: [0.60, 0, 0.16], wear: 0.7, shade: 0.8 });
    rig.add('web', rbox(0.0026, 0.0200, 0.0400, 0.0008), { p: [-0.0322, -0.1074, -0.0010], r: [0.72, 0, 0.16], wear: 0.2, shade: 0.82, grime: 1.3 });
  }

  /* ---- the small stuff that keeps a silhouette from reading as a box */
  _bGreeble(rig) {
    // rear-of-receiver serial / roll-mark plate
    rig.add('recv', rbox(0.0030, 0.0090, 0.0300, 0.0008), { p: [-0.0202, -0.0300, -0.0620], shade: 0.9, wear: 0.3 });
    // ejection-port hinge pin
    rig.add('bright', tube(0.0018, 0.0018, 0.0640, 8), { p: [0.0206, -0.0022, -0.0620], wear: 0.8 });
    // charging-handle guide slot in the rear boss
    rig.add('poly', rbox(0.0230, 0.0070, 0.0150, 0.0010), { p: [0, 0.0250, 0.0270], shade: 0.25, ao: 1.3 });
    // sling loop on the receiver end plate — moved forward onto the plate
    // itself; at z=0.030 it was buried inside the stock's front collar
    rig.add('steel', new THREE.TorusGeometry(0.0054, 0.0016, 5, 10), { p: [-0.0196, 0.0040, 0.0222], r: [0, Math.PI * 0.5, 0], wear: 0.85 });
    // trigger-pin heads
    for (const z of [-0.0560, -0.0800]) {
      rig.add('bright', cyl(0.0026, 0.0026, 0.0356, 8), { p: [0, -0.0300, z], r: [0, 0, Math.PI * 0.5], wear: 0.9 });
    }
  }

  /** Build a rig into a pivoted group parented to the gun. */
  _grp(rig, name, px, py, pz) {
    const g = new THREE.Group();
    g.name = name;
    g.position.set(px, py, pz);
    rig.build(g, k => this._matFor(k), name);
    this.gun.add(g);
    return g;
  }

  /* ---- optic glass + collimated reticle ---------------------------- */
  _bLens() {
    const cy = 0.0695, cz = -0.1200;
    const grp = new THREE.Group();
    grp.name = 'opticGlass';
    grp.position.set(0, cy, cz);

    // Shallow concave objective — a lathed cap so the highlight bends.
    // The UVs are re-derived as a planar projection down the bore so the
    // coating/vignette texture is genuinely radial about the optical axis,
    // rather than depending on how LatheGeometry happened to lay out v.
    const R = 0.0150;
    const mkLens = (sag) => {
      const pts = [];
      for (let i = 0; i <= 10; i++) {
        const r = (i / 10) * R;
        pts.push(new THREE.Vector2(r, sag * (r / R) * (r / R)));
      }
      const g = new THREE.LatheGeometry(pts, 28);
      g.rotateX(-Math.PI * 0.5);
      const p = g.attributes.position;
      const uv = new Float32Array(p.count * 2);
      for (let i = 0; i < p.count; i++) {
        uv[i * 2] = 0.5 + p.getX(i) / (2 * R);
        uv[i * 2 + 1] = 0.5 + p.getY(i) / (2 * R);
      }
      g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      return g;
    };
    const lensRig = new Rig();
    lensRig.add('lens', mkLens(0.0022), { p: [0, 0, -0.0272], r: [Math.PI, 0, 0], shade: 1.0, wear: 0, grime: 0, dust: 0 });
    lensRig.add('lens', mkLens(0.0016), { p: [0, 0, 0.0272], shade: 0.92, wear: 0, grime: 0, dust: 0 });
    lensRig.build(grp, () => this.matLens, 'lens');
    for (const ch of grp.children) { ch.renderOrder = 3; ch.castShadow = false; ch.receiveShadow = false; }

    // ---- coating reflection --------------------------------------------
    // A stretched sky highlight lying ON the objective. Without it the glass
    // is only a tint and the lens still reads as an open hole; a real coated
    // objective always catches one bright oblique smear.
    const coat = new THREE.Mesh(
      new THREE.CircleGeometry(R * 0.94, 26),
      new THREE.MeshBasicMaterial({
        map: this.texGlow, color: 0x86a6c6, transparent: true, opacity: 0.19,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true, toneMapped: true,
      }),
    );
    coat.position.set(-0.0034, 0.0042, -0.0281);
    coat.scale.set(1.0, 0.44, 1);
    coat.rotation.z = -0.62;
    coat.renderOrder = 4;
    grp.add(coat);
    this._lensCoat = coat;

    // ---- reticle: hot core + bloom skirt + outer flare -------------------
    // The old reticle was a 0.82mm disc: physically honest for 2 MOA and
    // completely invisible — it landed on two pixels at the aiming distance,
    // which is why the sight read as having no reticle at all. Every shipped
    // shooter draws the dot several times over-size with a bloom skirt, and
    // that is what is done here.
    // Discs, not quads: the dot texture is radial, so a square carrier can only
    // ever contribute corners that do not belong to the emitter.
    const mkDot = (rad, op, ord) => {
      const m = this.matReticle.clone();
      m.map = this.texDot;
      m.opacity = op;
      const mesh = new THREE.Mesh(new THREE.CircleGeometry(rad, 24), m);
      mesh.renderOrder = ord;
      mesh.frustumCulled = false;
      return mesh;
    };
    const dot = mkDot(0.00260, 1.00, 8);
    dot.position.set(0, 0, -0.0238);
    grp.add(dot);
    const halo = mkDot(0.00760, 0.34, 7);
    halo.material.color.setRGB(2.30, 0.16, 0.05);
    halo.position.set(0, 0, -0.0240);
    grp.add(halo);
    const flare = mkDot(0.01760, 0.13, 6);
    flare.material.color.setRGB(1.05, 0.07, 0.02);
    flare.position.set(0, 0, -0.0242);
    grp.add(flare);
    this._reticleDot = dot;
    this._reticleHalo = halo;
    this._reticleFlare = flare;

    this.gun.add(grp);
    this.opticGlass = grp;
  }

  /* ---- curved 30-round magazine ------------------------------------ */
  _magRig(rig) {
    const segs = 11, segLen = 0.0176;
    let px = 0, py = 0, pz = 0, th = 0.0800;
    for (let i = 0; i < segs; i++) {
      const dy = -Math.cos(th), dz = -Math.sin(th);
      const cx = px + dy * 0 , cyy = py + dy * segLen * 0.5, czz = pz + dz * segLen * 0.5;
      const taper = 1 - i * 0.004;
      rig.add('fde', rbox(0.0246 * taper, segLen * 1.06, 0.0352 * taper, 0.0022), {
        p: [cx, cyy, czz], r: [th, 0, 0], wear: 0.24 + i * 0.014,
      });
      // grip ribs down the sides — proud enough to actually cast a line, with
      // a shadowed valley under each one. At 0.8mm of relief they vanished and
      // the magazine read as a smooth tan curve.
      if (i > 1 && i < segs - 1 && (i & 1) === 0) {
        rig.add('fde', rbox(0.0294 * taper, 0.0054, 0.0304, 0.0014), {
          p: [cx, cyy, czz], r: [th, 0, 0], wear: 0.62, edgeHi: 0.44,
        });
        rig.add('fde', rbox(0.0268 * taper, 0.0038, 0.0322, 0.0010), {
          p: [cx, cyy - 0.0052 * Math.cos(th), czz - 0.0052 * Math.sin(th)], r: [th, 0, 0],
          shade: 0.66, ao: 1.4, wear: 0.2,
        });
      }
      // moulded index arrow / lot stamp on the outboard face, every third seg
      if (i === 3 || i === 7) {
        rig.add('poly', rbox(0.0030, 0.0074, 0.0170, 0.0008), {
          p: [-0.0130 * taper, cyy, czz], r: [th, 0, 0], shade: 0.30, ao: 1.4, wear: 0.3,
        });
      }
      px += 0; py += dy * segLen; pz += dz * segLen;
      th += 0.0345;
    }
    // feed lips + follower peeking out of the top
    rig.add('fde', rbox(0.0250, 0.0100, 0.0356, 0.0018), { p: [0, 0.0030, 0.0000], r: [0.0700, 0, 0], wear: 0.62 });
    rig.add('poly', rbox(0.0176, 0.0070, 0.0250, 0.0012), { p: [0, 0.0072, -0.0010], r: [0.0700, 0, 0], shade: 0.5, wear: 0.4 });
    rig.add('bright', rbox(0.0150, 0.0058, 0.0250, 0.0016), { p: [0, 0.0108, -0.0016], r: [0.0700, 0, 0], wear: 0.9 });
    // floorplate
    const dyE = -Math.cos(th), dzE = -Math.sin(th);
    rig.add('poly', rbox(0.0272, 0.0120, 0.0384, 0.0026), {
      p: [0, py + dyE * 0.0056, pz + dzE * 0.0056], r: [th, 0, 0], wear: 0.70,
    });
    rig.add('poly', rbox(0.0290, 0.0056, 0.0180, 0.0016), {
      p: [0, py + dyE * 0.0110, pz + dzE * 0.0110 - 0.0090], r: [th, 0, 0], wear: 0.75,
    });
    // witness-hole strip — BOTH flanks. The single right-hand strip faced away
    // from the eye, so the visible side of the magazine carried no detail at all.
    for (let i = 0; i < 4; i++) {
      const hy = -0.0300 - i * 0.0330, hz = -0.0060 - i * 0.0110;
      for (const sx of [-1, 1]) {
        rig.add('poly', cyl(0.0026, 0.0026, 0.0034, 8), {
          p: [sx * 0.0124, hy, hz], r: [0, 0, Math.PI * 0.5], shade: 0.22, ao: 1.45,
        });
        rig.add('fde', new THREE.TorusGeometry(0.0036, 0.0009, 5, 10), {
          p: [sx * 0.0130, hy, hz], r: [0, Math.PI * 0.5, 0], wear: 0.55, edgeHi: 0.5,
        });
      }
    }
    return { endY: py, endZ: pz, endTh: th };
  }

  _bMagazine() {
    const rig = new Rig();
    this._magInfo = this._magRig(rig);
    this.magMesh = this._grp(rig, 'magazine', 0, -0.0520, -0.1300);
  }

  /* ---- charging handle --------------------------------------------- */
  _bChargingHandle() {
    const rig = new Rig();
    rig.add('bright', rbox(0.0270, 0.0072, 0.0470, 0.0014), { p: [0, 0.0248, 0.0300], wear: 0.62 });
    rig.add('bright', rbox(0.0104, 0.0110, 0.0130, 0.0016), { p: [0, 0.0250, 0.0090], wear: 0.55 });
    // extended latch on the left, the surface a shooter actually touches
    // Wear pulled back off the latch. At 1.0 on a bright-steel base the whole
    // charging handle blew out into one white smear on the left silhouette.
    rig.add('bright', rbox(0.0300, 0.0092, 0.0210, 0.0022, 2), { p: [-0.0210, 0.0252, 0.0416], r: [0, 0, 0.06], wear: 0.40, edgeHi: 0.5, shade: 0.86 });
    for (let i = 0; i < 5; i++) {
      rig.add('bright', rbox(0.0026, 0.0110, 0.0170, 0.0006), { p: [-0.0300 + i * 0.0058, 0.0256, 0.0420], wear: 0.38, shade: 0.90 });
    }
    rig.add('bright', cyl(0.0028, 0.0028, 0.0090, 8), { p: [-0.0070, 0.0250, 0.0430], r: [0, 0, Math.PI * 0.5], wear: 0.7 });
    this.chargingHandle = this._grp(rig, 'chargingHandle', 0, 0, 0);
  }

  /* ---- bolt carrier seen through the port -------------------------- */
  _bBolt() {
    const rig = new Rig();
    // carrier body — parkerised, not bright: it should sit a value below the
    // chrome bolt face so the port cavity has internal contrast
    rig.add('steel', rbox(0.0150, 0.0230, 0.0700, 0.0026), { p: [0.0074, 0.0122, -0.0600], wear: 0.62, shade: 0.80 });
    // machined relief cuts along the carrier — visible as the bolt cycles
    for (let i = 0; i < 3; i++) {
      rig.add('steel', rbox(0.0164, 0.0044, 0.0110, 0.0009), { p: [0.0074, 0.0196, -0.0400 - i * 0.0180], shade: 0.55, ao: 1.3, wear: 0.5 });
    }
    rig.add('bright', tube(0.0088, 0.0088, 0.0180, 12), { p: [0.0088, 0.0124, -0.0900], wear: 0.85, shade: 0.92 });
    rig.add('poly', tube(0.0044, 0.0044, 0.0130, 10), { p: [0.0088, 0.0124, -0.0960], shade: 0.1, ao: 1.4 });
    // extractor claw
    rig.add('bright', rbox(0.0060, 0.0070, 0.0130, 0.0012), { p: [0.0136, 0.0180, -0.0910], wear: 0.9 });
    // chambered round — a sliver of warm brass in the port that no amount of
    // grey primitive can fake
    rig.add('brass', tube(0.0047, 0.0049, 0.0230, 12), { p: [0.0100, 0.0128, -0.1055], wear: 0.45, shade: 0.95 });
    rig.add('brass', tube(0.0049, 0.0044, 0.0040, 12), { p: [0.0100, 0.0128, -0.0928], wear: 0.9, edgeHi: 0.55 });
    this.bolt = this._grp(rig, 'bolt', 0, 0, 0);
  }

  /* ---- ejection-port dust cover (hinged on the port pin) ----------- */
  _bDustCover() {
    const rig = new Rig();
    // outer skin, inner ribbed face, rolled lip and the spring-loaded catch
    rig.add('recv', rbox(0.0046, 0.0264, 0.0584, 0.0016), { p: [0.0024, 0.0140, 0], wear: 0.62 });
    rig.add('recv', rbox(0.0026, 0.0210, 0.0500, 0.0008), { p: [-0.0004, 0.0140, 0], shade: 0.62, ao: 1.35, wear: 0.3 });
    for (let i = 0; i < 3; i++) {
      rig.add('recv', rbox(0.0018, 0.0026, 0.0480, 0.0006), { p: [-0.0012, 0.0064 + i * 0.0076, 0], shade: 0.7, wear: 0.35 });
    }
    rig.add('recv', rbox(0.0030, 0.0072, 0.0604, 0.0012), { p: [0.0046, 0.0252, 0], wear: 0.78, edgeHi: 0.46 });
    rig.add('bright', rbox(0.0036, 0.0062, 0.0092, 0.0012), { p: [0.0052, 0.0060, 0.0300], wear: 0.95, edgeHi: 0.5 });
    // torsion spring wrapped on the hinge pin
    for (let i = 0; i < 4; i++) {
      rig.add('bright', new THREE.TorusGeometry(0.0030, 0.0007, 4, 10), { p: [0.0018, -0.0004, -0.0260 + i * 0.0026], r: [0, Math.PI * 0.5, 0], wear: 0.8, shade: 0.85 });
    }
    this.dustCover = this._grp(rig, 'dustCover', 0.0206, -0.0022, -0.0620);
  }

  /* ---- trigger ------------------------------------------------------ */
  _bTrigger() {
    const rig = new Rig();
    rig.add('bright', rbox(0.0064, 0.0170, 0.0062, 0.0016), { p: [0, -0.0110, -0.0016], r: [0.16, 0, 0], wear: 0.85 });
    rig.add('bright', rbox(0.0070, 0.0058, 0.0090, 0.0018), { p: [0, -0.0196, -0.0044], r: [0.42, 0, 0], wear: 1.0 });
    this.trigger = this._grp(rig, 'trigger', 0, -0.0290, -0.0560);
  }

  // ==================================================================
  //  GLOVED HANDS
  //  Canonical frame: the gripped cylinder runs along +Y, the hand sits on
  //  sx*X, the palm covers the back (+Z) and the fingers wrap the front (-Z).
  // ==================================================================

  _capsule(rig, key, ax, ay, az, bx, by, bz, r, o) {
    _v3a.set(bx - ax, by - ay, bz - az);
    const len = _v3a.length();
    if (len < 1e-5) return;
    _v3a.multiplyScalar(1 / len);
    _v3b.set(0, 1, 0);
    _q4.setFromUnitVectors(_v3b, _v3a);
    const g = new THREE.CapsuleGeometry(r, Math.max(0.0005, len - r * 0.62), 4, 10);
    rig.add(key, g, {
      p: [(ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5],
      q: _q4.clone(), wear: o?.wear ?? 0.06, shade: o?.shade ?? 1, edgeHi: 0.78,
      grime: o?.grime ?? 1.25, dust: o?.dust ?? 1,
    });
  }

  /** A seam / stitch band wrapped around a limb segment at a point. */
  _seam(rig, key, x, y, z, dx, dy, dz, r, o) {
    _v3a.set(dx, dy, dz);
    if (_v3a.lengthSq() < 1e-10) return;
    _v3a.normalize();
    _v3b.set(0, 0, 1);
    _q4.setFromUnitVectors(_v3b, _v3a);
    rig.add(key, new THREE.TorusGeometry(r, o?.t ?? 0.0016, 5, 12), {
      p: [x, y, z], q: _q4.clone(), wear: o?.wear ?? 0.16, shade: o?.shade ?? 0.86,
      edgeHi: 0.8, grime: 1.2,
    });
  }

  /**
   * A gloved hand.
   *   sx    +1 = hand sits on the +X side of the gripped cylinder
   *   R     radius of the thing being gripped
   *   o.trigger  route the index finger up into the trigger guard instead of
   *              wrapping it round the grip
   *
   * The fingers are deliberately thinner than the knuckle spacing so there is
   * daylight between them: a hand modelled with touching digits reads as a
   * mitten no matter how many segments it has.
   */
  _handRig(rig, sx, R, o) {
    const SEG = [0.0344, 0.0238, 0.0176];
    // Slimmer than the knuckle pitch on purpose. At 8.8mm radius against a
    // 23.4mm pitch the digits touched and the hand read as a mitten; 8.2 against
    // 24.6 leaves 8mm of daylight between every finger at every pose.
    const RAD = [0.0082, 0.0073, 0.0062];
    const HY = [0.0368, 0.0122, -0.0126, -0.0372];
    const trig = !!o?.trigger;
    // Contact radii. R alone cannot drive both, because the palm sits on the
    // FLAT of the grip while the fingers ride a circle around it — deriving
    // both from one number buried the palm inside the handguard and left the
    // fingers floating a couple of millimetres clear of it.
    const PALM = o?.palm ?? (R * 0.66 + 0.0122);
    const WRAP = o?.wrap ?? (R + 0.0116);

    // ---- palm: a slim wedge hugging the back and outboard face ------
    const pcx = sx * PALM, pcz = R * 0.40;
    rig.add('glove', rbox(0.0238, 0.0860, 0.0404, 0.0100, 3), {
      p: [pcx, 0.0020, pcz], r: [0, sx * -0.40, 0], wear: 0.05, shade: 0.98, grime: 1.3,
    });
    // glove back panel — a separate darker piece with a raised seam border,
    // which is the single clearest "this is a glove, not a mitten" cue
    rig.add('gloveDark', rbox(0.0060, 0.0700, 0.0330, 0.0050, 2), {
      p: [sx * (PALM + 0.0108), 0.0040, pcz + 0.0010], r: [0, sx * -0.40, 0],
      wear: 0.14, shade: 0.90, grime: 1.4,
    });
    for (const sy of [-1, 1]) {
      rig.add('gloveDark', rbox(0.0044, 0.0040, 0.0350, 0.0012), {
        p: [sx * (PALM + 0.0112), 0.0040 + sy * 0.0372, pcz + 0.0010], r: [0, sx * -0.40, 0],
        wear: 0.24, shade: 0.80,
      });
    }
    // knuckle pad across the front of the hand + one moulded pad per knuckle
    rig.add('gloveDark', rbox(0.0142, 0.0760, 0.0186, 0.0064, 2), {
      p: [sx * (PALM - 0.0018), 0.0068, -R * 0.14],
      r: [0, sx * -0.28, 0], wear: 0.12, shade: 0.92, grime: 1.4,
    });
    for (let f = 0; f < 4; f++) {
      const a0 = -0.46 - f * 0.088;
      const kr = WRAP - 0.0014;
      rig.add('gloveDark', new THREE.SphereGeometry(0.0108 - f * 0.0005, 10, 7), {
        p: [sx * Math.cos(a0) * kr, HY[f] + 0.0016, Math.sin(a0) * kr],
        s: [1, 0.86, 1.05], wear: 0.20, shade: 0.88, edgeHi: 0.8, grime: 1.4,
      });
    }
    // thenar / thumb muscle
    rig.add('glove', rbox(0.0188, 0.0290, 0.0258, 0.0080, 2), {
      p: [sx * (PALM - 0.0052), 0.0290, R * 0.20], r: [0, sx * -0.48, 0], wear: 0.05, grime: 1.3,
    });
    // ---- palm-side grip patch --------------------------------------
    // The suede/silicone palm panel, on the face that is actually loaded
    // against the grip. Without it the hand had no visible contact surface and
    // read as hovering rather than gripping.
    rig.add('gloveDark', rbox(0.0088, 0.0640, 0.0250, 0.0038, 2), {
      p: [sx * (PALM - 0.0126), 0.0000, R * 0.16], r: [0, sx * -0.34, 0],
      wear: 0.30, shade: 0.84, ao: 1.35, grime: 1.5, edgeHi: 0.7,
    });
    // stitched border around that patch — two beads top and bottom
    for (const sy of [-1, 1]) {
      rig.add('gloveDark', rbox(0.0060, 0.0034, 0.0242, 0.0010), {
        p: [sx * (PALM - 0.0130), sy * 0.0332, R * 0.16], r: [0, sx * -0.34, 0],
        wear: 0.34, shade: 0.74,
      });
    }
    // metacarpal tendon ridges across the back of the hand, running out of the
    // knuckle pads — the cheapest possible "this is a hand" signal
    for (let f = 0; f < 4; f++) {
      rig.add('glove', rbox(0.0044, 0.0074, 0.0230, 0.0020), {
        p: [sx * (PALM + 0.0058), HY[f] * 0.62 + 0.0030, pcz - 0.0110],
        r: [0, sx * -0.40, 0], wear: 0.10, shade: 1.04, grime: 1.2,
      });
    }

    // ---- wrist + cuff + sleeved forearm -----------------------------
    const wx = pcx, wy = -0.0446, wz = pcz + 0.0020;
    // Forearm axis. Optional because it must NOT ride the grip roll: rolling
    // the support hand around the handguard also swung its forearm from
    // down-and-back to straight-out-left, which laid a thick olive tube
    // horizontally across the middle of the frame. o.arm is supplied
    // pre-derotated by _makeHand so the arm keeps dropping out of frame no
    // matter how the wrist is rolled.
    const dx = o?.arm ? o.arm[0] : sx * 0.34;
    const dy = o?.arm ? o.arm[1] : -0.87;
    const dz = o?.arm ? o.arm[2] : -0.35;
    this._capsule(rig, 'gloveDark', wx, wy, wz,
      wx + dx * 0.024, wy + dy * 0.024, wz + dz * 0.024, 0.0192, { wear: 0.14, shade: 0.88 });
    // elasticated cuff, then the hook-and-loop closure tab over it
    this._seam(rig, 'gloveDark', wx + dx * 0.026, wy + dy * 0.026, wz + dz * 0.026, dx, dy, dz, 0.0202, { t: 0.0034, shade: 0.78, wear: 0.24 });
    rig.add('web', rbox(0.0300, 0.0150, 0.0058, 0.0014), {
      p: [wx + dx * 0.034 - sx * 0.0130, wy + dy * 0.034, wz + dz * 0.034],
      r: [1.20, 0, sx * 0.30], wear: 0.24, shade: 0.78, grime: 1.4,
    });
    // Sleeve: an OLIVE combat shirt, not more glove. Three tapering sections
    // running well past the frame edge, so the arm reads as continuing into a
    // body instead of terminating in a stump.
    // Radii pulled in ~9%. At 28mm the last section was a 113mm-diameter
    // forearm — thicker than the receiver is long is not an arm, it is a pillar,
    // and it was wide enough to swallow the whole handguard behind it.
    this._capsule(rig, 'sleeve',
      wx + dx * 0.030, wy + dy * 0.030, wz + dz * 0.030,
      wx + dx * 0.098, wy + dy * 0.098, wz + dz * 0.098, 0.0206, { wear: 0.05, shade: 0.96, grime: 1.4 });
    this._capsule(rig, 'sleeve',
      wx + dx * 0.092, wy + dy * 0.092, wz + dz * 0.092,
      wx + dx * 0.215, wy + dy * 0.215, wz + dz * 0.215, 0.0238, { wear: 0.04, shade: 0.92, grime: 1.5 });
    this._capsule(rig, 'sleeve',
      wx + dx * 0.205, wy + dy * 0.205, wz + dz * 0.205,
      wx + dx * 0.345, wy + dy * 0.345, wz + dz * 0.345, 0.0258, { wear: 0.03, shade: 0.86, grime: 1.5 });
    // the sleeve's own cuff seam, where the shirt overlaps the glove gauntlet
    this._seam(rig, 'sleeve', wx + dx * 0.036, wy + dy * 0.036, wz + dz * 0.036, dx, dy, dz, 0.0246, { t: 0.0034, shade: 0.76, wear: 0.16 });
    this._seam(rig, 'sleeve', wx + dx * 0.096, wy + dy * 0.096, wz + dz * 0.096, dx, dy, dz, 0.0258, { t: 0.0030, shade: 0.80, wear: 0.14 });
    // Two rolled fabric folds along the forearm. There were six evenly spaced
    // rings of 2.4mm section here and the arm read as a length of bamboo; a
    // sleeve creases where the elbow pulls it, not on a regular pitch.
    this._seam(rig, 'sleeve', wx + dx * 0.132, wy + dy * 0.132, wz + dz * 0.132, dx, dy, dz, 0.0280, { t: 0.0017, shade: 0.90, wear: 0.10 });
    this._seam(rig, 'sleeve', wx + dx * 0.246, wy + dy * 0.246, wz + dz * 0.246, dx, dy, dz, 0.0296, { t: 0.0015, shade: 0.92, wear: 0.08 });
    // webbing cuff strap with a buckle, high on the forearm — one hard,
    // man-made edge to sell the soft fabric around it
    rig.add('web', rbox(0.0072, 0.0530, 0.0530, 0.0016), {
      p: [wx + dx * 0.152 + sx * 0.0110, wy + dy * 0.152, wz + dz * 0.152],
      r: [0.55, 0, sx * 0.22], wear: 0.18, shade: 0.80, grime: 1.5,
    });
    rig.add('gloveDark', rbox(0.0044, 0.0130, 0.0088, 0.0012), {
      p: [wx + dx * 0.150 + sx * 0.0170, wy + dy * 0.150, wz + dz * 0.150 - 0.0080],
      r: [0.55, 0, sx * 0.22], wear: 0.40, shade: 0.72, edgeHi: 0.6,
    });

    // ---- four fingers curling around the front ----------------------
    for (let f = 0; f < 4; f++) {
      if (trig && f === 0) continue;                 // index goes to the trigger
      const hy = HY[f];
      const a0 = -0.46 - f * 0.088;
      const kr = WRAP;
      let px = sx * Math.cos(a0) * kr;
      let py = hy;
      let pz = Math.sin(a0) * kr;
      let ang = a0 - Math.PI * 0.60;
      const spread = 1 + f * 0.03;
      // Per-finger curl. Every digit used to close at the same rate, so the
      // four tips landed on a regular pitch and the hand read as a stack of
      // identical lumps. A real grip closes harder toward the little finger:
      // index barely past vertical, pinky fully rolled under.
      const curl = 0.780 + f * 0.062;
      // gauntlet ring at the base of the finger — where the glove's back panel
      // is stitched onto the digit. Also the darkest band on the finger, which
      // is what gives the eye an unambiguous joint to read.
      this._seam(rig, 'gloveDark', px, py, pz,
        sx * Math.cos(ang), 0, Math.sin(ang), RAD[0] + 0.0014, { t: 0.0016, shade: 0.80, wear: 0.24 });
      for (let s = 0; s < 3; s++) {
        const L = SEG[s] * spread * (f === 3 ? 0.86 : 1) * (f === 0 ? 1.04 : 1);
        const nx = px + sx * Math.cos(ang) * L;
        const ny = py - 0.0018 * s;
        const nz = pz + Math.sin(ang) * L;
        this._capsule(rig, 'glove', px, py, pz, nx, ny, nz, RAD[s], { wear: 0.06, shade: s === 2 ? 0.96 : 1 });
        // padded panel down the BACK of each segment — a slightly proud, darker
        // capsule offset outward from the wrap circle. Two materials on one
        // digit is what separates a gloved finger from a sausage.
        {
          const mxp = (px + nx) * 0.5, myp = (py + ny) * 0.5, mzp = (pz + nz) * 0.5;
          const rr = Math.hypot(mxp, mzp) || 1;
          const oxu = mxp / rr, ozu = mzp / rr, off = RAD[s] * 0.52;
          this._capsule(rig, 'gloveDark',
            px + oxu * off, py + 0.0004, pz + ozu * off,
            nx + oxu * off, ny + 0.0004, nz + ozu * off,
            RAD[s] * 0.66, { wear: 0.20, shade: 0.90 });
        }
        // stitched seam at each joint
        if (s > 0) this._seam(rig, 'gloveDark', px, py, pz, nx - px, ny - py, nz - pz, RAD[s] + 0.0011, { t: 0.0012, shade: 0.86, wear: 0.20 });
        px = nx; py = ny; pz = nz;
        ang -= curl - s * 0.055;
      }
      // reinforced fingertip pad — tucked INSIDE the capsule cap so it reads as
      // a panel on the tip, not a bulb stuck on the end
      rig.add('gloveDark', new THREE.SphereGeometry(RAD[2] * 0.96, 9, 7), {
        p: [px, py, pz], s: [1, 0.86, 1], wear: 0.26, shade: 0.94, edgeHi: 0.85,
      });
    }

    // ---- trigger finger: out of the wrap, up into the guard ---------
    if (trig) {
      const CP = [
        [sx * 0.0262, 0.0352, -0.0148],
        [sx * 0.0214, 0.0478, -0.0296],
        [sx * 0.0092, 0.0562, -0.0262],
        [sx * 0.0018, 0.0584, -0.0104],
      ];
      for (let s = 0; s < 3; s++) {
        const a = CP[s], b = CP[s + 1];
        this._capsule(rig, 'glove', a[0], a[1], a[2], b[0], b[1], b[2], RAD[s], { wear: 0.07, shade: s === 2 ? 0.96 : 1 });
        if (s > 0) this._seam(rig, 'gloveDark', a[0], a[1], a[2], b[0] - a[0], b[1] - a[1], b[2] - a[2], RAD[s] + 0.0009, { t: 0.0010, shade: 0.90 });
      }
      rig.add('gloveDark', new THREE.SphereGeometry(RAD[2] * 0.96, 9, 7), {
        p: CP[3], s: [1, 0.86, 1], wear: 0.28, shade: 0.94, edgeHi: 0.85,
      });
    }

    // ---- thumb ------------------------------------------------------
    // Parameterised in round 3. On the firing hand the thumb wraps back and
    // down across the grip (thumbDY < 0, hard curl). On the SUPPORT hand it has
    // to do the opposite: lie along the top of the handguard pointing FORWARD
    // with almost no curl. A thumb over the bore is the single most legible
    // "this hand is holding this weapon" cue there is, and with the hand rolled
    // so the palm is underneath it is the only part of the grip the camera can
    // see at all.
    {
      const tA = o?.thumbA ?? 0.75;
      const tCurl = o?.thumbCurl ?? 0.62;
      // thumbLat splits each segment between wrapping AROUND the gripped
      // cylinder and running ALONG it. 1.0 (default) is the firing hand's
      // thumb curling over the grip; ~0.3 with an explicit axial travel is the
      // support thumb lying down the top of the handguard toward the muzzle.
      const tLat = o?.thumbLat ?? 1.0;
      const tAx = o?.thumbAx || null;
      // thumbOff is the angle between the thumb's root ray and the direction it
      // sets off in. The firing hand's 0.72*PI drives the tip INWARD onto the
      // grip, which is what you want when the thumb is closing on something.
      // A support thumb laid on a handguard has to leave tangentially (-PI/2)
      // or it buries itself in the polymer and renders as a stub.
      const tOff = o?.thumbOff ?? (Math.PI * 0.72);
      const kr = WRAP - 0.0026;
      let px = sx * Math.cos(tA) * kr, py = o?.thumbY0 ?? 0.0400, pz = Math.sin(tA) * kr;
      let ang = tA - tOff;
      const L = [0.0330, 0.0250];
      const TR = [0.0106, 0.0090];
      for (let s = 0; s < 2; s++) {
        const wl = L[s] * tLat;
        const nx = px + sx * Math.cos(ang) * wl;
        const ny = py + (tAx ? tAx[s] : (-0.0130 - s * 0.0080));
        const nz = pz + Math.sin(ang) * wl;
        this._capsule(rig, 'glove', px, py, pz, nx, ny, nz, TR[s], { wear: 0.07, shade: s === 1 ? 0.96 : 1 });
        if (s === 1) this._seam(rig, 'gloveDark', px, py, pz, nx - px, ny - py, nz - pz, TR[s] + 0.0010, { t: 0.0011, shade: 0.90 });
        px = nx; py = ny; pz = nz;
        ang -= tCurl;
      }
      rig.add('gloveDark', new THREE.SphereGeometry(TR[1] * 0.96, 9, 7), {
        p: [px, py, pz], s: [1, 0.86, 1], wear: 0.26, shade: 0.94, edgeHi: 0.85,
      });
    }
  }

  _makeHand(sx, R, roll, o) {
    const outer = new THREE.Group();
    const inner = new THREE.Group();
    inner.rotation.y = roll;
    let opt = o;
    // o.armGun is a forearm direction expressed in the GUN's frame. Undo the
    // outer Rx(-90) that _buildHands applies, then the inner grip roll, so the
    // rig can author it in hand space and the arm ends up pointing where the
    // art direction wants regardless of how the wrist is rolled.
    if (o?.armGun) {
      const g = o.armGun;
      const a = g[0], b = -g[2], c = g[1];        // gun -> post-roll hand space
      const cr = Math.cos(roll), sr = Math.sin(roll);
      opt = Object.assign({}, o, { arm: [a * cr - c * sr, b, a * sr + c * cr] });
    }
    const rig = new Rig();
    this._handRig(rig, sx, R, opt);
    rig.build(inner, k => this._matFor(k), sx > 0 ? 'handR' : 'handL');
    outer.add(inner);
    return outer;
  }

  _buildHands() {
    // firing hand on the pistol grip (grip raked back 24.6 deg), index on the
    // trigger rather than wrapped with the other three
    // grip is a 19mm ellipse: palm face at 19+11.9, fingers ride 19+8.8
    this.handR = this._makeHand(1, 0.0190, -0.10, { trigger: true, palm: 0.0302, wrap: 0.0280 });
    this.handR.position.set(0.0000, -0.0840, -0.0330);
    this.handR.rotation.set(-0.4300, 0.0000, 0.0000);
    this.gun.add(this.handR);

    // Support hand on the handguard. The handguard's panelled octagon has an
    // outer radius of 27.4mm, so the palm plane sits at 27.4+11.9 and the
    // fingers ride a 27.4+8.8 circle — anything looser and the hand hovers.
    //
    // ROUND 3 — the roll. This is the number that decided whether the weapon
    // read as held or as floating, and it was wrong by about 127 degrees.
    //
    // Work it out from where the eye actually is. With the hip pose the camera
    // sits on the weapon's LEFT and slightly above it, so the visible arc of
    // the handguard cross-section runs roughly 74deg..254deg measured from the
    // weapon's +X axis. At the old roll of +0.68 the wrist landed at 126deg —
    // ABOVE the handguard — so the forearm rose UP to meet the weapon and then
    // hung back down past it, and the four fingers wrapped from 167deg round to
    // 288deg, i.e. straight out of sight underneath. What the camera got was a
    // featureless tan dome on top of the handguard with a tube under it: a
    // sausage, exactly as the panel called it.
    //
    // At -1.530 the whole hand rotates a third of a turn: the heel of the palm
    // and the wrist drop to 252deg — genuinely UNDER the handguard, so the
    // forearm now approaches from below where an arm belongs — the fingers
    // wrap 294deg round to 55deg so their tips crest the top far edge and are
    // visible over the handguard, and the thumb (see thumbA) lies along the
    // top-left shoulder pointing at the muzzle. Palm under, fingertips over the
    // top, thumb along the bore: that is a support grip, and every part of it
    // that the camera can see is a part that says "holding".
    //
    // wrap opened 1.8mm to 0.0378 because the digits are polylines, not arcs:
    // the chords cut inside the wrap circle and at 0.0360 the middle phalanges
    // sank into the polymer instead of lying on it.
    this.handL = this._makeHand(-1, 0.0268, -1.3400, {
      palm: 0.0391, wrap: 0.0378, armGun: [-0.232, -0.845, 0.482],
      thumbA: 1.6020, thumbOff: -Math.PI * 0.5, thumbCurl: -0.42,
      thumbLat: 0.520, thumbY0: 0.0080, thumbAx: [0.0230, 0.0170],
    });
    this.handL.position.set(0.0000, -0.0020, -0.3200);
    this.handL.rotation.set(-Math.PI * 0.5, 0.0000, 0.0000);
    this.gun.add(this.handL);

    this._handLBase = this.handL.position.clone();
    this._handLBaseRot = this.handL.rotation.clone();
  }

  // ==================================================================
  //  MUZZLE FLASH
  // ==================================================================

  _canvas(size) {
    const c = document.createElement('canvas');
    c.width = c.height = size;
    return c;
  }

  _tex(canvas) {
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = true;
    t.needsUpdate = true;
    return t;
  }

  _makeFlashTextures() {
    // --- soft radial glow -------------------------------------------
    const N = 256;
    const gc = this._canvas(N), g = gc.getContext('2d');
    let grd = g.createRadialGradient(N / 2, N / 2, 0, N / 2, N / 2, N / 2);
    grd.addColorStop(0.00, 'rgba(255,252,236,1)');
    grd.addColorStop(0.13, 'rgba(255,226,150,0.92)');
    grd.addColorStop(0.34, 'rgba(255,163,58,0.44)');
    grd.addColorStop(0.62, 'rgba(214,96,20,0.13)');
    grd.addColorStop(1.00, 'rgba(120,40,0,0)');
    g.fillStyle = grd; g.fillRect(0, 0, N, N);
    this.texGlow = this._tex(gc);

    // --- star / petal burst -----------------------------------------
    // Same rule as texDot: alpha carries the shape, never an opaque black base,
    // or the flash quads punch a black square into the viewmodel alpha.
    const sc = this._canvas(N), s = sc.getContext('2d');
    s.clearRect(0, 0, N, N);
    s.globalCompositeOperation = 'lighter';
    s.translate(N / 2, N / 2);
    const petals = 7;
    for (let i = 0; i < petals; i++) {
      const a = (i / petals) * Math.PI * 2 + fhash(i, 3, 7) * 0.5;
      const len = N * (0.24 + fhash(i, 11, 5) * 0.24);
      const wid = N * (0.030 + fhash(i, 17, 2) * 0.040);
      s.save(); s.rotate(a);
      const lg = s.createLinearGradient(0, 0, len, 0);
      lg.addColorStop(0.0, 'rgba(255,248,220,0.95)');
      lg.addColorStop(0.35, 'rgba(255,190,90,0.52)');
      lg.addColorStop(1.0, 'rgba(200,70,10,0)');
      s.fillStyle = lg;
      s.beginPath();
      s.moveTo(0, -wid); s.quadraticCurveTo(len * 0.55, -wid * 0.35, len, 0);
      s.quadraticCurveTo(len * 0.55, wid * 0.35, 0, wid);
      s.closePath(); s.fill();
      s.restore();
    }
    grd = s.createRadialGradient(0, 0, 0, 0, 0, N * 0.19);
    grd.addColorStop(0, 'rgba(255,255,244,1)');
    grd.addColorStop(0.45, 'rgba(255,214,130,0.62)');
    grd.addColorStop(1, 'rgba(255,120,20,0)');
    s.fillStyle = grd;
    s.beginPath(); s.arc(0, 0, N * 0.19, 0, Math.PI * 2); s.fill();
    this.texStar = this._tex(sc);
  }

  /**
   * Coated-lens texture. RGB is the anti-reflective bloom you see looking into
   * a red dot — teal in the centre, going violet then near-black at the rim.
   * ALPHA is the tube vignette: the glass is nearly clear on axis and almost
   * opaque at the edge, so the sight picture darkens toward the tube wall the
   * way a real ocular does instead of being a hole cut in the weapon.
   */
  _makeOpticTextures() {
    const N = 256, c = this._canvas(N), x = c.getContext('2d');
    x.clearRect(0, 0, N, N);
    // Concentric ANNULI, not discs. Filled circles stacked centre-last
    // composite their alpha on top of one another, so a nominally 15%-opaque
    // core came out fully opaque after forty layers and the sight rendered as
    // a solid blue plate you could not see through — worse than the hollow
    // hoop it replaced. Each band has to be a ring with a hole in it.
    const B = 44;
    for (let i = 0; i < B; i++) {
      const r0 = i / B, r1 = (i + 1) / B;
      const r = (i + 0.5) / B;                 // 0 centre .. 1 rim
      const rr = r * r;
      // coating hue: teal core -> indigo -> a DARK plum crush at the rim. The
      // rim has to go down in value as well as over in hue, or the vignette
      // stops being a tube shadow and becomes a purple ring painted on glass.
      const dk = 1 - 0.52 * rr * rr;
      const cr = (10 + 52 * rr * rr) * dk;
      const cg = (42 + 24 * rr - 30 * rr * rr) * dk;
      const cb = (66 + 58 * rr - 44 * rr * rr) * dk;
      // vignette: near-clear across the useful field, then a long smooth shut
      const a = 0.115 + 0.885 * Math.pow(clamp01((r - 0.30) / 0.70), 2.30);
      x.beginPath();
      x.arc(N / 2, N / 2, r1 * N * 0.5, 0, Math.PI * 2);
      if (i > 0) x.arc(N / 2, N / 2, r0 * N * 0.5, 0, Math.PI * 2, true);
      x.fillStyle = `rgba(${cr | 0},${cg | 0},${cb | 0},${a.toFixed(3)})`;
      x.fill();
    }
    // faint wiped smear across the glass so it is not a perfect optical surface
    x.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 5; i++) {
      const a = fhash(i, 5, 2) * Math.PI;
      const g = x.createLinearGradient(0, 0, N, N);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(0.5, `rgba(70,96,120,${(0.020 + fhash(i, 9, 4) * 0.030).toFixed(3)})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      x.save();
      x.translate(N / 2, N / 2); x.rotate(a); x.translate(-N / 2, -N / 2);
      x.fillStyle = g;
      x.fillRect(0, N * 0.30, N, N * (0.05 + fhash(i, 13, 6) * 0.10));
      x.restore();
    }
    this.texLens = this._tex(c);

    // --- reticle sprite: hot core, soft bloom skirt, faint outer flare ------
    // NOTE: no opaque black base fill. The viewmodel renders to its own RGBA
    // target and the composite does mix( world, vm.rgb, vm.a ) — so any texel
    // with alpha 1 REPLACES the world, even where its RGB is black. Priming
    // this canvas with opaque black made the halo and flare quads stamp two
    // hard-edged rectangles of dead colour across the sight picture. Alpha has
    // to carry the shape.
    const M = 128, rc = this._canvas(M), r = rc.getContext('2d');
    r.clearRect(0, 0, M, M);
    r.globalCompositeOperation = 'lighter';
    const grd = r.createRadialGradient(M / 2, M / 2, 0, M / 2, M / 2, M / 2);
    grd.addColorStop(0.000, 'rgba(255,244,236,1)');
    grd.addColorStop(0.085, 'rgba(255,120,64,0.96)');
    grd.addColorStop(0.190, 'rgba(255,34,10,0.52)');
    grd.addColorStop(0.420, 'rgba(214,16,4,0.15)');
    grd.addColorStop(1.000, 'rgba(150,8,0,0)');
    r.fillStyle = grd; r.fillRect(0, 0, M, M);
    this.texDot = this._tex(rc);
  }

  _buildMuzzleFlash() {
    const grp = new THREE.Group();
    grp.name = 'muzzleFlash';
    grp.position.copy(this.muzzleLocal);
    grp.visible = false;
    grp.renderOrder = 20;

    const mk = (tex, size, color, op) => {
      const m = new THREE.MeshBasicMaterial({
        map: tex, color, transparent: true, opacity: op,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true,
        side: THREE.DoubleSide, toneMapped: true,
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), m);
      mesh.renderOrder = 20;
      return mesh;
    };

    // two petal layers with independent roll
    this._flashPetal = [mk(this.texStar, 0.30, 0xffe6b0, 1), mk(this.texStar, 0.20, 0xffd08a, 0.85)];
    this._flashPetal[0].position.z = -0.012;
    this._flashPetal[1].position.z = -0.020;
    grp.add(this._flashPetal[0], this._flashPetal[1]);

    // hot core — a physical burst of gas at the crown
    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.026, 1),
      new THREE.MeshBasicMaterial({
        color: 0xfff4d8, transparent: true, opacity: 0.9,
        blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: true,
      }),
    );
    core.scale.set(1, 1, 2.1);
    core.position.z = -0.018;
    core.renderOrder = 21;
    grp.add(core);
    this._flashCore = core;

    // soft bloom-feeding halo, always facing the viewer
    const glow = mk(this.texGlow, 0.46, 0xffc478, 0.55);
    glow.position.z = -0.010;
    this._flashGlow = glow;
    grp.add(glow);

    this.gun.add(grp);
    this.muzzleFlash = grp;

    // The flash light lives OUTSIDE the (hidden) flash group and stays visible
    // with zero intensity, so NUM_POINT_LIGHTS never changes and no material
    // ever recompiles mid-firefight.
    const l = new THREE.PointLight(0xffcf8c, 0, 1.5, 2.0);
    l.name = 'vmMuzzleLight';
    l.position.set(0, 0.012, this.muzzleLocal.z + 0.055);
    l.castShadow = false;
    this.gun.add(l);
    this._flashLightVm = l;
  }

  // ==================================================================
  //  WORLD-SCENE FX — muzzle light, brass, dropped magazines
  // ==================================================================

  _buildWorldFx() {
    const scene = this.g.scene;

    // --- world muzzle light (same zero-intensity trick) --------------
    this.flashLight = new THREE.PointLight(0xffc98a, 0, 16, 2.0);
    this.flashLight.name = 'muzzleFlashLight';
    this.flashLight.castShadow = false;
    this.flashLight.position.set(0, -50, 0);
    scene?.add(this.flashLight);

    // --- brass: a lathed 5.56 case, instanced ------------------------
    const cp = [
      new THREE.Vector2(0.0000, -0.0230),
      new THREE.Vector2(0.0048, -0.0230),
      new THREE.Vector2(0.0049, -0.0222),
      new THREE.Vector2(0.0042, -0.0214),
      new THREE.Vector2(0.0046, -0.0196),
      new THREE.Vector2(0.0047, -0.0060),
      new THREE.Vector2(0.0044, -0.0022),
      new THREE.Vector2(0.0030, 0.0032),
      new THREE.Vector2(0.0029, 0.0090),
      new THREE.Vector2(0.0025, 0.0092),
      new THREE.Vector2(0.0000, 0.0092),
    ];
    const shellGeo = new THREE.LatheGeometry(cp, 10);
    shellGeo.rotateX(-Math.PI * 0.5);
    const sRig = new Rig();
    sRig.add('brass', shellGeo, { wear: 0.5, edgeHi: 0.6 });
    const sHold = new THREE.Group();
    sRig.build(sHold, () => this.matBrass, 'shell');
    const sMesh = sHold.children[0];

    this.SHELLS = 24;
    this.shells = new THREE.InstancedMesh(sMesh.geometry, this.matBrass, this.SHELLS);
    this.shells.name = 'brass';
    this.shells.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.shells.frustumCulled = false;
    this.shells.castShadow = false;
    this.shells.receiveShadow = false;
    scene?.add(this.shells);

    this._shellPool = [];
    for (let i = 0; i < this.SHELLS; i++) {
      this._shellPool.push({
        alive: false, life: 0,
        p: new THREE.Vector3(), v: new THREE.Vector3(),
        q: new THREE.Quaternion(), w: new THREE.Vector3(),
        s: new THREE.Vector3(1, 1, 1), rest: false, ground: 0,
      });
    }
    this._shellHead = 0;
    this._shellQ = new THREE.Quaternion();
    this._shellM = new THREE.Matrix4();
    this._shellZero = new THREE.Vector3(0, 0, 0);
    this._shellNil = new THREE.Vector3(0.0001, 0.0001, 0.0001);
    for (let i = 0; i < this.SHELLS; i++) {
      this._shellM.compose(this._shellZero, this._shellQ.identity(), this._shellNil);
      this.shells.setMatrixAt(i, this._shellM);
    }
    this.shells.instanceMatrix.needsUpdate = true;

    // --- dropped magazines -------------------------------------------
    const mRig = new Rig();
    this._magRig(mRig);
    const mHold = new THREE.Group();
    mRig.build(mHold, () => this.matDropMag, 'dropmag');
    const mGeos = mHold.children.map(c => c.geometry);
    const dropGeo = mGeos.length > 1 ? (mergeGeometries(mGeos, false) || mGeos[0]) : mGeos[0];
    dropGeo.computeBoundingSphere();

    this.DROPMAGS = 3;
    this._dropMags = [];
    for (let i = 0; i < this.DROPMAGS; i++) {
      const m = new THREE.Mesh(dropGeo, this.matDropMag);
      m.name = 'droppedMag' + i;
      m.visible = false;
      m.castShadow = true;
      m.receiveShadow = true;
      m.frustumCulled = true;
      scene?.add(m);
      this._dropMags.push({
        mesh: m, alive: false, life: 0,
        v: new THREE.Vector3(), w: new THREE.Vector3(), ground: 0, rest: false,
      });
    }
    this._dropHead = 0;
  }

  // ==================================================================
  //  VIEWMODEL LIGHT RIG
  //  Mirrors the world rig (warm key, cool skylight, warm ground bounce)
  //  but lives in camera space, so the gun relights as the player turns.
  // ==================================================================

  _buildLightRig() {
    const S = this.viewmodelScene;

    // Round-1 note: the whole rig came down about a third. It was pushing the
    // viewmodel brighter than the sunlit sand behind it, which is what flattened
    // every surface into one washed semi-gloss. A carbine in hand is in the
    // shooter's own shadow more often than not; letting it sit a stop under the
    // plate makes the specular breakup on the metal visible at all.
    const key = new THREE.DirectionalLight(0xffd9a0, 2.05);
    key.name = 'vmKey';
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.05;
    key.shadow.camera.far = 2.6;
    key.shadow.camera.left = -0.42;
    key.shadow.camera.right = 0.42;
    key.shadow.camera.top = 0.42;
    key.shadow.camera.bottom = -0.42;
    key.shadow.bias = -0.00055;
    key.shadow.normalBias = 0.0032;
    key.shadow.camera.updateProjectionMatrix();
    const keyT = new THREE.Object3D();
    keyT.position.set(0.09, -0.10, -0.30);
    S.add(keyT); S.add(key);
    key.target = keyT;
    this._vmKey = key; this._vmKeyT = keyT;

    // Cool skylight fill dialled back from 0.82. On a 96%-metal receiver the
    // diffuse term is essentially zero, so the flank's whole colour came from
    // whatever light was hitting it — and the flank that faces the camera in
    // the hip pose faces AWAY from the warm key, so the cool fill was setting
    // its hue on its own. That is where "flat blue-grey slab" came from.
    const fill = new THREE.DirectionalLight(0x7fa6d9, 0.62);
    fill.name = 'vmFill';
    fill.castShadow = false;
    const fillT = new THREE.Object3D();
    fillT.position.copy(keyT.position);
    S.add(fillT); S.add(fill);
    fill.target = fillT;
    this._vmFill = fill; this._vmFillT = fillT;

    // warm sand bounce lifted to answer the reduced cool fill — the weapon is
    // held a metre over a sunlit dirt street and that is where its shadow side
    // gets its colour from
    const bounce = new THREE.DirectionalLight(0xd8a066, 0.82);
    bounce.name = 'vmBounce';
    bounce.castShadow = false;
    const bT = new THREE.Object3D();
    bT.position.copy(keyT.position);
    S.add(bT); S.add(bounce);
    bounce.target = bT;
    this._vmBounce = bounce; this._vmBounceT = bT;

    // rim from behind-left keeps the silhouette off the background
    const rim = new THREE.DirectionalLight(0xbcd2f0, 0.42);
    rim.name = 'vmRim';
    rim.castShadow = false;
    rim.position.set(-0.55, 0.35, 0.65);
    const rT = new THREE.Object3D();
    rT.position.copy(keyT.position);
    S.add(rT); S.add(rim);
    rim.target = rT;
    this._vmRim = rim;

    const hemi = new THREE.HemisphereLight(0x9cc2ef, 0x6f5334, 0.62);
    hemi.name = 'vmHemi';
    S.add(hemi);
    this._vmHemi = hemi;

    // Near-field lift. The stock and buffer tube sit a hand's width from the
    // lens, in the shadow of the receiver, and were crushing to a black slab —
    // which is both a hard "no pure blacks" violation and the reason that
    // corner of the frame read as an untextured placeholder. A weak, fast
    // falloff point light at the eye lifts only what is closest to camera.
    const eye = new THREE.PointLight(0xf0dcc0, 0.115, 1.05, 2.0);
    eye.name = 'vmEyeFill';
    eye.castShadow = false;
    eye.position.set(0.02, 0.02, 0.10);
    S.add(eye);
    this._vmEye = eye;

    // Second near-field lift aimed low and outboard, at the stock / butt pad /
    // buffer tube cluster in the bottom-right sixth of the frame. That cluster
    // faces away from every other light in the rig and was rendering as one
    // black slab with black ribs — a pure-black violation and the reason the
    // corner read as a placeholder.
    const butt = new THREE.PointLight(0xe6cfae, 0.034, 0.55, 2.0);
    butt.name = 'vmButtFill';
    butt.castShadow = false;
    butt.position.set(0.242, -0.085, -0.200);
    S.add(butt);
    this._vmButt = butt;
  }

  // ==================================================================
  //  INPUT
  // ==================================================================

  _locked() {
    return typeof document !== 'undefined' && !!document.pointerLockElement;
  }

  _bindInput() {
    if (this._bound || typeof window === 'undefined') return;
    this._bound = true;

    this._onDown = (e) => {
      if (!this._locked()) return;
      if (e.button === 0) { this.triggerHeld = true; this._fireTimer = Math.min(this._fireTimer, 0); }
      else if (e.button === 2) { this.setAds(true); }
    };
    this._onUp = (e) => {
      if (e.button === 0) this.triggerHeld = false;
      else if (e.button === 2) this.setAds(false);
    };
    this._onKey = (e) => {
      if (!this._locked()) return;
      if (e.code === 'KeyR') { this.reload(); }
      else if (e.code === 'KeyI') { this.inspect(); }
    };
    this._onBlur = () => { this.triggerHeld = false; };
    this._onCtx = (e) => { e.preventDefault(); };

    window.addEventListener('mousedown', this._onDown);
    window.addEventListener('mouseup', this._onUp);
    window.addEventListener('keydown', this._onKey);
    window.addEventListener('blur', this._onBlur);
    window.addEventListener('contextmenu', this._onCtx);
  }

  // ==================================================================
  //  OVERLAY RENDER PASS
  // ==================================================================

  _hookRender() {
    const scene = this.g.scene;
    if (!scene || scene.onAfterRender === this._afterHook) return;
    if (this._hookAttempts >= 8) return;
    this._hookAttempts++;
    const prev = (typeof scene.onAfterRender === 'function') ? scene.onAfterRender : null;
    this._prevAfter = prev;
    const self = this;
    this._afterHook = function (renderer, sc, camera, rt) {
      if (self._prevAfter) {
        try { self._prevAfter.call(this, renderer, sc, camera, rt); }
        catch (e) { console.error('[Weapon] chained scene.onAfterRender threw', e); }
      }
      if (!self.autoRender || !self.enabled) return;
      if (camera !== self.g.camera) return;
      // Only ever self-render over the DEFAULT framebuffer. Other modules render the
      // main scene with the player camera into their own offscreen targets several
      // times a frame (Sky's god-ray occlusion buffer, Particles' soft-particle depth
      // prepass). Drawing the viewmodel into one of those is not just wasted work:
      // renderViewmodel() calls clearDepth() first, which wiped the occluder depth out
      // of Sky's occlusion buffer and let the sun disc pass its depth test through
      // solid walls — god rays blooming on the inside of a building.
      // three passes only (renderer, scene, camera) to onAfterRender, so the bound
      // target has to be read off the renderer.
      if (renderer.getRenderTarget() !== null) return;
      const f = self.g.frame || 0;
      if (f <= self._extUntilFrame) return;      // PostFX is driving the pass
      if (self._afterFrame === f) return;
      self._afterFrame = f;
      self._inHook = true;
      try { self.renderViewmodel(renderer); }
      catch (e) { console.error('[Weapon] viewmodel pass', e); }
      finally { self._inHook = false; }
    };
    this._afterHook.__blacksiteWeapon = true;
    scene.onAfterRender = this._afterHook;
  }

  /**
   * Draw the viewmodel over whatever is currently bound. Clears depth first so
   * the gun can never intersect world geometry. Safe to call from PostFX; doing
   * so switches the module out of self-driving mode within one frame.
   */
  renderViewmodel(renderer) {
    const r = renderer || this.g.renderer?.renderer;
    if (!r || !this.enabled || !this.viewmodelScene) return;
    if (!this._inHook) this._extUntilFrame = (this.g.frame || 0) + 3;
    const ac = r.autoClear;
    r.autoClear = false;
    r.clearDepth();
    r.render(this.viewmodelScene, this.viewmodelCamera);
    r.autoClear = ac;
  }

  // ==================================================================
  //  PUBLIC ACTIONS
  // ==================================================================

  /** Debug API used by the screenshot critic — must keep working. */
  setAds(v) {
    const on = !!v;
    if (on && (this.sprintT > 0.6 || this._animKind === 'inspect')) this._cancelAnim();
    if (on === this.ads) return;
    this.ads = on;
    this.g.bus?.emit('ads', { active: on });
  }

  reload() {
    if (this.reloading) return false;
    if (this.mag >= this.magSize) return false;
    if (this.reserve <= 0) return false;
    this._emptyReload = this.mag <= 0;
    this.reloading = true;
    this._animKind = 'reload';
    this._animTrack = this._emptyReload ? RELOAD_EMPTY : RELOAD_TAC;
    this._animDur = this._animTrack[this._animTrack.length - 1].t;
    this._animT = 0;
    this._phaseIdx = 0;
    this._phases = this._emptyReload
      ? [[0.00, 'start'], [0.46, 'magout'], [1.06, 'magin'], [2.02, 'chamber'], [2.62, 'end']]
      : [[0.00, 'start'], [0.42, 'magout'], [1.00, 'magin'], [1.95, 'end']];
    this.triggerHeld = false;
    return true;
  }

  inspect() {
    if (this.reloading || this._animKind === 'inspect') return false;
    this._animKind = 'inspect';
    this._animTrack = INSPECT;
    this._animDur = INSPECT[INSPECT.length - 1].t;
    this._animT = 0;
    this._phases = null;
    return true;
  }

  _cancelAnim() {
    if (this._animKind === 'inspect') { this._animKind = ''; this._animTrack = null; this._animT = 0; }
  }

  // ==================================================================
  //  FIRING
  // ==================================================================

  _canFire() {
    return this.enabled && !this.reloading && this.mag > 0 && this.sprintT < 0.55;
  }

  fire() {
    if (!this._canFire()) {
      if (!this.reloading && this.mag <= 0) this.reload();
      return false;
    }
    this._cancelAnim();

    const cam = this.g.camera;
    const i = this._shotIndex;
    const jitter = (a) => (Math.random() * 2 - 1) * a;

    // ---- ammo ------------------------------------------------------
    this.mag--;
    this.g.bus?.emit('ammo', { mag: this.mag, reserve: this.reserve });

    // ---- shot ray --------------------------------------------------
    const slot = this._shotRing[this._shotRingI];
    this._shotRingI = (this._shotRingI + 1) % this._shotRing.length;
    slot.origin.setFromMatrixPosition(cam.matrixWorld);
    cam.getWorldDirection(slot.dir);
    const sp = this.spread;
    if (sp > 1e-5) {
      const a = Math.random() * Math.PI * 2;
      const rr = Math.sqrt(Math.random()) * sp;
      this._tmpV.set(Math.cos(a) * rr, Math.sin(a) * rr, 0).applyQuaternion(cam.quaternion);
      slot.dir.add(this._tmpV).normalize();
    }
    this.g.bus?.emit('shot', slot);

    // ---- recoil impulses (learned pattern + jitter) -----------------
    const vm = V_PATTERN[i % V_PATTERN.length] * (this.ads ? 0.72 : 1.0);
    const hm = H_PATTERN[i % H_PATTERN.length] * (this.ads ? 0.66 : 1.0);
    this._rRise[1] += (0.92 * vm + jitter(0.10)) * (1 - this.adsT * 0.18);
    this._rYaw[1] += (0.44 * hm + jitter(0.085));
    this._rRoll[1] += (-0.55 * hm + jitter(0.22));
    this._rKick[1] += 0.560 * (0.86 + Math.random() * 0.28) * (1 - this.adsT * 0.24);
    this._rLift[1] += 0.150 * vm * (1 - this.adsT * 0.35);

    // ---- camera kick -----------------------------------------------
    const camPitch = 0.0128 * vm * (1 - this.adsT * 0.30);
    const camYaw = 0.0056 * hm * (1 - this.adsT * 0.30) + jitter(0.0013);
    const c = this.g.controller;
    if (c) {
      if (typeof c.addRecoil === 'function') c.addRecoil(camPitch, camYaw);
      else if (typeof c.addShake === 'function') c.addShake(camPitch, camYaw, 0.10);
    }

    // ---- fx ---------------------------------------------------------
    this._flashT = 0;
    this._flashRoll = Math.random() * Math.PI * 2;
    this._flashScale = 0.82 + Math.random() * 0.46;
    this._boltT = 0;
    this._dust = 1;
    this._triggerT = 1;
    this._sinceShot = 0;
    this._shotIndex++;
    this.spread = Math.min(this.spread + (this.ads ? 0.0016 : 0.0072), this.ads ? 0.0170 : 0.0620);
    this._spawnShell();

    if (this.mag <= 0) this.reload();
    return true;
  }

  // ==================================================================
  //  BRASS + DROPPED MAGS
  // ==================================================================

  _groundY() {
    const p = this.g.controller?.pos;
    const eh = this.g.config.player.eyeHeight;
    if (p) return p.y - eh + 0.006;
    return 0.006;
  }

  _spawnShell() {
    const s = this._shellPool[this._shellHead];
    this._shellHead = (this._shellHead + 1) % this.SHELLS;
    const cam = this.g.camera;

    this.viewmodelScene.updateMatrixWorld(true);
    this._tmpV.copy(this.portLocal);
    this.gun.localToWorld(this._tmpV);
    this._tmpV.applyMatrix4(cam.matrixWorld);
    s.p.copy(this._tmpV);

    // world basis from the camera
    const e = cam.matrixWorld.elements;
    const rx = e[0], ry = e[1], rz = e[2];      // right
    const ux = e[4], uy = e[5], uz = e[6];      // up
    const fx = -e[8], fy = -e[9], fz = -e[10];  // forward

    const kr = 2.55 + Math.random() * 0.85;
    const ku = 1.35 + Math.random() * 0.55;
    const kf = 0.30 + Math.random() * 0.45;
    s.v.set(rx * kr + ux * ku + fx * kf, ry * kr + uy * ku + fy * kf, rz * kr + uz * ku + fz * kf);
    const pv = this.g.controller?.vel;
    if (pv && Number.isFinite(pv.x)) s.v.addScaledVector(pv, 0.85);

    s.q.setFromAxisAngle(this._tmpV2.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize(), Math.random() * 6.28);
    s.w.set((Math.random() - 0.5) * 44, (Math.random() - 0.5) * 30, (Math.random() - 0.5) * 44);
    s.s.set(1, 1, 1);
    s.alive = true; s.rest = false; s.life = 0;
    s.ground = this._groundY();
  }

  _updateShells(dt) {
    if (!this.shells) return;
    let dirty = false;
    for (let i = 0; i < this.SHELLS; i++) {
      const s = this._shellPool[i];
      if (!s.alive) continue;
      s.life += dt;
      if (!s.rest) {
        s.v.y -= 9.81 * dt;
        s.v.multiplyScalar(1 - 1.35 * dt);
        s.p.addScaledVector(s.v, dt);
        const wl = s.w.length();
        if (wl > 1e-4) {
          this._shellQ.setFromAxisAngle(this._tmpV2.copy(s.w).multiplyScalar(1 / wl), wl * dt);
          s.q.premultiply(this._shellQ).normalize();
        }
        if (s.p.y <= s.ground) {
          s.p.y = s.ground;
          if (Math.abs(s.v.y) < 0.55) { s.rest = true; s.v.set(0, 0, 0); s.w.set(0, 0, 0); }
          else {
            s.v.y = -s.v.y * 0.34;
            s.v.x *= 0.52; s.v.z *= 0.52;
            s.w.multiplyScalar(0.42);
          }
        }
      }
      // fade the oldest brass out rather than popping it
      const fade = s.life > 7.0 ? clamp01((8.2 - s.life) / 1.2) : 1;
      if (s.life > 8.2) { s.alive = false; s.s.copy(this._shellNil); }
      else s.s.set(fade, fade, fade);
      this._shellM.compose(s.p, s.q, s.s);
      this.shells.setMatrixAt(i, this._shellM);
      dirty = true;
    }
    if (dirty) this.shells.instanceMatrix.needsUpdate = true;
  }

  _dropMagazine() {
    const d = this._dropMags?.[this._dropHead];
    if (!d) return;
    this._dropHead = (this._dropHead + 1) % this.DROPMAGS;
    const cam = this.g.camera;
    this.viewmodelScene.updateMatrixWorld(true);

    this._tmpV.set(0, -0.052, -0.130);
    this.gun.localToWorld(this._tmpV);
    this._tmpV.applyMatrix4(cam.matrixWorld);
    d.mesh.position.copy(this._tmpV);
    d.mesh.quaternion.setFromRotationMatrix(cam.matrixWorld);
    d.mesh.rotateX(-0.25);
    d.mesh.visible = true;

    const e = cam.matrixWorld.elements;
    d.v.set(-e[8] * 0.35 + e[0] * 0.12, -0.55, -e[10] * 0.35 + e[2] * 0.12);
    const pv = this.g.controller?.vel;
    if (pv && Number.isFinite(pv.x)) d.v.addScaledVector(pv, 0.9);
    d.w.set((Math.random() - 0.5) * 6.5, (Math.random() - 0.5) * 4, (Math.random() - 0.5) * 6.5);
    d.alive = true; d.rest = false; d.life = 0;
    d.ground = this._groundY() - 0.006;
  }

  _updateDropMags(dt) {
    if (!this._dropMags) return;
    for (const d of this._dropMags) {
      if (!d.alive) continue;
      d.life += dt;
      if (!d.rest) {
        d.v.y -= 12.5 * dt;
        d.v.multiplyScalar(1 - 1.1 * dt);
        d.mesh.position.addScaledVector(d.v, dt);
        const wl = d.w.length();
        if (wl > 1e-4) {
          this._shellQ.setFromAxisAngle(this._tmpV2.copy(d.w).multiplyScalar(1 / wl), wl * dt);
          d.mesh.quaternion.premultiply(this._shellQ).normalize();
        }
        if (d.mesh.position.y <= d.ground) {
          d.mesh.position.y = d.ground;
          if (Math.abs(d.v.y) < 0.7) { d.rest = true; d.v.set(0, 0, 0); d.w.set(0, 0, 0); }
          else { d.v.y = -d.v.y * 0.22; d.v.x *= 0.45; d.v.z *= 0.45; d.w.multiplyScalar(0.35); }
        }
      }
      if (d.life > 22) { d.alive = false; d.mesh.visible = false; }
    }
  }

  // ==================================================================
  //  FRAME
  // ==================================================================

  _spring(s, k, c, dt) {
    const h = dt * 0.5;
    for (let i = 0; i < 2; i++) {
      s[1] += (-k * s[0] - c * s[1]) * h;
      s[0] += s[1] * h;
    }
  }

  update(dt, t) {
    if (!this.gun || !this.enabled) return;
    if (!(dt > 0)) dt = 1 / 120;
    if (dt > 0.05) dt = 0.05;

    const cfg = this.g.config;
    const cam = this.g.camera;
    const ctrl = this.g.controller;
    const P = this._sPos, R = this._sRot;

    // ---------------- player motion ---------------------------------
    if (ctrl?.pos) {
      if (this._havePrev) {
        const dx = ctrl.pos.x - this._prevPos.x, dz = ctrl.pos.z - this._prevPos.z;
        const inst = Math.sqrt(dx * dx + dz * dz) / Math.max(1e-4, dt);
        this._speed = (inst < 40) ? approach(this._speed, inst, 12, dt) : this._speed;
      } else this._havePrev = true;
      this._prevPos.copy(ctrl.pos);
    }
    const walk = cfg.player.walkSpeed;
    const wantSprint = (ctrl?.sprinting ?? (this._speed > walk * 1.22))
      && this._speed > walk * 0.9 && !this.reloading && !this.ads && this._animKind !== 'inspect';
    this.sprintT = approach(this.sprintT, wantSprint ? 1 : 0, wantSprint ? 8.5 : 12, dt);
    const sprintE = easeInOutCubic(clamp01(this.sprintT));

    // ---------------- ADS blend + camera FOV ------------------------
    const adsGoal = (this.ads && !this.reloading && this.sprintT < 0.5 && this._animKind !== 'inspect') ? 1 : 0;
    const step = dt / Math.max(0.02, cfg.weapon.adsTime);
    this._adsRaw = clamp01(this._adsRaw + (adsGoal ? step : -step * 1.12));
    this.adsT = adsGoal ? easeOutBack(this._adsRaw, 0.55) : easeInOutCubic(this._adsRaw);
    const adsE = this.adsT;
    const adsC = clamp01(this._adsRaw);

    const fovGoal = lerp(cfg.weapon.fovHip, cfg.weapon.fovAds, easeInOutCubic(adsC));
    if (Math.abs(cam.fov - fovGoal) > 0.008) { cam.fov = fovGoal; cam.updateProjectionMatrix(); }
    // The viewmodel lens narrows hard on ADS. That magnifies the optic while
    // flattening the perspective divergence of the receiver in the near field —
    // the same cheat every AAA shooter uses to keep an aimed shot clean.
    const vmFov = cfg.weapon.viewmodelFov - 14.0 * easeInOutCubic(adsC);
    if (Math.abs(this.viewmodelCamera.fov - vmFov) > 0.008) {
      this.viewmodelCamera.fov = vmFov;
      this.viewmodelCamera.updateProjectionMatrix();
    }

    // ---------------- animation timeline ----------------------------
    const A = this._anim;
    A[0] = A[1] = A[2] = A[3] = A[4] = A[5] = 0;
    let handOff = null;
    if (this._animTrack) {
      this._animT += dt;
      sampleTrack(this._animTrack, this._animT, A);
      if (this._phases) {
        while (this._phaseIdx < this._phases.length && this._animT >= this._phases[this._phaseIdx][0]) {
          this._onPhase(this._phases[this._phaseIdx][1]);
          this._phaseIdx++;
        }
      }
      handOff = this._emptyReload ? HANDL_EMPTY : HANDL_TAC;
      if (this._animKind === 'inspect') handOff = null;
      if (this._animT >= this._animDur) {
        this._animTrack = null; this._animKind = ''; this._animT = 0; this._phases = null;
        this.reloading = false;
      }
    }

    // ---------------- recoil springs --------------------------------
    this._spring(this._rKick, 600, 34, dt);
    this._spring(this._rRise, 430, 26, dt);
    this._spring(this._rYaw, 380, 27, dt);
    this._spring(this._rRoll, 340, 25, dt);
    this._spring(this._rLift, 520, 31, dt);
    this._sinceShot += dt;
    if (this._sinceShot > 0.55) this._shotIndex = 0;

    // ---------------- sway / lag from look input --------------------
    let dyaw = 0, dpit = 0;
    if (ctrl && typeof ctrl.yaw === 'number') {
      dyaw = ctrl.yaw - this._prevYaw;
      while (dyaw > Math.PI) dyaw -= Math.PI * 2;
      while (dyaw < -Math.PI) dyaw += Math.PI * 2;
      dpit = ctrl.pitch - this._prevPitch;
      this._prevYaw = ctrl.yaw; this._prevPitch = ctrl.pitch;
      if (Math.abs(dyaw) > 0.6) dyaw = 0;      // teleport, not a flick
      if (Math.abs(dpit) > 0.6) dpit = 0;
    }
    const wy = dyaw / Math.max(1e-4, dt);
    const wp = dpit / Math.max(1e-4, dt);
    // ADS damping raised across sway, bob and breathing. A collimated dot sits
    // where the BORE points, not where the tube happens to be, so any residual
    // wobble shows up as the dot crawling around inside the housing. Locking
    // the aimed pose down harder both steadies the sight picture and makes
    // aiming feel like it has weight behind it instead of drifting.
    const swayDamp = 1 - adsC * 0.80;
    const tgtY = clamp(-wy * 0.0290, -0.130, 0.130) * swayDamp;
    const tgtX = clamp(wp * 0.0250, -0.100, 0.100) * swayDamp;
    this._swayVY += (tgtY - this._swayY) * 320 * dt;
    this._swayVX += (tgtX - this._swayX) * 320 * dt;
    const sd = Math.exp(-15 * dt);
    this._swayVY *= sd; this._swayVX *= sd;
    this._swayY += this._swayVY * dt;
    this._swayX += this._swayVX * dt;

    // ---------------- walk bob + idle breathing ---------------------
    const spN = clamp(this._speed / Math.max(0.1, walk), 0, 1.9);
    this._bobPhase += dt * (6.6 + spN * 3.4) * clamp01(spN * 1.4);
    const bobK = spN * (1 - adsC * 0.90);
    const bp = this._bobPhase;
    const bobPX = Math.sin(bp) * 0.0082 * bobK;
    const bobPY = (Math.sin(bp * 2 + 0.4) * 0.0058 - 0.0016) * bobK;
    const bobRZ = Math.sin(bp) * 0.0250 * bobK;
    const bobRX = Math.sin(bp * 2 + 0.9) * 0.0140 * bobK;
    const brX = Math.sin(t * 1.31) * 0.0021 * (1 - adsC * 0.80);
    const brY = Math.sin(t * 0.87 + 1.2) * 0.0016 * (1 - adsC * 0.80);
    const brR = Math.sin(t * 0.63 + 0.4) * 0.0075 * (1 - adsC * 0.84);

    // ---------------- compose the pose ------------------------------
    const ADS = this.POSE_ADS || this.poseHip;
    const HIP = this.poseHip;
    P.set(
      lerp(HIP.p[0], ADS.p[0], adsE),
      lerp(HIP.p[1], ADS.p[1], adsE),
      lerp(HIP.p[2], ADS.p[2], adsE),
    );
    R.set(
      lerp(HIP.r[0], ADS.r[0], adsE),
      lerp(HIP.r[1], ADS.r[1], adsE),
      lerp(HIP.r[2], ADS.r[2], adsE),
    );
    if (sprintE > 0.001) {
      P.x = lerp(P.x, POSE_SPRINT.p[0], sprintE);
      P.y = lerp(P.y, POSE_SPRINT.p[1], sprintE);
      P.z = lerp(P.z, POSE_SPRINT.p[2], sprintE);
      R.x = lerp(R.x, POSE_SPRINT.r[0], sprintE);
      R.y = lerp(R.y, POSE_SPRINT.r[1], sprintE);
      R.z = lerp(R.z, POSE_SPRINT.r[2], sprintE);
    }

    P.x += A[0] + bobPX + brY + this._swayY * 0.062;
    P.y += A[1] + bobPY + brX + this._rLift[0] - Math.abs(this._swayX) * 0.030;
    P.z += A[2] + this._rKick[0];
    R.x += A[3] + bobRX + this._rRise[0] + this._swayX;
    R.y += A[4] + this._swayY + brR * 0.5;
    R.z += A[5] + bobRZ + this._rRoll[0] + this._swayY * 0.55 + brR;

    this.root.position.copy(P);
    this.root.rotation.copy(R);

    // ---------------- moving parts ----------------------------------
    this._updateParts(dt);

    // ---------------- left-hand offset ------------------------------
    if (handOff) {
      sampleTrack(handOff, this._animT, this._handTmp);
      const h = this._handTmp;
      this.handL.position.set(this._handLBase.x + h[0], this._handLBase.y + h[1], this._handLBase.z + h[2]);
      this.handL.rotation.set(this._handLBaseRot.x + h[3], this._handLBaseRot.y + h[4], this._handLBaseRot.z + h[5]);
    } else if (this._handDirty) {
      this.handL.position.copy(this._handLBase);
      this.handL.rotation.copy(this._handLBaseRot);
    }
    this._handDirty = !!handOff;

    // ---------------- fire control ----------------------------------
    this._fireTimer -= dt;
    this.firing = false;
    if (this.triggerHeld && this._canFire()) {
      let guard = 0;
      while (this._fireTimer <= 0 && guard++ < 4) {
        if (!this._canFire()) break;
        this.fire();
        this._fireTimer += this.shotInterval;
        this.firing = true;
      }
      if (this._fireTimer < 0) this._fireTimer = 0;
    } else if (this._fireTimer < 0) this._fireTimer = 0;

    // ---------------- spread ----------------------------------------
    const spBase = this.ads ? 0.00220 : 0.01650;
    const spMove = clamp(this._speed / 6, 0, 1) * (this.ads ? 0.0042 : 0.0180) + sprintE * 0.030;
    const spTarget = spBase + spMove;
    this.spread = Math.max(spTarget, this.spread - dt * (this.ads ? 0.060 : 0.170));
    this.crosshairSpread = clamp01((this.spread - 0.0022) / 0.062);

    // ---------------- world-space derived state ---------------------
    this.viewmodelScene.updateMatrixWorld(true);
    this._tmpV.copy(this.muzzleLocal);
    this.gun.localToWorld(this._tmpV);
    this._tmpV.applyMatrix4(cam.matrixWorld);
    this.muzzleWorld.copy(this._tmpV);

    this._collimateReticle();
    this._updateFlash(dt);
    this._updateShells(dt);
    this._updateDropMags(dt);
    this._updateVmLights();

    if ((this.g.frame & 31) === 0) this._hookRender();
  }

  /* ---- reload phase events ---------------------------------------- */
  _onPhase(phase) {
    const bus = this.g.bus;
    if (phase === 'magout') {
      this._magVis = 0;
      this.magMesh.visible = false;
      this._dropMagazine();
      bus?.emit('reload', { phase: 'magout' });
      return;
    }
    if (phase === 'magin') {
      const take = Math.min(this.reserve, this.magSize - this.mag);
      this.mag += take;
      this.reserve -= take;
      this._magVis = 1;
      this.magMesh.visible = true;
      bus?.emit('ammo', { mag: this.mag, reserve: this.reserve });
      bus?.emit('reload', { phase: 'magin' });
      return;
    }
    if (phase === 'end') {
      this.reloading = false;
      this._shotIndex = 0;
      bus?.emit('reload', { phase: 'end' });
      return;
    }
    bus?.emit('reload', { phase });
  }

  /* ---- bolt / dust cover / charging handle / trigger / mag --------- */
  _updateParts(dt) {
    const kind = this._animKind;
    const aT = this._animT;

    // --- bolt: reciprocates on each shot, locks back on empty --------
    let boltBack = 0;
    this._boltT += dt;
    const CY = 0.058;
    if (this._boltT < CY) {
      const u = this._boltT / CY;
      boltBack = u < 0.38 ? easeOutCubic(u / 0.38) : 1 - easeOutCubic((u - 0.38) / 0.62);
    }
    const lockOpen = (this.mag <= 0 && !(kind === 'reload' && this._phaseIdx > 2)) ? 1 : 0;
    boltBack = Math.max(boltBack, lockOpen);
    // empty-reload bolt release: the carrier slams home on the charging pull
    if (kind === 'reload' && this._emptyReload) {
      const rel = clamp01((aT - 2.02) / 0.09);
      boltBack = Math.max(0, boltBack * (1 - rel));
    }
    this.bolt.position.z = boltBack * 0.0380;

    // --- dust cover ---------------------------------------------------
    this._dust = Math.max(boltBack, this._dust - dt * 1.35);
    this.dustCover.rotation.z = -1.42 * easeOutCubic(clamp01(this._dust));

    // --- charging handle ---------------------------------------------
    let ch = 0;
    if (kind === 'reload' && this._emptyReload) {
      if (aT > 1.58 && aT < 2.02) ch = easeOutCubic(clamp01((aT - 1.58) / 0.30));
      else if (aT >= 2.02 && aT < 2.14) ch = 1 - easeOutCubic(clamp01((aT - 2.02) / 0.10));
    } else if (kind === 'inspect') {
      if (aT > 1.20 && aT < 1.55) ch = Math.sin(clamp01((aT - 1.20) / 0.35) * Math.PI) * 0.55;
    }
    this._chZ = ch;
    this.chargingHandle.position.z = ch * 0.0720;
    if (ch > 0.02) this.bolt.position.z = Math.max(this.bolt.position.z, ch * 0.0380);

    // --- trigger -------------------------------------------------------
    this._triggerT = Math.max(this.triggerHeld && this._canFire() ? 1 : 0, this._triggerT - dt * 7);
    this.trigger.rotation.x = -0.300 * easeOutCubic(clamp01(this._triggerT));

    // --- magazine ------------------------------------------------------
    const M = this.magMesh;
    if (kind === 'reload' && this._phases) {
      const tIn = this._phases[2][0];
      const enter = tIn - 0.20;
      if (aT >= enter && aT < tIn + 0.22) {
        M.visible = true;
        const u = clamp01((aT - enter) / 0.22);
        const e = easeOutBack(u, 1.1);
        M.position.set(0.0140 * (1 - e), -0.0520 - 0.1000 * (1 - e), -0.1300 + 0.0300 * (1 - e));
        M.rotation.set(0.4200 * (1 - e), 0, 0.2400 * (1 - e));
        // seat tap
        const s = clamp01((aT - (tIn + 0.16)) / 0.14);
        if (s > 0) M.position.y -= Math.sin(s * Math.PI) * 0.0035;
      } else if (aT < enter && this._magVis === 0) {
        M.visible = false;
      } else if (aT >= tIn + 0.22) {
        M.visible = true;
        M.position.set(0, -0.0520, -0.1300);
        M.rotation.set(0, 0, 0);
      }
    } else if (this._magVis) {
      M.visible = true;
      M.position.set(0, -0.0520, -0.1300);
      M.rotation.set(0, 0, 0);
    }

    // --- reticle brightness: only really visible through the glass ----
    // A dot emitter does not dim when you take your eye off it, but a
    // viewmodel one has to: at hip the dot is off-axis and would otherwise
    // smear a red bloom across the receiver. The core stays hot, the skirt is
    // what fades, so the dot never loses its hard centre.
    const rv = clamp01(this._adsRaw);
    const rvE = 0.42 + 0.58 * rv;
    this._reticleDot.material.opacity = rvE;
    this._reticleHalo.material.opacity = 0.12 + 0.30 * rv;
    this._reticleFlare.material.opacity = 0.030 + 0.115 * rv;
    if (this._lensCoat) this._lensCoat.material.opacity = 0.19 - 0.09 * rv;
    this.opticGlass.visible = true;
  }

  /**
   * Collimate the dot.
   *
   * A red dot is not a light bulb sitting behind the glass — the emitter is at
   * the focus of the objective so the dot leaves the tube as parallel rays and
   * appears at infinity. That is the whole point of the device: it does not
   * shift against the target when your head moves. Drawing it as a quad at a
   * fixed depth inside the tube reproduces exactly the parallax a real sight is
   * built to eliminate, and with the viewmodel breathing and swaying under it
   * the dot crawled visibly off the tube axis — which is what the critic saw.
   *
   * Fix: every frame, park the dot on the ray that leaves the eye PARALLEL to
   * the optic's bore. A point on that ray projects to the same pixel as a point
   * at infinity down the bore, so the dot is pinned to the optical axis no
   * matter where the tube has swung to. Off axis the solution runs outside the
   * glass, so it is clamped to the lens aperture and the tube wall occludes it
   * the way a real one loses the dot when you come off the weapon.
   *
   * Zero allocation: three pooled vectors, and worldToLocal uses three's own
   * module-level scratch matrix.
   */
  _collimateReticle() {
    const g = this.opticGlass;
    if (!g || !this._reticleDot) return;
    const O = this._tmpV.setFromMatrixPosition(g.matrixWorld);
    const f = this._tmpV2.set(0, 0, -1).transformDirection(this.gun.matrixWorld);
    const fl = f.length();
    if (fl < 1e-6) return;
    f.multiplyScalar(1 / fl);
    const D = this._retV.copy(f).multiplyScalar(O.dot(f));
    g.worldToLocal(D);
    const APER = 0.0122;                       // usable glass radius
    const d = Math.hypot(D.x, D.y);
    if (d > APER) { const k = APER / d; D.x *= k; D.y *= k; }
    this._reticleDot.position.set(D.x, D.y, -0.0238);
    this._reticleHalo.position.set(D.x, D.y, -0.0240);
    this._reticleFlare.position.set(D.x, D.y, -0.0242);
  }

  /* ---- muzzle flash ------------------------------------------------ */
  _updateFlash(dt) {
    this._flashT += dt;
    const f = this._flashT / this._flashDur;
    const F = this.muzzleFlash;
    if (f >= 1 || f < 0) {
      if (F.visible) F.visible = false;
      if (this._flashLightVm.intensity !== 0) this._flashLightVm.intensity = 0;
      if (this.flashLight && this.flashLight.intensity !== 0) this.flashLight.intensity = 0;
      return;
    }
    F.visible = true;
    const env = f < 0.20 ? (f / 0.20) : (1 - (f - 0.20) / 0.80);
    const e2 = env * env;
    const sc = this._flashScale * (0.42 + 0.82 * Math.sqrt(Math.max(0, env)));

    const p0 = this._flashPetal[0], p1 = this._flashPetal[1];
    p0.rotation.z = this._flashRoll;
    p1.rotation.z = -this._flashRoll * 1.7 + 0.9;
    p0.scale.setScalar(sc);
    p1.scale.setScalar(sc * (1.24 + 0.5 * (1 - env)));
    p0.material.opacity = env;
    p1.material.opacity = 0.80 * e2;

    this._flashCore.scale.set(sc * 0.92, sc * 0.92, sc * 1.95);
    this._flashCore.material.opacity = 0.95 * Math.pow(env, 0.6);
    this._flashGlow.scale.setScalar(sc * (1.5 + 0.9 * (1 - env)));
    this._flashGlow.material.opacity = 0.55 * e2;

    this._flashLightVm.intensity = 30 * e2;
    if (this.flashLight) {
      this.flashLight.position.copy(this.muzzleWorld);
      this.flashLight.intensity = 120 * e2;
    }
  }

  /* ---- viewmodel lighting, driven from the world rig --------------- */
  _updateVmLights() {
    const cam = this.g.camera;
    const S = this.viewmodelScene;

    // sun direction, world -> camera space
    this._tmpQ.copy(cam.quaternion).invert();
    const wd = this.g.renderer?.sunDir || this.g.sky?.sunDirection;
    if (wd) this._sunView.copy(wd).applyQuaternion(this._tmpQ);
    else this._sunView.set(-0.45, 0.42, -0.79);
    if (this._sunView.lengthSq() < 1e-6) this._sunView.set(-0.45, 0.42, -0.79);
    this._sunView.normalize();

    const tp = this._vmKeyT.position;
    this._vmKey.position.set(tp.x + this._sunView.x * 1.05, tp.y + this._sunView.y * 1.05, tp.z + this._sunView.z * 1.05);
    // cool skylight from the opposite side, always from above
    this._vmFill.position.set(tp.x - this._sunView.x * 1.0, tp.y + 0.85 + Math.abs(this._sunView.y) * 0.3, tp.z - this._sunView.z * 1.0);
    // warm bounce off the ground, biased toward the sun
    this._vmBounce.position.set(tp.x + this._sunView.x * 0.35, tp.y - 1.05, tp.z + this._sunView.z * 0.35);

    // colours track the atmosphere so the gun always belongs to the shot
    const sky = this.g.sky;
    if (sky) {
      if (sky.sunColor) this._vmKey.color.copy(sky.sunColor);
      if (sky.skyColor) { this._vmFill.color.copy(sky.skyColor); this._vmHemi.color.copy(sky.skyColor); }
      if (sky.groundColor) { this._vmBounce.color.copy(sky.groundColor); this._vmHemi.groundColor.copy(sky.groundColor); }
    }
    // the sun is behind the player as often as not — never let the gun go flat
    const facing = clamp01(this._sunView.z * -0.5 + 0.5);
    this._vmKey.intensity = 1.32 + 1.10 * facing;
    this._vmRim.intensity = 0.24 + 0.44 * (1 - facing);

    // IBL: share the world probe, rotated back into world orientation so the
    // reflections on the receiver agree with what the player is looking at.
    const env = this.g.scene?.environment || null;
    if (S.environment !== env) S.environment = env;
    if ('environmentIntensity' in S) S.environmentIntensity = this.g.scene?.environmentIntensity ?? 1;
    S.environmentRotation.setFromQuaternion(cam.quaternion);
  }

  // ==================================================================
  //  HOUSEKEEPING
  // ==================================================================

  resize(w, h) {
    const a = Math.max(0.1, (w || innerWidth) / Math.max(1, (h || innerHeight)));
    this.viewmodelCamera.aspect = a;
    this.viewmodelCamera.updateProjectionMatrix();
  }

  dispose() {
    if (typeof window !== 'undefined' && this._bound) {
      window.removeEventListener('mousedown', this._onDown);
      window.removeEventListener('mouseup', this._onUp);
      window.removeEventListener('keydown', this._onKey);
      window.removeEventListener('blur', this._onBlur);
      window.removeEventListener('contextmenu', this._onCtx);
      this._bound = false;
    }
    const scene = this.g.scene;
    if (scene && scene.onAfterRender === this._afterHook) scene.onAfterRender = this._prevAfter || function () { };
    this.viewmodelScene?.traverse(o => { o.geometry?.dispose?.(); });
    this.shells?.geometry?.dispose?.();
    this.texGlow?.dispose?.();
    this.texStar?.dispose?.();
    this.texLens?.dispose?.();
    this.texDot?.dispose?.();
  }
}

export default Weapon;
