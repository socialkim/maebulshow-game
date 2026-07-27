// ============================================================================
// BLACKSITE — src/physics/Physics.js
// Owner: Physics agent.  Collision + rigidbody simulation.  No external library.
//
// WHAT THIS FILE IS
//   A static-world collision system built on a binned-SAH BVH over a merged
//   triangle soup extracted from game.registry.colliders, plus a small pooled
//   rigidbody solver for shells and debris.  Everything steps at a fixed 120 Hz
//   with render-time interpolation, and every query is allocation-free on its
//   hot path.
//
// ---------------------------------------------------------------------------
// PUBLIC API  (collaborators — this is the contract)
// ---------------------------------------------------------------------------
//   game.physics.ready              bool   BVH is built and queryable
//   game.physics.stats              { tris, nodes, buildMs, depth, leaves,
//                                     bodies, awake, rays, sweeps, stepMs }
//
//   -- QUERIES ---------------------------------------------------------------
//   raycast(origin, dir, maxDist, out?)          -> Hit
//   raycastAny(origin, dir, maxDist)             -> bool     (early-out, LOS)
//   sphereCast(origin, dir, radius, maxDist, out?) -> Hit    (exact swept sphere)
//   closestPoint(point, maxDist, out?)           -> Hit      (nearest surface)
//   groundHeight(x, z, fromY, maxDrop)           -> number   (NaN if no ground)
//   overlapSphere(center, radius, out?)          -> Hit[]    (one per object)
//
//   Hit = { hit:bool, point:Vector3, normal:Vector3, distance:number,
//           surface:string, object:Object3D|null, triangle:int, material }
//
//   `out` is optional on every query.  Pass a Hit you own to make the call
//   fully allocation-free; omit it and you get a freshly allocated Hit.
//   dir does NOT need to be normalised.
//
//   -- CHARACTER -------------------------------------------------------------
//   capsuleSweep(from, to, radius, height, opts?) -> SweepResult   (SHARED —
//       read it immediately, it is overwritten by the next call)
//
//     `from`/`to` are the capsule BASE (feet) position, `height` is the full
//     standing height (Config.player.height).  The internal segment runs from
//     base+radius to base+height-radius.
//
//     opts (all optional):
//       grounded   bool    were you grounded last frame (enables ground snap)
//       stepHeight number  max step-up, default 0.35 m, 0 disables
//       snapDown   number  max ground snap, default 0.35 when grounded
//       slopeLimit number  degrees, default 50
//       skipStep   bool    force-disable step-up for this call
//
//     SweepResult = {
//       position     Vector3  resolved base position
//       grounded     bool
//       normal       Vector3  ground normal when grounded, else wall normal
//       groundNormal Vector3
//       wallNormal   Vector3
//       touched      bool     any contact at all this move
//       hitWall      bool
//       hitCeiling   bool
//       steppedUp    number   metres gained by step-up (0 if none)
//       slid         number   how much of the requested motion was lost
//       groundSurface string
//       groundObject Object3D|null
//     }
//
//     Typical Controller use:
//       const r = game.physics.capsuleSweep(pos, desired, radius, height,
//                                           { grounded: this.grounded });
//       pos.copy(r.position);
//       if (r.grounded) vel.y = Math.max(vel.y, 0);
//       if (r.hitCeiling) vel.y = Math.min(vel.y, 0);
//       // deflect velocity along the surfaces we actually slid on:
//       game.physics.projectVelocity(vel);
//
//   projectVelocity(vel)   clips `vel` (in place) against the contact planes
//                          recorded by the LAST capsuleSweep call.
//
//   -- RIGIDBODIES -----------------------------------------------------------
//   spawnBody(opts) -> body | null     (null when the 200-body cap is full)
//     opts: { x,y,z | position, vx,vy,vz | velocity, radius, mass,
//             restitution, friction, drag, angularDrag, gravityScale,
//             spin (rad/s scalar) | angularVelocity, life, surface,
//             object3D, onImpact(body, hit), impactEvents:bool, userData }
//   releaseBody(body)
//   applyExplosion(center, radius, power)     also driven by bus 'explosion'
//   bodies / awakeCount
//
//   -- WORLD -----------------------------------------------------------------
//   rebuild()              re-extract + rebuild the BVH (call after you add
//                          geometry to registry.colliders mid-game)
//   addCollider(obj) / removeCollider(obj)
//   surfaceOfObject(obj) -> string
//   setDebugDraw(bool, depth)   BVH wireframe, off by default
//
// ---------------------------------------------------------------------------
// SURFACE TAGGING
//   Every triangle carries a surface string resolved once at build time from,
//   in order:  mesh.userData.surface -> material.userData.bsImpact ->
//   materials.surfaceFor(material) -> parent chain userData.surface ->
//   'concrete'.  Canonical values are the six Bus 'impact' buckets:
//   concrete | metal | sand | wood | glass | flesh.
//   Flag a mesh with userData.noCollide = true to keep it out of the BVH.
// ---------------------------------------------------------------------------

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  Tunables (physics-only; anything with a Config key is read from it)
 * ------------------------------------------------------------------ */

const FIXED_DT = 1 / 120;          // deterministic substep
const MAX_SUBSTEPS = 8;            // spiral-of-death guard
const BINS = 12;                   // SAH bins per axis
const LEAF_SIZE = 8;               // triangles per leaf
const SAH_MIN_COUNT = 64;          // below this a median split is cheaper than it is worse
const MAX_DEPTH = 48;
const MAX_TRIS = 500000;
const MAX_BODIES = 200;
const BUILD_BUDGET_MS = 220;       // past this the builder degrades to median splits
const SAH_ALL_AXES = 3072;         // nodes smaller than this try all three axes
const QUERY_BUFFER = 4096;         // box-query result buffer
const MAX_CONTACT_TRIS = 512;      // capsule broadphase cap (bounds the solver cost)
const CONTACT_PLANES = 8;          // planes remembered for projectVelocity
const EPS = 1e-9;

const CANONICAL_SURFACES = ['concrete', 'metal', 'sand', 'wood', 'glass', 'flesh'];

// Alias table so a Level author writing userData.surface = 'stone' still lands
// on a real impact bucket rather than silently becoming concrete-by-accident.
const SURFACE_ALIAS = {
  stone: 'concrete', cement: 'concrete', plaster: 'concrete', brick: 'concrete',
  asphalt: 'concrete', road: 'concrete', tile: 'concrete', rock: 'concrete',
  steel: 'metal', iron: 'metal', rustmetal: 'metal', gunmetal: 'metal',
  aluminum: 'metal', aluminium: 'metal', container: 'metal', barrel: 'metal',
  dirt: 'sand', ground: 'sand', gravel: 'sand', dust: 'sand', soil: 'sand',
  plank: 'wood', crate: 'wood', timber: 'wood', plywood: 'wood', foliage: 'wood',
  window: 'glass', mirror: 'glass',
  skin: 'flesh', body: 'flesh', head: 'flesh', meat: 'flesh',
  fabric: 'concrete', cloth: 'concrete', cloth_tan: 'concrete',
  rubber: 'concrete', polymer: 'concrete',
};

/* ------------------------------------------------------------------ *
 *  Module-scope scratch — build-time only, never touched by update()
 * ------------------------------------------------------------------ */

const _binCount = new Int32Array(BINS);
const _binBox = new Float32Array(BINS * 6);
const _sweepArea = new Float64Array(BINS);
const _sweepCount = new Int32Array(BINS);
const _sweepBox = new Float64Array(BINS * 6);   // left-prefix box per bin
const _bestL = new Float64Array(6);             // child triangle bounds of the winning split
const _bestR = new Float64Array(6);
const _cenL = new Float64Array(6);              // child CENTROID bounds, accumulated
const _cenR = new Float64Array(6);              // during the partition — see _buildBVH

/** Swap two positional primitive records (index + AABB row). */
function swapPrim(idx, box, i, j) {
  const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  const a = i * 6, b = j * 6;
  let v = box[a]; box[a] = box[b]; box[b] = v;
  v = box[a + 1]; box[a + 1] = box[b + 1]; box[b + 1] = v;
  v = box[a + 2]; box[a + 2] = box[b + 2]; box[b + 2] = v;
  v = box[a + 3]; box[a + 3] = box[b + 3]; box[b + 3] = v;
  v = box[a + 4]; box[a + 4] = box[b + 4]; box[b + 4] = v;
  v = box[a + 5]; box[a + 5] = box[b + 5]; box[b + 5] = v;
}

function resetCenAccum() {
  _cenL[0] = _cenL[1] = _cenL[2] = Infinity; _cenL[3] = _cenL[4] = _cenL[5] = -Infinity;
  _cenR[0] = _cenR[1] = _cenR[2] = Infinity; _cenR[3] = _cenR[4] = _cenR[5] = -Infinity;
}
function accumCen(a, x, y, z) {
  if (x < a[0]) a[0] = x; if (x > a[3]) a[3] = x;
  if (y < a[1]) a[1] = y; if (y > a[4]) a[4] = y;
  if (z < a[2]) a[2] = z; if (z > a[5]) a[5] = z;
}
const _pt3 = new Float64Array(3);
const _ss7 = new Float64Array(7);

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/* ------------------------------------------------------------------ *
 *  Geometry primitives (raw scalars, zero allocation)
 * ------------------------------------------------------------------ */

/** Closest point on triangle abc to point p. Ericson, Real-Time Collision Detection. */
function closestPtTri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, out) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { out[0] = ax; out[1] = ay; out[2] = az; return; }

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { out[0] = bx; out[1] = by; out[2] = bz; return; }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3 || EPS);
    out[0] = ax + abx * v; out[1] = ay + aby * v; out[2] = az + abz * v; return;
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { out[0] = cx; out[1] = cy; out[2] = cz; return; }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6 || EPS);
    out[0] = ax + acx * w; out[1] = ay + acy * w; out[2] = az + acz * w; return;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6) || EPS);
    out[0] = bx + (cx - bx) * w; out[1] = by + (cy - by) * w; out[2] = bz + (cz - bz) * w; return;
  }

  const denom = 1 / (va + vb + vc || EPS);
  const v = vb * denom, w = vc * denom;
  out[0] = ax + abx * v + acx * w;
  out[1] = ay + aby * v + acy * w;
  out[2] = az + abz * v + acz * w;
}

/** Closest points between segments p1q1 and p2q2 -> out[c1(3), c2(3), dist2]. */
function segSeg(p1x, p1y, p1z, q1x, q1y, q1z, p2x, p2y, p2z, q2x, q2y, q2z, out) {
  const d1x = q1x - p1x, d1y = q1y - p1y, d1z = q1z - p1z;
  const d2x = q2x - p2x, d2y = q2y - p2y, d2z = q2z - p2z;
  const rx = p1x - p2x, ry = p1y - p2y, rz = p1z - p2z;
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  let s, t;

  if (a <= EPS && e <= EPS) { s = 0; t = 0; }
  else if (a <= EPS) { s = 0; t = f / e; t = t < 0 ? 0 : t > 1 ? 1 : t; }
  else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= EPS) { t = 0; s = -c / a; s = s < 0 ? 0 : s > 1 ? 1 : s; }
    else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const den = a * e - b * b;
      s = den > EPS ? (b * f - c * e) / den : 0;
      s = s < 0 ? 0 : s > 1 ? 1 : s;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = -c / a; s = s < 0 ? 0 : s > 1 ? 1 : s; }
      else if (t > 1) { t = 1; s = (b - c) / a; s = s < 0 ? 0 : s > 1 ? 1 : s; }
    }
  }
  const c1x = p1x + d1x * s, c1y = p1y + d1y * s, c1z = p1z + d1z * s;
  const c2x = p2x + d2x * t, c2y = p2y + d2y * t, c2z = p2z + d2z * t;
  const dx = c1x - c2x, dy = c1y - c2y, dz = c1z - c2z;
  out[0] = c1x; out[1] = c1y; out[2] = c1z;
  out[3] = c2x; out[4] = c2y; out[5] = c2z;
  out[6] = dx * dx + dy * dy + dz * dz;
}

function surfaceKey(name) {
  if (typeof name !== 'string') return null;
  const k = name.toLowerCase().trim();
  if (!k) return null;
  if (CANONICAL_SURFACES.indexOf(k) >= 0) return k;
  if (SURFACE_ALIAS[k]) return SURFACE_ALIAS[k];
  const cut = k.split(/[^a-z]+/)[0];
  if (CANONICAL_SURFACES.indexOf(cut) >= 0) return cut;
  if (SURFACE_ALIAS[cut]) return SURFACE_ALIAS[cut];
  return null;
}

/* ================================================================== *
 *  Physics
 * ================================================================== */

export class Physics {

  constructor(game) {
    this.g = game;
    this.THREE = THREE;

    const P = (game && game.config && game.config.player) || {};
    this.gravity = P.gravity ?? -18.5;
    this.defaultRadius = P.radius ?? 0.32;
    this.defaultHeight = P.height ?? 1.75;

    // ---- world -----------------------------------------------------------
    this.ready = false;
    this.triCount = 0;
    this.tri = null;            // Float32Array, 9 floats per triangle
    this.triN = null;           // Float32Array, 3 floats per triangle (unit normal)
    this.triSurf = null;        // Uint8Array  -> index into this.surfaces
    this.triObj = null;         // Uint32Array -> index into this.objects
    this.triIdx = null;         // Uint32Array, BVH primitive permutation
    this.objects = [];
    this.surfaces = CANONICAL_SURFACES.slice();
    this.syntheticGround = false;

    // BVH, flat arrays
    this.nBounds = null;        // Float32Array, 6 per node
    this.nLeft = null;          // Int32Array, first child (right = left+1), -1 for leaf
    this.nStart = null;         // Int32Array
    this.nCount = null;         // Int32Array, 0 => interior node
    this.nodeTotal = 0;

    // build scratch (nulled after the build)
    this._triBox = null;
    this._triCen = null;

    this.bounds = new THREE.Box3(
      new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
    this.voidY = -80;

    // ---- traversal scratch (allocation-free) -----------------------------
    this._stack = new Int32Array(256);
    this._buildStack = new Int32Array(1024);
    this._qhits = new Int32Array(QUERY_BUFFER);
    this._qcount = 0;
    this._cc = new Float64Array(8);

    // ---- capsule solver state -------------------------------------------
    this._rx = 0; this._ry = 0; this._rz = 0;
    this._rTouch = false; this._rGrounded = false; this._rCeil = false; this._rWall = false;
    this._rGnx = 0; this._rGny = 1; this._rGnz = 0;
    this._rWnx = 0; this._rWny = 0; this._rWnz = 0;
    this._rGroundTri = -1;
    this._planes = new Float32Array(CONTACT_PLANES * 3);
    this._planeCount = 0;

    this.slopeLimitDeg = 50;
    this.stepHeight = 0.35;
    this.snapDown = 0.35;
    this.skinWidth = 0.002;

    this.sweepResult = {
      position: new THREE.Vector3(),
      normal: new THREE.Vector3(0, 1, 0),
      groundNormal: new THREE.Vector3(0, 1, 0),
      wallNormal: new THREE.Vector3(),
      grounded: false,
      touched: false,
      hitWall: false,
      hitCeiling: false,
      steppedUp: 0,
      slid: 0,
      groundSurface: 'concrete',
      groundObject: null,
    };

    // ---- shared query results -------------------------------------------
    this._hitA = this._makeHit();
    this._hitB = this._makeHit();
    this._hitC = this._makeHit();
    this._hitD = this._makeHit();
    this._overlapOut = [];
    this._overlapPool = [];
    this._seenIdx = new Map();

    // ---- rigidbodies -----------------------------------------------------
    this.bodies = [];
    this._free = [];
    this.awakeCount = 0;
    for (let i = 0; i < MAX_BODIES; i++) {
      const b = this._makeBody(i);
      this.bodies.push(b);
      this._free.push(b);
    }
    this._qa = new THREE.Quaternion();
    this._qb = new THREE.Quaternion();
    this._qc = new THREE.Quaternion();

    // impact event ring — capped so a pile of shells never floods the bus
    this._evtPool = [];
    for (let i = 0; i < 16; i++) {
      this._evtPool.push({
        point: new THREE.Vector3(), normal: new THREE.Vector3(),
        material: null, surface: 'concrete', power: 0, body: null,
      });
    }
    this._evtI = 0;
    this._evtBudget = 0;
    this._evtClock = 0;

    // ---- stepping --------------------------------------------------------
    this._acc = 0;
    this.simTime = 0;
    this.steps = 0;

    this.stats = {
      tris: 0, nodes: 0, leaves: 0, depth: 0, buildMs: 0, meshes: 0,
      bodies: 0, awake: 0, rays: 0, sweeps: 0, stepMs: 0, skipped: 0,
    };
    this._rays = 0; this._sweeps = 0;

    // ---- rebuild watch ---------------------------------------------------
    this._builtLen = -1;
    this._builtVersion = -1;
    this._watchFrame = 0;

    // ---- debug -----------------------------------------------------------
    this._debugObj = null;
    this.debugDraw = false;

    // ---- bus -------------------------------------------------------------
    this._onExplosion = (e) => {
      if (!e || !e.point) return;
      this.applyExplosion(e.point, e.radius ?? 4, e.power ?? 900);
    };
    this.g?.bus?.on?.('explosion', this._onExplosion);
  }

  // ======================================================================
  //  LIFECYCLE
  // ======================================================================

  async init() {
    this.rebuild();
  }

  dispose() {
    this.g?.bus?.off?.('explosion', this._onExplosion);
    this.setDebugDraw(false);
    this.tri = this.triN = this.triSurf = this.triObj = this.triIdx = null;
    this.nBounds = this.nLeft = this.nStart = this.nCount = null;
    this.objects.length = 0;
    this.ready = false;
  }

  resize() { /* nothing view-dependent */ }

  // ======================================================================
  //  WORLD BUILD
  // ======================================================================

  addCollider(obj) {
    const reg = this.g?.registry;
    if (!reg || !obj) return;
    if (!Array.isArray(reg.colliders)) reg.colliders = [];
    if (reg.colliders.indexOf(obj) < 0) reg.colliders.push(obj);
    this._dirty = true;
  }

  removeCollider(obj) {
    const list = this.g?.registry?.colliders;
    if (!list) return;
    const i = list.indexOf(obj);
    if (i >= 0) list.splice(i, 1);
    this._dirty = true;
  }

  /** Full re-extract + BVH rebuild. Cheap enough to call on a level change. */
  rebuild() {
    const t0 = now();
    this._extract();
    const t1 = now();
    this._buildBVH();
    this.stats.extractMs = +(t1 - t0).toFixed(1);
    this.stats.bvhMs = +(now() - t1).toFixed(1);
    this._builtLen = this.g?.registry?.colliders?.length ?? 0;
    this._builtVersion = this.g?.registry?.collidersVersion ?? 0;
    this._dirty = false;
    this.ready = true;
    this.stats.buildMs = +(now() - t0).toFixed(1);
    this.stats.tris = this.triCount;
    this.stats.nodes = this.nodeTotal;
    if (this._debugObj) { this.setDebugDraw(false); this.setDebugDraw(true, this._debugDepth); }
    return this.stats.buildMs;
  }

  _collidable(o) {
    if (!o || !o.isMesh) return false;
    if (o.userData && (o.userData.noCollide === true || o.userData.collide === false)) return false;
    const geo = o.geometry;
    if (!geo || !geo.attributes || !geo.attributes.position) return false;
    if (geo.attributes.position.count < 3) return false;
    return true;
  }

  /** Resolve the impact surface bucket for a mesh, once, at build time. */
  surfaceOfObject(o) {
    if (!o) return 'concrete';
    let k = surfaceKey(o.userData && o.userData.surface);
    if (k) return k;
    // Level tags its merged buckets with the Materials surface key
    k = surfaceKey(o.userData && o.userData.bsSurface);
    if (k) return k;
    const mats = this.g?.materials;
    const m = o.material;
    const one = Array.isArray(m) ? m[0] : m;
    if (one && one.userData) {
      k = surfaceKey(one.userData.bsImpact) || surfaceKey(one.userData.surface);
      if (k) return k;
      k = surfaceKey(one.userData.bsSurface);
      if (k) return k;
      if (one.userData.bsSurface && mats?.surfaceFor) {
        k = surfaceKey(mats.surfaceFor(one.userData.bsSurface));
        if (k) return k;
      }
    }
    if (one && mats?.surfaceFor) {
      k = surfaceKey(mats.surfaceFor(one));
      if (k) return k;
    }
    if (one && one.name) { k = surfaceKey(String(one.name).replace(/^bs:/, '')); if (k) return k; }
    // walk up the parent chain — Level authors often tag the group, not the mesh
    let p = o.parent, guard = 0;
    while (p && guard++ < 8) {
      k = surfaceKey(p.userData && p.userData.surface);
      if (k) return k;
      p = p.parent;
    }
    return 'concrete';
  }

  _surfaceIndex(name) {
    const k = surfaceKey(name) || 'concrete';
    let i = this.surfaces.indexOf(k);
    if (i < 0) {
      if (this.surfaces.length >= 255) return 0;
      i = this.surfaces.push(k) - 1;
    }
    return i;
  }

  /** Merge every registered collider mesh into one indexed triangle soup. */
  _extract() {
    const reg = this.g?.registry;
    const roots = (reg && Array.isArray(reg.colliders)) ? reg.colliders : [];

    const meshes = this._meshScratch || (this._meshScratch = []);
    meshes.length = 0;
    let skipped = 0;

    const visit = (o) => {
      if (!o) return;
      if (o.isBatchedMesh) { skipped++; return; }
      if (this._collidable(o)) meshes.push(o);
    };

    for (let i = 0; i < roots.length; i++) {
      const root = roots[i];
      if (!root || !root.isObject3D) continue;
      try { root.updateMatrixWorld(true); } catch (e) { /* detached node */ }
      root.traverse(visit);
    }

    // ---- pass 1: count -----------------------------------------------------
    let total = 0;
    for (let i = 0; i < meshes.length; i++) {
      const m = meshes[i];
      const geo = m.geometry;
      const n = geo.index ? (geo.index.count / 3) | 0 : (geo.attributes.position.count / 3) | 0;
      const inst = m.isInstancedMesh ? Math.max(0, m.count | 0) : 1;
      total += n * inst;
    }
    if (total > MAX_TRIS) total = MAX_TRIS;

    if (total <= 0) {
      this._makeSyntheticGround();
      this.stats.meshes = meshes.length;
      this.stats.skipped = skipped;
      return;
    }

    // ---- pass 2: fill ------------------------------------------------------
    this.syntheticGround = false;
    const tri = new Float32Array(total * 9);
    const triN = new Float32Array(total * 3);
    const triSurf = new Uint8Array(total);
    const triObj = new Uint32Array(total);
    this.objects.length = 0;
    this.surfaces.length = 0;
    for (let i = 0; i < CANONICAL_SURFACES.length; i++) this.surfaces.push(CANONICAL_SURFACES[i]);

    const mat4 = this._mat4 || (this._mat4 = new THREE.Matrix4());
    const imat = this._imat || (this._imat = new THREE.Matrix4());

    let t = 0;

    for (let mi = 0; mi < meshes.length && t < total; mi++) {
      const mesh = meshes[mi];
      const geo = mesh.geometry;
      const pos = geo.attributes.position;
      const arr = pos.array;
      const itemSize = pos.itemSize || 3;
      const index = geo.index ? geo.index.array : null;
      const triCountGeo = index ? (index.length / 3) | 0 : (pos.count / 3) | 0;
      if (triCountGeo <= 0) continue;

      const objIdx = this.objects.length;
      this.objects.push(mesh);
      const sIdx = this._surfaceIndex(this.surfaceOfObject(mesh));

      const instances = mesh.isInstancedMesh ? Math.max(0, mesh.count | 0) : 1;

      for (let inst = 0; inst < instances && t < total; inst++) {
        if (mesh.isInstancedMesh) {
          mesh.getMatrixAt(inst, imat);
          mat4.multiplyMatrices(mesh.matrixWorld, imat);
        } else {
          mat4.copy(mesh.matrixWorld);
        }
        const e = mat4.elements;
        const e0 = e[0], e1 = e[1], e2 = e[2];
        const e4 = e[4], e5 = e[5], e6 = e[6];
        const e8 = e[8], e9 = e[9], e10 = e[10];
        const e12 = e[12], e13 = e[13], e14 = e[14];

        for (let f = 0; f < triCountGeo && t < total; f++) {
          const i0 = index ? index[f * 3] : f * 3;
          const i1 = index ? index[f * 3 + 1] : f * 3 + 1;
          const i2 = index ? index[f * 3 + 2] : f * 3 + 2;

          const a0 = i0 * itemSize, a1 = i1 * itemSize, a2 = i2 * itemSize;
          const lx0 = arr[a0], ly0 = arr[a0 + 1], lz0 = arr[a0 + 2];
          const lx1 = arr[a1], ly1 = arr[a1 + 1], lz1 = arr[a1 + 2];
          const lx2 = arr[a2], ly2 = arr[a2 + 1], lz2 = arr[a2 + 2];

          const x0 = e0 * lx0 + e4 * ly0 + e8 * lz0 + e12;
          const y0 = e1 * lx0 + e5 * ly0 + e9 * lz0 + e13;
          const z0 = e2 * lx0 + e6 * ly0 + e10 * lz0 + e14;
          const x1 = e0 * lx1 + e4 * ly1 + e8 * lz1 + e12;
          const y1 = e1 * lx1 + e5 * ly1 + e9 * lz1 + e13;
          const z1 = e2 * lx1 + e6 * ly1 + e10 * lz1 + e14;
          const x2 = e0 * lx2 + e4 * ly2 + e8 * lz2 + e12;
          const y2 = e1 * lx2 + e5 * ly2 + e9 * lz2 + e13;
          const z2 = e2 * lx2 + e6 * ly2 + e10 * lz2 + e14;

          const ux = x1 - x0, uy = y1 - y0, uz = z1 - z0;
          const vx = x2 - x0, vy = y2 - y0, vz = z2 - z0;
          let nx = uy * vz - uz * vy;
          let ny = uz * vx - ux * vz;
          let nz = ux * vy - uy * vx;
          const l2 = nx * nx + ny * ny + nz * nz;
          if (l2 < 1e-14) continue;                  // degenerate sliver
          const inv = 1 / Math.sqrt(l2);
          nx *= inv; ny *= inv; nz *= inv;

          const o9 = t * 9;
          tri[o9] = x0; tri[o9 + 1] = y0; tri[o9 + 2] = z0;
          tri[o9 + 3] = x1; tri[o9 + 4] = y1; tri[o9 + 5] = z1;
          tri[o9 + 6] = x2; tri[o9 + 7] = y2; tri[o9 + 8] = z2;
          const o3 = t * 3;
          triN[o3] = nx; triN[o3 + 1] = ny; triN[o3 + 2] = nz;
          triSurf[t] = sIdx;
          triObj[t] = objIdx;
          t++;
        }
      }
    }

    if (t <= 0) { this._makeSyntheticGround(); return; }

    this.tri = t === total ? tri : tri.subarray(0, t * 9);
    this.triN = t === total ? triN : triN.subarray(0, t * 3);
    this.triSurf = t === total ? triSurf : triSurf.subarray(0, t);
    this.triObj = t === total ? triObj : triObj.subarray(0, t);
    this.triCount = t;
    // world bounds come from the BVH root, which the builder computes anyway
    this.stats.meshes = meshes.length;
    this.stats.skipped = skipped;
    meshes.length = 0;
  }

  /**
   * No colliders registered (Level not up yet, or a build failure): give the
   * world a 600 m ground plane at y = 0 so the player never falls into the void
   * and every downstream module still gets sane answers.
   */
  _makeSyntheticGround() {
    const S = 300;
    const tri = new Float32Array(18);
    tri.set([-S, 0, -S, -S, 0, S, S, 0, S, -S, 0, -S, S, 0, S, S, 0, -S]);
    this.tri = tri;
    this.triN = new Float32Array([0, 1, 0, 0, 1, 0]);
    this.surfaces.length = 0;
    for (let i = 0; i < CANONICAL_SURFACES.length; i++) this.surfaces.push(CANONICAL_SURFACES[i]);
    const si = this.surfaces.indexOf('sand');
    this.triSurf = new Uint8Array([si, si]);
    this.triObj = new Uint32Array([0, 0]);
    this.objects.length = 0;
    this.objects.push(null);
    this.triCount = 2;
    this.syntheticGround = true;
    this.bounds.min.set(-S, -1, -S);
    this.bounds.max.set(S, 1, S);
    this.voidY = -60;
  }

  // ---------------------------------------------------------------------
  //  BVH — binned SAH with a hard time budget and a median-split fallback
  // ---------------------------------------------------------------------

  _buildBVH() {
    const N = this.triCount;
    const tri = this.tri;

    const idx = this.triIdx = new Uint32Array(N);
    // POSITIONAL primitive AABBs: row i belongs to whichever triangle currently
    // sits at position i in triIdx, and the partition swaps both together. Every
    // build pass then walks a contiguous slice instead of chasing triIdx through
    // a 5 MB array — on a 190k-triangle level that is the difference between a
    // build dominated by cache misses and one dominated by arithmetic.
    // The split axis is binned on the AABB centre rather than the barycentre;
    // they differ by a fraction of a triangle and give equivalent trees, but the
    // centre comes free out of the row we are already reading.
    const box = this._triBox = new Float32Array(N * 6);

    let rcx0 = Infinity, rcy0 = Infinity, rcz0 = Infinity;
    let rcx1 = -Infinity, rcy1 = -Infinity, rcz1 = -Infinity;
    for (let i = 0; i < N; i++) {
      idx[i] = i;
      const o = i * 9;
      const x0 = tri[o], y0 = tri[o + 1], z0 = tri[o + 2];
      const x1 = tri[o + 3], y1 = tri[o + 4], z1 = tri[o + 5];
      const x2 = tri[o + 6], y2 = tri[o + 7], z2 = tri[o + 8];
      let mnx = x0, mxx = x0; if (x1 < mnx) mnx = x1; else if (x1 > mxx) mxx = x1;
      if (x2 < mnx) mnx = x2; else if (x2 > mxx) mxx = x2;
      let mny = y0, mxy = y0; if (y1 < mny) mny = y1; else if (y1 > mxy) mxy = y1;
      if (y2 < mny) mny = y2; else if (y2 > mxy) mxy = y2;
      let mnz = z0, mxz = z0; if (z1 < mnz) mnz = z1; else if (z1 > mxz) mxz = z1;
      if (z2 < mnz) mnz = z2; else if (z2 > mxz) mxz = z2;
      const b = i * 6;
      box[b] = mnx; box[b + 3] = mxx;
      box[b + 1] = mny; box[b + 4] = mxy;
      box[b + 2] = mnz; box[b + 5] = mxz;
      const cx = mnx + mxx, cy = mny + mxy, cz = mnz + mxz;   // 2 * centre
      if (cx < rcx0) rcx0 = cx; if (cx > rcx1) rcx1 = cx;
      if (cy < rcy0) rcy0 = cy; if (cy > rcy1) rcy1 = cy;
      if (cz < rcz0) rcz0 = cz; if (cz > rcz1) rcz1 = cz;
    }

    const cap = Math.max(64, Math.min(2 * N + 8, ((2 * N) / LEAF_SIZE | 0) + 256));
    this._allocNodes(cap);
    this.nodeTotal = 1;
    this.nStart[0] = 0;
    this.nCount[0] = N;
    this.nLeft[0] = -1;
    this._nodeBounds(0, 0, N);
    // Centroid bounds live alongside the node and are produced by the parent's
    // partition loop, which has to compute every centre anyway. Scanning for
    // them per node instead costs a whole extra random-access pass over the
    // primitives at every level of the tree — about a third of the build.
    const nc0 = this.nCen;
    nc0[0] = rcx0; nc0[1] = rcy0; nc0[2] = rcz0;
    nc0[3] = rcx1; nc0[4] = rcy1; nc0[5] = rcz1;

    const stack = this._buildStack;
    let sp = 0;
    stack[sp++] = 0; stack[sp++] = 0;

    const deadline = now() + BUILD_BUDGET_MS;
    let overBudget = false;
    let tick = 0;
    let maxDepth = 0, leaves = 0;
    const useAllAxes = N <= 60000;

    while (sp > 0) {
      const depth = stack[--sp];
      const ni = stack[--sp];
      const count = this.nCount[ni];
      const start = this.nStart[ni];
      if (depth > maxDepth) maxDepth = depth;

      if (count <= LEAF_SIZE || depth >= MAX_DEPTH) { leaves++; continue; }

      // performance.now() is not free; 120k nodes' worth of calls is a
      // measurable slice of the budget we are trying to protect.
      if (!overBudget && (tick++ & 255) === 0 && now() > deadline) overBudget = true;

      let split = -1;
      let sahBounds = false;
      resetCenAccum();
      if (!overBudget && count > SAH_MIN_COUNT) {
        split = this._sahSplit(ni, start, count, useAllAxes);
        sahBounds = split >= 0;
      }
      if (split < 0) { resetCenAccum(); split = this._medianSplit(ni, start, count); }
      if (split <= start || split >= start + count) { leaves++; continue; }

      this._ensureNodes(this.nodeTotal + 2);
      const l = this.nodeTotal;
      this.nodeTotal += 2;

      this.nStart[l] = start; this.nCount[l] = split - start; this.nLeft[l] = -1;
      this.nStart[l + 1] = split; this.nCount[l + 1] = start + count - split; this.nLeft[l + 1] = -1;
      const lo = l * 6, ro = (l + 1) * 6;
      if (sahBounds) {
        // The binning sweep already computed both child boxes exactly; copying
        // them here avoids two more random-access passes over the primitives.
        const nb = this.nBounds;
        for (let k = 0; k < 6; k++) { nb[lo + k] = _bestL[k]; nb[ro + k] = _bestR[k]; }
      } else {
        this._nodeBounds(l, start, split - start);
        this._nodeBounds(l + 1, split, start + count - split);
      }
      const ncn = this.nCen;
      for (let k = 0; k < 6; k++) { ncn[lo + k] = _cenL[k]; ncn[ro + k] = _cenR[k]; }

      this.nLeft[ni] = l;
      this.nCount[ni] = 0;

      if (sp + 4 > stack.length) {
        const bigger = new Int32Array(stack.length * 2);
        bigger.set(stack);
        this._buildStack = bigger;
        return this._buildBVH();     // restart with room (effectively never happens)
      }
      stack[sp++] = l; stack[sp++] = depth + 1;
      stack[sp++] = l + 1; stack[sp++] = depth + 1;
    }

    this._triBox = null;
    this.nCen = null;
    const nb = this.nBounds;
    this.bounds.min.set(nb[0], nb[1], nb[2]);
    this.bounds.max.set(nb[3], nb[4], nb[5]);
    this.voidY = nb[1] - 60;
    this.stats.depth = maxDepth;
    this.stats.leaves = leaves;
    this.stats.nodes = this.nodeTotal;
    this.stats.sahTruncated = overBudget;
  }

  _allocNodes(cap) {
    this.nBounds = new Float32Array(cap * 6);
    this.nLeft = new Int32Array(cap);
    this.nStart = new Int32Array(cap);
    this.nCount = new Int32Array(cap);
    this.nCen = new Float32Array(cap * 6);   // build-only, freed when the tree is done
    this._nodeCap = cap;
  }

  _ensureNodes(n) {
    if (n <= this._nodeCap) return;
    const cap = Math.max(n, (this._nodeCap * 1.7) | 0);
    const nb = new Float32Array(cap * 6); nb.set(this.nBounds);
    const nl = new Int32Array(cap); nl.set(this.nLeft);
    const ns = new Int32Array(cap); ns.set(this.nStart);
    const nc = new Int32Array(cap); nc.set(this.nCount);
    this.nBounds = nb; this.nLeft = nl; this.nStart = ns; this.nCount = nc;
    if (this.nCen) { const ce = new Float32Array(cap * 6); ce.set(this.nCen); this.nCen = ce; }
    this._nodeCap = cap;
  }

  _nodeBounds(ni, start, count) {
    const box = this._triBox, nb = this.nBounds;
    let minx = Infinity, miny = Infinity, minz = Infinity;
    let maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
    const end = start + count;
    for (let i = start; i < end; i++) {
      const b = i * 6;
      if (box[b] < minx) minx = box[b];
      if (box[b + 1] < miny) miny = box[b + 1];
      if (box[b + 2] < minz) minz = box[b + 2];
      if (box[b + 3] > maxx) maxx = box[b + 3];
      if (box[b + 4] > maxy) maxy = box[b + 4];
      if (box[b + 5] > maxz) maxz = box[b + 5];
    }
    const o = ni * 6;
    nb[o] = minx; nb[o + 1] = miny; nb[o + 2] = minz;
    nb[o + 3] = maxx; nb[o + 4] = maxy; nb[o + 5] = maxz;
  }

  _sahSplit(ni, start, count, useAllAxes) {
    const box = this._triBox, idx = this.triIdx;
    const end = start + count;

    // centroid bounds, in "2 * centre" units to match the partition key
    const nc = this.nCen, co = ni * 6;
    const cminx = nc[co], cminy = nc[co + 1], cminz = nc[co + 2];
    const cmaxx = nc[co + 3], cmaxy = nc[co + 4], cmaxz = nc[co + 5];
    const ex = cmaxx - cminx, ey = cmaxy - cminy, ez = cmaxz - cminz;
    if (ex < 1e-7 && ey < 1e-7 && ez < 1e-7) return -1;

    const nb = this.nBounds, no = ni * 6;
    const px = nb[no + 3] - nb[no], py = nb[no + 4] - nb[no + 1], pz = nb[no + 5] - nb[no + 2];
    const parentArea = Math.max(1e-8, 2 * (px * py + py * pz + pz * px));

    const longest = (ex > ey && ex > ez) ? 0 : (ey > ez ? 1 : 2);
    const all = useAllAxes && count <= SAH_ALL_AXES;

    let bestCost = count;            // cost of leaving this node a leaf
    let bestAxis = -1, bestBin = -1;

    for (let ax = 0; ax < 3; ax++) {
      if (!all && ax !== longest) continue;
      const cmin = ax === 0 ? cminx : ax === 1 ? cminy : cminz;
      const ext = ax === 0 ? ex : ax === 1 ? ey : ez;
      if (ext < 1e-7) continue;
      const k = (BINS * 0.999999) / ext;

      for (let b = 0; b < BINS; b++) {
        _binCount[b] = 0;
        const o = b * 6;
        _binBox[o] = _binBox[o + 1] = _binBox[o + 2] = Infinity;
        _binBox[o + 3] = _binBox[o + 4] = _binBox[o + 5] = -Infinity;
      }

      for (let i = start; i < end; i++) {
        const tb = i * 6;
        let b = ((box[tb + ax] + box[tb + 3 + ax] - cmin) * k) | 0;
        if (b < 0) b = 0; else if (b >= BINS) b = BINS - 1;
        _binCount[b]++;
        const ob = b * 6;
        if (box[tb] < _binBox[ob]) _binBox[ob] = box[tb];
        if (box[tb + 1] < _binBox[ob + 1]) _binBox[ob + 1] = box[tb + 1];
        if (box[tb + 2] < _binBox[ob + 2]) _binBox[ob + 2] = box[tb + 2];
        if (box[tb + 3] > _binBox[ob + 3]) _binBox[ob + 3] = box[tb + 3];
        if (box[tb + 4] > _binBox[ob + 4]) _binBox[ob + 4] = box[tb + 4];
        if (box[tb + 5] > _binBox[ob + 5]) _binBox[ob + 5] = box[tb + 5];
      }

      // left sweep
      let lminx = Infinity, lminy = Infinity, lminz = Infinity;
      let lmaxx = -Infinity, lmaxy = -Infinity, lmaxz = -Infinity;
      let lc = 0;
      for (let b = 0; b < BINS - 1; b++) {
        const o = b * 6;
        if (_binCount[b] > 0) {
          if (_binBox[o] < lminx) lminx = _binBox[o];
          if (_binBox[o + 1] < lminy) lminy = _binBox[o + 1];
          if (_binBox[o + 2] < lminz) lminz = _binBox[o + 2];
          if (_binBox[o + 3] > lmaxx) lmaxx = _binBox[o + 3];
          if (_binBox[o + 4] > lmaxy) lmaxy = _binBox[o + 4];
          if (_binBox[o + 5] > lmaxz) lmaxz = _binBox[o + 5];
          lc += _binCount[b];
        }
        if (lc === 0) { _sweepArea[b] = 0; _sweepCount[b] = 0; continue; }
        const dx = lmaxx - lminx, dy = lmaxy - lminy, dz = lmaxz - lminz;
        _sweepArea[b] = 2 * (dx * dy + dy * dz + dz * dx);
        _sweepCount[b] = lc;
        const sb = b * 6;
        _sweepBox[sb] = lminx; _sweepBox[sb + 1] = lminy; _sweepBox[sb + 2] = lminz;
        _sweepBox[sb + 3] = lmaxx; _sweepBox[sb + 4] = lmaxy; _sweepBox[sb + 5] = lmaxz;
      }

      // right sweep + cost
      let rminx = Infinity, rminy = Infinity, rminz = Infinity;
      let rmaxx = -Infinity, rmaxy = -Infinity, rmaxz = -Infinity;
      let rc = 0;
      for (let b = BINS - 1; b > 0; b--) {
        const o = b * 6;
        if (_binCount[b] > 0) {
          if (_binBox[o] < rminx) rminx = _binBox[o];
          if (_binBox[o + 1] < rminy) rminy = _binBox[o + 1];
          if (_binBox[o + 2] < rminz) rminz = _binBox[o + 2];
          if (_binBox[o + 3] > rmaxx) rmaxx = _binBox[o + 3];
          if (_binBox[o + 4] > rmaxy) rmaxy = _binBox[o + 4];
          if (_binBox[o + 5] > rmaxz) rmaxz = _binBox[o + 5];
          rc += _binCount[b];
        }
        const li = b - 1;
        const lcount = _sweepCount[li];
        if (lcount === 0 || rc === 0) continue;
        const dx = rmaxx - rminx, dy = rmaxy - rminy, dz = rmaxz - rminz;
        const rArea = 2 * (dx * dy + dy * dz + dz * dx);
        const cost = 0.5 + (_sweepArea[li] * lcount + rArea * rc) / parentArea;
        if (cost < bestCost) {
          bestCost = cost; bestAxis = ax; bestBin = li;
          const sb = li * 6;
          _bestL[0] = _sweepBox[sb]; _bestL[1] = _sweepBox[sb + 1]; _bestL[2] = _sweepBox[sb + 2];
          _bestL[3] = _sweepBox[sb + 3]; _bestL[4] = _sweepBox[sb + 4]; _bestL[5] = _sweepBox[sb + 5];
          _bestR[0] = rminx; _bestR[1] = rminy; _bestR[2] = rminz;
          _bestR[3] = rmaxx; _bestR[4] = rmaxy; _bestR[5] = rmaxz;
        }
      }
    }

    if (bestAxis < 0) return -1;                     // let the median splitter decide

    // partition in place around the chosen bin boundary
    const cmin = bestAxis === 0 ? cminx : bestAxis === 1 ? cminy : cminz;
    const ext = bestAxis === 0 ? ex : bestAxis === 1 ? ey : ez;
    const k = (BINS * 0.999999) / ext;
    // Partition, accumulating each child's centroid bounds as we go — every
    // centre is already in a register here, so the children get their bounds
    // for free instead of paying for a scan of their own.
    let i = start, j = end - 1;
    while (i <= j) {
      const tb = i * 6;
      const cx = box[tb] + box[tb + 3];
      const cy = box[tb + 1] + box[tb + 4];
      const cz = box[tb + 2] + box[tb + 5];
      const key = bestAxis === 0 ? cx : bestAxis === 1 ? cy : cz;
      let b = ((key - cmin) * k) | 0;
      if (b < 0) b = 0; else if (b >= BINS) b = BINS - 1;
      if (b <= bestBin) { accumCen(_cenL, cx, cy, cz); i++; }
      else { accumCen(_cenR, cx, cy, cz); swapPrim(idx, box, i, j); j--; }
    }
    return i;
  }

  /** Spatial-median partition, falling back to an object median via quickselect. */
  _medianSplit(ni, start, count) {
    const box = this._triBox, idx = this.triIdx;
    const nc = this.nCen, co = ni * 6;
    const ex = nc[co + 3] - nc[co], ey = nc[co + 4] - nc[co + 1], ez = nc[co + 5] - nc[co + 2];
    const ax = (ex > ey && ex > ez) ? 0 : (ey > ez ? 1 : 2);
    const mid = (nc[co + ax] + nc[co + 3 + ax]) * 0.5;   // already in 2*centre units

    let i = start, j = start + count - 1;
    while (i <= j) {
      const tb = i * 6;
      const cx = box[tb] + box[tb + 3];
      const cy = box[tb + 1] + box[tb + 4];
      const cz = box[tb + 2] + box[tb + 5];
      const key = ax === 0 ? cx : ax === 1 ? cy : cz;
      if (key < mid) { accumCen(_cenL, cx, cy, cz); i++; }
      else { accumCen(_cenR, cx, cy, cz); swapPrim(idx, box, i, j); j--; }
    }
    if (i > start && i < start + count) return i;

    // Everything landed on one side (a slab of coplanar geometry): fall back to
    // an object median so the tree cannot degenerate into a linked list.
    const half = start + (count >> 1);
    this._nth(start, count, half, ax);
    resetCenAccum();
    for (let m = start; m < start + count; m++) {
      const tb = m * 6;
      accumCen(m < half ? _cenL : _cenR,
        box[tb] + box[tb + 3], box[tb + 1] + box[tb + 4], box[tb + 2] + box[tb + 5]);
    }
    return half;
  }

  /** Hoare quickselect over the positional primitive array by AABB centre. */
  _nth(start, count, n, ax) {
    const box = this._triBox, idx = this.triIdx;
    const lo0 = ax, hi0 = 3 + ax;
    let lo = start, hi = start + count - 1;
    let guard = 0;
    while (lo < hi && guard++ < 96) {
      const m = (lo + hi) >> 1;
      const pivot = box[m * 6 + lo0] + box[m * 6 + hi0];
      let i = lo, j = hi;
      while (i <= j) {
        while (box[i * 6 + lo0] + box[i * 6 + hi0] < pivot) i++;
        while (box[j * 6 + lo0] + box[j * 6 + hi0] > pivot) j--;
        if (i <= j) { swapPrim(idx, box, i, j); i++; j--; }
      }
      if (n <= j) hi = j;
      else if (n >= i) lo = i;
      else break;
    }
  }

  // ======================================================================
  //  HIT RECORDS
  // ======================================================================

  _makeHit() {
    return {
      hit: false,
      point: new THREE.Vector3(),
      normal: new THREE.Vector3(0, 1, 0),
      distance: Infinity,
      surface: 'concrete',
      object: null,
      triangle: -1,
      material: null,
    };
  }

  _clearHit(h, maxDist) {
    h.hit = false;
    h.distance = maxDist;
    h.surface = 'concrete';
    h.object = null;
    h.triangle = -1;
    h.material = null;
    return h;
  }

  _fillHit(h, t, px, py, pz, nx, ny, nz, dist) {
    h.hit = true;
    h.point.set(px, py, pz);
    h.normal.set(nx, ny, nz);
    h.distance = dist;
    h.triangle = t;
    h.surface = this.surfaces[this.triSurf[t]] || 'concrete';
    const o = this.objects[this.triObj[t]] || null;
    h.object = o;
    const m = o ? o.material : null;
    h.material = Array.isArray(m) ? m[0] : m || null;
    return h;
  }

  // ======================================================================
  //  RAYCAST
  // ======================================================================

  /**
   * Closest hit along a ray. `dir` need not be normalised. Returns `out` when
   * supplied (fully allocation-free), otherwise a freshly allocated Hit.
   */
  raycast(origin, dir, maxDist, out) {
    const h = out || this._makeHit();
    const max = (typeof maxDist === 'number' && maxDist > 0) ? maxDist : 1000;
    this._clearHit(h, max);
    if (!this.ready || this.triCount === 0 || !origin || !dir) return h;
    this._rays++;

    let dx = dir.x, dy = dir.y, dz = dir.z;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-12) return h;
    const idl = 1 / dl;
    dx *= idl; dy *= idl; dz *= idl;

    const t = this._traceRay(origin.x, origin.y, origin.z, dx, dy, dz, max, false);
    if (t < 0) return h;

    const ti = this._hitTri;
    const dist = this._hitT;
    const o3 = ti * 3;
    let nx = this.triN[o3], ny = this.triN[o3 + 1], nz = this.triN[o3 + 2];
    if (nx * dx + ny * dy + nz * dz > 0) { nx = -nx; ny = -ny; nz = -nz; }
    this._fillHit(h, ti,
      origin.x + dx * dist, origin.y + dy * dist, origin.z + dz * dist,
      nx, ny, nz, dist);
    return h;
  }

  /** Fast boolean occlusion test — bails on the first hit. Ideal for AI LOS. */
  raycastAny(origin, dir, maxDist) {
    if (!this.ready || this.triCount === 0 || !origin || !dir) return false;
    this._rays++;
    let dx = dir.x, dy = dir.y, dz = dir.z;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-12) return false;
    const idl = 1 / dl;
    return this._traceRay(origin.x, origin.y, origin.z, dx * idl, dy * idl, dz * idl,
      (typeof maxDist === 'number' && maxDist > 0) ? maxDist : 1000, true) >= 0;
  }

  /** True when nothing blocks the straight line between a and b. */
  lineOfSight(a, b) {
    if (!a || !b) return true;
    const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (d < 1e-5) return true;
    const inv = 1 / d;
    return this._traceRay(a.x, a.y, a.z, dx * inv, dy * inv, dz * inv, d - 0.02, true) < 0;
  }

  /**
   * Core BVH ray traversal. Front-to-back ordered, early-exiting.
   * Writes this._hitTri / this._hitT; returns the hit triangle or -1.
   */
  _traceRay(ox, oy, oz, dx, dy, dz, maxDist, anyHit) {
    const nb = this.nBounds, nc = this.nCount, nl = this.nLeft, nsArr = this.nStart;
    const idx = this.triIdx, tri = this.tri;
    const st = this._stack;

    const ix = dx !== 0 ? 1 / dx : 1e30;
    const iy = dy !== 0 ? 1 / dy : 1e30;
    const iz = dz !== 0 ? 1 / dz : 1e30;

    let best = maxDist;
    let bestTri = -1;
    let sp = 0;
    st[sp++] = 0;

    while (sp > 0) {
      const ni = st[--sp];
      const b = ni * 6;

      // slab test
      let t0 = (nb[b] - ox) * ix, t1 = (nb[b + 3] - ox) * ix;
      let tmin = t0 < t1 ? t0 : t1, tmax = t0 < t1 ? t1 : t0;
      t0 = (nb[b + 1] - oy) * iy; t1 = (nb[b + 4] - oy) * iy;
      const ymin = t0 < t1 ? t0 : t1, ymax = t0 < t1 ? t1 : t0;
      if (ymin > tmin) tmin = ymin;
      if (ymax < tmax) tmax = ymax;
      t0 = (nb[b + 2] - oz) * iz; t1 = (nb[b + 5] - oz) * iz;
      const zmin = t0 < t1 ? t0 : t1, zmax = t0 < t1 ? t1 : t0;
      if (zmin > tmin) tmin = zmin;
      if (zmax < tmax) tmax = zmax;
      if (tmax < 0 || tmin > tmax || tmin > best) continue;

      const count = nc[ni];
      if (count > 0) {
        const s = nsArr[ni];
        for (let i = 0; i < count; i++) {
          const t = idx[s + i];
          const o = t * 9;
          const ax = tri[o], ay = tri[o + 1], az = tri[o + 2];
          const e1x = tri[o + 3] - ax, e1y = tri[o + 4] - ay, e1z = tri[o + 5] - az;
          const e2x = tri[o + 6] - ax, e2y = tri[o + 7] - ay, e2z = tri[o + 8] - az;
          const px = dy * e2z - dz * e2y;
          const py = dz * e2x - dx * e2z;
          const pz = dx * e2y - dy * e2x;
          const det = e1x * px + e1y * py + e1z * pz;
          if (det > -1e-10 && det < 1e-10) continue;      // parallel
          const invDet = 1 / det;
          const tx = ox - ax, ty = oy - ay, tz = oz - az;
          const u = (tx * px + ty * py + tz * pz) * invDet;
          if (u < -1e-6 || u > 1.000001) continue;
          const qx = ty * e1z - tz * e1y;
          const qy = tz * e1x - tx * e1z;
          const qz = tx * e1y - ty * e1x;
          const v = (dx * qx + dy * qy + dz * qz) * invDet;
          if (v < -1e-6 || u + v > 1.000001) continue;
          const tt = (e2x * qx + e2y * qy + e2z * qz) * invDet;
          if (tt <= 1e-5 || tt >= best) continue;
          best = tt; bestTri = t;
          if (anyHit) { this._hitTri = t; this._hitT = tt; return t; }
        }
      } else {
        const l = nl[ni];
        if (l < 0) continue;
        // order the children so the nearer box is popped first
        const lb = l * 6, rb = (l + 1) * 6;
        const ltn = this._slabEnter(nb, lb, ox, oy, oz, ix, iy, iz);
        const rtn = this._slabEnter(nb, rb, ox, oy, oz, ix, iy, iz);
        if (sp + 2 >= st.length) continue;
        if (ltn <= rtn) { st[sp++] = l + 1; st[sp++] = l; }
        else { st[sp++] = l; st[sp++] = l + 1; }
      }
    }

    this._hitTri = bestTri;
    this._hitT = best;
    return bestTri;
  }

  _slabEnter(nb, b, ox, oy, oz, ix, iy, iz) {
    let t0 = (nb[b] - ox) * ix, t1 = (nb[b + 3] - ox) * ix;
    let tmin = t0 < t1 ? t0 : t1, tmax = t0 < t1 ? t1 : t0;
    t0 = (nb[b + 1] - oy) * iy; t1 = (nb[b + 4] - oy) * iy;
    let a = t0 < t1 ? t0 : t1, c = t0 < t1 ? t1 : t0;
    if (a > tmin) tmin = a; if (c < tmax) tmax = c;
    t0 = (nb[b + 2] - oz) * iz; t1 = (nb[b + 5] - oz) * iz;
    a = t0 < t1 ? t0 : t1; c = t0 < t1 ? t1 : t0;
    if (a > tmin) tmin = a; if (c < tmax) tmax = c;
    return (tmax < 0 || tmin > tmax) ? 1e30 : (tmin < 0 ? 0 : tmin);
  }

  // ======================================================================
  //  SPHERE CAST — exact swept sphere vs triangle (face, 3 edges, 3 vertices)
  // ======================================================================

  sphereCast(origin, dir, radius, maxDist, out) {
    const h = out || this._makeHit();
    const max = (typeof maxDist === 'number' && maxDist > 0) ? maxDist : 100;
    this._clearHit(h, max);
    if (!this.ready || this.triCount === 0 || !origin || !dir) return h;

    let dx = dir.x, dy = dir.y, dz = dir.z;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-12) return h;
    const inv = 1 / dl;
    dx *= inv; dy *= inv; dz *= inv;

    const r = Math.max(1e-4, radius);
    const t = this._traceSphere(origin.x, origin.y, origin.z, dx, dy, dz, r, max);
    if (t < 0) return h;

    const dist = this._hitT;
    const cx = origin.x + dx * dist, cy = origin.y + dy * dist, cz = origin.z + dz * dist;
    // contact point = closest point on the triangle to the sphere centre at impact
    const o = t * 9;
    closestPtTri(cx, cy, cz,
      this.tri[o], this.tri[o + 1], this.tri[o + 2],
      this.tri[o + 3], this.tri[o + 4], this.tri[o + 5],
      this.tri[o + 6], this.tri[o + 7], this.tri[o + 8], _pt3);
    let nx = cx - _pt3[0], ny = cy - _pt3[1], nz = cz - _pt3[2];
    const nl2 = nx * nx + ny * ny + nz * nz;
    if (nl2 > 1e-12) { const k = 1 / Math.sqrt(nl2); nx *= k; ny *= k; nz *= k; }
    else {
      const o3 = t * 3;
      nx = this.triN[o3]; ny = this.triN[o3 + 1]; nz = this.triN[o3 + 2];
      if (nx * dx + ny * dy + nz * dz > 0) { nx = -nx; ny = -ny; nz = -nz; }
    }
    this._fillHit(h, t, _pt3[0], _pt3[1], _pt3[2], nx, ny, nz, dist);
    return h;
  }

  _traceSphere(ox, oy, oz, dx, dy, dz, r, maxDist) {
    const nb = this.nBounds, nc = this.nCount, nl = this.nLeft, nsArr = this.nStart;
    const idx = this.triIdx;
    const st = this._stack;

    // conservative broadphase: the ray with boxes inflated by r
    const ix = dx !== 0 ? 1 / dx : 1e30;
    const iy = dy !== 0 ? 1 / dy : 1e30;
    const iz = dz !== 0 ? 1 / dz : 1e30;

    let best = maxDist;
    let bestTri = -1;
    let sp = 0;
    st[sp++] = 0;

    while (sp > 0) {
      const ni = st[--sp];
      const b = ni * 6;
      let t0 = (nb[b] - r - ox) * ix, t1 = (nb[b + 3] + r - ox) * ix;
      let tmin = t0 < t1 ? t0 : t1, tmax = t0 < t1 ? t1 : t0;
      t0 = (nb[b + 1] - r - oy) * iy; t1 = (nb[b + 4] + r - oy) * iy;
      let a = t0 < t1 ? t0 : t1, c = t0 < t1 ? t1 : t0;
      if (a > tmin) tmin = a; if (c < tmax) tmax = c;
      t0 = (nb[b + 2] - r - oz) * iz; t1 = (nb[b + 5] + r - oz) * iz;
      a = t0 < t1 ? t0 : t1; c = t0 < t1 ? t1 : t0;
      if (a > tmin) tmin = a; if (c < tmax) tmax = c;
      if (tmax < 0 || tmin > tmax || tmin > best) continue;

      const count = nc[ni];
      if (count > 0) {
        const s = nsArr[ni];
        for (let i = 0; i < count; i++) {
          const t = idx[s + i];
          const tt = this._sweepSphereTri(t, ox, oy, oz, dx, dy, dz, r, best);
          if (tt >= 0 && tt < best) { best = tt; bestTri = t; }
        }
      } else {
        const l = nl[ni];
        if (l < 0 || sp + 2 >= st.length) continue;
        st[sp++] = l; st[sp++] = l + 1;
      }
    }
    this._hitTri = bestTri;
    this._hitT = best;
    return bestTri;
  }

  /** Sweep a sphere of radius r from o along unit d; earliest touch time or -1. */
  _sweepSphereTri(t, ox, oy, oz, dx, dy, dz, r, tmax) {
    const tri = this.tri, o = t * 9, o3 = t * 3;
    const ax = tri[o], ay = tri[o + 1], az = tri[o + 2];
    const bx = tri[o + 3], by = tri[o + 4], bz = tri[o + 5];
    const cx = tri[o + 6], cy = tri[o + 7], cz = tri[o + 8];
    let nx = this.triN[o3], ny = this.triN[o3 + 1], nz = this.triN[o3 + 2];

    // face the normal toward the sphere so a double-sided wall behaves
    let sd = (ox - ax) * nx + (oy - ay) * ny + (oz - az) * nz;
    if (sd < 0) { nx = -nx; ny = -ny; nz = -nz; sd = -sd; }

    const nd = nx * dx + ny * dy + nz * dz;
    let best = -1;

    // --- 1. plane / face region -------------------------------------------
    if (sd <= r) {
      // already touching the plane; check the interior directly
      const px = ox - nx * sd, py = oy - ny * sd, pz = oz - nz * sd;
      if (this._pointInTri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz)) return 0;
    } else if (nd < -1e-9) {
      const tt = (sd - r) / -nd;
      if (tt >= 0 && tt <= tmax) {
        const ccx = ox + dx * tt, ccy = oy + dy * tt, ccz = oz + dz * tt;
        const px = ccx - nx * r, py = ccy - ny * r, pz = ccz - nz * r;
        if (this._pointInTri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz)) return tt;
      }
    }

    const r2 = r * r;

    // --- 2. vertices (ray vs sphere) --------------------------------------
    best = this._raySphere(ox, oy, oz, dx, dy, dz, ax, ay, az, r2, tmax, best);
    best = this._raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, r2, tmax, best);
    best = this._raySphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, r2, tmax, best);

    // --- 3. edges (ray vs finite cylinder) --------------------------------
    best = this._rayCylinder(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, r2, tmax, best);
    best = this._rayCylinder(ox, oy, oz, dx, dy, dz, bx, by, bz, cx, cy, cz, r2, tmax, best);
    best = this._rayCylinder(ox, oy, oz, dx, dy, dz, cx, cy, cz, ax, ay, az, r2, tmax, best);

    return best;
  }

  _pointInTri(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz) {
    let ex = bx - ax, ey = by - ay, ez = bz - az;
    let vx = px - ax, vy = py - ay, vz = pz - az;
    if ((ey * vz - ez * vy) * nx + (ez * vx - ex * vz) * ny + (ex * vy - ey * vx) * nz < -1e-7) return false;
    ex = cx - bx; ey = cy - by; ez = cz - bz;
    vx = px - bx; vy = py - by; vz = pz - bz;
    if ((ey * vz - ez * vy) * nx + (ez * vx - ex * vz) * ny + (ex * vy - ey * vx) * nz < -1e-7) return false;
    ex = ax - cx; ey = ay - cy; ez = az - cz;
    vx = px - cx; vy = py - cy; vz = pz - cz;
    if ((ey * vz - ez * vy) * nx + (ez * vx - ex * vz) * ny + (ex * vy - ey * vx) * nz < -1e-7) return false;
    return true;
  }

  _raySphere(ox, oy, oz, dx, dy, dz, sx, sy, sz, r2, tmax, best) {
    const mx = ox - sx, my = oy - sy, mz = oz - sz;
    const b = mx * dx + my * dy + mz * dz;
    const c = mx * mx + my * my + mz * mz - r2;
    if (c > 0 && b > 0) return best;
    const disc = b * b - c;
    if (disc < 0) return best;
    let t = -b - Math.sqrt(disc);
    if (t < 0) t = 0;
    if (t > tmax) return best;
    return (best < 0 || t < best) ? t : best;
  }

  _rayCylinder(ox, oy, oz, dx, dy, dz, px, py, pz, qx, qy, qz, r2, tmax, best) {
    const ax = qx - px, ay = qy - py, az = qz - pz;
    const mx = ox - px, my = oy - py, mz = oz - pz;
    const dd = ax * ax + ay * ay + az * az;
    if (dd < 1e-12) return best;
    const md = mx * ax + my * ay + mz * az;
    const nd = dx * ax + dy * ay + dz * az;
    const nn = 1;                              // d is unit length
    const mn = mx * dx + my * dy + mz * dz;
    const A = dd * nn - nd * nd;
    const k = mx * mx + my * my + mz * mz - r2;
    const C = dd * k - md * md;
    if (A < 1e-12) return best;                // parallel to the edge; vertices cover it
    const B = dd * mn - nd * md;
    const disc = B * B - A * C;
    if (disc < 0) return best;
    let t = (-B - Math.sqrt(disc)) / A;
    if (t < 0) t = 0;
    if (t > tmax) return best;
    const s = md + t * nd;                     // projection onto the edge, scaled by dd
    if (s < 0 || s > dd) return best;          // outside the segment; vertices cover it
    return (best < 0 || t < best) ? t : best;
  }

  // ======================================================================
  //  BOX QUERY  (broadphase for the capsule solver and rigidbodies)
  // ======================================================================

  _queryBox(minx, miny, minz, maxx, maxy, maxz, limit) {
    const hits = this._qhits;
    const cap = limit ? Math.min(limit, hits.length) : hits.length;
    let n = 0;
    if (!this.ready || this.triCount === 0) { this._qcount = 0; return 0; }

    const nb = this.nBounds, nc = this.nCount, nl = this.nLeft, nsArr = this.nStart;
    const idx = this.triIdx, st = this._stack;
    let sp = 0;
    st[sp++] = 0;

    while (sp > 0) {
      const ni = st[--sp];
      const b = ni * 6;
      if (nb[b] > maxx || nb[b + 3] < minx ||
          nb[b + 1] > maxy || nb[b + 4] < miny ||
          nb[b + 2] > maxz || nb[b + 5] < minz) continue;
      const count = nc[ni];
      if (count > 0) {
        const s = nsArr[ni];
        for (let i = 0; i < count && n < cap; i++) hits[n++] = idx[s + i];
        if (n >= cap) break;
      } else {
        const l = nl[ni];
        if (l < 0 || sp + 2 >= st.length) continue;
        st[sp++] = l; st[sp++] = l + 1;
      }
    }
    this._qcount = n;
    return n;
  }

  // ======================================================================
  //  CLOSEST POINT / OVERLAP
  // ======================================================================

  /** Nearest surface point within maxDist. Used for explosion normals and FX. */
  closestPoint(point, maxDist, out) {
    const h = out || this._makeHit();
    const max = (typeof maxDist === 'number' && maxDist > 0) ? maxDist : 4;
    this._clearHit(h, max);
    if (!this.ready || this.triCount === 0 || !point) return h;

    const px = point.x, py = point.y, pz = point.z;
    const n = this._queryBox(px - max, py - max, pz - max, px + max, py + max, pz + max);
    let bd2 = max * max, bt = -1, bx = 0, by = 0, bz = 0;
    const tri = this.tri, hits = this._qhits;
    for (let i = 0; i < n; i++) {
      const t = hits[i], o = t * 9;
      closestPtTri(px, py, pz,
        tri[o], tri[o + 1], tri[o + 2],
        tri[o + 3], tri[o + 4], tri[o + 5],
        tri[o + 6], tri[o + 7], tri[o + 8], _pt3);
      const dx = px - _pt3[0], dy = py - _pt3[1], dz = pz - _pt3[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bd2) { bd2 = d2; bt = t; bx = _pt3[0]; by = _pt3[1]; bz = _pt3[2]; }
    }
    if (bt < 0) return h;
    const d = Math.sqrt(bd2);
    let nx = px - bx, ny = py - by, nz = pz - bz;
    if (d > 1e-7) { nx /= d; ny /= d; nz /= d; }
    else { const o3 = bt * 3; nx = this.triN[o3]; ny = this.triN[o3 + 1]; nz = this.triN[o3 + 2]; }
    this._fillHit(h, bt, bx, by, bz, nx, ny, nz, d);
    return h;
  }

  /**
   * All colliders overlapping a sphere, one contact per distinct object,
   * nearest-first. Explosions use this to find what to shake, scorch and shred.
   * Records are pooled — valid until the next overlapSphere call.
   */
  overlapSphere(center, radius, out) {
    const list = out || this._overlapOut;
    list.length = 0;
    if (!this.ready || this.triCount === 0 || !center) return list;

    const r = Math.max(0.01, radius);
    const px = center.x, py = center.y, pz = center.z;
    const n = this._queryBox(px - r, py - r, pz - r, px + r, py + r, pz + r);
    const r2 = r * r;
    const tri = this.tri, hits = this._qhits;
    const seen = this._seenIdx;
    seen.clear();
    let used = 0;

    for (let i = 0; i < n; i++) {
      const t = hits[i], o = t * 9;
      closestPtTri(px, py, pz,
        tri[o], tri[o + 1], tri[o + 2],
        tri[o + 3], tri[o + 4], tri[o + 5],
        tri[o + 6], tri[o + 7], tri[o + 8], _pt3);
      const dx = px - _pt3[0], dy = py - _pt3[1], dz = pz - _pt3[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) continue;

      const oi = this.triObj[t];
      const at = seen.get(oi);
      if (at !== undefined) {
        // keep only the nearest contact per object
        const rec = list[at];
        if (rec && d2 < rec.distance * rec.distance) this._writeOverlap(rec, t, _pt3, dx, dy, dz, d2);
        continue;
      }
      if (list.length >= 64) continue;
      let rec = this._overlapPool[used];
      if (!rec) { rec = this._makeHit(); this._overlapPool[used] = rec; }
      used++;
      this._writeOverlap(rec, t, _pt3, dx, dy, dz, d2);
      seen.set(oi, list.length);
      list.push(rec);
    }
    seen.clear();
    list.sort(this._byDistance);
    return list;
  }

  _writeOverlap(rec, t, p, dx, dy, dz, d2) {
    const d = Math.sqrt(d2);
    let nx = dx, ny = dy, nz = dz;
    if (d > 1e-7) { nx /= d; ny /= d; nz /= d; }
    else { const o3 = t * 3; nx = this.triN[o3]; ny = this.triN[o3 + 1]; nz = this.triN[o3 + 2]; }
    this._fillHit(rec, t, p[0], p[1], p[2], nx, ny, nz, d);
  }

  _byDistance(a, b) { return a.distance - b.distance; }

  /** Ground height under (x, z), searching downward from fromY. NaN if none. */
  groundHeight(x, z, fromY, maxDrop) {
    const o = this._gv || (this._gv = new THREE.Vector3());
    const d = this._gd || (this._gd = new THREE.Vector3(0, -1, 0));
    o.set(x, (typeof fromY === 'number' ? fromY : 50), z);
    d.set(0, -1, 0);
    const h = this.raycast(o, d, (typeof maxDrop === 'number' ? maxDrop : 200), this._hitC);
    return h.hit ? h.point.y : NaN;
  }

  // ======================================================================
  //  CAPSULE — character collision
  // ======================================================================

  /**
   * Move a capsule from `from` to `to`, resolving against the world.
   * Returns the SHARED sweepResult (see the header block).
   */
  capsuleSweep(from, to, radius, height, opts) {
    const res = this.sweepResult;
    const r = Math.max(0.02, radius ?? this.defaultRadius);
    const hgt = Math.max(r * 2 + 0.01, height ?? this.defaultHeight);

    res.grounded = false; res.touched = false; res.hitWall = false; res.hitCeiling = false;
    res.steppedUp = 0; res.slid = 0;
    res.groundNormal.set(0, 1, 0);
    res.wallNormal.set(0, 0, 0);
    res.normal.set(0, 1, 0);
    res.groundSurface = 'concrete';
    res.groundObject = null;

    if (!from) { res.position.set(0, 0, 0); return res; }
    const fx = from.x, fy = from.y, fz = from.z;
    if (!to) { res.position.set(fx, fy, fz); return res; }

    if (!this.ready || this.triCount === 0) { res.position.set(to.x, to.y, to.z); return res; }
    this._sweeps++;

    const o = opts || null;
    const wasGrounded = o ? !!o.grounded : false;
    const slopeDeg = (o && typeof o.slopeLimit === 'number') ? o.slopeLimit : this.slopeLimitDeg;
    const slopeCos = Math.cos(slopeDeg * Math.PI / 180);
    const stepH = (o && typeof o.stepHeight === 'number') ? o.stepHeight : this.stepHeight;
    const allowStep = stepH > 0 && !(o && o.skipStep);

    let dx = to.x - fx, dy = to.y - fy, dz = to.z - fz;
    const wantH = Math.sqrt(dx * dx + dz * dz);

    this._planeCount = 0;
    this._recordPlanes = true;

    // ---- primary attempt --------------------------------------------------
    this._slide(fx, fy, fz, dx, dy, dz, r, hgt, slopeCos, allowStep ? stepH : 0);
    this._recordPlanes = false;
    let px = this._rx, py = this._ry, pz = this._rz;
    let grounded = this._rGrounded;
    let gnx = this._rGnx, gny = this._rGny, gnz = this._rGnz;
    let gTri = this._rGroundTri;
    let touched = this._rTouch, wall = this._rWall, ceil = this._rCeil;
    let wnx = this._rWnx, wny = this._rWny, wnz = this._rWnz;
    let gotH = Math.sqrt((px - fx) * (px - fx) + (pz - fz) * (pz - fz));

    // ---- step-up ----------------------------------------------------------
    // The capsule's rounded bottom already rolls over anything shorter than its
    // radius (see the convex-edge branch in _depenetrate). This handles the rest:
    // blocked by a face while standing on walkable ground, with a walkable
    // surface just ahead and no more than stepHeight up. Probe it, lift onto it,
    // re-run the move. That is what carries the player over sandbags, kerbs and
    // rubble piles without a jump input.
    if (allowStep && touched && wantH > 1e-4 && gotH < wantH - 0.004 &&
        (grounded || wasGrounded) && dy <= 0.02) {
      const hx = dx / wantH, hz = dz / wantH;
      const pv = this._sv || (this._sv = new THREE.Vector3());
      const pd = this._sd || (this._sd = new THREE.Vector3(0, -1, 0));
      pv.set(fx + hx * (r + 0.07), fy + stepH + 0.12, fz + hz * (r + 0.07));
      pd.set(0, -1, 0);
      const gp = this.raycast(pv, pd, stepH + 0.20, this._hitB);
      if (gp.hit && gp.normal.y >= slopeCos) {
        // snapshot: _hitB is reused by the lip probe further down
        const tgtNx = gp.normal.x, tgtNy = gp.normal.y, tgtNz = gp.normal.z, tgtTri = gp.triangle;
        const rise = gp.point.y - fy;
        if (rise > 0.008 && rise <= stepH) {
          // does the capsule actually fit at the raised height, right here?
          const lifted = fy + rise + 0.006;
          this._depenetrate(fx, lifted, fz, r, hgt, slopeCos, 0);
          const slipX = this._rx - fx, slipY = this._ry - lifted, slipZ = this._rz - fz;
          if (slipX * slipX + slipY * slipY + slipZ * slipZ < 4e-4) {
            this._slide(fx, lifted, fz, dx, Math.max(0, dy), dz, r, hgt, slopeCos, stepH);
            let ax = this._rx, ay = this._ry, az = this._rz;
            const stepGotH = Math.sqrt((ax - fx) * (ax - fx) + (az - fz) * (az - fz));
            if (stepGotH > gotH + 0.004) {
              // Settle back down onto whatever is under the raised capsule —
              // usually the rounded lip of the obstacle itself. Accepting a
              // non-walkable lip here is deliberate: it is a transient pose the
              // player passes through while mounting the kerb, and refusing it
              // is what makes step-up feel like it teleports you into the air.
              pv.set(ax, ay + r, az);
              const lipDrop = Math.min(rise + 0.02, stepH + 0.02);
              const lp = this.sphereCast(pv, pd, r - 0.004, lipDrop + 0.004, this._hitD);
              if (lp.hit) ay -= Math.max(0, Math.min(lp.distance, lipDrop) - 0.0008);
              res.steppedUp = Math.max(0, ay - py);
              px = ax; py = ay; pz = az;
              grounded = true;
              if (this._rGroundTri >= 0) {
                gnx = this._rGnx; gny = this._rGny; gnz = this._rGnz; gTri = this._rGroundTri;
              } else {
                gnx = tgtNx; gny = tgtNy; gnz = tgtNz; gTri = tgtTri;
              }
              // While mounting a kerb the contact really is the rounded lip, but
              // reporting its steep normal would have the Controller treat the
              // step as an unwalkable slope and shed the player's speed. The
              // surface being climbed ONTO is what matters here.
              if (gny < slopeCos) { gnx = tgtNx; gny = tgtNy; gnz = tgtNz; }
              touched = true; wall = this._rWall; ceil = ceil || this._rCeil;
              gotH = stepGotH;
            }
          }
        }
      }
    }

    // ---- ground probe / snap ---------------------------------------------
    // A short downward sphere probe is far more stable than contact-only
    // grounding: it survives the frame where gravity has lifted you 2 mm off
    // the floor, and it glues you to descending stairs and slopes.
    let snap = (o && typeof o.snapDown === 'number') ? o.snapDown
      : ((wasGrounded && dy <= 0.001) ? this.snapDown : 0);
    const probe = grounded ? 0.02 : Math.max(0.06, snap);
    if (dy <= 0.001 && probe > 0) {
      const oV = this._pv || (this._pv = new THREE.Vector3());
      const dV = this._pd || (this._pd = new THREE.Vector3(0, -1, 0));
      oV.set(px, py + r, pz);
      dV.set(0, -1, 0);
      let gh = this.sphereCast(oV, dV, r - 0.004, probe + 0.004, this._hitB);
      let drop = gh.hit ? gh.distance : -1;
      if (!gh.hit || gh.normal.y < slopeCos) {
        // The swept sphere can catch the rounded lip of a ledge the player is
        // walking off, and a lip normal is never walkable. Fall back to a plain
        // ray straight down the capsule axis, which finds the actual floor the
        // player's feet are over.
        oV.set(px, py + r, pz);
        const gr = this.raycast(oV, dV, probe + r + 0.004, this._hitB);
        if (gr.hit && gr.normal.y >= slopeCos) { gh = gr; drop = gr.distance - r; }
        else drop = -1;
      }
      if (drop >= 0 && gh.normal.y >= slopeCos) {
        if (!grounded) {
          py -= Math.max(0, drop - 0.0005);
          grounded = true;
        }
        gnx = gh.normal.x; gny = gh.normal.y; gnz = gh.normal.z;
        gTri = gh.triangle;
        touched = true;
        // the floor is a real constraint even when the capsule is not strictly
        // penetrating it — projectVelocity must know about it
        this._recordPlanes = true;
        this._addPlane(gnx, gny, gnz);
        this._recordPlanes = false;
      }
    }

    // ---- write the result -------------------------------------------------
    if (py < this.voidY) py = this.voidY;
    res.position.set(px, py, pz);
    res.grounded = grounded;
    res.touched = touched;
    res.hitWall = wall;
    res.hitCeiling = ceil;
    res.slid = Math.max(0, wantH - gotH);
    res.groundNormal.set(gnx, gny, gnz);
    res.wallNormal.set(wnx, wny, wnz);
    res.normal.copy(grounded ? res.groundNormal : (wall ? res.wallNormal : res.groundNormal));
    if (grounded && gTri >= 0) {
      res.groundSurface = this.surfaces[this.triSurf[gTri]] || 'concrete';
      res.groundObject = this.objects[this.triObj[gTri]] || null;
    }
    return res;
  }

  /**
   * Move + depenetrate. Splits the motion so a sprinting player can never
   * tunnel, then runs a Gauss-Seidel depenetration pass at each waypoint.
   * Writes the _r* fields.
   */
  _slide(x, y, z, dx, dy, dz, r, hgt, slopeCos, stepH) {
    this._rTouch = false; this._rGrounded = false; this._rCeil = false; this._rWall = false;
    this._rGnx = 0; this._rGny = 1; this._rGnz = 0;
    this._rWnx = 0; this._rWny = 0; this._rWnz = 0;
    this._rGroundTri = -1;

    let touched = false, wall = false, ceil = false, grounded = false;
    let gnx = 0, gny = 0, gnz = 0, gTri = -1;
    let wnx = 0, wny = 0, wnz = 0;

    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    let steps = 1;
    if (len > r * 0.7) steps = Math.min(24, Math.ceil(len / (r * 0.7)));
    const sx = dx / steps, sy = dy / steps, sz = dz / steps;

    let px = x, py = y, pz = z;
    for (let s = 0; s < steps; s++) {
      px += sx; py += sy; pz += sz;
      this._depenetrate(px, py, pz, r, hgt, slopeCos, stepH);
      px = this._rx; py = this._ry; pz = this._rz;
      // sticky contact flags: a wall touched at ANY waypoint counts for the
      // whole move, otherwise a corner grazed mid-step vanishes from the result
      if (this._rTouch) touched = true;
      if (this._rWall) { wall = true; wnx = this._rWnx; wny = this._rWny; wnz = this._rWnz; }
      if (this._rCeil) ceil = true;
      if (this._rGrounded && this._rGny > gny) {
        grounded = true; gnx = this._rGnx; gny = this._rGny; gnz = this._rGnz; gTri = this._rGroundTri;
      }
    }
    this._rTouch = touched;
    this._rWall = wall;
    this._rCeil = ceil;
    this._rGrounded = grounded;
    if (gTri >= 0) { this._rGnx = gnx; this._rGny = gny; this._rGnz = gnz; this._rGroundTri = gTri; }
    if (wall) { this._rWnx = wnx; this._rWny = wny; this._rWnz = wnz; }
  }

  /**
   * Push the capsule out of every triangle it overlaps. Sequential
   * (Gauss-Seidel) so acute corners converge instead of oscillating; steep
   * faces are horizontalised so the player cannot ride a 70-degree wall
   * upward; ceilings get the full push so you cannot pop through a soffit.
   */
  _depenetrate(x, y, z, r, hgt, slopeCos, stepH) {
    const pad = 0.06;
    const step = stepH > 0 ? stepH : 0;
    const n = this._queryBox(
      x - r - pad, y - pad, z - r - pad,
      x + r + pad, y + hgt + pad, z + r + pad, MAX_CONTACT_TRIS);

    this._rx = x; this._ry = y; this._rz = z;
    this._rTouch = false; this._rWall = false; this._rCeil = false; this._rGrounded = false;
    this._rGnx = 0; this._rGny = 1; this._rGnz = 0; this._rGroundTri = -1;
    this._rWnx = 0; this._rWny = 0; this._rWnz = 0;
    if (n === 0) return;

    const tri = this.tri, triN = this.triN, hits = this._qhits, cc = this._cc;
    const segLo = Math.min(r, hgt * 0.5);
    const segHi = Math.max(hgt - r, segLo);

    let bestGroundY = slopeCos;
    let gnx = 0, gny = 1, gnz = 0, gTri = -1;
    let wall = false, wnx = 0, wny = 0, wnz = 0, wallDepth = 0;
    let ceil = false;
    let touched = false;

    for (let iter = 0; iter < 6; iter++) {
      let maxPen = 0;

      for (let i = 0; i < n; i++) {
        const t = hits[i];
        // Re-read the position for EVERY triangle (Gauss-Seidel). Doing this
        // once per iteration instead would double-correct any contact that lands
        // on an edge shared by two triangles — which is exactly what happens
        // when you stand on the diagonal seam of a quad, and it launches you.
        const ax = this._rx, ay = this._ry, az = this._rz;
        const b0x = ax, b0y = ay + segLo, b0z = az;
        const b1x = ax, b1y = ay + segHi, b1z = az;
        const d2 = this._capsuleTri(t, b0x, b0y, b0z, b1x, b1y, b1z, cc);
        if (d2 >= r * r) continue;

        let nx, ny, nz, depth;
        if (d2 > 1e-10) {
          const d = Math.sqrt(d2);
          nx = (cc[4] - cc[1]) / d; ny = (cc[5] - cc[2]) / d; nz = (cc[6] - cc[3]) / d;
          depth = r - d;
        } else {
          // axis intersects the face — push along the plane normal, toward the
          // side the capsule centre already favours
          const o3 = t * 3;
          nx = triN[o3]; ny = triN[o3 + 1]; nz = triN[o3 + 2];
          const o = t * 9;
          const mx = (b0x + b1x) * 0.5 - tri[o];
          const my = (b0y + b1y) * 0.5 - tri[o + 1];
          const mz = (b0z + b1z) * 0.5 - tri[o + 2];
          if (nx * mx + ny * my + nz * mz < 0) { nx = -nx; ny = -ny; nz = -nz; }
          depth = r + cc[7];
        }
        if (depth <= 1e-6) continue;
        touched = true;
        if (depth > maxPen) maxPen = depth;

        if (ny >= slopeCos) {
          if (ny > bestGroundY || gTri < 0) { bestGroundY = ny; gnx = nx; gny = ny; gnz = nz; gTri = t; }
          this._rGrounded = true;
        } else if (ny < -0.30) {
          ceil = true;
        } else if (ny > 0.04 && step > 0 && cc[2] <= y + step && this._isEdgeContact(t, nx, ny, nz)) {
          // CONVEX EDGE within step height — the top lip of a kerb, sandbag or
          // rubble pile. The push is genuinely "up and over", not "up a wall":
          // the contact sits on a triangle EDGE and points markedly more
          // skyward than that triangle's own face does. Let the rounded bottom
          // of the capsule roll over it, and call it ground so the player does
          // not read as airborne while climbing.
          if (gTri < 0) { gnx = 0; gny = 1; gnz = 0; gTri = t; }
          this._rGrounded = true;
        } else {
          wall = true;
          if (depth > wallDepth) { wallDepth = depth; wnx = nx; wny = ny; wnz = nz; }
          // strip the vertical component so a too-steep face cannot lift us
          const hl = Math.sqrt(nx * nx + nz * nz);
          if (hl > 1e-4) { nx /= hl; nz /= hl; ny = 0; }
        }

        const push = depth + this.skinWidth;
        this._rx += nx * push;
        this._ry += ny * push;
        this._rz += nz * push;
        this._addPlane(nx, ny, nz);
      }
      if (maxPen < 2e-4) break;
    }

    this._rTouch = touched;
    this._rWall = wall;
    this._rCeil = ceil;
    this._rWnx = wnx; this._rWny = wny; this._rWnz = wnz;
    if (gTri >= 0) { this._rGnx = gnx; this._rGny = gny; this._rGnz = gnz; this._rGroundTri = gTri; }
  }

  /**
   * True when the resolved contact normal differs appreciably from the
   * triangle's own face normal — i.e. the capsule is resting on the triangle's
   * BOUNDARY (the lip of a kerb), not on its interior. Deliberately symmetric:
   * a kerb's top lip is shared by the vertical face (whose normal is far more
   * horizontal than the contact) and the horizontal top (whose normal is far
   * more vertical), and both must agree that this is an edge — otherwise
   * whichever triangle the BVH happened to return first would decide whether
   * the player can climb, which is exactly the kind of order-dependence that
   * makes a controller feel broken. A planar surface, of any steepness, always
   * fails this test, so it can never be abused to walk up a wall.
   */
  _isEdgeContact(t, nx, ny, nz) {
    const o3 = t * 3;
    const tx = this.triN[o3], ty = this.triN[o3 + 1], tz = this.triN[o3 + 2];
    const d = tx * nx + ty * ny + tz * nz;
    const tny = d < 0 ? -ty : ty;
    return Math.abs(ny - tny) > 0.10;
  }

  _addPlane(nx, ny, nz) {
    if (!this._recordPlanes) return;
    const p = this._planes;
    for (let i = 0; i < this._planeCount; i++) {
      const o = i * 3;
      if (p[o] * nx + p[o + 1] * ny + p[o + 2] * nz > 0.985) return;  // duplicate
    }
    if (this._planeCount >= CONTACT_PLANES) return;
    const o = this._planeCount * 3;
    p[o] = nx; p[o + 1] = ny; p[o + 2] = nz;
    this._planeCount++;
  }

  /**
   * Clip a velocity against the contact planes recorded by the last
   * capsuleSweep. Call it right after the sweep so the player loses the
   * component of speed that went into a wall instead of accumulating it.
   */
  projectVelocity(vel) {
    if (!vel || this._planeCount === 0) return vel;
    const p = this._planes;
    for (let pass = 0; pass < 2; pass++) {
      let changed = false;
      for (let i = 0; i < this._planeCount; i++) {
        const o = i * 3;
        const nx = p[o], ny = p[o + 1], nz = p[o + 2];
        const d = vel.x * nx + vel.y * ny + vel.z * nz;
        if (d < -1e-5) {
          vel.x -= nx * d; vel.y -= ny * d; vel.z -= nz * d;
          changed = true;
        }
      }
      if (!changed) break;
    }
    return vel;
  }

  /**
   * Squared distance between a capsule axis segment and triangle `t`.
   * out: [ _, triPt(3), segPt(3), signedPenetrationExtra ]
   * Exact for disjoint features; the crossing case is flagged with d2 = 0 and
   * out[7] holding the extra depth past the plane.
   */
  _capsuleTri(t, p0x, p0y, p0z, p1x, p1y, p1z, out) {
    const tri = this.tri, o = t * 9, o3 = t * 3;
    const ax = tri[o], ay = tri[o + 1], az = tri[o + 2];
    const bx = tri[o + 3], by = tri[o + 4], bz = tri[o + 5];
    const cx = tri[o + 6], cy = tri[o + 7], cz = tri[o + 8];
    const nx = this.triN[o3], ny = this.triN[o3 + 1], nz = this.triN[o3 + 2];

    // --- crossing test: axis passes through the face -----------------------
    const s0 = (p0x - ax) * nx + (p0y - ay) * ny + (p0z - az) * nz;
    const s1 = (p1x - ax) * nx + (p1y - ay) * ny + (p1z - az) * nz;
    if ((s0 > 0) !== (s1 > 0)) {
      const u = s0 / (s0 - s1);
      const ix = p0x + (p1x - p0x) * u;
      const iy = p0y + (p1y - p0y) * u;
      const iz = p0z + (p1z - p0z) * u;
      if (this._pointInTri(ix, iy, iz, ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz)) {
        out[1] = ix; out[2] = iy; out[3] = iz;
        out[4] = ix; out[5] = iy; out[6] = iz;
        out[7] = Math.min(Math.abs(s0), Math.abs(s1));
        return 0;
      }
    }
    out[7] = 0;

    let best = Infinity;
    let tpx = 0, tpy = 0, tpz = 0, spx = 0, spy = 0, spz = 0;

    // endpoint vs face
    closestPtTri(p0x, p0y, p0z, ax, ay, az, bx, by, bz, cx, cy, cz, _pt3);
    let ddx = p0x - _pt3[0], ddy = p0y - _pt3[1], ddz = p0z - _pt3[2];
    let d2 = ddx * ddx + ddy * ddy + ddz * ddz;
    if (d2 < best) { best = d2; tpx = _pt3[0]; tpy = _pt3[1]; tpz = _pt3[2]; spx = p0x; spy = p0y; spz = p0z; }

    closestPtTri(p1x, p1y, p1z, ax, ay, az, bx, by, bz, cx, cy, cz, _pt3);
    ddx = p1x - _pt3[0]; ddy = p1y - _pt3[1]; ddz = p1z - _pt3[2];
    d2 = ddx * ddx + ddy * ddy + ddz * ddz;
    if (d2 < best) { best = d2; tpx = _pt3[0]; tpy = _pt3[1]; tpz = _pt3[2]; spx = p1x; spy = p1y; spz = p1z; }

    // axis vs the three edges
    segSeg(p0x, p0y, p0z, p1x, p1y, p1z, ax, ay, az, bx, by, bz, _ss7);
    if (_ss7[6] < best) { best = _ss7[6]; spx = _ss7[0]; spy = _ss7[1]; spz = _ss7[2]; tpx = _ss7[3]; tpy = _ss7[4]; tpz = _ss7[5]; }
    segSeg(p0x, p0y, p0z, p1x, p1y, p1z, bx, by, bz, cx, cy, cz, _ss7);
    if (_ss7[6] < best) { best = _ss7[6]; spx = _ss7[0]; spy = _ss7[1]; spz = _ss7[2]; tpx = _ss7[3]; tpy = _ss7[4]; tpz = _ss7[5]; }
    segSeg(p0x, p0y, p0z, p1x, p1y, p1z, cx, cy, cz, ax, ay, az, _ss7);
    if (_ss7[6] < best) { best = _ss7[6]; spx = _ss7[0]; spy = _ss7[1]; spz = _ss7[2]; tpx = _ss7[3]; tpy = _ss7[4]; tpz = _ss7[5]; }

    out[1] = tpx; out[2] = tpy; out[3] = tpz;
    out[4] = spx; out[5] = spy; out[6] = spz;
    return best;
  }

  // ======================================================================
  //  RIGIDBODIES
  // ======================================================================

  _makeBody(id) {
    return {
      id, active: false, sleeping: false, sleepTimer: 0,
      px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0,
      ppx: 0, ppy: 0, ppz: 0,
      qx: 0, qy: 0, qz: 0, qw: 1,
      pqx: 0, pqy: 0, pqz: 0, pqw: 1,
      wx: 0, wy: 0, wz: 0,
      radius: 0.03, mass: 0.02, invMass: 50,
      restitution: 0.32, friction: 0.55, drag: 0.06, angularDrag: 0.9,
      gravityScale: 1, life: 0, maxLife: 8,
      grounded: false, surface: 'metal',
      object3D: null, onImpact: null, impactEvents: true,
      impactCooldown: 0, userData: null,
    };
  }

  /**
   * Grab a pooled rigidbody. Returns null when all 200 are in flight — always
   * null-check; a full pool is the normal, expected state during heavy fire.
   */
  spawnBody(opts) {
    const b = this._free.pop();
    if (!b) return null;
    const o = opts || {};

    const p = o.position;
    b.px = p ? p.x : (o.x || 0);
    b.py = p ? p.y : (o.y || 0);
    b.pz = p ? p.z : (o.z || 0);
    b.ppx = b.px; b.ppy = b.py; b.ppz = b.pz;

    const v = o.velocity;
    b.vx = v ? v.x : (o.vx || 0);
    b.vy = v ? v.y : (o.vy || 0);
    b.vz = v ? v.z : (o.vz || 0);

    const q = o.quaternion;
    b.qx = q ? q.x : 0; b.qy = q ? q.y : 0; b.qz = q ? q.z : 0; b.qw = q ? q.w : 1;
    b.pqx = b.qx; b.pqy = b.qy; b.pqz = b.qz; b.pqw = b.qw;

    const w = o.angularVelocity;
    if (w) { b.wx = w.x; b.wy = w.y; b.wz = w.z; }
    else if (typeof o.spin === 'number') {
      // deterministic axis derived from the id — no Math.random in the sim path
      const a = b.id * 2.399963;
      b.wx = Math.cos(a) * o.spin; b.wy = Math.sin(a * 1.7) * o.spin; b.wz = Math.sin(a) * o.spin;
    } else { b.wx = 0; b.wy = 0; b.wz = 0; }

    b.radius = Math.max(0.004, o.radius ?? 0.03);
    b.mass = Math.max(0.001, o.mass ?? 0.02);
    b.invMass = 1 / b.mass;
    b.restitution = o.restitution ?? 0.32;
    b.friction = o.friction ?? 0.55;
    b.drag = o.drag ?? 0.06;
    b.angularDrag = o.angularDrag ?? 0.9;
    b.gravityScale = o.gravityScale ?? 1;
    b.maxLife = o.life ?? 8;
    b.life = 0;
    b.surface = o.surface || 'metal';
    b.object3D = o.object3D || null;
    b.onImpact = o.onImpact || null;
    b.impactEvents = o.impactEvents !== false;
    b.impactCooldown = 0;
    b.userData = o.userData || null;
    b.sleeping = false;
    b.sleepTimer = 0;
    b.grounded = false;
    b.active = true;
    this.awakeCount++;
    return b;
  }

  releaseBody(b) {
    if (!b || !b.active) return;
    b.active = false;
    b.object3D = null;
    b.onImpact = null;
    b.userData = null;
    this.awakeCount = Math.max(0, this.awakeCount - 1);
    this._free.push(b);
  }

  releaseAllBodies() {
    for (let i = 0; i < this.bodies.length; i++) if (this.bodies[i].active) this.releaseBody(this.bodies[i]);
  }

  /** Blast impulse on every awake-able body inside the radius. */
  applyExplosion(center, radius, power) {
    const r = Math.max(0.1, radius || 4);
    const p = power || 900;
    const r2 = r * r;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      if (!b.active) continue;
      const dx = b.px - center.x, dy = b.py - center.y, dz = b.pz - center.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) continue;
      const d = Math.sqrt(d2) || 1e-3;
      const fall = 1 - d / r;
      const k = (p * fall * fall) * b.invMass * 0.001;
      b.vx += (dx / d) * k;
      b.vy += (dy / d) * k + fall * 1.6;
      b.vz += (dz / d) * k;
      b.wx += (dy / d) * fall * 26;
      b.wz += (dx / d) * fall * 26;
      b.sleeping = false;
      b.sleepTimer = 0;
    }
  }

  _stepBodies(h) {
    const g = this.gravity;
    const bodies = this.bodies;
    let awake = 0;

    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      if (!b.active) continue;

      b.life += h;
      if (b.life >= b.maxLife || b.py < this.voidY) { this.releaseBody(b); continue; }
      if (b.impactCooldown > 0) b.impactCooldown -= h;

      b.ppx = b.px; b.ppy = b.py; b.ppz = b.pz;
      b.pqx = b.qx; b.pqy = b.qy; b.pqz = b.qz; b.pqw = b.qw;
      if (b.sleeping) continue;
      awake++;

      // --- integrate ------------------------------------------------------
      b.vy += g * b.gravityScale * h;
      const dampV = 1 - Math.min(0.5, b.drag * h);
      b.vx *= dampV; b.vy *= dampV; b.vz *= dampV;

      let mx = b.vx * h, my = b.vy * h, mz = b.vz * h;
      const moveLen = Math.sqrt(mx * mx + my * my + mz * mz);

      // --- translate, with a swept test when the step is long -------------
      let needOverlap = true;
      if (moveLen > b.radius * 0.55 && moveLen > 1e-6) {
        const oV = this._bo || (this._bo = new THREE.Vector3());
        const dV = this._bd || (this._bd = new THREE.Vector3());
        oV.set(b.px, b.py, b.pz);
        dV.set(mx / moveLen, my / moveLen, mz / moveLen);
        const hit = this.sphereCast(oV, dV, b.radius, moveLen, this._hitA);
        if (hit.hit) {
          const back = Math.max(0, hit.distance - 0.0015);
          b.px += dV.x * back; b.py += dV.y * back; b.pz += dV.z * back;
          this._bodyResponse(b, hit.normal.x, hit.normal.y, hit.normal.z, hit.point, hit.triangle);
        } else {
          b.px += mx; b.py += my; b.pz += mz;
          // The sweep covered the whole move and found nothing, and the body
          // started outside the world, so it cannot be touching anything now.
          // Skipping the overlap query here halves the cost of airborne debris.
          needOverlap = false;
        }
      } else {
        b.px += mx; b.py += my; b.pz += mz;
      }

      // --- resting / shallow contacts -------------------------------------
      b.grounded = false;
      const r = b.radius;
      const n = needOverlap
        ? this._queryBox(b.px - r, b.py - r, b.pz - r, b.px + r, b.py + r, b.pz + r, 96)
        : 0;
      if (n > 0) {
        const tri = this.tri, hits = this._qhits;
        for (let pass = 0; pass < 2; pass++) {
          let hitAny = false;
          for (let k = 0; k < n; k++) {
            const t = hits[k], o = t * 9;
            closestPtTri(b.px, b.py, b.pz,
              tri[o], tri[o + 1], tri[o + 2],
              tri[o + 3], tri[o + 4], tri[o + 5],
              tri[o + 6], tri[o + 7], tri[o + 8], _pt3);
            let nx = b.px - _pt3[0], ny = b.py - _pt3[1], nz = b.pz - _pt3[2];
            let d2 = nx * nx + ny * ny + nz * nz;
            if (d2 >= r * r) continue;
            let d = Math.sqrt(d2);
            if (d < 1e-7) {
              const o3 = t * 3;
              nx = this.triN[o3]; ny = this.triN[o3 + 1]; nz = this.triN[o3 + 2];
              d = 0;
            } else { nx /= d; ny /= d; nz /= d; }
            const pen = r - d;
            b.px += nx * pen; b.py += ny * pen; b.pz += nz * pen;
            this._bodyResponse(b, nx, ny, nz, null, t);
            hitAny = true;
          }
          if (!hitAny) break;
        }
      }

      // --- orientation -----------------------------------------------------
      const wl = Math.sqrt(b.wx * b.wx + b.wy * b.wy + b.wz * b.wz);
      if (wl > 1e-5) {
        const ha = wl * h * 0.5;
        const s = Math.sin(ha) / wl, c = Math.cos(ha);
        const dqx = b.wx * s, dqy = b.wy * s, dqz = b.wz * s, dqw = c;
        const nqx = dqw * b.qx + dqx * b.qw + dqy * b.qz - dqz * b.qy;
        const nqy = dqw * b.qy - dqx * b.qz + dqy * b.qw + dqz * b.qx;
        const nqz = dqw * b.qz + dqx * b.qy - dqy * b.qx + dqz * b.qw;
        const nqw = dqw * b.qw - dqx * b.qx - dqy * b.qy - dqz * b.qz;
        const il = 1 / (Math.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw) || 1);
        b.qx = nqx * il; b.qy = nqy * il; b.qz = nqz * il; b.qw = nqw * il;
      }
      const dampW = Math.max(0, 1 - b.angularDrag * h);
      b.wx *= dampW; b.wy *= dampW; b.wz *= dampW;

      // --- sleeping --------------------------------------------------------
      const sp2 = b.vx * b.vx + b.vy * b.vy + b.vz * b.vz;
      const w2 = b.wx * b.wx + b.wy * b.wy + b.wz * b.wz;
      if (b.grounded && sp2 < 0.020 && w2 < 1.2) {
        b.sleepTimer += h;
        if (b.sleepTimer > 0.35) {
          b.sleeping = true;
          b.vx = b.vy = b.vz = 0;
          b.wx = b.wy = b.wz = 0;
        }
      } else {
        b.sleepTimer = 0;
      }
    }
    this.awakeCount = awake;
  }

  _bodyResponse(b, nx, ny, nz, point, triIndex) {
    const vn = b.vx * nx + b.vy * ny + b.vz * nz;
    if (ny > 0.55) b.grounded = true;
    if (vn >= 0) return;

    const speed = -vn;
    // restitution dies out at low speed so debris settles instead of buzzing
    const e = speed < 0.55 ? 0 : b.restitution;
    const j = (1 + e) * speed;
    b.vx += nx * j; b.vy += ny * j; b.vz += nz * j;

    // Coulomb friction on the tangential component
    let tx = b.vx - nx * (b.vx * nx + b.vy * ny + b.vz * nz);
    let ty = b.vy - ny * (b.vx * nx + b.vy * ny + b.vz * nz);
    let tz = b.vz - nz * (b.vx * nx + b.vy * ny + b.vz * nz);
    const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (tl > 1e-5) {
      const maxF = b.friction * j;
      const k = Math.min(tl, maxF) / tl;
      b.vx -= tx * k; b.vy -= ty * k; b.vz -= tz * k;
      // convert the scrubbed tangential motion into spin (rolling approximation)
      const rs = (1 / Math.max(0.004, b.radius)) * 0.7;
      b.wx += (ny * tz - nz * ty) * rs * k;
      b.wy += (nz * tx - nx * tz) * rs * k;
      b.wz += (nx * ty - ny * tx) * rs * k;
    }

    // --- impact feedback ---------------------------------------------------
    if (speed > 1.1 && b.impactCooldown <= 0) {
      b.impactCooldown = 0.06;
      const surf = (triIndex >= 0 && this.triSurf) ? (this.surfaces[this.triSurf[triIndex]] || 'concrete') : 'concrete';
      if (b.onImpact) {
        try { b.onImpact(b, surf, speed, nx, ny, nz); } catch (e) { /* collaborator callback */ }
      }
      if (b.impactEvents && this._evtBudget > 0) {
        this._evtBudget--;
        const ev = this._evtPool[this._evtI = (this._evtI + 1) & 15];
        if (point) ev.point.copy(point);
        else ev.point.set(b.px - nx * b.radius, b.py - ny * b.radius, b.pz - nz * b.radius);
        ev.normal.set(nx, ny, nz);
        ev.surface = surf;
        ev.power = speed;
        ev.body = b;
        const o = (triIndex >= 0 && this.triObj) ? this.objects[this.triObj[triIndex]] : null;
        const m = o ? o.material : null;
        ev.material = Array.isArray(m) ? m[0] : m || null;
        this.g?.bus?.emit?.('impact', ev);
      }
    }
  }

  _applyBodyTransforms(alpha) {
    const qa = this._qa, qb = this._qb, qc = this._qc;
    const bodies = this.bodies;
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      if (!b.active || !b.object3D) continue;
      const o = b.object3D;
      if (b.sleeping) {
        o.position.set(b.px, b.py, b.pz);
        o.quaternion.set(b.qx, b.qy, b.qz, b.qw);
        continue;
      }
      o.position.set(
        b.ppx + (b.px - b.ppx) * alpha,
        b.ppy + (b.py - b.ppy) * alpha,
        b.ppz + (b.pz - b.ppz) * alpha);
      qa.set(b.pqx, b.pqy, b.pqz, b.pqw);
      qb.set(b.qx, b.qy, b.qz, b.qw);
      qc.copy(qa).slerp(qb, alpha);
      o.quaternion.copy(qc);
    }
  }

  // ======================================================================
  //  FRAME
  // ======================================================================

  update(dt, t) {
    // --- lazily pick up geometry registered after our init ------------------
    if ((this._watchFrame++ & 15) === 0) {
      const reg = this.g?.registry;
      const len = reg?.colliders?.length ?? 0;
      const ver = reg?.collidersVersion ?? 0;
      if (this._dirty || len !== this._builtLen || ver !== this._builtVersion) {
        if (len > 0 || this._dirty) this.rebuild();
        else { this._builtLen = len; this._builtVersion = ver; }
      }
    }

    if (!(dt > 0)) { this._applyBodyTransforms(1); return; }
    const t0 = now();

    // impact-event budget: 14/s, refilled continuously
    this._evtClock += dt;
    if (this._evtClock >= 0.25) { this._evtClock = 0; this._evtBudget = 4; }

    this._acc += dt;
    let n = 0;
    while (this._acc >= FIXED_DT && n < MAX_SUBSTEPS) {
      this._stepBodies(FIXED_DT);
      this.simTime += FIXED_DT;
      this._acc -= FIXED_DT;
      n++;
      this.steps++;
    }
    if (n >= MAX_SUBSTEPS) this._acc = 0;      // never spiral

    this._applyBodyTransforms(this._acc / FIXED_DT);

    const s = this.stats;
    s.stepMs = +(now() - t0).toFixed(2);
    s.bodies = MAX_BODIES - this._free.length;
    s.awake = this.awakeCount;
    s.rays = this._rays; s.sweeps = this._sweeps;
    this._rays = 0; this._sweeps = 0;
    void t;
  }

  // ======================================================================
  //  DEBUG
  // ======================================================================

  /** BVH wireframe. Off by default; nothing is added to the scene until asked. */
  setDebugDraw(on, depth) {
    const scene = this.g?.scene;
    if (!on) {
      if (this._debugObj && scene) scene.remove(this._debugObj);
      this._debugObj?.geometry?.dispose?.();
      this._debugObj?.material?.dispose?.();
      this._debugObj = null;
      this.debugDraw = false;
      return;
    }
    if (!scene || !this.ready) return;
    this.setDebugDraw(false);
    this._debugDepth = depth ?? 6;

    const boxes = [];
    const stack = [0, 0];
    while (stack.length) {
      const d = stack.pop(), ni = stack.pop();
      if (d <= this._debugDepth) boxes.push(ni);
      if (d >= this._debugDepth) continue;
      if (this.nCount[ni] === 0) {
        const l = this.nLeft[ni];
        if (l >= 0) { stack.push(l, d + 1, l + 1, d + 1); }
      }
    }
    const verts = new Float32Array(boxes.length * 24 * 3);
    const E = [
      0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
    ];
    let vi = 0;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i] * 6;
      const x0 = this.nBounds[b], y0 = this.nBounds[b + 1], z0 = this.nBounds[b + 2];
      const x1 = this.nBounds[b + 3], y1 = this.nBounds[b + 4], z1 = this.nBounds[b + 5];
      const cx = [x0, x1, x1, x0, x0, x1, x1, x0];
      const cy = [y0, y0, y0, y0, y1, y1, y1, y1];
      const cz = [z0, z0, z1, z1, z0, z0, z1, z1];
      for (let e = 0; e < E.length; e++) {
        const c = E[e];
        verts[vi++] = cx[c]; verts[vi++] = cy[c]; verts[vi++] = cz[c];
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x39ff88, transparent: true, opacity: 0.35, fog: false, toneMapped: false });
    const obj = new THREE.LineSegments(geo, mat);
    obj.name = 'physics_bvh_debug';
    obj.frustumCulled = false;
    obj.renderOrder = 999;
    scene.add(obj);
    this._debugObj = obj;
    this.debugDraw = true;
  }
}

export default Physics;
