// ============================================================================
// BLACKSITE — src/art/Level.js
// Owner: Level agent.  The playable environment: a dusty Middle-Eastern urban
// compound, ~90x90 m, late afternoon (~16:20), sun low in the WNW.
//
// Everything here is procedural geometry built once at init().  No file loads.
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators depend on exactly this)
// ---------------------------------------------------------------------------
//   game.level.root           THREE.Group          everything the level owns
//   game.level.spawnPoints    [{pos:Vector3, yaw:number}]   yaw = camera.rotation.y
//   game.level.navBounds      { min:Vector3, max:Vector3, center:Vector3, size:Vector3 }
//   game.level.coverPoints    [{pos:Vector3, dir:Vector3, height:number}]
//   game.level.colliders      Mesh[]   (also pushed into game.registry.colliders)
//   game.level.getSpawn(i)    -> {pos, yaw}
//   game.level.sampleGroundY(x, z) -> number      cheap floor query for AI/props
//   game.level.stats          { draws, tris, buckets, instances }
//
// ---------------------------------------------------------------------------
// WORLD ORIENTATION  (all camera presets in tools/shoot.mjs were solved against
// this; yaw is camera.rotation.y, so forward = (-sin yaw, 0, -cos yaw))
//   +Z = "north"   +X = "east"
//   sun bearing ~ WNW, elevation ~10.7 deg -> direction TO sun = (-0.87, 0.19, 0.43)
//   therefore: WEST-facing surfaces are hot, NORTH-facing warm, EAST/SOUTH in shade.
//   The main street runs N-S at x in [-7, 7]; the east row is sunlit above ~6 m,
//   the west row is backlit.  The market plaza (z 23.4 .. 40) is open to the NW
//   and is the golden light pool every "look north" camera preset is aimed into.
// ============================================================================

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* ------------------------------------------------------------------ *
 *  deterministic math
 * ------------------------------------------------------------------ */

function mulberry32(a) {
  a = (a >>> 0) || 0x9e3779b9;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
function smooth(e0, e1, x) {
  let t = (x - e0) / ((e1 - e0) || 1e-6);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}

function ihash3(x, y, z) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(z | 0, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** cheap trilinear value noise, [0,1] */
function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  let fx = x - xi, fy = y - yi, fz = z - zi;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
  const c000 = ihash3(xi, yi, zi), c100 = ihash3(xi + 1, yi, zi);
  const c010 = ihash3(xi, yi + 1, zi), c110 = ihash3(xi + 1, yi + 1, zi);
  const c001 = ihash3(xi, yi, zi + 1), c101 = ihash3(xi + 1, yi, zi + 1);
  const c011 = ihash3(xi, yi + 1, zi + 1), c111 = ihash3(xi + 1, yi + 1, zi + 1);
  const x00 = c000 + (c100 - c000) * fx, x10 = c010 + (c110 - c010) * fx;
  const x01 = c001 + (c101 - c001) * fx, x11 = c011 + (c111 - c011) * fx;
  const y0 = x00 + (x10 - x00) * fy, y1 = x01 + (x11 - x01) * fy;
  return y0 + (y1 - y0) * fz;
}

/* ------------------------------------------------------------------ *
 *  box builder — world-anchored UVs so tiling is continuous across the
 *  level and adjacent boxes never repeat the same texel patch.
 * ------------------------------------------------------------------ */

const FACE_DEF = [
  { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
  { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
  { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
  { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
  { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
];
const QUAD_IDX = [0, 1, 2, 0, 2, 3];

function boxGeo(cx, cy, cz, w, h, d, ts, faces) {
  const hx = w * 0.5, hy = h * 0.5, hz = d * 0.5;
  const mask = faces === undefined ? 63 : faces;
  let nf = 0;
  for (let f = 0; f < 6; f++) if (mask & (1 << f)) nf++;
  const pos = new Float32Array(nf * 12);
  const nor = new Float32Array(nf * 12);
  const uvs = new Float32Array(nf * 8);
  const idx = new Uint16Array(nf * 6);
  const inv = 1 / ts;
  let vo = 0, io = 0, base = 0;

  for (let f = 0; f < 6; f++) {
    if (!(mask & (1 << f))) continue;
    const F = FACE_DEF[f];
    const n = F.n, u = F.u, v = F.v;
    const hu = Math.abs(u[0] * hx + u[1] * hy + u[2] * hz);
    const hv = Math.abs(v[0] * hx + v[1] * hy + v[2] * hz);
    const hn = Math.abs(n[0] * hx + n[1] * hy + n[2] * hz);
    const ox = cx + n[0] * hn, oy = cy + n[1] * hn, oz = cz + n[2] * hn;
    for (let k = 0; k < 4; k++) {
      const su = (k === 0 || k === 3) ? -hu : hu;
      const sv = (k < 2) ? -hv : hv;
      const px = ox + u[0] * su + v[0] * sv;
      const py = oy + u[1] * su + v[1] * sv;
      const pz = oz + u[2] * su + v[2] * sv;
      pos[vo] = px; pos[vo + 1] = py; pos[vo + 2] = pz;
      nor[vo] = n[0]; nor[vo + 1] = n[1]; nor[vo + 2] = n[2];
      const t2 = (vo / 3) * 2;
      uvs[t2] = (u[0] * px + u[1] * py + u[2] * pz) * inv;
      uvs[t2 + 1] = (v[0] * px + v[1] * py + v[2] * pz) * inv;
      vo += 3;
    }
    for (let k = 0; k < 6; k++) idx[io + k] = base + QUAD_IDX[k];
    io += 6; base += 4;
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

const _mA = new THREE.Matrix4();
const _mB = new THREE.Matrix4();
const _mC = new THREE.Matrix4();
const _axis = new THREE.Vector3();

function rotYg(geo, cx, cz, ang) {
  if (!ang) return geo;
  _mA.makeTranslation(cx, 0, cz);
  _mB.makeRotationY(ang);
  _mC.makeTranslation(-cx, 0, -cz);
  geo.applyMatrix4(_mA.multiply(_mB).multiply(_mC));
  return geo;
}

function rotAxisG(geo, cx, cy, cz, ax, ay, az, ang) {
  if (!ang) return geo;
  _axis.set(ax, ay, az).normalize();
  _mA.makeTranslation(cx, cy, cz);
  _mB.makeRotationAxis(_axis, ang);
  _mC.makeTranslation(-cx, -cy, -cz);
  geo.applyMatrix4(_mA.multiply(_mB).multiply(_mC));
  return geo;
}

function scaleUv(geo, su, sv) {
  const uv = geo.attributes.uv;
  if (!uv) return geo;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  uv.needsUpdate = true;
  return geo;
}

function ensureIndexed(geo) {
  if (geo.index) return geo;
  const n = geo.attributes.position.count;
  const idx = n > 65535 ? new Uint32Array(n) : new Uint16Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  return geo;
}

function trimAttrs(geo) {
  for (const k of Object.keys(geo.attributes)) {
    if (k !== 'position' && k !== 'normal' && k !== 'uv' && k !== 'color' && k !== 'aWind') {
      geo.deleteAttribute(k);
    }
  }
  return geo;
}

/* ------------------------------------------------------------------ *
 *  real-world scale constants
 * ------------------------------------------------------------------ */

const T_WALL = 0.34;
const STOREY = 3.10;
const DOOR_H = 2.05;
const KERB_H = 0.15;
const KERB_W = 0.32;
const ROAD_HW = 5.6;
const HALL_DECK = 2.58;      // market-hall roof deck: the "vista" preset stands here
const WALK_H = 0.17;         // raised sidewalk strips ringing the plaza
const INT_FY = 0.14;         // W3 interior screed level — clears the dirt noise
const CHAN_HW = 0.78;        // half width of the drainage channel down the street centre

const NO_COLLIDE = new Set(['detail', 'cloth', 'wire', 'sign', 'scar', 'trim', 'intdet']);

/* sun, matching the Sky module's late-afternoon key (unit, pointing TO the sun) */
const SUN = [-0.879, 0.192, 0.434];

/**
 * Hero camera stations, mirroring the presets in tools/shoot.mjs.
 * The dressing pass must FRAME these sightlines, never curtain them off, so
 * every station owns a short wedge that tall props keep out of.
 *
 * The depth matters and is not arbitrary: a 2.3 m stack at 3 m puts its top
 * ~12 deg above the horizon and swallows the gate arch whole, but the same
 * stack at 10 m tops out below 4 deg and merely crops the arch's footing —
 * which is good composition, not a blindfold. So the wedges are deliberately
 * SHORT. Only the first few metres are protected; past that the tall kit is
 * exactly what gives the frame its cover read.
 *   near = depth the wedge guards (m)
 */
const D2R = Math.PI / 180;
const HERO_VIEWS = [
  { x: 0.0, z: 18.0, yaw: 180 * D2R, near: 6.5 },   // spawn
  { x: 2.5, z: 6.0, yaw: 195 * D2R, near: 6.0 },   // street / ads
  { x: 0.0, z: 14.0, yaw: 178 * D2R, near: 7.5 },   // weapon (pitched down, sees more floor)
  { x: 0.0, z: 10.0, yaw: 118 * D2R, near: 5.0 },   // sun
  { x: 11.0, z: -2.0, yaw: 250 * D2R, near: 4.0 },   // alley
  { x: -14.0, z: 22.0, yaw: 150 * D2R, near: 4.0 },   // vista (perched on the hall deck)
];
const WEDGE_BASE = 0.65;     // half-width of the wedge at the camera
const WEDGE_SLOPE = 0.26;    // extra half-width per metre of depth

export class Level {

  constructor(game) {
    this.g = game;
    this.THREE = THREE;

    this.root = new THREE.Group();
    this.root.name = 'level';

    this.colliders = [];
    this.spawnPoints = [];
    this.coverPoints = [];
    this.meshes = [];
    // dressLow/dressMid/dressSkip let the art pass be tuned without guessing:
    // if dressSkip dwarfs the other two the floor reservations are too greedy.
    this.stats = {
      draws: 0, tris: 0, buckets: 0, instances: 0,
      dressLow: 0, dressMid: 0, dressSkip: 0,
    };

    this.navBounds = {
      min: new THREE.Vector3(-30, -1, -32),
      max: new THREE.Vector3(24, 18, 46),
      center: new THREE.Vector3(-3, 8.5, 7),
      size: new THREE.Vector3(54, 19, 78),
    };

    this._buckets = new Map();
    this._instPools = new Map();
    this._stains = [];
    this._occ = [];              // flat [x, z, r, ...] footprint reservations
    this._mats = {};
    this._tsz = {};
    this._zone = 'misc';
    this._tallGuard = false;     // set while a >1 m silhouette is looking for a home
    this._rng = mulberry32(0x8AC51E);
    // Separate deterministic streams for the round-3 façade + interior passes.
    // Drawing them from the main stream would reshuffle every prop placement
    // the previous rounds settled on, so they get their own generators.
    this._drng = mulberry32(0x2F19B7);
    this._irng = mulberry32(0xC41D05);
    this._IR = null;             // interior-room constants (see _interiorRoom)
    this._windU = { value: 0 };
    this._t = 0;

    game.scene.add(this.root);
  }

  _r() { return this._rng(); }
  _rr(a, b) { return a + (b - a) * this._rng(); }
  _ri(a, b) { return a + Math.floor(this._rng() * (b - a + 1)); }

  _dr() { return this._drng(); }
  _drr(a, b) { return a + (b - a) * this._drng(); }
  _dri(a, b) { return a + Math.floor(this._drng() * (b - a + 1)); }

  _ir() { return this._irng(); }
  _irr(a, b) { return a + (b - a) * this._irng(); }
  _iri(a, b) { return a + Math.floor(this._irng() * (b - a + 1)); }

  // =====================================================================
  //  LIFECYCLE
  // =====================================================================

  async init() {
    const yield_ = () => new Promise(r => setTimeout(r, 0));
    this._buildMaterials();
    await yield_();
    this._buildArchitecture();
    await yield_();
    this._buildProps();
    await yield_();
    this._buildDetail();
    await yield_();
    this._buildGround();
    await yield_();
    this._finalize();
  }

  update(dt, t) {
    this._windU.value = t;
  }

  resize() { }

  getSpawn(i) {
    const n = this.spawnPoints.length;
    if (!n) return null;
    return this.spawnPoints[((i | 0) % n + n) % n];
  }

  /** Cheap analytic floor height. Matches the ground mesh to within a cm. */
  sampleGroundY(x, z) {
    // W3 ground-floor room: a real screed floor lifted clear of the dirt
    if (x > -15.26 && x < -7.34 && z > -11.46 && z < -1.74) return INT_FY;
    if (x > -21.4 && x < -8.0 && z > 13.6 && z < 23.8) return HALL_DECK;  // market hall deck
    if (Math.abs(x) < ROAD_HW && z > -31 && z < 24.4) return this._hRoad(x);
    if (x > -7.02 && x < -5.58 && z > -31 && z < 24.4) return KERB_H;
    if (x > 5.58 && x < 7.02 && z > -31 && z < 24.4) return KERB_H;
    if (x > 5.30 && x < 6.62 && z > 23.90 && z < 38.70) return WALK_H;    // arcade walkway
    if (z > 38.30 && z < 39.72 && x > -21.0 && x < 5.62) return WALK_H;   // north walkway
    return 0;
  }

  /** road cross-section: cambered carriageway cut by a central drainage channel */
  _hRoad(x) {
    const ax = x < 0 ? -x : x;
    return 0.028 - 0.026 * (x / ROAD_HW) * (x / ROAD_HW)
      - 0.100 * (1 - smooth(0.20, CHAN_HW, ax));
  }

  /**
   * 1 where the compound is paved (carriageway + sidewalks + plaza apron),
   * 0 out in the loose dirt, with a soft shoulder between the two.
   */
  _hardMask(x, z) {
    const ax = x < 0 ? -x : x;
    // street corridor
    let m = (1 - smooth(ROAD_HW + 0.15, ROAD_HW + 1.85, ax))
      * smooth(-31.8, -30.3, z) * (1 - smooth(23.9, 25.2, z));
    // plaza apron + arcade forecourt
    const p = smooth(-21.9, -20.3, x) * (1 - smooth(5.0, 6.5, x))
      * smooth(22.9, 24.0, z) * (1 - smooth(38.8, 40.1, z));
    return m > p ? m : p;
  }

  /**
   * Undulating compound dirt floor — shared by the ground mesh and the dressing
   * pass. It is pushed hard below every paved surface: left to its own devices
   * the noise crests over the road and the apron and swallows the kerbs, the
   * drainage channel and all the low debris into one continuous sand plane.
   */
  _hCore(x, z) {
    let y = vnoise(x * 0.075, 7, z * 0.075) * 0.11 - 0.05;
    y += vnoise(x * 0.31, 11, z * 0.31) * 0.035;
    y -= 0.06 * Math.exp(-((x + 8) * (x + 8) + (z - 31) * (z - 31)) / 380);
    const m = this._hardMask(x, z);
    if (m > 0.001) y = y * (1 - m) - 0.115 * m;
    return y;
  }

  /** height of whatever surface a prop should stand on at (x,z) */
  _surfY(x, z, mode) {
    if (mode === 'core') return this._hCore(x, z);
    const s = this.sampleGroundY(x, z);
    return s !== 0 ? s : this._hCore(x, z);
  }

  // ------------------------------------------------------------------
  //  footprint reservations — keeps the dressing pass out of doorways,
  //  spawns, firing lanes and the hero props that were placed first.
  // ------------------------------------------------------------------

  _reserve(x, z, r) { this._occ.push(x, z, r); }

  /**
   * k scales the *stored* radii. Hero props reserve generously so they never
   * interpenetrate; the small floor dressing only needs to physically miss
   * them, so it queries with k < 1 and is allowed to nestle right up close.
   */
  _clearAt(x, z, r, k) {
    const s = k === undefined ? 1 : k;
    const o = this._occ;
    for (let i = 0; i < o.length; i += 3) {
      const dx = x - o[i], dz = z - o[i + 1], rr = r + o[i + 2] * s;
      if (dx * dx + dz * dz < rr * rr) return false;
    }
    return true;
  }

  /** never dress on top of a camera station or a player spawn */
  _onStation(x, z, r) {
    for (let i = 0; i < HERO_VIEWS.length; i++) {
      const dx = x - HERO_VIEWS[i].x, dz = z - HERO_VIEWS[i].z;
      const d = 1.5 + r;
      if (dx * dx + dz * dz < d * d) return true;
    }
    for (let i = 0; i < this.spawnPoints.length; i++) {
      const p = this.spawnPoints[i].pos;
      const dx = x - p.x, dz = z - p.z, d = 1.5 + r;
      if (dx * dx + dz * dz < d * d) return true;
    }
    return false;
  }

  /**
   * Authored prop positions are intent, not gospel — if the exact spot is
   * taken, walk a short golden-angle spiral for the nearest free one so the
   * dressing density actually lands instead of silently dropping out.
   */
  _findSpot(x, z, r, span, k) {
    if (this._ok(x, z, r, k)) return [x, z];
    const S = span || 2.3;
    for (let i = 1; i <= 40; i++) {
      const a = i * 2.399963, d = S * Math.sqrt(i / 40);
      const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
      if (this._ok(px, pz, r, k)) return [px, pz];
    }
    return null;
  }

  _ok(x, z, r, k) {
    if (!this._clearAt(x, z, r, k)) return false;
    return !(this._tallGuard && this._camBlocked(x, z, r));
  }

  /**
   * True when a tall silhouette at (x,z) would stand inside a hero camera's
   * wedge. This is the difference between "dressed" and "blindfolded": in a
   * MWII street frame the tall cover lives at the edges of the picture and the
   * centre lane stays legible all the way to the far landmark.
   */
  _camBlocked(x, z, r) {
    const R = r || 0;
    for (let i = 0; i < HERO_VIEWS.length; i++) {
      const V = HERO_VIEWS[i];
      const dx = x - V.x, dz = z - V.z;
      const fx = -Math.sin(V.yaw), fz = -Math.cos(V.yaw);
      const along = dx * fx + dz * fz;
      if (along < -1.8 || along > V.near) continue;
      const lat = Math.abs(dx * fz - dz * fx);
      const a = along > 0 ? along : 0;
      if (lat < WEDGE_BASE + a * WEDGE_SLOPE + R) return true;
    }
    return false;
  }

  // =====================================================================
  //  MATERIALS — every surface comes from game.materials
  // =====================================================================

  _mk(name, surface, ts, opts) {
    const mats = this.g.materials;
    let m;
    if (mats && typeof mats.make === 'function') {
      m = mats.make(surface, opts || {});
    } else {
      m = new THREE.MeshStandardMaterial({ color: (opts && opts.tint) || 0x9a8d78, roughness: 0.92 });
    }
    m.vertexColors = true;
    m.name = 'lvl:' + name;
    this._mats[name] = m;
    this._tsz[name] = ts;
    return m;
  }

  _buildMaterials() {
    // --- architecture ------------------------------------------------
    this._mk('plasterA', 'plaster', 2.55, { tint: 0xf3e6cd, roughness: 0.98 });
    this._mk('plasterB', 'plaster', 2.70, { tint: 0xd8c4a2, roughness: 1.0 });
    this._mk('plasterC', 'plaster', 2.40, { tint: 0xc4ab88, roughness: 1.0 });
    this._mk('plasterD', 'plaster', 2.85, { tint: 0xb9bcb4, roughness: 1.0 });
    this._mk('brickA', 'brick', 1.70, { tint: 0xe4c9a6 });
    this._mk('brickB', 'brick', 1.55, { tint: 0xc79a76 });
    this._mk('concrete', 'concrete', 2.90, { tint: 0xd8d0c2 });
    this._mk('concreteB', 'concrete', 2.10, { tint: 0xb7b0a4 });
    // Window voids / charred cabins. Kept very dark but never black: these are
    // shaded down again per-instance (shade 0.20-0.34), and the art direction
    // forbids a pure #000 anywhere in frame.
    this._mk('dark', 'concrete', 3.20, { tint: 0x2b2621, roughness: 1.0 });      // window voids
    this._mk('interior', 'plaster', 2.30, { tint: 0x8d8071 });

    // --- interior shell (own tiling scale so the room never shares a
    //     repeat period with the exterior stucco) --------------------------
    this._mk('intWall', 'plaster', 1.85, { tint: 0xc6b498, roughness: 1.0 });
    this._mk('intDado', 'plaster', 1.55, { tint: 0x7f8f8c, roughness: 1.0 });
    this._mk('intFloor', 'concrete', 2.35, { tint: 0xb9ad97 });
    this._mk('intScreed', 'plaster', 1.65, { tint: 0x9d8f79, roughness: 1.0 });
    this._mk('tile', 'concrete', 0.58, { tint: 0xa6a49a, roughness: 0.68 });
    this._mk('tileAlt', 'concrete', 0.61, { tint: 0x8d8b83, roughness: 0.66 });

    // --- ground ------------------------------------------------------
    this._mk('sand', 'sand', 4.20, { tint: 0xe8d6b4 });
    this._mk('asphalt', 'asphalt', 5.00, { tint: 0xbdb4a8 });
    this._mk('dirt', 'sand', 3.05, { tint: 0xcbb18d, roughness: 1.0 });     // wall-base spoil

    // --- metal / props -----------------------------------------------
    this._mk('metal', 'metal', 1.55, { tint: 0xbfc3c6, roughness: 0.85 });
    this._mk('rust', 'rustmetal', 1.95, { tint: 0xc7a184 });
    this._mk('rustDark', 'rustmetal', 1.45, { tint: 0x6f5a4a });
    // Charred sheet steel is soot, not a mirror. Left at the rustmetal default
    // (metalness 1) the wreck's flat door/bonnet panels took no diffuse at all
    // and read as pure black holes punched through the market frame. Soot is a
    // dielectric coating over the steel, so the metal term is pulled right down
    // and the tint lifted until the panels carry the key light again.
    this._mk('burnt', 'rustmetal', 1.70,
      { tint: 0x6e6053, roughness: 1.0, metalness: 0.22, envMapIntensity: 0.85 });
    this._mk('burlap', 'cloth_tan', 1.05, { tint: 0xd7c096, roughness: 1.0 });
    this._mk('gunmetal', 'gunmetal', 0.95, { tint: 0xa8a8ac });
    this._mk('wood', 'wood', 1.05, { tint: 0xd3b489 });
    this._mk('woodDark', 'wood', 0.95, { tint: 0x8a6a48 });
    this._mk('rubber', 'rubber', 0.85, { tint: 0x9a9691 });
    this._mk('polymer', 'polymer', 1.20, { tint: 0x4c5257 });
    this._mk('glass', 'glass', 2.0, { tint: 0x9fb2b8, opacity: 0.24, roughness: 0.35 });
    this._mk('foliage', 'foliage', 1.10, { tint: 0xbcc08a });

    // --- fabric ------------------------------------------------------
    this._mk('tarpRed', 'fabric', 1.35, { tint: 0xc06a4a, side: THREE.DoubleSide });
    this._mk('tarpBlue', 'fabric', 1.35, { tint: 0x6d8ba6, side: THREE.DoubleSide });
    this._mk('tarpTan', 'cloth_tan', 1.45, { tint: 0xe0cba4, side: THREE.DoubleSide });
    this._mk('tarpGreen', 'fabric', 1.30, { tint: 0x8a9464, side: THREE.DoubleSide });

    // --- signage -----------------------------------------------------
    this._mk('signBoard', 'metal', 1.1, { tint: 0x2e6f6a, roughness: 1.0 });
    this._mk('signInk', 'metal', 0.6, { tint: 0xe8e0cc, roughness: 1.0 });
    this._mk('signBoard2', 'metal', 1.1, { tint: 0xa03c2e, roughness: 1.0 });

    // --- wind-animated cloth (hanging laundry / loose tarps) ---------
    for (const [n, s, tint] of [
      ['clothA', 'cloth_tan', 0xf0e2c6], ['clothB', 'fabric', 0x8fa7bd],
      ['clothC', 'fabric', 0xc2705c], ['clothD', 'cloth_tan', 0xd8cbb0],
    ]) {
      const m = this._mk(n, s, 1.15, { tint, side: THREE.DoubleSide, roughness: 1.0 });
      this._windify(m);
    }
  }

  /** vertex-shader wind for hanging cloth; chains Materials' detail-normal patch */
  _windify(mat) {
    const prev = mat.onBeforeCompile;
    const uni = this._windU;
    mat.onBeforeCompile = function (shader, renderer) {
      if (typeof prev === 'function') prev.call(this, shader, renderer);
      shader.uniforms.uWindTime = uni;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>',
          '#include <common>\nuniform float uWindTime;\nattribute float aWind;')
        .replace('#include <begin_vertex>', `#include <begin_vertex>
  float wA = aWind;
  if ( wA > 0.001 ) {
    float wp = transformed.x * 1.7 + transformed.z * 2.3;
    float s1 = sin( uWindTime * 1.35 + wp );
    float s2 = sin( uWindTime * 2.70 + wp * 2.1 + 1.7 );
    transformed.x += ( s1 * 0.085 + s2 * 0.028 ) * wA;
    transformed.z += ( cos( uWindTime * 1.05 + wp * 0.8 ) * 0.075 ) * wA;
    transformed.y -= abs( s1 ) * 0.020 * wA;
  }`);
    };
    mat.customProgramCacheKey = () => 'bs-detail-1-wind';
    return mat;
  }

  // =====================================================================
  //  GEOMETRY COLLECTION
  // =====================================================================

  _zoneSet(z) { this._zone = z; return z; }

  /** bake vertex colours: contact grime low, sun-bleach high, blotchy breakup */
  _paint(geo, o) {
    const pos = geo.attributes.position;
    const n = pos.count;
    const arr = geo.attributes.color ? geo.attributes.color.array : new Float32Array(n * 3);
    const shade = (o && o.shade !== undefined) ? o.shade : 1;
    const base = (o && o.base !== undefined) ? o.base : 0;
    const grime = (o && o.grime !== undefined) ? o.grime : 1;
    const tint = o && o.tint;
    const tr = tint ? tint[0] : 1, tg = tint ? tint[1] : 1, tb = tint ? tint[2] : 1;
    const p = pos.array;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      const h = y - base;
      const g = Math.exp(-(h > 0 ? h : 0) * 1.30) * grime;
      let s = 1 - 0.40 * g;
      s *= 1 + 0.13 * smooth(3.0, 13.0, y);
      s *= 0.885 + 0.235 * vnoise(x * 0.19, y * 0.12, z * 0.19);
      s *= 0.94 + 0.12 * vnoise(x * 0.9 + 31.7, y * 0.9, z * 0.9 + 11.3);
      s *= shade;
      arr[i * 3] = clamp(s * tr * (1 + 0.055 * g), 0, 1.6);
      arr[i * 3 + 1] = clamp(s * tg, 0, 1.6);
      arr[i * 3 + 2] = clamp(s * tb * (1 - 0.16 * g), 0, 1.6);
    }
    if (!geo.attributes.color) geo.setAttribute('color', new THREE.BufferAttribute(arr, 3));
    else geo.attributes.color.needsUpdate = true;
    return geo;
  }

  _add(mat, geo, o) {
    const zone = (o && o.zone) || this._zone;
    const key = mat + '|' + zone;
    let b = this._buckets.get(key);
    if (!b) {
      b = {
        mat, zone, geos: [],
        collide: !NO_COLLIDE.has(zone) && !(o && o.noCollide),
        cast: !(o && o.noCast), recv: true,
      };
      this._buckets.set(key, b);
    }
    if (o && o.noCast) b.cast = false;
    if (!(o && o.raw)) this._paint(geo, o);
    else if (!geo.attributes.color) {
      const n = geo.attributes.position.count;
      const c = new Float32Array(n * 3); c.fill(1);
      geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
    }
    trimAttrs(ensureIndexed(geo));
    b.geos.push(geo);
    return geo;
  }

  _box(mat, cx, cy, cz, w, h, d, o) {
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const g = boxGeo(cx, cy, cz, w, h, d, ts, o && o.faces);
    if (o && o.rotY) rotYg(g, (o.pivotX !== undefined ? o.pivotX : cx), (o.pivotZ !== undefined ? o.pivotZ : cz), o.rotY);
    if (o && o.tilt) rotAxisG(g, cx, cy, cz, o.tilt[0], o.tilt[1], o.tilt[2], o.tilt[3]);
    return this._add(mat, g, o);
  }

  /** box spanning an explicit AABB */
  _bbox(mat, x0, y0, z0, x1, y1, z1, o) {
    return this._box(mat, (x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5,
      Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0), o);
  }

  _cyl(mat, cx, cy, cz, rt, rb, h, seg, o) {
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const open = !!(o && o.open);
    const g = new THREE.CylinderGeometry(rt, rb, h, seg || 12, 1, open);
    scaleUv(g, (Math.PI * (rt + rb)) / ts, h / ts);
    g.translate(cx, cy, cz);
    if (o && o.tilt) rotAxisG(g, cx, cy, cz, o.tilt[0], o.tilt[1], o.tilt[2], o.tilt[3]);
    if (o && o.rotY) rotYg(g, cx, cz, o.rotY);
    return this._add(mat, g, o);
  }

  _plane(mat, cx, cy, cz, w, h, o) {
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const g = new THREE.PlaneGeometry(w, h, (o && o.sx) || 1, (o && o.sy) || 1);
    scaleUv(g, w / ts, h / ts);
    if (o && o.flat) g.rotateX(-Math.PI * 0.5);
    g.translate(cx, cy, cz);
    if (o && o.rotY) rotYg(g, cx, cz, o.rotY);
    return this._add(mat, g, o);
  }

  /** irregular rock/rubble chunk */
  _chunk(mat, cx, cy, cz, r, o) {
    const g = new THREE.IcosahedronGeometry(r, 0);
    const p = g.attributes.position;
    const sx = 1 + this._rr(-0.35, 0.55), sy = 0.45 + this._r() * 0.5, sz = 1 + this._rr(-0.35, 0.55);
    for (let i = 0; i < p.count; i++) {
      const j = 1 + this._rr(-0.28, 0.28);
      p.setXYZ(i, p.getX(i) * sx * j, p.getY(i) * sy * j, p.getZ(i) * sz * j);
    }
    g.computeVertexNormals();
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const uv = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) {
      uv[i * 2] = (cx + p.getX(i)) / ts;
      uv[i * 2 + 1] = (cz + p.getZ(i) + cy + p.getY(i)) / ts;
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.translate(cx, cy, cz);
    if (o && o.rotY) rotYg(g, cx, cz, o.rotY);
    return this._add(mat, g, o);
  }

  // ------------------------------------------------------------------
  //  instancing
  // ------------------------------------------------------------------

  _pool(name, mat, geoFn, opts) {
    let p = this._instPools.get(name);
    if (!p) {
      const geo = geoFn();
      trimAttrs(ensureIndexed(geo));
      if (!geo.attributes.color) {
        const n = geo.attributes.position.count;
        const c = new Float32Array(n * 3); c.fill(1);
        geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
      }
      p = {
        name, mat, geo, items: [],
        cast: !(opts && opts.noCast), recv: true,
        collide: !!(opts && opts.collide),
      };
      this._instPools.set(name, p);
    }
    return p;
  }

  _instAdd(name, m4, shade) {
    const p = this._instPools.get(name);
    if (!p) return;
    p.items.push({ m: m4.clone(), s: shade === undefined ? 1 : shade });
  }

  // =====================================================================
  //  MODULAR BUILDING KIT
  //  Every hard edge gets trim: plinth, string course, cornice, quoins,
  //  window sills + lintels, parapet coping. A bare extruded box never
  //  appears anywhere in this level.
  // =====================================================================

  _faceFrame(b, face) {
    const T = b.t || T_WALL;
    if (face === 'E') return { ax: 'z', a0: b.z0, a1: b.z1, p: b.x1, sgn: -1, T };
    if (face === 'W') return { ax: 'z', a0: b.z0, a1: b.z1, p: b.x0, sgn: +1, T };
    if (face === 'N') return { ax: 'x', a0: b.x0 + T, a1: b.x1 - T, p: b.z1, sgn: -1, T };
    return { ax: 'x', a0: b.x0 + T, a1: b.x1 - T, p: b.z0, sgn: +1, T };
  }

  /** box in wall space: a = along-face centre, len = along-face length,
   *  yc/hh = vertical centre/height, d0/d1 = depth range measured from the
   *  outer plane (negative = proud of the wall, positive = into it). */
  _wput(mat, F, a, len, yc, hh, d0, d1, o) {
    const c = F.p + F.sgn * (d0 + d1) * 0.5;
    const th = Math.abs(d1 - d0);
    if (len <= 0.001 || hh <= 0.001 || th <= 0.001) return null;
    if (F.ax === 'z') return this._box(mat, c, yc, a, th, hh, len, o);
    return this._box(mat, a, yc, c, len, hh, th, o);
  }

  _windowUnit(b, F, a, w, ysill, yhead, o) {
    const T = F.T, M = b.mat, TR = b.trim || 'concrete';
    const h = yhead - ysill;
    // sill: proud lip with a drip, plus the bearing course inside the reveal
    this._wput(TR, F, a, w + 0.36, ysill - 0.06, 0.12, -0.15, 0.06);
    this._wput(TR, F, a, w + 0.10, ysill - 0.05, 0.10, 0.06, T);
    // lintel above the head
    this._wput(TR, F, a, w + 0.30, yhead + 0.09, 0.18, -0.09, 0.05);
    // dark cavity so the interior never reads as see-through (skipped on
    // genuinely open apertures — those are load-bearing light sources)
    if (!(o && o.open)) {
      this._wput('dark', F, a, w + 0.02, ysill + h * 0.5, h + 0.02, T - 0.10, T + 0.02, { noCast: true });
    }
    // frame set back into the reveal
    const fd = 0.20, fw = 0.065;
    this._wput('woodDark', F, a, w, ysill + fw * 0.5, fw, fd, fd + 0.05);
    this._wput('woodDark', F, a, w, yhead - fw * 0.5, fw, fd, fd + 0.05);
    this._wput('woodDark', F, a - w * 0.5 + fw * 0.5, fw, ysill + h * 0.5, h, fd, fd + 0.05);
    this._wput('woodDark', F, a + w * 0.5 - fw * 0.5, fw, ysill + h * 0.5, h, fd, fd + 0.05);
    if (h > 1.1) this._wput('woodDark', F, a, w, ysill + h * 0.52, 0.055, fd, fd + 0.04);
    // glazing (many are blown out)
    const rr = this._r();
    if (rr < (o && o.glass !== undefined ? o.glass : 0.45)) {
      this._wput('glass', F, a, w - 0.09, ysill + h * 0.5, h - 0.09, fd + 0.02, fd + 0.045, { noCast: true, noCollide: true, zone: 'detail' });
    }
    // shutters, hinged open against the wall
    if (this._r() < (o && o.shutters !== undefined ? o.shutters : 0.42)) {
      const sw = w * 0.5, ang = this._rr(1.05, 1.48);
      for (let s = -1; s <= 1; s += 2) {
        const hinge = a + s * (w * 0.5 + 0.03);
        const g = this._wput('woodDark', F, hinge + s * sw * 0.5, sw, ysill + h * 0.5, h - 0.04, -0.10, -0.04,
          { zone: 'detail', noCollide: true });
        if (g) {
          const hx = F.ax === 'z' ? F.p : hinge;
          const hz = F.ax === 'z' ? hinge : F.p;
          rotYg(g, hx, hz, (F.ax === 'z' ? F.sgn : -F.sgn) * s * ang);
        }
      }
    }
  }

  /** one storey of one façade: piers, spandrels, lintel band, window units */
  _facade(b, face, si) {
    const F = this._faceFrame(b, face);
    const cfg = b.win && b.win[face];
    const y0 = b.y + si * b.sh, y1 = y0 + b.sh;
    const span = F.a1 - F.a0;
    if (span <= 0.2) return;

    // ground floor: shopfront / doorway override the window grid
    if (si === 0 && b.shop === face) { this._shopfront(b, F, y0, y1); return; }
    if (si === 0 && b.door === face) { this._doorway(b, F, y0, y1); return; }

    const spec = Array.isArray(cfg) ? cfg[Math.min(si, cfg.length - 1)] : cfg;
    if (!spec || !spec.n) {
      this._wput(b.mat, F, (F.a0 + F.a1) * 0.5, span, (y0 + y1) * 0.5, b.sh, 0, F.T);
      return;
    }

    const n = spec.n;
    const w = spec.w || 1.22;
    const wh = spec.h || 1.58;
    const sill = y0 + (spec.sill !== undefined ? spec.sill : 0.95);
    const head = sill + wh;
    const pitch = span / n;
    const pier = pitch - w;
    if (pier < 0.5) {
      this._wput(b.mat, F, (F.a0 + F.a1) * 0.5, span, (y0 + y1) * 0.5, b.sh, 0, F.T);
      return;
    }

    // piers: half-width at the ends, full width between window centres
    for (let i = 0; i <= n; i++) {
      const cc = (i === 0) ? F.a0 + pier * 0.25
        : (i === n) ? F.a1 - pier * 0.25
          : F.a0 + i * pitch;
      const cw = (i === 0 || i === n) ? pier * 0.5 : pier;
      this._wput(b.mat, F, cc, cw + 0.02, (y0 + y1) * 0.5, b.sh, 0, F.T);
    }
    for (let i = 0; i < n; i++) {
      const a = F.a0 + pitch * (i + 0.5);
      this._wput(b.mat, F, a, w + 0.02, (y0 + sill) * 0.5, sill - y0, 0, F.T);        // spandrel
      this._wput(b.mat, F, a, w + 0.02, (head + y1) * 0.5, y1 - head, 0, F.T);        // lintel band
      this._windowUnit(b, F, a, w, sill, head, spec);
      if (spec.balcony && b.balconyFace === face) this._balcony(b, F, a, w, sill - 0.04);
      if (spec.ac && this._r() < 0.34) this._acUnit(b, F, a + w * 0.5 + 0.42, sill + 0.55);
    }
  }

  /** ground-floor wall with a real doorway punched through it */
  _doorway(b, F, y0, y1) {
    const T = F.T, TR = b.trim || 'concrete';
    const a = b.doorA !== undefined ? b.doorA : (F.a0 + F.a1) * 0.5;
    const w = b.doorW || 1.15;
    const head = y0 + DOOR_H;
    const l0 = F.a0, l1 = a - w * 0.5, r0 = a + w * 0.5, r1 = F.a1;
    if (l1 > l0) this._wput(b.mat, F, (l0 + l1) * 0.5, l1 - l0, (y0 + y1) * 0.5, b.sh, 0, T);
    if (r1 > r0) this._wput(b.mat, F, (r0 + r1) * 0.5, r1 - r0, (y0 + y1) * 0.5, b.sh, 0, T);
    this._wput(b.mat, F, a, w + 0.02, (head + y1) * 0.5 + 0.10, y1 - head - 0.20, 0, T);
    // lintel, jamb reveals, threshold
    this._wput(TR, F, a, w + 0.44, head + 0.10, 0.20, -0.10, T);
    this._wput(TR, F, a - w * 0.5 - 0.08, 0.16, y0 + DOOR_H * 0.5, DOOR_H, -0.08, 0.02);
    this._wput(TR, F, a + w * 0.5 + 0.08, 0.16, y0 + DOOR_H * 0.5, DOOR_H, -0.08, 0.02);
    this._wput(TR, F, a, w + 0.5, y0 + 0.05, 0.12, -0.34, T);
    this._addCover(F, a, 1.2, 0.2, 1.6);
  }

  _balcony(b, F, a, w, y) {
    const dep = 1.06, bw = w + 0.94;
    // slab + corbels
    this._wput('concrete', F, a, bw, y - 0.09, 0.18, -dep, 0.0);
    for (let s = -1; s <= 1; s++) {
      this._wput('concrete', F, a + s * (bw * 0.5 - 0.22), 0.20, y - 0.30, 0.26, -dep * 0.62, -0.02);
    }
    // solid balustrade + metal rail + balusters
    this._wput('concrete', F, a, bw, y + 0.24, 0.48, -dep, -dep + 0.13);
    this._wput('concrete', F, a - bw * 0.5 + 0.07, 0.14, y + 0.24, 0.48, -dep, 0.0);
    this._wput('concrete', F, a + bw * 0.5 - 0.07, 0.14, y + 0.24, 0.48, -dep, 0.0);
    this._wput('rust', F, a, bw, y + 0.92, 0.055, -dep - 0.02, -dep + 0.05, { zone: 'detail' });
    this._wput('rust', F, a, bw, y + 0.55, 0.035, -dep - 0.01, -dep + 0.04, { zone: 'detail' });
    const nb = Math.max(3, Math.round(bw / 0.24));
    for (let i = 0; i <= nb; i++) {
      const t = i / nb;
      this._wput('rust', F, a - bw * 0.5 + t * bw, 0.032, y + 0.70, 0.44, -dep - 0.005, -dep + 0.035,
        { zone: 'detail' });
    }
    this._addCover(F, a, y + 1.0, -dep * 0.5, 1.0);
  }

  _acUnit(b, F, a, y) {
    this._wput('metal', F, a, 0.78, y, 0.60, -0.62, -0.04, { zone: 'detail' });
    for (let i = 0; i < 5; i++) {
      this._wput('metal', F, a, 0.70, y - 0.22 + i * 0.11, 0.055, -0.665, -0.63, { zone: 'detail', shade: 0.72 });
    }
    this._wput('rust', F, a, 0.86, y - 0.36, 0.06, -0.66, -0.10, { zone: 'detail' });
    this._wput('rust', F, a - 0.30, 0.05, y - 0.55, 0.40, -0.30, -0.04, { zone: 'detail' });
    this._wput('rust', F, a + 0.30, 0.05, y - 0.55, 0.40, -0.30, -0.04, { zone: 'detail' });
  }

  _shopfront(b, F, y0, y1) {
    const T = F.T, span = F.a1 - F.a0;
    const n = Math.max(1, Math.round(span / 4.6));
    const pitch = span / n;
    const ow = Math.min(3.3, pitch - 1.25);
    const oh = 2.62;
    for (let i = 0; i <= n; i++) {
      const cc = (i === 0) ? F.a0 + (pitch - ow) * 0.25
        : (i === n) ? F.a1 - (pitch - ow) * 0.25 : F.a0 + i * pitch;
      const cw = (i === 0 || i === n) ? (pitch - ow) * 0.5 : (pitch - ow);
      this._wput(b.mat, F, cc, cw + 0.02, (y0 + y1) * 0.5, b.sh, 0, T);
    }
    for (let i = 0; i < n; i++) {
      const a = F.a0 + pitch * (i + 0.5);
      this._wput(b.mat, F, a, ow + 0.02, (y0 + oh + y1) * 0.5, y1 - y0 - oh, 0, T);
      this._wput('concrete', F, a, ow + 0.44, y0 + oh + 0.13, 0.26, -0.11, 0.05);   // head beam
      this._wput('dark', F, a, ow, y0 + oh * 0.5, oh, T - 0.06, T + 0.30, { noCast: true });
      const roll = this._r();
      if (roll < 0.42) {
        // corrugated roller shutter, part-open
        const sh = oh * this._rr(0.55, 1.0);
        this._wput('rust', F, a, ow - 0.06, y0 + oh - sh * 0.5, sh, 0.07, 0.13);
        const ribs = Math.round(sh / 0.16);
        for (let k = 0; k < ribs; k++) {
          this._wput('rust', F, a, ow - 0.09, y0 + oh - sh + 0.08 + k * 0.16, 0.055, 0.03, 0.075,
            { zone: 'detail', shade: 0.86 });
        }
      } else {
        // open shop: counter + goods behind
        this._wput('woodDark', F, a, ow - 0.30, y0 + 0.48, 0.96, T + 0.10, T + 0.48, { zone: 'detail' });
      }
      this._wput('concrete', F, a, ow + 0.5, y0 + 0.06, 0.13, -0.46, 0.0);           // step
      if (i % 2 === 0) this._awning(b, F, a, Math.min(ow + 0.9, pitch - 0.4), y0 + oh + 0.32);
      if (this._r() < 0.6) this._sign(F, a, y0 + oh + 0.62, Math.min(ow + 0.2, 2.9));
    }
  }

  _awning(b, F, a, w, y) {
    const dep = 1.28;
    const mats = ['tarpRed', 'tarpBlue', 'tarpGreen', 'tarpTan'];
    const m = mats[this._ri(0, 3)];
    // sloped canopy — a thin slab tilted down and away from the wall
    const g = this._wput(m, F, a, w, y - 0.20, 0.05, -dep, -0.02,
      { zone: 'detail', noCollide: true, ts: 1.3 });
    if (g) {
      const px = F.ax === 'z' ? F.p : a, pz = F.ax === 'z' ? a : F.p;
      const ang = 0.30 * (F.ax === 'z' ? (F.sgn > 0 ? 1 : -1) : (F.sgn > 0 ? -1 : 1));
      const ax = F.ax === 'z' ? [0, 0, 1] : [1, 0, 0];
      rotAxisG(g, px, y - 0.20, pz, ax[0], ax[1], ax[2], ang);
    }
    // scalloped valance + support arms
    this._wput(m, F, a, w, y - 0.60, 0.24, -dep - 0.02, -dep + 0.03, { zone: 'detail', noCollide: true });
    for (let s = -1; s <= 1; s += 2) {
      this._wput('rust', F, a + s * (w * 0.5 - 0.08), 0.05, y - 0.10, 0.05, -dep, 0.0, { zone: 'detail' });
      this._wput('rust', F, a + s * (w * 0.5 - 0.08), 0.05, y - 0.42, 0.62, -0.10, -0.04, { zone: 'detail' });
    }
  }

  /** painted board with abstract naskh-like script picked out in relief */
  _sign(F, a, y, w) {
    const board = this._r() < 0.5 ? 'signBoard' : 'signBoard2';
    const h = Math.min(0.62, w * 0.30);
    this._wput(board, F, a, w, y, h, -0.10, -0.03, { zone: 'sign', noCollide: true });
    this._wput('rust', F, a, w + 0.06, y + h * 0.5 + 0.03, 0.055, -0.12, -0.02, { zone: 'sign', noCollide: true });
    // strokes: a baseline with connected letterforms and diacritic dots
    let x = a + w * 0.5 - 0.14;
    const base = y - h * 0.16;
    let guard = 0;
    while (x > a - w * 0.5 + 0.12 && guard++ < 26) {
      const lw = this._rr(0.07, 0.20);
      const asc = this._r() < 0.3;
      this._wput('signInk', F, x - lw * 0.5, lw, base, 0.045, -0.125, -0.10, { zone: 'sign', noCollide: true, raw: true });
      if (asc) {
        this._wput('signInk', F, x - lw * 0.5, 0.05, base + h * 0.22, h * 0.44, -0.125, -0.10,
          { zone: 'sign', noCollide: true, raw: true });
      } else if (this._r() < 0.5) {
        this._wput('signInk', F, x - lw * 0.5, lw * 0.8, base + 0.09, 0.04, -0.125, -0.10,
          { zone: 'sign', noCollide: true, raw: true });
      }
      if (this._r() < 0.35) {
        this._wput('signInk', F, x - lw * 0.5, 0.04, base - 0.09, 0.04, -0.125, -0.10,
          { zone: 'sign', noCollide: true, raw: true });
      }
      x -= lw + 0.035;
    }
  }

  _quoins(b, top) {
    const TR = b.trim || 'concrete';
    const corners = [[b.x0, b.z0], [b.x1, b.z0], [b.x0, b.z1], [b.x1, b.z1]];
    for (const [cx, cz] of corners) {
      const sx = cx < (b.x0 + b.x1) * 0.5 ? 1 : -1;
      const sz = cz < (b.z0 + b.z1) * 0.5 ? 1 : -1;
      let y = b.y + 0.5, k = 0;
      while (y < top - 0.5) {
        const long = (k & 1) === 0;
        const lx = long ? 0.62 : 0.40, lz = long ? 0.40 : 0.62;
        this._box(TR, cx + sx * lx * 0.5 - sx * 0.03, y + 0.21, cz + sz * lz * 0.5 - sz * 0.03,
          lx + 0.06, 0.42, lz + 0.06, { shade: 1.02 });
        y += 0.48; k++;
      }
    }
  }

  _parapet(b, top) {
    const T = 0.24, ph = b.parapet || 0.95;
    const sides = [
      { ax: 'x', a0: b.x0, a1: b.x1, p: b.z1, sgn: -1 },
      { ax: 'x', a0: b.x0, a1: b.x1, p: b.z0, sgn: +1 },
      { ax: 'z', a0: b.z0, a1: b.z1, p: b.x1, sgn: -1 },
      { ax: 'z', a0: b.z0, a1: b.z1, p: b.x0, sgn: +1 },
    ];
    for (let s = 0; s < sides.length; s++) {
      if (b.noParapet && b.noParapet.indexOf('NSEW'[s]) >= 0) continue;
      const S = sides[s];
      const span = S.a1 - S.a0;
      const segs = Math.max(2, Math.round(span / 1.15));
      const step = span / segs;
      for (let i = 0; i < segs; i++) {
        const a = S.a0 + step * (i + 0.5);
        let h = ph;
        const dmg = b.broken ? this._r() : 1;
        if (dmg < 0.20) continue;
        if (dmg < 0.42) h = ph * this._rr(0.30, 0.72);
        const cy = top + h * 0.5;
        const c = S.p + S.sgn * T * 0.5;
        if (S.ax === 'x') {
          this._box(b.mat, a, cy, c, step + 0.01, h, T, { base: top });
          if (h > ph * 0.9) this._box(b.trim || 'concrete', a, top + h + 0.045, c, step + 0.02, 0.09, T + 0.11, { base: top });
        } else {
          this._box(b.mat, c, cy, a, T, h, step + 0.01, { base: top });
          if (h > ph * 0.9) this._box(b.trim || 'concrete', c, top + h + 0.045, a, T + 0.11, 0.09, step + 0.02, { base: top });
        }
        if (h < ph * 0.8) this._rebar(S.ax === 'x' ? a : c, top + h, S.ax === 'x' ? c : a, 2);
      }
    }
  }

  /** exposed reinforcement bars bent out of a broken concrete edge */
  _rebar(x, y, z, n) {
    for (let i = 0; i < n; i++) {
      const px = x + this._rr(-0.09, 0.09), pz = z + this._rr(-0.09, 0.09);
      const g = this._cyl('rust', px, y + 0.17, pz, 0.011, 0.013, 0.36, 5,
        { zone: 'detail', noCollide: true, noCast: true, ts: 0.5, raw: true });
      rotAxisG(g, px, y, pz, this._rr(-1, 1), 0, this._rr(-1, 1), this._rr(0.2, 0.85));
    }
  }

  // =====================================================================
  //  FAÇADE DRESSING PASS
  //  A CoD elevation is never a tiled plane with holes punched in it. It is a
  //  base course, a string course with a drip mould, pilasters, dentils under
  //  the cornice, sill drips, patched render showing the block underneath,
  //  downpipes and their stains, conduit and meter boxes, condensers, dishes,
  //  awnings, hanging cloth and painted signage. Everything below is thin,
  //  merges into the 'trim' bucket and never collides, so nothing here can
  //  trap the player or confuse Physics.
  // =====================================================================

  /** a dark weathering wash painted straight onto the wall plane */
  _wash(mat, F, a, w, y0, y1, shade) {
    if (y1 - y0 < 0.05) return;
    this._wput(mat, F, a, w, (y0 + y1) * 0.5, y1 - y0, -0.016, -0.005,
      { zone: 'trim', noCast: true, shade, grime: 0, base: y0 });
  }

  /** outward world normal + face-centre of a wall frame */
  _faceOut(F) {
    if (F.ax === 'z') return [-F.sgn, 0, F.p, (F.a0 + F.a1) * 0.5];
    return [0, -F.sgn, (F.a0 + F.a1) * 0.5, F.p];
  }

  /** cast-iron downpipe: hopper head, square stack, brackets, swan neck, stain */
  _downpipe(b, F, a, y0, y1) {
    const dp = -0.21;
    this._wput('rust', F, a, 0.135, (y0 + y1) * 0.5, y1 - y0, dp, dp + 0.135,
      { zone: 'trim', base: b.y, grime: 1.2 });
    this._wput('rust', F, a, 0.36, y1 - 0.15, 0.32, dp - 0.07, dp + 0.16,
      { zone: 'trim', base: b.y, shade: 1.04 });
    for (let y = y0 + 0.85; y < y1 - 0.5; y += 1.62) {
      this._wput('rustDark', F, a, 0.27, y, 0.055, dp - 0.025, 0.0,
        { zone: 'trim', base: b.y, shade: 0.88 });
    }
    this._wput('rust', F, a, 0.155, y0 + 0.20, 0.36, dp - 0.11, dp + 0.06,
      { zone: 'trim', base: b.y, grime: 1.5 });
    this._wash(b.mat, F, a, 0.46, y0, y1 - 0.4, 0.74);
    this._wash(b.mat, F, a, 0.86, y0, y0 + 1.5, 0.70);
  }

  /** surface conduit: horizontal run, drop leg, meter box, clips */
  _conduit(b, F, y, holes) {
    const span = F.a1 - F.a0, mid = (F.a0 + F.a1) * 0.5;
    if (span < 2.2) return;
    this._wput('rustDark', F, mid, span * 0.88, y, 0.05, -0.095, -0.045,
      { zone: 'trim', base: b.y, shade: 0.9 });
    for (let t = 0.10; t < 0.92; t += 0.155) {
      this._wput('rustDark', F, F.a0 + span * t, 0.065, y, 0.10, -0.105, -0.02,
        { zone: 'trim', base: b.y, shade: 0.84 });
    }
    let a = null;
    for (let i = 0; i < 10 && a === null; i++) {
      const c = F.a0 + span * this._drr(0.16, 0.84);
      if (!holes || this._wallClear(holes, c, y - 0.9, 0.6, 1.9)) a = c;
    }
    if (a === null) return;
    this._wput('rustDark', F, a, 0.05, y - 0.85, 1.70, -0.09, -0.045,
      { zone: 'trim', base: b.y, shade: 0.9 });
    this._wput('polymer', F, a, 0.36, y - 1.72, 0.46, -0.17, -0.02, { zone: 'trim', base: b.y });
    this._wput('metal', F, a, 0.30, y - 1.72, 0.40, -0.185, -0.16,
      { zone: 'trim', base: b.y, shade: 1.06 });
    this._wash(b.mat, F, a, 0.5, b.y + 0.5, y - 1.9, 0.80);
  }

  /** rectangle of render fallen away, blockwork showing through */
  _renderPatch(b, F, a, yc, w, h) {
    const bm = this._dr() < 0.5 ? 'brickA' : 'brickB';
    this._wput(bm, F, a, w, yc, h, -0.022, -0.004,
      { zone: 'trim', noCast: true, base: b.y, shade: 0.93, grime: 1.2 });
    // a second overlapping rectangle so the silhouette is never a clean box
    this._wput(bm, F, a + this._drr(-0.35, 0.35) * w, yc + this._drr(-0.4, 0.4) * h,
      w * this._drr(0.35, 0.75), h * this._drr(0.4, 0.8), -0.022, -0.004,
      { zone: 'trim', noCast: true, base: b.y, shade: 0.90, grime: 1.2 });
    // ragged bead of surviving render around it
    const t = 0.055;
    this._wput(b.mat, F, a, w + t * 2, yc + h * 0.5, t, -0.05, -0.02, { zone: 'trim', base: b.y, shade: 1.05 });
    this._wput(b.mat, F, a, w + t * 2, yc - h * 0.5, t, -0.05, -0.02, { zone: 'trim', base: b.y, shade: 0.96 });
    this._wput(b.mat, F, a - w * 0.5, t, yc, h, -0.05, -0.02, { zone: 'trim', base: b.y, shade: 1.02 });
    this._wput(b.mat, F, a + w * 0.5, t, yc, h, -0.05, -0.02, { zone: 'trim', base: b.y, shade: 0.99 });
    this._wash(b.mat, F, a, w * 0.8, yc - h * 0.5 - 1.1, yc - h * 0.5, 0.80);
  }

  /** wall-mounted satellite dish on a stand-off bracket */
  _wallDish(F, a, y) {
    const [nx, nz, wx, wz] = this._faceOut(F);
    const px = F.ax === 'z' ? F.p + nx * 0.55 : a;
    const pz = F.ax === 'z' ? a : F.p + nz * 0.55;
    void wx; void wz;
    this._wput('rustDark', F, a, 0.10, y, 0.10, -0.56, -0.02, { zone: 'trim', shade: 0.85 });
    this._wput('rustDark', F, a, 0.26, y, 0.26, -0.10, -0.02, { zone: 'trim', shade: 0.9 });
    this._dish(px, y, pz, Math.atan2(nx, nz) + Math.PI + this._drr(-0.5, 0.5));
  }

  /** louvred wall vent recess */
  _vent(b, F, a, y) {
    this._wput('dark', F, a, 0.52, y, 0.40, 0.02, 0.09,
      { zone: 'trim', raw: true, noCast: true, shade: 0.28 });
    for (let i = 0; i < 4; i++) {
      this._wput('rustDark', F, a, 0.50, y - 0.14 + i * 0.095, 0.05, -0.02, 0.02,
        { zone: 'trim', base: b.y, shade: 0.82 });
    }
    const TR = b.trim || 'concrete';
    this._wput(TR, F, a, 0.68, y + 0.245, 0.075, -0.05, -0.01, { zone: 'trim', base: b.y, shade: 1.05 });
    this._wput(TR, F, a, 0.68, y - 0.245, 0.075, -0.05, -0.01, { zone: 'trim', base: b.y, shade: 1.0 });
    this._wput(TR, F, a - 0.30, 0.075, y, 0.50, -0.05, -0.01, { zone: 'trim', base: b.y, shade: 1.02 });
    this._wput(TR, F, a + 0.30, 0.075, y, 0.50, -0.05, -0.01, { zone: 'trim', base: b.y, shade: 1.02 });
    this._wash(b.mat, F, a, 0.5, y - 1.3, y - 0.2, 0.82);
  }

  /** iron balcony / window guard grille — pure silhouette, very cheap */
  _windowGuard(F, a, w, y0, y1) {
    const n = Math.max(3, Math.round(w / 0.20));
    for (let i = 0; i <= n; i++) {
      this._wput('rustDark', F, a - w * 0.5 + (i / n) * w, 0.028, (y0 + y1) * 0.5, y1 - y0,
        -0.115, -0.085, { zone: 'trim', shade: 0.8 });
    }
    for (const y of [y0 + 0.03, y1 - 0.03, (y0 + y1) * 0.5]) {
      this._wput('rustDark', F, a, w, y, 0.030, -0.118, -0.082, { zone: 'trim', shade: 0.86 });
    }
  }

  /** clothes-line strung from a window head across the elevation */
  _faceCloth(F, a, y, w, h) {
    const [nx, nz] = this._faceOut(F);
    const px = F.ax === 'z' ? F.p + nx * 0.22 : a;
    const pz = F.ax === 'z' ? a : F.p + nz * 0.22;
    this._clothSheet(px, y, pz, w, h, F.ax === 'z' ? Math.PI * 0.5 : 0);
  }

  /**
   * Projecting timber screen box (mashrabiya). The strongest single silhouette
   * an elevation can carry: 0.9 m of overhang throwing a hard shadow down the
   * wall, with a latticed face and a dark void behind it.
   */
  _oriel(b, F, a, w, y0, h) {
    const TR = b.trim || 'concrete';
    const dep = -0.88;
    for (const s of [-1, 0, 1]) {
      this._wput(TR, F, a + s * (w * 0.42), 0.19, y0 - 0.30, 0.38, dep * 0.60, -0.02,
        { zone: 'trim', base: b.y, shade: 1.02 });
    }
    this._wput(TR, F, a, w + 0.30, y0 - 0.06, 0.16, dep - 0.07, 0.0,
      { zone: 'trim', base: b.y, shade: 1.06 });
    this._wput('woodDark', F, a, w, y0 + 0.32, 0.60, dep, dep + 0.10, { zone: 'trim', base: b.y });
    for (const s of [-1, 1]) {
      this._wput('woodDark', F, a + s * (w * 0.5 - 0.05), 0.10, y0 + h * 0.5, h, dep, 0.0,
        { zone: 'trim', base: b.y });
    }
    this._wput('dark', F, a, w - 0.14, y0 + h * 0.60, h * 0.56, dep + 0.10, dep + 0.16,
      { zone: 'trim', raw: true, noCast: true, shade: 0.22 });
    const nb = Math.max(4, Math.round(w / 0.17));
    for (let i = 0; i <= nb; i++) {
      this._wput('woodDark', F, a - w * 0.5 + (i / nb) * w, 0.038, y0 + h * 0.60, h * 0.56,
        dep - 0.012, dep + 0.055, { zone: 'trim', base: b.y, shade: 0.92 });
    }
    for (let k = 0; k < 3; k++) {
      this._wput('woodDark', F, a, w, y0 + h * 0.34 + k * h * 0.26, 0.042, dep - 0.014, dep + 0.055,
        { zone: 'trim', base: b.y, shade: 0.95 });
    }
    this._wput('woodDark', F, a, w + 0.14, y0 + h - 0.10, 0.22, dep - 0.05, 0.0,
      { zone: 'trim', base: b.y });
    this._wput(TR, F, a, w + 0.38, y0 + h + 0.08, 0.14, dep - 0.16, 0.0,
      { zone: 'trim', base: b.y, shade: 1.08 });
    this._wash(b.mat, F, a, w * 0.9, y0 - 1.7, y0 - 0.22, 0.76);
  }

  /** rooftop TV aerial — pure silhouette against the sky, near-free */
  _roofAerial(x, y, z, h, ry) {
    this._cyl('rust', x, y + h * 0.5, z, 0.030, 0.046, h, 6,
      { base: y, zone: 'trim', ts: 0.6, shade: 0.9 });
    const n = this._dri(4, 7);
    for (let i = 0; i < n; i++) {
      const yy = y + h * (0.42 + 0.56 * (i / n));
      const len = 1.05 - 0.10 * i;
      this._box('rust', x, yy, z, len, 0.028, 0.028,
        { base: y, zone: 'trim', rotY: ry, pivotX: x, pivotZ: z, ts: 0.5, shade: 0.9 });
    }
    this._box('rust', x, y + h * 0.40, z, 0.09, 0.09, 0.60,
      { base: y, zone: 'trim', rotY: ry, pivotX: x, pivotZ: z, shade: 0.88 });
  }

  /** roofline silhouette kit — aerials, a flagpole, a leaning conduit mast */
  _roofSkyline(b) {
    const w = b.x1 - b.x0, d = b.z1 - b.z0;
    if (w < 4 || d < 4) return;
    const n = this._dri(1, 3);
    for (let i = 0; i < n; i++) {
      const x = b.x0 + 0.7 + this._dr() * (w - 1.4);
      const z = b.z0 + 0.7 + this._dr() * (d - 1.4);
      this._roofAerial(x, b.top + (b.parapet || 0.95) * 0.4, z, this._drr(2.1, 3.9), this._drr(0, 3.1));
    }
    if (this._dr() < 0.5) {
      const x = b.x0 + 0.6 + this._dr() * (w - 1.2);
      const z = b.z0 + 0.6 + this._dr() * (d - 1.2);
      const hh = this._drr(3.2, 5.0);
      this._cyl('rust', x, b.top + hh * 0.5, z, 0.030, 0.055, hh, 6,
        { base: b.top, zone: 'trim', ts: 0.7, tilt: [this._drr(-1, 1), 0, this._drr(-1, 1), this._drr(0.03, 0.11)] });
      this._clothSheet(x + 0.32, b.top + hh - 0.16, z, this._drr(0.6, 0.95), this._drr(0.5, 0.8),
        this._drr(0, 3.1));
    }
  }

  /** aperture rectangles (a0,a1,y0,y1) on one elevation — trim must dodge them */
  _winRects(b, face, F) {
    const out = [];
    const cfg = b.win && b.win[face];
    const span = F.a1 - F.a0;
    for (let si = 0; si < b.st; si++) {
      const y0 = b.y + si * b.sh;
      if (si === 0 && (b.shop === face || b.door === face)) {
        out.push([F.a0 - 0.2, F.a1 + 0.2, y0 - 0.2, y0 + 3.05]);
        continue;
      }
      const sc = Array.isArray(cfg) ? cfg[Math.min(si, cfg.length - 1)] : cfg;
      if (!sc || !sc.n) continue;
      const pitch = span / sc.n, w = sc.w || 1.22;
      const s = y0 + (sc.sill !== undefined ? sc.sill : 0.95);
      const h = sc.h || 1.58;
      for (let i = 0; i < sc.n; i++) {
        const a = F.a0 + pitch * (i + 0.5);
        out.push([a - w * 0.5 - 0.30, a + w * 0.5 + 0.30, s - 0.28, s + h + 0.32]);
      }
    }
    return out;
  }

  _wallClear(rects, a, yc, w, h) {
    const a0 = a - w * 0.5, a1 = a + w * 0.5, y0 = yc - h * 0.5, y1 = yc + h * 0.5;
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      if (a1 > r[0] && a0 < r[1] && y1 > r[2] && y0 < r[3]) return false;
    }
    return true;
  }

  /**
   * Dress ONE elevation. `heavy` is set for the elevations the play space
   * actually looks at; the backs get the mouldings but not the ironmongery,
   * so the trim budget goes where the camera is.
   */
  _faceDress(b, face, style, heavy, lod) {
    const F = this._faceFrame(b, face);
    const span = F.a1 - F.a0;
    if (span < 2.4) return;
    const TR = b.trim || 'concrete';
    const top = b.top, mid = (F.a0 + F.a1) * 0.5;
    const isShop = b.shop === face, isDoor = b.door === face;

    // --------------------------------------------------------------------
    // LOD 1 — the outskirt row. It fills half the vista frame so it cannot be
    // a bare plane, but at 30-55 m only the big forms survive the haze: base
    // course, storey drip mould, cornice, pilasters and tonal patchwork.
    // --------------------------------------------------------------------
    if (lod) {
      this._wput(TR, F, mid, span + 0.22, b.y + 0.68, 0.46, -0.17, 0.02,
        { zone: 'trim', base: b.y, grime: 1.5 });
      this._wput(TR, F, mid, span + 0.28, b.y + 0.95, 0.11, -0.24, 0.02,
        { zone: 'trim', base: b.y, shade: 1.08 });
      for (let si = 1; si < b.st; si++) {
        const y = b.y + si * b.sh;
        this._wput(TR, F, mid, span + 0.30, y - 0.22, 0.17, -0.34, -0.05,
          { zone: 'trim', base: b.y, shade: 1.05 });
        this._wput(TR, F, mid, span + 0.16, y + 0.20, 0.09, -0.22, -0.04,
          { zone: 'trim', base: b.y, shade: 1.09 });
      }
      this._wput(TR, F, mid, span + 0.24, top - 0.70, 0.26, -0.32, -0.05,
        { zone: 'trim', base: b.y, shade: 1.07 });
      const lcfg = b.win && b.win[face];
      const lspec = Array.isArray(lcfg) ? (lcfg[lcfg.length - 1] || lcfg[0]) : lcfg;
      if (lspec && lspec.n >= 1) {
        const pitch = span / lspec.n, ww = lspec.w || 1.1, pier = pitch - ww;
        if (lspec.n >= 2 && pier > 0.70) {
          for (let i = 1; i < lspec.n; i++) {
            const a = F.a0 + i * pitch;
            const py0 = b.y + 1.0, py1 = top - 1.0;
            this._wput(b.mat, F, a, Math.min(0.56, pier - 0.20), (py0 + py1) * 0.5, py1 - py0,
              -0.17, 0.0, { zone: 'trim', base: b.y });
          }
        }
        // projecting sill + lintel on every opening: without these the windows
        // read as holes punched in a plane rather than as openings in a wall
        const wh = lspec.h || 1.4;
        for (let si = 0; si < b.st; si++) {
          const sy = b.y + si * b.sh + (lspec.sill !== undefined ? lspec.sill : 1.0);
          for (let i = 0; i < lspec.n; i++) {
            const a = F.a0 + pitch * (i + 0.5);
            this._wput(TR, F, a, ww + 0.40, sy - 0.10, 0.16, -0.26, 0.02,
              { zone: 'trim', base: b.y, shade: 1.07 });
            this._wput(TR, F, a, ww + 0.34, sy + wh + 0.13, 0.20, -0.24, 0.02,
              { zone: 'trim', base: b.y, shade: 1.05 });
            this._wash(b.mat, F, a, ww * 0.85, sy - 0.95, sy - 0.16, 0.70);
          }
        }
      }
      const T2 = ['plasterA', 'plasterB', 'plasterC', 'plasterD', 'concreteB'];
      for (let i = 0, n = this._dri(2, 4); i < n; i++) {
        const pw = this._drr(1.6, 4.6), ph = this._drr(1.2, 3.0);
        if (span < pw * 1.15) continue;
        let tone = T2[this._dri(0, 4)];
        if (tone === b.mat) tone = T2[(this._dri(0, 4) + 1) % 5];
        this._wput(tone, F, this._drr(F.a0 + pw * 0.45, F.a1 - pw * 0.45), pw,
          this._drr(b.y + 1.2, Math.max(b.y + 1.6, top - 1.2)), ph, -0.010, -0.001,
          { zone: 'trim', noCast: true, base: b.y, shade: this._drr(0.88, 1.06), grime: 0.7 });
      }
      for (let i = 0, n = this._dri(1, 3); i < n; i++) {
        const pw = this._drr(0.7, 1.9), ph = this._drr(0.6, 1.5);
        if (span < pw * 1.8) continue;
        this._renderPatch(b, F, this._drr(F.a0 + pw * 0.6, F.a1 - pw * 0.6),
          this._drr(b.y + 1.2, Math.max(b.y + 1.6, top - 1.2)), pw, ph);
      }
      for (let i = 0, n = this._dri(3, 6); i < n; i++) {
        this._wash(b.mat, F, this._drr(F.a0 + 0.3, F.a1 - 0.3), this._drr(0.18, 0.6),
          top - this._drr(1.4, 3.4), top - 0.30, this._drr(0.54, 0.74));
      }
      if (this._dr() < 0.5) this._downpipe(b, F, F.a0 + span * this._drr(0.06, 0.18), b.y + 0.05, top - 0.4);
      return;
    }

    // ---- base course: two stepped bands standing on the plinth -----------
    this._wput(TR, F, mid, span + 0.22, b.y + 0.66, 0.42, -0.17, 0.02,
      { zone: 'trim', base: b.y, grime: 1.55 });
    this._wput(TR, F, mid, span + 0.28, b.y + 0.92, 0.10, -0.23, 0.02,
      { zone: 'trim', base: b.y, shade: 1.08 });
    this._wash(b.mat, F, mid, span * 0.97, b.y + 0.98, b.y + 1.9, 0.86);

    // ---- rendered dado in a second tone: a big tonal block reads at 40 m --
    if ((style & 1) && !isShop && !isDoor) {
      this._wput(style & 4 ? 'plasterD' : 'concreteB', F, mid, span + 0.06, b.y + 1.55, 1.16,
        -0.055, -0.005, { zone: 'trim', base: b.y, grime: 1.35, shade: 0.93 });
      this._wput(TR, F, mid, span + 0.14, b.y + 2.16, 0.075, -0.13, -0.01,
        { zone: 'trim', base: b.y, shade: 1.06 });
    }

    // ---- string courses: a real drip mould at every storey line ----------
    for (let si = 1; si < b.st; si++) {
      const y = b.y + si * b.sh;
      this._wput(TR, F, mid, span + 0.30, y - 0.22, 0.16, -0.34, -0.05,
        { zone: 'trim', base: b.y, shade: 1.05 });
      this._wput(TR, F, mid, span + 0.20, y + 0.19, 0.09, -0.24, -0.04,
        { zone: 'trim', base: b.y, shade: 1.09 });
      this._wput(TR, F, mid, span + 0.10, y - 0.50, 0.07, -0.18, -0.03,
        { zone: 'trim', base: b.y, shade: 1.02 });
    }

    // ---- dentil / modillion band under the cornice -----------------------
    const dn = Math.max(4, Math.round(span / 0.74));
    const dw = span / dn;
    for (let i = 0; i < dn; i++) {
      this._wput(TR, F, F.a0 + (i + 0.5) * dw, dw * 0.44, top - 0.66, 0.24, -0.36, -0.06,
        { zone: 'trim', base: b.y, shade: 1.07 });
    }
    this._wput(TR, F, mid, span + 0.16, top - 0.90, 0.10, -0.26, -0.04,
      { zone: 'trim', base: b.y, shade: 1.03 });

    // ---- pilasters between the window bays -------------------------------
    // Verticals are the single biggest thing a bare elevation is missing:
    // they cast a full-height shadow and give the bay rhythm a hard edge.
    const cfg = b.win && b.win[face];
    const spec = Array.isArray(cfg) ? (cfg[cfg.length - 1] || cfg[0]) : cfg;
    const nw = spec && spec.n ? spec.n : 0;
    if (nw >= 2) {
      const pitch = span / nw, pier = pitch - (spec.w || 1.22);
      if (pier > 0.70) {
        for (let i = 1; i < nw; i++) {
          const a = F.a0 + i * pitch;
          const py0 = b.y + 0.98, py1 = top - 0.98;
          const pw = Math.min(0.58, pier - 0.20);
          this._wput(b.mat, F, a, pw, (py0 + py1) * 0.5, py1 - py0, -0.17, 0.0,
            { zone: 'trim', base: b.y });
          this._wput(TR, F, a, pw + 0.18, py1 + 0.12, 0.24, -0.26, 0.0,
            { zone: 'trim', base: b.y, shade: 1.09 });
          this._wput(TR, F, a, pw + 0.18, py0 - 0.11, 0.22, -0.26, 0.0,
            { zone: 'trim', base: b.y, shade: 1.04 });
        }
      }
    }

    // ---- sill drips under every window ----------------------------------
    if (nw >= 1 && spec) {
      const pitch = span / nw;
      const ww = spec.w || 1.22;
      for (let si = (isShop || isDoor) ? 1 : 0; si < b.st; si++) {
        const sc = Array.isArray(cfg) ? cfg[Math.min(si, cfg.length - 1)] : cfg;
        if (!sc || !sc.n) continue;
        const sy = b.y + si * b.sh + (sc.sill !== undefined ? sc.sill : 0.95);
        for (let i = 0; i < nw; i++) {
          const a = F.a0 + pitch * (i + 0.5);
          this._wash(b.mat, F, a - ww * 0.5 - 0.08, 0.17, sy - 1.25, sy - 0.12, 0.66);
          this._wash(b.mat, F, a + ww * 0.5 + 0.08, 0.17, sy - 1.05, sy - 0.12, 0.68);
          this._wash(b.mat, F, a, ww * 0.8, sy - 0.62, sy - 0.14, 0.80);
        }
      }
    }

    // ---- large-scale render tone patches --------------------------------
    // Big soft blocks of a different render batch. Nothing else on the
    // elevation breaks a 4 m span of wall, and at 30 m these are what stop it
    // reading as one poured plane.
    const TONES = ['plasterA', 'plasterB', 'plasterC', 'plasterD', 'concreteB'];
    for (let i = 0, n = this._dri(3, 6); i < n; i++) {
      const pw = this._drr(1.8, 5.2), ph = this._drr(1.2, 3.4);
      if (span < pw * 1.15) continue;
      const a = this._drr(F.a0 + pw * 0.45, F.a1 - pw * 0.45);
      const yc = this._drr(b.y + 1.2, Math.max(b.y + 1.6, top - 1.2));
      let tone = TONES[this._dri(0, 4)];
      if (tone === b.mat) tone = TONES[(this._dri(0, 4) + 1) % 5];
      this._wput(tone, F, a, pw, yc, ph, -0.010, -0.001,
        { zone: 'trim', noCast: true, base: b.y, shade: this._drr(0.90, 1.06), grime: 0.7 });
    }

    // ---- patched render --------------------------------------------------
    const holes = this._winRects(b, face, F);
    const np = heavy ? this._dri(4, 8) : this._dri(2, 4);
    for (let i = 0; i < np; i++) {
      let placed = false;
      for (let a2 = 0; a2 < 8 && !placed; a2++) {
        const pw = this._drr(0.55, 1.90), ph = this._drr(0.45, 1.55);
        if (span < pw * 1.8) continue;
        const a = this._drr(F.a0 + pw * 0.6, F.a1 - pw * 0.6);
        const yc = this._drr(b.y + 1.0, Math.max(b.y + 1.4, top - 1.1));
        if (!this._wallClear(holes, a, yc, pw + 0.16, ph + 0.16)) continue;
        this._renderPatch(b, F, a, yc, pw, ph);
        placed = true;
      }
    }

    if (!heavy) return;

    // ---- downpipes -------------------------------------------------------
    const dps = span > 9 ? 2 : 1;
    for (let i = 0; i < dps; i++) {
      const a = F.a0 + span * (dps === 1 ? this._drr(0.06, 0.16) : (i ? 0.90 : 0.09));
      this._downpipe(b, F, a, b.y + 0.05, top - 0.34);
    }

    // ---- conduit + meter, tucked under the first string course -----------
    if (this._dr() < 0.85) this._conduit(b, F, b.y + b.sh - this._drr(0.38, 0.50), holes);

    // ---- condensers, dishes, vents, guards, awnings, cloth ---------------
    if (nw >= 1 && spec) {
      const pitch = span / nw, ww = spec.w || 1.22;
      for (let si = 1; si < b.st; si++) {
        const sc = Array.isArray(cfg) ? cfg[Math.min(si, cfg.length - 1)] : cfg;
        if (!sc || !sc.n) continue;
        const sy = b.y + si * b.sh + (sc.sill !== undefined ? sc.sill : 0.95);
        for (let i = 0; i < nw; i++) {
          const a = F.a0 + pitch * (i + 0.5);
          const wh = sc.h || 1.58;
          const roll = this._dr();
          if (roll < 0.20) {
            this._acUnit(b, F, a + ww * 0.5 + this._drr(0.34, 0.5), sy + 0.62);
            this._wash(b.mat, F, a + ww * 0.5 + 0.42, 0.36, sy - 1.4, sy + 0.26, 0.68);
          } else if (roll < 0.33) {
            this._windowGuard(F, a, ww + 0.10, sy + 0.05, sy + wh - 0.05);
          } else if (roll < 0.47 && pitch - ww > 0.9) {
            this._oriel(b, F, a, Math.min(ww + 0.55, pitch - 0.30), sy - 0.32, wh + 0.72);
          } else if (roll < 0.58) {
            this._awning(b, F, a, Math.min(ww + 0.85, pitch - 0.35), sy + wh + 0.30);
            if (this._dr() < 0.5) this._faceCloth(F, a, sy - 0.10, this._drr(0.6, 1.0), this._drr(0.7, 1.2));
          } else if (roll < 0.68) {
            this._faceCloth(F, a, sy - 0.10, this._drr(0.6, 1.0), this._drr(0.7, 1.25));
            this._windowGuard(F, a, ww + 0.10, sy + 0.02, sy + 0.52);
          }
        }
      }
      // vents in the piers between the bays
      if (this._dr() < 0.8) {
        for (let a2 = 0; a2 < 6; a2++) {
          const a = F.a0 + pitch * this._dri(1, Math.max(1, nw - 1));
          const y = b.y + this._drr(1.9, 2.5) + this._dri(0, Math.max(0, b.st - 2)) * b.sh;
          if (!this._wallClear(holes, a, y, 0.9, 0.8)) continue;
          this._vent(b, F, a, y);
          break;
        }
      }
    }
    for (let i = 0, n = this._dri(1, 2); i < n; i++) {
      for (let a2 = 0; a2 < 6; a2++) {
        const a = this._drr(F.a0 + 0.9, F.a1 - 0.9);
        const y = b.y + this._drr(1.5, 2.1) + this._dri(1, Math.max(1, b.st - 1)) * b.sh;
        if (!this._wallClear(holes, a, y, 1.2, 1.0)) continue;
        this._wallDish(F, a, y);
        break;
      }
    }
    // painted signage over the ground floor
    if (!isShop && this._dr() < 0.6) {
      for (let a2 = 0; a2 < 6; a2++) {
        const w = this._drr(1.8, 3.1);
        if (span < w * 1.5) break;
        const a = this._drr(F.a0 + w * 0.6, F.a1 - w * 0.6);
        const y = b.y + b.sh - this._drr(0.85, 1.15);
        if (!this._wallClear(holes, a, y, w + 0.2, 0.8)) continue;
        this._sign(F, a, y, w);
        break;
      }
    }
    // parapet drip stains, so the roofline is never a clean edge
    for (let i = 0, n = this._dri(4, 9); i < n; i++) {
      const a = this._drr(F.a0 + 0.3, F.a1 - 0.3);
      this._wash(b.mat, F, a, this._drr(0.16, 0.60), top - this._drr(1.4, 3.6), top - 0.30,
        this._drr(0.52, 0.72));
    }
    // and a long-run stain down each string course, where the drip mould sheds
    for (let si = 1; si < b.st; si++) {
      const y = b.y + si * b.sh;
      for (let i = 0, n = this._dri(2, 5); i < n; i++) {
        const a = this._drr(F.a0 + 0.3, F.a1 - 0.3);
        this._wash(b.mat, F, a, this._drr(0.14, 0.42), y - this._drr(0.9, 2.4), y - 0.32,
          this._drr(0.58, 0.78));
      }
    }
  }

  /**
   * Every elevation in the level gets a trim pass — including the outskirt
   * blocks. Those are NOT background filler: from the market-hall deck the
   * outskirt row fills the whole right-hand half of the vista frame, which is
   * exactly the untrimmed stucco plane the critic pulled us up on. The far
   * ones only skip the ironmongery.
   */
  _facadeDetail() {
    const prev = this._zone;
    this._zoneSet('trim');
    for (const b of (this._blds || [])) {
      const cbx = (b.x0 + b.x1) * 0.5, cbz = (b.z0 + b.z1) * 0.5;
      const lod = b.id === 'O' ? 1 : 0;
      const style = this._dri(0, 7);
      for (const face of ['E', 'W', 'N', 'S']) {
        const F = this._faceFrame(b, face);
        const [nx, nz, cx, cz] = this._faceOut(F);
        // "public" = the elevation actually turned toward the play space
        const tx = -3 - cx, tz = 8 - cz;
        const l = Math.hypot(tx, tz) || 1;
        const pub = (nx * tx + nz * tz) / l;
        // an outskirt block only pays for the elevations that can be seen
        if (lod && pub < -0.25) continue;
        this._faceDress(b, face, style, pub > -0.05, lod);
      }
      if (!lod || Math.hypot(cbx + 3, cbz - 8) < 46) this._roofSkyline(b);
    }
    this._zoneSet(prev);
  }

  /**
   * The whole building: plinth, storeys, string courses, cornice, roof deck,
   * parapet, quoins, and a solid dark interior mass so nothing is see-through.
   */
  _building(spec) {
    const b = Object.assign({
      y: 0, sh: STOREY, st: 3, mat: 'plasterB', trim: 'concrete',
      t: T_WALL, parapet: 0.95, quoins: true, win: {},
    }, spec);
    b.top = b.y + b.st * b.sh;
    const TR = b.trim;

    // Plinth. On a hollow-ground building the plinth used to be ONE solid box
    // spanning the whole footprint up to +0.46 — so its top face became the
    // visible "floor" of the interior, buried every piece of furniture standing
    // on the real slab below it, showed the bare 2.9 m concrete repeat as a
    // grid across the room, and walled the bottom half-metre of the doorway
    // off. Hollow-ground buildings get a perimeter plinth with a real threshold
    // instead, so the dressed floor inside is what you actually see.
    if (b.hollowGround) {
      const P = 0.13, py0 = b.y - 0.30, py1 = b.y + 0.46, t = b.t;
      const PO = { grime: 1.35 };
      this._bbox(TR, b.x0 - P, py0, b.z1 - t, b.x1 + P, py1, b.z1 + P, PO);
      this._bbox(TR, b.x0 - P, py0, b.z0 - P, b.x1 + P, py1, b.z0 + t, PO);
      this._bbox(TR, b.x0 - P, py0, b.z0 + t, b.x0 + t, py1, b.z1 - t, PO);
      const dA = b.door === 'E' ? (b.doorA !== undefined ? b.doorA : (b.z0 + b.z1) * 0.5) : null;
      if (dA === null) {
        this._bbox(TR, b.x1 - t, py0, b.z0 + t, b.x1 + P, py1, b.z1 - t, PO);
      } else {
        const hw = (b.doorW || 1.15) * 0.5 + 0.03;
        this._bbox(TR, b.x1 - t, py0, b.z0 + t, b.x1 + P, py1, dA - hw, PO);
        this._bbox(TR, b.x1 - t, py0, dA + hw, b.x1 + P, py1, b.z1 - t, PO);
        // worn stone threshold running out through the opening
        this._bbox(TR, b.x1 - t - 0.06, py0, dA - hw, b.x1 + P + 0.26, b.y + 0.118, dA + hw,
          { grime: 1.7, shade: 1.03 });
      }
    } else {
      this._bbox(TR, b.x0 - 0.13, b.y - 0.30, b.z0 - 0.13, b.x1 + 0.13, b.y + 0.46, b.z1 + 0.13,
        { grime: 1.35 });
    }
    // walls
    for (let si = 0; si < b.st; si++) {
      for (const f of ['E', 'W', 'N', 'S']) this._facade(b, f, si);
      if (si < b.st - 1) {
        const y = b.y + (si + 1) * b.sh;
        this._bbox(TR, b.x0 - 0.09, y - 0.08, b.z0 - 0.09, b.x1 + 0.09, y + 0.09, b.z1 + 0.09, { base: b.y });
      }
    }
    // cornice + roof deck
    this._bbox(TR, b.x0 - 0.26, b.top - 0.30, b.z0 - 0.26, b.x1 + 0.26, b.top - 0.02, b.z1 + 0.26, { base: b.y, shade: 1.04 });
    this._bbox(TR, b.x0 - 0.14, b.top - 0.46, b.z0 - 0.14, b.x1 + 0.14, b.top - 0.30, b.z1 + 0.14, { base: b.y, shade: 1.02 });
    this._bbox(b.deckMat || 'concreteB', b.x0 + 0.02, b.top - 0.05, b.z0 + 0.02, b.x1 - 0.02, b.top, b.z1 - 0.02, { base: b.y });
    // interior mass
    const iy = b.hollowGround ? b.y + b.sh : b.y - 0.2;
    if (b.top - iy > 0.3) {
      this._bbox('dark', b.x0 + b.t + 0.01, iy, b.z0 + b.t + 0.01, b.x1 - b.t - 0.01, b.top - 0.5, b.z1 - b.t - 0.01,
        { noCast: true, raw: true, shade: 0.25 });
    }
    if (b.quoins) this._quoins(b, b.top);
    this._parapet(b, b.top);

    if (!this._blds) this._blds = [];
    this._blds.push(b);
    return b;
  }

  _addCover(F, a, y, depth, h) {
    const x = F.ax === 'z' ? F.p + F.sgn * depth : a;
    const z = F.ax === 'z' ? a : F.p + F.sgn * depth;
    const nx = F.ax === 'z' ? -F.sgn : 0, nz = F.ax === 'z' ? 0 : -F.sgn;
    this.coverPoints.push({
      pos: new THREE.Vector3(x, y, z), dir: new THREE.Vector3(nx, 0, nz), height: h || 1.0,
    });
  }

  // =====================================================================
  //  ARCHITECTURE LAYOUT
  // =====================================================================

  _buildArchitecture() {
    // primary spawns, in the order the camera presets expect them
    const D = Math.PI / 180;
    this.spawnPoints.push({ pos: new THREE.Vector3(0, 0, 18.0), yaw: 180 * D });
    this.spawnPoints.push({ pos: new THREE.Vector3(2.5, 0, 6.0), yaw: 195 * D });
    this.spawnPoints.push({ pos: new THREE.Vector3(-2.0, 0, -12.0), yaw: 172 * D });
    this.spawnPoints.push({ pos: new THREE.Vector3(3.6, 0, -22.0), yaw: 186 * D });

    const W3 = 1.22, WH = 1.58;
    const gridA = { n: 3, w: W3, h: WH, sill: 0.95, shutters: 0.45, glass: 0.4, ac: true };
    const gridB = { n: 4, w: 1.10, h: 1.50, sill: 1.0, shutters: 0.30, glass: 0.5 };
    const gridC = { n: 2, w: 1.35, h: 1.70, sill: 0.90, shutters: 0.55, glass: 0.3 };
    const balc = { n: 3, w: 1.22, h: 1.90, sill: 0.55, shutters: 0.25, glass: 0.35, balcony: true };

    // ---------------- WEST ROW (backlit, in shade) --------------------
    this._zoneSet('westA');
    this._building({
      id: 'W1', x0: -19.4, x1: -7.0, z0: -30.0, z1: -19.2, st: 3, mat: 'plasterB',
      win: { E: gridA, S: gridB, N: gridB, W: gridC }, balconyFace: 'E', broken: 0,
    });
    this._building({
      id: 'W2', x0: -17.2, x1: -7.4, z0: -18.2, z1: -12.4, st: 2, mat: 'brickA', trim: 'concreteB',
      win: { E: gridC, S: gridC, N: gridC, W: gridC }, parapet: 0.78,
    });

    this._zoneSet('westB');
    // W3 — the interior the "interior" camera preset stands in.
    this._building({
      id: 'W3', x0: -15.6, x1: -7.0, z0: -11.8, z1: -1.4, st: 2, mat: 'plasterA',
      hollowGround: true, parapet: 0.92, door: 'E', doorA: -4.4, doorW: 1.18,
      win: {
        E: [null, gridA],
        W: [{ n: 2, w: 1.45, h: 1.62, sill: 0.98, shutters: 0.0, glass: 0.0, open: true }, gridC],
        N: [null, gridC], S: [null, gridC],
      },
    });
    this._building({
      id: 'W4', x0: -18.0, x1: -7.6, z0: 0.0, z1: 11.0, st: 3, mat: 'plasterC',
      win: { E: [gridA, balc, balc], W: gridB, N: gridC, S: gridC },
      balconyFace: 'E', shop: 'E',
    });

    // ---------------- MARKET HALL (the "vista" perch) -----------------
    this._zoneSet('hall');
    this._marketHall();

    // ---------------- EAST ROW (sunlit façades) -----------------------
    this._zoneSet('eastA');
    this._building({
      id: 'E1', x0: 7.0, x1: 20.4, z0: -30.0, z1: -16.0, st: 4, mat: 'plasterC',
      win: { W: [gridA, gridA, balc, gridA], S: gridB, N: gridB, E: gridB },
      balconyFace: 'W', shop: 'W',
    });
    this._building({
      id: 'E2', x0: 7.0, x1: 19.2, z0: -15.0, z1: -3.9, st: 3, mat: 'brickB', trim: 'concreteB',
      win: { W: [gridA, balc, gridA], N: gridC, S: gridC, E: gridC },
      balconyFace: 'W', broken: 0,
    });

    this._zoneSet('eastB');
    this._building({
      id: 'E3', x0: 7.0, x1: 19.2, z0: -0.4, z1: 11.2, st: 3, mat: 'plasterA',
      win: { W: [gridA, balc, gridA], N: gridC, S: gridC, E: gridC },
      balconyFace: 'W', shop: 'W',
    });
    this._building({
      id: 'E4', x0: 7.4, x1: 21.0, z0: 12.0, z1: 24.2, st: 4, mat: 'plasterB',
      win: { W: [gridA, gridA, balc, gridA], N: gridA, S: gridA, E: gridB },
      balconyFace: 'W', shop: 'W', broken: 1,
    });

    this._zoneSet('eastC');
    this._arcade();

    // ---------------- NORTH BLOCK (closes the vista) ------------------
    this._zoneSet('north');
    this._building({
      id: 'N1', x0: -22.0, x1: -3.6, z0: 40.0, z1: 47.6, st: 3, mat: 'plasterC',
      win: { S: gridA, E: gridC, W: gridB, N: gridB }, broken: 1,
    });
    this._building({
      id: 'N2', x0: 2.4, x1: 20.0, z0: 39.2, z1: 47.6, st: 3, mat: 'brickA', trim: 'concreteB',
      win: { S: gridA, W: gridC, E: gridB, N: gridB },
    });
    this._minaret(-5.6, 43.2);
    this._gateArch();

    // ---------------- boundary walls & outskirts ----------------------
    this._zoneSet('outskirt');
    this._compoundWalls();
    this._outskirts();
  }

  /** single-storey covered market. Roof deck at exactly HALL_DECK. */
  _marketHall() {
    const x0 = -21.0, x1 = -8.4, z0 = 14.0, z1 = 23.4;
    const deck = HALL_DECK, slab = 0.30;
    const clear = deck - slab;
    // plinth / floor
    this._bbox('concrete', x0 - 0.16, -0.30, z0 - 0.16, x1 + 0.16, 0.16, z1 + 0.16, { grime: 1.4 });
    // solid west + south walls
    this._bbox('plasterB', x0, 0.10, z0, x0 + 0.34, deck, z1, {});
    this._bbox('plasterB', x0, 0.10, z0, x1, deck, z0 + 0.34, {});
    // piers on the open north + east sides
    const pier = 0.44;
    for (let x = x0 + 1.5; x <= x1 - 0.4; x += 2.42) {
      this._box('concrete', x, clear * 0.5 + 0.1, z1 - pier * 0.5, pier, clear, pier, {});
      this._box('concrete', x, clear + 0.16, z1 - pier * 0.5, pier + 0.16, 0.16, pier + 0.16, { base: 0 });
    }
    for (let z = z0 + 2.0; z <= z1 - 1.2; z += 2.35) {
      this._box('concrete', x1 - pier * 0.5, clear * 0.5 + 0.1, z, pier, clear, pier, {});
      this._box('concrete', x1 - pier * 0.5, clear + 0.16, z, pier + 0.16, 0.16, pier + 0.16, { base: 0 });
    }
    // lintel beams over the colonnade
    this._bbox('concrete', x0, clear, z1 - 0.5, x1, deck, z1, {});
    this._bbox('concrete', x1 - 0.5, clear, z0, x1, deck, z1, {});
    // roof slab with a cornice lip
    this._bbox('concreteB', x0 - 0.22, deck - slab, z0 - 0.22, x1 + 0.22, deck - 0.04, z1 + 0.22, { base: 0, shade: 1.03 });
    this._bbox('concreteB', x0 - 0.05, deck - 0.06, z0 - 0.05, x1 + 0.05, deck, z1 + 0.05, { base: 0 });
    // parapet, with a collapsed run on the north face right in front of the vista cam
    const b = { x0: x0 - 0.05, x1: x1 + 0.05, z0: z0 - 0.05, z1: z1 + 0.05, y: 0, mat: 'plasterB', trim: 'concrete', parapet: 0.95, broken: 1, noParapet: 'S' };
    this._parapet(b, deck);
    // external stair up the south face
    this._stair(-9.6, 0, 13.9, 10, 0.258, 0.30, 1.5, 'S');
    this.spawnPoints.push({ pos: new THREE.Vector3(-14.0, deck, 21.4), yaw: 150 * Math.PI / 180 });
    this.coverPoints.push({ pos: new THREE.Vector3(-13.0, deck + 1.0, 22.6), dir: new THREE.Vector3(0, 0, 1), height: 0.95 });
    this._hall = { x0, x1, z0, z1, deck };
  }

  /** stair run; face = direction the treads climb toward */
  _stair(cx, y0, z0, n, rise, run, width, face) {
    for (let i = 0; i < n; i++) {
      const y = y0 + rise * (i + 0.5);
      const h = rise * (i + 1);
      if (face === 'S') {
        this._box('concrete', cx, y0 + h * 0.5, z0 - run * (i + 0.5), width, h, run, { grime: 1.2 });
      } else if (face === 'N') {
        this._box('concrete', cx, y0 + h * 0.5, z0 + run * (i + 0.5), width, h, run, { grime: 1.2 });
      } else if (face === 'E') {
        this._box('concrete', cx + run * (i + 0.5), y0 + h * 0.5, z0, run, h, width, { grime: 1.2 });
      } else {
        this._box('concrete', cx - run * (i + 0.5), y0 + h * 0.5, z0, run, h, width, { grime: 1.2 });
      }
      void y;
    }
  }

  /** two-storey arcade closing the plaza's east side — sunlit columns */
  _arcade() {
    const x0 = 6.4, x1 = 18.0, z0 = 25.0, z1 = 38.4;
    const b = this._building({
      id: 'E5', x0: x0 + 2.2, x1, z0, z1, st: 2, mat: 'plasterD', trim: 'concrete',
      win: { W: [null, { n: 5, w: 1.05, h: 1.45, sill: 0.85, shutters: 0.4, glass: 0.4 }], N: { n: 3, w: 1.1, h: 1.5, sill: 0.9 }, S: { n: 3, w: 1.1, h: 1.5, sill: 0.9 }, E: { n: 4, w: 1.0, h: 1.4, sill: 1.0 } },
    });
    // colonnade in front of the ground floor — west-facing, so the low sun
    // rakes across the columns and throws a rhythm of long shadows
    const top = 3.35, cxp = x0 + 0.42, pitch = 2.30, arcR = pitch * 0.5;
    const zs = [];
    for (let z = z0 + 1.05; z <= z1 - 0.9; z += pitch) zs.push(z);
    for (const z of zs) {
      this._box('concrete', cxp, 0.22, z, 0.88, 0.44, 0.88, { grime: 1.45 });
      this._cyl('concrete', cxp, 0.44 + (top - 0.94) * 0.5, z, 0.29, 0.33, top - 0.94, 12, { base: 0 });
      this._box('concrete', cxp, top - 0.36, z, 0.84, 0.26, 0.84, { base: 0, shade: 1.06 });
      this._addCover({ ax: 'z', p: cxp, sgn: -1 }, z, 1.2, 0.45, 1.4);
    }
    // round-headed arches springing between adjacent columns
    const seg = 7;
    for (let i = 0; i + 1 < zs.length; i++) {
      const zc = (zs[i] + zs[i + 1]) * 0.5;
      for (let k = 0; k < seg; k++) {
        const th = Math.PI * (k + 0.5) / seg;
        const zz = zc - Math.cos(th) * arcR;
        const yy = top - 0.14 + Math.sin(th) * arcR;
        this._box('plasterD', cxp, yy, zz, 0.70, 0.34, arcR * Math.PI / seg + 0.06,
          { base: 0, tilt: [1, 0, 0, th - Math.PI * 0.5], shade: 1.02 });
      }
      this._bbox('plasterD', cxp - 0.32, top - 0.14 + arcR * 0.70, zs[i] - 0.05,
        cxp + 0.32, top + arcR + 0.20, zs[i + 1] + 0.05, { base: 0, shade: 0.98 });
    }
    // entablature + roof slab covering the arcade walkway
    const eaveY = top + arcR + 0.20;
    this._bbox('concrete', cxp - 0.62, eaveY, z0 - 0.1, x0 + 2.3, eaveY + 0.34, z1 + 0.1,
      { base: 0, shade: 1.07 });
    this._bbox('concreteB', cxp - 0.48, eaveY + 0.34, z0, x0 + 2.3, eaveY + 0.46, z1, { base: 0 });
    this._bbox('plasterD', cxp - 0.30, eaveY + 0.46, z0, cxp + 0.34, eaveY + 1.05, z1, { base: 0 });
    void b;
  }

  _minaret(x, z) {
    const r0 = 1.55, h1 = 12.0, h2 = 17.4;
    this._cyl('concrete', x, 0.30, z, r0 + 0.42, r0 + 0.55, 0.62, 16, { grime: 1.4 });
    this._cyl('plasterA', x, 0.6 + h1 * 0.5, z, r0 * 0.86, r0, h1, 16, { base: 0 });
    // string courses
    for (let k = 1; k <= 3; k++) {
      const y = 0.6 + h1 * (k / 4);
      this._cyl('concrete', x, y, z, r0 * 0.99, r0 * 0.99, 0.16, 16, { base: 0, shade: 1.04 });
    }
    // balcony gallery
    const gy = 0.6 + h1;
    this._cyl('concrete', x, gy + 0.12, z, r0 + 0.78, r0 + 0.62, 0.24, 16, { base: 0, shade: 1.05 });
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2;
      this._box('concrete', x + Math.cos(a) * (r0 + 0.62), gy + 0.62, z + Math.sin(a) * (r0 + 0.62),
        0.10, 0.78, 0.10, { base: 0, rotY: -a });
    }
    this._cyl('concrete', x, gy + 1.06, z, r0 + 0.70, r0 + 0.70, 0.12, 16, { base: 0 });
    // upper shaft, cornice and cap
    this._cyl('plasterA', x, gy + 0.6 + (h2 - h1) * 0.5, z, r0 * 0.62, r0 * 0.74, h2 - h1, 14, { base: 0 });
    this._cyl('concrete', x, 0.6 + h2 + 0.14, z, r0 * 0.86, r0 * 0.70, 0.28, 14, { base: 0, shade: 1.06 });
    this._cyl('plasterA', x, 0.6 + h2 + 1.15, z, 0.10, r0 * 0.74, 1.75, 14, { base: 0 });
    this._cyl('rust', x, 0.6 + h2 + 2.35, z, 0.035, 0.05, 0.7, 6, { base: 0, zone: 'detail' });
  }

  /** ceremonial gate closing the north end of the street */
  _gateArch() {
    const z0 = 40.4, z1 = 43.0, cx = -0.6, half = 3.0, pw = 1.6;
    const spring = 4.4;
    for (const s of [-1, 1]) {
      const px = cx + s * (half + pw * 0.5);
      this._box('brickA', px, 3.3, (z0 + z1) * 0.5, pw, 6.6, z1 - z0, { grime: 1.3 });
      this._box('concrete', px, 0.30, (z0 + z1) * 0.5, pw + 0.30, 0.60, z1 - z0 + 0.30, { grime: 1.5 });
      this._box('concrete', px, spring - 0.10, (z0 + z1) * 0.5, pw + 0.22, 0.22, z1 - z0 + 0.22,
        { base: 0, shade: 1.05 });
    }
    // voussoir ring: a semicircle spanning the opening, rotating about Z
    const seg = 11;
    for (let k = 0; k < seg; k++) {
      const th = Math.PI * (k + 0.5) / seg;
      const px = cx - Math.cos(th) * half;
      const py = spring + Math.sin(th) * half;
      this._box('brickA', px, py, (z0 + z1) * 0.5, 0.62, half * Math.PI / seg + 0.06, z1 - z0,
        { base: 0, tilt: [0, 0, 1, Math.PI * 0.5 - th], shade: 1.02 });
    }
    // spandrels, entablature, crowning cornice
    for (const s of [-1, 1]) {
      this._box('brickA', cx + s * (half - 0.4), spring + half * 0.55, (z0 + z1) * 0.5,
        1.4, half * 1.1, z1 - z0, { base: 0, shade: 0.99 });
    }
    this._bbox('brickA', cx - half - pw, spring + half + 0.25, z0, cx + half + pw, spring + half + 1.45, z1, { base: 0 });
    this._bbox('concrete', cx - half - pw - 0.28, spring + half + 1.45, z0 - 0.24,
      cx + half + pw + 0.28, spring + half + 1.86, z1 + 0.24, { base: 0, shade: 1.06 });
    this._bbox('concrete', cx - half - pw - 0.16, spring + half + 1.86, z0 - 0.14,
      cx + half + pw + 0.16, spring + half + 2.12, z1 + 0.14, { base: 0, shade: 1.04 });
    this._sign({ ax: 'x', p: z0, sgn: 1, T: 0.3 }, cx, spring + half + 0.9, 4.4);
    this._pockWall(cx - half - pw * 0.5, 0.5, 4.0, z0, 0, -1, 18, 0.7);
    this._pockWall(cx + half + pw * 0.5, 0.5, 4.0, z0, 0, -1, 18, 0.7);
  }

  _compoundWalls() {
    // west perimeter — deliberately low so the sun floods the plaza and the
    // west windows of the interior room
    for (let z = -14; z < 46; z += 2.4) {
      const h = z > 22 ? this._rr(1.9, 2.35) : this._rr(1.35, 1.85);
      if (this._r() < 0.13) continue;                       // breach
      const x = z > 22 ? -22.4 : -27.5;
      this._box('brickB', x, h * 0.5, z + 1.2, 0.32, h, 2.4, { grime: 1.3 });
      this._box('concrete', x, h + 0.06, z + 1.2, 0.46, 0.12, 2.42, { base: 0, shade: 1.05 });
    }
    // north-east yard wall
    for (let x = 20.6; x < 24; x += 2.4) {
      this._box('brickB', x + 1.2, 1.15, 24.6, 2.4, 2.3, 0.30, { grime: 1.3 });
    }
    // south checkpoint: blast walls staggered across the street
    for (let i = 0; i < 5; i++) {
      const x = -4.6 + i * 2.3, z = -25.5 + (i % 2) * 1.6;
      this._tWall(x, z, 0.08 * (i % 2 ? 1 : -1));
    }
  }

  /** precast T-wall blast barrier */
  _tWall(x, z, rot) {
    this._box('concreteB', x, 0.12, z, 2.35, 0.24, 0.92, { rotY: rot, grime: 1.4 });
    this._box('concreteB', x, 1.60, z, 2.05, 2.72, 0.26, { rotY: rot, base: 0 });
    this._box('concreteB', x, 2.98, z, 2.20, 0.12, 0.38, { rotY: rot, base: 0, shade: 1.04 });
    this.coverPoints.push({ pos: new THREE.Vector3(x, 1.4, z), dir: new THREE.Vector3(0, 0, -1), height: 1.4 });
  }

  /** low ruined blocks that fade into the haze behind the vista camera's view */
  _outskirts() {
    const blocks = [
      [-34, 30, 9, 8, 2], [-40, 40, 11, 9, 3], [-30, 43, 8, 7, 2],
      [-44, 22, 10, 8, 2], [-36, 12, 9, 7, 1], [-46, 33, 12, 10, 3],
      [-33, -8, 10, 9, 1], [-42, -2, 11, 8, 2], [-38, -18, 12, 9, 2],
      [26, 30, 10, 9, 3], [28, 12, 11, 10, 4], [25, -12, 12, 10, 3],
      [-16, -40, 12, 8, 3], [4, -40, 14, 9, 3], [22, -38, 12, 8, 2],
    ];
    for (const [x, z, w, d, st] of blocks) {
      const mat = this._r() < 0.5 ? 'plasterC' : 'brickB';
      this._building({
        id: 'O', x0: x - w * 0.5, x1: x + w * 0.5, z0: z - d * 0.5, z1: z + d * 0.5,
        st, mat, quoins: false, broken: 1, parapet: 0.8,
        win: { E: { n: 3, w: 1.1, h: 1.4, sill: 1.0, glass: 0.1, shutters: 0.1 }, W: { n: 3, w: 1.1, h: 1.4, sill: 1.0, glass: 0.1, shutters: 0.1 }, N: { n: 2, w: 1.1, h: 1.4, sill: 1.0, glass: 0.1, shutters: 0.1 }, S: { n: 2, w: 1.1, h: 1.4, sill: 1.0, glass: 0.1, shutters: 0.1 } },
      });
    }
  }

  // =====================================================================
  //  INSTANCED CLUTTER LIBRARY
  // =====================================================================

  _mergeList(list) {
    const clean = list.map(g => trimAttrs(ensureIndexed(g)));
    return clean.length === 1 ? clean[0] : mergeGeometries(clean, false);
  }

  _gBox(cx, cy, cz, w, h, d, ts) { return boxGeo(cx, cy, cz, w, h, d, ts); }

  _definePools() {
    const P = (n, mat, fn, o) => this._pool(n, mat, fn, o);

    P('crate', 'wood', () => {
      const L = [], s = 0.60, t = 0.055;
      L.push(this._gBox(0, s * 0.5, 0, s - t, s - t * 2, s - t, 0.55));
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        L.push(this._gBox(sx * (s * 0.5 - t * 0.5), s * 0.5, sz * (s * 0.5 - t * 0.5), t, s, t, 0.4));
      }
      for (const sy of [0.08, s - 0.08]) for (const sx of [-1, 1]) {
        L.push(this._gBox(sx * (s * 0.5 - t * 0.5), sy, 0, t, 0.08, s, 0.4));
        L.push(this._gBox(0, sy, sx * (s * 0.5 - t * 0.5), s, 0.08, t, 0.4));
      }
      return this._mergeList(L);
    }, { collide: true });

    P('crateBig', 'woodDark', () => {
      const L = [], w = 1.05, h = 0.72, d = 0.66, t = 0.06;
      L.push(this._gBox(0, h * 0.5, 0, w - t, h - t * 2, d - t, 0.6));
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        L.push(this._gBox(sx * (w * 0.5 - t * 0.5), h * 0.5, sz * (d * 0.5 - t * 0.5), t, h, t, 0.4));
      }
      L.push(this._gBox(0, h - 0.03, 0, w, 0.06, d, 0.5));
      for (const sy of [0.10, h - 0.12]) L.push(this._gBox(0, sy, d * 0.5 - t * 0.4, w, 0.09, t, 0.4));
      return this._mergeList(L);
    }, { collide: true });

    P('barrel', 'rust', () => {
      const L = [];
      const b = new THREE.CylinderGeometry(0.285, 0.285, 0.88, 14, 1, false);
      scaleUv(b, (Math.PI * 0.57) / 1.9, 0.88 / 1.9); b.translate(0, 0.44, 0); L.push(b);
      for (const y of [0.24, 0.64]) {
        const r = new THREE.CylinderGeometry(0.305, 0.305, 0.075, 14, 1, true);
        scaleUv(r, 1.0, 0.05); r.translate(0, y, 0); L.push(r);
      }
      const lid = new THREE.CylinderGeometry(0.245, 0.265, 0.05, 12, 1, false);
      scaleUv(lid, 0.8, 0.05); lid.translate(0, 0.905, 0); L.push(lid);
      return this._mergeList(L);
    }, { collide: true });

    P('tyre', 'rubber', () => {
      const g = new THREE.TorusGeometry(0.315, 0.115, 7, 14);
      g.rotateX(Math.PI * 0.5); scaleUv(g, 2.2, 0.8); g.translate(0, 0.115, 0);
      return g;
    }, { collide: true });

    P('sandbag', 'tarpTan', () => {
      const g = new THREE.SphereGeometry(0.30, 9, 6);
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) p.setXYZ(i, p.getX(i) * 1.02, p.getY(i) * 0.40, p.getZ(i) * 0.62);
      g.computeVertexNormals(); scaleUv(g, 0.55, 0.34);
      return g;
    }, { collide: false });

    P('rubbleS', 'concreteB', () => {
      const L = [];
      for (let i = 0; i < 2; i++) {
        const g = new THREE.IcosahedronGeometry(0.16 + i * 0.04, 0);
        const p = g.attributes.position;
        for (let k = 0; k < p.count; k++) {
          const j = 1 + this._rr(-0.4, 0.4);
          p.setXYZ(k, p.getX(k) * j * 1.2, p.getY(k) * j * 0.6, p.getZ(k) * j * 1.1);
        }
        g.computeVertexNormals();
        const uv = new Float32Array(p.count * 2);
        for (let k = 0; k < p.count; k++) { uv[k * 2] = p.getX(k) * 1.6; uv[k * 2 + 1] = p.getZ(k) * 1.6; }
        g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
        g.translate(this._rr(-0.22, 0.22), 0.09 + i * 0.02, this._rr(-0.22, 0.22));
        L.push(g);
      }
      return this._mergeList(L);
    }, { noCast: true });

    P('rubbleL', 'concreteB', () => {
      const g = new THREE.IcosahedronGeometry(0.52, 0);
      const p = g.attributes.position;
      for (let k = 0; k < p.count; k++) {
        const j = 1 + this._rr(-0.35, 0.35);
        p.setXYZ(k, p.getX(k) * j * 1.15, p.getY(k) * j * 0.55, p.getZ(k) * j * 1.05);
      }
      g.computeVertexNormals();
      const uv = new Float32Array(p.count * 2);
      for (let k = 0; k < p.count; k++) { uv[k * 2] = p.getX(k) * 0.9; uv[k * 2 + 1] = p.getZ(k) * 0.9; }
      g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      g.translate(0, 0.24, 0);
      return g;
    }, {});

    P('brickBit', 'brickB', () => this._gBox(0, 0.033, 0, 0.21, 0.066, 0.10, 0.35), { noCast: true });
    P('shard', 'concreteB', () => this._gBox(0, 0.018, 0, 0.30, 0.036, 0.22, 0.6), { noCast: true });
    P('pock', 'dark', () => {
      const g = new THREE.CylinderGeometry(0.005, 0.055, 0.05, 6, 1, false);
      g.rotateX(-Math.PI * 0.5); scaleUv(g, 0.1, 0.1);
      return g;
    }, { noCast: true });
    P('spall', 'concreteB', () => {
      const g = new THREE.CircleGeometry(0.19, 7);
      scaleUv(g, 0.4, 0.4);
      return g;
    }, { noCast: true });
    P('jerry', 'metal', () => {
      const L = [];
      L.push(this._gBox(0, 0.24, 0, 0.34, 0.47, 0.16, 0.45));
      L.push(this._gBox(0, 0.50, 0, 0.20, 0.06, 0.13, 0.3));
      L.push(this._gBox(-0.10, 0.47, 0, 0.05, 0.09, 0.14, 0.3));
      return this._mergeList(L);
    }, { collide: false });
    P('bucket', 'polymer', () => {
      const g = new THREE.CylinderGeometry(0.15, 0.115, 0.28, 10, 1, true);
      scaleUv(g, 0.7, 0.28); g.translate(0, 0.14, 0);
      return g;
    }, { noCast: false });

    // ---- gameplay-scale dressing props -------------------------------

    P('pallet', 'wood', () => {
      const L = [], w = 1.16, d = 0.82;
      for (const z of [-d * 0.5 + 0.07, 0, d * 0.5 - 0.07]) L.push(this._gBox(0, 0.019, z, w, 0.038, 0.13, 0.5));
      for (const x of [-w * 0.5 + 0.06, 0, w * 0.5 - 0.06]) L.push(this._gBox(x, 0.078, 0, 0.12, 0.080, d, 0.5));
      for (let i = 0; i < 6; i++) {
        L.push(this._gBox(0, 0.133, -d * 0.5 + 0.065 + i * (d - 0.13) / 5, w, 0.030, 0.105, 0.5));
      }
      return this._mergeList(L);
    }, { collide: false });

    P('cinder', 'concreteB', () => {
      const L = [], w = 0.40, h = 0.19, d = 0.20, t = 0.037;
      for (const sz of [-1, 1]) L.push(this._gBox(0, h * 0.5, sz * (d * 0.5 - t * 0.5), w, h, t, 0.5));
      for (const x of [-w * 0.5 + t * 0.5, 0, w * 0.5 - t * 0.5]) L.push(this._gBox(x, h * 0.5, 0, t, h, d, 0.5));
      L.push(this._gBox(0, h - 0.012, 0, w + 0.008, 0.024, d + 0.008, 0.5));
      return this._mergeList(L);
    }, { noCast: true });

    P('plank', 'woodDark', () => {
      const L = [];
      L.push(this._gBox(0, 0.028, 0, 1.72, 0.055, 0.175, 0.9));
      L.push(this._gBox(0.1, 0.075, 0.02, 1.30, 0.048, 0.155, 0.9));
      return this._mergeList(L);
    }, { noCast: true });

    // precast Jersey barrier — stepped taper reads as a bevel at distance
    P('jersey', 'concreteB', () => {
      const L = [], len = 2.18;
      L.push(this._gBox(0, 0.065, 0, len, 0.13, 0.66, 1.0));
      L.push(this._gBox(0, 0.225, 0, len, 0.19, 0.55, 1.0));
      L.push(this._gBox(0, 0.405, 0, len, 0.17, 0.43, 1.0));
      L.push(this._gBox(0, 0.680, 0, len, 0.38, 0.29, 1.0));
      L.push(this._gBox(0, 0.890, 0, len, 0.05, 0.245, 1.0));
      L.push(this._gBox(0, 0.928, 0, len + 0.06, 0.035, 0.31, 1.0));  // capping strip
      for (const sx of [-1, 1]) {
        L.push(this._gBox(sx * 0.48, 0.968, 0, 0.055, 0.06, 0.055, 0.3));  // lifting hooks
        L.push(this._gBox(sx * (len * 0.5 - 0.025), 0.47, 0, 0.05, 0.79, 0.32, 0.5)); // end rib
      }
      return this._mergeList(L);
    }, { collide: true });

    // concrete culvert section — big readable cylinder, waist-high cover
    P('pipe', 'concrete', () => {
      const L = [], len = 1.95, ro = 0.45;
      const o = new THREE.CylinderGeometry(ro, ro, len, 16, 1, false);
      scaleUv(o, 2.7, len / 2.4); o.rotateZ(Math.PI * 0.5); o.translate(0, ro, 0); L.push(o);
      for (const s of [-1, 1]) {
        const c = new THREE.CylinderGeometry(ro + 0.055, ro + 0.055, 0.11, 16, 1, false);
        scaleUv(c, 2.9, 0.1); c.rotateZ(Math.PI * 0.5); c.translate(s * (len * 0.5 - 0.055), ro, 0); L.push(c);
      }
      return this._mergeList(L);
    }, { collide: true });

    // the dark bore that makes a pipe read as a pipe at 40 m
    P('pipeBore', 'dark', () => {
      const L = [], len = 1.95, ri = 0.335;
      for (const s of [-1, 1]) {
        const c = new THREE.CylinderGeometry(ri, ri, 0.30, 13, 1, true);
        scaleUv(c, 1.0, 0.3); c.rotateZ(Math.PI * 0.5);
        c.translate(s * (len * 0.5 - 0.14), 0.45, 0); L.push(c);
        const d = new THREE.CircleGeometry(ri, 13);
        scaleUv(d, 0.4, 0.4);
        d.rotateY(s > 0 ? Math.PI * 0.5 : -Math.PI * 0.5);
        d.translate(s * (len * 0.5 - 0.29), 0.45, 0); L.push(d);
      }
      return this._mergeList(L);
    }, { noCast: true });

    P('spool', 'woodDark', () => {
      const L = [];
      for (const y of [0.055, 0.755]) {
        const g = new THREE.CylinderGeometry(0.68, 0.68, 0.11, 14, 1, false);
        scaleUv(g, 2.1, 0.12); g.translate(0, y, 0); L.push(g);
      }
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * Math.PI * 2;
        L.push(this._gBox(Math.cos(a) * 0.46, 0.405, Math.sin(a) * 0.46, 0.09, 0.60, 0.09, 0.5));
      }
      return this._mergeList(L);
    }, { collide: true });

    P('coil', 'rubber', () => {
      const g = new THREE.CylinderGeometry(0.50, 0.50, 0.56, 14, 1, false);
      scaleUv(g, 2.6, 0.5); g.translate(0, 0.405, 0);
      return g;
    }, {});

    P('sack', 'burlap', () => {
      const g = new THREE.SphereGeometry(0.27, 9, 7);
      const p = g.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const v = p.getY(i) / 0.27;
        p.setXYZ(i, p.getX(i) * (0.92 - v * 0.22), p.getY(i) * 1.42 + 0.02, p.getZ(i) * (0.86 - v * 0.20));
      }
      g.computeVertexNormals(); scaleUv(g, 0.6, 0.5);
      g.translate(0, 0.40, 0);
      return g;
    }, {});

    P('basket', 'wood', () => {
      const L = [];
      const g = new THREE.CylinderGeometry(0.29, 0.21, 0.36, 11, 1, true);
      scaleUv(g, 1.1, 0.34); g.translate(0, 0.18, 0); L.push(g);
      const r = new THREE.CylinderGeometry(0.315, 0.295, 0.055, 11, 1, false);
      scaleUv(r, 1.1, 0.06); r.translate(0, 0.37, 0); L.push(r);
      return this._mergeList(L);
    }, {});

    // small market kiosk, split into two pools so body and roof keep their materials
    P('kioskBody', 'woodDark', () => {
      const L = [], w = 1.72, d = 1.22, h = 2.02, t = 0.075;
      L.push(this._gBox(0, 0.055, 0, w, 0.11, d, 0.8));
      L.push(this._gBox(0, h * 0.5, -d * 0.5 + t * 0.5, w, h, t, 0.8));
      for (const s of [-1, 1]) L.push(this._gBox(s * (w * 0.5 - t * 0.5), h * 0.5, 0, t, h, d, 0.8));
      L.push(this._gBox(0, 0.50, d * 0.5 - t * 0.5, w, 0.79, t, 0.8));
      L.push(this._gBox(0, 0.945, d * 0.5 - 0.10, w + 0.18, 0.085, 0.40, 0.5));
      L.push(this._gBox(0, h - 0.22, d * 0.5 - t * 0.5, w, 0.44, t, 0.8));
      for (const s of [-1, 1]) L.push(this._gBox(s * (w * 0.5 - 0.16), 1.42, d * 0.5 - t * 0.5, 0.10, 0.52, t * 1.2, 0.5));
      return this._mergeList(L);
    }, { collide: true });

    P('kioskRoof', 'rust', () => {
      const L = [], w = 2.02, d = 1.56;
      L.push(this._gBox(0, 2.06, 0, w, 0.055, d, 1.0));
      for (let i = 0; i < 8; i++) {
        L.push(this._gBox(-w * 0.5 + 0.13 + i * (w - 0.26) / 7, 2.105, 0, 0.055, 0.05, d, 0.5));
      }
      L.push(this._gBox(0, 2.02, d * 0.5 + 0.02, w, 0.10, 0.05, 0.5));
      return this._mergeList(L);
    }, {});
  }

  _iPlace(name, x, y, z, ry, s, shade, tiltAng, tiltAx) {
    const p = this._instPools.get(name);
    if (!p) return;
    const q = new THREE.Quaternion();
    if (tiltAng) {
      const e = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(tiltAx ? tiltAx[0] : 1, 0, tiltAx ? tiltAx[2] : 0).normalize(), tiltAng);
      q.setFromEuler(new THREE.Euler(0, ry || 0, 0)).premultiply(e);
    } else {
      q.setFromEuler(new THREE.Euler(0, ry || 0, 0));
    }
    const sc = Array.isArray(s) ? new THREE.Vector3(s[0], s[1], s[2])
      : new THREE.Vector3(s || 1, s || 1, s || 1);
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), q, sc);
    p.items.push({ m, s: shade === undefined ? 1 : shade });
  }

  _stain(x, z, r, k) { this._stains.push([x, z, r, k === undefined ? 0.5 : k]); }

  _rubblePile(x, z, r, n, big, yBase) {
    const G = (px, pz) => (yBase !== undefined ? yBase : this._surfY(px, pz));
    for (let i = 0; i < n; i++) {
      const a = this._r() * Math.PI * 2, d = Math.pow(this._r(), 0.6) * r;
      const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
      const drop = 1 - d / r;
      const sc = 0.55 + drop * 0.9;
      this._iPlace('rubbleS', px, G(px, pz) + 0.02 + drop * (big ? 0.55 : 0.22), pz,
        this._r() * 6.28, sc, 0.78 + this._r() * 0.28, this._rr(-0.3, 0.3));
      if (this._r() < 0.35) {
        const bx = px + this._rr(-0.3, 0.3), bz = pz + this._rr(-0.3, 0.3);
        this._iPlace('brickBit', bx, G(bx, bz) + 0.02 + drop * (big ? 0.5 : 0.2), bz,
          this._r() * 6.28, 1, 0.85 + this._r() * 0.3, this._rr(-0.4, 0.4));
      }
    }
    if (big) {
      for (let i = 0; i < Math.max(2, n / 6); i++) {
        const a = this._r() * Math.PI * 2, d = this._r() * r * 0.7;
        const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
        this._iPlace('rubbleL', px, G(px, pz) + 0.05 + this._r() * 0.35, pz,
          this._r() * 6.28, 0.7 + this._r() * 0.8, 0.8 + this._r() * 0.25, this._rr(-0.35, 0.35));
      }
    }
    for (let i = 0; i < n * 0.45; i++) {
      const a = this._r() * Math.PI * 2, d = r * (0.7 + this._r() * 0.9);
      const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
      this._iPlace('shard', px, G(px, pz) + 0.015, pz,
        this._r() * 6.28, 0.6 + this._r() * 0.9, 0.8 + this._r() * 0.25);
    }
    this._stain(x, z, r * 1.5, 0.55);
  }

  /** bullet pocks + spall patches scattered over a vertical wall rectangle */
  _pockWall(x, y0, y1, z, nx, nz, count, spread) {
    const ry = Math.atan2(nx, nz);
    for (let i = 0; i < count; i++) {
      const t = this._r();
      const px = x + (nz) * this._rr(-spread, spread);
      const pz = z + (-nx) * this._rr(-spread, spread);
      const py = y0 + (y1 - y0) * (t * t * 0.85 + 0.05);
      this._iPlace('pock', px + nx * 0.012, py, pz + nz * 0.012, ry, 0.7 + this._r() * 0.9, 0.5 + this._r() * 0.3);
      if (this._r() < 0.30) {
        const g = new THREE.CircleGeometry(this._rr(0.10, 0.26), 7);
        scaleUv(g, 0.5, 0.5);
        g.lookAt(new THREE.Vector3(nx, 0, nz));
        g.translate(px + nx * 0.016, py, pz + nz * 0.016);
        this._add('concreteB', g, { zone: 'detail', noCollide: true, noCast: true, shade: 1.12, base: 0 });
      }
    }
  }

  // =====================================================================
  //  PROPS
  // =====================================================================

  _buildProps() {
    this._definePools();

    this._zoneSet('props');
    this._streetFurniture();
    this._alley();
    this._plaza();
    this._technical(-16.6, 29.8, 0.62);
    this._wreck(-30.5, -6.0, 1.9);
    this._wreck(-25.0, 34.5, -0.8);

    // set dressing runs last so it can steer clear of everything above
    this._dressStreet();
    this._dressPlaza();
    this._fillCourtyard();
    this._dressEdges();
    this._nearFieldFloor();
    this._zoneSet('props');
  }

  _streetFurniture() {
    // kerbs + sidewalks, both sides, with broken segments and sand drifts
    for (const s of [-1, 1]) {
      const kx = s * ROAD_HW;
      for (let z = -31; z < 24.4; z += 1.6) {
        if (this._r() < 0.06) continue;
        const tilt = this._r() < 0.14 ? this._rr(-0.14, 0.14) : 0;
        const drop = this._r() < 0.10 ? this._rr(0.02, 0.06) : 0;
        this._box('concrete', kx + s * KERB_W * 0.5, KERB_H * 0.5 - drop, z + 0.8, KERB_W, KERB_H + 0.08, 1.58,
          { grime: 1.5, tilt: tilt ? [0, 0, 1, tilt] : null, shade: 1.03 });
        // gutter stone in the road, hard against the kerb foot
        this._box('concreteB', kx - s * 0.19, 0.012, z + 0.8, 0.38, 0.05, 1.56,
          { grime: 1.6, shade: 0.94, noCast: true });
      }
      this._bbox('concrete', Math.min(kx + s * KERB_W, kx + s * 1.44), 0, -31,
        Math.max(kx + s * KERB_W, kx + s * 1.44), KERB_H, 24.4, { grime: 1.4 });
      // flagstone joints across the sidewalk so it is not one poured ribbon
      for (let z = -30.6; z < 24.2; z += 1.42) {
        this._box('concreteB', kx + s * 0.86, KERB_H - 0.005, z, 1.18, 0.03, 0.07,
          { grime: 1.5, shade: 0.9, noCast: true, zone: 'detail' });
      }
    }

    // ---- drainage channel down the crown of the street ----------------
    for (let z = -31; z < 24.4; z += 1.42) {
      for (const s of [-1, 1]) {
        this._box('concrete', s * CHAN_HW, this._hRoad(CHAN_HW) + 0.015, z + 0.71, 0.26, 0.10, 1.40,
          { grime: 1.55, shade: 1.04 });
      }
    }
    for (let z = -27.4; z < 23; z += 7.1) {
      const y = this._hRoad(0);
      this._box('gunmetal', 0, y + 0.045, z, 0.70, 0.06, 0.56, { grime: 1.7, shade: 0.82, base: y });
      for (let i = 0; i < 5; i++) {
        this._box('dark', 0, y + 0.062, z - 0.20 + i * 0.10, 0.58, 0.05, 0.045,
          { raw: true, shade: 0.26, noCast: true, zone: 'detail' });
      }
      this._stain(0, z, 1.3, 0.5);
    }
    // sump where the channel dies at the plaza mouth
    this._box('concrete', 0, this._hRoad(0) + 0.10, 24.15, 1.9, 0.30, 0.55, { grime: 1.6, shade: 1.02 });
    this._box('dark', 0, this._hRoad(0) + 0.06, 24.02, 1.4, 0.24, 0.10, { raw: true, shade: 0.22, noCast: true });

    // ---- raised walkways ringing the plaza ----------------------------
    this._bbox('concreteB', 5.30, 0, 23.90, 6.62, WALK_H, 38.70, { grime: 1.35, shade: 0.99 });
    for (let z = 23.90; z < 38.62; z += 1.46) {
      const l = Math.min(1.44, 38.70 - z);
      this._box('concrete', 5.14, WALK_H * 0.5 - (this._r() < 0.1 ? 0.04 : 0), z + l * 0.5, 0.32, WALK_H + 0.07, l - 0.02,
        { grime: 1.5, shade: 1.03, tilt: this._r() < 0.12 ? [0, 0, 1, this._rr(-0.1, 0.1)] : null });
    }
    this._bbox('concreteB', -21.0, 0, 38.30, 5.62, WALK_H, 39.72, { grime: 1.35, shade: 0.99 });
    for (let x = -21.0; x < 5.5; x += 1.46) {
      const l = Math.min(1.44, 5.62 - x);
      this._box('concrete', x + l * 0.5, WALK_H * 0.5 - (this._r() < 0.1 ? 0.04 : 0), 38.14, l - 0.02, WALK_H + 0.07, 0.32,
        { grime: 1.5, shade: 1.03, tilt: this._r() < 0.12 ? [1, 0, 0, this._rr(-0.1, 0.1)] : null });
    }
    // street poles + catenary power lines down the west side
    for (let z = -26; z < 24; z += 11.5) {
      const x = -6.4;
      this._cyl('rust', x, 2.6, z, 0.075, 0.11, 5.2, 8, { grime: 1.4 });
      this._box('rust', x, 5.15, z, 0.10, 0.10, 1.5, { base: 0 });
      this._box('metal', x, 5.05, z + 0.8, 0.44, 0.16, 0.30, { base: 0, zone: 'detail' });
      this._stain(x, z, 0.7, 0.5);
      if (z > -25) this._wire(x, 5.15, z - 11.5, x, 5.15, z, 0.85);
      // cross-street span to the east row
      if (this._r() < 0.7) this._wire(x, 4.9, z, 7.2, 4.4 + this._r(), z + this._rr(-1.5, 1.5), 1.1);
    }
    // bollards + a burnt-out kiosk at the south checkpoint
    for (let i = 0; i < 7; i++) {
      this._cyl('concreteB', -4.9 + i * 1.6, 0.34, -22.4, 0.11, 0.13, 0.68, 8, { grime: 1.5 });
    }
    this._box('rustDark', 8.6, 1.05, -20.0, 2.2, 2.1, 1.9, { rotY: 0.3, grime: 1.4, shade: 0.6 });
    this._box('rustDark', 8.6, 2.16, -20.0, 2.5, 0.12, 2.2, { rotY: 0.3, base: 0, shade: 0.55 });
    this._rubblePile(9.4, -18.6, 1.5, 16, false);
  }

  _wire(x0, y0, z0, x1, y1, z1, sag) {
    const pts = [];
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      pts.push(new THREE.Vector3(
        x0 + (x1 - x0) * t,
        y0 + (y1 - y0) * t - Math.sin(Math.PI * t) * sag,
        z0 + (z1 - z0) * t));
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const g = new THREE.TubeGeometry(curve, 12, 0.022, 4, false);
    scaleUv(g, 3, 0.2);
    this._add('rustDark', g, { zone: 'wire', noCollide: true, noCast: true, shade: 0.55, base: 0 });
  }

  // =====================================================================
  //  INTERIOR LIGHTING BAKE
  //  The room is a shaded box with two open west apertures. Rather than hope
  //  a shadow cascade lands the shaft in the right place, the sun patch is
  //  traced analytically per vertex — through the west aperture, past the
  //  partition doorway, with the window mullion's bar in it — and baked into
  //  vertex colour. Every surface AND every prop in the room uses the same
  //  function, so a crate standing in the beam is hot and the same crate two
  //  metres deeper is cool, and the pool has real falloff into the room.
  // =====================================================================

  /**
   * 0..1 fraction of the sun disc reaching a point inside the W3 room.
   * `b` widens every soft edge: pass 0 for the hard shaft, ~0.7 for the
   * bounce halo that spills off it onto the surrounding floor and walls.
   */
  _sunLit(x, y, z, b) {
    const I = this._IR;
    if (!I) return 0;
    const DX = 0.879, DY = 0.192, DZ = 0.434;      // toward the sun
    const B = b || 0;
    let jam = 1;
    if (x > I.px) {
      const t = (x - I.px) / DX;
      const yy = y + DY * t, zz = z + DZ * t;
      if (yy < I.ptop) {
        if (yy > I.dhead + B) return 0;
        const e = Math.min(zz - I.dz0, I.dz1 - zz) + B;
        if (e <= 0) return 0;
        jam = smooth(0, 0.15 + B, e) * (1 - smooth(I.dhead + B - 0.20, I.dhead + B, yy));
        if (jam <= 0) return 0;
      }
    }
    const t2 = (x - I.wx) / DX;
    if (t2 < 0) return 0;
    const yy = y + DY * t2, zz = z + DZ * t2;
    let f = smooth(I.sill - B, I.sill + 0.15, yy) * (1 - smooth(I.head - 0.20, I.head + B, yy));
    if (f <= 0) return 0;
    let g = 0;
    for (let i = 0; i < I.wins.length; i++) {
      const v = 1 - smooth(I.whw - 0.12, I.whw + B, Math.abs(zz - I.wins[i]));
      if (v > g) g = v;
    }
    f *= g;
    if (f <= 0) return 0;
    if (!B) f *= 0.34 + 0.66 * smooth(0.035, 0.115, Math.abs(yy - I.mull));
    return f * jam;
  }

  /** warm sun + cool skylight fill, as an rgb multiplier on the albedo */
  _intShade(x, y, z, nx, ny, nz) {
    const I = this._IR;
    const c = this._c3 || (this._c3 = [1, 1, 1]);
    if (!I) { c[0] = c[1] = c[2] = 1; return c; }
    const px = x + nx * 0.03, py = y + ny * 0.03, pz = z + nz * 0.03;
    const hard = this._sunLit(px, py, pz, 0);
    const soft = this._sunLit(px, py, pz, 0.75);      // bounce spilling off the shaft
    const ndl = Math.max(0, nx * -0.879 + ny * 0.192 + nz * 0.434);
    const dw = Math.exp(-Math.max(0, x - I.wx) * 0.20);       // west apertures
    const dd = Math.exp(-Math.hypot(x - I.x1, z - I.doorz) * 0.27);  // street door
    let amb = 0.32 + 0.30 * dw + 0.30 * dd + 0.34 * soft;
    amb *= 0.72 + 0.44 * Math.max(0, ny) + 0.16 * Math.max(0, -nz);
    amb *= 0.90 + 0.22 * vnoise(x * 0.63 + 5.1, y * 0.63, z * 0.63);
    const w = hard * (0.58 + 0.98 * ndl);
    c[0] = clamp(amb * 0.98 + w * 1.48, 0, 2.4);
    c[1] = clamp(amb * 0.99 + w * 1.18, 0, 2.4);
    c[2] = clamp(amb * 1.12 + w * 0.74, 0, 2.4);
    return c;
  }

  _intLum(x, y, z) {
    const c = this._intShade(x, y, z, 0, 1, 0);
    return clamp(0.32 * c[0] + 0.58 * c[1] + 0.10 * c[2], 0.16, 2.2);
  }

  _paintLit(geo, k) {
    const pos = geo.attributes.position, nor = geo.attributes.normal;
    const n = pos.count, m = k === undefined ? 1 : k;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const c = this._intShade(pos.getX(i), pos.getY(i), pos.getZ(i),
        nor ? nor.getX(i) : 0, nor ? nor.getY(i) : 1, nor ? nor.getZ(i) : 0);
      arr[i * 3] = c[0] * m; arr[i * 3 + 1] = c[1] * m; arr[i * 3 + 2] = c[2] * m;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(arr, 3));
    return geo;
  }

  /** box lit by the interior bake */
  _iB(mat, cx, cy, cz, w, h, d, o) {
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const g = boxGeo(cx, cy, cz, w, h, d, ts, o && o.faces);
    if (o && o.rotY) rotYg(g, (o.pivotX !== undefined ? o.pivotX : cx), (o.pivotZ !== undefined ? o.pivotZ : cz), o.rotY);
    if (o && o.tilt) rotAxisG(g, cx, cy, cz, o.tilt[0], o.tilt[1], o.tilt[2], o.tilt[3]);
    this._paintLit(g, o && o.k);
    return this._add(mat, g, Object.assign({ raw: true }, o));
  }

  /** cylinder lit by the interior bake */
  _iCyl(mat, cx, cy, cz, rt, rb, h, seg, o) {
    const ts = (o && o.ts) || this._tsz[mat] || 2.0;
    const g = new THREE.CylinderGeometry(rt, rb, h, seg || 10, 1, !!(o && o.open));
    scaleUv(g, (Math.PI * (rt + rb)) / ts, h / ts);
    g.translate(cx, cy, cz);
    if (o && o.tilt) rotAxisG(g, cx, cy, cz, o.tilt[0], o.tilt[1], o.tilt[2], o.tilt[3]);
    if (o && o.rotY) rotYg(g, cx, cz, o.rotY);
    this._paintLit(g, o && o.k);
    return this._add(mat, g, Object.assign({ raw: true }, o));
  }

  _litFix(geo, nx, ny, nz) {
    const p = geo.attributes.position.array, ix = geo.index.array;
    const a = ix[0] * 3, b = ix[1] * 3, c = ix[2] * 3;
    const e1x = p[b] - p[a], e1y = p[b + 1] - p[a + 1], e1z = p[b + 2] - p[a + 2];
    const e2x = p[c] - p[a], e2y = p[c + 1] - p[a + 1], e2z = p[c + 2] - p[a + 2];
    const cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
    if (cx * nx + cy * ny + cz * nz < 0) {
      for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; }
    }
    return geo;
  }

  _litMesh(mat, nu, nv, ptFn, nx, ny, nz, uvFn, o) {
    const vw = nu + 1, vh = nv + 1, cnt = vw * vh;
    const pos = new Float32Array(cnt * 3), nor = new Float32Array(cnt * 3);
    const uvs = new Float32Array(cnt * 2), col = new Float32Array(cnt * 3);
    for (let j = 0; j < vh; j++) {
      for (let i = 0; i < vw; i++) {
        const k = j * vw + i;
        const P = ptFn(i / nu, j / nv);
        pos[k * 3] = P[0]; pos[k * 3 + 1] = P[1]; pos[k * 3 + 2] = P[2];
        nor[k * 3] = nx; nor[k * 3 + 1] = ny; nor[k * 3 + 2] = nz;
        const U = uvFn(P[0], P[1], P[2]);
        uvs[k * 2] = U[0]; uvs[k * 2 + 1] = U[1];
        const c = this._intShade(P[0], P[1], P[2], nx, ny, nz);
        col[k * 3] = c[0]; col[k * 3 + 1] = c[1]; col[k * 3 + 2] = c[2];
      }
    }
    const idx = cnt > 65535 ? new Uint32Array(nu * nv * 6) : new Uint16Array(nu * nv * 6);
    let q = 0;
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        const a = j * vw + i, b = a + 1, c = a + vw, d = c + 1;
        idx[q++] = a; idx[q++] = c; idx[q++] = b;
        idx[q++] = b; idx[q++] = c; idx[q++] = d;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    this._litFix(g, nx, ny, nz);
    return this._add(mat, g, Object.assign({ raw: true }, o));
  }

  /** vertical lining panel: ax='x' -> plane at constant x, `a` runs in z */
  _litPanel(mat, ax, p, ns, a0, a1, y0, y1, cell, o) {
    if (a1 - a0 < 0.03 || y1 - y0 < 0.03) return;
    const nu = Math.max(1, Math.round((a1 - a0) / cell));
    const nv = Math.max(1, Math.round((y1 - y0) / cell));
    const ts = this._tsz[mat] || 2.0;
    const nx = ax === 'x' ? ns : 0, nz = ax === 'z' ? ns : 0;
    this._litMesh(mat, nu, nv,
      (u, v) => {
        const a = a0 + (a1 - a0) * u, y = y0 + (y1 - y0) * v;
        return ax === 'x' ? [p, y, a] : [a, y, p];
      },
      nx, 0, nz,
      (X, Y, Z) => [(ax === 'x' ? Z : X) / ts, Y / ts], o);
  }

  /** horizontal lining panel (floor ny=+1, ceiling ny=-1) */
  _litSlab(mat, x0, z0, x1, z1, y, ny, cell, o) {
    if (x1 - x0 < 0.03 || z1 - z0 < 0.03) return;
    const nu = Math.max(1, Math.round((x1 - x0) / cell));
    const nv = Math.max(1, Math.round((z1 - z0) / cell));
    const ts = this._tsz[mat] || 2.0;
    this._litMesh(mat, nu, nv,
      (u, v) => [x0 + (x1 - x0) * u, y, z0 + (z1 - z0) * v],
      0, ny, 0, (X, Y, Z) => [X / ts, Z / ts], o);
  }

  /** wall panel with apertures cut out of it */
  _litWallHoles(mat, ax, p, ns, a0, a1, y0, y1, holes, cell, o) {
    const hs = (holes || []).slice().sort((A, B) => A[0] - B[0]);
    let cur = a0;
    for (let i = 0; i < hs.length; i++) {
      const h = hs[i];
      const ha0 = Math.max(a0, h[0]), ha1 = Math.min(a1, h[1]);
      if (ha1 <= ha0) continue;
      if (ha0 > cur) this._litPanel(mat, ax, p, ns, cur, ha0, y0, y1, cell, o);
      if (h[2] > y0) this._litPanel(mat, ax, p, ns, ha0, ha1, y0, Math.min(h[2], y1), cell, o);
      if (h[3] < y1) this._litPanel(mat, ax, p, ns, ha0, ha1, Math.max(h[3], y0), y1, cell, o);
      cur = Math.max(cur, ha1);
    }
    if (cur < a1) this._litPanel(mat, ax, p, ns, cur, a1, y0, y1, cell, o);
  }

  /** the room the "interior" preset stands in: bright west windows, deep shade */
  _interiorRoom() {
    const x0 = -15.6 + T_WALL, x1 = -7.0 - T_WALL, z0 = -11.8 + T_WALL, z1 = -1.4 - T_WALL;
    const FY = INT_FY, CEIL = 2.84;
    // The partition opening is placed so the shaft coming through the north
    // west aperture clears both jambs: the beam crosses this plane between
    // z = -6.49 and z = -5.04, so the opening has to bracket that.
    const PX = -11.70, PT = 0.16, DZ0 = -7.20, DZ1 = -4.88, DH = FY + 2.10;
    const DA = -4.4, DW = 1.18;
    this._IR = {
      x0, x1, z0, z1, fy: FY, wx: x0,
      sill: 0.98, head: 2.60, mull: 0.98 + 1.62 * 0.52,
      wins: [-9.2, -4.0], whw: 0.66,
      px: PX, ptop: CEIL - 0.04, dz0: DZ0, dz1: DZ1, dhead: DH, doorz: DA,
    };
    const prevZone = this._zone;
    this._zoneSet('interior');
    const R = (a, b) => this._irr(a, b);
    const RI = (a, b) => this._iri(a, b);
    const LIN = { zone: 'interior', noCast: true };
    const DET = { zone: 'intdet', noCast: true };
    const CW = 0.30;

    // ---------------------------------------------------------------- shell
    // Solid sub-slab: this is the collider the player actually stands on, and
    // it also caps the dirt height-field, which used to poke its 1 m grid
    // through the old 2 cm floor box (that was the "regular grid pattern").
    this._bbox('concreteB', x0, -0.34, z0, x1, FY - 0.014, z1, { grime: 1.25, shade: 0.86 });
    // visible screed carrying the baked sun pool
    this._litSlab('intFloor', x0, z0, x1, z1, FY, 1, 0.30, LIN);

    // surviving tile field in the east room, so the beam rakes across grout
    const TS = 0.55, TG = 0.05;
    for (let tx = PX + 0.34; tx < x1 - TS; tx += TS) {
      for (let tz = z0 + 0.22; tz < z1 - TS; tz += TS) {
        const q = this._ir();
        if (q < 0.14) continue;                       // tile gone: screed shows
        this._iB(q < 0.55 ? 'tile' : 'tileAlt', tx + TS * 0.5, FY + (q < 0.19 ? 0.030 : 0.018),
          tz + TS * 0.5, TS - TG, 0.022, TS - TG,
          Object.assign({ faces: 4, ts: 0.58 }, DET,
            q < 0.19 ? { tilt: [1, 0, 0.45, R(-0.055, 0.055)] } : null));
      }
    }

    // wall linings, apertures cut out of them
    this._litWallHoles('intWall', 'x', x0 + 0.016, 1, z0, z1, FY, CEIL,
      [[-9.88, -8.52, 0.98, 2.60], [-4.68, -3.32, 0.98, 2.60]], CW, LIN);
    this._litWallHoles('intWall', 'x', x1 - 0.016, -1, z0, z1, FY, CEIL,
      [[DA - DW * 0.5 - 0.02, DA + DW * 0.5 + 0.02, FY, 2.05]], CW, LIN);
    this._litPanel('intWall', 'z', z0 + 0.016, 1, x0, x1, FY, CEIL, CW, LIN);
    this._litPanel('intWall', 'z', z1 - 0.016, -1, x0, x1, FY, CEIL, CW, LIN);

    // painted dado, skirting and a picture rail — a bare tan box has none of it
    for (const [ax, p, ns] of [['x', x0 + 0.030, 1], ['x', x1 - 0.030, -1]]) {
      this._litWallHoles('intDado', ax, p, ns, z0, z1, FY + 0.02, FY + 0.98,
        ax === 'x' && ns > 0 ? [[-9.88, -8.52, 0.90, 2.60], [-4.68, -3.32, 0.90, 2.60]]
          : [[DA - DW * 0.5 - 0.02, DA + DW * 0.5 + 0.02, FY, 2.05]], 0.5, DET);
    }
    this._litPanel('intDado', 'z', z0 + 0.030, 1, x0, x1, FY + 0.02, FY + 0.98, 0.5, DET);
    this._litPanel('intDado', 'z', z1 - 0.030, -1, x0, x1, FY + 0.02, FY + 0.98, 0.5, DET);
    this._iB('intWall', x0 + 0.05, FY + 0.07, (z0 + z1) * 0.5, 0.10, 0.14, z1 - z0, DET);
    this._iB('intWall', x1 - 0.05, FY + 0.07, (z0 + z1) * 0.5, 0.10, 0.14, z1 - z0, DET);
    this._iB('intWall', (x0 + x1) * 0.5, FY + 0.07, z0 + 0.05, x1 - x0, 0.14, 0.10, DET);
    this._iB('intWall', (x0 + x1) * 0.5, FY + 0.07, z1 - 0.05, x1 - x0, 0.14, 0.10, DET);
    // picture rail capping the dado
    this._iB('intWall', x0 + 0.06, FY + 1.02, (z0 + z1) * 0.5, 0.07, 0.07, z1 - z0, DET);
    this._iB('intWall', x1 - 0.06, FY + 1.02, (z0 + z1) * 0.5, 0.07, 0.07, z1 - z0, DET);

    // fallen render: blockwork showing through, plus damp blooms
    for (let i = 0; i < 9; i++) {
      const wall = RI(0, 3);
      const w = R(0.5, 1.7), h = R(0.4, 1.3);
      const y = R(FY + 0.5, CEIL - 0.7);
      if (wall < 2) {
        const px = wall ? x1 - 0.055 : x0 + 0.055;
        const pz = R(z0 + 1.0, z1 - 1.0);
        if (Math.abs(pz - DA) < 1.2 && wall) continue;
        this._iB('brickB', px, y, pz, 0.05, h, w, { zone: 'intdet', ts: 1.5, noCast: true, k: 0.94 });
      } else {
        const pz = wall === 2 ? z0 + 0.055 : z1 - 0.055;
        this._iB('brickB', R(x0 + 1.0, x1 - 1.0), y, pz, w, h, 0.05,
          { zone: 'intdet', ts: 1.5, noCast: true, k: 0.94 });
      }
    }
    for (let i = 0; i < 12; i++) {
      const pz = R(z0 + 0.4, z1 - 0.4);
      this._iB('intScreed', (this._ir() < 0.5 ? x0 + 0.042 : x1 - 0.042), R(FY + 0.6, CEIL - 0.4), pz,
        0.02, R(0.5, 1.6), R(0.5, 2.0), { zone: 'intdet', ts: 1.7, noCast: true, k: R(0.78, 0.96) });
    }

    // ------------------------------------------------------------ partition
    this._bbox('interior', PX - PT, FY - 0.06, z0, PX + PT, CEIL - 0.02, DZ0, {});
    this._bbox('interior', PX - PT, FY - 0.06, DZ1, PX + PT, CEIL - 0.02, z1, {});
    this._bbox('interior', PX - PT, DH, DZ0, PX + PT, CEIL - 0.02, DZ1, {});
    const PH = [[DZ0, DZ1, FY, DH]];
    this._litWallHoles('intWall', 'x', PX - PT - 0.014, -1, z0, z1, FY, CEIL - 0.04, PH, CW, LIN);
    this._litWallHoles('intWall', 'x', PX + PT + 0.014, 1, z0, z1, FY, CEIL - 0.04, PH, CW, LIN);
    // real frame + threshold in the opening
    for (const s of [-1, 1]) {
      this._iB('woodDark', PX, FY + (DH - FY) * 0.5, s > 0 ? DZ1 + 0.05 : DZ0 - 0.05,
        PT * 2 + 0.07, DH - FY, 0.11, { ts: 0.9 });
    }
    this._iB('woodDark', PX, DH + 0.055, (DZ0 + DZ1) * 0.5, PT * 2 + 0.07, 0.11, DZ1 - DZ0 + 0.20, { ts: 0.9 });
    this._iB('concreteB', PX, FY + 0.022, (DZ0 + DZ1) * 0.5, PT * 2 + 0.18, 0.05, DZ1 - DZ0 + 0.06,
      Object.assign({ ts: 1.2 }, DET));
    // the door itself, off its hinges and leaning in the opening
    this._iB('woodDark', PX + 0.42, FY + 0.92, DZ0 + 0.55, 0.09, 1.88, 0.82,
      { ts: 1.0, tilt: [0, 0, 1, 0.30], rotY: 0.22, k: 0.92 });

    // -------------------------------------------------------------- ceiling
    this._litSlab('intWall', x0, z0, x1, z1, CEIL, -1, 0.5, LIN);
    this._bbox('concreteB', x0, CEIL + 0.012, z0, x1, CEIL + 0.24, z1,
      { base: 0, shade: 0.64, noCast: true });
    // timber joists spanning the short way, one sagging run near the damage
    for (let z = z0 + 0.5; z < z1 - 0.15; z += 0.63) {
      const sag = z > z1 - 3.4 ? R(0.0, 0.09) : R(0.0, 0.02);
      this._iB('woodDark', (x0 + x1) * 0.5, CEIL - 0.115 - sag, z, x1 - x0, 0.20, 0.115,
        { ts: 0.95, noCast: true, k: R(0.92, 1.05) });
    }
    // spine beam on a timber post — real depth in the middle of the room
    this._iB('woodDark', -9.55, CEIL - 0.36, (z0 + z1) * 0.5, 0.28, 0.36, z1 - z0, { ts: 1.1 });
    this._iB('woodDark', -9.55, FY + (CEIL - 0.54 - FY) * 0.5, -9.05, 0.20, CEIL - 0.54 - FY, 0.20, { ts: 0.9 });
    this._iB('metal', -9.55, FY + 0.05, -9.05, 0.32, 0.10, 0.32, { ts: 0.6 });
    // conduit run + pendant lamp
    this._iB('rustDark', -10.55, CEIL - 0.06, (z0 + z1) * 0.5, 0.05, 0.05, z1 - z0 - 0.7,
      Object.assign({ ts: 0.6 }, DET));
    this._iCyl('rustDark', -10.55, CEIL - 0.34, -7.30, 0.008, 0.008, 0.52, 5,
      Object.assign({ ts: 0.4 }, DET));
    this._iCyl('glass', -10.55, CEIL - 0.66, -7.30, 0.075, 0.05, 0.16, 8,
      Object.assign({ ts: 0.4, k: 1.5 }, DET));
    // lath and plaster hanging out of the damaged corner
    for (let i = 0; i < 5; i++) {
      this._iB('woodDark', R(x1 - 2.6, x1 - 0.5), CEIL - R(0.25, 0.55), R(z1 - 2.4, z1 - 0.35),
        R(0.3, 0.9), 0.035, 0.09,
        { ts: 0.7, rotY: R(-1, 1), tilt: [1, 0, R(-1, 1), R(0.25, 0.8)], zone: 'intdet', noCast: true, k: 0.85 });
    }

    // ------------------------------------------------------- street doorway
    for (const s of [-1, 1]) {
      this._iB('woodDark', x1 + 0.17, FY + (2.05 - FY) * 0.5, DA + s * (DW * 0.5 - 0.02),
        0.34, 2.05 - FY, 0.10, { ts: 0.9 });
    }
    this._iB('woodDark', x1 + 0.17, 2.05 - 0.06, DA, 0.34, 0.12, DW + 0.16, { ts: 0.9 });
    this._iB('sand', x1 - 0.62, FY + 0.020, DA, 1.30, 0.036, DW + 0.60,
      Object.assign({ ts: 1.4 }, DET));
    // the door, wrenched off and hanging into the room
    this._iB('woodDark', x1 + 0.12, FY + 1.00, DA + 0.52, 0.06, 1.96, 0.94,
      { pivotX: x1 + 0.1, pivotZ: DA + 0.98, rotY: -0.9, zone: 'intdet', k: 0.9 });

    // ================================================================ WEST ROOM
    // the lit half: fighting position at the north window, a bench under the
    // south one, shelving on the partition, sleeping kit against the wall.
    const sbShade = this._intLum(-14.6, FY + 0.4, -4.0);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 6; i++) {
        const zz = -5.55 + i * 0.52 + ((c & 1) ? 0.26 : 0);
        this._iPlace('sandbag', -14.68 + R(-0.05, 0.05), FY + 0.02 + c * 0.20 + 0.10, zz,
          R(-0.09, 0.09), [1, 1, 1], sbShade * R(0.92, 1.08), R(-0.05, 0.05));
      }
    }
    this._iB('woodDark', -14.05, FY + 0.30, -2.85, 0.62, 0.32, 0.44, { ts: 0.8, rotY: 0.3 });
    this._iPlace('jerry', -13.95, FY, -3.55, 0.4, 1, this._intLum(-13.95, FY + 0.3, -3.55));
    // workbench under the south window
    const bx = -14.20, bz = -9.30;
    this._iB('woodDark', bx, FY + 0.80, bz, 0.78, 0.075, 2.35, { ts: 1.0 });
    this._iB('woodDark', bx, FY + 0.60, bz, 0.70, 0.05, 2.10, { ts: 1.0, k: 0.9 });
    for (const [ox, oz] of [[-0.32, -1.05], [0.32, -1.05], [-0.32, 1.05], [0.32, 1.05]]) {
      this._iB('woodDark', bx + ox, FY + 0.39, bz + oz, 0.085, 0.78, 0.085, { ts: 0.6 });
    }
    this._iB('metal', bx - 0.05, FY + 0.92, bz - 0.65, 0.42, 0.19, 0.60, { ts: 0.7 });
    this._iB('metal', bx + 0.10, FY + 0.90, bz + 0.35, 0.30, 0.15, 0.42, { ts: 0.7, rotY: 0.4 });
    this._iB('polymer', bx - 0.10, FY + 0.90, bz + 0.95, 0.22, 0.15, 0.30, { ts: 0.6, rotY: -0.3 });
    this._iPlace('bucket', -13.35, FY, -10.55, 0.7, 1, this._intLum(-13.35, FY + 0.2, -10.55));
    // shelving against the partition
    const sx = PX - 0.42;
    for (const oz of [-9.9, -8.1]) {
      this._iB('woodDark', sx, FY + 0.95, oz, 0.16, 1.90, 0.10, { ts: 0.8 });
    }
    for (let k = 0; k < 4; k++) {
      this._iB('woodDark', sx, FY + 0.30 + k * 0.50, -9.0, 0.44, 0.045, 1.84, { ts: 1.0 });
      for (let i = 0; i < 3; i++) {
        if (this._ir() < 0.35) continue;
        this._iB(this._ir() < 0.5 ? 'metal' : 'polymer', sx + R(-0.05, 0.05),
          FY + 0.30 + k * 0.50 + 0.13, -9.72 + i * 0.62 + R(-0.08, 0.08),
          R(0.24, 0.36), 0.22, R(0.28, 0.46), { ts: 0.7, rotY: R(-0.3, 0.3) });
      }
    }
    // sleeping kit + stove
    this._iB('tarpTan', -13.20, FY + 0.09, -2.90, 1.05, 0.14, 1.95, { ts: 1.3, rotY: 0.07 });
    this._iB('tarpBlue', -13.20, FY + 0.19, -2.55, 1.00, 0.09, 1.10, { ts: 1.1, rotY: 0.12, k: 0.95 });
    this._iCyl('rust', -12.35, FY + 0.44, -4.60, 0.29, 0.29, 0.88, 12, { ts: 1.9 });
    this._iCyl('rustDark', -12.35, FY + 1.60, -4.60, 0.055, 0.055, 1.45, 8, { ts: 0.8 });
    this._iB('rustDark', -12.05, FY + 2.32, -4.60, 0.65, 0.09, 0.09, { ts: 0.6 });
    this._iB('woodDark', -12.55, FY + 0.22, -6.10, 0.44, 0.42, 0.44, { ts: 0.7, rotY: 0.8, tilt: [1, 0, 0.4, 1.35] });
    this._iPlace('crate', -12.55, FY, -10.85, 0.35, 1, this._intLum(-12.55, FY + 0.3, -10.85));
    this._iPlace('crate', -12.50, FY + 0.60, -10.80, 0.95, 0.92, this._intLum(-12.5, FY + 0.9, -10.8));

    // ================================================================ EAST ROOM
    // the beam from the north window lands here — dress it so the pool has
    // something to fall across.
    const tx2 = -9.70, tz2 = -7.55;
    this._iB('woodDark', tx2, FY + 0.74, tz2, 1.55, 0.075, 0.92, { ts: 1.0, rotY: 0.22 });
    this._iB('woodDark', tx2, FY + 0.68, tz2, 1.35, 0.05, 0.72, { ts: 1.0, rotY: 0.22, k: 0.92 });
    for (const [ox, oz] of [[-0.64, -0.34], [0.64, -0.34], [-0.64, 0.34], [0.64, 0.34]]) {
      this._iB('woodDark', tx2 + ox, FY + 0.36, tz2 + oz, 0.075, 0.72, 0.075,
        { ts: 0.6, rotY: 0.22, pivotX: tx2, pivotZ: tz2 });
    }
    this._iB('metal', tx2 - 0.25, FY + 0.86, tz2 - 0.12, 0.34, 0.17, 0.26, { ts: 0.6, rotY: 0.5 });
    this._iB('polymer', tx2 + 0.35, FY + 0.83, tz2 + 0.16, 0.22, 0.11, 0.16, { ts: 0.5, rotY: -0.2 });
    // chairs
    this._iB('woodDark', -10.55, FY + 0.22, -6.45, 0.46, 0.42, 0.46,
      { ts: 0.7, rotY: 0.9, tilt: [1, 0, 0.4, 1.35] });
    this._iB('woodDark', -8.80, FY + 0.24, -8.15, 0.44, 0.05, 0.44, { ts: 0.7, rotY: -0.4 });
    for (const [ox, oz] of [[-0.17, -0.17], [0.17, -0.17], [-0.17, 0.17], [0.17, 0.17]]) {
      this._iB('woodDark', -8.80 + ox, FY + 0.11, -8.15 + oz, 0.05, 0.24, 0.05, { ts: 0.5 });
    }
    this._iB('woodDark', -8.80, FY + 0.50, -8.34, 0.44, 0.48, 0.05, { ts: 0.7 });
    // steel bed frame + mattress against the north wall
    this._iB('metal', -8.55, FY + 0.42, -3.30, 1.00, 0.06, 1.96, { ts: 1.0, rotY: 0.05 });
    this._iB('metal', -8.55, FY + 0.72, -2.36, 1.00, 0.62, 0.05, { ts: 1.0 });
    this._iB('metal', -8.55, FY + 0.62, -4.26, 1.00, 0.42, 0.05, { ts: 1.0 });
    for (const [ox, oz] of [[-0.45, -0.9], [0.45, -0.9], [-0.45, 0.9], [0.45, 0.9]]) {
      this._iB('metal', -8.55 + ox, FY + 0.21, -3.30 + oz, 0.05, 0.42, 0.05, { ts: 0.5 });
    }
    this._iB('tarpTan', -8.55, FY + 0.53, -3.45, 1.02, 0.14, 1.62, { ts: 1.3 });
    // cabinet, locker, shelf
    this._iB('woodDark', x1 - 0.36, FY + 0.86, -10.30, 0.58, 1.72, 1.05, { ts: 1.0, rotY: -0.06 });
    this._iB('woodDark', x1 - 0.36, FY + 1.75, -10.30, 0.66, 0.07, 1.12, { ts: 1.0, k: 1.05 });
    // steel locker, kept well clear of the interior camera station at
    // (-7.5, -6.0): anything within ~1.5 m of it fills the whole frame
    this._iB('metal', -10.95, FY + 0.92, -8.55, 0.52, 1.84, 0.94, { ts: 1.1, rotY: 0.09 });
    this._iB('metal', -10.69, FY + 1.05, -8.90, 0.05, 1.40, 0.42, { ts: 0.9, rotY: -0.55, pivotX: -10.69, pivotZ: -9.02 });
    this._iB('dark', -10.74, FY + 0.95, -8.50, 0.03, 1.45, 0.72, { ts: 0.9, k: 0.35 });
    this._iB('woodDark', -9.35, FY + 1.52, z1 - 0.12, 1.75, 0.05, 0.30, { ts: 1.0 });
    for (const [ox, oz] of [[-0.75, 0], [0.75, 0]]) {
      this._iB('woodDark', -9.35 + ox, FY + 1.38, z1 - 0.12 + oz, 0.06, 0.24, 0.24, { ts: 0.5 });
    }
    for (let i = 0; i < 4; i++) {
      this._iB(this._ir() < 0.5 ? 'polymer' : 'metal', -10.05 + i * 0.45, FY + 1.66, z1 - 0.16,
        R(0.22, 0.32), 0.24, R(0.20, 0.26), { ts: 0.6, rotY: R(-0.3, 0.3) });
    }
    // crates, pallet and drum stack — cover-readable, and they catch the beam
    this._iPlace('pallet', -10.65, FY + 0.01, -10.20, 0.32, 1, this._intLum(-10.65, FY + 0.2, -10.2));
    this._iPlace('crateBig', -10.65, FY + 0.14, -10.20, 0.34, 1, this._intLum(-10.65, FY + 0.5, -10.2));
    this._iPlace('crate', -10.50, FY + 0.86, -10.30, 0.9, 1, this._intLum(-10.5, FY + 1.1, -10.3));
    this._iPlace('crate', -10.90, FY + 0.02, -9.15, 0.2, 1, this._intLum(-10.9, FY + 0.3, -9.15));
    this._iPlace('barrel', -8.20, FY + 0.02, -9.55, 0.5, 1, this._intLum(-8.2, FY + 0.45, -9.55));
    this._iPlace('barrel', -8.72, FY + 0.02, -9.95, 1.2, 1, this._intLum(-8.72, FY + 0.45, -9.95));
    this._iPlace('sack', -11.05, FY + 0.01, -4.05, 0.9, 1, this._intLum(-11.05, FY + 0.3, -4.05));
    this._iPlace('sack', -11.30, FY + 0.01, -3.55, 2.1, 1.05, this._intLum(-11.3, FY + 0.3, -3.55));
    this._iPlace('basket', -10.60, FY + 0.01, -2.55, 0.6, 1, this._intLum(-10.6, FY + 0.2, -2.55));
    this._iPlace('bucket', -9.20, FY, -5.10, 0.7, 1, this._intLum(-9.2, FY + 0.15, -5.1));
    this._iPlace('jerry', -7.90, FY, -8.65, -0.5, 1, this._intLum(-7.9, FY + 0.25, -8.65));
    this._iPlace('tyre', -8.40, FY + 0.02, -6.35, 1.1, 1, this._intLum(-8.4, FY + 0.1, -6.35));

    // ------------------------------------------------- south wall: counter run
    // This is the wall the interior camera faces on its right-hand side; bare
    // it was the flattest thing in the room.
    const cz = z0 + 0.31, cxm = -10.15;
    this._iB('intWall', cxm, FY + 0.44, cz, 2.30, 0.88, 0.62, { ts: 1.4 });
    this._iB('concreteB', cxm, FY + 0.92, cz, 2.44, 0.07, 0.70, { ts: 1.2, k: 1.06 });
    this._iB('dark', cxm, FY + 0.48, cz + 0.30, 1.94, 0.62, 0.05, { ts: 1.0, k: 0.30 });
    for (let i = 0; i < 3; i++) {
      this._iB('woodDark', cxm - 0.76 + i * 0.76, FY + 0.46, cz + 0.31, 0.06, 0.74, 0.06, { ts: 0.5 });
    }
    for (let i = 0; i < 5; i++) {
      this._iCyl(this._ir() < 0.5 ? 'metal' : 'polymer', cxm - 0.90 + i * 0.46 + R(-0.06, 0.06),
        FY + 1.06, cz + R(-0.14, 0.14), R(0.07, 0.13), R(0.07, 0.13), R(0.14, 0.28), 9, { ts: 0.5 });
    }
    this._iCyl('rust', cxm, FY + 1.30, z0 + 0.10, 0.032, 0.032, 3.1, 7,
      { ts: 0.6, tilt: [0, 0, 1, Math.PI * 0.5] });
    this._iB('woodDark', cxm, FY + 1.64, z0 + 0.16, 2.10, 0.05, 0.30, { ts: 1.0 });
    this._iB('woodDark', cxm, FY + 2.06, z0 + 0.16, 2.10, 0.05, 0.30, { ts: 1.0 });
    for (const ox of [-1.02, 0, 1.02]) {
      this._iB('woodDark', cxm + ox, FY + 1.85, z0 + 0.16, 0.05, 0.46, 0.30, { ts: 0.5 });
    }
    for (let i = 0; i < 7; i++) {
      if (this._ir() < 0.25) continue;
      this._iB(this._ir() < 0.5 ? 'metal' : 'polymer', cxm - 0.92 + i * 0.30 + R(-0.05, 0.05),
        FY + (this._ir() < 0.5 ? 1.79 : 2.21), z0 + 0.16 + R(-0.05, 0.05),
        R(0.16, 0.26), 0.24, R(0.16, 0.24), { ts: 0.5, rotY: R(-0.3, 0.3) });
    }
    // hanging carpet + a wall hook rail further along the same wall
    this._iB('tarpRed', -8.55, FY + 1.52, z0 + 0.075, 1.55, 1.38, 0.05, { ts: 1.5, k: 1.0 });
    this._iB('woodDark', -8.55, FY + 2.24, z0 + 0.10, 1.72, 0.07, 0.07, { ts: 0.6 });
    this._iB('woodDark', -12.90, FY + 1.72, z0 + 0.09, 1.10, 0.07, 0.07, { ts: 0.6 });
    for (let i = 0; i < 3; i++) {
      this._iB('rustDark', -13.30 + i * 0.40, FY + 1.63, z0 + 0.12, 0.035, 0.14, 0.035, { ts: 0.4 });
    }

    // ---------------------------------------------------------- floor litter
    for (let i = 0; i < 46; i++) {
      const px = R(x0 + 0.35, x1 - 0.35), pz = R(z0 + 0.35, z1 - 0.35);
      if (Math.abs(px - PX) < 0.30) continue;
      this._iPlace('shard', px, FY + 0.014, pz, R(0, 6.28),
        [0.45 + this._ir() * 0.55, 0.28, 0.45 + this._ir() * 0.55],
        this._intLum(px, FY + 0.05, pz) * R(0.9, 1.1));
    }
    for (let i = 0; i < 26; i++) {
      const px = R(x0 + 0.3, x1 - 0.3), pz = R(z0 + 0.3, z1 - 0.3);
      if (Math.abs(px - PX) < 0.30) continue;
      this._iPlace(this._ir() < 0.55 ? 'rubbleS' : 'brickBit', px, FY + 0.012, pz, R(0, 6.28),
        0.45 + this._ir() * 0.7, this._intLum(px, FY + 0.08, pz) * R(0.9, 1.05), R(-0.3, 0.3));
    }
    // dust drift banked against the walls
    for (let i = 0; i < 10; i++) {
      const s = this._ir() < 0.5;
      this._iB('sand', s ? x0 + 0.30 : x1 - 0.30, FY + 0.022, R(z0 + 0.6, z1 - 0.6),
        R(0.35, 0.65), 0.038, R(0.7, 2.0), Object.assign({ ts: 1.5 }, DET));
    }

    // main-stream helpers, kept to a fixed count so nothing downstream shifts
    this._pockWall(PX + PT, FY + 0.35, 2.45, -8.30, 1, 0, 16, 1.5);
    this._pockWall(x1 - 0.02, FY + 0.35, 2.30, -8.60, -1, 0, 12, 1.6);
    this._rubblePile(x1 - 1.35, z1 - 1.15, 1.7, 26, true, FY + 0.01);
    this._rubblePile(x0 + 0.95, z0 + 0.75, 1.1, 12, false, FY + 0.01);
    this._clothSheet(x0 + 0.24, 2.52, -4.0, 1.15, 1.45, Math.PI * 0.5);
    this._clothSheet(x0 + 0.24, 2.52, -9.2, 1.10, 1.20, Math.PI * 0.5);

    this.spawnPoints.push({ pos: new THREE.Vector3(-9.4, FY, -6.4), yaw: 70 * Math.PI / 180 });
    this.coverPoints.push({ pos: new THREE.Vector3(tx2, FY + 0.8, tz2), dir: new THREE.Vector3(1, 0, 0), height: 0.8 });
    this.coverPoints.push({ pos: new THREE.Vector3(-10.65, FY + 0.9, -10.2), dir: new THREE.Vector3(1, 0, 0), height: 0.95 });
    this.coverPoints.push({ pos: new THREE.Vector3(-14.68, FY + 0.6, -4.3), dir: new THREE.Vector3(-1, 0, 0), height: 0.6 });
    this.coverPoints.push({ pos: new THREE.Vector3(-8.10, FY + 1.1, -5.9), dir: new THREE.Vector3(-1, 0, 0), height: 1.2 });
    this._zoneSet(prevZone);
  }

  /** the "alley" preset: narrow, laundry, AC units, pipework, debris */
  _alley() {
    const z0 = -3.9, z1 = -0.4, y = 0;
    this._zoneSet('alley');
    // ground: broken concrete slabs, a gutter channel
    this._bbox('concreteB', 7.0, y - 0.02, z0, 21.5, y + 0.04, z1, { grime: 1.5, shade: 0.92 });
    this._bbox('concrete', 7.0, y + 0.02, (z0 + z1) * 0.5 - 0.16, 21.5, y + 0.06, (z0 + z1) * 0.5 + 0.16, { grime: 1.6, shade: 0.85 });
    // a lit side yard opening in the north wall lets a shaft down onto the floor
    // pipework running down both walls
    for (const [x, z, s] of [[19.2, z0, 1], [19.2, z0, 1]]) { void x; void z; void s; }
    for (let i = 0; i < 5; i++) {
      const px = 8.4 + i * 2.5;
      this._cyl('rust', px, 4.2, z1 - 0.16, 0.055, 0.055, 8.4, 6, { base: 0, zone: 'detail' });
      this._box('metal', px, 1.5, z1 - 0.24, 0.16, 0.05, 0.16, { base: 0, zone: 'detail' });
      this._box('metal', px, 4.5, z1 - 0.24, 0.16, 0.05, 0.16, { base: 0, zone: 'detail' });
      if (i % 2 === 0) {
        this._cyl('rust', px + 0.9, 2.55, z0 + 0.14, 0.04, 0.04, 5.1, 6, { base: 0, zone: 'detail' });
      }
    }
    // horizontal pipe run + valves
    this._cyl('rust', 14.0, 2.35, z0 + 0.2, 0.06, 0.06, 12.6, 6,
      { base: 0, zone: 'detail', tilt: [0, 0, 1, Math.PI * 0.5], rotY: Math.PI * 0.5 });
    for (const px of [10.5, 15.8]) this._box('metal', px, 2.35, z0 + 0.2, 0.18, 0.22, 0.18, { base: 0, zone: 'detail' });
    // AC units + brackets on the shaded wall
    for (const [px, py] of [[9.6, 3.4], [13.1, 5.9], [16.4, 3.2], [11.8, 8.4]]) {
      this._acUnit({ mat: 'plasterA' }, { ax: 'x', p: z1, sgn: -1, T: T_WALL }, px, py);
    }
    // laundry lines across the alley
    for (let i = 0; i < 4; i++) {
      const px = 9.2 + i * 2.9, py = 3.6 + (i % 2) * 1.5;
      this._wire(px, py, z0 + 0.1, px, py - 0.05, z1 - 0.1, 0.18);
      const n = 2 + this._ri(0, 2);
      for (let k = 0; k < n; k++) {
        const t = 0.2 + (k + 0.5) / (n + 0.2) * 0.6;
        const cz = z0 + (z1 - z0) * t;
        this._clothSheet(px, py - 0.28 - Math.sin(Math.PI * t) * 0.16, cz,
          this._rr(0.5, 0.9), this._rr(0.6, 1.15), Math.PI * 0.5);
      }
    }
    // debris, dumpster, tyres, crates
    this._box('rustDark', 17.4, 0.55, z0 + 0.85, 1.7, 1.05, 1.05, { rotY: 0.06, grime: 1.5 });
    this._box('rustDark', 17.4, 1.12, z0 + 0.85, 1.76, 0.10, 1.1, { rotY: 0.06, base: 0, shade: 0.9 });
    this._stain(17.4, z0 + 0.85, 1.4, 0.6);
    for (let i = 0; i < 5; i++) {
      this._iPlace('tyre', 10.2 + this._rr(-0.2, 0.2), 0.02 + i * 0.20, z0 + 0.72 + this._rr(-0.15, 0.15),
        this._r() * 6.28, 1, 0.8 + this._r() * 0.2, this._rr(-0.06, 0.06));
    }
    this._iPlace('barrel', 12.4, 0.02, z1 - 0.62, 0.4, 1, 0.85);
    this._iPlace('barrel', 12.95, 0.02, z1 - 0.55, 1.1, 1, 0.78);
    this._iPlace('crate', 15.1, 0.02, z0 + 0.6, 0.25, 1, 0.85);
    this._iPlace('crate', 15.15, 0.62, z0 + 0.66, 0.9, 0.92, 0.9);
    this._iPlace('crateBig', 20.2, 0.02, z0 + 0.9, -0.2, 1, 0.85);
    this._rubblePile(19.0, z1 - 0.8, 1.3, 18, false, 0.04);
    this._rubblePile(8.2, z0 + 0.7, 1.0, 12, false, 0.04);
    this._pockWall(0, 0.3, 3.2, z1, 0, -1, 40, 6.0);
    this._pockWall(0, 0.3, 2.6, z0, 0, 1, 26, 6.0);
    for (let i = 0; i < 26; i++) {
      this._iPlace('shard', this._rr(7.4, 21.2), 0.02, this._rr(z0 + 0.2, z1 - 0.2),
        this._r() * 6.28, 0.7 + this._r() * 0.8, 0.85 + this._r() * 0.2);
    }
    this.spawnPoints.push({ pos: new THREE.Vector3(11.0, 0, -2.1), yaw: 250 * Math.PI / 180 });
    this._zoneSet('props');
  }

  /** hanging cloth with a wind-animated vertex weight */
  _clothSheet(cx, cy, cz, w, h, ry) {
    const mats = ['clothA', 'clothB', 'clothC', 'clothD'];
    const m = mats[this._ri(0, 3)];
    const g = new THREE.PlaneGeometry(w, h, 3, 5);
    const p = g.attributes.position;
    const wind = new Float32Array(p.count);
    for (let i = 0; i < p.count; i++) {
      const t = 0.5 - p.getY(i) / h;                       // 0 at the line, 1 at the hem
      wind[i] = t * t;
      p.setZ(i, p.getZ(i) + Math.sin(p.getX(i) * 5.0) * 0.035 * t);
    }
    g.computeVertexNormals();
    scaleUv(g, w / 1.15, h / 1.15);
    g.setAttribute('aWind', new THREE.BufferAttribute(wind, 1));
    g.translate(cx, cy - h * 0.5, cz);
    if (ry) rotYg(g, cx, cz, ry);
    return this._add(m, g, { zone: 'cloth', noCollide: true, base: cy - h });
  }

  // ---------------------------------------------------------------- plaza

  _plaza() {
    this._zoneSet('plaza');
    // concrete apron ringing the plaza + a raised planter kerb
    this._bbox('concreteB', -21.0, -0.02, 23.6, 5.6, 0.05, 39.4, { grime: 1.3, shade: 0.97 });
    for (let x = -20.4; x < 5.2; x += 2.1) {
      this._box('concrete', x, 0.09, 23.7, 2.05, 0.18, 0.34, { grime: 1.5 });
    }
    this._fountain(-7.4, 31.6);
    // market stalls, two rows leaving a firing lane down the middle
    const stalls = [
      [-19.4, 25.6, 0.05], [-19.6, 29.2, -0.09], [-19.2, 32.9, 0.12], [-19.8, 36.4, 0.02],
      [-13.4, 36.9, 3.24], [-9.0, 37.1, 3.10], [-4.4, 36.8, 3.30],
      [1.8, 27.4, 1.60], [1.9, 31.2, 1.52],
    ];
    for (const [x, z, r] of stalls) this._stall(x, z, r);
    // sandbag emplacement covering the street mouth
    this._sandbagWall(1.6, 24.9, 0.16, 4.2, 3);
    this._sandbagWall(-0.9, 26.6, 1.45, 2.6, 3);
    this._iPlace('crateBig', 0.2, 0.02, 25.9, 0.2, 1, 0.9);
    this._iPlace('crate', 1.5, 0.02, 26.2, -0.5, 1, 0.88);
    // barrels / tyres / crate clusters
    const clusters = [[-11.8, 25.4], [-4.2, 27.9], [-14.6, 34.8], [3.4, 34.2], [-8.8, 38.2]];
    for (const [cx, cz] of clusters) this._propCluster(cx, cz);
    // palms for a spot of cool green against all the tan
    this._palm(-2.2, 34.6, 4.4);
    this._palm(-18.2, 22.0, 3.9);
    this._palm(4.2, 21.0, 4.1);
    // rubble from the blown corner of the north block spilling south
    this._rubblePile(-6.0, 39.0, 3.4, 44, true);
    this._rubblePile(-19.6, 23.2, 2.4, 26, true);
    this._rubblePile(4.8, 19.4, 2.0, 20, true);
    this.spawnPoints.push({ pos: new THREE.Vector3(-9.5, 0, 30.0), yaw: 200 * Math.PI / 180 });
    this.spawnPoints.push({ pos: new THREE.Vector3(2.0, 0, 35.5), yaw: 175 * Math.PI / 180 });
    this._zoneSet('props');
  }

  _fountain(cx, cz) {
    const R = 2.35, wallH = 0.86;
    const N = 8;
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2 + Math.PI / N;
      const w = 2 * R * Math.tan(Math.PI / N);
      const px = cx + Math.cos(a) * R, pz = cz + Math.sin(a) * R;
      const broken = this._r() < 0.22;
      const h = broken ? wallH * this._rr(0.35, 0.7) : wallH;
      this._box('concrete', px, h * 0.5, pz, 0.30, h, w + 0.04, { rotY: -a, grime: 1.5 });
      if (!broken) {
        this._box('concreteB', px, h + 0.055, pz, 0.44, 0.11, w + 0.06, { rotY: -a, base: 0, shade: 1.06 });
      } else {
        this._rebar(px, h, pz, 2);
      }
    }
    this._cyl('concreteB', cx, 0.05, cz, R + 0.05, R + 0.14, 0.16, N * 2, { grime: 1.5 });
    // dry, cracked basin floor with drifted sand
    this._cyl('sand', cx, 0.14, cz, R - 0.22, R - 0.22, 0.14, N * 2, { grime: 1.6, shade: 0.94 });
    // central pedestal + broken spout
    this._cyl('concrete', cx, 0.36, cz, 0.46, 0.58, 0.62, 10, { base: 0.14 });
    this._cyl('concrete', cx, 0.78, cz, 0.20, 0.30, 0.30, 10, { base: 0.14 });
    this._cyl('concrete', cx, 1.02, cz, 0.62, 0.16, 0.22, 12, { base: 0.14, shade: 1.05 });
    this._cyl('rust', cx + 0.1, 1.22, cz, 0.05, 0.06, 0.34, 6, { base: 0.14, tilt: [1, 0, 0.4, 0.5], zone: 'detail' });
    for (let i = 0; i < 12; i++) {
      const a = this._r() * 6.28, d = this._r() * (R - 0.5);
      this._iPlace('rubbleS', cx + Math.cos(a) * d, 0.16, cz + Math.sin(a) * d, this._r() * 6.28,
        0.6 + this._r() * 0.5, 0.9);
    }
    this._stain(cx, cz, R + 1.4, 0.4);
    this._reserve(cx, cz, R + 0.55);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, 0.9, cz - R), dir: new THREE.Vector3(0, 0, -1), height: 0.9 });
    this.coverPoints.push({ pos: new THREE.Vector3(cx, 0.9, cz + R), dir: new THREE.Vector3(0, 0, 1), height: 0.9 });
  }

  _stall(cx, cz, rot) {
    const sp = this._findSpot(cx, cz, 1.45, 2.2); if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 1.65);
    const w = 2.5, d = 1.7, h = 2.25;
    const legs = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
    for (const [sx, sz] of legs) {
      this._box('wood', cx + sx * w * 0.5, h * 0.5, cz + sz * d * 0.5, 0.085, h, 0.085,
        { rotY: rot, pivotX: cx, pivotZ: cz, grime: 1.4 });
    }
    // frame
    this._box('wood', cx, h, cz - d * 0.5, w + 0.1, 0.075, 0.075, { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    this._box('wood', cx, h, cz + d * 0.5, w + 0.1, 0.075, 0.075, { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    this._box('wood', cx - w * 0.5, h, cz, 0.075, 0.075, d, { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    this._box('wood', cx + w * 0.5, h, cz, 0.075, 0.075, d, { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    // sagging canopy
    this._tarp(cx, h + 0.10, cz, w + 0.55, d + 0.5, 0.24, rot);
    // counter + goods
    this._box('wood', cx, 0.86, cz + d * 0.18, w - 0.15, 0.07, d * 0.6,
      { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    this._box('woodDark', cx, 0.43, cz + d * 0.18, w - 0.35, 0.78, 0.08,
      { rotY: rot, pivotX: cx, pivotZ: cz });
    const cr = this._ri(2, 4);
    for (let i = 0; i < cr; i++) {
      const t = (i + 0.5) / cr - 0.5;
      const lx = cx + Math.cos(rot) * (t * (w - 0.6)) - Math.sin(rot) * (-d * 0.24);
      const lz = cz + Math.sin(rot) * (t * (w - 0.6)) + Math.cos(rot) * (-d * 0.24);
      this._iPlace('crate', lx, 0.02, lz, rot + this._rr(-0.2, 0.2), 1, 0.86 + this._r() * 0.2);
      if (this._r() < 0.5) this._iPlace('crate', lx, 0.62, lz, rot + this._rr(-0.3, 0.3), 0.92, 0.9);
    }
    // hanging cloth on the shaded back edge
    if (this._r() < 0.8) {
      const bx = cx - Math.sin(rot) * (d * 0.5), bz = cz + Math.cos(rot) * (d * 0.5);
      this._clothSheet(bx, h - 0.12, bz, 1.0, 1.25, rot);
    }
    this._stain(cx, cz, 2.0, 0.45);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, 1.0, cz), dir: new THREE.Vector3(Math.sin(rot), 0, Math.cos(rot)), height: 1.0 });
  }

  _tarp(cx, cy, cz, w, d, sag, rot) {
    const mats = ['tarpRed', 'tarpBlue', 'tarpGreen', 'tarpTan'];
    const m = mats[this._ri(0, 3)];
    const g = new THREE.PlaneGeometry(w, d, 6, 5);
    g.rotateX(-Math.PI * 0.5);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const u = p.getX(i) / w + 0.5, v = p.getZ(i) / d + 0.5;
      p.setY(i, p.getY(i) - Math.sin(Math.PI * u) * Math.sin(Math.PI * v) * sag);
    }
    g.computeVertexNormals();
    scaleUv(g, w / 1.35, d / 1.35);
    g.translate(cx, cy, cz);
    if (rot) rotYg(g, cx, cz, rot);
    return this._add(m, g, { zone: 'detail', noCollide: true, base: 0 });
  }

  _sandbagWall(cx, cz, ang, len, courses) {
    const bw = 0.56, ch = 0.20;
    const n = Math.max(2, Math.round(len / bw));
    for (let c = 0; c < courses; c++) {
      const off = (c & 1) ? bw * 0.5 : 0;
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) * bw + off - len * 0.5;
        const x = cx + Math.cos(ang) * t, z = cz + Math.sin(ang) * t;
        this._iPlace('sandbag', x, 0.02 + c * ch + ch * 0.5, z,
          -ang + this._rr(-0.09, 0.09), [1, 1, 1], 0.86 + this._r() * 0.22, this._rr(-0.05, 0.05));
      }
    }
    this._stain(cx, cz, len * 0.7, 0.45);
    this._reserve(cx, cz, len * 0.5 + 0.25);
    this.coverPoints.push({
      pos: new THREE.Vector3(cx, courses * ch, cz),
      dir: new THREE.Vector3(Math.sin(ang), 0, -Math.cos(ang)), height: courses * ch,
    });
  }

  _propCluster(cx, cz) {
    const k = this._ri(0, 2);
    if (k === 0) {
      for (let i = 0; i < 4; i++) {
        this._iPlace('barrel', cx + this._rr(-0.7, 0.7), 0.02, cz + this._rr(-0.7, 0.7),
          this._r() * 6.28, 1, 0.82 + this._r() * 0.25);
      }
    } else if (k === 1) {
      const st = this._ri(3, 6);
      for (let i = 0; i < st; i++) {
        this._iPlace('tyre', cx + this._rr(-0.09, 0.09), 0.02 + i * 0.205, cz + this._rr(-0.09, 0.09),
          this._r() * 6.28, 1, 0.78 + this._r() * 0.2, this._rr(-0.04, 0.04));
      }
      this._iPlace('tyre', cx + 0.9, 0.02, cz - 0.5, this._r() * 6.28, 1, 0.85, 1.5, [1, 0, 0.2]);
    } else {
      this._iPlace('crateBig', cx, 0.02, cz, this._r() * 6.28, 1, 0.88);
      this._iPlace('crate', cx + 0.3, 0.74, cz - 0.1, this._r() * 6.28, 1, 0.9);
      this._iPlace('crate', cx - 0.75, 0.02, cz + 0.5, this._r() * 6.28, 1, 0.86);
      this._iPlace('jerry', cx + 0.95, 0.02, cz + 0.55, this._r() * 6.28, 1, 0.85);
    }
    this._stain(cx, cz, 1.5, 0.5);
    this._reserve(cx, cz, 1.35);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, 0.9, cz), dir: new THREE.Vector3(0, 0, -1), height: 0.9 });
  }

  _palm(cx, cz, h) {
    const seg = 5;
    for (let i = 0; i < seg; i++) {
      const t = i / seg;
      const r = 0.19 - t * 0.07;
      this._cyl('woodDark', cx + Math.sin(t * 2.2) * 0.22, 0.1 + h * (t + 0.5 / seg), cz,
        r, r + 0.02, h / seg + 0.02, 8, { base: 0, ts: 0.6 });
    }
    const top = 0.1 + h;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2 + this._rr(-0.2, 0.2);
      const droop = this._rr(0.5, 1.0);
      const g = new THREE.PlaneGeometry(0.44, 2.3, 1, 3);
      const p = g.attributes.position;
      for (let k = 0; k < p.count; k++) {
        const v = p.getY(k) / 2.3 + 0.5;
        p.setZ(k, p.getZ(k) - v * v * droop * 1.5);
        p.setX(k, p.getX(k) * (1.15 - v * 0.75));
      }
      g.computeVertexNormals();
      scaleUv(g, 0.4, 2.0);
      g.rotateX(-Math.PI * 0.5 + 0.45);
      g.translate(0, 0, -1.15);
      g.rotateY(a);
      g.translate(cx + Math.sin(1.0) * 0.22, top, cz);
      this._add('foliage', g, { zone: 'detail', noCollide: true, base: 0, shade: 0.95 });
    }
    this._stain(cx, cz, 1.1, 0.35);
    this._reserve(cx, cz, 1.0);
  }

  // ------------------------------------------------------- hero vehicles

  /** shot-up technical: the plaza's focal prop */
  _technical(cx, cz, rot) {
    const zn = 'vehicle';
    this._zoneSet(zn);
    const B = (mat, x, y, z, w, h, d, o) => this._box(mat, cx + x, y, cz + z, w, h, d,
      Object.assign({ rotY: rot, pivotX: cx, pivotZ: cz, base: 0 }, o || {}));
    const body = 'rustDark';
    // chassis rails + bumper
    B(body, 0, 0.52, 0, 4.85, 0.20, 1.78, { grime: 1.3 });
    B('metal', 0, 0.30, 0, 4.2, 0.14, 1.55, { shade: 0.6 });
    B('rust', -2.42, 0.62, 0, 0.22, 0.36, 1.86, {});
    B('rust', -2.62, 0.86, 0, 0.20, 0.95, 1.70, { tilt: [0, 0, 1, 0.12] });   // bull bar
    // bonnet + cab
    B(body, -1.55, 0.86, 0, 1.72, 0.50, 1.80, {});
    B(body, -1.55, 1.13, 0, 1.60, 0.06, 1.66, { shade: 1.05 });
    B(body, -0.30, 1.28, 0, 1.82, 1.34, 1.80, {});
    B(body, -0.30, 2.02, 0, 1.72, 0.10, 1.72, { shade: 1.05 });
    B('dark', -1.05, 1.62, 0, 0.10, 0.62, 1.52, { raw: true, shade: 0.32 });   // blown windscreen
    B('dark', -0.30, 1.62, 0.88, 1.60, 0.62, 0.10, { raw: true, shade: 0.32 });
    B('dark', -0.30, 1.62, -0.88, 1.60, 0.62, 0.10, { raw: true, shade: 0.32 });
    for (const s of [-1, 1]) {
      B(body, -1.08, 1.62, s * 0.86, 0.09, 0.66, 0.10, {});                    // A pillars
      B(body, 0.55, 1.62, s * 0.86, 0.09, 0.66, 0.10, {});
      B(body, -0.30, 1.62, s * 0.90, 1.70, 0.06, 0.06, {});
    }
    // open driver's door
    const dg = this._box(body, cx - 0.42, 1.05, cz + 0.94, 1.05, 1.02, 0.08,
      { rotY: rot, pivotX: cx, pivotZ: cz, base: 0 });
    void dg;
    // flatbed
    B(body, 1.42, 0.98, 0, 2.5, 0.10, 1.80, {});
    B(body, 1.42, 1.24, 0.88, 2.5, 0.55, 0.09, {});
    B(body, 1.42, 1.24, -0.88, 2.5, 0.55, 0.09, {});
    B(body, 2.65, 1.10, 0, 0.09, 0.30, 1.80, { tilt: [0, 0, 1, -0.9] });        // tailgate down
    for (let i = 0; i < 4; i++) {
      B('rust', 0.4 + i * 0.62, 1.62, 0.90, 0.06, 0.85, 0.06, {});
      B('rust', 0.4 + i * 0.62, 1.62, -0.90, 0.06, 0.85, 0.06, {});
    }
    B('rust', 1.42, 2.03, 0.90, 2.5, 0.05, 0.05, {});
    B('rust', 1.42, 2.03, -0.90, 2.5, 0.05, 0.05, {});
    // pintle-mounted heavy machine gun
    this._cyl('gunmetal', cx + Math.cos(rot) * 1.35, 1.42, cz + Math.sin(rot) * 1.35, 0.055, 0.075, 0.78, 8, { base: 0 });
    B('gunmetal', 1.35, 1.92, 0, 0.85, 0.20, 0.16, {});
    B('gunmetal', 1.05, 2.05, 0, 0.30, 0.16, 0.14, {});
    this._cyl('gunmetal', cx + Math.cos(rot) * 2.15, 1.94, cz + Math.sin(rot) * 2.15, 0.032, 0.038, 1.25, 8,
      { base: 0, tilt: [0, 0, 1, Math.PI * 0.5 - 0.06], rotY: rot });
    B('gunmetal', 1.62, 1.98, 0, 0.06, 0.55, 0.72, {});                         // shield
    B('metal', 1.20, 1.72, 0.26, 0.30, 0.26, 0.20, {});                         // ammo can
    // wheels — two shredded flat
    const wz = 0.92, wx = [-1.62, 1.55];
    for (let i = 0; i < 2; i++) for (const s of [-1, 1]) {
      const flat = (i === 1 && s < 0) || (i === 0 && s > 0);
      const px = cx + Math.cos(rot) * wx[i] - Math.sin(rot) * (s * wz);
      const pz = cz + Math.sin(rot) * wx[i] + Math.cos(rot) * (s * wz);
      const r = 0.40, hh = flat ? 0.26 : 0.40;
      this._cyl('rubber', px, hh, pz, r, r, 0.30, 12,
        { base: 0, tilt: [1, 0, 0, Math.PI * 0.5], rotY: rot, shade: flat ? 0.8 : 1 });
      this._cyl('metal', px, hh, pz, r * 0.55, r * 0.55, 0.33, 10,
        { base: 0, tilt: [1, 0, 0, Math.PI * 0.5], rotY: rot, shade: 0.9 });
      B(body, wx[i], hh + r * 0.55, s * wz, 1.05, 0.14, 0.42, {});               // arch
    }
    // damage
    this._pockWall(cx - Math.sin(rot) * 0.95, 0.7, 1.9, cz + Math.cos(rot) * 0.95,
      -Math.sin(rot), Math.cos(rot), 22, 1.9);
    this._pockWall(cx + Math.sin(rot) * 0.95, 0.7, 1.9, cz - Math.cos(rot) * 0.95,
      Math.sin(rot), -Math.cos(rot), 14, 1.9);
    this._stain(cx, cz, 3.4, 0.62);
    this._rubblePile(cx + 2.6, cz - 1.2, 1.2, 10, false);
    this._reserve(cx, cz, 2.9);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, 1.1, cz), dir: new THREE.Vector3(-Math.sin(rot), 0, Math.cos(rot)), height: 1.2 });
    this._zoneSet('props');
  }

  /**
   * Burnt-out civilian sedan husk. A real silhouette: sills, wheel arches,
   * bonnet, boot, six pillars, blown glass, two shredded tyres, one door
   * hanging open. This is the workhorse floor-breaker in the street frames.
   */
  _wreck(cx, cz, rot) {
    const prevZone = this._zone;
    this._zoneSet('vehicle');
    const gy = this._surfY(cx, cz);
    const B = (mat, x, y, z, w, h, d, o) => this._box(mat, cx + x, gy + y, cz + z, w, h, d,
      Object.assign({ rotY: rot, pivotX: cx, pivotZ: cz, base: gy, shade: 0.68 }, o || {}));
    const M = 'burnt';

    // floor pan, sills, lower body
    B(M, 0, 0.30, 0, 4.02, 0.14, 1.58, { grime: 1.4 });
    for (const s of [-1, 1]) B(M, 0, 0.44, s * 0.80, 3.60, 0.30, 0.10, {});
    B(M, 0, 0.60, 0, 4.12, 0.42, 1.66, {});
    B(M, 0, 0.83, 0, 4.06, 0.09, 1.60, { shade: 0.80 });               // waistline crease
    // bonnet (buckled open at the back edge) + boot lid
    B(M, -1.42, 0.91, 0, 1.24, 0.09, 1.54, { shade: 0.86 });
    B(M, -0.92, 1.06, 0, 0.70, 0.07, 1.42, { tilt: [0, 0, 1, -0.34], shade: 0.92 });
    B(M, 1.56, 0.94, 0, 1.00, 0.09, 1.50, { shade: 0.86 });
    // bumpers, grille, light buckets
    B('rustDark', -2.06, 0.57, 0, 0.20, 0.34, 1.64, {});
    B('rustDark', 2.06, 0.59, 0, 0.20, 0.32, 1.60, {});
    B('dark', -2.10, 0.74, 0, 0.09, 0.22, 1.14, { raw: true, shade: 0.30, noCast: true });
    for (const s of [-1, 1]) {
      B('dark', -2.08, 0.86, s * 0.60, 0.10, 0.19, 0.32, { raw: true, shade: 0.34, noCast: true });
      B('dark', 2.08, 0.86, s * 0.58, 0.10, 0.17, 0.30, { raw: true, shade: 0.34, noCast: true });
    }
    // cabin: charred void, pillars, roof skin, rails
    B('dark', 0.12, 1.20, 0, 2.06, 0.62, 1.40, { raw: true, shade: 0.20, noCast: true });
    B(M, 0.12, 1.54, 0, 1.94, 0.08, 1.44, { shade: 0.88 });
    for (const s of [-1, 1]) {
      B(M, -0.90, 1.22, s * 0.71, 0.11, 0.64, 0.10, { tilt: [0, 0, 1, 0.30] });
      B(M, 0.26, 1.22, s * 0.72, 0.08, 0.64, 0.10, {});
      B(M, 1.06, 1.22, s * 0.69, 0.12, 0.64, 0.10, { tilt: [0, 0, 1, -0.26] });
      B(M, 0.12, 1.22, s * 0.74, 2.02, 0.07, 0.08, {});
      B(M, 0.12, 1.50, s * 0.72, 1.96, 0.07, 0.09, { shade: 0.82 });
      B(M, -1.32, 0.76, s * 0.80, 1.00, 0.15, 0.13, {});                // front arch
      B(M, 1.30, 0.76, s * 0.80, 0.98, 0.15, 0.13, {});                 // rear arch
      B(M, -0.62, 0.60, s * 0.84, 0.34, 0.26, 0.06, { shade: 0.78 });   // mirror stalk
    }
    // rear door still on its hinge, driver's door swung wide open
    B(M, 0.62, 0.72, 0.86, 1.00, 0.80, 0.07, { shade: 0.62 });
    {
      const th = 0.92, hx = -1.05, hz = 0.86, dl = 0.98;
      const lx = hx + dl * 0.5 * Math.cos(th), lz = hz + dl * 0.5 * Math.sin(th);
      const wx = cx + lx * Math.cos(rot) + lz * Math.sin(rot);
      const wz = cz - lx * Math.sin(rot) + lz * Math.cos(rot);
      this._box(M, wx, gy + 0.74, wz, dl, 0.84, 0.07,
        { rotY: rot - th, base: gy, shade: 0.66 });
      this._box('dark', wx, gy + 1.14, wz, dl - 0.14, 0.30, 0.05,
        { rotY: rot - th, base: gy, raw: true, shade: 0.24, noCast: true });
    }
    // wheels — two burnt down to the rim
    for (let i = 0; i < 2; i++) for (const s of [-1, 1]) {
      const flat = (i === 0 && s > 0) || (i === 1 && s < 0);
      const wx = -1.32 + i * 2.62;
      const px = cx + Math.cos(rot) * wx - Math.sin(rot) * (s * 0.79);
      const pz = cz + Math.sin(rot) * wx + Math.cos(rot) * (s * 0.79);
      if (!flat) {
        this._cyl('rubber', px, gy + 0.32, pz, 0.32, 0.32, 0.24, 11,
          { base: gy, tilt: [1, 0, 0, Math.PI * 0.5], rotY: rot, shade: 0.55 });
      }
      this._cyl('metal', px, gy + (flat ? 0.20 : 0.32), pz, 0.205, 0.205, 0.26, 10,
        { base: gy, tilt: [1, 0, 0, Math.PI * 0.5], rotY: rot, shade: flat ? 0.62 : 0.78 });
    }
    // scorch, glass litter and blast damage
    this._pockWall(cx - Math.sin(rot) * 0.86, gy + 0.55, gy + 1.45, cz + Math.cos(rot) * 0.86,
      -Math.sin(rot), Math.cos(rot), 12, 1.6);
    for (let i = 0; i < 16; i++) {
      const a = this._r() * 6.28, d = 1.6 + this._r() * 2.2;
      this._iPlace('shard', cx + Math.cos(a) * d, gy + 0.014, cz + Math.sin(a) * d,
        this._r() * 6.28, [0.4 + this._r() * 0.4, 0.25, 0.4 + this._r() * 0.4], 0.9);
    }
    this._rubblePile(cx + Math.cos(rot + 1.9) * 2.5, cz + Math.sin(rot + 1.9) * 2.5, 1.0, 8, false);
    this._stain(cx, cz, 3.2, 0.68);
    this._reserve(cx, cz, 2.5);
    this.coverPoints.push({
      pos: new THREE.Vector3(cx, gy + 1.0, cz),
      dir: new THREE.Vector3(-Math.sin(rot), 0, Math.cos(rot)), height: 1.1,
    });
    this._zoneSet(prevZone);
  }

  // =====================================================================
  //  SET DRESSING
  //  (a) a spoil/rubble skirt everywhere a wall meets the floor,
  //  (b) gameplay-scale props clustered into readable cover lines,
  //  (c) hard-surface breakup of the floor plane itself.
  //  Nothing here may sit on a spawn, a doorway, or an existing hero prop —
  //  every placement goes through _clearAt / _reserve.
  // =====================================================================

  /**
   * Dirt-and-rubble skirt along a wall/ground intersection.
   * (x0,z0)->(x1,z1) is the wall base line, (nx,nz) points away from the wall.
   * Emits a three-row berm strip (the ground locally lifts 5-15 cm against the
   * wall so the seam is buried) plus 3-6 loose chunks per 4 m of run.
   */
  _skirtRun(x0, z0, x1, z1, nx, nz, o) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    if (len < 0.7) return;
    const mode = (o && o.mode) || 'surf';
    const wMax = (o && o.w) || 1.02;
    const hMax = (o && o.h) || 0.155;
    const inner = (o && o.inner !== undefined) ? o.inner : -0.28;
    const dens = (o && o.dens !== undefined) ? o.dens : 1;
    const n = Math.max(2, Math.round(len / 1.05));
    const rows = 3, vw = n + 1, cnt = vw * rows;
    const pos = new Float32Array(cnt * 3);
    const uvs = new Float32Array(cnt * 2);
    const col = new Float32Array(cnt * 3);
    const ts = this._tsz.dirt || 3.0;

    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const bx = x0 + (x1 - x0) * t, bz = z0 + (z1 - z0) * t;
      const w = wMax * (0.40 + 0.82 * vnoise(bx * 0.62 + 4.3, 2.5, bz * 0.62));
      const h = hMax * (0.32 + 0.86 * vnoise(bx * 0.47 + 11.9, 6.5, bz * 0.47));
      for (let r = 0; r < rows; r++) {
        const d = r === 0 ? inner : (r === 1 ? w * 0.40 : w);
        const X = bx + nx * d, Z = bz + nz * d;
        const g0 = this._surfY(X, Z, mode);
        const Y = r === 0 ? g0 + h : (r === 1 ? g0 + h * 0.58 : g0 + 0.012);
        const k = r * vw + i;
        pos[k * 3] = X; pos[k * 3 + 1] = Y; pos[k * 3 + 2] = Z;
        uvs[k * 2] = X / ts; uvs[k * 2 + 1] = Z / ts;
        const s = this._groundShade(X, Z) * (r === 0 ? 0.79 : r === 1 ? 0.88 : 0.97);
        col[k * 3] = clamp(s * 1.025, 0, 1.6);
        col[k * 3 + 1] = clamp(s * 0.985, 0, 1.6);
        col[k * 3 + 2] = clamp(s * 0.920, 0, 1.6);
      }
    }
    const idx = new Uint16Array(n * (rows - 1) * 6);
    let p = 0;
    for (let r = 0; r < rows - 1; r++) {
      for (let i = 0; i < n; i++) {
        const a = r * vw + i, b = a + 1, c = a + vw, d2 = c + 1;
        idx[p++] = a; idx[p++] = c; idx[p++] = b;
        idx[p++] = b; idx[p++] = c; idx[p++] = d2;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    // winding depends on which way the run travels — force normals up
    const na = geo.attributes.normal.array;
    let sy = 0;
    for (let i = 1; i < na.length; i += 3) sy += na[i];
    if (sy < 0) {
      const ix = geo.index.array;
      for (let i = 0; i < ix.length; i += 3) { const q = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = q; }
      geo.computeVertexNormals();
    }
    this._add('dirt', geo, { zone: 'skirt', raw: true, noCollide: true, noCast: true });

    // loose chunks broken off the wall above
    const chunks = Math.max(2, Math.round(len * 0.25 * this._rr(3, 5.2) * dens));
    for (let i = 0; i < chunks; i++) {
      const t = this._r();
      const bx = x0 + (x1 - x0) * t, bz = z0 + (z1 - z0) * t;
      const d = this._rr(-0.06, wMax * 1.55);
      const X = bx + nx * d + nz * this._rr(-0.35, 0.35);
      const Z = bz + nz * d - nx * this._rr(-0.35, 0.35);
      const gy = this._surfY(X, Z, mode);
      const ry = this._r() * 6.28, tl = this._rr(-0.32, 0.32);
      const k = this._r();
      if (k < 0.40) this._iPlace('rubbleS', X, gy + 0.015, Z, ry, 0.55 + this._r() * 0.85, 0.78 + this._r() * 0.26, tl);
      else if (k < 0.60) this._iPlace('brickBit', X, gy + 0.012, Z, ry, 0.8 + this._r() * 0.9, 0.84 + this._r() * 0.26, tl);
      else if (k < 0.74) this._iPlace('rubbleL', X, gy - 0.07, Z, ry, 0.55 + this._r() * 0.7, 0.80 + this._r() * 0.22, tl * 0.8);
      else if (k < 0.86) this._iPlace('cinder', X, gy + 0.005, Z, ry, 1, 0.86 + this._r() * 0.2, tl * 0.5);
      else this._iPlace('shard', X, gy + 0.010, Z, ry, 0.6 + this._r() * 0.9, 0.86 + this._r() * 0.2);
    }
  }

  /** every wall in the playable compound gets grounded */
  _dressWallBases() {
    const prev = this._zone;
    this._zoneSet('skirt');
    const E = 0.15;
    for (const b of (this._blds || [])) {
      if (b.id === 'O') continue;                    // hazed outskirts, not worth it
      const x0 = b.x0 - E, x1 = b.x1 + E, z0 = b.z0 - E, z1 = b.z1 + E;
      this._skirtRun(x0, z1, x1, z1, 0, 1, null);
      this._skirtRun(x1, z0, x0, z0, 0, -1, null);
      this._skirtRun(x1, z1, x1, z0, 1, 0, null);
      this._skirtRun(x0, z0, x0, z1, -1, 0, null);
    }
    const H = this._hall;
    if (H) {
      const o = { mode: 'core' };
      this._skirtRun(H.x0 - 0.22, H.z1 + 0.22, H.x1 + 0.22, H.z1 + 0.22, 0, 1, o);
      this._skirtRun(H.x1 + 0.22, H.z0 - 0.22, H.x0 - 0.22, H.z0 - 0.22, 0, -1, o);
      this._skirtRun(H.x1 + 0.22, H.z1 + 0.22, H.x1 + 0.22, H.z0 - 0.22, 1, 0, o);
      this._skirtRun(H.x0 - 0.22, H.z0 - 0.22, H.x0 - 0.22, H.z1 + 0.22, -1, 0, o);
    }
    // arcade colonnade footing line, gate piers, compound walls
    this._skirtRun(6.02, 25.0, 6.02, 38.4, -1, 0, { w: 0.78, h: 0.12 });
    this._skirtRun(-4.42, 40.2, -4.42, 43.2, -1, 0, { w: 0.85, h: 0.14 });
    this._skirtRun(3.22, 40.2, 3.22, 43.2, 1, 0, { w: 0.85, h: 0.14 });
    this._skirtRun(-4.4, 40.25, 3.2, 40.25, 0, -1, { w: 0.7, h: 0.11, dens: 0.6 });
    this._skirtRun(-27.32, -12.8, -27.32, 23.0, 1, 0, { w: 0.88, h: 0.13, mode: 'core', dens: 0.55 });
    this._skirtRun(-22.22, 23.2, -22.22, 47.2, 1, 0, { w: 0.88, h: 0.13, mode: 'core', dens: 0.55 });
    this._skirtRun(20.6, 24.42, 25.2, 24.42, 0, -1, { w: 0.8, h: 0.12, dens: 0.6 });
    // minaret plinth gets a conical spoil ring rather than four straight runs
    this._cyl('dirt', -5.6, 0.05, 43.2, 2.42, 3.15, 0.22, 16,
      { base: 0, zone: 'skirt', noCollide: true, noCast: true, ts: 3.0, grime: 1.5, shade: 0.9 });
    for (let i = 0; i < 22; i++) {
      const a = this._r() * 6.28, d = 2.2 + this._r() * 1.5;
      this._iPlace(this._r() < 0.6 ? 'rubbleS' : 'brickBit', -5.6 + Math.cos(a) * d, 0.03, 43.2 + Math.sin(a) * d,
        this._r() * 6.28, 0.6 + this._r() * 0.8, 0.82 + this._r() * 0.22, this._rr(-0.3, 0.3));
    }
    // blast-wall feet at the south checkpoint
    for (let i = 0; i < 5; i++) {
      const x = -4.6 + i * 2.3, z = -25.5 + (i % 2) * 1.6;
      this._skirtRun(x - 1.1, z + 0.5, x + 1.1, z + 0.5, 0, 1, { w: 0.55, h: 0.10, dens: 0.8 });
      this._skirtRun(x + 1.1, z - 0.5, x - 1.1, z - 0.5, 0, -1, { w: 0.55, h: 0.10, dens: 0.8 });
    }
    this._zoneSet(prev);
  }

  /** low earth mound — kills the perfect straight silhouette of the floor plane */
  _mound(cx, cz, r, h) {
    const sp = this._findSpot(cx, cz, r * 0.34, 2.4); if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, r * 0.40);
    this._gridMesh('dirt', cx - r, cz - r, cx + r, cz + r, r / 4, (x, z) => {
      const dx = (x - cx) / r, dz = (z - cz) / r;
      const d2 = dx * dx + dz * dz;
      const k = d2 < 1 ? (1 - d2) : 0;
      return this._surfY(x, z) + h * k * k * (0.72 + 0.62 * vnoise(x * 0.8, 3, z * 0.8)) + 0.010;
    }, { zone: 'terrain' });
    const n = Math.round(r * 5);
    for (let i = 0; i < n; i++) {
      const a = this._r() * 6.28, d = this._r() * r * 0.9;
      const px = cx + Math.cos(a) * d, pz = cz + Math.sin(a) * d;
      const k = 1 - (d / r) * (d / r);
      this._iPlace(this._r() < 0.6 ? 'rubbleS' : 'brickBit', px, this._surfY(px, pz) + h * k * k * 0.8, pz,
        this._r() * 6.28, 0.6 + this._r() * 0.9, 0.82 + this._r() * 0.24, this._rr(-0.3, 0.3));
    }
    this._stain(cx, cz, r * 1.3, 0.30);
  }

  /** lifted / sunken paving slabs so no floor edge ever reads as a ruled line */
  _pavingBreak(count) {
    const prev = this._zone;
    this._zoneSet('skirt');
    for (let i = 0; i < count; i++) {
      let x = 0, z = 0, ok = false;
      for (let a = 0; a < 14 && !ok; a++) {
        if (this._r() < 0.5) { x = this._rr(-5.1, 5.1); z = this._rr(-29, 23.4); }
        else { x = this._rr(-20, 5.2); z = this._rr(24.2, 39.0); }
        ok = this._clearAt(x, z, 1.5);
      }
      if (!ok) continue;
      const gy = this._surfY(x, z);
      const w = this._rr(0.85, 1.9), d = this._rr(0.65, 1.5), th = this._rr(0.08, 0.14);
      const lift = this._rr(0.018, 0.052);
      this._box(this._r() < 0.45 ? 'asphalt' : 'concreteB', x, gy + lift - th * 0.5, z, w, th, d, {
        rotY: this._r() * 3.14,
        tilt: [this._rr(-1, 1), 0, this._rr(-1, 1), this._rr(0.015, 0.07)],
        grime: 1.5, shade: 0.96, noCast: true, base: gy,
      });
      for (let k = 0; k < 4; k++) {
        const a = this._r() * 6.28, dd = this._rr(0.5, 1.3);
        this._iPlace(this._r() < 0.5 ? 'shard' : 'brickBit', x + Math.cos(a) * dd, gy + 0.012, z + Math.sin(a) * dd,
          this._r() * 6.28, 0.6 + this._r() * 0.8, 0.86 + this._r() * 0.2, this._rr(-0.25, 0.25));
      }
      this._stain(x, z, 1.2, 0.22);
    }
    this._zoneSet(prev);
  }

  /**
   * Tonal breakup of the floor plane itself. A prop every few metres is only
   * half the job — an unresurfaced Middle-Eastern street is a patchwork of
   * dark tar repairs, pale dust drifts, blown-out shoulders and long cracks.
   * Without this the carriageway reads as one poured sheet no matter how much
   * clutter stands on it. Everything here is a 3 cm lip, so it never trips the
   * player and never z-fights the road mesh.
   */
  _roadScars() {
    const prev = this._zone;
    this._zoneSet('scar');

    // dark bitumen repair patches, biased to the wheel tracks
    for (let i = 0; i < 78; i++) {
      const plaza = this._r() < 0.34;
      const x = plaza ? this._rr(-20.4, 5.2) : this._rr(-5.1, 5.1);
      const z = plaza ? this._rr(24.0, 38.8) : this._rr(-30.2, 23.8);
      const gy = this._surfY(x, z);
      const w = this._rr(1.1, 3.6), d = this._rr(0.8, 2.9);
      this._box('asphalt', x, gy + 0.040, z, w, 0.030, d, {
        rotY: this._r() * 3.14, base: gy, grime: 1.0,
        shade: 0.44 + this._r() * 0.16, noCast: true, zone: 'scar',
      });
      // ragged shoulder so the patch is not a clean rectangle
      for (let k = 0; k < 3; k++) {
        const a = this._r() * 6.28, dd = this._rr(0.35, 0.75);
        this._box('asphalt', x + Math.cos(a) * w * dd, gy + 0.038, z + Math.sin(a) * d * dd,
          this._rr(0.4, 1.1), 0.026, this._rr(0.35, 0.95), {
          rotY: this._r() * 3.14, base: gy, shade: 0.46 + this._r() * 0.18,
          noCast: true, zone: 'scar',
        });
      }
    }

    // long cracks running with the street, and a few crossing it
    for (let i = 0; i < 46; i++) {
      const along = this._r() < 0.7;
      const x = this._rr(-5.0, 5.0), z = this._rr(-30.0, 23.6);
      const gy = this._surfY(x, z);
      const len = this._rr(1.6, 5.4);
      const ang = (along ? 0 : Math.PI * 0.5) + this._rr(-0.45, 0.45);
      const segs = Math.max(2, Math.round(len / 1.1));
      for (let s = 0; s < segs; s++) {
        const t = (s + 0.5) / segs - 0.5;
        const px = x + Math.cos(ang) * t * len, pz = z + Math.sin(ang) * t * len;
        this._box('dark', px, this._surfY(px, pz) + 0.030, pz,
          len / segs + 0.05, 0.024, this._rr(0.035, 0.075), {
          rotY: -ang + this._rr(-0.18, 0.18), base: gy, raw: true,
          shade: 0.34 + this._r() * 0.16, noCast: true, zone: 'scar',
        });
      }
    }

    // pale dust drifts feathering across the carriageway and the apron
    for (let i = 0; i < 58; i++) {
      const plaza = this._r() < 0.42;
      const x = plaza ? this._rr(-20.4, 5.2) : this._rr(-5.2, 5.2);
      const z = plaza ? this._rr(24.0, 38.8) : this._rr(-30.2, 23.8);
      const gy = this._surfY(x, z);
      const w = this._rr(1.4, 4.2), d = this._rr(0.7, 2.2);
      this._box('sand', x, gy + 0.048, z, w, 0.036, d, {
        rotY: this._rr(-0.5, 0.5) + (this._r() < 0.5 ? 0 : 1.57), base: gy,
        grime: 0.7, shade: 1.06 + this._r() * 0.10, noCast: true, zone: 'scar',
      });
    }

    // blown-out shoulders where the tarmac has failed against the kerb
    for (const s of [-1, 1]) {
      for (let z = -29.0; z < 23.0; z += this._rr(3.4, 7.0)) {
        const x = s * this._rr(4.15, 5.05);
        const gy = this._surfY(x, z);
        this._box('concreteB', x, gy + 0.036, z, this._rr(0.7, 1.5), 0.028, this._rr(1.1, 2.6), {
          rotY: this._rr(-0.25, 0.25), base: gy, grime: 1.5,
          shade: 0.66 + this._r() * 0.16, noCast: true, zone: 'scar',
        });
        for (let k = 0; k < 4; k++) {
          const px = x + this._rr(-0.7, 0.7), pz = z + this._rr(-1.1, 1.1);
          this._iPlace(this._r() < 0.5 ? 'shard' : 'brickBit', px, this._surfY(px, pz) + 0.012, pz,
            this._r() * 6.28, 0.7 + this._r() * 0.8, 0.84 + this._r() * 0.22, this._rr(-0.28, 0.28));
        }
      }
    }
    this._zoneSet(prev);
  }

  // ------------------------------------------------------------ prop kits

  _kioskProp(cx, cz, rot, toppled) {
    const g0 = this._tallGuard; this._tallGuard = true;
    const sp = this._findSpot(cx, cz, 1.25, 2.6);
    this._tallGuard = g0;
    if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 1.45);
    const gy = this._surfY(cx, cz);
    if (toppled) {
      const ax = Math.cos(rot), az = Math.sin(rot);
      this._iPlace('kioskBody', cx, gy + 0.62, cz, rot, 1, 0.80, 1.49, [ax, 0, az]);
      this._iPlace('kioskRoof', cx + az * 0.6, gy + 0.05, cz - ax * 0.6, rot + 0.4, 1, 0.86, 1.62, [ax, 0, az]);
      this._rubblePile(cx + az * 1.4, cz - ax * 1.4, 1.1, 10, false);
    } else {
      this._iPlace('kioskBody', cx, gy, cz, rot, 1, 0.92);
      this._iPlace('kioskRoof', cx, gy, cz, rot, 1, 0.97);
      this._iPlace('crate', cx - Math.sin(rot) * 1.15, gy, cz - Math.cos(rot) * 1.15, rot + 0.4, 1, 0.9);
      this._iPlace('basket', cx - Math.sin(rot) * 1.1 + 0.6, gy, cz - Math.cos(rot) * 1.1 + 0.3, rot, 1, 0.92);
    }
    this._stain(cx, cz, 2.1, 0.45);
    this.coverPoints.push({
      pos: new THREE.Vector3(cx, gy + (toppled ? 0.7 : 1.0), cz),
      dir: new THREE.Vector3(-Math.sin(rot), 0, -Math.cos(rot)), height: toppled ? 0.7 : 1.1,
    });
  }

  _spoolProp(cx, cz, rot) {
    const sp = this._findSpot(cx, cz, 0.85, 2.0); if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 1.0);
    const gy = this._surfY(cx, cz);
    this._iPlace('spool', cx, gy, cz, rot, 1, 0.90);
    this._iPlace('coil', cx, gy, cz, rot, 1, 0.72);
    if (this._r() < 0.6) this._iPlace('plank', cx + this._rr(-1.3, 1.3), gy + 0.01, cz + this._rr(-1.3, 1.3), this._r() * 6.28, 1, 0.86);
    this._stain(cx, cz, 1.4, 0.42);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 0.8, cz), dir: new THREE.Vector3(0, 0, -1), height: 0.85 });
  }

  _tyreStack(cx, cz, n) {
    const sp = this._findSpot(cx, cz, 0.7, 1.8); if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 0.85);
    const gy = this._surfY(cx, cz);
    for (let i = 0; i < n; i++) {
      this._iPlace('tyre', cx + this._rr(-0.10, 0.10), gy + 0.02 + i * 0.205, cz + this._rr(-0.10, 0.10),
        this._r() * 6.28, 1, 0.76 + this._r() * 0.2, this._rr(-0.045, 0.045));
    }
    for (let i = 0; i < 2; i++) {
      const a = this._r() * 6.28, d = this._rr(0.7, 1.2);
      this._iPlace('tyre', cx + Math.cos(a) * d, gy + 0.03, cz + Math.sin(a) * d,
        this._r() * 6.28, 1, 0.82, this._r() < 0.5 ? 1.5 : 0, [1, 0, 0.2]);
    }
    this._stain(cx, cz, 1.5, 0.5);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + n * 0.2, cz), dir: new THREE.Vector3(0, 0, -1), height: n * 0.2 });
  }

  _drumCluster(cx, cz, n) {
    const sp = this._findSpot(cx, cz, 0.85, 2.0); if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 1.05);
    const gy = this._surfY(cx, cz);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * 6.28 + this._rr(-0.4, 0.4), d = this._rr(0.05, 0.62);
      const px = cx + Math.cos(a) * d, pz = cz + Math.sin(a) * d;
      const down = this._r() < 0.22;
      if (down) this._iPlace('barrel', px, gy + 0.29, pz, this._r() * 6.28, 1, 0.80 + this._r() * 0.2, 1.55, [1, 0, 0.15]);
      else this._iPlace('barrel', px, gy + 0.02, pz, this._r() * 6.28, 1, 0.80 + this._r() * 0.25);
    }
    if (this._r() < 0.6) this._iPlace('jerry', cx + this._rr(-1.1, 1.1), gy + 0.02, cz + this._rr(-1.1, 1.1), this._r() * 6.28, 1, 0.85);
    if (this._r() < 0.5) this._iPlace('cinder', cx + this._rr(-1.2, 1.2), gy + 0.01, cz + this._rr(-1.2, 1.2), this._r() * 6.28, 1, 0.88);
    this._stain(cx, cz, 1.7, 0.55);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 0.85, cz), dir: new THREE.Vector3(0, 0, -1), height: 0.9 });
  }

  /** stack of concrete culvert sections dumped in the street */
  _pipeStack(cx, cz, rot, n) {
    const g0 = this._tallGuard; this._tallGuard = true;
    const sp = this._findSpot(cx, cz, 1.05, 2.4);
    this._tallGuard = g0;
    if (!sp) return;
    cx = sp[0]; cz = sp[1];
    this._reserve(cx, cz, 1.25);
    const gy = this._surfY(cx, cz);
    const ux = Math.cos(rot), uz = -Math.sin(rot);        // pipe long axis
    const vx = -uz, vz = ux;                              // across the stack
    let placed = 0;
    for (let row = 0; row < 3 && placed < n; row++) {
      const cnt = 3 - row;
      for (let i = 0; i < cnt && placed < n; i++) {
        const off = (i - (cnt - 1) * 0.5) * 0.96;
        const px = cx + vx * off + ux * this._rr(-0.12, 0.12);
        const pz = cz + vz * off + uz * this._rr(-0.12, 0.12);
        this._iPlace('pipe', px, gy + row * 0.86, pz, rot + this._rr(-0.04, 0.04), 1, 0.88 + this._r() * 0.16);
        this._iPlace('pipeBore', px, gy + row * 0.86, pz, rot + this._rr(-0.04, 0.04), 1, 0.55);
        placed++;
      }
    }
    // one rolled clear of the stack
    const ox = cx + vx * 2.1, oz = cz + vz * 2.1;
    if (this._clearAt(ox, oz, 0.7)) {
      this._reserve(ox, oz, 0.9);
      this._iPlace('pipe', ox, this._surfY(ox, oz), oz, rot + this._rr(0.5, 1.1), 1, 0.9);
      this._iPlace('pipeBore', ox, this._surfY(ox, oz), oz, rot + 0.8, 1, 0.55);
    }
    this._stain(cx, cz, 2.2, 0.42);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 0.9, cz), dir: new THREE.Vector3(vx, 0, vz), height: 0.9 });
  }

  /**
   * Free-standing ruined wall stub. Reads as a hard silhouette and a cover
   * line, and its rubble spill breaks the floor plane at both ends.
   */
  _ruinWall(x0, z0, x1, z1, h, mat) {
    const prev = this._zone;
    this._zoneSet('dress');
    const len = Math.hypot(x1 - x0, z1 - z0);
    const segs = Math.max(3, Math.round(len / 0.95));
    const M = mat || (this._r() < 0.5 ? 'brickB' : 'plasterC');
    const ry = Math.atan2(-(z1 - z0), (x1 - x0));
    const nx = Math.sin(ry), nz = Math.cos(ry);
    const T = 0.36;
    // footing
    const mx = (x0 + x1) * 0.5, mz = (z0 + z1) * 0.5;
    const gy0 = this._surfY(mx, mz);
    this._box('concrete', mx, gy0 + 0.10, mz, len + 0.3, 0.34, T + 0.24,
      { rotY: ry, grime: 1.55, base: gy0 });
    for (let i = 0; i < segs; i++) {
      const t = (i + 0.5) / segs;
      const px = x0 + (x1 - x0) * t, pz = z0 + (z1 - z0) * t;
      const gy = this._surfY(px, pz);
      // collapse profile: tall in the middle, broken away at the ends
      const edge = Math.min(t, 1 - t) * 2;
      let hh = h * (0.30 + 0.72 * smooth(0.0, 0.55, edge)) * (0.78 + 0.34 * vnoise(px * 0.9, 1, pz * 0.9));
      hh = Math.max(0.35, hh);
      this._box(M, px, gy + hh * 0.5 + 0.2, pz, len / segs + 0.02, hh, T,
        { rotY: ry, grime: 1.4, base: gy });
      if (hh > h * 0.55) {
        this._box('concrete', px, gy + hh + 0.24, pz, len / segs + 0.06, 0.10, T + 0.12,
          { rotY: ry, base: gy, shade: 1.05 });
      } else {
        this._rebar(px, gy + hh + 0.2, pz, 2);
      }
      if (this._r() < 0.35) {
        this._pockWall(px, gy + 0.4, gy + hh, pz, nx, nz, 5, 0.4);
      }
    }
    this._rubblePile(x0 - nx * 0.2, z0 - nz * 0.2, 1.5, 16, true);
    this._rubblePile(x1 + nx * 0.2, z1 + nz * 0.2, 1.4, 14, true);
    this._skirtRun(x0, z0, x1, z1, nx, nz, { w: 0.8, h: 0.12, dens: 1.1 });
    this._skirtRun(x1, z1, x0, z0, -nx, -nz, { w: 0.8, h: 0.12, dens: 1.1 });
    this._reserve(mx, mz, len * 0.45);
    this._reserve(x0, z0, 1.0);
    this._reserve(x1, z1, 1.0);
    this.coverPoints.push({ pos: new THREE.Vector3(mx, gy0 + h * 0.6, mz), dir: new THREE.Vector3(nx, 0, nz), height: h * 0.7 });
    this.coverPoints.push({ pos: new THREE.Vector3(mx, gy0 + h * 0.6, mz), dir: new THREE.Vector3(-nx, 0, -nz), height: h * 0.7 });
    this._zoneSet(prev);
  }

  /** run of Jersey barriers between two points — the level's cover-line primitive */
  _barrierLine(x0, z0, x1, z1) {
    const len = Math.hypot(x1 - x0, z1 - z0);
    const n = Math.max(1, Math.round(len / 2.18));
    const ry = Math.atan2(-(z1 - z0), (x1 - x0));
    for (let i = 0; i < n; i++) {
      const t = (i + 0.5) / n;
      let x = x0 + (x1 - x0) * t, z = z0 + (z1 - z0) * t;
      const sp = this._findSpot(x, z, 0.95, 1.1);
      if (!sp) continue;
      x = sp[0]; z = sp[1];
      this._reserve(x, z, 1.00);
      const gy = this._surfY(x, z);
      this._iPlace('jersey', x, gy, z, ry + this._rr(-0.05, 0.05), 1, 0.88 + this._r() * 0.16);
      this.coverPoints.push({
        pos: new THREE.Vector3(x, gy + 0.78, z),
        dir: new THREE.Vector3(-Math.sin(ry), 0, -Math.cos(ry)), height: 0.8,
      });
      const k = this._r();
      if (k < 0.30) this._iPlace('cinder', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-0.9, 0.9), this._r() * 6.28, 1, 0.88);
      else if (k < 0.45) this._iPlace('tyre', x + this._rr(-1.3, 1.3), gy + 0.02, z + this._rr(-0.9, 0.9), this._r() * 6.28, 1, 0.84);
    }
    this._stain((x0 + x1) * 0.5, (z0 + z1) * 0.5, len * 0.6 + 0.6, 0.32);
  }

  /**
   * A pile of goods stacked against a wall. (nx,nz) points away from the wall.
   * Six flavours so a row of these never repeats visibly.
   */
  _wallPile(x, z, nx, nz) {
    const sp = this._findSpot(x, z, 0.95, 1.3); if (!sp) return;
    x = sp[0]; z = sp[1];
    this._reserve(x, z, 1.02);
    const gy = this._surfY(x, z);
    const ry = Math.atan2(nx, nz);
    // wall-local frame: u runs along the wall, v runs away from it
    const U = (u, v) => [x + nz * u + nx * v, z - nx * u + nz * v];
    const k = this._r();

    if (k < 0.19) {                                   // pallets + crate stack
      let p = U(-0.25, 0.10);
      this._iPlace('pallet', p[0], gy + 0.005, p[1], ry + this._rr(-0.25, 0.25), 1, 0.90);
      this._iPlace('crate', p[0] - 0.14, gy + 0.148, p[1] + 0.10, ry + this._rr(-0.3, 0.3), 1, 0.86);
      this._iPlace('crate', p[0] + 0.24, gy + 0.148, p[1] - 0.16, ry + this._rr(-0.2, 0.2), 0.96, 0.92);
      if (this._r() < 0.65) this._iPlace('crate', p[0] + 0.05, gy + 0.752, p[1] - 0.02, ry + 0.34, 0.9, 0.96);
      p = U(0.85, 0.05);
      this._iPlace('pallet', p[0], gy + 0.005, p[1], ry + 1.25, 1, 0.86, 0.10);
      p = U(1.05, 0.34);
      this._iPlace('plank', p[0], gy + 0.01, p[1], ry + this._rr(-0.5, 0.5), 1, 0.84);
    } else if (k < 0.37) {                            // oil drums against the wall
      for (let i = 0; i < 3; i++) {
        const p = U(-0.6 + i * 0.64 + this._rr(-0.08, 0.08), this._rr(0.02, 0.26));
        this._iPlace('barrel', p[0], gy + 0.02, p[1], this._r() * 6.28, 1, 0.80 + this._r() * 0.22);
      }
      if (this._r() < 0.5) {
        const p = U(0.95, 0.42);
        this._iPlace('barrel', p[0], gy + 0.29, p[1], this._r() * 6.28, 1, 0.78, 1.55, [1, 0, 0.2]);
      }
      const p = U(-1.05, 0.18);
      this._iPlace('jerry', p[0], gy + 0.01, p[1], this._r() * 6.28, 1, 0.85);
    } else if (k < 0.52) {                            // stacked tyres + a loose one
      for (let i = 0; i < 4; i++) {
        const p = U(this._rr(-0.09, 0.09), this._rr(-0.09, 0.09));
        this._iPlace('tyre', p[0], gy + 0.02 + i * 0.205, p[1], this._r() * 6.28, 1, 0.78 + this._r() * 0.18, this._rr(-0.04, 0.04));
      }
      const p = U(0.86, 0.30);
      this._iPlace('tyre', p[0], gy + 0.03, p[1], this._r() * 6.28, 1, 0.84, 1.5, [1, 0, 0.2]);
      const q = U(-0.8, 0.12);
      this._iPlace('cinder', q[0], gy + 0.01, q[1], this._r() * 6.28, 1, 0.88);
    } else if (k < 0.70) {                            // market goods: sacks + baskets
      for (let i = 0; i < 4; i++) {
        const p = U(-0.72 + i * 0.46 + this._rr(-0.06, 0.06), this._rr(0.0, 0.22));
        this._iPlace('sack', p[0], gy + 0.01, p[1], this._r() * 6.28, 0.9 + this._r() * 0.25, 0.86 + this._r() * 0.2, this._rr(-0.12, 0.12));
      }
      const p = U(0.4, 0.05);
      this._iPlace('sack', p[0], gy + 0.62, p[1], this._r() * 6.28, 0.92, 0.94, this._rr(-0.2, 0.2));
      const q = U(-1.1, 0.30);
      this._iPlace('basket', q[0], gy + 0.01, q[1], this._r() * 6.28, 1, 0.9);
      const r2 = U(1.0, 0.24);
      this._iPlace('basket', r2[0], gy + 0.01, r2[1], this._r() * 6.28, 1.1, 0.88);
    } else if (k < 0.86) {                            // builder's spoil: blocks + planks
      for (let i = 0; i < 5; i++) {
        const p = U(-0.5 + (i % 3) * 0.44, 0.05 + Math.floor(i / 3) * 0.05);
        this._iPlace('cinder', p[0], gy + 0.01 + Math.floor(i / 3) * 0.195, p[1],
          ry + this._rr(-0.12, 0.12), 1, 0.86 + this._r() * 0.2);
      }
      for (let i = 0; i < 2; i++) {
        const p = U(this._rr(-1.1, 1.1), this._rr(0.15, 0.55));
        this._iPlace('plank', p[0], gy + 0.01, p[1], ry + this._rr(-0.7, 0.7), 1, 0.84);
      }
      const q = U(0.95, 0.2);
      this._iPlace('bucket', q[0], gy + 0.01, q[1], this._r() * 6.28, 1, 0.9);
      this._rubblePile(x + nx * 0.35, z + nz * 0.35, 0.85, 8, false);
    } else {                                          // sandbag stub + crate
      const ang = ry + Math.PI * 0.5;
      const bw = 0.56, ch = 0.20, cn = 3;
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < cn; i++) {
          const t = (i + 0.5) * bw + ((c & 1) ? bw * 0.5 : 0) - cn * bw * 0.5;
          const p = U(t, 0.12);
          this._iPlace('sandbag', p[0], gy + 0.02 + c * ch + ch * 0.5, p[1],
            -ang + this._rr(-0.09, 0.09), [1, 1, 1], 0.86 + this._r() * 0.2, this._rr(-0.05, 0.05));
        }
      }
      const p = U(1.15, 0.16);
      this._iPlace('crateBig', p[0], gy + 0.01, p[1], ry + this._rr(-0.3, 0.3), 1, 0.88);
    }
    this._stain(x, z, 1.7, 0.42);
    this.coverPoints.push({
      pos: new THREE.Vector3(x + nx * 0.5, gy + 0.8, z + nz * 0.5),
      dir: new THREE.Vector3(nx, 0, nz), height: 0.85,
    });
  }

  // ---------------------------------------------- tall mid-ground fillers
  //  Anything under ~1 m disappears behind the viewmodel; these are the
  //  1.5-2.6 m silhouettes that actually break the floor plane at 5-25 m.

  _awningFrame(cx, cz, rot) {
    const prev = this._zone;
    this._zoneSet('dress');
    const gy = this._surfY(cx, cz), w = 2.9, d = 2.1, h = 2.32;
    const O = { rotY: rot, pivotX: cx, pivotZ: cz, base: gy };
    for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      this._box('wood', cx + sx * w * 0.5, gy + h * 0.5, cz + sz * d * 0.5, 0.11, h, 0.11,
        Object.assign({ grime: 1.45 }, O));
    }
    for (const sz of [-1, 1]) {
      this._box('wood', cx, gy + h, cz + sz * d * 0.5, w + 0.22, 0.09, 0.09, O);
    }
    this._box('wood', cx - w * 0.5, gy + h, cz, 0.09, 0.09, d, O);
    this._box('wood', cx + w * 0.5, gy + h, cz, 0.09, 0.09, d, O);
    this._tarp(cx, gy + h + 0.11, cz, w + 0.55, d + 0.45, 0.30, rot);
    this._clothSheet(cx - Math.sin(rot) * (d * 0.5), gy + h - 0.15, cz + Math.cos(rot) * (d * 0.5),
      1.1, 1.35, rot);
    this._iPlace('crateBig', cx + this._rr(-0.7, 0.7), gy + 0.01, cz + this._rr(-0.5, 0.5), rot, 1, 0.86);
    this._iPlace('sack', cx + this._rr(-1.0, 1.0), gy + 0.01, cz + this._rr(-0.6, 0.6), this._r() * 6.28, 1.05, 0.88);
    this._stain(cx, cz, 2.4, 0.45);
    this._zoneSet(prev);
  }

  _waterTank(cx, cz, rot) {
    const prev = this._zone;
    this._zoneSet('dress');
    const gy = this._surfY(cx, cz);
    for (const [ox, oz] of [[-0.50, -0.50], [0.50, -0.50], [-0.50, 0.50], [0.50, 0.50]]) {
      this._box('rust', cx + ox, gy + 0.55, cz + oz, 0.085, 1.10, 0.085, { base: gy, rotY: rot, pivotX: cx, pivotZ: cz });
    }
    this._box('rust', cx, gy + 0.55, cz - 0.50, 1.0, 0.06, 0.06, { base: gy, rotY: rot, pivotX: cx, pivotZ: cz });
    this._box('rust', cx, gy + 1.08, cz, 1.26, 0.09, 1.26, { base: gy, rotY: rot, pivotX: cx, pivotZ: cz, shade: 1.03 });
    this._cyl('polymer', cx, gy + 1.74, cz, 0.58, 0.60, 1.22, 12, { base: gy });
    this._cyl('polymer', cx, gy + 2.40, cz, 0.22, 0.32, 0.13, 10, { base: gy });
    this._cyl('rust', cx + 0.62, gy + 1.20, cz + 0.30, 0.030, 0.030, 2.0, 6,
      { base: gy, zone: 'detail', tilt: [0, 0, 1, 0.16] });
    this._stain(cx, cz, 1.8, 0.5);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 1.0, cz), dir: new THREE.Vector3(0, 0, -1), height: 1.0 });
    this._zoneSet(prev);
  }

  _crateTower(cx, cz, rot) {
    const gy = this._surfY(cx, cz);
    this._iPlace('pallet', cx, gy + 0.005, cz, rot, 1, 0.88);
    this._iPlace('crateBig', cx, gy + 0.14, cz, rot + this._rr(-0.08, 0.08), 1, 0.88);
    this._iPlace('crateBig', cx + 0.06, gy + 0.86, cz - 0.04, rot + this._rr(-0.14, 0.14), 0.97, 0.93);
    this._iPlace('crate', cx - 0.18, gy + 1.58, cz + 0.10, rot + this._rr(-0.3, 0.3), 1, 0.97);
    this._iPlace('crate', cx + 0.30, gy + 1.58, cz - 0.14, rot + this._rr(-0.3, 0.3), 0.94, 1.0);
    this._iPlace('crate', cx + 0.04, gy + 2.16, cz - 0.02, rot + this._rr(-0.4, 0.4), 0.9, 1.02);
    this._iPlace('sack', cx + this._rr(-1.1, 1.1), gy + 0.01, cz + this._rr(-1.0, 1.0), this._r() * 6.28, 1, 0.88);
    this._iPlace('basket', cx + this._rr(-1.1, 1.1), gy + 0.01, cz + this._rr(-1.0, 1.0), this._r() * 6.28, 1, 0.9);
    this._stain(cx, cz, 1.8, 0.45);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 1.0, cz), dir: new THREE.Vector3(0, 0, -1), height: 1.1 });
  }

  _drumTower(cx, cz, rot) {
    const gy = this._surfY(cx, cz);
    this._iPlace('pallet', cx, gy + 0.005, cz, rot, 1, 0.86);
    for (let i = 0; i < 3; i++) {
      const a = rot + i * 2.09;
      this._iPlace('barrel', cx + Math.cos(a) * 0.30, gy + 0.14, cz + Math.sin(a) * 0.30,
        this._r() * 6.28, 1, 0.80 + this._r() * 0.22);
    }
    this._iPlace('barrel', cx, gy + 1.02, cz, this._r() * 6.28, 1, 0.94);
    this._iPlace('tyre', cx + this._rr(-1.2, 1.2), gy + 0.02, cz + this._rr(-1.0, 1.0), this._r() * 6.28, 1, 0.84);
    this._stain(cx, cz, 1.9, 0.55);
    this.coverPoints.push({ pos: new THREE.Vector3(cx, gy + 0.9, cz), dir: new THREE.Vector3(0, 0, -1), height: 0.95 });
  }

  /** one tall silhouette, kind chosen by an index so a run never repeats */
  _tallProp(x, z, k, rot) {
    this._tallGuard = true;
    // 0.78 — hero props reserve far more ground than they physically occupy;
    // at full clearance the late fill passes find nowhere left to stand
    const sp = this._findSpot(x, z, 1.18, 3.6, 0.78);
    if (!sp) { this._tallGuard = false; return false; }
    x = sp[0]; z = sp[1];
    const r = rot === undefined ? this._r() * 6.28 : rot;
    switch (k % 8) {
      case 0: this._reserve(x, z, 1.7); this._awningFrame(x, z, r); break;
      case 1: this._reserve(x, z, 1.5); this._waterTank(x, z, r); break;
      case 2: this._reserve(x, z, 1.5); this._crateTower(x, z, r); break;
      case 3: this._pipeStack(x, z, r, 6); break;
      case 4: this._kioskProp(x, z, r, this._r() < 0.28); break;
      case 5: this._reserve(x, z, 1.4); this._drumTower(x, z, r); break;
      case 6: this._stall(x, z, r); break;
      default: {
        const ux = Math.cos(r) * 1.7, uz = Math.sin(r) * 1.7;
        this._ruinWall(x - ux, z - uz, x + ux, z + uz, 1.15 + this._r() * 0.55);
        break;
      }
    }
    this._tallGuard = false;
    return true;
  }

  /**
   * The critic's "60 metres of empty carpet" fix: a jittered grid of tall
   * silhouettes across the courtyard and plaza so the eye always has a
   * readable object every few metres between here and the archway.
   */
  _fillCourtyard() {
    const prev = this._zone;
    this._zoneSet('dress');
    this._heroForeground();
    let k = this._ri(0, 7);
    // Deliberate anchors that FRAME the hero presets. Every one of these sits
    // hard against a kerb line, outside the camera wedge, so it breaks the
    // floor plane and reads as cover without hiding the arch behind it.
    this._tallProp(-4.85, 17.40, 0, 1.55);    // weapon: right of frame, near
    this._tallProp(4.90, 20.20, 2, 4.60);     // weapon: left of frame, near
    this._tallProp(-4.95, 24.40, 4, 0.85);    // weapon: right of frame, mid
    this._tallProp(4.80, 26.80, 1, 4.10);     // weapon + spawn: left, mid
    this._tallProp(-4.90, 12.40, 5, 2.30);    // street: right of frame
    this._tallProp(4.95, 8.60, 2, 3.90);      // street: left of frame
    this._tallProp(-4.95, 1.40, 1, 1.20);     // street: deep right
    this._tallProp(4.85, -6.60, 7, 0.60);     // street: deep left
    this._tallProp(-8.60, 25.60, 6, 0.35);    // plaza: closes the west side
    this._tallProp(-15.40, 27.40, 3, 1.15);   // vista foreground
    // A few mid-band silhouettes standing IN the lane. At 12-20 m these top
    // out below the arch springing line, so they read as cover and give the
    // eye depth cues without shutting the sightline down. Kept deliberately
    // few and staggered off-axis: the lane between them is the composition.
    this._tallProp(-2.90, 27.90, 2, 2.55);    // crate tower, 14 m from weapon
    this._tallProp(1.30, 31.60, 4, 0.40);     // kiosk, 18 m
    this._tallProp(3.10, 12.60, 3, 2.10);     // street lane, mid
    this._tallProp(-2.30, -11.20, 1, 2.85);
    this._tallProp(2.70, -18.40, 4, 1.05);
    // plaza: a loose scatter, not a grid of obstacles. The plaza has to stay
    // walkable and the gate has to stay visible from the hall deck, so the
    // spacing is deliberately wide — the wall piles and the low tier fill in
    // between these.
    for (let z = 21.4; z < 39.0; z += 6.10) {
      for (let x = -19.4; x < 5.6; x += 6.30) {
        const px = x + this._rr(-1.6, 1.6), pz = z + this._rr(-1.6, 1.6);
        if (pz > 22.8 && pz < 24.3) continue;               // keep the kerb line clear
        if (this._r() < 0.22) continue;
        if (this._tallProp(px, pz, k, this._r() * 6.28)) k++;
      }
    }
    // street corridor: sparse, and only narrow kinds — it is the firing lane
    const NARROW = [1, 2, 3, 5];
    for (let z = -27.0; z < 19.0; z += 9.6) {
      for (let x = -4.2; x < 4.6; x += 4.4) {
        const px = x + this._rr(-1.0, 1.0), pz = z + this._rr(-2.2, 2.2);
        if (this._r() < 0.52) continue;
        if (this._tallProp(px, pz, NARROW[k % 4], this._r() * 6.28)) k++;
      }
    }
    this._zoneSet(prev);
  }

  /**
   * Hand-authored anchors for the hero frames. Everything here is under eye
   * height (1.62 m), so none of it can rise above the horizon and hide the
   * gate arch. They sit on the SHOULDERS of the street — hard against the
   * kerb line — so they frame the lane rather than filling it: a MWII street
   * puts its cover against the kerbs and keeps the crown of the road open.
   * Placed before the tall anchors so it owns those shoulders.
   */
  _heroForeground() {
    const prev = this._zone;
    this._zoneSet('dress');
    const put = (x, z, kind, rot) => {
      let px = x, pz = z;
      // k = 0.8: only nestle slightly inside a neighbour's reservation. At the
      // old 0.32 these piled straight on top of the hero props and turned the
      // near field into a solid wall of clutter.
      let ok = !this._onStation(px, pz, 1.1) && this._clearAt(px, pz, 1.05, 0.8);
      for (let i = 1; i <= 14 && !ok; i++) {
        const a = i * 2.399963, d = 1.7 * Math.sqrt(i / 14);
        px = x + Math.cos(a) * d; pz = z + Math.sin(a) * d;
        ok = !this._onStation(px, pz, 1.1) && this._clearAt(px, pz, 1.05, 0.8);
      }
      if (!ok) return;
      this._reserve(px, pz, 1.45);
      this._midBreaker(px, pz, kind, rot);
      this.stats.dressMid++;
    };

    // --- weapon + spawn lane, running north to the gate ----------------
    put(4.35, 16.60, 2, 0.85);     // crate stack, screen left
    put(-4.30, 18.80, 3, 1.90);    // oil drums, screen right
    put(4.55, 22.60, 6, 1.35);     // low ruin stub, screen left
    put(-4.45, 24.20, 1, 0.55);    // culverts, screen right
    put(3.30, 29.20, 4, 0.05);     // sandbags, plaza mouth
    put(-3.20, 32.80, 5, 2.35);    // tyres + spool

    // --- street + sun lane, running south ------------------------------
    put(4.30, 9.60, 0, 0.35);
    put(-4.40, 5.20, 1, 1.30);
    put(4.45, -0.80, 4, 0.15);
    put(-4.35, -8.40, 2, 1.90);
    put(4.40, -15.20, 0, 0.25);
    put(-4.45, -23.10, 3, 0.85);

    // --- vista foreground: the plaza floor read from the hall deck ------
    put(-11.80, 24.90, 0, 1.35);
    put(-15.20, 30.60, 4, 2.65);
    put(-8.20, 31.40, 3, 0.45);
    put(-18.10, 26.20, 2, 1.05);
    this._zoneSet(prev);
  }

  // ------------------------------------------------- near-field floor breakup
  //  Everything above is >1 m and lives outside the camera wedges. This is what
  //  goes INSIDE them: knee-height silhouettes every few metres so the floor is
  //  never a continuous unbroken plane, but the sightline stays open.

  /** one sub-knee-height readable object at (x,z); nothing here exceeds ~0.9 m */
  _lowBreaker(x, z) {
    const gy = this._surfY(x, z);
    const ry = this._r() * 6.28;
    const k = this._r();

    if (k < 0.11) {
      // concrete culvert lying on its side — the classic street floor-breaker
      this._iPlace('pipe', x, gy, z, ry, 1, 0.86 + this._r() * 0.16);
      this._iPlace('pipeBore', x, gy, z, ry, 1, 0.55);
      if (this._r() < 0.5) {
        this._iPlace('cinder', x + this._rr(-1.2, 1.2), gy + 0.01, z + this._rr(-1.2, 1.2),
          this._r() * 6.28, 1, 0.88);
      }
    } else if (k < 0.24) {
      // spilled kerbstones / broken curb run, dropped out of line
      const ca = Math.cos(ry), sa = Math.sin(ry);
      for (let i = 0; i < 4; i++) {
        const t = (i - 1.5) * 0.64 + this._rr(-0.06, 0.06);
        const px = x + ca * t, pz = z + sa * t;
        const g0 = this._surfY(px, pz);
        this._box('concrete', px, g0 + 0.088 - (this._r() < 0.3 ? 0.04 : 0), pz,
          0.60, 0.20, 0.32, {
          rotY: -ry + this._rr(-0.12, 0.12), base: g0, grime: 1.5, shade: 1.03,
          tilt: [ca, 0, sa, this._rr(-0.16, 0.16)],
        });
      }
      this._iPlace('brickBit', x + this._rr(-1.1, 1.1), gy + 0.012, z + this._rr(-1.1, 1.1),
        this._r() * 6.28, 1, 0.86, this._rr(-0.3, 0.3));
    } else if (k < 0.36) {
      // builder's block pile + planks
      for (let i = 0; i < 6; i++) {
        const row = Math.floor(i / 3);
        const px = x + Math.cos(ry) * ((i % 3) - 1) * 0.44 + this._rr(-0.05, 0.05);
        const pz = z + Math.sin(ry) * ((i % 3) - 1) * 0.44 + this._rr(-0.05, 0.05);
        this._iPlace('cinder', px, this._surfY(px, pz) + 0.008 + row * 0.196, pz,
          ry + this._rr(-0.14, 0.14), 1, 0.85 + this._r() * 0.2);
      }
      this._iPlace('plank', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-1.3, 1.3),
        this._r() * 6.28, 1, 0.85);
      this._iPlace('bucket', x + this._rr(-1.1, 1.1), gy + 0.01, z + this._rr(-1.1, 1.1),
        this._r() * 6.28, 1, 0.9);
    } else if (k < 0.47) {
      // tyres: one short stack, one on its side, one leaning
      const n = this._ri(2, 3);
      for (let i = 0; i < n; i++) {
        this._iPlace('tyre', x + this._rr(-0.09, 0.09), gy + 0.02 + i * 0.205, z + this._rr(-0.09, 0.09),
          this._r() * 6.28, 1, 0.76 + this._r() * 0.2, this._rr(-0.05, 0.05));
      }
      this._iPlace('tyre', x + this._rr(0.7, 1.3) * (this._r() < 0.5 ? -1 : 1), gy + 0.03,
        z + this._rr(-1.2, 1.2), this._r() * 6.28, 1, 0.84, 1.5, [1, 0, 0.2]);
    } else if (k < 0.58) {
      // low sandbag stub — two courses, 0.4 m, pure silhouette
      const ang = ry;
      const bw = 0.56, ch = 0.20, cn = this._ri(3, 5);
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < cn; i++) {
          const t = (i + 0.5) * bw + ((c & 1) ? bw * 0.5 : 0) - cn * bw * 0.5;
          const px = x + Math.cos(ang) * t, pz = z + Math.sin(ang) * t;
          this._iPlace('sandbag', px, this._surfY(px, pz) + 0.02 + c * ch + ch * 0.5, pz,
            -ang + this._rr(-0.09, 0.09), [1, 1, 1], 0.64 + this._r() * 0.16, this._rr(-0.05, 0.05));
        }
      }
      this._stain(x, z, 1.6, 0.42);
      this.coverPoints.push({
        pos: new THREE.Vector3(x, gy + 0.42, z),
        dir: new THREE.Vector3(Math.sin(ang), 0, -Math.cos(ang)), height: 0.42,
      });
    } else if (k < 0.68) {
      // pallet + market goods dumped in the open
      this._iPlace('pallet', x, gy + 0.006, z, ry, 1, 0.89, this._rr(-0.05, 0.05));
      this._iPlace('sack', x + this._rr(-0.3, 0.3), gy + 0.145, z + this._rr(-0.25, 0.25),
        this._r() * 6.28, 0.95, 0.9, this._rr(-0.15, 0.15));
      if (this._r() < 0.6) {
        this._iPlace('basket', x + this._rr(-1.0, 1.0), gy + 0.01, z + this._rr(-1.0, 1.0),
          this._r() * 6.28, 1.05, 0.9);
      }
      this._iPlace('crate', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-1.3, 1.3),
        this._r() * 6.28, 1, 0.87);
    } else if (k < 0.78) {
      // spoil ridge — a long low bank of broken masonry, not a round pile
      const ca = Math.cos(ry), sa = Math.sin(ry);
      for (let i = 0; i < 12; i++) {
        const t = this._rr(-1.5, 1.5);
        const o = this._rr(-0.42, 0.42);
        const px = x + ca * t - sa * o, pz = z + sa * t + ca * o;
        const drop = 1 - Math.abs(o) / 0.5;
        this._iPlace(this._r() < 0.55 ? 'rubbleS' : 'brickBit', px,
          this._surfY(px, pz) + 0.015 + drop * 0.16, pz,
          this._r() * 6.28, 0.6 + this._r() * 0.9, 0.80 + this._r() * 0.24, this._rr(-0.32, 0.32));
      }
      this._iPlace('rubbleL', x, gy + 0.02, z, ry, 0.8 + this._r() * 0.6, 0.84, this._rr(-0.25, 0.25));
      this._stain(x, z, 2.0, 0.38);
    } else if (k < 0.87) {
      // storm drain / inspection cover sunk into the floor
      this._box('concrete', x, gy + 0.028, z, 1.22, 0.16, 0.92,
        { rotY: ry, base: gy, grime: 1.6, shade: 1.02 });
      this._box('gunmetal', x, gy + 0.085, z, 0.74, 0.06, 0.60,
        { rotY: ry, pivotX: x, pivotZ: z, base: gy, grime: 1.7, shade: 0.8 });
      for (let i = 0; i < 5; i++) {
        this._box('dark', x, gy + 0.10, z - 0.21 + i * 0.105, 0.62, 0.05, 0.048,
          { rotY: ry, pivotX: x, pivotZ: z, raw: true, shade: 0.24, noCast: true, zone: 'detail' });
      }
      this._stain(x, z, 1.5, 0.5);
    } else {
      // oil drum knocked flat + a jerry can
      this._iPlace('barrel', x, gy + 0.29, z, this._r() * 6.28, 1, 0.80 + this._r() * 0.2, 1.55, [1, 0, 0.2]);
      this._iPlace('jerry', x + this._rr(-1.0, 1.0), gy + 0.01, z + this._rr(-1.0, 1.0),
        this._r() * 6.28, 1, 0.85);
      this._stain(x, z, 1.6, 0.55);
    }
  }

  /**
   * The mid tier: 0.9-1.5 m. Anything under eye height (1.62 m) can never rise
   * above the horizon line, so these are safe to stand right in the middle of a
   * sightline — they read as cover and break the floor without hiding the far
   * landmark. This is the tier that actually does the work in a CoD street.
   */
  _midBreaker(x, z, kind, rot) {
    const gy = this._surfY(x, z);
    const ry = rot === undefined ? this._r() * 6.28 : rot;
    const K = kind === undefined ? this._ri(0, 6) : ((kind % 7) + 7) % 7;
    const k = [0.10, 0.25, 0.40, 0.52, 0.64, 0.76, 0.90][K];

    if (k < 0.17) {
      // Jersey barrier, sometimes a second one shunted out of line
      this._iPlace('jersey', x, gy, z, ry, 1, 0.86 + this._r() * 0.18);
      if (this._r() < 0.55) {
        const a = ry + Math.PI * 0.5 + this._rr(-0.35, 0.35), d = this._rr(2.0, 2.6);
        const px = x + Math.cos(a) * d, pz = z - Math.sin(a) * d;
        if (this._clearAt(px, pz, 0.8)) {
          this._reserve(px, pz, 0.85);
          this._iPlace('jersey', px, this._surfY(px, pz), pz, ry + this._rr(-0.5, 0.5), 1, 0.9);
        }
      }
      this._iPlace('cinder', x + this._rr(-1.4, 1.4), gy + 0.01, z + this._rr(-1.0, 1.0),
        this._r() * 6.28, 1, 0.88);
      this.coverPoints.push({
        pos: new THREE.Vector3(x, gy + 0.78, z),
        dir: new THREE.Vector3(-Math.sin(ry), 0, -Math.cos(ry)), height: 0.8,
      });
    } else if (k < 0.31) {
      // pair of culverts side by side + one rolled clear
      const ux = Math.cos(ry), uz = -Math.sin(ry);
      for (let i = 0; i < 2; i++) {
        const off = (i - 0.5) * 0.98;
        const px = x - uz * off, pz = z + ux * off;
        this._iPlace('pipe', px, this._surfY(px, pz), pz, ry + this._rr(-0.05, 0.05), 1, 0.88 + this._r() * 0.14);
        this._iPlace('pipeBore', px, this._surfY(px, pz), pz, ry, 1, 0.55);
      }
      this._iPlace('plank', x + this._rr(-1.5, 1.5), gy + 0.01, z + this._rr(-1.2, 1.2),
        this._r() * 6.28, 1, 0.85);
      this.coverPoints.push({
        pos: new THREE.Vector3(x, gy + 0.85, z),
        dir: new THREE.Vector3(-uz, 0, ux), height: 0.85,
      });
    } else if (k < 0.45) {
      // crate stack on a pallet — 1.45 m of hard silhouette
      this._iPlace('pallet', x, gy + 0.006, z, ry, 1, 0.88);
      this._iPlace('crateBig', x, gy + 0.14, z, ry + this._rr(-0.1, 0.1), 1, 0.88);
      this._iPlace('crate', x + this._rr(-0.2, 0.2), gy + 0.86, z + this._rr(-0.2, 0.2),
        ry + this._rr(-0.35, 0.35), 1, 0.95);
      if (this._r() < 0.5) {
        this._iPlace('crate', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-1.2, 1.2),
          this._r() * 6.28, 1, 0.87);
      }
      this._iPlace('sack', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-1.2, 1.2),
        this._r() * 6.28, 1, 0.88, this._rr(-0.15, 0.15));
      this._stain(x, z, 1.7, 0.42);
      this.coverPoints.push({ pos: new THREE.Vector3(x, gy + 0.9, z), dir: new THREE.Vector3(0, 0, -1), height: 0.95 });
    } else if (k < 0.58) {
      // oil drums, one on its side
      const n = this._ri(3, 4);
      for (let i = 0; i < n; i++) {
        const a = (i / n) * 6.28 + this._rr(-0.4, 0.4), d = this._rr(0.10, 0.66);
        const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
        this._iPlace('barrel', px, this._surfY(px, pz) + 0.02, pz, this._r() * 6.28, 1, 0.80 + this._r() * 0.24);
      }
      this._iPlace('barrel', x + this._rr(-1.6, 1.6), gy + 0.29, z + this._rr(-1.2, 1.2),
        this._r() * 6.28, 1, 0.78, 1.55, [1, 0, 0.2]);
      this._iPlace('jerry', x + this._rr(-1.3, 1.3), gy + 0.01, z + this._rr(-1.1, 1.1), this._r() * 6.28, 1, 0.85);
      this._stain(x, z, 1.8, 0.55);
      this.coverPoints.push({ pos: new THREE.Vector3(x, gy + 0.88, z), dir: new THREE.Vector3(0, 0, -1), height: 0.9 });
    } else if (k < 0.70) {
      // three-course sandbag emplacement with a crate shoulder
      const bw = 0.56, ch = 0.20, cn = this._ri(5, 7);
      for (let c = 0; c < 3; c++) {
        const drop = c === 2 && this._r() < 0.4 ? 1 : 0;      // top course partly spilled
        for (let i = drop; i < cn; i++) {
          const t = (i + 0.5) * bw + ((c & 1) ? bw * 0.5 : 0) - cn * bw * 0.5;
          const px = x + Math.cos(ry) * t, pz = z + Math.sin(ry) * t;
          this._iPlace('sandbag', px, this._surfY(px, pz) + 0.02 + c * ch + ch * 0.5, pz,
            -ry + this._rr(-0.09, 0.09), [1, 1, 1], 0.62 + this._r() * 0.18, this._rr(-0.05, 0.05));
        }
      }
      const ex = x + Math.cos(ry) * (cn * bw * 0.5 + 0.6), ez = z + Math.sin(ry) * (cn * bw * 0.5 + 0.6);
      this._iPlace('crateBig', ex, this._surfY(ex, ez) + 0.01, ez, ry, 1, 0.88);
      this._stain(x, z, 2.0, 0.45);
      this.coverPoints.push({
        pos: new THREE.Vector3(x, gy + 0.6, z),
        dir: new THREE.Vector3(Math.sin(ry), 0, -Math.cos(ry)), height: 0.6,
      });
    } else if (k < 0.82) {
      // tyre stack + cable spool
      const n = this._ri(4, 5);
      for (let i = 0; i < n; i++) {
        this._iPlace('tyre', x + this._rr(-0.10, 0.10), gy + 0.02 + i * 0.205, z + this._rr(-0.10, 0.10),
          this._r() * 6.28, 1, 0.76 + this._r() * 0.2, this._rr(-0.045, 0.045));
      }
      const sx = x + Math.cos(ry) * 1.5, sz = z + Math.sin(ry) * 1.5;
      if (this._clearAt(sx, sz, 0.8)) {
        this._reserve(sx, sz, 0.85);
        this._iPlace('spool', sx, this._surfY(sx, sz), sz, this._r() * 6.28, 1, 0.90);
        this._iPlace('coil', sx, this._surfY(sx, sz), sz, this._r() * 6.28, 1, 0.72);
      }
      this._stain(x, z, 1.7, 0.5);
      this.coverPoints.push({ pos: new THREE.Vector3(x, gy + 1.0, z), dir: new THREE.Vector3(0, 0, -1), height: 1.0 });
    } else {
      // a short ruined wall stub — the strongest floor-breaker in the kit
      const h = 0.95 + this._r() * 0.40;
      const ux = Math.cos(ry) * this._rr(1.2, 1.9), uz = Math.sin(ry) * this._rr(1.2, 1.9);
      this._ruinWall(x - ux, z - uz, x + ux, z + uz, h);
    }
  }

  /**
   * Sweep each hero camera wedge and drop a breaker every 1-1.6 m of depth,
   * alternating sides — low kit right under the muzzle, mid kit from ~3.5 m
   * out. Runs dead last so it lands on exactly the floor nothing else claimed,
   * which is where the critic found "60 metres of empty carpet".
   */
  /** rectangles the scatter must never land in: building shells, colonnades,
   *  the alley (dressed by hand), the gate piers */
  _noGoRects() {
    if (this._noGo) return this._noGo;
    const R = [];
    for (const b of (this._blds || [])) R.push([b.x0 - 0.5, b.z0 - 0.5, b.x1 + 0.5, b.z1 + 0.5]);
    const H = this._hall;
    if (H) R.push([H.x0 - 0.45, H.z0 - 0.45, H.x1 + 0.45, H.z1 + 0.45]);
    R.push([5.85, 24.5, 8.90, 38.9]);     // arcade colonnade walkway
    R.push([6.80, -4.40, 21.8, 0.00]);    // alley floor — dressed by _alley()
    R.push([-4.70, 39.9, 3.50, 43.6]);    // gate piers
    R.push([-9.90, 13.5, -8.60, 17.0]);   // market-hall external stair
    this._noGo = R;
    return R;
  }

  _inNoGo(x, z) {
    const R = this._noGoRects();
    for (let i = 0; i < R.length; i++) {
      const r = R[i];
      if (x > r[0] && x < r[2] && z > r[1] && z < r[3]) return true;
    }
    return false;
  }

  _nearFieldFloor() {
    const prev = this._zone;
    this._zoneSet('dress');
    const place = (x, z, r, mid) => {
      if (x < -21.6 || x > 21.0 || z < -30.2 || z > 39.2) return false;
      // stay off the raised sidewalk strips flanking the carriageway
      const ax = x < 0 ? -x : x;
      if (ax > 5.12 && ax < 7.30 && z > -31 && z < 24.1) return false;
      if (this._inNoGo(x, z)) return false;
      if (this._onStation(x, z, mid ? 1.4 : 0.9)) return false;
      if (!this._clearAt(x, z, r, 0.70)) { this.stats.dressSkip++; return false; }
      this._reserve(x, z, r + 0.40);
      if (mid) { this._midBreaker(x, z); this.stats.dressMid++; }
      else { this._lowBreaker(x, z); this.stats.dressLow++; }
      return true;
    };

    // One pass down each hero sightline. The interval is wide on purpose:
    // the job is a readable object every few metres at the SIDES of frame,
    // not a carpet. Nothing lands closer than 5 m or nearer than 1.9 m to the
    // optical axis, so the muzzle is never buried and the lane stays legible.
    for (let v = 0; v < HERO_VIEWS.length; v++) {
      const V = HERO_VIEWS[v];
      const fx = -Math.sin(V.yaw), fz = -Math.cos(V.yaw);
      const rx = fz, rz = -fx;                        // camera-right in world XZ
      const far = V.near + 13.0;
      let side = this._r() < 0.5 ? -1 : 1;
      for (let along = 5.0; along < far; along += this._rr(2.6, 4.1)) {
        const spread = Math.min(5.0, 2.6 + along * 0.20);
        const lat = side * this._rr(1.9, spread);
        side = -side;                                  // alternate shoulders
        const mid = this._r() < 0.55;
        place(V.x + fx * along + rx * lat, V.z + fz * along + rz * lat, mid ? 1.15 : 0.75, mid);
      }
    }
    // general scatter, so the wedges are not the only interrupted floor
    for (let i = 0; i < 300; i++) {
      const mid = this._r() < 0.30;
      const r = mid ? 1.15 : 0.78;
      if (this._r() < 0.44) place(this._rr(-5.0, 5.0), this._rr(-29.0, 23.4), r, mid);
      else place(this._rr(-20.5, 5.2), this._rr(24.2, 38.6), r, mid);
    }
    this._zoneSet(prev);
  }

  // ------------------------------------------------------- dressing passes

  _dressStreet() {
    const prev = this._zone;
    this._zoneSet('dress');
    for (const s of this.spawnPoints) this._reserve(s.pos.x, s.pos.z, 1.5);
    for (const V of HERO_VIEWS) this._reserve(V.x, V.z, 1.35);   // camera stations stay clear
    this._reserve(-7.5, -4.4, 1.8);      // W3 street doorway must stay walkable
    this._reserve(-9.6, 13.2, 1.8);      // market hall external stair
    this._reserve(0, 42.0, 2.4);         // gate opening
    this._reserve(11.0, -2.1, 1.5);      // alley mouth

    // ---- hero silhouettes down the carriageway -----------------------
    this._wreck(3.30, 18.70, 1.74);
    this._wreck(-3.45, 8.20, 0.44);
    this._wreck(3.55, -13.90, 2.55);
    this._wreck(-3.60, -1.70, 1.18);

    // ---- ruined wall stubs: the strongest floor-breakers we have ------
    this._ruinWall(-5.30, 16.20, -2.00, 16.95, 1.32, 'brickB');
    this._ruinWall(1.60, -20.40, 5.00, -19.55, 1.22, 'plasterC');

    // ---- stacked culverts --------------------------------------------
    this._pipeStack(3.55, 15.15, 0.32, 5);
    this._pipeStack(-4.30, 4.30, 1.42, 4);
    this._pipeStack(4.30, -24.60, 0.10, 4);

    // ---- cover lines across the firing lane (always leave a lane) -----
    this._barrierLine(-5.30, 21.55, 1.15, 21.10);
    this._barrierLine(5.25, -9.60, -0.55, -9.20);
    this._barrierLine(-5.15, -18.30, 1.05, -17.90);
    this._barrierLine(5.30, 2.95, 1.30, 2.55);
    this._barrierLine(-5.30, -26.70, -0.90, -26.45);
    this._barrierLine(5.20, 11.80, 2.10, 11.45);

    // ---- hard props ---------------------------------------------------
    this._kioskProp(4.60, 12.45, 2.15, false);
    this._kioskProp(-4.45, -6.10, 0.62, true);
    this._kioskProp(4.45, -17.30, 3.30, false);
    this._spoolProp(-4.20, 19.40, 0.55);
    this._spoolProp(4.60, -4.30, 2.30);
    this._spoolProp(-4.55, -11.60, 1.05);
    this._tyreStack(-4.85, 10.60, 5);
    this._tyreStack(4.90, -1.45, 4);
    this._tyreStack(-4.95, -22.60, 5);
    this._tyreStack(2.35, 23.05, 4);
    this._drumCluster(4.70, 6.90, 4);
    this._drumCluster(-4.60, -20.60, 3);
    this._drumCluster(-2.60, 14.10, 4);
    this._drumCluster(3.90, -7.10, 3);
    this._sandbagWall(-3.15, 22.40, 0.10, 3.0, 3);
    this._sandbagWall(4.55, 20.20, 1.50, 2.4, 3);
    this._sandbagWall(-4.30, -14.30, 1.50, 2.6, 2);
    this._mound(-2.35, 20.15, 1.7, 0.22);
    this._mound(2.05, 11.30, 1.6, 0.19);
    this._mound(-4.20, -16.40, 1.6, 0.19);
    this._mound(1.45, -4.40, 1.5, 0.17);
    this._mound(-1.60, -24.00, 1.5, 0.18);
    this._mound(0.90, 6.60, 1.4, 0.15);

    // and a scatter of loose items out in the carriageway itself
    for (let i = 0; i < 26; i++) {
      const x = this._rr(-4.9, 4.9), z = this._rr(-29, 23.4);
      if (!this._clearAt(x, z, 0.75)) continue;
      this._reserve(x, z, 0.8);
      const gy = this._surfY(x, z);
      const k = this._r();
      if (k < 0.25) this._iPlace('tyre', x, gy + 0.02, z, this._r() * 6.28, 1, 0.84, this._r() < 0.6 ? 1.5 : 0, [1, 0, 0.2]);
      else if (k < 0.45) this._iPlace('cinder', x, gy + 0.01, z, this._r() * 6.28, 1, 0.88, this._rr(-0.2, 0.2));
      else if (k < 0.62) this._iPlace('pallet', x, gy + 0.01, z, this._r() * 6.28, 1, 0.88, this._rr(-0.1, 0.1));
      else if (k < 0.76) this._iPlace('barrel', x, gy + 0.29, z, this._r() * 6.28, 1, 0.80, 1.55, [1, 0, 0.2]);
      else if (k < 0.88) this._iPlace('plank', x, gy + 0.01, z, this._r() * 6.28, 1, 0.85);
      else this._rubblePile(x, z, 1.1, 10, true);
    }
    this._zoneSet(prev);
  }

  _dressPlaza() {
    const prev = this._zone;
    this._zoneSet('dress');
    // more stalls trading in the arcade's shade
    this._stall(4.15, 26.90, 1.62);
    this._stall(4.35, 31.10, 1.57);
    this._stall(-11.30, 34.30, 0.22);

    this._wreck(-4.85, 35.90, 0.96);
    this._wreck(2.70, 24.60, 2.35);

    this._ruinWall(-12.40, 27.00, -8.80, 28.30, 1.45, 'brickB');
    this._ruinWall(0.40, 35.20, 3.80, 36.55, 1.35, 'plasterC');
    this._ruinWall(-20.60, 33.40, -20.30, 37.20, 1.30, 'plasterC');

    this._pipeStack(-5.60, 25.85, 0.20, 5);
    this._pipeStack(-11.00, 33.60, 1.10, 4);

    this._barrierLine(-5.00, 38.60, 1.90, 38.35);
    this._barrierLine(-8.40, 24.25, -3.40, 24.45);

    this._kioskProp(-11.60, 30.60, 0.75, true);
    this._kioskProp(-4.60, 24.60, 3.05, false);
    this._kioskProp(-16.40, 37.20, 1.90, false);
    this._spoolProp(-2.40, 28.40, 1.10);
    this._spoolProp(-16.90, 24.90, 0.35);
    this._tyreStack(-16.20, 32.80, 5);
    this._tyreStack(2.80, 37.60, 4);
    this._tyreStack(-1.20, 26.40, 4);
    this._drumCluster(-13.20, 26.10, 4);
    this._drumCluster(0.90, 33.30, 3);
    this._drumCluster(-8.40, 35.20, 3);
    this._drumCluster(-19.20, 34.90, 3);

    this._mound(-6.30, 27.30, 2.2, 0.28);
    this._mound(1.90, 30.60, 1.9, 0.23);
    this._mound(-13.80, 37.20, 2.4, 0.30);
    this._mound(-17.90, 28.70, 2.0, 0.24);
    this._mound(-2.60, 32.90, 1.8, 0.22);
    this._mound(3.60, 21.60, 1.7, 0.21);

    this._zoneSet(prev);
  }

  /**
   * Goods and spoil stacked along every hard edge. Runs last so the big
   * silhouettes get first claim on the floor and these fill in around them.
   */
  _dressEdges() {
    const prev = this._zone;
    this._zoneSet('dress');
    for (let z = -29.4; z < 23.8; z += this._rr(3.4, 5.2)) {
      this._wallPile(6.42 + this._rr(-0.12, 0.12), z, -1, 0);
    }
    for (let z = -28.1; z < 23.8; z += this._rr(3.4, 5.2)) {
      this._wallPile(-6.42 + this._rr(-0.12, 0.12), z, 1, 0);
    }
    for (let x = -19.4; x < 4.6; x += this._rr(2.5, 3.8)) this._wallPile(x, 24.42, 0, 1);
    for (let z = 25.2; z < 38.2; z += this._rr(2.6, 4.0)) this._wallPile(5.78, z, -1, 0);
    for (let x = -19.4; x < 3.8; x += this._rr(2.8, 4.3)) this._wallPile(x, 37.90, 0, -1);
    for (let z = 25.0; z < 38.4; z += this._rr(3.2, 5.0)) this._wallPile(-20.55, z, 1, 0);
    this._zoneSet(prev);
  }

  // =====================================================================
  //  DETAIL PASS — rooftops, damage, clutter, grounding
  // =====================================================================

  _buildDetail() {
    // ground every wall in the compound before anything else is scattered
    this._dressWallBases();
    this._pavingBreak(48);
    this._roadScars();

    this._zoneSet('roof');
    for (const b of (this._blds || [])) {
      if (b.id === 'O') continue;
      this._roofKit(b);
    }
    this._hallRoof();

    this._zoneSet('detail');
    // laundry lines strung across the street between the two rows
    for (const z of [-13.5, -8.0, 2.4, 7.6, 16.4]) {
      const y = 4.4 + this._rr(-0.5, 0.9);
      this._wire(-6.9, y, z, 6.9, y - 0.15, z + this._rr(-1.2, 1.2), 1.05);
      const n = this._ri(3, 6);
      for (let k = 0; k < n; k++) {
        const t = 0.14 + (k + 0.5) / (n + 0.4) * 0.72;
        const x = -6.9 + 13.8 * t;
        const cz = z + this._rr(-0.3, 0.3) + t * 0.6;
        const cw = this._rr(0.55, 1.05), ch = this._rr(0.7, 1.4);
        // A 1.4 m sheet hung 1.6 m in front of a camera station fills half the
        // frame with a flat tan quad. Keep the stations' near field clear.
        if (this._onStation(x, cz, 2.2)) continue;
        this._clothSheet(x, y - 0.20 - Math.sin(Math.PI * t) * 1.0, cz, cw, ch, 0);
      }
    }
    // rubble banked against the base of the street walls
    for (const [x, z, r, n] of [
      [-7.6, -16.0, 1.8, 22], [-8.2, 5.2, 2.1, 26], [7.7, -9.0, 1.7, 20],
      [7.6, 14.5, 2.0, 24], [-7.4, -24.0, 1.5, 18], [6.9, 26.6, 1.9, 22],
      [-21.3, 18.0, 1.7, 20], [-3.2, 40.2, 2.6, 30], [3.0, 39.4, 2.2, 26],
    ]) this._rubblePile(x, z, r, n, true);

    // damage passes on the façades the hero cameras see
    this._pockWall(7.0, 0.4, 4.6, 0, -1, 0, 90, 12.0);      // east row, street side
    this._pockWall(7.0, 0.4, 3.6, -20, -1, 0, 50, 8.0);
    this._pockWall(-7.0, 0.4, 4.2, 5, 1, 0, 70, 11.0);      // west row, street side
    this._pockWall(-7.0, 0.4, 3.4, -22, 1, 0, 46, 8.0);
    this._pockWall(6.4, 0.5, 4.0, 31, -1, 0, 46, 6.0);      // arcade
    this._pockWall(-8.4, 0.5, 2.5, 19, -1, 0, 34, 4.4);     // market hall east wall
    this._pockWall(-3.6, 0.6, 5.0, 40.0, 0, 1, 40, 8.0);    // north block

    // sand drifted against the kerbs and wall bases
    for (let z = -30; z < 24; z += this._rr(2.2, 5.0)) {
      for (const s of [-1, 1]) {
        const x = s * (ROAD_HW - 0.30);
        this._box('sand', x, this._surfY(x, z) + 0.055, z, this._rr(1.0, 2.4), 0.13, this._rr(1.4, 3.4),
          { rotY: this._rr(-0.2, 0.2), grime: 1.2, shade: 1.04, noCast: true, zone: 'detail' });
      }
    }
    for (let i = 0; i < 26; i++) {
      const x = this._rr(-21, 5), z = this._rr(23.8, 39);
      this._box('sand', x, this._surfY(x, z) + 0.05, z, this._rr(1.2, 3.2), 0.11, this._rr(1.2, 3.0),
        { rotY: this._r() * 3, grime: 1.2, shade: 1.05, noCast: true, zone: 'detail' });
    }
    // loose debris field across the plaza and street
    for (let i = 0; i < 190; i++) {
      const x = this._rr(-22, 8), z = this._rr(-28, 40);
      if (Math.abs(x) < 5 && z < 22 && this._r() < 0.5) continue;
      this._iPlace(this._r() < 0.55 ? 'shard' : 'brickBit', x, this._surfY(x, z) + 0.015, z,
        this._r() * 6.28, 0.5 + this._r() * 0.9, 0.85 + this._r() * 0.25, this._rr(-0.2, 0.2));
    }

    // The W3 interior and the façade trim run dead last: both consume their own
    // RNG streams, so dressing them cannot reshuffle the exterior ground pass
    // that round 2 settled on.
    this._interiorRoom();
    this._facadeDetail();
    this._zoneSet('props');
  }

  _roofKit(b) {
    const top = b.top;
    const cx = (b.x0 + b.x1) * 0.5, cz = (b.z0 + b.z1) * 0.5;
    const w = b.x1 - b.x0, d = b.z1 - b.z0;
    if (w < 4 || d < 4) return;
    const R = (fx, fz) => [b.x0 + 0.9 + fx * (w - 1.8), b.z0 + 0.9 + fz * (d - 1.8)];

    // stair head / roof access box
    const [sx, sz] = R(0.30, 0.70);
    this._box(b.mat, sx, top + 1.15, sz, 2.1, 2.3, 1.9, { base: top });
    this._box('concrete', sx, top + 2.36, sz, 2.34, 0.14, 2.14, { base: top, shade: 1.05 });
    this._box('dark', sx - 1.02, top + 1.02, sz, 0.09, 2.0, 0.95, { base: top, raw: true, shade: 0.3 });

    // water tanks on an angle-iron stand
    const [tx, tz] = R(0.72, 0.28);
    for (let i = 0; i < (this._r() < 0.5 ? 2 : 1); i++) {
      const px = tx + i * 1.25;
      for (const [ox, oz] of [[-0.42, -0.42], [0.42, -0.42], [-0.42, 0.42], [0.42, 0.42]]) {
        this._box('rust', px + ox, top + 0.30, tz + oz, 0.06, 0.60, 0.06, { base: top });
      }
      this._cyl('polymer', px, top + 1.15, tz, 0.55, 0.55, 1.10, 12, { base: top });
      this._cyl('polymer', px, top + 1.74, tz, 0.20, 0.30, 0.12, 10, { base: top });
      this._cyl('rust', px + 0.5, top + 0.9, tz + 0.4, 0.028, 0.028, 1.8, 6,
        { base: top, zone: 'detail', tilt: [0, 0, 1, 0.25] });
    }

    // satellite dishes on the parapet
    for (let i = 0; i < this._ri(1, 3); i++) {
      const [dx, dz] = R(this._r(), this._r());
      this._dish(dx, top + 1.0, dz, this._rr(-2.6, -1.4));
    }
    // AC condensers + vents
    for (let i = 0; i < this._ri(1, 3); i++) {
      const [ax, az] = R(this._r(), this._r());
      this._box('metal', ax, top + 0.36, az, 0.9, 0.72, 0.72, { base: top, rotY: this._r() * 3 });
      this._box('metal', ax, top + 0.76, az, 0.94, 0.08, 0.76, { base: top, shade: 0.9 });
    }
    // rooftop laundry line
    if (this._r() < 0.75) {
      const y = top + 1.5;
      this._cyl('rust', b.x0 + 1.0, top + 0.75, b.z0 + 1.2, 0.035, 0.045, 1.5, 6, { base: top });
      this._cyl('rust', b.x1 - 1.0, top + 0.75, b.z1 - 1.2, 0.035, 0.045, 1.5, 6, { base: top });
      this._wire(b.x0 + 1.0, y, b.z0 + 1.2, b.x1 - 1.0, y, b.z1 - 1.2, 0.22);
      const n = this._ri(2, 4);
      for (let k = 0; k < n; k++) {
        const t = 0.2 + (k + 0.5) / (n + 0.3) * 0.6;
        this._clothSheet(b.x0 + 1.0 + (b.x1 - b.x0 - 2) * t, y - 0.16,
          b.z0 + 1.2 + (b.z1 - b.z0 - 2.4) * t, this._rr(0.5, 0.9), this._rr(0.6, 1.1), 0.4);
      }
    }
    // scattered rooftop clutter
    for (let i = 0; i < this._ri(2, 5); i++) {
      const [px, pz] = R(this._r(), this._r());
      const k = this._r();
      if (k < 0.35) this._iPlace('crate', px, top, pz, this._r() * 6.28, 1, 0.95);
      else if (k < 0.6) this._iPlace('barrel', px, top, pz, this._r() * 6.28, 1, 0.92);
      else if (k < 0.8) this._iPlace('tyre', px, top, pz, this._r() * 6.28, 1, 0.9, this._rr(-0.05, 0.05));
      else this._iPlace('bucket', px, top, pz, this._r() * 6.28, 1, 0.95);
    }
    if (b.broken) {
      const [px, pz] = R(this._r() * 0.3 + 0.6, this._r() * 0.3 + 0.6);
      this._rubblePile(px, pz, 1.6, 16, false, top);
      for (let i = 0; i < 5; i++) this._rebar(px + this._rr(-1, 1), top + 0.1, pz + this._rr(-1, 1), 1);
    }
  }

  _dish(x, y, z, ry) {
    const g = new THREE.SphereGeometry(0.44, 12, 7, 0, Math.PI * 2, 0, 0.62);
    scaleUv(g, 1.4, 0.7);
    g.rotateX(Math.PI * 0.5 - 0.45);
    g.translate(x, y + 0.15, z);
    rotYg(g, x, z, ry);
    this._add('metal', g, { base: y - 1.0, shade: 1.06, zone: 'detail', noCollide: true });
    this._cyl('rust', x, y - 0.45, z, 0.035, 0.05, 1.0, 6, { base: y - 1.0, zone: 'detail' });
    this._box('rust', x, y - 1.0, z, 0.30, 0.08, 0.30, { base: y - 1.0, zone: 'detail' });
    const ax = x + Math.sin(ry) * 0.42, az = z + Math.cos(ry) * 0.42;
    this._cyl('metal', ax, y + 0.22, az, 0.02, 0.02, 0.5, 5,
      { base: y - 1.0, zone: 'detail', tilt: [1, 0, 0, 0.9], rotY: ry });
    this._box('polymer', ax, y + 0.42, az, 0.09, 0.09, 0.09, { base: y - 1.0, zone: 'detail' });
  }

  /** the market-hall roof: this is the "vista" camera's foreground */
  _hallRoof() {
    const H = this._hall;
    if (!H) return;
    const deck = H.deck;
    this._zoneSet('roof');
    // a fighting position of sandbags on the north edge, right of the vista camera
    this._sandbagWallY(-17.2, 22.7, 0.0, 3.0, 3, deck);
    this._sandbagWallY(-11.6, 22.6, 0.0, 2.0, 2, deck);
    this._iPlace('crateBig', -12.6, deck, 21.2, 0.3, 1, 0.95);
    this._iPlace('crate', -12.4, deck + 0.74, 21.1, -0.4, 0.95, 1.0);
    this._iPlace('barrel', -19.4, deck, 21.6, 0.5, 1, 0.92);
    this._iPlace('barrel', -19.9, deck, 21.1, 1.2, 1, 0.9);
    this._iPlace('tyre', -10.2, deck, 16.6, 0.4, 1, 0.9);
    // water tank + dish + AC on the far side so the skyline reads
    for (const [ox, oz] of [[-0.46, -0.46], [0.46, -0.46], [-0.46, 0.46], [0.46, 0.46]]) {
      this._box('rust', -19.6 + ox, deck + 0.34, 16.6 + oz, 0.07, 0.68, 0.07, { base: deck });
    }
    this._cyl('polymer', -19.6, deck + 1.24, 16.6, 0.62, 0.62, 1.18, 12, { base: deck });
    this._cyl('polymer', -19.6, deck + 1.88, 16.6, 0.22, 0.32, 0.13, 10, { base: deck });
    this._dish(-16.0, deck + 1.05, 15.6, -2.2);
    this._box('metal', -13.0, deck + 0.38, 15.3, 0.95, 0.76, 0.74, { base: deck, rotY: 0.4 });
    // laundry across the deck — catches the low sun translucently
    this._cyl('rust', -20.2, deck + 0.85, 19.4, 0.035, 0.05, 1.7, 6, { base: deck });
    this._cyl('rust', -9.6, deck + 0.85, 19.0, 0.035, 0.05, 1.7, 6, { base: deck });
    this._wire(-20.2, deck + 1.62, 19.4, -9.6, deck + 1.62, 19.0, 0.30);
    for (let k = 0; k < 6; k++) {
      const t = 0.10 + k * 0.155;
      this._clothSheet(-20.2 + 10.6 * t, deck + 1.50 - Math.sin(Math.PI * t) * 0.28, 19.4 - 0.4 * t,
        this._rr(0.6, 1.1), this._rr(0.75, 1.45), 0.06);
    }
    this._rubblePile(-20.0, 22.4, 1.4, 14, false, deck);
    this.coverPoints.push({ pos: new THREE.Vector3(-17.2, deck + 0.6, 22.7), dir: new THREE.Vector3(0, 0, 1), height: 0.6 });
    this._zoneSet('detail');
  }

  _sandbagWallY(cx, cz, ang, len, courses, y) {
    const bw = 0.56, ch = 0.20;
    const n = Math.max(2, Math.round(len / bw));
    for (let c = 0; c < courses; c++) {
      const off = (c & 1) ? bw * 0.5 : 0;
      for (let i = 0; i < n; i++) {
        const t = (i + 0.5) * bw + off - len * 0.5;
        this._iPlace('sandbag', cx + Math.cos(ang) * t, y + c * ch + ch * 0.5, cz + Math.sin(ang) * t,
          -ang + this._rr(-0.09, 0.09), [1, 1, 1], 0.9 + this._r() * 0.2, this._rr(-0.05, 0.05));
      }
    }
  }

  // =====================================================================
  //  GROUND
  // =====================================================================

  _groundShade(x, z) {
    let s = 0.94 + 0.16 * vnoise(x * 0.085, 0.5, z * 0.085);
    s *= 0.93 + 0.14 * vnoise(x * 0.34 + 7.1, 1.5, z * 0.34);
    // grime pooling against the building lines
    const wallEdge = Math.min(
      Math.abs(Math.abs(x) - 7.0) + (Math.abs(x) > 7.4 ? 8 : 0),
      Math.abs(z - 23.4) + (z > 24 || z < 14 ? 8 : 0));
    s *= 1 - 0.14 * Math.exp(-wallEdge * 1.1);
    const st = this._stains;
    for (let i = 0; i < st.length; i++) {
      const dx = x - st[i][0], dz = z - st[i][1], r = st[i][2];
      const d2 = (dx * dx + dz * dz) / (r * r);
      if (d2 < 1) {
        const f = 1 - d2;
        s *= 1 - st[i][3] * f * f;
      }
    }
    return s;
  }

  _gridMesh(mat, x0, z0, x1, z1, cell, hFn, o) {
    const nx = Math.max(1, Math.round((x1 - x0) / cell));
    const nz = Math.max(1, Math.round((z1 - z0) / cell));
    const dx = (x1 - x0) / nx, dz = (z1 - z0) / nz;
    const vw = nx + 1, vh = nz + 1;
    const cnt = vw * vh;
    const pos = new Float32Array(cnt * 3);
    const nor = new Float32Array(cnt * 3);
    const uvs = new Float32Array(cnt * 2);
    const col = new Float32Array(cnt * 3);
    const ts = this._tsz[mat] || 3.0;
    const shadeK = (o && o.shade) || 1;
    for (let j = 0; j < vh; j++) {
      for (let i = 0; i < vw; i++) {
        const k = j * vw + i;
        const x = x0 + i * dx, z = z0 + j * dz;
        const y = hFn(x, z);
        pos[k * 3] = x; pos[k * 3 + 1] = y; pos[k * 3 + 2] = z;
        const hx = (hFn(x + 0.25, z) - hFn(x - 0.25, z)) / 0.5;
        const hz = (hFn(x, z + 0.25) - hFn(x, z - 0.25)) / 0.5;
        const l = 1 / Math.sqrt(hx * hx + hz * hz + 1);
        nor[k * 3] = -hx * l; nor[k * 3 + 1] = l; nor[k * 3 + 2] = -hz * l;
        uvs[k * 2] = x / ts; uvs[k * 2 + 1] = z / ts;
        const s = this._groundShade(x, z) * shadeK;
        col[k * 3] = clamp(s * 1.015, 0, 1.6);
        col[k * 3 + 1] = clamp(s, 0, 1.6);
        col[k * 3 + 2] = clamp(s * 0.955, 0, 1.6);
      }
    }
    const idx = cnt > 65535 ? new Uint32Array(nx * nz * 6) : new Uint16Array(nx * nz * 6);
    let p = 0;
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const a = j * vw + i, b = a + 1, c = a + vw, d = c + 1;
        idx[p++] = a; idx[p++] = c; idx[p++] = b;
        idx[p++] = b; idx[p++] = c; idx[p++] = d;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    return this._add(mat, g, { zone: (o && o.zone) || this._zone, raw: true, noCast: true });
  }

  _buildGround() {
    this._zoneSet('ground');

    // --- outer desert: cheap, mostly eaten by the haze ----------------
    // Built as a RING around the compound, never across it. The previous
    // single 380 m plane evaluated to +0.9 .. +1.4 m over the playspace and
    // buried the road, the kerbs, the wall skirts and every piece of set
    // dressing under one continuous sheet of sand — which is exactly the
    // "unbroken prop-free plane" the critic saw. A ring cannot do that, and
    // its inner edge is pinned to the compound's own height field so the
    // two meshes meet without a crack or a step.
    const CX0 = -34, CZ0 = -34, CX1 = 30, CZ1 = 50;   // compound floor extents
    const dune = (x, z) => (
      Math.sin(x * 0.011) * 0.9 + Math.cos(z * 0.009) * 0.8 +
      vnoise(x * 0.012, 3, z * 0.012) * 2.4 - 1.2);
    const hDesert = (x, z) => {
      const ex = Math.max(0, Math.abs(x + 2) - 32);
      const ez = Math.max(0, Math.abs(z - 8) - 42);
      const k = smooth(6, 150, Math.sqrt(ex * ex + ez * ez));
      return this._hCore(x, z) - 0.02 + dune(x, z) * 1.30 * k;
    };
    // shared cell size + coincident bounds => the four strips share vertices
    this._gridMesh('sand', -258, -258, 254, CZ0, 8, hDesert, { shade: 1.0 });
    this._gridMesh('sand', -258, CZ1, 254, 274, 8, hDesert, { shade: 1.0 });
    this._gridMesh('sand', -258, CZ0, CX0, CZ1, 8, hDesert, { shade: 1.0 });
    this._gridMesh('sand', CX1, CZ0, 254, CZ1, 8, hDesert, { shade: 1.0 });

    // --- compound floor ----------------------------------------------
    // 1 m proud of the ring's inner edge on every side: the ring sits 2 cm
    // lower, so the overlap guarantees the seam can never open a slit, and
    // the separation is far above depth precision at that range.
    this._gridMesh('sand', CX0 - 1, CZ0 - 1, CX1 + 1, CZ1 + 1, 1.0, (x, z) => this._hCore(x, z), {});

    // --- road: two carriageways either side of a concrete drainage channel
    const potholes = [];
    for (let i = 0; i < 18; i++) {
      const s = this._r() < 0.5 ? -1 : 1;
      potholes.push([s * this._rr(1.3, 4.7), this._rr(-29, 23), this._rr(0.5, 1.5), this._rr(0.05, 0.13)]);
    }
    const hRoad = (x, z) => {
      let y = this._hRoad(x);
      y += vnoise(x * 0.42, 21, z * 0.42) * 0.020;
      for (let i = 0; i < potholes.length; i++) {
        const dx = x - potholes[i][0], dz = z - potholes[i][1], r = potholes[i][2];
        const d2 = (dx * dx + dz * dz) / (r * r);
        if (d2 < 1) y -= potholes[i][3] * (1 - d2) * (1 - d2);
      }
      return y;
    };
    // identical cell size everywhere so the three strips share edge vertices
    const RC = 0.40;
    this._gridMesh('asphalt', -ROAD_HW, -31, -CHAN_HW, 24.4, RC, hRoad, {});
    this._gridMesh('asphalt', CHAN_HW, -31, ROAD_HW, 24.4, RC, hRoad, {});
    this._gridMesh('concreteB', -CHAN_HW, -31, CHAN_HW, 24.4, RC, hRoad, { shade: 1.02 });
    // road continues north through the plaza as a worn track
    this._gridMesh('asphalt', -ROAD_HW + 0.6, 24.4, ROAD_HW - 0.6, 30.0, 0.9,
      (x, z) => 0.028 - 0.026 * (x / ROAD_HW) * (x / ROAD_HW) - (z - 24.4) * 0.004, { shade: 1.03 });

    // --- market-hall interior floor -----------------------------------
    const H = this._hall;
    if (H) {
      this._gridMesh('concreteB', H.x0 + 0.34, H.z0 + 0.34, H.x1 - 0.5, H.z1 - 0.5, 1.4,
        () => 0.16, { shade: 0.95 });
    }
    this._zoneSet('props');
  }

  // =====================================================================
  //  FINALIZE — merge buckets, build instanced meshes, register colliders
  // =====================================================================

  _finalize() {
    const reg = this.g.registry;

    for (const b of this._buckets.values()) {
      if (!b.geos.length) continue;
      let merged;
      try {
        merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
      } catch (e) {
        console.warn('[Level] merge failed for', b.mat, b.zone, e);
        continue;
      }
      if (!merged) continue;
      merged.computeBoundingSphere();
      merged.computeBoundingBox();
      const mesh = new THREE.Mesh(merged, this._mats[b.mat]);
      mesh.name = 'lvl_' + b.zone + '_' + b.mat;
      mesh.castShadow = b.cast;
      mesh.receiveShadow = b.recv;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      mesh.userData.bsSurface = this._mats[b.mat]?.userData?.bsSurface;
      this.root.add(mesh);
      this.meshes.push(mesh);
      this.stats.tris += merged.index ? merged.index.count / 3 : 0;
      if (b.collide) this.colliders.push(mesh);
      b.geos.length = 0;
    }
    this.stats.buckets = this._buckets.size;

    for (const p of this._instPools.values()) {
      const n = p.items.length;
      if (!n) continue;
      const im = new THREE.InstancedMesh(p.geo, this._mats[p.mat], n);
      im.name = 'lvl_inst_' + p.name;
      im.castShadow = p.cast;
      im.receiveShadow = p.recv;
      im.instanceMatrix.setUsage(THREE.StaticDrawUsage);
      const col = new THREE.Color();
      im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
      for (let i = 0; i < n; i++) {
        im.setMatrixAt(i, p.items[i].m);
        const s = p.items[i].s;
        col.setRGB(s, s * 0.995, s * 0.975);
        im.setColorAt(i, col);
      }
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.computeBoundingSphere();
      im.frustumCulled = true;
      this.root.add(im);
      this.meshes.push(im);
      this.stats.instances += n;
      this.stats.tris += n * (p.geo.index ? p.geo.index.count / 3 : 0);
      if (p.collide) this.colliders.push(im);
      p.items.length = 0;
    }

    this.stats.draws = this.meshes.length;
    this.root.updateMatrixWorld(true);

    if (reg && Array.isArray(reg.colliders)) {
      for (const m of this.colliders) reg.colliders.push(m);
    }
    if (reg) {
      reg.level = this;
      reg.navBounds = this.navBounds;
      reg.spawnPoints = this.spawnPoints;
      reg.coverPoints = this.coverPoints;
    }
  }
}

export default Level;
