// ============================================================================
// BLACKSITE — src/fx/Decals.js
// Owner: Decals agent.  Persistent projected surface marks.
//
// WHAT THIS IS
//   A pooled, single-draw-call projected-decal system.  Decals are not quads
//   stuck onto a plane — every decal CLIPS the real level triangles inside a
//   projector box (Sutherland–Hodgman against the 6 box planes) and re-emits
//   them displaced a hair along their own surface normal.  That is why a
//   bullet hole placed on a wall/floor junction WRAPS the corner instead of
//   hovering, and why nothing z-fights: the decal geometry IS the surface
//   geometry, offset ~2.5 mm and biased in the depth test.
//
//   Every decal texture is generated procedurally on a 2D canvas at boot:
//   concrete craters with bright spall rims and radial cracking, metal
//   punch-through with peeled bright petals, wood splintering, glass
//   spiderweb fracture, sand craters, scorch bursts, directional blood
//   splatter with throw and drips, plus lived-in world dressing (run-off
//   grime streaks, oil pools, weathered posters).  Albedo, height→normal and
//   ORM (AO / roughness / metalness) are painted for each mark from the SAME
//   shape sequence, so a fresh crater reads as rough exposed aggregate and a
//   metal petal reads as bare, shiny steel under the golden-hour key.
//
// PUBLIC API (collaborators — this is stable)
//   game.decals.addImpact(point, normal, surface, opts) -> slot | -1
//        surface: 'concrete'|'metal'|'sand'|'wood'|'flesh'|'glass' (or any
//        Materials key / material instance — resolved through Materials).
//        opts: { size, angle, tint, life, opacity, tileIndex, minGap }
//   game.decals.addScorch(point, normal, radius)          -> slot | -1
//   game.decals.addBlood(point, normal, size)             -> slot | -1
//   game.decals.addBloodSpray(point, dirVec3, scale)      -> slot | -1
//   game.decals.addDecal(tile, point, normal, w, h, opts) -> slot | -1
//   game.decals.clear(alsoScene = false)
//   game.decals.setEnabled(bool)
//   game.decals.mesh    THREE.Mesh   (one draw call, every decal)
//   game.decals.stats   { active, scene, dynamic, capacity, tris, verts, soup, buildMs }
//
//   Opt an object out of receiving decals:  obj.userData.bsNoDecal = true
//
// BUS
//   listens: 'shot' (caches the fire direction so blood throws the right way),
//            'impact', 'hit', 'kill', 'explosion'.   emits: nothing.
//
// PERFORMANCE
//   One Mesh, one material, one draw call.  Zero allocation in update() — the
//   whole system runs out of preallocated typed arrays.  Attribute uploads
//   are partial (addUpdateRange) so spawning a decal never re-uploads the
//   vertex pool.  Per-decal opacity/tint lives in a 528x1 data texture read in
//   the vertex shader, so fading 400 decals costs one 2 KB upload per frame.
// ============================================================================

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  Small math / PRNG toolbox
 * ------------------------------------------------------------------ */

const TAU = Math.PI * 2;
const EMPTY = {};
const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
function sstep(e0, e1, x) {
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}
function mulberry32(a) {
  a = (a >>> 0) || 0x9e3779b9;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ *
 *  Pool / atlas constants
 * ------------------------------------------------------------------ */

const TILE_PX = 256;                     // one decal texture tile
const ATLAS_COLS = 5;
const ATLAS_ROWS = 5;
const TILE_COUNT = ATLAS_COLS * ATLAS_ROWS;
const UV_INSET = 0.030;                  // keep sampling off the tile border (mip bleed guard)

const SLOT_VERTS = 144;                  // 48 clipped triangles per decal — enough for a corner wrap
const DYN_CAP = 400;                     // rolling gameplay decals (brief: ~400)
const SCENE_CAP = 190;                   // permanent world-dressing decals placed at init
const CAP = DYN_CAP + SCENE_CAP;

const MAX_TRI = 520000;                  // level triangle-soup ceiling
const MIN_TRI_AREA = 1.4e-4;             // skip sub-centimetre greeble triangles
const CELL = 2.5;                        // XZ broadphase cell size, metres
const OVERSIZE_CELLS = 220;              // a triangle spanning more cells than this is always-tested

// flat-card fallback corners (module scope so the spawn path allocates nothing)
const FB_X = [-1, 1, 1, -1, 1, -1];
const FB_Y = [-1, -1, 1, -1, 1, 1];

// Tile indices — the whole system refers to marks by these.  5x5 atlas.
const T_CONCRETE_A = 0, T_CONCRETE_B = 1, T_CONCRETE_C = 2, T_CONCRETE_D = 3;
const T_PLASTER_A = 4, T_PLASTER_B = 5, T_BRICK_A = 6, T_BRICK_B = 7;
const T_METAL_A = 8, T_METAL_B = 9, T_METAL_C = 10;
const T_WOOD_A = 11, T_WOOD_B = 12, T_GLASS = 13, T_SAND = 14, T_FLESH = 15;
const T_SCORCH_S = 16, T_SCORCH_L = 17, T_BLOOD_A = 18, T_BLOOD_B = 19;
const T_GRIME = 20, T_OIL = 21, T_POSTER_A = 22, T_POSTER_B = 23, T_DUSTFALL = 24;

// painter dispatch + per-tile post-process grain strength
const TILE_DEFS = [
  { fn: '_tConcrete', v: 0, grain: 0.32, seed: 1201 },
  { fn: '_tConcrete', v: 1, grain: 0.36, seed: 1277 },
  { fn: '_tConcrete', v: 2, grain: 0.30, seed: 1319 },
  { fn: '_tConcrete', v: 3, grain: 0.38, seed: 1409 },
  { fn: '_tPlaster', v: 0, grain: 0.30, seed: 1523 },
  { fn: '_tPlaster', v: 1, grain: 0.34, seed: 1607 },
  { fn: '_tBrick', v: 0, grain: 0.34, seed: 1721 },
  { fn: '_tBrick', v: 1, grain: 0.30, seed: 1831 },
  { fn: '_tMetal', v: 0, grain: 0.22, seed: 2311 },
  { fn: '_tMetal', v: 1, grain: 0.26, seed: 2389 },
  { fn: '_tMetalDent', v: 0, grain: 0.24, seed: 2477 },
  { fn: '_tWood', v: 0, grain: 0.30, seed: 3407 },
  { fn: '_tWood', v: 1, grain: 0.34, seed: 3491 },
  { fn: '_tGlass', v: 0, grain: 0.10, seed: 4513 },
  { fn: '_tSand', v: 0, grain: 0.42, seed: 5623 },
  { fn: '_tFlesh', v: 0, grain: 0.26, seed: 6733 },
  { fn: '_tScorch', v: 0, grain: 0.46, seed: 7841 },
  { fn: '_tScorch', v: 1, grain: 0.52, seed: 7907 },
  { fn: '_tBlood', v: 0, grain: 0.24, seed: 8971 },
  { fn: '_tBlood', v: 1, grain: 0.28, seed: 9043 },
  { fn: '_tGrime', v: 0, grain: 0.50, seed: 10111 },
  { fn: '_tOil', v: 0, grain: 0.28, seed: 11213 },
  { fn: '_tPoster', v: 0, grain: 0.30, seed: 12329 },
  { fn: '_tPoster', v: 1, grain: 0.34, seed: 12433 },
  { fn: '_tDustfall', v: 0, grain: 0.44, seed: 13537 },
];

// Fine-grained surface identities. The Bus only speaks the six coarse buckets,
// but Materials tags every material with its real key (plaster / brick /
// asphalt / rustmetal ...), and a hit reads completely differently on each.
const FINE = ['concrete', 'plaster', 'brick', 'asphalt', 'sand',
  'metal', 'rustmetal', 'wood', 'glass', 'flesh'];
// Materials key -> fine surface
const FINE_MAP = {
  concrete: 'concrete', plaster: 'plaster', brick: 'brick', asphalt: 'asphalt',
  sand: 'sand', metal: 'metal', rustmetal: 'rustmetal', gunmetal: 'metal',
  polymer: 'metal', rubber: 'asphalt', wood: 'wood', foliage: 'wood',
  fabric: 'plaster', cloth_tan: 'plaster', glass: 'glass', flesh: 'flesh',
};
// fine -> the six canonical Bus buckets
const COARSE = {
  concrete: 'concrete', plaster: 'concrete', brick: 'concrete', asphalt: 'concrete',
  sand: 'sand', metal: 'metal', rustmetal: 'metal', wood: 'wood',
  glass: 'glass', flesh: 'flesh',
};

const HOLE_TILES = {
  concrete: [T_CONCRETE_A, T_CONCRETE_B, T_CONCRETE_C, T_CONCRETE_D],
  plaster: [T_PLASTER_A, T_PLASTER_B, T_CONCRETE_C, T_PLASTER_A],
  brick: [T_BRICK_A, T_BRICK_B, T_CONCRETE_C],
  asphalt: [T_CONCRETE_D, T_CONCRETE_C, T_CONCRETE_D],
  metal: [T_METAL_A, T_METAL_B, T_METAL_C],
  rustmetal: [T_METAL_C, T_METAL_B, T_METAL_A],
  wood: [T_WOOD_A, T_WOOD_B],
  glass: [T_GLASS],
  sand: [T_SAND],
  flesh: [T_FLESH],
};
// world size range, metres — deliberately wide so no two hits stamp alike
const HOLE_SIZE = {
  concrete: [0.19, 0.40], plaster: [0.21, 0.44], brick: [0.17, 0.35],
  asphalt: [0.19, 0.37], metal: [0.14, 0.29], rustmetal: [0.16, 0.32],
  wood: [0.19, 0.39], glass: [0.50, 0.80], sand: [0.30, 0.52], flesh: [0.14, 0.22],
};
// a decal is tinted toward whatever it sits on so it never reads as a sticker
const SURF_TINT = {
  concrete: [1.00, 0.98, 0.95], plaster: [1.00, 0.99, 0.96], brick: [1.00, 0.94, 0.88],
  asphalt: [0.96, 0.96, 0.99], sand: [1.00, 0.97, 0.90], metal: [0.95, 0.96, 1.00],
  rustmetal: [1.00, 0.94, 0.86], wood: [1.00, 0.96, 0.88],
  flesh: [1.00, 0.97, 0.96], glass: [0.97, 0.99, 1.00],
};
const BUCKETS = FINE;

/* ------------------------------------------------------------------ *
 *  Impact palettes.  Every bullet mark is the same four-zone structure —
 *  soft dust halo -> darker soot bruise -> BRIGHT chipped substrate ->
 *  small dark entry pocket — with the numbers swapped per material.
 *  Nothing here is allowed to reach the grade's black floor: the darkest
 *  value in any palette is well above #000 and carries a warm/cool cast so
 *  it still takes ambient bounce.
 * ------------------------------------------------------------------ */
const CRATER = {
  // concrete: pale grey aggregate under a dusty skin, deep dish, long cracks
  concrete: {
    rOut: 118, rRim: 50, rCore: 17, irr: 0.32,
    dust: [232, 226, 210], dustA: 0.34,
    soot: [70, 62, 53], sootA: 0.48,
    hot: [246, 242, 232], mid: [212, 205, 188], edge: [172, 163, 146],
    pit: [58, 51, 43], pitLo: [86, 78, 67], bounce: [104, 110, 122],
    rgh: 0.99, mt: 0, chips: 44, cracks: 16, crackLen: 1.5,
    hCore: 0.09, hMid: 0.31, hRim: 0.82,
  },
  // plaster / render: thin bright skim blown off, DARK coarse block behind it
  plaster: {
    rOut: 120, rRim: 60, rCore: 22, irr: 0.40,
    dust: [238, 232, 218], dustA: 0.30,
    soot: [88, 79, 68], sootA: 0.34,
    hot: [250, 246, 236], mid: [176, 166, 150], edge: [128, 118, 104],
    pit: [56, 49, 42], pitLo: [82, 73, 62], bounce: [100, 108, 122],
    rgh: 0.98, mt: 0, chips: 30, cracks: 11, crackLen: 1.1,
    hCore: 0.10, hMid: 0.38, hRim: 0.84,
  },
  // brick: sharp pale chip face plus a red-brown dust bloom
  brick: {
    rOut: 112, rRim: 46, rCore: 14, irr: 0.44,
    dust: [176, 108, 76], dustA: 0.34,
    soot: [96, 62, 45], sootA: 0.36,
    hot: [236, 214, 190], mid: [206, 152, 118], edge: [162, 100, 72],
    pit: [64, 44, 34], pitLo: [96, 66, 50], bounce: [110, 106, 112],
    rgh: 0.97, mt: 0, chips: 36, cracks: 9, crackLen: 0.85,
    hCore: 0.10, hMid: 0.34, hRim: 0.82,
  },
  // asphalt / rubber: shallow scuffed dish, grey-black grit, little spall
  asphalt: {
    rOut: 110, rRim: 44, rCore: 13, irr: 0.34,
    dust: [148, 144, 138], dustA: 0.30,
    soot: [62, 59, 56], sootA: 0.46,
    hot: [186, 181, 172], mid: [146, 141, 134], edge: [104, 100, 95],
    pit: [50, 47, 44], pitLo: [74, 70, 66], bounce: [96, 102, 114],
    rgh: 1.0, mt: 0, chips: 40, cracks: 7, crackLen: 0.7,
    hCore: 0.20, hMid: 0.40, hRim: 0.68,
  },
};

/* ================================================================== *
 *  Decals
 * ================================================================== */

export class Decals {

  constructor(game) {
    this.g = game;
    this.enabled = true;
    this._ready = false;

    const q = game?.config?.quality || 'ultra';
    this._q = q;
    this.dynCap = q === 'low' ? 180 : DYN_CAP;
    this.sceneCap = q === 'low' ? 56 : SCENE_CAP;
    this.cap = CAP;
    this.tilePx = q === 'low' ? 128 : TILE_PX;
    this.atlasPx = this.tilePx * ATLAS_COLS;

    // ---- per-slot state (all preallocated, never resized) ----------------
    this.sAge = new Float32Array(CAP);
    this.sLife = new Float32Array(CAP);
    this.sFade = new Float32Array(CAP);
    this.sUsed = new Uint8Array(CAP);
    this.sPerm = new Uint8Array(CAP);
    this.sVerts = new Int32Array(CAP);
    this.sPx = new Float32Array(CAP);
    this.sPy = new Float32Array(CAP);
    this.sPz = new Float32Array(CAP);
    this.sRad = new Float32Array(CAP);
    this.sBoost = new Float32Array(CAP);

    this._active = new Int32Array(CAP);
    this._activeAt = new Int32Array(CAP).fill(-1);
    this._activeN = 0;
    this._ring = 0;
    this._sceneN = 0;
    this._highWater = 0;

    // ---- per-decal GPU state row (rgb = tint, a = opacity) ---------------
    this._stateBytes = new Uint8Array(CAP * 4);
    for (let i = 0; i < CAP; i++) {
      this._stateBytes[i * 4] = 255;
      this._stateBytes[i * 4 + 1] = 255;
      this._stateBytes[i * 4 + 2] = 255;
      this._stateBytes[i * 4 + 3] = 0;
    }

    // ---- clipping scratch ------------------------------------------------
    this._pA = new Float32Array(96);
    this._pB = new Float32Array(96);
    this._qBuf = new Int32Array(6144);
    this._qStamp = null;
    this._stamp = 0;

    // ---- triangle soup / broadphase -------------------------------------
    this.triCount = 0;
    this.triV = null; this.triN = null; this.triS = null; this.triA = null;
    this._grid = null;
    this._floorTri = null; this._floorN = 0; this._floorMax = 1;
    this._wallTri = null; this._wallN = 0; this._wallMax = 1;

    // ---- misc scratch (zero allocation at runtime) ----------------------
    this._shotDir = new THREE.Vector3(0, 0, -1);
    this._nm3 = new THREE.Matrix3();
    this._rnd = mulberry32(0x51A2C7);
    this._hit = { t: 0, nx: 0, ny: 1, nz: 0, x: 0, y: 0, z: 0, tri: -1 };
    this._sample = { x: 0, y: 0, z: 0, nx: 0, ny: 1, nz: 0, surf: 'concrete', tri: -1 };
    this._streakP = { x: 0, y: 0, z: 0 };
    this._clusterP = { x: 0, y: 0, z: 0 };

    this.stats = {
      active: 0, scene: 0, dynamic: 0, capacity: CAP,
      tris: 0, verts: 0, soup: 0, buildMs: 0,
    };

    // Bus wiring happens now so nothing is missed; every handler no-ops until
    // init() has finished building the pool.
    const bus = game?.bus;
    if (bus) {
      bus.on('shot', (e) => this._onShot(e));
      bus.on('impact', (e) => this._onImpact(e));
      bus.on('explosion', (e) => this._onExplosion(e));
      bus.on('hit', (e) => this._onHit(e));
      bus.on('kill', (e) => this._onKill(e));
    }
  }

  /* ================================================================ *
   *  LIFECYCLE
   * ================================================================ */

  async init() {
    const t0 = this._now();

    this._buildAtlases();
    await this._yield();
    this._buildMesh();
    await this._yield();
    this._buildSoup();
    await this._yield();
    this._buildGrid();
    this._classifyTris();
    await this._yield();

    this._ready = true;
    this._placeSceneDecals();

    this.stats.buildMs = +(this._now() - t0).toFixed(1);
    this.stats.soup = this.triCount;
  }

  _now() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()); }
  _yield() { return new Promise(r => setTimeout(r, 0)); }

  update(dt) {
    if (!this._ready || !this.enabled) return;
    if (this._activeN === 0) {
      this.stats.active = this._sceneN;
      this.stats.dynamic = 0;
      return;
    }

    const bytes = this._stateBytes;

    for (let i = this._activeN - 1; i >= 0; i--) {
      const s = this._active[i];
      const age = this.sAge[s] + dt;
      this.sAge[s] = age;
      const life = this.sLife[s];

      if (age >= life) { this._freeSlot(s); continue; }

      // Fade IN over 50 ms so an impact reads instantly without popping, then
      // hold, then dissolve on a smooth curve over the tail of the lifetime.
      let op = this.sBoost[s];
      if (age < 0.05) op *= age * 20;
      const fade = this.sFade[s];
      const tail = life - age;
      if (tail < fade) { const k = tail / fade; op *= k * k * (3 - 2 * k); }

      bytes[s * 4 + 3] = (clamp01(op) * 255) | 0;
    }

    this._stateTex.needsUpdate = true;
    this.stats.dynamic = this._activeN;
    this.stats.active = this._activeN + this._sceneN;
  }

  resize() { /* screen independent */ }

  setEnabled(on) {
    this.enabled = !!on;
    if (this.mesh) this.mesh.visible = this.enabled;
  }

  dispose() {
    this.mesh?.parent?.remove(this.mesh);
    this.geometry?.dispose();
    this.material?.dispose();
    this._mapTex?.dispose();
    this._nrmTex?.dispose();
    this._ormTex?.dispose();
    this._stateTex?.dispose();
  }

  /* ================================================================ *
   *  BUS HANDLERS
   * ================================================================ */

  _onShot(e) {
    const d = e?.dir;
    if (d && typeof d.x === 'number') {
      this._shotDir.set(d.x, d.y, d.z);
      if (this._shotDir.lengthSq() > 1e-8) this._shotDir.normalize();
    }
  }

  _onImpact(e) {
    if (!this._ready || !this.enabled || !e || !e.point) return;
    const surf = this._bucket(e.surface !== undefined ? e.surface : e.material);
    if (surf === 'flesh') {
      // No decal on a moving body — throw the blood onto whatever is behind it.
      this.addBloodSpray(e.point, this._shotDir, 0.55 + this._rnd() * 0.5);
      return;
    }
    this.addImpact(e.point, e.normal, surf);
  }

  _onHit(e) {
    if (!this._ready || !this.enabled || !e?.point) return;
    this.addBloodSpray(e.point, this._shotDir, e.headshot ? 1.15 : 0.62);
  }

  _onKill(e) {
    if (!this._ready || !this.enabled) return;
    const en = e?.enemy;
    const p = en?.position || en?.pos || e?.point;
    if (!p || typeof p.x !== 'number') return;
    if (this._rayHit(p.x, p.y + 0.6, p.z, 0, -1, 0, 3.2)) {
      this.addDecal(this._rnd() < 0.5 ? T_BLOOD_A : T_BLOOD_B, this._hit, this._hit,
        1.15 + this._rnd() * 0.75, 0, {
        angle: this._rnd() * TAU, life: 240, fade: 60, tint: 'flesh', opacity: 0.9,
      });
    }
  }

  _onExplosion(e) {
    if (!this._ready || !this.enabled || !e?.point) return;
    const p = e.point;
    const R = Math.max(0.6, e.radius ?? 3.5);

    // Primary scorch on the ground under the blast.
    if (this._rayHit(p.x, p.y, p.z, 0, -1, 0, Math.max(2.5, R))) {
      this.addDecal(T_SCORCH_L, this._hit, this._hit, R * 1.45, 0, {
        angle: this._rnd() * TAU, life: 300, fade: 70, opacity: 0.92,
      });
    }
    // Secondary sooting sprayed onto whatever surrounds the blast.
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * TAU + this._rnd() * 0.8;
      const el = -0.25 + this._rnd() * 0.85;
      const cl = Math.cos(el);
      const reach = R * 1.35;
      if (!this._rayHit(p.x, p.y, p.z, Math.cos(a) * cl, Math.sin(el), Math.sin(a) * cl, reach)) continue;
      const f = 1 - this._hit.t / reach;
      this.addDecal(this._rnd() < 0.62 ? T_SCORCH_S : T_SCORCH_L, this._hit, this._hit,
        R * (0.45 + f * 0.75), 0, {
        angle: this._rnd() * TAU, life: 260, fade: 60, opacity: 0.35 + f * 0.5,
      });
    }
  }

  /* ================================================================ *
   *  PUBLIC SPAWN API
   * ================================================================ */

  /** Bullet hole. `surface` is any Materials key, impact bucket, or material. */
  addImpact(point, normal, surface, opts) {
    if (!this._ready || !this.enabled || !point) return -1;
    const o = opts || EMPTY;
    let b = this._bucket(surface);

    // Refine against the geometry actually under the hit: the Bus flattens
    // plaster / brick / asphalt into 'concrete', and those three chip in
    // completely different colours.
    let nx0 = normal ? (normal.nx !== undefined ? normal.nx : normal.x) : 0;
    let ny0 = normal ? (normal.ny !== undefined ? normal.ny : normal.y) : 1;
    let nz0 = normal ? (normal.nz !== undefined ? normal.nz : normal.z) : 0;
    const nl0 = Math.hypot(nx0, ny0, nz0) || 1;
    nx0 /= nl0; ny0 /= nl0; nz0 /= nl0;
    if (o.surfaceLocked !== true) {
      const geo = this._fineAt(point.x, point.y, point.z, nx0, ny0, nz0);
      if (geo && (b === 'concrete' || COARSE[geo] === COARSE[b])) b = geo;
    }

    const tiles = HOLE_TILES[b] || HOLE_TILES.concrete;
    const tile = o.tileIndex !== undefined ? o.tileIndex : tiles[(this._rnd() * tiles.length) | 0];
    const rng = HOLE_SIZE[b] || HOLE_SIZE.concrete;
    // biased low so most hits are modest and the occasional one is a chunk
    const size = o.size || lerp(rng[0], rng[1], Math.pow(this._rnd(), 1.45));

    const slot = this.addDecal(tile, point, normal, size, size, {
      angle: o.angle !== undefined ? o.angle : this._rnd() * TAU,
      mirror: o.mirror !== undefined ? o.mirror : ((this._rnd() * 4) | 0),
      life: o.life || 150,
      fade: o.fade || 35,
      tint: o.tint || b,
      opacity: o.opacity !== undefined ? o.opacity : (0.70 + this._rnd() * 0.30),
      minGap: o.minGap !== undefined ? o.minGap : size * 0.22,
      depth: o.depth,
      permanent: o.permanent,
    });

    // Dust and pulverised substrate run down the wall under the hit. Only on
    // near-vertical faces, and only sometimes, so it never reads as a stamp.
    if (slot >= 0 && o.noStreak !== true && Math.abs(ny0) < 0.55 && this._rnd() < 0.62) {
      const sw = size * (0.55 + this._rnd() * 0.55);
      const sh = size * (1.5 + this._rnd() * 1.6);
      // The tile's head is at its TOP edge, so the card hangs from the hole:
      // centre it half a height below, and tuck the head just under the crater.
      this._streakP.x = point.x;
      this._streakP.y = point.y - sh * 0.5 + size * 0.16;
      this._streakP.z = point.z;
      this.addDecal(T_DUSTFALL, this._streakP, normal, sw, sh, {
        upAlign: true, angle: 0, depth: o.depth,
        mirror: (this._rnd() * 2) | 0,
        life: o.life || 150, fade: o.fade || 35,
        tint: b, opacity: 0.14 + this._rnd() * 0.26,
        permanent: o.permanent,
      });
    }
    return slot;
  }

  /** Soot burst. */
  addScorch(point, normal, radius) {
    const r = Math.max(0.35, radius || 1.2);
    return this.addDecal(r > 1.6 ? T_SCORCH_L : T_SCORCH_S, point, normal, r * 1.5, 0, {
      angle: this._rnd() * TAU, life: 280, fade: 60,
      opacity: 0.75 + this._rnd() * 0.2,
    });
  }

  /** Blood pool / splatter directly at a point. */
  addBlood(point, normal, size) {
    return this.addDecal(this._rnd() < 0.5 ? T_BLOOD_A : T_BLOOD_B, point, normal, size || 0.9, 0, {
      upAlign: true, angle: 0, life: 200, fade: 50, tint: 'flesh',
      opacity: 0.82 + this._rnd() * 0.18,
    });
  }

  /**
   * Cast from a body hit along the bullet path and splatter the first surface
   * behind it. This is what leaves a story on the walls after a firefight.
   */
  addBloodSpray(point, dir, scale) {
    if (!this._ready || !this.enabled || !point) return -1;
    let dx = dir?.x ?? 0, dy = dir?.y ?? 0, dz = dir?.z ?? -1;
    const l = Math.hypot(dx, dy, dz);
    if (l < 1e-5) { dx = 0; dy = 0; dz = -1; } else { dx /= l; dy /= l; dz /= l; }

    const k = scale || 0.7;
    const n = k > 1 ? 3 : 2;
    let placed = -1;
    for (let i = 0; i < n; i++) {
      const sx = dx + (this._rnd() - 0.5) * 0.42;
      const sy = dy + (this._rnd() - 0.5) * 0.42;
      const sz = dz + (this._rnd() - 0.5) * 0.42;
      const sl = Math.hypot(sx, sy, sz) || 1;
      if (!this._rayHit(point.x, point.y, point.z, sx / sl, sy / sl, sz / sl, 4.2)) continue;
      const f = 1 - this._hit.t / 4.2;
      const r = this.addDecal(this._rnd() < 0.5 ? T_BLOOD_A : T_BLOOD_B, this._hit, this._hit,
        (0.55 + f * 0.85) * k, 0, {
        upAlign: true, angle: 0, life: 190, fade: 45, tint: 'flesh',
        opacity: 0.55 + f * 0.42, minGap: 0.10,
      });
      if (r >= 0) placed = r;
    }
    return placed;
  }

  /**
   * The generic entry point.
   *   tile    atlas tile index
   *   point   {x,y,z}
   *   normal  {x,y,z} (or {nx,ny,nz}) surface normal
   *   w, h    world size in metres (h = 0 -> square)
   *   opts    { angle, upAlign, life, fade, opacity, tint, depth, minGap, permanent }
   */
  addDecal(tile, point, normal, w, h, opts) {
    if (!this._ready || !this.enabled || !point) return -1;
    const o = opts || EMPTY;

    const width = Math.max(0.02, w || 0.25);
    const height = Math.max(0.02, h || width);
    const depth = o.depth || Math.max(0.12, Math.min(width, height) * 0.85);

    let nx = normal ? (normal.nx !== undefined ? normal.nx : normal.x) : 0;
    let ny = normal ? (normal.ny !== undefined ? normal.ny : normal.y) : 1;
    let nz = normal ? (normal.nz !== undefined ? normal.nz : normal.z) : 0;
    if (typeof nx !== 'number' || typeof ny !== 'number' || typeof nz !== 'number') {
      nx = 0; ny = 1; nz = 0;
    }
    const nl = Math.hypot(nx, ny, nz);
    if (nl < 1e-6) { nx = 0; ny = 1; nz = 0; } else { nx /= nl; ny /= nl; nz /= nl; }

    const px = point.x, py = point.y, pz = point.z;
    if (!isFinite(px) || !isFinite(py) || !isFinite(pz)) return -1;

    const gap = o.minGap || 0;
    if (gap > 0 && this._tooClose(px, py, pz, gap)) return -1;

    const slot = o.permanent ? this._allocScene() : this._allocDyn();
    if (slot < 0) return -1;

    const angle = o.angle !== undefined ? o.angle : 0;
    const mirror = (o.mirror | 0) & 3;
    const verts = this._project(slot, px, py, pz, nx, ny, nz,
      width, height, depth, angle, tile | 0, !!o.upAlign, mirror);

    this.sVerts[slot] = verts;
    if (verts < 3) { this._freeSlotSilent(slot); return -1; }

    // --- slot bookkeeping ---
    this.sUsed[slot] = 1;
    this.sPx[slot] = px; this.sPy[slot] = py; this.sPz[slot] = pz;
    this.sRad[slot] = Math.max(width, height) * 0.5;
    this.sAge[slot] = 0;
    this.sPerm[slot] = o.permanent ? 1 : 0;
    this.sLife[slot] = o.permanent ? Infinity : (o.life || 150);
    this.sFade[slot] = o.fade || 30;
    this.sBoost[slot] = clamp01(o.opacity !== undefined ? o.opacity : 1);

    // --- tint: pull the mark toward the colour of what it sits on ---
    let tr = 1, tg = 1, tb = 1;
    const tint = o.tint;
    if (typeof tint === 'string') {
      const c = SURF_TINT[tint]; if (c) { tr = c[0]; tg = c[1]; tb = c[2]; }
    } else if (Array.isArray(tint)) {
      tr = tint[0]; tg = tint[1]; tb = tint[2];
    } else if (typeof tint === 'number') {
      tr = ((tint >> 16) & 255) / 255; tg = ((tint >> 8) & 255) / 255; tb = (tint & 255) / 255;
    }
    const j = 0.93 + this._rnd() * 0.10;
    const b = this._stateBytes, o4 = slot * 4;
    b[o4] = (clamp01(tr * j) * 255) | 0;
    b[o4 + 1] = (clamp01(tg * j) * 255) | 0;
    b[o4 + 2] = (clamp01(tb * j) * 255) | 0;
    b[o4 + 3] = o.permanent ? (clamp01(this.sBoost[slot]) * 255) | 0 : 0;
    this._stateTex.needsUpdate = true;

    if (o.permanent) this._sceneN++;
    else this._pushActive(slot);

    const end = slot * SLOT_VERTS + verts;
    if (end > this._highWater) {
      this._highWater = end;
      this.geometry.setDrawRange(0, this._highWater);
      this.stats.verts = this._highWater;
      this.stats.tris = (this._highWater / 3) | 0;
    }
    return slot;
  }

  clear(alsoScene) {
    if (!this._ready) return;
    for (let i = this._activeN - 1; i >= 0; i--) this._freeSlot(this._active[i]);
    if (alsoScene) {
      for (let s = this.dynCap; s < CAP; s++) if (this.sUsed[s]) this._freeSlotSilent(s);
      this._sceneN = 0;
    }
    this._stateTex.needsUpdate = true;
  }

  /* ================================================================ *
   *  SLOT ALLOCATION
   * ================================================================ */

  _allocDyn() {
    const cap = this.dynCap;
    // Pre-expire what the ring is about to reach so the oldest decal
    // DISSOLVES a couple of seconds ahead instead of vanishing mid-frame.
    for (let k = 0; k < 3; k++) {
      const s = (this._ring + 26 + k) % cap;
      if (this.sUsed[s] && this.sLife[s] - this.sAge[s] > 2.2) {
        this.sLife[s] = this.sAge[s] + 2.2;
        this.sFade[s] = 2.2;
      }
    }
    const slot = this._ring;
    this._ring = (this._ring + 1) % cap;
    if (this.sUsed[slot]) this._freeSlot(slot);
    return slot;
  }

  _allocScene() {
    if (this._sceneN >= this.sceneCap) return -1;
    for (let s = this.dynCap; s < CAP; s++) if (!this.sUsed[s]) return s;
    return -1;
  }

  _pushActive(slot) {
    if (this._activeAt[slot] >= 0) return;
    this._active[this._activeN] = slot;
    this._activeAt[slot] = this._activeN;
    this._activeN++;
  }

  _popActive(slot) {
    const at = this._activeAt[slot];
    if (at < 0) return;
    const last = this._activeN - 1;
    const moved = this._active[last];
    this._active[at] = moved;
    this._activeAt[moved] = at;
    this._activeN = last;
    this._activeAt[slot] = -1;
  }

  _freeSlot(slot) {
    this._popActive(slot);
    this._freeSlotSilent(slot);
  }

  _freeSlotSilent(slot) {
    const n = this.sVerts[slot];
    if (n > 0 && this._pos) {
      const base = slot * SLOT_VERTS;
      this._pos.fill(0, base * 3, (base + n) * 3);
      this._nrm.fill(0, base * 3, (base + n) * 3);
      this._dcl.fill(0, base * 2, (base + n) * 2);
      this._aPos.addUpdateRange?.(base * 3, n * 3);
      this._aNrm.addUpdateRange?.(base * 3, n * 3);
      this._aDcl.addUpdateRange?.(base * 2, n * 2);
      this._aPos.needsUpdate = true;
      this._aNrm.needsUpdate = true;
      this._aDcl.needsUpdate = true;
    }
    this.sUsed[slot] = 0;
    this.sVerts[slot] = 0;
    this.sPerm[slot] = 0;
    this._stateBytes[slot * 4 + 3] = 0;
  }

  /** Reject a decal landing on top of the one we just placed (burst fire). */
  _tooClose(x, y, z, gap) {
    const g2 = gap * gap;
    const cap = this.dynCap;
    for (let k = 1; k <= 8; k++) {
      const s = (this._ring - k + cap * 2) % cap;
      if (!this.sUsed[s]) continue;
      const dx = this.sPx[s] - x, dy = this.sPy[s] - y, dz = this.sPz[s] - z;
      if (dx * dx + dy * dy + dz * dz < g2) return true;
    }
    return false;
  }

  /** Resolve anything a collaborator hands us to one of the FINE surfaces. */
  _bucket(x) {
    if (typeof x === 'string') {
      const k = x.toLowerCase();
      if (FINE_MAP[k]) return FINE_MAP[k];
      if (FINE.indexOf(k) >= 0) return k;
      const r = this.g?.materials?.surfaceFor?.(k);
      return FINE_MAP[r] || 'concrete';
    }
    if (x && typeof x === 'object') {
      const u = x.userData;
      const k = u && (u.bsSurface || u.bsImpact);
      if (k && FINE_MAP[k]) return FINE_MAP[k];
      const r = this.g?.materials?.surfaceFor?.(x);
      if (r) return FINE_MAP[r] || r;
    }
    return 'concrete';
  }

  /**
   * What is the level actually made of right here?  The Bus only carries the
   * six coarse buckets, so a hit on rendered plaster and a hit on bare brick
   * both arrive as 'concrete'.  One small broadphase query recovers the real
   * material from the triangle soup so the mark can respond to it.
   */
  _fineAt(px, py, pz, nx, ny, nz) {
    if (!this.triCount || !this._grid) return null;
    const n = this._query(px, pz, 0.42);
    if (!n) return null;
    const V = this.triV, N = this.triN, S = this.triS;
    let best = 1e9, bi = -1;
    for (let qi = 0; qi < n; qi++) {
      const t = this._qBuf[qi], t3 = t * 3;
      const fnx = N[t3], fny = N[t3 + 1], fnz = N[t3 + 2];
      if (fnx * nx + fny * ny + fnz * nz < 0.55) continue;
      const o = t * 9;
      // squared distance to the triangle centroid — good enough to identify the
      // host material and far cheaper than an exact closest-point test
      const cxx = (V[o] + V[o + 3] + V[o + 6]) / 3 - px;
      const cyy = (V[o + 1] + V[o + 4] + V[o + 7]) / 3 - py;
      const czz = (V[o + 2] + V[o + 5] + V[o + 8]) / 3 - pz;
      const d = cxx * cxx + cyy * cyy + czz * czz;
      if (d < best) { best = d; bi = t; }
    }
    if (bi < 0 || best > 0.36) return null;
    return FINE[S[bi]] || null;
  }

  /* ================================================================ *
   *  PROJECTION — clip level triangles against the projector box
   * ================================================================ */

  _project(slot, px, py, pz, nx, ny, nz, w, h, depth, angle, tile, upAlign, mirror) {
    // ---- decal basis: T = +u, B = +v, N = surface normal ----------------
    let ax, ay, az;
    if (upAlign) { ax = 0; ay = 1; az = 0; }
    else if (Math.abs(ny) > 0.90) { ax = 1; ay = 0; az = 0; }
    else { ax = 0; ay = 1; az = 0; }

    const d0 = ax * nx + ay * ny + az * nz;
    let vx = ax - nx * d0, vy = ay - ny * d0, vz = az - nz * d0;
    let vl = Math.hypot(vx, vy, vz);
    if (vl < 1e-5) {
      vx = 1 - nx * nx; vy = -nx * ny; vz = -nx * nz;
      vl = Math.hypot(vx, vy, vz) || 1;
    }
    vx /= vl; vy /= vl; vz /= vl;
    const ux = vy * nz - vz * ny, uy = vz * nx - vx * nz, uz = vx * ny - vy * nx;

    const ca = Math.cos(angle), sa = Math.sin(angle);
    const tx = ux * ca + vx * sa, ty = uy * ca + vy * sa, tz = uz * ca + vz * sa;
    const bx = -ux * sa + vx * ca, by = -uy * sa + vy * ca, bz = -uz * sa + vz * ca;

    const hw = w * 0.5, hh = h * 0.5, hd = depth * 0.5;
    const rad = Math.sqrt(hw * hw + hh * hh + hd * hd);

    // ---- atlas rect for this tile ---------------------------------------
    const ti = ((tile % TILE_COUNT) + TILE_COUNT) % TILE_COUNT;
    const col = ti % ATLAS_COLS, row = (ti / ATLAS_COLS) | 0;
    const tuw = 1 / ATLAS_COLS, tvh = 1 / ATLAS_ROWS;
    let u0 = (col + UV_INSET) * tuw, uS = (1 - UV_INSET * 2) * tuw;
    let v0 = (row + UV_INSET) * tvh, vS = (1 - UV_INSET * 2) * tvh;
    // Mirroring the tile quadruples the apparent shape library for free, which
    // is what stops repeated hits on one wall from reading as the same sticker.
    if (mirror & 1) { u0 += uS; uS = -uS; }
    if (mirror & 2) { v0 += vS; vS = -vS; }

    const base = slot * SLOT_VERTS;
    const pos = this._pos, nrm = this._nrm, uvs = this._uv, dcl = this._dcl;
    // Lift proportional to the mark so a big blast decal cannot punch back
    // through its host, while a small hole still hugs the surface.
    const bias = 0.0032 + Math.min(w, h) * 0.009;
    const invW = 1 / w, invH = 1 / h, invD = 1 / Math.max(1e-4, hd);
    let vi = 0;

    const count = this.triCount ? this._query(px, pz, rad) : 0;
    const tV = this.triV, tN = this.triN;
    const pA = this._pA, pB = this._pB;

    outer:
    for (let qi = 0; qi < count; qi++) {
      const t = this._qBuf[qi];
      const t3 = t * 3;
      const fnx = tN[t3], fny = tN[t3 + 1], fnz = tN[t3 + 2];
      const facing = fnx * nx + fny * ny + fnz * nz;
      if (facing < 0.16) continue;                       // back / edge-on face

      const t9 = t * 9;
      let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, minz = 1e9, maxz = -1e9;
      for (let k = 0; k < 3; k++) {
        const k3 = t9 + k * 3;
        const dx = tV[k3] - px, dy = tV[k3 + 1] - py, dz = tV[k3 + 2] - pz;
        const lx = dx * tx + dy * ty + dz * tz;
        const ly = dx * bx + dy * by + dz * bz;
        const lz = dx * nx + dy * ny + dz * nz;
        const o = k * 3;
        pA[o] = lx; pA[o + 1] = ly; pA[o + 2] = lz;
        if (lx < minx) minx = lx; if (lx > maxx) maxx = lx;
        if (ly < miny) miny = ly; if (ly > maxy) maxy = ly;
        if (lz < minz) minz = lz; if (lz > maxz) maxz = lz;
      }
      if (minx > hw || maxx < -hw || miny > hh || maxy < -hh || minz > hd || maxz < -hd) continue;

      // Sutherland–Hodgman against the 6 projector-box planes.
      let n = 3;
      n = this._clipPlane(pA, n, 0, 1, hw, pB); if (n < 3) continue;
      n = this._clipPlane(pB, n, 0, -1, hw, pA); if (n < 3) continue;
      n = this._clipPlane(pA, n, 1, 1, hh, pB); if (n < 3) continue;
      n = this._clipPlane(pB, n, 1, -1, hh, pA); if (n < 3) continue;
      n = this._clipPlane(pA, n, 2, 1, hd, pB); if (n < 3) continue;
      n = this._clipPlane(pB, n, 2, -1, hd, pA); if (n < 3) continue;

      const angFade = sstep(0.16, 0.52, facing);
      const obx = px + fnx * bias, oby = py + fny * bias, obz = pz + fnz * bias;

      for (let f = 1; f < n - 1; f++) {
        if (vi + 3 > SLOT_VERTS) break outer;
        for (let c = 0; c < 3; c++) {
          const src = (c === 0 ? 0 : (c === 1 ? f : f + 1)) * 3;
          const lx = pA[src], ly = pA[src + 1], lz = pA[src + 2];
          const o = base + vi;
          pos[o * 3] = obx + tx * lx + bx * ly + nx * lz;
          pos[o * 3 + 1] = oby + ty * lx + by * ly + ny * lz;
          pos[o * 3 + 2] = obz + tz * lx + bz * ly + nz * lz;
          nrm[o * 3] = fnx; nrm[o * 3 + 1] = fny; nrm[o * 3 + 2] = fnz;
          uvs[o * 2] = u0 + (lx * invW + 0.5) * uS;
          uvs[o * 2 + 1] = v0 + (ly * invH + 0.5) * vS;
          dcl[o * 2] = slot;
          dcl[o * 2 + 1] = angFade * (1 - sstep(0.62, 1.0, Math.abs(lz) * invD));
          vi++;
        }
      }
    }

    // Nothing to clip against (no soup yet, or a dynamic object) — fall back
    // to a flat card so a mark still appears.
    if (vi < 3) {
      vi = 0;
      const obx = px + nx * bias, oby = py + ny * bias, obz = pz + nz * bias;
      for (let c = 0; c < 6; c++) {
        const lx = FB_X[c] * hw, ly = FB_Y[c] * hh;
        const o = base + vi;
        pos[o * 3] = obx + tx * lx + bx * ly;
        pos[o * 3 + 1] = oby + ty * lx + by * ly;
        pos[o * 3 + 2] = obz + tz * lx + bz * ly;
        nrm[o * 3] = nx; nrm[o * 3 + 1] = ny; nrm[o * 3 + 2] = nz;
        uvs[o * 2] = u0 + (lx * invW + 0.5) * uS;
        uvs[o * 2 + 1] = v0 + (ly * invH + 0.5) * vS;
        dcl[o * 2] = slot;
        dcl[o * 2 + 1] = 1;
        vi++;
      }
    }

    // clear the unused tail of the slot so a smaller decal leaves no ghost
    const prev = this.sVerts[slot];
    if (prev > vi) {
      pos.fill(0, (base + vi) * 3, (base + prev) * 3);
      nrm.fill(0, (base + vi) * 3, (base + prev) * 3);
      dcl.fill(0, (base + vi) * 2, (base + prev) * 2);
    }
    const up = prev > vi ? prev : vi;
    this._aPos.addUpdateRange?.(base * 3, up * 3);
    this._aNrm.addUpdateRange?.(base * 3, up * 3);
    this._aUv.addUpdateRange?.(base * 2, up * 2);
    this._aDcl.addUpdateRange?.(base * 2, up * 2);
    this._aPos.needsUpdate = true;
    this._aNrm.needsUpdate = true;
    this._aUv.needsUpdate = true;
    this._aDcl.needsUpdate = true;

    return vi;
  }

  /** Clip a polygon of xyz triples against `sign * coord <= limit`. */
  _clipPlane(src, n, axis, sign, limit, dst) {
    let m = 0;
    let ai = (n - 1) * 3;
    let di = limit - sign * src[ai + axis];
    for (let i = 0; i < n; i++) {
      const bi = i * 3;
      const dj = limit - sign * src[bi + axis];
      const inI = di >= 0, inJ = dj >= 0;
      if (inI !== inJ) {
        const t = di / (di - dj);
        const m3 = m * 3;
        dst[m3] = src[ai] + (src[bi] - src[ai]) * t;
        dst[m3 + 1] = src[ai + 1] + (src[bi + 1] - src[ai + 1]) * t;
        dst[m3 + 2] = src[ai + 2] + (src[bi + 2] - src[ai + 2]) * t;
        m++;
      }
      if (inJ) {
        const m3 = m * 3;
        dst[m3] = src[bi]; dst[m3 + 1] = src[bi + 1]; dst[m3 + 2] = src[bi + 2];
        m++;
      }
      if (m >= 30) break;
      ai = bi; di = dj;
    }
    return m;
  }

  /* ================================================================ *
   *  LEVEL TRIANGLE SOUP + XZ BROADPHASE
   * ================================================================ */

  _buildSoup() {
    const scene = this.g?.scene;
    if (!scene) return;

    const V = new Float32Array(MAX_TRI * 9);
    const N = new Float32Array(MAX_TRI * 3);
    const S = new Uint8Array(MAX_TRI);
    const A = new Float32Array(MAX_TRI);
    let n = 0;

    const m4 = new THREE.Matrix4();
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const na = new THREE.Vector3(), nb = new THREE.Vector3(), nc = new THREE.Vector3();
    const nm3 = this._nm3;
    const mats = this.g?.materials;

    // Half a million triangles go through here at boot, so the inner loop is
    // hand-rolled against the raw typed arrays: Vector3.fromBufferAttribute /
    // applyMatrix4 per vertex costs several times as much for identical maths.
    const P = new Float64Array(9);
    const NR = new Float64Array(3);

    const pushGeo = (geo, matrix, surfId) => {
      const posA = geo.attributes?.position;
      if (!posA) return;
      const nrmA = geo.attributes.normal || null;
      const idx = geo.index;
      const triN = idx ? (idx.count / 3) | 0 : (posA.count / 3) | 0;
      nm3.setFromMatrix4(matrix);
      const nme = nm3.elements;
      const me = matrix.elements;
      const m0 = me[0], m1 = me[1], m2 = me[2], m4a = me[4], m5 = me[5], m6 = me[6];
      const m8 = me[8], m9 = me[9], m10 = me[10], m12 = me[12], m13 = me[13], m14 = me[14];

      // fast path: plain, non-interleaved, un-normalised float attributes
      const pa = posA.array;
      const fastP = !posA.isInterleavedBufferAttribute && posA.itemSize === 3 &&
        !posA.normalized && pa && pa.BYTES_PER_ELEMENT === 4 && pa.length >= posA.count * 3;
      const naA = nrmA ? nrmA.array : null;
      const fastN = nrmA && !nrmA.isInterleavedBufferAttribute && nrmA.itemSize === 3 &&
        !nrmA.normalized && naA && naA.BYTES_PER_ELEMENT === 4;
      const ia = idx ? idx.array : null;
      const fastI = !idx || (!idx.isInterleavedBufferAttribute && ia);

      for (let t = 0; t < triN; t++) {
        if (n >= MAX_TRI) return;
        const b3 = t * 3;
        const i0 = idx ? (fastI ? ia[b3] : idx.getX(b3)) : b3;
        const i1 = idx ? (fastI ? ia[b3 + 1] : idx.getX(b3 + 1)) : b3 + 1;
        const i2 = idx ? (fastI ? ia[b3 + 2] : idx.getX(b3 + 2)) : b3 + 2;

        if (fastP) {
          const a0 = i0 * 3, a1 = i1 * 3, a2 = i2 * 3;
          for (let k = 0; k < 3; k++) {
            const src = k === 0 ? a0 : (k === 1 ? a1 : a2);
            const x = pa[src], y = pa[src + 1], z = pa[src + 2];
            const o = k * 3;
            P[o] = m0 * x + m4a * y + m8 * z + m12;
            P[o + 1] = m1 * x + m5 * y + m9 * z + m13;
            P[o + 2] = m2 * x + m6 * y + m10 * z + m14;
          }
        } else {
          v0.fromBufferAttribute(posA, i0).applyMatrix4(matrix);
          v1.fromBufferAttribute(posA, i1).applyMatrix4(matrix);
          v2.fromBufferAttribute(posA, i2).applyMatrix4(matrix);
          P[0] = v0.x; P[1] = v0.y; P[2] = v0.z;
          P[3] = v1.x; P[4] = v1.y; P[5] = v1.z;
          P[6] = v2.x; P[7] = v2.y; P[8] = v2.z;
        }

        const e1x = P[3] - P[0], e1y = P[4] - P[1], e1z = P[5] - P[2];
        const e2x = P[6] - P[0], e2y = P[7] - P[1], e2z = P[8] - P[2];
        let fx = e1y * e2z - e1z * e2y;
        let fy = e1z * e2x - e1x * e2z;
        let fz = e1x * e2y - e1y * e2x;
        const twice = Math.sqrt(fx * fx + fy * fy + fz * fz);
        if (twice < MIN_TRI_AREA * 2) continue;    // sub-centimetre greeble
        const inv = 1 / twice;
        fx *= inv; fy *= inv; fz *= inv;

        // Keep the winding and the shading normal in agreement so FrontSide
        // rendering and the backface rejection both stay correct.
        let flip = false;
        if (nrmA) {
          let sx, sy, sz;
          if (fastN) {
            const b0 = i0 * 3, b1 = i1 * 3, b2 = i2 * 3;
            sx = naA[b0] + naA[b1] + naA[b2];
            sy = naA[b0 + 1] + naA[b1 + 1] + naA[b2 + 1];
            sz = naA[b0 + 2] + naA[b1 + 2] + naA[b2 + 2];
          } else {
            na.fromBufferAttribute(nrmA, i0);
            nb.fromBufferAttribute(nrmA, i1);
            nc.fromBufferAttribute(nrmA, i2);
            sx = na.x + nb.x + nc.x; sy = na.y + nb.y + nc.y; sz = na.z + nb.z + nc.z;
          }
          NR[0] = nme[0] * sx + nme[3] * sy + nme[6] * sz;
          NR[1] = nme[1] * sx + nme[4] * sy + nme[7] * sz;
          NR[2] = nme[2] * sx + nme[5] * sy + nme[8] * sz;
          const l2 = NR[0] * NR[0] + NR[1] * NR[1] + NR[2] * NR[2];
          if (l2 > 1e-10 && (NR[0] * fx + NR[1] * fy + NR[2] * fz) < 0) flip = true;
        }

        const o9 = n * 9;
        V[o9] = P[0]; V[o9 + 1] = P[1]; V[o9 + 2] = P[2];
        if (flip) {
          V[o9 + 3] = P[6]; V[o9 + 4] = P[7]; V[o9 + 5] = P[8];
          V[o9 + 6] = P[3]; V[o9 + 7] = P[4]; V[o9 + 8] = P[5];
          fx = -fx; fy = -fy; fz = -fz;
        } else {
          V[o9 + 3] = P[3]; V[o9 + 4] = P[4]; V[o9 + 5] = P[5];
          V[o9 + 6] = P[6]; V[o9 + 7] = P[7]; V[o9 + 8] = P[8];
        }
        const o3 = n * 3;
        N[o3] = fx; N[o3 + 1] = fy; N[o3 + 2] = fz;
        S[n] = surfId;
        A[n] = twice * 0.5;
        n++;
      }
    };

    scene.updateMatrixWorld(true);
    scene.traverse((obj) => {
      if (n >= MAX_TRI) return;
      if (obj === this.mesh) return;
      if (!obj.visible) return;
      if (obj.userData && obj.userData.bsNoDecal) return;
      if (typeof obj.name === 'string' && obj.name.indexOf('sky_') === 0) return;
      if (!obj.isMesh || obj.isSkinnedMesh) return;

      const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
      if (!mat) return;
      if (mat.isShaderMaterial || mat.isRawShaderMaterial) return;      // atmosphere / fx layers
      if (mat.transparent && !(mat.userData && mat.userData.bsSurface)) return;
      const geo = obj.geometry;
      if (!geo || !geo.attributes || !geo.attributes.position) return;

      // Store the FINE material, not the coarse impact bucket — plaster, brick
      // and asphalt all report as 'concrete' and must not look alike.
      let surfId = 0;
      const key = (mat.userData && (mat.userData.bsSurface || mat.userData.bsImpact))
        || (obj.userData && obj.userData.bsSurface);
      if (key) {
        let f = FINE_MAP[key];
        if (!f) {
          const b = mats?.surfaceFor ? mats.surfaceFor(key) : key;
          f = FINE_MAP[b] || (FINE.indexOf(b) >= 0 ? b : null);
        }
        const bi = f ? FINE.indexOf(f) : -1;
        if (bi >= 0) surfId = bi;
      }

      if (obj.isInstancedMesh) {
        const cnt = Math.min(obj.count, 900);
        for (let i = 0; i < cnt && n < MAX_TRI; i++) {
          obj.getMatrixAt(i, m4);
          m4.premultiply(obj.matrixWorld);
          pushGeo(geo, m4, surfId);
        }
      } else {
        pushGeo(geo, obj.matrixWorld, surfId);
      }
    });

    this.triCount = n;
    if (!n) return;
    this.triV = V.slice(0, n * 9);
    this.triN = N.slice(0, n * 3);
    this.triS = S.slice(0, n);
    this.triA = A.slice(0, n);
    this._qStamp = new Int32Array(n);
  }

  _buildGrid() {
    const n = this.triCount;
    if (!n) return;
    const V = this.triV;

    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const o = i * 9;
      for (let k = 0; k < 3; k++) {
        const x = V[o + k * 3], z = V[o + k * 3 + 2];
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
    }
    const gw = Math.max(1, Math.min(768, Math.ceil((maxX - minX) / CELL) + 1));
    const gh = Math.max(1, Math.min(768, Math.ceil((maxZ - minZ) / CELL) + 1));
    const cells = gw * gh;
    const counts = new Int32Array(cells + 1);
    const over = [];

    const cx0 = (v) => { const c = ((v - minX) / CELL) | 0; return c < 0 ? 0 : (c >= gw ? gw - 1 : c); };
    const cz0 = (v) => { const c = ((v - minZ) / CELL) | 0; return c < 0 ? 0 : (c >= gh ? gh - 1 : c); };

    for (let pass = 0; pass < 2; pass++) {
      let items = null, cur = null;
      if (pass === 1) {
        for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];
        items = new Int32Array(counts[cells]);
        cur = counts.slice(0, cells);
        this._gridItems = items;
        this._gridCur = cur;
      }
      for (let i = 0; i < n; i++) {
        const o = i * 9;
        let x0 = V[o], x1 = x0, z0 = V[o + 2], z1 = z0;
        for (let k = 1; k < 3; k++) {
          const x = V[o + k * 3], z = V[o + k * 3 + 2];
          if (x < x0) x0 = x; if (x > x1) x1 = x;
          if (z < z0) z0 = z; if (z > z1) z1 = z;
        }
        const i0 = cx0(x0), i1 = cx0(x1), j0 = cz0(z0), j1 = cz0(z1);
        if ((i1 - i0 + 1) * (j1 - j0 + 1) > OVERSIZE_CELLS) {
          if (pass === 0) over.push(i);
          continue;
        }
        if (pass === 0) {
          for (let j = j0; j <= j1; j++) {
            const rb = j * gw;
            for (let ii = i0; ii <= i1; ii++) counts[rb + ii + 1]++;
          }
        } else {
          for (let j = j0; j <= j1; j++) {
            const rb = j * gw;
            for (let ii = i0; ii <= i1; ii++) items[cur[rb + ii]++] = i;
          }
        }
      }
    }

    this._grid = {
      minX, minZ, gw, gh,
      starts: counts,
      items: this._gridItems,
      over: Int32Array.from(over),
    };
    this._gridCur = null;
  }

  /** Fill _qBuf with unique triangle indices whose cell overlaps the query disc. */
  _query(x, z, r) {
    const G = this._grid;
    if (!G) return 0;
    const buf = this._qBuf, cap = buf.length;
    const st = this._qStamp;
    const stamp = ++this._stamp;
    let m = 0;

    let i0 = (((x - r) - G.minX) / CELL) | 0, i1 = (((x + r) - G.minX) / CELL) | 0;
    let j0 = (((z - r) - G.minZ) / CELL) | 0, j1 = (((z + r) - G.minZ) / CELL) | 0;
    if (i0 < 0) i0 = 0; if (j0 < 0) j0 = 0;
    if (i1 >= G.gw) i1 = G.gw - 1; if (j1 >= G.gh) j1 = G.gh - 1;

    for (let j = j0; j <= j1; j++) {
      const rb = j * G.gw;
      for (let i = i0; i <= i1; i++) {
        const c = rb + i;
        const s = G.starts[c], e = G.starts[c + 1];
        for (let k = s; k < e; k++) {
          const t = G.items[k];
          if (st[t] === stamp) continue;
          st[t] = stamp;
          if (m < cap) buf[m++] = t;
        }
      }
    }
    const ov = G.over;
    for (let k = 0; k < ov.length; k++) {
      const t = ov[k];
      if (st[t] === stamp) continue;
      st[t] = stamp;
      if (m < cap) buf[m++] = t;
    }
    return m;
  }

  /**
   * Nearest triangle along a ray; the result lands in this._hit, which doubles
   * as both a point ({x,y,z}) and a normal ({nx,ny,nz}) argument. No allocation.
   */
  _rayHit(ox, oy, oz, dx, dy, dz, maxD) {
    const G = this._grid;
    if (!this.triCount || !G) return false;
    const V = this.triV, N = this.triN;
    const buf = this._qBuf, cap = buf.length;
    const st = this._qStamp;
    const stamp = ++this._stamp;
    let m = 0;

    const steps = Math.min(12, Math.max(2, Math.ceil(maxD / CELL) + 1));
    for (let s = 0; s <= steps; s++) {
      const t = (s / steps) * maxD;
      const x = ox + dx * t, z = oz + dz * t;
      let i0 = ((x - CELL - G.minX) / CELL) | 0, i1 = ((x + CELL - G.minX) / CELL) | 0;
      let j0 = ((z - CELL - G.minZ) / CELL) | 0, j1 = ((z + CELL - G.minZ) / CELL) | 0;
      if (i0 < 0) i0 = 0; if (j0 < 0) j0 = 0;
      if (i1 >= G.gw) i1 = G.gw - 1; if (j1 >= G.gh) j1 = G.gh - 1;
      for (let j = j0; j <= j1; j++) {
        const rb = j * G.gw;
        for (let i = i0; i <= i1; i++) {
          const c = rb + i;
          const a = G.starts[c], b = G.starts[c + 1];
          for (let k = a; k < b; k++) {
            const t2 = G.items[k];
            if (st[t2] === stamp) continue;
            st[t2] = stamp;
            if (m < cap) buf[m++] = t2;
          }
        }
      }
    }
    const ov = G.over;
    for (let k = 0; k < ov.length; k++) {
      const t2 = ov[k];
      if (st[t2] === stamp) continue;
      st[t2] = stamp;
      if (m < cap) buf[m++] = t2;
    }

    // Möller–Trumbore
    let best = maxD, hit = -1;
    for (let qi = 0; qi < m; qi++) {
      const t2 = buf[qi];
      const o = t2 * 9;
      const axx = V[o], ayy = V[o + 1], azz = V[o + 2];
      const e1x = V[o + 3] - axx, e1y = V[o + 4] - ayy, e1z = V[o + 5] - azz;
      const e2x = V[o + 6] - axx, e2y = V[o + 7] - ayy, e2z = V[o + 8] - azz;
      const pvx = dy * e2z - dz * e2y, pvy = dz * e2x - dx * e2z, pvz = dx * e2y - dy * e2x;
      const det = e1x * pvx + e1y * pvy + e1z * pvz;
      if (det > -1e-9 && det < 1e-9) continue;
      const inv = 1 / det;
      const tvx = ox - axx, tvy = oy - ayy, tvz = oz - azz;
      const u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv;
      if (u < -1e-5 || u > 1 + 1e-5) continue;
      const qx = tvy * e1z - tvz * e1y, qy = tvz * e1x - tvx * e1z, qz = tvx * e1y - tvy * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * inv;
      if (v < -1e-5 || u + v > 1 + 1e-5) continue;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (tt > 0.001 && tt < best) { best = tt; hit = t2; }
    }

    if (hit < 0) return false;
    const h = this._hit;
    h.t = best; h.tri = hit;
    const o3 = hit * 3;
    let hnx = N[o3], hny = N[o3 + 1], hnz = N[o3 + 2];
    if (hnx * dx + hny * dy + hnz * dz > 0) { hnx = -hnx; hny = -hny; hnz = -hnz; }
    h.nx = hnx; h.ny = hny; h.nz = hnz;
    h.x = ox + dx * best + hnx * 0.001;
    h.y = oy + dy * best + hny * 0.001;
    h.z = oz + dz * best + hnz * 0.001;
    return true;
  }

  /* ================================================================ *
   *  WORLD DRESSING — pre-placed decals so the level is lived-in
   * ================================================================ */

  _classifyTris() {
    const n = this.triCount;
    if (!n) return;
    const N = this.triN, A = this.triA;
    const floor = new Int32Array(n), wall = new Int32Array(n);
    let fi = 0, wi = 0;
    for (let i = 0; i < n; i++) {
      const ny = N[i * 3 + 1], a = A[i];
      if (a < 0.12) continue;
      if (ny > 0.74) floor[fi++] = i;
      else if (ny > -0.36 && ny < 0.36) wall[wi++] = i;
    }
    this._floorTri = floor.subarray(0, fi); this._floorN = fi;
    this._wallTri = wall.subarray(0, wi); this._wallN = wi;
    this._floorCdf = this._buildCdf(this._floorTri, fi);
    this._wallCdf = this._buildCdf(this._wallTri, wi);
  }

  /**
   * Cumulative clamped-area table for O(log n) area-weighted sampling.
   * The old rejection sampler divided by the LARGEST triangle in the set — one
   * 100 m2 ground slab made every 1 m2 wall face a 1% shot, 96 tries in a row
   * missed, and the world-dressing pass bailed out after four decals. Clamping
   * each triangle's weight also stops the courtyard floor from swallowing the
   * whole budget.
   */
  _buildCdf(list, count) {
    const cdf = new Float32Array(count);
    const A = this.triA;
    let acc = 0;
    for (let i = 0; i < count; i++) {
      const a = A[list[i]];
      acc += a > 6 ? 6 : a;
      cdf[i] = acc;
    }
    return cdf;
  }

  /** Area-weighted sample of a point on a classified triangle. Never starves. */
  _sampleSurface(list, cdf, count, rnd, minY, maxY, out) {
    if (!count || !cdf) return false;
    const V = this.triV, N = this.triN;
    const total = cdf[count - 1];
    if (!(total > 0)) return false;
    for (let tries = 0; tries < 240; tries++) {
      const target = rnd() * total;
      let lo = 0, hi = count - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < target) lo = mid + 1; else hi = mid; }
      const t = list[lo];
      let a = rnd(), b = rnd();
      if (a + b > 1) { a = 1 - a; b = 1 - b; }
      const o = t * 9;
      const y = V[o + 1] + (V[o + 4] - V[o + 1]) * a + (V[o + 7] - V[o + 1]) * b;
      if (y < minY || y > maxY) continue;
      out.x = V[o] + (V[o + 3] - V[o]) * a + (V[o + 6] - V[o]) * b;
      out.y = y;
      out.z = V[o + 2] + (V[o + 5] - V[o + 2]) * a + (V[o + 8] - V[o + 2]) * b;
      const o3 = t * 3;
      out.nx = N[o3]; out.ny = N[o3 + 1]; out.nz = N[o3 + 2];
      out.surf = FINE[this.triS[t]] || 'concrete';
      out.tri = t;
      return true;
    }
    return false;
  }

  _placeSceneDecals() {
    if (!this.triCount) return;
    const rnd = mulberry32(0x0BADC0DE);
    const p = this._sample;
    const W = this._wallTri, WC = this._wallCdf, WN = this._wallN;
    const F = this._floorTri, FC = this._floorCdf, FN = this._floorN;

    // --- run-off grime bleeding down from ledges and sills ---------------
    for (let i = 0; i < 8; i++) {
      if (!this._sampleSurface(W, WC, WN, rnd, 1.6, 6.2, p)) continue;
      const w = 0.7 + rnd() * 0.9, h = 1.4 + rnd() * 1.3;
      p.y -= h * 0.34;
      this.addDecal(T_GRIME, p, p, w, h, {
        permanent: true, upAlign: true, angle: 0, depth: 0.50,
        mirror: (rnd() * 2) | 0,
        opacity: 0.22 + rnd() * 0.30, tint: p.surf,
      });
    }

    // --- older, wider wall staining --------------------------------------
    for (let i = 0; i < 3; i++) {
      if (!this._sampleSurface(W, WC, WN, rnd, 0.4, 2.6, p)) continue;
      this.addDecal(T_SCORCH_L, p, p, 1.0 + rnd() * 1.0, 0, {
        permanent: true, angle: rnd() * TAU, depth: 0.55,
        opacity: 0.10 + rnd() * 0.14, tint: p.surf,
      });
    }

    // --- oil / fluid pools on the ground ---------------------------------
    for (let i = 0; i < 5; i++) {
      if (!this._sampleSurface(F, FC, FN, rnd, -50, 3.0, p)) continue;
      this.addDecal(T_OIL, p, p, 0.7 + rnd() * 1.0, 0, {
        permanent: true, angle: rnd() * TAU, depth: 0.32,
        opacity: 0.55 + rnd() * 0.40,
      });
    }

    // --- burn / drag / scuff marks ---------------------------------------
    for (let i = 0; i < 6; i++) {
      if (!this._sampleSurface(F, FC, FN, rnd, -50, 3.0, p)) continue;
      this.addDecal(rnd() < 0.55 ? T_SCORCH_S : T_SCORCH_L, p, p, 0.6 + rnd() * 1.1, 0, {
        permanent: true, angle: rnd() * TAU, depth: 0.30,
        opacity: 0.16 + rnd() * 0.32, tint: p.surf,
      });
    }

    // --- weathered notices pasted on walls -------------------------------
    for (let i = 0; i < 6; i++) {
      if (!this._sampleSurface(W, WC, WN, rnd, 1.0, 2.5, p)) continue;
      const w = 0.46 + rnd() * 0.22;
      this.addDecal(rnd() < 0.5 ? T_POSTER_A : T_POSTER_B, p, p, w, w * (1.28 + rnd() * 0.24), {
        permanent: true, upAlign: true, angle: 0, depth: 0.20,
        opacity: 0.80 + rnd() * 0.18, minGap: 0.9,
      });
    }

    // --- old battle damage, so the compound has a history ----------------
    // Rounds arrive in bursts, so the marks arrive in loose CLUSTERS walking
    // across a wall rather than as an even scatter of identical dots.
    const cp = this._clusterP;
    let budget = this.sceneCap - this._sceneN - 8;
    for (let g = 0; g < 26 && budget > 0; g++) {
      const onWall = rnd() < 0.74;
      const ok = onWall
        ? this._sampleSurface(W, WC, WN, rnd, 0.45, 4.2, p)
        : this._sampleSurface(F, FC, FN, rnd, -50, 3.0, p);
      if (!ok) continue;

      // build a tangent frame on the hit face so the burst walks ALONG it
      const nx = p.nx, ny = p.ny, nz = p.nz;
      let ax, ay, az;
      if (Math.abs(ny) > 0.9) { ax = 1; ay = 0; az = 0; } else { ax = 0; ay = 1; az = 0; }
      const d0 = ax * nx + ay * ny + az * nz;
      let vx = ax - nx * d0, vy = ay - ny * d0, vz = az - nz * d0;
      const vl = Math.hypot(vx, vy, vz) || 1;
      vx /= vl; vy /= vl; vz /= vl;
      const ux = vy * nz - vz * ny, uy = vz * nx - vx * nz, uz = vx * ny - vy * nx;

      const n = 1 + ((rnd() * rnd() * 4) | 0);        // 1..4, mostly small
      const walk = rnd() * TAU;                        // burst direction on the face
      const cw = Math.cos(walk), sw = Math.sin(walk);
      const spread = 0.16 + rnd() * 0.55;
      for (let k = 0; k < n && budget > 0; k++) {
        const along = (k - (n - 1) * 0.5) * spread * (0.7 + rnd() * 0.7);
        const off = (rnd() - 0.5) * spread * 0.8;
        const su = cw * along - sw * off, sv = sw * along + cw * off;
        cp.x = p.x + ux * su + vx * sv;
        cp.y = p.y + uy * su + vy * sv;
        cp.z = p.z + uz * su + vz * sv;
        const before = this._sceneN;
        this.addImpact(cp, p, p.surf, {
          permanent: true, surfaceLocked: true, minGap: 0.045,
          opacity: 0.34 + rnd() * 0.46,
          mirror: (rnd() * 4) | 0,
          size: lerp(HOLE_SIZE[p.surf] ? HOLE_SIZE[p.surf][0] : 0.19,
            HOLE_SIZE[p.surf] ? HOLE_SIZE[p.surf][1] : 0.40, Math.pow(rnd(), 1.3)) * (0.85 + rnd() * 0.6),
        });
        budget -= Math.max(1, this._sceneN - before);
      }
    }

    this.stats.scene = this._sceneN;
    this.stats.active = this._sceneN;
  }

  /* ================================================================ *
   *  GEOMETRY + MATERIAL
   * ================================================================ */

  _buildMesh() {
    const total = CAP * SLOT_VERTS;
    this._pos = new Float32Array(total * 3);
    this._nrm = new Float32Array(total * 3);
    this._uv = new Float32Array(total * 2);
    this._dcl = new Float32Array(total * 2);

    const geo = new THREE.BufferGeometry();
    this._aPos = new THREE.BufferAttribute(this._pos, 3).setUsage(THREE.DynamicDrawUsage);
    this._aNrm = new THREE.BufferAttribute(this._nrm, 3).setUsage(THREE.DynamicDrawUsage);
    this._aUv = new THREE.BufferAttribute(this._uv, 2).setUsage(THREE.DynamicDrawUsage);
    this._aDcl = new THREE.BufferAttribute(this._dcl, 2).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this._aPos);
    geo.setAttribute('normal', this._aNrm);
    geo.setAttribute('uv', this._aUv);
    geo.setAttribute('aDecal', this._aDcl);
    geo.setDrawRange(0, 0);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e5);
    geo.boundingBox = new THREE.Box3(
      new THREE.Vector3(-1e5, -1e5, -1e5), new THREE.Vector3(1e5, 1e5, 1e5));
    this.geometry = geo;

    // per-decal state row: rgb = tint, a = opacity
    const st = new THREE.DataTexture(this._stateBytes, CAP, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    st.minFilter = THREE.NearestFilter;
    st.magFilter = THREE.NearestFilter;
    st.wrapS = st.wrapT = THREE.ClampToEdgeWrapping;
    st.generateMipmaps = false;
    st.colorSpace = THREE.NoColorSpace;
    st.needsUpdate = true;
    this._stateTex = st;

    const mat = new THREE.MeshStandardMaterial({
      name: 'bs:decals',
      map: this._mapTex || null,
      normalMap: this._nrmTex || null,
      roughnessMap: this._ormTex || null,
      metalnessMap: this._ormTex || null,
      aoMap: this._ormTex || null,
      color: 0xffffff,
      roughness: 1.0,
      metalness: 1.0,
      aoMapIntensity: 0.95,
      transparent: true,
      alphaTest: 0.004,
      depthWrite: false,
      depthTest: true,
      side: THREE.FrontSide,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -8,
      dithering: true,
      envMapIntensity: 1.0,
    });
    // The crater rim has to catch the raking key hard enough to read as
    // geometry from across the courtyard, not as printed shading.
    if (mat.normalScale) mat.normalScale.set(1.35, 1.35);

    const uState = { value: st };
    const uCount = { value: CAP };
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uDecalState = uState;
      shader.uniforms.uDecalCount = uCount;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\n' + /* glsl */`
attribute vec2 aDecal;
uniform sampler2D uDecalState;
uniform float uDecalCount;
varying vec4 vDecalState;
`)
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + /* glsl */`
{
  vec4 bsState = texture2D( uDecalState, vec2( ( aDecal.x + 0.5 ) / uDecalCount, 0.5 ) );
  vDecalState = vec4( bsState.rgb, bsState.a * aDecal.y );
}
`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\n' + /* glsl */`
varying vec4 vDecalState;
`)
        .replace('#include <map_fragment>', '#include <map_fragment>\n' + /* glsl */`
diffuseColor.rgb *= vDecalState.rgb;
diffuseColor.a *= vDecalState.a;
`);
    };
    mat.customProgramCacheKey = () => 'bs-decal-1';
    this.material = mat;

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'decals';
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.updateMatrix();
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    mesh.renderOrder = 3;
    mesh.userData.bsNoDecal = true;
    this.mesh = mesh;
    this.g?.scene?.add(mesh);
  }

  /* ================================================================ *
   *  PROCEDURAL DECAL ATLASES
   *  Three canvases per tile — albedo(+alpha), height, ORM — painted from
   *  the SAME shape sequence (identical PRNG seed) so they register exactly.
   * ================================================================ */

  /**
   * Channel-aware fill style.
   *   ch 0: albedo    o.r/o.g/o.b/o.a
   *   ch 1: height    o.h (0.5 = flat), alpha o.ha (defaults to o.a when h is given)
   *   ch 2: ORM       o.ao / o.rg / o.mt, alpha o.ra (defaults to o.a)
   */
  _c(ch, o) {
    if (ch === 0) {
      return 'rgba(' + (o.r | 0) + ',' + (o.g | 0) + ',' + (o.b | 0) + ',' + (o.a !== undefined ? o.a : 1) + ')';
    }
    if (ch === 1) {
      const v = Math.round(clamp01(o.h !== undefined ? o.h : 0.5) * 255);
      const a = o.ha !== undefined ? o.ha : (o.h !== undefined ? (o.a !== undefined ? o.a : 1) : 0);
      return 'rgba(' + v + ',' + v + ',' + v + ',' + a + ')';
    }
    const ao = Math.round(clamp01(o.ao !== undefined ? o.ao : 1) * 255);
    const rg = Math.round(clamp01(o.rg !== undefined ? o.rg : 0.9) * 255);
    const mt = Math.round(clamp01(o.mt !== undefined ? o.mt : 0) * 255);
    const a = o.ra !== undefined ? o.ra : (o.a !== undefined ? o.a : 1);
    return 'rgba(' + ao + ',' + rg + ',' + mt + ',' + a + ')';
  }

  _canvas(px) {
    try {
      if (typeof document !== 'undefined' && document.createElement) {
        const c = document.createElement('canvas');
        c.width = px; c.height = px;
        return c;
      }
      if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(px, px);
    } catch (e) { /* fall through */ }
    return null;
  }

  _buildAtlases() {
    const TP = this.tilePx, AP = this.atlasPx;
    const alb = new Uint8Array(AP * AP * 4);
    const orm = new Uint8Array(AP * AP * 4);
    const hgt = new Float32Array(AP * AP);
    hgt.fill(0.5);
    for (let i = 0; i < AP * AP; i++) {
      const o = i * 4;
      orm[o] = 255; orm[o + 1] = 230; orm[o + 2] = 0; orm[o + 3] = 255;
    }

    const cv = this._canvas(TP);
    let ctx = null;
    try { ctx = cv ? cv.getContext('2d', { willReadFrequently: true }) : null; } catch (e) { ctx = null; }

    if (ctx) {
      const grain = this._grainField(128);
      const scale = TP / 256;
      for (let t = 0; t < TILE_COUNT; t++) {
        const def = TILE_DEFS[t];
        const col = t % ATLAS_COLS, row = (t / ATLAS_COLS) | 0;
        for (let ch = 0; ch < 3; ch++) {
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.globalCompositeOperation = 'source-over';
          ctx.globalAlpha = 1;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.clearRect(0, 0, TP, TP);
          if (ch === 1) { ctx.fillStyle = 'rgb(128,128,128)'; ctx.fillRect(0, 0, TP, TP); }
          else if (ch === 2) { ctx.fillStyle = 'rgb(255,230,0)'; ctx.fillRect(0, 0, TP, TP); }

          const rnd = mulberry32(def.seed);
          ctx.save();
          ctx.scale(scale, scale);           // painters author in a 256 x 256 space
          try { this[def.fn](ctx, ch, rnd, def.v); }
          catch (e) { console.warn('[Decals] tile paint failed', def.fn, e); }
          ctx.restore();
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.globalCompositeOperation = 'source-over';

          const img = ctx.getImageData(0, 0, TP, TP).data;
          this._blit(img, ch, alb, orm, hgt, AP, TP, col * TP, row * TP, def.grain, grain, def.seed);
        }
      }
      this._bleedAlpha(alb, AP, 5);
    }

    // Painted shapes have knife edges; a 2 px blur turns them into bevels so the
    // normal map reads as a moulded crater rather than a stamped emboss.
    this._blurField(hgt, AP, 2);
    const nrm = this._sobelAtlas(hgt, AP, TP);
    this._mapTex = this._tex(alb, AP, true);
    this._nrmTex = this._tex(nrm, AP, false);
    this._ormTex = this._tex(orm, AP, false);
  }

  _tex(data, N, srgb) {
    const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = Math.max(1, Math.min(4, this.g?.config?.render?.maxAnisotropy ?? 4));
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
  }

  /** Tileable value-noise field used to break up every painted mark. */
  _grainField(N) {
    const f = new Float32Array(N * N);
    const periods = [4, 8, 16, 32, 64];
    const amps = [1, 0.6, 0.36, 0.2, 0.12];
    const lat = [];
    for (let o = 0; o < periods.length; o++) {
      const p = periods[o];
      const g = new Float32Array(p * p);
      const r = mulberry32(7717 + o * 977);
      for (let i = 0; i < g.length; i++) g[i] = r();
      lat.push(g);
    }
    let mn = 1e9, mx = -1e9;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let s = 0;
        for (let o = 0; o < periods.length; o++) {
          const p = periods[o], g = lat[o];
          const fx = (x / N) * p, fy = (y / N) * p;
          const ix = Math.floor(fx), iy = Math.floor(fy);
          const tx = fx - ix, ty = fy - iy;
          const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
          const x0 = ix % p, x1 = (ix + 1) % p, y0 = iy % p, y1 = (iy + 1) % p;
          const a = g[y0 * p + x0] + (g[y0 * p + x1] - g[y0 * p + x0]) * sx;
          const b = g[y1 * p + x0] + (g[y1 * p + x1] - g[y1 * p + x0]) * sx;
          s += (a + (b - a) * sy) * amps[o];
        }
        f[y * N + x] = s;
        if (s < mn) mn = s; if (s > mx) mx = s;
      }
    }
    const d = 1 / Math.max(1e-6, mx - mn);
    for (let i = 0; i < f.length; i++) f[i] = (f[i] - mn) * d;
    return { f, N };
  }

  /**
   * Copy one painted tile into the atlas. Rows are flipped so decal-space +v
   * points UP (posters and drips must not be upside down), the grain field
   * modulates coverage, and the height / ORM channels composite against their
   * neutral defaults using the canvas alpha.
   */
  _blit(img, ch, alb, orm, hgt, AP, TP, ox, oy, grainK, grain, seed) {
    const GF = grain.f, GN = grain.N;
    const gx0 = (seed * 7) % GN, gy0 = (seed * 13) % GN;
    for (let y = 0; y < TP; y++) {
      const srow = (TP - 1 - y) * TP * 4;         // vertical flip
      const drow = (oy + y) * AP + ox;
      const gyr = ((y + gy0) % GN) * GN;
      for (let x = 0; x < TP; x++) {
        const s = srow + x * 4;
        // two decorrelated scales: broad blotching + fine speckle. This is what
        // stops a painted shape from reading as clean vector art.
        const gc = GF[gyr + ((x + gx0) % GN)];
        const gf = GF[(((y * 3 + gy0 + 41) % GN) * GN) + ((x * 3 + gx0 + 17) % GN)];
        const gv = gc * 0.62 + gf * 0.38;
        const d = drow + x;
        const a = img[s + 3] / 255;
        if (ch === 0) {
          let al = a;
          if (grainK > 0) al = clamp01(a * (1 - grainK * 0.85 + grainK * 1.7 * gv));
          const shade = 1 - grainK * 0.30 * (0.5 - gv);
          const o4 = d * 4;
          alb[o4] = Math.min(255, img[s] * shade) | 0;
          alb[o4 + 1] = Math.min(255, img[s + 1] * shade) | 0;
          alb[o4 + 2] = Math.min(255, img[s + 2] * shade) | 0;
          alb[o4 + 3] = (clamp01(al) * 255) | 0;
        } else if (ch === 1) {
          const v = img[s] / 255;
          hgt[d] = 0.5 + (v - 0.5) * a + (gv - 0.5) * 0.045 * a;
        } else {
          const o4 = d * 4;
          orm[o4] = lerp(255, img[s], a) | 0;
          orm[o4 + 1] = (clamp01(lerp(230, img[s + 1], a) / 255 + (gv - 0.5) * 0.08 * a) * 255) | 0;
          orm[o4 + 2] = lerp(0, img[s + 2], a) | 0;
          orm[o4 + 3] = 255;
        }
      }
    }
  }

  /**
   * Push colour outward into fully transparent texels. Without this, bilinear
   * filtering pulls the canvas's black transparent background into every decal
   * edge and every mark gets a dark halo — the classic hobby tell.
   */
  _bleedAlpha(alb, N, iters) {
    const n = N * N;
    const solid = new Uint8Array(n);
    for (let i = 0; i < n; i++) solid[i] = alb[i * 4 + 3] > 2 ? 1 : 0;
    const next = new Uint8Array(n);
    for (let it = 0; it < iters; it++) {
      next.set(solid);
      for (let y = 0; y < N; y++) {
        const yb = y * N;
        const ym = y > 0 ? yb - N : yb;
        const yp = y < N - 1 ? yb + N : yb;
        for (let x = 0; x < N; x++) {
          const i = yb + x;
          if (solid[i]) continue;
          const xm = x > 0 ? x - 1 : x, xp = x < N - 1 ? x + 1 : x;
          let r = 0, g = 0, b = 0, c = 0;
          let j = ym + x;
          if (solid[j]) { r += alb[j * 4]; g += alb[j * 4 + 1]; b += alb[j * 4 + 2]; c++; }
          j = yp + x;
          if (solid[j]) { r += alb[j * 4]; g += alb[j * 4 + 1]; b += alb[j * 4 + 2]; c++; }
          j = yb + xm;
          if (solid[j]) { r += alb[j * 4]; g += alb[j * 4 + 1]; b += alb[j * 4 + 2]; c++; }
          j = yb + xp;
          if (solid[j]) { r += alb[j * 4]; g += alb[j * 4 + 1]; b += alb[j * 4 + 2]; c++; }
          if (!c) continue;
          alb[i * 4] = (r / c) | 0;
          alb[i * 4 + 1] = (g / c) | 0;
          alb[i * 4 + 2] = (b / c) | 0;
          next[i] = 1;
        }
      }
      solid.set(next);
    }
  }

  /** Clamped separable box blur over the height atlas. */
  _blurField(H, N, r) {
    const tmp = new Float32Array(N * N);
    const w = 2 * r + 1, inv = 1 / w;
    for (let y = 0; y < N; y++) {
      const o = y * N;
      for (let x = 0; x < N; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) {
          let xx = x + k; if (xx < 0) xx = 0; else if (xx >= N) xx = N - 1;
          s += H[o + xx];
        }
        tmp[o + x] = s * inv;
      }
    }
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        let s = 0;
        for (let k = -r; k <= r; k++) {
          let yy = y + k; if (yy < 0) yy = 0; else if (yy >= N) yy = N - 1;
          s += tmp[yy * N + x];
        }
        H[y * N + x] = s * inv;
      }
    }
  }

  /** Height field -> tangent-space normal atlas (per tile, clamped at borders). */
  _sobelAtlas(H, N, TP) {
    const out = new Uint8Array(N * N * 4);
    const k = N * 0.0022;
    for (let ty = 0; ty < ATLAS_ROWS; ty++) {
      for (let tx = 0; tx < ATLAS_COLS; tx++) {
        const x0 = tx * TP, y0 = ty * TP, x1 = x0 + TP - 1, y1 = y0 + TP - 1;
        for (let y = y0; y <= y1; y++) {
          const ym = (y > y0 ? y - 1 : y0) * N, yp = (y < y1 ? y + 1 : y1) * N, yc = y * N;
          for (let x = x0; x <= x1; x++) {
            const xm = x > x0 ? x - 1 : x0, xp = x < x1 ? x + 1 : x1;
            const h00 = H[ym + xm], h10 = H[ym + x], h20 = H[ym + xp];
            const h01 = H[yc + xm], h21 = H[yc + xp];
            const h02 = H[yp + xm], h12 = H[yp + x], h22 = H[yp + xp];
            const gx = (h20 + 2 * h21 + h22) - (h00 + 2 * h01 + h02);
            const gy = (h02 + 2 * h12 + h22) - (h00 + 2 * h10 + h20);
            let vx = -gx * k, vy = -gy * k;
            const inv = 1 / Math.sqrt(vx * vx + vy * vy + 1);
            vx *= inv; vy *= inv;
            const o = (yc + x) << 2;
            out[o] = ((vx * 0.5 + 0.5) * 255) | 0;
            out[o + 1] = ((vy * 0.5 + 0.5) * 255) | 0;
            out[o + 2] = ((inv * 0.5 + 0.5) * 255) | 0;
            out[o + 3] = 255;
          }
        }
      }
    }
    return out;
  }

  /* ---------------------------------------------------------------- *
   *  Canvas drawing helpers (author space is always 256 x 256)
   * ---------------------------------------------------------------- */

  /**
   * Closed organic blob. Low-order harmonics only, no per-vertex jitter (that
   * is what turns a blob into a star), and the outline is smoothed through
   * quadratic midpoints so the silhouette has no visible facets.
   */
  _blob(ctx, cx, cy, r, irr, lobes, rnd, squashY) {
    const K = 48;
    const p1 = rnd() * TAU, p2 = rnd() * TAU, p3 = rnd() * TAU, p4 = rnd() * TAU;
    const h4 = 6 + (((lobes | 0) + (rnd() * 3 | 0)) % 5);
    const sq = squashY || 1;
    const pts = this._bpts || (this._bpts = new Float32Array(128));
    for (let i = 0; i < K; i++) {
      const a = (i / K) * TAU;
      const rr = r * (1 + irr * (
        Math.sin(a * 2 + p1) * 0.46 +
        Math.sin(a * 3 + p2) * 0.27 +
        Math.sin(a * 5 + p3) * 0.16 +
        Math.sin(a * h4 + p4) * 0.09));
      pts[i * 2] = cx + Math.cos(a) * rr;
      pts[i * 2 + 1] = cy + Math.sin(a) * rr * sq;
    }
    ctx.beginPath();
    ctx.moveTo((pts[0] + pts[(K - 1) * 2]) * 0.5, (pts[1] + pts[(K - 1) * 2 + 1]) * 0.5);
    for (let i = 0; i < K; i++) {
      const j = (i + 1) % K;
      ctx.quadraticCurveTo(pts[i * 2], pts[i * 2 + 1],
        (pts[i * 2] + pts[j * 2]) * 0.5, (pts[i * 2 + 1] + pts[j * 2 + 1]) * 0.5);
    }
    ctx.closePath();
  }

  /**
   * Soft-edged vertical run: a stack of horizontally-graded slabs. Hard-sided
   * polygons are the reason procedural streaks usually read as vector art.
   */
  _softStreak(ctx, ch, x, w, L, alpha, wob, col) {
    const SEG = 18;
    for (let s = 0; s < SEG; s++) {
      const t = s / SEG;
      const fall = (1 - t) * (1 - t * 0.62);
      if (fall <= 0.002) break;
      const ww = w * (1 - t * 0.5);
      const xc = x + wob * t * t;
      const y0 = t * L, y1 = ((s + 1) / SEG) * L;
      const g = ctx.createLinearGradient(xc - ww, 0, xc + ww, 0);
      g.addColorStop(0, this._c(ch, col(0)));
      g.addColorStop(0.28, this._c(ch, col(alpha * fall * 0.55)));
      g.addColorStop(0.5, this._c(ch, col(alpha * fall)));
      g.addColorStop(0.72, this._c(ch, col(alpha * fall * 0.55)));
      g.addColorStop(1, this._c(ch, col(0)));
      ctx.fillStyle = g;
      ctx.fillRect(xc - ww, y0, ww * 2, y1 - y0 + 1.2);
    }
  }

  /** A soft round puff — the building block of every cloud-like mark. */
  _puff(ctx, ch, x, y, r, o) {
    const g = this._radial(ctx, x, y, 0, r);
    g.addColorStop(0, this._c(ch, o));
    const mid = Object.assign({}, o);
    mid.a = (o.a !== undefined ? o.a : 1) * 0.55;
    if (mid.ha !== undefined) mid.ha *= 0.55;
    if (mid.ra !== undefined) mid.ra *= 0.55;
    g.addColorStop(0.55, this._c(ch, mid));
    const out = Object.assign({}, o);
    out.a = 0; out.ha = 0; out.ra = 0;
    g.addColorStop(1, this._c(ch, out));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    ctx.fill();
  }

  /** Jagged crack polyline. Returns the tip so a branch can continue from it. */
  _crack(ctx, x, y, ang, len, wob, rnd, segs) {
    const n = segs || 7;
    const step = len / n;
    ctx.beginPath();
    ctx.moveTo(x, y);
    let a = ang, px = x, py = y;
    for (let i = 0; i < n; i++) {
      a += (rnd() - 0.5) * wob;
      px += Math.cos(a) * step;
      py += Math.sin(a) * step;
      ctx.lineTo(px, py);
    }
    ctx.stroke();
    return { x: px, y: py, a };
  }

  _radial(ctx, cx, cy, r0, r1) { return ctx.createRadialGradient(cx, cy, r0, cx, cy, r1); }

  /** Soft round falloff so a decal never shows its square tile boundary. */
  _softEdge(ctx, ch, inner, outer) {
    if (ch !== 0) return;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    const g = this._radial(ctx, 128, 128, inner, outer);
    g.addColorStop(0, 'rgba(0,0,0,0)');
    g.addColorStop(0.72, 'rgba(0,0,0,0.35)');
    g.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    ctx.restore();
  }

  /* ---------------------------------------------------------------- *
   *  TILE PAINTERS
   * ---------------------------------------------------------------- */

  /**
   * THE IMPACT.  Every bullet mark on a mineral surface is built from the same
   * four zones, driven by a per-material palette in CRATER:
   *
   *     dust bloom  ->  darker soot bruise  ->  BRIGHT chipped substrate  ->
   *     small dark entry pocket (never black, always warm, with a cool
   *     skylight crescent on its upper lip)
   *
   * The height channel is authored as a real DISH — smooth fall from a raised
   * lip down to the pocket — instead of a flat plate with a cliff at the hole.
   * That is what makes the whole crater, not just a 2 px rim, respond to the
   * raking golden-hour key: lit on the sun side, filled with cool sky bounce
   * on the other, and completely different again when the wall is in shadow.
   */
  _crater(ctx, ch, rnd, P, variant) {
    const V = (variant | 0) & 3;
    const S = 0.80 + V * 0.12;
    const cx = 128 + (rnd() - 0.5) * 13, cy = 128 + (rnd() - 0.5) * 13;
    const rOut = Math.min(120, P.rOut * (0.92 + rnd() * 0.16));
    const rRim = Math.min(94, P.rRim * S * (0.86 + rnd() * 0.30));
    const rCore = Math.max(5.5, P.rCore * S * (0.76 + rnd() * 0.58));
    const D = P.dust, So = P.soot, HO = P.hot, MI = P.mid, ED = P.edge;
    const KP = P.pit, KL = P.pitLo, BO = P.bounce;
    const rg = P.rgh, mt = P.mt;

    // ---- 1. pulverised dust thrown clear, dying smoothly into nothing -----
    let g = this._radial(ctx, cx, cy, rRim * 0.80, rOut);
    g.addColorStop(0.00, this._c(ch, { r: D[0], g: D[1], b: D[2], a: P.dustA, ha: 0, ao: 0.94, rg, mt, ra: P.dustA * 0.85 }));
    g.addColorStop(0.30, this._c(ch, { r: D[0], g: D[1], b: D[2], a: P.dustA * 0.50, ha: 0, ao: 0.97, rg, mt, ra: P.dustA * 0.44 }));
    g.addColorStop(0.66, this._c(ch, { r: D[0], g: D[1], b: D[2], a: P.dustA * 0.18, ha: 0, ao: 0.99, rg, mt, ra: P.dustA * 0.16 }));
    g.addColorStop(1.00, this._c(ch, { r: D[0], g: D[1], b: D[2], a: 0, ha: 0, ao: 1, rg, mt, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // ---- 2. soot / bruise ring hugging the outside of the chipped zone ----
    g = this._radial(ctx, cx, cy, rRim * 0.66, rRim * 2.25);
    g.addColorStop(0.00, this._c(ch, { r: So[0], g: So[1], b: So[2], a: P.sootA * 0.92, ha: 0, ao: 0.74, rg, mt, ra: P.sootA }));
    g.addColorStop(0.26, this._c(ch, { r: So[0], g: So[1], b: So[2], a: P.sootA, ha: 0, ao: 0.72, rg, mt, ra: P.sootA }));
    g.addColorStop(0.62, this._c(ch, { r: So[0], g: So[1], b: So[2], a: P.sootA * 0.36, ha: 0, ao: 0.90, rg, mt, ra: P.sootA * 0.34 }));
    g.addColorStop(1.00, this._c(ch, { r: So[0], g: So[1], b: So[2], a: 0, ha: 0, ao: 1, rg, mt, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // ---- 3. cracking, reaching well past the chipped zone -----------------
    const nC = P.cracks + ((rnd() * 7) | 0);
    for (let i = 0; i < nC; i++) {
      const a = (i / nC) * TAU + (rnd() - 0.5) * 0.8;
      const len = rRim * (0.35 + Math.pow(rnd(), 1.6) * P.crackLen);
      const heavy = rnd() < 0.32;
      ctx.strokeStyle = this._c(ch, {
        r: KP[0] + 6, g: KP[1] + 5, b: KP[2] + 4,
        a: heavy ? (0.34 + rnd() * 0.30) : (0.12 + rnd() * 0.22),
        h: heavy ? 0.28 : 0.38, ao: heavy ? 0.62 : 0.82, rg, mt,
      });
      ctx.lineWidth = heavy ? (0.9 + rnd() * 1.8) : (0.35 + rnd() * 0.85);
      const end = this._crack(ctx, cx + Math.cos(a) * rCore * 0.9, cy + Math.sin(a) * rCore * 0.9,
        a, len, 0.58, rnd, 9);
      if (rnd() < 0.66) {
        ctx.lineWidth = 0.3 + rnd() * 0.7;
        ctx.strokeStyle = this._c(ch, { r: KL[0], g: KL[1], b: KL[2], a: 0.12 + rnd() * 0.16, h: 0.40, ao: 0.86, rg, mt });
        this._crack(ctx, end.x, end.y, end.a + (rnd() - 0.5) * 1.5, len * (0.22 + rnd() * 0.5), 0.9, rnd, 5);
      }
    }

    // ---- 4. the dish: fresh, BRIGHT substrate under a smooth height bowl --
    ctx.save();
    this._blob(ctx, cx, cy, rRim, P.irr, 3 + V, rnd);
    ctx.clip();
    g = this._radial(ctx, cx, cy, 0, rRim * 1.02);
    const hK = P.hCore, hM = P.hMid, hR = P.hRim;
    g.addColorStop(0.00, this._c(ch, { r: HO[0], g: HO[1], b: HO[2], a: 0.97, h: hK, ha: 1, ao: 0.56, rg, mt }));
    g.addColorStop(0.22, this._c(ch, { r: HO[0], g: HO[1], b: HO[2], a: 0.97, h: hK + (hM - hK) * 0.30, ha: 1, ao: 0.64, rg, mt }));
    g.addColorStop(0.52, this._c(ch, { r: MI[0], g: MI[1], b: MI[2], a: 0.96, h: hM, ha: 1, ao: 0.78, rg, mt }));
    g.addColorStop(0.80, this._c(ch, { r: MI[0], g: MI[1], b: MI[2], a: 0.90, h: hM + (hR - hM) * 0.72, ha: 1, ao: 0.90, rg, mt }));
    g.addColorStop(0.93, this._c(ch, { r: ED[0], g: ED[1], b: ED[2], a: 0.66, h: hR, ha: 0.92, ao: 0.96, rg, mt }));
    g.addColorStop(1.00, this._c(ch, { r: ED[0], g: ED[1], b: ED[2], a: 0, h: 0.5, ha: 0, ao: 1, rg, mt, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // exposed aggregate: some grains stand proud and catch the key, some tore
    // out and left pits. This is the micro-relief that sells it at 2 m.
    for (let i = 0; i < P.chips; i++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * rRim * 0.97;
      const sx = cx + Math.cos(a) * rr, sy = cy + Math.sin(a) * rr;
      const sr = 1.4 + rnd() * (rRim * 0.075);
      const proud = rnd() < 0.55;
      ctx.fillStyle = proud
        ? this._c(ch, { r: HO[0] + 6, g: HO[1] + 5, b: HO[2] + 5, a: 0.80, h: 0.70, ao: 1, rg: rg * 0.92, mt })
        : this._c(ch, { r: KL[0] + 14, g: KL[1] + 12, b: KL[2] + 10, a: 0.62, h: 0.24, ao: 0.50, rg, mt });
      ctx.beginPath();
      ctx.ellipse(sx, sy, sr, sr * (0.55 + rnd() * 0.75), rnd() * TAU, 0, TAU);
      ctx.fill();
    }
    ctx.restore();

    // ---- 5. broken chipped lip standing proud of the wall -----------------
    const nA = 8 + ((rnd() * 6) | 0);
    for (let i = 0; i < nA; i++) {
      const a0 = (i / nA) * TAU + rnd() * 0.3;
      const sweep = (0.22 + rnd() * 0.6) * (TAU / nA);
      const rr = rRim * (0.72 + rnd() * 0.28);
      ctx.strokeStyle = this._c(ch, {
        r: HO[0] + 4, g: HO[1] + 4, b: HO[2] + 4,
        a: 0.34 + rnd() * 0.34, h: 0.86, ao: 1, rg: rg * 0.9, mt,
      });
      ctx.lineWidth = 2.0 + rnd() * (rRim * 0.075);
      ctx.beginPath();
      ctx.arc(cx, cy, rr, a0, a0 + sweep);
      ctx.stroke();
    }

    // ---- 6. entry pocket: the darkest value in the whole mark, and it still
    //         sits well above the grade's black floor and takes sky bounce ---
    const ox = cx + (rnd() - 0.5) * rCore * 0.5, oy = cy + (rnd() - 0.5) * rCore * 0.5;
    this._blob(ctx, ox, oy, rCore, 0.26, 6, rnd);
    g = this._radial(ctx, ox - rCore * 0.22, oy + rCore * 0.30, 0, rCore * 1.35);
    g.addColorStop(0.00, this._c(ch, { r: KP[0], g: KP[1], b: KP[2], a: 0.98, h: 0.02, ao: 0.18, rg: 1.0, mt: mt * 0.3 }));
    g.addColorStop(0.58, this._c(ch, { r: KL[0], g: KL[1], b: KL[2], a: 0.97, h: 0.09, ao: 0.26, rg: 1.0, mt: mt * 0.5 }));
    g.addColorStop(1.00, this._c(ch, { r: KL[0] + 22, g: KL[1] + 20, b: KL[2] + 18, a: 0.92, h: 0.20, ao: 0.42, rg: 0.99, mt }));
    ctx.fillStyle = g;
    ctx.fill();
    // cool skylight crescent on the far inner lip — the tell that it is a hole
    ctx.save();
    this._blob(ctx, ox, oy, rCore, 0.26, 6, rnd);
    ctx.clip();
    g = this._radial(ctx, ox + rCore * 0.30, oy - rCore * 0.52, 0, rCore * 1.15);
    g.addColorStop(0.00, this._c(ch, { r: BO[0], g: BO[1], b: BO[2], a: 0.50, h: 0.24, ao: 0.55, rg: 0.98, mt }));
    g.addColorStop(1.00, this._c(ch, { r: BO[0], g: BO[1], b: BO[2], a: 0, ha: 0, ao: 1, rg: 0.98, mt, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    ctx.restore();

    // ---- 7. ejecta specks flung clear of the mark -------------------------
    for (let i = 0; i < 34; i++) {
      const a = rnd() * TAU, rr = rRim * (1.0 + Math.pow(rnd(), 0.7) * 1.25);
      if (rr > rOut * 0.96) continue;
      const bright = rnd() < 0.45;
      ctx.fillStyle = bright
        ? this._c(ch, { r: MI[0], g: MI[1], b: MI[2], a: 0.20 + rnd() * 0.40, h: 0.62, ao: 1, rg, mt })
        : this._c(ch, { r: So[0], g: So[1], b: So[2], a: 0.16 + rnd() * 0.34, h: 0.44, ao: 0.72, rg, mt });
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr,
        0.6 + rnd() * 2.4, 0.6 + rnd() * 1.8, rnd() * TAU, 0, TAU);
      ctx.fill();
    }

    this._softEdge(ctx, ch, 104, 127);
  }

  // --- CONCRETE: pale aggregate crater. v3 is the grimy asphalt palette ----
  _tConcrete(ctx, ch, rnd, variant) {
    this._crater(ctx, ch, rnd, variant === 3 ? CRATER.asphalt : CRATER.concrete, variant);
  }

  // --- PLASTER / RENDER: thin bright skim punched through to a dark core ---
  _tPlaster(ctx, ch, rnd, variant) {
    this._crater(ctx, ch, rnd, CRATER.plaster, variant);
    // flaked, undercut edge where the skim coat has let go of the block
    const cx = 128, cy = 128;
    const r = 52 + variant * 8;
    for (let i = 0; i < 16; i++) {
      const a = rnd() * TAU;
      const rr = r * (0.86 + rnd() * 0.46);
      const w = 4 + rnd() * 13;
      ctx.strokeStyle = this._c(ch, { r: 252, g: 248, b: 238, a: 0.20 + rnd() * 0.30, h: 0.80, ao: 1, rg: 0.94 });
      ctx.lineWidth = 1.2 + rnd() * 2.6;
      ctx.beginPath();
      ctx.arc(cx, cy, rr, a, a + w / rr);
      ctx.stroke();
      ctx.strokeStyle = this._c(ch, { r: 96, g: 87, b: 75, a: 0.16 + rnd() * 0.22, h: 0.30, ao: 0.60, rg: 0.99 });
      ctx.lineWidth = 1.0 + rnd() * 2.0;
      ctx.beginPath();
      ctx.arc(cx, cy, rr * 0.94, a, a + w / rr);
      ctx.stroke();
    }
    this._softEdge(ctx, ch, 106, 127);
  }

  // --- BRICK: sharp pale chip face plus a red-brown dust bloom -------------
  _tBrick(ctx, ch, rnd, variant) {
    this._crater(ctx, ch, rnd, CRATER.brick, variant);
    // angular conchoidal chip flakes — brick breaks in facets, not dishes
    const cx = 128, cy = 128;
    for (let i = 0; i < 7; i++) {
      const a = rnd() * TAU, rr = 16 + rnd() * 30;
      const x = cx + Math.cos(a) * rr * 0.6, y = cy + Math.sin(a) * rr * 0.6;
      const s = 4.5 + rnd() * 9;
      ctx.beginPath();
      const K = 4 + ((rnd() * 3) | 0);
      for (let k = 0; k <= K; k++) {
        const aa = (k / K) * TAU + rnd() * 0.4;
        const r2 = s * (0.55 + rnd() * 0.7);
        const xx = x + Math.cos(aa) * r2, yy = y + Math.sin(aa) * r2;
        if (k === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.closePath();
      ctx.fillStyle = this._c(ch, { r: 234, g: 214, b: 190, a: 0.24 + rnd() * 0.26, h: 0.66, ao: 0.94, rg: 0.96 });
      ctx.fill();
      ctx.strokeStyle = this._c(ch, { r: 118, g: 74, b: 54, a: 0.28, h: 0.34, ao: 0.66, rg: 0.97 });
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
    this._softEdge(ctx, ch, 104, 127);
  }

  // --- METAL: punched hole, peeled bright petals, heat temper, bruise -----
  _tMetal(ctx, ch, rnd, variant) {
    const cx = 128 + (rnd() - 0.5) * 8, cy = 128 + (rnd() - 0.5) * 8;
    const hole = 13 + rnd() * 5;
    const petal = 44 + rnd() * 14 + variant * 6;

    // dished deformation around the entry — the panel is pushed IN, so it
    // catches a dark, wide gradient that reads as a dent under raking light.
    let g = this._radial(ctx, cx, cy, hole, 118);
    g.addColorStop(0, this._c(ch, { r: 58, g: 56, b: 54, a: 0.72, h: 0.30, ao: 0.55, rg: 0.46, mt: 0.95 }));
    g.addColorStop(0.34, this._c(ch, { r: 92, g: 90, b: 88, a: 0.40, h: 0.40, ao: 0.78, rg: 0.50, mt: 0.92 }));
    g.addColorStop(0.7, this._c(ch, { r: 128, g: 126, b: 122, a: 0.16, h: 0.47, ao: 0.92, rg: 0.62, mt: 0.85 }));
    g.addColorStop(1, this._c(ch, { r: 140, g: 136, b: 130, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // heat-tinted temper rings — broken arcs, low contrast
    for (let i = 0; i < 7; i++) {
      const a0 = rnd() * TAU, sw = 0.4 + rnd() * 1.5;
      const warm = rnd() < 0.6;
      ctx.strokeStyle = warm
        ? this._c(ch, { r: 158, g: 98, b: 50, a: 0.14 + rnd() * 0.18, ha: 0, ao: 0.92, rg: 0.40, mt: 1 })
        : this._c(ch, { r: 68, g: 92, b: 136, a: 0.10 + rnd() * 0.14, ha: 0, ao: 0.95, rg: 0.32, mt: 1 });
      ctx.lineWidth = 3 + rnd() * 6;
      ctx.beginPath();
      ctx.arc(cx, cy, petal * (0.42 + rnd() * 0.42), a0, a0 + sw);
      ctx.stroke();
    }

    // scoring — short, low contrast, concentrated near the entry
    for (let i = 0; i < 13; i++) {
      const a = rnd() * TAU;
      const r0 = hole * 1.1 + rnd() * 8, r1 = r0 + 6 + rnd() * 34;
      const lg0 = ctx.createLinearGradient(
        cx + Math.cos(a) * r0, cy + Math.sin(a) * r0,
        cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      lg0.addColorStop(0, this._c(ch, { r: 216, g: 214, b: 208, a: 0.16 + rnd() * 0.22, h: 0.58, ao: 1, rg: 0.20, mt: 1 }));
      lg0.addColorStop(1, this._c(ch, { r: 180, g: 178, b: 174, a: 0, ha: 0, rg: 0.4, ra: 0, mt: 1 }));
      ctx.strokeStyle = lg0;
      ctx.lineWidth = 0.6 + rnd() * 1.4;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }

    // Torn burrs of bare metal peeled back around the entry. Many small,
    // ragged, unequal slivers — a handful of clean symmetric petals reads as a
    // flower, which is the single most common giveaway of a fake bullet hole.
    const nP = 11 + ((rnd() * 7) | 0);
    for (let i = 0; i < nP; i++) {
      const a = (i / nP) * TAU + (rnd() - 0.5) * 0.65;
      const L = hole + (petal - hole) * (0.12 + Math.pow(rnd(), 2.1) * 0.62);
      const wdt = (0.34 + rnd() * 0.34);
      const axx = Math.cos(a), ayy = Math.sin(a);
      const pxx = -ayy, pyy = axx;
      const r0 = hole * 0.82;
      const skew = (rnd() - 0.5) * 0.7;
      const w0 = Math.max(2.2, (L - r0) * wdt), w1 = w0 * (0.34 + rnd() * 0.42);
      ctx.beginPath();
      ctx.moveTo(cx + axx * r0 + pxx * w0, cy + ayy * r0 + pyy * w0);
      ctx.quadraticCurveTo(
        cx + axx * (r0 + (L - r0) * 0.55) + pxx * (w0 * 0.85 + skew * w0),
        cy + ayy * (r0 + (L - r0) * 0.55) + pyy * (w0 * 0.85 + skew * w0),
        cx + axx * L + pxx * w1, cy + ayy * L + pyy * w1);
      ctx.lineTo(cx + axx * (L * 0.94) - pxx * w1 * 1.6, cy + ayy * (L * 0.94) - pyy * w1 * 1.6);
      ctx.quadraticCurveTo(
        cx + axx * (r0 + (L - r0) * 0.5) - pxx * (w0 * 0.9 - skew * w0),
        cy + ayy * (r0 + (L - r0) * 0.5) - pyy * (w0 * 0.9 - skew * w0),
        cx + axx * r0 - pxx * w0 * 0.9, cy + ayy * r0 - pyy * w0 * 0.9);
      ctx.closePath();
      const lg = ctx.createLinearGradient(cx + axx * r0, cy + ayy * r0, cx + axx * L, cy + ayy * L);
      lg.addColorStop(0, this._c(ch, { r: 252, g: 251, b: 247, a: 1, h: 0.92, ao: 1, rg: 0.12, mt: 1 }));
      lg.addColorStop(0.40, this._c(ch, { r: 206, g: 204, b: 200, a: 0.96, h: 0.74, ao: 0.95, rg: 0.20, mt: 1 }));
      lg.addColorStop(0.75, this._c(ch, { r: 138, g: 136, b: 133, a: 0.70, h: 0.58, ao: 0.88, rg: 0.32, mt: 1 }));
      lg.addColorStop(1, this._c(ch, { r: 86, g: 84, b: 82, a: 0.10, h: 0.50, ao: 0.84, rg: 0.44, mt: 1 }));
      ctx.fillStyle = lg;
      ctx.fill();
      // lit top edge on one side only, so the burr reads as bent, not printed
      ctx.strokeStyle = this._c(ch, { r: 255, g: 255, b: 252, a: 0.42, h: 0.98, ao: 1, rg: 0.09, mt: 1 });
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(cx + axx * r0 + pxx * w0, cy + ayy * r0 + pyy * w0);
      ctx.quadraticCurveTo(
        cx + axx * (r0 + (L - r0) * 0.55) + pxx * (w0 * 0.85 + skew * w0),
        cy + ayy * (r0 + (L - r0) * 0.55) + pyy * (w0 * 0.85 + skew * w0),
        cx + axx * L + pxx * w1, cy + ayy * L + pyy * w1);
      ctx.stroke();
    }

    // the hole itself — dark, but lifted off the grade's black floor and given
    // a cool sky crescent on the far lip so it takes ambient bounce
    this._blob(ctx, cx, cy, hole, 0.22, 5, rnd);
    g = this._radial(ctx, cx - hole * 0.2, cy + hole * 0.3, 0, hole * 1.25);
    g.addColorStop(0, this._c(ch, { r: 42, g: 41, b: 40, a: 0.98, h: 0.02, ao: 0.16, rg: 0.90, mt: 0.15 }));
    g.addColorStop(0.7, this._c(ch, { r: 56, g: 55, b: 54, a: 0.98, h: 0.08, ao: 0.22, rg: 0.72, mt: 0.5 }));
    g.addColorStop(1, this._c(ch, { r: 84, g: 83, b: 82, a: 0.96, h: 0.20, ao: 0.34, rg: 0.55, mt: 0.8 }));
    ctx.fillStyle = g;
    ctx.fill();
    ctx.save();
    this._blob(ctx, cx, cy, hole, 0.22, 5, rnd);
    ctx.clip();
    g = this._radial(ctx, cx + hole * 0.3, cy - hole * 0.5, 0, hole * 1.1);
    g.addColorStop(0, this._c(ch, { r: 118, g: 126, b: 140, a: 0.42, h: 0.24, ao: 0.55, rg: 0.60, mt: 0.9 }));
    g.addColorStop(1, this._c(ch, { r: 118, g: 126, b: 140, a: 0, ha: 0, ao: 1, rg: 0.6, mt: 0.9, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    ctx.restore();

    this._softEdge(ctx, ch, 104, 127);
  }

  // --- METAL DENT: no penetration. A pushed-in bowl with the paint scuffed
  //     off the rim down to bright bare steel that flares under the key. -----
  _tMetalDent(ctx, ch, rnd) {
    const cx = 128 + (rnd() - 0.5) * 12, cy = 128 + (rnd() - 0.5) * 12;
    const R = 40 + rnd() * 16;

    // 1. wide, shallow deformation — the whole panel is pulled in
    let g = this._radial(ctx, cx, cy, 0, 116);
    g.addColorStop(0.00, this._c(ch, { r: 74, g: 74, b: 75, a: 0.66, h: 0.10, ha: 1, ao: 0.44, rg: 0.52, mt: 0.95 }));
    g.addColorStop(0.30, this._c(ch, { r: 104, g: 104, b: 104, a: 0.44, h: 0.26, ha: 1, ao: 0.62, rg: 0.56, mt: 0.92 }));
    g.addColorStop(0.52, this._c(ch, { r: 148, g: 148, b: 146, a: 0.18, h: 0.62, ha: 0.9, ao: 0.90, rg: 0.62, mt: 0.88 }));
    g.addColorStop(0.74, this._c(ch, { r: 150, g: 150, b: 148, a: 0.07, h: 0.53, ha: 0.4, ao: 0.97, rg: 0.72, mt: 0.85 }));
    g.addColorStop(1.00, this._c(ch, { r: 150, g: 150, b: 148, a: 0, ha: 0, ao: 1, rg: 0.9, mt: 0.8, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // 2. bare-metal scrape where the coating tore off, brightest at the rim
    ctx.save();
    this._blob(ctx, cx, cy, R * 0.82, 0.30, 4, rnd);
    ctx.clip();
    g = this._radial(ctx, cx, cy, 0, R * 0.9);
    g.addColorStop(0.00, this._c(ch, { r: 196, g: 196, b: 194, a: 0.60, h: 0.16, ha: 1, ao: 0.50, rg: 0.30, mt: 1 }));
    g.addColorStop(0.62, this._c(ch, { r: 226, g: 226, b: 223, a: 0.72, h: 0.34, ha: 1, ao: 0.74, rg: 0.20, mt: 1 }));
    g.addColorStop(1.00, this._c(ch, { r: 248, g: 248, b: 245, a: 0.55, h: 0.62, ha: 1, ao: 0.94, rg: 0.13, mt: 1 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    ctx.restore();

    // 3. bright radial scratches biting outward from the strike
    for (let i = 0; i < 26; i++) {
      const a = rnd() * TAU;
      const r0 = R * (0.18 + rnd() * 0.5), r1 = r0 + R * (0.25 + rnd() * 0.85);
      const lg = ctx.createLinearGradient(
        cx + Math.cos(a) * r0, cy + Math.sin(a) * r0,
        cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      lg.addColorStop(0, this._c(ch, { r: 250, g: 250, b: 247, a: 0.22 + rnd() * 0.34, h: 0.70, ao: 1, rg: 0.10, mt: 1 }));
      lg.addColorStop(1, this._c(ch, { r: 190, g: 190, b: 188, a: 0, ha: 0, ao: 1, rg: 0.4, mt: 1, ra: 0 }));
      ctx.strokeStyle = lg;
      ctx.lineWidth = 0.5 + rnd() * 1.6;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }

    // 4. lead smear + a shallow warm-dark seat, never black
    this._blob(ctx, cx, cy, R * 0.30, 0.34, 5, rnd);
    g = this._radial(ctx, cx, cy, 0, R * 0.36);
    g.addColorStop(0, this._c(ch, { r: 74, g: 72, b: 70, a: 0.80, h: 0.06, ao: 0.26, rg: 0.62, mt: 0.7 }));
    g.addColorStop(1, this._c(ch, { r: 128, g: 126, b: 122, a: 0.30, h: 0.18, ao: 0.60, rg: 0.44, mt: 0.9 }));
    ctx.fillStyle = g;
    ctx.fill();

    this._softEdge(ctx, ch, 100, 127);
  }

  // --- WOOD: torn fibres, pale raised splinters ---------------------------
  _tWood(ctx, ch, rnd, variant) {
    const cx = 128 + (rnd() - 0.5) * 10, cy = 128 + (rnd() - 0.5) * 10;
    const hole = 12 + rnd() * 6 + (variant | 0) * 4;

    let g = this._radial(ctx, cx, cy, hole, 116);
    g.addColorStop(0, this._c(ch, { r: 86, g: 62, b: 36, a: 0.68, h: 0.36, ao: 0.62, rg: 0.94 }));
    g.addColorStop(0.5, this._c(ch, { r: 116, g: 88, b: 56, a: 0.30, h: 0.45, ao: 0.86, rg: 0.92 }));
    g.addColorStop(1, this._c(ch, { r: 140, g: 112, b: 76, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    const grainA = rnd() * TAU;
    const nS = 26 + ((rnd() * 14) | 0);
    for (let i = 0; i < nS; i++) {
      const along = rnd() < 0.72;
      const flip = rnd() < 0.5;
      const a = along ? grainA + (rnd() - 0.5) * 0.85 + (flip ? 0 : Math.PI) : rnd() * TAU;
      const L = 14 + Math.pow(rnd(), 1.4) * 62;
      const wdt = 2.2 + rnd() * 6.5;
      const axx = Math.cos(a), ayy = Math.sin(a), pxx = -ayy, pyy = axx;
      const r0 = hole * 0.55 + rnd() * 5;
      const bend = (rnd() - 0.5) * 0.55;
      const tipW = wdt * (0.15 + rnd() * 0.4);
      ctx.beginPath();
      ctx.moveTo(cx + axx * r0 + pxx * wdt, cy + ayy * r0 + pyy * wdt);
      ctx.quadraticCurveTo(
        cx + axx * (r0 + L * 0.55) + pxx * (wdt * 0.7 + L * bend),
        cy + ayy * (r0 + L * 0.55) + pyy * (wdt * 0.7 + L * bend),
        cx + axx * (r0 + L) + pxx * tipW, cy + ayy * (r0 + L) + pyy * tipW);
      ctx.lineTo(cx + axx * (r0 + L) - pxx * tipW, cy + ayy * (r0 + L) - pyy * tipW);
      ctx.quadraticCurveTo(
        cx + axx * (r0 + L * 0.5) - pxx * (wdt * 0.75 - L * bend),
        cy + ayy * (r0 + L * 0.5) - pyy * (wdt * 0.75 - L * bend),
        cx + axx * r0 - pxx * wdt, cy + ayy * r0 - pyy * wdt);
      ctx.closePath();
      const bright = rnd();
      ctx.fillStyle = bright < 0.58
        ? this._c(ch, { r: 214 - bright * 74, g: 178 - bright * 66, b: 124 - bright * 52, a: 0.92, h: 0.78, ao: 1, rg: 0.84 })
        : this._c(ch, { r: 66, g: 45, b: 26, a: 0.86, h: 0.26, ao: 0.54, rg: 0.95 });
      ctx.fill();
    }

    for (let i = 0; i < 30; i++) {
      const a = rnd() * TAU;
      ctx.strokeStyle = this._c(ch, { r: 226, g: 192, b: 138, a: 0.40 + rnd() * 0.45, h: 0.76, ao: 1, rg: 0.84 });
      ctx.lineWidth = 0.6 + rnd() * 1.4;
      this._crack(ctx, cx + Math.cos(a) * hole, cy + Math.sin(a) * hole, a, 16 + rnd() * 56, 0.9, rnd, 5);
    }

    this._blob(ctx, cx, cy, hole, 0.30, 5, rnd);
    g = this._radial(ctx, cx - hole * 0.2, cy + hole * 0.3, 0, hole * 1.3);
    g.addColorStop(0, this._c(ch, { r: 47, g: 36, b: 26, a: 0.98, h: 0.02, ao: 0.18, rg: 0.96 }));
    g.addColorStop(1, this._c(ch, { r: 74, g: 56, b: 38, a: 0.95, h: 0.18, ao: 0.36, rg: 0.94 }));
    ctx.fillStyle = g;
    ctx.fill();
    ctx.save();
    this._blob(ctx, cx, cy, hole, 0.30, 5, rnd);
    ctx.clip();
    g = this._radial(ctx, cx + hole * 0.3, cy - hole * 0.5, 0, hole * 1.1);
    g.addColorStop(0, this._c(ch, { r: 108, g: 104, b: 104, a: 0.36, h: 0.24, ao: 0.58, rg: 0.95 }));
    g.addColorStop(1, this._c(ch, { r: 108, g: 104, b: 104, a: 0, ha: 0, ao: 1, rg: 0.95, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    ctx.restore();

    this._softEdge(ctx, ch, 108, 127);
  }

  /**
   * DUSTFALL — pulverised substrate and soot running down the wall directly
   * under an impact. Placed upAlign so it always hangs vertically, and kept
   * faint: it is the thing that stops a hole reading as a sticker, not a
   * feature in its own right. Ledge (the impact) is at canvas y = 0.
   */
  _tDustfall(ctx, ch, rnd) {
    // Everything here is built from horizontally-graded slabs and soft puffs.
    // A single flat fillRect would leave a visible rectangular halo on the
    // wall — the exact tell this decal exists to avoid.
    const pale = (al) => ({
      r: 224, g: 216, b: 198, a: al, h: al > 0 ? 0.525 : 0.5, ha: al * 0.45,
      ao: 1 - al * 0.10, rg: 0.99, ra: al,
    });
    const dark = (al) => ({
      r: 78, g: 70, b: 59, a: al, h: al > 0 ? 0.478 : 0.5, ha: al * 0.6,
      ao: 1 - al * 0.35, rg: 0.99, ra: al,
    });

    // broad pale wash bleeding straight down out of the hole
    for (let i = 0; i < 9; i++) {
      const x = 128 + (rnd() - 0.5) * 52;
      this._softStreak(ctx, ch, x, 16 + rnd() * 34, 110 + rnd() * 120,
        0.14 + rnd() * 0.22, (rnd() - 0.5) * 16, pale);
    }
    // finer, darker run-off fingers
    for (let i = 0; i < 14; i++) {
      const x = 128 + (rnd() - 0.5) * 92;
      this._softStreak(ctx, ch, x, 1.2 + rnd() * 6.0, 55 + rnd() * 145,
        0.08 + rnd() * 0.26, (rnd() - 0.5) * 22, dark);
    }
    // a soft head of dust hanging right under the crater
    for (let i = 0; i < 5; i++) {
      this._puff(ctx, ch, 128 + (rnd() - 0.5) * 34, 8 + rnd() * 22, 16 + rnd() * 20, {
        r: 228, g: 220, b: 202, a: 0.14 + rnd() * 0.16, h: 0.53, ha: 0.25,
        ao: 0.95, rg: 0.99, ra: 0.2,
      });
    }
    // grit that fell out of the hole and lodged on the way down
    for (let i = 0; i < 40; i++) {
      const t = Math.pow(rnd(), 0.7);
      const x = 128 + (rnd() - 0.5) * (34 + t * 88);
      const y = 6 + t * 196;
      ctx.fillStyle = rnd() < 0.5
        ? this._c(ch, { r: 232, g: 224, b: 206, a: 0.14 + rnd() * 0.30, h: 0.58, ao: 1, rg: 0.99 })
        : this._c(ch, { r: 78, g: 70, b: 59, a: 0.12 + rnd() * 0.26, h: 0.42, ao: 0.70, rg: 0.99 });
      ctx.beginPath();
      ctx.ellipse(x, y, 0.6 + rnd() * 1.9, 0.6 + rnd() * 1.5, rnd() * TAU, 0, TAU);
      ctx.fill();
    }

    if (ch === 0) {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      const gx = ctx.createLinearGradient(0, 0, 256, 0);
      gx.addColorStop(0, 'rgba(0,0,0,1)');
      gx.addColorStop(0.34, 'rgba(0,0,0,0)');
      gx.addColorStop(0.66, 'rgba(0,0,0,0)');
      gx.addColorStop(1, 'rgba(0,0,0,1)');
      ctx.fillStyle = gx; ctx.fillRect(0, 0, 256, 256);
      const gy = ctx.createLinearGradient(0, 110, 0, 250);
      gy.addColorStop(0, 'rgba(0,0,0,0)');
      gy.addColorStop(1, 'rgba(0,0,0,1)');
      ctx.fillStyle = gy; ctx.fillRect(0, 110, 256, 146);
      const gt = ctx.createLinearGradient(0, 0, 0, 14);
      gt.addColorStop(0, 'rgba(0,0,0,1)');
      gt.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gt; ctx.fillRect(0, 0, 256, 14);
      ctx.restore();
    }
  }

  // --- GLASS: crushed core, long radial fractures, concentric webbing -----
  _tGlass(ctx, ch, rnd) {
    const cx = 128 + (rnd() - 0.5) * 6, cy = 128 + (rnd() - 0.5) * 6;
    const core = 7 + rnd() * 3;
    const N = 11 + ((rnd() * 5) | 0);

    let g = this._radial(ctx, cx, cy, core, 116);
    g.addColorStop(0, this._c(ch, { r: 226, g: 236, b: 240, a: 0.30, h: 0.55, ao: 0.95, rg: 0.36 }));
    g.addColorStop(0.5, this._c(ch, { r: 220, g: 232, b: 238, a: 0.10, h: 0.52, ao: 1, rg: 0.24 }));
    g.addColorStop(1, this._c(ch, { r: 220, g: 232, b: 238, a: 0, ha: 0, rg: 0.12, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    const ang = new Float64Array(N), rad = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      ang[i] = (i / N) * TAU + (rnd() - 0.5) * 0.34;
      rad[i] = 58 + rnd() * 66;
    }
    for (let i = 0; i < N; i++) {
      const a = ang[i], L = rad[i];
      const mx = cx + Math.cos(a + (rnd() - 0.5) * 0.12) * L * 0.55;
      const my = cy + Math.sin(a + (rnd() - 0.5) * 0.12) * L * 0.55;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * core * 0.6, cy + Math.sin(a) * core * 0.6);
      ctx.quadraticCurveTo(mx, my, cx + Math.cos(a) * L, cy + Math.sin(a) * L);
      ctx.strokeStyle = this._c(ch, { r: 244, g: 250, b: 252, a: 0.62, h: 0.30, ao: 0.80, rg: 0.50 });
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.strokeStyle = this._c(ch, { r: 150, g: 172, b: 182, a: 0.50, h: 0.22, ao: 0.70, rg: 0.55 });
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    // Concentric webbing — deliberately incomplete. Every segment can be
    // missing, and the radii wander, so it never reads as a drawn spiderweb.
    const rings = 5;
    for (let r = 1; r <= rings; r++) {
      const f = (r / (rings + 0.7)) * (0.8 + rnd() * 0.5);
      ctx.strokeStyle = this._c(ch, { r: 238, g: 246, b: 250, a: 0.36 - f * 0.16, h: 0.32, ao: 0.85, rg: 0.50 });
      ctx.lineWidth = 1.6 - f * 0.8;
      for (let i = 0; i < N; i++) {
        const skip = rnd() < 0.34;
        const a0 = ang[i], a1 = ang[(i + 1) % N] + (i === N - 1 ? TAU : 0);
        const r0 = rad[i] * f * (0.75 + rnd() * 0.5);
        const r1 = rad[(i + 1) % N] * f * (0.75 + rnd() * 0.5);
        const am = (a0 + a1) * 0.5, rm = (r0 + r1) * 0.5 * (0.68 + rnd() * 0.3);
        if (skip) continue;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a0) * r0, cy + Math.sin(a0) * r0);
        ctx.quadraticCurveTo(cx + Math.cos(am) * rm, cy + Math.sin(am) * rm,
          cx + Math.cos(a1) * r1, cy + Math.sin(a1) * r1);
        ctx.stroke();
      }
    }

    this._blob(ctx, cx, cy, core * 1.5, 0.40, 6, rnd);
    ctx.fillStyle = this._c(ch, { r: 228, g: 238, b: 242, a: 0.88, h: 0.62, ao: 0.75, rg: 0.66 });
    ctx.fill();
    this._blob(ctx, cx, cy, core * 0.75, 0.40, 5, rnd);
    ctx.fillStyle = this._c(ch, { r: 56, g: 64, b: 70, a: 0.90, h: 0.10, ao: 0.25, rg: 0.70 });
    ctx.fill();
    for (let i = 0; i < 22; i++) {
      const a = rnd() * TAU, rr = core + Math.sqrt(rnd()) * 44;
      ctx.fillStyle = this._c(ch, { r: 250, g: 254, b: 255, a: 0.3 + rnd() * 0.5, h: 0.80, ao: 1, rg: 0.10 });
      ctx.beginPath();
      ctx.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, 0.6 + rnd() * 1.4, 0, TAU);
      ctx.fill();
    }

    this._softEdge(ctx, ch, 104, 127);
  }

  // --- SAND: soft crater with a raised lip and an ejecta apron ------------
  _tSand(ctx, ch, rnd) {
    const cx = 128 + (rnd() - 0.5) * 10, cy = 128 + (rnd() - 0.5) * 10;
    const r = 46 + rnd() * 16;

    let g = this._radial(ctx, cx, cy, r * 0.8, r * 2.6);
    g.addColorStop(0, this._c(ch, { r: 196, g: 172, b: 128, a: 0.55, h: 0.58, ao: 0.95, rg: 0.98 }));
    g.addColorStop(0.5, this._c(ch, { r: 188, g: 164, b: 122, a: 0.22, h: 0.54, ao: 1, rg: 0.98 }));
    g.addColorStop(1, this._c(ch, { r: 184, g: 160, b: 118, a: 0, ha: 0, rg: 0.95, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 40; i++) {
      const a = rnd() * TAU, r0 = r * (0.9 + rnd() * 0.4), L = 6 + rnd() * 44;
      ctx.strokeStyle = this._c(ch, { r: 206, g: 184, b: 142, a: 0.10 + rnd() * 0.3, h: 0.60, ao: 1, rg: 0.98 });
      ctx.lineWidth = 1 + rnd() * 3.4;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * (r0 + L), cy + Math.sin(a) * (r0 + L));
      ctx.stroke();
    }

    // raised lip, broken into arcs
    for (let i = 0; i < 11; i++) {
      const a0 = rnd() * TAU, sw = 0.3 + rnd() * 1.0;
      ctx.strokeStyle = this._c(ch, { r: 222, g: 202, b: 162, a: 0.35 + rnd() * 0.45, h: 0.82, ao: 1, rg: 0.98 });
      ctx.lineWidth = 4 + rnd() * 8;
      ctx.beginPath();
      ctx.arc(cx, cy, r * (0.82 + rnd() * 0.2), a0, a0 + sw);
      ctx.stroke();
    }

    this._blob(ctx, cx, cy, r * 0.86, 0.22, 5, rnd);
    ctx.save();
    ctx.clip();
    g = this._radial(ctx, cx, cy, 0, r);
    g.addColorStop(0, this._c(ch, { r: 74, g: 60, b: 40, a: 0.92, h: 0.08, ao: 0.32, rg: 0.99 }));
    g.addColorStop(0.5, this._c(ch, { r: 124, g: 102, b: 70, a: 0.80, h: 0.26, ao: 0.60, rg: 0.99 }));
    g.addColorStop(1, this._c(ch, { r: 172, g: 148, b: 108, a: 0.30, h: 0.46, ao: 0.88, rg: 0.98 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);
    // coarse grit collected in the bowl
    for (let i = 0; i < 70; i++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * r * 0.9;
      ctx.fillStyle = rnd() < 0.5
        ? this._c(ch, { r: 208, g: 186, b: 144, a: 0.4 + rnd() * 0.4, h: 0.60, ao: 1, rg: 0.98 })
        : this._c(ch, { r: 62, g: 50, b: 34, a: 0.3 + rnd() * 0.4, h: 0.20, ao: 0.5, rg: 0.99 });
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr,
        0.9 + rnd() * 2.6, 0.9 + rnd() * 2.0, rnd() * TAU, 0, TAU);
      ctx.fill();
    }
    ctx.restore();

    this._softEdge(ctx, ch, 100, 127);
  }

  // --- FLESH: small wound + tight mist ------------------------------------
  _tFlesh(ctx, ch, rnd) {
    const cx = 128, cy = 128;
    let g = this._radial(ctx, cx, cy, 6, 84);
    g.addColorStop(0, this._c(ch, { r: 96, g: 16, b: 14, a: 0.80, h: 0.40, ao: 0.60, rg: 0.40 }));
    g.addColorStop(0.4, this._c(ch, { r: 74, g: 12, b: 11, a: 0.28, h: 0.46, ao: 0.85, rg: 0.40 }));
    g.addColorStop(1, this._c(ch, { r: 60, g: 10, b: 9, a: 0, ha: 0, rg: 0.50, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    for (let i = 0; i < 46; i++) {
      const a = rnd() * TAU, rr = 10 + Math.pow(rnd(), 0.6) * 96;
      const s = 0.8 + rnd() * 3.4 * (1 - rr / 130);
      ctx.fillStyle = this._c(ch, { r: 108, g: 18, b: 15, a: 0.3 + rnd() * 0.6, h: 0.56, ao: 0.80, rg: 0.30 });
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, Math.max(0.4, s * 1.6), Math.max(0.3, s), a, 0, TAU);
      ctx.fill();
    }

    this._blob(ctx, cx, cy, 11, 0.40, 5, rnd);
    ctx.fillStyle = this._c(ch, { r: 62, g: 16, b: 14, a: 0.95, h: 0.18, ao: 0.30, rg: 0.28 });
    ctx.fill();

    this._softEdge(ctx, ch, 100, 127);
  }

  // --- SCORCH: soot burst with tongues, ash bloom and spall flecks --------
  _tScorch(ctx, ch, rnd, variant) {
    const cx = 128, cy = 128;
    const R = variant ? 104 : 82;

    // 1. the burn itself: an accumulation of soft soot puffs, densest at the
    //    seat of the blast. Overlapping gradients give a cloudy, filthy edge
    //    that no single shape can fake.
    const nP = 120;
    for (let i = 0; i < nP; i++) {
      const a = rnd() * TAU;
      const d = Math.pow(rnd(), 0.5) * R * 0.98;
      const x = cx + Math.cos(a) * d, y = cy + Math.sin(a) * d;
      const rr = R * (0.10 + rnd() * 0.26) * (1.05 - d / (R * 1.9));
      const near = 1 - d / (R * 1.15);
      this._puff(ctx, ch, x, y, Math.max(3, rr), {
        r: 44, g: 39, b: 34, a: (0.10 + rnd() * 0.20) * (0.35 + near),
        h: 0.472, ha: (0.05 + rnd() * 0.08),
        ao: 0.55 + 0.3 * (1 - near), rg: 1.0, ra: (0.10 + rnd() * 0.22) * (0.35 + near),
      });
    }

    // 2. dense charred seat
    let g = this._radial(ctx, cx, cy, 0, R * 0.52);
    g.addColorStop(0, this._c(ch, { r: 38, g: 34, b: 30, a: 0.88, h: 0.42, ao: 0.20, rg: 1.0 }));
    g.addColorStop(0.5, this._c(ch, { r: 46, g: 41, b: 36, a: 0.62, h: 0.45, ao: 0.36, rg: 1.0 }));
    g.addColorStop(1, this._c(ch, { r: 58, g: 52, b: 45, a: 0, ha: 0, rg: 1.0, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // 3. fine soot streaks blown outward — thin, tapered, mostly transparent
    const nS = 48;
    for (let i = 0; i < nS; i++) {
      const a = rnd() * TAU;
      const r0 = R * (0.20 + rnd() * 0.35), r1 = r0 + R * (0.25 + rnd() * 0.72);
      const lg = ctx.createLinearGradient(
        cx + Math.cos(a) * r0, cy + Math.sin(a) * r0,
        cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      lg.addColorStop(0, this._c(ch, { r: 42, g: 37, b: 33, a: 0.16 + rnd() * 0.20, ha: 0, ao: 0.62, rg: 1.0 }));
      lg.addColorStop(1, this._c(ch, { r: 58, g: 52, b: 45, a: 0, ha: 0, rg: 1.0, ra: 0 }));
      ctx.strokeStyle = lg;
      ctx.lineWidth = 1.2 + rnd() * 6.5;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }

    // 4. pale ash bloom, CLIPPED to the burn so it cannot halo over clean wall
    if (variant) {
      ctx.save();
      this._blob(ctx, cx, cy, R * 0.52, 0.30, 4, rnd);
      ctx.clip();
      g = this._radial(ctx, cx - 6, cy + 4, 0, R * 0.5);
      g.addColorStop(0, this._c(ch, { r: 126, g: 116, b: 102, a: 0.34, h: 0.53, ao: 0.72, rg: 0.99 }));
      g.addColorStop(1, this._c(ch, { r: 104, g: 96, b: 85, a: 0, ha: 0, rg: 0.99, ra: 0 }));
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 256, 256);
      ctx.restore();
    }

    // 5. spalled flecks and grit thrown clear of the burn
    for (let i = 0; i < 90; i++) {
      const a = rnd() * TAU, rr = Math.sqrt(rnd()) * R * 1.18;
      ctx.fillStyle = this._c(ch, { r: 40, g: 36, b: 31, a: 0.14 + rnd() * 0.45, h: 0.44, ao: 0.55, rg: 1.0 });
      ctx.beginPath();
      ctx.ellipse(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr,
        0.7 + rnd() * 3.0, 0.7 + rnd() * 2.2, rnd() * TAU, 0, TAU);
      ctx.fill();
    }

    this._softEdge(ctx, ch, 104, 127);
  }

  // --- BLOOD: directional throw (up in uv), lobed pool, running drips -----
  _tBlood(ctx, ch, rnd, variant) {
    const cx = 128, cy = 148 - variant * 10;
    const R = 30 + rnd() * 14 + variant * 6;

    const DR = 74, DG = 15, DB = 12;     // deep and oxygenated, never black
    const ER = 46, EG = 13, EB = 11;     // dried edge
    const WR = 108, WG = 22, WB = 17;    // wet highlight

    let g = this._radial(ctx, cx, cy, R * 0.6, R * 2.4);
    g.addColorStop(0, this._c(ch, { r: ER, g: EG, b: EB, a: 0.50, ha: 0, ao: 0.78, rg: 0.66 }));
    g.addColorStop(1, this._c(ch, { r: ER, g: EG, b: EB, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    // main mass — three overlapping films, so the silhouette is a run of blood
    // rather than one tidy amoeba
    for (let k = 0; k < 3; k++) {
      const ox = cx + (rnd() - 0.5) * R * 0.75;
      const oy = cy + (rnd() - 0.5) * R * 0.60;
      const rr = R * (0.62 + rnd() * 0.48);
      this._blob(ctx, ox, oy, rr, 0.36, 4, rnd, 0.82 + rnd() * 0.22);
      g = this._radial(ctx, ox - rr * 0.22, oy - rr * 0.26, 0, rr * 1.2);
      g.addColorStop(0, this._c(ch, { r: WR, g: WG, b: WB, a: 0.97, h: 0.545, ao: 0.88, rg: 0.26 }));
      g.addColorStop(0.55, this._c(ch, { r: DR, g: DG, b: DB, a: 0.96, h: 0.535, ao: 0.72, rg: 0.30 }));
      g.addColorStop(1, this._c(ch, { r: ER, g: EG, b: EB, a: 0.88, h: 0.52, ao: 0.58, rg: 0.44 }));
      ctx.fillStyle = g;
      ctx.fill();
    }

    // small satellites clinging to the main mass
    for (let i = 0; i < 7; i++) {
      const a = rnd() * TAU, rr = R * (0.80 + rnd() * 0.55);
      this._blob(ctx, cx + Math.cos(a) * rr, cy + Math.sin(a) * rr * 0.8,
        R * (0.07 + rnd() * 0.15), 0.55, 4, rnd);
      ctx.fillStyle = this._c(ch, { r: DR, g: DG, b: DB, a: 0.86, h: 0.536, ao: 0.78, rg: 0.30 });
      ctx.fill();
    }

    // DIRECTIONAL THROW — droplets flung toward canvas -y (world up). Each is a
    // round bead with a soft tapered tail; hard triangles read as glass shards.
    const nD = 40 + ((rnd() * 26) | 0);
    for (let i = 0; i < nD; i++) {
      const a = -Math.PI * 0.5 + (rnd() - 0.5) * 1.6;
      const dist = R * (0.85 + Math.pow(rnd(), 0.6) * 3.2);
      const x = cx + Math.cos(a) * dist, y = cy + Math.sin(a) * dist;
      const s = Math.max(0.6, (3.9 - 2.6 * (dist / (R * 4.1))) * Math.pow(rnd(), 1.7) * 1.5);
      const el = 1 + rnd() * 1.15;
      const tail = rnd() < 0.55;
      const tl = s * (2.2 + rnd() * 3.4);
      const al = 0.62 + rnd() * 0.38;
      if (y < -16 || y > 272) continue;
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(a + Math.PI * 0.5);
      if (tail) {
        // tapered comet tail drawn as a rounded stroke, not a polygon
        const tg = ctx.createLinearGradient(0, 0, 0, -tl);
        tg.addColorStop(0, this._c(ch, { r: DR, g: DG, b: DB, a: al * 0.75, h: 0.535, ao: 0.88, rg: 0.32 }));
        tg.addColorStop(1, this._c(ch, { r: ER, g: EG, b: EB, a: 0, ha: 0, rg: 0.6, ra: 0 }));
        ctx.strokeStyle = tg;
        ctx.lineWidth = s * 0.9;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -tl);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.ellipse(0, 0, s, s * el, 0, 0, TAU);
      ctx.fillStyle = this._c(ch, { r: DR + 12, g: DG, b: DB, a: al, h: 0.548, ao: 0.85, rg: 0.26 });
      ctx.fill();
      ctx.restore();
    }

    // DRIPS — running toward canvas +y, which is world DOWN after the flip
    const nR = 4 + ((rnd() * 4) | 0);
    for (let i = 0; i < nR; i++) {
      const ox = cx + (rnd() - 0.5) * R * 1.7;
      const oy = cy + R * (0.55 + rnd() * 0.35);
      const L = 16 + rnd() * 92;
      const w = 1.2 + rnd() * 2.6;
      const wob = (rnd() - 0.5) * 16;
      const lg = ctx.createLinearGradient(ox, oy, ox, oy + L);
      lg.addColorStop(0, this._c(ch, { r: DR, g: DG, b: DB, a: 0.92, h: 0.545, ao: 0.80, rg: 0.26 }));
      lg.addColorStop(0.8, this._c(ch, { r: ER, g: EG, b: EB, a: 0.78, h: 0.535, ao: 0.85, rg: 0.32 }));
      lg.addColorStop(1, this._c(ch, { r: ER, g: EG, b: EB, a: 0, ha: 0, rg: 0.50, ra: 0 }));
      ctx.fillStyle = lg;
      ctx.beginPath();
      ctx.moveTo(ox - w, oy);
      ctx.quadraticCurveTo(ox - w * 0.5 + wob, oy + L * 0.6, ox + wob * 0.7 - w * 0.22, oy + L);
      ctx.lineTo(ox + wob * 0.7 + w * 0.22, oy + L);
      ctx.quadraticCurveTo(ox + w * 0.5 + wob, oy + L * 0.6, ox + w, oy);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ox + wob * 0.7, oy + L, w * 0.72, 0, TAU);
      ctx.fillStyle = this._c(ch, { r: DR, g: DG, b: DB, a: 0.85, h: 0.575, ao: 0.85, rg: 0.22 });
      ctx.fill();
    }

    if (ch === 0) this._rectFade(ctx, 0.09, 0.08);
  }

  // --- GRIME: run-off streaking down a wall (ledge is at canvas y = 0) ----
  _tGrime(ctx, ch, rnd) {
    const cols = 34;
    for (let i = 0; i < cols; i++) {
      const x = (i + rnd() * 1.2) * (256 / cols);
      const w = 1.6 + rnd() * 11;
      const L = 80 + rnd() * 176;
      const a = 0.20 + rnd() * 0.55;
      const wob = (rnd() - 0.5) * 14;
      this._softStreak(ctx, ch, x, w, L, a, wob,
        (al) => ({ r: 46, g: 38, b: 29, a: al, h: al > 0 ? 0.455 : 0.5, ha: al * 0.9, ao: 1 - al * 0.3, rg: 0.99, ra: al }));
    }

    // heavier pooled band right under the ledge
    let lg = ctx.createLinearGradient(0, 0, 0, 68);
    lg.addColorStop(0, this._c(ch, { r: 34, g: 28, b: 22, a: 0.62, h: 0.42, ao: 0.70, rg: 0.99 }));
    lg.addColorStop(1, this._c(ch, { r: 48, g: 40, b: 31, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = lg;
    ctx.fillRect(0, 0, 256, 68);

    // mineral tide-lines crossing the streaks
    for (let i = 0; i < 16; i++) {
      const y = rnd() * 210;
      ctx.strokeStyle = this._c(ch, { r: 128, g: 118, b: 100, a: 0.08 + rnd() * 0.12, h: 0.56, ao: 1, rg: 0.98 });
      ctx.lineWidth = 0.8 + rnd() * 2.2;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.bezierCurveTo(85, y + (rnd() - 0.5) * 18, 170, y + (rnd() - 0.5) * 18, 256, y + (rnd() - 0.5) * 10);
      ctx.stroke();
    }

    if (ch === 0) {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      const g2 = ctx.createLinearGradient(0, 0, 256, 0);
      g2.addColorStop(0, 'rgba(0,0,0,1)');
      g2.addColorStop(0.14, 'rgba(0,0,0,0)');
      g2.addColorStop(0.86, 'rgba(0,0,0,0)');
      g2.addColorStop(1, 'rgba(0,0,0,1)');
      ctx.fillStyle = g2; ctx.fillRect(0, 0, 256, 256);
      const g3 = ctx.createLinearGradient(0, 200, 0, 256);
      g3.addColorStop(0, 'rgba(0,0,0,0)');
      g3.addColorStop(1, 'rgba(0,0,0,1)');
      ctx.fillStyle = g3; ctx.fillRect(0, 200, 256, 56);
      const g4 = ctx.createLinearGradient(0, 0, 0, 10);
      g4.addColorStop(0, 'rgba(0,0,0,1)');
      g4.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g4; ctx.fillRect(0, 0, 256, 10);
      ctx.restore();
    }
  }

  // --- OIL: dark soaked pool, glossy film, splashed satellites ------------
  _tOil(ctx, ch, rnd) {
    const cx = 128 + (rnd() - 0.5) * 14, cy = 128 + (rnd() - 0.5) * 14;

    let g = this._radial(ctx, cx, cy, 40, 118);
    g.addColorStop(0, this._c(ch, { r: 46, g: 40, b: 33, a: 0.50, ha: 0, ao: 0.78, rg: 0.92 }));
    g.addColorStop(0.6, this._c(ch, { r: 52, g: 45, b: 37, a: 0.20, ha: 0, ao: 0.90, rg: 0.90 }));
    g.addColorStop(1, this._c(ch, { r: 56, g: 49, b: 40, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 256, 256);

    for (let k = 0; k < 3; k++) {
      const ox = cx + (rnd() - 0.5) * 34, oy = cy + (rnd() - 0.5) * 34;
      const rr = 30 + rnd() * 34;
      this._blob(ctx, ox, oy, rr, 0.36, 3, rnd, 0.82);
      ctx.fillStyle = this._c(ch, { r: 39, g: 35, b: 31, a: 0.88, h: 0.54, ao: 0.50, rg: 0.10 });
      ctx.fill();
    }
    ctx.strokeStyle = this._c(ch, { r: 62, g: 55, b: 44, a: 0.40, h: 0.60, ao: 0.70, rg: 0.30 });
    ctx.lineWidth = 2.4;
    this._blob(ctx, cx, cy, 52, 0.30, 3, rnd, 0.82);
    ctx.stroke();

    for (let i = 0; i < 26; i++) {
      const a = rnd() * TAU, rr = 46 + Math.pow(rnd(), 0.7) * 72;
      const s = 1 + rnd() * 5;
      this._blob(ctx, cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, s, 0.5, 4, rnd);
      ctx.fillStyle = this._c(ch, { r: 42, g: 38, b: 33, a: 0.5 + rnd() * 0.45, h: 0.53, ao: 0.60, rg: 0.14 });
      ctx.fill();
    }

    this._softEdge(ctx, ch, 104, 127);
  }

  // --- POSTER: sun-bleached paper notice, torn and weathered --------------
  _tPoster(ctx, ch, rnd, variant) {
    const x0 = 26, y0 = 14, w = 204, h = 228;
    const pr = variant ? 206 : 222, pg = variant ? 176 : 214, pb = variant ? 132 : 196;
    const ir = variant ? 122 : 54, ig = variant ? 40 : 52, ib = variant ? 30 : 50;

    let g = ctx.createLinearGradient(x0, y0, x0 + w, y0 + h);
    g.addColorStop(0, this._c(ch, { r: pr, g: pg, b: pb, a: 0.97, h: 0.62, ao: 1, rg: 0.86 }));
    g.addColorStop(0.55, this._c(ch, { r: pr - 22, g: pg - 22, b: pb - 20, a: 0.97, h: 0.60, ao: 0.98, rg: 0.88 }));
    g.addColorStop(1, this._c(ch, { r: pr - 46, g: pg - 44, b: pb - 40, a: 0.96, h: 0.58, ao: 0.92, rg: 0.90 }));
    ctx.fillStyle = g;
    ctx.fillRect(x0, y0, w, h);

    ctx.strokeStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.55, h: 0.60, ao: 1, rg: 0.88 });
    ctx.lineWidth = 3;
    ctx.strokeRect(x0 + 10, y0 + 10, w - 20, h - 20);

    ctx.fillStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.72, h: 0.61, ao: 1, rg: 0.90 });
    ctx.fillRect(x0 + 22, y0 + 24, w - 44, 22);

    if (variant) {
      ctx.beginPath();
      ctx.arc(x0 + w * 0.5, y0 + 96, 34, 0, TAU);
      ctx.strokeStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.60, h: 0.61, ao: 1, rg: 0.90 });
      ctx.lineWidth = 5; ctx.stroke();
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * TAU - Math.PI / 2;
        const xx = x0 + w * 0.5 + Math.cos(a) * 20, yy = y0 + 96 + Math.sin(a) * 20;
        if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.closePath();
      ctx.fillStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.45, h: 0.61, ao: 1, rg: 0.90 });
      ctx.fill();
    } else {
      ctx.fillStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.30, h: 0.61, ao: 1, rg: 0.90 });
      ctx.fillRect(x0 + 30, y0 + 62, w - 60, 62);
      ctx.fillStyle = this._c(ch, { r: pr - 60, g: pg - 58, b: pb - 54, a: 0.60, h: 0.61, ao: 1, rg: 0.90 });
      ctx.fillRect(x0 + 40, y0 + 74, w - 80, 38);
    }

    // rows of abstract type blocks
    let ty = y0 + (variant ? 148 : 140);
    while (ty < y0 + h - 34) {
      let tx = x0 + 24 + rnd() * 8;
      const lim = x0 + w - 24 - rnd() * 26;
      while (tx < lim) {
        const bw = 4 + rnd() * 17;
        if (tx + bw > lim) break;
        ctx.fillStyle = this._c(ch, { r: ir, g: ig, b: ib, a: 0.34 + rnd() * 0.3, h: 0.605, ao: 1, rg: 0.90 });
        ctx.fillRect(tx, ty, bw, 4.5);
        tx += bw + 3 + rnd() * 4;
      }
      ty += 11 + rnd() * 3;
    }

    // fold crease
    const fx = x0 + w * (0.3 + rnd() * 0.4);
    const lg = ctx.createLinearGradient(fx - 6, 0, fx + 6, 0);
    lg.addColorStop(0, this._c(ch, { r: 255, g: 255, b: 255, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    lg.addColorStop(0.5, this._c(ch, { r: 120, g: 112, b: 100, a: 0.32, h: 0.44, ao: 0.85, rg: 0.90 }));
    lg.addColorStop(1, this._c(ch, { r: 255, g: 255, b: 255, a: 0, ha: 0, rg: 0.90, ra: 0 }));
    ctx.fillStyle = lg;
    ctx.fillRect(fx - 6, y0, 12, h);

    // water staining
    for (let i = 0; i < 5; i++) {
      const sx = x0 + rnd() * w, sy = y0 + rnd() * h;
      const rr = 14 + rnd() * 42;
      const rg2 = this._radial(ctx, sx, sy, rr * 0.3, rr);
      rg2.addColorStop(0, this._c(ch, { r: 128, g: 108, b: 78, a: 0.16, ha: 0, ao: 0.90, rg: 0.92 }));
      rg2.addColorStop(1, this._c(ch, { r: 128, g: 108, b: 78, a: 0, ha: 0, rg: 0.90, ra: 0 }));
      ctx.fillStyle = rg2;
      ctx.beginPath(); ctx.arc(sx, sy, rr, 0, TAU); ctx.fill();
    }

    // lifted top edge catching the sun
    ctx.fillStyle = this._c(ch, { r: 246, g: 240, b: 226, a: 0.30, h: 0.80, ao: 1, rg: 0.84 });
    ctx.fillRect(x0, y0, w, 5);

    // TEARS — punch the paper away
    const erase = ch === 0 ? 'rgba(0,0,0,1)' : (ch === 1 ? 'rgb(128,128,128)' : 'rgb(255,230,0)');
    if (ch === 0) ctx.globalCompositeOperation = 'destination-out';
    for (let k = 0; k < 3; k++) {
      const px = k === 0 ? x0 + w : (k === 1 ? x0 : x0);
      const py = k === 0 ? y0 : (k === 1 ? y0 + h : y0 + h * (0.3 + rnd() * 0.4));
      const rr = k === 0 ? 34 : (k === 1 ? 26 : 20);
      ctx.beginPath();
      const K = 14;
      for (let i = 0; i <= K; i++) {
        const a = (i / K) * TAU;
        const r2 = rr * (0.55 + rnd() * 0.75);
        const xx = px + Math.cos(a) * r2, yy = py + Math.sin(a) * r2;
        if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.closePath();
      ctx.fillStyle = erase;
      ctx.fill();
    }
    if (ch === 0) ctx.globalCompositeOperation = 'source-over';

    // everything outside the sheet is empty
    if (ch === 0) {
      ctx.save();
      ctx.globalCompositeOperation = 'destination-in';
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.fillRect(x0 - 1, y0 - 1, w + 2, h + 2);
      ctx.restore();
    }
  }

  /** Rectangular alpha falloff (fractions of the tile). */
  _rectFade(ctx, fy, fx) {
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    const a = ctx.createLinearGradient(0, 0, 0, 256);
    a.addColorStop(0, 'rgba(0,0,0,1)');
    a.addColorStop(fy, 'rgba(0,0,0,0)');
    a.addColorStop(1 - fy, 'rgba(0,0,0,0)');
    a.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.fillStyle = a; ctx.fillRect(0, 0, 256, 256);
    const b = ctx.createLinearGradient(0, 0, 256, 0);
    b.addColorStop(0, 'rgba(0,0,0,1)');
    b.addColorStop(fx, 'rgba(0,0,0,0)');
    b.addColorStop(1 - fx, 'rgba(0,0,0,0)');
    b.addColorStop(1, 'rgba(0,0,0,1)');
    ctx.fillStyle = b; ctx.fillRect(0, 0, 256, 256);
    ctx.restore();
  }
}

export default Decals;
