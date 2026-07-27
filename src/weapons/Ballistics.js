// ============================================================================
// BLACKSITE — src/weapons/Ballistics.js
// Owner: Ballistics agent.  Everything that happens AFTER the trigger.
//
// Responsibilities
//   * spread / bloom model (fire duration, movement, hip vs ADS, crouch, air)
//   * the trace itself (physics raycast, with a hardened internal fallback)
//   * hitbox resolution against game.registry.enemies (head 2.4 / torso 1 / limb 0.75)
//   * damage falloff across Config.weapon.range
//   * wall penetration — thickness measured with a reverse probe, damage
//     attenuated by material cost, exit impacts spawned on the far side
//   * ricochets at shallow angles against hard surfaces, with a deflected trace
//   * tracers (a minority of rounds, wall-clock lifetimes) and near-miss whizz-by
//     scheduling
//   * enemy return fire through the same pipeline, emitting 'damage' on the
//     player with a world direction for the HUD's directional hit indicator
//
// It spawns NO impact visuals.  Decals / Particles / Audio listen for 'impact'.
// The only thing this module draws is the tracer billboard batch.
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators — read this)
// ---------------------------------------------------------------------------
//   game.ballistics.fire(opts)                 fire one round through the pipeline
//        opts: { origin:Vector3, dir:Vector3, damage, range, spread (rad),
//                pellets, penetration (0..2), tracer:bool, shooter, weapon }
//   game.ballistics.enemyFire({origin, dir, shooter, damage, accuracy, target, ...})
//        AI shooting MUST go through this so it obeys the same rules.
//   game.ballistics.raycast(origin, dir, maxDist[, skipObject]) -> hit | null
//        hit = { t, point:Vector3, normal:Vector3, object, material, surface, key }
//        Records come from a ring buffer — copy what you need synchronously.
//   game.ballistics.losClear(fromVec3, toVec3) -> bool     (world line of sight)
//   game.ballistics.getSpread()                -> cone HALF-angle in radians
//   game.ballistics.spreadDeg                  -> the same, in degrees
//   game.ballistics.getCrosshairPixels(h)      -> crosshair half-gap in px
//   game.ballistics.registerHitboxes(enemy, list)
//   game.ballistics.playerHp / .playerMaxHp    (authoritative only if unclaimed)
//   game.ballistics.stats                      { shots, hits, headshots, kills, ... }
//   game.ballistics.lastHitTime / .lastKillTime / .lastPlayerHitTime
//   game.ballistics.demoBurst(n)               debug: rip a burst down the camera axis
//
// Bus events EMITTED here
//   'impact'  {point, normal, material, surface, key, object, distance, source,
//              exit, ricochet, penetrated, energy, incidence, dir}
//   'hit'     {enemy, point, damage, headshot, part, distance, source}
//   'kill'    {enemy, headshot, distance, point, source}
//   'damage'  {amount, fromDir, from, angle, point, headshot, distance, shooter}
//   'health'  {hp, max}    — only when no other module owns player health
//   'whizz'   {point, dir, distance, speed}   — round passed close to the player
//   'ricochet'{point, normal, dir, surface, key, source}
//
// CONTRACT FOR THE ENEMIES MODULE
//   * Put enemies in game.registry.enemies.  Anything works: an Object3D, a plain
//     object with .position, or an object exposing .object3D/.root/.mesh/.model.
//   * Implement takeDamage(amount, info) to own the reaction; otherwise .hp /
//     .health is decremented here.  Do NOT emit 'kill' yourself — Ballistics owns
//     'hit' and 'kill' so each fires exactly once.
//   * Optional: .hitboxes[] ({part|name, object|box|center+radius, mul}), .height,
//     .crouched, .yaw, .accuracy (0..1), .alive/.dead, .muzzle (Object3D).
//   * Fire with game.ballistics.enemyFire({ shooter:this, origin, dir }).
//
// The 'impact' / 'hit' payloads allocate a couple of Vector3 each (~13/s at
// 780 rpm) on purpose: listeners are written by other agents and a pooled vector
// would be a landmine.  update() itself is strictly allocation-free.
// ============================================================================

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  Material behaviour table
 *    cost   penetration budget consumed per metre of this material
 *    hard   0..1 how badly a shallow round wants to skip off it
 *    ric    base ricochet probability at a fully grazing angle
 *    max    metres the reverse thickness probe will look through
 *    spall  extra energy shed as fragments on exit
 * ------------------------------------------------------------------ */
const SURF = {
  concrete:  { cost: 4.60, hard: 1.00, ric: 0.30, max: 0.90, spall: 0.16 },
  plaster:   { cost: 1.85, hard: 0.35, ric: 0.06, max: 1.10, spall: 0.08 },
  brick:     { cost: 3.90, hard: 0.85, ric: 0.22, max: 0.95, spall: 0.14 },
  sand:      { cost: 5.60, hard: 0.15, ric: 0.05, max: 0.70, spall: 0.05 },
  asphalt:   { cost: 5.00, hard: 0.80, ric: 0.26, max: 0.80, spall: 0.12 },
  metal:     { cost: 7.00, hard: 1.00, ric: 0.44, max: 0.35, spall: 0.20 },
  rustmetal: { cost: 4.30, hard: 0.90, ric: 0.34, max: 0.40, spall: 0.18 },
  gunmetal:  { cost: 7.60, hard: 1.00, ric: 0.46, max: 0.30, spall: 0.20 },
  wood:      { cost: 1.15, hard: 0.25, ric: 0.05, max: 1.20, spall: 0.10 },
  fabric:    { cost: 0.30, hard: 0.00, ric: 0.00, max: 1.60, spall: 0.02 },
  cloth_tan: { cost: 0.30, hard: 0.00, ric: 0.00, max: 1.60, spall: 0.02 },
  rubber:    { cost: 2.20, hard: 0.20, ric: 0.08, max: 0.80, spall: 0.06 },
  glass:     { cost: 0.45, hard: 0.90, ric: 0.08, max: 0.40, spall: 0.22 },
  polymer:   { cost: 1.60, hard: 0.40, ric: 0.12, max: 0.60, spall: 0.08 },
  flesh:     { cost: 1.50, hard: 0.00, ric: 0.00, max: 0.60, spall: 0.05 },
  foliage:   { cost: 0.18, hard: 0.00, ric: 0.00, max: 2.20, spall: 0.01 },
};

// fine surface key -> the six canonical Bus 'impact' buckets
const BUCKET = {
  concrete: 'concrete', plaster: 'concrete', brick: 'concrete', asphalt: 'concrete',
  fabric: 'concrete', cloth_tan: 'concrete', rubber: 'concrete', polymer: 'concrete',
  sand: 'sand', metal: 'metal', rustmetal: 'metal', gunmetal: 'metal',
  wood: 'wood', foliage: 'wood', glass: 'glass', flesh: 'flesh',
};

// loose names other agents might put on userData.surface / material.name
const KEYALIAS = {
  stone: 'concrete', cement: 'concrete', rock: 'concrete', kerb: 'concrete', curb: 'concrete',
  wall: 'plaster', stucco: 'plaster', paint: 'plaster', painted: 'plaster', tile: 'plaster',
  ceramic: 'plaster', ceiling: 'plaster', masonry: 'brick', clay: 'brick',
  dirt: 'sand', ground: 'sand', gravel: 'sand', dust: 'sand', soil: 'sand', terrain: 'sand',
  road: 'asphalt', tarmac: 'asphalt', street: 'asphalt',
  steel: 'metal', iron: 'metal', aluminum: 'metal', aluminium: 'metal', chrome: 'metal',
  rust: 'rustmetal', corrugated: 'rustmetal', container: 'rustmetal', barrel: 'rustmetal',
  plank: 'wood', crate: 'wood', timber: 'wood', plywood: 'wood', door: 'wood',
  cloth: 'cloth_tan', canvas: 'fabric', tarp: 'fabric', carpet: 'fabric', burlap: 'fabric',
  tan: 'cloth_tan', nylon: 'cloth_tan', webbing: 'cloth_tan',
  tyre: 'rubber', tire: 'rubber', mat: 'rubber',
  window: 'glass', mirror: 'glass', gun: 'gunmetal', weapon: 'gunmetal',
  plastic: 'polymer', abs: 'polymer', skin: 'flesh', body: 'flesh', head: 'flesh',
  leaf: 'foliage', leaves: 'foliage', grass: 'foliage', palm: 'foliage', bush: 'foliage',
};

/* ------------------------------------------------------------------ *
 *  Humanoid hitbox proxy, in FRACTIONS OF BODY HEIGHT.
 *  Local frame: +x right, +y up from the feet, +z forward.
 *  mul === -1 means "resolve against Config.weapon.headMul" so the
 *  designer-facing tunable stays the single source of truth.
 * ------------------------------------------------------------------ */
const RIG_PARTS = [
  { n: 'head',   k: 1, x: 0.000, y: 0.9075, z: 0.010, r: 0.0765, mul: -1 },
  { n: 'neck',   k: 0, x0: -0.046, y0: 0.838, z0: -0.042, x1: 0.046, y1: 0.880, z1: 0.042, mul: 1.45 },
  { n: 'chest',  k: 0, x0: -0.126, y0: 0.640, z0: -0.079, x1: 0.126, y1: 0.845, z1: 0.081, mul: 1.00 },
  { n: 'torso',  k: 0, x0: -0.109, y0: 0.508, z0: -0.072, x1: 0.109, y1: 0.640, z1: 0.074, mul: 1.00 },
  { n: 'pelvis', k: 0, x0: -0.104, y0: 0.398, z0: -0.068, x1: 0.104, y1: 0.508, z1: 0.070, mul: 1.00 },
  { n: 'armL',   k: 0, x0: -0.203, y0: 0.560, z0: -0.056, x1: -0.126, y1: 0.846, z1: 0.060, mul: 0.75 },
  { n: 'armR',   k: 0, x0: 0.126, y0: 0.560, z0: -0.056, x1: 0.203, y1: 0.846, z1: 0.060, mul: 0.75 },
  { n: 'legL',   k: 0, x0: -0.108, y0: 0.008, z0: -0.062, x1: -0.012, y1: 0.400, z1: 0.068, mul: 0.75 },
  { n: 'legR',   k: 0, x0: 0.012, y0: 0.008, z0: -0.062, x1: 0.108, y1: 0.400, z1: 0.068, mul: 0.75 },
];

const HEAD_NAMES = /head|skull|helmet|cranium/i;
const ARM_NAMES = /arm|hand|elbow|forearm|shoulderpad/i;
const LEG_NAMES = /leg|foot|feet|shin|thigh|calf|knee/i;
const TORSO_NAMES = /chest|torso|body|spine|stomach|pelvis|hip|abdomen/i;
const SKIP_NAMES = /viewmodel|view_model|tracer|decal|particle|muzzle|godray|god_ray|helper|hud|debug|crosshair|gizmo|impact|flash|smoke|dustfx|__/i;

/* ------------------------------------------------------------------ *
 *  Tuning with no Config key of its own.
 * ------------------------------------------------------------------ */
const T = {
  // --- spread: cone HALF-angle, radians ---
  adsBase: 0.0022,          // ~0.13 deg — rifle at the shoulder
  hipBase: 0.0270,          // ~1.55 deg — hip fire, standing still
  bloomPerShot: 0.0036,
  bloomMax: 0.0260,
  bloomDecay: 0.052,        // rad/s
  bloomHold: 0.085,         // s before recovery begins
  burstGrow: 0.0110,        // extra cone per second of sustained fire
  burstGrowMax: 0.0140,
  burstReset: 0.34,         // gap that ends a burst
  moveScale: 0.0210,        // penalty at full walk speed
  sprintExtra: 1.70,
  crouchMul: 0.72,
  airMul: 2.35,
  adsBloomMul: 0.16,        // at the shoulder, recoil (Weapon's job) dominates
  adsBurstMul: 0.28,
  moveAdsMul: 0.30,
  coneMax: 0.16,

  // --- ballistics ---
  falloffStart: 0.20,       // fraction of range still doing full damage
  falloffFloor: 0.42,       // damage multiplier at maximum range
  penBudget: 1.00,
  penDamageFloor: 0.22,
  maxSegments: 6,
  ricAngleCos: 0.315,       // |cos(angle to normal)| below which a skip is possible
  ricSpread: 0.115,
  ricDamage: 0.52,
  ricRange: 0.55,
  bodyPassCost: 0.55,

  // --- tracers ---
  // Only a minority of rounds are tracered; a belt where every other round
  // glows reads as a laser show, not as rifle fire.
  tracerEvery: 5,
  tracerEveryEnemy: 4,
  tracerMinDist: 3.0,
  tracerSpeed: 470,
  tracerSpeedEnemy: 330,
  tracerLen: 5.6,           // short dart, not a rope
  tracerLenEnemy: 7.0,
  tracerWidth: 0.062,
  tracerWidthEnemy: 0.074,
  // Lifetimes are measured on the WALL clock, never on the simulation clock —
  // see _updateTracers for why that distinction is the whole ballgame.
  tracerRealLife: 0.34,     // real seconds: hard release, whatever the sim thinks
  tracerStaleFrame: 0.10,   // a frame slower than this cannot show a live tracer

  // --- near miss ---
  whizzRadius: 2.30,
  whizzMinDist: 6.0,

  // --- enemy fire ---
  enemyDamage: 14.0,
  enemyHeadMul: 2.00,
  enemyLimbMul: 0.80,
  enemyFalloffFloor: 0.45,
  enemySpreadBase: 0.0125,
  enemySpreadSlop: 0.0300,
  enemyGraceShots: 2,       // opening rounds of a burst are deliberately wide
  enemyGraceCone: 0.055,

  // --- player health, used only when nobody else owns it ---
  regenDelay: 4.6,
  regenRate: 24.0,
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);

/* ================================================================== *
 *  Ballistics
 * ================================================================== */

export class Ballistics {

  constructor(game) {
    this.g = game;
    this.cfg = game?.config ?? {};

    this.t = 0;
    this.dt = 1 / 60;
    this._vh = (typeof innerHeight === 'number' ? innerHeight : 1080);

    // ---- spread state -------------------------------------------------
    this._bloom = 0;
    this._lastShot = -10;
    this._burstStart = -10;
    this._adsEvt = false;
    this._adsBlend = 0;
    this.spread = T.hipBase;
    this.spreadDeg = this.spread * 180 / Math.PI;

    // ---- deterministic PRNG (no Math.random jitter, reproducible runs) --
    this._seed = 0x9e3779b9 | 0;

    // ---- counters -------------------------------------------------------
    this._round = 0;
    this._roundEnemy = 0;
    this.stats = {
      shots: 0, enemyShots: 0, hits: 0, headshots: 0, kills: 0,
      pens: 0, ricochets: 0, tracers: 0, impacts: 0, playerHits: 0,
    };
    this.lastHitTime = -10;
    this.lastKillTime = -10;
    this.lastPlayerHitTime = -10;

    // ---- player health (claimed lazily, see _hurtPlayer) ----------------
    this.playerMaxHp = 100;
    this.playerHp = 100;
    this.ownsPlayerHealth = null;
    this._regenAt = -1;
    this._healthPay = { hp: 100, max: 100 };

    // ---- scratch: strict discipline, one set per call path --------------
    // The player and AI fire paths get separate vectors: an enemy's takeDamage()
    // may return fire synchronously from inside a player shot, and the two must
    // not be able to overwrite each other's origin/direction mid-burst.
    this._org = new THREE.Vector3();
    this._dir = new THREE.Vector3();
    this._cone = new THREE.Vector3();
    this._muz = new THREE.Vector3();
    this._eOrg = new THREE.Vector3();
    this._eDir = new THREE.Vector3();
    this._eCone = new THREE.Vector3();
    this._eMuz = new THREE.Vector3();
    this._sideA = new THREE.Vector3();
    this._sideB = new THREE.Vector3();
    this._camPos = new THREE.Vector3();
    this._camFwd = new THREE.Vector3();
    this._camRight = new THREE.Vector3();
    this._prevCam = new THREE.Vector3();
    this._prevCamValid = false;
    this._estSpeed = 0;
    this._up = new THREE.Vector3(0, 1, 0);
    this._pbO = new THREE.Vector3();
    this._pbD = new THREE.Vector3();
    this._mtO = new THREE.Vector3();
    this._mtD = new THREE.Vector3();
    this._aimT = new THREE.Vector3();
    this._rgP = new THREE.Vector3();
    this._rgQ = new THREE.Vector3();
    this._plr = new THREE.Vector3();
    this._hpF = new THREE.Vector3();
    this._nm = new THREE.Matrix3();
    this._box = new THREE.Box3();
    this._physOut = {};
    this._exitHit = null;

    // per-recursion-depth trace scratch, so a nested segment can never
    // clobber the segment that spawned it
    this._st = [];
    for (let i = 0; i <= T.maxSegments + 1; i++) {
      this._st.push({
        o: new THREE.Vector3(), d: new THREE.Vector3(),
        a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(),
      });
    }

    // ---- hit-record ring -------------------------------------------------
    this._hits = [];
    for (let i = 0; i < 32; i++) this._hits.push(this._mkHit());
    this._hitIdx = 0;

    // ---- world raycast fallback -------------------------------------------
    this._rc = new THREE.Raycaster();
    this._rc.near = 0;
    this._rc.far = 1000;
    if (this._rc.params) {
      if (this._rc.params.Points) this._rc.params.Points.threshold = 0;
      if (this._rc.params.Line) this._rc.params.Line.threshold = 0;
    }
    this._rcHits = [];
    this._castList = [];
    this._castStamp = -1e9;
    this._castChildren = -1;
    this._physMode = -2;           // -2 unprobed, -1 unusable, >=0 signature index
    this._physName = null;
    this._physProbes = 0;
    this._physDisagree = 0;
    this._physNextProbe = 0;
    this._ray = null;

    // ---- enemy rigs --------------------------------------------------------
    this._rigs = new WeakMap();
    this._hpFallback = new WeakMap();
    this._hitboxOverride = new WeakMap();
    this._aiBurst = new WeakMap();
    this._enemyRoots = new Set();      // Object3Ds that belong to an enemy
    this._enemyCount = -1;
    this._enemyStamp = -1e9;

    // ---- tracer pool -------------------------------------------------------
    this._tN = 96;
    this._rtPrev = -1;             // wall-clock seconds at the last tracer update
    this._rdt = 1 / 60;
    this._allocTracers(this._tN);
    this._mesh = null;
    this._geo = null;

    // ---- whizz queue --------------------------------------------------------
    this._wN = 24;
    this._wT = new Float32Array(this._wN).fill(-1);
    this._wX = new Float32Array(this._wN);
    this._wY = new Float32Array(this._wN);
    this._wZ = new Float32Array(this._wN);
    this._wDist = new Float32Array(this._wN);
    this._wS = new Float32Array(this._wN);
    this._wDX = new Float32Array(this._wN);
    this._wDY = new Float32Array(this._wN);
    this._wDZ = new Float32Array(this._wN);
    this._wCur = 0;
    this._wPay = [];
    for (let i = 0; i < 8; i++) {
      this._wPay.push({ point: new THREE.Vector3(), dir: new THREE.Vector3(), distance: 0, speed: 0 });
    }
    this._wPayIdx = 0;

    // ---- bus wiring ----------------------------------------------------------
    this._unsub = [];
    const bus = game?.bus;
    if (bus) {
      this._unsub.push(bus.on('shot', d => this._onShot(d)));
      this._unsub.push(bus.on('ads', d => { this._adsEvt = !!(d && d.active); }));
    }
  }

  /* ---------------------------------------------------------------- *
   *  Lifecycle
   * ---------------------------------------------------------------- */

  async init() {
    this._buildTracerBatch();
  }

  resize(w, h) { if (h) this._vh = h; }

  update(dt, t) {
    this.t = t;
    this.dt = dt;
    this._trackPlayer(dt);
    this._updateSpread(dt);
    this._updateTracers(dt);
    this._updateWhizz(t);
    this._updateHealth(dt);
  }

  dispose() {
    for (let i = 0; i < this._unsub.length; i++) {
      try { this._unsub[i](); } catch (e) { /* listener already gone */ }
    }
    this._unsub.length = 0;
    if (this._mesh) {
      this._mesh.parent?.remove(this._mesh);
      this._mesh.geometry?.dispose?.();
      this._mesh.material?.dispose?.();
      this._mesh = null;
    }
  }

  /* ================================================================ *
   *  RNG
   * ================================================================ */

  _rand() {
    // xorshift32 — deterministic, allocation-free, no Math.random cost spikes
    let x = this._seed | 0;
    x ^= x << 13; x |= 0;
    x ^= x >>> 17;
    x ^= x << 5; x |= 0;
    this._seed = x;
    return (x >>> 0) / 4294967296;
  }

  /* ================================================================ *
   *  PLAYER STATE PROBES (Controller / Weapon belong to other agents)
   * ================================================================ */

  _trackPlayer(dt) {
    const cam = this.g?.camera;
    if (!cam) return;
    cam.getWorldPosition(this._camPos);
    if (this._prevCamValid && dt > 1e-5) {
      const dx = this._camPos.x - this._prevCam.x;
      const dy = this._camPos.y - this._prevCam.y;
      const dz = this._camPos.z - this._prevCam.z;
      const inst = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt;
      // a debug teleport must not read as a 400 m/s sprint
      this._estSpeed = inst > 40 ? 0 : lerp(this._estSpeed, inst, clamp01(dt * 9));
    }
    this._prevCam.copy(this._camPos);
    this._prevCamValid = true;
  }

  get ads() {
    const w = this.g?.weapon;
    if (w) {
      if (typeof w.ads === 'boolean') return w.ads || this._adsEvt;
      if (typeof w.isAds === 'boolean') return w.isAds || this._adsEvt;
      if (typeof w.aiming === 'boolean') return w.aiming || this._adsEvt;
      if (typeof w.adsActive === 'boolean') return w.adsActive || this._adsEvt;
      if (typeof w.adsAmount === 'number') return w.adsAmount > 0.55 || this._adsEvt;
    }
    return this._adsEvt;
  }

  get crouched() {
    const c = this.g?.controller;
    if (!c) return false;
    if (typeof c.crouched === 'boolean') return c.crouched;
    if (typeof c.isCrouched === 'boolean') return c.isCrouched;
    if (typeof c.crouching === 'boolean') return c.crouching;
    if (typeof c.crouch === 'boolean') return c.crouch;
    if (typeof c.stance === 'string') return c.stance === 'crouch' || c.stance === 'crouched';
    if (typeof c.crouchAmount === 'number') return c.crouchAmount > 0.5;
    return false;
  }

  get grounded() {
    const c = this.g?.controller;
    if (!c) return true;
    if (typeof c.onGround === 'boolean') return c.onGround;
    if (typeof c.grounded === 'boolean') return c.grounded;
    if (typeof c.isGrounded === 'boolean') return c.isGrounded;
    if (typeof c.inAir === 'boolean') return !c.inAir;
    return true;
  }

  get sprinting() {
    const c = this.g?.controller;
    if (c) {
      if (typeof c.sprinting === 'boolean') return c.sprinting;
      if (typeof c.isSprinting === 'boolean') return c.isSprinting;
      if (typeof c.sprint === 'boolean') return c.sprint;
    }
    return this._playerSpeed() > (this.cfg?.player?.walkSpeed ?? 3.4) * 1.25;
  }

  _playerSpeed() {
    const c = this.g?.controller;
    if (c) {
      if (typeof c.speed === 'number' && isFinite(c.speed)) return c.speed;
      const v = c.velocity ?? c.vel;
      if (v && typeof v.x === 'number') return Math.sqrt(v.x * v.x + v.z * v.z);
    }
    return this._estSpeed;
  }

  /* ================================================================ *
   *  SPREAD MODEL
   * ================================================================ */

  _updateSpread(dt) {
    const ads = this.ads;
    this._adsBlend = lerp(this._adsBlend, ads ? 1 : 0, clamp01(dt * 12));

    if (this.t - this._lastShot > T.bloomHold && this._bloom > 0) {
      const walk = this.cfg?.player?.walkSpeed ?? 3.4;
      const moveF = clamp01(this._playerSpeed() / Math.max(0.5, walk));
      let rate = T.bloomDecay * (ads ? 2.1 : 1.0) * (this.crouched ? 1.25 : 1.0);
      rate *= lerp(1.35, 0.62, moveF);         // settling is what recovers accuracy
      this._bloom = Math.max(0, this._bloom - rate * dt);
    }
    if (this.t - this._lastShot > T.burstReset) this._burstStart = this.t;

    this.spread = this._computeSpread();
    this.spreadDeg = this.spread * 180 / Math.PI;
  }

  _computeSpread() {
    const ads = this._adsBlend;
    const walk = this.cfg?.player?.walkSpeed ?? 3.4;
    const moveF = clamp01(this._playerSpeed() / Math.max(0.5, walk));

    let cone = lerp(T.hipBase, T.adsBase, smooth(ads));

    // movement barely matters at the shoulder, dominates from the hip
    cone += moveF * T.moveScale * lerp(1.0, T.moveAdsMul, ads);
    if (this.sprinting) cone *= lerp(T.sprintExtra, 1.15, ads);

    // sustained fire
    if (this.t - this._lastShot < T.burstReset) {
      const burst = clamp(this.t - this._burstStart, 0, 2.0);
      cone += Math.min(burst * T.burstGrow, T.burstGrowMax) * lerp(1.0, T.adsBurstMul, ads);
    }

    // per-shot bloom
    cone += this._bloom * lerp(1.0, T.adsBloomMul, ads);

    // stance
    if (this.crouched) cone *= lerp(T.crouchMul, 0.86, ads * 0.5);
    if (!this.grounded) cone *= T.airMul;

    return clamp(cone, T.adsBase * 0.75, T.coneMax);
  }

  getSpread() { return this.spread; }

  /** Half-gap for a spread-driven crosshair, in pixels, at viewport height h. */
  getCrosshairPixels(h) {
    const H = h || this._vh || 1080;
    const cam = this.g?.camera;
    const fov = (cam?.fov ?? this.cfg?.weapon?.fovHip ?? 68) * Math.PI / 180;
    const focal = (H * 0.5) / Math.tan(fov * 0.5);
    return Math.tan(this.spread) * focal;
  }

  /* ================================================================ *
   *  SHOT ENTRY POINTS
   * ================================================================ */

  _onShot(d) {
    if (!d) { this.fire(null); return; }
    const w = (d.weapon && typeof d.weapon === 'object') ? d.weapon : null;
    const src = d.source ?? d.faction ?? d.team ?? w?.faction ?? w?.source ?? null;
    const enemy = (src === 'enemy' || src === 'ai' || src === 'hostile') ||
      d.enemy === true || this._isEnemyEntity(d.shooter);
    if (enemy) this.enemyFire(d);
    else this.fire(d);
  }

  _isEnemyEntity(x) {
    if (!x || typeof x !== 'object') return false;
    const list = this.g?.registry?.enemies;
    if (!list) return false;
    for (let i = 0; i < list.length; i++) {
      const raw = list[i];
      if (raw === x) return true;
      if (raw && typeof raw === 'object' && this._entityOf(raw) === x) return true;
    }
    return false;
  }

  /**
   * Fire one round (or `pellets` rounds).  Every option has a sane default —
   * called with nothing it fires straight down the camera axis.
   */
  fire(opts) {
    const o = opts || 0;
    const cam = this.g?.camera;
    if (!cam) return;

    const org = this._org;
    if (o.origin && typeof o.origin.x === 'number') org.copy(o.origin);
    else cam.getWorldPosition(org);

    const dir = this._dir;
    if (o.dir && typeof o.dir.x === 'number') dir.copy(o.dir);
    else cam.getWorldDirection(dir);
    if (!isFinite(dir.x) || dir.lengthSq() < 1e-8) dir.set(0, 0, -1);
    dir.normalize();

    const cone = (typeof o.spread === 'number') ? o.spread : this.spread;
    const pellets = Math.max(1, (o.pellets | 0) || 1);
    const range = o.range ?? this.cfg?.weapon?.range ?? 220;
    const dmg = o.damage ?? this.cfg?.weapon?.damage ?? 34;
    const pen = (o.penetration ?? 1) * T.penBudget;

    // the muzzle is cosmetic: rounds are traced from the eye, drawn from the barrel
    const muzzle = this._muzzle(this._muz);

    for (let p = 0; p < pellets; p++) {
      this.stats.shots++;
      this._round++;
      this._coneDir(dir, cone, this._cone);
      const tracer = o.tracer === true ||
        (o.tracer !== false && (this._round % T.tracerEvery) === 0);
      this._trace(org, this._cone, range, dmg, pen, 'player', o.shooter ?? null,
        tracer, muzzle, 0, null, 0);
    }

    // bloom is per trigger pull, not per pellet
    this._bloom = Math.min(T.bloomMax, this._bloom + T.bloomPerShot);
    if (this.t - this._lastShot > T.burstReset) this._burstStart = this.t;
    this._lastShot = this.t;
  }

  /**
   * AI return fire — the same trace pipeline with an accuracy model and the
   * player as the valid target.
   */
  enemyFire(opts) {
    const o = opts || 0;
    const shooter = o.shooter ?? o.enemy ?? o.from ?? null;

    const org = this._eOrg;
    if (o.origin && typeof o.origin.x === 'number') org.copy(o.origin);
    else if (!this._entityMuzzle(shooter, org)) return;

    const dir = this._eDir;
    if (o.dir && typeof o.dir.x === 'number') dir.copy(o.dir);
    else if (o.target && typeof o.target.x === 'number') dir.copy(o.target).sub(org);
    else if (!this._aimAtPlayer(org, dir)) return;
    if (!isFinite(dir.x) || dir.lengthSq() < 1e-8) return;
    dir.normalize();

    this.stats.enemyShots++;
    this._roundEnemy++;

    const acc = clamp01(o.accuracy ?? shooter?.accuracy ?? 0.55);
    const sight = this.cfg?.ai?.sightRange ?? 90;
    const range = o.range ?? Math.max(60, sight * 1.5);
    let cone = (typeof o.spread === 'number')
      ? o.spread
      : T.enemySpreadBase + (1 - acc) * T.enemySpreadSlop;

    // Burst grace: the opening rounds are deliberately wide so the player gets
    // a directional cue and a reaction window before the damage starts landing.
    if (shooter && typeof shooter === 'object') {
      let b = this._aiBurst.get(shooter);
      if (!b) { b = { n: 0, t: -10 }; this._aiBurst.set(shooter, b); }
      if (this.t - b.t > 0.85) b.n = 0;
      b.t = this.t;
      if (b.n < T.enemyGraceShots) cone += T.enemyGraceCone * (1 - b.n / T.enemyGraceShots);
      b.n++;
    }

    this._coneDir(dir, cone, this._eCone);

    const dmg = o.damage ?? (typeof shooter?.damage === 'number' ? shooter.damage : T.enemyDamage);
    const pen = (o.penetration ?? 0.85) * T.penBudget;
    const tracer = o.tracer === true ||
      (o.tracer !== false && (this._roundEnemy % T.tracerEveryEnemy) === 0);

    this._eMuz.copy(org);
    this._trace(org, this._eCone, range, dmg, pen, 'enemy', shooter, tracer, this._eMuz, 0, null, 0);
  }

  /** Debug: rip a fanned burst down the camera axis so tracers are visible. */
  demoBurst(n = 8, spacing = 0.055) {
    const cam = this.g?.camera;
    if (!cam) return;
    const count = Math.max(1, n | 0);
    const S = this._st[T.maxSegments + 1];
    for (let i = 0; i < count; i++) {
      cam.getWorldDirection(S.a);
      const yaw = (i - (count - 1) * 0.5) * spacing;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const x = S.a.x * cy + S.a.z * sy;
      const z = -S.a.x * sy + S.a.z * cy;
      S.a.set(x, S.a.y + (i % 3 - 1) * 0.01, z).normalize();
      this.fire({ dir: S.a, tracer: true });
    }
  }

  /* ================================================================ *
   *  CONE SAMPLING
   * ================================================================ */

  /** out = dir perturbed inside a cone of half-angle `cone` (radians). */
  _coneDir(dir, cone, out) {
    out.copy(dir);
    if (!(cone > 1e-6)) return out;

    const a = this._sideA, b = this._sideB;
    if (Math.abs(dir.y) < 0.94) a.set(0, 1, 0); else a.set(1, 0, 0);
    a.crossVectors(dir, a).normalize();
    b.crossVectors(dir, a).normalize();

    // area-weighted disc, pulled slightly toward the centre so most rounds
    // still go where the sights are pointing
    const r = Math.pow(this._rand(), 0.62) * cone;
    const th = this._rand() * Math.PI * 2;
    out.addScaledVector(a, Math.tan(Math.cos(th) * r));
    out.addScaledVector(b, Math.tan(Math.sin(th) * r));
    return out.normalize();
  }

  /* ================================================================ *
   *  THE TRACE
   * ================================================================ */

  /**
   * One bullet segment.  Recurses (bounded by T.maxSegments) through
   * penetrations, body pass-throughs and ricochets.  Impacts, damage and
   * tracers are all emitted from here.
   */
  _trace(origin, dir, range, damage, pen, source, shooter, wantTracer, tracerFrom, depth, ignore, travelled) {
    if (depth >= T.maxSegments || range <= 0.05 || damage <= 0.5) return;
    const flown = travelled || 0;

    // freeze this segment's ray into depth-local storage
    const S = this._st[depth];
    S.o.copy(origin);
    S.d.copy(dir);
    const ox = S.o.x, oy = S.o.y, oz = S.o.z;
    const dx = S.d.x, dy = S.d.y, dz = S.d.z;

    const isPlayerShot = source === 'player';
    const world = this._raycastWorld(S.o, S.d, range, null);
    const ch = isPlayerShot
      ? this._raycastEnemies(S.o, S.d, range, ignore)
      : this._raycastPlayer(S.o, S.d, range);

    const wT = world ? world.t : Infinity;
    const cT = ch ? ch.t : Infinity;

    // ---------------------------------------------------------------
    // clean miss — the round flies out to maximum range
    // ---------------------------------------------------------------
    if (!world && !ch) {
      if (!isPlayerShot) this._scheduleWhizz(S.o, S.d, range, T.tracerSpeedEnemy);
      if (wantTracer) {
        S.a.set(ox + dx * range, oy + dy * range, oz + dz * range);
        this._spawnTracer(tracerFrom, S.a, source, depth);
      }
      return;
    }

    // ---------------------------------------------------------------
    // a body is in the way
    // ---------------------------------------------------------------
    if (ch && cT <= wT) {
      const dist = this._segDistance(tracerFrom, ch.point, depth, cT, flown);
      if (wantTracer) this._spawnTracer(tracerFrom, ch.point, source, depth);

      if (isPlayerShot) this._hitEnemy(ch, S.d, damage, dist, source, depth);
      else this._hurtPlayer(ch, S.d, damage, dist, shooter, S.o);

      this._emitImpact(ch, 'flesh', dist, source, false, false,
        clamp01(damage / 40), S.d, depth);

      // over-penetration: rifle rounds go through people
      const left = pen - T.bodyPassCost;
      if (left > 0.06 && isPlayerShot) {
        S.a.set(ch.point.x + dx * 0.35, ch.point.y + dy * 0.35, ch.point.z + dz * 0.35);
        S.c.copy(ch.point);
        this._trace(S.a, S.d, range - cT - 0.35, damage * 0.55, left,
          source, shooter, false, S.c, depth + 1, ch.entity ?? ch.enemy, dist + 0.35);
      }
      return;
    }

    // ---------------------------------------------------------------
    // world geometry
    // ---------------------------------------------------------------
    if (!isPlayerShot) this._scheduleWhizz(S.o, S.d, wT, T.tracerSpeedEnemy);
    if (wantTracer) this._spawnTracer(tracerFrom, world.point, source, depth);

    const dist = this._segDistance(tracerFrom, world.point, depth, wT, flown);
    const surf = SURF[world.key] || SURF.concrete;
    const ud = world.object ? world.object.userData : null;

    // incidence: 1 = square on, 0 = perfectly grazing
    const inc = clamp01(-(dx * world.nx + dy * world.ny + dz * world.nz));
    const energy = clamp01(damage / Math.max(1, this.cfg?.weapon?.damage ?? 34));

    // ---- ricochet ---------------------------------------------------
    if (!ud?.noRicochet && inc < T.ricAngleCos && surf.ric > 0 && depth < 2) {
      const graze = 1 - inc / T.ricAngleCos;
      if (this._rand() < surf.ric * (0.35 + 0.65 * graze) * surf.hard) {
        this.stats.ricochets++;
        this._emitImpact(world, world.key, dist, source, false, true, energy, S.d, depth);
        this.g?.bus?.emit('ricochet', {
          point: world.point.clone(), normal: world.normal.clone(),
          dir: new THREE.Vector3(dx, dy, dz),
          surface: world.surface, key: world.key, source,
        });

        S.a.copy(S.d).reflect(world.normal);
        this._coneDir(S.a, T.ricSpread * (1.2 - graze * 0.6), S.b);
        S.c.copy(world.point);
        S.a.copy(world.point).addScaledVector(world.normal, 0.02);
        // NEVER tracer the deflected leg. Two streaks meeting at the impact
        // point draw one bent polyline with a hard corner in it, which reads as
        // a debug gizmo rather than as gunfire. The skip itself is sold by the
        // 'ricochet' spark and its audio, not by a second glowing line.
        this._trace(S.a, S.b, (range - wT) * T.ricRange, damage * T.ricDamage,
          pen * 0.35, source, shooter, false, S.c, depth + 1, ignore, dist);
        return;
      }
    }

    // ---- penetration -------------------------------------------------
    if (pen > 0.02 && !ud?.noPenetration) {
      const maxT = Math.min(surf.max, typeof ud?.thickness === 'number' ? ud.thickness : 99);
      const thick = this._measureThickness(world, S.d, maxT);
      if (thick > 0) {
        // a shallow angle means a longer path through the same slab
        const path = thick / Math.max(0.22, inc);
        const cost = path * surf.cost * (typeof ud?.penCost === 'number' ? ud.penCost : 1);
        if (cost < pen) {
          this.stats.pens++;
          this._emitImpact(world, world.key, dist, source, false, false, energy, S.d, depth);

          const keep = clamp(Math.pow(1 - cost / pen, 0.75), T.penDamageFloor, 1);
          const exit = this._exitHit;
          this._emitImpact(exit, exit.key, dist + thick, source, true, false,
            energy * keep, S.d, depth);

          S.c.copy(exit.point);
          S.a.set(exit.point.x + dx * 0.01, exit.point.y + dy * 0.01, exit.point.z + dz * 0.01);
          this._trace(S.a, S.d, range - wT - thick,
            damage * keep * (1 - surf.spall * 0.5), pen - cost,
            source, shooter, false, S.c, depth + 1, ignore, dist + thick);
          return;
        }
      }
    }

    // ---- absorbed -----------------------------------------------------
    this._emitImpact(world, world.key, dist, source, false, false, energy, S.d, depth);
  }

  /** Total distance flown from the muzzle — what damage falloff is measured on. */
  _segDistance(from, to, depth, segT, flown) {
    if (depth === 0 && from) {
      const dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return flown + segT;
  }

  /**
   * Thickness of the slab we just entered.  Probes backwards from the far side
   * so a front-face-only raycaster still reports the exit plane.  On success
   * this._exitHit holds a full hit record for the exit face.
   */
  _measureThickness(entry, dir, maxT) {
    if (!(maxT > 0.002)) return 0;
    this._mtD.set(-dir.x, -dir.y, -dir.z);
    this._mtO.set(entry.point.x + dir.x * maxT,
      entry.point.y + dir.y * maxT,
      entry.point.z + dir.z * maxT);

    const h = this._raycastWorld(this._mtO, this._mtD, maxT * 1.001, null);
    if (!h) return 0;
    const thick = maxT - h.t;
    if (thick <= 0.002 || thick >= maxT * 0.998) return 0;
    this._exitHit = h;
    return thick;
  }

  /* ================================================================ *
   *  IMPACT / DAMAGE EMISSION
   * ================================================================ */

  _emitImpact(h, key, dist, source, exit, ricochet, energy, dir, depth) {
    const bus = this.g?.bus;
    if (!bus || !h) return;
    this.stats.impacts++;
    const k = key || h.key || 'concrete';
    bus.emit('impact', {
      point: h.point.clone(),
      normal: h.normal.clone(),
      material: h.material ?? null,
      surface: BUCKET[k] ?? 'concrete',
      key: k,
      object: h.object ?? null,
      distance: dist,
      source,
      exit: !!exit,
      ricochet: !!ricochet,
      penetrated: depth > 0,
      energy: clamp01(energy),
      // 1 = square on, 0 = grazing.  Absolute, because an exit face's normal
      // points along the round's travel rather than against it.
      incidence: clamp01(Math.abs(dir.x * h.nx + dir.y * h.ny + dir.z * h.nz)),
      dir: new THREE.Vector3(dir.x, dir.y, dir.z),
    });
  }

  _falloff(dist, range, floor) {
    const near = range * T.falloffStart;
    if (dist <= near) return 1;
    if (dist >= range) return floor;
    return lerp(1, floor, smooth((dist - near) / Math.max(1e-3, range - near)));
  }

  _hitEnemy(ch, dir, baseDamage, dist, source, depth) {
    const target = ch.entity ?? ch.enemy;
    if (!target) return;

    const range = this.cfg?.weapon?.range ?? 220;
    const headMul = this.cfg?.weapon?.headMul ?? 2.4;
    const mul = ch.mul < 0 ? headMul : ch.mul;
    const headshot = ch.part === 'head';
    const dmg = baseDamage * mul * this._falloff(dist, range, T.falloffFloor);

    const before = this._enemyHp(target);
    const wasAlive = this._enemyAlive(target);

    const info = {
      damage: dmg, amount: dmg,
      point: ch.point.clone(), normal: ch.normal.clone(),
      dir: new THREE.Vector3(dir.x, dir.y, dir.z),
      part: ch.part, headshot, distance: dist, source, penetrated: depth > 0,
    };
    this._applyEnemyDamage(target, dmg, info);

    this.stats.hits++;
    if (headshot) this.stats.headshots++;
    this.lastHitTime = this.t;

    const bus = this.g?.bus;
    bus?.emit('hit', {
      enemy: target, point: ch.point.clone(), damage: dmg, headshot,
      part: ch.part, distance: dist, source, penetrated: depth > 0,
      hpBefore: before, hpAfter: this._enemyHp(target),
    });

    if (wasAlive && !this._enemyAlive(target) && !target.__bsKilled) {
      target.__bsKilled = true;
      this.stats.kills++;
      this.lastKillTime = this.t;
      bus?.emit('kill', {
        enemy: target, headshot, distance: dist,
        point: ch.point.clone(), source,
      });
    }
  }

  _applyEnemyDamage(e, dmg, info) {
    if (!e || typeof e !== 'object') return;
    const fns = ['takeDamage', 'applyDamage', 'hurt', 'onHit', 'receiveDamage', 'damage'];
    for (let i = 0; i < fns.length; i++) {
      const f = e[fns[i]];
      if (typeof f === 'function') {
        try { f.call(e, dmg, info); } catch (err) { console.error('[ballistics] enemy takeDamage', err); }
        return;
      }
    }
    if (typeof e.hp === 'number') { e.hp -= dmg; return; }
    if (typeof e.health === 'number') { e.health -= dmg; return; }
    // the enemy tracks nothing at all — keep a shadow pool so kills still work
    const max = typeof e.maxHp === 'number' ? e.maxHp : 100;
    const cur = this._hpFallback.has(e) ? this._hpFallback.get(e) : max;
    const next = cur - dmg;
    this._hpFallback.set(e, next);
    if (next <= 0) { e.dead = true; e.alive = false; }
  }

  _enemyHp(e) {
    if (!e || typeof e !== 'object') return null;
    if (typeof e.hp === 'number') return e.hp;
    if (typeof e.health === 'number') return e.health;
    if (this._hpFallback.has(e)) return this._hpFallback.get(e);
    return null;
  }

  _enemyAlive(e) {
    if (!e || typeof e !== 'object') return false;
    if (e.dead === true || e.isDead === true || e.alive === false) return false;
    if (typeof e.state === 'string' && (e.state === 'dead' || e.state === 'dying')) return false;
    const hp = this._enemyHp(e);
    return !(hp !== null && hp <= 0);
  }

  /* ---- the player takes a round ---------------------------------- */

  _hurtPlayer(ch, dir, baseDamage, dist, shooter, origin) {
    const sight = this.cfg?.ai?.sightRange ?? 90;
    const mul = ch.mul < 0 ? T.enemyHeadMul : ch.mul;
    const amount = baseDamage * mul * this._falloff(dist, sight * 1.6, T.enemyFalloffFloor);
    const headshot = ch.part === 'head';

    this.stats.playerHits++;
    this.lastPlayerHitTime = this.t;

    // world direction the round came FROM (player -> shooter), for the HUD arc
    const from = new THREE.Vector3(origin.x, origin.y, origin.z);
    const fromDir = new THREE.Vector3(origin.x - this._plr.x, 0, origin.z - this._plr.z);
    if (fromDir.lengthSq() < 1e-6) fromDir.set(-dir.x, 0, -dir.z);
    if (fromDir.lengthSq() < 1e-6) fromDir.set(0, 0, -1);
    fromDir.normalize();

    // signed angle relative to where the player is looking: 0 ahead, + to the right
    let angle = 0;
    const cam = this.g?.camera;
    if (cam) {
      cam.getWorldDirection(this._hpF);
      this._hpF.y = 0;
      if (this._hpF.lengthSq() > 1e-6) {
        this._hpF.normalize();
        const dot = clamp(this._hpF.x * fromDir.x + this._hpF.z * fromDir.z, -1, 1);
        const cross = this._hpF.x * fromDir.z - this._hpF.z * fromDir.x;
        angle = Math.atan2(cross, dot);
      }
    }

    const bus = this.g?.bus;
    bus?.emit('damage', {
      amount, damage: amount, fromDir, from, angle,
      point: ch.point.clone(), headshot, part: ch.part,
      distance: dist, shooter: shooter ?? null, source: 'enemy',
    });

    // Only own player health if nobody else does.
    const c = this.g?.controller;
    if (this.ownsPlayerHealth === null) {
      this.ownsPlayerHealth = !(c && (typeof c.takeDamage === 'function' ||
        typeof c.applyDamage === 'function' ||
        typeof c.hp === 'number' || typeof c.health === 'number'));
    }
    if (c) {
      try {
        if (typeof c.takeDamage === 'function') c.takeDamage(amount, { fromDir, angle, shooter, headshot });
        else if (typeof c.applyDamage === 'function') c.applyDamage(amount, { fromDir, angle, shooter, headshot });
        else if (typeof c.hp === 'number') c.hp = Math.max(0, c.hp - amount);
        else if (typeof c.health === 'number') c.health = Math.max(0, c.health - amount);
      } catch (e) { console.error('[ballistics] player damage', e); }
    }

    if (this.ownsPlayerHealth) {
      this.playerHp = Math.max(0, this.playerHp - amount);
      this._regenAt = this.t + T.regenDelay;
      this._healthPay.hp = this.playerHp;
      this._healthPay.max = this.playerMaxHp;
      bus?.emit('health', this._healthPay);
    }
  }

  _updateHealth(dt) {
    if (!this.ownsPlayerHealth) return;
    if (this.playerHp <= 0 || this.playerHp >= this.playerMaxHp) return;
    if (this.t < this._regenAt) return;
    const before = this.playerHp;
    this.playerHp = Math.min(this.playerMaxHp, this.playerHp + T.regenRate * dt);
    if ((this.playerHp | 0) !== (before | 0)) {
      this._healthPay.hp = this.playerHp;
      this._healthPay.max = this.playerMaxHp;
      this.g?.bus?.emit('health', this._healthPay);
    }
  }

  /* ================================================================ *
   *  WORLD RAYCAST — physics first, hardened fallback second
   * ================================================================ */

  _mkHit() {
    return {
      t: 0, px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
      point: new THREE.Vector3(), normal: new THREE.Vector3(0, 1, 0),
      object: null, material: null, surface: 'concrete', key: 'concrete',
      enemy: null, entity: null, part: 'torso', mul: 1,
    };
  }

  _nextHit() {
    const h = this._hits[this._hitIdx];
    this._hitIdx = (this._hitIdx + 1) % this._hits.length;
    h.object = null; h.material = null; h.enemy = null; h.entity = null;
    h.surface = 'concrete'; h.key = 'concrete'; h.part = 'torso'; h.mul = 1;
    return h;
  }

  /**
   * Public world raycast.  The record comes from a ring buffer — copy what you
   * need synchronously.  `skip` excludes a single object.
   */
  raycast(origin, dir, maxDist, skip) {
    if (!origin || !dir) return null;
    return this._raycastWorld(origin, dir,
      maxDist ?? (this.cfg?.weapon?.range ?? 220), skip ?? null);
  }

  /** True when nothing solid sits between the two world points. */
  losClear(a, b) {
    if (!a || !b) return false;
    const S = this._st[T.maxSegments + 1];
    S.b.set(b.x - a.x, b.y - a.y, b.z - a.z);
    const d = S.b.length();
    if (d < 1e-4) return true;
    S.b.multiplyScalar(1 / d);
    S.a.copy(a);
    return !this._raycastWorld(S.a, S.b, d - 0.02, null);
  }

  _raycastWorld(origin, dir, maxDist, skip) {
    if (!(maxDist > 0)) return null;

    const phys = this.g?.physics;
    if (phys && this._physMode !== -1) {
      if (this._physMode === -2) this._probePhysics(phys);
      if (this._physMode >= 0) {
        const r = this._callPhysics(phys, origin, dir, maxDist, skip);
        if (r) return r;
        // A silently-wrong signature returns null forever. Cross-check a few
        // misses against the scene graph before trusting them.
        if (this._physDisagree < 3) {
          const f = this._fallbackRay(origin, dir, maxDist, skip);
          if (f) {
            this._physDisagree++;
            if (this._physDisagree >= 3) {
              this._physMode = -1;
              // info, not warn: a screenshot build treats warnings as failures and
              // this is a successful recovery, not a fault
              console.info('[ballistics] physics raycast disagreed with the scene 3x — using the internal raycaster');
            }
            return f;
          }
        }
        return null;
      }
    }
    return this._fallbackRay(origin, dir, maxDist, skip);
  }

  _probePhysics(phys) {
    if (this.t < this._physNextProbe) return;
    this._physNextProbe = this.t + 2.0;
    this._physProbes++;

    const name = ['raycast', 'raycastClosest', 'rayTest', 'castRay', 'intersectRay']
      .find(n => typeof phys[n] === 'function');
    if (!name) {
      if (this._physProbes > 4) this._physMode = -1;
      return;
    }
    this._physName = name;

    const cam = this.g?.camera;
    if (cam) cam.getWorldPosition(this._pbO); else this._pbO.set(0, 3, 0);
    this._pbD.set(0, -1, 0);

    for (let mode = 0; mode < 5; mode++) {
      this._physMode = mode;
      let r = null;
      try { r = this._callPhysics(phys, this._pbO, this._pbD, 60, null); } catch (e) { continue; }
      if (r) return;                       // this signature produced a real hit
    }
    // Nothing hit — the level may simply be empty under the camera. Keep the
    // most likely signature and let the disagreement counter sort it out.
    this._physMode = 0;
  }

  _callPhysics(phys, origin, dir, maxDist, skip) {
    const fn = phys[this._physName];
    if (typeof fn !== 'function') { this._physMode = -1; return null; }
    let raw = null;
    try {
      switch (this._physMode) {
        case 0:
          raw = fn.call(phys, origin, dir, maxDist);
          break;
        case 1: {
          const o = this._physOut;
          o.origin = origin; o.dir = dir; o.direction = dir;
          o.maxDistance = maxDist; o.far = maxDist; o.distance = maxDist;
          o.skip = skip; o.exclude = skip;
          raw = fn.call(phys, o);
          break;
        }
        case 2:
          raw = fn.call(phys, origin, dir, maxDist, { skip, exclude: skip });
          break;
        case 3: {
          if (!this._ray) this._ray = new THREE.Ray();
          this._ray.origin.copy(origin);
          this._ray.direction.copy(dir);
          raw = fn.call(phys, this._ray, maxDist);
          break;
        }
        case 4: {
          const o = this._physOut;
          raw = fn.call(phys, origin, dir, maxDist, o);
          if (raw === true) raw = o;
          break;
        }
      }
    } catch (e) {
      return null;
    }
    return this._normalizePhysHit(raw, origin, dir, maxDist, skip);
  }

  _normalizePhysHit(raw, origin, dir, maxDist, skip) {
    if (!raw || typeof raw !== 'object') return null;
    if (raw.hit === false || raw.hasHit === false) return null;

    const p = raw.point ?? raw.position ?? raw.hitPoint ?? raw.p;
    let t = raw.distance ?? raw.dist ?? raw.t ?? raw.toi;
    if (typeof t !== 'number' || !isFinite(t)) {
      if (!p || typeof p.x !== 'number') return null;
      const dx = p.x - origin.x, dy = p.y - origin.y, dz = p.z - origin.z;
      t = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (t < 0 || t > maxDist + 1e-3) return null;

    const obj = raw.object ?? raw.obj ?? raw.mesh ?? raw.collider ?? raw.body ?? null;
    if (skip && obj === skip) return null;
    // characters are resolved by hitbox, never as world geometry
    if (obj && obj.isObject3D && this._isEnemyObject(obj)) return null;

    const h = this._nextHit();
    h.t = t;
    if (p && typeof p.x === 'number') h.point.set(p.x, p.y, p.z);
    else h.point.set(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);

    const n = raw.normal ?? raw.n ?? raw.hitNormal ?? raw.faceNormal;
    if (n && typeof n.x === 'number') h.normal.set(n.x, n.y, n.z);
    else h.normal.set(-dir.x, -dir.y, -dir.z);
    if (h.normal.lengthSq() < 1e-8) h.normal.set(-dir.x, -dir.y, -dir.z);
    h.normal.normalize();
    if (h.normal.dot(dir) > 0) h.normal.negate();

    h.px = h.point.x; h.py = h.point.y; h.pz = h.point.z;
    h.nx = h.normal.x; h.ny = h.normal.y; h.nz = h.normal.z;
    h.object = obj && obj.isObject3D ? obj : null;
    h.material = raw.material ?? this._materialOf(h.object, null) ?? null;
    h.key = this._surfaceKey(typeof raw.surface === 'string' ? raw.surface : raw.key, h.material, h.object);
    h.surface = BUCKET[h.key] ?? 'concrete';
    return h;
  }

  /* ---- internal raycaster ---------------------------------------- */

  _fallbackRay(origin, dir, maxDist, skip) {
    const scene = this.g?.scene;
    if (!scene) return null;
    this._refreshCastList();
    const list = this._castList;
    if (!list.length) return null;

    const rc = this._rc;
    rc.ray.origin.copy(origin);
    rc.ray.direction.copy(dir);
    rc.near = 0;
    rc.far = maxDist;
    rc.camera = this.g?.camera ?? rc.camera;

    const out = this._rcHits;
    out.length = 0;
    try {
      rc.intersectObjects(list, true, out);
    } catch (e) {
      out.length = 0;
      return null;
    }
    if (!out.length) return null;

    let result = null;
    for (let i = 0; i < out.length; i++) {
      const it = out[i];
      const obj = it.object;
      if (!obj || obj === skip) continue;
      if (it.distance < 1e-4) continue;
      if (this._isSkipped(obj)) continue;

      const h = this._nextHit();
      h.t = it.distance;
      h.point.copy(it.point);
      h.px = h.point.x; h.py = h.point.y; h.pz = h.point.z;

      if (it.normal && typeof it.normal.x === 'number') {
        h.normal.copy(it.normal);
      } else if (it.face) {
        h.normal.copy(it.face.normal);
        this._nm.getNormalMatrix(obj.matrixWorld);
        h.normal.applyMatrix3(this._nm);
      } else {
        h.normal.set(-dir.x, -dir.y, -dir.z);
      }
      if (h.normal.lengthSq() < 1e-8) h.normal.set(-dir.x, -dir.y, -dir.z);
      h.normal.normalize();
      if (h.normal.dot(dir) > 0) h.normal.negate();
      h.nx = h.normal.x; h.ny = h.normal.y; h.nz = h.normal.z;

      h.object = obj;
      h.material = this._materialOf(obj, it);
      h.key = this._surfaceKey(null, h.material, obj);
      h.surface = BUCKET[h.key] ?? 'concrete';
      result = h;
      break;
    }
    out.length = 0;
    return result;
  }

  _materialOf(obj, it) {
    if (!obj) return null;
    const m = obj.material;
    if (!m) return null;
    if (Array.isArray(m)) return m[it?.face?.materialIndex ?? 0] ?? m[0] ?? null;
    return m;
  }

  _isSkipped(obj) {
    if (!obj.visible) return true;
    if (obj.isSprite || obj.isPoints || obj.isLine || obj.isLineSegments) return true;
    const roots = this._enemySet();
    let o = obj, depth = 0;
    while (o && depth < 10) {
      // A character's render mesh must never act as a wall — hitboxes are
      // authoritative for bodies, and a mesh hit would land a hair in front of
      // the hitbox and swallow every shot.
      if (roots.size && roots.has(o)) return true;
      const ud = o.userData;
      if (ud && (ud.noRaycast || ud.viewmodel || ud.isViewmodel || ud.isDecal || ud.fx)) return true;
      if (o.name && SKIP_NAMES.test(o.name)) return true;
      o = o.parent; depth++;
    }
    return false;
  }

  /** True when this object (or an ancestor) belongs to a registered enemy. */
  _isEnemyObject(obj) {
    const roots = this._enemySet();
    if (!roots.size) return false;
    let o = obj, depth = 0;
    while (o && depth < 10) {
      if (roots.has(o)) return true;
      o = o.parent; depth++;
    }
    return false;
  }

  /** Set of Object3D roots owned by registered enemies. Rebuilt lazily. */
  _enemySet() {
    const list = this.g?.registry?.enemies;
    const n = list ? list.length : 0;
    if (n !== this._enemyCount || this.t - this._enemyStamp > 1.0) {
      this._enemyCount = n;
      this._enemyStamp = this.t;
      this._enemyRoots.clear();
      for (let i = 0; i < n; i++) {
        const raw = list[i];
        if (!raw || typeof raw !== 'object') continue;
        const o = this._objOf(raw);
        if (o) this._enemyRoots.add(o);
        const e = this._entityOf(raw);
        if (e && e !== raw) {
          const eo = this._objOf(e);
          if (eo) this._enemyRoots.add(eo);
        }
      }
    }
    return this._enemyRoots;
  }

  _refreshCastList() {
    const scene = this.g?.scene;
    if (!scene) return;
    const n = scene.children.length;
    if (this._castList.length && this.t - this._castStamp < 2.0 && n === this._castChildren) return;
    this._castStamp = this.t;
    this._castChildren = n;

    // The renderer refreshes world matrices once per frame; a level built after
    // the last render (or a headless tool that never rendered) would otherwise
    // be raycast at identity. Cheap here — this runs at most every two seconds.
    try { scene.updateMatrixWorld(false); } catch (e) { /* exotic scene graph */ }

    const arr = this._castList;
    arr.length = 0;

    // 1) explicit colliders, if any module published them
    const cols = this.g?.registry?.colliders;
    if (cols && cols.length) {
      for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (!c || typeof c !== 'object') continue;
        const o = c.isObject3D ? c : (c.mesh ?? c.object ?? c.obj ?? c.node);
        if (o && o.isObject3D && !this._isSkipped(o)) arr.push(o);
      }
    }
    if (arr.length) return;

    // 2) otherwise walk the visible scene graph
    for (let i = 0; i < scene.children.length; i++) {
      const c = scene.children[i];
      if (!c || !c.visible || c.isLight || c.isCamera) continue;
      if (c.layers && (c.layers.mask & 1) === 0) continue;   // viewmodel layers stay out
      if (this._isSkipped(c)) continue;
      arr.push(c);
    }
  }

  _surfaceKey(explicit, material, object) {
    let k = typeof explicit === 'string' ? explicit : null;
    if (!k && object) {
      const ud = object.userData;
      if (ud) k = ud.bsSurface ?? ud.surface ?? ud.surfaceKey ?? (typeof ud.material === 'string' ? ud.material : null);
      if (!k && object.parent && object.parent.userData) {
        const pu = object.parent.userData;
        k = pu.bsSurface ?? pu.surface ?? pu.surfaceKey ?? null;
      }
    }
    if (!k && material) {
      k = material.userData?.bsSurface ?? material.userData?.surface ?? null;
      if (!k && typeof material.name === 'string' && material.name) {
        k = material.name.startsWith('bs:') ? material.name.slice(3) : material.name;
      }
    }
    if (typeof k !== 'string' || !k) return 'concrete';
    const s = k.toLowerCase().trim();
    if (SURF[s]) return s;
    if (KEYALIAS[s]) return KEYALIAS[s];
    const cut = s.split(/[^a-z]+/)[0];
    if (SURF[cut]) return cut;
    if (KEYALIAS[cut]) return KEYALIAS[cut];
    // last resort: Materials knows its own aliases
    const bucket = this.g?.materials?.surfaceFor?.(material ?? s);
    if (typeof bucket === 'string' && SURF[bucket]) return bucket;
    return 'concrete';
  }

  /* ================================================================ *
   *  ENEMY HITBOXES
   * ================================================================ */

  /** Explicit hitboxes for one enemy (Enemies may call this at spawn time). */
  registerHitboxes(enemy, list) {
    if (!enemy || typeof enemy !== 'object' || !Array.isArray(list)) return;
    this._hitboxOverride.set(enemy, list);
    this._rigs.delete(enemy);
  }

  _raycastEnemies(origin, dir, maxDist, ignore) {
    const list = this.g?.registry?.enemies;
    if (!list || !list.length) return null;

    let best = null, bestT = maxDist;

    for (let i = 0; i < list.length; i++) {
      const raw = list[i];
      if (!raw || typeof raw !== 'object' || raw === ignore) continue;
      const entity = this._entityOf(raw);
      if (entity === ignore || !this._enemyAlive(entity)) continue;

      const rig = this._rig(raw, entity);
      if (!rig) continue;

      // cheap bounding-sphere reject around the standing body
      const cx = rig.x, cy = rig.y + rig.h * 0.5 * rig.ys, cz = rig.z;
      const rx = cx - origin.x, ry = cy - origin.y, rz = cz - origin.z;
      const proj = rx * dir.x + ry * dir.y + rz * dir.z;
      if (proj < -rig.rad || proj > bestT + rig.rad) continue;
      const px = rx - dir.x * proj, py = ry - dir.y * proj, pz = rz - dir.z * proj;
      if (px * px + py * py + pz * pz > rig.rad * rig.rad) continue;

      const h = this._rigRay(rig, entity, origin, dir, bestT);
      if (h && h.t < bestT) { bestT = h.t; best = h; }
    }
    return best;
  }

  /** The logical entity behind a registry entry. */
  _entityOf(raw) {
    if (!raw) return null;
    if (raw.isObject3D) {
      const ud = raw.userData;
      if (ud) {
        const e = ud.enemy ?? ud.entity ?? ud.agent ?? ud.owner ?? ud.controller;
        if (e && typeof e === 'object') return e;
      }
      return raw;
    }
    return raw;
  }

  /** The Object3D carrying the enemy's transform, if any. */
  _objOf(raw) {
    if (!raw || typeof raw !== 'object') return null;
    if (raw.isObject3D) return raw;
    const c0 = raw.object3D, c1 = raw.object, c2 = raw.root, c3 = raw.group;
    const c4 = raw.mesh, c5 = raw.model, c6 = raw.obj, c7 = raw.body, c8 = raw.node;
    if (c0 && c0.isObject3D) return c0;
    if (c1 && c1.isObject3D) return c1;
    if (c2 && c2.isObject3D) return c2;
    if (c3 && c3.isObject3D) return c3;
    if (c4 && c4.isObject3D) return c4;
    if (c5 && c5.isObject3D) return c5;
    if (c6 && c6.isObject3D) return c6;
    if (c7 && c7.isObject3D) return c7;
    if (c8 && c8.isObject3D) return c8;
    return null;
  }

  _entityPos(raw, out) {
    const obj = this._objOf(raw);
    if (obj) { obj.getWorldPosition(out); return true; }
    const p = raw?.position ?? raw?.pos ?? raw?.p;
    if (p && typeof p.x === 'number') { out.set(p.x, p.y, p.z); return true; }
    return false;
  }

  _entityYaw(raw) {
    if (typeof raw?.yaw === 'number') return raw.yaw;
    if (typeof raw?.heading === 'number') return raw.heading;
    const obj = this._objOf(raw);
    if (obj) {
      obj.updateWorldMatrix(true, false);
      const e = obj.matrixWorld.elements;
      return Math.atan2(e[8], e[10]);
    }
    if (typeof raw?.rotation?.y === 'number') return raw.rotation.y;
    return 0;
  }

  /**
   * Cached hitbox rig.  Dimensions are measured from the scene graph once (so a
   * feet-origin and a centre-origin character both work) and then only the
   * transform is refreshed per shot.
   */
  _rig(raw, entity) {
    let rig = this._rigs.get(raw);
    if (!rig) {
      rig = { h: 1.8, foot: 0, rad: 1.55, x: 0, y: 0, z: 0, cos: 1, sin: 0, ys: 1, custom: null, stamp: -1e9 };
      this._rigs.set(raw, rig);
      this._measureRig(raw, entity, rig);
    } else if (this.t - rig.stamp > 3.0) {
      this._measureRig(raw, entity, rig);
    }

    if (!this._entityPos(raw, this._rgP)) return null;
    rig.x = this._rgP.x;
    rig.y = this._rgP.y + rig.foot;
    rig.z = this._rgP.z;

    const yaw = this._entityYaw(raw);
    rig.cos = Math.cos(yaw);
    rig.sin = Math.sin(yaw);

    const e = entity;
    const crouch = e ? (e.crouched === true || e.isCrouched === true ||
      e.crouching === true || e.stance === 'crouch') : false;
    const prone = e ? (e.prone === true || e.stance === 'prone') : false;
    rig.ys = prone ? 0.34 : (crouch ? 0.66 : 1.0);
    return rig;
  }

  _measureRig(raw, entity, rig) {
    rig.stamp = this.t;
    rig.custom = this._hitboxOverride.get(raw) ??
      (entity && typeof entity === 'object' ? this._hitboxOverride.get(entity) : null) ??
      (Array.isArray(entity?.hitboxes) ? entity.hitboxes : null) ??
      (Array.isArray(raw?.hitboxes) ? raw.hitboxes : null) ?? null;

    let h = (typeof entity?.height === 'number' ? entity.height : null) ??
      (typeof raw?.height === 'number' ? raw.height : null);
    let foot = null;

    const obj = this._objOf(raw);
    if (obj) {
      try {
        this._box.makeEmpty();
        this._box.setFromObject(obj);
        if (!this._box.isEmpty()) {
          const bh = this._box.max.y - this._box.min.y;
          if (bh > 0.6 && bh < 4.0) {
            if (h === null) h = bh;
            obj.getWorldPosition(this._rgQ);
            foot = this._box.min.y - this._rgQ.y;
          }
        }
      } catch (e) { /* geometry-less rigs fall through to the defaults */ }
    }

    if (h === null || !isFinite(h) || h <= 0) h = 1.8;
    if (foot === null || !isFinite(foot)) {
      // no measurable bounds: assume a feet origin unless the entity is
      // obviously anchored at torso height
      foot = (this._entityPos(raw, this._rgQ) && this._rgQ.y > h * 0.42) ? -h * 0.5 : 0;
      if (typeof entity?.eyeHeight === 'number') foot = -entity.eyeHeight;
    }
    if (typeof entity?.hitboxOffsetY === 'number') foot = entity.hitboxOffsetY;

    rig.h = h;
    rig.foot = foot;
    rig.rad = h * 0.72 + 0.30;
  }

  /** Ray vs one enemy rig.  Returns a hit record or null. */
  _rigRay(rig, entity, origin, dir, maxT) {
    // into the enemy's yaw-local frame, origin at the feet
    const rx = origin.x - rig.x, ry = origin.y - rig.y, rz = origin.z - rig.z;
    const c = rig.cos, s = rig.sin;
    const ox = rx * c - rz * s, oy = ry, oz = rx * s + rz * c;
    const dx = dir.x * c - dir.z * s, dy = dir.y, dz = dir.x * s + dir.z * c;

    let bestT = maxT, bestPart = null, bestMul = 1;
    let bnx = 0, bny = 0, bnz = 0;

    if (rig.custom) {
      const cl = rig.custom;
      for (let i = 0; i < cl.length; i++) {
        const hb = cl[i];
        if (!hb || typeof hb !== 'object') continue;
        const t = this._customHit(hb, origin, dir, bestT);
        if (t >= 0 && t < bestT) {
          bestT = t;
          bestPart = this._partName(hb);
          bestMul = this._partMul(hb, bestPart);
          bnx = -dir.x; bny = -dir.y; bnz = -dir.z;
        }
      }
      if (bestPart === null) return null;
      return this._fillCharHit(bestT, origin, dir, bnx, bny, bnz, entity, bestPart, bestMul, false);
    }

    const H = rig.h, ys = rig.ys;
    for (let i = 0; i < RIG_PARTS.length; i++) {
      const P = RIG_PARTS[i];
      let t = -1, nx = 0, ny = 0, nz = 0;

      if (P.k === 1) {
        const cx = P.x * H, cy = P.y * H * ys, cz = P.z * H;
        t = this._raySphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, P.r * H, bestT);
        if (t >= 0) {
          nx = (ox + dx * t) - cx; ny = (oy + dy * t) - cy; nz = (oz + dz * t) - cz;
          const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
          nx /= l; ny /= l; nz /= l;
        }
      } else {
        const x0 = P.x0 * H, x1 = P.x1 * H;
        const y0 = P.y0 * H * ys, y1 = P.y1 * H * ys;
        const z0 = P.z0 * H, z1 = P.z1 * H;
        t = this._rayBox(ox, oy, oz, dx, dy, dz, x0, y0, z0, x1, y1, z1, bestT);
        if (t >= 0) {
          // face normal = whichever slab plane we landed closest to
          const hx = ox + dx * t, hy = oy + dy * t, hz = oz + dz * t;
          let m = Math.abs(hx - x0); nx = -1; ny = 0; nz = 0;
          let v = Math.abs(hx - x1); if (v < m) { m = v; nx = 1; ny = 0; nz = 0; }
          v = Math.abs(hy - y0); if (v < m) { m = v; nx = 0; ny = -1; nz = 0; }
          v = Math.abs(hy - y1); if (v < m) { m = v; nx = 0; ny = 1; nz = 0; }
          v = Math.abs(hz - z0); if (v < m) { m = v; nx = 0; ny = 0; nz = -1; }
          v = Math.abs(hz - z1); if (v < m) { nx = 0; ny = 0; nz = 1; }
        }
      }

      if (t >= 0 && t < bestT) {
        bestT = t;
        bestPart = (P.n === 'armL' || P.n === 'armR') ? 'arm'
          : ((P.n === 'legL' || P.n === 'legR') ? 'leg' : P.n);
        bestMul = P.mul;
        bnx = nx; bny = ny; bnz = nz;
      }
    }
    if (bestPart === null) return null;

    // local normal back to world
    const wnx = bnx * c + bnz * s;
    const wnz = -bnx * s + bnz * c;
    return this._fillCharHit(bestT, origin, dir, wnx, bny, wnz, entity, bestPart, bestMul, false);
  }

  _fillCharHit(t, origin, dir, nx, ny, nz, entity, part, mul, isPlayer) {
    const h = this._nextHit();
    h.t = t;
    h.point.set(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);
    h.px = h.point.x; h.py = h.point.y; h.pz = h.point.z;

    let l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l < 1e-6) { nx = -dir.x; ny = -dir.y; nz = -dir.z; l = 1; }
    h.normal.set(nx / l, ny / l, nz / l);
    if (h.normal.dot(dir) > 0) h.normal.negate();
    h.nx = h.normal.x; h.ny = h.normal.y; h.nz = h.normal.z;

    h.enemy = isPlayer ? null : entity;
    h.entity = isPlayer ? null : entity;
    h.part = part;
    h.mul = mul;
    h.key = 'flesh';
    h.surface = 'flesh';
    h.object = isPlayer ? null : this._objOf(entity);
    h.material = null;
    return h;
  }

  _partName(hb) {
    const n = hb.part ?? hb.name ?? hb.tag ?? hb.object?.name ?? '';
    if (typeof n === 'string' && n) {
      if (HEAD_NAMES.test(n)) return 'head';
      if (ARM_NAMES.test(n)) return 'arm';
      if (LEG_NAMES.test(n)) return 'leg';
      if (TORSO_NAMES.test(n)) return 'torso';
    }
    return 'torso';
  }

  _partMul(hb, part) {
    const m = hb.mul ?? hb.multiplier ?? hb.damageMul ?? hb.damageMultiplier;
    if (typeof m === 'number' && isFinite(m)) return m;
    if (part === 'head') return -1;                  // -1 -> Config.weapon.headMul
    if (part === 'arm' || part === 'leg') return 0.75;
    return 1.0;
  }

  /** Ray vs one caller-supplied hitbox description (world space). */
  _customHit(hb, origin, dir, maxT) {
    const box = hb.box ?? hb.aabb ?? (hb.isBox3 ? hb : null);
    if (box && box.min && box.max) {
      return this._rayBox(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
        box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z, maxT);
    }
    const c = hb.center ?? hb.c;
    const r = hb.radius ?? hb.r;
    if (c && typeof c.x === 'number' && typeof r === 'number') {
      return this._raySphere(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
        c.x, c.y, c.z, r, maxT);
    }
    const obj = hb.object ?? hb.mesh ?? hb.node ?? (hb.isObject3D ? hb : null);
    if (obj && obj.isObject3D) {
      try {
        this._box.makeEmpty();
        this._box.setFromObject(obj);
        if (this._box.isEmpty()) return -1;
        return this._rayBox(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
          this._box.min.x, this._box.min.y, this._box.min.z,
          this._box.max.x, this._box.max.y, this._box.max.z, maxT);
      } catch (e) { return -1; }
    }
    return -1;
  }

  /* ---- the player as a target ------------------------------------ */

  _raycastPlayer(origin, dir, maxDist) {
    const cam = this.g?.camera;
    if (!cam) return null;
    cam.getWorldPosition(this._plr);
    const px = this._plr.x, py = this._plr.y, pz = this._plr.z;
    if (!isFinite(px)) return null;

    const pc = this.cfg?.player ?? {};
    const eye = pc.eyeHeight ?? 1.62;
    const rad = (pc.radius ?? 0.32) * 1.02;
    const c = this.g?.controller;
    const crouch = !!(c && (c.crouched === true || c.isCrouched === true || c.crouching === true));
    const drop = crouch ? eye * ((pc.crouchHeight ?? 1.0) / (pc.height ?? 1.75)) : eye;
    const feetY = py - drop;
    const headY = py - 0.06;
    const shoulderY = py - 0.24;

    const th = this._raySphere(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
      px, headY, pz, 0.145, maxDist);
    const tb = this._rayVertCyl(origin.x, origin.y, origin.z, dir.x, dir.y, dir.z,
      px, pz, feetY, shoulderY, rad, maxDist);

    if (th < 0 && tb < 0) return null;
    const head = (th >= 0 && (tb < 0 || th <= tb));
    const t = head ? th : tb;

    const hx = origin.x + dir.x * t, hy = origin.y + dir.y * t, hz = origin.z + dir.z * t;
    const nx = hx - px, nz = hz - pz;
    const ny = head ? (hy - headY) : 0;

    const h = this._fillCharHit(t, origin, dir, nx, ny, nz, null,
      head ? 'head' : 'torso', head ? -1 : 1, true);

    if (!head) {
      // Limb vs centre mass is decided by how far the ray LINE passes from the
      // body axis, not by where it touched the cylinder skin (that is always
      // exactly one radius out, which would call every hit an arm).
      const A = dir.x * dir.x + dir.z * dir.z;
      if (A > 1e-9) {
        const ex = origin.x - px, ez = origin.z - pz;
        const off = Math.abs(ex * dir.z - ez * dir.x) / Math.sqrt(A);
        if (off > rad * 0.62) { h.part = 'arm'; h.mul = T.enemyLimbMul; }
      }
      if (hy < feetY + (shoulderY - feetY) * 0.34) { h.part = 'leg'; h.mul = T.enemyLimbMul; }
    }
    return h;
  }

  /* ---- primitive intersection ------------------------------------ */

  _rayBox(ox, oy, oz, dx, dy, dz, x0, y0, z0, x1, y1, z1, maxT) {
    const idx = 1 / (dx !== 0 ? dx : 1e-9);
    const idy = 1 / (dy !== 0 ? dy : 1e-9);
    const idz = 1 / (dz !== 0 ? dz : 1e-9);

    let a = (x0 - ox) * idx, b = (x1 - ox) * idx;
    let tmin = a < b ? a : b;
    let tmax = a < b ? b : a;

    a = (y0 - oy) * idy; b = (y1 - oy) * idy;
    const ymin = a < b ? a : b, ymax = a < b ? b : a;
    if (ymin > tmin) tmin = ymin;
    if (ymax < tmax) tmax = ymax;

    a = (z0 - oz) * idz; b = (z1 - oz) * idz;
    const zmin = a < b ? a : b, zmax = a < b ? b : a;
    if (zmin > tmin) tmin = zmin;
    if (zmax < tmax) tmax = zmax;

    if (tmax < tmin || tmax < 0) return -1;
    const t = tmin >= 0 ? tmin : tmax;
    return (t < 0 || t > maxT) ? -1 : t;
  }

  _raySphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, r, maxT) {
    const lx = cx - ox, ly = cy - oy, lz = cz - oz;
    const tca = lx * dx + ly * dy + lz * dz;
    const d2 = lx * lx + ly * ly + lz * lz - tca * tca;
    const r2 = r * r;
    if (d2 > r2) return -1;
    const thc = Math.sqrt(r2 - d2);
    let t = tca - thc;
    if (t < 0) t = tca + thc;
    return (t < 0 || t > maxT) ? -1 : t;
  }

  _rayVertCyl(ox, oy, oz, dx, dy, dz, cx, cz, y0, y1, r, maxT) {
    const ex = ox - cx, ez = oz - cz;
    const A = dx * dx + dz * dz;
    const B = 2 * (ex * dx + ez * dz);
    const C = ex * ex + ez * ez - r * r;

    if (A > 1e-9) {
      const disc = B * B - 4 * A * C;
      if (disc >= 0) {
        const sq = Math.sqrt(disc);
        const inv = 0.5 / A;
        let t = (-B - sq) * inv;
        for (let i = 0; i < 2; i++) {
          if (t >= 0 && t <= maxT) {
            const y = oy + dy * t;
            if (y >= y0 && y <= y1) return t;
          }
          t = (-B + sq) * inv;
        }
      }
    }
    if (Math.abs(dy) > 1e-9) {                 // end caps
      for (let i = 0; i < 2; i++) {
        const t = ((i === 0 ? y0 : y1) - oy) / dy;
        if (t >= 0 && t <= maxT) {
          const hx = ox + dx * t - cx, hz = oz + dz * t - cz;
          if (hx * hx + hz * hz <= r * r) return t;
        }
      }
    }
    return -1;
  }

  /* ================================================================ *
   *  MUZZLE / AIM HELPERS
   * ================================================================ */

  _muzzle(out) {
    const w = this.g?.weapon;
    if (w) {
      if (typeof w.getMuzzleWorldPosition === 'function') {
        try {
          const r = w.getMuzzleWorldPosition(out);
          if (r && r.isVector3 && r !== out) out.copy(r);
          if (isFinite(out.x)) return out;
        } catch (e) { /* fall through */ }
      }
      if (typeof w.getMuzzlePosition === 'function') {
        try {
          const r = w.getMuzzlePosition(out);
          if (r && r.isVector3 && r !== out) out.copy(r);
          if (isFinite(out.x)) return out;
        } catch (e) { /* fall through */ }
      }
      const mw = w.muzzleWorld ?? w.muzzlePos ?? w.muzzlePosition;
      if (mw && typeof mw.x === 'number' && (mw.x || mw.y || mw.z)) { out.copy(mw); return out; }
      const mo = w.muzzle ?? w.muzzleTip ?? w.muzzleNode;
      if (mo && mo.isObject3D) { mo.getWorldPosition(out); return out; }
    }
    // plausible barrel position hung off the camera
    const cam = this.g?.camera;
    if (!cam) { out.set(0, 1.6, 0); return out; }
    cam.getWorldPosition(out);
    cam.getWorldDirection(this._camFwd);
    this._camRight.crossVectors(this._camFwd, this._up);
    if (this._camRight.lengthSq() < 1e-8) this._camRight.set(1, 0, 0);
    this._camRight.normalize();
    const ads = this._adsBlend;
    out.addScaledVector(this._camFwd, 0.62)
      .addScaledVector(this._camRight, lerp(0.145, 0.008, ads))
      .addScaledVector(this._up, lerp(-0.135, -0.045, ads));
    return out;
  }

  _entityMuzzle(e, out) {
    if (!e || typeof e !== 'object') return false;
    const w = e.muzzle ?? e.weaponMuzzle ?? e.gunMuzzle;
    if (w && w.isObject3D) { w.getWorldPosition(out); return true; }
    if (w && typeof w.x === 'number') { out.set(w.x, w.y, w.z); return true; }
    if (!this._entityPos(e, out)) return false;
    const rig = this._rigs.get(e);
    const h = rig ? rig.h : (typeof e.height === 'number' ? e.height : 1.8);
    const foot = rig ? rig.foot : 0;
    out.y += foot + h * 0.78;
    return true;
  }

  _aimAtPlayer(org, out) {
    const cam = this.g?.camera;
    if (!cam) return false;
    cam.getWorldPosition(this._aimT);
    this._aimT.y -= 0.35;                       // centre mass, not the eyeball
    out.copy(this._aimT).sub(org);
    const l = out.length();
    if (l < 1e-4) return false;
    out.multiplyScalar(1 / l);
    return true;
  }

  /* ================================================================ *
   *  NEAR-MISS WHIZZ
   * ================================================================ */

  _scheduleWhizz(origin, dir, segLen, speed) {
    if (!this._prevCamValid || !(segLen > 0)) return;
    const rx = this._camPos.x - origin.x;
    const ry = this._camPos.y - origin.y;
    const rz = this._camPos.z - origin.z;
    const proj = rx * dir.x + ry * dir.y + rz * dir.z;
    if (proj < T.whizzMinDist || proj > segLen) return;
    const cx = rx - dir.x * proj, cy = ry - dir.y * proj, cz = rz - dir.z * proj;
    const d2 = cx * cx + cy * cy + cz * cz;
    if (d2 > T.whizzRadius * T.whizzRadius) return;

    const i = this._wCur;
    this._wCur = (this._wCur + 1) % this._wN;
    this._wT[i] = this.t + proj / Math.max(50, speed);
    this._wX[i] = origin.x + dir.x * proj;
    this._wY[i] = origin.y + dir.y * proj;
    this._wZ[i] = origin.z + dir.z * proj;
    this._wDist[i] = Math.sqrt(d2);
    this._wS[i] = speed;
    this._wDX[i] = dir.x; this._wDY[i] = dir.y; this._wDZ[i] = dir.z;
  }

  _updateWhizz(t) {
    const bus = this.g?.bus;
    for (let i = 0; i < this._wN; i++) {
      const wt = this._wT[i];
      if (wt < 0 || t < wt) continue;
      this._wT[i] = -1;
      if (t - wt > 0.4 || !bus) continue;          // stale (backgrounded tab)
      const p = this._wPay[this._wPayIdx];
      this._wPayIdx = (this._wPayIdx + 1) % this._wPay.length;
      p.point.set(this._wX[i], this._wY[i], this._wZ[i]);
      p.dir.set(this._wDX[i], this._wDY[i], this._wDZ[i]);
      p.distance = this._wDist[i];
      p.speed = this._wS[i];
      bus.emit('whizz', p);
    }
  }

  /* ================================================================ *
   *  TRACERS
   * ================================================================ */

  _allocTracers(n) {
    this._tOn = new Uint8Array(n);
    this._tSx = new Float32Array(n); this._tSy = new Float32Array(n); this._tSz = new Float32Array(n);
    this._tDx = new Float32Array(n); this._tDy = new Float32Array(n); this._tDz = new Float32Array(n);
    this._tTot = new Float32Array(n);
    this._tTrav = new Float32Array(n);
    this._tSpd = new Float32Array(n);
    this._tLen = new Float32Array(n);
    this._tAge = new Float32Array(n);
    this._tRt = new Float32Array(n);     // REAL seconds alive — the authoritative lifetime
    this._tPh = new Float32Array(n);     // flicker phase
    this._tR = new Float32Array(n); this._tG = new Float32Array(n); this._tB = new Float32Array(n);
    this._tW = new Float32Array(n);
    this._tA = new Float32Array(n);
    this._tCore = new Float32Array(n);
    this._tNext = 0;

    this._iHead = new Float32Array(n * 3);
    this._iTail = new Float32Array(n * 3);
    this._iCol = new Float32Array(n * 3);
    this._iPar = new Float32Array(n * 3);
  }

  _buildTracerBatch() {
    const scene = this.g?.scene;
    if (!scene) return;

    const geo = new THREE.InstancedBufferGeometry();
    // position.xy is the billboard corner: x across (-1..1), y along (0 tail, 1 head).
    // The strip is subdivided along y so the quad can carry a real comet
    // silhouette — thin tail, full body, rounded tip — instead of a constant
    // width ribbon whose ends stop dead.
    const SEG = 10;
    const pos = new Float32Array((SEG + 1) * 2 * 3);
    const idx = new Uint16Array(SEG * 6);
    for (let r = 0; r <= SEG; r++) {
      const y = r / SEG, o = r * 6;
      pos[o] = -1; pos[o + 1] = y; pos[o + 2] = 0;
      pos[o + 3] = 1; pos[o + 4] = y; pos[o + 5] = 0;
    }
    for (let r = 0; r < SEG; r++) {
      const a = r * 2, o = r * 6;
      idx[o] = a; idx[o + 1] = a + 1; idx[o + 2] = a + 3;
      idx[o + 3] = a; idx[o + 4] = a + 3; idx[o + 5] = a + 2;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));

    const aHead = new THREE.InstancedBufferAttribute(this._iHead, 3);
    const aTail = new THREE.InstancedBufferAttribute(this._iTail, 3);
    const aCol = new THREE.InstancedBufferAttribute(this._iCol, 3);
    const aPar = new THREE.InstancedBufferAttribute(this._iPar, 3);
    aHead.setUsage(THREE.DynamicDrawUsage);
    aTail.setUsage(THREE.DynamicDrawUsage);
    aCol.setUsage(THREE.DynamicDrawUsage);
    aPar.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('iHead', aHead);
    geo.setAttribute('iTail', aTail);
    geo.setAttribute('iCol', aCol);
    geo.setAttribute('iPar', aPar);
    geo.instanceCount = 0;

    const mat = new THREE.ShaderMaterial({
      name: 'bs:tracer',
      uniforms: {
        uPixScale: { value: 900 },   // (viewportH/2) / tan(fov/2)
        uMinPx: { value: 2.80 },     // minimum on-screen HALF width, pixels
        uMaxPx: { value: 6.00 },     // maximum — a round passing the ear is a
                                     // fast blur, never a searchlight
      },
      vertexShader: /* glsl */`
        uniform float uPixScale;
        uniform float uMinPx;
        uniform float uMaxPx;

        attribute vec3 iHead;
        attribute vec3 iTail;
        attribute vec3 iCol;
        attribute vec3 iPar;    // x width, y alpha, z head-core boost

        varying vec2  vC;
        varying vec3  vCol;
        varying float vAlpha;
        varying float vCore;

        void main() {
          vec4 hv = modelViewMatrix * vec4( iHead, 1.0 );
          vec4 tv = modelViewMatrix * vec4( iTail, 1.0 );
          float along = position.y;
          vec3 p  = mix( tv.xyz, hv.xyz, along );

          vec3 axis = hv.xyz - tv.xyz;
          float al  = length( axis );
          vec3 ad   = al > 1e-5 ? axis / al : vec3( 0.0, 1.0, 0.0 );

          vec3 toEye = normalize( -p );
          vec3 side  = cross( ad, toEye );
          float sl   = length( side );
          side = sl > 1e-4 ? side / sl : normalize( cross( ad, vec3( 0.0, 0.0, 1.0 ) ) + vec3( 1e-4 ) );

          // Comet silhouette: a thin wake that swells behind the round and
          // rounds off at the tip, so neither end is a flat cut.
          float prof = mix( 0.20, 1.0, smoothstep( 0.0, 0.58, along ) )
                     * ( 1.0 - 0.52 * smoothstep( 0.86, 1.0, along ) );

          float dist = max( length( p ), 0.05 );
          float wGeo = iPar.x * prof * ( 1.0 + dist * 0.004 );
          float wMin = dist * uMinPx / uPixScale;
          float wMax = dist * uMaxPx / uPixScale;
          float w    = clamp( wGeo, wMin, wMax );

          // Widening a sub-pixel streak up to the pixel floor has to COST
          // brightness. Without this a distant round keeps full intensity while
          // being forced to one pixel wide, which is precisely the aliased
          // hairline read we are trying to kill.
          float comp = clamp( wGeo / max( w, 1e-6 ), 0.14, 1.0 );

          p += side * ( position.x * w );

          vC     = vec2( position.x, along );
          vCol   = iCol;
          // sail past the eye and you get a fast dim blur, not a searchlight
          vAlpha = iPar.y * comp * smoothstep( 0.25, 1.80, dist );
          vCore  = iPar.z;
          gl_Position = projectionMatrix * vec4( p, 1.0 );
        }
      `,
      fragmentShader: /* glsl */`
        varying vec2  vC;
        varying vec3  vCol;
        varying float vAlpha;
        varying float vCore;

        void main() {
          float x     = vC.x;
          float along = clamp( vC.y, 0.0, 1.0 );

          // Gaussian cross-section. Both the soft halo and the filament inside
          // it reach zero before the quad edge, so there is no hard flank.
          float glow = max( exp( -3.2 * x * x ) - 0.0408, 0.0 ) * 1.0426;
          float core = exp( -22.0 * x * x );

          // Alpha taper at BOTH ends — the wake dissolves, the tip dissolves.
          float tail = smoothstep( 0.0, 0.34, along );
          float tip  = 1.0 - smoothstep( 0.80, 1.0, along );
          float body = tail * tip * ( 0.28 + 0.72 * along );
          float hot  = smoothstep( 0.52, 0.92, along ) * tip * vCore;

          float a = vAlpha * body * ( glow * 0.70 + core * 0.46 );
          if ( a < 0.0025 ) discard;

          // HDR: the filament runs well over 1.0 so the grade rolls it off warm
          // instead of clipping it to a flat #ffffff wire.
          vec3 c = vCol * ( 0.32 + glow * 0.72 + core * 1.10 + hot * 0.85 )
                 + vec3( core * hot * 0.55 );
          gl_FragColor = vec4( c, a );

          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: true,
      fog: false,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = 'bs_tracers';
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.renderOrder = 12;
    mesh.visible = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.userData.noRaycast = true;
    mesh.userData.fx = true;
    scene.add(mesh);

    this._mesh = mesh;
    this._geo = geo;
    this._aHead = aHead; this._aTail = aTail; this._aCol = aCol; this._aPar = aPar;
  }

  _spawnTracer(from, to, source, depth) {
    if (!from || !to || !this._mesh) return;
    const dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(d > T.tracerMinDist)) return;

    // first free slot, otherwise steal the oldest
    let i = -1;
    for (let k = 0; k < this._tN; k++) {
      const s = (this._tNext + k) % this._tN;
      if (!this._tOn[s]) { i = s; break; }
    }
    if (i < 0) {
      let oldest = 0, age = -1;
      for (let s = 0; s < this._tN; s++) if (this._tAge[s] > age) { age = this._tAge[s]; oldest = s; }
      i = oldest;
    }
    this._tNext = (i + 1) % this._tN;

    const enemy = source === 'enemy';
    const inv = 1 / d;
    this._tOn[i] = 1;
    this._tSx[i] = from.x; this._tSy[i] = from.y; this._tSz[i] = from.z;
    this._tDx[i] = dx * inv; this._tDy[i] = dy * inv; this._tDz[i] = dz * inv;
    this._tTot[i] = d;
    this._tTrav[i] = 0;
    this._tAge[i] = 0;
    this._tRt[i] = 0;
    this._tPh[i] = this._rand() * 6.2831853;
    this._tSpd[i] = (enemy ? T.tracerSpeedEnemy : T.tracerSpeed) * (0.94 + this._rand() * 0.12);
    this._tLen[i] = (enemy ? T.tracerLenEnemy : T.tracerLen) * (0.85 + this._rand() * 0.30);

    if (enemy) {
      // hostile rounds: orange-red with a hot core
      this._tR[i] = 1.00; this._tG[i] = 0.33; this._tB[i] = 0.11;
      this._tW[i] = T.tracerWidthEnemy;
      this._tA[i] = 1.06;
      this._tCore[i] = 1.15;
    } else {
      // friendly 5.56: warm gold, sits inside the golden-hour grade
      this._tR[i] = 1.00; this._tG[i] = 0.80; this._tB[i] = 0.40;
      this._tW[i] = T.tracerWidth;
      this._tA[i] = 0.96;
      this._tCore[i] = 1.00;
    }
    if (depth > 0) { this._tA[i] *= 0.60; this._tW[i] *= 0.85; }
    this.stats.tracers++;
  }

  /** Release every live tracer and take the batch off screen this frame. */
  _flushTracers() {
    if (this._tOn) this._tOn.fill(0);
    if (this._geo) this._geo.instanceCount = 0;
    if (this._mesh) this._mesh.visible = false;
  }

  _updateTracers(dt) {
    const mesh = this._mesh;
    if (!mesh) return;

    // ------------------------------------------------------------------
    // WALL CLOCK, not the simulation clock.
    //
    // main.js hands us `dt = min(realDelta, 0.05)`. That clamp is correct for
    // gameplay, but it means the sim clock runs arbitrarily far behind real
    // time whenever the frame rate collapses (a hitch, a backgrounded tab, a
    // software rasteriser). A tracer's entire existence is ~100 ms. Aged on the
    // clamped clock it survives for seconds of real time and ends up frozen in
    // the frame as a stray streak with no relationship to anything being fired
    // — a stale line primitive, not an effect. So the pool ages on the wall
    // clock and is released on the wall clock.
    // ------------------------------------------------------------------
    const now = (typeof performance !== 'undefined' && performance.now)
      ? performance.now() * 0.001
      : (this._rtPrev < 0 ? 0 : this._rtPrev + dt);
    let rdt = (this._rtPrev < 0) ? dt : (now - this._rtPrev);
    this._rtPrev = now;
    if (!(rdt >= 0) || !isFinite(rdt)) rdt = dt;
    this._rdt = rdt;

    // A frame that took longer to produce than a tracer exists for cannot
    // honestly show one: whatever is still in the pool is stale data. Drop it
    // all rather than smear a frozen streak across the presented image.
    if (rdt > T.tracerStaleFrame) { this._flushTracers(); return; }

    const head = this._iHead, tail = this._iTail, col = this._iCol, par = this._iPar;
    let count = 0;

    for (let i = 0; i < this._tN; i++) {
      if (!this._tOn[i]) continue;

      // authoritative lifetime first — nothing outlives it
      this._tRt[i] += rdt;
      if (this._tRt[i] > T.tracerRealLife) { this._tOn[i] = 0; continue; }

      this._tAge[i] += dt;
      this._tTrav[i] += this._tSpd[i] * dt;

      const total = this._tTot[i];
      const len = this._tLen[i];
      const trav = this._tTrav[i];
      const hd = trav < total ? trav : total;
      const tl = trav - len;

      if (tl >= total || this._tAge[i] > 3.0) { this._tOn[i] = 0; continue; }
      const t0 = tl > 0 ? tl : 0;
      if (hd - t0 < 0.02) { this._tOn[i] = 0; continue; }

      const sx = this._tSx[i], sy = this._tSy[i], sz = this._tSz[i];
      const dx = this._tDx[i], dy = this._tDy[i], dz = this._tDz[i];

      const o = count * 3;
      head[o] = sx + dx * hd; head[o + 1] = sy + dy * hd; head[o + 2] = sz + dz * hd;
      tail[o] = sx + dx * t0; tail[o + 1] = sy + dy * t0; tail[o + 2] = sz + dz * t0;
      col[o] = this._tR[i]; col[o + 1] = this._tG[i]; col[o + 2] = this._tB[i];

      // fade in off the muzzle, fade out once the round has landed
      let a = this._tA[i] * clamp01(trav / 2.4);
      if (trav > total) a *= clamp01(1 - (trav - total) / Math.max(0.5, len * 0.9));
      // and fade out on the wall clock as the slot approaches release, so a
      // tracer can never be culled mid-brightness and pop
      a *= clamp01((T.tracerRealLife - this._tRt[i]) / (T.tracerRealLife * 0.34));

      // Burning propellant is not a steady lamp — jitter the output every frame.
      const fl = 0.80 + 0.20 * Math.sin(now * 213.0 + this._tPh[i]);
      a *= fl;

      par[o] = this._tW[i];
      par[o + 1] = a;
      par[o + 2] = this._tCore[i] * (trav < total ? 1 : 0.35) * (0.86 + 0.28 * fl);
      count++;
    }

    if (count > 0) {
      this._aHead.needsUpdate = true;
      this._aTail.needsUpdate = true;
      this._aCol.needsUpdate = true;
      this._aPar.needsUpdate = true;

      // keep the minimum-pixel-width term correct through ADS fov changes
      const cam = this.g?.camera;
      const fov = ((cam && cam.isPerspectiveCamera ? cam.fov : null) ??
        this.cfg?.weapon?.fovHip ?? 68) * Math.PI / 180;
      mesh.material.uniforms.uPixScale.value = (this._vh * 0.5) / Math.tan(fov * 0.5);
    }
    this._geo.instanceCount = count;
    mesh.visible = count > 0;
  }
}

export default Ballistics;
