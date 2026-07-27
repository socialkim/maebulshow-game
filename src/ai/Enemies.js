// ============================================================================
// BLACKSITE — src/ai/Enemies.js
// Owner: AI / enemy-combatant agent.
//
// Everything about the hostiles lives here and nowhere else:
//
//   * A procedurally generated 1.80 m humanoid soldier — plate carrier with
//     pouches, radio + whip antenna, ballistic helmet with NVG shroud and
//     headset cups, gloves, knee pads, boots, and a rifle held in a real
//     two-handed grip.  ~3k triangles, two draw calls (soft kit / hard kit),
//     one shared geometry across every soldier in the game.
//
//   * A real skinned rig: 21 THREE.Bone objects, procedurally generated skin
//     weights (up to three influences per vertex, smooth-stepped across every
//     joint), THREE.Skeleton + THREE.SkinnedMesh.  Animation is joint rotation,
//     never object hopping.
//
//   * Procedural animation: idle-alert with weapon-ready breathing, walk, run,
//     crouch-walk, fire with recoil driven into the shoulder, reload, take-cover
//     crouch, flinch and death.  Legs are IK'd to world-locked foot plants so
//     the feet never slide, and the left hand is IK'd onto the rifle handguard
//     every frame so the two-handed grip survives recoil, aim and reload.  The
//     muzzle is closed onto the target by an aim feedback loop through the
//     spine, so where the rifle points is where the bullet goes.
//
//   * Behaviour: PATROL -> SUSPECT -> ALERT -> ENGAGE -> SUPPRESSED -> RELOAD
//     -> SEEK COVER -> FLANK, with a vision cone from Config.ai, line-of-sight
//     through game.physics, per-soldier reaction delay, squad alert
//     propagation, cover scored against real level geometry, and suppression
//     that makes them duck instead of trade shots.
//
//   * Damage: {head, torso, limbs} world hitboxes registered into
//     game.registry.enemies, a 'hit' bus listener, directional flinch, and on
//     death a verlet ragdoll built on the bone hierarchy that collapses,
//     settles, lingers, fades and respawns.  Emits 'kill'.
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators — this is the whole contract)
// ---------------------------------------------------------------------------
//   game.registry.enemies -> Array<EnemyHandle>   (stable, built during init)
//
//   EnemyHandle = {
//     id, alive, dead, hp, maxHp, team,
//     position   : Vector3  world, chest height, LIVE (do not keep a copy)
//     head       : Box3     world, updated every frame while alive
//     torso      : Box3
//     limbs      : Box3[]   [armL, armR, legL, legR]
//     hitboxes   : [{ part, box, mul }]   part: 'head'|'torso'|'limb'
//     object3D / object / mesh : THREE.SkinnedMesh
//     radius, height,
//     raycast(origin, dir, maxDist) -> {point, distance, part, headshot, mul} | null
//     applyDamage(amount, opts) -> boolean killed   opts: {point, dir, headshot, part}
//     damage(...) / hit(...)      aliases of applyDamage
//   }
//
//   Ballistics may resolve enemy hits EITHER by calling handle.applyDamage(...)
//   OR by emitting bus 'hit' { enemy, point, damage, headshot }.  Doing both in
//   the same frame is de-duplicated here, so you cannot double-damage.
//
//   Enemy fire is published as bus 'shot'
//     { origin, dir, weapon:'enemy_rifle', source:'enemy', enemy, damage, spread }
//   If your Ballistics wants to own the resolution of enemy bullets against the
//   player, set `game.ballistics.handlesEnemyFire = true` and this module stops
//   emitting 'damage' itself.  Otherwise it resolves the trace against the
//   player capsule and emits 'damage' { amount, fromDir, from, source, enemy }.
//
//   game.enemies.alertAll(pos)          — force the whole squad onto a position
//   game.enemies.showcase(x, z, yaw)    — line the squad up in front of a camera
//   game.enemies.setStance(name)        — 'idle'|'patrol'|'aim'|'fire'|'crouch'|'reload'
//   game.enemies.stats                  — { alive, dead, tris, drawCalls }
// ============================================================================

import * as THREE from 'three';

/* =========================================================================== *
 *  CONSTANTS
 * =========================================================================== */

const DEG = Math.PI / 180;

// Bone table.  parent = -1 for the root.  `axis` is which local axis points at
// the child: +1 => local +Y, -1 => local -Y.  Offsets are in the parent's local
// frame, in metres, and are the proportions of a real 1.80 m male.
const B_PELVIS = 0, B_SPINE = 1, B_CHEST = 2, B_NECK = 3, B_HEAD = 4,
  B_CLAVL = 5, B_ARML = 6, B_FOREL = 7, B_HANDL = 8,
  B_CLAVR = 9, B_ARMR = 10, B_FORER = 11, B_HANDR = 12,
  B_THIGHL = 13, B_SHINL = 14, B_FOOTL = 15, B_TOEL = 16,
  B_THIGHR = 17, B_SHINR = 18, B_FOOTR = 19, B_TOER = 20;
const BONE_COUNT = 21;

const BONES = [
  /*  0 */ { name: 'pelvis', parent: -1, off: [0, 0.960, 0], axis: 1 },
  /*  1 */ { name: 'spine', parent: B_PELVIS, off: [0, 0.135, 0], axis: 1 },
  /*  2 */ { name: 'chest', parent: B_SPINE, off: [0, 0.195, 0], axis: 1 },
  /*  3 */ { name: 'neck', parent: B_CHEST, off: [0, 0.165, 0.012], axis: 1 },
  /*  4 */ { name: 'head', parent: B_NECK, off: [0, 0.095, 0], axis: 1 },
  /*  5 */ { name: 'clavL', parent: B_CHEST, off: [0.046, 0.150, 0.014], axis: 1 },
  /*  6 */ { name: 'armL', parent: B_CLAVL, off: [0.140, -0.020, 0], axis: -1 },
  /*  7 */ { name: 'foreL', parent: B_ARML, off: [0, -0.285, 0], axis: -1 },
  /*  8 */ { name: 'handL', parent: B_FOREL, off: [0, -0.255, 0], axis: -1 },
  /*  9 */ { name: 'clavR', parent: B_CHEST, off: [-0.046, 0.150, 0.014], axis: 1 },
  /* 10 */ { name: 'armR', parent: B_CLAVR, off: [-0.140, -0.020, 0], axis: -1 },
  /* 11 */ { name: 'foreR', parent: B_ARMR, off: [0, -0.285, 0], axis: -1 },
  /* 12 */ { name: 'handR', parent: B_FORER, off: [0, -0.255, 0], axis: -1 },
  /* 13 */ { name: 'thighL', parent: B_PELVIS, off: [0.098, -0.055, 0], axis: -1 },
  /* 14 */ { name: 'shinL', parent: B_THIGHL, off: [0, -0.435, 0], axis: -1 },
  /* 15 */ { name: 'footL', parent: B_SHINL, off: [0, -0.415, 0], axis: -1 },
  /* 16 */ { name: 'toeL', parent: B_FOOTL, off: [0, -0.032, 0.118], axis: -1 },
  /* 17 */ { name: 'thighR', parent: B_PELVIS, off: [-0.098, -0.055, 0], axis: -1 },
  /* 18 */ { name: 'shinR', parent: B_THIGHR, off: [0, -0.435, 0], axis: -1 },
  /* 19 */ { name: 'footR', parent: B_SHINR, off: [0, -0.415, 0], axis: -1 },
  /* 20 */ { name: 'toeR', parent: B_FOOTR, off: [0, -0.032, 0.118], axis: -1 },
];

// Bind-pose local rotations (radians, XYZ).  This is a REAL bladed fighting
// stance with the weapon shouldered: hips and chest turned so the support
// shoulder leads, spine leaned ~8 deg forward, head brought back around onto
// the stock.  Keeping the bind pose identical to the pose the soldier holds
// 90% of the time is what stops linear-blend skinning from folding.
// Both arms and both hands stay at zero here — they are solved by IK at build
// time so the stock lands in the shoulder pocket, the firing hand lands on the
// pistol grip and the support hand lands on the handguard, with elbows bent
// 65-75 deg.  There is no straight-arm anywhere in this rig.
const BIND_ROT = {
  [B_PELVIS]: [0.010, -0.100, 0],
  [B_SPINE]: [0.055, -0.140, 0.012],
  [B_CHEST]: [0.065, -0.330, 0.022],
  [B_NECK]: [-0.075, 0.255, -0.030],
  [B_HEAD]: [-0.030, 0.205, 0.055],
  [B_CLAVL]: [0.020, -0.130, -0.085],
  [B_CLAVR]: [-0.015, 0.135, 0.100],
  [B_THIGHL]: [-0.055, 0.135, -0.025],
  [B_SHINL]: [0.080, 0, 0],
  [B_FOOTL]: [-0.032, 0, 0],
  [B_THIGHR]: [0.045, 0.045, 0.025],
  [B_SHINR]: [0.062, 0, 0],
  [B_FOOTR]: [-0.024, 0, 0],
};

// Rifle geometry is authored in "rifle space": origin at the pistol-grip web,
// +Z toward the muzzle, +Y up.  These points drive the whole grip.
const RIFLE_HANDGUARD = new THREE.Vector3(0.000, 0.032, 0.248);  // support-hand wrist
const RIFLE_MUZZLE = new THREE.Vector3(0.000, 0.098, 0.672);
const RIFLE_BUTT = new THREE.Vector3(0.000, 0.062, -0.206);      // recoil pad centre
const RIFLE_WRIST = new THREE.Vector3(-0.012, -0.046, -0.022);   // firing-hand wrist
// grip axis (web -> heel of the pistol grip) in rifle space
const RIFLE_GRIPDIR = new THREE.Vector3(0.000, -0.9553, -0.2955);

// Palette — authored in sRGB, converted to the linear working space on upload.
//
// These are TRUE ALBEDO values, not "colours that looked right in the viewport".
// The shader below normalises the tiling albedo map to unit mean, so what is
// written here is exactly what the surface reflects; previously the tan cloth
// map multiplied these a second time and knocked every soldier down to roughly
// 40% of the intended value, which is why a backlit man collapsed into one
// charcoal mass.  A sun-bleached desert uniform is a genuinely light material —
// around 0.35 linear — and it has to be authored that way to read at 30 m.
//
// ALBEDO_SCALE is the one exposure knob for the whole squad.  The palette below
// is authored as honest material reflectance; this scales it in LINEAR space to
// sit under this level's key intensity, so a sunlit soldier lands just below a
// sunlit stucco wall instead of punching a white hole in the frame.  Scaling in
// linear preserves every relationship in the palette — value separation between
// kit, plate, boot and skin survives intact.
const ALBEDO_SCALE = 0.33;

const COL = {
  uniform: 0xa2977a, uniformDk: 0x8d846b, uniformLt: 0xb4a98a,
  carrier: 0x635d4a, pouch: 0x5b5544, webbing: 0x4e493a,
  skin: 0xba8f6b, skinDk: 0x9d7452, skinLt: 0xcaa07b, balaclava: 0x565046,
  brow: 0x8f6a4c, beard: 0x4b4038,
  helmet: 0x8a8470, helmetCover: 0x968d74,
  nvg: 0x525046, rail: 0x4c4a41, headset: 0x48463e,
  boot: 0x4f483c, glove: 0x524b40, knee: 0x4b463b,
  // The weapon is phosphated steel and glass-filled polymer, not chrome: its
  // base colour doubles as metal reflectance, so it stays a stop below the kit
  // or the carbine turns into a white bar hanging off a dark man.
  gunBody: 0x585b61, gunPoly: 0x54524a, gunMag: 0x4d4f46,
  optic: 0x45454b, lens: 0x33445e, antenna: 0x474740,
  strapDk: 0x4d483c, sole: 0x3d382f, butt: 0x424139,
  scarf: 0xa3977d, scarfDk: 0x8c8269,
  boonie: 0x9e9476, boonieDk: 0x8a816b,
  // a mid tone used to break the leg into readable bands instead of one slab
  legBand: 0x807761, legShadow: 0x7b7259,
};

// Kit variants.  Four different silhouettes in a squad: helmet vs boonie vs
// bare balaclava, plate carrier vs slick chest rig, sleeves rolled or down.
// Colours shift a full value step between them — one bleached tan, one olive,
// one grey, one dark coyote — so no two men in a frame read as the same asset.
const KITS = [
  { // 0 — full kit: ballistic helmet, NVG shroud, plate carrier, sleeves down
    head: 'helmet', nvg: true, headset: true, radio: true, hydration: true,
    rig: 'carrier', sleeves: 'down', scarf: false, knee: true, shoulder: true,
    beard: false, eyepro: true,
    uni: 0xa2977a, uniDk: 0x8d846b, uniLt: 0xb4a98a, rigCol: 0x655f4a, pouchCol: 0x5c5644,
  },
  { // 1 — boonie hat, slick chest rig, sleeves rolled, olive drab
    head: 'boonie', nvg: false, headset: false, radio: true, hydration: false,
    rig: 'chestrig', sleeves: 'up', scarf: false, knee: false, shoulder: false,
    beard: true, eyepro: false,
    uni: 0x8d8f68, uniDk: 0x797b58, uniLt: 0x9fa179, rigCol: 0x53553d, pouchCol: 0x4b4d36,
  },
  { // 2 — bare balaclava + shemagh, plate carrier, sleeves down, dusty grey
    head: 'bare', nvg: false, headset: true, radio: false, hydration: true,
    rig: 'carrier', sleeves: 'down', scarf: true, knee: true, shoulder: false,
    beard: false, eyepro: true,
    uni: 0x89836c, uniDk: 0x75705c, uniLt: 0x99937b, rigCol: 0x565241, pouchCol: 0x605b4a,
  },
  { // 3 — stripped helmet, chest rig, sleeves rolled, scarf, pale tan
    head: 'helmet', nvg: false, headset: false, radio: true, hydration: false,
    rig: 'chestrig', sleeves: 'up', scarf: true, knee: true, shoulder: true,
    beard: true, eyepro: false,
    uni: 0xa19367, uniDk: 0x8b7f59, uniLt: 0xb1a37a, rigCol: 0x6a6348, pouchCol: 0x605a42,
  },
];
const KIT_COUNT = KITS.length;

// [absolute roughness, metalness].  aMR.x is no longer a multiplier on whatever
// the shared roughness map happened to contain — it is the surface's real
// roughness, with the map allowed only +/-20% of variation on top.  That is the
// only way to guarantee a matte helmet: the gunmetal roughness map is polished,
// and multiplying by it was what gave every soldier a wet plastic dome.
const MR = {
  cloth: [0.95, 0.00], carrier: [0.90, 0.00], webbing: [0.97, 0.00],
  skin: [0.62, 0.00], boot: [0.74, 0.00], helmet: [0.72, 0.00],
  helmetCloth: [0.90, 0.00],
  gunMetal: [0.42, 0.88], gunPoly: [0.56, 0.00], lens: [0.14, 0.25],
  rubber: [0.86, 0.00],
};

// Pose archetypes.  Every soldier is dealt one of these, and it changes how he
// stands, where the weapon sits, how far he is turned off his own facing and how
// wide he scans.  Without this a squad is five copies of one animation clip.
//   carry  : weapon-down amount 0 (shouldered) .. 1 (slung low)
//   lean   : lateral weight shift, radians into the spine
//   twist  : torso rotation off the feet
//   cant   : weapon yawed across the chest at the firing wrist
//   scan   : head sweep multiplier
//   crouch : posture floor
const POSE = [
  { carry: 0.14, lean: -0.05, twist: 0.13, cant: -0.13, scan: 0.55, crouch: 0.00, elbow: 0.07, chin: 0.06 },
  { carry: 0.60, lean: 0.09, twist: -0.09, cant: 0.09, scan: 0.85, crouch: 0.00, elbow: -0.05, chin: -0.04 },
  { carry: 0.28, lean: 0.30, twist: 0.24, cant: -0.18, scan: 0.35, crouch: 0.62, elbow: 0.11, chin: 0.10 },
  { carry: 0.46, lean: -0.14, twist: -0.30, cant: 0.05, scan: 1.35, crouch: 0.00, elbow: 0.00, chin: -0.02 },
  { carry: 0.82, lean: 0.13, twist: 0.06, cant: 0.15, scan: 0.95, crouch: 0.00, elbow: -0.11, chin: 0.03 },
];
const POSE_COUNT = POSE.length;

// AI behaviour states.
const S_IDLE = 0, S_PATROL = 1, S_SUSPECT = 2, S_ALERT = 3, S_ENGAGE = 4,
  S_SUPPRESSED = 5, S_RELOAD = 6, S_SEEKCOVER = 7, S_FLANK = 8, S_DEAD = 9;
const STATE_NAMES = ['idle', 'patrol', 'suspect', 'alert', 'engage', 'suppressed',
  'reload', 'seek-cover', 'flank', 'dead'];

// How far the muzzle drops, in radians about the firing wrist, at full "carry"
// (weapon down / compressed ready).  The support arm follows through grip IK.
const CARRY_DROP = 0.58;
// Share of the aim solution taken by the weapon pivoting at the wrist rather
// than by bending the spine.  Real shooters elevate with the arms first.
const AIM_WRIST_SHARE = 0.62;

const MAG_SIZE = 30;
const BURST_RPM = 640;
const ENEMY_DAMAGE = 11;
const CORPSE_LINGER = 17.0;
const CORPSE_FADE = 2.6;
const RESPAWN_DELAY = 9.0;

/* =========================================================================== *
 *  helpers
 * =========================================================================== */

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
const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
const lerp = (a, b, t) => a + (b - a) * t;
function sstep(e0, e1, x) {
  if (e1 === e0) return x < e0 ? 0 : 1;
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}
function angDelta(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}
// critically-damped implicit spring — unconditionally stable at any dt
function springX(cur, vel, target, omega, dt) {
  const f = 1 + 2 * dt * omega;
  const oo = omega * omega, hoo = dt * oo, hhoo = dt * hoo;
  const det = 1 / (f + hhoo);
  return (f * cur + dt * vel + hhoo * target) * det;
}
function springV(cur, vel, target, omega, dt) {
  const f = 1 + 2 * dt * omega;
  const oo = omega * omega, hoo = dt * oo, hhoo = dt * hoo;
  const det = 1 / (f + hhoo);
  return (vel + hoo * (target - cur)) * det;
}

/* =========================================================================== *
 *  ENEMIES
 * =========================================================================== */

export class Enemies {

  constructor(game) {
    this.g = game;
    this.THREE = THREE;
    const cfg = game?.config ?? {};
    this.cfgAI = cfg.ai ?? { count: 14, sightRange: 90, fov: 120, reactionTime: 0.28 };

    this.count = Math.max(0, this.cfgAI.count | 0);
    this.sightRange = this.cfgAI.sightRange ?? 90;
    this.fovCos = Math.cos((this.cfgAI.fov ?? 120) * 0.5 * DEG);
    this.reactionTime = this.cfgAI.reactionTime ?? 0.28;

    this.rand = mulberry32(0x51ac9d);
    this.units = [];
    this.handles = [];
    this._handleMap = new Map();

    this.stats = { alive: 0, dead: 0, tris: 0, drawCalls: 0 };

    // ---------------------------------------------------------------- scratch
    this._v1 = new THREE.Vector3(); this._v2 = new THREE.Vector3();
    this._v3 = new THREE.Vector3(); this._v4 = new THREE.Vector3();
    this._v5 = new THREE.Vector3(); this._v6 = new THREE.Vector3();
    this._v7 = new THREE.Vector3(); this._v8 = new THREE.Vector3();
    this._v9 = new THREE.Vector3(); this._v10 = new THREE.Vector3();
    this._v11 = new THREE.Vector3(); this._v12 = new THREE.Vector3();
    this._q1 = new THREE.Quaternion(); this._q2 = new THREE.Quaternion();
    this._q3 = new THREE.Quaternion();
    this._m1 = new THREE.Matrix4(); this._m2 = new THREE.Matrix4();
    this._nm3 = new THREE.Matrix3();
    this._e1 = new THREE.Euler();
    this._ray = new THREE.Ray();
    this._boxScratch = new THREE.Box3();
    this._col = new THREE.Color();
    this._playerPos = new THREE.Vector3();
    this._playerEye = new THREE.Vector3();
    this._lastPlayerPos = new THREE.Vector3();
    this._playerVel = new THREE.Vector3();
    this._wScratch = new Float32Array(6);
    this._rqW = [];
    for (let i = 0; i < BONE_COUNT; i++) this._rqW.push(new THREE.Quaternion());

    // shot / damage payload rings — reused but deep enough that a listener
    // holding one for a bullet's lifetime never sees it stomped
    this._shotRing = [];
    for (let i = 0; i < 16; i++) {
      this._shotRing.push({
        origin: new THREE.Vector3(), dir: new THREE.Vector3(),
        weapon: 'enemy_rifle', source: 'enemy', enemy: null,
        damage: ENEMY_DAMAGE, spread: 0, tracer: false,
      });
    }
    this._shotIdx = 0;
    this._dmgRing = [];
    for (let i = 0; i < 8; i++) {
      this._dmgRing.push({
        amount: 0, fromDir: new THREE.Vector3(), from: new THREE.Vector3(),
        source: 'enemy', enemy: null, point: new THREE.Vector3(),
      });
    }
    this._dmgIdx = 0;
    this._killMsg = { enemy: null, headshot: false, distance: 0 };
    this._reloadMsg = { phase: 'start', enemy: null, source: 'enemy' };

    // --------------------------------------------------------------- container
    this.group = new THREE.Group();
    this.group.name = 'enemies';
    game?.scene?.add(this.group);

    // ------------------------------------------------------------ cover cache
    this._cover = [];
    this._coverFaces = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    this._coverBuilt = false;
    this._coverCheckAt = 0;
    this._lastColliderCount = -1;
    this._posts = [];
    this._physWarned = false;

    this._t = 0;
    this._alarm = 0;
    this._alarmPos = new THREE.Vector3();
    this._hasAlarmPos = false;

    // Shared sun-direction uniform for the character kicker.  Seeded with the
    // art-direction key (low warm sun raking in from the west) so the soldiers
    // read correctly even if Sky never publishes one.
    this._sunU = { value: new THREE.Vector3(-0.72, 0.34, -0.60).normalize() };
    this._sunPollT = 0;

    this._buildMaterials();
    this._buildRig();
    this._buildFX();
  }

  /* ======================================================================== *
   *  MATERIALS
   * ======================================================================== */

  /**
   * Mean LINEAR colour of a tiling albedo map.  Used to divide the map back out
   * of the albedo so the texture contributes detail and grime but not a second
   * global tint — see the map_fragment patch below.
   */
  _mapMean(tex) {
    const out = new THREE.Vector3(0.5, 0.5, 0.5);
    const d = tex?.image?.data;
    if (!d || !d.length || !d.BYTES_PER_ELEMENT) return out;
    const px = (d.length / 4) | 0;
    if (px < 4) return out;
    const step = Math.max(1, (px / 8192) | 0);
    const srgb = tex.colorSpace === THREE.SRGBColorSpace;
    const s2l = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < px; i += step) {
      const o = i * 4;
      let cr = d[o] / 255, cg = d[o + 1] / 255, cb = d[o + 2] / 255;
      if (srgb) { cr = s2l(cr); cg = s2l(cg); cb = s2l(cb); }
      r += cr; g += cg; b += cb; n++;
    }
    if (!n) return out;
    out.set(
      clamp(r / n, 0.03, 1.0),
      clamp(g / n, 0.03, 1.0),
      clamp(b / n, 0.03, 1.0));
    return out;
  }

  _makeBase(surfaceKey, envI, repeat) {
    const M = this.g?.materials;
    let src = null;
    try {
      src = M?.make ? M.make(surfaceKey, { repeat: [repeat, repeat] })
        : (M?.get ? M.get(surfaceKey) : null);
    } catch (e) { src = null; }

    // 1 / mean albedo of the shared map, so the texture modulates around 1.0.
    const mean = this._mapMean(src?.map);
    const norm = { value: new THREE.Vector3(1 / mean.x, 1 / mean.y, 1 / mean.z) };

    const mat = new THREE.MeshStandardMaterial({
      name: 'enemy:' + surfaceKey,
      map: src?.map ?? null,
      normalMap: src?.normalMap ?? null,
      roughnessMap: src?.roughnessMap ?? null,
      metalnessMap: null,               // metalness comes from the aMR attribute
      aoMap: src?.aoMap ?? null,
      color: 0xffffff,
      roughness: 1.0,
      metalness: 1.0,
      vertexColors: true,
      envMapIntensity: envI,
      dithering: true,
      aoMapIntensity: 0.85,
    });
    if (mat.normalMap) mat.normalScale.set(0.85, 0.85);

    // Per-vertex roughness multiplier + absolute metalness.  One material then
    // carries cordura, skin, rubber and phosphated steel at once, which is what
    // keeps a whole soldier down to two draw calls.
    //
    // On top of that this shader adds the two things a character needs that a
    // stock MeshStandardMaterial will not give you in a golden-hour scene:
    //
    //   * a CHARACTER FILL — a hemispheric term, cool skylight from above and
    //     warm dust bounce from below, added straight into indirect diffuse so
    //     the shadow side of the carrier / uniform never crushes to a black
    //     cutout at 40 m.  It is proportional to albedo, so it lifts the floor
    //     without washing out anything already lit.
    //
    //   * a SKY-SIDE RIM / KICKER — a fresnel edge weighted toward the upper
    //     hemisphere, in skylight blue.  This is what separates the helmet, the
    //     shoulder line and the weapon silhouette from a sunlit wall behind.
    mat.onBeforeCompile = (shader) => {
      // One shared uniform object across every soldier material, so a single
      // per-frame write in update() re-points the kicker on the whole squad.
      shader.uniforms.bsSunDir = this._sunU;
      shader.uniforms.bsMapNorm = norm;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nattribute vec2 aMR;\nvarying vec2 vMR;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n\tvMR = aMR;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>',
          '#include <common>\nvarying vec2 vMR;\nuniform vec3 bsSunDir;\nuniform vec3 bsMapNorm;')
        // ---- albedo: the map is DETAIL, the vertex colour is the ALBEDO -------
        // One shared cloth/gunmetal texture has to serve uniform, webbing, boot
        // leather, skin and phosphated steel.  Multiplying its tan average into
        // all of them tinted and darkened every material on the man by the same
        // amount, which is precisely what made a squad read as one charcoal
        // silhouette.  Normalising the map to unit mean keeps the weave, the
        // grime and the bleaching, and throws away the global tint.
        .replace('#include <map_fragment>', `#ifdef USE_MAP
	vec4 bsTexel = texture2D( map, vMapUv );
	vec3 bsDetail = mix( vec3( 1.0 ), bsTexel.rgb * bsMapNorm, 0.80 );
	diffuseColor.rgb *= clamp( bsDetail, vec3( 0.32 ), vec3( 1.50 ) );
#endif`)
        // ---- roughness: aMR.x is absolute, the map is +/-20% variation --------
        .replace('#include <roughnessmap_fragment>',
          '#include <roughnessmap_fragment>\n\troughnessFactor = clamp( vMR.x * ( 0.80 + 0.42 * roughnessFactor ), 0.055, 1.0 );')
        .replace('#include <metalnessmap_fragment>',
          '#include <metalnessmap_fragment>\n\tmetalnessFactor = vMR.y;')
        .replace('#include <aomap_fragment>', `#include <aomap_fragment>
	{
		// view-space normal back into world space (viewMatrix is orthonormal)
		vec3 bsWN = normalize( vec3(
			dot( viewMatrix[ 0 ].xyz, geometryNormal ),
			dot( viewMatrix[ 1 ].xyz, geometryNormal ),
			dot( viewMatrix[ 2 ].xyz, geometryNormal ) ) );
		float bsHemi = clamp( bsWN.y * 0.5 + 0.5, 0.0, 1.0 );
		float bsNL = dot( bsWN, bsSunDir );

		// CHARACTER FILL — strong warm dust bounce from below, pale warm-grey
		// skylight from above.  Golden hour over a sand compound has almost no
		// blue in it at ground level; the old deep-blue upper term is what turned
		// every soldier into a steel-blue cut-out standing in a warm street.
		vec3 bsFill = mix( vec3( 0.1080, 0.0730, 0.0390 ), vec3( 0.0650, 0.0672, 0.0735 ), bsHemi );
		reflectedLight.indirectDiffuse += diffuseColor.rgb * bsFill;

		// SUN WRAP — a figure in open desert shade still catches a wide warm
		// wrap off the key.  Half-lambert, weak, and warm: this is what keeps the
		// shadow side of the uniform in the same colour family as the world.
		float bsWrap = clamp( bsNL * 0.5 + 0.5, 0.0, 1.0 );
		reflectedLight.indirectDiffuse += diffuseColor.rgb * vec3( 0.1550, 0.1010, 0.0530 ) * ( bsWrap * bsWrap );

		// SILHOUETTE EDGE — a tight, quiet cool sky edge plus a broad warm sun
		// kicker.  The warm one carries the read: it is what puts a lit contour
		// on a helmet, a shoulder and a weapon standing against a sunlit wall.
		float bsF = 1.0 - clamp( dot( geometryNormal, geometryViewDir ), 0.0, 1.0 );
		float bsF2 = bsF * bsF;
		float bsF3 = bsF2 * bsF;
		float bsEdge = bsF2 * bsF2;
		float bsSky = clamp( bsWN.y * 0.75 + 0.30, 0.0, 1.0 );
		float bsSun = clamp( bsNL, 0.0, 1.0 );
		bsSun = bsSun * bsSun;
		reflectedLight.indirectSpecular += vec3( 0.260, 0.345, 0.470 ) * ( bsEdge * bsSky * 0.30 );
		reflectedLight.indirectSpecular += vec3( 1.000, 0.660, 0.330 ) * ( bsF3 * bsSun * 0.80 );
	}`);
    };
    mat.customProgramCacheKey = () => 'bs-enemy-mr-4';
    return mat;
  }

  _buildMaterials() {
    // Two prototypes; every soldier clones them so a corpse fades out on its own
    // without dragging the squad's opacity with it.  Identical parameters plus a
    // shared customProgramCacheKey means a single compiled program.
    // scene.environmentIntensity is 0.45 in this build, so these are pushed up
    // to land at a sane effective strength.  They came DOWN in round 3 because
    // the albedo is no longer double-darkened by the tiling map: the same env
    // strength on a correct albedo was blowing the kit out and, worse, pouring
    // the sky's blue into every shadowed uniform.
    this.matSoftProto = this._makeBase('cloth_tan', 0.92, 3);
    this.matHardProto = this._makeBase('gunmetal', 1.40, 4);
  }

  /* ======================================================================== *
   *  RIG — bones, bind pose, skinned geometry
   * ======================================================================== */

  _buildRig() {
    // ---- 1. a throwaway bone tree used purely to resolve the bind pose ------
    const bones = [];
    for (let i = 0; i < BONE_COUNT; i++) {
      const d = BONES[i];
      const b = new THREE.Bone();
      b.name = d.name;
      b.position.set(d.off[0], d.off[1], d.off[2]);
      const r = BIND_ROT[i];
      if (r) b.rotation.set(r[0], r[1], r[2]);
      bones.push(b);
    }
    for (let i = 0; i < BONE_COUNT; i++) {
      const p = BONES[i].parent;
      if (p >= 0) bones[p].add(bones[i]);
    }
    const holder = new THREE.Object3D();
    holder.add(bones[0]);
    holder.updateMatrixWorld(true);

    // bone lengths (distance to the first child offset)
    this._boneLen = new Float32Array(BONE_COUNT);
    for (let i = 0; i < BONE_COUNT; i++) {
      let L = 0;
      for (let j = 0; j < BONE_COUNT; j++) {
        if (BONES[j].parent === i) {
          const o = BONES[j].off;
          const l = Math.hypot(o[0], o[1], o[2]);
          if (l > L) L = l;
        }
      }
      this._boneLen[i] = L || 0.16;
    }

    // ---- 2. weapon mount, then BOTH arms solved onto it ---------------------
    //
    // The weapon is placed first, from the body: the recoil pad goes into the
    // right shoulder pocket, the bore sits level and slightly canted inboard.
    // Only then are the arms solved onto it.  That ordering is the whole reason
    // the stance reads as a shouldered carbine instead of a scarecrow holding a
    // stick at arm's length — the elbows end up wherever anatomy puts them, and
    // anatomy puts them at 65-75 deg of flexion.
    const shoulderRW = new THREE.Vector3().setFromMatrixPosition(bones[B_ARMR].matrixWorld);
    const shoulderLW = new THREE.Vector3().setFromMatrixPosition(bones[B_ARML].matrixWorld);
    const pocket = shoulderRW.clone().add(new THREE.Vector3(0.034, -0.028, 0.086));

    const rifleQ = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(0.030, -0.045, -0.115, 'YXZ'));           // level, canted inboard
    const rifleP = pocket.clone().sub(RIFLE_BUTT.clone().applyQuaternion(rifleQ));
    const rifleM = new THREE.Matrix4().compose(rifleP, rifleQ, new THREE.Vector3(1, 1, 1));
    this._rifleBindMatrix = rifleM;

    const hgW = RIFLE_HANDGUARD.clone().applyMatrix4(rifleM);
    const muzW = RIFLE_MUZZLE.clone().applyMatrix4(rifleM);
    const wristRW = RIFLE_WRIST.clone().applyMatrix4(rifleM);
    const boreW = new THREE.Vector3(0, 0, 1).applyQuaternion(rifleQ).normalize();
    const gripDirW = RIFLE_GRIPDIR.clone().applyQuaternion(rifleQ).normalize();
    const rifleUpW = new THREE.Vector3(0, 1, 0).applyQuaternion(rifleQ).normalize();
    const rifleRightW = new THREE.Vector3(1, 0, 0).applyQuaternion(rifleQ).normalize();

    // firing arm: elbow driven down, out to the soldier's right and slightly
    // back — the classic tucked shooting elbow, never flared to shoulder height
    const poleR = shoulderRW.clone().add(new THREE.Vector3(-0.52, -0.80, -0.26));
    this._ikTwoBone(bones[B_ARMR], bones[B_FORER],
      this._boneLen[B_ARMR], this._boneLen[B_FORER], wristRW, poleR, true);
    this._aimBoneWorld(bones[B_HANDR], gripDirW, rifleRightW, -1);
    bones[B_HANDR].updateMatrixWorld(true);

    // support arm: elbow down and slightly forward, tucked under the handguard
    const poleL = shoulderLW.clone().add(new THREE.Vector3(0.30, -0.86, 0.10));
    this._ikTwoBone(bones[B_ARML], bones[B_FOREL],
      this._boneLen[B_ARML], this._boneLen[B_FOREL], hgW, poleL, true);
    // support hand wraps the angled foregrip: fingers follow the foregrip axis
    const fgDir = new THREE.Vector3(0, -0.9004, 0.4350).applyQuaternion(rifleQ).normalize();
    this._aimBoneWorld(bones[B_HANDL], fgDir, rifleUpW, -1);
    holder.updateMatrixWorld(true);

    // ---- 3. capture the bind pose ------------------------------------------
    this._bindWorld = [];
    this._bindPos = [];
    this._bindDir = [];
    this._bindQuat = [];
    for (let i = 0; i < BONE_COUNT; i++) {
      const m = bones[i].matrixWorld.clone();
      this._bindWorld.push(m);
      this._bindPos.push(new THREE.Vector3().setFromMatrixPosition(m));
      const q = new THREE.Quaternion().setFromRotationMatrix(m);
      this._bindQuat.push(q);
      this._bindDir.push(new THREE.Vector3(0, BONES[i].axis, 0).applyQuaternion(q).normalize());
    }
    this._bindInv = this._bindWorld.map(m => new THREE.Matrix4().copy(m).invert());

    // bind-pose local eulers, used as the animation rest pose
    this._restEuler = new Float32Array(BONE_COUNT * 3);
    const eTmp = new THREE.Euler();
    for (let i = 0; i < BONE_COUNT; i++) {
      eTmp.setFromQuaternion(bones[i].quaternion, 'XYZ');
      this._restEuler[i * 3] = eTmp.x;
      this._restEuler[i * 3 + 1] = eTmp.y;
      this._restEuler[i * 3 + 2] = eTmp.z;
    }

    // muzzle / handguard in handR local space — one matrix multiply at runtime
    // gives the world muzzle no matter what the arm is doing
    const rotInvR = new THREE.Matrix4().extractRotation(this._bindInv[B_HANDR]);
    this._muzzleLocal = muzW.clone().applyMatrix4(this._bindInv[B_HANDR]);
    this._handguardLocal = hgW.clone().applyMatrix4(this._bindInv[B_HANDR]);
    // TRUE bore axis, not handguard->muzzle: the handguard sits 8 cm under the
    // barrel, so the old chord was 12 deg off and the aim closure converged onto
    // a weapon that visibly pointed below the target.
    this._rifleAxisLocal = boreW.clone().applyMatrix4(rotInvR).normalize();

    // World +X expressed in handR's local frame.  Post-multiplying handR by a
    // rotation about this axis pitches the WHOLE weapon about the firing wrist,
    // which is how a real shooter drops to low ready and rides recoil — the
    // support arm then follows for free through the grip IK.
    const qHandR = new THREE.Quaternion().setFromRotationMatrix(this._bindWorld[B_HANDR]);
    const qHandRInv = qHandR.clone().invert();
    this._carryAxis = new THREE.Vector3(1, 0, 0)
      .applyQuaternion(qHandRInv).normalize();
    // World +Y in handR's local frame — rotating about this yaws the weapon
    // across the chest at the wrist, which is the difference between "carbine
    // shouldered square" and "carbine canted across the body at compressed
    // ready".  One number per soldier, and no two men hold it the same way.
    this._cantAxis = new THREE.Vector3(0, 1, 0)
      .applyQuaternion(qHandRInv).normalize();
    this._qCarry = new THREE.Quaternion();
    this._qCant = new THREE.Quaternion();

    holder.remove(bones[0]);

    // ---- 4. geometry — one per kit variant ----------------------------------
    this.geometries = [];
    for (let k = 0; k < KIT_COUNT; k++) this.geometries.push(this._buildBody(KITS[k]));
    this.geometry = this.geometries[0];
    this.stats.tris = this.geometry.index.count / 3;
  }

  /**
   * Analytic two-bone IK.  bA/bB must be a parent/child pair whose local -Y
   * points at the child.  Writes local quaternions and refreshes the subtree.
   */
  _ikTwoBone(bA, bB, lenA, lenB, target, pole, updateWorld) {
    const pA = this._v1.setFromMatrixPosition(bA.matrixWorld);
    const d = this._v2.copy(target).sub(pA);
    let dist = d.length();
    if (dist < 1e-5) { d.set(0, -1, 0); dist = 1e-5; }
    d.multiplyScalar(1 / dist);
    const maxL = (lenA + lenB) * 0.9975;
    const minL = Math.abs(lenA - lenB) * 1.02 + 1e-3;
    dist = clamp(dist, minL, maxL);

    const cosA = clamp((lenA * lenA + dist * dist - lenB * lenB) / (2 * lenA * dist), -1, 1);
    const angA = Math.acos(cosA);

    const u = this._v3.copy(pole).sub(pA);
    u.addScaledVector(d, -u.dot(d));
    if (u.lengthSq() < 1e-8) {
      u.set(0, 0, 1).addScaledVector(d, -d.z);
      if (u.lengthSq() < 1e-8) u.set(1, 0, 0).addScaledVector(d, -d.x);
    }
    u.normalize();

    const dirA = this._v4.copy(d).multiplyScalar(Math.cos(angA)).addScaledVector(u, Math.sin(angA)).normalize();
    const elbow = this._v5.copy(pA).addScaledVector(dirA, lenA);
    const dirB = this._v6.copy(target).sub(elbow);
    if (dirB.lengthSq() < 1e-10) dirB.copy(dirA); else dirB.normalize();

    // dirA/dirB live in _v4/_v6, the pole hint in _v3 — _aimBoneWorld only uses
    // _v7.._v9 plus _m1/_m2/_q1/_q2, so nothing here is clobbered.
    this._aimBoneWorld(bA, dirA, u, -1);
    bA.updateMatrixWorld(true);
    this._aimBoneWorld(bB, dirB, u, -1);
    if (updateWorld !== false) bB.updateMatrixWorld(true);
  }

  /**
   * Orient a bone in world space so its local (axisSign * Y) axis lies along
   * `dir`, with the joint bulging toward `poleU` (used as the local -Z hint).
   */
  _aimBoneWorld(bone, dir, poleU, axisSign) {
    const y = this._v7.copy(dir);
    if (axisSign < 0) y.multiplyScalar(-1);
    y.normalize();
    const z = this._v8.copy(poleU).multiplyScalar(-1);
    z.addScaledVector(y, -z.dot(y));
    if (z.lengthSq() < 1e-9) {
      z.set(0, 0, 1).addScaledVector(y, -y.z);
      if (z.lengthSq() < 1e-9) z.set(1, 0, 0).addScaledVector(y, -y.x);
    }
    z.normalize();
    const x = this._v9.crossVectors(y, z).normalize();
    this._m1.makeBasis(x, y, z);
    this._q1.setFromRotationMatrix(this._m1);
    if (bone.parent) {
      this._m2.extractRotation(bone.parent.matrixWorld);
      this._q2.setFromRotationMatrix(this._m2).invert();
      bone.quaternion.copy(this._q2).multiply(this._q1);
    } else {
      bone.quaternion.copy(this._q1);
    }
  }

  /* ======================================================================== *
   *  GEOMETRY BUILDER
   *
   *  Everything is authored in BIND-POSE WORLD SPACE (soldier faces +Z, feet at
   *  y = 0), which makes the numbers readable, then given up to three skin
   *  influences per vertex.
   * ======================================================================== */

  _newBuilder() {
    return { pos: [], nor: [], uv: [], col: [], mr: [], si: [], sw: [], idxSoft: [], idxHard: [], v: 0 };
  }

  _emit(b, geo, mtx, opt) {
    const pa = geo.attributes.position, na = geo.attributes.normal;
    const index = geo.index;
    const nm = this._nm3.getNormalMatrix(mtx);
    const base = b.v;
    const P = this._v11, N = this._v12;
    const w = this._wScratch;
    const uvScale = opt.uv ?? 2.4;
    const c = this._col.setHex(opt.color, THREE.SRGBColorSpace);
    const cr = c.r * ALBEDO_SCALE, cg = c.g * ALBEDO_SCALE, cb = c.b * ALBEDO_SCALE;
    const mr0 = opt.mr ? opt.mr[0] : 1, mr1 = opt.mr ? opt.mr[1] : 0;

    for (let i = 0; i < pa.count; i++) {
      P.fromBufferAttribute(pa, i).applyMatrix4(mtx);
      N.fromBufferAttribute(na, i).applyMatrix3(nm).normalize();

      b.pos.push(P.x, P.y, P.z);
      b.nor.push(N.x, N.y, N.z);

      // triplanar UV picked by the dominant normal axis — constant texel density
      // across the whole body without authoring a real unwrap
      const ax = Math.abs(N.x), ay = Math.abs(N.y), az = Math.abs(N.z);
      let uu, vv;
      if (ay >= ax && ay >= az) { uu = P.x; vv = P.z; }
      else if (ax >= az) { uu = P.z; vv = P.y; }
      else { uu = P.x; vv = P.y; }
      b.uv.push(uu * uvScale, vv * uvScale);

      b.col.push(cr, cg, cb);
      b.mr.push(mr0, mr1);

      this._weights(P.x, P.y, P.z, opt, w);
      b.si.push(w[0], w[2], w[4], 0);
      b.sw.push(w[1], w[3], w[5], 0);
    }

    const dst = opt.hard ? b.idxHard : b.idxSoft;
    if (index) { for (let i = 0; i < index.count; i++) dst.push(base + index.getX(i)); }
    else { for (let i = 0; i < pa.count; i++) dst.push(base + i); }
    b.v += pa.count;
  }

  /** Skin weights for one bind-space point.  Writes [i0,w0,i1,w1,i2,w2]. */
  _weights(x, y, z, opt, out) {
    const bi = opt.bone;
    const mode = opt.blend;

    if (mode === 'torso') {
      // pelvis -> spine -> chest driven by height, so the plate carrier deforms
      // as one continuous slab instead of three stacked boxes
      const yS = this._bindPos[B_SPINE].y, yC = this._bindPos[B_CHEST].y, yP = this._bindPos[B_PELVIS].y;
      if (y >= yC) {
        const t = sstep(yC, yC + 0.14, y);
        out[0] = B_CHEST; out[1] = 0.72 + 0.28 * t;
        out[2] = B_SPINE; out[3] = 1 - out[1];
        out[4] = 0; out[5] = 0;
      } else if (y >= yS) {
        const t = sstep(yS, yC, y);
        let a = 0.16 + 0.56 * t;                 // chest
        let c2 = 0.32 * (1 - t);                 // pelvis
        let s = 1 - a - c2;                      // spine
        if (s < 0.05) { s = 0.05; const k = (1 - s) / (a + c2); a *= k; c2 *= k; }
        out[0] = B_CHEST; out[1] = a;
        out[2] = B_SPINE; out[3] = s;
        out[4] = B_PELVIS; out[5] = c2;
      } else if (y >= yP - 0.04) {
        const t = sstep(yP - 0.04, yS, y);
        out[0] = B_SPINE; out[1] = 0.18 + 0.52 * t;
        out[2] = B_PELVIS; out[3] = 1 - out[1];
        out[4] = 0; out[5] = 0;
      } else {
        out[0] = B_PELVIS; out[1] = 1; out[2] = 0; out[3] = 0; out[4] = 0; out[5] = 0;
      }
      return;
    }

    if (mode === 'limb') {
      const o = this._bindPos[bi], d = this._bindDir[bi], L = this._boneLen[bi];
      const t = ((x - o.x) * d.x + (y - o.y) * d.y + (z - o.z) * d.z) / L;
      const parent = BONES[bi].parent;
      const child = (opt.child !== undefined) ? opt.child : this._firstChild(bi);
      const wp = (parent >= 0) ? 0.46 * sstep(0.20, -0.06, t) : 0;
      const wc = (child >= 0) ? 0.46 * sstep(0.80, 1.06, t) : 0;
      const ws = Math.max(0.04, 1 - wp - wc);
      const s = 1 / (wp + wc + ws);
      out[0] = bi; out[1] = ws * s;
      out[2] = parent >= 0 ? parent : bi; out[3] = wp * s;
      out[4] = child >= 0 ? child : bi; out[5] = wc * s;
      return;
    }

    out[0] = bi; out[1] = 1; out[2] = 0; out[3] = 0; out[4] = 0; out[5] = 0;
  }

  _firstChild(i) {
    for (let j = 0; j < BONE_COUNT; j++) if (BONES[j].parent === i) return j;
    return -1;
  }

  // ---- primitive helpers (all in bind space) -----------------------------

  _box(b, w, h, d, px, py, pz, rx, ry, rz, opt, seg) {
    const g = new THREE.BoxGeometry(w, h, d, seg || 1, seg || 1, seg || 1);
    this._m2.makeRotationFromEuler(this._e1.set(rx || 0, ry || 0, rz || 0, 'YXZ'));
    this._m2.setPosition(px, py, pz);
    this._emit(b, g, this._m2, opt);
    g.dispose();
  }

  /** Tapered tube from A to B. */
  _tube(b, ax, ay, az, bx, by, bz, r0, r1, rad, opt, hSeg) {
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const len = Math.hypot(dx, dy, dz) || 1e-4;
    const g = new THREE.CylinderGeometry(r0, r1, len, rad || 10, hSeg || 3, false);
    g.translate(0, -len * 0.5, 0);             // y = 0 is now the A end
    this._v7.set(-dx / len, -dy / len, -dz / len);   // local +Y
    this._v8.set(0, 0, 1).addScaledVector(this._v7, -this._v7.z);
    if (this._v8.lengthSq() < 1e-8) this._v8.set(1, 0, 0).addScaledVector(this._v7, -this._v7.x);
    this._v8.normalize();
    this._v9.crossVectors(this._v7, this._v8).normalize();
    this._m2.makeBasis(this._v9, this._v7, this._v8);
    this._m2.setPosition(ax, ay, az);
    this._emit(b, g, this._m2, opt);
    g.dispose();
  }

  _sphere(b, r, px, py, pz, sx, sy, sz, opt, wSeg, hSeg, thetaL) {
    const g = new THREE.SphereGeometry(r, wSeg || 12, hSeg || 9, 0, Math.PI * 2, 0,
      thetaL === undefined ? Math.PI : thetaL);
    this._m2.makeScale(sx === undefined ? 1 : sx, sy === undefined ? 1 : sy, sz === undefined ? 1 : sz);
    this._m2.setPosition(px, py, pz);
    this._emit(b, g, this._m2, opt);
    g.dispose();
  }

  _cyl(b, r0, r1, h, px, py, pz, rx, ry, rz, opt, rad) {
    const g = new THREE.CylinderGeometry(r0, r1, h, rad || 10, 1, false);
    this._m2.makeRotationFromEuler(this._e1.set(rx || 0, ry || 0, rz || 0, 'YXZ'));
    this._m2.setPosition(px, py, pz);
    this._emit(b, g, this._m2, opt);
    g.dispose();
  }

  /* ---------------------------------------------------------------- BODY --- */

  _buildBody(K) {
    K = K || KITS[0];
    const b = this._newBuilder();
    const BP = this._bindPos;
    const C_UNI = K.uni, C_UNIDK = K.uniDk, C_UNILT = K.uniLt;
    const C_RIG = K.rigCol, C_POUCH = K.pouchCol;

    // =====================================================================
    // LEGS
    // =====================================================================
    for (let s = 0; s < 2; s++) {
      const th = s ? B_THIGHR : B_THIGHL, sh = s ? B_SHINR : B_SHINL;
      const ft = s ? B_FOOTR : B_FOOTL, to = s ? B_TOER : B_TOEL;
      const hip = BP[th], knee = BP[sh], ank = BP[ft], toe = BP[to];
      const sgn = s ? -1 : 1;

      // Thigh and shin carry real mass. The old 19 cm thigh read as a stilt
      // under a bulky carrier, which is half of why the legs came across as one
      // flat beige block hung off a dark torso.
      this._tube(b, hip.x, hip.y + 0.042, hip.z, knee.x, knee.y, knee.z,
        0.117, 0.087, 12, { bone: th, blend: 'limb', color: C_UNI, mr: MR.cloth, uv: 2.2 }, 4);
      // seat / glute mass so the hip-to-thigh join is not a hole
      this._sphere(b, 0.098, hip.x + sgn * 0.012, hip.y - 0.020, hip.z - 0.038, 1.0, 0.86, 1.06,
        { bone: th, blend: 'limb', child: -1, color: C_UNIDK, mr: MR.cloth, uv: 2.4 }, 10, 8);
      this._tube(b, knee.x, knee.y, knee.z, ank.x, ank.y + 0.100, ank.z,
        0.092, 0.068, 12, { bone: sh, blend: 'limb', color: C_UNIDK, mr: MR.cloth, uv: 2.2 }, 4);
      // calf mass, sitting high and to the back of the shin — a straight taper
      // from knee to ankle is a broomstick, and it reads as one at range
      this._sphere(b, 0.072,
        lerp(knee.x, ank.x, 0.30), lerp(knee.y, ank.y, 0.30), lerp(knee.z, ank.z, 0.30) - 0.020,
        1.00, 1.42, 1.08, { bone: sh, blend: 'limb', color: C_UNIDK, mr: MR.cloth, uv: 2.8 }, 10, 8);
      // knee joint volume under the pad
      this._sphere(b, 0.078, knee.x, knee.y, knee.z + 0.010, 1.02, 0.94, 1.06,
        { bone: sh, blend: 'limb', color: C_UNI, mr: MR.cloth, uv: 3.0 }, 10, 8);
      // mid-thigh retention strap: a dark band across the widest part of the leg
      this._cyl(b, 0.107, 0.101, 0.026, lerp(hip.x, knee.x, 0.52), lerp(hip.y, knee.y, 0.52),
        lerp(hip.z, knee.z, 0.52), 0.04, 0, 0,
        { bone: th, blend: 'limb', color: COL.webbing, mr: MR.carrier, uv: 4.6 }, 12);
      // calf band
      this._cyl(b, 0.093, 0.091, 0.024, ank.x, ank.y + 0.300, ank.z, 0, 0, 0,
        { bone: sh, blend: 'limb', color: COL.legBand, mr: MR.cloth, uv: 4.0 }, 12);
      // knee pad — a real silhouette read at range
      if (K.knee) {
        this._box(b, 0.129, 0.156, 0.084, knee.x + sgn * 0.004, knee.y - 0.052, knee.z + 0.058,
          -0.06, 0, 0, { bone: sh, blend: 'limb', hard: true, color: COL.knee, mr: MR.rubber, uv: 3.4 }, 2);
        this._box(b, 0.135, 0.026, 0.030, knee.x + sgn * 0.004, knee.y - 0.130, knee.z + 0.060,
          -0.06, 0, 0, { bone: sh, blend: 'limb', color: COL.webbing, mr: MR.carrier, uv: 4.6 });
      } else {
        // reinforced cloth knee: lower relief, but the tonal step is still there
        this._box(b, 0.119, 0.132, 0.064, knee.x + sgn * 0.004, knee.y - 0.046, knee.z + 0.046,
          -0.05, 0, 0, { bone: sh, blend: 'limb', color: COL.legShadow, mr: MR.cloth, uv: 3.2 }, 2);
      }
      // blousing band above the boot — reads as a hard line at 40 m
      this._cyl(b, 0.091, 0.089, 0.038, ank.x, ank.y + 0.138, ank.z, 0, 0, 0,
        { bone: sh, blend: 'limb', color: C_UNILT, mr: MR.cloth, uv: 4.0 }, 12);
      // cargo pocket
      this._box(b, 0.090, 0.142, 0.060, hip.x + sgn * 0.092, hip.y - 0.168, hip.z + 0.026,
        0, 0, sgn * 0.05, { bone: th, blend: 'limb', color: C_UNILT, mr: MR.cloth, uv: 3.0 });
      // pocket flap on its own value so the pocket reads as a pocket, not a smear
      this._box(b, 0.094, 0.034, 0.066, hip.x + sgn * 0.092, hip.y - 0.100, hip.z + 0.026,
        0, 0, sgn * 0.05, { bone: th, blend: 'limb', color: COL.legShadow, mr: MR.cloth, uv: 3.8 });

      // Boot.  Built as a real boot rather than a wedge: blousing cuff, padded
      // ankle collar, laced upper, eyelets, a lighter welt line, a proud rubber
      // outsole and a separate heel block.  The ankle break and the sole shadow
      // are the two things that stop a leg from ending in a stump at 30 m.
      const bootOpt = { bone: ft, blend: 'rigid', hard: true, color: COL.boot, mr: MR.boot, uv: 3.6 };
      const soleOpt = { bone: ft, blend: 'rigid', hard: true, color: COL.sole, mr: MR.rubber, uv: 5.0 };
      this._cyl(b, 0.084, 0.076, 0.038, ank.x, ank.y + 0.116, ank.z - 0.004, 0, 0, 0,
        { bone: ft, blend: 'rigid', color: COL.legShadow, mr: MR.cloth, uv: 4.4 }, 12);
      this._box(b, 0.127, 0.056, 0.152, ank.x, ank.y + 0.084, ank.z - 0.014,
        0.04, 0, 0, bootOpt, 2);
      this._box(b, 0.119, 0.126, 0.144, ank.x, ank.y + 0.032, ank.z - 0.012,
        0.04, 0, 0, bootOpt, 2);
      // lacing panel down the instep
      this._box(b, 0.056, 0.130, 0.040, ank.x, ank.y + 0.054, ank.z + 0.062,
        0.16, 0, 0, { bone: ft, blend: 'rigid', hard: true, color: COL.sole, mr: MR.boot, uv: 6.0 });
      for (let e = 0; e < 3; e++) {
        this._box(b, 0.074, 0.009, 0.014, ank.x, ank.y + 0.014 + e * 0.038, ank.z + 0.076,
          0.16, 0, 0, { bone: ft, blend: 'rigid', hard: true, color: COL.rail, mr: MR.gunMetal, uv: 7.0 });
      }
      this._box(b, 0.117, 0.080, 0.240, ank.x, ank.y - 0.014, ank.z + 0.052, 0, 0, 0, bootOpt, 2);
      // welt — a light line between the leather and the rubber
      this._box(b, 0.126, 0.016, 0.250, ank.x, ank.y - 0.050, ank.z + 0.050, 0, 0, 0,
        { bone: ft, blend: 'rigid', hard: true, color: COL.legShadow, mr: MR.boot, uv: 6.0 });
      // outsole, proud of the upper on every side, plus a stepped heel block
      this._box(b, 0.128, 0.030, 0.264, ank.x, ank.y - 0.068, ank.z + 0.048, 0, 0, 0, soleOpt);
      this._box(b, 0.118, 0.030, 0.094, ank.x, ank.y - 0.086, ank.z - 0.036, 0, 0, 0, soleOpt);
      this._box(b, 0.109, 0.062, 0.088, toe.x, toe.y + 0.012, toe.z + 0.026,
        0.10, 0, 0, { bone: to, blend: 'rigid', hard: true, color: COL.boot, mr: MR.boot, uv: 3.6 });
      this._box(b, 0.115, 0.024, 0.086, toe.x, toe.y - 0.020, toe.z + 0.026,
        0.06, 0, 0, { bone: to, blend: 'rigid', hard: true, color: COL.sole, mr: MR.rubber, uv: 5.4 });
    }

    // =====================================================================
    // PELVIS / TORSO
    // =====================================================================
    const pel = BP[B_PELVIS], spn = BP[B_SPINE], cst = BP[B_CHEST], nck = BP[B_NECK];

    this._sphere(b, 0.150, 0, pel.y + 0.010, 0.004, 1.03, 0.78, 0.80,
      { bone: B_PELVIS, blend: 'torso', color: C_UNI, mr: MR.cloth, uv: 2.2 }, 14, 10);
    // waist -> ribcage as a tapered oval, not a box
    this._tube(b, 0, spn.y - 0.055, 0.004, 0, cst.y + 0.075, 0.012,
      0.148, 0.176, 14, { bone: B_CHEST, blend: 'torso', color: C_UNI, mr: MR.cloth, uv: 2.2 }, 5);
    // shoulder yoke — squares the top of the silhouette
    this._tube(b, -0.150, cst.y + 0.128, 0.006, 0.150, cst.y + 0.128, 0.006,
      0.088, 0.088, 10, { bone: B_CHEST, blend: 'torso', color: C_UNIDK, mr: MR.cloth, uv: 2.6 }, 3);
    // clavicle filler so the neck-to-shoulder transition is continuous.  Kept a
    // value step DOWN from the uniform: it is a top-lit horizontal surface right
    // under the chin, and at the uniform's own value it flared into a white
    // collar that cut the head off the body at 30 m.
    this._tube(b, -0.098, cst.y + 0.150, 0.020, 0.098, cst.y + 0.150, 0.020,
      0.058, 0.058, 8, { bone: B_CHEST, blend: 'torso', color: C_UNIDK, mr: MR.cloth, uv: 2.8 }, 2);

    const carrier = { bone: B_CHEST, blend: 'torso', color: C_RIG, mr: MR.carrier, uv: 2.8 };
    const pouchOpt = { bone: B_CHEST, blend: 'torso', color: C_POUCH, mr: MR.carrier, uv: 3.6 };
    const webOpt = { bone: B_CHEST, blend: 'torso', color: COL.webbing, mr: MR.carrier, uv: 4.5 };

    if (K.rig === 'carrier') {
      // ---- full plate carrier ------------------------------------------------
      this._box(b, 0.300, 0.365, 0.085, 0, cst.y - 0.045, 0.152, 0.02, 0, 0, carrier, 2);
      this._box(b, 0.300, 0.385, 0.080, 0, cst.y - 0.050, -0.150, -0.02, 0, 0, carrier, 2);
      this._box(b, 0.115, 0.185, 0.245, 0.155, cst.y - 0.140, 0.000, 0, 0, 0.04, carrier);
      this._box(b, 0.115, 0.185, 0.245, -0.155, cst.y - 0.140, 0.000, 0, 0, -0.04, carrier);
      for (let s = 0; s < 2; s++) {
        const x = s ? -0.093 : 0.093;
        this._box(b, 0.092, 0.055, 0.360, x, cst.y + 0.188, 0.000, 0, 0, 0,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
        this._box(b, 0.088, 0.145, 0.050, x, cst.y + 0.108, 0.170, 0.10, 0, 0,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
        this._box(b, 0.088, 0.150, 0.048, x, cst.y + 0.103, -0.168, -0.10, 0, 0,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
      }
      // triple mag shingle — the pouch stack is the read at 60 m
      for (let i = 0; i < 3; i++) {
        const x = (i - 1) * 0.093;
        this._box(b, 0.086, 0.170, 0.062, x, cst.y - 0.128, 0.216, 0.03, 0, (i - 1) * -0.05, pouchOpt);
        this._box(b, 0.080, 0.028, 0.024, x, cst.y - 0.052, 0.238, 0.03, 0, (i - 1) * -0.05, webOpt);
      }
      this._box(b, 0.115, 0.100, 0.048, 0.095, cst.y + 0.048, 0.205, 0, 0, 0, pouchOpt);
    } else {
      // ---- slick chest rig: no plates, a harness and four small pouches ------
      // narrower torso, so the silhouette is obviously a different soldier
      for (let s = 0; s < 2; s++) {
        const x = s ? -0.072 : 0.072;
        this._box(b, 0.062, 0.048, 0.300, x, cst.y + 0.176, 0.010, 0, 0, 0,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
        this._box(b, 0.058, 0.230, 0.044, x, cst.y + 0.055, 0.166, 0.06, 0, 0,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
        // X-back straps
        this._box(b, 0.052, 0.240, 0.040, x, cst.y + 0.050, -0.162, -0.06, 0, s ? -0.30 : 0.30,
          { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.carrier, uv: 3.2 });
      }
      this._box(b, 0.290, 0.115, 0.062, 0, cst.y - 0.062, 0.170, 0.03, 0, 0, carrier, 2);
      for (let i = 0; i < 2; i++) {
        const x = (i ? -1 : 1) * 0.078;
        this._box(b, 0.088, 0.150, 0.058, x, cst.y - 0.152, 0.184, 0.04, 0, (i ? 1 : -1) * 0.05, pouchOpt);
        this._box(b, 0.082, 0.026, 0.022, x, cst.y - 0.084, 0.204, 0.04, 0, (i ? 1 : -1) * 0.05, webOpt);
      }
      // admin pouch high on the left chest
      this._box(b, 0.098, 0.086, 0.040, 0.108, cst.y + 0.030, 0.176, 0, 0, 0.05, pouchOpt);
      // belt-line dump pouch
      this._box(b, 0.112, 0.132, 0.088, -0.150, spn.y - 0.120, 0.052, 0, 0, -0.10,
        { bone: B_PELVIS, blend: 'torso', color: C_POUCH, mr: MR.carrier, uv: 3.4 });
    }

    // ---- two-point sling -----------------------------------------------------
    // A hard diagonal across the chest, over the firing shoulder and down the
    // back to the off hip.  Every other line on a soldier is horizontal or
    // vertical; this is the one that isn't, and it does more for reading the
    // torso as loaded infantry than another pouch would.
    {
      const sling = { bone: B_CHEST, blend: 'torso', color: COL.strapDk, mr: MR.webbing, uv: 4.2 };
      this._box(b, 0.046, 0.430, 0.028, 0.006, cst.y - 0.008, 0.206, 0, 0, 0.69, sling);
      this._box(b, 0.044, 0.400, 0.026, 0.004, cst.y - 0.020, -0.196, 0, 0, 0.69, sling);
      // the run over the shoulder that joins the two, plus its hardware
      this._box(b, 0.058, 0.052, 0.220, -0.126, cst.y + 0.170, 0.006, 0, 0, -0.12, sling);
      this._box(b, 0.036, 0.048, 0.030, 0.086, cst.y - 0.128, 0.216, 0, 0, 0.69,
        { bone: B_CHEST, blend: 'torso', hard: true, color: COL.rail, mr: MR.gunMetal, uv: 6.0 });
    }

    if (K.radio) {
      // radio on the upper back + whip antenna
      this._box(b, 0.088, 0.150, 0.062, -0.100, cst.y + 0.020, -0.198, 0, 0, 0.04,
        { bone: B_CHEST, blend: 'torso', hard: true, color: COL.nvg, mr: MR.gunPoly, uv: 3.8 });
      this._cyl(b, 0.0075, 0.0035, 0.430, -0.108, cst.y + 0.290, -0.240, -0.30, 0, 0.10,
        { bone: B_CHEST, blend: 'torso', hard: true, color: COL.antenna, mr: MR.gunPoly, uv: 6.0 }, 6);
    }
    if (K.hydration) {
      this._box(b, 0.235, 0.290, 0.085, 0.020, cst.y - 0.060, -0.212, -0.02, 0, 0,
        { bone: B_CHEST, blend: 'torso', color: C_POUCH, mr: MR.carrier, uv: 2.8 }, 2);
    }
    if (K.shoulder) {
      // shoulder brassards — squares off the top of the silhouette
      for (let s = 0; s < 2; s++) {
        const x = s ? -0.168 : 0.168;
        this._box(b, 0.070, 0.130, 0.170, x, cst.y + 0.108, 0.006, 0, 0, s ? -0.14 : 0.14,
          { bone: B_CHEST, blend: 'torso', color: C_RIG, mr: MR.carrier, uv: 3.0 });
      }
    }
    // hip pouches
    this._box(b, 0.105, 0.145, 0.095, -0.165, spn.y - 0.095, -0.030, 0, 0, -0.08,
      { bone: B_PELVIS, blend: 'torso', color: C_POUCH, mr: MR.carrier, uv: 3.4 });
    this._box(b, 0.095, 0.120, 0.085, 0.168, spn.y - 0.085, 0.030, 0, 0, 0.08,
      { bone: B_PELVIS, blend: 'torso', color: C_POUCH, mr: MR.carrier, uv: 3.4 });
    // belt
    this._tube(b, 0, pel.y + 0.062, 0.004, 0, pel.y + 0.106, 0.004,
      0.152, 0.150, 14, { bone: B_PELVIS, blend: 'torso', color: COL.webbing, mr: MR.carrier, uv: 4.0 }, 1);

    // =====================================================================
    // NECK / HEAD / HELMET
    // =====================================================================
    this._tube(b, 0, nck.y - 0.055, 0.006, 0, nck.y + 0.070, 0.004,
      0.056, 0.052, 9, { bone: B_NECK, blend: 'limb', color: COL.balaclava, mr: MR.cloth, uv: 3.4 }, 2);

    // =====================================================================
    // FACE
    //
    // Zero facial detail is what makes a soldier read as a shop dummy, and it
    // reads that way at every range — a head with no features is a ball, and the
    // eye goes straight to it.  What is needed at 30 m is not anatomy, it is a
    // VALUE LADDER on the front of the skull: a lit brow, a shadowed eye band, a
    // lit cheek/nose, a shadowed mouth, a lit chin.  Five bands, authored as
    // separate volumes so they self-shadow and hold up under any key direction.
    // =====================================================================
    const hd = BP[B_HEAD];
    const sk = { bone: B_HEAD, blend: 'rigid', color: COL.skin, mr: MR.skin, uv: 4.4 };
    const skD = { bone: B_HEAD, blend: 'rigid', color: COL.skinDk, mr: MR.skin, uv: 5.0 };
    const skL = { bone: B_HEAD, blend: 'rigid', color: COL.skinLt, mr: MR.skin, uv: 5.2 };
    const bal = { bone: B_HEAD, blend: 'rigid', color: COL.balaclava, mr: MR.cloth, uv: 3.2 };
    const socket = { bone: B_HEAD, blend: 'rigid', color: COL.beard, mr: MR.cloth, uv: 5.6 };

    // cranium + the back of the balaclava
    this._sphere(b, 0.100, hd.x, hd.y + 0.062, hd.z + 0.004, 1.00, 1.16, 1.10, bal, 14, 11);
    this._box(b, 0.126, 0.092, 0.140, hd.x, hd.y + 0.006, hd.z + 0.028, 0.10, 0, 0,
      { bone: B_HEAD, blend: 'rigid', color: COL.balaclava, mr: MR.cloth, uv: 3.6 });

    // brow ridge — proud of the face, so it throws the eye band into shadow
    this._box(b, 0.128, 0.030, 0.046, hd.x, hd.y + 0.102, hd.z + 0.080, -0.16, 0, 0,
      { bone: B_HEAD, blend: 'rigid', color: COL.brow, mr: MR.skin, uv: 5.0 });
    // eye sockets, recessed and dark — two of them, not one band
    for (let s = 0; s < 2; s++) {
      const x = s ? -0.036 : 0.036;
      this._box(b, 0.046, 0.026, 0.022, hd.x + x, hd.y + 0.080, hd.z + 0.082, 0.06, 0, 0, socket);
    }
    // temples / cheekbones — the widest point of a real face
    for (let s = 0; s < 2; s++) {
      const x = s ? -0.052 : 0.052;
      this._sphere(b, 0.036, hd.x + x, hd.y + 0.060, hd.z + 0.062, 1.0, 0.90, 0.80, skD, 8, 6);
    }
    // nose: bridge then tip, both catching the light
    this._box(b, 0.026, 0.052, 0.034, hd.x, hd.y + 0.074, hd.z + 0.100, 0.14, 0, 0, skL);
    this._sphere(b, 0.020, hd.x, hd.y + 0.050, hd.z + 0.112, 1.0, 0.85, 0.95, sk, 8, 6);
    // mid-face plane between nose and mouth
    this._box(b, 0.096, 0.034, 0.036, hd.x, hd.y + 0.052, hd.z + 0.092, 0.05, 0, 0, sk);
    // mouth line — a dark step, so the lower face is not one flat slab
    this._box(b, 0.062, 0.014, 0.024, hd.x, hd.y + 0.030, hd.z + 0.096, 0.06, 0, 0,
      { bone: B_HEAD, blend: 'rigid', color: COL.skinDk, mr: MR.skin, uv: 6.0 });
    // jaw + chin: a tapered wedge, not a box, so the profile has an angle
    this._box(b, 0.100, 0.046, 0.062, hd.x, hd.y + 0.012, hd.z + 0.074, 0.16, 0, 0, skD);
    this._sphere(b, 0.030, hd.x, hd.y + 0.002, hd.z + 0.086, 1.10, 0.80, 0.90, sk, 8, 6);
    // jaw hinge, both sides — stops the head reading as a sphere from 3/4
    for (let s = 0; s < 2; s++) {
      const x = s ? -0.050 : 0.050;
      this._sphere(b, 0.030, hd.x + x, hd.y + 0.024, hd.z + 0.028, 0.90, 1.05, 1.10, skD, 8, 6);
    }
    if (K.beard) {
      // a full beard is the single loudest silhouette change you can put on a
      // head, and it instantly tells the player these are not the same four men
      this._sphere(b, 0.062, hd.x, hd.y + 0.016, hd.z + 0.058, 1.05, 0.86, 0.96,
        { bone: B_HEAD, blend: 'rigid', color: COL.beard, mr: MR.cloth, uv: 4.2 }, 12, 8);
      this._box(b, 0.104, 0.052, 0.040, hd.x, hd.y + 0.028, hd.z + 0.066, 0.10, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.beard, mr: MR.cloth, uv: 4.6 });
    } else {
      // balaclava pulled up over the jaw instead
      this._box(b, 0.112, 0.058, 0.056, hd.x, hd.y + 0.018, hd.z + 0.062, 0.12, 0, 0, bal);
    }

    // ---- headgear -----------------------------------------------------------
    const shellY = hd.y + 0.086;
    // A cloth-covered ballistic shell is a MATTE object.  It was reading as wet
    // plastic because its roughness was a multiplier on the polished gunmetal
    // map; aMR.x is an absolute roughness now, and the cover is authored as
    // cloth, so the dome finally takes the sun as a broad soft terminator
    // instead of a hotspot.
    const hel = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.helmet, mr: MR.helmetCloth, uv: 3.0 };
    const helCover = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.helmetCover, mr: MR.helmetCloth, uv: 3.0 };
    const helDark = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.nvg, mr: MR.gunPoly, uv: 3.6 };
    const helRail = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.rail, mr: MR.gunMetal, uv: 4.4 };
    const hdSet = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.headset, mr: MR.rubber, uv: 4.0 };

    if (K.head === 'helmet') {
      this._sphere(b, 0.134, hd.x, shellY, hd.z - 0.004, 1.00, 0.94, 1.06, hel, 18, 12, Math.PI * 0.58);
      this._cyl(b, 0.1355, 0.1385, 0.055, hd.x, shellY - 0.012, hd.z - 0.004, 0, 0, 0, helCover, 18);
      // rear occiput flare
      this._sphere(b, 0.108, hd.x, shellY - 0.030, hd.z - 0.062, 1.02, 0.70, 0.80, hel, 12, 8);
      // brim
      this._box(b, 0.150, 0.022, 0.052, hd.x, shellY - 0.028, hd.z + 0.118, -0.14, 0, 0, helCover);
      if (K.nvg) {
        // NVG shroud + mount arm
        this._box(b, 0.072, 0.052, 0.030, hd.x, shellY + 0.010, hd.z + 0.124, -0.10, 0, 0, helDark);
        this._box(b, 0.040, 0.078, 0.026, hd.x, shellY + 0.058, hd.z + 0.132, 0.22, 0, 0, helDark);
        this._cyl(b, 0.012, 0.012, 0.048, hd.x, shellY + 0.092, hd.z + 0.118, Math.PI * 0.5, 0, 0, helRail, 8);
      } else {
        // bare shroud plate + an IR strobe puck instead
        this._box(b, 0.058, 0.020, 0.032, hd.x, shellY + 0.014, hd.z + 0.122, -0.10, 0, 0, helDark);
        this._cyl(b, 0.020, 0.020, 0.024, hd.x - 0.062, shellY + 0.044, hd.z - 0.048, 0.3, 0, 0.2, helDark, 8);
      }
      for (let s = 0; s < 2; s++) {
        const x = s ? -0.128 : 0.128;
        this._box(b, 0.020, 0.030, 0.170, hd.x + x, shellY + 0.004, hd.z + 0.006, 0, 0, s ? 0.10 : -0.10, helRail);
      }
      // counterweight pouch at the back — a real profile break
      this._box(b, 0.090, 0.062, 0.048, hd.x, shellY - 0.006, hd.z - 0.124, 0.10, 0, 0,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: C_POUCH, mr: MR.carrier, uv: 4.0 });
      // cover seams — front-to-back and ear-to-ear, plus the loop tape panels.
      // Tiny geometry, but it is the difference between a helmet and a bowl.
      const seam = { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.helmet, mr: MR.helmetCloth, uv: 5.4 };
      this._box(b, 0.018, 0.012, 0.244, hd.x, shellY + 0.114, hd.z - 0.004, 0, 0, 0, seam);
      this._box(b, 0.244, 0.012, 0.018, hd.x, shellY + 0.110, hd.z + 0.010, 0, 0, 0, seam);
      this._box(b, 0.072, 0.010, 0.052, hd.x, shellY + 0.058, hd.z + 0.104, -0.42, 0, 0,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.webbing, mr: MR.webbing, uv: 5.6 });
    } else if (K.head === 'boonie') {
      // soft crown + a wide floppy brim: completely different head silhouette
      this._sphere(b, 0.128, hd.x, shellY - 0.012, hd.z - 0.002, 1.00, 0.80, 1.02,
        { bone: B_HEAD, blend: 'rigid', color: COL.boonie, mr: MR.cloth, uv: 3.0 }, 16, 10, Math.PI * 0.55);
      this._cyl(b, 0.132, 0.136, 0.048, hd.x, shellY - 0.038, hd.z - 0.002, 0, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.boonieDk, mr: MR.cloth, uv: 3.4 }, 16);
      this._cyl(b, 0.214, 0.206, 0.016, hd.x, shellY - 0.062, hd.z + 0.004, 0.09, 0, 0.04,
        { bone: B_HEAD, blend: 'rigid', color: COL.boonie, mr: MR.cloth, uv: 2.6 }, 18);
      this._cyl(b, 0.140, 0.140, 0.014, hd.x, shellY - 0.050, hd.z - 0.002, 0, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.webbing, mr: MR.cloth, uv: 4.6 }, 16);
    } else {
      // bare head: watch cap over the balaclava, plus a low-profile headband
      this._sphere(b, 0.108, hd.x, shellY - 0.028, hd.z - 0.002, 1.02, 0.86, 1.04,
        { bone: B_HEAD, blend: 'rigid', color: COL.balaclava, mr: MR.cloth, uv: 3.4 }, 14, 10, Math.PI * 0.56);
      this._cyl(b, 0.112, 0.114, 0.040, hd.x, shellY - 0.056, hd.z - 0.002, 0, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.strapDk, mr: MR.cloth, uv: 4.0 }, 14);
    }

    // Eye protection — only on half the squad now.  A bare, browed, bearded face
    // and a goggled one are two completely different reads at 30 m, and running
    // ballistic glasses on every man was quietly erasing the face work above.
    if (K.eyepro) {
      this._box(b, 0.140, 0.036, 0.032, hd.x, hd.y + 0.086, hd.z + 0.084, 0.02, 0, 0,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.lens, mr: MR.lens, uv: 5.2 });
      this._box(b, 0.152, 0.018, 0.024, hd.x, hd.y + 0.104, hd.z + 0.074, 0.05, 0, 0,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.rail, mr: MR.gunPoly, uv: 5.6 });
      // retention strap around the back of the skull
      this._box(b, 0.150, 0.020, 0.180, hd.x, hd.y + 0.090, hd.z - 0.010, 0.02, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.strapDk, mr: MR.webbing, uv: 5.4 });
    }

    // Chin strap.  Four-point retention: two risers off the shell, a nape strap
    // and a chin cup.  It is a small thing, but a helmet with nothing holding it
    // on is the most obvious "prop" tell a character model can have.
    if (K.head !== 'boonie') {
      const strap = { bone: B_HEAD, blend: 'rigid', color: COL.strapDk, mr: MR.webbing, uv: 5.6 };
      for (let s = 0; s < 2; s++) {
        const x = s ? -1 : 1;
        this._box(b, 0.016, 0.108, 0.020, hd.x + x * 0.086, hd.y + 0.052, hd.z + 0.052,
          0.10, 0, x * 0.26, strap);
        this._box(b, 0.016, 0.092, 0.020, hd.x + x * 0.090, hd.y + 0.050, hd.z - 0.038,
          -0.16, 0, x * 0.22, strap);
      }
      this._box(b, 0.070, 0.038, 0.044, hd.x, hd.y - 0.004, hd.z + 0.052, 0.20, 0, 0,
        { bone: B_HEAD, blend: 'rigid', color: COL.webbing, mr: MR.webbing, uv: 5.0 });
    }

    if (K.headset) {
      for (let s = 0; s < 2; s++) {
        const x = s ? -0.128 : 0.128;
        this._cyl(b, 0.049, 0.045, 0.036, hd.x + x * 0.86, hd.y + 0.038, hd.z - 0.012,
          0, 0, Math.PI * 0.5, hdSet, 12);
      }
      this._box(b, 0.026, 0.030, 0.026, hd.x + 0.088, hd.y + 0.002, hd.z + 0.052, 0, 0, 0.5,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.headset, mr: MR.rubber, uv: 4.6 });
      this._box(b, 0.230, 0.016, 0.014, hd.x, hd.y - 0.012, hd.z + 0.058, 0, 0, 0,
        { bone: B_HEAD, blend: 'rigid', hard: true, color: COL.webbing, mr: MR.carrier, uv: 4.6 });
    }

    if (K.scarf) {
      // shemagh bunched at the neck with a flap over the shoulder — the single
      // cheapest way to break a helmeted-clone silhouette
      this._sphere(b, 0.118, 0, nck.y - 0.036, 0.010, 1.18, 0.62, 1.10,
        { bone: B_NECK, blend: 'limb', color: COL.scarf, mr: MR.cloth, uv: 3.0 }, 14, 9);
      this._box(b, 0.150, 0.170, 0.060, 0.058, nck.y - 0.140, 0.128, 0.18, 0, -0.24,
        { bone: B_CHEST, blend: 'torso', color: COL.scarfDk, mr: MR.cloth, uv: 3.0 });
      this._box(b, 0.120, 0.130, 0.050, -0.070, nck.y - 0.120, -0.120, -0.14, 0, 0.20,
        { bone: B_CHEST, blend: 'torso', color: COL.scarf, mr: MR.cloth, uv: 3.0 });
    }

    // =====================================================================
    // ARMS
    // =====================================================================
    const sleevesUp = K.sleeves === 'up';
    for (let s = 0; s < 2; s++) {
      const ar = s ? B_ARMR : B_ARML, fo = s ? B_FORER : B_FOREL, hn = s ? B_HANDR : B_HANDL;
      const sh = BP[ar], el = BP[fo], wr = BP[hn];
      const sgn = s ? -1 : 1;

      // deltoid cap — no child weighting so the shoulder does not collapse
      this._sphere(b, 0.086, sh.x + sgn * 0.010, sh.y + 0.016, sh.z, 1.0, 1.02, 1.0,
        { bone: ar, blend: 'limb', child: -1, color: C_UNI, mr: MR.cloth, uv: 2.8 }, 12, 9);
      // Upper arm: a real taper, wide at the deltoid and narrow at the elbow,
      // with a bicep mass on top of it.  A constant-radius tube is the single
      // clearest "mannequin" tell there is, and both arms had one.
      this._tube(b, sh.x, sh.y, sh.z, el.x, el.y, el.z,
        0.074, sleevesUp ? 0.056 : 0.051, 10,
        { bone: ar, blend: 'limb', color: C_UNI, mr: MR.cloth, uv: 2.6 }, 4);
      this._sphere(b, 0.060,
        lerp(sh.x, el.x, 0.34), lerp(sh.y, el.y, 0.34), lerp(sh.z, el.z, 0.34) + 0.006,
        1.05, 1.35, 1.10,
        { bone: ar, blend: 'limb', color: C_UNI, mr: MR.cloth, uv: 3.0 }, 10, 8);
      // shoulder admin / IFF pouch on the outside of the bicep
      this._box(b, 0.038, 0.086, 0.066,
        lerp(sh.x, el.x, 0.30) + sgn * 0.052, lerp(sh.y, el.y, 0.30), lerp(sh.z, el.z, 0.30),
        0, 0, sgn * 0.08,
        { bone: ar, blend: 'limb', color: C_POUCH, mr: MR.carrier, uv: 4.0 });
      // elbow: a joint VOLUME under the pad, so the arm bends at a lump rather
      // than creasing like a hose
      this._sphere(b, 0.056, el.x, el.y, el.z - 0.008, 1.02, 1.0, 1.08,
        { bone: fo, blend: 'limb', color: C_UNIDK, mr: MR.cloth, uv: 3.4 }, 10, 8);
      this._box(b, 0.084, 0.090, 0.058,
        lerp(sh.x, el.x, 0.95), lerp(sh.y, el.y, 0.95), lerp(sh.z, el.z, 0.95) - 0.030,
        0, 0, 0, { bone: fo, blend: 'limb', hard: true, color: COL.knee, mr: MR.rubber, uv: 4.0 });
      if (sleevesUp) {
        // rolled cuff above a bare forearm
        this._cyl(b, 0.064, 0.060, 0.048,
          lerp(sh.x, el.x, 0.88), lerp(sh.y, el.y, 0.88), lerp(sh.z, el.z, 0.88), 0, 0, 0,
          { bone: ar, blend: 'limb', color: C_UNILT, mr: MR.cloth, uv: 4.0 }, 10);
        // forearm: fat at the elbow, wristy at the hand
        this._tube(b, el.x, el.y, el.z, wr.x, wr.y, wr.z,
          0.055, 0.033, 10, { bone: fo, blend: 'limb', color: COL.skin, mr: MR.skin, uv: 3.0 }, 4);
        this._sphere(b, 0.044,
          lerp(el.x, wr.x, 0.26), lerp(el.y, wr.y, 0.26), lerp(el.z, wr.z, 0.26),
          1.05, 1.25, 1.05, { bone: fo, blend: 'limb', color: COL.skin, mr: MR.skin, uv: 3.4 }, 10, 7);
      } else {
        this._tube(b, el.x, el.y, el.z, wr.x, wr.y, wr.z,
          0.058, 0.037, 10, { bone: fo, blend: 'limb', color: C_UNIDK, mr: MR.cloth, uv: 2.8 }, 4);
        this._sphere(b, 0.048,
          lerp(el.x, wr.x, 0.24), lerp(el.y, wr.y, 0.24), lerp(el.z, wr.z, 0.24),
          1.05, 1.22, 1.05, { bone: fo, blend: 'limb', color: C_UNIDK, mr: MR.cloth, uv: 3.2 }, 10, 7);
      }
      // glove: palm + thumb ridge so the hand reads as gripping
      this._box(b, 0.062, 0.108, 0.084, wr.x, wr.y - 0.026, wr.z + 0.006, 0, 0, 0,
        { bone: hn, blend: 'rigid', color: COL.glove, mr: MR.rubber, uv: 4.4 }, 2);
      this._box(b, 0.036, 0.058, 0.046, wr.x + sgn * 0.030, wr.y - 0.040, wr.z + 0.034, 0, 0, sgn * 0.35,
        { bone: hn, blend: 'rigid', color: COL.glove, mr: MR.rubber, uv: 5.0 });
      // knuckle guard — a hard, dark shape on the back of the hand
      this._box(b, 0.052, 0.040, 0.030, wr.x - sgn * 0.010, wr.y - 0.062, wr.z - 0.024, 0.20, 0, 0,
        { bone: hn, blend: 'rigid', hard: true, color: COL.knee, mr: MR.rubber, uv: 5.4 });
      this._cyl(b, 0.048, 0.046, 0.030, wr.x, wr.y + 0.020, wr.z, 0, 0, 0,
        { bone: hn, blend: 'rigid', color: COL.webbing, mr: MR.webbing, uv: 5.0 }, 10);
    }

    // =====================================================================
    // RIFLE — rigid to the weapon hand
    // =====================================================================
    this._buildRifle(b);

    // ---- assemble -----------------------------------------------------------
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(b.pos, 3));
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(b.nor, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(b.uv, 2));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(b.col, 3));
    geo.setAttribute('aMR', new THREE.Float32BufferAttribute(b.mr, 2));
    geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(b.si, 4));
    geo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(b.sw, 4));

    const nSoft = b.idxSoft.length, nHard = b.idxHard.length;
    const idx = new Uint32Array(nSoft + nHard);
    idx.set(b.idxSoft, 0);
    idx.set(b.idxHard, nSoft);
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.clearGroups();
    geo.addGroup(0, nSoft, 0);
    geo.addGroup(nSoft, nHard, 1);

    // Bind-pose bounds are useless for a skinned mesh in motion; inflate once so
    // frustum culling is conservative but still cheap.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0.95, 0), 2.4);
    geo.boundingBox = new THREE.Box3(
      new THREE.Vector3(-1.3, -0.6, -1.3), new THREE.Vector3(1.3, 2.3, 1.3));
    return geo;
  }

  _buildRifle(b) {
    const M = this._rifleBindMatrix;
    const put = (geo, lx, ly, lz, rx, ry, rz, opt) => {
      this._m2.makeRotationFromEuler(this._e1.set(rx || 0, ry || 0, rz || 0, 'YXZ'));
      this._m2.setPosition(lx, ly, lz);
      this._m1.multiplyMatrices(M, this._m2);
      this._emit(b, geo, this._m1, opt);
      geo.dispose();
    };
    const gun = { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.gunBody, mr: MR.gunMetal, uv: 5.5 };
    const poly = { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.gunPoly, mr: MR.gunPoly, uv: 5.5 };
    const mag = { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.gunMag, mr: MR.gunPoly, uv: 5.5 };
    const optic = { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.optic, mr: MR.gunPoly, uv: 6.0 };

    // lower receiver + magwell, upper receiver
    put(new THREE.BoxGeometry(0.052, 0.086, 0.165), 0, 0.038, 0.062, 0, 0, 0, gun);
    put(new THREE.BoxGeometry(0.048, 0.062, 0.235), 0, 0.098, 0.098, 0, 0, 0, gun);
    // top rail with tooth detail
    put(new THREE.BoxGeometry(0.026, 0.014, 0.330), 0, 0.134, 0.152, 0, 0, 0, gun);
    for (let i = 0; i < 7; i++) {
      put(new THREE.BoxGeometry(0.030, 0.010, 0.012), 0, 0.140, 0.020 + i * 0.045, 0, 0, 0, gun);
    }
    // ejection port cover, brass deflector, charging handle
    put(new THREE.BoxGeometry(0.014, 0.038, 0.070), -0.030, 0.098, 0.100, 0, 0, 0, poly);
    put(new THREE.BoxGeometry(0.020, 0.032, 0.030), -0.030, 0.122, 0.060, 0, 0, 0.3, gun);
    put(new THREE.BoxGeometry(0.058, 0.014, 0.030), 0, 0.126, -0.020, 0, 0, 0, gun);

    // handguard — octagonal free-float tube
    for (let i = 0; i < 4; i++) {
      const a = i * Math.PI * 0.5 + Math.PI * 0.25;
      const r = 0.038;
      put(new THREE.BoxGeometry(0.030, 0.030, 0.270),
        Math.cos(a) * r, 0.098 + Math.sin(a) * r, 0.318, 0, 0, a, poly);
    }
    put(new THREE.BoxGeometry(0.052, 0.052, 0.272), 0, 0.098, 0.318, 0, 0, 0, poly);
    for (let i = 0; i < 5; i++) {
      put(new THREE.BoxGeometry(0.028, 0.010, 0.014), 0, 0.130, 0.212 + i * 0.052, 0, 0, 0, gun);
    }
    // angled foregrip — where the support hand sits (matches RIFLE_HANDGUARD)
    put(new THREE.BoxGeometry(0.032, 0.090, 0.054), 0, 0.036, 0.248, -0.45, 0, 0, poly);
    // hand stop just ahead of it
    put(new THREE.BoxGeometry(0.034, 0.026, 0.020), 0, 0.062, 0.294, 0, 0, 0, poly);

    // barrel, gas block, muzzle brake
    put(new THREE.CylinderGeometry(0.0115, 0.0105, 0.130, 10), 0, 0.098, 0.512, Math.PI * 0.5, 0, 0, gun);
    put(new THREE.BoxGeometry(0.026, 0.030, 0.036), 0, 0.108, 0.462, 0, 0, 0, gun);
    put(new THREE.CylinderGeometry(0.019, 0.017, 0.062, 10), 0, 0.098, 0.612, Math.PI * 0.5, 0, 0, gun);
    put(new THREE.CylinderGeometry(0.0135, 0.0135, 0.020, 8), 0, 0.098, 0.648, Math.PI * 0.5, 0, 0, gun);

    // magazine — three stacked, progressively tilted boxes make a real curve
    for (let i = 0; i < 3; i++) {
      const t = i / 2;
      put(new THREE.BoxGeometry(0.030, 0.098, 0.072 - t * 0.006),
        0, -0.012 - i * 0.088, 0.050 + t * 0.028 + i * 0.006, 0.16 + t * 0.10, 0, 0, mag);
    }
    put(new THREE.BoxGeometry(0.034, 0.018, 0.070), 0, -0.262, 0.104, 0.30, 0, 0, poly);

    // pistol grip + trigger guard
    put(new THREE.BoxGeometry(0.036, 0.115, 0.052), 0, -0.052, -0.024, 0.30, 0, 0, poly);
    put(new THREE.BoxGeometry(0.030, 0.010, 0.060), 0, -0.006, 0.020, 0, 0, 0, gun);
    put(new THREE.BoxGeometry(0.012, 0.036, 0.010), 0, 0.008, 0.048, -0.2, 0, 0, gun);

    // stock: buffer tube, cheek riser, butt pad
    put(new THREE.CylinderGeometry(0.021, 0.021, 0.170, 10), 0, 0.086, -0.086, Math.PI * 0.5, 0, 0, gun);
    put(new THREE.BoxGeometry(0.046, 0.062, 0.130), 0, 0.078, -0.128, 0, 0, 0, poly);
    put(new THREE.BoxGeometry(0.050, 0.106, 0.026), 0, 0.062, -0.196, 0.12, 0, 0, poly);
    put(new THREE.BoxGeometry(0.054, 0.112, 0.016), 0, 0.060, -0.210, 0.12, 0, 0,
      { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.butt, mr: MR.rubber, uv: 5.5 });

    // optic: body, hood, lens, mount, folded offset iron
    put(new THREE.BoxGeometry(0.040, 0.030, 0.052), 0, 0.156, 0.118, 0, 0, 0, optic);
    put(new THREE.CylinderGeometry(0.024, 0.024, 0.078, 12), 0, 0.180, 0.118, Math.PI * 0.5, 0, 0, optic);
    put(new THREE.CylinderGeometry(0.021, 0.021, 0.006, 12), 0, 0.180, 0.156, Math.PI * 0.5, 0, 0,
      { bone: B_HANDR, blend: 'rigid', hard: true, color: COL.lens, mr: MR.lens, uv: 6.0 });
    put(new THREE.BoxGeometry(0.046, 0.028, 0.060), 0, 0.142, 0.118, 0, 0, 0, gun);
    put(new THREE.BoxGeometry(0.012, 0.034, 0.014), 0.024, 0.146, 0.286, 0, 0, 0.6, gun);

    // sling from stock to handguard
    put(new THREE.BoxGeometry(0.012, 0.010, 0.470), 0.026, 0.036, 0.130, 0, -0.10, 0.2,
      { bone: B_HANDR, blend: 'rigid', color: COL.webbing, mr: MR.carrier, uv: 5.0 });
  }

  /* ======================================================================== *
   *  FX — muzzle flash sprite, tracers, muzzle light
   * ======================================================================== */

  _buildFX() {
    // flash texture: hot core, four-point star, soft falloff
    const N = 64, data = new Uint8Array(N * N * 4);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const dx = (x + 0.5) / N - 0.5, dy = (y + 0.5) / N - 0.5;
        const r = Math.sqrt(dx * dx + dy * dy) * 2;
        const ang = Math.atan2(dy, dx);
        const star = Math.pow(Math.abs(Math.cos(ang * 2)), 6) * 0.55 + 0.45;
        let a = Math.pow(clamp01(1 - r / star), 2.4);
        a += Math.pow(clamp01(1 - r * 3.1), 5) * 0.9;
        a = clamp01(a);
        const i = (y * N + x) * 4;
        data[i] = 255;
        data[i + 1] = (255 * clamp01(0.72 + a * 0.30)) | 0;
        data[i + 2] = (255 * clamp01(0.32 + a * 0.50)) | 0;
        data[i + 3] = (a * 255) | 0;
      }
    }
    const tex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    this.flashTex = tex;

    this.flashMat = new THREE.MeshBasicMaterial({
      map: tex, transparent: true, blending: THREE.AdditiveBlending,
      depthWrite: false, depthTest: true, fog: false, toneMapped: true,
      side: THREE.DoubleSide, color: 0xffc98a,
    });
    this.flashMat.name = 'enemy_muzzleflash';
    const q1 = new THREE.PlaneGeometry(1, 1);
    const q2 = new THREE.PlaneGeometry(1, 1).rotateY(Math.PI * 0.5);
    const q3 = new THREE.PlaneGeometry(1, 1).rotateY(Math.PI * 0.25).rotateZ(0.4);
    this.flashGeo = this._mergeSimple([q1, q2, q3]);
    q1.dispose(); q2.dispose(); q3.dispose();

    // tracer pool: instanced billboarded quads, one draw call
    this.TRACER_MAX = 64;
    const tg = new THREE.PlaneGeometry(1, 1);
    const tm = new THREE.MeshBasicMaterial({
      color: 0xffb066, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false, depthTest: true,
      fog: false, toneMapped: true, side: THREE.DoubleSide,
    });
    tm.name = 'enemy_tracer';
    this.tracers = new THREE.InstancedMesh(tg, tm, this.TRACER_MAX);
    this.tracers.frustumCulled = false;
    this.tracers.castShadow = false;
    this.tracers.receiveShadow = false;
    this.tracers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.tracers.name = 'enemy_tracers';
    this.group.add(this.tracers);

    this._tracer = [];
    this._m1.makeScale(0, 0, 0);
    for (let i = 0; i < this.TRACER_MAX; i++) {
      this._tracer.push({
        active: false, t: 0, life: 0,
        from: new THREE.Vector3(), dir: new THREE.Vector3(),
        speed: 320, len: 7, width: 0.030, travel: 0, maxDist: 200,
      });
      this.tracers.setMatrixAt(i, this._m1);
    }
    this.tracers.instanceMatrix.needsUpdate = true;
    this._tracerIdx = 0;
    this._tracerDirty = false;

    // one shared muzzle point light — a constant light count means no runtime
    // shader recompiles
    this.muzzleLight = new THREE.PointLight(0xffb45e, 0, 16, 2.0);
    this.muzzleLight.name = 'enemyMuzzleLight';
    this.muzzleLight.castShadow = false;
    this.group.add(this.muzzleLight);
    this._muzzleLightT = 0;
    const lights = this.g?.registry?.lights;
    if (Array.isArray(lights)) lights.push(this.muzzleLight);
  }

  _mergeSimple(geos) {
    let vc = 0, ic = 0;
    for (const g of geos) {
      vc += g.attributes.position.count;
      ic += g.index ? g.index.count : g.attributes.position.count;
    }
    const pos = new Float32Array(vc * 3), nor = new Float32Array(vc * 3), uv = new Float32Array(vc * 2);
    const idx = new Uint16Array(ic);
    let vo = 0, io = 0;
    for (const g of geos) {
      const p = g.attributes.position, n = g.attributes.normal, u = g.attributes.uv;
      for (let i = 0; i < p.count; i++) {
        pos[(vo + i) * 3] = p.getX(i); pos[(vo + i) * 3 + 1] = p.getY(i); pos[(vo + i) * 3 + 2] = p.getZ(i);
        nor[(vo + i) * 3] = n.getX(i); nor[(vo + i) * 3 + 1] = n.getY(i); nor[(vo + i) * 3 + 2] = n.getZ(i);
        uv[(vo + i) * 2] = u.getX(i); uv[(vo + i) * 2 + 1] = u.getY(i);
      }
      if (g.index) { for (let i = 0; i < g.index.count; i++) idx[io++] = vo + g.index.getX(i); }
      else { for (let i = 0; i < p.count; i++) idx[io++] = vo + i; }
      vo += p.count;
    }
    const out = new THREE.BufferGeometry();
    out.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    out.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    out.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    out.setIndex(new THREE.BufferAttribute(idx, 1));
    return out;
  }

  /* ======================================================================== *
   *  SPAWNING
   * ======================================================================== */

  async init() {
    this._makePosts();
    for (let i = 0; i < this.count; i++) {
      const u = this._makeUnit(i);
      this.units.push(u);
      this.handles.push(u.handle);
    }
    const reg = this.g?.registry;
    if (reg) {
      if (!Array.isArray(reg.enemies)) reg.enemies = [];
      for (const h of this.handles) reg.enemies.push(h);
    }

    for (let i = 0; i < this.units.length; i++) {
      const p = this._posts[i % this._posts.length];
      this.units[i].postIndex = i % this._posts.length;
      this._respawnUnit(this.units[i], p.x, p.z, p.yaw);
    }

    const bus = this.g?.bus;
    if (bus) {
      this._onHit = (d) => this._handleHit(d);
      this._onShot = (d) => this._handleShot(d);
      this._onImpact = (d) => this._handleImpact(d);
      this._onExplosion = (d) => this._handleExplosion(d);
      bus.on('hit', this._onHit);
      bus.on('shot', this._onShot);
      bus.on('impact', this._onImpact);
      bus.on('explosion', this._onExplosion);
    }

    this.stats.drawCalls = this.units.length * 2 + 1;
  }

  _makeUnit(id) {
    const bones = [];
    for (let i = 0; i < BONE_COUNT; i++) {
      const d = BONES[i];
      const b = new THREE.Bone();
      b.name = d.name;
      b.position.set(d.off[0], d.off[1], d.off[2]);
      b.rotation.set(this._restEuler[i * 3], this._restEuler[i * 3 + 1], this._restEuler[i * 3 + 2]);
      bones.push(b);
    }
    for (let i = 0; i < BONE_COUNT; i++) {
      const p = BONES[i].parent;
      if (p >= 0) bones[p].add(bones[i]);
    }

    const root = new THREE.Group();
    root.name = 'enemy' + id;
    root.add(bones[0]);
    root.updateMatrixWorld(true);

    const skeleton = new THREE.Skeleton(bones);

    const matSoft = this.matSoftProto.clone();
    const matHard = this.matHardProto.clone();
    matSoft.onBeforeCompile = this.matSoftProto.onBeforeCompile;
    matHard.onBeforeCompile = this.matHardProto.onBeforeCompile;
    matSoft.customProgramCacheKey = this.matSoftProto.customProgramCacheKey;
    matHard.customProgramCacheKey = this.matHardProto.customProgramCacheKey;
    // per-soldier weathering so the squad is not fourteen identical clones
    const r = this.rand;
    // Deal kits from a shuffled deck rather than sampling independently: every
    // block of four men contains all four silhouettes, so any group the player
    // can see at once is guaranteed to be mixed instead of eleven of one variant.
    if (!this._kitDeck || this._kitDeck.length === 0) {
      this._kitDeck = this._kitDeck || [];
      for (let k = 0; k < KIT_COUNT; k++) this._kitDeck.push(k);
      for (let k = this._kitDeck.length - 1; k > 0; k--) {
        const j = (r() * (k + 1)) | 0;
        const tmp = this._kitDeck[k]; this._kitDeck[k] = this._kitDeck[j]; this._kitDeck[j] = tmp;
      }
    }
    const kitIdx = this._kitDeck.pop();
    // Per-man weathering: value AND hue drift, so one man is bleached pale, the
    // next is dust-caked and warm, the next has a greener issue of the same
    // uniform.  Kept inside +/-15% so nobody leaves the desert palette.
    const tint = 0.84 + r() * 0.24;
    const hue = r();
    // Blue is always held BELOW red.  The old spread let a man come out cooler
    // than neutral, and a cool figure standing in a warm street is the loudest
    // "pasted in" signal there is — it was reading as grey plastic.
    matSoft.color.setRGB(
      tint * (1.00 + hue * 0.05),
      tint * (0.950 + hue * 0.035),
      tint * (0.790 + (1 - hue) * 0.130));
    matHard.color.setScalar(0.92 + r() * 0.18);

    // Pose archetype, dealt from a shuffled deck for the same reason kits are:
    // any group of five the player can see at once is guaranteed to contain a
    // high ready, a low ready, a cover lean and a scanning turn.
    if (!this._poseDeck || this._poseDeck.length === 0) {
      this._poseDeck = this._poseDeck || [];
      for (let k = 0; k < POSE_COUNT; k++) this._poseDeck.push(k);
      for (let k = this._poseDeck.length - 1; k > 0; k--) {
        const j = (r() * (k + 1)) | 0;
        const tmp = this._poseDeck[k]; this._poseDeck[k] = this._poseDeck[j]; this._poseDeck[j] = tmp;
      }
    }
    const archeIdx = this._poseDeck.pop();
    const P = POSE[archeIdx];

    // SkinnedMesh uses AttachedBindMode, so bindMatrixInverse cancels the mesh's
    // own world matrix every frame: moving `root` moves mesh and bones together
    // with no double transform, and frustum culling still works off the group.
    const mesh = new THREE.SkinnedMesh(this.geometries[kitIdx], [matSoft, matHard]);
    mesh.name = 'enemyMesh' + id;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = true;
    root.add(mesh);
    root.updateMatrixWorld(true);
    mesh.bind(skeleton, new THREE.Matrix4());

    const flash = new THREE.Mesh(this.flashGeo, this.flashMat);
    flash.frustumCulled = false;
    flash.visible = false;
    flash.castShadow = false;
    bones[B_HANDR].add(flash);
    flash.position.copy(this._muzzleLocal);
    flash.scale.setScalar(0.28);

    this.group.add(root);

    const u = {
      id, root, mesh, bones, skeleton, matSoft, matHard, flash,
      alive: true, dead: false, hp: 100, maxHp: 100,
      kit: kitIdx,

      // ---- pose archetype ---------------------------------------------------
      arche: archeIdx,
      pCarry: P.carry, pLean: P.lean, pTwist: P.twist, pCant: P.cant,
      pElbow: P.elbow, pChin: P.chin, pCrouch: P.crouch,
      // low-amplitude weapon sway, on its own clock — nobody is frozen
      swayW: 0.30 + r() * 0.34, swayWPh: r() * 6.283,

      // ---- per-instance identity ------------------------------------------
      // +/- 6 cm of height on a 1.80 m man, so the squad's heads never line up
      hScale: 1 + (r() - 0.5) * 0.078,
      // idle machinery: nobody shares a breath phase, a weight shift or a scan
      swayPhase: r() * 6.283, swayRate: 0.42 + r() * 0.38,
      scanPhase: r() * 6.283, scanRate: 0.30 + r() * 0.34,
      scanAmp: (0.24 + r() * 0.42) * P.scan,
      idleAimYaw: (r() - 0.5) * 0.24, idleAimPitch: -0.06 - r() * 0.16,
      // constant postural bias — slouch, shoulder drop, head cant
      biasSpine: (r() - 0.5) * 0.135,
      biasChestY: (r() - 0.5) * 0.265,
      biasHeadZ: (r() - 0.5) * 0.190,
      biasClav: (r() - 0.5) * 0.120,
      // Standing weight is never even. One hip carries it, the spine answers
      // with a counter-lean, and the whole man sits slightly off his own axis.
      braceLean: (r() - 0.5) * 0.42,
      // Slow idle beat: a weapon shift, a shoulder roll, a sector check or a
      // weight change. Rate and phase are per-man, so a static squad never
      // freeze-frames into a row of identical statues.
      gestPhase: r(),
      gestRate: 0.052 + r() * 0.062,
      gestKind: (r() * 4) | 0,
      // 0 standing, 1 crouched behind cover, 2 kneeling brace — the cover-lean
      // archetype always crouches, the rest keep their own bias
      postureStyle: P.crouch > 0.4 ? 1 : (r() < 0.18 ? 2 : 0),
      carryBias: r(),
      carry: 0.7, carryVel: 0,
      weaponPitch: 0, weaponCant: 0,
      // fighting-stance footwork, lead foot varies per man
      leadSide: r() < 0.78 ? 0 : 1,
      stanceLead: 0.145 + r() * 0.075,
      stanceTrail: -0.105 - r() * 0.075,
      stanceWidth: 0.134 + r() * 0.054,
      footSplay: [0.16 + r() * 0.16, -0.20 - r() * 0.20],
      // Nobody squares up dead-on to the threat. The feet sit off-axis and the
      // torso makes up the difference, which is what breaks a firing line out of
      // "five copies of the same cutout facing the camera".
      yawBias: (r() - 0.5) * 0.55,

      // transform / motion
      pos: new THREE.Vector3(),
      vel: new THREE.Vector3(),
      yaw: 0, yawTarget: 0,
      speed: 0, speedTarget: 0,
      stance: 0, stanceTarget: 0,
      groundY: 0,
      moveTo: new THREE.Vector3(),
      hasMoveTo: false,
      pathBlockT: 0,

      // animation
      phase: r(),
      lean: 0, leanVel: 0,
      bob: 0,
      breathe: r() * 6.28,
      pose: new Float32Array(BONE_COUNT * 3),
      aimYaw: 0, aimPitch: 0,
      headYaw: 0, headPitch: 0,
      recoil: 0, recoilVel: 0,
      flinch: 0, flinchVel: 0, flinchDir: 0,
      reloadT: -1,
      footPlant: [new THREE.Vector3(), new THREE.Vector3()],
      footPrev: [new THREE.Vector3(), new THREE.Vector3()],
      footNext: [new THREE.Vector3(), new THREE.Vector3()],
      footDown: [true, true],
      footInit: false,
      pelvisDrop: 0,
      flashT: 0,
      ragAcc: 0,

      // AI
      state: S_IDLE, stateT: 0,
      awareness: 0, alerted: false, reactT: 0,
      canSee: false, losT: 0, losOK: false,
      lastSeen: new THREE.Vector3(),
      hasLastSeen: false,
      searchT: 0,
      suppress: 0,
      ammo: MAG_SIZE, burst: 0, burstT: 0, fireT: 0,
      aggression: 0.55 + r() * 0.45,
      accuracy: 0.30 + r() * 0.35,
      reaction: this.reactionTime * (0.7 + r() * 0.9),
      thinkT: r() * 0.4,
      coverIdx: -1,
      postIndex: 0,
      strafeDir: r() < 0.5 ? -1 : 1,
      distToPlayer: 999,
      lod: 3,
      animAcc: 0,
      seed: r(),

      // death
      deathT: 0, fadeT: 0, respawnT: 0, settled: 0, settleTimer: 0,
      rag: null,
      handle: null,

      // hitboxes
      hbHead: new THREE.Box3(), hbTorso: new THREE.Box3(),
      hbLimb: [new THREE.Box3(), new THREE.Box3(), new THREE.Box3(), new THREE.Box3()],
      hbPos: new THREE.Vector3(),
      directDmgFrame: -1, directDmgAmt: 0,
    };

    u.rag = this._makeRagdoll();
    u.handle = this._makeHandle(u);
    this._handleMap.set(u.handle, u);
    this._handleMap.set(mesh, u);
    this._handleMap.set(root, u);
    return u;
  }

  _makeRagdoll() {
    const n = Enemies.RAG.length;
    const p = [], q = [], f = [];
    for (let i = 0; i < n; i++) { p.push(new THREE.Vector3()); q.push(new THREE.Vector3()); f.push(new THREE.Vector3()); }
    return { p, prev: q, tmp: f, rest: null, links: null, radius: new Float32Array(n), active: false, energy: 0 };
  }

  _makeHandle(u) {
    const self = this;
    return {
      id: u.id,
      team: 'opfor',
      alive: true,
      dead: false,
      hp: u.hp,
      maxHp: u.maxHp,
      health: u.hp,
      radius: 0.34,
      height: 1.80,
      position: u.hbPos,
      head: u.hbHead,
      torso: u.hbTorso,
      limbs: u.hbLimb,
      hitboxes: [
        { part: 'head', box: u.hbHead, mul: 1 },
        { part: 'torso', box: u.hbTorso, mul: 1 },
        { part: 'limb', box: u.hbLimb[0], mul: 0.72 },
        { part: 'limb', box: u.hbLimb[1], mul: 0.72 },
        { part: 'limb', box: u.hbLimb[2], mul: 0.72 },
        { part: 'limb', box: u.hbLimb[3], mul: 0.72 },
      ],
      object3D: u.mesh, object: u.mesh, mesh: u.mesh, root: u.root,
      raycast(origin, dir, maxDist) { return self._handleRaycast(u, origin, dir, maxDist); },
      applyDamage(amount, opts) { return self._damage(u, amount, opts, true); },
      damage(amount, opts) { return self._damage(u, amount, opts, true); },
      hit(amount, opts) { return self._damage(u, amount, opts, true); },
      getState() { return STATE_NAMES[u.state]; },
    };
  }

  _makePosts() {
    // A compound-shaped default layout.  Once the level's colliders exist the
    // posts get snapped onto real cover, but this pattern already reads as a
    // defended position on an empty plate.
    const r = this.rand;
    this._posts.length = 0;
    const rings = [
      { rad: 12, n: 4, z: 27 },
      { rad: 21, n: 5, z: 35 },
      { rad: 32, n: 5, z: 46 },
    ];
    for (const ring of rings) {
      for (let i = 0; i < ring.n; i++) {
        const a = (i / Math.max(1, ring.n - 1) - 0.5) * 1.45 + (r() - 0.5) * 0.20;
        const x = Math.sin(a) * ring.rad;
        const z = ring.z - Math.cos(a) * ring.rad * 0.40;
        this._posts.push({ x, z, yaw: Math.atan2(-x, -z) + (r() - 0.5) * 0.45 });
      }
    }
    this._posts.push({ x: -19, z: 13, yaw: 2.1 });
    this._posts.push({ x: 20, z: 11, yaw: -2.1 });
  }

  /* ======================================================================== *
   *  SPAWN / RESPAWN
   * ======================================================================== */

  _respawnUnit(u, x, z, yaw) {
    u.alive = true; u.dead = false;
    u.hp = u.maxHp;
    u.handle.alive = true; u.handle.dead = false; u.handle.hp = u.hp; u.handle.health = u.hp;
    u.state = S_PATROL; u.stateT = 0;
    u.awareness = 0; u.alerted = false; u.suppress = 0; u.reactT = 0;
    u.ammo = MAG_SIZE; u.burst = 0; u.burstT = 0; u.reloadT = -1;
    u.hasLastSeen = false; u.hasMoveTo = false; u.searchT = 0;
    this._releaseCover(u);
    u.deathT = 0; u.fadeT = 0; u.respawnT = 0; u.settled = 0; u.settleTimer = 0;
    u.rag.active = false; u.ragAcc = 0;
    u.flinch = 0; u.flinchVel = 0; u.recoil = 0; u.recoilVel = 0;
    u.stance = 0; u.stanceTarget = 0;
    u.speed = 0; u.speedTarget = 0; u.vel.set(0, 0, 0);
    u.footInit = false;
    u.pelvisDrop = 0; u.lean = 0; u.leanVel = 0; u.bob = 0;
    u.aimYaw = u.idleAimYaw; u.aimPitch = u.idleAimPitch;
    u.headYaw = 0; u.headPitch = 0;
    u.carry = u.pCarry; u.carryVel = 0;
    u.weaponPitch = u.carry * CARRY_DROP; u.weaponCant = u.pCant;
    u.flash.visible = false; u.flashT = 0;

    u.groundY = this._groundAt(x, z, 0);
    u.pos.set(x, u.groundY, z);
    u.yaw = yaw; u.yawTarget = yaw;
    u.root.position.copy(u.pos);
    u.root.quaternion.identity();
    u.root.rotation.set(0, yaw, 0);
    u.root.scale.setScalar(u.hScale);
    u.mesh.visible = true;

    for (let i = 0; i < BONE_COUNT; i++) {
      u.pose[i * 3] = this._restEuler[i * 3];
      u.pose[i * 3 + 1] = this._restEuler[i * 3 + 1];
      u.pose[i * 3 + 2] = this._restEuler[i * 3 + 2];
      u.bones[i].rotation.set(u.pose[i * 3], u.pose[i * 3 + 1], u.pose[i * 3 + 2]);
      u.bones[i].position.set(BONES[i].off[0], BONES[i].off[1], BONES[i].off[2]);
    }
    u.matSoft.transparent = false; u.matSoft.opacity = 1; u.matSoft.depthWrite = true;
    u.matHard.transparent = false; u.matHard.opacity = 1; u.matHard.depthWrite = true;
    u.root.updateMatrixWorld(true);
    this._updateHitboxes(u);
  }

  /* ======================================================================== *
   *  WORLD QUERIES (tolerant of a missing / half-built Physics)
   * ======================================================================== */

  _rayFirst(origin, dir, maxDist) {
    const ph = this.g?.physics;
    if (!ph) return null;
    try {
      let r = null;
      if (typeof ph.raycast === 'function') r = ph.raycast(origin, dir, maxDist);
      else if (typeof ph.ray === 'function') r = ph.ray(origin, dir, maxDist);
      else if (typeof ph.castRay === 'function') r = ph.castRay(origin, dir, maxDist);
      else return null;
      if (!r || r === true) return null;
      if (typeof r === 'number') return (r >= 0 && r <= maxDist) ? r : null;
      if (r.hit === false) return null;
      let d = r.distance;
      if (d === undefined) d = r.dist;
      if (d === undefined) d = r.t;
      if (d === undefined && r.point) d = origin.distanceTo(r.point);
      if (d === undefined || d === null) return null;
      return d <= maxDist ? r : null;
    } catch (e) {
      if (!this._physWarned) {
        this._physWarned = true;
        console.warn('[Enemies] physics raycast unavailable — using flat ground / clear LOS');
      }
      return null;
    }
  }

  _hitDistance(r, origin) {
    if (typeof r === 'number') return r;
    if (r.distance !== undefined) return r.distance;
    if (r.dist !== undefined) return r.dist;
    if (r.t !== undefined) return r.t;
    if (r.point) return origin.distanceTo(r.point);
    return 0;
  }

  /** Ground height under (x,z).  Falls back to a flat plane at y = 0. */
  _groundAt(x, z, fallback) {
    const base = fallback ?? 0;
    const o = this._v10.set(x, base + 3.2, z);
    const d = this._v11.set(0, -1, 0);
    const r = this._rayFirst(o, d, 9.0);
    if (r) {
      const dist = this._hitDistance(r, o);
      if (dist >= 0 && dist <= 9.0) return o.y - dist;
    }
    return base;
  }

  /** true if something solid sits between a and b. */
  _blocked(a, b) {
    const d = this._v12.copy(b).sub(a);
    const len = d.length();
    if (len < 0.05) return false;
    d.multiplyScalar(1 / len);
    return !!this._rayFirst(a, d, len - 0.15);
  }

  /* ---- cover -------------------------------------------------------------- */

  _boxOf(c, out) {
    if (!c) return false;
    if (c.isBox3) { out.copy(c); return true; }
    const b = c.box3 || c.box || c.aabb || c.bounds;
    if (b) {
      if (b.isBox3) { out.copy(b); return true; }
      if (b.min && b.max) { out.min.copy(b.min); out.max.copy(b.max); return true; }
    }
    if (c.min && c.max && c.min.isVector3) { out.min.copy(c.min); out.max.copy(c.max); return true; }
    const h = c.halfExtents || c.half || c.extents;
    if (c.center && h) {
      out.min.set(c.center.x - h.x, c.center.y - h.y, c.center.z - h.z);
      out.max.set(c.center.x + h.x, c.center.y + h.y, c.center.z + h.z);
      return true;
    }
    const obj = c.isObject3D ? c : (c.mesh?.isObject3D ? c.mesh : (c.object?.isObject3D ? c.object : null));
    if (obj) { try { out.setFromObject(obj); return !out.isEmpty(); } catch (e) { return false; } }
    return false;
  }

  _buildCover() {
    const cols = this.g?.registry?.colliders;
    this._cover.length = 0;
    this._coverBuilt = true;
    if (!Array.isArray(cols) || cols.length === 0) return;

    const box = this._boxScratch;
    const MAXC = 300;
    for (let i = 0; i < cols.length && this._cover.length < MAXC; i++) {
      if (!this._boxOf(cols[i], box)) continue;
      const sx = box.max.x - box.min.x, sy = box.max.y - box.min.y, sz = box.max.z - box.min.z;
      if (!isFinite(sx) || !isFinite(sy) || !isFinite(sz)) continue;
      if (sy < 0.55 || sy > 6.0) continue;      // not cover: too low, or a whole building face
      if (sx < 0.5 && sz < 0.5) continue;       // a post is not cover
      if (sx > 26 || sz > 26) continue;         // ground plane
      const cx = (box.min.x + box.max.x) * 0.5, cz = (box.min.z + box.max.z) * 0.5;
      const cy = box.min.y;
      const h = Math.min(sy, 2.2);
      for (let f = 0; f < 4; f++) {
        const nx = this._coverFaces[f][0], nz = this._coverFaces[f][1];
        const ex = nx !== 0 ? sx * 0.5 + 0.62 : 0;
        const ez = nz !== 0 ? sz * 0.5 + 0.62 : 0;
        const along = nx !== 0 ? sz : sx;
        const steps = along > 3.2 ? 3 : (along > 1.4 ? 2 : 1);
        for (let s = 0; s < steps; s++) {
          const t = steps === 1 ? 0 : (s / (steps - 1) - 0.5) * Math.min(along * 0.62, 3.4);
          this._cover.push({
            x: cx + nx * ex + (nx !== 0 ? 0 : t),
            y: cy,
            z: cz + nz * ez + (nz !== 0 ? 0 : t),
            nx, nz, h, taken: -1,
          });
          if (this._cover.length >= MAXC) break;
        }
        if (this._cover.length >= MAXC) break;
      }
    }
  }

  _snapPosts() {
    if (this._cover.length === 0) return;
    const n = Math.min(this._posts.length, this._cover.length);
    for (let i = 0; i < n; i++) {
      const c = this._cover[(i * 5 + 1) % this._cover.length];
      this._posts[i].x = c.x + c.nx * 0.25;
      this._posts[i].z = c.z + c.nz * 0.25;
      this._posts[i].yaw = Math.atan2(c.nx, c.nz);
    }
  }

  /* ======================================================================== *
   *  BUS HANDLERS
   * ======================================================================== */

  _resolveUnit(x) {
    if (!x) return null;
    let u = this._handleMap.get(x);
    if (u) return u;
    if (typeof x === 'number') return this.units[x] ?? null;
    if (x.isObject3D) {
      let o = x;
      for (let i = 0; i < 6 && o; i++) { u = this._handleMap.get(o); if (u) return u; o = o.parent; }
      return null;
    }
    if (x.enemy) return this._resolveUnit(x.enemy);
    if (x.handle) return this._resolveUnit(x.handle);
    return null;
  }

  _handleHit(d) {
    if (!d) return;
    const u = this._resolveUnit(d.enemy ?? d.target ?? d.object);
    if (!u || !u.alive) return;
    const amt = d.damage ?? d.amount ?? 0;
    // de-dup: Ballistics may both call applyDamage() and emit 'hit'
    if (u.directDmgFrame === (this.g?.frame ?? -1) && Math.abs(u.directDmgAmt - amt) < 1e-4) return;
    this._damage(u, amt, d, false);
  }

  _handleShot(d) {
    if (!d || d.source === 'enemy') return;
    const o = d.origin, dir = d.dir ?? d.direction;
    if (!o || !dir) return;
    // suppression + alert from rounds cracking past
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) continue;
      const px = u.pos.x - o.x, py = (u.pos.y + 1.1) - o.y, pz = u.pos.z - o.z;
      const t = px * dir.x + py * dir.y + pz * dir.z;
      if (t < 0 || t > 260) continue;
      const cx = px - dir.x * t, cy = py - dir.y * t, cz = pz - dir.z * t;
      const d2 = cx * cx + cy * cy + cz * cz;
      if (d2 < 6.25) {
        const near = 1 - Math.sqrt(d2) / 2.5;
        u.suppress = Math.min(1.6, u.suppress + 0.30 + near * 0.55);
        u.awareness = Math.min(1.35, u.awareness + 0.50 + near * 0.6);
        if (!u.alerted) this._alertUnit(u, o.x, o.y, o.z);
      }
    }
    this._alarm = Math.min(1, this._alarm + 0.12);
  }

  _handleImpact(d) {
    if (!d?.point) return;
    const p = d.point;
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) continue;
      const dx = u.pos.x - p.x, dz = u.pos.z - p.z, dy = (u.pos.y + 1.0) - p.y;
      if (dx * dx + dy * dy + dz * dz < 16) {
        u.suppress = Math.min(1.6, u.suppress + 0.30);
        u.awareness = Math.min(1.35, u.awareness + 0.40);
        if (!u.alerted) this._alertUnit(u, p.x, p.y, p.z);
      }
    }
  }

  _handleExplosion(d) {
    if (!d?.point) return;
    const p = d.point, R = d.radius ?? 6, power = d.power ?? 100;
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) continue;
      const dx = u.pos.x - p.x, dy = (u.pos.y + 1.0) - p.y, dz = u.pos.z - p.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > R * 1.8) continue;
      const f = clamp01(1 - dist / (R * 1.8));
      u.suppress = Math.min(1.8, u.suppress + f * 1.4);
      this._alertUnit(u, p.x, p.y, p.z);
      if (dist < R) {
        this._v1.set(dx, dy + 0.3, dz);
        if (this._v1.lengthSq() < 1e-6) this._v1.set(0, 1, 0);
        this._v1.normalize();
        this._damage(u, power * f * 0.9, { dir: this._v1, point: p, part: 'torso' }, false);
      }
    }
  }

  /* ======================================================================== *
   *  DAMAGE
   * ======================================================================== */

  _damage(u, amount, opts, direct) {
    if (!u || !u.alive) return false;
    const amt = Math.max(0, amount || 0);
    if (direct) { u.directDmgFrame = this.g?.frame ?? -1; u.directDmgAmt = amt; }

    const headshot = !!(opts && (opts.headshot || opts.part === 'head'));
    u.hp -= amt;
    u.handle.hp = u.hp; u.handle.health = u.hp;

    const dir = opts?.dir ?? opts?.direction ?? null;
    if (dir) {
      const c = Math.cos(-u.yaw), s = Math.sin(-u.yaw);
      u.flinchDir = clamp(dir.x * c - dir.z * s, -1, 1);
    } else {
      u.flinchDir = u.seed < 0.5 ? -1 : 1;
    }

    if (u.hp <= 0) { this._kill(u, headshot, opts); return true; }

    u.flinch = Math.min(1.2, u.flinch + 0.55 + Math.min(0.5, amt / 60));
    u.suppress = Math.min(1.8, u.suppress + 0.55);
    u.awareness = 1.35;
    const p = this._playerPos;
    this._alertUnit(u, p.x, p.y, p.z);
    if (u.state !== S_SEEKCOVER && u.hp < u.maxHp * 0.6 && Math.random() < 0.6) {
      this._setState(u, S_SEEKCOVER);
    }
    return false;
  }

  _kill(u, headshot, opts) {
    u.alive = false; u.dead = true; u.hp = 0;
    u.handle.alive = false; u.handle.dead = true; u.handle.hp = 0; u.handle.health = 0;
    u.state = S_DEAD; u.stateT = 0;
    u.deathT = 0; u.settled = 0; u.settleTimer = 0; u.ragAcc = 0;
    u.flash.visible = false;
    u.burst = 0;
    this._releaseCover(u);

    this._startRagdoll(u, opts, headshot);

    const msg = this._killMsg;
    msg.enemy = u.handle;
    msg.headshot = !!headshot;
    msg.distance = u.pos.distanceTo(this._playerPos);
    this.g?.bus?.emit('kill', msg);

    this._alarm = Math.min(1, this._alarm + 0.4);
    for (let i = 0; i < this.units.length; i++) {
      const o = this.units[i];
      if (!o.alive || o === u) continue;
      if (o.pos.distanceToSquared(u.pos) < 900) {
        o.awareness = Math.min(1.35, o.awareness + 0.7);
        if (this._hasAlarmPos) this._alertUnit(o, this._alarmPos.x, this._alarmPos.y, this._alarmPos.z);
      }
    }
  }

  _handleRaycast(u, origin, dir, maxDist) {
    if (!u.alive) return null;
    this._ray.origin.copy(origin);
    this._ray.direction.copy(dir).normalize();
    const md = maxDist ?? 500;
    let best = md, bestPart = null, bestMul = 1;
    const hit = this._v1, keep = this._v2;

    if (this._ray.intersectBox(u.hbHead, hit)) {
      const d = hit.distanceTo(origin);
      if (d < best) { best = d; bestPart = 'head'; bestMul = 1; keep.copy(hit); }
    }
    if (this._ray.intersectBox(u.hbTorso, hit)) {
      const d = hit.distanceTo(origin);
      if (d < best) { best = d; bestPart = 'torso'; bestMul = 1; keep.copy(hit); }
    }
    for (let i = 0; i < 4; i++) {
      if (this._ray.intersectBox(u.hbLimb[i], hit)) {
        const d = hit.distanceTo(origin);
        if (d < best) { best = d; bestPart = 'limb'; bestMul = 0.72; keep.copy(hit); }
      }
    }
    if (!bestPart) return null;
    const out = this._rcOut || (this._rcOut = {
      point: new THREE.Vector3(), distance: 0, part: '', headshot: false, mul: 1, enemy: null,
    });
    out.point.copy(keep);
    out.distance = best;
    out.part = bestPart;
    out.headshot = bestPart === 'head';
    out.mul = bestMul;
    out.enemy = u.handle;
    return out;
  }

  /* ======================================================================== *
   *  RAGDOLL — verlet chain on the bone hierarchy
   * ======================================================================== */

  static RAG = [
    B_PELVIS, B_CHEST, B_HEAD,
    B_ARML, B_FOREL, B_HANDL,
    B_ARMR, B_FORER, B_HANDR,
    B_THIGHL, B_SHINL, B_FOOTL,
    B_THIGHR, B_SHINR,
  ];

  static RAGLINKS = [
    [0, 1], [1, 2],
    [1, 3], [3, 4], [4, 5],
    [1, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11],
    [0, 12], [12, 13],
    // braces so the corpse keeps a body shape instead of turning into a noodle
    [0, 2], [3, 6], [0, 3], [0, 6], [2, 3], [2, 6],
    [9, 12], [1, 9], [1, 12], [0, 10], [0, 13], [5, 8],
  ];
  static RAG_STRUCT = 13;   // links below this index are stiff

  _startRagdoll(u, opts, headshot) {
    const R = Enemies.RAG, rag = u.rag;
    u.root.updateMatrixWorld(true);

    for (let i = 0; i < R.length; i++) {
      rag.p[i].setFromMatrixPosition(u.bones[R[i]].matrixWorld);
      rag.prev[i].copy(rag.p[i]);
      rag.radius[i] = (R[i] === B_HEAD) ? 0.145
        : (R[i] === B_CHEST || R[i] === B_PELVIS) ? 0.165 : 0.085;
    }

    rag.links = Enemies.RAGLINKS;
    if (!rag.rest) rag.rest = new Float32Array(rag.links.length);
    for (let i = 0; i < rag.links.length; i++) {
      rag.rest[i] = rag.p[rag.links[i][0]].distanceTo(rag.p[rag.links[i][1]]);
    }

    // launch impulse: bullet momentum + a slight lift on the upper body
    const dir = opts?.dir ?? opts?.direction;
    const push = this._v1;
    if (dir) { push.copy(dir); push.y = dir.y * 0.30 + 0.12; }
    else push.set(0, 0.25, -0.2);
    if (push.lengthSq() < 1e-6) push.set(0, 1, 0);
    push.normalize();
    const mag = headshot ? 2.6 : 1.7;
    const dt = 1 / 90;
    for (let i = 0; i < R.length; i++) {
      const w = (R[i] === B_HEAD ? 1.55 : R[i] === B_CHEST ? 1.15 : R[i] === B_PELVIS ? 0.85 : 0.60);
      rag.prev[i].addScaledVector(push, -mag * w * dt);
      rag.prev[i].x -= u.vel.x * dt;
      rag.prev[i].z -= u.vel.z * dt;
    }
    rag.active = true;
    rag.energy = 1;

    // one ground query — a corpse does not need per-frame terrain tracking
    u.groundY = this._groundAt(u.pos.x, u.pos.z, u.groundY);
  }

  _stepRagdoll(u, dt) {
    const rag = u.rag;
    const R = Enemies.RAG, n = R.length;
    const g = -18.5, dt2 = dt * dt, damp = 0.986, gy = u.groundY;

    let energy = 0;
    for (let i = 0; i < n; i++) {
      const p = rag.p[i], q = rag.prev[i], t = rag.tmp[i];
      t.copy(p);
      const vx = (p.x - q.x) * damp, vy = (p.y - q.y) * damp, vz = (p.z - q.z) * damp;
      p.x += vx; p.y += vy + g * dt2; p.z += vz;
      q.copy(t);
      energy += vx * vx + vy * vy + vz * vz;
    }

    const links = rag.links, rest = rag.rest, STRUCT = Enemies.RAG_STRUCT;
    for (let it = 0; it < 7; it++) {
      for (let i = 0; i < links.length; i++) {
        const a = links[i][0], bb = links[i][1];
        const pa = rag.p[a], pb = rag.p[bb];
        let dx = pb.x - pa.x, dy = pb.y - pa.y, dz = pb.z - pa.z;
        let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < 1e-6) { d = 1e-6; dx = 1e-6; }
        const stiff = i < STRUCT ? 0.5 : 0.24;
        const diff = ((d - rest[i]) / d) * stiff;
        dx *= diff; dy *= diff; dz *= diff;
        pa.x += dx; pa.y += dy; pa.z += dz;
        pb.x -= dx; pb.y -= dy; pb.z -= dz;
      }
      for (let i = 0; i < n; i++) {
        const p = rag.p[i], q = rag.prev[i];
        const floor = gy + rag.radius[i];
        if (p.y < floor) {
          p.y = floor;
          const fx = (p.x - q.x) * 0.60, fz = (p.z - q.z) * 0.60;
          q.x = p.x - fx; q.z = p.z - fz; q.y = p.y;
        }
      }
    }

    const speed = Math.sqrt(energy / n) / Math.max(dt, 1e-4);
    rag.energy = rag.energy * 0.85 + speed * 0.15;
    if (rag.energy < 0.25) { u.settleTimer += dt; if (u.settleTimer > 0.9) u.settled = 1; }
    else u.settleTimer = 0;

    this._ragdollToBones(u);
  }

  _ragdollToBones(u) {
    const rag = u.rag, W = this._rqW, P = rag.p;

    // the corpse is authored in world space: park the container under the pelvis
    // with no rotation so bone locals are trivially world-aligned
    u.root.position.copy(P[0]);
    u.root.quaternion.identity();
    u.pos.copy(P[0]); u.pos.y = u.groundY;

    const spineDir = this._v1.copy(P[1]).sub(P[0]);
    if (spineDir.lengthSq() < 1e-8) spineDir.set(0, 1, 0); else spineDir.normalize();
    const sideRef = this._v2.copy(P[6]).sub(P[3]);
    if (sideRef.lengthSq() < 1e-8) sideRef.set(1, 0, 0); else sideRef.normalize();
    const poleBack = this._v3.copy(spineDir).multiplyScalar(-1);

    this._ragAim(W, B_PELVIS, P[0], P[1], 1, sideRef);
    W[B_SPINE].copy(W[B_PELVIS]);
    this._ragAim(W, B_CHEST, P[1], P[2], 1, sideRef);
    W[B_NECK].copy(W[B_CHEST]);
    W[B_HEAD].copy(W[B_CHEST]);
    W[B_CLAVL].copy(W[B_CHEST]);
    W[B_CLAVR].copy(W[B_CHEST]);
    this._ragAim(W, B_ARML, P[3], P[4], -1, poleBack);
    this._ragAim(W, B_FOREL, P[4], P[5], -1, poleBack);
    W[B_HANDL].copy(W[B_FOREL]);
    this._ragAim(W, B_ARMR, P[6], P[7], -1, poleBack);
    this._ragAim(W, B_FORER, P[7], P[8], -1, poleBack);
    W[B_HANDR].copy(W[B_FORER]);
    this._ragAim(W, B_THIGHL, P[9], P[10], -1, spineDir);
    this._ragAim(W, B_SHINL, P[10], P[11], -1, spineDir);
    W[B_FOOTL].copy(W[B_SHINL]);
    W[B_TOEL].copy(W[B_SHINL]);
    this._ragAim(W, B_THIGHR, P[12], P[13], -1, spineDir);
    W[B_SHINR].copy(W[B_THIGHR]);
    W[B_FOOTR].copy(W[B_THIGHR]);
    W[B_TOER].copy(W[B_THIGHR]);

    for (let i = 0; i < BONE_COUNT; i++) {
      const p = BONES[i].parent;
      if (p < 0) u.bones[i].quaternion.copy(W[i]);
      else { this._q3.copy(W[p]).invert(); u.bones[i].quaternion.copy(this._q3).multiply(W[i]); }
    }
    u.bones[B_PELVIS].position.set(0, 0, 0);
    u.root.updateMatrixWorld(true);
  }

  _ragAim(W, bi, from, to, axis, pole) {
    const d = this._v4.copy(to).sub(from);
    if (d.lengthSq() < 1e-8) { W[bi].copy(this._bindQuat[bi]); return; }
    d.normalize();
    const y = this._v5.copy(d);
    if (axis < 0) y.multiplyScalar(-1);
    const z = this._v6.copy(pole);
    z.addScaledVector(y, -z.dot(y));
    if (z.lengthSq() < 1e-8) {
      z.set(0, 0, 1).addScaledVector(y, -y.z);
      if (z.lengthSq() < 1e-8) z.set(1, 0, 0).addScaledVector(y, -y.x);
    }
    z.normalize();
    const x = this._v7.crossVectors(y, z).normalize();
    this._m1.makeBasis(x, y, z);
    W[bi].setFromRotationMatrix(this._m1);
  }

  /* ======================================================================== *
   *  PERCEPTION + BEHAVIOUR
   * ======================================================================== */

  _alertUnit(u, x, y, z) {
    u.alerted = true;
    u.lastSeen.set(x, y ?? 1.5, z);
    u.hasLastSeen = true;
    u.searchT = 0;
    this._alarmPos.set(x, y ?? 1.5, z);
    this._hasAlarmPos = true;
    if (u.state === S_IDLE || u.state === S_PATROL) this._setState(u, S_ALERT);
  }

  alertAll(pos) {
    const p = pos ?? this._playerPos;
    for (const u of this.units) if (u.alive) this._alertUnit(u, p.x, p.y, p.z);
    this._alarm = 1;
  }

  _setState(u, s) {
    if (u.state === s) return;
    u.state = s;
    u.stateT = 0;
    if (s !== S_SEEKCOVER && s !== S_FLANK) u.hasMoveTo = false;
  }

  _perceive(u, dt) {
    const p = this._playerPos;
    const dx = p.x - u.pos.x, dz = p.z - u.pos.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    u.distToPlayer = dist;

    if (dist > this.sightRange) {
      u.canSee = false;
      u.awareness = Math.max(0, u.awareness - dt * 0.35);
      return;
    }

    const inv = 1 / Math.max(dist, 1e-4);
    const fx = Math.sin(u.yaw), fz = Math.cos(u.yaw);
    const dot = (dx * inv) * fx + (dz * inv) * fz;
    const inCone = dot > this.fovCos;
    const peripheral = dist < 9 && dot > -0.3;   // a man is not blind at arm's length

    u.losT -= dt;
    if (u.losT <= 0) {
      u.losT = 0.10 + u.lod * 0.16 + u.seed * 0.05;
      if (inCone || peripheral) {
        this._v8.copy(u.pos); this._v8.y += 1.42;
        u.losOK = !this._blocked(this._v8, this._playerEye);
      } else u.losOK = false;
    }

    u.canSee = (inCone || peripheral) && u.losOK;

    if (u.canSee) {
      const prox = 1 - clamp01(dist / this.sightRange);
      const centre = clamp01((dot - this.fovCos) / Math.max(1e-3, 1 - this.fovCos));
      const motion = clamp01(this._playerVel.length() * 0.22);
      const gain = (0.55 + prox * 1.9 + centre * 0.8 + motion * 0.7) / Math.max(0.05, u.reaction * 3.4);
      u.awareness = Math.min(1.35, u.awareness + gain * dt);
      u.lastSeen.copy(p); u.hasLastSeen = true;
      if (u.awareness >= 1 && !u.alerted) {
        this._alertUnit(u, p.x, p.y, p.z);
        this._alarm = Math.min(1, this._alarm + 0.35);
      }
    } else {
      u.awareness = Math.max(0, u.awareness - dt * (u.alerted ? 0.12 : 0.42));
    }
  }

  _think(u, dt) {
    const p = this._playerPos;
    u.stateT += dt;

    // squad alarm bleed: a shot pulls everyone in after their own reaction delay
    if (!u.alerted && this._alarm > 0.5 && this._hasAlarmPos) {
      u.reactT += dt;
      if (u.reactT > u.reaction * 3.2 + 0.5) {
        this._alertUnit(u, this._alarmPos.x, this._alarmPos.y, this._alarmPos.z);
      }
    }

    u.suppress = Math.max(0, u.suppress - dt * 0.55);

    switch (u.state) {

      case S_IDLE:
      case S_PATROL: {
        u.stanceTarget = u.postureStyle === 1 ? 0.70 : (u.postureStyle === 2 ? 0.28 : 0);
        const post = this._posts[u.postIndex % this._posts.length];
        if (!u.hasMoveTo) {
          const a = (u.seed * 6.283 + this._t * 0.05) % (Math.PI * 2);
          u.moveTo.set(post.x + Math.cos(a) * 3.4, 0, post.z + Math.sin(a) * 3.4);
          u.hasMoveTo = true;
        }
        const d = Math.hypot(u.moveTo.x - u.pos.x, u.moveTo.z - u.pos.z);
        if (d < 0.9) {
          u.hasMoveTo = false;
          u.speedTarget = 0;
          if (Math.random() < 0.35) u.yawTarget = post.yaw + (Math.random() - 0.5) * 1.6;
        } else {
          u.speedTarget = 1.35;
          u.yawTarget = Math.atan2(u.moveTo.x - u.pos.x, u.moveTo.z - u.pos.z);
        }
        if (u.awareness > 0.35) this._setState(u, S_SUSPECT);
        break;
      }

      case S_SUSPECT: {
        u.speedTarget = 0;
        u.stanceTarget = u.postureStyle === 1 ? 0.55 : 0;
        if (u.hasLastSeen) u.yawTarget = Math.atan2(u.lastSeen.x - u.pos.x, u.lastSeen.z - u.pos.z);
        if (u.awareness >= 1) this._setState(u, S_ALERT);
        else if (u.awareness < 0.12) this._setState(u, S_PATROL);
        break;
      }

      case S_ALERT: {
        u.stanceTarget = u.postureStyle === 1 ? 0.72 : (u.postureStyle === 2 ? 0.34 : 0);
        if (u.hasLastSeen) u.yawTarget = Math.atan2(u.lastSeen.x - u.pos.x, u.lastSeen.z - u.pos.z);
        if (u.canSee && u.awareness >= 1) { this._setState(u, S_ENGAGE); break; }
        u.searchT += dt;
        if (u.hasLastSeen) {
          const d = Math.hypot(u.lastSeen.x - u.pos.x, u.lastSeen.z - u.pos.z);
          if (d > 6) { u.moveTo.copy(u.lastSeen); u.hasMoveTo = true; u.speedTarget = 3.2; }
          else { u.hasMoveTo = false; u.speedTarget = 0.9; }
        }
        if (u.searchT > 9 && !u.canSee) { u.alerted = false; u.searchT = 0; this._setState(u, S_PATROL); }
        break;
      }

      case S_ENGAGE: {
        u.yawTarget = Math.atan2(p.x - u.pos.x, p.z - u.pos.z);
        if (u.suppress > 0.85) { this._setState(u, S_SUPPRESSED); break; }
        if (u.ammo <= 0) { this._setState(u, S_RELOAD); break; }
        if (!u.canSee) {
          u.stanceTarget = 0;
          if (u.stateT > 0.7 + u.reaction) this._setState(u, Math.random() < 0.5 ? S_FLANK : S_ALERT);
          break;
        }
        const inCover = u.coverIdx >= 0;
        const postureFloor = u.postureStyle === 1 ? 0.78 : (u.postureStyle === 2 ? 0.42 : 0);
        u.stanceTarget = Math.max(postureFloor,
          inCover ? (u.burst > 0 ? 0.25 : 0.75) : (u.suppress > 0.35 ? 0.55 : 0));
        const range = u.distToPlayer;
        if (range > 42) { u.moveTo.copy(p); u.hasMoveTo = true; u.speedTarget = 3.6; }
        else if (range < 7) {
          this._v1.copy(u.pos).sub(p); this._v1.y = 0;
          if (this._v1.lengthSq() < 1e-4) this._v1.set(0, 0, 1);
          this._v1.normalize().multiplyScalar(9);
          u.moveTo.copy(u.pos).add(this._v1); u.hasMoveTo = true; u.speedTarget = 3.0;
        } else if (!inCover) {
          if (u.stateT > 1.4) {
            u.stateT = 0;
            if (Math.random() < 0.45) { this._setState(u, S_SEEKCOVER); break; }
            u.strafeDir = -u.strafeDir;
          }
          // sidestep so he is never a static target
          this._v1.set(Math.cos(u.yaw) * u.strafeDir, 0, -Math.sin(u.yaw) * u.strafeDir).multiplyScalar(2.6);
          u.moveTo.copy(u.pos).add(this._v1); u.hasMoveTo = true; u.speedTarget = 1.7;
        } else { u.hasMoveTo = false; u.speedTarget = 0; }
        break;
      }

      case S_SUPPRESSED: {
        u.stanceTarget = 1;
        u.speedTarget = 0;
        u.hasMoveTo = false;
        if (u.hasLastSeen) u.yawTarget = Math.atan2(u.lastSeen.x - u.pos.x, u.lastSeen.z - u.pos.z);
        if (u.suppress < 0.35) {
          this._setState(u, (u.stateT > 3.5 && Math.random() < 0.6) ? S_SEEKCOVER : S_ENGAGE);
        }
        break;
      }

      case S_RELOAD: {
        u.speedTarget = u.coverIdx >= 0 ? 0 : 1.4;
        u.stanceTarget = u.coverIdx >= 0 ? 0.85 : 0.3;
        const before = u.reloadT;
        if (u.reloadT < 0) { u.reloadT = 0; this._emitReload('start', u); }
        u.reloadT += dt;
        if (before < 1.05 && u.reloadT >= 1.05) this._emitReload('magout', u);
        if (before < 1.85 && u.reloadT >= 1.85) this._emitReload('magin', u);
        if (u.reloadT > 2.75) {
          u.ammo = MAG_SIZE; u.reloadT = -1;
          this._emitReload('end', u);
          this._setState(u, u.canSee ? S_ENGAGE : S_ALERT);
        }
        break;
      }

      case S_SEEKCOVER: {
        u.stanceTarget = 0.15;
        if (u.coverIdx < 0) this._pickCover(u);
        if (u.coverIdx < 0) { this._setState(u, S_ENGAGE); break; }
        const c = this._cover[u.coverIdx];
        u.moveTo.set(c.x, 0, c.z); u.hasMoveTo = true;
        u.speedTarget = 5.4;
        const d = Math.hypot(c.x - u.pos.x, c.z - u.pos.z);
        u.yawTarget = d > 1.2
          ? Math.atan2(c.x - u.pos.x, c.z - u.pos.z)
          : Math.atan2(p.x - u.pos.x, p.z - u.pos.z);
        if (d < 0.75) { u.hasMoveTo = false; this._setState(u, u.ammo <= 0 ? S_RELOAD : S_ENGAGE); }
        else if (u.stateT > 7) { this._releaseCover(u); this._setState(u, S_ENGAGE); }
        break;
      }

      case S_FLANK: {
        u.stanceTarget = 0;
        if (!u.hasMoveTo || u.stateT > 5.5) {
          const a = Math.atan2(u.pos.x - p.x, u.pos.z - p.z) + u.strafeDir * (0.8 + Math.random() * 0.5);
          const rr = clamp(u.distToPlayer, 9, 26);
          u.moveTo.set(p.x + Math.sin(a) * rr, 0, p.z + Math.cos(a) * rr);
          u.hasMoveTo = true;
          u.stateT = 0;
        }
        u.speedTarget = 5.6;
        u.yawTarget = Math.atan2(u.moveTo.x - u.pos.x, u.moveTo.z - u.pos.z);
        const d = Math.hypot(u.moveTo.x - u.pos.x, u.moveTo.z - u.pos.z);
        if (d < 1.2 || u.canSee) { u.hasMoveTo = false; this._setState(u, S_ENGAGE); }
        break;
      }
    }

    if (u.ammo <= 0 && u.state !== S_RELOAD && u.state !== S_SUPPRESSED && u.state !== S_DEAD) {
      this._setState(u, S_RELOAD);
    }
  }

  _emitReload(phase, u) {
    const m = this._reloadMsg;
    m.phase = phase; m.enemy = u.handle;
    this.g?.bus?.emit('reload', m);
  }

  _releaseCover(u) {
    if (u.coverIdx >= 0 && this._cover[u.coverIdx]) this._cover[u.coverIdx].taken = -1;
    u.coverIdx = -1;
  }

  _pickCover(u) {
    this._releaseCover(u);
    const N = this._cover.length;
    if (N === 0) return;
    const p = this._playerPos;
    let best = -1, bestScore = -1e9;
    const samples = Math.min(N, 44);
    const start = ((u.id * 7 + (this._t * 3 | 0)) % N + N) % N;
    for (let s = 0; s < samples; s++) {
      const i = (start + s * 3) % N;
      const c = this._cover[i];
      if (c.taken >= 0 && c.taken !== u.id) continue;
      const dxp = p.x - c.x, dzp = p.z - c.z;
      const distP = Math.sqrt(dxp * dxp + dzp * dzp);
      if (distP < 4) continue;
      const dxu = c.x - u.pos.x, dzu = c.z - u.pos.z;
      const distU = Math.sqrt(dxu * dxu + dzu * dzu);
      if (distU > 34) continue;
      // the face normal must point at the threat — that is what makes it cover
      const facing = (dxp * c.nx + dzp * c.nz) / Math.max(distP, 1e-3);
      let score = facing * 44;
      score += (c.h > 1.25 ? 16 : 9);
      score -= distU * 1.5;
      score -= Math.abs(distP - 19) * 0.85;   // prefer a mid-range firing position
      if (distU < 1.0) score += 8;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    if (best >= 0) { u.coverIdx = best; this._cover[best].taken = u.id; }
  }

  /* ---- firing ------------------------------------------------------------ */

  _tryFire(u, dt) {
    u.fireT -= dt;
    if (u.burst > 0) {
      if (u.fireT <= 0) {
        this._fireRound(u);
        u.burst--;
        u.fireT = 60 / BURST_RPM;
        if (u.burst <= 0) u.burstT = 0.55 + Math.random() * 1.5 * (1.4 - u.aggression);
      }
      return;
    }
    u.burstT -= dt;
    if (u.burstT <= 0 && u.canSee && u.ammo > 0 && u.awareness >= 1 && u.state === S_ENGAGE) {
      // first contact is deliberately slow — nobody snaps onto a target
      if (u.stateT < u.reaction) return;
      u.burst = Math.min(u.ammo, 3 + (Math.random() * 4 | 0));
      u.fireT = 0;
    }
  }

  _fireRound(u) {
    if (u.ammo <= 0) return;
    u.ammo--;

    const hand = u.bones[B_HANDR];
    const mz = this._v1.copy(this._muzzleLocal).applyMatrix4(hand.matrixWorld);
    this._m1.extractRotation(hand.matrixWorld);
    const ax = this._v2.copy(this._rifleAxisLocal).applyMatrix4(this._m1).normalize();

    const tgt = this._v3.copy(this._playerPos);
    tgt.y += 0.95;
    tgt.addScaledVector(this._playerVel, 0.10 + u.distToPlayer * 0.0025);   // small lead
    const dir = this._v4.copy(tgt).sub(mz);
    const dist = Math.max(dir.length(), 0.001);
    dir.multiplyScalar(1 / dist);

    let spread = 0.020 + (1 - u.accuracy) * 0.030;
    spread *= 1 + u.suppress * 1.5;
    spread *= 1 - u.stance * 0.28;
    spread *= 1 + clamp01((dist - 25) / 60) * 1.1;
    spread *= 1 + clamp01(u.speed / 4) * 1.4;
    spread *= 1 + (1 - clamp01(ax.dot(dir))) * 2.4;   // he must be pointing it, too

    const a = Math.random() * Math.PI * 2, rr = Math.sqrt(Math.random()) * spread;
    const side = this._v5.set(-dir.z, 0, dir.x);
    if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
    side.normalize();
    const upv = this._v6.crossVectors(dir, side).normalize();
    dir.addScaledVector(side, Math.cos(a) * rr).addScaledVector(upv, Math.sin(a) * rr).normalize();

    // --- publish ------------------------------------------------------------
    this._shotIdx = (this._shotIdx + 1) & 15;
    const msg = this._shotRing[this._shotIdx];
    msg.origin.copy(mz);
    msg.dir.copy(dir);
    msg.enemy = u.handle;
    msg.damage = ENEMY_DAMAGE;
    msg.spread = spread;
    this.g?.bus?.emit('shot', msg);

    // --- resolve against the player unless Ballistics claims enemy fire -----
    if (!this.g?.ballistics?.handlesEnemyFire) {
      const pp = this._v7.copy(this._playerPos);
      pp.y += 0.90;
      const to = this._v8.copy(pp).sub(mz);
      const t = to.dot(dir);
      if (t > 0 && t < 240) {
        const cx = to.x - dir.x * t, cy = to.y - dir.y * t, cz = to.z - dir.z * t;
        const rad = Math.sqrt(cx * cx + cy * cy * 0.45 + cz * cz);
        if (rad < 0.42) {
          this._v9.copy(mz);
          if (!this._blocked(this._v9, pp)) {
            this._dmgIdx = (this._dmgIdx + 1) & 7;
            const d = this._dmgRing[this._dmgIdx];
            d.amount = ENEMY_DAMAGE * (rad < 0.20 ? 1.0 : 0.75);
            d.fromDir.copy(mz).sub(pp); d.fromDir.y = 0;
            if (d.fromDir.lengthSq() < 1e-6) d.fromDir.set(0, 0, 1);
            d.fromDir.normalize();
            d.from.copy(mz);
            d.point.copy(pp);
            d.enemy = u.handle;
            this.g?.bus?.emit('damage', d);
          }
        }
      }
    }

    // --- local FX ------------------------------------------------------------
    u.recoilVel += 9.0;
    u.flash.visible = true;
    u.flash.rotation.z = Math.random() * 6.283;
    u.flash.scale.setScalar(0.22 + Math.random() * 0.13);
    u.flashT = 0.045;

    this.muzzleLight.position.copy(mz).addScaledVector(dir, 0.25);
    this._muzzleLightT = 0.055;

    this._spawnTracer(mz, dir, dist);
  }

  _spawnTracer(from, dir, dist) {
    // every fourth round is a tracer, the way a real magazine is loaded
    this._tracerIdx++;
    if ((this._tracerIdx & 3) !== 0) return;
    let slot = -1;
    for (let i = 0; i < this.TRACER_MAX; i++) {
      const k = (this._tracerIdx + i) % this.TRACER_MAX;
      if (!this._tracer[k].active) { slot = k; break; }
    }
    if (slot < 0) return;
    const t = this._tracer[slot];
    t.active = true; t.t = 0;
    t.from.copy(from); t.dir.copy(dir);
    t.speed = 320; t.len = 6.5 + Math.random() * 2.5; t.width = 0.030;
    t.maxDist = Math.min(dist + 4, 190);
    t.life = t.maxDist / t.speed;
    t.travel = 0;
  }

  /* ======================================================================== *
   *  ANIMATION
   * ======================================================================== */

  _setPose(u, i, x, y, z, k, dt) {
    const b = i * 3;
    const w = 1 - Math.exp(-k * dt);
    u.pose[b] += (x - u.pose[b]) * w;
    u.pose[b + 1] += (y - u.pose[b + 1]) * w;
    u.pose[b + 2] += (z - u.pose[b + 2]) * w;
  }

  _animate(u, dt) {
    const R = this._restEuler;
    const moving = u.speed > 0.12;
    const run = clamp01((u.speed - 2.2) / 3.4);
    const walkAmt = clamp01(u.speed / 2.2) * (1 - run);
    const crouch = u.stance;

    // gait clock
    const stride = lerp(0.78, 1.42, run) * lerp(1, 0.72, crouch);
    const freq = moving ? clamp(u.speed / stride, 0.35, 3.2) : 0;
    u.phase = (u.phase + freq * dt) % 1;
    const ph = u.phase * Math.PI * 2;

    // springs
    const rc = u.recoil, rv = u.recoilVel;
    u.recoil = springX(rc, rv, 0, 34, dt); u.recoilVel = springV(rc, rv, 0, 34, dt);
    const fc = u.flinch, fv = u.flinchVel;
    u.flinch = springX(fc, fv, 0, 11, dt); u.flinchVel = springV(fc, fv, 0, 11, dt);
    const lc = u.lean, lv = u.leanVel;
    const leanTarget = moving ? run * 0.16 + walkAmt * 0.05 : 0;
    u.lean = springX(lc, lv, leanTarget, 9, dt); u.leanVel = springV(lc, lv, leanTarget, 9, dt);

    u.breathe += dt * (1.1 + u.suppress * 1.6 + run * 1.9);
    const breath = Math.sin(u.breathe) * 0.5 + Math.sin(u.breathe * 2.13) * 0.16;
    const breathAmp = lerp(0.027, 0.060, clamp01(u.suppress + run));

    // reload timeline
    let reloadArm = 0, reloadDip = 0;
    if (u.reloadT >= 0) {
      reloadArm = sstep(0, 0.35, u.reloadT) * (1 - sstep(2.1, 2.7, u.reloadT));
      reloadDip = Math.sin(clamp01(u.reloadT / 2.75) * Math.PI) * 0.55;
    }

    // aimPitch is POSITIVE = muzzle up.  Raising the muzzle means pitching the
    // chest BACK, i.e. a negative X rotation, so every aim term below subtracts.
    // (The old build added it, which inverted the closed loop and drove every
    // soldier to the clamp — that is why they all stood in an identical arch
    // with the rifle locked out at shoulder height.)
    const aimP = clamp(u.aimPitch, -0.85, 0.75);
    const aimY = clamp(u.aimYaw, -0.75, 0.75);
    const engaged = (u.state === S_ENGAGE || u.state === S_ALERT ||
      u.state === S_SUPPRESSED || u.state === S_RELOAD || u.state === S_SUSPECT);
    const ready = engaged ? 1 : 0;
    const twist = u.flinch * u.flinchDir;

    // ---- weapon carry: shouldered <-> compressed ready ----------------------
    const carryT = this._carryTarget(u);
    const cc = u.carry, cv = u.carryVel;
    u.carry = springX(cc, cv, carryT, 7.5, dt);
    u.carryVel = springV(cc, cv, carryT, 7.5, dt);
    // the weapon pivots at the firing wrist: carry drops it, the aim solution
    // lifts it, recoil kicks it up
    u.weaponPitch = u.carry * CARRY_DROP - aimP * AIM_WRIST_SHARE
      - u.recoil * 0.22 - reloadDip * 0.30;
    const aimSpine = aimP * (1 - AIM_WRIST_SHARE);

    // ---- idle life: weight shift, nobody standing at attention --------------
    const still = 1 - clamp01(u.speed * 1.6);
    const t = this._t;
    const sw1 = Math.sin(t * u.swayRate + u.swayPhase) * still;
    const sw2 = Math.sin(t * u.swayRate * 0.61 + u.swayPhase * 1.7 + 1.1) * still;
    const sw3 = Math.sin(t * u.swayRate * 1.37 + u.swayPhase * 0.6) * still;

    // Slow gesture beat. Each man fires a single soft pulse on his own clock and
    // his own channel: shift the weapon, roll the shoulders, check his sector or
    // change which hip he is standing on. Four channels x an unshared phase means
    // that in any one frame no two soldiers are in the same pose.
    const gcyc = (t * u.gestRate + u.gestPhase) % 1;
    const gp = gcyc < 0.30 ? Math.sin((gcyc * 3.3333333) * Math.PI) : 0;
    const gpulse = gp * gp * still;
    const gWeapon = u.gestKind === 0 ? gpulse : 0;
    const gShoulder = u.gestKind === 1 ? gpulse : 0;
    const gCheck = u.gestKind === 2 ? gpulse : 0;
    const gHip = u.gestKind === 3 ? gpulse : 0;
    // standing weight is carried on one hip, and the spine counter-leans
    const brace = u.braceLean * still * (1 - crouch * 0.55);

    // ---- pose archetype -----------------------------------------------------
    // The archetype owns the man's posture whenever he is not actually looking
    // through the sight; it fades to a third of strength as the weapon comes up
    // so a firing stance is still a firing stance.  This is what makes a line of
    // soldiers read as four different men doing four different jobs.
    const aw = 1 - 0.62 * ready * (1 - clamp01(u.carry * 1.4));
    const pLean = u.pLean * aw * (1 - crouch * 0.30);
    const pTwist = u.pTwist * aw;

    // ---- weapon idle sway ---------------------------------------------------
    // Two detuned sines on the muzzle and a slower one on the cant.  Amplitude
    // rides with how relaxed the carry is, so a man at low ready visibly breathes
    // the weapon around while a man on the sight holds it nearly still.
    const wsw = Math.sin(t * u.swayW + u.swayWPh) * 0.030
      + Math.sin(t * u.swayW * 1.63 + u.swayWPh * 1.7) * 0.012;
    u.weaponPitch += gWeapon * 0.24 + wsw * (0.34 + u.carry * 0.66) * (0.35 + still * 0.65);
    u.weaponCant = u.pCant * aw
      + Math.sin(t * u.swayW * 0.71 + u.swayWPh * 2.3) * 0.034 * (0.3 + still * 0.7)
      - twist * 0.10;

    this._setPose(u, B_PELVIS,
      R[B_PELVIS * 3] + crouch * 0.12 + u.lean * 0.35 + sw2 * 0.028,
      R[B_PELVIS * 3 + 1] + Math.sin(ph) * 0.05 * (walkAmt + run) - aimY * 0.10 + sw1 * 0.050,
      R[B_PELVIS * 3 + 2] + Math.sin(ph) * 0.045 * (walkAmt + run * 1.4) + sw1 * 0.062
      + brace * 0.42 + gHip * 0.10 + pLean * 0.34,
      18, dt);

    this._setPose(u, B_SPINE,
      R[B_SPINE * 3] + u.biasSpine + u.lean * 0.55 + crouch * 0.24 - aimSpine * 0.42
      + breath * breathAmp * 0.5 + u.recoil * 0.035 + reloadDip * 0.10 - u.suppress * 0.10
      + sw3 * 0.016,
      R[B_SPINE * 3 + 1] + aimY * 0.34 + twist * 0.16 + sw2 * 0.030 + pTwist * 0.38,
      R[B_SPINE * 3 + 2] - Math.sin(ph) * 0.030 * (walkAmt + run) + twist * 0.10 - sw1 * 0.040
      - brace * 0.26 - gHip * 0.07 + pLean * 0.30,
      14, dt);

    this._setPose(u, B_CHEST,
      R[B_CHEST * 3] - aimSpine * 0.58 + crouch * 0.10 + breath * breathAmp
      + u.recoil * 0.085 + reloadDip * 0.16 + u.lean * 0.30 - u.flinch * 0.10,
      R[B_CHEST * 3 + 1] + u.biasChestY + aimY * 0.62 - Math.sin(ph) * 0.055 * (walkAmt + run)
      + twist * 0.10 + sw3 * 0.026 + pTwist * 0.56,
      R[B_CHEST * 3 + 2] + twist * 0.14 + u.recoil * 0.03 - sw1 * 0.026 - brace * 0.18
      + pLean * 0.22,
      16, dt);

    // head tracks the threat independently of the torso; hp > 0 = looking up
    const hp = clamp(u.headPitch, -0.5, 0.5), hy = clamp(u.headYaw, -0.9, 0.9);
    // The head counter-rotates against the archetype's torso twist, so a man
    // turned 20 deg off his sector is still looking down it — which is exactly
    // what makes the twist read as a decision instead of a modelling error.
    this._setPose(u, B_NECK,
      R[B_NECK * 3] - hp * 0.45 - crouch * 0.10 - u.recoil * 0.05 + gCheck * 0.20
      + u.pChin * aw * 0.55,
      R[B_NECK * 3 + 1] + hy * 0.40 + gCheck * 0.22 - pTwist * 0.34,
      R[B_NECK * 3 + 2] - Math.sin(ph) * 0.02 + sw2 * 0.020 - pLean * 0.16,
      13, dt);
    this._setPose(u, B_HEAD,
      R[B_HEAD * 3] - hp * 0.55 - u.flinch * 0.12 + gCheck * 0.26 + u.pChin * aw * 0.45,
      R[B_HEAD * 3 + 1] + hy * 0.55 + gCheck * 0.30 - pTwist * 0.30,
      R[B_HEAD * 3 + 2] + u.biasHeadZ + twist * 0.10 + sw1 * 0.030 + gCheck * 0.14,
      12, dt);

    // weapon arm: recoil driven back into the shoulder
    this._setPose(u, B_CLAVR,
      R[B_CLAVR * 3] - u.recoil * 0.10 - u.carry * 0.075,
      R[B_CLAVR * 3 + 1] + u.recoil * 0.06 + u.carry * 0.05,
      R[B_CLAVR * 3 + 2] + u.biasClav + u.recoil * 0.05 - u.flinch * 0.06 + u.carry * 0.09
      + gShoulder * 0.13,
      24, dt);
    this._setPose(u, B_CLAVL,
      R[B_CLAVL * 3] - u.recoil * 0.05 + reloadArm * 0.10 - u.carry * 0.05,
      R[B_CLAVL * 3 + 1] - reloadArm * 0.12,
      R[B_CLAVL * 3 + 2] - u.flinch * 0.05 - u.biasClav * 0.6 - gShoulder * 0.12,
      22, dt);

    // The bind pose IS the shouldered pose, so these are small trims only —
    // the elbows never leave 65-80 deg of flexion.
    this._setPose(u, B_ARMR,
      R[B_ARMR * 3] + ready * 0.030 + u.recoil * 0.075 + breath * breathAmp * 0.6
      - crouch * 0.04 + u.carry * 0.10,
      R[B_ARMR * 3 + 1] - u.recoil * 0.05,
      R[B_ARMR * 3 + 2] + u.recoil * 0.045 - u.carry * 0.06,
      20, dt);
    this._setPose(u, B_FORER,
      R[B_FORER * 3] + u.recoil * 0.12 - u.carry * 0.12 + u.pElbow * aw,
      R[B_FORER * 3 + 1], R[B_FORER * 3 + 2], 20, dt);
    this._setPose(u, B_HANDR,
      R[B_HANDR * 3] - u.recoil * 0.06,
      R[B_HANDR * 3 + 1], R[B_HANDR * 3 + 2], 20, dt);

    // support arm — only matters when the grip IK is off (LOD 2)
    if (u.lod >= 2) {
      this._setPose(u, B_ARML, R[B_ARML * 3] + reloadArm * 0.55 + u.carry * 0.10,
        R[B_ARML * 3 + 1] - reloadArm * 0.30, R[B_ARML * 3 + 2] + reloadArm * 0.20, 14, dt);
      this._setPose(u, B_FOREL, R[B_FOREL * 3] + reloadArm * 0.45 - u.carry * 0.10,
        R[B_FOREL * 3 + 1], R[B_FOREL * 3 + 2], 14, dt);
      this._legsFK(u, dt, ph, walkAmt, run, crouch);
    }

    // pelvis vertical: gait bob + crouch + IK relief
    const bobAmt = (walkAmt * 0.026 + run * 0.055) * (1 - crouch * 0.4);
    u.bob = -Math.abs(Math.sin(ph)) * bobAmt - crouch * 0.36 - u.pelvisDrop
      - Math.abs(sw1) * 0.012;

    for (let i = 0; i < BONE_COUNT; i++) {
      const b = i * 3;
      u.bones[i].rotation.set(u.pose[b], u.pose[b + 1], u.pose[b + 2]);
    }
    // pivot the whole weapon about the firing wrist (see CARRY_DROP), then yaw
    // it across the chest by the archetype's cant.  The support arm follows both
    // for free through the grip IK, so the two-handed hold survives intact.
    if (this._carryAxis) {
      this._qCarry.setFromAxisAngle(this._carryAxis, u.weaponPitch);
      u.bones[B_HANDR].quaternion.multiply(this._qCarry);
    }
    if (this._cantAxis) {
      this._qCant.setFromAxisAngle(this._cantAxis, u.weaponCant);
      u.bones[B_HANDR].quaternion.multiply(this._qCant);
    }
    u.bones[B_PELVIS].position.set(sw1 * 0.020, BONES[B_PELVIS].off[1] + u.bob,
      u.lean * 0.05 + sw2 * 0.012);
  }

  /**
   * How far the weapon is off the shoulder for this soldier, 0..1.
   * Everything except an actual sight picture is blended toward the soldier's
   * pose archetype, so an alert squad still contains a high ready, a low ready
   * and a man with the carbine slung across his gut rather than five identical
   * compressed-ready clones.
   */
  _carryTarget(u) {
    const A = u.pCarry;
    switch (u.state) {
      case S_ENGAGE: return u.canSee ? 0.0 : lerp(0.20, A, 0.35);
      case S_SUPPRESSED: return lerp(0.30, A, 0.30);
      case S_RELOAD: return 0.75;
      case S_ALERT: return lerp(0.16 + u.carryBias * 0.16, A, 0.45);
      case S_SUSPECT: return lerp(0.28 + u.carryBias * 0.18, A, 0.55);
      default: return clamp(A + (u.carryBias - 0.5) * 0.18, 0.05, 0.95);
    }
  }

  /** Cheap sinusoidal legs for distant soldiers (no IK, no rays). */
  _legsFK(u, dt, ph, walkAmt, run, crouch) {
    const R = this._restEuler;
    const amp = walkAmt * 0.42 + run * 0.85;
    const sL = Math.sin(ph), sR = Math.sin(ph + Math.PI);
    const kL = Math.max(0, -Math.cos(ph)) * (walkAmt * 0.5 + run * 1.1);
    const kR = Math.max(0, -Math.cos(ph + Math.PI)) * (walkAmt * 0.5 + run * 1.1);
    this._setPose(u, B_THIGHL, R[B_THIGHL * 3] - sL * amp - crouch * 0.85, R[B_THIGHL * 3 + 1], R[B_THIGHL * 3 + 2], 18, dt);
    this._setPose(u, B_SHINL, R[B_SHINL * 3] + kL + crouch * 1.55, R[B_SHINL * 3 + 1], R[B_SHINL * 3 + 2], 18, dt);
    this._setPose(u, B_FOOTL, R[B_FOOTL * 3] + sL * amp * 0.35 - crouch * 0.55, 0, 0, 18, dt);
    this._setPose(u, B_THIGHR, R[B_THIGHR * 3] - sR * amp - crouch * 0.85, R[B_THIGHR * 3 + 1], R[B_THIGHR * 3 + 2], 18, dt);
    this._setPose(u, B_SHINR, R[B_SHINR * 3] + kR + crouch * 1.55, R[B_SHINR * 3 + 1], R[B_SHINR * 3 + 2], 18, dt);
    this._setPose(u, B_FOOTR, R[B_FOOTR * 3] + sR * amp * 0.35 - crouch * 0.55, 0, 0, 18, dt);
  }

  /**
   * Foot planting.  Feet are locked to world positions during stance and swung
   * along an arc to the next predicted plant, then the leg is IK'd.  The pelvis
   * is dropped whenever a leg would over-extend, which is what stops the classic
   * "skating on tiptoes" look.
   */
  _legsIK(u, dt) {
    const run = clamp01((u.speed - 2.2) / 3.4);
    const stride = lerp(0.72, 1.32, run) * lerp(1, 0.70, u.stance);
    const width = u.stanceWidth + u.stance * 0.050;
    const moving = u.speed > 0.14;
    const c = Math.cos(u.yaw), s = Math.sin(u.yaw);

    if (!u.footInit) {
      for (let i = 0; i < 2; i++) {
        const lx = (i ? -1 : 1) * width, lz = (i ? -0.06 : 0.06);
        const wx = u.pos.x + lx * c + lz * s;
        const wz = u.pos.z - lx * s + lz * c;
        u.footPlant[i].set(wx, this._groundAt(wx, wz, u.groundY), wz);
        u.footPrev[i].copy(u.footPlant[i]);
        u.footNext[i].copy(u.footPlant[i]);
      }
      u.footInit = true;
    }

    for (let i = 0; i < 2; i++) {
      const p = (u.phase + i * 0.5) % 1;
      const lx = (i ? -1 : 1) * width;

      if (moving) {
        const swinging = p > 0.60;
        if (swinging) {
          const k = (p - 0.60) / 0.40;
          if (u.footDown[i]) {
            u.footDown[i] = false;
            u.footPrev[i].copy(u.footPlant[i]);
            const ahead = stride * 0.55 + u.speed * 0.10;
            const wx = u.pos.x + lx * c + ahead * s;
            const wz = u.pos.z - lx * s + ahead * c;
            u.footNext[i].set(wx, this._groundAt(wx, wz, u.groundY), wz);
          }
          const e = k * k * (3 - 2 * k);
          u.footPlant[i].lerpVectors(u.footPrev[i], u.footNext[i], e);
          u.footPlant[i].y += Math.sin(k * Math.PI) * (0.085 + u.speed * 0.022);
        } else if (!u.footDown[i]) {
          u.footDown[i] = true;
          u.footPlant[i].copy(u.footNext[i]);
        }
      } else {
        // Standing: this is a FIGHTING stance, not a parade rest.  The support
        // side foot leads, the firing side foot trails and turns out, and the
        // whole thing is jittered per soldier so no two men plant identically.
        const lead = (i === u.leadSide);
        const lz = lead ? u.stanceLead : u.stanceTrail;
        const wx = u.pos.x + lx * c + lz * s;
        const wz = u.pos.z - lx * s + lz * c;
        const w = 1 - Math.exp(-6 * dt);
        u.footPlant[i].x += (wx - u.footPlant[i].x) * w;
        u.footPlant[i].z += (wz - u.footPlant[i].z) * w;
        u.footPlant[i].y += (u.groundY - u.footPlant[i].y) * w;
        u.footDown[i] = true;
      }
    }

    // pelvis relief: drop the hips until both ankles are reachable
    const hipL = this._v10.setFromMatrixPosition(u.bones[B_THIGHL].matrixWorld);
    const hipR = this._v11.setFromMatrixPosition(u.bones[B_THIGHR].matrixWorld);
    const legLen = (this._boneLen[B_THIGHL] + this._boneLen[B_SHINL]) * 0.985;
    const ankleUp = 0.055;
    let drop = 0;
    for (let i = 0; i < 2; i++) {
      const hip = i ? hipR : hipL;
      const d = Math.hypot(u.footPlant[i].x - hip.x,
        u.footPlant[i].y + ankleUp - hip.y, u.footPlant[i].z - hip.z);
      if (d > legLen) drop = Math.max(drop, d - legLen);
    }
    u.pelvisDrop += (Math.min(drop, 0.26) - u.pelvisDrop) * (1 - Math.exp(-14 * dt));

    for (let i = 0; i < 2; i++) {
      const th = i ? B_THIGHR : B_THIGHL, sh = i ? B_SHINR : B_SHINL, ft = i ? B_FOOTR : B_FOOTL;
      const hip = i ? hipR : hipL;
      // knee pole: forward from the soldier's facing, splayed a little outward
      const pole = this._v8.set(
        hip.x + s * 1.6 + c * (i ? -0.45 : 0.45),
        hip.y - 0.20,
        hip.z + c * 1.6 - s * (i ? -0.45 : 0.45));
      const target = this._v9.set(u.footPlant[i].x, u.footPlant[i].y + ankleUp, u.footPlant[i].z);
      this._ikTwoBone(u.bones[th], u.bones[sh], this._boneLen[th], this._boneLen[sh], target, pole, true);

      // lay the sole on the ground: build the desired world orientation directly
      const foot = u.bones[ft];
      const toeLift = u.footDown[i] ? 0 : 0.30;
      const splay = moving ? 0 : u.footSplay[i === u.leadSide ? 0 : 1];
      this._m1.extractRotation(u.bones[sh].matrixWorld);
      this._q1.setFromRotationMatrix(this._m1).invert();
      this._q2.setFromEuler(this._e1.set(-1.30 + toeLift * 0.5, u.yaw + splay, 0, 'YXZ'));
      foot.quaternion.copy(this._q1).multiply(this._q2);
      foot.updateMatrixWorld(true);
    }
  }

  /** Support-hand IK: put the left hand on the handguard, every frame. */
  _gripIK(u) {
    const hand = u.bones[B_HANDR];
    const target = this._v8.copy(this._handguardLocal).applyMatrix4(hand.matrixWorld);
    const sh = this._v9.setFromMatrixPosition(u.bones[B_ARML].matrixWorld);
    const c = Math.cos(u.yaw), s = Math.sin(u.yaw);
    const pole = this._v10.set(sh.x + c * 0.80 + s * 0.35, sh.y - 0.90, sh.z - s * 0.80 + c * 0.35);
    this._ikTwoBone(u.bones[B_ARML], u.bones[B_FOREL],
      this._boneLen[B_ARML], this._boneLen[B_FOREL], target, pole, true);
  }

  /* ---- aim closure -------------------------------------------------------- */

  _updateAim(u, dt) {
    const p = this._playerPos;
    const engaged = u.state === S_ENGAGE || u.state === S_SUPPRESSED ||
      u.state === S_ALERT || u.state === S_RELOAD || u.state === S_SUSPECT;

    if (!engaged) {
      // Not on a threat: stop closing the muzzle onto anything, let the weapon
      // sit at its carry angle and let the head scan the sector on its own
      // clock.  Each man has his own rate, phase and sweep, so a line of
      // patrolling soldiers never shares a head angle.
      const k = 1 - Math.exp(-2.2 * dt);
      u.aimYaw += (u.idleAimYaw - u.aimYaw) * k;
      u.aimPitch += (u.idleAimPitch - u.aimPitch) * k;
      const t = this._t;
      const scan = Math.sin(t * u.scanRate + u.scanPhase)
        + Math.sin(t * u.scanRate * 0.37 + u.scanPhase * 1.9) * 0.45;
      const nod = Math.sin(t * u.scanRate * 0.53 + u.scanPhase * 2.7);
      const hk = 1 - Math.exp(-3.0 * dt);
      u.headYaw += (scan * u.scanAmp * 0.62 - u.headYaw) * hk;
      u.headPitch += (nod * 0.09 - 0.05 - u.headPitch) * hk;
      return;
    }

    const tx = p.x;
    const ty = p.y + 0.95;
    const tz = p.z;

    const hand = u.bones[B_HANDR];
    const mz = this._v1.copy(this._muzzleLocal).applyMatrix4(hand.matrixWorld);
    this._m1.extractRotation(hand.matrixWorld);
    const ax = this._v2.copy(this._rifleAxisLocal).applyMatrix4(this._m1).normalize();
    const want = this._v3.set(tx - mz.x, ty - mz.y, tz - mz.z);
    if (want.lengthSq() < 1e-6) return;
    want.normalize();

    const curYaw = Math.atan2(ax.x, ax.z), curPitch = Math.asin(clamp(ax.y, -1, 1));
    const wantYaw = Math.atan2(want.x, want.z), wantPitch = Math.asin(clamp(want.y, -1, 1));
    const dYaw = angDelta(wantYaw, curYaw);
    const dPitch = wantPitch - curPitch;

    const rate = 5.5 + u.aggression * 5.0;
    const k = 1 - Math.exp(-rate * dt);
    u.aimYaw = clamp(u.aimYaw + dYaw * k, -0.85, 0.85);
    u.aimPitch = clamp(u.aimPitch + dPitch * k, -0.90, 0.80);

    // the eyes lead the weapon; both are "+ = up" now, so the head only has to
    // make up whatever elevation the torso did not
    const eyeYaw = angDelta(Math.atan2(tx - u.pos.x, tz - u.pos.z), u.yaw);
    const eyePitch = Math.atan2(ty - (u.pos.y + 1.55), Math.max(0.3, Math.hypot(tx - u.pos.x, tz - u.pos.z)));
    const hk = 1 - Math.exp(-7 * dt);
    u.headYaw += (clamp(eyeYaw, -1.1, 1.1) - u.aimYaw * 0.7 - u.headYaw) * hk;
    u.headPitch += (clamp(eyePitch, -0.6, 0.6) - u.aimPitch * 0.30 - u.headPitch) * hk;
  }

  /* ======================================================================== *
   *  MOVEMENT
   * ======================================================================== */

  _move(u, dt) {
    let wantX = 0, wantZ = 0;
    if (u.hasMoveTo && u.speedTarget > 0.05) {
      let dx = u.moveTo.x - u.pos.x, dz = u.moveTo.z - u.pos.z;
      const d = Math.hypot(dx, dz);
      if (d > 0.05) {
        dx /= d; dz /= d;
        // separation so a squad never stacks into one man
        for (let i = 0; i < this.units.length; i++) {
          const o = this.units[i];
          if (o === u || !o.alive) continue;
          const ox = u.pos.x - o.pos.x, oz = u.pos.z - o.pos.z;
          const od2 = ox * ox + oz * oz;
          if (od2 < 1.44 && od2 > 1e-4) {
            const dd = Math.sqrt(od2);
            const w = (1.2 - dd) * 1.4 / dd;
            dx += ox * w; dz += oz * w;
          }
        }
        // wall probe: steer around instead of grinding into geometry
        u.pathBlockT -= dt;
        if (u.pathBlockT <= 0 && u.lod <= 1) {
          u.pathBlockT = 0.18;
          this._v1.set(u.pos.x, u.pos.y + 0.95, u.pos.z);
          this._v2.set(dx, 0, dz).normalize();
          if (this._rayFirst(this._v1, this._v2, 1.5)) {
            const ca = Math.cos(u.strafeDir * 1.05), sa = Math.sin(u.strafeDir * 1.05);
            const nx = dx * ca + dz * sa, nz = -dx * sa + dz * ca;
            dx = nx; dz = nz;
          }
        }
        const n = Math.hypot(dx, dz) || 1;
        const sp = Math.min(u.speedTarget, d * 2.6 + 0.35);
        wantX = dx / n * sp; wantZ = dz / n * sp;
      }
    }

    const k = 1 - Math.exp(-14 * dt);
    u.vel.x += (wantX - u.vel.x) * k;
    u.vel.z += (wantZ - u.vel.z) * k;
    if (Math.abs(u.vel.x) < 0.02) u.vel.x = 0;
    if (Math.abs(u.vel.z) < 0.02) u.vel.z = 0;

    u.pos.x += u.vel.x * dt;
    u.pos.z += u.vel.z * dt;
    u.speed = Math.hypot(u.vel.x, u.vel.z);

    u.groundY += (this._groundAt(u.pos.x, u.pos.z, u.groundY) - u.groundY) * (1 - Math.exp(-12 * dt));
    u.pos.y = u.groundY;

    // A soldier's feet follow his eyes — but never dead square on the threat.
    // The per-man yaw bias is what stops a firing line from looking like a row
    // of identical cutouts all facing the camera; the aim closure makes up the
    // difference through the torso, which is what a real shooter does.
    const dy = angDelta(u.yawTarget + (u.speed > 0.5 ? 0 : u.yawBias), u.yaw);
    const turnRate = (u.state === S_ENGAGE || u.state === S_SUPPRESSED) ? 7.5 : 4.5;
    u.yaw += dy * (1 - Math.exp(-turnRate * dt));

    u.stance += (u.stanceTarget - u.stance) * (1 - Math.exp(-7 * dt));

    u.root.position.copy(u.pos);
    u.root.rotation.set(0, u.yaw, 0);
  }

  /* ======================================================================== *
   *  HITBOXES
   * ======================================================================== */

  _updateHitboxes(u) {
    const b = u.bones;
    const P = this._v1.setFromMatrixPosition(b[B_HEAD].matrixWorld);
    u.hbHead.min.set(P.x - 0.115, P.y - 0.035, P.z - 0.125);
    u.hbHead.max.set(P.x + 0.115, P.y + 0.215, P.z + 0.135);

    const a = this._v2.setFromMatrixPosition(b[B_PELVIS].matrixWorld);
    const c = this._v3.setFromMatrixPosition(b[B_NECK].matrixWorld);
    u.hbTorso.min.set(Math.min(a.x, c.x) - 0.235, Math.min(a.y, c.y) - 0.115, Math.min(a.z, c.z) - 0.190);
    u.hbTorso.max.set(Math.max(a.x, c.x) + 0.235, Math.max(a.y, c.y) + 0.055, Math.max(a.z, c.z) + 0.190);
    u.hbPos.set((a.x + c.x) * 0.5, (a.y + c.y) * 0.5, (a.z + c.z) * 0.5);

    this._limbBox(u.hbLimb[0], b[B_ARML], b[B_HANDL], 0.095);
    this._limbBox(u.hbLimb[1], b[B_ARMR], b[B_HANDR], 0.095);
    this._limbBox(u.hbLimb[2], b[B_THIGHL], b[B_FOOTL], 0.115);
    this._limbBox(u.hbLimb[3], b[B_THIGHR], b[B_FOOTR], 0.115);
  }

  _limbBox(box, b0, b1, r) {
    const a = this._v4.setFromMatrixPosition(b0.matrixWorld);
    const b = this._v5.setFromMatrixPosition(b1.matrixWorld);
    box.min.set(Math.min(a.x, b.x) - r, Math.min(a.y, b.y) - r, Math.min(a.z, b.z) - r);
    box.max.set(Math.max(a.x, b.x) + r, Math.max(a.y, b.y) + r, Math.max(a.z, b.z) + r);
  }

  /* ======================================================================== *
   *  FRAME
   * ======================================================================== */

  update(dt, t) {
    if (!this.units.length) return;
    this._t = t;

    // ---- player ------------------------------------------------------------
    const ctrl = this.g?.controller;
    const cam = this.g?.camera;
    if (ctrl?.pos) this._playerEye.copy(ctrl.pos);
    else if (cam) this._playerEye.copy(cam.position);
    this._playerPos.copy(this._playerEye);
    this._playerPos.y -= (this.g?.config?.player?.eyeHeight ?? 1.62);
    if (dt > 1e-5) {
      this._playerVel.copy(this._playerPos).sub(this._lastPlayerPos).multiplyScalar(1 / dt);
      if (this._playerVel.lengthSq() > 400) this._playerVel.set(0, 0, 0);   // teleport
    }
    this._lastPlayerPos.copy(this._playerPos);

    // ---- cover cache --------------------------------------------------------
    this._coverCheckAt -= dt;
    if (!this._coverBuilt || this._coverCheckAt <= 0) {
      this._coverCheckAt = 3.0;
      const n = this.g?.registry?.colliders?.length ?? 0;
      if (n !== this._lastColliderCount) {
        this._lastColliderCount = n;
        this._buildCover();
        this._snapPosts();
      }
    }

    this._alarm = Math.max(0, this._alarm - dt * 0.06);

    // ---- keep the character kicker pointed at the real sun -------------------
    // Cheap and rate-limited; the sun barely moves, but reading it means the warm
    // edge on a helmet always agrees with the key light the level is lit by.
    this._sunPollT -= dt;
    if (this._sunPollT <= 0) {
      this._sunPollT = 0.75;
      const sd = this.g?.sky?.sunDirection;
      if (sd && isFinite(sd.x) && (sd.x * sd.x + sd.y * sd.y + sd.z * sd.z) > 1e-4) {
        this._sunU.value.copy(sd).normalize();
      }
    }

    // ---- units --------------------------------------------------------------
    let alive = 0, dead = 0;
    const camPos = cam ? cam.position : this._playerEye;

    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      const dx = u.pos.x - camPos.x, dy = u.pos.y - camPos.y, dz = u.pos.z - camPos.z;
      const dcam = Math.sqrt(dx * dx + dy * dy + dz * dz);
      u.lod = dcam < 26 ? 0 : dcam < 55 ? 1 : dcam < 95 ? 2 : 3;

      if (u.alive) { alive++; this._updateAlive(u, dt); }
      else { dead++; this._updateDead(u, dt, dcam); }
    }

    this.stats.alive = alive;
    this.stats.dead = dead;

    this._updateFX(dt);
  }

  _updateAlive(u, dt) {
    // think, rate-limited by distance
    u.thinkT -= dt;
    const thinkStep = u.lod === 0 ? 0.05 : u.lod === 1 ? 0.09 : 0.20;
    if (u.thinkT <= 0) {
      const step = Math.min(0.5, thinkStep - u.thinkT);
      u.thinkT = thinkStep;
      this._perceive(u, step);
      this._think(u, step);
    }
    // firing needs frame resolution to keep the cadence honest
    if (u.state === S_ENGAGE) this._tryFire(u, dt);

    this._move(u, dt);

    // animation rate by LOD
    const animStep = u.lod === 0 ? 0 : u.lod === 1 ? 1 / 30 : 1 / 12;
    u.animAcc += dt;
    let step = dt;
    if (animStep > 0) {
      if (u.animAcc < animStep) { u.mesh.visible = u.lod < 3; return; }
      step = u.animAcc;
    }
    u.animAcc = 0;

    if (u.lod >= 3) { u.mesh.visible = false; return; }
    u.mesh.visible = true;

    this._animate(u, step);
    u.root.updateMatrixWorld(true);

    if (u.lod <= 1) {
      this._legsIK(u, step);
      this._gripIK(u);
    }
    this._updateAim(u, step);
    this._updateHitboxes(u);
  }

  _updateDead(u, dt, dcam) {
    u.deathT += dt;
    if (dcam > 130) { u.mesh.visible = false; }
    else if (u.fadeT <= 0) u.mesh.visible = true;

    if (u.rag.active && !u.settled) {
      // fixed-step the verlet so the collapse is identical at any framerate
      u.ragAcc += Math.min(dt, 0.05);
      let guard = 0;
      while (u.ragAcc > 1 / 90 && guard++ < 6) {
        this._stepRagdoll(u, 1 / 90);
        u.ragAcc -= 1 / 90;
      }
    }

    if (u.deathT > CORPSE_LINGER) {
      u.fadeT += dt;
      const a = clamp01(1 - u.fadeT / CORPSE_FADE);
      if (!u.matSoft.transparent) {
        u.matSoft.transparent = true; u.matHard.transparent = true;
        u.matSoft.depthWrite = false; u.matHard.depthWrite = false;
      }
      u.matSoft.opacity = a; u.matHard.opacity = a;
      if (a <= 0.001) {
        u.mesh.visible = false;
        u.respawnT += dt;
        if (u.respawnT > RESPAWN_DELAY) {
          // come back from a post the player is not staring at
          let best = 0, bestD = -1;
          for (let i = 0; i < this._posts.length; i++) {
            const p = this._posts[i];
            const d = Math.hypot(p.x - this._playerPos.x, p.z - this._playerPos.z);
            if (d > 22 && d > bestD) { bestD = d; best = i; }
          }
          const p = this._posts[best];
          u.postIndex = best;
          this._respawnUnit(u, p.x, p.z, p.yaw);
          if (this._alarm > 0.4 && this._hasAlarmPos) {
            this._alertUnit(u, this._alarmPos.x, this._alarmPos.y, this._alarmPos.z);
          }
        }
      }
    }
  }

  _updateFX(dt) {
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.flash.visible) continue;
      u.flashT -= dt;
      if (u.flashT <= 0) u.flash.visible = false;
    }

    if (this._muzzleLightT > 0) {
      this._muzzleLightT -= dt;
      const k = clamp01(this._muzzleLightT / 0.055);
      this.muzzleLight.intensity = 26 * k * k;
    } else if (this.muzzleLight.intensity !== 0) {
      this.muzzleLight.intensity = 0;
    }

    const cam = this.g?.camera;
    let any = false;
    for (let i = 0; i < this.TRACER_MAX; i++) {
      const tr = this._tracer[i];
      if (!tr.active) continue;
      any = true;
      tr.t += dt;
      tr.travel += tr.speed * dt;
      if (tr.t > tr.life || tr.travel > tr.maxDist) {
        tr.active = false;
        this._m1.makeScale(0, 0, 0);
        this.tracers.setMatrixAt(i, this._m1);
        continue;
      }
      const tail = Math.min(tr.len, tr.travel);
      const head = this._v1.copy(tr.from).addScaledVector(tr.dir, tr.travel);
      const mid = this._v2.copy(head).addScaledVector(tr.dir, -tail * 0.5);
      const toCam = this._v3.copy(cam ? cam.position : this._playerEye).sub(mid);
      const side = this._v4.crossVectors(tr.dir, toCam);
      if (side.lengthSq() < 1e-8) side.set(1, 0, 0); else side.normalize();
      const up = this._v5.crossVectors(side, tr.dir).normalize();
      // the quad's local +X is its width, +Y its length, +Z the facing normal
      this._m2.makeBasis(
        this._v6.copy(side).multiplyScalar(tr.width),
        this._v7.copy(tr.dir).multiplyScalar(tail),
        this._v8.copy(up));
      this._m2.setPosition(mid.x, mid.y, mid.z);
      this.tracers.setMatrixAt(i, this._m2);
    }
    if (any || this._tracerDirty) {
      this.tracers.instanceMatrix.needsUpdate = true;
      this._tracerDirty = any;
    }
  }

  /* ======================================================================== *
   *  DEBUG / AUTHORING HELPERS
   * ======================================================================== */

  /** Line the squad up in front of a camera at graduated distances. */
  showcase(cx = 0, cz = 18, yaw = Math.PI) {
    // `yaw` is the CAMERA yaw as used by Controller.teleport: a camera looks
    // along (-sin, 0, -cos), so that is the axis the squad is laid out on.
    const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
    const rx = Math.cos(yaw), rz = -Math.sin(yaw);
    const layout = [
      [-1.5, 5.0], [1.1, 6.5], [3.0, 8.5],
      [-3.6, 11], [1.4, 13], [4.8, 12],
      [-6.2, 19], [2.2, 21], [7.4, 24],
      [-9.5, 33], [4.4, 36], [11.0, 41],
      [-14.0, 55], [9.0, 63],
    ];
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      const l = layout[i % layout.length];
      const x = cx + fx * l[1] + rx * l[0];
      const z = cz + fz * l[1] + rz * l[0];
      // face back down the camera axis
      this._respawnUnit(u, x, z, Math.atan2(-fx, -fz) + (i % 5 - 2) * 0.22);
      u.state = i % 3 === 0 ? S_ENGAGE : (i % 3 === 1 ? S_ALERT : S_PATROL);
      u.alerted = true; u.awareness = 1;
      u.stanceTarget = (i % 5 === 3) ? 0.9 : (u.postureStyle === 1 ? 0.7 : 0);
      u.stance = u.stanceTarget;
      u.speedTarget = 0;
      u.hasMoveTo = false;
    }
    return this.units.length;
  }

  /** Force one visual state across the squad (screenshot helper). */
  setStance(name) {
    const map = {
      idle: S_IDLE, patrol: S_PATROL, aim: S_ALERT, fire: S_ENGAGE,
      crouch: S_SUPPRESSED, reload: S_RELOAD,
    };
    const s = map[name];
    if (s === undefined) return null;
    for (const u of this.units) {
      if (!u.alive) continue;
      this._setState(u, s);
      u.alerted = true; u.awareness = 1;
      if (s === S_RELOAD) { u.reloadT = 0.9; u.ammo = 0; }
      if (s === S_SUPPRESSED) u.suppress = 1.2;
    }
    return name;
  }

  resize() { }

  dispose() {
    const bus = this.g?.bus;
    if (bus) {
      if (this._onHit) bus.off('hit', this._onHit);
      if (this._onShot) bus.off('shot', this._onShot);
      if (this._onImpact) bus.off('impact', this._onImpact);
      if (this._onExplosion) bus.off('explosion', this._onExplosion);
    }
    for (const u of this.units) { u.matSoft.dispose(); u.matHard.dispose(); u.skeleton?.dispose?.(); }
    if (this.geometries) for (const g of this.geometries) g?.dispose?.();
    this.geometry = null;
    this.flashGeo?.dispose();
    this.flashMat?.dispose();
    this.flashTex?.dispose();
    this.tracers?.geometry?.dispose();
    this.tracers?.material?.dispose();
    this.matSoftProto?.dispose();
    this.matHardProto?.dispose();
    this.g?.scene?.remove(this.group);
  }
}

export default Enemies;
