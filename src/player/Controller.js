// ============================================================================
// BLACKSITE — src/player/Controller.js
// Owner: Player agent.  Movement, collision resolution and the CAMERA FEEL.
//
// The rendered image is somebody else's job.  This file owns the half of "looks
// like Call of Duty" that lives in the time domain: how the camera weighs, lags,
// settles, rolls and breathes.  Everything here is a curve with a timing on it —
// there is not a single linear lerp of a pose in this module.
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators — this is stable, read it)
// ---------------------------------------------------------------------------
//   game.controller.teleport(x, y, z, yaw, pitch)   place the CAMERA (eye) exactly
//        at x,y,z with the given orientation in radians.  Freezes the simulation
//        until real input arrives, so an automated screenshot never drifts.
//   game.controller.addShake(amount, duration, freq)
//        Summed decaying sine octaves — NOT random jitter.  amount 1.0 peaks at
//        ~0.8 deg of view pitch, ~1.2 deg of roll and ~2 cm of translation.
//        Weapon fire: (0.05..0.12, 0.10, 30).  Explosion at 5 m: (1.4, 0.9, 9).
//        Landing, mantling and sliding drive it internally.  Bus 'explosion' and
//        'damage' are already wired in — do not double-shake for those.
//   game.controller.setAds(bool)          force ADS state (also mirrors bus 'ads')
//   game.controller.getForward(outVec3)   unit forward including pitch
//   game.controller.getRight(outVec3)     unit right, horizontal
//   game.controller.getEyePosition(out)   final composed camera world position
//   game.controller.getMuzzleOrigin(out, fwdOffset, downOffset)
//
//   Live, read-only state every frame:
//     .pos          THREE.Vector3  final camera (eye) world position
//     .feet         THREE.Vector3  capsule base (foot) world position
//     .velocity     THREE.Vector3  world velocity, m/s
//     .speed        number         horizontal speed, m/s
//     .grounded     bool
//     .isAds        bool
//     .isSprinting  bool
//     .isCrouching  bool
//     .isSliding    bool
//     .isMantling   bool
//     .yaw / .pitch radians
//     .capsule      { radius, height }   height is the LIVE (crouch-blended) height
//     .surface      string  surface key under the player, from physics
//
// ---------------------------------------------------------------------------
// PHYSICS CONTRACT (what this module asks of game.physics)
// ---------------------------------------------------------------------------
//   physics.capsuleSweep(from, to, radius, height, out?) where `from`/`to` are the
//   capsule BASE (feet) position in world space.  Any of these return shapes is
//   understood, and a missing/short return simply means "no hit":
//       falsy | true/false
//       THREE.Vector3                      -> resolved end position
//       { position|pos|point|end, normal|n, hit|collided, fraction|t|toi,
//         grounded|onGround, surface|material }
//   The optional 5th argument is a reusable scratch object this module owns; a
//   physics implementation may fill `out.position` / `out.normal` / `out.hit` /
//   `out.fraction` / `out.surface` instead of allocating a return value.
//
//   Optional, used when present:  physics.surfaceAt(x, y, z) -> string
//
//   If physics is absent, throws, or never resolves anything, this module falls
//   back to its own AABB world built from game.registry.colliders plus a ground
//   plane at y = 0.  The player is never left floating and never falls forever.
// ============================================================================

import * as THREE from 'three';

/* ------------------------------------------------------------------ *
 *  Constants — everything that has no Config key lives here.
 * ------------------------------------------------------------------ */

const DEG = Math.PI / 180;
const TAU = Math.PI * 2;

// Look
const PITCH_LIMIT = Math.PI * 0.5 - 0.5 * DEG;   // "just under 90 degrees"

// Locomotion
const STOP_SPEED = 1.25;      // Quake-lineage friction floor: kills the last cm of drift
const STEP_HEIGHT = 0.45;     // stairs / kerbs this tall are walked over, not blocked
const GROUND_SNAP = 0.14;     // downward probe that keeps the capsule glued to slopes
const AIR_WISH_CAP = 1.05;    // air-strafe wishspeed cap (metres/s) — Quake air control
const COYOTE_TIME = 0.11;     // grace after walking off a ledge
const JUMP_BUFFER = 0.13;     // press-early grace before landing
const SPRINT_SPIN_UP = 0.34;  // seconds from walk to full sprint
const SPRINT_SPIN_DN = 0.16;

// Slide
const SLIDE_ENTER_SPEED = 4.3;
const SLIDE_BOOST = 1.17;
const SLIDE_FRICTION = 3.1;
const SLIDE_MIN_SPEED = 2.5;
const SLIDE_MAX_TIME = 1.45;
const SLIDE_COOLDOWN = 0.42;
const SLIDE_STEER = 62 * DEG; // radians/second the slide can be steered

// Mantle / vault
const MANTLE_REACH = 0.92;
const MANTLE_MIN_RISE = 0.34;
const MANTLE_MAX_RISE = 1.45;
const MANTLE_PROBE_R = 0.26;
const MANTLE_PROBE_H = 0.42;

// Lean
const LEAN_OFFSET = 0.34;     // metres the eye slides sideways
const LEAN_ROLL = 13.5 * DEG;
const LEAN_DROP = 0.055;      // you dip a little as you lean

// View bob — figure-8 Lissajous, phase advanced by distance travelled
const BOB_STRIDE = 3.72;      // metres per full 2-PI cycle (= 2 footfalls)
const BOB_PHASE_PER_M = TAU / BOB_STRIDE;
const BOB_AMP_X = 0.0335;     // metres of lateral sway at walk speed
const BOB_AMP_Y = 0.0215;     // metres of vertical bounce at walk speed
const BOB_ROLL = 0.62 * DEG;
const BOB_PITCH = 0.26 * DEG;
const BOB_ADS = 0.16;         // hard damping when aiming down sights
const FOOT_PHASE = Math.PI * 0.75;  // sin(2p) minimum -> the foot plant

// Landing spring
const LAND_K = 148;           // spring constant
const LAND_ZETA = 0.74;       // slightly underdamped: it settles with one small rebound
const LAND_MAX = 0.30;        // metres of dip at terminal impact

// Camera roll from strafing
const STRAFE_ROLL = 1.2 * DEG;

// Idle breathing sway
const SWAY_HIP = 0.17 * DEG;
const SWAY_ADS = 0.052 * DEG;

// Shake
const SHAKE_SLOTS = 16;
const SHAKE_OCTAVES = 3;
const SHAKE_PITCH_K = 0.0142; // radians per unit amount, per octave-sum
const SHAKE_YAW_K = 0.0118;
const SHAKE_ROLL_K = 0.0205;
const SHAKE_POS_K = 0.0215;

// Fallback collision world
const MAX_FB_BOXES = 3000;
const MAX_FB_CAND = 256;
const FB_GROUND_Y = 0.0;

/* ------------------------------------------------------------------ *
 *  Small math helpers (module scope so update() never builds closures)
 * ------------------------------------------------------------------ */

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

function smoothstep(e0, e1, x) {
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}

function smootherstep(x) {
  const t = x < 0 ? 0 : x > 1 ? 1 : x;
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// Frame-rate independent exponential approach factor. rate ~= 1/timeConstant.
function damp(rate, dt) {
  return 1 - Math.exp(-rate * dt);
}

/* ================================================================== *
 *  Controller
 * ================================================================== */

export class Controller {

  constructor(game) {
    this.g = game;
    const P = game?.config?.player ?? {};

    // ---- tunables pulled from Config (never hardcode what Config declares) --
    this.cfgP = P;
    this.standHeight = P.height ?? 1.75;
    this.standEye = P.eyeHeight ?? 1.62;
    this.crouchHeight = P.crouchHeight ?? 1.0;
    // The head sits the same distance below the top of the capsule in either
    // stance — that is what makes the crouch blend read as a body, not a scale.
    this.headDrop = this.standHeight - this.standEye;
    this.crouchEye = Math.max(0.42, this.crouchHeight - this.headDrop);
    this.slideHeight = Math.max(0.72, this.crouchHeight * 0.82);
    this.slideEye = Math.max(0.34, this.slideHeight - this.headDrop * 0.82);
    this.radius = P.radius ?? 0.32;

    // ---- orientation -------------------------------------------------------
    this.yaw = Math.PI;
    this.pitch = 0;

    // ---- position / motion -------------------------------------------------
    // `pos` is the EYE (camera) position — the stub's contract, and what the
    // screenshot critic teleports.  `feet` is the capsule base we simulate.
    this.pos = new THREE.Vector3(0, this.standEye, 18);
    this.feet = new THREE.Vector3(0, 0, 18);
    this.velocity = new THREE.Vector3();
    this.speed = 0;
    this.grounded = true;
    this.surface = 'concrete';

    this.capsule = { radius: this.radius, height: this.standHeight };
    this._capH = this.standHeight;
    this._eye = this.standEye;

    // ---- stance ------------------------------------------------------------
    this.isAds = false;
    this.isSprinting = false;
    this.isCrouching = false;
    this.isSliding = false;
    this.isMantling = false;
    this.frozen = true;                 // released by the first real input

    this._sprintT = 0;                  // 0..1 sprint spin-up
    this._slideT = 0;
    this._slideCool = 0;
    this._airTime = 0;
    this._coyote = 0;
    this._jumpBuf = 0;
    this._jumpEdge = false;
    this._fallVel = 0;
    this._wasGrounded = true;
    this._uncrouchBlocked = false;
    this._forwardHold = 0;              // seconds pushing into a wall (auto-vault)

    // ---- mantle ------------------------------------------------------------
    this._mtT = 0; this._mtDur = 0.44;
    this._mtFrom = new THREE.Vector3();
    this._mtTo = new THREE.Vector3();
    this._mtCool = 0;

    // ---- camera feel state -------------------------------------------------
    this._bobPhase = 0;
    this._bobAmp = 0;
    this._footIndex = 0;
    this._landY = 0; this._landV = 0;
    this._stepSmooth = 0;
    this._roll = 0;
    this._lean = 0; this._leanLimitL = 1; this._leanLimitR = 1;
    this._breath = Math.random() * 100;
    this._adsBlend = 0;
    this._recoverT = 0;

    // ---- shake pool (fixed, zero allocation) -------------------------------
    this._shAmp = new Float32Array(SHAKE_SLOTS);
    this._shDur = new Float32Array(SHAKE_SLOTS);
    this._shT = new Float32Array(SHAKE_SLOTS);
    this._shFrq = new Float32Array(SHAKE_SLOTS);
    this._shPh = new Float32Array(SHAKE_SLOTS * 3);
    this._shOn = new Uint8Array(SHAKE_SLOTS);
    this._shNext = 0;
    this._shakePitch = 0; this._shakeYaw = 0; this._shakeRoll = 0;
    this._shakePX = 0; this._shakePY = 0;

    // ---- input -------------------------------------------------------------
    this.keys = {
      f: false, b: false, l: false, r: false,
      jump: false, sprint: false, crouch: false, leanL: false, leanR: false,
    };
    this._locked = false;
    this._mouseDX = 0;
    this._mouseDY = 0;
    this._rmb = false;
    this._adsSelf = false;
    this._anyInput = false;

    // ---- scratch (every vector update() touches is preallocated) -----------
    this._vFwd = new THREE.Vector3(0, 0, -1);
    this._vRight = new THREE.Vector3(1, 0, 0);
    this._vFlat = new THREE.Vector3();
    this._vWish = new THREE.Vector3();
    this._vA = new THREE.Vector3();
    this._vB = new THREE.Vector3();
    this._vC = new THREE.Vector3();
    this._vFrom = new THREE.Vector3();
    this._vTo = new THREE.Vector3();
    this._vEye = new THREE.Vector3();
    this._qTmp = new THREE.Quaternion();

    // sweep results
    this._swEnd = new THREE.Vector3();
    this._swAdv = 1;        // fraction of the requested motion actually achieved
    this._swHit = false;    // obstructed along the intended direction
    this._swRich = false;   // backend reported explicit wall/ceiling/ground flags
    this._swWall = false;
    this._swCeil = false;
    this._swGround = false;
    this._swStep = 0;       // metres the backend stepped us up by
    this._swNX = 0; this._swNY = 0; this._swNZ = 0;
    this._swSurface = null;
    this._swOut = {
      position: new THREE.Vector3(),
      normal: new THREE.Vector3(),
      hit: false, fraction: 1, grounded: false, surface: null,
    };
    this._physBad = false;
    this._physOk = false;

    // fallback AABB world
    this._fbBoxes = null;
    this._fbSurf = null;
    this._fbCount = 0;
    this._fbCand = new Int32Array(MAX_FB_CAND);
    this._fbCandN = 0;
    this._fbLastLen = -1;
    this._fbTimer = 0;
    this._fbNormAxis = -1;
    this._fbNormSign = 0;

    // reusable bus payloads — emitted synchronously, consumed immediately
    this._pFoot = { surface: 'concrete', speed: 0 };
    this._pAds = { active: false };

    this._surfTimer = 0;
    this._leanTimer = 0;
    this._fovWritten = -1;
    this._fovContested = false;

    this._bound = false;
    this._bindInput();
    this._bindBus();
  }

  async init() {
    // Drop the capsule onto whatever the level put under the spawn point, so the
    // very first frame is already resting on the ground rather than mid-fall.
    this._syncFallbackWorld(true);
    this.feet.set(this.pos.x, this.pos.y - this.standEye, this.pos.z);
    this._probeGround(true);
    this._composeCamera(0, true);
  }

  // ==========================================================================
  //  DEBUG / TOOL API — the screenshot critic depends on this exact behaviour
  // ==========================================================================

  /**
   * Place the CAMERA at (x, y, z) with the given orientation, in radians.
   * The simulation is frozen at that pose until real player input arrives, so a
   * capture at 4.2 m in the air does not fall out of frame while TAA converges.
   */
  teleport(x, y, z, yaw, pitch) {
    this.pos.set(x, y, z);
    if (yaw !== undefined && yaw !== null && isFinite(yaw)) this.yaw = yaw;
    if (pitch !== undefined && pitch !== null && isFinite(pitch)) {
      this.pitch = clamp(pitch, -PITCH_LIMIT, PITCH_LIMIT);
    }

    // Reset every dynamic system so the pose is exactly the pose asked for.
    this.velocity.set(0, 0, 0);
    this.speed = 0;
    this.isSliding = false;
    this.isMantling = false;
    this.isSprinting = false;
    this._sprintT = 0;
    this._slideT = 0;
    this._slideCool = 0;
    this._mtT = 0;
    this._airTime = 0;
    this._fallVel = 0;
    this._bobPhase = 0;
    this._bobAmp = 0;
    this._landY = 0; this._landV = 0;
    this._stepSmooth = 0;
    this._roll = 0;
    this._lean = 0;
    this._forwardHold = 0;
    for (let i = 0; i < SHAKE_SLOTS; i++) this._shOn[i] = 0;
    this._shakePitch = this._shakeYaw = this._shakeRoll = 0;
    this._shakePX = this._shakePY = 0;

    this._eye = this.isCrouching ? this.crouchEye : this.standEye;
    this._capH = this.isCrouching ? this.crouchHeight : this.standHeight;
    this.capsule.height = this._capH;
    this.feet.set(x, y - this._eye, z);
    this.grounded = true;
    this._wasGrounded = true;
    this._coyote = COYOTE_TIME;

    this.frozen = true;
    this._anyInput = false;

    this._composeCamera(0, true);
    return this;
  }

  /**
   * Camera shake as summed decaying sine octaves. Deterministic per call, with a
   * per-slot phase offset so repeated calls never phase-lock into a buzz.
   *   amount    peak magnitude; 1.0 ~= 1 degree of pitch
   *   duration  seconds (default 0.35)
   *   freq      base frequency in Hz (default 14)
   */
  addShake(amount, duration, freq) {
    const a = +amount;
    if (!isFinite(a) || a <= 0) return this;
    const d = isFinite(duration) && duration > 0 ? +duration : 0.35;
    const f = isFinite(freq) && freq > 0 ? +freq : 14;

    // Prefer a free slot; otherwise steal the weakest one that is nearly done.
    let slot = -1, worst = -1, worstScore = Infinity;
    for (let i = 0; i < SHAKE_SLOTS; i++) {
      if (!this._shOn[i]) { slot = i; break; }
      const rem = 1 - this._shT[i] / this._shDur[i];
      const score = this._shAmp[i] * rem * rem;
      if (score < worstScore) { worstScore = score; worst = i; }
    }
    if (slot < 0) {
      if (worstScore >= a) return this;   // everything running is stronger
      slot = worst;
    }

    this._shOn[slot] = 1;
    this._shAmp[slot] = Math.min(a, 6);
    this._shDur[slot] = Math.min(d, 4);
    this._shT[slot] = 0;
    this._shFrq[slot] = Math.min(f, 60);
    const n = this._shNext++;
    this._shPh[slot * 3 + 0] = (n * 2.399963) % TAU;
    this._shPh[slot * 3 + 1] = (n * 4.129871 + 1.7) % TAU;
    this._shPh[slot * 3 + 2] = (n * 5.834017 + 3.1) % TAU;
    return this;
  }

  /** Force the aim-down-sights state. Emits bus 'ads' if it actually changed. */
  setAds(v) { this._setAds(!!v, true); return this; }

  getForward(out) {
    const o = out || this._vA;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    // camera looks down local -Z with Euler order YXZ
    o.set(-sy * cp, sp, -cy * cp);
    return o;
  }

  getRight(out) {
    const o = out || this._vB;
    o.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    return o;
  }

  getEyePosition(out) { return (out || this._vA).copy(this.pos); }

  /** Muzzle-ish origin: the eye, pushed forward and dropped a little. */
  getMuzzleOrigin(out, fwd, down) {
    const o = out || this._vA;
    this.getForward(this._vC);
    o.copy(this.pos).addScaledVector(this._vC, fwd ?? 0.25);
    o.y -= (down ?? 0.06);
    return o;
  }

  // ==========================================================================
  //  INPUT
  // ==========================================================================

  _bindInput() {
    if (this._bound || typeof window === 'undefined') return;
    this._bound = true;

    const el = this.g?.renderer?.renderer?.domElement
      || document.querySelector('#app canvas')
      || document.body;
    this._el = el;

    this._onKeyDown = (e) => {
      if (e.repeat) { this._maybePrevent(e); return; }
      const k = Controller.KEYMAP[e.code];
      if (!k) return;
      this.keys[k] = true;
      if (k === 'jump') { this._jumpEdge = true; this._jumpBuf = JUMP_BUFFER; }
      this._wake();
      this._maybePrevent(e);
    };

    this._onKeyUp = (e) => {
      const k = Controller.KEYMAP[e.code];
      if (!k) return;
      this.keys[k] = false;
      this._maybePrevent(e);
    };

    this._onMouseMove = (e) => {
      if (!this._locked) return;
      // RAW deltas. No smoothing, no acceleration, no filtering — ever.
      const dx = e.movementX || 0, dy = e.movementY || 0;
      // A single absurd delta is a browser hiccup on lock acquisition, not input.
      if (Math.abs(dx) > 900 || Math.abs(dy) > 900) return;
      this._mouseDX += dx;
      this._mouseDY += dy;
      if (dx || dy) this._wake();
    };

    this._onMouseDown = (e) => {
      if (e.button === 2) { this._rmb = true; this._setAds(true, true); this._wake(); }
      else if (e.button === 0) {
        this._wake();
        if (!this._locked) this._requestLock();
      }
    };

    this._onMouseUp = (e) => {
      if (e.button === 2) { this._rmb = false; this._setAds(false, true); }
    };

    this._onContext = (e) => { e.preventDefault(); };

    this._onLockChange = () => {
      const locked = document.pointerLockElement === this._el;
      this._locked = locked;
      if (locked) { this._mouseDX = 0; this._mouseDY = 0; this._wake(); }
    };

    this._onLockError = () => { this._locked = false; };

    this._onBlur = () => {
      for (const k in this.keys) this.keys[k] = false;
      this._rmb = false;
      this._setAds(false, true);
      this._mouseDX = 0; this._mouseDY = 0;
    };

    window.addEventListener('keydown', this._onKeyDown, false);
    window.addEventListener('keyup', this._onKeyUp, false);
    window.addEventListener('blur', this._onBlur, false);
    document.addEventListener('mousemove', this._onMouseMove, false);
    document.addEventListener('pointerlockchange', this._onLockChange, false);
    document.addEventListener('pointerlockerror', this._onLockError, false);
    el.addEventListener('mousedown', this._onMouseDown, false);
    window.addEventListener('mouseup', this._onMouseUp, false);
    el.addEventListener('contextmenu', this._onContext, false);
  }

  _maybePrevent(e) {
    // Only swallow keys when we actually own the input focus, so devtools and
    // the address bar keep working while the pointer is free.
    if (this._locked || e.target === document.body || e.target === this._el) {
      if (e.code === 'Space' || e.code === 'Tab' ||
          e.code === 'ArrowUp' || e.code === 'ArrowDown' ||
          e.code === 'ArrowLeft' || e.code === 'ArrowRight') e.preventDefault();
    }
  }

  _requestLock() {
    const el = this._el;
    if (!el || !el.requestPointerLock) return;
    try {
      const p = el.requestPointerLock({ unadjustedMovement: true });
      if (p && typeof p.then === 'function') {
        p.catch(() => {
          try {
            const q = el.requestPointerLock();
            if (q && typeof q.then === 'function') q.catch(() => {});
          } catch (e) { /* user gesture rules — silently ignore */ }
        });
      }
    } catch (e) { /* older signature or denied — ignore, never log */ }
  }

  /** Any genuine player input releases the teleport freeze. */
  _wake() {
    this._anyInput = true;
    if (this.frozen) {
      this.frozen = false;
      this._coyote = COYOTE_TIME;
    }
  }

  _bindBus() {
    const bus = this.g?.bus;
    if (!bus || typeof bus.on !== 'function') return;

    bus.on('ads', (d) => {
      if (this._adsSelf) return;
      const a = !!(d && d.active);
      if (a !== this.isAds) this._setAds(a, false);
    });

    // A single shared shake system: explosions and incoming damage feed it too.
    bus.on('explosion', (d) => {
      if (!d || !d.point) return;
      const dx = d.point.x - this.pos.x, dy = d.point.y - this.pos.y, dz = d.point.z - this.pos.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const power = d.power ?? 1;
      const radius = d.radius ?? 6;
      const falloff = 1 / (1 + (dist / Math.max(1.2, radius)) * (dist / Math.max(1.2, radius)));
      const amt = clamp(power * falloff * 2.6, 0, 3.4);
      if (amt < 0.02) return;
      this.addShake(amt, 0.42 + Math.min(0.55, amt * 0.28), 8.5 + 6 / (1 + amt));
    });

    bus.on('damage', (d) => {
      const amt = clamp((d?.amount ?? 10) / 34, 0.08, 1.6);
      this.addShake(amt * 0.9, 0.30, 19);
      // a short punch away from the hit direction
      this._landV -= amt * 0.9;
    });
  }

  _setAds(v, emit) {
    if (v === this.isAds) return;
    this.isAds = v;
    if (v) { this.isSprinting = false; this._sprintT = 0; }
    if (emit) {
      this._pAds.active = v;
      this._adsSelf = true;
      try { this.g?.bus?.emit?.('ads', this._pAds); }
      finally { this._adsSelf = false; }
    }
  }

  // ==========================================================================
  //  FRAME
  // ==========================================================================

  update(dt, t) {
    const d = (isFinite(dt) && dt > 0) ? Math.min(dt, 0.05) : 0;

    this._applyLook();

    if (d > 0) {
      this._syncAdsFromWeapon();
      this._fbTimer -= d;
      if (this._fbTimer <= 0) { this._fbTimer = 0.5; this._syncFallbackWorld(false); }

      if (!this.frozen) {
        this._updateStance(d);
        if (this.isMantling) this._updateMantle(d);
        else this._integrate(d);
        this._updateSurface(d);
      }
      this._updateShake(d);
      if (!this.frozen) this._updateCameraFeel(d);
      this._updateFov(d);
    }

    this._composeCamera(d, this.frozen);
  }

  resize(/* w, h */) { }

  // --------------------------------------------------------------------------
  //  LOOK — raw, unsmoothed, unaccelerated
  // --------------------------------------------------------------------------

  _applyLook() {
    const dx = this._mouseDX, dy = this._mouseDY;
    this._mouseDX = 0; this._mouseDY = 0;
    if (dx === 0 && dy === 0) return;

    const P = this.cfgP;
    let sens = P.mouseSensitivity ?? 0.0022;
    if (this.isAds) sens *= (P.adsSensMul ?? 0.62);
    // Slides and mantles slightly restrict turn authority — you are committed.
    if (this.isSliding) sens *= 0.78;
    else if (this.isMantling) sens *= 0.55;

    this.yaw -= dx * sens;
    this.pitch -= dy * sens;

    // keep yaw in a sane range without ever snapping mid-frame visibly
    if (this.yaw > Math.PI) this.yaw -= TAU;
    else if (this.yaw < -Math.PI) this.yaw += TAU;
    this.pitch = clamp(this.pitch, -PITCH_LIMIT, PITCH_LIMIT);
  }

  _syncAdsFromWeapon() {
    // While the pointer is locked, RMB is the truth. Otherwise (the screenshot
    // harness, or a scripted sequence) mirror whatever the weapon says.
    if (this._locked) return;
    const w = this.g?.weapon;
    if (!w) return;
    const ext = (typeof w.isAds === 'boolean') ? w.isAds
      : (typeof w.ads === 'boolean') ? w.ads
        : (typeof w.aiming === 'boolean') ? w.aiming : null;
    if (ext !== null && ext !== this.isAds) this._setAds(ext, false);
  }

  // --------------------------------------------------------------------------
  //  STANCE — sprint spin-up, crouch blend, slide state machine
  // --------------------------------------------------------------------------

  _updateStance(dt) {
    const K = this.keys;
    const P = this.cfgP;

    this._slideCool = Math.max(0, this._slideCool - dt);
    this._mtCool = Math.max(0, this._mtCool - dt);
    this._jumpBuf = Math.max(0, this._jumpBuf - dt);

    const wantFwd = K.f && !K.b;

    // ---- sprint --------------------------------------------------------
    const canSprint = K.sprint && wantFwd && !this.isAds && !this.isSliding &&
      !this._uncrouchBlocked && (this.grounded || this._sprintT > 0.05);
    this.isSprinting = canSprint && this._sprintT > 0.12;
    const spinRate = canSprint ? 1 / SPRINT_SPIN_UP : -1 / SPRINT_SPIN_DN;
    this._sprintT = clamp(this._sprintT + spinRate * dt, 0, 1);

    // ---- slide ---------------------------------------------------------
    if (this.isSliding) {
      this._slideT += dt;
      const tooSlow = this.speed < SLIDE_MIN_SPEED;
      const tooLong = this._slideT > SLIDE_MAX_TIME;
      const airborne = !this.grounded && this._airTime > 0.28;
      if (tooSlow || tooLong || airborne || !K.crouch) this._endSlide();
    } else if (K.crouch && K.sprint && this.grounded && !this.isAds &&
      this._slideCool <= 0 && this.speed > SLIDE_ENTER_SPEED && wantFwd) {
      this._beginSlide();
    }

    // ---- crouch --------------------------------------------------------
    const wantCrouch = this.isSliding || (K.crouch && !this.isMantling);
    if (!wantCrouch && this.isCrouching) {
      // Only stand up if there is actually room for the tall capsule.
      this._uncrouchBlocked = !this._hasHeadroom(this.standHeight + 0.04);
      if (!this._uncrouchBlocked) this.isCrouching = false;
    } else if (wantCrouch) {
      this.isCrouching = true;
      this._uncrouchBlocked = false;
    }

    // ---- capsule + eye height blend ------------------------------------
    let targetH, targetEye;
    if (this.isSliding) { targetH = this.slideHeight; targetEye = this.slideEye; }
    else if (this.isCrouching) { targetH = this.crouchHeight; targetEye = this.crouchEye; }
    else { targetH = this.standHeight; targetEye = this.standEye; }

    // Going down is quicker than coming up — that asymmetry is the whole feel.
    const goingDown = targetEye < this._eye;
    const rate = this.isSliding ? 19 : (goingDown ? 15.5 : 10.5);
    const k = damp(rate, dt);
    this._eye += (targetEye - this._eye) * k;
    this._capH += (targetH - this._capH) * k;
    if (Math.abs(targetEye - this._eye) < 0.0008) this._eye = targetEye;
    if (Math.abs(targetH - this._capH) < 0.0008) this._capH = targetH;
    this.capsule.height = this._capH;
    this.capsule.radius = this.radius;

    // ---- ads blend (drives bob damping and sway tightening) ------------
    const adsT = this.g?.config?.weapon?.adsTime ?? 0.17;
    this._adsBlend += ((this.isAds ? 1 : 0) - this._adsBlend) * damp(1 / Math.max(0.04, adsT * 0.62), dt);
  }

  _beginSlide() {
    this.isSliding = true;
    this.isCrouching = true;
    this._slideT = 0;
    const v = this.velocity;
    const sp = Math.hypot(v.x, v.z);
    if (sp > 0.001) {
      const cap = (this.cfgP.sprintSpeed ?? 6.2) * 1.34;
      const boosted = Math.min(sp * SLIDE_BOOST, cap);
      const s = boosted / sp;
      v.x *= s; v.z *= s;
    }
    this._landV -= 0.55;                    // the drop into the slide
    this.addShake(0.30, 0.30, 11);
    this._emitFootstep(this.speed * 1.25);
  }

  _endSlide() {
    if (!this.isSliding) return;
    this.isSliding = false;
    this._slideCool = SLIDE_COOLDOWN;
    this._slideT = 0;
    if (!this.keys.crouch) {
      this._uncrouchBlocked = !this._hasHeadroom(this.standHeight + 0.04);
      if (!this._uncrouchBlocked) this.isCrouching = false;
    }
  }

  // --------------------------------------------------------------------------
  //  MOVEMENT — Quake-lineage acceleration / friction, resolved by capsule sweeps
  // --------------------------------------------------------------------------

  _integrate(dt) {
    const P = this.cfgP;
    const K = this.keys;
    const v = this.velocity;

    // ---- wish direction in the camera's horizontal basis ----------------
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    // forward (yaw only): the camera's -Z projected onto the ground plane
    const fx = -sy, fz = -cy;
    const rx = cy, rz = -sy;

    let ix = (K.r ? 1 : 0) - (K.l ? 1 : 0);
    let iz = (K.f ? 1 : 0) - (K.b ? 1 : 0);

    const w = this._vWish;
    w.set(fx * iz + rx * ix, 0, fz * iz + rz * ix);
    const wlen = Math.hypot(w.x, w.z);
    if (wlen > 1e-5) { w.x /= wlen; w.z /= wlen; }

    // ---- target speed ---------------------------------------------------
    const walk = P.walkSpeed ?? 3.4;
    const sprint = P.sprintSpeed ?? 6.2;
    const crouchSp = P.crouchSpeed ?? 1.9;
    let wishSpeed;
    if (this.isCrouching && !this.isSliding) wishSpeed = crouchSp;
    else wishSpeed = walk + (sprint - walk) * smootherstep(this._sprintT);
    if (this.isAds) wishSpeed *= (P.adsSpeedMul ?? 0.42);
    if (iz < 0) wishSpeed *= 0.82;                 // backpedal is slower
    else if (iz === 0 && ix !== 0) wishSpeed *= 0.92;
    if (wlen < 1e-5) wishSpeed = 0;

    // Freecam debug flight — no collision, no gravity.
    if (this.g?.config?.debug?.freecam) {
      this.getForward(this._vA);
      const spd = (K.sprint ? 18 : 7) * dt;
      this.feet.addScaledVector(this._vA, iz * spd);
      this.getRight(this._vB);
      this.feet.addScaledVector(this._vB, ix * spd);
      this.feet.y += ((K.jump ? 1 : 0) - (K.crouch ? 1 : 0)) * spd;
      v.set(0, 0, 0);
      this.speed = 0;
      this.grounded = false;
      return;
    }

    // ---- jump (buffered + coyote) ---------------------------------------
    const gravity = P.gravity ?? -18.5;
    if (this._jumpBuf > 0 && (this.grounded || this._coyote > 0) && !this.isMantling) {
      // Sliding into a jump keeps the slide momentum — the classic slide-hop.
      if (this.isSliding) this._endSlide();
      v.y = P.jumpVel ?? 4.6;
      this.grounded = false;
      this._coyote = 0;
      this._jumpBuf = 0;
      this._airTime = 0;
      this._landV += 0.28;                 // tiny rise as you leave the floor
      this._emitFootstep(this.speed);
    }

    // ---- mantle test -----------------------------------------------------
    // Triggered by a jump press into a ledge, or by holding forward into one.
    if (!this.isMantling && this._mtCool <= 0 && wlen > 0.4) {
      const jumpAsk = this._jumpEdge && !this.grounded && this._airTime < 0.32;
      const pushAsk = this._forwardHold > 0.17 && this.grounded;
      if (jumpAsk || pushAsk || (this._jumpEdge && this.grounded)) {
        if (this._tryMantle(w)) { this._jumpEdge = false; return; }
      }
    }
    this._jumpEdge = false;

    // ---- accelerate ------------------------------------------------------
    if (this.grounded) {
      if (this.isSliding) {
        this._slideAccel(dt, w, wlen);
      } else {
        this._friction(dt, P.friction ?? 11);
        this._accelerate(dt, w, wishSpeed, P.accel ?? 48);
      }
    } else {
      // Air control: a hard cap on the wish speed is what makes air-strafing
      // feel like control rather than a second ground.
      this._accelerate(dt, w, Math.min(wishSpeed, AIR_WISH_CAP), P.airAccel ?? 9);
      // and a whisper of drag so bunny-hopping cannot run away with the speed
      const drag = 1 - 0.35 * dt;
      v.x *= drag; v.z *= drag;
    }

    // ---- gravity ---------------------------------------------------------
    if (!this.grounded) {
      v.y += gravity * dt;
      if (v.y < -60) v.y = -60;
      this._airTime += dt;
    } else {
      this._airTime = 0;
      if (v.y < 0) v.y = 0;
    }

    // ---- resolve ---------------------------------------------------------
    this._wasGrounded = this.grounded;
    this._fallVel = v.y;

    this._moveHorizontal(dt);
    this._moveVertical(dt);
    this._probeGround(false);

    // Safety net: a collaborator's collision backend that silently reports
    // "never blocked" would drop the player out of the world forever. Catch it
    // once, disown the backend, and put the player back on the ground.
    if (!isFinite(this.feet.y) || this.feet.y < -60) {
      this._physBad = true;
      this.feet.set(isFinite(this.feet.x) ? this.feet.x : 0, FB_GROUND_Y,
        isFinite(this.feet.z) ? this.feet.z : 0);
      v.set(0, 0, 0);
      this.grounded = true;
      this._fallVel = 0;
    }

    // ---- landing ---------------------------------------------------------
    if (this.grounded && !this._wasGrounded) this._onLand();
    if (!this.grounded && this._wasGrounded) this._coyote = COYOTE_TIME;
    else if (!this.grounded) this._coyote = Math.max(0, this._coyote - dt);
    else this._coyote = COYOTE_TIME;

    this.speed = Math.hypot(v.x, v.z);

    // "pushing into a wall while trying to move forward" — the auto-vault cue
    if (this._blockedThisFrame && wlen > 0.4 && this.keys.f) this._forwardHold += dt;
    else this._forwardHold = 0;

    this.pos.set(this.feet.x, this.feet.y + this._eye, this.feet.z);
  }

  _friction(dt, friction) {
    const v = this.velocity;
    const sp = Math.hypot(v.x, v.z);
    if (sp < 1e-4) { v.x = 0; v.z = 0; return; }
    const control = sp < STOP_SPEED ? STOP_SPEED : sp;
    const drop = control * friction * dt;
    let ns = sp - drop;
    if (ns < 0) ns = 0;
    ns /= sp;
    v.x *= ns; v.z *= ns;
  }

  _accelerate(dt, wish, wishSpeed, accel) {
    if (wishSpeed <= 0) return;
    const v = this.velocity;
    const cur = v.x * wish.x + v.z * wish.z;
    const add = wishSpeed - cur;
    if (add <= 0) return;
    let as = accel * dt * wishSpeed;
    if (as > add) as = add;
    v.x += wish.x * as;
    v.z += wish.z * as;
  }

  // A slide keeps its momentum and can only be steered, never re-accelerated.
  _slideAccel(dt, wish, wlen) {
    const v = this.velocity;
    const sp = Math.hypot(v.x, v.z);
    if (sp < 1e-4) return;
    let dirx = v.x / sp, dirz = v.z / sp;

    if (wlen > 0.4) {
      // rotate the velocity toward the wish direction, capped in degrees/second
      const cross = dirx * wish.z - dirz * wish.x;
      const dot = clamp(dirx * wish.x + dirz * wish.z, -1, 1);
      const ang = Math.atan2(cross, dot);
      const maxA = SLIDE_STEER * dt;
      const use = clamp(ang, -maxA, maxA);
      const c = Math.cos(use), s = Math.sin(use);
      const nx = dirx * c - dirz * s;
      const nz = dirx * s + dirz * c;
      dirx = nx; dirz = nz;
    }

    // low friction that ramps up as the slide ages
    const fr = SLIDE_FRICTION * (0.62 + 0.85 * (this._slideT / SLIDE_MAX_TIME));
    const ns = Math.max(0, sp - fr * dt);
    v.x = dirx * ns;
    v.z = dirz * ns;
  }

  _onLand() {
    const impact = clamp(-this._fallVel / 11.5, 0, 1);
    if (impact > 0.02) {
      // Drive the spring rather than setting a pose — the recovery is a curve.
      this._landV -= impact * 3.4;
      if (impact > 0.30) this.addShake(impact * 0.85, 0.26 + impact * 0.18, 17);
      this._emitFootstep(Math.max(this.speed, impact * 6));
    }
    // Reset the bob to a foot-plant so the first step after landing lands right.
    this._bobPhase = FOOT_PHASE;
    this._footIndex = Math.floor((this._bobPhase - FOOT_PHASE) / Math.PI);
    this._airTime = 0;
  }

  // --------------------------------------------------------------------------
  //  COLLISION RESOLUTION
  // --------------------------------------------------------------------------

  _moveHorizontal(dt) {
    const v = this.velocity;
    const dx = v.x * dt, dz = v.z * dt;
    this._blockedThisFrame = false;
    if (Math.abs(dx) < 1e-7 && Math.abs(dz) < 1e-7) return;

    const f = this.feet;
    const r = this.radius, h = this._capH;

    // 1. combined move
    this._sweep(f.x, f.y, f.z, f.x + dx, f.y, f.z + dz, r, h);

    // A full collision solver already stepped, slid and depenetrated for us —
    // take its answer verbatim and only harvest what the CAMERA needs from it.
    if (this._swRich) {
      // Only a genuine STEP gets camera smoothing. A slope also raises us every
      // frame, and smoothing that would leave the camera permanently sunk into
      // the floor on every ramp — so the threshold scales with how far we moved.
      let rise = this._swStep;
      if (rise <= 0.02) {
        const dy = this._swEnd.y - f.y;
        const slopeMax = Math.max(0.08, Math.hypot(dx, dz) * 0.72);
        rise = dy > slopeMax ? dy : 0;
      }
      if (rise > 0.02 && rise <= STEP_HEIGHT + 0.05) this._stepSmooth -= rise;
      f.copy(this._swEnd);
      this._blockedThisFrame = this._swWall;
      if (this._swWall) this._bleedIntoWall(dx, dz);
      return;
    }

    if (!this._swHit) { f.copy(this._swEnd); return; }

    this._blockedThisFrame = true;
    const flatX = this._swEnd.x, flatY = this._swEnd.y, flatZ = this._swEnd.z;
    const flatDist2 = (flatX - f.x) * (flatX - f.x) + (flatZ - f.z) * (flatZ - f.z);

    // 2. try to step over it (stairs, kerbs, rubble) before sliding along it
    if (this.grounded && !this.isSliding) {
      this._sweep(f.x, f.y, f.z, f.x, f.y + STEP_HEIGHT, f.z, r, h);
      const upY = this._swEnd.y;
      if (upY - f.y > 0.06) {
        this._sweep(f.x, upY, f.z, f.x + dx, upY, f.z + dz, r, h);
        const sx = this._swEnd.x, sz = this._swEnd.z;
        const stepDist2 = (sx - f.x) * (sx - f.x) + (sz - f.z) * (sz - f.z);
        if (stepDist2 > flatDist2 + 0.0009) {
          // drop back down onto whatever we just climbed onto
          this._sweep(sx, upY, sz, sx, f.y - 0.02, sz, r, h);
          const dy = this._swEnd.y - f.y;
          if (this._swHit && dy > -0.02 && dy <= STEP_HEIGHT + 0.02) {
            // Camera lags the step by dy and catches up on a curve — this is
            // the single fix that stops stairs from strobing the view.
            if (dy > 0.02) this._stepSmooth -= dy;
            f.copy(this._swEnd);
            return;
          }
        }
      }
    }

    // 3. Slide along the obstruction: re-run the move one axis at a time from
    //    the ORIGINAL position. Against an axis-aligned wall this recovers the
    //    full tangential motion without ever needing a contact normal.
    const ox = f.x, oy = f.y, oz = f.z;

    this._sweep(ox, oy, oz, ox + dx, oy, oz, r, h);
    const ax = this._swEnd.x, ay = this._swEnd.y, az = this._swEnd.z;

    this._sweep(ax, ay, az, ax, ay, az + dz, r, h);
    const bx = this._swEnd.x, by = this._swEnd.y, bz = this._swEnd.z;

    const gotX = ax - ox;
    const gotZ = bz - az;
    const axis2 = (bx - ox) * (bx - ox) + (bz - oz) * (bz - oz);

    if (axis2 >= flatDist2) f.set(bx, by, bz);
    else f.set(flatX, flatY, flatZ);

    // Bleed off the velocity component that got fully stopped, so we do not keep
    // accelerating into the wall (which reads as a stuck, buzzing player) while
    // leaving the tangential component completely untouched.
    if (Math.abs(dx) > 1e-6 && Math.abs(gotX) < Math.abs(dx) * 0.2) v.x *= 0.03;
    if (Math.abs(dz) > 1e-6 && Math.abs(gotZ) < Math.abs(dz) * 0.2) v.z *= 0.03;
  }

  /**
   * Remove the velocity component pushing into a wall so we stop accelerating
   * into it, while leaving the tangential component completely intact — that is
   * what makes running along a wall feel smooth instead of sticky.
   */
  _bleedIntoWall(dx, dz) {
    const v = this.velocity;
    const nx = this._swNX, nz = this._swNZ;
    const nl = Math.sqrt(nx * nx + nz * nz);
    if (nl > 1e-4) {
      const ux = nx / nl, uz = nz / nl;
      const into = v.x * ux + v.z * uz;
      if (into < 0) { v.x -= ux * into; v.z -= uz * into; }
      return;
    }
    // No usable normal — fall back to killing whichever axis made no progress.
    const gx = this._swEnd.x - (this.feet.x);
    const gz = this._swEnd.z - (this.feet.z);
    if (Math.abs(dx) > 1e-6 && Math.abs(gx) < Math.abs(dx) * 0.2) v.x *= 0.03;
    if (Math.abs(dz) > 1e-6 && Math.abs(gz) < Math.abs(dz) * 0.2) v.z *= 0.03;
  }

  _moveVertical(dt) {
    const v = this.velocity;
    const dy = v.y * dt;
    if (Math.abs(dy) < 1e-7) return;
    const f = this.feet;
    this._sweep(f.x, f.y, f.z, f.x, f.y + dy, f.z, this.radius, this._capH);

    // Only take Y from a vertical sweep. A solver's lateral depenetration is
    // legitimate during a real move, but applying it here every frame while the
    // player stands still would creep them across the floor.
    const endY = this._swEnd.y;

    if (dy < 0) {
      // Landing: trust the solver's ground flag when it has one, otherwise the
      // measured shortfall.
      const landed = this._swRich ? (this._swGround || this._swHit) : this._swHit;
      f.y = endY;
      if (landed) { this.grounded = true; v.y = 0; }
    } else {
      // Rising: `grounded` is meaningless here (it is true for the whole first
      // few centimetres of a jump), so only a real ceiling contact stops us.
      const bumped = this._swRich ? this._swCeil : this._swHit;
      f.y = endY;
      if (bumped) v.y = 0;
    }
  }

  /**
   * Short downward probe that keeps the capsule glued to the floor across steps
   * and slopes, and reads the surface we are standing on.
   */
  _probeGround(force) {
    const v = this.velocity;
    if (!force && v.y > 0.6) { this.grounded = false; return; }

    const f = this.feet;
    const reach = force ? 6.0 : GROUND_SNAP;
    this._sweep(f.x, f.y + 0.02, f.z, f.x, f.y + 0.02 - reach, f.z, this.radius * 0.985, this._capH);
    const found = this._swRich ? (this._swGround || this._swHit) : this._swHit;
    if (found) {
      const gy = this._swEnd.y;
      const drop = (f.y + 0.02) - gy;
      if (force || drop <= GROUND_SNAP + 0.03) {
        if (!this.grounded && drop > 0.03 && !force) {
          // stepping down a kerb: let the camera trail the drop
          this._stepSmooth += Math.min(drop, STEP_HEIGHT) * 0.55;
        }
        f.y = gy;
        this.grounded = true;
        if (v.y < 0) v.y = 0;
        if (this._swSurface) this.surface = this._swSurface;
        return;
      }
    }
    if (!force) this.grounded = false;
  }

  /** Is there room for a capsule of `h` at the current feet position? */
  _hasHeadroom(h) {
    const f = this.feet;
    // sweep a tall capsule up by a hair; if it cannot rise, the ceiling is low
    this._sweep(f.x, f.y, f.z, f.x, f.y + 0.06, f.z, this.radius * 0.94, h);
    if (this._swRich) return !this._swCeil && this._swAdv > 0.75;
    return !this._swHit || (this._swEnd.y - f.y) > 0.055;
  }

  // --------------------------------------------------------------------------
  //  MANTLE / VAULT
  // --------------------------------------------------------------------------

  _tryMantle(wish) {
    const f = this.feet;
    const r = MANTLE_PROBE_R, ph = MANTLE_PROBE_H;
    const wx = wish.x, wz = wish.z;
    if (Math.abs(wx) + Math.abs(wz) < 0.2) return false;

    const reach = MANTLE_REACH;

    // A) is anything actually in front of us at shin height?
    this._sweep(f.x, f.y + 0.22, f.z, f.x + wx * reach, f.y + 0.22, f.z + wz * reach, r, ph);
    const blockedDist = Math.hypot(this._swEnd.x - f.x, this._swEnd.z - f.z);
    if (blockedDist > reach * 0.72) return false;

    // B) walk up in 12 cm increments looking for the first free height
    let found = -1;
    for (let h = MANTLE_MIN_RISE; h <= MANTLE_MAX_RISE + 0.001; h += 0.12) {
      this._sweep(f.x, f.y + h + 0.04, f.z, f.x + wx * reach, f.y + h + 0.04, f.z + wz * reach, r, ph);
      const dist = Math.hypot(this._swEnd.x - f.x, this._swEnd.z - f.z);
      if (dist > reach * 0.90) { found = h; break; }
    }
    if (found < 0) return false;

    // C) drop onto the ledge top to find its exact height
    const lx = f.x + wx * reach * 0.94;
    const lz = f.z + wz * reach * 0.94;
    this._sweep(lx, f.y + found + 0.20, lz, lx, f.y - 0.05, lz, r, ph);
    if (!this._swHit) return false;
    const topY = this._swEnd.y;
    const rise = topY - f.y;
    if (rise < MANTLE_MIN_RISE || rise > MANTLE_MAX_RISE) return false;

    // D) does the player actually fit up there?
    const need = this.isCrouching ? this.crouchHeight : this.standHeight;
    this._sweep(lx, topY + 0.25, lz, lx, topY + 0.02, lz, this.radius * 0.94, need);
    if (this._swHit && this._swEnd.y > topY + 0.10) return false;

    // ---- commit -----------------------------------------------------------
    this.isMantling = true;
    this._mtT = 0;
    this._mtDur = 0.34 + clamp(rise / MANTLE_MAX_RISE, 0, 1) * 0.24;
    this._mtFrom.copy(f);
    this._mtTo.set(lx + wx * 0.16, topY + 0.01, lz + wz * 0.16);
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    this.isSliding = false;
    this._sprintT *= 0.4;

    // the grab: a short, sharp shake and a downward camera loading
    this.addShake(0.34, 0.18, 21);
    this._landV -= 0.75;
    return true;
  }

  _updateMantle(dt) {
    this._mtT += dt;
    const u = clamp(this._mtT / this._mtDur, 0, 1);

    // Vertical leads, horizontal follows — you pull up, THEN step over. A single
    // blended lerp reads as being sucked through the wall; this reads as a body.
    const uy = smootherstep(clamp(u / 0.66, 0, 1));
    const ux = smootherstep(clamp((u - 0.26) / 0.74, 0, 1));

    const a = this._mtFrom, b = this._mtTo;
    this.feet.set(
      a.x + (b.x - a.x) * ux,
      a.y + (b.y - a.y) * uy,
      a.z + (b.z - a.z) * ux,
    );
    this.pos.set(this.feet.x, this.feet.y + this._eye, this.feet.z);

    // the view rises as the body is pulled up, then drops onto the ledge below
    this._landV += Math.sin(u * Math.PI) * dt * 4.5;

    if (u >= 1) {
      this.isMantling = false;
      this._mtCool = 0.22;
      this.grounded = true;
      this._probeGround(false);
      this.velocity.set(0, 0, 0);
      this._landV -= 0.9;
      this._emitFootstep(2.4);
      this.addShake(0.22, 0.20, 15);
      this._forwardHold = 0;
    }
  }

  // --------------------------------------------------------------------------
  //  CAMERA FEEL
  // --------------------------------------------------------------------------

  _updateCameraFeel(dt) {
    const P = this.cfgP;
    const walk = P.walkSpeed ?? 3.4;
    const v = this.velocity;
    const adsK = 1 - this._adsBlend * (1 - BOB_ADS);

    // ---- view bob on a figure-8 ----------------------------------------
    const moving = this.grounded && !this.isSliding && !this.isMantling;
    const sp = moving ? this.speed : 0;
    if (moving && sp > 0.12) {
      // Phase advances with DISTANCE, not time — so a slow walk and a sprint
      // both plant a foot every stride, and speed changes never skip a beat.
      const strideK = this.isCrouching ? 1.32 : 1.0;
      this._bobPhase += sp * dt * BOB_PHASE_PER_M * strideK;
      if (this._bobPhase > 1e6) this._bobPhase -= 1e6;
    }
    const ampTarget = moving ? clamp(sp / walk, 0, 1.85) : 0;
    this._bobAmp += (ampTarget - this._bobAmp) * damp(moving ? 8.5 : 6.0, dt);

    const bp = this._bobPhase;
    const amp = this._bobAmp * adsK;
    const s1 = Math.sin(bp), c1 = Math.cos(bp);
    const s2 = Math.sin(bp * 2), c2 = Math.cos(bp * 2);

    this._bobX = s1 * amp * BOB_AMP_X;
    this._bobY = s2 * amp * BOB_AMP_Y;
    this._bobRoll = s1 * amp * BOB_ROLL;
    this._bobPitch = c2 * amp * BOB_PITCH;

    // foot plant: sin(2p) bottoms out twice per cycle
    if (this._bobAmp > 0.14 && moving) {
      const idx = Math.floor((bp - FOOT_PHASE) / Math.PI);
      if (idx !== this._footIndex) {
        this._footIndex = idx;
        this._emitFootstep(sp);
      }
    } else {
      this._footIndex = Math.floor((bp - FOOT_PHASE) / Math.PI);
    }

    // ---- landing spring -------------------------------------------------
    // Critically-ish damped second order system. Never a scripted keyframe.
    const c = 2 * Math.sqrt(LAND_K) * LAND_ZETA;
    this._landV += (-LAND_K * this._landY - c * this._landV) * dt;
    this._landY += this._landV * dt;
    if (this._landY < -LAND_MAX) { this._landY = -LAND_MAX; if (this._landV < 0) this._landV = 0; }
    else if (this._landY > LAND_MAX * 0.45) { this._landY = LAND_MAX * 0.45; if (this._landV > 0) this._landV = 0; }
    if (Math.abs(this._landY) < 1e-5 && Math.abs(this._landV) < 1e-4) { this._landY = 0; this._landV = 0; }

    // ---- step smoothing -------------------------------------------------
    this._stepSmooth -= this._stepSmooth * damp(11.5, dt);
    if (Math.abs(this._stepSmooth) < 1e-4) this._stepSmooth = 0;
    this._stepSmooth = clamp(this._stepSmooth, -STEP_HEIGHT, STEP_HEIGHT);

    // ---- strafe roll ----------------------------------------------------
    // Driven by the actual lateral velocity, not the key state, so it eases in
    // and out with the movement rather than snapping with the keypress.
    const rx = Math.cos(this.yaw), rz = -Math.sin(this.yaw);
    const lateral = v.x * rx + v.z * rz;
    let rollTarget = -clamp(lateral / walk, -1.35, 1.35) * STRAFE_ROLL;
    if (this.isSliding) rollTarget += -clamp(lateral / walk, -1, 1) * 2.6 * DEG;
    rollTarget *= (1 - this._adsBlend * 0.55);
    this._roll += (rollTarget - this._roll) * damp(7.5, dt);

    // ---- lean -----------------------------------------------------------
    let leanTarget = (this.keys.leanR ? 1 : 0) - (this.keys.leanL ? 1 : 0);
    if (this.isSliding || this.isMantling || this.isSprinting) leanTarget = 0;
    // Do not lean into geometry. Probed at 12 Hz — this is not a per-frame cost.
    this._leanTimer -= dt;
    if (this._leanTimer <= 0) {
      this._leanTimer = 0.083;
      this._leanLimitR = this._leanClearance(1);
      this._leanLimitL = this._leanClearance(-1);
    }
    if (leanTarget > 0) leanTarget = Math.min(leanTarget, this._leanLimitR);
    else if (leanTarget < 0) leanTarget = Math.max(leanTarget, -this._leanLimitL);
    this._lean += (leanTarget - this._lean) * damp(10.5, dt);
    if (Math.abs(this._lean) < 1e-4) this._lean = 0;

    // ---- idle breathing sway --------------------------------------------
    // Slower and much tighter when aiming: a held breath, not a resting one.
    const breathRate = this.isAds ? 1.42 : 1.02;
    this._breath += dt * breathRate;
    const idleK = 1 - clamp(this.speed / walk, 0, 1) * 0.72;
    const swayA = (SWAY_HIP + (SWAY_ADS - SWAY_HIP) * this._adsBlend) * idleK;
    const b = this._breath;
    this._swayPitch = (Math.sin(b * 0.93) * 1.0 + Math.sin(b * 2.31 + 1.1) * 0.34) * swayA;
    this._swayYaw = (Math.sin(b * 0.71 + 2.1) * 1.15 + Math.sin(b * 1.77 + 0.4) * 0.28) * swayA;
    this._swayPos = Math.sin(b * 0.61 + 0.9) * swayA * 0.55;
  }

  /** 0..1 fraction of the lean offset that fits before hitting geometry. */
  _leanClearance(sign) {
    const rx = Math.cos(this.yaw) * sign, rz = -Math.sin(this.yaw) * sign;
    const f = this.feet;
    const ey = f.y + this._eye - 0.14;
    const d = LEAN_OFFSET + this.radius * 0.4;
    this._sweep(f.x, ey, f.z, f.x + rx * d, ey, f.z + rz * d, 0.11, 0.20);
    // Purely distance-based so it behaves identically on either backend.
    const got = Math.hypot(this._swEnd.x - f.x, this._swEnd.z - f.z);
    if (got >= d - 0.004) return 1;
    return clamp((got - this.radius * 0.4) / LEAN_OFFSET, 0, 1);
  }

  // --------------------------------------------------------------------------
  //  SHAKE — summed decaying sine octaves, never random jitter
  // --------------------------------------------------------------------------

  _updateShake(dt) {
    let ap = 0, ay = 0, ar = 0;
    let any = false;

    for (let i = 0; i < SHAKE_SLOTS; i++) {
      if (!this._shOn[i]) continue;
      let t = this._shT[i] + dt;
      const dur = this._shDur[i];
      if (t >= dur) { this._shOn[i] = 0; continue; }
      this._shT[i] = t;
      any = true;

      const u = t / dur;
      const env = (1 - u) * (1 - u);           // quadratic decay to a clean zero
      const amp = this._shAmp[i] * env;
      const w = TAU * this._shFrq[i] * t;
      const p0 = this._shPh[i * 3], p1 = this._shPh[i * 3 + 1], p2 = this._shPh[i * 3 + 2];

      let sp = 0, sy = 0, sr = 0;
      for (let k = 1; k <= SHAKE_OCTAVES; k++) {
        const inv = 1 / k;
        // The three axes run on deliberately incommensurate frequency ratios so
        // the motion never resolves into a visible circle or a straight line.
        sp += Math.sin(w * k + p0 * k) * inv;
        sy += Math.sin(w * k * 1.317 + p1 * k) * inv;
        sr += Math.sin(w * k * 0.773 + p2 * k) * inv;
      }
      ap += sp * amp;
      ay += sy * amp;
      ar += sr * amp;
    }

    if (!any) {
      this._shakePitch = this._shakeYaw = this._shakeRoll = 0;
      this._shakePX = this._shakePY = 0;
      return;
    }

    this._shakePitch = ap * SHAKE_PITCH_K;
    this._shakeYaw = ay * SHAKE_YAW_K;
    this._shakeRoll = ar * SHAKE_ROLL_K;
    this._shakePX = ay * SHAKE_POS_K;
    this._shakePY = ap * SHAKE_POS_K;
  }

  // --------------------------------------------------------------------------
  //  FOV — sprint punch, only while nobody else owns camera.fov
  // --------------------------------------------------------------------------

  _updateFov(dt) {
    if (this._fovContested) return;
    const cam = this.g?.camera;
    if (!cam || !cam.isPerspectiveCamera) return;

    if (this._fovWritten >= 0 && Math.abs(cam.fov - this._fovWritten) > 1e-3) {
      // Weapon (or anything else) is animating the FOV. Back off permanently —
      // two modules fighting over one value is a visible stutter.
      this._fovContested = true;
      return;
    }
    if (this.isAds || this._adsBlend > 0.01) {
      // ADS FOV belongs to the weapon; if it never claims it, hip FOV is correct.
      if (this._fovWritten >= 0) {
        const base = this.g.config?.weapon?.fovHip ?? 68;
        if (Math.abs(cam.fov - base) > 1e-3) {
          cam.fov += (base - cam.fov) * damp(9, dt);
          cam.updateProjectionMatrix();
          this._fovWritten = cam.fov;
        }
      }
      return;
    }

    const base = this.g.config?.weapon?.fovHip ?? 68;
    const punch = smootherstep(this._sprintT) * 0.062 + (this.isSliding ? 0.055 : 0);
    const target = base * (1 + punch);
    if (Math.abs(cam.fov - target) < 1e-3) { this._fovWritten = cam.fov; return; }
    cam.fov += (target - cam.fov) * damp(6.5, dt);
    cam.updateProjectionMatrix();
    this._fovWritten = cam.fov;
  }

  // --------------------------------------------------------------------------
  //  CAMERA COMPOSITION
  // --------------------------------------------------------------------------

  _composeCamera(dt, plain) {
    const cam = this.g?.camera;
    if (!cam) return;

    if (plain) {
      // Frozen (post-teleport) — exactly the requested pose, dead still, so the
      // screenshot critic gets the framing it asked for.
      cam.position.copy(this.pos);
      cam.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
      cam.updateMatrixWorld();
      return;
    }

    const f = this.feet;
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    const rx = cy, rz = -sy;                     // right, horizontal
    const fx = -sy, fz = -cy;                    // forward, horizontal

    let px = f.x, py = f.y + this._eye, pz = f.z;

    // vertical: step smoothing + landing spring
    py += this._stepSmooth + this._landY;

    // lateral: bob + lean + sway, all in the camera's horizontal basis
    const lat = (this._bobX || 0) + this._lean * LEAN_OFFSET + (this._swayPos || 0);
    px += rx * lat;
    pz += rz * lat;
    py += (this._bobY || 0) - Math.abs(this._lean) * LEAN_DROP;

    // shake displacement (small, and mostly lateral — vertical shake reads cheap)
    px += rx * this._shakePX;
    pz += rz * this._shakePX;
    py += this._shakePY * 0.55;

    // slide pushes the camera slightly forward, like the body leading the head
    if (this.isSliding) {
      const k = 0.10 * (1 - this._slideT / SLIDE_MAX_TIME);
      px += fx * k; pz += fz * k;
    }

    this.pos.set(px, py, pz);

    const pitch = clamp(
      this.pitch + (this._bobPitch || 0) + (this._swayPitch || 0) + this._shakePitch
      + this._landY * 0.42,
      -PITCH_LIMIT - 0.05, PITCH_LIMIT + 0.05,
    );
    const yaw = this.yaw + (this._swayYaw || 0) + this._shakeYaw;
    const roll = this._roll + (this._bobRoll || 0) + this._shakeRoll
      - this._lean * LEAN_ROLL;

    cam.position.set(px, py, pz);
    cam.rotation.set(pitch, yaw, roll, 'YXZ');
    cam.updateMatrixWorld();

    // published basis for collaborators (weapon tracers, AI awareness, audio)
    this.getForward(this._vFwd);
    this._vRight.set(rx, 0, rz);
  }

  // --------------------------------------------------------------------------
  //  SURFACE + EVENTS
  // --------------------------------------------------------------------------

  _updateSurface(dt) {
    this._surfTimer -= dt;
    if (this._surfTimer > 0) return;
    this._surfTimer = 0.14;

    const p = this.g?.physics;
    const f = this.feet;
    let s = null;
    if (p) {
      try {
        if (typeof p.surfaceAt === 'function') s = p.surfaceAt(f.x, f.y - 0.05, f.z);
        else if (typeof p.getSurface === 'function') s = p.getSurface(f.x, f.y - 0.05, f.z);
        else if (typeof p.surfaceUnder === 'function') s = p.surfaceUnder(f.x, f.y - 0.05, f.z);
      } catch (e) { /* a collaborator's probe threw — never fatal here */ }
    }
    if (typeof s === 'string' && s.length) this.surface = s;
    else if (this._swSurface) this.surface = this._swSurface;
  }

  _emitFootstep(speed) {
    const bus = this.g?.bus;
    if (!bus) return;
    this._pFoot.surface = this.surface || 'concrete';
    this._pFoot.speed = speed || 0;
    bus.emit('footstep', this._pFoot);
  }

  // ==========================================================================
  //  SWEEP BACKEND
  // ==========================================================================

  /**
   * Sweep the capsule from (fx,fy,fz) to (tx,ty,tz).  Writes this._swEnd,
   * _swAdv, _swHit, _swRich, _swWall/_swCeil/_swGround, _swStep, _swN* and
   * _swSurface.  Never allocates.
   */
  _sweep(fx, fy, fz, tx, ty, tz, radius, height) {
    this._swHit = false;
    this._swRich = false;
    this._swWall = false;
    this._swCeil = false;
    this._swGround = false;
    this._swStep = 0;
    this._swAdv = 1;
    this._swNX = 0; this._swNY = 0; this._swNZ = 0;
    this._swSurface = null;

    const p = this.g?.physics;
    if (p && !this._physBad && typeof p.capsuleSweep === 'function') {
      if (this._physSweep(p, fx, fy, fz, tx, ty, tz, radius, height)) return;
    }
    this._fallbackSweep(fx, fy, fz, tx, ty, tz, radius, height);
  }

  /**
   * How far did we actually get ALONG the direction we asked to move?
   * Measuring the projection rather than the raw endpoint distance is what
   * makes this immune to the few millimetres of depenetration drift a real
   * collision solver applies perpendicular to the sweep every single frame.
   */
  _measureAdvance(fx, fy, fz, tx, ty, tz) {
    const dx = tx - fx, dy = ty - fy, dz = tz - fz;
    const req = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (req < 1e-6) { this._swAdv = 1; return false; }
    const gx = this._swEnd.x - fx, gy = this._swEnd.y - fy, gz = this._swEnd.z - fz;
    const proj = (gx * dx + gy * dy + gz * dz) / req;
    this._swAdv = clamp(proj / req, -1, 1);
    // tolerance: 4 mm absolute, or 6% of the requested distance, whichever is larger
    return proj < req - Math.max(0.004, req * 0.06);
  }

  _physSweep(p, fx, fy, fz, tx, ty, tz, radius, height) {
    const from = this._vFrom.set(fx, fy, fz);
    const to = this._vTo.set(tx, ty, tz);
    const o = this._swOut;
    o.hit = false; o.fraction = 1; o.grounded = false; o.surface = null;
    o.position.set(tx, ty, tz);
    o.normal.set(0, 0, 0);

    let r;
    try {
      r = p.capsuleSweep(from, to, radius, height, o);
    } catch (e) {
      // One throw and we never call it again this session. A collaborator's
      // half-built module must not be able to take the player down with it.
      this._physBad = true;
      if (!this._physWarned) { this._physWarned = true; console.warn('[Controller] physics.capsuleSweep threw — using internal collision', e); }
      return false;
    }
    this._physOk = true;

    // ---- parse whatever came back ---------------------------------------
    let end = null, frac = -1, hit = null, nx = 0, ny = 0, nz = 0, surf = null;
    let rich = false, wall = false, ceil = false, ground = false, stepped = 0;

    if (r && typeof r === 'object') {
      if (r.isVector3) end = r;
      else {
        const ep = r.position || r.pos || r.point || r.end || r.origin;
        if (ep && typeof ep.x === 'number') end = ep;

        // Explicit, per-feature flags from a full collision solver. These are
        // FAR better than any distance heuristic, so they win outright.
        if (typeof r.hitWall === 'boolean') { wall = r.hitWall; rich = true; }
        if (typeof r.hitCeiling === 'boolean') { ceil = r.hitCeiling; rich = true; }
        if (typeof r.grounded === 'boolean') { ground = r.grounded; rich = true; }
        else if (typeof r.onGround === 'boolean') { ground = r.onGround; rich = true; }
        if (typeof r.steppedUp === 'number') { stepped = r.steppedUp; rich = true; }
        else if (typeof r.stepUp === 'number') { stepped = r.stepUp; rich = true; }

        // Prefer a wall normal for the slide response; fall back to the generic one.
        const n = r.wallNormal && (r.wallNormal.x || r.wallNormal.y || r.wallNormal.z)
          ? r.wallNormal : (r.normal || r.n || r.groundNormal);
        if (n && typeof n.x === 'number') { nx = n.x; ny = n.y; nz = n.z; }

        // NOTE: `touched` and `grounded` are deliberately NOT treated as "hit" —
        // both are true every frame while simply standing on the floor.
        if (typeof r.hit === 'boolean') hit = r.hit;
        else if (typeof r.collided === 'boolean') hit = r.collided;
        else if (typeof r.hits === 'boolean') hit = r.hits;

        const fr = (typeof r.fraction === 'number') ? r.fraction
          : (typeof r.t === 'number') ? r.t
            : (typeof r.toi === 'number') ? r.toi : -1;
        if (fr >= 0) frac = fr;

        if (typeof r.groundSurface === 'string') surf = r.groundSurface;
        else if (typeof r.surface === 'string') surf = r.surface;
        else if (typeof r.material === 'string') surf = r.material;
        else if (r.object?.material?.userData?.bsSurface) surf = r.object.material.userData.bsSurface;
      }
    } else if (r === true) {
      hit = true;
    }

    // nothing returned — maybe the implementation filled our scratch object
    if (!end && frac < 0 && hit === null) {
      if (o.hit || o.fraction < 1 ||
        o.position.x !== tx || o.position.y !== ty || o.position.z !== tz) {
        end = o.position;
        if (o.fraction < 1) frac = o.fraction;
        if (o.hit) hit = true;
        if (o.normal.lengthSq() > 1e-8) { nx = o.normal.x; ny = o.normal.y; nz = o.normal.z; }
        if (typeof o.surface === 'string') surf = o.surface;
      } else if (r === undefined && !this._physProven) {
        // The call returned nothing and touched nothing. If that keeps happening
        // for a sweep that should obviously be blocked, the module is a stub —
        // let the fallback handle it instead of letting the player fall forever.
        this._physNull = (this._physNull || 0) + 1;
        if (this._physNull > 30) { this._physBad = true; return false; }
        return false;
      }
    }
    this._physProven = true;

    if (frac >= 0 && !end) {
      const t = clamp(frac, 0, 1);
      this._swEnd.set(fx + (tx - fx) * t, fy + (ty - fy) * t, fz + (tz - fz) * t);
    } else if (end) {
      this._swEnd.set(end.x, end.y, end.z);
    } else {
      this._swEnd.set(tx, ty, tz);
    }
    if (!isFinite(this._swEnd.x) || !isFinite(this._swEnd.y) || !isFinite(this._swEnd.z)) {
      this._swEnd.set(fx, fy, fz);
    }

    const shortfall = this._measureAdvance(fx, fy, fz, tx, ty, tz);
    if (hit === null) hit = rich ? (wall || ceil || shortfall) : shortfall;

    this._swHit = !!hit;
    this._swRich = rich;
    this._swWall = wall;
    this._swCeil = ceil;
    this._swGround = ground;
    this._swStep = stepped > 0 ? stepped : 0;
    this._swNX = nx; this._swNY = ny; this._swNZ = nz;
    this._swSurface = surf;
    return true;
  }

  // ==========================================================================
  //  FALLBACK COLLISION WORLD
  //  A flat array of AABBs harvested from game.registry.colliders, plus an
  //  infinite ground plane. Not a physics engine — a guarantee that the player
  //  module is playable and screenshotable on its own.
  // ==========================================================================

  _syncFallbackWorld(force) {
    const reg = this.g?.registry;
    const list = reg?.colliders;
    if (!Array.isArray(list)) return;
    if (!force && list.length === this._fbLastLen) return;
    this._fbLastLen = list.length;

    const n = Math.min(list.length, MAX_FB_BOXES);
    if (!this._fbBoxes || this._fbBoxes.length < n * 6) {
      this._fbBoxes = new Float32Array(Math.max(64, n) * 6);
      this._fbSurf = new Array(Math.max(64, n));
    }
    const B = this._fbBoxes;
    let c = 0;
    const box = this._fbScratchBox || (this._fbScratchBox = new THREE.Box3());

    for (let i = 0; i < n; i++) {
      const e = list[i];
      if (!e) continue;
      let min = null, max = null, surf = null;
      try {
        if (e.isBox3) { min = e.min; max = e.max; }
        else if (e.box && e.box.isBox3) { min = e.box.min; max = e.box.max; surf = e.surface || null; }
        else if (e.min && e.max && typeof e.min.x === 'number') { min = e.min; max = e.max; surf = e.surface || null; }
        else if (e.isObject3D) {
          if (e.userData?.noCollide) continue;
          box.makeEmpty();
          box.setFromObject(e, true);
          if (box.isEmpty()) continue;
          min = box.min; max = box.max;
          surf = e.userData?.surface
            || (Array.isArray(e.material) ? e.material[0]?.userData?.bsSurface : e.material?.userData?.bsSurface)
            || null;
        } else if (e.object?.isObject3D) {
          box.makeEmpty();
          box.setFromObject(e.object, true);
          if (box.isEmpty()) continue;
          min = box.min; max = box.max;
          surf = e.surface || e.object.userData?.surface || null;
        }
      } catch (err) { continue; }
      if (!min || !max) continue;
      if (!isFinite(min.x) || !isFinite(max.x)) continue;
      // reject degenerate / absurd volumes
      if (max.x - min.x > 4000 || max.y - min.y > 4000 || max.z - min.z > 4000) continue;

      const o = c * 6;
      B[o] = min.x; B[o + 1] = min.y; B[o + 2] = min.z;
      B[o + 3] = max.x; B[o + 4] = max.y; B[o + 5] = max.z;
      this._fbSurf[c] = surf;
      c++;
    }
    this._fbCount = c;
  }

  _fbGather(minX, minY, minZ, maxX, maxY, maxZ) {
    const B = this._fbBoxes;
    let n = 0;
    if (!B) { this._fbCandN = 0; return; }
    const cnt = this._fbCount;
    for (let i = 0; i < cnt; i++) {
      const o = i * 6;
      if (B[o + 3] < minX || B[o] > maxX) continue;
      if (B[o + 4] < minY || B[o + 1] > maxY) continue;
      if (B[o + 5] < minZ || B[o + 2] > maxZ) continue;
      this._fbCand[n++] = i;
      if (n >= MAX_FB_CAND) break;
    }
    this._fbCandN = n;
  }

  _fallbackSweep(fx, fy, fz, tx, ty, tz, radius, height) {
    const dx = tx - fx, dy = ty - fy, dz = tz - fz;
    const pad = radius + 0.05;

    this._fbGather(
      Math.min(fx, tx) - pad, Math.min(fy, ty) - 0.05, Math.min(fz, tz) - pad,
      Math.max(fx, tx) + pad, Math.max(fy, ty) + height + 0.05, Math.max(fz, tz) + pad,
    );

    let px = fx, py = fy, pz = fz;
    let hit = false;
    let surf = null;
    const B = this._fbBoxes;
    const N = this._fbCandN;
    const EPSP = 0.0015;

    // ---- Y ---------------------------------------------------------------
    if (dy !== 0) {
      let ny = py + dy;
      // infinite ground plane
      if (dy < 0 && ny < FB_GROUND_Y) { ny = FB_GROUND_Y; hit = true; surf = surf || null; }
      for (let i = 0; i < N; i++) {
        const o = this._fbCand[i] * 6;
        if (px + radius <= B[o] || px - radius >= B[o + 3]) continue;
        if (pz + radius <= B[o + 2] || pz - radius >= B[o + 5]) continue;
        if (ny + height <= B[o + 1] || ny >= B[o + 4]) continue;
        if (dy > 0) {
          const lim = B[o + 1] - height - EPSP;
          if (lim < ny) { ny = lim; hit = true; }
        } else {
          const lim = B[o + 4] + EPSP;
          if (lim > ny) { ny = lim; hit = true; surf = this._fbSurf[this._fbCand[i]] || surf; }
        }
      }
      if (dy > 0 && ny < py) ny = py;
      if (dy < 0 && ny > py) ny = py;
      py = ny;
    } else if (py < FB_GROUND_Y) {
      py = FB_GROUND_Y;
    }

    // ---- X ---------------------------------------------------------------
    if (dx !== 0) {
      let nx = px + dx;
      for (let i = 0; i < N; i++) {
        const o = this._fbCand[i] * 6;
        if (py + height <= B[o + 1] || py >= B[o + 4]) continue;
        if (pz + radius <= B[o + 2] || pz - radius >= B[o + 5]) continue;
        if (nx + radius <= B[o] || nx - radius >= B[o + 3]) continue;
        if (dx > 0) { const lim = B[o] - radius - EPSP; if (lim < nx) { nx = lim; hit = true; } }
        else { const lim = B[o + 3] + radius + EPSP; if (lim > nx) { nx = lim; hit = true; } }
      }
      if (dx > 0 && nx < px) nx = px;
      if (dx < 0 && nx > px) nx = px;
      px = nx;
    }

    // ---- Z ---------------------------------------------------------------
    if (dz !== 0) {
      let nz = pz + dz;
      for (let i = 0; i < N; i++) {
        const o = this._fbCand[i] * 6;
        if (py + height <= B[o + 1] || py >= B[o + 4]) continue;
        if (px + radius <= B[o] || px - radius >= B[o + 3]) continue;
        if (nz + radius <= B[o + 2] || nz - radius >= B[o + 5]) continue;
        if (dz > 0) { const lim = B[o + 2] - radius - EPSP; if (lim < nz) { nz = lim; hit = true; } }
        else { const lim = B[o + 5] + radius + EPSP; if (lim > nz) { nz = lim; hit = true; } }
      }
      if (dz > 0 && nz < pz) nz = pz;
      if (dz < 0 && nz > pz) nz = pz;
      pz = nz;
    }

    this._swEnd.set(px, py, pz);
    this._measureAdvance(fx, fy, fz, tx, ty, tz);
    this._swHit = hit;
    this._swRich = false;                // the movement code runs its own logic
    this._swGround = hit && dy < 0;
    this._swCeil = hit && dy > 0;
    this._swWall = hit && (dx !== 0 || dz !== 0) && dy === 0;
    this._swStep = 0;
    this._swSurface = surf;
    if (hit) {
      // axis-aligned normal, good enough for the fallback path
      this._swNX = 0; this._swNY = (dy < 0 ? 1 : dy > 0 ? -1 : 0); this._swNZ = 0;
    }
  }

  // ==========================================================================
  dispose() {
    if (!this._bound) return;
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onBlur);
    window.removeEventListener('mouseup', this._onMouseUp);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    document.removeEventListener('pointerlockerror', this._onLockError);
    this._el?.removeEventListener('mousedown', this._onMouseDown);
    this._el?.removeEventListener('contextmenu', this._onContext);
    this._bound = false;
  }
}

Controller.KEYMAP = {
  KeyW: 'f', ArrowUp: 'f',
  KeyS: 'b', ArrowDown: 'b',
  KeyA: 'l', ArrowLeft: 'l',
  KeyD: 'r', ArrowRight: 'r',
  Space: 'jump',
  ShiftLeft: 'sprint', ShiftRight: 'sprint',
  ControlLeft: 'crouch', ControlRight: 'crouch', KeyC: 'crouch',
  KeyQ: 'leanL', KeyE: 'leanR',
};

export default Controller;
