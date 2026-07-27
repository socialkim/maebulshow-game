// ============================================================================
// BLACKSITE — src/audio/Audio.js
// Owner: Audio agent.  100% procedural WebAudio.  No files, no fetches, no CDN.
//
// Everything you hear is synthesised in this file at runtime:
//   * weapon fire   — layered transient / body / sub / mechanical click, convolved
//                     through a procedurally generated impulse response, with a few
//                     percent of per-shot pitch, level and cutoff jitter so sustained
//                     fire never degenerates into a looped sample.
//   * distant fire  — a separate heavily-lowpassed "crack-thump" whose arrival delay
//                     is the real speed of sound over the real distance (343 m/s).
//   * tails         — three IRs (tight interior, open street, alley flutter) built from
//                     exponentially decaying filtered noise + sparse early reflections,
//                     crossfaded from a raycast probe of the player's surroundings.
//   * impacts, footsteps (randomised gait), shell casings, mag drops, bolt mechanics
//   * full 3D spatialisation via PannerNode + a distance-driven air-absorption lowpass
//   * enemy radio chatter: formant-filtered noise syllables through a squelchy radio chain
//   * player state: sprint breathing, low-health heartbeat, concussion lowpass, tinnitus
//   * a golden-hour ambience bed — wind, low city rumble, occasional distant contacts
//
// ---------------------------------------------------------------------------
// PUBLIC API (collaborators — read this)
// ---------------------------------------------------------------------------
//   game.audio.ready                 bool    — context exists and is running
//   game.audio.ctx                   AudioContext | null
//   game.audio.resume()              force-create / resume (safe to call anywhere)
//   game.audio.setMuted(bool) / game.audio.muted
//   game.audio.setMasterVolume(v)    0..1  (also writes Config.audio.master)
//   game.audio.gunshot(opts)         {x,y,z, self:bool, distance, weapon, suppressed}
//   game.audio.distantReport(x,y,z)  crack-thump with speed-of-sound delay
//   game.audio.impact(surface, x,y,z, power)
//   game.audio.footstep(surface, speed, x,y,z, self)
//   game.audio.shellCasing(x,y,z, delay)
//   game.audio.magDrop(x,y,z, surface)
//   game.audio.bolt(kind)            'back'|'release'|'magout'|'magin'|'click'
//   game.audio.chatter(x,y,z, kind)  'contact'|'suppress'|'move'|'hit'|'radio'
//   game.audio.explosion(x,y,z, radius, power)
//   game.audio.tinnitus(strength, seconds)
//   game.audio.concuss(strength)
//   game.audio.setSpace('auto'|'interior'|'street'|'alley')
//   game.audio.stats                 {state, space, voices, shots, probeHits}
//
// Everything is a no-op (never a throw, never a warn) until a real user gesture
// creates the AudioContext.  The headless screenshot critic runs with no gesture,
// so in that environment this module allocates nothing and does nothing at all.
// ============================================================================

import * as THREE from 'three';

const SPEED_OF_SOUND = 343;        // m/s at ~25 C
const MAX_AUDIBLE = 320;           // metres — past this we do not schedule anything
const SLOT_COUNT = 28;             // spatial voice slots

// Master gain structure, measured off the real graph rather than guessed at.
// Gunfire is the loudest recurring event and sets the ceiling; everything else
// is placed underneath it. Changing one of these moves that whole family only.
const SHOT_LEVEL = 0.66;
const FOOT_LEVEL = 0.50;
const IMPACT_LEVEL = 0.62;
const EXP_LEVEL = 0.72;

/* ------------------------------------------------------------------ *
 *  Deterministic PRNG — reproducible runs, no Math.random surprises.
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
const lerp = (a, b, t) => a + (b - a) * t;
// Collaborators pass whatever they have. A single NaN reaching an AudioParam
// throws a DOMException and takes the frame with it, so every number that can
// come from outside this file goes through here first.
const num = (v, d) => (typeof v === 'number' && v === v && v !== Infinity && v !== -Infinity) ? v : d;

/* ------------------------------------------------------------------ *
 *  Surface tables.  The Bus contract guarantees one of six surface
 *  buckets, but Materials has fourteen keys and collaborators will pass
 *  whatever they have — so everything funnels through SURF_MAP first.
 * ------------------------------------------------------------------ */
const SURF_MAP = {
  concrete: 'concrete', plaster: 'concrete', brick: 'concrete', asphalt: 'concrete',
  stone: 'concrete', cement: 'concrete', rock: 'concrete', tile: 'concrete', stucco: 'concrete',
  road: 'concrete', tarmac: 'concrete', street: 'concrete', wall: 'concrete',
  sand: 'sand', dirt: 'sand', gravel: 'sand', ground: 'sand', dust: 'sand', soil: 'sand',
  metal: 'metal', rustmetal: 'metal', gunmetal: 'metal', steel: 'metal', iron: 'metal',
  rust: 'metal', container: 'metal', barrel: 'metal', corrugated: 'metal',
  wood: 'wood', plank: 'wood', crate: 'wood', plywood: 'wood', timber: 'wood',
  glass: 'glass', window: 'glass', mirror: 'glass',
  flesh: 'flesh', skin: 'flesh', body: 'flesh', head: 'flesh',
  fabric: 'soft', cloth: 'soft', cloth_tan: 'soft', rubber: 'soft', foliage: 'soft',
  polymer: 'soft', carpet: 'soft', tarp: 'soft', water: 'soft',
};

/**
 * Impact recipes.  Each is a single generic synth driven by these numbers, so
 * every surface is tuned in one place rather than in six copy-pasted functions.
 *   tf/tq   transient bandpass centre / Q          td   transient decay
 *   bf      body lowpass start (swept down to be)  bd   body decay
 *   sub     sub-thump frequency (0 = none)         subG sub gain
 *   ring    metallic partial gain (0 = none)       rf   ring base frequency
 *   grit    granular debris amount                 wet  reverb send
 */
const IMPACT = {
  concrete: { tf: 2800, tq: 1.1, td: 0.045, bf: 1800, be: 260, bd: 0.10, sub: 62, subG: 0.30, ring: 0, rf: 0, grit: 0.85, wet: 0.55, g: 1.00 },
  sand:     { tf: 1100, tq: 0.7, td: 0.055, bf: 900,  be: 130, bd: 0.13, sub: 48, subG: 0.22, ring: 0, rf: 0, grit: 0.55, wet: 0.28, g: 0.80 },
  metal:    { tf: 4200, tq: 2.2, td: 0.030, bf: 3000, be: 700, bd: 0.07, sub: 90, subG: 0.14, ring: 0.42, rf: 1750, grit: 0.35, wet: 0.72, g: 1.05 },
  wood:     { tf: 1900, tq: 1.4, td: 0.038, bf: 1300, be: 210, bd: 0.11, sub: 78, subG: 0.26, ring: 0.10, rf: 520, grit: 0.45, wet: 0.42, g: 0.92 },
  glass:    { tf: 6200, tq: 3.0, td: 0.022, bf: 5200, be: 1600, bd: 0.05, sub: 0, subG: 0, ring: 0.55, rf: 3400, grit: 1.00, wet: 0.65, g: 0.95 },
  flesh:    { tf: 700,  tq: 0.9, td: 0.030, bf: 520,  be: 95,  bd: 0.12, sub: 55, subG: 0.42, ring: 0, rf: 0, grit: 0.18, wet: 0.22, g: 1.00 },
  soft:     { tf: 1300, tq: 0.8, td: 0.030, bf: 800,  be: 150, bd: 0.09, sub: 52, subG: 0.16, ring: 0, rf: 0, grit: 0.25, wet: 0.30, g: 0.72 },
};

/**
 * Footstep recipes.  Two-stage: heel strike then a shorter toe scuff, both
 * shaped from the same noise bank.
 */
const FOOT = {
  concrete: { hf: 1450, hq: 1.0, hd: 0.055, sf: 3200, sq: 1.6, sd: 0.075, sub: 78, subG: 0.30, scuff: 0.30, g: 1.00, wet: 0.55 },
  sand:     { hf: 780,  hq: 0.6, hd: 0.075, sf: 2100, sq: 0.9, sd: 0.110, sub: 58, subG: 0.20, scuff: 0.70, g: 0.86, wet: 0.22 },
  metal:    { hf: 1900, hq: 1.5, hd: 0.060, sf: 4200, sq: 2.4, sd: 0.140, sub: 96, subG: 0.24, scuff: 0.25, g: 1.05, wet: 0.70 },
  wood:     { hf: 1100, hq: 1.2, hd: 0.060, sf: 2600, sq: 1.4, sd: 0.090, sub: 88, subG: 0.34, scuff: 0.28, g: 0.95, wet: 0.45 },
  glass:    { hf: 3800, hq: 2.4, hd: 0.045, sf: 6200, sq: 2.8, sd: 0.090, sub: 70, subG: 0.16, scuff: 0.55, g: 0.90, wet: 0.60 },
  flesh:    { hf: 620,  hq: 0.7, hd: 0.060, sf: 1500, sq: 0.9, sd: 0.070, sub: 50, subG: 0.30, scuff: 0.20, g: 0.85, wet: 0.30 },
  soft:     { hf: 900,  hq: 0.7, hd: 0.070, sf: 1900, sq: 1.0, sd: 0.090, sub: 55, subG: 0.18, scuff: 0.45, g: 0.72, wet: 0.28 },
};

/**
 * Impulse-response specs.  time = tail length in seconds, decay = exponential
 * rate, pow = extra polynomial fade so the tail truly reaches silence, lpf =
 * one-pole lowpass coefficient on the noise (darker = smaller), early = sparse
 * discrete reflections [timeSec, amplitude], flutter = comb period for the alley.
 */
const IR_SPECS = {
  interior: {
    time: 0.85, decay: 7.2, pow: 1.6, lpf: 0.16, hpf: 0.986, gain: 1.00, flutter: 0,
    early: [[0.006, 0.72], [0.011, 0.58], [0.017, 0.46], [0.024, 0.38], [0.033, 0.30],
            [0.041, 0.24], [0.052, 0.19], [0.068, 0.14], [0.085, 0.10]],
  },
  street: {
    time: 2.40, decay: 3.1, pow: 1.15, lpf: 0.36, hpf: 0.9955, gain: 1.00, flutter: 0,
    early: [[0.019, 0.44], [0.037, 0.36], [0.058, 0.30], [0.079, 0.24], [0.108, 0.19],
            [0.145, 0.15], [0.190, 0.12], [0.245, 0.09], [0.320, 0.07], [0.410, 0.05]],
  },
  alley: {
    time: 1.55, decay: 4.4, pow: 1.3, lpf: 0.26, hpf: 0.992, gain: 1.00, flutter: 0.0295,
    early: [[0.009, 0.62], [0.021, 0.52], [0.034, 0.44], [0.050, 0.36], [0.071, 0.28],
            [0.098, 0.22], [0.132, 0.17], [0.178, 0.12]],
  },
};

/* ================================================================== *
 *  Audio
 * ================================================================== */

export class Audio {

  constructor(game) {
    this.g = game;
    this.ctx = null;
    this.ready = false;
    this.muted = false;
    this.booting = false;
    this.bootError = null;

    const cfg = game?.config?.audio ?? {};
    this._volMaster = cfg.master ?? 0.8;
    this._volSfx = cfg.sfx ?? 1.0;
    this._volMusic = cfg.music ?? 0.35;

    this._rnd = mulberry32(0xB1AC5175);

    // ---- listener / player state (tracked even with no context, so the first
    //      frame after a gesture already has a warm, correct state) -----------
    this._camPrev = new THREE.Vector3();
    this._camHasPrev = false;
    this._playerSpeed = 0;
    this._playerSpeedSm = 0;
    this._sprinting = false;
    this._grounded = true;
    this._exertion = 0;         // 0..1, drives breathing
    this._hp = 1;
    this._hpMax = 100;
    this._ads = false;
    this._mag = -1;
    this._reserve = -1;

    // ---- effect state -----------------------------------------------------
    this._concuss = 0;          // 0..1 muffle amount
    this._concussApplied = -1;
    this._tinnitusUntil = 0;
    this._tinnitusNodes = null;
    this._duckUntil = 0;

    // ---- schedulers (absolute ctx times) ----------------------------------
    this._nextBreath = 0;
    this._breathPhase = 0;      // 0 = inhale, 1 = exhale
    this._nextHeart = 0;
    this._heartPhase = 0;
    this._nextAmbEvent = 0;
    this._nextStep = 0;
    this._stepFoot = 0;
    this._autoFootsteps = true; // disabled permanently once a real 'footstep' arrives
    this._lastShotAt = -1;

    // ---- reverb space -----------------------------------------------------
    this._spaceMode = 'auto';
    this._spaceW = { interior: 0, street: 1, alley: 0 };
    this._spaceTarget = { interior: 0, street: 1, alley: 0 };
    this._spaceSend = 1.0;
    this._probeT = 0;
    this._probeHits = 0;
    this._probeCeiling = 0;
    this._probeList = [];
    this._probeLen = -1;
    this._ray = new THREE.Raycaster();
    this._ray.far = 20;
    this._rayHits = [];
    this._probeDirs = [
      new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0.707, 0, 0.707), new THREE.Vector3(-0.707, 0, -0.707),
      new THREE.Vector3(0.707, 0, -0.707), new THREE.Vector3(-0.707, 0, 0.707),
      new THREE.Vector3(0, 1, 0),
    ];
    this._probeOrigin = new THREE.Vector3();
    this._probeNear = new Float32Array(8);
    this._vTmp = new THREE.Vector3();
    this._vTmp2 = new THREE.Vector3();
    this._lastGroundSurface = 'concrete';

    // ---- pools ------------------------------------------------------------
    this._slots = [];
    this._slotCursor = 0;
    this._buf = null;

    this._unbind = [];
    this._gestureHandler = null;
    this._gestureEvents = ['pointerdown', 'mousedown', 'touchstart', 'keydown', 'wheel'];

    this.stats = { state: 'none', space: 'street', voices: 0, shots: 0, steps: 0, probeHits: 0, ir: 0 };
  }

  /* ================================================================ *
   *  Lifecycle
   * ================================================================ */

  async init() {
    this._bindBus();

    // The autoplay policy makes an ungestured AudioContext a console warning in
    // Chrome, and the headless critic treats warnings as build failures — so we
    // do not touch WebAudio at all until the page has genuinely been activated.
    const activated = (typeof navigator !== 'undefined' && navigator.userActivation)
      ? !!navigator.userActivation.hasBeenActive : false;

    if (activated) {
      this._boot();
    } else if (typeof window !== 'undefined') {
      this._gestureHandler = () => this.resume();
      for (const e of this._gestureEvents) {
        try { window.addEventListener(e, this._gestureHandler, { passive: true, capture: true }); } catch (_) { }
      }
      try {
        document.addEventListener('pointerlockchange', this._gestureHandler, false);
        this._unbind.push(() => document.removeEventListener('pointerlockchange', this._gestureHandler, false));
      } catch (_) { }
    }
  }

  /** Force the context up. Safe to call at any time, from anywhere, repeatedly. */
  resume() {
    if (!this.ctx) { this._boot(); return; }
    if (this.ctx.state === 'suspended') {
      try { const p = this.ctx.resume(); if (p && p.catch) p.catch(() => { }); } catch (_) { }
    }
  }

  _dropGestureListeners() {
    if (!this._gestureHandler || typeof window === 'undefined') return;
    for (const e of this._gestureEvents) {
      try { window.removeEventListener(e, this._gestureHandler, { capture: true }); } catch (_) { }
    }
    this._gestureHandler = null;
  }

  _boot() {
    if (this.ctx || this.booting) return;
    if (typeof window === 'undefined') return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;

    this.booting = true;
    try {
      const ctx = new AC({ latencyHint: 'interactive' });
      this.ctx = ctx;
      this._nyq = ctx.sampleRate * 0.5;
      this._maxLP = Math.min(20000, this._nyq * 0.92);

      this._buildBuffers();
      this._buildGraph();
      this._buildSlots();
      this._startAmbience();

      const now = ctx.currentTime;
      this._nextBreath = now + 1.2;
      this._nextHeart = now + 1.0;
      this._nextAmbEvent = now + 5 + this._rnd() * 8;

      if (ctx.state === 'suspended') {
        try { const p = ctx.resume(); if (p && p.catch) p.catch(() => { }); } catch (_) { }
      }
      this.ready = true;
      this._dropGestureListeners();
    } catch (e) {
      // Never throw out of audio. A missing/blocked context just means silence.
      this.bootError = e;
      this.ctx = null;
      this.ready = false;
    }
    this.booting = false;
  }

  /* ================================================================ *
   *  Buffer bank — noise flavours + impulse responses
   * ================================================================ */

  _buildBuffers() {
    const ctx = this.ctx, sr = ctx.sampleRate, r = this._rnd;
    const b = {};

    // --- white: flat, the raw material for transients ---------------------
    b.white = ctx.createBuffer(1, (sr * 2.0) | 0, sr);
    {
      const d = b.white.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = r() * 2 - 1;
    }

    // --- pink: -3 dB/oct via the Paul Kellett economy filter. This is the
    //     workhorse for bodies and tails — white noise alone sounds like a
    //     hiss, pink sounds like air being displaced.
    b.pink = ctx.createBuffer(1, (sr * 2.0) | 0, sr);
    {
      const d = b.pink.getChannelData(0);
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
      let peak = 1e-6;
      for (let i = 0; i < d.length; i++) {
        const w = r() * 2 - 1;
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520;
        b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.0168980;
        const v = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
        b6 = w * 0.115926;
        d[i] = v;
        const a = v < 0 ? -v : v; if (a > peak) peak = a;
      }
      const k = 0.92 / peak;
      for (let i = 0; i < d.length; i++) d[i] *= k;
    }

    // --- brown: -6 dB/oct, for sub rumble and explosion bodies -------------
    b.brown = ctx.createBuffer(1, (sr * 2.0) | 0, sr);
    {
      const d = b.brown.getChannelData(0);
      let last = 0, peak = 1e-6;
      for (let i = 0; i < d.length; i++) {
        last = (last + 0.021 * (r() * 2 - 1)) / 1.021;
        d[i] = last;
        const a = last < 0 ? -last : last; if (a > peak) peak = a;
      }
      const k = 0.90 / peak;
      for (let i = 0; i < d.length; i++) d[i] *= k;
    }

    // --- grain: sparse impulsive crackle. Concrete spall, glass shards,
    //     gravel underfoot, debris after a blast. A dense noise burst does not
    //     sound like fragments — discrete transients do.
    b.grain = ctx.createBuffer(1, (sr * 1.5) | 0, sr);
    {
      const d = b.grain.getChannelData(0);
      const n = d.length;
      const count = (n / sr) * 2200 | 0;
      for (let k = 0; k < count; k++) {
        const at = (r() * (n - 64)) | 0;
        const amp = Math.pow(r(), 2.6) * (r() < 0.5 ? -1 : 1);
        const len = 4 + (r() * 40) | 0;
        const dec = 1 / len;
        for (let i = 0; i < len; i++) {
          const e = 1 - i * dec;
          d[at + i] += amp * e * e * (r() * 2 - 1);
        }
      }
      let peak = 1e-6;
      for (let i = 0; i < n; i++) { const a = d[i] < 0 ? -d[i] : d[i]; if (a > peak) peak = a; }
      const k2 = 0.90 / peak;
      for (let i = 0; i < n; i++) d[i] *= k2;
    }

    // --- wind: 6 s of band-limited pink with a matched head/tail crossfade so
    //     it loops seamlessly forever with zero clicks.
    b.wind = ctx.createBuffer(2, (sr * 6.0) | 0, sr);
    for (let ch = 0; ch < 2; ch++) {
      const d = b.wind.getChannelData(ch);
      let lp1 = 0, lp2 = 0, lp3 = 0, hp = 0, prev = 0, peak = 1e-6;
      for (let i = 0; i < d.length; i++) {
        const w = r() * 2 - 1;
        lp1 += (w - lp1) * 0.055;
        lp2 += (lp1 - lp2) * 0.055;
        lp3 += (lp2 - lp3) * 0.12;
        hp = 0.994 * (hp + lp3 - prev); prev = lp3;
        d[i] = hp;
        const a = hp < 0 ? -hp : hp; if (a > peak) peak = a;
      }
      const k = 0.85 / peak;
      for (let i = 0; i < d.length; i++) d[i] *= k;
      this._loopFade(d, (sr * 0.35) | 0);
    }

    // --- rumble: 8 s of very low brown noise, likewise loop-matched ---------
    b.rumble = ctx.createBuffer(2, (sr * 8.0) | 0, sr);
    for (let ch = 0; ch < 2; ch++) {
      const d = b.rumble.getChannelData(ch);
      let last = 0, lp = 0, peak = 1e-6;
      for (let i = 0; i < d.length; i++) {
        last = (last + 0.014 * (r() * 2 - 1)) / 1.014;
        lp += (last - lp) * 0.02;
        d[i] = lp;
        const a = lp < 0 ? -lp : lp; if (a > peak) peak = a;
      }
      const k = 0.85 / peak;
      for (let i = 0; i < d.length; i++) d[i] *= k;
      this._loopFade(d, (sr * 0.5) | 0);
    }

    this._buf = b;

    // --- waveshaper curve, shared by every radio-chatter voice --------------
    const N = 1024;
    const curve = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * 2 - 1;
      // asymmetric soft clip — the grit of a cheap handheld transceiver
      curve[i] = Math.tanh(x * 3.4) * 0.82 + Math.tanh(x * 11) * 0.18;
    }
    this._radioCurve = curve;

    // --- impulse responses --------------------------------------------------
    this._ir = {
      interior: this._makeIR(IR_SPECS.interior),
      street: this._makeIR(IR_SPECS.street),
      alley: this._makeIR(IR_SPECS.alley),
    };
    this.stats.ir = 3;
  }

  /** Blend the tail of a buffer into its head so `loop = true` is click-free. */
  _loopFade(d, fade) {
    const n = d.length;
    if (fade * 2 >= n) return;
    for (let i = 0; i < fade; i++) {
      const t = i / fade;
      const s = t * t * (3 - 2 * t);
      const head = d[i], tail = d[n - fade + i];
      d[i] = head * s + tail * (1 - s);
    }
    // the last `fade` samples are now duplicated information; taper them out
    for (let i = 0; i < fade; i++) {
      const t = i / fade;
      d[n - fade + i] *= (1 - t);
    }
  }

  /**
   * Procedural impulse response: exponentially decaying, one-pole filtered
   * noise for the diffuse late field, plus discrete early reflections and an
   * optional flutter comb.  Decorrelated per channel so the tail is genuinely
   * stereo rather than a mono blob in the middle of the image.
   * Energy-normalised, so swapping IRs never changes perceived loudness.
   */
  _makeIR(spec) {
    const ctx = this.ctx, sr = ctx.sampleRate, r = this._rnd;
    const len = Math.max(64, (sr * spec.time) | 0);
    const buf = ctx.createBuffer(2, len, sr);

    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      let lp = 0, hp = 0, prev = 0;
      const lpC = spec.lpf, hpC = spec.hpf;
      const invLen = 1 / len;
      for (let i = 0; i < len; i++) {
        const t = i * invLen;
        const env = Math.pow(1 - t, spec.pow) * Math.exp(-t * spec.decay);
        const w = r() * 2 - 1;
        lp += (w - lp) * lpC;
        hp = hpC * (hp + lp - prev); prev = lp;
        d[i] = hp * env;
      }
      // early reflections — slightly different arrival per ear, alternating sign
      const skew = ch === 0 ? 0.972 : 1.028;
      const spec_e = spec.early;
      for (let k = 0; k < spec_e.length; k++) {
        const idx = (spec_e[k][0] * sr * skew) | 0;
        if (idx > 0 && idx < len) {
          const s = (k & 1) ? -1 : 1;
          d[idx] += spec_e[k][1] * s * (0.72 + r() * 0.56);
          if (idx + 1 < len) d[idx + 1] += spec_e[k][1] * s * 0.45;
          if (idx + 2 < len) d[idx + 2] += spec_e[k][1] * s * 0.18;
        }
      }
      // flutter echo — parallel walls of a narrow alley
      if (spec.flutter > 0) {
        const per = spec.flutter * (ch === 0 ? 1 : 1.037);
        for (let k = 1; k <= 14; k++) {
          const idx = (per * k * sr) | 0;
          if (idx >= len) break;
          d[idx] += 0.44 * Math.pow(0.68, k) * ((k & 1) ? -1 : 1);
        }
      }
      // tiny pre-delay taper so the direct sound is never smeared
      const pre = (sr * 0.0035) | 0;
      for (let i = 0; i < pre && i < len; i++) d[i] *= i / pre;
    }

    // energy normalisation across both channels
    let e = 0;
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) e += d[i] * d[i];
    }
    const scale = (spec.gain || 1) / Math.max(1e-6, Math.sqrt(e / sr));
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] *= scale;
    }
    return buf;
  }

  /* ================================================================ *
   *  Signal graph
   * ================================================================ */

  _buildGraph() {
    const ctx = this.ctx;

    // ---- master chain -----------------------------------------------------
    // A DynamicsCompressor is a compressor, not a brickwall — with a 1.5 ms
    // attack it still let gunshot transients out at +3 dBFS, which the device
    // then hard-clips. So the last node in the chain is a soft-clip curve that
    // is transparent below -3 dBFS and asymptotes to full scale above it:
    // guaranteed no digital clipping, and what little saturation happens reads
    // as loudness rather than as crunch.
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -9;
    limiter.knee.value = 6;
    limiter.ratio.value = 16;
    limiter.attack.value = 0.0015;
    limiter.release.value = 0.22;
    this._limiter = limiter;

    // The curve's domain is [-1,1], so pre-scale by 1/3 and bake the x3 into
    // the curve: signals up to +9.5 dBFS are shaped instead of clamped.
    const CN = 4096;
    const cc = new Float32Array(CN);
    for (let i = 0; i < CN; i++) {
      const s = ((i / (CN - 1)) * 2 - 1) * 3;
      const a = s < 0 ? -s : s;
      const y = a <= 0.72 ? a : 0.72 + 0.28 * Math.tanh((a - 0.72) / 0.28);
      cc[i] = (s < 0 ? -y : y) * 0.995;
    }
    const preClip = ctx.createGain();
    preClip.gain.value = 1 / 3;
    const softClip = ctx.createWaveShaper();
    softClip.curve = cc;
    softClip.oversample = '2x';
    limiter.connect(preClip);
    preClip.connect(softClip);
    softClip.connect(ctx.destination);
    this._softClip = softClip;

    const master = ctx.createGain();
    master.gain.value = this.muted ? 0 : this._volMaster;
    master.connect(limiter);
    this.master = master;

    // ---- concussion / muffle -----------------------------------------------
    // One shared lowpass across the whole SFX bus: dropping it to a few hundred
    // hertz is the "your ears are ringing and the world went underwater" effect.
    const muffle = ctx.createBiquadFilter();
    muffle.type = 'lowpass';
    muffle.frequency.value = this._maxLP;
    muffle.Q.value = 0.55;
    const muffleTilt = ctx.createBiquadFilter();
    muffleTilt.type = 'highshelf';
    muffleTilt.frequency.value = 2400;
    muffleTilt.gain.value = 0;
    muffle.connect(muffleTilt);
    this._muffle = muffle;
    this._muffleTilt = muffleTilt;

    const sfx = ctx.createGain();
    sfx.gain.value = this._volSfx;
    muffleTilt.connect(sfx);
    sfx.connect(master);
    this.sfxOut = sfx;
    this.sfxIn = muffle;          // everything spatial lands here

    // ---- ambience bus (ducked under gunfire) --------------------------------
    const amb = ctx.createGain();
    amb.gain.value = this._volMusic;
    const ambDuck = ctx.createGain();
    ambDuck.gain.value = 1;
    ambDuck.connect(amb);
    amb.connect(master);
    this.ambOut = amb;
    this.ambIn = ambDuck;
    this._ambDuck = ambDuck;

    // ---- reverb: shared pre-delay into three parallel convolvers ------------
    const revIn = ctx.createGain();
    revIn.gain.value = 1;
    const preDelay = ctx.createDelay(0.25);
    preDelay.delayTime.value = 0.014;
    const revTilt = ctx.createBiquadFilter();
    revTilt.type = 'highpass';    // keep the mud out of the tail
    revTilt.frequency.value = 165;
    revTilt.Q.value = 0.6;
    revIn.connect(preDelay);
    preDelay.connect(revTilt);
    this.reverbIn = revIn;

    this._conv = {};
    this._convGain = {};
    for (const key of ['interior', 'street', 'alley']) {
      const c = ctx.createConvolver();
      c.normalize = false;
      c.buffer = this._ir[key];
      const g = ctx.createGain();
      g.gain.value = key === 'street' ? 0.5 : 0.0;
      revTilt.connect(c);
      c.connect(g);
      g.connect(muffle);          // the tail gets muffled too — correct, and free
      this._conv[key] = c;
      this._convGain[key] = g;
    }

    // ---- local (head-relative) chain: breathing, heartbeat, self-mechanics ---
    const localPan = ctx.createStereoPanner();
    localPan.pan.value = 0;
    const local = ctx.createGain();
    local.gain.value = 1;
    local.connect(localPan);
    localPan.connect(muffle);
    this.localIn = local;
    this._localPan = localPan;

    // ---- self weapon chain: dry + a hot reverb send, no HRTF -----------------
    const selfIn = ctx.createGain();
    const selfPan = ctx.createStereoPanner();
    selfPan.pan.value = 0.16;     // the weapon sits right of the eye line
    const selfDry = ctx.createGain();
    selfDry.gain.value = 1;
    const selfSend = ctx.createGain();
    selfSend.gain.value = 0.42;
    selfIn.connect(selfPan);
    selfPan.connect(selfDry);
    selfDry.connect(muffle);
    selfIn.connect(selfSend);
    selfSend.connect(revIn);
    this.selfIn = selfIn;
    this._selfSend = selfSend;
    this._selfPanNode = selfPan;

    // ---- tinnitus bus (bypasses the muffle: ringing is *in* your head) -------
    const tin = ctx.createGain();
    tin.gain.value = 0;
    tin.connect(master);
    this._tinBus = tin;

    this._listenerReady = !!ctx.listener;
  }

  _buildSlots() {
    const ctx = this.ctx;
    const hrtf = (this.g?.config?.quality === 'low') ? 'equalpower' : 'HRTF';
    for (let i = 0; i < SLOT_COUNT; i++) {
      const input = ctx.createGain();
      input.gain.value = 1;

      // air absorption: high frequencies die over distance, which is most of
      // what makes a far-away gunshot read as far away
      const air = ctx.createBiquadFilter();
      air.type = 'lowpass';
      air.frequency.value = this._maxLP;
      air.Q.value = 0.5;

      const panner = ctx.createPanner();
      panner.panningModel = hrtf;
      panner.distanceModel = 'inverse';
      panner.refDistance = 3.2;
      panner.maxDistance = MAX_AUDIBLE;
      panner.rolloffFactor = 1.15;
      panner.coneInnerAngle = 360;
      panner.coneOuterAngle = 360;
      panner.coneOuterGain = 1;

      const dry = ctx.createGain();
      dry.gain.value = 1;
      const send = ctx.createGain();
      send.gain.value = 0.3;

      input.connect(air);
      air.connect(panner);
      panner.connect(dry);
      dry.connect(this.sfxIn);
      air.connect(send);
      send.connect(this.reverbIn);

      this._slots.push({ input, air, panner, dry, send, freeAt: 0 });
    }
  }

  /* ================================================================ *
   *  Primitives
   * ================================================================ */

  _when(t) {
    const now = this.ctx.currentTime;
    const v = num(t, -1);
    return (v < now + 0.002) ? now + 0.002 : v;
  }

  _gain(v) { const g = this.ctx.createGain(); g.gain.value = v; return g; }

  _bq(type, f, q) {
    const b = this.ctx.createBiquadFilter();
    b.type = type;
    b.frequency.value = clamp(f, 12, this._maxLP);
    if (q !== undefined) b.Q.value = q;
    return b;
  }

  _osc(type, f) {
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.value = clamp(f, 0.5, this._maxLP);
    return o;
  }

  /** Percussive attack-decay envelope. setTargetAtTime gives the natural,
   *  never-quite-zero exponential release that a linear ramp cannot. */
  _envAD(param, when, peak, atk, dec) {
    param.cancelScheduledValues(when);
    param.setValueAtTime(0.0001, when);
    if (atk > 0.0008) param.linearRampToValueAtTime(peak, when + atk);
    else { param.setValueAtTime(peak, when + 0.0006); atk = 0.0006; }
    param.setTargetAtTime(0.0, when + atk, Math.max(0.003, dec * 0.32));
  }

  /** Exponential parameter sweep, guarded against the zero-value trap. */
  _sweep(param, when, from, to, time) {
    const a = Math.max(1e-4, from), b = Math.max(1e-4, to);
    param.cancelScheduledValues(when);
    param.setValueAtTime(a, when);
    param.exponentialRampToValueAtTime(b, when + Math.max(0.002, time));
  }

  /** One-shot slice of a noise buffer at a randomised offset. */
  _noise(kind, when, dur, rate) {
    const b = this._buf[kind] || this._buf.white;
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    const rt = rate || 1;
    s.playbackRate.value = rt;
    const need = dur * rt + 0.02;
    const maxOff = Math.max(0, b.duration - need - 0.005);
    s.start(when, this._rnd() * maxOff, need);
    return s;
  }

  /* ---------------------------------------------------------------- *
   *  Spatial voice allocation
   * ---------------------------------------------------------------- */

  _setPannerPos(p, x, y, z) {
    if (p.positionX) { p.positionX.value = x; p.positionY.value = y; p.positionZ.value = z; }
    else if (p.setPosition) p.setPosition(x, y, z);
  }

  /**
   * Grab a spatial slot, aim it, and return it. Connect your sources into
   * `slot.input`. Returns null if the sound is inaudible or the graph is down.
   *   opts.send   reverb send 0..1 (scaled by the space)
   *   opts.gain   linear input gain
   *   opts.ref    reference distance
   *   opts.roll   rolloff factor
   *   opts.dur    expected lifetime (drives slot recycling)
   *   opts.air    air-absorption multiplier (1 = normal, 0 = none)
   */
  _slot(x, y, z, opts) {
    if (!this.ctx || !this._slots.length) return null;
    const cam = this.g?.camera;
    x = num(x, cam ? cam.position.x : 0);
    y = num(y, cam ? cam.position.y : 1.6);
    z = num(z, cam ? cam.position.z : 0);
    let dist = 1;
    if (cam) {
      const dx = x - cam.position.x, dy = y - cam.position.y, dz = z - cam.position.z;
      dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (!(dist <= MAX_AUDIBLE)) return null;   // also rejects NaN

    const now = this.ctx.currentTime;
    const pool = this._slots;
    let slot = null;
    for (let i = 0; i < pool.length; i++) {
      const k = (this._slotCursor + i) % pool.length;
      if (pool[k].freeAt <= now) { slot = pool[k]; this._slotCursor = (k + 1) % pool.length; break; }
    }
    if (!slot) {
      let bt = Infinity, best = 0;
      for (let i = 0; i < pool.length; i++) if (pool[i].freeAt < bt) { bt = pool[i].freeAt; best = i; }
      slot = pool[best];
      this._slotCursor = (best + 1) % pool.length;
    }

    this._setPannerPos(slot.panner, x, y, z);
    slot.panner.refDistance = opts.ref ?? 3.2;
    slot.panner.rolloffFactor = opts.roll ?? 1.15;

    // Air absorption. 20 kHz at the muzzle, ~2 kHz at 120 m, ~900 Hz at 250 m.
    const airK = opts.air ?? 1;
    const cut = this._maxLP * Math.exp(-dist * 0.0135 * airK);
    slot.air.frequency.value = clamp(cut, 420, this._maxLP);

    slot.input.gain.value = opts.gain ?? 1;
    // Distant sources are mostly tail: the further away, the wetter.
    const distWet = clamp(dist / 55, 0, 1) * 0.55;
    slot.send.gain.value = clamp(((opts.send ?? 0.3) + distWet) * this._spaceSend, 0, 2.2);
    slot.dry.gain.value = 1;

    slot.freeAt = now + (opts.dur ?? 0.6);
    slot.dist = dist;
    return slot;
  }

  /** Duck the ambience bed under a loud event. */
  _duck(amount, hold) {
    const now = this.ctx.currentTime;
    const g = this._ambDuck.gain;
    g.cancelScheduledValues(now);
    g.setTargetAtTime(clamp(1 - amount, 0.1, 1), now, 0.02);
    g.setTargetAtTime(1, now + (hold ?? 0.14), 0.30);
    this._duckUntil = now + (hold ?? 0.14) + 1.0;
  }

  /* ================================================================ *
   *  WEAPON FIRE
   * ================================================================ */

  /**
   * @param {object} o  {x,y,z}=muzzle world position, self=player's own weapon,
   *                    suppressed, gain, when
   */
  gunshot(o) {
    if (!this.ready || !this.ctx) return;
    const ctx = this.ctx;
    const now = ctx.currentTime;
    // Anti-storm clamp only: 780 RPM is 77 ms between rounds, so this never
    // drops a legitimate shot — it exists so a caller that emits a whole burst
    // inside one tick cannot stack sixty identical transients on one sample.
    if (now - this._lastShotAt < 0.012) return;
    this._lastShotAt = now;

    o = o || {};
    const self = !!o.self;
    const supp = !!o.suppressed;
    const when = this._when(o.when);
    const r = this._rnd;

    // --- per-shot variance: this is the whole ballgame. Same sample twice in a
    //     row is what makes hobby FPS audio sound like a toy.
    const p = 1 + (r() - 0.5) * 0.085;             // +-4.25% pitch
    const lvl = num(o.gain, 1) * SHOT_LEVEL * (0.90 + r() * 0.20) * (supp ? 0.34 : 1);
    const fj = 0.92 + r() * 0.16;                  // cutoff jitter

    let dest, dur;
    if (self) {
      dest = this.selfIn;
      dur = 0.9;
      // slight stereo wander so consecutive shots do not stack in one spot
      this._selfPanJitter(r);
    } else {
      const s = this._slot(num(o.x, 0), num(o.y, 1.4), num(o.z, 0), {
        send: supp ? 0.35 : 0.85, gain: 1, ref: 4.5, roll: 1.05, dur: 1.1,
      });
      if (!s) return;
      dest = s.input;
      dur = 1.1;
    }

    // ---- 1. transient crack: the supersonic snap off the muzzle -----------
    {
      const n = this._noise('white', when, 0.10, 0.85 + r() * 0.35);
      const lp = this._bq('lowpass', 9500 * p * fj, 0.85);
      this._sweep(lp.frequency, when, clamp(11000 * p * fj, 200, this._maxLP), 1350 * p, supp ? 0.085 : 0.052);
      const hp = this._bq('highpass', supp ? 260 : 430, 0.72);
      const g = this._gain(0);
      this._envAD(g.gain, when, (supp ? 0.55 : 1.05) * lvl, 0.0007, supp ? 0.10 : 0.070);
      n.connect(lp); lp.connect(hp); hp.connect(g); g.connect(dest);
    }

    // ---- 2. body: two detuned saws crashing down through a resonant lowpass,
    //        blended with pink noise. Saws alone are a synth; noise alone is a
    //        hiss; together they are a gunshot.
    {
      const o1 = this._osc('sawtooth', 232 * p);
      const o2 = this._osc('sawtooth', 197 * p);
      const o3 = this._osc('square', 118 * p);
      this._sweep(o1.frequency, when, 246 * p, 63 * p, 0.085);
      this._sweep(o2.frequency, when, 204 * p, 52 * p, 0.105);
      this._sweep(o3.frequency, when, 126 * p, 41 * p, 0.120);
      const mix = this._gain(0.62);
      const sq = this._gain(0.28);
      const nz = this._noise('pink', when, 0.26, 0.9 + r() * 0.25);
      const nzG = this._gain(0.75);
      const lp = this._bq('lowpass', 2400 * p * fj, supp ? 1.6 : 3.6);
      this._sweep(lp.frequency, when, clamp(3100 * p * fj, 200, this._maxLP), (supp ? 300 : 430) * p, 0.135);
      const g = this._gain(0);
      this._envAD(g.gain, when, (supp ? 0.62 : 0.95) * lvl, 0.0022, supp ? 0.10 : 0.155);
      o1.connect(mix); o2.connect(mix); o3.connect(sq); sq.connect(mix);
      nz.connect(nzG); nzG.connect(mix);
      mix.connect(lp); lp.connect(g); g.connect(dest);
      o1.start(when); o2.start(when); o3.start(when);
      o1.stop(when + 0.34); o2.stop(when + 0.34); o3.stop(when + 0.34);
    }

    // ---- 3. sub thump: the punch you feel rather than hear -----------------
    if (!supp) {
      const s = this._osc('sine', 94 * p);
      this._sweep(s.frequency, when, 102 * p, 36 * p, 0.13);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.60 * lvl, 0.004, 0.20);
      s.connect(g); g.connect(dest);
      s.start(when); s.stop(when + 0.46);
    }

    // ---- 4. mechanical action: bolt cycling, tiny and bright ---------------
    {
      const t2 = when + 0.006 + r() * 0.004;
      const n = this._noise('grain', t2, 0.045, 1.1 + r() * 0.5);
      const bp = this._bq('bandpass', 3000 + r() * 1400, 3.8);
      const g = this._gain(0);
      this._envAD(g.gain, t2, 0.42 * lvl, 0.0006, 0.030);
      n.connect(bp); bp.connect(g); g.connect(dest);

      const t3 = when + 0.030 + r() * 0.012;
      const n2 = this._noise('white', t3, 0.030, 1.0);
      const bp2 = this._bq('bandpass', 1650 + r() * 700, 5.5);
      const g2 = this._gain(0);
      this._envAD(g2.gain, t3, 0.26 * lvl, 0.0006, 0.022);
      n2.connect(bp2); bp2.connect(g2); g2.connect(dest);
    }

    // ---- 5. local tail: feeds the convolvers hard and gives the shot its
    //        room before the convolution even arrives.
    {
      const t2 = when + 0.018;
      const n = this._noise('pink', t2, 0.40, 0.95);
      const bp = this._bq('bandpass', 820 * p, 0.75);
      const g = this._gain(0);
      this._envAD(g.gain, t2, (supp ? 0.10 : 0.26) * lvl, 0.012, 0.34);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }

    this.stats.shots++;
    this._duck(supp ? 0.14 : 0.42, 0.11);

    // Player's own weapon ejects brass and, at range, is heard by nobody else.
    if (self) {
      const cam = this.g?.camera;
      const ex = (cam ? cam.position.x : 0), ey = (cam ? cam.position.y : 1.6), ez = (cam ? cam.position.z : 0);
      this.shellCasing(ex + 0.35, ey - 1.35, ez, 0.30 + r() * 0.22);
    }
  }

  // The weapon does not sit in exactly the same place twice; wander the self
  // chain's stereo placement a hair per shot so a burst has width.
  _selfPanJitter(r) {
    const node = this._selfPanNode;
    if (node) node.pan.value = (this._ads ? 0.04 : 0.16) + (r() - 0.5) * 0.08;
  }

  /**
   * A rifle at range: the ballistic crack arrives first (if the round passes
   * near you), the muzzle thump follows over the speed of sound. Both are
   * heavily lowpassed — distance is a lowpass filter with a delay line.
   */
  distantReport(x, y, z, opts) {
    if (!this.ready || !this.ctx) return;
    const cam = this.g?.camera;
    const px = cam ? cam.position.x : 0, py = cam ? cam.position.y : 1.6, pz = cam ? cam.position.z : 0;
    x = num(x, px + 80); y = num(y, py); z = num(z, pz - 80);
    const dx = x - px, dy = y - py, dz = z - pz;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(dist <= MAX_AUDIBLE)) return;

    const r = this._rnd;
    const o = opts || {};
    // Speed of sound over the real distance, plus any burst spacing the caller
    // wants folded in (so a 6-round burst arrives as a burst, not as one bang).
    const flight = dist / SPEED_OF_SOUND + num(o.extra, 0);
    const base = this._when(this.ctx.currentTime + flight);
    const lvl = num(o.gain, 1) * clamp(1.25 - dist / 300, 0.18, 1.1);
    const p = 1 + (r() - 0.5) * 0.09;

    // The thump: everything above ~700 Hz has been eaten by the air and the
    // buildings between you and the shooter.
    const s = this._slot(x, y, z, { send: 1.05, gain: 1, ref: 22, roll: 0.72, dur: 1.6, air: 1.35 });
    if (!s) return;
    const dest = s.input;

    {
      const n = this._noise('brown', base, 0.55, 0.85 + r() * 0.3);
      const lp = this._bq('lowpass', 520 * p, 1.1);
      this._sweep(lp.frequency, base, 760 * p, 165 * p, 0.16);
      const g = this._gain(0);
      this._envAD(g.gain, base, 1.15 * lvl, 0.010, 0.22);
      n.connect(lp); lp.connect(g); g.connect(dest);
    }
    {
      const sub = this._osc('sine', 74 * p);
      this._sweep(sub.frequency, base, 82 * p, 33 * p, 0.20);
      const g = this._gain(0);
      this._envAD(g.gain, base, 0.55 * lvl, 0.012, 0.26);
      sub.connect(g); g.connect(dest);
      sub.start(base); sub.stop(base + 0.6);
    }
    // slap-back off whatever is behind the shooter — sells the urban canyon
    {
      const t2 = base + 0.075 + r() * 0.085;
      const n = this._noise('pink', t2, 0.45, 0.9);
      const bp = this._bq('bandpass', 430 * p, 0.85);
      const g = this._gain(0);
      this._envAD(g.gain, t2, 0.42 * lvl, 0.020, 0.34);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }
    // the ballistic crack — arrives EARLIER than the muzzle thump because the
    // round is supersonic. Only if the shot is roughly toward the player.
    if (dist > 30 && (o.crack ?? true)) {
      const lead = clamp(dist * 0.00098, 0.01, 0.35);
      const tc = this._when(this.ctx.currentTime + Math.max(0.004, flight - lead));
      const n = this._noise('white', tc, 0.05, 1.0);
      const bp = this._bq('bandpass', 1500 + r() * 900, 1.5);
      const hp = this._bq('highpass', 700, 0.7);
      const g = this._gain(0);
      this._envAD(g.gain, tc, 0.55 * lvl, 0.0006, 0.035);
      n.connect(bp); bp.connect(hp); hp.connect(g); g.connect(dest);
    }
  }

  /* ================================================================ *
   *  IMPACTS
   * ================================================================ */

  impact(surface, x, y, z, power) {
    if (!this.ready || !this.ctx) return;
    const key = SURF_MAP[String(surface || 'concrete').toLowerCase()] || 'concrete';
    const d = IMPACT[key] || IMPACT.concrete;
    const r = this._rnd;
    const pw = clamp(num(power, 1), 0.15, 3);
    const when = this._when();
    const p = 0.88 + r() * 0.26;
    const lvl = d.g * IMPACT_LEVEL * pw * (0.85 + r() * 0.3);

    const s = this._slot(x, y, z, {
      send: d.wet, gain: 1, ref: 2.4, roll: 1.5, dur: 0.7,
    });
    if (!s) return;
    const dest = s.input;

    // transient — the ricochet snap
    {
      const n = this._noise(d.grit > 0.6 ? 'grain' : 'white', when, 0.09, 0.9 + r() * 0.5);
      const bp = this._bq('bandpass', d.tf * p, d.tq);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.95 * lvl, 0.0006, d.td);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }
    // body — material displacement
    {
      const n = this._noise('pink', when + 0.001, 0.25, p);
      const lp = this._bq('lowpass', d.bf * p, 1.6);
      this._sweep(lp.frequency, when, d.bf * p, d.be * p, d.bd * 1.3);
      const g = this._gain(0);
      this._envAD(g.gain, when + 0.001, 0.75 * lvl, 0.0015, d.bd);
      n.connect(lp); lp.connect(g); g.connect(dest);
    }
    // sub — the wall taking the round
    if (d.sub > 0) {
      const o = this._osc('sine', d.sub * p);
      this._sweep(o.frequency, when, d.sub * p * 1.25, d.sub * p * 0.55, 0.10);
      const g = this._gain(0);
      this._envAD(g.gain, when, d.subG * lvl, 0.003, 0.13);
      o.connect(g); g.connect(dest);
      o.start(when); o.stop(when + 0.3);
    }
    // metallic / vitreous ring — three inharmonic partials, the fingerprint of
    // struck metal and shattering glass
    if (d.ring > 0) {
      const partials = key === 'glass' ? 4 : 3;
      for (let i = 0; i < partials; i++) {
        const f = d.rf * p * (1 + i * (0.71 + r() * 0.55));
        if (f > this._maxLP) break;
        const o = this._osc(i === 0 ? 'triangle' : 'sine', f);
        const g = this._gain(0);
        const dec = (key === 'glass' ? 0.16 : 0.30) / (1 + i * 0.55);
        this._envAD(g.gain, when + i * 0.002, d.ring * lvl * (0.9 / (1 + i * 0.85)), 0.0012, dec);
        o.connect(g); g.connect(dest);
        o.start(when); o.stop(when + dec * 4 + 0.05);
      }
    }
    // debris — spall, shards, gravel skittering away
    if (d.grit > 0.2) {
      const t2 = when + 0.02 + r() * 0.03;
      const n = this._noise('grain', t2, 0.32, 0.7 + r() * 0.9);
      const hp = this._bq('highpass', key === 'sand' ? 700 : 2100, 0.7);
      const g = this._gain(0);
      this._envAD(g.gain, t2, 0.30 * d.grit * lvl, 0.008, 0.26);
      n.connect(hp); hp.connect(g); g.connect(dest);
    }
  }

  /* ================================================================ *
   *  FOOTSTEPS
   * ================================================================ */

  /**
   * @param {string} surface
   * @param {number} speed  m/s — drives level, brightness and scuff length
   * @param self  true = the player's own boots (head-relative, alternating ears)
   */
  footstep(surface, speed, x, y, z, self) {
    if (!this.ready || !this.ctx) return;
    const key = SURF_MAP[String(surface || 'concrete').toLowerCase()] || 'concrete';
    const d = FOOT[key] || FOOT.concrete;
    const r = this._rnd;
    const sp = clamp(num(speed, 3), 0.4, 8);
    const hard = clamp(sp / 6.2, 0.25, 1.25);
    const when = this._when();
    const p = 0.90 + r() * 0.22;                     // per-step timbre variance
    const lvl = d.g * FOOT_LEVEL * (0.42 + hard * 0.75) * (0.86 + r() * 0.28);

    let dest;
    if (self) {
      this._stepFoot ^= 1;
      // Head-relative with a small L/R alternation: your own footsteps should
      // not be HRTF-panned to a point in the world behind you.
      const pan = this._stepFoot ? 0.22 : -0.22;
      const node = this._stepPanNode || (this._stepPanNode = (() => {
        const sp2 = this.ctx.createStereoPanner();
        sp2.connect(this.localIn);
        return sp2;
      })());
      node.pan.value = pan + (r() - 0.5) * 0.06;
      dest = node;
    } else {
      const s = this._slot(x ?? 0, y ?? 0.1, z ?? 0, {
        send: d.wet, gain: 1, ref: 2.0, roll: 1.9, dur: 0.5,
      });
      if (!s) return;
      dest = s.input;
    }
    this.stats.steps++;

    // heel strike
    {
      const n = this._noise(key === 'sand' ? 'pink' : 'white', when, 0.13, 0.85 + r() * 0.35);
      const lp = this._bq('lowpass', d.hf * p, d.hq + 0.4);
      this._sweep(lp.frequency, when, d.hf * p * 1.5, d.hf * p * 0.55, d.hd * 2.2);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.9 * lvl, 0.0012, d.hd);
      n.connect(lp); lp.connect(g); g.connect(dest);
    }
    // body thud
    if (d.sub > 0) {
      const o = this._osc('sine', d.sub * p);
      this._sweep(o.frequency, when, d.sub * p * 1.2, d.sub * p * 0.6, 0.075);
      const g = this._gain(0);
      this._envAD(g.gain, when, d.subG * lvl * hard, 0.004, 0.085);
      o.connect(g); g.connect(dest);
      o.start(when); o.stop(when + 0.22);
    }
    // toe scuff / grit — offset a few ms so the step has a front and a back
    {
      const t2 = when + 0.016 + r() * 0.020;
      const n = this._noise(d.scuff > 0.4 ? 'grain' : 'white', t2, 0.20, 0.8 + r() * 0.6);
      const bp = this._bq('bandpass', d.sf * p, d.sq);
      const g = this._gain(0);
      this._envAD(g.gain, t2, 0.38 * d.scuff * lvl * (0.6 + hard * 0.6), 0.004, d.sd);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }
    // gear rattle on the player only — sling swivel, mag pouches
    if (self && r() < 0.55) {
      const t3 = when + 0.03 + r() * 0.05;
      const n = this._noise('grain', t3, 0.10, 1.6 + r() * 0.8);
      const bp = this._bq('bandpass', 4200 + r() * 2200, 2.6);
      const g = this._gain(0);
      this._envAD(g.gain, t3, 0.09 * lvl, 0.003, 0.06);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }
  }

  /* ================================================================ *
   *  MECHANICS — brass, magazines, bolt
   * ================================================================ */

  /** Brass hitting the ground: two or three bright, inharmonic tinkles. */
  shellCasing(x, y, z, delay) {
    if (!this.ready || !this.ctx) return;
    const r = this._rnd;
    const t0 = this._when(this.ctx.currentTime + clamp(num(delay, 0.35), 0, 4));
    const s = this._slot(x, y, z, { send: 0.55, gain: 1, ref: 1.6, roll: 2.4, dur: 1.4 });
    if (!s) return;
    const dest = s.input;

    const bounces = 2 + ((r() * 2.6) | 0);
    let t = t0, amp = 1;
    for (let i = 0; i < bounces; i++) {
      const f = 2600 + r() * 3200;
      // metallic body: two inharmonic partials plus a noise tick
      for (let k = 0; k < 2; k++) {
        const o = this._osc(k ? 'sine' : 'triangle', clamp(f * (1 + k * 1.63), 60, this._maxLP));
        const g = this._gain(0);
        this._envAD(g.gain, t, 0.16 * amp / (1 + k), 0.0008, 0.055 + r() * 0.05);
        o.connect(g); g.connect(dest);
        o.start(t); o.stop(t + 0.30);
      }
      const n = this._noise('grain', t, 0.05, 1.5 + r());
      const bp = this._bq('bandpass', 5200 + r() * 3000, 2.2);
      const g2 = this._gain(0);
      this._envAD(g2.gain, t, 0.12 * amp, 0.0006, 0.030);
      n.connect(bp); bp.connect(g2); g2.connect(dest);

      t += 0.055 + r() * 0.085;
      amp *= 0.52 + r() * 0.16;
    }
  }

  /** A steel magazine dropped on the deck. */
  magDrop(x, y, z, surface) {
    if (!this.ready || !this.ctx) return;
    const r = this._rnd;
    const key = SURF_MAP[String(surface || 'concrete').toLowerCase()] || 'concrete';
    const soft = (key === 'sand' || key === 'soft');
    const t0 = this._when(this.ctx.currentTime + 0.34 + r() * 0.10);
    const s = this._slot(x, y, z, { send: 0.6, gain: 1, ref: 1.8, roll: 2.0, dur: 1.2 });
    if (!s) return;
    const dest = s.input;

    let t = t0, amp = 1;
    const hits = soft ? 1 : 2 + ((r() * 2) | 0);
    for (let i = 0; i < hits; i++) {
      const n = this._noise(soft ? 'pink' : 'white', t, 0.16, 0.8 + r() * 0.4);
      const lp = this._bq('lowpass', soft ? 900 : 3600, 1.2);
      const g = this._gain(0);
      this._envAD(g.gain, t, 0.55 * amp, 0.001, soft ? 0.10 : 0.065);
      n.connect(lp); lp.connect(g); g.connect(dest);

      if (!soft) {
        const o = this._osc('triangle', 320 + r() * 260);
        const g2 = this._gain(0);
        this._envAD(g2.gain, t, 0.22 * amp, 0.001, 0.11);
        o.connect(g2); g2.connect(dest);
        o.start(t); o.stop(t + 0.4);
        const o2 = this._osc('sine', 92 + r() * 40);
        const g3 = this._gain(0);
        this._envAD(g3.gain, t, 0.26 * amp, 0.003, 0.09);
        o2.connect(g3); g3.connect(dest);
        o2.start(t); o2.stop(t + 0.3);
      }
      t += 0.075 + r() * 0.09;
      amp *= 0.48;
    }
  }

  /**
   * Bolt / charging-handle mechanics. Kinds:
   *   'magout'  release catch + magazine sliding free
   *   'magin'   fresh magazine seated, hard
   *   'back'    charging handle drawn to the rear (scrape + clack)
   *   'release' bolt slamming home
   *   'click'   dry fire on an empty chamber
   *   'select'  fire selector detent
   */
  bolt(kind) {
    if (!this.ready || !this.ctx) return;
    const r = this._rnd;
    const dest = this.localIn;
    const when = this._when();

    const clack = (t, f, q, amp, dec, sub) => {
      const n = this._noise('white', t, 0.06, 0.9 + r() * 0.3);
      const bp = this._bq('bandpass', f * (0.92 + r() * 0.16), q);
      const g = this._gain(0);
      this._envAD(g.gain, t, amp, 0.0005, dec);
      n.connect(bp); bp.connect(g); g.connect(dest);
      if (sub) {
        const o = this._osc('triangle', sub * (0.94 + r() * 0.12));
        const g2 = this._gain(0);
        this._envAD(g2.gain, t, amp * 0.55, 0.0012, dec * 1.7);
        o.connect(g2); g2.connect(dest);
        o.start(t); o.stop(t + dec * 6 + 0.05);
      }
    };
    // sliding steel-on-steel: noise through a bandpass whose centre sweeps
    const scrape = (t, dur, f0, f1, amp) => {
      const n = this._noise('white', t, dur, 0.9 + r() * 0.2);
      const bp = this._bq('bandpass', f0, 2.6);
      this._sweep(bp.frequency, t, f0, f1, dur);
      const g = this._gain(0);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.linearRampToValueAtTime(amp, t + dur * 0.28);
      g.gain.setTargetAtTime(0, t + dur * 0.62, dur * 0.22);
      n.connect(bp); bp.connect(g); g.connect(dest);
    };

    switch (kind) {
      case 'magout':
        clack(when, 3400, 5.0, 0.34, 0.020, 900);
        scrape(when + 0.045, 0.115, 1900, 780, 0.13);
        break;
      case 'magin':
        scrape(when, 0.085, 900, 1700, 0.11);
        clack(when + 0.085, 1500, 3.0, 0.55, 0.055, 240);
        clack(when + 0.108, 4200, 6.0, 0.24, 0.016, 0);
        break;
      case 'back':
        scrape(when, 0.135, 1200, 2800, 0.20);
        clack(when + 0.135, 2600, 4.2, 0.42, 0.028, 520);
        break;
      case 'release':
        scrape(when, 0.045, 2600, 1300, 0.16);
        clack(when + 0.042, 1800, 3.2, 0.72, 0.045, 190);
        clack(when + 0.056, 5200, 6.5, 0.26, 0.014, 0);
        break;
      case 'select':
        clack(when, 5200, 8.0, 0.18, 0.010, 0);
        break;
      case 'click':
      default:
        clack(when, 2800, 6.0, 0.30, 0.014, 640);
        break;
    }
  }

  /** Cloth / kit movement — ADS transitions, sprint starts, reload handling. */
  cloth(amount) {
    if (!this.ready || !this.ctx) return;
    const r = this._rnd;
    const when = this._when();
    const a = clamp(num(amount, 1), 0.1, 2);
    const n = this._noise('white', when, 0.24, 0.6 + r() * 0.4);
    const bp = this._bq('bandpass', 2400 + r() * 1600, 0.85);
    const hp = this._bq('highpass', 900, 0.7);
    const g = this._gain(0);
    g.gain.setValueAtTime(0.0001, when);
    g.gain.linearRampToValueAtTime(0.075 * a, when + 0.035);
    g.gain.setTargetAtTime(0, when + 0.06, 0.05);
    n.connect(bp); bp.connect(hp); hp.connect(g); g.connect(this.localIn);
  }

  /* ================================================================ *
   *  VOICES — radio chatter
   * ================================================================ */

  /**
   * Enemy comms. Not words — filtered-noise syllables with a formant contour,
   * squashed through a squelchy narrowband radio chain. Reads as human speech
   * at distance without ever being intelligible, which is exactly what you want.
   */
  chatter(x, y, z, kind) {
    if (!this.ready || !this.ctx) return;
    const ctx = this.ctx, r = this._rnd;
    const when = this._when(ctx.currentTime + r() * 0.10);
    const s = this._slot(x ?? 0, y ?? 1.6, z ?? 0, {
      send: 0.42, gain: 1, ref: 5.0, roll: 1.35, dur: 1.6, air: 0.7,
    });
    if (!s) return;

    // --- radio chain: narrowband + soft clip + a little presence bump --------
    const hp = this._bq('highpass', 420, 0.8);
    const lp = this._bq('lowpass', 2750, 0.9);
    const pres = this._bq('peaking', 1700, 1.4);
    pres.gain.value = 6.5;
    const shaper = ctx.createWaveShaper();
    shaper.curve = this._radioCurve;
    shaper.oversample = '2x';
    const outG = this._gain(0.42);
    hp.connect(lp); lp.connect(pres); pres.connect(shaper); shaper.connect(outG);
    outG.connect(s.input);

    // --- squelch open --------------------------------------------------------
    {
      const n = this._noise('white', when, 0.05, 1.0);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.22, 0.0006, 0.030);
      n.connect(g); g.connect(hp);
    }

    // --- syllables -----------------------------------------------------------
    const urgency = kind === 'contact' ? 1.25 : kind === 'suppress' ? 1.1 : kind === 'hit' ? 1.4 : 0.85;
    const count = 2 + ((r() * (kind === 'radio' ? 5 : 3.2)) | 0);
    let t = when + 0.055;
    const base = 105 + r() * 55;              // "vocal" fundamental region
    for (let i = 0; i < count; i++) {
      const dur = (0.075 + r() * 0.13) / urgency;
      const open = 0.35 + r() * 0.65;         // vowel openness -> formant pair

      // buzz: pulse-ish source, pitch drifting through the syllable
      const o = this._osc('sawtooth', base);
      const bend = 0.86 + r() * 0.30;
      this._sweep(o.frequency, t, base * (0.95 + r() * 0.2), base * bend, dur);
      const oG = this._gain(0);
      this._envAD(oG.gain, t, 0.30 * urgency, 0.012, dur * 0.9);

      // two formants define the vowel
      const f1 = this._bq('bandpass', lerp(320, 760, open), 7.5);
      const f2 = this._bq('bandpass', lerp(2100, 1180, open), 9.0);
      const f1g = this._gain(1.0), f2g = this._gain(0.55);
      o.connect(oG); oG.connect(f1); oG.connect(f2);
      f1.connect(f1g); f2.connect(f2g);
      f1g.connect(hp); f2g.connect(hp);
      o.start(t); o.stop(t + dur + 0.08);

      // fricative breath on top so consonants exist
      if (r() < 0.62) {
        const tf = t + dur * (0.55 + r() * 0.4);
        const n = this._noise('white', tf, 0.07, 1.0);
        const bp = this._bq('bandpass', 2600 + r() * 2600, 1.4);
        const g = this._gain(0);
        this._envAD(g.gain, tf, 0.11 * urgency, 0.004, 0.045);
        n.connect(bp); bp.connect(g); g.connect(hp);
      }

      t += dur + 0.018 + r() * 0.055;
    }

    // --- squelch close -------------------------------------------------------
    {
      const n = this._noise('white', t + 0.02, 0.06, 1.0);
      const g = this._gain(0);
      this._envAD(g.gain, t + 0.02, 0.26, 0.0008, 0.040);
      n.connect(g); g.connect(hp);
    }
    s.freeAt = t + 0.4;
  }

  /* ================================================================ *
   *  EXPLOSION / PLAYER STATE
   * ================================================================ */

  explosion(x, y, z, radius, power) {
    if (!this.ready || !this.ctx) return;
    const r = this._rnd;
    const cam = this.g?.camera;
    const px = cam ? cam.position.x : 0, py = cam ? cam.position.y : 1.6, pz = cam ? cam.position.z : 0;
    x = num(x, px); y = num(y, py); z = num(z, pz);
    const dx = x - px, dy = y - py, dz = z - pz;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(dist <= MAX_AUDIBLE)) return;

    const rad = clamp(num(radius, 6), 0.5, 200);
    const raw = clamp(num(power, 1), 0.2, 4);
    const pw = raw * EXP_LEVEL;
    const flight = dist / SPEED_OF_SOUND;
    const when = this._when(this.ctx.currentTime + flight);
    const p = 0.9 + r() * 0.22;

    const s = this._slot(x, y, z, { send: 1.2, gain: 1, ref: 12, roll: 0.85, dur: 3.2, air: 0.9 });
    if (!s) return;
    const dest = s.input;

    // 1 — the sub: a body blow
    {
      const o = this._osc('sine', 68 * p);
      this._sweep(o.frequency, when, 74 * p, 21 * p, 0.85);
      const g = this._gain(0);
      this._envAD(g.gain, when, 1.5 * pw, 0.006, 0.85);
      o.connect(g); g.connect(dest);
      o.start(when); o.stop(when + 2.0);
    }
    // 2 — the body: brown noise falling through a collapsing lowpass
    {
      const n = this._noise('brown', when, 1.6, 0.85 + r() * 0.3);
      const lp = this._bq('lowpass', 1100 * p, 1.3);
      this._sweep(lp.frequency, when, 1500 * p, 85 * p, 1.05);
      const g = this._gain(0);
      this._envAD(g.gain, when, 1.35 * pw, 0.008, 0.75);
      n.connect(lp); lp.connect(g); g.connect(dest);
    }
    // 3 — the crack: the detonation front, gone in 60 ms
    {
      const n = this._noise('white', when, 0.14, 1.0);
      const hp = this._bq('highpass', 900, 0.7);
      const lp = this._bq('lowpass', 9000, 0.8);
      this._sweep(lp.frequency, when, this._maxLP * 0.85, 1300, 0.09);
      const g = this._gain(0);
      this._envAD(g.gain, when, 1.1 * pw, 0.0008, 0.075);
      n.connect(hp); hp.connect(lp); lp.connect(g); g.connect(dest);
    }
    // 4 — debris raining down for a second and a half
    {
      const t2 = when + 0.10;
      const n = this._noise('grain', t2, 1.5, 0.6 + r() * 0.7);
      const hp = this._bq('highpass', 700, 0.6);
      const g = this._gain(0);
      g.gain.setValueAtTime(0.0001, t2);
      g.gain.linearRampToValueAtTime(0.34 * pw, t2 + 0.06);
      g.gain.setTargetAtTime(0, t2 + 0.12, 0.42);
      n.connect(hp); hp.connect(g); g.connect(dest);
    }
    // 5 — the tail, fed almost entirely to the convolvers
    {
      const t2 = when + 0.05;
      const n = this._noise('pink', t2, 1.8, 0.9);
      const bp = this._bq('bandpass', 420 * p, 0.6);
      const g = this._gain(0);
      g.gain.setValueAtTime(0.0001, t2);
      g.gain.linearRampToValueAtTime(0.42 * pw, t2 + 0.05);
      g.gain.setTargetAtTime(0, t2 + 0.10, 0.55);
      n.connect(bp); bp.connect(g); g.connect(dest);
    }

    this._duck(0.75, 0.55);

    // Close blasts wreck your hearing.
    const near = clamp(1 - dist / Math.max(4, rad * 2.6), 0, 1) * raw;
    if (near > 0.05) {
      this.concuss(clamp(near * 1.25, 0, 1));
      this.tinnitus(clamp(near * 1.1, 0, 1), 5 + near * 9);
    }
  }

  /** Muffle everything, as if your eardrums just took a pressure spike. */
  concuss(strength) {
    this._concuss = clamp(Math.max(this._concuss, num(strength, 0.6)), 0, 1);
  }

  /** Start (or reinforce) the ringing. */
  tinnitus(strength, seconds) {
    if (!this.ready || !this.ctx) return;
    const ctx = this.ctx;
    const now = ctx.currentTime;
    const s = clamp(num(strength, 0.6), 0, 1);
    const dur = clamp(num(seconds, 8), 0.5, 40);

    if (!this._tinnitusNodes) {
      const o1 = this._osc('sine', 4180);
      const o2 = this._osc('sine', 6420);
      const o3 = this._osc('sine', 2960);
      const g1 = this._gain(1.0), g2 = this._gain(0.42), g3 = this._gain(0.28);
      // a slow beat between the partials keeps it alive rather than sterile
      const lfo = this._osc('sine', 0.23);
      const lfoG = this._gain(0.09);
      lfo.connect(lfoG); lfoG.connect(g2.gain);
      o1.connect(g1); o2.connect(g2); o3.connect(g3);
      g1.connect(this._tinBus); g2.connect(this._tinBus); g3.connect(this._tinBus);
      o1.start(now); o2.start(now); o3.start(now); lfo.start(now);
      this._tinnitusNodes = { o1, o2, o3, lfo };
    }

    const g = this._tinBus.gain;
    const peak = 0.055 * s;
    g.cancelScheduledValues(now);
    g.setValueAtTime(Math.max(g.value, 0.0001), now);
    g.linearRampToValueAtTime(Math.max(peak, g.value), now + 0.04);
    g.setTargetAtTime(0.0, now + 0.25, dur * 0.30);
    this._tinnitusUntil = now + dur;
  }

  /* ================================================================ *
   *  AMBIENCE BED
   * ================================================================ */

  _startAmbience() {
    const ctx = this.ctx, r = this._rnd;
    const now = ctx.currentTime;
    const dest = this.ambIn;

    // --- wind: two decorrelated band-limited layers, each with its own slow
    //     LFO on the filter so the bed never sits still.
    for (let i = 0; i < 2; i++) {
      const src = ctx.createBufferSource();
      src.buffer = this._buf.wind;
      src.loop = true;
      src.playbackRate.value = 0.82 + i * 0.27;
      const bp = this._bq('bandpass', i ? 620 : 310, 0.55);
      const lfo = this._osc('sine', 0.031 + i * 0.019);
      const lfoG = this._gain(i ? 260 : 130);
      lfo.connect(lfoG); lfoG.connect(bp.frequency);
      const g = this._gain(i ? 0.075 : 0.115);
      const pan = ctx.createStereoPanner();
      pan.pan.value = i ? 0.45 : -0.42;
      // a second, slower LFO breathes the level — gusts
      const lfo2 = this._osc('sine', 0.017 + i * 0.011);
      const lfo2G = this._gain(i ? 0.038 : 0.058);
      lfo2.connect(lfo2G); lfo2G.connect(g.gain);
      src.connect(bp); bp.connect(g); g.connect(pan); pan.connect(dest);
      src.start(now + i * 0.13);
      lfo.start(now); lfo2.start(now);
    }

    // --- distant city / generator rumble ------------------------------------
    {
      const src = ctx.createBufferSource();
      src.buffer = this._buf.rumble;
      src.loop = true;
      src.playbackRate.value = 0.93;
      const lp = this._bq('lowpass', 130, 0.7);
      const g = this._gain(0.155);
      src.connect(lp); lp.connect(g); g.connect(dest);
      src.start(now);
    }

    // --- a faint 50 Hz-ish hum from somewhere, plus its fifth. Cities hum. ---
    {
      const o1 = this._osc('sine', 51.3);
      const o2 = this._osc('sine', 77.4);
      const g1 = this._gain(0.008), g2 = this._gain(0.0035);
      const wob = this._osc('sine', 0.09);
      const wobG = this._gain(0.55);
      wob.connect(wobG); wobG.connect(o1.frequency);
      o1.connect(g1); o2.connect(g2);
      g1.connect(dest); g2.connect(dest);
      o1.start(now); o2.start(now); wob.start(now);
    }

    // --- high air / insect shimmer, barely there but it fills the top end ----
    {
      const src = ctx.createBufferSource();
      src.buffer = this._buf.wind;
      src.loop = true;
      src.playbackRate.value = 1.9;
      const bp = this._bq('bandpass', 5200, 1.4);
      const g = this._gain(0.018);
      const lfo = this._osc('sine', 0.043);
      const lfoG = this._gain(0.010);
      lfo.connect(lfoG); lfoG.connect(g.gain);
      src.connect(bp); bp.connect(g); g.connect(dest);
      src.start(now + 0.4); lfo.start(now);
    }

    this._ambStarted = true;
  }

  /** Occasional far-off contacts: the war is bigger than your street. */
  _ambientEvent() {
    const r = this._rnd;
    const cam = this.g?.camera;
    const px = cam ? cam.position.x : 0, pz = cam ? cam.position.z : 0;
    const ang = r() * Math.PI * 2;
    const dist = 90 + r() * 170;
    const x = px + Math.cos(ang) * dist;
    const z = pz + Math.sin(ang) * dist;
    const y = 1.5 + r() * 8;

    const roll = r();
    if (roll < 0.62) {
      // a distant burst of automatic fire
      const shots = 2 + ((r() * 6) | 0);
      const rpm = 600 + r() * 350;
      const gap = 60 / rpm;
      for (let i = 0; i < shots; i++) {
        this._queueDistant(x, z, y, i * gap, 0.55 + r() * 0.35);
      }
    } else if (roll < 0.85) {
      // a single, deliberate rifle shot
      this._queueDistant(x, z, y, 0, 0.7 + r() * 0.4);
    } else {
      // a very distant, very low thump — armour, a door, a detonation
      const s = this._slot(x, y, z, { send: 1.2, gain: 1, ref: 30, roll: 0.6, dur: 2.2, air: 1.5 });
      if (!s) return;
      const when = this._when(this.ctx.currentTime + dist / SPEED_OF_SOUND);
      const n = this._noise('brown', when, 1.0, 0.7);
      const lp = this._bq('lowpass', 190, 1.0);
      this._sweep(lp.frequency, when, 260, 70, 0.5);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.75, 0.02, 0.55);
      n.connect(lp); lp.connect(g); g.connect(s.input);
    }
  }

  _queueDistant(x, z, y, offset, gain) {
    // distantReport applies the speed-of-sound delay itself; `extra` is the
    // spacing inside the burst. Everything is scheduled on the audio clock —
    // no setTimeout, so a stalled main thread never smears a burst.
    this.distantReport(x, y, z, { gain, crack: false, extra: offset });
  }

  /* ================================================================ *
   *  BUS WIRING
   * ================================================================ */

  _bindBus() {
    const bus = this.g?.bus;
    if (!bus) return;
    const on = (evt, fn) => { const off = bus.on(evt, fn); if (off) this._unbind.push(off); };

    on('shot', (d) => {
      if (!this.ready) return;
      const cam = this.g?.camera;
      const o = d?.origin;
      let selfShot = true, x = 0, y = 1.6, z = 0;
      if (o && cam) {
        x = o.x ?? 0; y = o.y ?? 1.6; z = o.z ?? 0;
        const dx = x - cam.position.x, dy = y - cam.position.y, dz = z - cam.position.z;
        const dd = dx * dx + dy * dy + dz * dz;
        selfShot = dd < 4.0;      // inside 2 m -> it is the player's weapon
      } else if (o) {
        x = o.x ?? 0; y = o.y ?? 1.6; z = o.z ?? 0;
        selfShot = false;
      }
      const supp = !!(d?.weapon && (d.weapon.suppressed || d.weapon === 'suppressed'));
      if (selfShot) {
        this.gunshot({ self: true, suppressed: supp });
      } else {
        const cx = cam ? cam.position.x : 0, cy = cam ? cam.position.y : 1.6, cz = cam ? cam.position.z : 0;
        const dx = x - cx, dy = y - cy, dz = z - cz;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > 26) this.distantReport(x, y, z, { gain: 1, crack: true });
        else this.gunshot({ x, y, z, self: false, suppressed: supp });
      }
    });

    on('impact', (d) => {
      if (!this.ready || !d) return;
      const p = d.point;
      const surf = d.surface || d.material?.userData?.bsImpact ||
        this.g?.materials?.surfaceFor?.(d.material) || 'concrete';
      this.impact(surf, p?.x ?? 0, p?.y ?? 1, p?.z ?? 0, d.power ?? 1);
    });

    on('hit', (d) => {
      if (!this.ready || !d) return;
      const p = d.point;
      this.impact('flesh', p?.x ?? 0, p?.y ?? 1.2, p?.z ?? 0, d.headshot ? 1.5 : 1.0);
      this._hitmarker(!!d.headshot);
    });

    on('kill', (d) => {
      if (!this.ready) return;
      this._hitmarker(true, true);
      if (this._rnd() < 0.65 && d?.enemy?.position) {
        const p = d.enemy.position;
        this.chatter(p.x, (p.y ?? 1) + 1.2, p.z, 'hit');
      }
    });

    on('damage', (d) => {
      if (!this.ready) return;
      this._playerHurt(clamp(num(d?.amount, 20) / 40, 0.2, 1.4));
    });

    on('reload', (d) => {
      if (!this.ready) return;
      const phase = d?.phase;
      const cam = this.g?.camera;
      switch (phase) {
        case 'start': this.cloth(1.1); this.bolt('select'); break;
        case 'magout': {
          this.bolt('magout');
          if (cam) this.magDrop(cam.position.x + 0.25, cam.position.y - 1.4, cam.position.z, this._lastGroundSurface || 'concrete');
          break;
        }
        case 'magin': this.cloth(0.8); this.bolt('magin'); break;
        case 'end': this.bolt('release'); this.cloth(0.5); break;
        default: this.cloth(0.7); break;
      }
    });

    on('ads', (d) => {
      if (!this.ready) return;
      this._ads = !!d?.active;
      this.cloth(this._ads ? 0.75 : 0.55);
      // ADS pulls the weapon to the eye — recentre the self chain
      const p = this._selfPanNode;
      if (p) p.pan.setTargetAtTime(this._ads ? 0.04 : 0.16, this.ctx.currentTime, 0.06);
    });

    on('footstep', (d) => {
      this._autoFootsteps = false;   // the Controller owns the gait from here on
      if (!this.ready || !d) return;
      const surf = d.surface || 'concrete';
      this._lastGroundSurface = surf;
      const cam = this.g?.camera;
      const p = d.point || d.position;
      if (p && cam) {
        const dx = p.x - cam.position.x, dz = p.z - cam.position.z;
        if (dx * dx + dz * dz > 2.25) { this.footstep(surf, d.speed ?? 3, p.x, p.y ?? 0, p.z, false); return; }
      }
      this.footstep(surf, d.speed ?? this._playerSpeedSm, 0, 0, 0, true);
    });

    on('explosion', (d) => {
      if (!this.ready || !d) return;
      const p = d.point;
      this.explosion(p?.x ?? 0, p?.y ?? 1, p?.z ?? 0, d.radius ?? 6, d.power ?? 1);
    });

    on('ammo', (d) => {
      if (!d) return;
      const prev = this._mag;
      this._mag = d.mag ?? this._mag;
      this._reserve = d.reserve ?? this._reserve;
      if (this.ready && prev > 0 && this._mag === 0) this.bolt('click');
    });

    on('health', (d) => {
      if (!d) return;
      this._hpMax = Math.max(1, num(d.max, this._hpMax));
      this._hp = clamp(num(d.hp, this._hpMax) / this._hpMax, 0, 1);
    });
  }

  /** Hit confirm — non-spatial, tiny, and the single most satisfying 20 ms in
   *  any shooter. Headshots and kills ring higher and a touch longer. */
  _hitmarker(head, kill) {
    if (!this.ready || !this.ctx) return;
    const when = this._when();
    const f = kill ? 1580 : head ? 1320 : 980;
    const o = this._osc('sine', f);
    const o2 = this._osc('triangle', f * 1.5);
    const g = this._gain(0), g2 = this._gain(0);
    this._envAD(g.gain, when, kill ? 0.13 : 0.085, 0.0008, kill ? 0.075 : 0.045);
    this._envAD(g2.gain, when, kill ? 0.055 : 0.030, 0.0008, 0.030);
    o.connect(g); o2.connect(g2);
    g.connect(this.localIn); g2.connect(this.localIn);
    o.start(when); o2.start(when);
    o.stop(when + 0.4); o2.stop(when + 0.3);
    if (kill) {
      const o3 = this._osc('sine', f * 2);
      const g3 = this._gain(0);
      this._envAD(g3.gain, when + 0.055, 0.075, 0.001, 0.075);
      o3.connect(g3); g3.connect(this.localIn);
      o3.start(when + 0.055); o3.stop(when + 0.4);
    }
  }

  /** Taking a round: the thud, the grunt, and a short duck of everything else. */
  _playerHurt(amount) {
    const r = this._rnd;
    const when = this._when();
    const dest = this.localIn;

    // impact on plate / body
    {
      const n = this._noise('white', when, 0.14, 0.8 + r() * 0.3);
      const lp = this._bq('lowpass', 1400, 1.4);
      this._sweep(lp.frequency, when, 2200, 300, 0.10);
      const g = this._gain(0);
      this._envAD(g.gain, when, 0.55 * amount, 0.001, 0.09);
      n.connect(lp); lp.connect(g); g.connect(dest);
      const o = this._osc('sine', 78);
      this._sweep(o.frequency, when, 86, 40, 0.12);
      const g2 = this._gain(0);
      this._envAD(g2.gain, when, 0.45 * amount, 0.004, 0.14);
      o.connect(g2); g2.connect(dest);
      o.start(when); o.stop(when + 0.35);
    }
    // grunt: a short vowel with a falling pitch
    {
      const t = when + 0.02 + r() * 0.03;
      const dur = 0.16 + r() * 0.12;
      const o = this._osc('sawtooth', 132);
      this._sweep(o.frequency, t, 148, 96, dur);
      const f1 = this._bq('bandpass', 620, 6.0);
      const f2 = this._bq('bandpass', 1180, 7.0);
      const f2g = this._gain(0.5);
      const g = this._gain(0);
      this._envAD(g.gain, t, 0.19 * amount, 0.02, dur);
      const nz = this._noise('white', t, dur, 1.0);
      const nzB = this._bq('bandpass', 1900, 1.2);
      const nzG = this._gain(0);
      this._envAD(nzG.gain, t, 0.05 * amount, 0.02, dur);
      o.connect(g); g.connect(f1); g.connect(f2); f2.connect(f2g);
      f1.connect(dest); f2g.connect(dest);
      nz.connect(nzB); nzB.connect(nzG); nzG.connect(dest);
      o.start(t); o.stop(t + dur + 0.15);
    }
    this._duck(0.30, 0.12);
    if (amount > 0.9) this.concuss(clamp((amount - 0.9) * 0.8, 0, 0.45));
  }

  /* ================================================================ *
   *  PER-FRAME
   * ================================================================ */

  update(dt, t) {
    // ---- state tracking that must work with or without a context ----------
    const g = this.g;
    const cam = g?.camera;
    if (cam && dt > 0) {
      const ctrl = g.controller;
      let sp = -1;
      const v = ctrl?.velocity || ctrl?.vel;
      if (v && typeof v.x === 'number') sp = Math.sqrt(v.x * v.x + v.z * v.z);
      else if (typeof ctrl?.speed === 'number') sp = ctrl.speed;
      if (sp < 0) {
        if (this._camHasPrev) {
          const dx = cam.position.x - this._camPrev.x, dz = cam.position.z - this._camPrev.z;
          sp = Math.sqrt(dx * dx + dz * dz) / dt;
          if (sp > 14) sp = 0;                     // teleport, not a sprint
        } else sp = 0;
      }
      this._camPrev.copy(cam.position);
      this._camHasPrev = true;
      sp = clamp(num(sp, 0), 0, 20);
      this._playerSpeed = sp;
      this._playerSpeedSm += (sp - this._playerSpeedSm) * clamp(dt * 8, 0, 1);

      const pc = g.config?.player;
      this._sprinting = ctrl?.sprinting === true ||
        (this._playerSpeedSm > (pc ? pc.walkSpeed * 1.22 : 4.2));
      this._grounded = ctrl?.grounded !== false;

      const targetEx = clamp(this._playerSpeedSm / (pc ? pc.sprintSpeed : 6.2), 0, 1);
      // exertion builds fast under load and recovers slowly — like lungs
      const k = targetEx > this._exertion ? 0.55 : 0.16;
      this._exertion += (targetEx - this._exertion) * clamp(dt * k, 0, 1);
    }

    if (!this.ctx) return;
    const ctx = this.ctx;
    this.stats.state = ctx.state;
    if (ctx.state !== 'running') return;

    this._updateListener(cam);
    this._updateConfig();
    this._updateSpace(dt);
    this._updateConcussion(dt);
    this._updateBreath();
    this._updateHeart();
    this._updateAutoGait(dt);

    // ambience events
    const now = ctx.currentTime;
    if (now >= this._nextAmbEvent) {
      this._nextAmbEvent = now + 7 + this._rnd() * 22;
      this._ambientEvent();
    }
    if (this._tinnitusUntil && now > this._tinnitusUntil + 2) {
      this._tinnitusUntil = 0;
      this._tinBus.gain.cancelScheduledValues(now);
      this._tinBus.gain.setTargetAtTime(0, now, 0.4);
    }

    // stats (no allocation — the object is reused)
    let busy = 0;
    for (let i = 0; i < this._slots.length; i++) if (this._slots[i].freeAt > now) busy++;
    this.stats.voices = busy;
    this.stats.probeHits = this._probeHits;
  }

  _updateListener(cam) {
    if (!cam) return;
    const L = this.ctx.listener;
    if (!L) return;
    const e = cam.matrixWorld.elements;
    const fx = -e[8], fy = -e[9], fz = -e[10];
    const ux = e[4], uy = e[5], uz = e[6];
    if (L.positionX) {
      L.positionX.value = cam.position.x;
      L.positionY.value = cam.position.y;
      L.positionZ.value = cam.position.z;
      L.forwardX.value = fx; L.forwardY.value = fy; L.forwardZ.value = fz;
      L.upX.value = ux; L.upY.value = uy; L.upZ.value = uz;
    } else {
      if (L.setPosition) L.setPosition(cam.position.x, cam.position.y, cam.position.z);
      if (L.setOrientation) L.setOrientation(fx, fy, fz, ux, uy, uz);
    }
  }

  _updateConfig() {
    const c = this.g?.config?.audio;
    if (!c) return;
    const m = clamp(num(c.master, this._volMaster), 0, 2);
    const s = clamp(num(c.sfx, this._volSfx), 0, 2);
    const mu = clamp(num(c.music, this._volMusic), 0, 2);
    if (m !== this._volMaster) {
      this._volMaster = m;
      if (!this.muted) this.master.gain.setTargetAtTime(m, this.ctx.currentTime, 0.03);
    }
    if (s !== this._volSfx) {
      this._volSfx = s;
      this.sfxOut.gain.setTargetAtTime(s, this.ctx.currentTime, 0.03);
    }
    if (mu !== this._volMusic) {
      this._volMusic = mu;
      this.ambOut.gain.setTargetAtTime(mu, this.ctx.currentTime, 0.05);
    }
  }

  /* ---------------------------------------------------------------- *
   *  Reverb space probe
   * ---------------------------------------------------------------- */

  _updateSpace(dt) {
    this._probeT -= dt;
    if (this._probeT <= 0) {
      this._probeT = 0.22;
      this._probe();
    }
    // smooth crossfade toward the target weights (unrolled — no iterator, no
    // allocation, this runs every frame)
    const now = this.ctx.currentTime;
    const w = this._spaceW, tg = this._spaceTarget;
    const k = clamp(dt * 2.6, 0, 1);
    const ni = w.interior + (tg.interior - w.interior) * k;
    const ns = w.street + (tg.street - w.street) * k;
    const na = w.alley + (tg.alley - w.alley) * k;
    const changed = Math.abs(ni - w.interior) > 0.002 ||
      Math.abs(ns - w.street) > 0.002 ||
      Math.abs(na - w.alley) > 0.002;
    w.interior = ni; w.street = ns; w.alley = na;
    // enclosed spaces want more send, the open street wants less (three multiplies,
    // so it is cheaper to keep it always correct than to gate it)
    this._spaceSend = 0.65 + w.interior * 0.75 + w.alley * 0.45;
    this.stats.space = (w.interior >= w.street && w.interior >= w.alley) ? 'interior'
      : (w.alley >= w.street) ? 'alley' : 'street';
    if (changed) {
      this._convGain.interior.gain.setTargetAtTime(w.interior * 0.62, now, 0.12);
      this._convGain.street.gain.setTargetAtTime(w.street * 0.50, now, 0.12);
      this._convGain.alley.gain.setTargetAtTime(w.alley * 0.58, now, 0.12);
    }
  }

  _probe() {
    if (this._spaceMode !== 'auto') return;
    const g = this.g;
    const cam = g?.camera;
    const cols = g?.registry?.colliders;
    const tg = this._spaceTarget;

    if (!cam || !cols || !cols.length) {
      tg.interior = 0; tg.street = 1; tg.alley = 0;
      return;
    }
    if (cols.length !== this._probeLen) {
      this._probeLen = cols.length;
      this._probeList.length = 0;
      for (let i = 0; i < cols.length; i++) {
        const o = cols[i];
        if (o && o.isObject3D) this._probeList.push(o);
      }
      // A short list is probably individual collider meshes and is cheap to
      // recurse; a long one is probably the whole level and is not.
      this._probeRecursive = this._probeList.length <= 64;
    }
    const list = this._probeList;
    if (!list.length) { tg.interior = 0; tg.street = 1; tg.alley = 0; return; }

    this._probeOrigin.copy(cam.position);
    const dirs = this._probeDirs;
    const hits = this._rayHits;
    const nearD = this._probeNear;
    const rec = !!this._probeRecursive;
    let close = 0, hitCount = 0, sumOpen = 0, ceiling = 0;
    const lat = 8;                       // first 8 dirs are horizontal

    for (let i = 0; i < dirs.length; i++) {
      this._ray.set(this._probeOrigin, dirs[i]);
      const far = (i === lat) ? 9 : 20;
      this._ray.far = far;
      hits.length = 0;
      let d = far;
      try {
        this._ray.intersectObjects(list, rec, hits);
        if (hits.length) d = hits[0].distance;
      } catch (_) { /* a collaborator's collider may not be raycastable */ }
      if (i < lat) {
        nearD[i] = d;
        if (d < far - 0.01) { hitCount++; if (d < 4.5) close++; }
        sumOpen += clamp(d / far, 0, 1);
      } else if (d < far - 0.1) {
        ceiling = clamp(1 - d / far, 0, 1);
      }
    }
    hits.length = 0;
    this._probeHits = hitCount;
    this._probeCeiling = ceiling;

    // 1 = standing in the open, 0 = boxed in on every side
    const openness = sumOpen / lat;
    const enclosure = clamp((1 - openness) * 0.7 + (close / lat) * 0.6, 0, 1);

    // Alley signature: two OPPOSITE walls close, the other axis running away.
    const axisA = Math.min(nearD[0], nearD[1]);  // +-X
    const axisB = Math.min(nearD[2], nearD[3]);  // +-Z
    const tight = Math.min(axisA, axisB);
    const wide = Math.max(axisA, axisB);
    const corridor = clamp((7 - tight) / 6, 0, 1) * clamp((wide - 8) / 10, 0, 1);

    // A roof over your head is the single strongest interior cue there is, so
    // ceiling drives this and enclosure only reinforces it. Requiring both
    // (the obvious formulation) leaves you in "street" under an open-sided
    // arcade, which sounds badly wrong.
    const interior = clamp(ceiling * (0.62 + 0.48 * enclosure), 0, 1);
    const alley = clamp(corridor * (1 - ceiling), 0, 1) * 1.15;
    // A little of the open street always bleeds in — the compound is outdoors.
    const street = Math.max(clamp(1 - interior * 1.45 - alley * 0.9, 0, 1), 0.10);

    const sum = interior + alley + street || 1;
    tg.interior = interior / sum;
    tg.alley = alley / sum;
    tg.street = street / sum;
  }

  /** Force a space (or return to probing). */
  setSpace(mode) {
    this._spaceMode = mode || 'auto';
    const tg = this._spaceTarget;
    if (mode === 'interior') { tg.interior = 1; tg.street = 0; tg.alley = 0; }
    else if (mode === 'street') { tg.interior = 0; tg.street = 1; tg.alley = 0; }
    else if (mode === 'alley') { tg.interior = 0; tg.street = 0.15; tg.alley = 0.85; }
    return this;
  }

  /* ---------------------------------------------------------------- *
   *  Player state audio
   * ---------------------------------------------------------------- */

  _updateConcussion(dt) {
    if (this._concuss > 0) {
      // ~4.5 s recovery, fast at first then lingering
      this._concuss = Math.max(0, this._concuss - dt * (0.18 + this._concuss * 0.30));
    }
    const c = this._concuss;
    if (Math.abs(c - this._concussApplied) < 0.006) return;
    this._concussApplied = c;
    const now = this.ctx.currentTime;
    const f = this._maxLP * Math.pow(340 / this._maxLP, c);
    this._muffle.frequency.setTargetAtTime(clamp(f, 200, this._maxLP), now, 0.08);
    this._muffleTilt.gain.setTargetAtTime(-16 * c, now, 0.08);
    // the world also gets quieter for a moment
    if (!this.muted) {
      this.master.gain.setTargetAtTime(this._volMaster * (1 - c * 0.32), now, 0.10);
    }
  }

  _updateBreath() {
    const now = this.ctx.currentTime;
    if (now < this._nextBreath) return;
    const r = this._rnd;
    const ex = this._exertion;
    const lowHp = clamp(1 - this._hp / 0.45, 0, 1);
    const drive = clamp(ex * 0.85 + lowHp * 0.5, 0, 1.2);

    // rate: ~3.6 s per cycle at rest, ~0.75 s flat out
    const period = lerp(1.85, 0.42, clamp(drive, 0, 1));
    this._nextBreath = now + period * (0.88 + r() * 0.24);

    const inhale = (this._breathPhase ^= 1) === 1;
    const level = (0.028 + drive * 0.115) * (inhale ? 1.0 : 0.82);
    if (level < 0.006) return;

    const dur = period * (inhale ? 0.55 : 0.68);
    const n = this._noise('white', this._when(), dur + 0.05, 0.85 + r() * 0.3);
    const bp = this._bq('bandpass', inhale ? (520 + drive * 260) : (380 + drive * 180), 0.9 + drive * 0.7);
    const hp = this._bq('highpass', 210, 0.7);
    const g = this._gain(0);
    const t0 = this._when();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.linearRampToValueAtTime(level, t0 + dur * (inhale ? 0.42 : 0.22));
    g.gain.setTargetAtTime(0, t0 + dur * (inhale ? 0.55 : 0.34), dur * 0.30);
    const pan = this.ctx.createStereoPanner();
    pan.pan.value = (r() - 0.5) * 0.12;
    n.connect(bp); bp.connect(hp); hp.connect(g); g.connect(pan); pan.connect(this.localIn);

    // hard exertion adds a voiced edge to the exhale
    if (!inhale && drive > 0.72) {
      const o = this._osc('sawtooth', 118 + r() * 26);
      const f1 = this._bq('bandpass', 480, 8);
      const og = this._gain(0);
      og.gain.setValueAtTime(0.0001, t0);
      og.gain.linearRampToValueAtTime(0.030 * (drive - 0.72) / 0.48, t0 + dur * 0.2);
      og.gain.setTargetAtTime(0, t0 + dur * 0.3, dur * 0.25);
      o.connect(og); og.connect(f1); f1.connect(this.localIn);
      o.start(t0); o.stop(t0 + dur + 0.2);
    }
  }

  _updateHeart() {
    if (this._hp > 0.38) return;
    const now = this.ctx.currentTime;
    if (now < this._nextHeart) return;
    const sev = clamp((0.38 - this._hp) / 0.38, 0, 1);
    const bpm = lerp(78, 148, sev);
    const beat = 60 / bpm;

    const lub = (this._heartPhase ^= 1) === 1;
    this._nextHeart = now + (lub ? beat * 0.30 : beat * 0.70);

    const t = this._when();
    const f = lub ? 58 : 46;
    const amp = (lub ? 0.16 : 0.11) * (0.35 + sev * 0.85);
    const o = this._osc('sine', f);
    this._sweep(o.frequency, t, f * 1.35, f * 0.75, 0.075);
    const g = this._gain(0);
    this._envAD(g.gain, t, amp, 0.006, 0.085);
    o.connect(g); g.connect(this.localIn);
    o.start(t); o.stop(t + 0.32);

    // a little valve thud on top so it is not a pure sine
    const n = this._noise('brown', t, 0.10, 0.7);
    const lp = this._bq('lowpass', 190, 1.1);
    const g2 = this._gain(0);
    this._envAD(g2.gain, t, amp * 0.55, 0.004, 0.06);
    n.connect(lp); lp.connect(g2); g2.connect(this.localIn);
  }

  /**
   * Fallback gait. If nobody ever emits 'footstep', synthesise one from the
   * player's actual speed with a naturally irregular stride — alternating feet
   * with a slightly uneven left/right interval, which is what real walking is.
   */
  _updateAutoGait(dt) {
    if (!this._autoFootsteps) return;
    if (!this._grounded) return;
    const sp = this._playerSpeedSm;
    if (sp < 0.45) { this._nextStep = 0; return; }
    const now = this.ctx.currentTime;
    if (!this._nextStep) { this._nextStep = now + 0.18; return; }
    if (now < this._nextStep) return;

    const r = this._rnd;
    // stride length ~0.78 m walking, stretching to ~1.15 m at a sprint
    const stride = lerp(0.72, 1.15, clamp(sp / 6.2, 0, 1));
    const interval = clamp(stride / Math.max(0.5, sp), 0.20, 1.1);
    // uneven gait: the trailing foot lands slightly late
    const skew = (this._stepFoot & 1) ? 1.06 : 0.94;
    this._nextStep = now + interval * skew * (0.95 + r() * 0.10);

    this.footstep(this._lastGroundSurface || this._guessSurface(), sp, 0, 0, 0, true);
    void dt;
  }

  /** Best-effort ground material under the player, from a short downward ray. */
  _guessSurface() {
    const cam = this.g?.camera;
    const list = this._probeList;
    if (!cam || !list.length) return 'concrete';
    this._vTmp.copy(cam.position);
    this._vTmp2.set(0, -1, 0);
    this._ray.set(this._vTmp, this._vTmp2);
    this._ray.far = 3.0;
    const hits = this._rayHits;
    hits.length = 0;
    let surf = 'concrete';
    try {
      this._ray.intersectObjects(list, !!this._probeRecursive, hits);
      if (hits.length) {
        const m = hits[0].object?.material;
        const mm = Array.isArray(m) ? m[0] : m;
        surf = mm?.userData?.bsImpact || mm?.userData?.bsSurface ||
          this.g?.materials?.surfaceFor?.(mm) || 'concrete';
      }
    } catch (_) { }
    hits.length = 0;
    this._lastGroundSurface = surf;
    return surf;
  }

  /* ================================================================ *
   *  Misc public
   * ================================================================ */

  setMuted(m) {
    this.muted = !!m;
    if (!this.ctx) return this;
    this.master.gain.setTargetAtTime(this.muted ? 0 : this._volMaster, this.ctx.currentTime, 0.03);
    return this;
  }

  setMasterVolume(v) {
    this._volMaster = clamp(num(v, 0.8), 0, 2);
    if (this.g?.config?.audio) this.g.config.audio.master = this._volMaster;
    if (this.ctx && !this.muted) {
      this.master.gain.setTargetAtTime(this._volMaster, this.ctx.currentTime, 0.03);
    }
    return this;
  }

  /** Generic one-shot at a world position — a thin façade over the recipes. */
  playAt(name, x, y, z, opts) {
    if (!this.ready) return;
    switch (name) {
      case 'gunshot': this.gunshot({ x, y, z, ...(opts || {}) }); break;
      case 'distant': this.distantReport(x, y, z, opts); break;
      case 'explosion': this.explosion(x, y, z, opts?.radius ?? 6, opts?.power ?? 1); break;
      case 'chatter': this.chatter(x, y, z, opts?.kind); break;
      case 'shell': this.shellCasing(x, y, z, opts?.delay); break;
      case 'mag': this.magDrop(x, y, z, opts?.surface); break;
      case 'footstep': this.footstep(opts?.surface, opts?.speed ?? 3, x, y, z, false); break;
      default: this.impact(name, x, y, z, opts?.power ?? 1); break;
    }
  }

  resize(/* w, h */) { }

  dispose() {
    this._dropGestureListeners();
    for (const off of this._unbind) { try { off(); } catch (_) { } }
    this._unbind.length = 0;
    if (this.ctx) {
      try { this.master.disconnect(); } catch (_) { }
      try { const p = this.ctx.close(); if (p && p.catch) p.catch(() => { }); } catch (_) { }
    }
    this.ctx = null;
    this.ready = false;
  }
}

export default Audio;
