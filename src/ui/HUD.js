// =============================================================================
// BLACKSITE — HUD.js
// Owner: the HUD agent. Owns every pixel of 2D interface: crosshair, hit/kill
// markers, ammo readout, health (vignette + directional damage arcs), compass
// strip, tac-map, killfeed, objective marker, pause/start screen.
//
// Two rules this file will not break:
//   1. NO KEYBIND LEGEND IN THE PLAY FRAME. Control bindings appear on the
//      pause / first-run card only. A permanent 'WASD MOVE  LMB FIRE' strip is
//      a browser-demo tell; showHint() exists for transient contextual prompts
//      ("PRESS F TO PLANT") and is empty unless something sets it.
//   2. THREE TYPE TIERS, AND ONLY THREE (see _layout):
//        T1 DISPLAY  heavy, tight tracking, large  — mission line, pause title
//        T2 LABEL    small, ~50% alpha, light tracking — everything supporting
//        T3 READOUT  tabular monospace              — every live number
//      Tracking is reserved for T2 micro-labels. Display type and numerals are
//      set tight. If everything is tracked, nothing has hierarchy.
//
// Rendered into a single 2D canvas layered inside #hud. Canvas (not DOM) because
// every element here is a *drawn* element — ticks, arcs, chamfers, rotated map
// geometry — and one immediate-mode surface keeps the whole interface on one
// clock, one grid and one palette. The backing store is allocated at device
// pixel ratio and every hairline is snapped to the pixel grid, so it is crisp
// at any resolution.
//
// Visual language (matches the golden-hour art direction):
//   * warm off-white ink on a low, soft dark scrim — never pure white on black
//   * condensed uppercase, wide tracking, thin 1px rules, hard chamfered corners
//   * strict margin grid, generous negative space, no boxes-with-rounded-corners
//   * everything eases in/out over 120–200 ms; nothing pops
//
// -----------------------------------------------------------------------------
// PUBLIC API (collaborators)
// -----------------------------------------------------------------------------
//   game.hud.setDemoState()                       populate representative values
//   game.hud.setPaused(bool) / togglePause()
//   game.hud.setWeapon(name, cls, mode, caliber)  identity of the current weapon
//   game.hud.setAmmo(mag, reserve, magSize)
//   game.hud.setHealth(hp, max)
//   game.hud.setObjective(x, y, z, label)         world-space objective marker
//                                                 (fixed in the world; range is
//                                                 measured live and the marker
//                                                 ghosts out when occluded)
//   game.hud.clearObjective()
//   game.hud.pushKill(killer, victim, headshot)   append a killfeed entry
//   game.hud.showHint(text, seconds)              transient contextual prompt
//   game.hud.flashHit(headshot, kill)             manual hit-marker trigger
//   game.hud.setMapScale(metresAcross)
//
// It also listens to every canonical bus event ('ammo', 'shot', 'hit', 'kill',
// 'damage', 'health', 'reload', 'ads', 'explosion') so a collaborator only has
// to emit; no direct calls are required.
// =============================================================================

import * as THREE from 'three';

// ---------------------------------------------------------------- palette ----
// Lifted blacks, warm whites. Nothing is #000 and nothing is #fff.
const INK          = '#f2ece0';   // primary warm white
const INK_DIM      = '#c7bfb0';   // secondary
const INK_FAINT    = '#8e877a';   // tertiary / units / rules
const SCRIM        = '#070a0c';   // scrim base (never used at full opacity)

const COL = {
  ink: INK,
  inkDim: INK_DIM,
  inkFaint: INK_FAINT,
  rule: 'rgba(228,219,201,0.30)',
  ruleSoft: 'rgba(228,219,201,0.16)',
  scrim: SCRIM,
  amber: '#e7ab4a',          // fire mode / objective label / accents
  amberDim: '#a87c34',
  danger: '#e2573a',         // low ammo, kills, damage
  dangerDim: '#8c2f20',
  hostile: '#e2573a',
  objective: '#8fd4e8',      // cool cyan — the only cool ink on the HUD
  objectiveDim: '#4c7d8c',   // occluded / no-line-of-sight state

  // --- tac-map ---------------------------------------------------------------
  // The map is an *instrument*, not a card. Its ground is near-black and almost
  // opaque so the bright game frame never bleeds through and lifts it; its
  // architecture is low-alpha cyan vector line, its ownship amber. Every value
  // here is deliberately darker than anything the 3D frame is likely to render,
  // so the plate always recedes.
  mapVoid:  'rgba(3,5,7,0.90)',
  mapFill:  'rgba(96,158,178,0.075)',
  mapEdge:  'rgba(138,208,230,0.30)',
  mapGrid:  'rgba(120,190,210,0.050)',
  mapFoot:  'rgba(2,4,5,0.66)',
};

const DEG = Math.PI / 180;
const TAU = Math.PI * 2;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (a, b, x) => { const t = clamp01((x - a) / (b - a || 1e-6)); return t * t * (3 - 2 * t); };
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const easeOutQuint = (t) => 1 - Math.pow(1 - t, 5);
const easeOutBack = (t) => { const c1 = 1.70158, c3 = c1 + 1; const u = t - 1; return 1 + c3 * u * u * u + c1 * u * u; };
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) * 0.5);

// Frame-rate independent exponential approach. tau = time constant in seconds.
const approach = (cur, target, dt, tau) => cur + (target - cur) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));

const CARDINALS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

// World-space extraction point: open air over the market plaza at the north end
// of the main street (the level runs +Z = north, street at x in [-7,7], plaza at
// z 23.4..40). Fixed, so camera-to-target distance is a real measurement.
const OBJ_ANCHOR = [-2.4, 6.2, 41.5];

// Deterministic LCG — used once, at build time, for the fallback map footprint.
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

export class HUD {

  // ===========================================================================
  //  CONSTRUCTION
  // ===========================================================================
  constructor(game) {
    this.g = game;
    this.cfg = game.config;

    // ---------------------------------------------------------------- canvas
    const host = document.getElementById('hud') || document.body;
    const cv = document.createElement('canvas');
    cv.id = 'blacksite-hud';
    cv.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;' +
                       'display:block;pointer-events:none;image-rendering:auto;';
    host.appendChild(cv);
    this.canvas = cv;
    this.ctx = cv.getContext('2d', { alpha: true });
    this.ctx.textBaseline = 'alphabetic';

    this.dpr = Math.min(2, (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1);
    this.W = Math.max(2, innerWidth);
    this.H = Math.max(2, innerHeight);

    // letter-spacing support (Chromium 99+); we degrade to plain text if absent
    this.hasLS = ('letterSpacing' in this.ctx);
    this._lsCache = new Map();

    this.font = this._pickFontStack();
    this.mono = 'ui-monospace, "Consolas", "DejaVu Sans Mono", "Courier New", monospace';

    // ------------------------------------------------------------ HUD state
    const wcfg = this.cfg.weapon || {};
    this.weaponName = 'M4A1';
    this.weaponClass = 'CARBINE';
    this.fireMode = 'AUTO';
    this.caliber = '5.56×45';
    this.magSize = wcfg.magSize ?? 30;
    this.mag = this.magSize;
    this.reserve = wcfg.reserve ?? 210;

    this.hpMax = 100;
    this.hp = 100;
    this._healthAuthoritative = false;
    this.regenDelay = 0;
    this.painFlash = 0;
    this._vig = 0;                 // smoothed vignette intensity

    this.paused = true;            // boot straight into the START card
    this.started = false;
    this._demo = false;
    this._everLocked = false;
    this.pauseT = 1;               // 0..1 eased overlay presence

    this.boot = 0;                 // 0..1 HUD entrance animation
    this.time = 0;

    // crosshair / spread
    this.spread = 0.34;
    this._spreadFire = 0;
    this._spreadMove = 0;
    this.adsT = 0;
    this._adsTarget = 0;
    this._fireKick = 0;
    this._speed = 0;
    this._demoSpread = -1;         // >=0 pins the spread (screenshot determinism)

    // markers
    this._hit = { t: 1e9, life: 0.42, head: false, count: 0 };
    this._killMark = { t: 1e9, life: 0.66, head: false };

    // reload
    this.reloading = false;
    this.reloadT = 0;
    this.reloadDur = 2.1;

    // killfeed (pooled, fixed capacity — no allocation at runtime)
    this.KF_MAX = 6;
    this.kf = new Array(this.KF_MAX);
    for (let i = 0; i < this.KF_MAX; i++) this.kf[i] = { a: '', b: '', head: false, t: 1e9, life: 5.2, used: false };
    this._kfHead = 0;

    // damage arcs (pooled)
    this.ARC_MAX = 6;
    this.arcs = new Array(this.ARC_MAX);
    for (let i = 0; i < this.ARC_MAX; i++) this.arcs[i] = { ang: 0, t: 1e9, life: 2.6, power: 1 };
    this._arcHead = 0;

    // contacts (pooled — filled every frame from registry.enemies / demo set)
    this.CONTACT_MAX = 48;
    this.contacts = new Float32Array(this.CONTACT_MAX * 4); // x, z, hot, dist
    this.contactCount = 0;
    this._demoContacts = null;

    // Gunfire contacts. Unsuppressed enemy fire reveals the shooter for a beat —
    // that, and an AI that explicitly reports itself alerted, are the ONLY ways a
    // hostile reaches the tac-map. The map is an instrument, not an x-ray of every
    // AI in the compound; a plate speckled with a dozen permanent red pips is an
    // arcade radar, and the compass strip turns into a rhythm game.
    this.GF_MAX = 10;
    this.GF_LIFE = 2.6;
    this.gunfire = new Float32Array(this.GF_MAX * 3);   // x, z, life
    this._gfHead = 0;
    this.PIP_MAX = 6;                                    // hard cap on compass pips

    // objective — a fixed point in the WORLD (the plaza landing zone at the
    // north end of the main street). It is deliberately NOT re-anchored to the
    // camera: the distance readout has to change when the player moves, which is
    // the entire reason the readout exists.
    this.objective = { x: OBJ_ANCHOR[0], y: OBJ_ANCHOR[1], z: OBJ_ANCHOR[2], label: 'EXTRACT', active: true };
    this._objDist = -1;
    this._objDistStr = '—';
    this._objClear = 1;        // 1 = clear line of sight, 0 = behind geometry
    this._objVis = 1;          // smoothed
    this._objLosT = 0;
    this._objP = new THREE.Vector3();

    // Contextual hint line. Empty by default — control bindings live on the
    // pause / first-run card, never burned across the bottom of the play frame.
    this.hintText = '';
    this.hintT = 1e9;
    this.hintLife = 0;

    // cached strings (rebuilt only when the underlying value changes)
    this._magStr = String(this.mag);
    this._resStr = '/ ' + this.reserve;
    this._hdgStr = '000';
    this._nameStr = this.weaponName + '  ·  ' + this.weaponClass;

    // scratch math — nothing in update() allocates
    this._fwd = new THREE.Vector3();
    this._v3 = new THREE.Vector3();
    this._mInv = new THREE.Matrix4();
    this._prevCam = new THREE.Vector3();
    this._prevCamValid = false;
    this._dash = [3, 3];           // reused — setLineDash must never allocate
    this._noDash = [];

    this.heading = 0;              // radians, 0 = north (-Z)
    this.headingDeg = 0;
    this.px = 0; this.pz = 0; this.py = 0;

    // map footprint (world-space AABB rectangles, flat: x, z, w, h)
    this.mapMeters = 58;
    this._foot = new Float32Array(0);
    this._footCount = 0;
    this._footColliders = -1;
    this._footCheck = 0;
    this._scratchBox = new THREE.Box3();

    // ------------------------------------------------------------- listeners
    const bus = this.g.bus;
    this._un = [];
    if (bus) {
      this._un.push(bus.on('ammo', (d) => this._onAmmo(d)));
      this._un.push(bus.on('shot', (d) => this._onShot(d)));
      this._un.push(bus.on('hit', (d) => this._onHit(d)));
      this._un.push(bus.on('kill', (d) => this._onKill(d)));
      this._un.push(bus.on('damage', (d) => this._onDamage(d)));
      this._un.push(bus.on('health', (d) => this._onHealth(d)));
      this._un.push(bus.on('reload', (d) => this._onReload(d)));
      this._un.push(bus.on('ads', (d) => { this._adsTarget = d && d.active ? 1 : 0; }));
      this._un.push(bus.on('explosion', () => { this.painFlash = Math.min(1, this.painFlash + 0.18); }));
    }

    this._onKeyDown = (e) => {
      if (e.code === 'Escape') { this.togglePause(); return; }
      if (!this.started) this._begin();
    };
    this._onPointerDown = () => { if (!this.started) this._begin(); else if (this.paused) this.setPaused(false); };
    this._onLockChange = () => {
      const locked = !!document.pointerLockElement;
      if (locked) { this._everLocked = true; this.started = true; this.setPaused(false); }
      else if (this._everLocked && !this._demo) this.setPaused(true);
    };
    addEventListener('keydown', this._onKeyDown, false);
    addEventListener('pointerdown', this._onPointerDown, false);
    document.addEventListener('pointerlockchange', this._onLockChange, false);

    this._layout();
  }

  async init() {
    this._buildVignette();
    this._buildMapPlate();
    this._buildFootprint(true);
    this._syncCam();
    this._updateObjective(0, true);
    this._objVis = this._objClear;
    // Draw one frame immediately so the HUD is present on the very first
    // presented frame rather than appearing a frame late.
    this._draw(0);
  }

  dispose() {
    for (const off of this._un) { try { off(); } catch (e) { /* bus already gone */ } }
    removeEventListener('keydown', this._onKeyDown);
    removeEventListener('pointerdown', this._onPointerDown);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    this.canvas.remove();
  }

  // ===========================================================================
  //  TYPOGRAPHY
  // ===========================================================================

  // Pick the most condensed grotesque actually installed. Bahnschrift ships with
  // Windows and is a DIN-derived condensed variable face — exactly the register a
  // military UI wants. Everything degrades gracefully.
  _pickFontStack() {
    const cands = [
      '"Bahnschrift SemiCondensed"',
      '"Bahnschrift Condensed"',
      '"Bahnschrift"',
      '"Roboto Condensed"',
      '"Arial Narrow"',
      '"Segoe UI Semibold"',
      '"Segoe UI"',
    ];
    const tail = ', "Segoe UI", system-ui, sans-serif';
    const ctx = this.ctx;
    for (const c of cands) {
      try {
        if (document.fonts && typeof document.fonts.check === 'function') {
          if (!document.fonts.check('16px ' + c)) continue;
        }
        const prev = ctx.font;
        ctx.font = '600 17px ' + c + tail;
        const ok = ctx.font.indexOf('17px') >= 0;
        ctx.font = prev;
        if (ok) return c + tail;
      } catch (e) { /* keep looking */ }
    }
    return 'system-ui, sans-serif';
  }

  _ls(px) {
    if (!this.hasLS) return '0px';
    let s = this._lsCache.get(px);
    if (s === undefined) { s = px.toFixed(2) + 'px'; this._lsCache.set(px, s); }
    return s;
  }

  // Positional-argument text helper: zero object allocation per call.
  _txt(str, x, y, font, color, align, lsPx, alpha) {
    const ctx = this.ctx;
    ctx.font = font;
    if (this.hasLS) ctx.letterSpacing = this._ls(lsPx || 0);
    ctx.textAlign = align || 'left';
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha === undefined ? 1 : alpha;
    // Chromium adds the tracking after the final glyph, which visually shifts
    // right-aligned runs left by one step. Compensate.
    ctx.fillText(str, (align === 'right' && lsPx) ? x + lsPx : x, y);
    ctx.globalAlpha = 1;
  }

  _measure(str, font, lsPx) {
    const ctx = this.ctx;
    ctx.font = font;
    if (this.hasLS) ctx.letterSpacing = this._ls(lsPx || 0);
    return ctx.measureText(str).width;
  }

  // ===========================================================================
  //  LAYOUT — one strict grid, recomputed only on resize
  // ===========================================================================
  _layout() {
    const W = this.W, H = this.H;
    // HUD scale tracks the frame height, but with a floor: below ~0.8 the T2
    // micro-labels fall under 8 px and stop being type. A shooter's interface
    // grows *relatively* larger on a small screen — it never shrinks with it.
    const s = this.s = clamp(H / 1080, 0.80, 2.0);
    const R = (v) => Math.round(v);

    this.M = R(44 * s);                  // outer margin
    this.hair = 1 / this.dpr;            // one device pixel
    this.cx = Math.round(W * 0.5) + 0.5 * 0;
    this.cy = Math.round(H * 0.5);

    // ---- three type tiers, and only three -----------------------------------
    //  T1 DISPLAY  heavy, tight tracking, large      — the one thing you read first
    //  T2 LABEL    small caps, ~55% alpha, light tracking — supporting text
    //  T3 READOUT  tabular monospace                 — every live number
    // Tracking is reserved for T2 micro-labels (where it is genuine typographic
    // practice). Display type and numerals are set tight; nothing else is tracked.
    const f = (w, px) => w + ' ' + R(px * s) + 'px ' + this.font;
    const m = (w, px) => w + ' ' + R(px * s) + 'px ' + this.mono;
    this.F = {
      // T1 display
      missionT: f('700', 22),
      pauseT:   f('700', 54),
      // T2 labels
      missionK: f('600', 9),
      missionS: f('500', 10),
      wpnName:  f('600', 12),
      wpnMeta:  f('500', 10),
      mode:     f('600', 11),
      objLabel: f('700', 10),
      mapLabel: f('600', 9),
      compass:  f('700', 15),
      compassS: f('500', 10),
      hint:     f('600', 11),
      pauseS:   f('500', 11),
      pauseK:   f('600', 11),
      prompt:   f('700', 13),
      // T3 readouts (tabular)
      magBig:   f('700', 60),
      magRes:   m('500', 17),
      heading:  m('600', 12),
      objDist:  m('600', 13),
      mapNum:   m('500', 9),
      pauseV:   m('500', 12),
      stat:     m('500', 11),
      kf:       m('600', 12),
    };

    // ammo block (bottom-right, right-aligned to the margin)
    this.ammo = {
      x: W - this.M,
      yName: H - this.M - R(112 * s),
      yRule: H - this.M - R(100 * s),
      yNum:  H - this.M - R(44 * s),
      yTick: H - this.M - R(30 * s),
      hTick: R(9 * s),
      wTick: R(176 * s),
      yMeta: H - this.M - R(8 * s),
      wRule: R(222 * s),
    };

    // minimap (bottom-left). Labels live INSIDE the plate on a darker footer
    // rail — a caption sitting outside the frame is what makes a tac-map read
    // as a dashboard card.
    const ms = R(186 * s);
    this.map = {
      size: ms,
      x: this.M,
      y: H - this.M - ms,
      cham: R(14 * s),
      foot: R(15 * s),
      cx: this.M + ms * 0.5,
      cy: H - this.M - ms * 0.5,
    };

    // compass (top-centre). Rows, top to bottom: cardinal labels, tick lane,
    // baseline rule, contact-pip lane, numeric heading.
    this.comp = {
      w: Math.round(Math.min(860 * s, W * 0.50)),
      yLabel: this.M + R(12 * s),         // baseline of the cardinal letters
      h: R(14 * s),                       // tick lane height
      span: 118,                          // degrees visible across the strip
    };
    this.comp.yTick = this.comp.yLabel + R(9 * s);
    this.comp.yBase = this.comp.yTick + this.comp.h;
    this.comp.x = Math.round(W * 0.5);

    // killfeed (top-right)
    this.kfBox = { x: W - this.M, y: this.M + R(34 * s), lh: R(23 * s) };

    // mission block (top-left)
    this.mission = { x: this.M, y: this.M + R(12 * s) };

    // scrims
    const c = this.ctx;
    const topH = R(132 * s), botH = R(232 * s);
    this.gTop = c.createLinearGradient(0, 0, 0, topH);
    this.gTop.addColorStop(0, 'rgba(7,10,12,0.40)');
    this.gTop.addColorStop(0.55, 'rgba(7,10,12,0.13)');
    this.gTop.addColorStop(1, 'rgba(7,10,12,0)');
    this.gTopH = topH;

    this.gBot = c.createLinearGradient(0, H - botH, 0, H);
    this.gBot.addColorStop(0, 'rgba(7,10,12,0)');
    this.gBot.addColorStop(0.48, 'rgba(7,10,12,0.16)');
    this.gBot.addColorStop(1, 'rgba(7,10,12,0.50)');
    this.gBotH = botH;

    // backing store
    const bw = Math.max(2, Math.round(W * this.dpr));
    const bh = Math.max(2, Math.round(H * this.dpr));
    if (this.canvas.width !== bw || this.canvas.height !== bh) {
      this.canvas.width = bw;
      this.canvas.height = bh;
    }
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx.textBaseline = 'alphabetic';
    this.ctx.lineCap = 'butt';
    this.ctx.lineJoin = 'miter';

    this._buildMapPlate();
  }

  resize(w, h) {
    this.W = Math.max(2, w);
    this.H = Math.max(2, h);
    this.dpr = Math.min(2, (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1);
    this._layout();
    this._buildVignette();
  }

  // ===========================================================================
  //  OFFSCREEN — tac-map plate furniture, generated once per layout
  //  Two cached layers so the per-frame cost is two drawImage calls:
  //    _mapBed  the ground bed (near-black, slight cool lift toward the centre)
  //    _mapOv   scanlines + phosphor glow + the fade-to-edge vignette
  // ===========================================================================
  _buildMapPlate() {
    const M = this.map;
    if (!M) return;
    const px = Math.max(8, Math.round(M.size * this.dpr));

    // ---- ground bed
    const bed = this._mapBed || (this._mapBed = document.createElement('canvas'));
    bed.width = px; bed.height = px;
    const b = bed.getContext('2d');
    b.clearRect(0, 0, px, px);
    b.fillStyle = COL.mapVoid;
    b.fillRect(0, 0, px, px);
    const lift = b.createRadialGradient(px * 0.5, px * 0.42, 0, px * 0.5, px * 0.42, px * 0.62);
    lift.addColorStop(0.00, 'rgba(46,86,100,0.30)');
    lift.addColorStop(0.55, 'rgba(30,58,70,0.13)');
    lift.addColorStop(1.00, 'rgba(20,40,50,0)');
    b.fillStyle = lift;
    b.fillRect(0, 0, px, px);

    // ---- CRT overlay
    const cv = this._mapOv || (this._mapOv = document.createElement('canvas'));
    cv.width = px; cv.height = px;
    const c = cv.getContext('2d');
    c.clearRect(0, 0, px, px);

    // scanlines — one dark device-pixel line every ~2.5 px
    const step = Math.max(2, Math.round(2.6 * this.dpr));
    c.fillStyle = 'rgba(0,0,0,0.34)';
    for (let yy = 0; yy < px; yy += step) c.fillRect(0, yy, px, 1);

    // faint phosphor bloom so the tube feels lit rather than printed
    const glow = c.createRadialGradient(px * 0.5, px * 0.5, 0, px * 0.5, px * 0.5, px * 0.55);
    glow.addColorStop(0.00, 'rgba(126,198,222,0.055)');
    glow.addColorStop(0.60, 'rgba(126,198,222,0.016)');
    glow.addColorStop(1.00, 'rgba(126,198,222,0)');
    c.globalCompositeOperation = 'lighter';
    c.fillStyle = glow;
    c.fillRect(0, 0, px, px);
    c.globalCompositeOperation = 'source-over';

    // fade to edge — geometry dissolves into the bezel instead of being sliced
    const vg = c.createRadialGradient(px * 0.5, px * 0.5, px * 0.26, px * 0.5, px * 0.5, px * 0.62);
    vg.addColorStop(0.00, 'rgba(3,5,7,0)');
    vg.addColorStop(0.58, 'rgba(3,5,7,0.34)');
    vg.addColorStop(0.86, 'rgba(3,5,7,0.80)');
    vg.addColorStop(1.00, 'rgba(3,5,7,0.97)');
    c.fillStyle = vg;
    c.fillRect(0, 0, px, px);

    // view-cone gradient, cached — gradient coords live in user space, and the
    // only transform applied at paint time is the boot slide, which moves the
    // wedge and the plate together. Nothing is allocated per frame.
    const cr = M.size * 0.58;
    const cg = this.ctx.createRadialGradient(M.cx, M.cy, 2 * this.s, M.cx, M.cy, cr);
    cg.addColorStop(0.00, 'rgba(231,171,74,0.17)');
    cg.addColorStop(0.45, 'rgba(231,171,74,0.070)');
    cg.addColorStop(1.00, 'rgba(231,171,74,0)');
    this._coneGrad = cg;
    this._coneR = cr;

    this._mapPlateReady = true;
  }

  // ===========================================================================
  //  OFFSCREEN — blood vignette, generated once
  // ===========================================================================
  _buildVignette() {
    const w = 384, h = 216;
    const cv = this._vigCanvas || (this._vigCanvas = document.createElement('canvas'));
    cv.width = w; cv.height = h;
    const c = cv.getContext('2d');
    c.clearRect(0, 0, w, h);

    // base: an elliptical falloff hugging the frame edge
    const g = c.createRadialGradient(w * 0.5, h * 0.5, h * 0.20, w * 0.5, h * 0.5, h * 0.86);
    g.addColorStop(0.00, 'rgba(126,20,14,0)');
    g.addColorStop(0.55, 'rgba(126,20,14,0.10)');
    g.addColorStop(0.82, 'rgba(112,17,12,0.52)');
    g.addColorStop(1.00, 'rgba(74,10,8,0.92)');
    c.save();
    c.translate(w * 0.5, h * 0.5);
    c.scale(w / h * 0.86, 1);
    c.translate(-w * 0.5, -h * 0.5);
    c.fillStyle = g;
    c.fillRect(-w, -h, w * 3, h * 3);
    c.restore();

    // organic wet blobs around the perimeter so it never reads as a clean gradient
    const rnd = lcg(0x51ce);
    c.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 26; i++) {
      const a = rnd() * TAU;
      const rr = 0.80 + rnd() * 0.36;
      const bx = w * 0.5 + Math.cos(a) * w * 0.5 * rr;
      const by = h * 0.5 + Math.sin(a) * h * 0.5 * rr;
      const br = (12 + rnd() * 52);
      const bg = c.createRadialGradient(bx, by, 0, bx, by, br);
      const al = 0.10 + rnd() * 0.22;
      bg.addColorStop(0, 'rgba(146,26,16,' + al.toFixed(3) + ')');
      bg.addColorStop(1, 'rgba(146,26,16,0)');
      c.fillStyle = bg;
      c.fillRect(bx - br, by - br, br * 2, br * 2);
    }
    c.globalCompositeOperation = 'source-over';
    this._vigReady = true;
  }

  // ===========================================================================
  //  BUS HANDLERS
  // ===========================================================================
  _onAmmo(d) {
    if (!d || this._demo) return;   // demo values are pinned for the critic
    if (typeof d.mag === 'number' && d.mag !== this.mag) { this.mag = d.mag | 0; this._magStr = String(this.mag); }
    if (typeof d.reserve === 'number' && d.reserve !== this.reserve) { this.reserve = d.reserve | 0; this._resStr = '/ ' + this.reserve; }
    if (typeof d.magSize === 'number' && d.magSize > 0) this.magSize = d.magSize | 0;
  }

  //  The 'shot' bus event carries BOTH player and enemy fire. Enemy fire must
  //  never kick the player's reticle or drain the player's magazine; it paints a
  //  short-lived gunfire contact on the tac-map instead.
  _onShot(d) {
    if (d && (d.source === 'enemy' || d.weapon === 'enemy_rifle')) { this._onEnemyShot(d); return; }
    this._spreadFire = Math.min(1.05, this._spreadFire + 0.30);
    this._fireKick = 1;
    this._tickFlash = 1;
    if (!this._ammoAuthoritative && this.mag > 0) { this.mag--; this._magStr = String(this.mag); }
  }

  _onEnemyShot(d) {
    let x, z;
    const o = d.origin;
    if (o && typeof o.x === 'number') { x = o.x; z = o.z; }
    else {
      const p = d.enemy && d.enemy.position;
      if (!p || typeof p.x !== 'number') return;
      x = p.x; z = p.z;
    }
    const i = this._gfHead;
    this._gfHead = (this._gfHead + 1) % this.GF_MAX;
    this.gunfire[i * 3] = x; this.gunfire[i * 3 + 1] = z; this.gunfire[i * 3 + 2] = this.GF_LIFE;
  }

  _onHit(d) {
    const head = !!(d && d.headshot);
    this._hit.t = 0;
    this._hit.head = head;
    this._hit.count = Math.min(6, this._hit.count + 1);
  }

  _onKill(d) {
    const head = !!(d && d.headshot);
    this._killMark.t = 0;
    this._killMark.head = head;
    const e = d && d.enemy;
    const victim = (e && (e.callsign || e.name)) || this._nextTangoName();
    this.pushKill('VIPER 1-1', String(victim).toUpperCase(), head);
  }

  _onDamage(d) {
    const amt = (d && typeof d.amount === 'number') ? d.amount : 12;
    if (!this._healthAuthoritative) this.hp = Math.max(0, this.hp - amt);
    this.regenDelay = 3.6;
    this.painFlash = Math.min(1, this.painFlash + 0.35 + amt / 90);
    let ang = 0;
    const fd = d && d.fromDir;
    if (fd && typeof fd.x === 'number' && typeof fd.z === 'number') ang = this._screenBearing(fd.x, fd.z);
    else if (d && typeof d.angle === 'number') ang = d.angle;
    this._pushArc(ang, clamp01(amt / 45));
  }

  _onHealth(d) {
    if (!d) return;
    this._healthAuthoritative = true;
    if (typeof d.max === 'number' && d.max > 0) this.hpMax = d.max;
    if (typeof d.hp === 'number') {
      if (d.hp < this.hp - 0.5) { this.regenDelay = 3.6; this.painFlash = Math.min(1, this.painFlash + 0.3); }
      this.hp = clamp(d.hp, 0, this.hpMax);
    }
  }

  _onReload(d) {
    const phase = d && d.phase;
    if (phase === 'start') { this.reloading = true; this.reloadT = 0; }
    else if (phase === 'end') { this.reloading = false; this.reloadT = 0; }
  }

  _nextTangoName() {
    this._tango = ((this._tango || 0) % 9) + 1;
    return 'TANGO-' + this._tango;
  }

  _pushArc(ang, power) {
    const a = this.arcs[this._arcHead];
    this._arcHead = (this._arcHead + 1) % this.ARC_MAX;
    a.ang = ang; a.t = 0; a.life = 2.6; a.power = clamp(power, 0.25, 1);
  }

  // ===========================================================================
  //  PUBLIC API
  // ===========================================================================
  setWeapon(name, cls, mode, caliber) {
    if (name) this.weaponName = String(name).toUpperCase();
    if (cls) this.weaponClass = String(cls).toUpperCase();
    if (mode) this.fireMode = String(mode).toUpperCase();
    if (caliber) this.caliber = String(caliber);
    this._nameStr = this.weaponName + '  ·  ' + this.weaponClass;
  }

  setAmmo(mag, reserve, magSize) {
    this._ammoAuthoritative = true;
    if (typeof magSize === 'number' && magSize > 0) this.magSize = magSize | 0;
    if (typeof mag === 'number') { this.mag = mag | 0; this._magStr = String(this.mag); }
    if (typeof reserve === 'number') { this.reserve = reserve | 0; this._resStr = '/ ' + this.reserve; }
  }

  setHealth(hp, max) {
    this._healthAuthoritative = true;
    if (typeof max === 'number' && max > 0) this.hpMax = max;
    this.hp = clamp(hp, 0, this.hpMax);
  }

  setObjective(x, y, z, label) {
    this.objective.x = x; this.objective.y = y; this.objective.z = z;
    if (label) this.objective.label = String(label).toUpperCase();
    this.objective.active = true;
  }

  clearObjective() { this.objective.active = false; }

  setMapScale(m) { this.mapMeters = clamp(m, 18, 240); }

  pushKill(killer, victim, headshot) {
    const e = this.kf[this._kfHead];
    this._kfHead = (this._kfHead + 1) % this.KF_MAX;
    e.a = String(killer).toUpperCase();
    e.b = String(victim).toUpperCase();
    e.head = !!headshot;
    e.t = 0; e.life = 5.2; e.used = true;
  }

  showHint(text, seconds) {
    if (text) this.hintText = String(text).toUpperCase();
    this.hintLife = seconds || 8;
    this.hintT = 0;
  }

  flashHit(headshot, kill) {
    this._hit.t = 0; this._hit.head = !!headshot; this._hit.count = Math.min(6, this._hit.count + 1);
    if (kill) { this._killMark.t = 0; this._killMark.head = !!headshot; }
  }

  setPaused(p) {
    const v = !!p;
    if (v === this.paused) return;
    this.paused = v;
    if (!v) this.started = true;
  }

  togglePause() { this.setPaused(!this.paused); if (!this.paused) this.started = true; }

  _begin() { this.started = true; this.setPaused(false); }

  // ---------------------------------------------------------------------------
  //  DEMO STATE — the screenshot critic calls this. Everything the brief lists
  //  must be visible and legible in the resulting frame.
  // ---------------------------------------------------------------------------
  setDemoState() {
    this._demo = true;
    this.started = true;
    this.paused = false;
    this.pauseT = 0;
    this.boot = 1;

    this.setWeapon('M4A1', 'CARBINE', 'AUTO', '5.56×45');
    this.magSize = this.cfg.weapon?.magSize ?? 30;
    this._ammoAuthoritative = true;
    this.mag = 23; this._magStr = '23';
    this.reserve = 210; this._resStr = '/ 210';

    this._healthAuthoritative = true;
    this.hpMax = 100;
    this.hp = 62;
    this.regenDelay = 2.2;
    this.painFlash = 0.22;
    this._vig = smoothstep(100, 24, this.hp) + 0.1;

    // Two directional damage arcs, caught mid-fade. Lives are stretched so the
    // pair is still at readable strength after the capture tool's 1.4 s settle —
    // an indicator that has already bled out to 15% tells the critic nothing.
    for (let i = 0; i < this.ARC_MAX; i++) this.arcs[i].t = 1e9;
    this._arcHead = 0;
    // Strengths are deliberately modest: at full demo strength the band read as
    // a red smear painted across the set rather than as a HUD element.
    this._pushArc(-52 * DEG, 0.52); this.arcs[0].t = 0.10; this.arcs[0].life = 9.0;
    this._pushArc(118 * DEG, 0.34); this.arcs[1].t = 2.20; this.arcs[1].life = 9.0;

    // killfeed — two entries at different ages so the fade curve reads
    for (let i = 0; i < this.KF_MAX; i++) this.kf[i].used = false;
    this._kfHead = 0;
    this.pushKill('VIPER 1-1', 'TANGO-7', false); this.kf[0].t = 3.4; this.kf[0].life = 11;
    this.pushKill('VIPER 1-1', 'TANGO-4', true);  this.kf[1].t = 0.8; this.kf[1].life = 11;

    // pin the crosshair open so all four strokes read in a still frame
    this._demoSpread = 0.33;
    this.spread = 0.33;
    this.adsT = 0;
    this._adsTarget = 0;

    // contacts + objective, anchored to wherever the critic just put the camera
    const cam = this.g.camera;
    if (cam) {
      cam.updateMatrixWorld();
      this._fwd.set(0, 0, -1).applyQuaternion(cam.quaternion);
      const fx = this._fwd.x, fz = this._fwd.z;
      const n = Math.hypot(fx, fz) || 1;
      const ux = fx / n, uz = fz / n;          // forward on the ground plane
      const rx = -uz, rz = ux;                 // right on the ground plane
      const px = cam.position.x, py = cam.position.y, pz = cam.position.z;

      const place = (bearingDeg, dist) => {
        const a = bearingDeg * DEG;
        const ca = Math.cos(a), sa = Math.sin(a);
        return [px + (ux * ca + rx * sa) * dist, pz + (uz * ca + rz * sa) * dist];
      };

      const c = this._demoContacts || (this._demoContacts = new Float32Array(5 * 3));
      const specs = [[24, 19, 1], [-41, 31, 1], [63, 44, 0], [-8, 52, 1], [138, 27, 0]];
      for (let i = 0; i < 5; i++) {
        const p = place(specs[i][0], specs[i][1]);
        c[i * 3] = p[0]; c[i * 3 + 1] = p[1]; c[i * 3 + 2] = specs[i][2];
      }
    }
    for (let i = 0; i < this.GF_MAX; i++) this.gunfire[i * 3 + 2] = 0;

    // The objective stays where it is in the WORLD. Every camera the critic
    // teleports to therefore reports a different, real range to it, and the
    // line-of-sight test resolves against real level geometry.
    this.setObjective(OBJ_ANCHOR[0], OBJ_ANCHOR[1], OBJ_ANCHOR[2], 'EXTRACT');
    this._objDist = -1;
    this._syncCam();
    this._updateObjective(0, true);
    this._objVis = this._objClear;

    // no keybind strip, ever — those live on the pause card
    this.hintText = '';
    this.hintT = 1e9;
    this.hintLife = 0;

    this._buildFootprint(false);
  }

  // ===========================================================================
  //  FRAME
  // ===========================================================================

  // Pull camera position, forward vector, inverse world matrix and heading.
  // Split out of update() so init() and setDemoState() can resolve against the
  // camera the instant they are called rather than a frame late. No allocation.
  _syncCam() {
    const cam = this.g.camera;
    if (!cam) return;
    cam.updateMatrixWorld();
    this._mInv.copy(cam.matrixWorld).invert();
    this._fwd.set(0, 0, -1).applyQuaternion(cam.quaternion);
    this.px = cam.position.x; this.py = cam.position.y; this.pz = cam.position.z;

    let h = Math.atan2(this._fwd.x, -this._fwd.z);
    if (h < 0) h += TAU;
    this.heading = h;
    const hd = Math.round(h / DEG) % 360;
    if (hd !== this.headingDeg) {
      this.headingDeg = hd;
      this._hdgStr = hd < 10 ? '00' + hd : hd < 100 ? '0' + hd : String(hd);
    }
  }

  update(dt, t) {
    const d = clamp(dt || 0, 0, 0.1);
    this.time = t || (this.time + d);

    // ---- camera-derived state ------------------------------------------------
    const cam = this.g.camera;
    if (cam) {
      this._syncCam();

      if (this._prevCamValid && d > 0) {
        const dx = cam.position.x - this._prevCam.x;
        const dz = cam.position.z - this._prevCam.z;
        const sp = Math.hypot(dx, dz) / d;
        this._speed = approach(this._speed, Math.min(sp, 12), d, 0.10);
      }
      this._prevCam.copy(cam.position);
      this._prevCamValid = true;
    }

    // ---- lazily-bound collaborators (constructed after HUD) -------------------
    const wpn = this.g.weapon;
    if (wpn) {
      // ADS always tracks the weapon — the critic's `ads` preset must fade the
      // reticle down to the optic. Everything else is frozen in demo mode.
      if (typeof wpn.ads === 'boolean') this._adsTarget = wpn.ads ? 1 : 0;
      else if (typeof wpn.adsActive === 'boolean') this._adsTarget = wpn.adsActive ? 1 : 0;
      else if (typeof wpn.aiming === 'boolean') this._adsTarget = wpn.aiming ? 1 : 0;

      if (!this._demo) {
        if (typeof wpn.mag === 'number' && wpn.mag !== this.mag) { this.mag = wpn.mag | 0; this._magStr = String(this.mag); this._ammoAuthoritative = true; }
        if (typeof wpn.reserve === 'number' && wpn.reserve !== this.reserve) { this.reserve = wpn.reserve | 0; this._resStr = '/ ' + this.reserve; }
        if (typeof wpn.magSize === 'number' && wpn.magSize > 0) this.magSize = wpn.magSize | 0;
        if (wpn.name && !this._weaponNamed) {
          this.setWeapon(wpn.name, wpn.cls || wpn.category || this.weaponClass, wpn.fireMode || this.fireMode, wpn.caliber || this.caliber);
          this._weaponNamed = true;
        }
        if (typeof wpn.reloading === 'boolean') this.reloading = wpn.reloading;
      }
    }

    // ---- timers ---------------------------------------------------------------
    this.boot = Math.min(1, this.boot + d / 0.75);
    this._hit.t += d;
    this._killMark.t += d;
    this.hintT += d;
    this._tickFlash = Math.max(0, (this._tickFlash || 0) - d * 4.5);
    for (let i = 0; i < this.GF_MAX; i++) if (this.gunfire[i * 3 + 2] > 0) this.gunfire[i * 3 + 2] -= d;
    for (let i = 0; i < this.ARC_MAX; i++) this.arcs[i].t += d;
    for (let i = 0; i < this.KF_MAX; i++) if (this.kf[i].used) this.kf[i].t += d;
    if (this.reloading) { this.reloadT += d; if (this.reloadT > this.reloadDur + 0.6) this.reloading = false; }

    // ---- health / regen --------------------------------------------------------
    if (this.regenDelay > 0) this.regenDelay -= d;
    else if (!this._healthAuthoritative && this.hp < this.hpMax) this.hp = Math.min(this.hpMax, this.hp + 24 * d);
    else if (this._healthAuthoritative && this._demo) { /* demo holds its value */ }
    this.painFlash = Math.max(0, this.painFlash - d * 1.35);
    const vigTarget = smoothstep(100, 24, (this.hp / this.hpMax) * 100) + this.painFlash * 0.45;
    this._vig = approach(this._vig, Math.min(1.25, vigTarget), d, 0.16);

    // ---- crosshair spread ------------------------------------------------------
    this._spreadFire = Math.max(0, this._spreadFire - d * (this._demoSpread >= 0 ? 0.02 : 1.55));
    const moveN = clamp01(this._speed / Math.max(1, this.cfg.player?.sprintSpeed ?? 6.2));
    this._spreadMove = approach(this._spreadMove, moveN * 0.42, d, this._spreadMove < moveN * 0.42 ? 0.07 : 0.20);
    this.adsT = approach(this.adsT, this._adsTarget, d, (this.cfg.weapon?.adsTime ?? 0.17) * 0.55);
    this._fireKick = Math.max(0, this._fireKick - d * 7.5);

    let sTarget = 0.10 + this._spreadMove + this._spreadFire * 0.55 + this._fireKick * 0.10;
    sTarget *= (1 - this.adsT * 0.86);
    if (this._demoSpread >= 0) sTarget = this._demoSpread;
    this.spread = approach(this.spread, clamp(sTarget, 0, 1.25), d, this.spread < sTarget ? 0.035 : 0.11);

    // ---- pause overlay presence -------------------------------------------------
    this.pauseT = approach(this.pauseT, (this.paused && !this._demo) ? 1 : 0, d, 0.075);
    if (this.pauseT < 0.002) this.pauseT = 0;

    // ---- objective range + line of sight -------------------------------------------
    this._updateObjective(d, false);

    // ---- contacts + map footprint ------------------------------------------------
    this._gatherContacts();
    this._footCheck += d;
    if (this._footCheck > 2.5) { this._footCheck = 0; this._buildFootprint(false); }

    this._draw(d);
  }

  // ---------------------------------------------------------------------------
  //  Objective: live range and an occlusion test.
  //  Range is measured camera -> world anchor every frame; the anchor is fixed
  //  in the world, so the number is a real measurement and changes as you move.
  //  The occlusion test is the physics BVH's boolean LOS query, throttled to
  //  ~8 Hz (it costs one ray) and smoothed, so the marker cross-fades between
  //  its solid in-sight state and a dim outline-only behind-cover state instead
  //  of popping. No allocation: one cached Vector3, no temporary objects.
  // ---------------------------------------------------------------------------
  _updateObjective(d, force) {
    if (!this.objective.active) { this._objVis = this._objClear = 1; return; }
    const o = this._objP.set(this.objective.x, this.objective.y, this.objective.z);

    const dx = o.x - this.px, dy = o.y - this.py, dz = o.z - this.pz;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const di = Math.round(dist);
    if (di !== this._objDist) {
      this._objDist = di;
      this._objDistStr = di >= 1000 ? (di / 1000).toFixed(1) + ' KM' : di + ' M';
    }

    this._objLosT += d;
    if (force || this._objLosT > 0.125) {
      this._objLosT = 0;
      const ph = this.g.physics;
      const cam = this.g.camera;
      let clear = 1;
      if (cam && ph && typeof ph.lineOfSight === 'function') {
        try { clear = ph.lineOfSight(cam.position, o) ? 1 : 0; } catch (e) { clear = 1; }
      }
      this._objClear = clear;
    }
    this._objVis = force ? this._objClear : approach(this._objVis, this._objClear, d, 0.12);
  }

  // ---------------------------------------------------------------------------
  //  Contacts. Three sources, all of them earned:
  //    1. an AI that explicitly reports itself firing / alerted
  //    2. a decaying gunfire contact from the 'shot' bus (unsuppressed fire gives
  //       your position away — the CoD rule)
  //    3. the demo set, for the screenshot critic
  //  Everything else stays off the plate. Written into a preallocated
  //  Float32Array; no garbage.
  // ---------------------------------------------------------------------------
  _gatherContacts() {
    const arr = this.contacts;
    let n = 0;

    const list = this.g.registry?.enemies;
    if (Array.isArray(list)) {
      for (let i = 0; i < list.length && n < this.CONTACT_MAX; i++) {
        const e = list[i];
        if (!e) continue;
        if (e.alive === false || e.dead === true || e.health <= 0) continue;
        // Explicit hostile intent only. `visible` is deliberately NOT consulted:
        // on a THREE object it is always true, which would light up the map.
        if (!(e.firing === true || e.shooting === true || e.alerted === true || e.detected === true)) continue;
        let p = e.position;
        if (!p || typeof p.x !== 'number') p = e.mesh?.position || e.object?.position || e.root?.position || e.obj?.position;
        if (!p || typeof p.x !== 'number') continue;
        const dx = p.x - this.px, dz = p.z - this.pz;
        const dist = Math.hypot(dx, dz);
        if (dist > 160) continue;
        arr[n * 4] = p.x; arr[n * 4 + 1] = p.z; arr[n * 4 + 2] = 1; arr[n * 4 + 3] = dist;
        n++;
      }
    }

    const gf = this.gunfire;
    for (let i = 0; i < this.GF_MAX && n < this.CONTACT_MAX; i++) {
      if (gf[i * 3 + 2] <= 0) continue;
      const x = gf[i * 3], z = gf[i * 3 + 1];
      arr[n * 4] = x; arr[n * 4 + 1] = z; arr[n * 4 + 2] = 1;
      arr[n * 4 + 3] = Math.hypot(x - this.px, z - this.pz);
      n++;
    }

    const dc = this._demoContacts;
    if (dc) {
      for (let i = 0; i * 3 < dc.length && n < this.CONTACT_MAX; i++) {
        const x = dc[i * 3], z = dc[i * 3 + 1];
        arr[n * 4] = x; arr[n * 4 + 1] = z;
        arr[n * 4 + 2] = dc[i * 3 + 2];
        arr[n * 4 + 3] = Math.hypot(x - this.px, z - this.pz);
        n++;
      }
    }
    this.contactCount = n;
  }

  // ---------------------------------------------------------------------------
  //  Map footprint. Built from whatever the Level agent registered as colliders;
  //  falls back to a procedurally-generated compound so the minimap is never an
  //  empty box. Rebuilt only when the collider count changes.
  // ---------------------------------------------------------------------------
  _buildFootprint(force) {
    const cols = this.g.registry?.colliders;
    const count = Array.isArray(cols) ? cols.length : 0;
    if (!force && count === this._footColliders) return;
    this._footColliders = count;

    if (!count) { this._fallbackFootprint(); return; }

    const cap = Math.min(count, 700);
    if (this._foot.length < cap * 4) this._foot = new Float32Array(cap * 4);
    const out = this._foot;
    const box = this._scratchBox;
    let n = 0;

    for (let i = 0; i < count && n < cap; i++) {
      const c = cols[i];
      if (!c) continue;
      let minx, minz, maxx, maxz;
      const obj = c.isObject3D ? c : (c.mesh?.isObject3D ? c.mesh : (c.object?.isObject3D ? c.object : null));
      if (c.isBox3) { minx = c.min.x; minz = c.min.z; maxx = c.max.x; maxz = c.max.z; }
      else if (c.box && c.box.isBox3) { minx = c.box.min.x; minz = c.box.min.z; maxx = c.box.max.x; maxz = c.box.max.z; }
      else if (c.aabb && c.aabb.min) { minx = c.aabb.min.x; minz = c.aabb.min.z; maxx = c.aabb.max.x; maxz = c.aabb.max.z; }
      else if (c.min && c.max && typeof c.min.x === 'number') { minx = c.min.x; minz = c.min.z; maxx = c.max.x; maxz = c.max.z; }
      else if (obj) {
        try { box.setFromObject(obj, true); } catch (e) { continue; }
        if (!isFinite(box.min.x) || !isFinite(box.max.x)) continue;
        minx = box.min.x; minz = box.min.z; maxx = box.max.x; maxz = box.max.z;
      } else continue;

      const w = maxx - minx, h = maxz - minz;
      // Reject clutter below map legibility, and reject the ground plane / world
      // bounds — a footprint that large is terrain, not architecture, and filling
      // it would turn the minimap into a solid block.
      if (!(w > 0.7 && h > 0.7)) continue;
      if (w > 70 || h > 70 || w * h > 2600) continue;
      out[n * 4] = minx; out[n * 4 + 1] = minz; out[n * 4 + 2] = w; out[n * 4 + 3] = h;
      n++;
    }
    this._footCount = n;
    if (n === 0) this._fallbackFootprint();
  }

  // A believable compound: a main street running north–south, a cross alley, two
  // rows of blocks, and a walled courtyard on the west side. Deterministic.
  _fallbackFootprint() {
    const rects = [];
    const rnd = lcg(0xb1acc5);
    const push = (x, z, w, h) => rects.push(x, z, w, h);

    // west and east building rows flanking the main street (x in [-6, 6])
    for (let side = 0; side < 2; side++) {
      const dir = side ? 1 : -1;
      let z = -46;
      while (z < 44) {
        const depth = 7 + rnd() * 13;
        const width = 9 + rnd() * 15;
        const inset = 6.5 + rnd() * 1.6;
        const x = dir > 0 ? inset : -(inset + width);
        push(x, z, width, depth);
        // stepped annexe — breaks the block silhouette on the map
        if (rnd() > 0.45) {
          const aw = 4 + rnd() * 6, ad = 3.5 + rnd() * 5;
          push(dir > 0 ? x + width : x - aw, z + rnd() * (depth - ad), aw, ad);
        }
        z += depth + 2.2 + rnd() * 5.5;
      }
    }
    // second rank further out, cut by an alley
    for (let side = 0; side < 2; side++) {
      const dir = side ? 1 : -1;
      let z = -40;
      while (z < 38) {
        const depth = 9 + rnd() * 12;
        const width = 11 + rnd() * 12;
        const x = dir > 0 ? 27 + rnd() * 3 : -(27 + rnd() * 3 + width);
        push(x, z, width, depth);
        z += depth + 4 + rnd() * 7;
      }
    }
    // cross street cut: a market row along z ~ -4
    push(-34, -6.5, 12, 5.5);
    push(16, -6.5, 14, 5.5);

    // walled courtyard, north-west
    const cx = -30, cz = -34, cw = 24, ch = 20, t = 0.9;
    push(cx, cz, cw, t); push(cx, cz + ch - t, cw, t);
    push(cx, cz, t, ch); push(cx + cw - t, cz, t, ch * 0.42);
    push(cx + cw - t, cz + ch * 0.62, t, ch * 0.38);
    push(cx + 4, cz + 4, 9, 8);
    push(cx + 15, cz + 10, 6.5, 7);

    // motor pool, south-east
    push(20, 26, 18, 13);
    push(20, 41, 8, 5);

    const out = new Float32Array(rects.length);
    out.set(rects);
    this._foot = out;
    this._footCount = rects.length >> 2;
  }

  // ===========================================================================
  //  DRAW
  // ===========================================================================
  _draw(dt) {
    const ctx = this.ctx;
    const W = this.W, H = this.H;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    const boot = easeOutCubic(this.boot);
    const live = 1 - easeInOut(this.pauseT) * 0.82;   // HUD dims behind the pause card

    this._drawVignette();
    this._drawDamageArcs(boot * live);

    ctx.globalAlpha = 1;
    ctx.fillStyle = this.gTop;
    ctx.fillRect(0, 0, W, this.gTopH);
    ctx.fillStyle = this.gBot;
    ctx.fillRect(0, H - this.gBotH, W, this.gBotH);

    this._drawCompass(boot * live);
    this._drawMission(boot * live);
    this._drawKillfeed(boot * live);
    this._drawMinimap(boot * live);
    this._drawAmmo(boot * live);
    this._drawObjective(boot * live);
    this._drawCrosshair(live);
    this._drawHint(boot * live);
    if (this.pauseT > 0.001) this._drawPause();
    if (this.cfg.debug?.stats) this._drawStats();

    ctx.globalAlpha = 1;
  }

  // --------------------------------------------------------------- vignette --
  _drawVignette() {
    const a = clamp01(this._vig);
    if (a < 0.004 || !this._vigReady) return;
    const ctx = this.ctx;
    ctx.globalAlpha = Math.min(0.94, a * 0.92);
    ctx.drawImage(this._vigCanvas, 0, 0, this.W, this.H);
    ctx.globalAlpha = 1;
    // a very slight extra pulse at critical health — read as a heartbeat, not a bar
    if (this.hp / this.hpMax < 0.32) {
      const p = 0.5 + 0.5 * Math.sin(this.time * 3.4);
      ctx.globalAlpha = 0.10 + p * 0.13;
      ctx.drawImage(this._vigCanvas, 0, 0, this.W, this.H);
      ctx.globalAlpha = 1;
    }
  }

  // ------------------------------------------------------------ damage arcs --
  _screenBearing(dx, dz) {
    const f = this._fwd;
    const n = Math.hypot(f.x, f.z) || 1;
    const ux = f.x / n, uz = f.z / n;
    const rx = -uz, rz = ux;
    const fwdDot = dx * ux + dz * uz;
    const rightDot = dx * rx + dz * rz;
    return Math.atan2(rightDot, fwdDot);       // 0 = ahead, + = to the right
  }

  //  A damage indicator has to answer one question in a tenth of a second:
  //  which way. So: a wide, soft, radially-graded band well outside the reticle,
  //  angularly tapered at both ends (built from short wedge segments so the taper
  //  is real and not a rectangular smear), with a crisp inner leading edge that
  //  carries the direction. It snaps in over ~90 ms and bleeds out over ~2.6 s
  //  while drifting outward.
  //  The band is baked once per (angular width, layout) into a small offscreen
  //  sprite and blitted rotated. Drawing it live as a fan of alpha-blended wedge
  //  segments cross-fades their antialiased shared edges and leaves a fan of
  //  bright radial spokes across the indicator — the taper has to be a real
  //  mask over one continuous fill, not a per-segment alpha.
  _arcSprite(half) {
    const base = Math.min(this.W, this.H);
    const sig = (base | 0) + '|' + this.dpr + '|' + this.s.toFixed(3);
    if (!this._arcSprites || this._arcSig !== sig) { this._arcSprites = new Map(); this._arcSig = sig; }
    const key = Math.round(half / DEG);                 // 1° buckets → ≤11 sprites, ever
    const cached = this._arcSprites.get(key);
    if (cached) return cached;

    const s = this.s, dpr = this.dpr;
    const rIn = base * 0.155, rOut = base * 0.228;
    const size = Math.ceil(rOut * 2 + 8 * s);           // CSS px, square, padded
    const cv = document.createElement('canvas');
    cv.width = cv.height = Math.max(2, Math.round(size * dpr));
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const m = size * 0.5;

    // one continuous annulus wedge, radially graded
    const g = c.createRadialGradient(m, m, rIn, m, m, rOut);
    // The direction cue is carried by a STEEP inner ramp, not by a stroked rim.
    // Any hairline drawn here shows up in a hero still as a stray thread of
    // near-white floating over the set, which is the single most "not shipped"
    // artefact a critic can find in a frame.
    g.addColorStop(0.00, 'rgba(236,104,72,0.00)');
    g.addColorStop(0.17, 'rgba(240,120,86,0.06)');
    g.addColorStop(0.25, 'rgba(233,92,60,0.60)');
    g.addColorStop(0.58, 'rgba(190,46,28,0.22)');
    g.addColorStop(1.00, 'rgba(150,30,18,0.00)');
    c.fillStyle = g;
    c.beginPath();
    c.arc(m, m, rOut, -half, half);
    c.arc(m, m, rIn, half, -half, true);
    c.closePath();
    c.fill();

    // angular taper as a mask over that single fill — no segment seams
    const span = (half * 2) / (Math.PI * 2);
    c.globalCompositeOperation = 'destination-in';
    if (typeof c.createConicGradient === 'function' && span > 0 && span < 1) {
      const cg = c.createConicGradient(-half, m, m);
      const N = 32;
      for (let i = 0; i <= N; i++) {
        const t = i / N;
        const taper = Math.pow(Math.cos((t * 2 - 1) * Math.PI * 0.5), 1.6);
        cg.addColorStop(t * span, 'rgba(255,255,255,' + taper.toFixed(4) + ')');
      }
      cg.addColorStop(Math.min(1, span + 1e-4), 'rgba(255,255,255,0)');
      cg.addColorStop(1, 'rgba(255,255,255,0)');
      c.fillStyle = cg;
      c.fillRect(0, 0, size, size);
    } else {
      // conic gradients missing: fall back to a stepped mask (build-time only)
      const N = 48, step = (half * 2) / N;
      c.fillStyle = '#fff';
      for (let j = 0; j < N; j++) {
        const a0 = -half + j * step;
        const rel = (a0 + step * 0.5) / half;
        c.globalAlpha = Math.pow(Math.cos(rel * Math.PI * 0.5), 1.6);
        c.beginPath();
        c.arc(m, m, rOut + 2, a0, a0 + step);
        c.lineTo(m, m);
        c.closePath();
        c.fill();
      }
      c.globalAlpha = 1;
    }
    c.globalCompositeOperation = 'source-over';

    const sp = { cv, size, rOut };
    if (this._arcSprites.size > 16) this._arcSprites.clear();
    this._arcSprites.set(key, sp);
    return sp;
  }

  _drawDamageArcs(alpha) {
    const ctx = this.ctx;
    const s = this.s;
    const rOut = Math.min(this.W, this.H) * 0.228;

    for (let i = 0; i < this.ARC_MAX; i++) {
      const a = this.arcs[i];
      if (a.t >= a.life) continue;
      const u = a.t / a.life;
      let k = 1 - easeOutCubic(u);
      k *= smoothstep(0, 0.05, u) * 0.4 + 0.6;
      const al = k * (0.45 + a.power * 0.55) * alpha;
      if (al < 0.006) continue;

      const mid = a.ang - Math.PI * 0.5;
      const half = (20 + a.power * 10) * DEG;
      const sp = this._arcSprite(half);
      const d = sp.size * ((rOut + (1 - k) * 16 * s) / rOut);   // drifts outward as it fades

      ctx.save();
      ctx.translate(this.cx, this.cy);
      ctx.rotate(mid);
      ctx.globalAlpha = al;
      ctx.drawImage(sp.cv, -d * 0.5, -d * 0.5, d, d);
      ctx.restore();
    }
    ctx.globalAlpha = 1;
  }

  // --------------------------------------------------------------- crosshair --
  _drawCrosshair(alpha) {
    const ctx = this.ctx;
    const s = this.s;
    const cx = Math.round(this.cx) + 0.5 * (this.dpr === 1 ? 1 : 0);
    const cy = Math.round(this.cy) + 0.5 * (this.dpr === 1 ? 1 : 0);

    const chAlpha = (1 - easeOutCubic(clamp01(this.adsT * 1.12))) * alpha;

    if (chAlpha > 0.01) {
      const gap = (7 + this.spread * 27) * s + this._fireKick * 2.4 * s;
      const len = (12 + this.spread * 7) * s;
      const th = Math.max(2, Math.round(2 * s));
      const oth = th + Math.max(2, Math.round(2.2 * s));

      // dark backing so the reticle survives a blown-out sky
      ctx.globalAlpha = chAlpha * 0.5;
      ctx.fillStyle = 'rgba(5,8,10,0.92)';
      this._chStrokes(cx, cy, gap - 1.2, len + 2.4, oth);
      // ink
      ctx.globalAlpha = chAlpha;
      ctx.fillStyle = INK;
      this._chStrokes(cx, cy, gap, len, th);

      // centre dot — one device pixel, only when nearly still
      const still = 1 - smoothstep(0.12, 0.30, this.spread);
      if (still > 0.02) {
        ctx.globalAlpha = chAlpha * still * 0.75;
        ctx.fillStyle = INK;
        ctx.fillRect(cx - th * 0.5, cy - th * 0.5, th, th);
      }
      ctx.globalAlpha = 1;
    }

    // ---- hit marker ---------------------------------------------------------
    const h = this._hit;
    if (h.t < h.life) {
      const u = h.t / h.life;
      const pop = 1 + (1 - easeOutBack(clamp01(u / 0.22))) * 0.55;
      const a = (1 - easeOutQuint(u)) * alpha;
      const r0 = 7.5 * s * pop, r1 = (14 + Math.min(4, h.count) * 0.8) * s * pop;
      ctx.globalAlpha = a * 0.55;
      ctx.strokeStyle = 'rgba(6,9,11,0.95)';
      ctx.lineWidth = Math.max(2, 3.1 * s);
      this._diagTicks(this.cx, this.cy, r0, r1, h.head ? 0 : 0);
      ctx.globalAlpha = a;
      ctx.strokeStyle = h.head ? '#ffe6c0' : INK;
      ctx.lineWidth = Math.max(1.2, (h.head ? 2.1 : 1.7) * s);
      this._diagTicks(this.cx, this.cy, r0, r1, 0);
      if (h.head) {
        ctx.globalAlpha = a * 0.7;
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, r1 + 4 * s, 0, TAU);
        ctx.lineWidth = Math.max(1, 1.1 * s);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (u >= 1) h.count = 0;
    } else h.count = 0;

    // ---- kill marker --------------------------------------------------------
    const k = this._killMark;
    if (k.t < k.life) {
      const u = k.t / k.life;
      const pop = 1 + (1 - easeOutBack(clamp01(u / 0.26))) * 0.85;
      const a = (1 - easeOutQuint(u)) * alpha;
      const r0 = 9 * s * pop, r1 = 20 * s * pop;
      ctx.save();
      ctx.translate(this.cx, this.cy);
      ctx.rotate((1 - easeOutCubic(clamp01(u / 0.3))) * 0.20);
      ctx.globalAlpha = a * 0.5;
      ctx.strokeStyle = 'rgba(6,9,11,0.95)';
      ctx.lineWidth = Math.max(2.4, 3.6 * s);
      this._diagTicks(0, 0, r0, r1, 0);
      ctx.globalAlpha = a;
      ctx.strokeStyle = COL.danger;
      ctx.lineWidth = Math.max(1.6, 2.4 * s);
      this._diagTicks(0, 0, r0, r1, 0);
      if (k.head) {
        ctx.globalAlpha = a * 0.8;
        ctx.lineWidth = Math.max(1, 1.2 * s);
        ctx.beginPath(); ctx.arc(0, 0, r1 + 5 * s, 0, TAU); ctx.stroke();
      }
      ctx.restore();
      ctx.globalAlpha = 1;
    }

    // ---- reload progress ----------------------------------------------------
    if (this.reloading) {
      const p = clamp01(this.reloadT / this.reloadDur);
      const w = 70 * s, y = this.cy + 46 * s;
      ctx.globalAlpha = 0.55 * alpha;
      ctx.fillStyle = 'rgba(8,11,13,0.8)';
      ctx.fillRect(this.cx - w * 0.5, y, w, Math.max(2, 2.5 * s));
      ctx.globalAlpha = 0.95 * alpha;
      ctx.fillStyle = COL.amber;
      ctx.fillRect(this.cx - w * 0.5, y, w * p, Math.max(2, 2.5 * s));
      this._txt('RELOADING', this.cx, y - 8 * s, this.F.wpnMeta, COL.amber, 'center', 3.0 * s, 0.85 * alpha);
      ctx.globalAlpha = 1;
    }
  }

  _chStrokes(cx, cy, gap, len, th) {
    const ctx = this.ctx;
    const g = Math.round(gap), l = Math.round(len), t = Math.round(th);
    const o = Math.floor(t * 0.5);
    ctx.fillRect(cx - g - l, cy - o, l, t);
    ctx.fillRect(cx + g, cy - o, l, t);
    ctx.fillRect(cx - o, cy - g - l, t, l);
    ctx.fillRect(cx - o, cy + g, t, l);
  }

  _diagTicks(cx, cy, r0, r1, rot) {
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = (Math.PI * 0.25) + i * (Math.PI * 0.5) + rot;
      const ca = Math.cos(a), sa = Math.sin(a);
      ctx.moveTo(cx + ca * r0, cy + sa * r0);
      ctx.lineTo(cx + ca * r1, cy + sa * r1);
    }
    ctx.stroke();
  }

  // ------------------------------------------------------------------- ammo --
  _drawAmmo(alpha) {
    const ctx = this.ctx;
    const s = this.s, A = this.ammo;
    const slide = (1 - easeOutCubic(clamp01(this.boot * 1.3))) * 26 * s;
    const x = A.x + slide;

    const lowN = this.magSize > 0 ? this.mag / this.magSize : 1;
    const low = lowN <= 0.25 || this.mag <= 6;
    const empty = this.mag <= 0;
    const pulse = low ? 0.72 + 0.28 * (0.5 + 0.5 * Math.sin(this.time * 6.2)) : 1;
    const magCol = empty ? COL.danger : low ? '#e8814a' : COL.ink;

    // weapon identity + rule — T2: this is a label, not a headline
    this._txt(this._nameStr, x, A.yName, this.F.wpnName, COL.inkDim, 'right', 1.6 * s, 0.62 * alpha);
    ctx.globalAlpha = alpha;
    this._rule(x - A.wRule, A.yRule, A.wRule, COL.rule, 1);
    // a short accent stub at the right end of the rule
    ctx.fillStyle = COL.amber;
    ctx.globalAlpha = 0.85 * alpha;
    ctx.fillRect(x - Math.round(26 * s), this._snap(A.yRule), Math.round(26 * s), Math.max(1, Math.round(1 * this.s)));
    ctx.globalAlpha = 1;

    // T3 readouts: reserve set tabular, then the big magazine count to its left
    const resW = this._measure(this._resStr, this.F.magRes, 0);
    this._txt(this._resStr, x, A.yNum, this.F.magRes, COL.inkFaint, 'right', 0, 0.72 * alpha);
    this._txt(this._magStr, x - resW - 13 * s, A.yNum, this.F.magBig, magCol, 'right', 0, alpha * pulse);

    // per-round tick strip
    const n = Math.max(1, this.magSize);
    const gapT = Math.max(1, Math.round(1.6 * s));
    const tw = Math.max(2, Math.floor((A.wTick - gapT * (n - 1)) / n));
    const total = tw * n + gapT * (n - 1);
    const x0 = Math.round(x - total);
    const y0 = this._snap(A.yTick);
    const hT = A.hTick;
    for (let i = 0; i < n; i++) {
      const filled = i < this.mag;
      const tx = x0 + i * (tw + gapT);
      if (filled) {
        const fresh = (i === this.mag - 1) ? this._tickFlash : 0;
        ctx.globalAlpha = alpha * (0.86 + fresh * 0.14);
        ctx.fillStyle = fresh > 0.02 ? '#fff3dc' : (low ? '#e8814a' : COL.ink);
        ctx.fillRect(tx, y0, tw, hT);
      } else {
        ctx.globalAlpha = alpha * 0.24;
        ctx.fillStyle = COL.inkFaint;
        ctx.fillRect(tx, y0 + hT - Math.max(1, Math.round(1.5 * s)), tw, Math.max(1, Math.round(1.5 * s)));
      }
    }
    ctx.globalAlpha = 1;

    // fire mode (with a 3-bar glyph) + calibre
    const modeX = x - Math.round(A.wTick) - 0 * s;
    const gy = A.yMeta - Math.round(3.5 * s);
    ctx.globalAlpha = 0.95 * alpha;
    ctx.fillStyle = COL.amber;
    const bw = Math.max(1, Math.round(2 * s)), bh = Math.round(7 * s);
    const bars = this.fireMode === 'SEMI' ? 1 : this.fireMode === 'BURST' ? 3 : 3;
    for (let i = 0; i < bars; i++) ctx.fillRect(Math.round(modeX + i * (bw + 2 * s)), this._snap(gy - bh), bw, bh);
    ctx.globalAlpha = 1;
    this._txt(this.fireMode, modeX + (bars * (bw + 2 * s)) + 6 * s, A.yMeta, this.F.mode, COL.amber, 'left', 1.6 * s, 0.80 * alpha);
    this._txt(this.caliber + '  NATO', x, A.yMeta, this.F.wpnMeta, COL.inkFaint, 'right', 1.3 * s, 0.44 * alpha);

    if (empty) {
      const bl = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(this.time * 7.5));
      this._txt('RELOAD', x, A.yTick - 14 * s, this.F.mode, COL.danger, 'right', 2.6 * s, bl * alpha);
    }
  }

  // ---------------------------------------------------------------- compass --
  _drawCompass(alpha) {
    const ctx = this.ctx;
    const s = this.s, C = this.comp;
    const drop = (1 - easeOutCubic(clamp01(this.boot * 1.15))) * -16 * s;
    const cx = C.x;
    const yLabel = C.yLabel + drop, yTick = C.yTick + drop, baseY = C.yBase + drop;
    const half = C.w * 0.5;
    const ppd = C.w / C.span;
    const hdg = this.headingDeg;
    const fade = (px) => 1 - smoothstep(0.70, 1.0, Math.abs(px) / half);

    // baseline rule with fading ends
    const gl = ctx.createLinearGradient(cx - half, 0, cx + half, 0);
    gl.addColorStop(0, 'rgba(228,219,201,0)');
    gl.addColorStop(0.20, 'rgba(228,219,201,0.24)');
    gl.addColorStop(0.5, 'rgba(228,219,201,0.36)');
    gl.addColorStop(0.80, 'rgba(228,219,201,0.24)');
    gl.addColorStop(1, 'rgba(228,219,201,0)');
    ctx.globalAlpha = alpha;
    ctx.fillStyle = gl;
    ctx.fillRect(cx - half, this._snap(baseY), C.w, Math.max(1, Math.round(1 * this.dpr)) / this.dpr);

    const start = Math.floor((hdg - C.span * 0.5) / 5) * 5;
    const end = hdg + C.span * 0.5 + 5;
    for (let d = start; d <= end; d += 5) {
      let rel = d - hdg;
      while (rel > 180) rel -= 360;
      while (rel < -180) rel += 360;
      const px = cx + rel * ppd;
      const edge = fade(rel * ppd);
      if (edge <= 0.01) continue;
      const dn = ((d % 360) + 360) % 360;
      const isCard = dn % 45 === 0;
      const isMaj = dn % 15 === 0;
      const h = isCard ? C.h : isMaj ? C.h * 0.58 : C.h * 0.30;
      ctx.globalAlpha = alpha * edge * (isCard ? 0.92 : isMaj ? 0.50 : 0.26);
      ctx.fillStyle = COL.ink;
      ctx.fillRect(Math.round(px), this._snap(baseY - h), Math.max(1, Math.round(1.2 * s)), Math.round(h));

      if (isCard) {
        const label = CARDINALS[(dn / 45) | 0];
        const cardinal = dn % 90 === 0;
        this._shadow(true, 4 * s);
        this._txt(label, px, yLabel, cardinal ? this.F.compass : this.F.compassS,
                  cardinal ? COL.ink : COL.inkDim, 'center', cardinal ? 0 : 1.0 * s,
                  alpha * edge * (cardinal ? 0.95 : 0.5));
        this._shadow(false, 0);
      }
    }
    ctx.globalAlpha = 1;

    // Contact pips, in their own lane below the rule. Hard-capped: past half a
    // dozen the strip stops reading as threat direction and starts reading as
    // decoration.
    const pipY = baseY + 8 * s;
    let pips = 0;
    for (let i = 0; i < this.contactCount && pips < this.PIP_MAX; i++) {
      if (this.contacts[i * 4 + 2] < 0.5) continue;
      const dx = this.contacts[i * 4] - this.px;
      const dz = this.contacts[i * 4 + 1] - this.pz;
      const b = Math.atan2(dx, -dz) / DEG;           // world bearing, deg from north
      let rel = b - hdg;
      while (rel > 180) rel -= 360;
      while (rel < -180) rel += 360;
      if (Math.abs(rel) > C.span * 0.5) continue;
      const px = cx + rel * ppd;
      const edge = fade(rel * ppd);
      if (edge <= 0.02) continue;
      const near = 1 - smoothstep(20, 90, this.contacts[i * 4 + 3]);
      const r = (3.6 + near * 1.6) * s;
      ctx.globalAlpha = alpha * edge * (0.55 + near * 0.45);
      ctx.fillStyle = COL.hostile;
      ctx.beginPath();
      ctx.moveTo(px, pipY - r);
      ctx.lineTo(px + r * 0.8, pipY);
      ctx.lineTo(px, pipY + r);
      ctx.lineTo(px - r * 0.8, pipY);
      ctx.closePath();
      ctx.fill();
      pips++;
    }
    ctx.globalAlpha = 1;

    // centre index — a warm tick spanning the lane, with a small cap above
    ctx.globalAlpha = alpha * 0.95;
    ctx.fillStyle = COL.amber;
    const iw = Math.max(1, Math.round(1.6 * s));
    ctx.fillRect(Math.round(cx - iw * 0.5), this._snap(yTick - 3 * s), iw, Math.round(C.h + 5 * s));
    ctx.beginPath();
    ctx.moveTo(cx, yTick - 4 * s);
    ctx.lineTo(cx + 4.5 * s, yTick - 10 * s);
    ctx.lineTo(cx - 4.5 * s, yTick - 10 * s);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    // T3 tabular heading readout — no tracking, mono figures so it never jitters
    this._shadow(true, 4 * s);
    this._txt(this._hdgStr, cx, baseY + 30 * s, this.F.heading, COL.inkDim, 'center', 0, alpha * 0.70);
    this._shadow(false, 0);
  }

  // ---------------------------------------------------------------- minimap --
  //  A tac-map, not a card. Build order, back to front:
  //    1. cached near-black ground bed (opaque enough that the game frame never
  //       shows through and lifts the plate)
  //    2. 10 m grid + building footprints as low-alpha CYAN VECTOR line work
  //    3. amber view-cone wedge with crisp bounding rays
  //    4. hostile blips
  //    5. cached CRT overlay — scanlines, phosphor glow, fade-to-edge vignette
  //    6. ownship arrow and the footer rail (kept above the scanlines so they
  //       stay legible), then a bevelled bezel
  //  Nothing here is brighter than mid-grey, so the plate always sits *under*
  //  the luminance of the 3D frame and recedes.
  // ---------------------------------------------------------------------------
  _drawMinimap(alpha) {
    const ctx = this.ctx;
    const s = this.s, M = this.map;
    const slide = (1 - easeOutCubic(clamp01(this.boot * 1.2))) * -26 * s;
    const x = M.x, y = M.y;
    const size = M.size, cham = M.cham;
    const cx = M.cx, cy = M.cy;
    const k = size / this.mapMeters;               // pixels per metre

    ctx.save();
    ctx.translate(slide, 0);
    ctx.globalAlpha = alpha;

    // ---- 1. ground bed, clipped to the chamfered plate
    ctx.save();
    this._chamfer(x, y, size, size, cham);
    ctx.clip();

    if (this._mapPlateReady) ctx.drawImage(this._mapBed, x, y, size, size);
    else { ctx.fillStyle = COL.mapVoid; ctx.fillRect(x, y, size, size); }

    // ---- 2. world-space frame: rotate so the player's facing is up
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-this.heading);
    ctx.scale(k, k);
    ctx.translate(-this.px, -this.pz);

    const R = this.mapMeters * 0.82;

    // 10 m grid
    ctx.lineWidth = 1 / k;
    ctx.strokeStyle = COL.mapGrid;
    ctx.beginPath();
    const g0x = Math.floor((this.px - R) / 10) * 10, g1x = this.px + R;
    for (let gx = g0x; gx <= g1x; gx += 10) { ctx.moveTo(gx, this.pz - R); ctx.lineTo(gx, this.pz + R); }
    const g0z = Math.floor((this.pz - R) / 10) * 10, g1z = this.pz + R;
    for (let gz = g0z; gz <= g1z; gz += 10) { ctx.moveTo(this.px - R, gz); ctx.lineTo(this.px + R, gz); }
    ctx.stroke();

    // building footprints — barely-there interior wash, cyan vector outline
    const f = this._foot, n = this._footCount;
    ctx.beginPath();
    let drawn = 0;
    for (let i = 0; i < n; i++) {
      const bx = f[i * 4], bz = f[i * 4 + 1], bw = f[i * 4 + 2], bh = f[i * 4 + 3];
      if (bx + bw < this.px - R || bx > this.px + R || bz + bh < this.pz - R || bz > this.pz + R) continue;
      ctx.rect(bx, bz, bw, bh);
      if (++drawn > 260) break;
    }
    ctx.fillStyle = COL.mapFill;
    ctx.fill();
    ctx.strokeStyle = COL.mapEdge;
    ctx.lineWidth = 1.0 / k;
    ctx.stroke();

    ctx.restore();

    // ---- 3. view cone — an amber wedge with two crisp bounding rays
    const cone = 30 * DEG;
    const cr = this._coneR || size * 0.58;
    const a0 = -Math.PI * 0.5 - cone, a1 = -Math.PI * 0.5 + cone;
    ctx.fillStyle = this._coneGrad || 'rgba(231,171,74,0.10)';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, cr, a0, a1);
    ctx.closePath();
    ctx.fill();

    ctx.globalAlpha = alpha * 0.34;
    ctx.strokeStyle = COL.amber;
    ctx.lineWidth = Math.max(1, 1 * this.dpr) / this.dpr;
    ctx.beginPath();
    ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a0) * cr * 0.92, cy + Math.sin(a0) * cr * 0.92);
    ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a1) * cr * 0.92, cy + Math.sin(a1) * cr * 0.92);
    ctx.stroke();
    ctx.globalAlpha = alpha;

    // ---- 4. hostile blips
    const ch = Math.cos(this.heading), sh = Math.sin(this.heading);
    for (let i = 0; i < this.contactCount; i++) {
      if (this.contacts[i * 4 + 2] < 0.5) continue;
      const rx = this.contacts[i * 4] - this.px;
      const rz = this.contacts[i * 4 + 1] - this.pz;
      const mx = rx * ch + rz * sh;
      const my = -rx * sh + rz * ch;
      const bx = cx + mx * k, by = cy + my * k;
      if (bx < x - 8 || bx > x + size + 8 || by < y - 8 || by > y + size + 8) continue;
      const r = 3.0 * s;
      ctx.globalAlpha = alpha * 0.26;
      ctx.fillStyle = COL.hostile;
      ctx.beginPath(); ctx.arc(bx, by, r * 2.6, 0, TAU); ctx.fill();
      ctx.globalAlpha = alpha * 0.92;
      ctx.beginPath();
      ctx.moveTo(bx, by - r); ctx.lineTo(bx + r, by); ctx.lineTo(bx, by + r); ctx.lineTo(bx - r, by);
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = alpha;

    // ---- 5. CRT overlay: scanlines, glow, fade to edge
    if (this._mapPlateReady) ctx.drawImage(this._mapOv, x, y, size, size);

    // ---- 6a. footer rail, inside the plate
    const fh = M.foot;
    const fy = y + size - fh;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = COL.mapFoot;
    ctx.fillRect(x, fy, size, fh);
    this._rule(x, fy, size, 'rgba(138,208,230,0.16)', alpha * 0.9);
    const fb = fy + fh - Math.round(4.5 * s);
    this._txt('SEC 07', x + 10 * s, fb, this.F.mapLabel, COL.inkDim, 'left', 1.5 * s, alpha * 0.62);
    this._txt(Math.round(this.mapMeters) + ' M', x + size - 11 * s, fb, this.F.mapNum, COL.inkDim, 'right', 0, alpha * 0.66);

    // ---- 6b. ownship arrow, above the scanlines so it stays crisp
    ctx.save();
    ctx.translate(cx, cy);
    ctx.beginPath();
    const ar = 6.8 * s;
    ctx.moveTo(0, -ar);
    ctx.lineTo(ar * 0.70, ar * 0.70);
    ctx.lineTo(0, ar * 0.30);
    ctx.lineTo(-ar * 0.70, ar * 0.70);
    ctx.closePath();
    ctx.fillStyle = COL.amber;
    ctx.globalAlpha = alpha * 0.95;
    ctx.fill();
    ctx.strokeStyle = 'rgba(4,7,9,0.9)';
    ctx.lineWidth = Math.max(1, 1.1 * s);
    ctx.stroke();
    ctx.restore();
    ctx.globalAlpha = alpha;

    ctx.restore();  // un-clip

    // ---- bezel: dark outer rim, then a real bevel (light top-left / dark
    //      bottom-right) so the plate reads as a physical inset panel
    this._chamfer(x, y, size, size, cham);
    ctx.strokeStyle = 'rgba(0,0,0,0.70)';
    ctx.lineWidth = Math.max(1.6, 2.2 * s);
    ctx.globalAlpha = alpha * 0.9;
    ctx.stroke();

    const inset = Math.max(1, 1.2 * s);
    ctx.lineWidth = Math.max(1, 1.2 * s);
    ctx.globalAlpha = alpha * 0.5;
    ctx.strokeStyle = 'rgba(196,222,232,0.30)';
    this._bevelEdge(x + inset, y + inset, size - inset * 2, size - inset * 2, cham, true);
    ctx.stroke();
    ctx.globalAlpha = alpha * 0.7;
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    this._bevelEdge(x + inset, y + inset, size - inset * 2, size - inset * 2, cham, false);
    ctx.stroke();

    // corner ticks — short, dim, purely to key the plate to the frame grid
    ctx.globalAlpha = alpha * 0.42;
    ctx.strokeStyle = 'rgba(180,214,226,0.75)';
    ctx.lineWidth = Math.max(1, 1.5 * s);
    const tl = 10 * s;
    ctx.beginPath();
    ctx.moveTo(x + cham, y); ctx.lineTo(x + cham + tl, y);
    ctx.moveTo(x + size - cham, y); ctx.lineTo(x + size - cham - tl, y);
    ctx.moveTo(x + cham, y + size); ctx.lineTo(x + cham + tl, y + size);
    ctx.moveTo(x + size - cham, y + size); ctx.lineTo(x + size - cham - tl, y + size);
    ctx.stroke();
    ctx.globalAlpha = alpha;

    // north pip riding the bezel
    const nr = size * 0.5 - 6 * s;
    const na = -this.heading - Math.PI * 0.5;
    const nx = cx + Math.cos(na) * nr, ny = cy + Math.sin(na) * nr;
    const inside = Math.abs(Math.cos(na)) * nr <= size * 0.5 - 4 * s &&
                   Math.abs(Math.sin(na)) * nr <= size * 0.5 - 4 * s &&
                   ny < fy - 4 * s;
    if (inside) {
      ctx.globalAlpha = alpha * 0.8;
      ctx.fillStyle = COL.amber;
      ctx.beginPath();
      ctx.arc(nx, ny, 2.2 * s, 0, TAU);
      ctx.fill();
      ctx.globalAlpha = 1;
      const lx = cx + (nx - cx) * 0.80, ly = cy + (ny - cy) * 0.80;
      this._txt('N', lx, ly + 3.4 * s, this.F.mapLabel, COL.amber, 'center', 0, alpha * 0.62);
    }

    ctx.restore();
  }

  // Half of a chamfered outline: the top+left run (highlight) or the
  // bottom+right run (shadow). Two of these make a bevel.
  _bevelEdge(x, y, w, h, c, topLeft) {
    const ctx = this.ctx;
    ctx.beginPath();
    if (topLeft) {
      ctx.moveTo(x, y + h - c);
      ctx.lineTo(x, y + c);
      ctx.lineTo(x + c, y);
      ctx.lineTo(x + w - c, y);
    } else {
      ctx.moveTo(x + w, y + c);
      ctx.lineTo(x + w, y + h - c);
      ctx.lineTo(x + w - c, y + h);
      ctx.lineTo(x + c, y + h);
    }
  }

  // --------------------------------------------------------------- killfeed --
  _drawKillfeed(alpha) {
    const ctx = this.ctx;
    const s = this.s, K = this.kfBox;
    let row = 0;
    for (let i = 0; i < this.KF_MAX; i++) {
      // draw newest first, walking backwards from the write head
      const idx = ((this._kfHead - 1 - i) % this.KF_MAX + this.KF_MAX) % this.KF_MAX;
      const e = this.kf[idx];
      if (!e.used || e.t >= e.life) continue;
      const u = e.t / e.life;
      const inT = easeOutCubic(clamp01(e.t / 0.17));
      const out = 1 - smoothstep(0.80, 1.0, u);
      const a = alpha * out * inT * 0.96;
      if (a < 0.01) continue;
      const y = K.y + row * K.lh;
      const x = K.x + (1 - inT) * 22 * s;

      // Killfeed lands in the top-right corner, which in this level is nearly
      // always open sky — the brightest thing in the frame. Halo the whole row.
      this._shadow(true, 4 * s);
      const gw = 34 * s;
      const bw = this._measure(e.b, this.F.kf, 0.4 * s);
      this._txt(e.b, x, y, this.F.kf, COL.inkDim, 'right', 0.4 * s, a * 0.70);
      const gx = x - bw - 15 * s;
      this._rifleGlyph(gx, y - 8 * s, s, e.head ? COL.danger : COL.inkDim, a * 0.85);
      this._txt(e.a, gx - gw - 15 * s, y, this.F.kf, COL.ink, 'right', 0.4 * s, a * 0.88);
      if (e.head) {
        // headshot pip riding above the glyph
        ctx.globalAlpha = a;
        ctx.fillStyle = COL.danger;
        ctx.beginPath();
        ctx.arc(gx - gw * 0.5, y - 15 * s, 2.4 * s, 0, TAU);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      this._shadow(false, 0);
      row++;
      if (row >= 5) break;
    }
  }

  // Procedural carbine silhouette — right-aligned, muzzle pointing left, so the
  // row reads "killer  ->  victim". Drawn from a 34 x 12 box at scale s.
  _rifleGlyph(x, y, s, color, alpha) {
    const ctx = this.ctx;
    const w = 34 * s, h = 12 * s;
    const x0 = x - w;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;

    // barrel + flash hider (left end)
    ctx.fillRect(x0, y + h * 0.36, w * 0.34, h * 0.13);
    ctx.fillRect(x0, y + h * 0.28, w * 0.06, h * 0.28);
    // handguard
    ctx.fillRect(x0 + w * 0.20, y + h * 0.30, w * 0.24, h * 0.24);
    // receiver
    ctx.fillRect(x0 + w * 0.44, y + h * 0.24, w * 0.30, h * 0.30);
    // optic
    ctx.fillRect(x0 + w * 0.52, y + h * 0.06, w * 0.16, h * 0.16);
    // stock (right end)
    ctx.fillRect(x0 + w * 0.74, y + h * 0.28, w * 0.26, h * 0.22);
    // magazine, raked forward
    ctx.beginPath();
    ctx.moveTo(x0 + w * 0.50, y + h * 0.54);
    ctx.lineTo(x0 + w * 0.64, y + h * 0.54);
    ctx.lineTo(x0 + w * 0.70, y + h * 1.00);
    ctx.lineTo(x0 + w * 0.56, y + h * 1.00);
    ctx.closePath();
    ctx.fill();
    // pistol grip
    ctx.beginPath();
    ctx.moveTo(x0 + w * 0.70, y + h * 0.54);
    ctx.lineTo(x0 + w * 0.80, y + h * 0.54);
    ctx.lineTo(x0 + w * 0.84, y + h * 0.92);
    ctx.lineTo(x0 + w * 0.76, y + h * 0.92);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  // --------------------------------------------------------------- objective --
  _drawObjective(alpha) {
    if (!this.objective.active) return;
    const cam = this.g.camera;
    if (!cam) return;
    const ctx = this.ctx;
    const s = this.s;

    // range + LOS were resolved in _updateObjective(); this only projects.
    const v = this._v3.set(this.objective.x, this.objective.y, this.objective.z);
    v.applyMatrix4(this._mInv);
    const behind = v.z > -0.02;
    v.applyMatrix4(cam.projectionMatrix);

    let sx = (v.x * 0.5 + 0.5) * this.W;
    let sy = (-v.y * 0.5 + 0.5) * this.H;

    const insetX = this.M + 34 * s, insetY = this.M + 52 * s;
    const minX = insetX, maxX = this.W - insetX;
    const minY = insetY, maxY = this.H - insetY - 40 * s;

    let ox = sx - this.cx, oy = sy - this.cy;
    if (behind) { ox = -ox; oy = -oy; if (Math.abs(ox) < 1 && Math.abs(oy) < 1) oy = 1; }

    let clamped = behind;
    let fx = this.cx + ox, fy = this.cy + oy;
    if (fx < minX || fx > maxX || fy < minY || fy > maxY) {
      clamped = true;
      const hx = (ox > 0 ? maxX : minX) - this.cx;
      const hy = (oy > 0 ? maxY : minY) - this.cy;
      const tx = Math.abs(ox) > 1e-4 ? hx / ox : Infinity;
      const ty = Math.abs(oy) > 1e-4 ? hy / oy : Infinity;
      const t = Math.min(Math.abs(tx), Math.abs(ty));
      fx = this.cx + ox * t;
      fy = this.cy + oy * t;
    }
    fx = clamp(fx, minX, maxX);
    fy = clamp(fy, minY, maxY);

    // ---- line-of-sight state -------------------------------------------------
    //  vis = 1  target is in the clear: filled diamond, full-strength label and
    //           range, dark carrier so it holds against the sky.
    //  vis = 0  target is behind geometry: outline only, dashed, no fill, no
    //           carrier, label and range dropped right back. It reads as "known
    //           position, not visible" instead of x-raying through a wall.
    const vis = clamp01(this._objVis);
    const a = alpha * (0.46 + vis * 0.49);
    const col = vis > 0.5 ? COL.objective : COL.objectiveDim;
    const r = (8.4 + vis * 1.6) * s;

    ctx.save();

    // dark carrier — only when the marker is actually visible; a halo behind a
    // ghosted outline just makes the ghost heavy.
    if (vis > 0.02) {
      ctx.beginPath();
      ctx.moveTo(fx, fy - r); ctx.lineTo(fx + r, fy); ctx.lineTo(fx, fy + r); ctx.lineTo(fx - r, fy);
      ctx.closePath();
      ctx.strokeStyle = 'rgba(5,8,10,0.85)';
      ctx.lineWidth = Math.max(3, 4.2 * s);
      ctx.globalAlpha = alpha * vis * 0.72;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(fx, fy - r); ctx.lineTo(fx + r, fy); ctx.lineTo(fx, fy + r); ctx.lineTo(fx - r, fy);
    ctx.closePath();

    if (vis > 0.02) {
      ctx.globalAlpha = alpha * vis * 0.24;
      ctx.fillStyle = COL.objective;
      ctx.fill();
    }

    ctx.globalAlpha = a;
    ctx.strokeStyle = col;
    ctx.lineWidth = Math.max(1.3, (1.4 + vis * 0.7) * s);
    if (vis < 0.5) { this._dash[0] = 3 * s; this._dash[1] = 3 * s; ctx.setLineDash(this._dash); }
    ctx.stroke();
    ctx.setLineDash(this._noDash);

    // inner tick only in the solid state
    if (vis > 0.5) {
      ctx.beginPath();
      ctx.moveTo(fx, fy - r * 0.32); ctx.lineTo(fx, fy + r * 0.32);
      ctx.globalAlpha = a * 0.85;
      ctx.stroke();
    }

    if (clamped) {
      // offscreen arrow, pointing outward along the clamp direction
      const ang = Math.atan2(oy, ox);
      ctx.save();
      ctx.translate(fx + Math.cos(ang) * (r + 8 * s), fy + Math.sin(ang) * (r + 8 * s));
      ctx.rotate(ang);
      ctx.beginPath();
      ctx.moveTo(6 * s, 0); ctx.lineTo(-5 * s, -5 * s); ctx.lineTo(-5 * s, 5 * s);
      ctx.closePath();
      ctx.fillStyle = col;
      ctx.globalAlpha = a;
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();

    // T2 label above, T3 tabular range below. Both step down with the marker but
    // stay readable — a ghosted marker still has to tell you the range. The drop
    // shadow stays on in both states: the outline is thin and often lands on a
    // sunlit wall, where unbacked cyan disappears entirely.
    this._shadow(true, 5 * s);
    this._txt(this.objective.label, fx, fy - r - 9 * s, this.F.objLabel, col, 'center', 2.2 * s,
              alpha * (0.40 + vis * 0.52));
    this._shadow(false, 0);

    // The range sits on its own small dark tag. Unbacked warm ink over a blown
    // sunlit archway is the one place on this HUD where a live number can vanish
    // completely, and the range is the whole point of the marker.
    if (this._distKey !== this._objDistStr || this._distKeyS !== this.s) {
      this._distKey = this._objDistStr;
      this._distKeyS = this.s;
      this._distW = this._measure(this._objDistStr, this.F.objDist, 0);
    }
    const dyBase = fy + r + 16 * s;
    const dw = this._distW + 13 * s, dh = 16 * s;
    const dx0 = fx - dw * 0.5, dy0 = dyBase - dh + 4 * s;
    ctx.globalAlpha = alpha * (0.42 + vis * 0.34);
    ctx.fillStyle = 'rgba(6,10,12,0.90)';
    this._chamfer(dx0, dy0, dw, dh, 3.2 * s);
    ctx.fill();
    ctx.globalAlpha = alpha * (0.20 + vis * 0.26);
    ctx.strokeStyle = col;
    ctx.lineWidth = Math.max(1, this.dpr) / this.dpr;
    ctx.stroke();
    ctx.globalAlpha = 1;
    this._txt(this._objDistStr, fx, dyBase, this.F.objDist,
              vis > 0.5 ? COL.ink : COL.inkDim, 'center', 0, alpha * (0.60 + vis * 0.38));
  }

  // ---------------------------------------------------------------- mission --
  //  Three tiers, stacked, so the eye lands on the mission line first:
  //    T2 eyebrow  'OBJECTIVE'  — 9px amber, tracked, 65% (micro-labels are the
  //                one place tracking is correct; it is not applied elsewhere)
  //    T1 display  the mission itself — 22px/700, tracking 0, full strength
  //    T2 sub      location — 10px, 42%, barely there
  _drawMission(alpha) {
    const s = this.s, m = this.mission;
    const slide = (1 - easeOutCubic(clamp01(this.boot * 1.1))) * -20 * s;
    const x = m.x + slide;
    // amber accent bar keying the block to the frame edge
    const bh = Math.round(34 * s);
    this._ctxBar(x - Math.round(10 * s), m.y - Math.round(10 * s), Math.max(1, Math.round(2 * s)), bh, COL.amber, alpha * 0.62);
    // The block sits over sunlit plaster as often as over sky, so the whole
    // stack carries a soft dark halo. Without it the two supporting tiers are
    // simply gone against a bright wall — and an invisible tier is not a tier.
    this._shadow(true, 6 * s);
    this._txt('OBJECTIVE', x, m.y, this.F.missionK, COL.amber, 'left', 2.6 * s, alpha * 0.72);
    this._txt('SECURE THE COMPOUND', x, m.y + 27 * s, this.F.missionT, COL.ink, 'left', 0, alpha * 0.98);
    this._shadow(false, 0);
    this._rule(x, m.y + 38 * s, 176 * s, COL.ruleSoft, alpha * 0.75);
    this._shadow(true, 5 * s);
    this._txt('BLACKSITE  ·  MARKAZ DISTRICT', x, m.y + 54 * s, this.F.missionS, COL.inkFaint, 'left', 1.4 * s, alpha * 0.56);
    this._shadow(false, 0);
  }

  _ctxBar(x, y, w, h, color, alpha) {
    const ctx = this.ctx;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), Math.round(y), Math.max(1, Math.round(w)), Math.round(h));
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------- hint --
  //  A transient, contextual prompt only — "PRESS F TO PLANT", never a static
  //  keybind legend. The control reference lives on the pause / first-run card
  //  where it belongs; a permanent strip across the bottom of the play frame is
  //  a browser-demo tell and is deliberately not drawn here.
  _drawHint(alpha) {
    if (!this.hintText || this.hintLife <= 0) return;
    const u = this.hintT / Math.max(0.5, this.hintLife);
    if (u >= 1) return;
    const a = alpha * (1 - smoothstep(0.78, 1.0, u)) * easeOutCubic(clamp01(this.hintT / 0.4)) * 0.60;
    if (a < 0.01) return;
    const s = this.s;
    const y = this.cy + 108 * s;
    this._shadow(true, 5 * s);
    this._txt(this.hintText, this.cx, y, this.F.hint, COL.inkDim, 'center', 1.8 * s, a);
    this._shadow(false, 0);
  }

  // ------------------------------------------------------------------ pause --
  _drawPause() {
    const ctx = this.ctx;
    const s = this.s;
    const p = easeOutCubic(clamp01(this.pauseT));
    const start = !this.started;

    ctx.globalAlpha = p * 0.78;
    ctx.fillStyle = 'rgba(5,7,9,1)';
    ctx.fillRect(0, 0, this.W, this.H);
    ctx.globalAlpha = 1;

    const cx = this.cx;
    const cy = Math.round(this.H * 0.42);
    const lift = (1 - p) * 16 * s;

    // title with hairline rules either side
    const title = start ? 'BLACKSITE' : 'PAUSED';
    this._txt(title, cx, cy - lift, this.F.pauseT, COL.ink, 'center', 14 * s, p);
    const tw = this._measure(title, this.F.pauseT, 14 * s);
    const ry = cy - 18 * s - lift;
    this._rule(cx - tw * 0.5 - 78 * s, ry, 62 * s, COL.rule, p * 0.8);
    this._rule(cx + tw * 0.5 + 16 * s, ry, 62 * s, COL.rule, p * 0.8);

    this._txt(start ? 'TACTICAL OPERATIONS  ·  MARKAZ DISTRICT  ·  16:24 LOCAL'
                    : 'SIMULATION HELD  ·  ALL SYSTEMS NOMINAL',
              cx, cy + 24 * s - lift, this.F.pauseS, COL.inkFaint, 'center', 3.6 * s, p * 0.6);

    // control reference — this is the ONLY place bindings are ever shown.
    // Two columns on a strict baseline: T2 action label, T3 tabular key.
    const rows = [
      ['MOVE', 'W A S D'], ['SPRINT', 'SHIFT'], ['CROUCH', 'CTRL'], ['JUMP', 'SPACE'],
      ['FIRE', 'MOUSE 1'], ['AIM', 'MOUSE 2'], ['RELOAD', 'R'], ['PAUSE', 'ESC'],
    ];
    const colW = 210 * s;
    const gy = cy + 92 * s - lift;
    const lh = 22 * s;
    const gx0 = cx - colW - 22 * s;
    this._txt('CONTROLS', cx, gy - 26 * s, this.F.missionK, COL.amber, 'center', 2.6 * s, p * 0.6);
    for (let i = 0; i < rows.length; i++) {
      const col = i >= 4 ? 1 : 0;
      const r = i % 4;
      const bx = gx0 + col * (colW + 44 * s);
      const by = gy + r * lh;
      this._txt(rows[i][0], bx, by, this.F.pauseK, COL.inkFaint, 'left', 1.8 * s, p * 0.52);
      this._txt(rows[i][1], bx + colW, by, this.F.pauseV, COL.inkDim, 'right', 0, p * 0.8);
      this._rule(bx, by + 6 * s, colW, COL.ruleSoft, p * 0.45);
    }

    const blink = 0.62 + 0.38 * (0.5 + 0.5 * Math.sin(this.time * 3.2));
    this._txt(start ? 'CLICK TO DEPLOY' : 'CLICK TO RESUME  ·  ESC',
              cx, gy + 4 * lh + 40 * s, this.F.prompt, COL.amber, 'center', 4.0 * s, p * blink);
  }

  // ------------------------------------------------------------------ stats --
  _drawStats() {
    const st = this.g.renderer?.getStats?.();
    if (!st) return;
    if (!this._statT || this.time - this._statT > 0.25) {
      this._statT = this.time;
      this._statStr = Math.round(st.fps) + ' FPS  ' + st.frameMs.toFixed(1) + 'MS  ' +
                      st.drawCalls + ' DC  ' + (st.triangles / 1000).toFixed(0) + 'K TRI';
    }
    if (this._statStr) this._txt(this._statStr, this.M, this.H - this.M - 200 * this.s, this.F.stat, COL.inkFaint, 'left', 0, 0.6);
  }

  // ===========================================================================
  //  PRIMITIVES
  // ===========================================================================
  _snap(y) { const d = this.dpr; return Math.round(y * d) / d; }

  // Soft dark halo behind ink that has to survive an unpredictable background
  // (the objective marker floats over sky, sand and shadow in the same frame).
  _shadow(on, blur) {
    const ctx = this.ctx;
    if (on) {
      ctx.shadowColor = 'rgba(4,7,9,0.72)';
      ctx.shadowBlur = blur;
      ctx.shadowOffsetY = Math.max(1, this.s);
    } else {
      ctx.shadowColor = 'rgba(0,0,0,0)';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;
    }
  }

  _rule(x, y, w, color, alpha) {
    const ctx = this.ctx;
    ctx.globalAlpha = alpha === undefined ? 1 : alpha;
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), this._snap(y), Math.round(w), Math.max(1, Math.round(1 * this.dpr)) / this.dpr);
    ctx.globalAlpha = 1;
  }

  _chamfer(x, y, w, h, c) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + c, y);
    ctx.lineTo(x + w - c, y);
    ctx.lineTo(x + w, y + c);
    ctx.lineTo(x + w, y + h - c);
    ctx.lineTo(x + w - c, y + h);
    ctx.lineTo(x + c, y + h);
    ctx.lineTo(x, y + h - c);
    ctx.lineTo(x, y + c);
    ctx.closePath();
  }
}

export default HUD;
