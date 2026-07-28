// ─────────────────────────────────────────────────────────────────────────────
//  TouchControls.js — 모바일 / 태블릿 터치 조작
//
//  Controller 는 키보드 + 포인터 락 마우스만 받는다. 이 모듈은 그 입력 경로에
//  터치를 직접 주입한다. Controller 를 수정하지 않는다.
//
//   · 왼쪽 반 : 가상 조이스틱  →  controller.keys.f/b/l/r + sprint
//   · 오른쪽 반 : 드래그       →  controller._mouseDX/_mouseDY (포인터 락 우회)
//   · 버튼      : 점프 / 앉기 / 시점전환 (+ 프로젝트별 액션 버튼)
//
//  터치 기기에서만 UI 를 띄운다. 데스크톱에서는 아무것도 하지 않는다.
//  (?touch=1 쿼리로 강제 표시 — 자동 테스트용)
// ─────────────────────────────────────────────────────────────────────────────

const DEAD = 0.16;          // 조이스틱 데드존 (0..1)
const SPRINT_AT = 0.86;     // 이 이상 밀면 달리기
const LOOK_SCALE = 1.45;    // 드래그 → 시야 회전 배율

export class TouchControls {
  constructor(game) {
    this.g = game;
    this.enabled = false;
    this.moveId = null;      // 조이스틱을 잡고 있는 터치 id
    this.lookId = null;      // 시야 드래그 터치 id
    this.mx = 0; this.my = 0;
    this.lookX = 0; this.lookY = 0;
    this.held = { f: false, b: false, l: false, r: false, sprint: false, crouch: false };
  }

  async init() {
    const forced = /[?&]touch=1/.test(location.search);
    const isTouch = forced ||
      (('ontouchstart' in window) || (navigator.maxTouchPoints || 0) > 0) &&
      !window.matchMedia('(pointer: fine)').matches;
    if (!isTouch) return;              // 데스크톱이면 조용히 빠진다
    this.enabled = true;
    this._buildUI();
    this._bind();
  }

  // ── UI ────────────────────────────────────────────────────────────────────
  _buildUI() {
    const el = document.createElement('div');
    el.id = 'touch';
    el.innerHTML = `
      <style>
        #touch { position:fixed; inset:0; z-index:30; touch-action:none;
                 -webkit-user-select:none; user-select:none; -webkit-tap-highlight-color:transparent;
                 font-family:"Pretendard","Malgun Gothic",system-ui,sans-serif; }
        #touch .zone { position:absolute; top:0; bottom:0; }
        #tz-move { left:0; width:46%; }
        #tz-look { right:0; width:54%; }
        #tstick { position:absolute; width:150px; height:150px; margin:-75px 0 0 -75px;
                  border-radius:50%; border:2px solid rgba(255,255,255,.24);
                  background:rgba(0,0,0,.20); opacity:0; transition:opacity .16s; }
        #tstick.on { opacity:1; }
        #tknob { position:absolute; left:50%; top:50%; width:62px; height:62px;
                 margin:-31px 0 0 -31px; border-radius:50%;
                 background:rgba(255,255,255,.30); border:2px solid rgba(255,255,255,.55); }
        #tbtns { position:absolute; right:18px; bottom:22px; display:flex; gap:12px; align-items:flex-end; }
        #tbtns .b { width:76px; height:76px; border-radius:50%; display:grid; place-items:center;
                    background:rgba(0,0,0,.34); border:2px solid rgba(255,255,255,.34);
                    color:#fff; font-size:14px; font-weight:800; }
        #tbtns .b.sm { width:60px; height:60px; font-size:12px; }
        #tbtns .b:active, #tbtns .b.act { background:rgba(255,201,77,.75); color:#12161c;
                                          border-color:rgba(255,201,77,.95); }
        #tview { position:absolute; right:18px; top:calc(env(safe-area-inset-top,0px) + 152px);
                 padding:10px 15px; border-radius:999px; background:rgba(0,0,0,.36);
                 border:1px solid rgba(255,255,255,.30); color:#fff; font-size:12.5px; font-weight:700; }
        #thint { position:absolute; left:50%; bottom:8px; transform:translateX(-50%);
                 font-size:11px; color:rgba(255,255,255,.45); }
        @media (max-height:430px){ #tbtns .b { width:62px; height:62px; } }
      </style>
      <div class="zone" id="tz-move"><div id="tstick"><div id="tknob"></div></div></div>
      <div class="zone" id="tz-look"></div>
      <div id="tview">시점 전환</div>
      <div id="tbtns">
        <div class="b sm" id="tb-crouch">앉기</div>
        <div class="b" id="tb-jump">점프</div>
      </div>
      <div id="thint">왼쪽 드래그 — 이동  ·  오른쪽 드래그 — 시야</div>`;
    document.body.appendChild(el);

    this.root  = el;
    this.zMove = el.querySelector('#tz-move');
    this.zLook = el.querySelector('#tz-look');
    this.stick = el.querySelector('#tstick');
    this.knob  = el.querySelector('#tknob');
    this.bJump = el.querySelector('#tb-jump');
    this.bCrouch = el.querySelector('#tb-crouch');
    this.bView = el.querySelector('#tview');
  }

  // ── 입력 ──────────────────────────────────────────────────────────────────
  _bind() {
    const c = () => this.g?.controller;

    const key = (code, down) => {
      const ctl = c(); if (!ctl) return;
      const ev = { code, repeat: false, preventDefault() {}, stopPropagation() {} };
      if (down) ctl._onKeyDown?.(ev); else ctl._onKeyUp?.(ev);
    };
    this._key = key;

    // 대화가 떠 있으면 조작 대신 대사 넘기기
    const paused = () => !!this.g?.story?.paused;

    // ── 이동 조이스틱
    const moveStart = (t) => {
      this.moveId = t.identifier;
      this.mx = t.clientX; this.my = t.clientY;
      this.stick.style.left = t.clientX + 'px';
      this.stick.style.top  = t.clientY + 'px';
      this.stick.classList.add('on');
    };
    const moveMove = (t) => {
      const R = 62;
      let dx = t.clientX - this.mx, dy = t.clientY - this.my;
      const d = Math.hypot(dx, dy);
      if (d > R) { dx = dx / d * R; dy = dy / d * R; }
      this.knob.style.transform = `translate(${dx}px, ${dy}px)`;
      const nx = dx / R, ny = dy / R;
      const mag = Math.min(1, Math.hypot(nx, ny));
      const want = {
        f: ny < -DEAD, b: ny > DEAD, l: nx < -DEAD, r: nx > DEAD,
        sprint: mag > SPRINT_AT, crouch: this.held.crouch,
      };
      for (const k of ['f', 'b', 'l', 'r']) {
        if (want[k] !== this.held[k]) {
          this.held[k] = want[k];
          key({ f: 'KeyW', b: 'KeyS', l: 'KeyA', r: 'KeyD' }[k], want[k]);
        }
      }
      if (want.sprint !== this.held.sprint) {
        this.held.sprint = want.sprint;
        key('ShiftLeft', want.sprint);
      }
    };
    const moveEnd = () => {
      this.moveId = null;
      this.knob.style.transform = '';
      this.stick.classList.remove('on');
      for (const k of ['f', 'b', 'l', 'r']) {
        if (this.held[k]) { this.held[k] = false; key({ f: 'KeyW', b: 'KeyS', l: 'KeyA', r: 'KeyD' }[k], false); }
      }
      if (this.held.sprint) { this.held.sprint = false; key('ShiftLeft', false); }
    };

    // ── 시야 드래그 : 포인터 락을 쓰지 않고 델타를 직접 누적한다
    const lookMove = (t) => {
      const ctl = c(); if (!ctl) return;
      const dx = (t.clientX - this.lookX) * LOOK_SCALE;
      const dy = (t.clientY - this.lookY) * LOOK_SCALE;
      this.lookX = t.clientX; this.lookY = t.clientY;
      ctl._mouseDX = (ctl._mouseDX || 0) + dx;
      ctl._mouseDY = (ctl._mouseDY || 0) + dy;
      ctl._wake?.();
    };

    const onStart = (e) => {
      for (const t of e.changedTouches) {
        if (paused()) { this.g.story.advance(); continue; }
        const left = t.clientX < innerWidth * 0.46;
        if (left && this.moveId === null) moveStart(t);
        else if (!left && this.lookId === null) {
          this.lookId = t.identifier; this.lookX = t.clientX; this.lookY = t.clientY;
        }
      }
      if (e.cancelable) e.preventDefault();
    };
    const onMove = (e) => {
      if (paused()) { if (e.cancelable) e.preventDefault(); return; }
      for (const t of e.changedTouches) {
        if (t.identifier === this.moveId) moveMove(t);
        else if (t.identifier === this.lookId) lookMove(t);
      }
      if (e.cancelable) e.preventDefault();
    };
    const onEnd = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === this.moveId) moveEnd();
        else if (t.identifier === this.lookId) this.lookId = null;
      }
      if (e.cancelable) e.preventDefault();
    };

    for (const [ev, fn] of [['touchstart', onStart], ['touchmove', onMove],
                            ['touchend', onEnd], ['touchcancel', onEnd]]) {
      this.root.addEventListener(ev, fn, { passive: false });
    }

    // ── 버튼
    const hold = (node, code, toggle) => {
      if (!node) return;
      node.addEventListener('touchstart', (e) => {
        e.stopPropagation(); if (e.cancelable) e.preventDefault();
        if (paused()) { this.g.story.advance(); return; }
        if (toggle) {
          this.held.crouch = !this.held.crouch;
          node.classList.toggle('act', this.held.crouch);
          key(code, this.held.crouch);
        } else { node.classList.add('act'); key(code, true); }
      }, { passive: false });
      node.addEventListener('touchend', (e) => {
        e.stopPropagation(); if (e.cancelable) e.preventDefault();
        if (!toggle) { node.classList.remove('act'); key(code, false); }
      }, { passive: false });
    };
    hold(this.bJump, 'Space', false);
    hold(this.bCrouch, 'ControlLeft', true);

    if (this.bView) {
      this.bView.addEventListener('touchstart', (e) => {
        e.stopPropagation(); if (e.cancelable) e.preventDefault();
        this.g?.thirdperson?.toggle?.();
      }, { passive: false });
    }
  }

  update() {
    // Story 는 이 모듈보다 나중에 만들어진다. UI 가 준비되면 한 번만 문구를 바꾼다.
    if (!this.enabled || this._fixedHint) return;
    const n = document.getElementById('st-next');
    if (n) { n.textContent = '화면을 탭하면 다음'; this._fixedHint = true; }
  }
  resize() {}
}

export default TouchControls;
