// ─────────────────────────────────────────────────────────────────────────────
//  Story.js — 매불쇼 신입 작가 생존기 : 내러티브 레이어
//
//  이 모듈은 BLACKSITE 엔진(렌더링·레벨·이동) 위에 얹히는 스토리 계층이다.
//  전투는 원본 그대로 두고, 다음을 담당한다.
//
//   · 원본 영문 HUD를 숨기고 한글 DOM HUD를 새로 그린다 (목표·시간·탄약·카드)
//   · 월드에 '사연 카드' 수집품을 배치하고 근접 획득을 처리한다
//   · 대화 박스를 띄우고 그동안 플레이어 입력을 얼어붙게 한다 (튜토리얼 진행)
//   · 남은 방송 시간 카운트다운과 엔딩을 관리한다
//
//  전투는 원본 BLACKSITE 엔진 그대로다 (사격·적 AI·탄도).
//
//  ⚠ 이 게임에 등장하는 모든 대사는 창작입니다. 실제 인물의 발언이 아닙니다.
//    등장하는 상대 팀은 원본 엔진의 익명 병사이며, 실존 인물을 형상화하거나
//    공격 대상으로 삼지 않습니다. 설정상 '서바이벌 특집 촬영'입니다.
// ─────────────────────────────────────────────────────────────────────────────

const CARD_COLOR = 0xffc94d;
const GOAL_COLOR = 0x4dd2ff;

// 수집품 배치 — 카메라 프리셋으로 검증된 보행 가능 구역 안에 둔다.
const CARDS = [
  { id: 1, x:   3.0, z:  13.5, label: '사연 카드 ①' },
  { id: 2, x:  -6.2, z:   8.0, label: '사연 카드 ②' },
  { id: 3, x:   9.8, z:   1.5, label: '사연 카드 ③' },
  { id: 4, x:  -7.2, z:  -4.6, label: '사연 카드 ④' },
  { id: 5, x:   0.6, z:  -0.5, label: '사연 카드 ⑤' },
];

const GOAL = { x: 0, z: 17.5 };

// ── 대사 데이터 ──────────────────────────────────────────────────────────────
const INTRO = [
  ['PD',       '야, 신입! 첫 출근인데 하필 오늘이 특집이야.'],
  ['PD',       '여기 \"매불쇼 서바이벌 특집\" 촬영장이다. 페인트탄이라 안 죽어. 아프기만 해.'],
  ['최욱',     '어이 신입. 근데 진짜 문제는 그게 아니야. 오늘 쓸 사연 카드가 세트에 다 흩어졌어.'],
  ['최욱',     '상대 팀이 세트 안에 쫙 깔렸다. 알아서 뚫고 다섯 장 회수해.'],
  ['PD',       '좌클릭 사격, 우클릭 조준, R 재장전. 카드 근처로 가면 알아서 주워진다.'],
  ['PD',       '방송까지 4분. 가자!'],
];

const PICKUP = {
  1: [['최욱', '첫 장 확보. …\'윗집에 편지를 썼더니 답장이 왔습니다\'? 오, 이거 물건인데.']],
  2: [['신입 작가', '두 번째 카드요! 근데 이거 손글씨인데요?'],
      ['PD',       '그거 최욱 씨가 새벽에 직접 쓴 거야. 읽지 말고 그냥 가져와.']],
  3: [['PD',   '그거 사이다 헤드라인 큐시트야! 그거 없으면 1부가 안 열려.'],
      ['최욱', '잘했어. 그건 진짜 없으면 안 되는 거야.']],
  4: [['최욱', '거기 안쪽이지? 조명 꺼진 데. 무서우면 뛰어. 대신 카드는 들고 뛰어.']],
  5: [['PD',   '마지막! 다섯 장 다 모았다.'],
      ['최욱', '좋아. 부스로 돌아와. 파란 표시 보이지?']],
};

const OUTRO = [
  ['PD',       '복귀 확인. 카드 다섯 장 전부 들어왔습니다.'],
  ['최욱',     '딱 맞췄네. 신입, 오늘 첫날치고 나쁘지 않았어.'],
  ['최욱',     '자, 그럼 시작합니다. 압도적 재미 — 매불쇼!'],
];

const TIMEUP = [
  ['PD',   '큐 들어갔어! 방송 이미 시작했다고!'],
  ['최욱', '괜찮아, 내가 오프닝 늘릴게. 신입 너는 카드나 계속 가져와.'],
];

export class Story {
  constructor(game) {
    this.g = game;
    this.THREE = game.THREE;

    this.state = 'boot';          // boot | intro | hunt | return | done
    this.collected = 0;
    this.total = CARDS.length;
    this.timeLeft = 240;          // 방송까지 남은 시간(초) — 0이 되어도 진행은 계속된다
    this.timeUpFired = false;
    this.elapsed = 0;

    this.paused = false;
    this.queue = [];
    this.line = null;
    this.typed = 0;
    this.typeT = 0;

    this.cards = [];
    this.goalMesh = null;
    this._tmp = new this.THREE.Vector3();
  }

  async init() {
    this._setupOverlay();
    this._buildUI();
    this._buildPickups();
    this._bindInput();
    // 인트로는 플레이어가 실제로 게임에 들어온 뒤(포인터 락 획득) 시작한다.
    this.state = 'boot';
  }

  // ── 표시 계층 준비 ─────────────────────────────────────────────────────────
  //  전투(사격·적 AI·탄도)는 원본 엔진 그대로 둔다. 여기서는 영문 HUD만 걷어내고
  //  한글 HUD 를 대신 그린다. 사격 입력은 건드리지 않는다.
  _setupOverlay() {
    // 플레이어가 게임 안으로 들어왔는지 기록 (포인터 락이 막힌 브라우저 대비).
    // 어떤 것도 막지 않고 표시만 남기므로 반드시 첫 번째 리스너여야 한다.
    this._entered = false;
    const mark = () => { this._entered = true; };
    for (const t of ['mousedown', 'pointerdown', 'touchstart', 'keydown']) {
      window.addEventListener(t, mark, true);
    }
    // 원본 영문 HUD(캔버스)를 숨긴다 — 조준점·탄약까지 한글 HUD 로 직접 그린다
    const hud = document.getElementById('hud');
    if (hud) hud.style.display = 'none';
  }

  // ── 입력 ──────────────────────────────────────────────────────────────────
  _bindInput() {
    this._block = (e) => {
      if (!this.paused) return;
      if (e.type === 'keydown' && (e.code === 'Space' || e.code === 'Enter' || e.code === 'KeyE')) {
        this.advance();
      } else if (e.type === 'mousedown') {
        this.advance();
      }
      e.stopImmediatePropagation();
      if (e.cancelable) e.preventDefault();
    };
    for (const t of ['keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove', 'wheel']) {
      window.addEventListener(t, this._block, true);
    }
  }

  // ── 수집품 ────────────────────────────────────────────────────────────────
  _buildPickups() {
    const T = this.THREE;
    const cardGeo = new T.BoxGeometry(0.42, 0.58, 0.03);
    const ringGeo = new T.TorusGeometry(0.52, 0.022, 8, 40);

    for (const c of CARDS) {
      const grp = new T.Group();
      grp.position.set(c.x, 1.15, c.z);

      const mat = new T.MeshStandardMaterial({
        color: CARD_COLOR, emissive: CARD_COLOR, emissiveIntensity: 1.5,
        roughness: 0.55, metalness: 0.0,
      });
      const card = new T.Mesh(cardGeo, mat);
      card.castShadow = false; card.receiveShadow = false;
      grp.add(card);

      const ring = new T.Mesh(ringGeo, new T.MeshBasicMaterial({
        color: CARD_COLOR, transparent: true, opacity: 0.55,
      }));
      ring.rotation.x = Math.PI / 2;
      grp.add(ring);

      // 멀리서도 보이도록 세로 광선 기둥
      const beam = new T.Mesh(
        new T.CylinderGeometry(0.06, 0.06, 6, 8, 1, true),
        new T.MeshBasicMaterial({ color: CARD_COLOR, transparent: true, opacity: 0.13,
                                  side: T.DoubleSide, depthWrite: false })
      );
      beam.position.y = 2.6;
      grp.add(beam);

      this.g.scene.add(grp);
      this.cards.push({ ...c, grp, card, ring, taken: false, phase: Math.random() * 6.28 });
    }

    // 복귀 지점 표시 — 다 모으기 전까지는 숨긴다
    const goal = new this.THREE.Group();
    goal.position.set(GOAL.x, 0.06, GOAL.z);
    const disc = new this.THREE.Mesh(
      new this.THREE.CylinderGeometry(1.5, 1.5, 0.04, 40),
      new this.THREE.MeshBasicMaterial({ color: GOAL_COLOR, transparent: true, opacity: 0.35 })
    );
    goal.add(disc);
    const pillar = new this.THREE.Mesh(
      new this.THREE.CylinderGeometry(0.10, 0.10, 9, 10, 1, true),
      new this.THREE.MeshBasicMaterial({ color: GOAL_COLOR, transparent: true, opacity: 0.16,
                                         side: this.THREE.DoubleSide, depthWrite: false })
    );
    pillar.position.y = 4.5;
    goal.add(pillar);
    goal.visible = false;
    this.g.scene.add(goal);
    this.goalMesh = goal;
  }

  // ── UI ────────────────────────────────────────────────────────────────────
  _buildUI() {
    const el = document.createElement('div');
    el.id = 'story';
    el.innerHTML = `
      <style>
        #story, #story * { box-sizing: border-box; }
        #story { position: fixed; inset: 0; z-index: 15; pointer-events: none;
                 font-family: "Pretendard","Malgun Gothic","Apple SD Gothic Neo",sans-serif;
                 color: #fff; text-shadow: 0 2px 12px rgba(0,0,0,.85); }
        #st-top { position: absolute; top: 26px; left: 30px; }
        #st-obj-l { font-size: 11px; letter-spacing: .26em; color: #ffc94d; font-weight: 700; }
        #st-obj { font-size: 25px; font-weight: 800; margin-top: 3px; letter-spacing: -.4px; }
        #st-sub { font-size: 13px; color: #cfcfcf; margin-top: 3px; }
        #st-clock { position: absolute; top: 26px; right: 30px; text-align: right; }
        #st-cl-l { font-size: 11px; letter-spacing: .24em; color: #ffc94d; font-weight: 700; }
        #st-cl { font-size: 34px; font-weight: 800; font-variant-numeric: tabular-nums;
                 letter-spacing: 1px; line-height: 1.1; }
        #st-cl.warn { color: #ff6a5a; }
        #st-cards { position: absolute; top: 96px; right: 30px; display: flex; gap: 7px; }
        #st-cards i { width: 26px; height: 34px; border-radius: 3px; display: block;
                      border: 1.5px solid rgba(255,255,255,.34); background: rgba(0,0,0,.28); }
        #st-cards i.on { background: #ffc94d; border-color: #ffc94d;
                         box-shadow: 0 0 14px rgba(255,201,77,.75); }
        #st-hint { position: absolute; left: 50%; bottom: 96px; transform: translateX(-50%);
                   font-size: 15px; color: #ffe9b0; opacity: 0; transition: opacity .3s; }
        #st-hint.on { opacity: 1; }
        #st-cross { position: absolute; left: 50%; top: 50%; width: 22px; height: 22px;
                    margin: -11px 0 0 -11px; }
        #st-cross i { position: absolute; background: rgba(255,255,255,.85); }
        #st-cross i:nth-child(1) { left: 10px; top: 0;  width: 2px; height: 7px; }
        #st-cross i:nth-child(2) { left: 10px; top: 15px; width: 2px; height: 7px; }
        #st-cross i:nth-child(3) { left: 0;  top: 10px; width: 7px; height: 2px; }
        #st-cross i:nth-child(4) { left: 15px; top: 10px; width: 7px; height: 2px; }
        #st-ammo { position: absolute; right: 30px; bottom: 26px; text-align: right; }
        #st-ammo .w { font-size: 12px; color: #cfcfcf; letter-spacing: .14em; font-weight: 700; }
        #st-ammo .n { font-size: 42px; font-weight: 800; line-height: 1.05;
                      font-variant-numeric: tabular-nums; }
        #st-ammo .n small { font-size: 17px; color: #9aa0a8; font-weight: 700; }
        #st-ammo .n.low { color: #ff6a5a; }
        #st-kill { position: absolute; left: 30px; bottom: 26px; font-size: 13px; color: #cfcfcf; }
        #st-kill b { color: #ffc94d; font-size: 18px; }
        #st-toast { position: absolute; left: 50%; top: 27%; transform: translateX(-50%);
                    font-size: 30px; font-weight: 800; color: #ffc94d; opacity: 0;
                    transition: opacity .35s, transform .35s; }
        #st-toast.on { opacity: 1; transform: translateX(-50%) translateY(-10px); }

        #st-dlg { position: absolute; left: 50%; bottom: 52px; transform: translateX(-50%);
                  width: min(900px, 88vw); background: rgba(8,10,14,.93);
                  border: 1px solid rgba(255,255,255,.14); border-radius: 16px;
                  padding: 24px 30px 22px; display: none; pointer-events: none;
                  box-shadow: 0 24px 70px rgba(0,0,0,.6); }
        #st-dlg.on { display: block; }
        #st-who { display: inline-block; background: #ffc94d; color: #12161c; font-weight: 800;
                  font-size: 13px; border-radius: 999px; padding: 4px 15px; margin-bottom: 13px; }
        #st-txt { font-size: 20px; line-height: 1.65; min-height: 66px; text-shadow: none;
                  color: #f2f2f2; }
        #st-next { text-align: right; font-size: 12.5px; color: #8e96a3; margin-top: 8px; }

        #st-end { position: absolute; inset: 0; background: rgba(6,8,11,.95); display: none;
                  align-items: center; justify-content: center; text-align: center; }
        #st-end.on { display: flex; }
        #st-end h2 { font-size: clamp(30px,6vw,58px); font-weight: 800; letter-spacing: .04em; }
        #st-end .t { font-size: 17px; color: #ffc94d; margin-top: 16px; }
        #st-end .n { font-size: 13px; color: #8e96a3; margin-top: 26px; line-height: 1.8; }
      </style>
      <div id="st-top">
        <div id="st-obj-l">현재 목표</div>
        <div id="st-obj">사연 카드를 찾아라</div>
        <div id="st-sub">야외 세트에 다섯 장이 흩어져 있다</div>
      </div>
      <div id="st-clock"><div id="st-cl-l">방송까지</div><div id="st-cl">4:00</div></div>
      <div id="st-cards"></div>
      <div id="st-cross"><i></i><i></i><i></i><i></i></div>
      <div id="st-ammo"><div class="w" id="st-wname">M4A1</div>
        <div class="n" id="st-mag">30<small> / 210</small></div></div>
      <div id="st-kill">제압 <b id="st-kills">0</b></div>
      <div id="st-toast"></div>
      <div id="st-hint"></div>
      <div id="st-dlg">
        <div id="st-who">PD</div>
        <div id="st-txt"></div>
        <div id="st-next">스페이스 / 클릭 — 다음</div>
      </div>
      <div id="st-end"><div>
        <h2>ON AIR</h2>
        <div class="t" id="st-end-t"></div>
        <div class="n">압도적 재미 — 매불쇼<br>이 게임은 팬메이드 패러디이며 모든 대사는 창작입니다.</div>
      </div></div>`;
    document.body.appendChild(el);

    this.ui = {
      obj:  document.getElementById('st-obj'),
      sub:  document.getElementById('st-sub'),
      cl:   document.getElementById('st-cl'),
      cards: document.getElementById('st-cards'),
      dlg:  document.getElementById('st-dlg'),
      who:  document.getElementById('st-who'),
      txt:  document.getElementById('st-txt'),
      hint: document.getElementById('st-hint'),
      toast: document.getElementById('st-toast'),
      end:  document.getElementById('st-end'),
      wname: document.getElementById('st-wname'),
      mag:  document.getElementById('st-mag'),
      kills: document.getElementById('st-kills'),
      endT: document.getElementById('st-end-t'),
      cross: document.getElementById('st-cross'),
    };
    for (let i = 0; i < this.total; i++) {
      this.ui.cards.appendChild(document.createElement('i'));
    }
  }

  _setObjective(title, sub) {
    if (this.ui.obj) this.ui.obj.textContent = title;
    if (this.ui.sub) this.ui.sub.textContent = sub;
  }

  _toast(text) {
    const t = this.ui.toast;
    if (!t) return;
    t.textContent = text;
    t.classList.add('on');
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => t.classList.remove('on'), 1600);
  }

  _hint(text) {
    const h = this.ui.hint;
    if (!h) return;
    if (!text) { h.classList.remove('on'); return; }
    h.textContent = text;
    h.classList.add('on');
  }

  // ── 대화 ──────────────────────────────────────────────────────────────────
  say(lines, done) {
    this.queue = lines.slice();
    this._onDone = done || null;
    this.paused = true;
    if (this.ui.dlg) this.ui.dlg.classList.add('on');
    if (this.ui.cross) this.ui.cross.style.display = 'none';
    this._next();
  }

  _next() {
    const l = this.queue.shift();
    if (!l) {
      this.paused = false;
      if (this.ui.dlg) this.ui.dlg.classList.remove('on');
      if (this.ui.cross) this.ui.cross.style.display = '';
      const d = this._onDone; this._onDone = null;
      if (d) d();
      return;
    }
    this.line = l;
    this.typed = 0;
    this.typeT = 0;
    if (this.ui.who) this.ui.who.textContent = l[0];
    if (this.ui.txt) this.ui.txt.textContent = '';
  }

  advance() {
    if (!this.line) return;
    // 아직 타이핑 중이면 즉시 전부 표시, 다 나왔으면 다음 줄로
    if (this.typed < this.line[1].length) {
      this.typed = this.line[1].length;
      if (this.ui.txt) this.ui.txt.textContent = this.line[1];
      return;
    }
    this._next();
  }

  // ── 프레임 ────────────────────────────────────────────────────────────────
  update(dt) {
    this.elapsed += dt;

    // 탄약 / 제압 수 — 원본 엔진 값을 그대로 읽어 한글 HUD 에 반영한다.
    // 인트로 이전에도 HUD 는 살아 있어야 하므로 boot 분기보다 위에 둔다.
    const w = this.g.weapon;
    if (w && this.ui.mag) {
      const mag = w.mag ?? 0, res = w.reserve ?? 0;
      this.ui.mag.innerHTML = mag + '<small> / ' + res + '</small>';
      this.ui.mag.classList.toggle('low', mag <= 6);
      if (this.ui.wname && w.name && this._wname !== w.name) {
        this._wname = w.name; this.ui.wname.textContent = w.name;
      }
    }
    const kills = this.g.ballistics?.stats?.kills ?? 0;
    if (kills !== this._kills) {
      this._kills = kills;
      if (this.ui.kills) this.ui.kills.textContent = kills;
      if (kills === 1 && !this._firstBlood && this.state !== 'boot') {
        this._firstBlood = true;
        this.say([['최욱', '오, 하나 잡았네. 페인트탄이니까 죄책감 갖지 마.'],
                  ['PD',   '상대 팀도 저러고 촬영비 받는 거야. 계속 가자.']]);
      }
    }

    // 시작 게이트를 지나 실제로 조작이 시작되면 인트로를 연다
    if (this.state === 'boot') {
      if ((document.pointerLockElement || this._entered) && this.elapsed > 0.15) {
        this.state = 'intro';
        this.say(INTRO, () => {
          this.state = 'hunt';
          this._setObjective('사연 카드를 찾아라', '야외 세트에 다섯 장이 흩어져 있다');
        });
      }
      return;
    }

    // 타이핑 연출
    if (this.paused && this.line && this.typed < this.line[1].length) {
      this.typeT += dt;
      const per = 0.022;
      while (this.typeT >= per && this.typed < this.line[1].length) {
        this.typeT -= per; this.typed++;
      }
      if (this.ui.txt) this.ui.txt.textContent = this.line[1].slice(0, this.typed);
    }

    // 카드 부유 애니메이션
    for (const c of this.cards) {
      if (c.taken) continue;
      c.phase += dt * 1.6;
      c.grp.position.y = 1.15 + Math.sin(c.phase) * 0.12;
      c.card.rotation.y += dt * 1.15;
      c.ring.rotation.z += dt * 0.7;
    }

    if (this.paused) return;

    // 카운트다운 — 0이 되어도 게임은 계속된다 (시연 중 하드 실패 방지)
    if (this.state === 'hunt' || this.state === 'return') {
      this.timeLeft = Math.max(0, this.timeLeft - dt);
      const m = Math.floor(this.timeLeft / 60);
      const s = Math.floor(this.timeLeft % 60);
      if (this.ui.cl) {
        this.ui.cl.textContent = m + ':' + String(s).padStart(2, '0');
        this.ui.cl.classList.toggle('warn', this.timeLeft < 60);
      }
      if (this.timeLeft <= 0 && !this.timeUpFired) {
        this.timeUpFired = true;
        this.say(TIMEUP);
      }
    }

    const p = this.g.camera?.position;
    if (!p) return;

    // 카드 근접 획득
    if (this.state === 'hunt') {
      let nearest = null, nd = 1e9;
      for (const c of this.cards) {
        if (c.taken) continue;
        const dx = p.x - c.x, dz = p.z - c.z;
        const d = Math.hypot(dx, dz);
        if (d < nd) { nd = d; nearest = c; }
        if (d < 2.0) { this._take(c); break; }
      }
      if (nearest && nd < 14) {
        this._hint('사연 카드까지 ' + nd.toFixed(0) + 'm');
      } else {
        this._hint('');
      }
    }

    // 부스 복귀
    if (this.state === 'return') {
      const d = Math.hypot(p.x - GOAL.x, p.z - GOAL.z);
      this._hint(d < 20 ? '부스까지 ' + d.toFixed(0) + 'm' : '');
      if (d < 2.6) {
        this.state = 'done';
        this._hint('');
        this.say(OUTRO, () => this._ending());
      }
    }
  }

  _take(c) {
    c.taken = true;
    c.grp.visible = false;
    this.collected++;
    const dots = this.ui.cards?.children;
    if (dots && dots[this.collected - 1]) dots[this.collected - 1].classList.add('on');
    this._toast('사연 카드 ' + this.collected + ' / ' + this.total);
    this._hint('');

    const lines = PICKUP[c.id];
    const last = this.collected >= this.total;
    this.say(lines || [], () => {
      if (last) {
        this.state = 'return';
        if (this.goalMesh) this.goalMesh.visible = true;
        this._setObjective('부스로 복귀하라', '파란 표시가 뜬 지점으로 돌아가면 방송을 시작한다');
      }
    });
  }

  _ending() {
    const used = Math.max(0, 240 - this.timeLeft);
    const m = Math.floor(used / 60), s = Math.floor(used % 60);
    if (this.ui.endT) {
      this.ui.endT.textContent =
        '사연 카드 ' + this.total + '장 회수 완료  ·  소요 시간 ' + m + '분 ' + String(s).padStart(2, '0') + '초';
    }
    if (this.ui.end) this.ui.end.classList.add('on');
    try { document.exitPointerLock?.(); } catch (e) {}
  }

  resize() {}
}

export default Story;
