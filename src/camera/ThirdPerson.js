// ─────────────────────────────────────────────────────────────────────────────
//  ThirdPerson.js — 3인칭 시점 + 플레이어 아바타
//
//  Controller 는 매 프레임 자기 상태(this.pos)를 camera.position 에 덮어쓴다.
//  따라서 Controller 보다 **뒤에** 업데이트되는 이 모듈에서 카메라를 뒤로 물리면
//  Controller 를 건드리지 않고도 3인칭이 된다. 다음 프레임이면 다시 원위치에서 시작한다.
//
//   · V 키 또는 game.thirdperson.toggle() 로 전환
//   · 아바타는 전부 코드로 만든 프리미티브 (외부 에셋 금지 규칙 유지)
//   · 걷기/달리기 속도에 따라 다리·팔이 흔들리는 절차적 애니메이션
//   · 카메라 뒤쪽에 벽이 있으면 레이캐스트로 당겨서 파고들지 않게 한다
// ─────────────────────────────────────────────────────────────────────────────

const DIST      = 4.35;   // 기본 카메라 거리 (m)
const HEIGHT    = 1.02;   // 어깨 위 오프셋 (m)
const SIDE      = 0.44;   // 오른쪽 어깨 너머 오프셋 (m)
const LERP      = 12.0;   // 시점 전환 보간 속도
const MIN_DIST  = 0.85;   // 벽에 붙었을 때 최소 거리

export class ThirdPerson {
  constructor(game) {
    const T = game.THREE;
    this.g = game;
    this.THREE = T;
    this.on = false;
    this.blend = 0;                 // 0 = 1인칭, 1 = 3인칭
    this.variant = game.config?.avatar?.variant || 'operator';

    this.avatar = null;
    this.parts = {};
    this.stride = 0;
    this._prev = new T.Vector3();
    this._eye = new T.Vector3();
    this._fwd = new T.Vector3();
    this._right = new T.Vector3();
    this._want = new T.Vector3();
    this._up = new T.Vector3(0, 1, 0);
    this._speed = 0;
  }

  async init() {
    this._build();
    this._bind();
    const p = this.g.camera?.position;
    if (p) this._prev.copy(p);
  }

  toggle() { this.setEnabled(!this.on); }

  setEnabled(v) {
    this.on = !!v;
    // 1인칭 무기는 3인칭에서 숨긴다 (매불쇼 에디션은 애초에 숨겨져 있다)
    const vm = this.g.weapon?.viewmodelScene;
    if (vm && this._vmOwned !== false) {
      if (this.on) { this._vmWas = vm.visible; vm.visible = false; }
      else if (this._vmWas !== undefined) { vm.visible = this._vmWas; }
    }
    const btn = document.getElementById('tview');
    if (btn) btn.textContent = this.on ? '1인칭으로' : '3인칭으로';
  }

  _bind() {
    this._onKey = (e) => {
      if (e.code === 'KeyV' && !e.repeat) {
        if (this.g?.story?.paused) return;
        this.toggle();
      }
    };
    window.addEventListener('keydown', this._onKey, false);
  }

  // ── 아바타 ────────────────────────────────────────────────────────────────
  _mat(hex, rough = 0.85, metal = 0.0) {
    const T = this.THREE;
    return new T.MeshStandardMaterial({ color: hex, roughness: rough, metalness: metal });
  }

  _box(w, h, d, mat, x = 0, y = 0, z = 0) {
    const m = new this.THREE.Mesh(new this.THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    m.castShadow = true; m.receiveShadow = true;
    return m;
  }

  _build() {
    const T = this.THREE;
    const staff = this.variant === 'staff';

    const cloth  = this._mat(staff ? 0x2f3946 : 0x6d6448, 0.95);
    const vest   = this._mat(staff ? 0xffc94d : 0x4a4636, 0.82);
    const dark   = this._mat(0x24262b, 0.9);
    const skin   = this._mat(0xb98d6a, 0.72);
    const boot   = this._mat(0x33302c, 0.8);

    const g = new T.Group();

    // 다리
    const legL = new T.Group(), legR = new T.Group();
    for (const [grp, sx] of [[legL, -0.13], [legR, 0.13]]) {
      grp.position.set(sx, 0.86, 0);
      grp.add(this._box(0.19, 0.46, 0.20, cloth, 0, -0.23, 0));           // 허벅지
      const shin = this._box(0.16, 0.44, 0.18, cloth, 0, -0.68, 0);
      grp.add(shin);
      grp.add(this._box(0.19, 0.13, 0.29, boot, 0, -0.94, 0.04));         // 군화
      g.add(grp);
    }

    // 몸통
    const torso = new T.Group();
    torso.position.set(0, 1.16, 0);
    torso.add(this._box(0.46, 0.54, 0.26, cloth, 0, 0.05, 0));
    torso.add(this._box(0.48, 0.46, 0.31, vest, 0, 0.07, 0.01));          // 조끼
    torso.add(this._box(0.40, 0.05, 0.32, dark, 0, -0.10, 0.02));         // 벨트
    torso.add(this._box(0.54, 0.09, 0.24, cloth, 0, 0.28, 0));            // 어깨선
    g.add(torso);

    // 팔
    const armL = new T.Group(), armR = new T.Group();
    for (const [grp, sx] of [[armL, -0.31], [armR, 0.31]]) {
      grp.position.set(sx, 1.44, 0);
      grp.add(this._box(0.14, 0.34, 0.15, cloth, 0, -0.17, 0));           // 상완
      grp.add(this._box(0.12, 0.32, 0.13, cloth, 0, -0.50, 0));           // 전완
      grp.add(this._box(0.12, 0.13, 0.13, dark, 0, -0.71, 0));            // 장갑
      g.add(grp);
    }

    // 머리
    const head = new T.Group();
    head.position.set(0, 1.53, 0);
    head.add(this._box(0.12, 0.07, 0.12, skin, 0, 0.03, 0));              // 목
    head.add(this._box(0.23, 0.26, 0.24, skin, 0, 0.19, 0));              // 두상
    if (staff) {
      head.add(this._box(0.26, 0.09, 0.26, dark, 0, 0.34, 0));            // 캡
      head.add(this._box(0.24, 0.03, 0.12, dark, 0, 0.31, 0.17));         // 챙
    } else {
      head.add(this._box(0.25, 0.14, 0.26, this._mat(0x4b4a42, 0.8), 0, 0.31, 0)); // 헬멧
    }
    g.add(head);

    // 스태프는 클립보드를 든다
    if (staff) {
      const clip = this._box(0.24, 0.32, 0.02, this._mat(0xd8d2c4, 0.9), 0, 0, 0);
      clip.rotation.x = -0.35;
      armL.add(clip);
      clip.position.set(0.02, -0.74, 0.14);
    }

    g.visible = false;
    this.g.scene.add(g);

    this.avatar = g;
    this.parts = { legL, legR, armL, armR, torso, head };
  }

  // ── 프레임 ────────────────────────────────────────────────────────────────
  update(dt) {
    const cam = this.g.camera;
    const ctl = this.g.controller;
    if (!cam) return;

    // 목표 블렌드로 부드럽게
    const goal = this.on ? 1 : 0;
    this.blend += (goal - this.blend) * Math.min(1, dt * LERP);
    if (this.blend < 0.001 && !this.on) {
      if (this.avatar) this.avatar.visible = false;
      this._prev.copy(cam.position);
      return;
    }
    if (this.avatar) this.avatar.visible = true;

    // Controller 가 이번 프레임에 확정한 눈 위치 — 이 값이 진짜 플레이어 위치다
    this._eye.copy(cam.position);

    // 이동 속도 (수평)
    const dx = this._eye.x - this._prev.x, dz = this._eye.z - this._prev.z;
    const inst = dt > 0 ? Math.hypot(dx, dz) / dt : 0;
    this._speed += (Math.min(inst, 9) - this._speed) * Math.min(1, dt * 9);
    this._prev.copy(this._eye);

    // 아바타 배치 — 발끝을 눈높이 아래로
    const eyeH = this.g.config?.player?.eyeHeight ?? 1.62;
    const yaw = Math.atan2(-(cam.matrixWorld.elements[8]), -(cam.matrixWorld.elements[10]));
    if (this.avatar) {
      this.avatar.position.set(this._eye.x, this._eye.y - eyeH, this._eye.z);
      this.avatar.rotation.y = yaw;
    }

    this._animate(dt);

    // 카메라를 뒤로 — 시선 반대 방향
    cam.getWorldDirection(this._fwd);
    this._right.crossVectors(this._fwd, this._up).normalize();

    let dist = DIST * this.blend;
    // 뒤쪽 벽 충돌 — 가능하면 레이캐스트로 당긴다
    const back = this._want.copy(this._fwd).multiplyScalar(-1).normalize();
    const rc = this.g.ballistics?.raycast;
    if (rc && dist > MIN_DIST) {
      try {
        const hit = rc.call(this.g.ballistics, this._eye, back, dist + 0.45);
        if (hit && typeof hit.distance === 'number') {
          dist = Math.max(MIN_DIST, hit.distance - 0.45);
        }
      } catch (e) { /* 레이캐스트가 없거나 형태가 다르면 그냥 무시 */ }
    }

    cam.position
      .addScaledVector(back, dist)
      .addScaledVector(this._right, SIDE * this.blend)
      .addScaledVector(this._up, HEIGHT * this.blend);
  }

  _animate(dt) {
    const p = this.parts;
    if (!p.legL) return;
    const moving = this._speed > 0.25;
    const rate = 1.9 + this._speed * 1.15;
    this.stride += dt * rate * (moving ? 1 : 0);
    const s = Math.sin(this.stride * Math.PI);
    const amp = Math.min(0.62, 0.13 + this._speed * 0.10);

    p.legL.rotation.x =  s * amp;
    p.legR.rotation.x = -s * amp;
    p.armL.rotation.x = -s * amp * 0.72;
    p.armR.rotation.x =  s * amp * 0.72;

    // 걷지 않을 때는 서서히 기본자세로
    if (!moving) {
      for (const k of ['legL', 'legR', 'armL', 'armR']) {
        p[k].rotation.x *= Math.max(0, 1 - dt * 6);
      }
    }
    // 상하 반동
    if (this.avatar) this.avatar.position.y += moving ? Math.abs(s) * 0.035 : 0;
    p.torso.rotation.y = s * 0.05;
    p.head.rotation.y = -s * 0.03;
  }

  resize() {}
}

export default ThirdPerson;
