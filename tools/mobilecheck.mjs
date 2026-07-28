// 3인칭 + 터치 조작 검증
//   node tools/mobilecheck.mjs <prefix>
import { chromium, devices } from 'playwright';
import { serve } from './serve.mjs';

const out = process.argv[2] || 'mob';
const { server, url } = await serve(0);
const GL = ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
            '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-sandbox', '--no-sandbox'];
const browser = await chromium.launch({ args: GL });

const logs = [];
const IGNORE = /GL Driver Message|ReadPixels|Slow read-back|SwiftShader|Context Lost/i;
const wire = (p) => {
  p.on('console', m => { const t = m.text();
    if ((m.type() === 'error' || m.type() === 'warning') && !IGNORE.test(t)) logs.push('[' + m.type() + '] ' + t); });
  p.on('pageerror', e => logs.push('[pageerror] ' + e.message));
};
const boot = async (p, q = '') => {
  await p.goto(url + q, { waitUntil: 'load', timeout: 90000 });
  await p.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 150000 });
  await p.waitForTimeout(2500);
};
const shot = (p, n) => p.screenshot({ path: `shots/${out}-${n}.png`, timeout: 180000 });
const drain = async (p) => {
  for (let i = 0; i < 14; i++) {
    if (!await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) return;
    await p.evaluate(() => window.__GAME?.story?.advance());
    await p.waitForTimeout(140);
  }
};

// ───────────────────────── 데스크톱: 3인칭 ─────────────────────────
const d = await browser.newPage({ viewport: { width: 960, height: 540 } });
wire(d);
await boot(d);
await d.mouse.click(480, 300);
await d.waitForTimeout(1200);
await drain(d);

await d.evaluate(() => window.__GAME.thirdperson.setEnabled(true));
await d.waitForTimeout(1400);
const tp = await d.evaluate(() => {
  const t = window.__GAME.thirdperson;
  return { on: t.on, blend: +t.blend.toFixed(2), avatar: !!t.avatar?.visible };
});
await shot(d, '01-thirdperson');

// 걷는 동안 아바타 애니메이션이 도는지
await d.evaluate(() => window.__GAME.controller._onKeyDown({ code: 'KeyW', repeat: false, preventDefault() {} }));
await d.waitForTimeout(1500);
const walking = await d.evaluate(() => ({
  speed: +window.__GAME.thirdperson._speed.toFixed(2),
  legX: +window.__GAME.thirdperson.parts.legL.rotation.x.toFixed(3),
}));
await shot(d, '02-walking');
await d.evaluate(() => window.__GAME.controller._onKeyUp({ code: 'KeyW', preventDefault() {} }));

await d.evaluate(() => window.__GAME.thirdperson.setEnabled(false));
await d.waitForTimeout(900);
const backTo1st = await d.evaluate(() => window.__GAME.thirdperson.on === false);
await d.close();

// ───────────────────────── 모바일: 터치 ─────────────────────────
const ctx = await browser.newContext({
  viewport: { width: 640, height: 360 }, deviceScaleFactor: 1,
  hasTouch: true, isMobile: true, userAgent: devices['Pixel 7'].userAgent,
});
const m = await ctx.newPage();
wire(m);
await boot(m, '?touch=1');
await m.tap('body').catch(() => {});
await m.waitForTimeout(1200);
await drain(m);

const uiUp = await m.evaluate(() => !!document.getElementById('touch'));
await shot(m, '03-touch-ui');

// 왼쪽 조이스틱을 위로 밀어 전진 → 위치가 실제로 변하는지
const before = await m.evaluate(() => { const p = window.__GAME.camera.position; return { x: +p.x.toFixed(2), z: +p.z.toFixed(2) }; });
const vw = m.viewportSize();
const sx = Math.round(vw.width * 0.22), sy = Math.round(vw.height * 0.72);
await m.touchscreen.tap(sx, sy);           // 조이스틱 활성화 지점 확보
await m.evaluate(([x, y]) => {
  const el = document.getElementById('touch');
  const mk = (type, id, cx, cy) => {
    const t = new Touch({ identifier: id, target: el, clientX: cx, clientY: cy });
    el.dispatchEvent(new TouchEvent(type, { touches: type === 'touchend' ? [] : [t],
      changedTouches: [t], bubbles: true, cancelable: true }));
  };
  mk('touchstart', 1, x, y);
  mk('touchmove', 1, x, y - 60);           // 위로 밀기 = 전진
  window.__MOVE_END = () => mk('touchend', 1, x, y - 60);
}, [sx, sy]);
await m.waitForTimeout(9000);
const frames = await m.evaluate(() => window.__GAME.frame);
const moved = await m.evaluate(() => { const p = window.__GAME.camera.position; return { x: +p.x.toFixed(2), z: +p.z.toFixed(2) }; });
const keysHeld = await m.evaluate(() => ({ ...window.__GAME.touch.held }));
await shot(m, '04-touch-move');
await m.evaluate(() => window.__MOVE_END?.());

// 오른쪽 드래그로 시야가 도는지
const yaw0 = await m.evaluate(() => window.__GAME.camera.rotation.y);
await m.evaluate(([x, y]) => {
  const el = document.getElementById('touch');
  const mk = (type, id, cx, cy) => {
    const t = new Touch({ identifier: id, target: el, clientX: cx, clientY: cy });
    el.dispatchEvent(new TouchEvent(type, { touches: type === 'touchend' ? [] : [t],
      changedTouches: [t], bubbles: true, cancelable: true }));
  };
  mk('touchstart', 2, x, y);
  mk('touchmove', 2, x + 120, y);
  mk('touchend', 2, x + 120, y);
}, [Math.round(vw.width * 0.78), Math.round(vw.height * 0.45)]);
await m.waitForTimeout(4000);
const yaw1 = await m.evaluate(() => window.__GAME.camera.rotation.y);

// 모바일에서 3인칭 버튼
await m.evaluate(() => window.__GAME.thirdperson.toggle());
await m.waitForTimeout(1300);
await shot(m, '05-mobile-thirdperson');
const mtp = await m.evaluate(() => window.__GAME.thirdperson.on);

console.log('\n=== 3인칭 · 터치 검증 ===');
console.log('3인칭 진입    :', tp.on, '| blend', tp.blend, '| 아바타 표시', tp.avatar);
console.log('걷기 애니메이션:', '속도', walking.speed, '| 다리 회전', walking.legX);
console.log('1인칭 복귀    :', backTo1st);
console.log('터치 UI       :', uiUp);
console.log('조이스틱 이동 :', `(${before.x}, ${before.z}) → (${moved.x}, ${moved.z})`,
            '| 이동거리', Math.hypot(moved.x - before.x, moved.z - before.z).toFixed(2), 'm | 누적프레임', frames);
console.log('눌린 키       :', JSON.stringify(keysHeld));
console.log('시야 드래그   :', yaw0.toFixed(3), '→', yaw1.toFixed(3), '| 변화', Math.abs(yaw1 - yaw0).toFixed(3));
console.log('모바일 3인칭  :', mtp);
console.log('콘솔 이슈     :', logs.length);
logs.slice(0, 8).forEach(l => console.log('   ', l));

await browser.close();
server.close();
