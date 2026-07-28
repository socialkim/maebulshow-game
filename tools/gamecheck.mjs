// 스폰 방향 · 목숨/사망/부활/실패 · 모바일 사격 버튼 검증
import { chromium, devices } from 'playwright';
import { serve } from './serve.mjs';
const { server, url } = await serve(0);
const GL = ['--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader',
            '--ignore-gpu-blocklist','--enable-webgl','--disable-gpu-sandbox','--no-sandbox'];
const b = await chromium.launch({ args: GL });
const logs = []; const IG = /GL Driver|ReadPixels|Slow read-back|SwiftShader|Context Lost/i;
const wire = (p) => {
  p.on('console', m => { const t = m.text();
    if ((m.type() === 'error' || m.type() === 'warning') && !IG.test(t)) logs.push(t); });
  p.on('pageerror', e => logs.push('[pageerror] ' + e.message));
};
const waitDlg = async (p, ms = 14000) => {   // 대사가 열릴 때까지 기다린다
  for (let i = 0; i < ms / 200; i++) {
    if (await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) return true;
    await p.waitForTimeout(200);
  }
  return false;
};
const drain = async (p) => {
  for (let i = 0; i < 26; i++) {
    if (!await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) return;
    await p.evaluate(() => window.__GAME.story.advance());
    await p.waitForTimeout(140);
  }
};

// ── 데스크톱: 스폰 방향 · 사망/부활/실패
const p = await b.newPage({ viewport: { width: 900, height: 506 } });
wire(p);
await p.goto(url, { waitUntil: 'load', timeout: 90000 });
await p.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 180000 });
await p.waitForTimeout(2500);
await p.mouse.click(450, 250);
await waitDlg(p);
await drain(p);
await p.waitForTimeout(1200);

const spawn = await p.evaluate(() => {
  const g = window.__GAME, T = g.THREE, cam = g.camera;
  const f = new T.Vector3(); cam.getWorldDirection(f);
  const angs = g.story.cards.map(c => {
    const dx = c.x - cam.position.x, dz = c.z - cam.position.z, d = Math.hypot(dx, dz);
    return { id: c.id, dist: +d.toFixed(1), ang: Math.round(Math.acos(Math.max(-1, Math.min(1, (dx*f.x+dz*f.z)/d))) * 180 / Math.PI) };
  });
  return { forward: [+f.x.toFixed(2), +f.z.toFixed(2)], angs, front: angs.filter(a => a.ang < 70).length };
});
await p.screenshot({ path: 'shots/game-01-spawn.png', timeout: 180000 });

// 사망 3회 → 실패
const deaths = [];
for (let i = 0; i < 3; i++) {
  await p.evaluate(() => { window.__GAME.ballistics.playerHp = 0; });
  await waitDlg(p, 8000);
  await p.waitForTimeout(400);
  deaths.push(await p.evaluate(() => ({
    lives: window.__GAME.story.lives, dead: window.__GAME.story.dead,
    dlg: !!document.querySelector('#st-dlg.on'), failed: window.__GAME.story.failed,
  })));
  await drain(p);
  await p.waitForTimeout(700);
}
const afterAll = await p.evaluate(() => ({
  lives: window.__GAME.story.lives, failed: window.__GAME.story.failed,
  failVisible: document.getElementById('st-fail-box')?.style.display !== 'none'
               && !!document.querySelector('#st-end.on'),
  hp: window.__GAME.ballistics.playerHp,
}));
await p.screenshot({ path: 'shots/game-02-fail.png', timeout: 180000 });
await p.close();

// ── 모바일: 사격 버튼
const ctx = await b.newContext({ viewport: { width: 780, height: 400 }, deviceScaleFactor: 1,
  hasTouch: true, isMobile: true, userAgent: devices['Pixel 7'].userAgent });
const m = await ctx.newPage();
wire(m);
await m.goto(url + '?touch=1', { waitUntil: 'load', timeout: 90000 });
await m.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 180000 });
await m.waitForTimeout(2500);
await m.touchscreen.tap(390, 200);
await waitDlg(m);
await drain(m);
await m.waitForTimeout(800);
const btns = await m.evaluate(() => ({
  fire: !!document.getElementById('tb-fire'),
  ads: !!document.getElementById('tb-ads'),
  reload: !!document.getElementById('tb-reload'),
}));
const magBefore = await m.evaluate(() => window.__GAME.weapon.mag);
await m.evaluate(() => {
  const el = document.getElementById('tb-fire');
  const mk = (t) => { const tt = new Touch({ identifier: 9, target: el, clientX: 700, clientY: 340 });
    el.dispatchEvent(new TouchEvent(t, { touches: t === 'touchend' ? [] : [tt], changedTouches: [tt], bubbles: true, cancelable: true })); };
  mk('touchstart');
  window.__FIRE_STOP = () => mk('touchend');
});
await m.waitForTimeout(4000);
await m.evaluate(() => window.__FIRE_STOP?.());
const magAfter = await m.evaluate(() => window.__GAME.weapon.mag);
await m.screenshot({ path: 'shots/game-03-mobile.png', timeout: 180000 });

console.log('\n=== 게임성 검증 ===');
console.log('스폰 방향   :', JSON.stringify(spawn.forward), '| 앞쪽(70도 이내) 카드', spawn.front, '/ 5');
console.log('카드 각도   :', spawn.angs.map(a => `${a.id}:${a.ang}도/${a.dist}m`).join('  '));
console.log('사망 진행   :', deaths.map(d => `목숨${d.lives}${d.dlg ? '(대사)' : ''}`).join(' → '));
console.log('최종        :', JSON.stringify(afterAll));
console.log('모바일 버튼 :', JSON.stringify(btns));
console.log('모바일 사격 : mag', magBefore, '→', magAfter);
console.log('콘솔 이슈   :', logs.length);
logs.slice(0, 6).forEach(l => console.log('   ', l));
await b.close(); server.close();
