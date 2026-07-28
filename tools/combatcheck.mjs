// 전투 복구 검증 — 적 스폰 / 사격 / 탄약 HUD / 스토리 동시 동작
import { chromium } from 'playwright';
import { serve } from './serve.mjs';

const { server, url } = await serve(0);
const b = await chromium.launch({ args: ['--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader',
  '--ignore-gpu-blocklist','--enable-webgl','--disable-gpu-sandbox','--no-sandbox'] });
const p = await b.newPage({ viewport: { width: 960, height: 540 } });
const logs = []; const IG = /GL Driver|ReadPixels|Slow read-back|SwiftShader|Context Lost/i;
p.on('console', m => { const t = m.text();
  if ((m.type() === 'error' || m.type() === 'warning') && !IG.test(t)) logs.push(t); });
p.on('pageerror', e => logs.push('[pageerror] ' + e.message));

await p.goto(url, { waitUntil: 'load', timeout: 90000 });
await p.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 180000 });
await p.waitForTimeout(2500);
await p.mouse.click(480, 300);

// 인트로 대사 소진
for (let i = 0; i < 60; i++) {
  if (await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) break;
  await p.waitForTimeout(200);
}
for (let i = 0; i < 24; i++) {
  if (!await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) break;
  await p.evaluate(() => window.__GAME.story.advance());
  await p.waitForTimeout(150);
}
await p.waitForTimeout(800);

const probe = await p.evaluate(() => ({
  entered: window.__GAME.story._entered, elapsed: +window.__GAME.story.elapsed.toFixed(2),
  lock: !!document.pointerLockElement, state: window.__GAME.story.state,
}));
console.log('진입 프로브:', JSON.stringify(probe));

const base = await p.evaluate(() => ({
  enemies: window.__GAME.registry.enemies.length,
  weaponVisible: window.__GAME.weapon?.viewmodelScene?.visible,
  mag: window.__GAME.weapon?.mag,
  state: window.__GAME.story.state,
  hudText: document.getElementById('st-mag')?.textContent,
}));

// 실제로 쏴 본다 — 원본 엔진의 발사 경로를 그대로 태운다
await p.evaluate(() => { for (let i = 0; i < 6; i++) window.__GAME.weapon?.fire?.(); });
await p.waitForTimeout(1600);
const fired = await p.evaluate(() => ({
  mag: window.__GAME.weapon?.mag,
  shots: window.__GAME.ballistics?.stats?.shots,
  hudText: document.getElementById('st-mag')?.textContent,
}));
await p.screenshot({ path: 'shots/combat-01.png', timeout: 180000 });

// 적 쪽을 바라보고 한 장 더
await p.evaluate(() => window.__GAME.controller?.teleport?.(2.5, 1.62, 6, Math.PI, -0.05));
await p.waitForTimeout(2200);
await p.screenshot({ path: 'shots/combat-02.png', timeout: 180000 });

console.log('\n=== 전투 복구 검증 ===');
console.log('적 스폰      :', base.enemies, '| 무기 표시:', base.weaponVisible, '| 스토리 상태:', base.state);
console.log('탄약 HUD     :', base.hudText, '→', fired.hudText);
console.log('사격         : mag', base.mag, '→', fired.mag, '| 발사 수', fired.shots);
console.log('콘솔 이슈    :', logs.length);
logs.slice(0, 6).forEach(l => console.log('   ', l));
await b.close(); server.close();
