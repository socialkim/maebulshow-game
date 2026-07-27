// 매불쇼 에디션 검증 하네스
//   node tools/storycheck.mjs <outPrefix>
// 부팅 → 인트로 대화 → 대사 넘기기 → 카드 순간이동 획득 → 엔딩까지 자동으로 밟아본다.
import { chromium } from 'playwright';
import { serve } from './serve.mjs';

const out = process.argv[2] || 'story';
const { server, url } = await serve(0);
const browser = await chromium.launch({
  args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
         '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-sandbox', '--no-sandbox'],
});
const page = await browser.newPage({ viewport: { width: 960, height: 540 } });

const logs = [];
const IGNORE = /GL Driver Message|ReadPixels|Slow read-back|SwiftShader|Context Lost/i;
page.on('console', m => {
  const t = m.text();
  if ((m.type() === 'error' || m.type() === 'warning') && !IGNORE.test(t)) logs.push(`[${m.type()}] ${t}`);
});
page.on('pageerror', e => logs.push('[pageerror] ' + e.message));

await page.goto(url, { waitUntil: 'load', timeout: 90000 });
let booted = true;
try { await page.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 150000 }); }
catch { booted = false; logs.push('[fatal] never reached ready=true'); }
await page.waitForTimeout(2500);

const shot = async (name) => {
  await page.screenshot({ path: `shots/${out}-${name}.png`, timeout: 180000 });
};

// 1) 게임 진입 → 인트로 대화가 떠야 한다
await page.mouse.click(480, 300);
await page.waitForTimeout(1800);
const introUp = await page.evaluate(() => !!document.querySelector('#st-dlg.on'));
await shot('01-intro');

// 2) 인트로 대사 전부 넘기기
for (let i = 0; i < 14; i++) {
  await page.evaluate(() => window.__GAME?.story?.advance());
  await page.waitForTimeout(160);
}
await page.waitForTimeout(600);
const state1 = await page.evaluate(() => window.__GAME?.story?.state);
await shot('02-hunt');

// 3) 카드 5장을 순간이동으로 회수
const drain = async () => {                 // 열려 있는 대화를 전부 넘긴다
  for (let i = 0; i < 12; i++) {
    const open = await page.evaluate(() => !!document.querySelector('#st-dlg.on'));
    if (!open) return;
    await page.evaluate(() => window.__GAME?.story?.advance());
    await page.waitForTimeout(150);
  }
};

const cards = await page.evaluate(() => window.__GAME.story.cards.map(c => ({ id: c.id, x: c.x, z: c.z })));
const missed = [];
for (const c of cards) {
  await drain();                            // 반드시 먼저 대화를 닫아야 획득 판정이 돈다
  await page.evaluate(([x, z]) => {
    window.__GAME.controller?.teleport?.(x, 1.62, z + 1.2, Math.PI, 0);
  }, [c.x, c.z]);
  // 최대 4초까지 기다리며, 중간에 대화가 뜨면 넘겨가며 획득을 확인한다
  let took = false;
  for (let t = 0; t < 20; t++) {
    await page.waitForTimeout(200);
    const s = await page.evaluate((id) => ({
      taken: window.__GAME.story.cards.find(k => k.id === id).taken,
      paused: window.__GAME.story.paused,
      dlg: !!document.querySelector('#st-dlg.on'),
      d: (() => { const p = window.__GAME.camera.position;
                  const k = window.__GAME.story.cards.find(q => q.id === id);
                  return +Math.hypot(p.x - k.x, p.z - k.z).toFixed(2); })(),
    }), c.id);
    if (s.taken) { took = true; break; }
    if (s.dlg) await page.evaluate(() => window.__GAME?.story?.advance());
    if (t === 19) console.log(`   [진단] 카드 ${c.id} 실패 — 거리 ${s.d}, paused=${s.paused}, dlg=${s.dlg}`);
  }
  if (!took) missed.push(c.id);
}
await drain();
const got = await page.evaluate(() => window.__GAME.story.collected);
const state2 = await page.evaluate(() => window.__GAME?.story?.state);
await shot('03-collected');

// 4) 부스로 복귀 → 엔딩
await page.evaluate(() => window.__GAME.controller?.teleport?.(0, 1.62, 17.5, Math.PI, 0));
await page.waitForTimeout(1200);
await drain();
await page.waitForTimeout(900);
const ended = await page.evaluate(() => !!document.querySelector('#st-end.on'));
await shot('04-ending');

// 5) 전투가 실제로 제거됐는지
const combat = await page.evaluate(() => ({
  enemies: window.__GAME.registry?.enemies?.length ?? -1,
  weaponHidden: window.__GAME.weapon?.viewmodelScene ? !window.__GAME.weapon.viewmodelScene.visible : null,
  hudHidden: document.getElementById('hud')?.style.display === 'none',
}));

console.log('\n=== 매불쇼 에디션 검증 ===');
console.log('booted        :', booted);
console.log('인트로 표시   :', introUp);
console.log('상태(인트로후):', state1, '→', state2);
console.log('카드 회수     :', got, '/ 5', missed.length ? ('미획득: ' + missed.join(',')) : '');
console.log('엔딩 도달     :', ended);
console.log('적 수         :', combat.enemies, '| 무기 숨김:', combat.weaponHidden, '| 원본HUD 숨김:', combat.hudHidden);
console.log('콘솔 이슈     :', logs.length);
logs.slice(0, 8).forEach(l => console.log('   ', l));

await browser.close();
server.close();
if (!booted || logs.length) process.exitCode = 1;
