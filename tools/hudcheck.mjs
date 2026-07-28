// 마커 · 체력 · 명중 표시 검증
import { chromium } from 'playwright';
import { serve } from './serve.mjs';
const { server, url } = await serve(0);
const b = await chromium.launch({ args: ['--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader',
  '--ignore-gpu-blocklist','--enable-webgl','--disable-gpu-sandbox','--no-sandbox'] });
const p = await b.newPage({ viewport: { width: 900, height: 506 } });
const logs = []; const IG = /GL Driver|ReadPixels|Slow read-back|SwiftShader|Context Lost/i;
p.on('console', m => { const t = m.text();
  if ((m.type() === 'error' || m.type() === 'warning') && !IG.test(t)) logs.push(t); });
p.on('pageerror', e => logs.push('[pageerror] ' + e.message));

await p.goto(url, { waitUntil: 'load', timeout: 90000 });
await p.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 180000 });
await p.waitForTimeout(2500);
await p.mouse.click(450, 250);
await p.waitForTimeout(1500);
for (let i = 0; i < 26; i++) {
  if (!await p.evaluate(() => !!document.querySelector('#st-dlg.on'))) break;
  await p.evaluate(() => window.__GAME.story.advance());
  await p.waitForTimeout(150);
}
await p.waitForTimeout(1200);

const markers = await p.evaluate(() => {
  const wp = document.getElementById('st-wp');
  const vis = [...wp.children].filter(e => e.style.display !== 'none');
  return {
    total: wp.children.length,
    visible: vis.length,
    cards: vis.filter(e => e.classList.contains('card')).length,
    enemies: vis.filter(e => e.classList.contains('enemy')).length,
    offscreen: vis.filter(e => e.classList.contains('off')).length,
    sample: vis.slice(0, 3).map(e => ({ cls: e.className, l: e.style.left, t: e.style.top,
                                        ic: e.firstChild.textContent, d: e.querySelector('.d')?.textContent })),
  };
});
await p.screenshot({ path: 'shots/hud-01-markers.png', timeout: 180000 });

// 적을 실제로 맞혀 명중/제압 표시 확인
const combat = await p.evaluate(async () => {
  const g = window.__GAME, B = g.ballistics, T = g.THREE;
  const origin = new T.Vector3().copy(g.camera.position);
  let chosen = null;
  for (const e of g.registry.enemies) {
    const tgt = new T.Vector3(e.position.x, e.position.y + 0.2, e.position.z);
    let clear = true; try { clear = B.losClear ? B.losClear(origin, tgt) : true; } catch {}
    if (clear && e.alive !== false) { chosen = { e, tgt }; break; }
  }
  if (!chosen) return { note: 'LOS 확보 실패' };
  const dir = chosen.tgt.clone().sub(origin).normalize();
  for (let i = 0; i < 3; i++) B.fire({ origin, dir, spread: 0 });
  return { hits: B.stats.hits, hp: chosen.e.hp, maxHp: chosen.e.maxHp };
});
await p.waitForTimeout(300);
const hitUi = await p.evaluate(() => ({
  hitOn: !!document.querySelector('#st-hit.on'),
  hpText: document.getElementById('st-hpn')?.textContent,
  hpWidth: document.getElementById('st-hpb')?.style.width,
  enemyBars: [...document.querySelectorAll('#st-wp .wp.enemy .hb')].filter(e => e.style.display !== 'none').length,
}));
await p.screenshot({ path: 'shots/hud-02-hit.png', timeout: 180000 });

console.log('\n=== HUD 검증 ===');
console.log('마커       :', JSON.stringify(markers));
console.log('전투       :', JSON.stringify(combat));
console.log('명중 표시  :', JSON.stringify(hitUi));
console.log('콘솔 이슈  :', logs.length);
logs.slice(0, 5).forEach(l => console.log('  ', l));
await b.close(); server.close();
