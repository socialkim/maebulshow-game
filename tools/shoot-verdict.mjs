// Verdict capture: same as shoot.mjs but with a long screenshot timeout for the software rasteriser.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';
import { CAMS } from './shoot.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'shots');

const args = process.argv.slice(2);
const label = args[0] && !args[0].startsWith('--') ? args[0] : 'shot';
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const W = Number(flag('w', 1280)), H = Number(flag('h', 720));
const SETTLE = Number(flag('settle', 5000));
const cams = String(flag('cam', 'spawn')).split(',');

fs.mkdirSync(OUT, { recursive: true });
const { server, url } = await serve(0);

const browser = await chromium.launch({
  args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
         '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-sandbox',
         '--enable-features=Vulkan', '--no-sandbox'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
page.setDefaultTimeout(300000);

const logs = [];
const IGNORE = /GL Driver Message|GPU stall due to ReadPixels|Slow read-back|SwiftShader|THREE.WebGLRenderer: Context Lost/i;
page.on('console', m => {
  const t = m.text();
  if ((m.type() === 'error' || m.type() === 'warning') && !IGNORE.test(t)) logs.push(`[${m.type()}] ${t}`);
});
page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));

await page.goto(url, { waitUntil: 'load', timeout: 120000 });

let booted = true;
try {
  await page.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 180000 });
} catch { booted = false; logs.push('[fatal] game never reached ready=true within 180s'); }

await page.waitForTimeout(SETTLE);

const results = [];
for (const cam of cams) {
  const p = CAMS[cam];
  if (!p) { console.error('unknown cam preset:', cam); continue; }
  await page.evaluate(([x, y, z, yaw, pitch, ads]) => {
    const g = window.__GAME;
    g?.controller?.teleport?.(x, y, z, yaw * Math.PI / 180, pitch * Math.PI / 180);
    g?.weapon?.setAds?.(ads);
    g?.hud?.setDemoState?.();
  }, p);
  await page.waitForTimeout(2500);
  const file = path.join(OUT, `${label}-${cam}.png`);
  try {
    await page.screenshot({ path: file, timeout: 300000, animations: 'allow', caret: 'initial' });
    results.push(file);
    console.log('captured', cam);
  } catch (e) {
    console.error('FAILED cam', cam, e.message);
  }
}

await browser.close();
server.close();

console.log('\n=== SHOOT REPORT: ' + label + ' ===');
console.log('booted:', booted);
console.log('images:\n  ' + results.join('\n  '));
if (logs.length) console.log('\nerrors:\n' + [...new Set(logs)].slice(0, 40).join('\n'));
else console.log('\nno console errors.');
