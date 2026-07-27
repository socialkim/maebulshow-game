// Screenshot harness for the visual-critic loop.
//   node tools/shoot.mjs <label> [--cam spawn,street,ads] [--all] [--w 1920] [--h 1080] [--settle 3500]
// Writes shots/<label>-<cam>.png and prints any console/page errors.
// A build with errors is a FAILED build — the critic treats a non-empty error list as an automatic reject.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'shots');

// Camera presets. [x, y, z, yawDeg, pitchDeg, ads]
export const CAMS = {
  spawn:    [0, 1.62, 18, 180, -2, false],
  street:   [2.5, 1.62, 6, 195, -3, false],
  vista:    [-14, 4.2, 22, 150, -8, false],
  interior: [-7.5, 1.55, -6, 70, 0, false],
  alley:    [11, 1.62, -2, 250, 1, false],
  ads:      [2.5, 1.62, 6, 195, -2, true],
  weapon:   [0, 1.62, 14, 178, -14, false],
  sun:      [0, 1.7, 10, 118, 6, false],
};

const args = process.argv.slice(2);
const label = args[0] && !args[0].startsWith('--') ? args[0] : 'shot';
const flag = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const W = Number(flag('w', 1920)), H = Number(flag('h', 1080));
const SETTLE = Number(flag('settle', 3500));
const cams = args.includes('--all') ? Object.keys(CAMS) : String(flag('cam', 'spawn')).split(',');

fs.mkdirSync(OUT, { recursive: true });
const { server, url } = await serve(0);

const browser = await chromium.launch({
  args: ['--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
         '--ignore-gpu-blocklist', '--enable-webgl', '--disable-gpu-sandbox',
         '--enable-features=Vulkan', '--no-sandbox'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });

const logs = [];
// Benign noise from the software rasteriser used for headless capture — not build failures.
const IGNORE = /GL Driver Message|GPU stall due to ReadPixels|Slow read-back|SwiftShader|THREE.WebGLRenderer: Context Lost/i;
page.on('console', m => {
  const t = m.text();
  if ((m.type() === 'error' || m.type() === 'warning') && !IGNORE.test(t)) logs.push(`[${m.type()}] ${t}`);
});
page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));

await page.goto(url, { waitUntil: 'load', timeout: 60000 });

// Wait for the game to finish booting (or time out and shoot anyway so we can see the failure).
let booted = true;
try {
  await page.waitForFunction(() => window.__GAME?.ready === true, null, { timeout: 90000 });
} catch { booted = false; logs.push('[fatal] game never reached ready=true within 90s'); }

await page.waitForTimeout(SETTLE); // let TAA converge, animations settle, streaming finish

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
  await page.waitForTimeout(1400); // re-converge TAA after the camera cut
  const file = path.join(OUT, `${label}-${cam}.png`);
  await page.screenshot({ path: file });
  results.push(file);
}

const fps = await page.evaluate(() => {
  const g = window.__GAME; return { frames: g?.frame ?? 0, time: +(g?.time ?? 0).toFixed(1) };
});

await browser.close();
server.close();

console.log('\n=== SHOOT REPORT: ' + label + ' ===');
console.log('booted:', booted, '| frames:', fps.frames, '| sim seconds:', fps.time);
console.log('images:\n  ' + results.join('\n  '));
if (logs.length) {
  console.log('\n!!! ' + logs.length + ' CONSOLE/PAGE ERRORS — THIS IS A FAILED BUILD !!!');
  console.log([...new Set(logs)].slice(0, 40).join('\n'));
  process.exitCode = 1;
} else {
  console.log('\nno console errors.');
}
