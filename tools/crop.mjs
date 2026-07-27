// node tools/crop.mjs <png> <x> <y> <w> <h> <scale> <out>
import { chromium } from 'playwright';
import fs from 'node:fs';
const [png, x, y, w, h, scale, out] = process.argv.slice(2);
const S = Number(scale) || 2;
const b64 = fs.readFileSync(png).toString('base64');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Number(w) * S, height: Number(h) * S } });
await page.setContent(`<style>html,body{margin:0}canvas{display:block;image-rendering:pixelated}</style><canvas id=c width=${Number(w) * S} height=${Number(h) * S}></canvas>`);
await page.evaluate(async ([b64, x, y, w, h, S]) => {
  const img = new Image();
  img.src = 'data:image/png;base64,' + b64;
  await img.decode();
  const c = document.getElementById('c').getContext('2d');
  c.imageSmoothingEnabled = false;
  c.drawImage(img, x, y, w, h, 0, 0, w * S, h * S);
}, [b64, Number(x), Number(y), Number(w), Number(h), S]);
await page.screenshot({ path: out });
await browser.close();
console.log('wrote', out);
