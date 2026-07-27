// Fast syntax/parse check for every module under src/ (no browser, no GPU).
// Usage: node tools/check.mjs
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = path.join(ROOT, 'src');
const walk = d => fs.readdirSync(d, { withFileTypes: true }).flatMap(e =>
  e.isDirectory() ? walk(path.join(d, e.name)) : (e.name.endsWith('.js') ? [path.join(d, e.name)] : []));

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'bschk-'));
let bad = 0;
for (const f of walk(SRC)) {
  const rel = path.relative(ROOT, f);
  const t = path.join(tmp, path.basename(f, '.js') + '.mjs');
  fs.writeFileSync(t, fs.readFileSync(f));
  try {
    execFileSync(process.execPath, ['--check', t], { stdio: 'pipe' });
    // Cheap contract check: every module must export a named class.
    const src = fs.readFileSync(f, 'utf8');
    const cls = path.basename(f, '.js');
    if (!new RegExp(`export\\s+(class|\\{[^}]*\\b${cls}\\b)`).test(src) && !src.includes(`export class ${cls}`))
      console.log(`WARN  ${rel} — no obvious \`export class ${cls}\``);
  } catch (e) {
    bad++;
    console.log(`FAIL  ${rel}\n${(e.stderr || '').toString().split('\n').slice(0, 6).join('\n')}`);
  }
}
fs.rmSync(tmp, { recursive: true, force: true });
console.log(bad ? `\n${bad} file(s) failed to parse.` : '\nall modules parse OK.');
process.exitCode = bad ? 1 : 0;
