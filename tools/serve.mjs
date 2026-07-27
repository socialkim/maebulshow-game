// Zero-dep static server for the project root. Picks an ephemeral port so many
// agents can run screenshot passes in parallel without colliding.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.css': 'text/css', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.wasm': 'application/wasm',
  '.hdr': 'application/octet-stream', '.bin': 'application/octet-stream',
};

export function serve(port = 0) {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]);
    if (p === '/' || p === '') p = '/index.html';
    const file = path.join(ROOT, path.normalize(p).replace(/^(\.\.[/\\])+/, ''));
    if (!file.startsWith(ROOT)) { res.writeHead(403).end('forbidden'); return; }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404, { 'content-type': 'text/plain' }).end('404 ' + p); return; }
      res.writeHead(200, {
        'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream',
        'cache-control': 'no-store',
      }).end(buf);
    });
  });
  return new Promise(resolve => server.listen(port, '127.0.0.1',
    () => resolve({ server, port: server.address().port, url: `http://127.0.0.1:${server.address().port}/` })));
}

if (import.meta.url === `file://${process.argv[1]}`.replace(/\\/g, '/')) {
  const { url } = await serve(Number(process.argv[2]) || 5173);
  console.log('serving', ROOT, '->', url);
}
