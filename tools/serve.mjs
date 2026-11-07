// Minimal static server for local play: node tools/serve.mjs [port]
// Serves the repository root (and models/ if you put a downloaded .onnx there: ?model=models/mmm_v6_soma_fp16.onnx).
// For local tooling only (it listens on 127.0.0.1): POST /save/<name> writes the body to eval_out/<name> (the
// evaluator's heat maps and scores), and files may be read from other origins (the Mixamo download helper).
import { createServer } from 'node:http';
import { createReadStream, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const port = Number(process.argv[2] ?? 8080);
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.bin': 'application/octet-stream' };
const CORS = { 'access-control-allow-origin': '*', 'access-control-allow-private-network': 'true', 'access-control-allow-headers': '*', 'access-control-allow-methods': 'GET, POST, OPTIONS' };

createServer((req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, CORS).end(); return; }
  const url = new URL(req.url, 'http://x');
  if (req.method === 'POST' && url.pathname.startsWith('/save/')) {
    const name = basename(decodeURIComponent(url.pathname.slice(6)));
    if (!/^[\w.-]+$/.test(name)) { res.writeHead(400, CORS).end('bad name'); return; }
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      mkdirSync(join(root, 'eval_out'), { recursive: true });
      writeFileSync(join(root, 'eval_out', name), Buffer.concat(chunks));
      res.writeHead(200, CORS).end('ok');
    });
    return;
  }
  const path = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '');
  const file = join(root, path || 'index.html');
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  let st;
  try { st = statSync(file); } catch { res.writeHead(404).end('not found'); return; }
  const target = st.isDirectory() ? join(file, 'index.html') : file;
  try { st = statSync(target); } catch { res.writeHead(404).end('not found'); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(target)] ?? 'application/octet-stream', 'content-length': st.size, 'cache-control': 'no-cache', ...CORS });
  createReadStream(target).pipe(res);
}).on('error', e => {
  if (e.code !== 'EADDRINUSE') throw e;
  console.log(`already running: http://127.0.0.1:${port}/ (not starting a second server)`);
}).listen(port, '127.0.0.1', () => console.log(`http://127.0.0.1:${port}/`));
