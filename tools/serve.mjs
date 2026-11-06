// Minimal static server for local play: node tools/serve.mjs [port]
// Serves the repository root (and models/ if you put a downloaded .onnx there: ?model=models/mmm_v6_soma_fp16.onnx).
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const port = Number(process.argv[2] ?? 8080);
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml' };

createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([/\\])+/, '');
  const file = join(root, path || 'index.html');
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  let st;
  try { st = statSync(file); } catch { res.writeHead(404).end('not found'); return; }
  const target = st.isDirectory() ? join(file, 'index.html') : file;
  try { st = statSync(target); } catch { res.writeHead(404).end('not found'); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(target)] ?? 'application/octet-stream', 'content-length': st.size, 'cache-control': 'no-cache' });
  createReadStream(target).pipe(res);
}).on('error', e => {
  if (e.code !== 'EADDRINUSE') throw e;
  console.log(`already running: http://127.0.0.1:${port}/ (not starting a second server)`);
}).listen(port, '127.0.0.1', () => console.log(`http://127.0.0.1:${port}/`));
