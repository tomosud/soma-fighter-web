// MotionBricks SOMA inference in a worker: ONNX Runtime Web on WebGPU (falls back to WASM).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session = null;
let backend = '';

const post = (type, data = {}) => self.postMessage({ type, ...data });

// The downloaded model is kept in IndexedDB as one Blob (Cache Storage fails on entries of a few hundred MB,
// and IndexedDB refuses ArrayBuffers over ~127 MB, but Blobs of this size are fine).
const DB = 'motionbricks-models';
const STORE = 'onnx';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idb(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const req = fn(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(req?.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}

async function readCached(db, url) {
  const blob = await idb(db, 'readonly', s => s.get(url));
  return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
}

async function writeCached(db, url, bytes) {
  try {
    await idb(db, 'readwrite', s => s.put(new Blob([bytes]), url));
  } catch (e) {
    console.warn('could not keep the model in IndexedDB:', e);
  }
}

async function fetchModel(url) {
  let db = null;
  try { db = await openDb(); } catch { /* no IndexedDB (private mode etc.) */ }
  for (const old of ['motionbricks-model-v1', 'motionbricks-model-v2']) self.caches?.delete(old)?.catch(() => {}); // earlier versions
  if (db) {
    const hit = await readCached(db, url).catch(() => null);
    if (hit) {
      post('progress', { phase: 'cache', loaded: 1, total: 1 });
      return hit;
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const buf = total ? new Uint8Array(total) : null;
  const parts = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && loaded + value.length <= total) buf.set(value, loaded); else parts.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - lastPost > 100) { post('progress', { phase: 'download', loaded, total }); lastPost = now; }
  }
  let bytes = buf && loaded === total ? buf : null;
  if (!bytes) {
    bytes = new Uint8Array(loaded);
    let o = 0;
    if (buf) { bytes.set(buf.subarray(0, Math.min(loaded, total)), 0); o = Math.min(loaded, total); }
    for (const p of parts) { bytes.set(p, o); o += p.length; }
  }
  post('progress', { phase: 'download', loaded, total: loaded });
  if (db) await writeCached(db, url, bytes);
  return bytes;
}

async function webgpuDevice() {
  if (!self.navigator?.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  // Raise the limits to the adapter's (the default of 8 storage buffers per stage is too low for some ops).
  const want = ['maxStorageBuffersPerShaderStage', 'maxBufferSize', 'maxStorageBufferBindingSize',
    'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY', 'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension'];
  const requiredLimits = {};
  for (const k of want) if (k in adapter.limits) requiredLimits[k] = adapter.limits[k];
  const requiredFeatures = ['shader-f16', 'timestamp-query', 'subgroups'].filter(f => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  const info = adapter.info ?? {};
  return { device, name: [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') };
}

async function load({ url, ep, batch }) {
  const bytes = await fetchModel(url);
  post('progress', { phase: 'session' });
  const errors = [];
  if (ep !== 'wasm') {
    try {
      const gpu = await webgpuDevice();
      if (!gpu) throw new Error('WebGPU is not available');
      ort.env.webgpu.device = gpu.device;
      // A fixed batch size lets ORT fold the shape computations, which otherwise run on the CPU between GPU kernels.
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all', freeDimensionOverrides: { batch_size: batch } });
      backend = `WebGPU${gpu.name ? ` (${gpu.name})` : ''}`;
    } catch (e) {
      errors.push(`WebGPU: ${e.message ?? e}`);
      session = null;
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all', freeDimensionOverrides: { batch_size: batch } });
    backend = 'WASM (CPU)';
  }
  post('ready', { backend, warnings: errors });
}

async function run({ id, n, inputs }) {
  const T = (type, data, dims) => new ort.Tensor(type, data, dims);
  const feeds = {
    global_root_values: T('float32', inputs.global_root_values, [n, 8, 5]),
    local_root_values: T('float32', inputs.local_root_values, [n, 8, 4]),
    local_poses: T('float32', inputs.local_poses, [n, 8, 267]),
    has_global_root_values: T('int32', inputs.has_global_root_values, [n, 8]),
    has_local_root_values: T('int32', inputs.has_local_root_values, [n, 8]),
    has_local_poses: T('int32', inputs.has_local_poses, [n, 8]),
    allowed_pred_num_tokens: T('int32', inputs.allowed_pred_num_tokens, [n, 11]),
  };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const motions = new Float32Array(out.pred_global_motions.data);
  const tokens = new Int32Array(out.pred_num_tokens.data);
  self.postMessage({ type: 'result', id, motions, tokens, ms: performance.now() - t0 }, [motions.buffer, tokens.buffer]);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'run') await run(data);
  } catch (e) {
    post('error', { id: data.id, message: String(e?.message ?? e) });
  }
};
