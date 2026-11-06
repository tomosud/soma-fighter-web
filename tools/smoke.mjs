// Node smoke test: encode authored poses, run the SOMA model on the CPU, decode, and check the result.
//   node tools/smoke.mjs <path to mmm_v6_soma_fp16.onnx or fp32>
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { Vector3 } from 'three';
import { Rig, emptyBatch, encodeRequest, decodeRequest, correctEndpoints, placePose, MAX_FRAMES } from '../src/soma.js';
import { PoseBook } from '../src/poses.js';

const modelPath = process.argv[2];
const rig = new Rig(JSON.parse(readFileSync(new URL('../assets/soma_rig.json', import.meta.url))));
const book = new PoseBook(rig);
const P = rig.positions(book.stance);
console.log('stance: hips', P[0].toArray().map(v => v.toFixed(3)), 'heading', (rig.heading(P) * 180 / Math.PI).toFixed(1),
  'minY', Math.min(...P.map(v => v.y)).toFixed(3), 'L hand', P[rig.index.LeftHand].toArray().map(v => v.toFixed(2)));

const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });

function toFeeds(batch) {
  const n = batch.n, T = (d, a, s) => new ort.Tensor(d, a, s);
  return {
    global_root_values: T('float32', batch.global_root_values, [n, 8, 5]),
    local_root_values: T('float32', batch.local_root_values, [n, 8, 4]),
    local_poses: T('float32', batch.local_poses, [n, 8, 267]),
    has_global_root_values: T('int32', batch.has_global_root_values, [n, 8]),
    has_local_root_values: T('int32', batch.has_local_root_values, [n, 8]),
    has_local_poses: T('int32', batch.has_local_poses, [n, 8]),
    allowed_pred_num_tokens: T('int32', batch.allowed_pred_num_tokens, [n, 11]),
  };
}

// Two requests in one batch: walk 1.2 m forward (facing +X in world), and a jab toward a point.
const ctx = [0, 1, 2, 3].map(() => placePose(book.stance, Math.PI / 2, new Vector3(), new Vector3(0, 0, 0)));
const walkT = [placePose(book.stance, Math.PI / 2, new Vector3(), new Vector3(1.2, 0, 0))];
const jabLocal = book.punch(new Vector3(0.12, 1.5, 0.75), true);
const jabT = [placePose(jabLocal, Math.PI / 2, new Vector3(), new Vector3(0.15, 0, 0))];
const batch = emptyBatch(2);
const a0 = encodeRequest(rig, batch, 0, ctx, walkT, [8, 9, 10, 11, 12]);
const a1 = encodeRequest(rig, batch, 1, ctx, jabT, [6]);
const t0 = performance.now();
const out = await session.run(toFeeds(batch));
console.log('inference ms', (performance.now() - t0).toFixed(0), 'tokens', Array.from(out.pred_num_tokens.data));
const motions = out.pred_global_motions.data, tokens = out.pred_num_tokens.data;
console.log('output dims', out.pred_global_motions.dims, MAX_FRAMES);

for (const [b, align, target, name] of [[0, a0, walkT[0], 'walk'], [1, a1, jabT[0], 'jab']]) {
  const { frames } = decodeRequest(rig, motions, tokens, b, align);
  const first = rig.positions(frames[0]), last = rig.positions(frames.at(-1));
  const c = rig.positions(ctx[3]), t = rig.positions(target);
  const err = (A, B) => Math.max(...A.map((v, j) => v.distanceTo(B[j])));
  console.log(`${name}: ${frames.length} frames; start err ${err(first, c).toFixed(3)} m, end err ${err(last, t).toFixed(3)} m;`,
    'hips path', frames.filter((_, i) => i % 6 === 0).map(f => f.root.x.toFixed(2)).join(' '),
    'minY', Math.min(...frames.flatMap(f => rig.positions(f).map(v => v.y))).toFixed(3));
  // Also compare the model's own joint positions (feature 5..92) with FK from its rotations.
  const o = (b * MAX_FRAMES + 3) * 366;
  const fk = rig.positions(frames[0]);
  const local = new Vector3(motions[o + 5 + 3 * (rig.index.LeftHand - 1)], motions[o + 6 + 3 * (rig.index.LeftHand - 1)], motions[o + 7 + 3 * (rig.index.LeftHand - 1)]);
  console.log('  L hand: model feature', local.toArray().map(v => v.toFixed(3)), 'FK (world)', fk[rig.index.LeftHand].toArray().map(v => v.toFixed(3)));
  const fixed = correctEndpoints(rig, frames, ctx[3], target);
  console.log('  corrected: start err', err(rig.positions(fixed[0]), c).toFixed(4), 'end err', err(rig.positions(fixed.at(-1)), t).toFixed(4));
}

// Chain: the walk's last four frames become the next context (in-distribution context -> small start error).
{
  const { frames } = decodeRequest(rig, motions, tokens, 0, a0);
  const fixed = correctEndpoints(rig, frames, ctx[3], walkT[0]);
  const ctx2 = fixed.slice(-4);
  const tgt2 = [placePose(book.stance, Math.PI / 2, new Vector3(), new Vector3(2.4, 0, 0.3))];
  const b2 = emptyBatch(1);
  const al = encodeRequest(rig, b2, 0, ctx2, tgt2, [8, 9, 10, 11, 12]);
  const o2 = await session.run(toFeeds(b2));
  const res = decodeRequest(rig, o2.pred_global_motions.data, o2.pred_num_tokens.data, 0, al);
  const A = rig.positions(res.frames[0]), C = rig.positions(ctx2[3]);
  const per = A.map((v, j) => [rig.names[j], v.distanceTo(C[j])]).sort((x, y) => y[1] - x[1]).slice(0, 5);
  console.log('chain: frames', res.frames.length, 'start err top', per.map(([n, e]) => `${n}:${e.toFixed(3)}`).join(' '),
    'root err', A[0].distanceTo(C[0]).toFixed(3));
}
