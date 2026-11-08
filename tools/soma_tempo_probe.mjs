// How SOMA's own output is paced, against the Mixamo tempo profiles: generates a jab and a kick on the CPU (as
// tools/smoke.mjs does), measures the striking limb's speed per frame as generated (rate 1), then warps the
// frames with src/tempo.js and measures again.
//   node tools/soma_tempo_probe.mjs [models/mmm_v6_soma_fp16.onnx]
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { Vector3 } from 'three';
import { Rig, emptyBatch, encodeRequest, decodeRequest, correctEndpoints, placePose } from '../src/soma.js';
import { PoseBook } from '../src/poses.js';
import { tempoRates, realizedProfile, profileTimeline, resampleProfile, correlation, findApex } from '../src/tempo.js';

import { fileURLToPath } from 'node:url';
const modelPath = process.argv[2] ?? fileURLToPath(new URL('../models/mmm_v6_soma_fp16.onnx', import.meta.url));
const rig = new Rig(JSON.parse(readFileSync(new URL('../assets/soma_rig.json', import.meta.url))));
const profiles = JSON.parse(readFileSync(new URL('../assets/tempo_profiles.json', import.meta.url))).families;
const book = new PoseBook(rig);
const I = rig.index;
const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
const T = (d, a, s) => new ort.Tensor(d, a, s);
const feeds = b => ({
  global_root_values: T('float32', b.global_root_values, [b.n, 8, 5]), local_root_values: T('float32', b.local_root_values, [b.n, 8, 4]),
  local_poses: T('float32', b.local_poses, [b.n, 8, 267]), has_global_root_values: T('int32', b.has_global_root_values, [b.n, 8]),
  has_local_root_values: T('int32', b.has_local_root_values, [b.n, 8]), has_local_poses: T('int32', b.has_local_poses, [b.n, 8]),
  allowed_pred_num_tokens: T('int32', b.allowed_pred_num_tokens, [b.n, 11]),
});
const r2 = v => +v.toFixed(2);

const ctx = [0, 1, 2, 3].map(() => placePose(book.stance, 0, new Vector3(), new Vector3()));
const jabT = [placePose(book.punch(new Vector3(0.12, 1.5, 0.75), true), 0, new Vector3(), new Vector3(0, 0, 0.15))];
const kickT = [placePose(book.kick(new Vector3(-0.1, 0.9, 0.9)), 0, new Vector3(), new Vector3(0, 0, 0.1))];
const batch = emptyBatch(2);
const a0 = encodeRequest(rig, batch, 0, ctx, jabT, [6]);
const a1 = encodeRequest(rig, batch, 1, ctx, kickT, [6]);
const t0 = performance.now();
const out = await session.run(feeds(batch));
console.log(`inference ${((performance.now() - t0) / 1000).toFixed(1)} s`);
const motions = out.pred_global_motions.data, tokens = out.pred_num_tokens.data;

for (const [b, align, target, name, part, fam] of [[0, a0, jabT[0], 'jab', 'LeftHand', 'straight'], [1, a1, kickT[0], 'kick', 'RightFoot', 'kick_front']]) {
  const { frames } = decodeRequest(rig, motions, tokens, b, align);
  const fixed = correctEndpoints(rig, frames, ctx[3], target);
  const P = fixed.map(p => rig.positions(p)[I[part]]);
  const Phips = fixed.map(p => rig.positions(p)[0]);
  const head = fixed.map(p => rig.positions(p)[I.Head]);
  const even = new Float32Array(P.length).fill(1);
  const raw = realizedProfile(P, even);
  const apex = findApex(P);
  const sp = raw.v.map(r2);
  console.log(`\n${name}: ${fixed.length} frames generated; ${part} path ${r2(raw.t[raw.t.length - 1])} s at rate 1, apex (path turns) at frame ${apex}`);
  console.log('  speed/frame (m/s) as generated:', sp.join(' '));
  console.log(`  as generated: peak ${r2(raw.peak)} m/s, mean ${r2(raw.mean)}, peak/mean ${r2(raw.peakToMean)}, time-to-peak ${r2(raw.timeToPeakFrac)}; hips peak ${r2(Math.max(...realizedProfile(Phips, even).v))} m/s, head peak ${r2(Math.max(...realizedProfile(head, even).v))} m/s`);
  const prof = profiles[fam];
  console.log(`  Mixamo ${fam}: peak ${prof.peakSpeed.med} m/s, peak/mean ${prof.peakToMean.med}, time-to-peak ${prof.timeToPeakFrac.med}, ant ${prof.dur.ant.med} s, accel ${prof.dur.accel.med} s`);
  const o = { apex: fixed.length - 1, phases: ['ant', 'accel'] };
  const rates = tempoRates(fixed, P, prof, o);
  const warped = realizedProfile(P, rates);
  const tl = profileTimeline(prof, ['ant', 'accel']);
  const c = correlation(resampleProfile(tl.t, tl.v, 24), resampleProfile(warped.t, warped.v, 24));
  const c0 = correlation(resampleProfile(tl.t, tl.v, 24), resampleProfile(raw.t, raw.v, 24));
  console.log(`  warped (ant+accel): ${r2(warped.duration)} s, onset frame ${o.out.onset}, peak ${r2(warped.peak)} m/s, peak/mean ${r2(warped.peakToMean)}, shape corr with Mixamo ${r2(c)} (as generated ${r2(c0)})`);
  console.log('  rates:', Array.from(rates, r2).join(' '));
  const o8 = { apex: fixed.length - 1, phases: ['ant', 'accel'], peakSpeed: 8, maxRate: 6 };
  const r8 = tempoRates(fixed, P, prof, o8);
  const w8 = realizedProfile(P, r8);
  console.log(`  warped to an 8 m/s contact: ${r2(w8.duration)} s, played peak ${r2(w8.peak)} m/s, rates ${r2(Math.min(...r8))}..${r2(Math.max(...r8))}`);
}
