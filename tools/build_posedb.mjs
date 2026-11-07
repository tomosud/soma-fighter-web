// Mixamo FBX clips -> the reference pose database for motion matching (assets/posedb.bin + .json).
//
//   node tools/build_posedb.mjs [fbx dir = tools/mixamo_src]
//
// Per clip: retarget to the SOMA rig (tools/mixamo_clips.mjs), give in-place stepping the travel its feet imply,
// repeat a looping clip a few times (so its frames have a future), put the first frame facing +Z with the hips
// over the origin. Entries every 2 frames (15 Hz) with the features of src/posedb.js, for the clip and its mirror
// image. Poses are stored once (30 fps, rotations as int16), mirrors are made when a pose is used.
import fs from 'fs';
import path from 'path';
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, clonePose } from '../src/soma.js';
import { DIM, AHEAD, TRAJ, clipCache, features, makeMirror } from '../src/posedb.js';
import { convert } from './mixamo_clips.mjs';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const rig = new Rig(JSON.parse(fs.readFileSync(path.join(root, 'assets/soma_rig.json'), 'utf8')));
const I = rig.index;
const mirror = makeMirror(rig);
const dir = process.argv[2] ?? path.join(root, 'tools/mixamo_src');
const flat = v => new Vector3(v.x, 0, v.z);

const toPoses = frames => frames.map(f => rig.fixRigid({ root: new Vector3(...f.root), rot: f.rot.map(q => new Quaternion(...q).normalize()) }));

/** In-place stepping: the planted (lower) foot stays put, so the body travels by the opposite of its slide. */
function withRootMotion(poses) {
  const far = Math.max(...poses.map(q => flat(q.root.clone().sub(poses[0].root)).length()));
  if (far > 0.15) return { poses, fixed: false };
  let slide = 0;
  const out = [clonePose(poses[0])];
  const off = new Vector3();
  for (let f = 1; f < poses.length; f++) {
    const A = rig.positions(poses[f - 1]), B = rig.positions(poses[f]);
    const foot = A[I.LeftFoot].y < A[I.RightFoot].y ? I.LeftFoot : I.RightFoot;
    const d = flat(B[foot].clone().sub(A[foot]));
    slide += d.length();
    off.sub(d);
    const q = clonePose(poses[f]);
    q.root.add(off);
    out.push(q);
  }
  // only stepping counts (an idle's planted foot barely moves)
  return slide > 0.5 ? { poses: out, fixed: true } : { poses, fixed: false };
}

/** Joint positions relative to the hips on the floor, facing +Z (to compare poses wherever they are). */
function localShape(p) {
  const P = rig.positions(p), h = rig.heading(P), c = Math.cos(-h), s = Math.sin(-h);
  return P.map(v => { const dx = v.x - P[0].x, dz = v.z - P[0].z; return new Vector3(dx * c + dz * s, v.y, -dx * s + dz * c); });
}
const shapeDist = (a, b) => a.reduce((acc, v, i) => acc + v.distanceTo(b[i]), 0) / a.length;

/** A clip whose last frame is (nearly) its first again is repeated, carried on from where each cycle ends. */
function extendLoop(poses, minFrames) {
  const n = poses.length;
  // a locomotion cycle (travels, and ends where it began) repeats; anything else holds its last pose
  const speed = flat(poses[n - 1].root.clone().sub(poses[0].root)).length() / (n / 30);
  if (n < 12 || speed < 0.3 || shapeDist(localShape(poses[0]), localShape(poses[n - 1])) > 0.04) {
    const out = poses.slice();
    while (out.length < minFrames) out.push(clonePose(poses[n - 1]));
    return { poses: out, loop: false };
  }
  const period = n - 1;
  const out = poses.slice();
  while (out.length < minFrames) {
    const last = out[out.length - 1], first = poses[0];
    const dy = rig.heading(rig.positions(last)) - rig.heading(rig.positions(first));
    const off = flat(last.root.clone().sub(first.root));
    for (let k = 1; k <= period; k++) out.push(placePose(poses[k], dy, flat(first.root), off));
  }
  return { poses: out, loop: true };
}

function canonical(poses) {
  const h = rig.heading(rig.positions(poses[0])), at = flat(poses[0].root);
  return poses.map(p => placePose(p, -h, at, at.clone().negate()));
}

// jumps are left out: their take-off looks like any quick rise of the hips (getting up from a crouch), and a jump
// should only come when it is clearly asked for
const EXCLUDE = /jump|hop|leap|kick|knee|stair|ladder|climb|ledge/i;
const files = fs.readdirSync(dir).filter(f => /^m[mx]_.*\.fbx$/i.test(f) && !/\(\d+\)/.test(f) && !EXCLUDE.test(f)).sort();
const clips = [], entries = [];
const needAfter = Math.max(AHEAD, TRAJ.at(-1)) + 1;
let skipped = 0, looped = 0, rooted = 0;
for (const file of files) {
  let frames;
  try { frames = convert(path.join(dir, file)).frames; } catch (e) { console.log('skip', file, e.message); skipped++; continue; }
  let poses = toPoses(frames);
  if (poses.length < 16) { skipped++; continue; }
  const rm = withRootMotion(poses); poses = rm.poses; if (rm.fixed) rooted++;
  const own = poses.length; // entries only on the clip's own frames (not on the repeated or held ones)
  const lp = extendLoop(poses, poses.length + needAfter + 6); poses = lp.poses; if (lp.loop) looped++;
  poses = canonical(poses);
  const name = file.replace(/^m[mx]_/, '').replace(/\.fbx$/i, '');
  const c = clips.length;
  clips.push({ name, frames: poses.length, own, loop: lp.loop, poses });
  // entries: skip the first and last frames (clips start and end in T-ish transitions or cut motion)
  for (const mirrored of [false, true]) {
    const src = mirrored ? poses.map(mirror) : poses;
    const C = clipCache(rig, src);
    const end = lp.loop ? own : own - 4;
    for (let f = 6; f < end && f + needAfter < src.length; f += 2) entries.push({ c, f, m: mirrored ? 1 : 0, x: features(rig, C, f) });
  }
}
console.log(`${clips.length} clips (${skipped} skipped, ${looped} looping, ${rooted} given root motion), ${entries.length} entries`);

// per-dimension scale: robust spread over the database (with a floor so a near-constant feature is not blown up)
const scale = new Float32Array(DIM);
for (let d = 0; d < DIM; d++) {
  // standard deviation (as in motion matching): the many idle entries would make a median-based spread of the
  // velocities tiny and let them dominate
  let m = 0, m2 = 0;
  for (const e of entries) { m += e.x[d]; m2 += e.x[d] * e.x[d]; }
  m /= entries.length;
  scale[d] = Math.max(0.05, Math.sqrt(Math.max(0, m2 / entries.length - m * m)));
}
const QUANT = 4096; // int16 holds +-8 scale units

// binary: poses, then entry clip / frame / mirror, then features
const nj = rig.names.length, frameBytes = 12 + nj * 8;
const totalFrames = clips.reduce((a, c) => a + c.frames, 0);
const poseBytes = totalFrames * frameBytes;
const align4 = x => (x + 3) & ~3;
const off = { poses: 0 };
off.clip = align4(poseBytes);
off.frame = off.clip + align4(entries.length * 2);
off.mirror = off.frame + align4(entries.length * 2);
off.feat = align4(off.mirror + entries.length);
const total = off.feat + entries.length * DIM * 2;
const buf = new ArrayBuffer(total);
const dv = new DataView(buf);
let o = 0, start = 0;
const meta = { about: 'Reference pose database for SOMA Pull (Mixamo clips on the SOMA rig, tools/build_posedb.mjs).', fps: 30, joints: rig.names, clips: [] };
for (const c of clips) {
  meta.clips.push({ name: c.name, start, frames: c.frames, own: c.own, loop: c.loop });
  start += c.frames;
  for (const p of c.poses) {
    dv.setFloat32(o, p.root.x, true); dv.setFloat32(o + 4, p.root.y, true); dv.setFloat32(o + 8, p.root.z, true); o += 12;
    for (const q of p.rot) {
      const qq = q.w < 0 ? [-q.x, -q.y, -q.z, -q.w] : [q.x, q.y, q.z, q.w];
      for (const v of qq) { dv.setInt16(o, Math.round(v * 32767), true); o += 2; }
    }
  }
}
const ec = new Uint16Array(buf, off.clip, entries.length), ef = new Uint16Array(buf, off.frame, entries.length), em = new Uint8Array(buf, off.mirror, entries.length);
const ft = new Int16Array(buf, off.feat, entries.length * DIM);
entries.forEach((e, i) => {
  ec[i] = e.c; ef[i] = e.f; em[i] = e.m;
  for (let d = 0; d < DIM; d++) ft[i * DIM + d] = Math.max(-32767, Math.min(32767, Math.round(e.x[d] / scale[d] * QUANT)));
});
meta.poses = { offset: 0, bytes: poseBytes };
meta.entries = { count: entries.length, clipOffset: off.clip, frameOffset: off.frame, mirrorOffset: off.mirror };
meta.features = { offset: off.feat, dim: DIM, quant: QUANT, scale: Array.from(scale), ahead: AHEAD, traj: TRAJ };
fs.writeFileSync(path.join(root, 'assets/posedb.bin'), Buffer.from(buf));
fs.writeFileSync(path.join(root, 'assets/posedb.json'), JSON.stringify(meta));
console.log(`assets/posedb.bin ${(total / 1048576).toFixed(1)} MB (${totalFrames} frames), features scale e.g. feet ${scale[0].toFixed(2)} hipsVel ${scale[12].toFixed(2)} traj ${scale[35].toFixed(2)}`);
