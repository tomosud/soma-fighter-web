// Mixamo FBX clips -> the reference pose database for motion matching (assets/posedb.bin + .json).
//
//   node tools/build_posedb.mjs [fbx dir = tools/mixamo_src] [--fight]
//
// --fight: the fighting game's database (assets/fightdb): also the fighting clips (mf_*), strikes included, every
// clip tagged with what it is (stance / attack / block / hit / fall / getup / move) and each strike's limb and top.
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
const FIGHT = process.argv.includes('--fight');
const dir = process.argv.slice(2).find(a => !a.startsWith('--')) ?? path.join(root, 'tools/mixamo_src');
const OUT = FIGHT ? 'fightdb' : 'posedb';
// what a clip is, from its name (first match wins)
const TAGS = ['move', 'stance', 'attack', 'block', 'hit', 'fall', 'getup'];
function tagOf(name) {
  // (the fighting clips' file names start with the category guessed when they were picked: not trusted)
  const n = name.toLowerCase().replace(/^(punch|kick|block|hit|fall|getup|stance)_/, '');
  if (/getting_up|get_up|stand_up|standing_up|kip_up/.test(n)) return 'getup';
  if (/receiv|getting_hit|getting_punched|hit_reaction|_hit_|hit_to|hit_on|reaction|react_|stagger|stumble|shove|pushed|impact|clipped|knocked_down|knocked_over/.test(n)) return /knocked|falling|fall/.test(n) ? 'fall' : 'hit';
  if (/fall|knocked_out|knock_out|knockout|collapse|dying|death/.test(n)) return 'fall';
  if (/stretch|warm_up|exercise|workout|bag/.test(n)) return 'move';
  if (/punch|jab|cross|hook|uppercut|elbow|haymaker|kick|knee/.test(n)) return 'attack';
  if (/block|dodge|evade|duck|weave/.test(n)) return 'block';
  if (/fight|boxing|bouncing|combat|stance|ready|side_step|step_forward|step_backward|pivot/.test(n)) return 'stance';
  return 'move';
}
const STRIKERS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'LeftShin', 'RightShin', 'LeftForeArm', 'RightForeArm'];
/** A strike: the limb that moves fastest away from the hips, and the frame it is farthest out (its top). */
function strikeOf(poses, own, name) {
  const P = poses.slice(0, own).map(p => rig.positions(p));
  const knee = /knee/i.test(name), elbow = /elbow/i.test(name), kick = /kick/i.test(name);
  const cands = STRIKERS.filter(n => (knee ? /Shin/.test(n) : elbow ? /ForeArm/.test(n) : kick ? /Foot/.test(n) : /Hand/.test(n)));
  let best = null;
  for (const part of cands) {
    const j = I[part];
    for (let f = 2; f < P.length - 2; f++) {
      const d = P[f][j].clone().sub(P[f][I.Spine2]).length();
      const v = P[f][j].distanceTo(P[f - 2][j]) * 15;
      const score = d + 0.08 * v;
      if (!best || score > best.score) best = { part, apex: f, score, speed: v };
    }
  }
  return best && { part: best.part, apex: best.apex, speed: +best.speed.toFixed(2) };
}
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

/**
 * Clips stand on floors of their own (some a little above or below): put each clip's floor at 0, the floor being
 * where its feet are lowest most of the time (ankle and toe against their heights at rest, 10th percentile).
 */
const ANK = rig.restPos[I.LeftFoot].y, TOE = rig.restPos[I.LeftToeBase].y;
function onFloor(poses, own) {
  const lows = poses.slice(0, own).map(p => { const P = rig.positions(p); return Math.min(P[I.LeftFoot].y - ANK, P[I.RightFoot].y - ANK, P[I.LeftToeBase].y - TOE, P[I.RightToeBase].y - TOE); }).sort((a, b) => a - b);
  const floor = lows[Math.floor(lows.length * 0.1)] ?? 0;
  return { poses: poses.map(p => { const q = clonePose(p); q.root.y -= floor; return q; }), floor };
}

function canonical(poses) {
  const h = rig.heading(rig.positions(poses[0])), at = flat(poses[0].root);
  return poses.map(p => placePose(p, -h, at, at.clone().negate()));
}

// jumps are left out: their take-off looks like any quick rise of the hips (getting up from a crouch), and a jump
// should only come when it is clearly asked for
const EXCLUDE = FIGHT ? /jump|hop|leap|stair|ladder|climb|ledge|mx_kick|mx_knee/i : /jump|hop|leap|kick|knee|stair|ladder|climb|ledge/i;
const files = fs.readdirSync(dir).filter(f => (FIGHT ? /^m[mxf]_.*\.fbx$/i : /^m[mx]_.*\.fbx$/i).test(f) && !/\(\d+\)/.test(f) && !EXCLUDE.test(f)).sort();
const clips = [], entries = [];
const needAfter = Math.max(AHEAD, TRAJ.at(-1)) + 1;
let skipped = 0, looped = 0, rooted = 0;
const floors = [];
for (const file of files) {
  let frames;
  try { frames = convert(path.join(dir, file)).frames; } catch (e) { console.log('skip', file, e.message); skipped++; continue; }
  let poses = toPoses(frames);
  if (poses.length < 16) { skipped++; continue; }
  const rm = withRootMotion(poses); poses = rm.poses; if (rm.fixed) rooted++;
  const own = poses.length; // entries only on the clip's own frames (not on the repeated or held ones)
  const lp = extendLoop(poses, poses.length + needAfter + 6); poses = lp.poses; if (lp.loop) looped++;
  poses = canonical(poses);
  const fl = onFloor(poses, own); poses = fl.poses; floors.push(fl.floor);
  // a clip standing well above or below the floor is on something (a ledge, a beam): not for walking on the floor
  if (Math.abs(fl.floor) > 0.15) { console.log(`  left out (floor ${(fl.floor * 100).toFixed(0)} cm): ${file}`); skipped++; continue; }
  const name = file.replace(/^m[mxf]_/, '').replace(/\.fbx$/i, '');
  const c = clips.length;
  const tag = FIGHT ? tagOf(name) : 'move';
  const strike = tag === 'attack' ? strikeOf(poses, own, name) : null;
  clips.push({ name, frames: poses.length, own, loop: lp.loop, poses, tag, strike });
  // entries: skip the first and last frames (clips start and end in T-ish transitions or cut motion)
  for (const mirrored of [false, true]) {
    const src = mirrored ? poses.map(mirror) : poses;
    const C = clipCache(rig, src);
    const end = lp.loop ? own : own - 4;
    for (let f = 6; f < end && f + needAfter < src.length; f += 2) entries.push({ c, f, m: mirrored ? 1 : 0, x: features(rig, C, f) });
  }
}
console.log(`${clips.length} clips (${skipped} skipped, ${looped} looping, ${rooted} given root motion), ${entries.length} entries`);
const fs2 = floors.map(Math.abs).sort((a, b) => a - b);
console.log(`floors moved: median ${(fs2[fs2.length >> 1] * 100).toFixed(1)} cm, largest ${(fs2.at(-1) * 100).toFixed(1)} cm`);

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
  meta.clips.push({ name: c.name, start, frames: c.frames, own: c.own, loop: c.loop, tag: c.tag, strike: c.strike });
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
fs.writeFileSync(path.join(root, `assets/${OUT}.bin`), Buffer.from(buf));
fs.writeFileSync(path.join(root, `assets/${OUT}.json`), JSON.stringify(meta));
if (FIGHT) { const cnt = {}; clips.forEach(c => { cnt[c.tag] = (cnt[c.tag] ?? 0) + 1; }); console.log('tags', JSON.stringify(cnt)); }
console.log(`assets/${OUT}.bin ${(total / 1048576).toFixed(1)} MB (${totalFrames} frames), features scale e.g. feet ${scale[0].toFixed(2)} hipsVel ${scale[12].toFixed(2)} traj ${scale[35].toFixed(2)}`);
