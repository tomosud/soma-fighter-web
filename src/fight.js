// SOMA Brawl: a fighting game on the same principle as SOMA Pull MM. Every fighter's motion is chosen by motion
// matching over Mixamo reference moments and made by MotionBricks (the player and one opponent in one batch).
//
// - Left button anywhere: the player's hips are pulled toward the pointer (on the floor plane). The player keeps
//   facing the opponent, so pulling away backs off and pulling sideways circles.
// - Right button: a strike. The reference strikes are searched for one that suits how the body is moving now and
//   whose striking limb reaches the opponent's head (or body) at its top.
// - A hit counts when a striking limb (fist, elbow, knee, foot) touches the other's head or body; its strength is
//   the kinetic energy of the closing speed (0.5 m v^2 with the limb's effective mass). The one hit is knocked
//   along the blow: a hit reaction, or a fall and getting up, again chosen from the references by that motion.
// - Opponents come one at a time and walk in; a knocked-out one stays down, then the next comes. The floor is
//   endless and the camera follows (no camera control).
import * as THREE from 'three';
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, clonePose, lerpPose, emptyBatch, encodeRequest, FPS } from './soma.js';
import { Track, Scheduler, BATCH } from './motion.js';
import { Stage, Mannequin } from './render.js';
import { rotateSubtree, twoBoneIK } from './poses.js';
import { PoseDB, queryFeatures, weightVector, meanHeading, bodyFrame, AHEAD, TRAJ, DIM } from './posedb.js';
import { Ragdoll } from './ragdoll.js';

const MB_REV = '54fc563aeaaf819a753d72cc9fdc6713989cdd43';
const MODELS = {
  fp16: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx`,
  fp32: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx`,
};
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const UP = new Vector3(0, 1, 0), ZERO = new Vector3();
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const flat = v => new Vector3(v.x, 0, v.z);
const rand = (a, b) => a + Math.random() * (b - a);

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const I = rig.index;
const ANKLE = rig.restPos[I.LeftFoot].y, TOE = rig.restPos[I.LeftToeBase].y;
const canvas = $('view');
const stage = new Stage(canvas);
stage.makeEndless();
const scene = stage.scene, camera = stage.camera;
const worker = new Worker(new URL('./infer.worker.js', import.meta.url), { type: 'module' });
const scheduler = new Scheduler(rig, worker);

const settings = {
  vmax: 3.5, replan: 0.2, stay: 0.85,
  spring: { k: 14, c: 2.6, coast: 3 }, // the player's hips on a spring to the pointer (1/s^2, 1/s): under-damped, so it swings
  trigger: params.get('trigger') ?? 'click', // 'click' (right button) or 'auto' (strikes set off by the motion, as before)
  enemies: clamp(Number(params.get('enemies') ?? 3), 1, 3), // how many opponents at once
  w: { feetPos: 1, feetVel: 0.75, hipsVel: 1, hipsH: 0.7, hipsVy: 0.8, hands: 0.3, head: 0.3, handsVel: 0.1, trajPos: 1, trajDir: 1.2, handle: 1.5, trajH: 1 },
  damage: { player: 0.9, enemy: 0.3 }, // hp per joule (the opponents hit softer)
  hipsMass: 12, // the part of the body's weight a moving body puts behind a blow (kg)
  regen: 3, // the player's hp back per second after 3 s without being hit
  knockdown: 45, // J in one blow that puts the one hit on the floor
};

// ---------------------------------------------------------------- the reference database

const TAGS = ['move', 'stance', 'attack', 'block', 'hit', 'fall', 'getup'];
const T = Object.fromEntries(TAGS.map((t, k) => [t, k]));
let db = null, entTag = null, change = null, sunk = null, strikes = [], entryAt = new Map(), standPose = null;
PoseDB.load(rig, new URL('../assets/fightdb', import.meta.url).href).then(d => {
  db = d;
  entTag = new Uint8Array(d.n);
  change = new Float32Array(d.n);
  // entries whose motion takes the hips low before its target (a fall, a trip, a slide, a crouch, lying down):
  // not for moving about (the body would sit down out of nothing)
  const hipsY = d.clips.map((c, k) => Float32Array.from({ length: c.frames }, (_, f) => d.pose(k, f).root.y));
  sunk = new Uint8Array(d.n);
  for (let i = 0; i < d.n; i++) {
    const ys = hipsY[d.entClip[i]], f0 = d.entFrame[i], f1 = Math.min(ys.length - 1, f0 + AHEAD + 3);
    for (let f = f0; f <= f1; f++) if (ys[f] < 0.72) { sunk[i] = 1; break; }
  }
  for (let i = 0; i < d.n; i++) {
    const c = d.clips[d.entClip[i]];
    entTag[i] = T[c.tag ?? 'move'];
    entryAt.set(`${d.entClip[i]}:${d.entMirror[i]}:${d.entFrame[i]}`, i);
    const x = d.entryFeatures(i);
    const hipsA = [x[58], x[59], x[60]];
    let ch = 0;
    [[43, 16], [46, 19], [49, 0], [52, 3], [55, 22]].forEach(([a, b]) => { ch += Math.hypot((x[a] - hipsA[0]) - x[b], (x[a + 1] - hipsA[1]) - (x[b + 1] - x[14]), (x[a + 2] - hipsA[2]) - x[b + 2]); });
    change[i] = ch;
    // strikes: entries 0.5-1 s before the clip's top, with where the striking limb is at the top (body frame of
    // the entry's frame)
    if (c.tag === 'attack' && c.strike) {
      const mirrored = !!d.entMirror[i];
      const lead = c.strike.apex - d.entFrame[i];
      if (lead < 14 || lead > 30) continue;
      const part = mirrored ? swapSide(c.strike.part) : c.strike.part;
      const Pf = rig.positions(d.pose(d.entClip[i], d.entFrame[i], mirrored));
      const Pa = rig.positions(d.pose(d.entClip[i], c.strike.apex, mirrored));
      const B = bodyFrame(Pf[0], rig.heading(Pf));
      strikes.push({ i, part, lead, at: B.point(Pa[I[part]]), name: c.name });
    }
  }
  buildKeys(d);
  const idle = d.clips.findIndex(c => /fight_idle|boxing_idle|bouncing_fight_idle|fighting_idle/.test(c.name));
  standPose = d.pose(Math.max(0, idle), 10);
  $('db-info').textContent = `${d.clips.length} クリップ / 攻撃 ${new Set(strikes.map(s => db.entClip[s.i])).size} 種`;
  start();
}).catch(e => { $('db-info').textContent = `読み込めません: ${e.message}`; });
// Key poses, as the original SOMA Fighter did: the moment a strike lands (four frames from the reference strike's
// top) and the moment a hit reaction is at its strongest. Each is put facing +Z with its hips over the origin.
const strikeKeys = [], hitKeys = [], fallKeys = [], guardKeys = [], getupKeys = [];
/** What kind of strike a reference clip is, from its name. */
function familyOf(name, kind) {
  const n = name.toLowerCase();
  if (/sweep|low_kick|leg_kick|to_the_knee/.test(n)) return 'low';
  if (/elbow/.test(n)) return 'elbow';
  if (/knee/.test(n)) return 'knee';
  if (/uppercut/.test(n)) return 'uppercut';
  if (/hook|haymaker/.test(n)) return 'hook';
  if (/front|snap|thrust|lead_foot|push_kick/.test(n)) return 'front';
  if (/roundhouse|side_kick|high_kick|medium_kick|360|round/.test(n)) return 'round';
  if (kind === 'kick') return 'round';
  return 'straight';
}
function keyFrames(d, c, f, mirrored, ref = f) {
  const p0 = d.pose(c, ref, mirrored);
  const h = rig.heading(rig.positions(p0)), at = flat(p0.root);
  return [0, 1, 2, 3].map(k => placePose(d.pose(c, f + k, mirrored), -h, at, at.clone().negate()));
}
/**
 * What a body is doing, in its own frame (hips at the origin, facing +Z, x to its left), for comparing a moment of
 * a strike clip with the body now: where the hands and feet are, the hands' speed, the trunk's twist, the hips'
 * height. Each part scaled to about 1.
 */
const FEAT_W = [1, 1, 1, 1, 1, 1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
const FEAT_SUM = FEAT_W.reduce((a, b) => a + b, 0);
function poseFeat(P, Pnext, dtFrames = 1) {
  const h = rig.heading(P), c = Math.cos(-h), sn = Math.sin(-h), o = P[0];
  const loc = v => [(v.x - o.x) * c + (v.z - o.z) * sn, v.y - o.y, -(v.x - o.x) * sn + (v.z - o.z) * c];
  const f = [];
  for (const sd of ['Left', 'Right']) { const q = loc(P[I[sd + 'Hand']]); f.push(q[0] / 0.4, q[1] / 0.4, q[2] / 0.4); }
  for (const sd of ['Left', 'Right']) { const q = loc(P[I[sd + 'Foot']]); f.push(q[0] / 0.35, (q[1] + o.y) / 0.2, q[2] / 0.35); }
  const sl = loc(P[I.LeftArm]).map((x, i) => x - loc(P[I.RightArm])[i]);
  f.push(Math.atan2(sl[2], sl[0]) / 0.5, (o.y - 0.95) / 0.1);
  for (const sd of ['Left', 'Right']) {
    const a = loc(P[I[sd + 'Hand']]), b = loc(Pnext[I[sd + 'Hand']]);
    f.push(...[0, 1, 2].map(i => ((b[i] - a[i]) * FPS / dtFrames) / 3));
  }
  return f;
}
const featCost = (a, b) => a.reduce((s2, x, i) => s2 + FEAT_W[i] * (x - b[i]) ** 2, 0) / FEAT_SUM;
function buildKeys(d) {
  d.clips.forEach((c, k) => {
    if (c.tag === 'attack' && c.strike && !/getting|hand_stand|restrain/.test(c.name)) {
      for (const mirrored of [false, true]) {
        const part = mirrored ? swapSide(c.strike.part) : c.strike.part;
        const frames = keyFrames(d, k, c.strike.apex, mirrored);
        const tip = rig.positions(frames[0])[I[part]];
        const kind = /Foot/.test(part) ? 'kick' : /Shin/.test(part) ? 'knee' : /ForeArm/.test(part) ? 'elbow' : 'punch';
        if (tip.z < 0.25) continue; // a strike that does not reach forward (a spin, a back kick)
        let lowest = 9;
        for (let f = Math.max(0, c.strike.apex - 10); f <= Math.min(c.frames - 1, c.strike.apex + 12); f++) lowest = Math.min(lowest, d.pose(k, f).root.y);
        if (lowest < 0.7) continue; // the hips drop (a lunge to the floor, a dive): it reads as sitting down
        // what comes after the top, in the same frame of reference: the swing carrying on (a miss) and the limb
        // coming back a little sooner (a blow that met something)
        const own = (c.own ?? c.frames) - 4;
        const miss = keyFrames(d, k, Math.min(own, c.strike.apex + 9), mirrored, c.strike.apex);
        const hit = keyFrames(d, k, Math.min(own, c.strike.apex + 5), mirrored, c.strike.apex);
        // the wind-up (6 and 10 frames before the top, in the top's frame): what the body looks like as the blow
        // starts, and how far the clip's hips travel into it
        const chamber = [6, 10].map(N => {
          const fr = keyFrames(d, k, Math.max(0, c.strike.apex - N), mirrored, c.strike.apex);
          return { N, pose: fr[0], feat: poseFeat(rig.positions(fr[0]), rig.positions(fr[1])) };
        });
        const travel = flat(chamber[0].pose.root).negate().divideScalar(6 / FPS); // (m/s, in the top's frame: +z forward)
        strikeKeys.push({ clip: k, name: c.name, part, kind, family: familyOf(c.name, kind), frames, tip, high: tip.y > 1.25, miss, hit, chamber, travel });
      }
    }
    if (c.tag === 'block' && !/restrain|transition/.test(c.name)) {
      // a guard or a dodge at its fullest (as for the hits below)
      const n = Math.min(c.own ?? c.frames, c.frames) - 4;
      if (n < 6) return;
      const P0 = rig.positions(d.pose(k, 0));
      let best = 0, bd = -1;
      for (let f = 2; f < n; f++) {
        const P = rig.positions(d.pose(k, f));
        const dd = P[I.LeftHand].distanceTo(P0[I.LeftHand]) + P[I.RightHand].distanceTo(P0[I.RightHand]) + P[I.Head].distanceTo(P0[I.Head]) * 2;
        if (dd > bd) { bd = dd; best = f; }
      }
      const name = c.name.toLowerCase();
      if (/hiding|crouched/.test(name)) return;
      const fam = /dodge|duck|weave|evade/.test(name) ? 'dodge' : /low|body|chest/.test(name) ? 'low' : 'high';
      for (const mirrored of [false, true]) guardKeys.push({ clip: k, name: c.name, family: fam, frames: keyFrames(d, k, best, mirrored) });
      return;
    }
    if (c.tag === 'getup' && !/sliding|looking_around/.test(c.name)) {
      const n = Math.min(c.own ?? c.frames, c.frames) - 4;
      let f0 = 0;
      for (let f = 0; f < Math.min(20, n); f++) if (d.pose(k, f).root.y < d.pose(k, f0).root.y) f0 = f;
      if (d.pose(k, f0).root.y > 0.35) return; // not from the floor
      // standing: from where the hips stay up (near the clip's own standing height); on a knee: where they first pass 0.5 m
      const ys = []; for (let f = 0; f < n; f++) ys.push(d.pose(k, f).root.y);
      const top = Math.max(...ys.slice(f0));
      let fs = n - 1;
      while (fs > f0 && ys[fs - 1] > 0.85 * top) fs--;
      let fk = f0;
      while (fk < fs && ys[fk] < 0.5) fk++;
      if (top < 0.75 || fs - f0 < 8 || fk >= fs - 3) return;
      for (const mirrored of [false, true]) {
        const P0 = rig.positions(d.pose(k, f0, mirrored));
        const ang = Math.atan2(P0[I.Head].x - P0[0].x, P0[I.Head].z - P0[0].z), at = flat(P0[0]);
        const canon = f => [0, 1, 2, 3].map(u => placePose(d.pose(k, f + u, mirrored), -ang, at, at.clone().negate()));
        // moments on the floor before getting up (stirring, rolling to the side, propping up on an elbow)
        const stir = [0.35, 0.6].map(u => canon(Math.round(f0 + (fk - f0) * u)));
        getupKeys.push({ clip: k, name: c.name, up: faceUp(P0), flashy: /kip/.test(c.name), kneel: canon(fk), stand: canon(fs), stir });
      }
      return;
    }
    if (c.tag === 'hit' || c.tag === 'fall') {
      // the strongest moment: where the upper body has moved farthest from how the clip began
      const n = Math.min(c.own ?? c.frames, c.frames) - 4;
      if (n < 6) return;
      const P0 = rig.positions(d.pose(k, 0));
      let best = 0, bd = -1;
      for (let f = 2; f < (c.tag === 'fall' ? n : Math.floor(n * 0.7)); f++) {
        const P = rig.positions(d.pose(k, f));
        const dd = (c.tag === 'fall' ? -P[0].y : 0) + P[I.Head].clone().sub(P[0]).distanceTo(P0[I.Head].clone().sub(P0[0])) + P[0].distanceTo(P0[0]) * 0.5;
        if (dd > bd) { bd = dd; best = f; }
      }
      const name = c.name.toLowerCase();
      // falls: only from standing to lying (not a landing from a height, not a dive), lying at the chosen moment
      if (c.tag === 'fall' && (/land|mid_air|in_the_air|falling_idle|jump|explosion|flying|dive|roll/.test(name) || d.pose(k, best).root.y > 0.45)) return;
      if (c.tag === 'hit' && d.pose(k, best).root.y < 0.72) return; // (a reaction that drops the hips reads as sitting down)
      const where = /head|face|jaw|chin|nose/.test(name) ? 'head' : /body|stomach|rib|kidney|liver|side|gut|chest/.test(name) ? 'body' : 'any';
      for (const mirrored of [false, true]) {
        const key = { clip: k, name: c.name, frames: keyFrames(d, k, best, mirrored, c.tag === 'fall' ? 0 : best), where, strength: bd };
        if (c.tag === 'fall') {
          // which way the body falls (where the head ends up, seen from the standing body) and a moment on the way down
          key.dir = flat(rig.positions(key.frames[0])[I.Head]);
          if (key.dir.lengthSq() < 0.04) continue;
          key.dir.normalize();
          const y0 = d.pose(k, 0, mirrored).root.y, yb = d.pose(k, best, mirrored).root.y;
          let fm = 1;
          while (fm < best && d.pose(k, fm, mirrored).root.y > (y0 + yb) / 2 + 0.12) fm++;
          key.mid = keyFrames(d, k, fm, mirrored, 0);
        }
        (c.tag === 'fall' ? fallKeys : hitKeys).push(key);
      }
    }
  });
}
/** Lying face up? (the chest's front, from the shoulder line and the spine) */
function faceUp(P) {
  const lr = P[I.LeftArm].clone().sub(P[I.RightArm]), up = P[I.Head].clone().sub(P[0]);
  return new Vector3().crossVectors(lr, up).y > 0;
}
const toLocal = p => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));
function fromLocalPose(root, L) {
  const rot = [];
  L.forEach((q, j) => { rot.push(rig.parents[j] < 0 ? q.clone() : rot[rig.parents[j]].clone().multiply(q)); });
  return rig.fixRigid({ root: root.clone(), rot });
}
/** Closest points of two segments (p1-q1, p2-q2): { d, a, b }. */
function segSeg(p1, q1, p2, q2) {
  const d1 = q1.clone().sub(p1), d2 = q2.clone().sub(p2), r = p1.clone().sub(p2);
  const a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
  let s2, t2;
  if (a < 1e-9 && e < 1e-9) { s2 = t2 = 0; }
  else if (a < 1e-9) { s2 = 0; t2 = clamp(f / e, 0, 1); }
  else {
    const c = d1.dot(r);
    if (e < 1e-9) { t2 = 0; s2 = clamp(-c / a, 0, 1); }
    else {
      const b = d1.dot(d2), den = a * e - b * b;
      s2 = den > 1e-9 ? clamp((b * f - c * e) / den, 0, 1) : 0;
      t2 = (b * s2 + f) / e;
      if (t2 < 0) { t2 = 0; s2 = clamp(-c / a, 0, 1); } else if (t2 > 1) { t2 = 1; s2 = clamp((b - c) / a, 0, 1); }
    }
  }
  const A = p1.clone().addScaledVector(d1, s2), B = p2.clone().addScaledVector(d2, t2);
  return { d: A.distanceTo(B), a: A, b: B };
}
function swapSide(n) { return n.startsWith('Left') ? `Right${n.slice(4)}` : n.startsWith('Right') ? `Left${n.slice(5)}` : n; }

// ---------------------------------------------------------------- pose helpers

function lowest(P) {
  let low = Math.min(P[I.LeftFoot].y - ANKLE, P[I.RightFoot].y - ANKLE, P[I.LeftToeBase].y - TOE, P[I.RightToeBase].y - TOE);
  for (let j = 0; j < P.length; j++) if (!FOOTJ.has(j)) low = Math.min(low, P[j].y - 0.05);
  return low;
}
const FOOTJ = new Set(['LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase'].map(n => I[n]));
function ground(t) {
  const low = lowest(rig.positions(t));
  if (Math.abs(low) > 0.01) { const q = clonePose(t); q.root.y -= low; return rig.fixRigid(q); }
  return t;
}
/** The generated motion kept on the floor (only the hips' height, smoothed; not while running). */
function groundFrames(frames, onlySink = false) {
  const n = frames.length, off = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const p = frames[k].pose, prev = frames[Math.max(0, k - 1)].pose.root;
    const speed = Math.hypot(p.root.x - prev.x, p.root.z - prev.z) * FPS;
    const low = lowest(rig.positions(p));
    off[k] = speed > 2 || (onlySink && low > 0) ? 0 : Math.abs(low) > 0.015 ? -low : 0;
  }
  for (let k = 0; k < n; k++) {
    let m = 0, c = 0;
    for (let u = Math.max(0, k - 3); u <= Math.min(n - 1, k + 3); u++) { m += off[u]; c++; }
    if (Math.abs(m / c) > 0.005) frames[k].pose.root.y += m / c;
  }
}
function legTo(p, side, goal) {
  const c = ['Leg', 'Shin', 'Foot'].map(n => I[side + n]);
  const P = rig.positions(p);
  const keep = p.rot[c[2]].clone();
  const knee = P[c[1]].clone().sub(P[c[0]].clone().lerp(P[c[2]], 0.5));
  const h = rig.heading(P), fwd = new Vector3(Math.sin(h), 0, Math.cos(h));
  twoBoneIK(rig, p, c[0], c[1], c[2], goal, knee.lengthSq() > 1e-6 ? knee.add(fwd.multiplyScalar(0.3)) : fwd);
  rotateSubtree(rig, p, c[2], keep.multiply(p.rot[c[2]].clone().invert()));
}
function plantedThrough(clip, f0, f1, mirrored) {
  const out = { Left: true, Right: true };
  let prev = null;
  for (let f = f0; f <= f1; f += 2) {
    const P = rig.positions(db.pose(clip, f, mirrored));
    for (const side of ['Left', 'Right']) {
      const a = P[I[side + 'Foot']];
      if (a.y > ANKLE + 0.04) out[side] = false;
      if (prev && Math.hypot(a.x - prev[side].x, a.z - prev[side].z) * FPS / 2 > 0.2) out[side] = false;
    }
    prev = { Left: P[I.LeftFoot].clone(), Right: P[I.RightFoot].clone() };
  }
  return out;
}
/** A limb's tip onto a point (within its reach), on one target. */
function reachLimb(p, part, goal, maxShift = 0.3) {
  const side = part.startsWith('Left') ? 'Left' : 'Right';
  const names = /Hand|ForeArm/.test(part) ? ['Arm', 'ForeArm', 'Hand'] : ['Leg', 'Shin', 'Foot'];
  const c = names.map(n => I[side + n]);
  const q = clonePose(p);
  const P = rig.positions(q);
  const tipNow = P[c[2]];
  let g = goal.clone();
  if (g.distanceTo(tipNow) > maxShift) g = tipNow.clone().addScaledVector(g.sub(tipNow).normalize(), maxShift);
  const len = P[c[1]].distanceTo(P[c[0]]) + P[c[2]].distanceTo(P[c[1]]);
  const to = g.clone().sub(P[c[0]]);
  const tip = to.length() > len * 0.98 ? P[c[0]].clone().addScaledVector(to.normalize(), len * 0.98) : g;
  const bend = P[c[1]].clone().sub(P[c[0]].clone().lerp(P[c[2]], 0.5));
  const keep = q.rot[c[2]].clone();
  twoBoneIK(rig, q, c[0], c[1], c[2], tip, bend.lengthSq() > 1e-6 ? bend : new Vector3(0, 0.5, 1));
  if (names[0] === 'Leg') rotateSubtree(rig, q, c[2], keep.multiply(q.rot[c[2]].clone().invert()));
  return rig.fixRigid(q);
}
// eyes on a point: head and neck, the chest turning a quarter of the way
function logq(q) { let { x, y, z, w } = q; if (w < 0) { x = -x; y = -y; z = -z; w = -w; } const s = Math.hypot(x, y, z); if (s < 1e-9) return new Vector3(); return new Vector3(x, y, z).multiplyScalar((2 * Math.atan2(s, w)) / s); }
function expq(v) { const a = v.length(); return a < 1e-9 ? new Quaternion() : new Quaternion().setFromAxisAngle(v.clone().divideScalar(a), a); }
const qpow = (q, k) => expq(logq(q).multiplyScalar(k));
const turn = (a, b) => new Quaternion().setFromUnitVectors(a.clone().normalize(), b.clone().normalize());
function gaze(p, L) {
  const P = rig.positions(p);
  const head = P[I.Head];
  const want = L.clone().sub(head);
  if (want.lengthSq() < 1e-4) return p;
  const b = rig.heading(P);
  const yaw = clamp(wrap(Math.atan2(want.x, want.z) - b), -1.3, 1.3), pt = clamp(Math.atan2(want.y, Math.hypot(want.x, want.z)), -0.4, 0.35);
  const fwdHead = new Vector3(0, 0, 1).applyQuaternion(p.rot[I.Head].clone().multiply(rig.restRot[I.Head].clone().invert()));
  rotateSubtree(rig, p, I.Chest, new Quaternion().setFromAxisAngle(UP, wrap(b + yaw - Math.atan2(fwdHead.x, fwdHead.z)) * 0.25));
  const fwd2 = new Vector3(0, 0, 1).applyQuaternion(p.rot[I.Head].clone().multiply(rig.restRot[I.Head].clone().invert()));
  const dir = new Vector3(Math.sin(b + yaw) * Math.cos(pt), Math.sin(pt), Math.cos(b + yaw) * Math.cos(pt));
  const q = turn(fwd2, dir);
  for (const [j, share] of [['Neck1', 0.4], ['Head', 0.6]]) rotateSubtree(rig, p, I[j], qpow(q, share));
  return p;
}

// ---------------------------------------------------------------- fighters

function approach(v, want, dt) {
  const s = v.length(), w = want.length();
  if (s > 0.2 && w > 0.2) {
    const a = Math.atan2(v.x, v.z), b = Math.atan2(want.x, want.z);
    const heading = a + clamp(wrap(b - a), -5 * dt, 5 * dt);
    const speed = s + clamp(w * Math.cos(clamp(wrap(b - a), -Math.PI / 2, Math.PI / 2)) - s, -3.6 * dt, 2.6 * dt);
    return v.set(Math.sin(heading) * speed, 0, Math.cos(heading) * speed);
  }
  const diff = want.clone().sub(v);
  const acc = (w < s ? 3.6 : 2.6) * dt;
  if (diff.length() > acc) diff.setLength(acc);
  return v.add(diff);
}

const fighters = [];
class Fighter {
  constructor(isPlayer, color, pos, yaw) {
    this.isPlayer = isPlayer;
    this.track = new Track(rig, { stance: standPose ?? rig.restPose() });
    this.track.reset(pos, yaw);
    this.body = new Mannequin(scene, rig, color);
    this.vel = new Vector3(); this.want = new Vector3();
    this.current = null; this.recent = [];
    this.state = 'free'; this.until = 0; // free, attack, hit, fall, down, getup, out
    this.hp = this.maxHp = isPlayer ? 100 : 60;
    this.attack = null; this.attackQueued = null;
    this.shown = clonePose(this.track.display); this.P = rig.positions(this.shown); this.prevP = this.P.map(v => v.clone());
    this.planTimer = 0; this.lastWant = new Vector3();
    this.nextAttack = clock + rand(2.5, 4); // (opponents)
    this.knock = new Vector3();
    this.lastMatch = null;
    this.rag = new Ragdoll(rig); this.ragK = this.rag.k.slice(); this.ragW = 0; this.ragOn = false;
  }
  get opponent() {
    let best = null, bd = Infinity;
    for (const f of fighters) if (f !== this && f.isPlayer !== this.isPlayer && f.state !== 'out') { const d = f.P[0].distanceTo(this.P[0]); if (d < bd) { bd = d; best = f; } }
    return best;
  }
  remove() { scene.remove(this.body.group); }
}

/** Hips path at TRAJ frames: velocity easing toward the wanted one, the facing turning toward the opponent. */
/**
 * The player's hips are a weight on a spring to the pointer: the force is k * stretch - c * velocity, so pulling
 * the other way makes the body swing back hard (like a yo-yo) and a body that is moving keeps its momentum.
 * Let go and it coasts to a stop.
 */
function springStep(v, pos, target, dt) {
  const sp = settings.spring;
  const a = target ? flat(target.clone().sub(pos)).multiplyScalar(sp.k).addScaledVector(v, -sp.c) : v.clone().multiplyScalar(-sp.coast);
  v.addScaledVector(a, dt);
  v.y = 0;
  if (v.length() > settings.vmax) v.setLength(settings.vmax);
  return v;
}
function predict(F, pose, heading) {
  let yaw = heading;
  const p = flat(pose.root), v = F.vel.clone();
  const opp = F.opponent;
  const out = [];
  // where it looks: the player at what the cursor is on (or where it goes), the opponents at the player
  const look = F.isPlayer ? (F.lookAt ?? (opp ? opp.P[0] : null)) : opp?.P[0];
  // standing still, the body turns only once the look is well off to one side (then it steps round); small drifts
  // are left alone (turning on the spot by small amounts only slides the feet)
  if (look && v.length() < 0.25) {
    const err = wrap(Math.atan2(look.x - p.x, look.z - p.z) - heading);
    if (Math.abs(err) < 0.5 && !F.turning) F.faceHold = true;
    else { F.faceHold = false; F.turning = Math.abs(err) > 0.15; }
  } else { F.faceHold = false; F.turning = false; }
  for (let k = 1; k <= TRAJ.at(-1); k++) {
    if (F.spring) springStep(v, p, F.spring.target, 1 / FPS); else approach(v, F.want, 1 / FPS);
    p.addScaledVector(v, 1 / FPS);
    const face = F.faceHold ? heading : look && F.state !== 'fall' && F.state !== 'down' && Math.hypot(look.x - p.x, look.z - p.z) > 0.3 ? Math.atan2(look.x - p.x, look.z - p.z) : (v.length() > 0.3 ? Math.atan2(v.x, v.z) : yaw);
    yaw += clamp(wrap(face - yaw), -4.5 / FPS, 4.5 / FPS);
    if (TRAJ.includes(k)) out.push({ pos: p.clone(), yaw });
  }
  return out;
}

const qbuf = new Float32Array(DIM);
const ALLOW = { move: [T.move, T.stance, T.block], hit: [T.hit], fall: [T.fall], getup: [T.getup] };
function costOf(i, q, w) {
  const F = db.feat, s = db.scale, k = 1 / db.q;
  let c = 0;
  for (let d = 0; d < DIM; d++) { if (!w[d]) continue; const e = F[i * DIM + d] * k - q[d] / s[d]; c += w[d] * e * e; }
  return c;
}
/** The opponent's vital point (the head) and where a strike should reach: 20 cm through it, so the limb is still fast
 * when it lands (a blow aimed to stop at the surface arrives at its slowest). */
const vital = opp => opp.P[I.Head].clone().add(new Vector3(0, -0.02, 0));
function through(F, opp) {
  const v = vital(opp), d = flat(v.clone().sub(F.P[0]));
  return d.lengthSq() > 1e-6 ? v.addScaledVector(d.normalize(), 0.2) : v;
}

/**
 * A plan for a fighter. kind: 'move' (normal motion matching), 'attack' (a strike at the opponent), 'hit' / 'fall'
 * (reacting along F.knock), 'getup'.
 */
function planReq(F, kind, mode = 'now') {
  return {
    kind: kind === 'move' ? 'move' : kind === 'attack' ? 'attack' : 'hit', mode, interrupt: kind !== 'move' && kind !== 'getup',
    build(tr, a, ctx) {
      const from = ctx[3];
      const Pn = rig.positions(from), Pp = rig.positions(ctx[2]);
      const heading = meanHeading(ctx.map(p => rig.heading(rig.positions(p))));
      if (kind === 'hit' || kind === 'fall') { F.vel.copy(F.knock); F.want.set(0, 0, 0); }
      const traj = predict(F, from, heading);
      const q = queryFeatures(rig, Pp, Pn, heading, traj, null, null, qbuf);
      const w = weightVector(settings.w);
      w.fill(0, 43, 61);
      if (kind === 'fall' || kind === 'getup') { w.fill(0.2, 37, 43); } // facing matters little on the floor
      const opp = F.opponent;
      let best = null, lead = AHEAD, strikePart = null;
      if (kind === 'attack' && opp && strikes.length) {
        // the strike that suits this motion and reaches the opponent's vital point at its top
        const B = bodyFrame(Pn[0], heading);
        const goal = B.point(through(F, opp));
        let bc = Infinity;
        for (const s of strikes) {
          const e = Math.hypot(s.at[0] - goal[0], s.at[1] - goal[1], s.at[2] - goal[2]);
          let c = 6 * e * e / 0.04 + 0.5 * change[s.i];
          if (c >= bc) continue;
          c += costOf(s.i, q, w);
          if (F.lastStrike === db.entClip[s.i]) c += 3; // not the same strike twice running
          if (c < bc) { bc = c; best = { i: s.i, cost: c, clip: db.entClip[s.i], frame: db.entFrame[s.i], mirrored: !!db.entMirror[s.i] }; lead = s.lead; strikePart = s.part; }
        }
        if (best) { F.lastStrike = best.clip; F.attack = { part: strikePart, id: Math.random(), hit: false }; }
      }
      if (!best) {
        const allow = ALLOW[kind === 'attack' ? 'move' : kind];
        let cont = null;
        if (F.current && kind === 'move') {
          const f = F.current.frame + (mode === 'end' ? AHEAD : a - F.current.anchor);
          const fe = f - (f % 2);
          const i = entryAt.get(`${F.current.clip}:${F.current.mirror ? 1 : 0}:${fe}`);
          if (i != null && allow.includes(entTag[i]) && !sunk[i]) cont = { i, frame: fe };
        }
        const recentSet = new Set(F.recent.filter(r => clock - r.t < 1.5).map(r => r.clip));
        const passive = F.vel.length() < 0.05 && F.want.length() < 0.05;
        const movingOn = kind === 'move' && F.isPlayer && !!F.spring?.target && F.vel.length() > 0.5;
        const calm = kind === 'move' ? (passive ? 2 : 0.4) : 0;
        const bias = i => {
          if (!allow.includes(entTag[i]) || (kind === 'move' && sunk[i])) return Infinity;
          let b = calm * change[i];
          if (movingOn && entTag[i] === T.stance) b += 1.5; // pulled along: not a clip that stays put
          if (cont && db.entClip[i] === F.current.clip && db.entFrame[i] < cont.frame && db.entFrame[i] > cont.frame - 40) b += 4;
          if (recentSet.has(db.entClip[i]) && !(cont && db.entClip[i] === F.current.clip)) b += 2;
          return b;
        };
        best = db.search(q, w, bias);
        if (best.i < 0) best = db.search(q, w, i => (ALLOW.move.includes(entTag[i]) && !sunk[i] ? 0 : Infinity)); // nothing of that kind
        if (cont) {
          const c = costOf(cont.i, q, w) + calm * change[cont.i];
          if (best.cost > (movingOn ? 1 : settings.stay) * c) best = { i: cont.i, cost: c, clip: F.current.clip, frame: cont.frame, mirrored: F.current.mirror, stayed: true };
        }
      }
      if (!F.current || F.current.clip !== best.clip) { F.recent.push({ clip: F.current?.clip ?? best.clip, t: clock }); if (F.recent.length > 20) F.recent.shift(); }
      F.current = { clip: best.clip, mirror: best.mirrored, frame: best.frame + (kind === 'attack' ? lead - AHEAD : 0), anchor: a };
      F.lastMatch = { ...best, name: db.clips[best.clip].name, kind };
      // place the matched moment on the body now; the frames `lead` later are the target
      const at = db.pose(best.clip, best.frame, best.mirrored);
      const hAt = meanHeading([-2, -1, 0, 1, 2].map(o => rig.heading(rig.positions(db.pose(best.clip, best.frame + o, best.mirrored)))));
      const yaw = wrap(heading - hAt), pivot = flat(at.root), offset = flat(from.root).sub(pivot);
      let targets = [0, 1, 2, 3].map(k => ground(placePose(db.pose(best.clip, best.frame + lead + k, best.mirrored), yaw, pivot, offset)));
      if (kind === 'move') {
        const keep = plantedThrough(best.clip, best.frame, best.frame + lead + 3, best.mirrored);
        for (const side of ['Left', 'Right']) {
          if (!keep[side]) continue;
          const now = Pn[I[side + 'Foot']].clone();
          targets = targets.map(t => { const r = clonePose(t); legTo(r, side, new Vector3(now.x, Math.max(now.y, ANKLE), now.z)); return rig.fixRigid(r); });
        }
      }
      // eyes on the opponent; a strike's limb onto the vital point (a little)
      const eyesOn = F.isPlayer && F.lookAt ? F.lookAt : opp ? vital(opp) : null;
      if (eyesOn && kind !== 'fall' && kind !== 'getup') { const r = clonePose(targets[0]); gaze(r, eyesOn); targets[0] = rig.fixRigid(r); }
      if (kind === 'attack' && strikePart && opp) targets[0] = reachLimb(targets[0], strikePart, through(F, opp));
      const moving = traj.at(-1).pos.distanceTo(flat(from.root)) > 0.25;
      const posedAll = kind === 'attack' || kind === 'fall';
      const out = targets.map((t, k) => { if (k === 0 || posedAll) return t; const r = clonePose(t); r.free = true; return r; });
      const t0 = Math.max(6, Math.round((lead + 2) / 4));
      const rate = kind === 'attack' ? 1.35 : kind === 'hit' ? 1.15 : 1;
      return {
        targets: out, tokens: kind === 'attack' ? [t0] : [6, 7], rate, fixEnd: kind === 'move' ? (moving ? 'root' : true) : true,
        tag: { kind: kind === 'move' ? (moving ? 'move' : 'idle') : kind === 'attack' ? 'attack' : 'hit' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          const left = (tr2.frames.length - 1 - tr2.t) / FPS / rate;
          if (kind === 'attack') { F.state = 'attack'; F.until = clock + frames.length / FPS / rate + 0.15; }
          if (kind === 'hit') { F.state = 'hit'; F.until = clock + left; }
          if (kind === 'fall') {
            // a moment on the floor, then up
            const end = tr2.frames[tr2.frames.length - 1];
            for (let k = 0; k < FPS * 0.8; k++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
            F.state = 'down'; F.until = clock + left + 0.8;
          }
          if (kind === 'getup') { F.state = 'getup'; F.until = clock + left; }
        },
      };
    },
  };
}

// ---------------------------------------------------------------- strikes and reactions (key poses, short, fast)

const RATE = { punch: 1.8, elbow: 1.8, knee: 1.6, kick: 1.5 };
const WIND = { punch: 0.5, elbow: 0.35, knee: 0.45, kick: 0.6 }; // how far a strike may step in (m)
/**
 * A strike now: one of the reference strikes at the moment it lands, turned to the opponent, stepping in so its
 * own reach is about right, its limb bent onto 8 cm through the target; generated short (6 tokens) and played
 * fast, as the original SOMA Fighter did. Which strike: one whose reach and height suit the distance and the
 * target, a little at random, not the same one twice running; moving in fast prefers a kick or a stepping blow.
 */
/**
 * Pace (per-frame playback rate) of a strike: the wind-up slow, the limb whipping in faster and faster, the
 * contact played fast enough that the limb is shown at about 8 m/s (so the energy follows the pace), then easing
 * out. Opponents wind up slower (it shows what is coming, so it can be stepped out of).
 */
function warpStrike(F, frames, part, base, ready = 0) {
  const j = I[part], n = frames.length;
  const pos = frames.map(f => rig.positions(f.pose)[j]);
  const sp = pos.map((q, i) => (i ? q.distanceTo(pos[i - 1]) * FPS : 0));
  let ip = 1;
  for (let i = 1; i < n; i++) if (sp[i] >= sp[ip]) ip = i;
  const slow = F.isPlayer ? 1.0 + 0.3 * ready : 0.62, ramp = Math.round(8 - 4 * ready); // (wound up already: a short, quick wind-up)
  for (let i = 0; i < n; i++) {
    let r;
    if (i < ip - ramp) r = slow;
    else if (i < ip) r = slow + (2.3 - slow) * ((i - (ip - ramp)) / ramp) ** 1.5;
    else if (i <= ip + 1) r = clamp(8 / Math.max(sp[i], 0.5), 1.4, 2.8);
    else r = 1.3;
    frames[i].rate = r * base / 1.7;
  }
  return frames.reduce((a, f) => a + 1 / (FPS * f.rate), 0);
}
/** Pace of a reaction: snapped by the blow, then drifting (the body carried, slowing down). */
function warpReact(frames, strength) {
  frames.forEach((f, k) => { f.rate = k < 3 ? 2.6 : Math.max(0.55 + 0.3 * (1 - strength), 2.6 * 0.8 ** (k - 2)); });
}
const LEGS = new Set(rig.names.map((n, j) => (/(Left|Right)(UpLeg|Leg|Shin|Foot|Toe)/.test(n) ? j : -1)).filter(j => j >= 0));
/**
 * A hand strike on walking legs: the arms, trunk and head from the strike's key pose, the legs (as they are in the
 * world) from where normal motion would have the body by then, so the footwork carries on under the blow.
 */
function onLegs(strikePose, legPose) {
  const q = clonePose(strikePose);
  for (const j of LEGS) q.rot[j] = legPose.rot[j].clone();
  q.root.y = 0.5 * (strikePose.root.y + legPose.root.y);
  return rig.fixRigid(q);
}
function strikeReq(F, family = null, mode = 'now', kindSel = null) {
  return {
    kind: 'attack', mode, interrupt: mode === 'now',
    build(tr, a, ctx) {
      const from = ctx[3];
      // the body as it will be when the strike starts (the request's context: now + the wait, or the end of a
      // strike being chained from), and its hands' motion
      const feat = poseFeat(rig.positions(ctx[2]), rig.positions(from));
      const hNow = rig.heading(rig.positions(from));
      const aimed = F.isPlayer ? F.aim : null;
      const opp = aimed?.enemy ?? F.opponent;
      const A = flat(from.root);
      let yaw = opp ? Math.atan2(opp.P[0].x - A.x, opp.P[0].z - A.z) : rig.heading(rig.positions(from));
      let c = Math.cos(-yaw), sn = Math.sin(-yaw);
      const local = v => { const dx = v.x - A.x, dz = v.z - A.z; return new Vector3(dx * c + dz * sn, v.y, -dx * sn + dz * c); };
      // when the blow arrives: the wait for the model, then the stretch up to the contact
      const tLand = scheduler.latency * 1.15 + 0.45;
      // where the opponent will be then: its own motion is already generated that far (the player's strikes); an
      // opponent's strike aims where the player is now, so a step aside in the wind-up makes it miss
      let OP = opp?.P;
      if (opp && F.isPlayer) OP = rig.positions(opp.track.frame(opp.track.now + tLand * FPS).pose);
      const head = opp ? OP[I.Head].clone().add(new Vector3(0, -0.02, 0)) : A.clone().add(new Vector3(Math.sin(yaw), 1.5, Math.cos(yaw)));
      const body = opp ? OP[I.Spine2].clone() : head.clone().setY(1.1);
      const speedIn = opp ? F.vel.dot(flat(opp.P[0].clone().sub(A)).normalize()) : 0;
      // where this body will be: the player's hips run on their spring (it may swing back), others glide on
      if (F.isPlayer && F.spring) {
        const v = F.vel.clone(), p = A.clone();
        for (let t = 0; t < tLand; t += 1 / 30) { springStep(v, p, F.spring.target, 1 / 30); p.addScaledVector(v, 1 / 30); }
        A.add(p.sub(A).clampLength(0, 1.2));
      } else A.addScaledVector(F.vel, tLand * 0.6);
      // face where the two will be (the player's strikes)
      if (opp && F.isPlayer) { yaw = Math.atan2(OP[0].x - A.x, OP[0].z - A.z); c = Math.cos(-yaw); sn = Math.sin(-yaw); }
      const low = opp ? OP[I.LeftShin].clone().lerp(OP[I.RightShin], 0.5) : body.clone().setY(0.45);
      // the swing: moving round the opponent (rather than at it), the blow that sweeps the same way (a hook or a
      // round kick from the side the body swings from), and the body's speed feeds it
      const rel = opp ? flat(opp.P[0].clone().sub(flat(from.root))).normalize() : new Vector3(0, 0, 1);
      const vrel = opp ? F.vel.clone().sub(opp.vel) : F.vel.clone();
      const radial = vrel.dot(rel), tang = vrel.clone().addScaledVector(rel, -radial).length();
      const lateral = vrel.dot(new Vector3(rel.z, 0, -rel.x)); // + to the body's left
      const swing = tang > 1.0 && tang > 1.2 * Math.max(radial, 0.3);
      const swingSide = lateral > 0 ? 'Right' : 'Left'; // moving left: the right side comes round
      // the aimed part, where it will be: a punch at the legs goes to the body; a kick at the legs is a low kick
      let aimPart = aimed?.part ?? null;
      if (kindSel === 'punch' && aimPart === 'legs') aimPart = 'body';
      const aimPt = aimed && opp ? OP[aimed.joint].clone().add(aimed.off) : null;
      if (aimPt && aimPart === 'body' && aimed.part === 'legs') aimPt.copy(OP[I.Spine2]);
      const aimOf = k => (aimPt ?? (k.family === 'low' ? low : k.high ? head : body));
      // the body's own motion in its frame (x left, z forward), for the clip whose own travel into the blow matches
      const mv = new Vector3(F.vel.x, 0, F.vel.z).applyAxisAngle(UP, -hNow);
      let best = null, bc = Infinity;
      const byKind = k => !kindSel || (kindSel === 'punch' ? k.kind === 'punch' || k.kind === 'elbow' : k.kind === 'kick' || k.kind === 'knee');
      const byPart = k => !aimPart || (aimPart === 'legs' ? k.family === 'low' || !k.high : aimPart === 'head' ? k.high || k.kind === 'punch' : k.family !== 'low');
      let pool = strikeKeys.filter(k => byKind(k) && (!family || k.family === family));
      if (pool.some(byPart)) pool = pool.filter(byPart);
      if (aimPart === 'legs' && pool.some(k => k.family === 'low')) pool = pool.filter(k => k.family === 'low'); // at the legs: a low kick, a sweep
      for (const k of (pool.length ? pool : strikeKeys)) {
        const goal = local(aimOf(k));
        const step = clamp(goal.z - k.tip.z + 0.08, -0.2, WIND[k.kind] + (F.isPlayer ? 0.25 : 0.1)); // (aim assist: it may step further in, the player most)
        let cost = Math.abs(goal.z - k.tip.z - step) * 4 + Math.abs(goal.y - k.tip.y) * 2 + Math.abs(goal.x - k.tip.x) * 1.5;
        cost += (k.clip === F.lastStrike ? 1.0 : 0) + (k.clip === F.lastStrike2 ? 0.5 : 0) + Math.random() * 0.25;
        // the pose now: the strike whose wind-up the body is already in (or near: a deeper wind-up costs a little more)
        const dPose = Math.min(featCost(feat, k.chamber[0].feat), 0.15 + featCost(feat, k.chamber[1].feat));
        cost += 2.5 * dPose;
        // the momentum: the clip's own travel into the blow like the body's motion now
        cost += 0.35 * new Vector3(mv.x - k.travel.x, 0, mv.z - k.travel.z).lengthSq() / 2;
        if (speedIn > 2.2 && (k.kind === 'kick' || k.kind === 'knee')) cost -= 0.6; // charging in: a kick or a knee
        else if (speedIn > 1.2 && step > 0.3) cost -= 0.3; // coming in: a stepping blow
        if (radial < -0.8) cost += k.family === 'front' || (k.family === 'straight' && k.travel.z < 0.5) ? -0.8 : /hook|uppercut/.test(k.family) ? 0.6 : 0; // backing off: a push kick or a jab
        if (swing && /hook|round|elbow/.test(k.family)) cost += k.part.startsWith(swingSide) ? -1.2 : 0.4; // swinging round: a sweeping blow
        if (cost < bc) { bc = cost; best = { k, step, goal, dPose }; }
      }
      if (!best) return planReq(F, 'move').build(tr, a, ctx);
      const { k, step, dPose } = best;
      F.lastStrike2 = F.lastStrike; F.lastStrike = k.clip;
      const fwd = new Vector3(Math.sin(yaw), 0, Math.cos(yaw));
      const at = A.clone().addScaledVector(fwd, step);
      const aim = aimOf(k).clone();
      let targets = k.frames.map(f => ground(placePose(f, yaw, ZERO, at)));
      // hand strikes: the legs keep walking (from normal motion at this moment), the arms strike
      if (k.kind === 'punch' || k.kind === 'elbow') {
        const keep = { current: F.current, lastMatch: F.lastMatch, recent: F.recent.slice() };
        const loco = planReq(F, 'move').build(tr, a, ctx);
        Object.assign(F, keep);
        targets = targets.map((t, i) => ground(onLegs(t, loco.targets[i])));
      }
      // the four target frames run through the target (from short of it to well past it): they carry the
      // limb's speed into the contact; the fist / foot is bent onto that line (an elbow or a knee is placed by the step)
      const through = [-0.22, -0.08, 0.06, 0.2];
      if (!/ForeArm|Shin/.test(k.part)) targets = targets.map((t, i) => reachLimb(t, k.part, aim.clone().addScaledVector(fwd, through[i]), F.isPlayer ? 0.5 : 0.35));
      // the one being struck sees it coming: a guard or a dodge, or it takes it
      if (opp) defend(opp, F, k);
      if (opp) targets = targets.map(t => { const r = clonePose(t); gaze(r, head); return rig.fixRigid(r); });
      // (it takes over when its motion is spliced in: until then the strike being played is the one that counts)
      const atk = { part: k.part, id: Math.random(), kind: k.kind, key: k, yaw, at, followed: false, landed: false, sweep: swing && /hook|round/.test(k.family), t0: clock };
      F.lastMatch = { name: k.name, kind: 'attack', cost: bc, frame: 0, dPose: +dPose.toFixed(2) };
      F.current = null;
      // already wound up that way: out sooner (less wind-up, a little faster)
      const ready = clamp(1 - dPose / 0.8, 0, 1);
      const rate = RATE[k.kind] * (1 + 0.25 * ready);
      atk.ready = ready;
      return {
        targets, tokens: [6], rate, fixEnd: true, tag: { kind: 'attack' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          F.attack = atk; atk.spliced = true;
          const dur = warpStrike(F, frames, k.part, rate, ready);
          F.state = 'attack'; F.until = clock + dur + 0.6;
        },
      };
    },
  };
}
/**
 * After the top of a strike, at once (no stop): a miss carries on along the reference's own swing (the follow-
 * through, with the body's momentum); a blow that met something comes back sooner. Soft (not held to the pose):
 * MotionBricks settles it, then normal motion carries on.
 */
function followReq(F, mode = 'end') {
  return {
    kind: 'attack', mode, interrupt: mode === 'now',
    build(tr, a, ctx) {
      const A = F.attack;
      if (!A?.key) return planReq(F, 'move', 'end').build(tr, a, ctx);
      const frames = A.landed ? A.key.hit : A.key.miss;
      const targets = frames.map(f => ground(placePose(f, A.yaw, ZERO, A.at)));
      F.lastMatch = { name: `${A.key.name} → ${A.landed ? '当たって戻る' : '振り抜く'}`, kind: 'follow', cost: 0, frame: 0 };
      const rate = A.landed ? 1.5 : 1.3;
      return {
        targets: targets.map((t, k) => { if (k === 0) return t; const r = clonePose(t); r.free = true; return r; }),
        tokens: [6], rate, fixEnd: false, tag: { kind: 'move' },
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          gen.forEach((f, i) => { f.rate = 1.6 - 0.7 * (i / Math.max(1, gen.length - 1)); }); // carried through, easing out
          if (F.attack) F.attack.followSpliced = clock;
          F.until = clock + gen.reduce((s2, f) => s2 + 1 / (FPS * f.rate), 0) - 0.15;
        },
      };
    },
  };
}
/**
 * The body where a blow puts it, built on the pose it has now: the head snaps along the blow (and turns away from
 * a blow from the side), a blow to the body bends the trunk (folding over one to the middle, sideways from the
 * side), the arms fly a little, the hips are pushed along the blow and, for a strong one, a foot steps back to
 * catch it. Then MotionBricks carries the body there and back.
 */
function impactPose(pose, dir, energy, isHead, hitAt) {
  const p = clonePose(pose);
  const P = rig.positions(p);
  const h = rig.heading(P), fwd = new Vector3(Math.sin(h), 0, Math.cos(h)), left = new Vector3(Math.cos(h), 0, -Math.sin(h));
  const d = flat(dir).normalize();
  const lateral = d.dot(left); // + pushed toward the body's left
  const k = clamp(Math.sqrt(energy) / 9, 0.15, 1.1);
  const axis = new Vector3().crossVectors(UP, d).normalize(); // tipping along the blow
  if (isHead) {
    rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(axis, 0.55 * k));
    rotateSubtree(rig, p, I.Head, new Quaternion().setFromAxisAngle(axis, 0.35 * k));
    rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(UP, -0.6 * k * lateral)); // a hook turns the face
    rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(axis, 0.2 * k));
  } else {
    const low = hitAt ? clamp((P[I.Chest].y - hitAt.y) / 0.4, 0, 1) : 0.5; // a blow to the middle folds the body over it
    const front = Math.max(0, -d.dot(fwd)) > 0.5 ? 1 : 0;
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(axis, (front ? -0.35 * low : 0.25) * k));
    rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(axis, (front ? -0.25 * low : 0.2) * k));
  }
  for (const [side, sg] of [['Left', 1], ['Right', -1]]) rotateSubtree(rig, p, I[side + 'Arm'], new Quaternion().setFromAxisAngle(fwd, sg * 0.35 * k));
  const feet = { Left: P[I.LeftFoot].clone(), Right: P[I.RightFoot].clone() };
  p.root.addScaledVector(d, 0.12 * k);
  p.root.y -= 0.05 * k;
  // the foot behind (along the blow) steps back to catch a strong one
  if (k > 0.5) {
    const back = feet.Left.clone().sub(P[0]).dot(d) > feet.Right.clone().sub(P[0]).dot(d) ? 'Left' : 'Right';
    feet[back].addScaledVector(d, 0.3 * k);
  }
  for (const side of ['Left', 'Right']) legTo(p, side, new Vector3(feet[side].x, Math.max(feet[side].y, ANKLE), feet[side].z));
  return rig.fixRigid(p);
}
/** Back to a guard where the body ended up, facing the opponent (soft; MotionBricks finds the way). */
function recoverReq(F) {
  return {
    kind: 'move', mode: 'end',
    build(tr, a, ctx) {
      const from = ctx[3];
      const opp = F.opponent;
      const A = flat(from.root);
      const yaw = opp ? Math.atan2(opp.P[0].x - A.x, opp.P[0].z - A.z) : rig.heading(rig.positions(from));
      const stance = standPose ?? from;
      const t = ground(placePose(stance, yaw - rig.heading(rig.positions(stance)), flat(stance.root), A.clone().sub(flat(stance.root))));
      return { targets: [t], tokens: [6, 7], rate: 1.2, fixEnd: 'root', tag: { kind: 'idle' }, after: (tr2, a0, gen) => { groundFrames(gen); F.until = Math.max(F.until, clock + 0.2); } };
    },
  };
}
// An unseen ragdoll takes the blow: the body as it is, an impulse where the blow lands, muscles and footing loosened
// for a moment; a third of a second on, where it has been knocked to is the target, and where its centre of mass is
// heading says whether a foot must step out to catch it (or whether it goes down).
const rd = new Ragdoll(rig);
function takeBlow(pose, at, impulse) {
  rd.reset(pose);
  const keepK = rd.k.slice(), keepRoot = rd.rootK;
  rd.k = rd.k.map(k => k * 0.3);
  rd.rootK = 60;
  const order = rd.x.map((x, i) => [x.distanceTo(at), i]).sort((a, b) => a[0] - b[0]).slice(0, 4);
  const wsum = order.reduce((a, [d]) => a + 1 / (d + 0.05), 0);
  for (const [d, i] of order) rd.v[i].addScaledVector(impulse, (1 / (d + 0.05)) / wsum / rd.mass[i]);
  for (let t = 0; t < 0.32; t += 1 / 120) rd.step(1 / 120, pose, [1, 1, 1, 1], null);
  const out = rd.pose(pose);
  const { pos, vel } = rd.com();
  const cp = pos.clone().addScaledVector(vel, 0.31);
  rd.k = keepK; rd.rootK = keepRoot;
  return { pose: out, cp: flat(cp), push: flat(vel) };
}
/**
 * Seeing a strike come: the opponents guard (high or low, as the strike comes) or dodge some of the time; the
 * player dodges when pulling away as it comes, and guards now and then when standing. A guard takes most of the
 * blow; a dodge moves the body out of its way (back and a little aside).
 */
function defend(B, A, k) {
  if (B.state !== 'free' && !(B.state === 'attack' && !B.attack?.spliced)) return;
  const away = flat(B.P[0].clone().sub(A.P[0])).normalize();
  let kind = null;
  if (B.isPlayer) {
    return; // the player keeps its own footwork: stepping out of the way is how it avoids a blow
  } else {
    const r = Math.random(), recentlyHit = clock - (B.hitAt ?? -9) < 1.2;
    if (!recentlyHit && r < 0.15) kind = 'dodge';
    else if (!recentlyHit && r < 0.35) kind = k.family === 'low' || !k.high ? 'low' : 'high';
  }
  if (!kind) return;
  B.track.request(guardReq(B, A, kind));
  B.state = 'guard'; B.until = clock + 0.9;
  B.guard = { kind, high: kind === 'high', until: clock + 0.9 };
}
function guardReq(F, A, kind) {
  return {
    kind: 'hit', mode: 'now', interrupt: true,
    build(tr, a, ctx) {
      const At = flat(ctx[3].root);
      const away = At.clone().sub(flat(A.P[0])).normalize();
      const pool = guardKeys.filter(g => g.family === kind);
      const g = (pool.length ? pool : guardKeys)[Math.floor(Math.random() * Math.max(1, (pool.length ? pool : guardKeys).length))];
      if (!g) return planReq(F, 'move').build(tr, a, ctx);
      const side = new Vector3(away.z, 0, -away.x).multiplyScalar(Math.random() < 0.5 ? 1 : -1);
      const dest = kind === 'dodge' ? At.clone().addScaledVector(away, 0.35).addScaledVector(side, 0.2) : At.clone().addScaledVector(away, 0.05);
      const yaw = Math.atan2(-away.x, -away.z);
      const targets = g.frames.map(f => ground(placePose(f, yaw, ZERO, dest)));
      F.current = null;
      F.lastMatch = { name: g.name, kind: kind === 'dodge' ? 'dodge' : 'guard', cost: 0, frame: 0 };
      return {
        targets, tokens: [6], rate: 2.0, fixEnd: true, tag: { kind: 'hit' },
        after: (tr2, a0, frames) => { groundFrames(frames); F.until = clock + (tr2.frames.length - 1 - tr2.t) / FPS / 2.0 + 0.1; tr2.request(recoverReq(F)); },
      };
    },
  };
}
/**
 * Knocked down. A fall that goes the way the blow pushes (a fall backwards for a blow from the front, sideways
 * for a hook, forwards for one from behind) is turned so it falls exactly along the blow, the turn as small as the
 * clips allow. How far the body goes is the blow's energy. A big blow lifts the body off its feet: first the
 * moment half way down, lifted and carried along the blow (the body in the air), then the body lying where it
 * lands; MotionBricks makes the flight, the landing and the slide. Then a moment on the floor (longer after a
 * bigger blow).
 */
function fallPlan(F, ctx, energy, dir) {
  const A = flat(ctx[3].root);
  const blow = flat(dir).normalize();
  const hV = rig.heading(rig.positions(ctx[3])), want = Math.atan2(blow.x, blow.z);
  let best = null, bc = Infinity, yaw = 0;
  for (const k of fallKeys) {
    const y = wrap(want - Math.atan2(k.dir.x, k.dir.z));
    const c = 1.2 * Math.abs(wrap(y - hV)) + Math.random() * 0.5 + (k.clip === F.lastReact ? 0.6 : 0);
    if (c < bc) { bc = c; best = k; yaw = y; }
  }
  F.lastReact = best.clip;
  const dist = clamp(0.5 + energy / 55, 0.6, 2.8);
  const placeAt = (frames, where) => {
    const f0 = placePose(frames[0], yaw, ZERO, ZERO);
    const move = where.clone().sub(flat(f0.root));
    return frames.map(f => ground(placePose(f, yaw, ZERO, move)));
  };
  const lying = placeAt(best.frames, A.clone().addScaledVector(blow, dist));
  const fly = energy > settings.knockdown * 1.2 && best.mid;
  F.current = null;
  F.lastMatch = { name: best.name + (fly ? '（吹き飛ぶ）' : ''), kind: 'fall', cost: bc, frame: 0 };
  const hold = 0.5 + clamp(energy / 200, 0, 0.8);
  const settle = rate => (tr2, a0, gen) => {
    groundFrames(gen);
    const left = (tr2.frames.length - 1 - tr2.t) / FPS / rate;
    const end = tr2.frames[tr2.frames.length - 1];
    // a breath on the floor, then (unless knocked out) stirring: the body does not lie there like a thing
    const still = F.ko ? hold + 1 : 0.15 + 0.25 * Math.random();
    for (let k = 0; k < FPS * still; k++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
    F.state = 'down'; F.until = clock + left + still;
    if (!F.ko) tr2.request(stirReq(F, hold));
  };
  if (!fly) return { targets: lying, tokens: [8, 9], rate: 1.6, fixEnd: true, tag: { kind: 'down' }, after: settle(1.6) };
  // in the air: the moment on the way down, half way along, lifted
  const lift = clamp((energy - settings.knockdown) / 220, 0.1, 0.45);
  const air = placeAt(best.mid, A.clone().addScaledVector(blow, dist * 0.5)).map(t => { const r = clonePose(t); r.root.y += lift; return r; });
  const landing = { kind: 'hit', mode: 'end', build: () => ({ targets: lying, tokens: [6, 7], rate: 1.5, fixEnd: true, tag: { kind: 'down' }, after: settle(1.5) }) };
  return {
    targets: air, tokens: [6], rate: 1.9, fixEnd: true, tag: { kind: 'down' },
    after: (tr2, a0, gen) => { groundFrames(gen, true); F.state = 'fall'; F.until = clock + 5; tr2.request(landing); },
  };
}
/**
 * On the floor: a moment of the getting-up clip that starts lying the way the body lies (stirring, rolling to the
 * side, propping up), laid along the body, played slowly; once or twice, then up (the same clip carries on).
 */
function stirReq(F, hold, n = 0) {
  return {
    kind: 'hit', mode: 'end',
    build(tr, a, ctx) {
      const Pn = rig.positions(ctx[3]), up = faceUp(Pn);
      const ang = Math.atan2(Pn[I.Head].x - Pn[0].x, Pn[I.Head].z - Pn[0].z), at = flat(Pn[0]);
      let pool = getupKeys.filter(g => g.up === up && !g.flashy);
      if (!pool.length) pool = getupKeys;
      const g = F.getupKey && pool.includes(F.getupKey) ? F.getupKey : pool[Math.floor(Math.random() * pool.length)];
      F.getupKey = g;
      const fr = g.stir[Math.min(n, g.stir.length - 1)];
      const targets = fr.map(f => ground(placePose(f, ang, ZERO, at)));
      const rate = 0.8 + 0.4 * Math.random();
      F.lastMatch = { name: `${g.name}（床で）`, kind: 'down', cost: 0, frame: 0 };
      return {
        targets, tokens: [7, 8], rate, fixEnd: true, tag: { kind: 'down' },
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          F.until = clock + (tr2.frames.length - 1 - tr2.t) / FPS / rate;
          // a bigger blow: a little longer on the floor (one more stir)
          if (n === 0 && hold > 0.9) tr2.request(stirReq(F, hold, 1));
        },
      };
    },
  };
}
/**
 * Up again, the way a person gets up from how the body lies (from the back or from the stomach): a getting-up
 * clip that starts lying the same way, laid along the body, its moment on a knee (or on the hands) and then its
 * moment standing; MotionBricks makes the way between. Then back to a guard facing the opponent.
 */
function getupReq(F) {
  return {
    kind: 'hit', mode: 'end',
    build(tr, a, ctx) {
      const Pn = rig.positions(ctx[3]), up = faceUp(Pn);
      const ang = Math.atan2(Pn[I.Head].x - Pn[0].x, Pn[I.Head].z - Pn[0].z), at = flat(Pn[0]);
      let pool = getupKeys.filter(g => g.up === up && (!g.flashy || (!F.isPlayer && Math.random() < 0.35)));
      if (F.getupKey && F.getupKey.up === up) pool = [F.getupKey]; // the same clip as the stirring: it carries on
      F.getupKey = null;
      if (!pool.length) pool = getupKeys.filter(g => g.up === up);
      if (!pool.length) pool = getupKeys;
      if (!pool.length) return planReq(F, 'getup', 'end').build(tr, a, ctx);
      const g = pool[Math.floor(Math.random() * pool.length)];
      const place = frames => frames.map(f => ground(placePose(f, ang, ZERO, at)));
      const kneel = place(g.kneel), stand = place(g.stand);
      F.current = null;
      F.lastMatch = { name: `${g.name} → 起き上がり`, kind: 'getup', cost: 0, frame: 0 };
      const rate = F.isPlayer ? 1.5 : 1.3;
      const rise = {
        kind: 'hit', mode: 'end',
        build: () => ({
          targets: stand.map((t, k) => { if (k === 0) return t; const r = clonePose(t); r.free = true; return r; }),
          tokens: [6, 7], rate, fixEnd: 'root', tag: { kind: 'idle' },
          after: (tr3, a1, gen) => { groundFrames(gen); F.until = clock + (tr3.frames.length - 1 - tr3.t) / FPS / rate; tr3.request(recoverReq(F)); },
        }),
      };
      return {
        targets: kneel, tokens: [7, 8], rate, fixEnd: true, tag: { kind: 'hit' },
        after: (tr2, a0, gen) => { groundFrames(gen); F.state = 'getup'; F.until = clock + 5; tr2.request(rise); },
      };
    },
  };
}
/** Hit: the reaction built from the blow, mixed with a reference reaction at its strongest; then back. */
function reactReq(F, from, energy, isHead, down, dir = null, hitAt = null) {
  return {
    kind: down ? 'down' : 'hit', mode: 'now', interrupt: true,
    build(tr, a, ctx) {
      const A = flat(ctx[3].root);
      const away = A.clone().sub(flat(from));
      if (away.lengthSq() < 1e-6) away.set(1, 0, 0);
      away.normalize();
      if (down && fallKeys.length) return fallPlan(F, ctx, energy, dir ?? away);
      const pool = down ? fallKeys : hitKeys.filter(h => h.where === (isHead ? 'head' : 'body') || h.where === 'any');
      const list = pool.length ? pool : hitKeys;
      // stronger blows: the bigger reactions
      const wantStrength = clamp(0.25 + energy / 120, 0.2, 1.2);
      let best = null, bc = Infinity;
      for (const h of list) { const c = Math.abs(h.strength - wantStrength) + Math.random() * 0.4 + (h.clip === F.lastReact ? 0.5 : 0); if (c < bc) { bc = c; best = h; } }
      F.lastReact = best?.clip;
      const back = down ? 0.35 + clamp(energy / 90, 0, 1.5) : clamp(0.1 + energy / 200, 0.1, 0.55); // a big blow sends the body flying
      const dest = A.clone().addScaledVector(away, back);
      const yaw = Math.atan2(-away.x, -away.z); // reactions face the attacker
      let targets = best.frames.map(f => ground(placePose(f, yaw, ZERO, dest)));
      if (!down) {
        // the body knocked by the blow (the unseen ragdoll), and, if its centre of mass is heading off its feet,
        // a foot stepping out under it; the head and trunk turns of the blow on top
        const J = flat(dir ?? away).normalize().multiplyScalar(Math.sqrt(2 * energy * 4)); // impulse of the limb (kg m/s), as felt
        J.y = 0.15 * J.length();
        const blow = takeBlow(ctx[3], hitAt ?? ctx[3].root.clone().setY(1.2), J);
        const Pk = rig.positions(blow.pose);
        const feet = { Left: Pk[I.LeftFoot], Right: Pk[I.RightFoot] };
        const off = Math.min(...['Left', 'Right'].map(sd => blow.cp.distanceTo(flat(feet[sd]))));
        if (off > 0.25) {
          const pushDir = blow.push.lengthSq() > 1e-6 ? blow.push.clone().normalize() : away;
          const back = feet.Left.clone().sub(Pk[0]).dot(pushDir) > feet.Right.clone().sub(Pk[0]).dot(pushDir) ? 'Left' : 'Right';
          const stepTo = flat(Pk[0]).addScaledVector(pushDir, clamp(off, 0.25, 0.6));
          legTo(blow.pose, back, new Vector3(stepTo.x, ANKLE, stepTo.z));
        }
        const imp = impactPose(blow.pose, dir ?? away, energy * 0.6, isHead, hitAt);
        const L = toLocal(imp), Lr = toLocal(targets[0]);
        L.forEach((q, j) => { if (j > 0) q.slerp(Lr[j], 0.4); });
        const mixed = fromLocalPose(imp.root.clone().lerp(targets[0].root, 0.3), L);
        // the body still moving along the blow on the next frames
        targets = [0, 1, 2, 3].map(k2 => { const t = clonePose(mixed); t.root.addScaledVector(away, 0.02 * k2); if (k2 > 0) t.free = true; return ground(rig.fixRigid(t)); });
      }
      F.current = null;
      F.lastMatch = { name: best.name, kind: down ? 'fall' : 'hit', cost: bc, frame: 0 };
      const rate = down ? 1.45 : energy > 40 ? 1.8 : 2.1;
      return {
        targets, tokens: down ? [8, 9] : [6], rate: down ? 1.6 : rate, fixEnd: true, tag: { kind: down ? 'down' : 'hit' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          if (!down) warpReact(frames, clamp(energy / 45, 0, 1));
          const left = down ? (tr2.frames.length - 1 - tr2.t) / FPS / rate : frames.reduce((s2, f) => s2 + 1 / (FPS * f.rate), 0);
          if (down) {
            const end = tr2.frames[tr2.frames.length - 1];
            for (let k = 0; k < FPS * 0.8; k++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
            F.state = 'down'; F.until = clock + left + 0.8;
          } else { F.state = 'hit'; F.until = clock + left + 0.5; tr2.request(recoverReq(F)); }
        },
      };
    },
  };
}

// ---------------------------------------------------------------- hits

// striking points and the effective mass behind each (kg)
const STRIKE_MASS = { LeftHand: 1.4, RightHand: 1.4, LeftForeArm: 2.0, RightForeArm: 2.0, LeftFoot: 3.5, RightFoot: 3.5, LeftShin: 4.5, RightShin: 4.5 };
const sparks = [];
function spark(at, energy, color = 0xffe9a8) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }));
  m.position.copy(at); m.renderOrder = 20; scene.add(m);
  sparks.push({ m, t: 0, s: 1 + Math.min(4, energy / 40) });
}
/**
 * Contact of A's striking limbs with B's head (sphere) or trunk (capsule). While a limb is in contact its fastest
 * closing speed is kept; the blow lands when the limb starts to slow down or leaves (so a fist that brushes the
 * body on its way out is not counted at its slowest moment). Energy = 0.5 m v^2 with the limb's effective mass.
 */
function checkHits(A, B, dt) {
  // one contact per target (a swing can go through several bodies)
  A.contacts ??= new Map();
  if (A.state === 'out' || B.state === 'out' || B.state === 'fall' || B.state === 'down' || B.state === 'getup') { A.contacts.delete(B); return; } // (no hits on the floor or while getting up)
  const striking = A.state === 'attack' || settings.trigger === 'ram';
  if (!striking) { A.contacts.delete(B); return; }
  const head = B.P[I.Head].clone().add(new Vector3(0, 0.06, 0)), hips = B.P[0], neck = B.P[I.Neck1];
  const vB = B.P[0].clone().sub(B.prevP[0]).divideScalar(dt), vA = A.P[0].clone().sub(A.prevP[0]).divideScalar(dt);
  const legSeg = ['Left', 'Right'].map(sd => [B.P[I[sd + 'Leg']], B.P[I[sd + 'Foot']]]);
  const side = A.attack?.part?.startsWith('Left') ? 'Left' : 'Right';
  let touching = null;
  for (const [part, m] of Object.entries(STRIKE_MASS)) {
    // the striking limb (and its own forearm / shin): not the other arm or leg brushing past
    if (A.state === 'attack' && A.attack && part !== A.attack.part && !(part.startsWith(side) && /ForeArm|Shin/.test(part))) continue;
    const p = A.P[I[part]], v = p.clone().sub(A.prevP[I[part]]).divideScalar(dt);
    A.peak = Math.max(A.peak ?? 0, v.length());
    const seg = neck.clone().sub(hips), tt = clamp(p.clone().sub(hips).dot(seg) / seg.lengthSq(), 0, 1);
    const onTrunk = hips.clone().addScaledVector(seg, tt);
    let dHead = p.distanceTo(head) - 0.12, dTrunk = p.distanceTo(onTrunk) - 0.16;
    let c = null;
    // an arm in the way (a guard, or just there): it takes the blow
    let onArm = false;
    for (const sd of ['Left', 'Right']) {
      const a0 = B.P[I[sd + 'Arm']], b0 = B.P[I[sd + 'Hand']];
      const sg = b0.clone().sub(a0), u = clamp(p.clone().sub(a0).dot(sg) / sg.lengthSq(), 0, 1), on = a0.clone().addScaledVector(sg, u);
      const dl = p.distanceTo(on) - 0.06;
      if (dl < dTrunk && dl < dHead - 0.04) { dTrunk = dl; c = on; onArm = true; }
    }
    // the legs (a sweep, a low kick)
    for (const [a0, b0] of legSeg) {
      const sg = b0.clone().sub(a0), u = clamp(p.clone().sub(a0).dot(sg) / sg.lengthSq(), 0, 1), on = a0.clone().addScaledVector(sg, u);
      const dl = p.distanceTo(on) - 0.09;
      if (dl < dTrunk && dl < dHead) { dTrunk = dl; c = on; onArm = false; }
    }
    const isHead = !c && dHead < dTrunk + 0.03; // a fist near both is a blow to the head
    const d = Math.min(dHead, dTrunk);
    if (d > 0.07) continue;
    c ??= isHead ? head : onTrunk;
    const n = c.clone().sub(p).normalize();
    const closing = Math.min(11, v.clone().sub(vB).dot(n)); // (a fist or foot of a person: at most about 11 m/s)
    const behind = clamp(vA.clone().sub(vB).dot(n), 0, 4); // the body coming in behind the blow
    if (!touching || closing > touching.closing) touching = { part, m: onArm ? m * 0.5 : m, closing, behind, isHead, n, at: p.clone(), onArm };
  }
  const id = A.attack?.id ?? `ram${Math.floor(clock * 2)}`;
  const C = A.contacts.get(B);
  if (touching && touching.closing > 0.5) {
    if (!C || C.id !== id) A.contacts.set(B, { id, best: touching, done: false });
    else if (!C.done) {
      if (touching.closing > C.best.closing) C.best = touching;
      else if (touching.closing < 0.7 * C.best.closing) settle(A, B, C);
    }
  } else if (C && !C.done) settle(A, B, C);
  if (C?.done && !C.landed) A.contacts.delete(B); // a brush that did not land: the real contact may still come
}
function settle(A, B, C) {
  C.done = true;
  const b = C.best;
  if (b.closing < 1.2 || B.lastHitBy === C.id) return;
  B.lastHitBy = C.id; C.landed = true;
  // the limb's energy, and the body's own momentum behind it (a swinging, charging body hits heavier)
  const e = 0.5 * b.m * b.closing * b.closing * (b.isHead ? 1.2 : 1) + 0.5 * settings.hipsMass * (A.isPlayer ? 1 : 0.3) * b.behind * b.behind; // (the opponents lunge lighter)
  land(A, B, e, b.n, b.at, b.isHead);
}
function land(A, B, energy, dir, at, isHead) {
  // a touch (a glancing contact, little in it): a small mark, a small sway of the body; no stop, no shake, no
  // reaction motion - the striker carries on through
  if (energy < 12 && B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy) > 0) {
    B.hp = Math.max(0, B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy));
    spark(at, energy * 0.3, 0xd8d8d8);
    const J = flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6) * 0.6);
    jolt(B, at, J, 0.7, 0.25);
    lastHit = { energy, dmg: 0, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, touch: true };
    hudFight();
    return;
  }
  // a guard up at the right height takes most of it; the strike is stopped (it comes back as a hit)
  if (B.guard && clock < B.guard.until && B.guard.kind !== 'dodge' && (B.guard.high === isHead || Math.random() < 0.4)) {
    energy *= 0.2;
    B.blocked = true;
    jolt(B, at, flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6)), 0.5, 0.3);
    if (A.attack) A.attack.landed = true;
    spark(at, energy, 0x9fd3ff);
    shake = Math.max(shake, 0.02);
    B.hp = Math.max(0, B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy));
    lastHit = { energy, dmg: 0, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, blocked: true };
    hudFight();
    return;
  }
  B.hitAt = clock;
  {
    const down0 = B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy) <= 0 || energy > settings.knockdown * (B.isPlayer ? 1.6 : 1);
    const J = flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6)); // the momentum the blow puts in (kg m/s)
    J.y = (isHead ? 0.25 : 0.1) * J.length();
    jolt(B, at, J, clamp(0.35 - energy / 300, down0 ? 0.04 : 0.08, 0.3), clamp(0.3 + energy / 120, 0.3, 0.9), down0);
    jolt(A, at, J.clone().multiplyScalar(-0.12), 0.75, 0.2);
  }
  const dmg = energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy);
  B.hp = Math.max(0, B.hp - dmg);
  spark(at, energy);
  const push = flat(dir).normalize().multiplyScalar(clamp(0.35 * Math.sqrt(energy), 0.6, 2.6));
  B.knock.copy(push);
  const down = B.hp <= 0 || energy > settings.knockdown * (B.isPlayer ? 1.6 : 1); // (the player stays up longer)
  B.state = down ? 'fall' : 'hit';
  B.until = clock + 3;
  B.track.request(reactReq(B, A.P[0], energy, isHead, down, dir, at));
  if (A.attack) {
    if (A.attack.followed && !A.attack.landed && !A.attack.sweep && !A.attack.chained) A.track.request(followReq(A, 'now'));
    A.attack.landed = true;
  }
  B.body.flash = 1;
  // the stop and the shake grow with the blow (a heavy one is felt, an ordinary one only marked)
  hitstop = energy < 45 ? clamp(0.02 + energy / 1500, 0.02, 0.05) : clamp(0.06 + energy / 1500, 0.06, 0.14);
  shake = Math.max(shake, energy < 45 ? clamp(energy / 900, 0.01, 0.04) : clamp(energy / 500, 0.06, 0.14));
  if (down || energy > 60) slowUntil = performance.now() + clamp(energy * 4, 160, 420); // a heavy one: a moment of slow motion
  if (!B.isPlayer && (energy > 45 || down)) { crew.watchUntil = clock + 0.9; crew.swapAt = clock + 0.9; } // the others watch, then the next comes
  if (B.state === 'attack' || B.attack) B.attack = null; // a hit cancels the one hit's own strike
  B.attackQueued = null;
  if (B.hp <= 0) B.ko = true;
  lastHit = { energy, dmg, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock };
  hudFight();
}
let lastHit = null, hitstop = 0, shake = 0, slowUntil = 0;
const jumps = [];

// ---------------------------------------------------------------- the player and the opponents

let player = null, kos = 0, spawnAt = 0;
function start() {
  player = new Fighter(true, 0xd9b26a, new Vector3(0, 0, 0), 0);
  fighters.push(player);
  spawnAt = clock + 0.5;
}
function spawn() {
  // in front of the player, a little to one side when others are already there
  const others = fighters.filter(f => !f.isPlayer && f.state !== 'out').length;
  const h = rig.heading(player.P) + (others ? (others % 2 ? 0.9 : -0.9) : 0), d = 5.5;
  const pos = flat(player.P[0]).add(new Vector3(Math.sin(h), 0, Math.cos(h)).multiplyScalar(d));
  const e = new Fighter(false, 0x7f97c9, pos, h + Math.PI);
  e.hp = e.maxHp = 50 + 10 * Math.min(kos, 5);
  fighters.push(e);
  hudFight();
}
/**
 * The opponents play as a group. One at a time is the attacker: it closes in, circles a little to look for an
 * opening, then lunges (its strike comes out of the lunge), sometimes only feints, and backs off after it has
 * struck (hit and away). The others take places around the player (to its sides and behind, where it is not
 * looking), keep their distance there and circle. After two goes, or when it is hurt, the attacker hands over;
 * whoever has the best angle (the player's blind side) is chosen next. When one takes a big hit or goes down, the
 * others step back a moment to watch, and then one comes in at once.
 */
const crew = { attacker: null, goes: 0, swapAt: 0, watchUntil: 0 };
function enemies() { return fighters.filter(f => !f.isPlayer && f.state !== 'out'); }
function pickAttacker() {
  const P = player, list = enemies().filter(e => e.state !== 'down' && e.state !== 'fall');
  if (!list.length) return null;
  const h = rig.heading(P.P), fwd = new Vector3(Math.sin(h), 0, Math.cos(h));
  // blind side first (behind and to the sides score higher), near ones a little more, not the same one again
  let best = null, bc = -Infinity;
  for (const e of list) {
    const to = flat(e.P[0].clone().sub(P.P[0])), d = to.length();
    const behind = -to.normalize().dot(fwd);
    const c = 0.8 * behind - 0.3 * d + (e === crew.attacker ? -0.8 : 0) + Math.random() * 0.6;
    if (c > bc) { bc = c; best = e; }
  }
  crew.attacker = best; crew.goes = 0; crew.swapAt = clock + rand(5, 8);
  best.phase = 'approach'; best.phaseUntil = clock + 3;
  return best;
}
function think(E, dt) {
  const P = player;
  if (!P || E.state !== 'free') { E.want.set(0, 0, 0); return; }
  const to = flat(P.P[0].clone().sub(E.P[0])), d = to.length();
  to.normalize();
  const side = new Vector3(to.z, 0, -to.x);
  const playerDown = P.state === 'down' || P.state === 'fall';
  if (!crew.attacker || crew.attacker.state === 'out' || clock > crew.swapAt || crew.goes >= 2) pickAttacker();
  // keep off each other
  const apart = new Vector3();
  for (const o of enemies()) if (o !== E) { const dd = flat(E.P[0].clone().sub(o.P[0])); const l = dd.length(); if (l < 1.4 && l > 1e-3) apart.addScaledVector(dd.normalize(), (1.4 - l) * 1.5); }
  if (clock < crew.watchUntil || playerDown) {
    // watching: back to a ring a little further off, side-stepping
    E.want.copy(to).multiplyScalar(clamp((d - 2.6) * 1.2, -1.2, 0.8)).addScaledVector(side, 0.3 * Math.sin(clock + E.maxHp)).add(apart);
    return;
  }
  if (E === crew.attacker) {
    const ph = E.phase ?? 'approach';
    if (ph === 'approach') {
      E.want.copy(to).multiplyScalar(clamp((d - 1.45) * 1.8, -0.6, 1.8)).add(apart);
      if (d < 1.7 || clock > E.phaseUntil) { E.phase = 'circle'; E.phaseUntil = clock + rand(0.5, 1.4); E.circleDir = Math.random() < 0.5 ? 1 : -1; }
    } else if (ph === 'circle') {
      // look for an opening: around the player at about 1.5 m
      E.want.copy(side).multiplyScalar(1.1 * E.circleDir).addScaledVector(to, clamp((d - 1.5) * 1.5, -0.8, 0.8)).add(apart);
      if (clock > E.phaseUntil) { E.phase = Math.random() < 0.25 ? 'feint' : 'lunge'; E.phaseUntil = clock + (E.phase === 'feint' ? 0.25 : 0.65); }
    } else if (ph === 'lunge') {
      E.want.copy(to).multiplyScalar(2.3);
      if (clock > E.phaseUntil) { E.phase = 'retreat'; E.phaseUntil = clock + rand(0.5, 0.8); crew.goes++; }
    } else if (ph === 'feint') {
      E.want.copy(to).multiplyScalar(1.8);
      if (clock > E.phaseUntil) { E.phase = 'retreat'; E.phaseUntil = clock + 0.4; }
    } else if (ph === 'retreat') {
      E.want.copy(to).multiplyScalar(-1.5).addScaledVector(side, 0.5 * (E.circleDir ?? 1)).add(apart);
      if (clock > E.phaseUntil) { E.phase = 'circle'; E.phaseUntil = clock + rand(0.6, 1.6); }
    }
    return;
  }
  // the others: a place around the player (to its sides and behind), kept, circling slowly
  const others = enemies().filter(e => e !== crew.attacker);
  const k = others.indexOf(E);
  const h = rig.heading(P.P) + (k === 0 ? 2.0 : -2.0) + 0.4 * Math.sin(clock * 0.3 + k);
  // round the player along the circle to the place (not through the player), keeping about 2.3 m
  const at = Math.atan2(-to.x, -to.z); // where this one is, seen from the player
  const e = wrap(h - at);
  // (moving along `side` lowers that angle, so -side goes round to a larger one)
  E.want.copy(side).multiplyScalar(-Math.sign(e) * Math.min(Math.abs(e) * 1.6, 1.5)).addScaledVector(to, clamp((d - 2.3) * 1.3, -1.2, 1.2)).add(apart);
  if (E.want.length() > 1.8) E.want.setLength(1.8);
}
// ---------------------------------------------------------------- input (no camera control)

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let scripted = null; // (tests: { point } in place of the cursor)
function floorPoint(x, y) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const out = new Vector3();
  return ray.ray.intersectPlane(new THREE.Plane(UP, -0.95), out) ? out : null; // the plane at hip height
}
// The mouse moves the body without a button: the hips are drawn on their spring toward the point under the cursor
// (at hip height); over an opponent they stop at striking distance (swept through fast, the body slams into it).
// Left button: a punch, right button: a kick, at what the cursor is on. Buttons are read from the change of
// e.buttons on every pointer event (a second button pressed while one is held gives no pointerdown).
const mouse = { x: 0, y: 0, seen: false };
let buttonsPrev = 0;
function onButtons(e) {
  const b = e.buttons ?? 0, pressed = b & ~buttonsPrev;
  buttonsPrev = b;
  mouse.x = e.clientX; mouse.y = e.clientY; mouse.seen = true;
  if (!player) return;
  if (pressed & 1) { player.strikeInput = { t: clock, kind: 'punch' }; cueStrike('punch'); }
  if (pressed & 2) { player.strikeInput = { t: clock, kind: 'kick' }; cueStrike('kick'); }
}
canvas.addEventListener('pointerdown', e => {
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  e.preventDefault();
  onButtons(e);
});
const flickTrail = [];
canvas.addEventListener('pointermove', e => {
  onButtons(e);
  flickTrail.push({ x: e.clientX, y: e.clientY, t: performance.now() });
  while (flickTrail.length && performance.now() - flickTrail[0].t > 150) flickTrail.shift();
});
/**
 * A quick short flick of the pointer near the opponent is a crisp strike, its kind from the flick's direction on
 * screen: toward the opponent a straight (or a front kick when very quick), across a hook or an elbow (a round kick
 * when very quick), downward a low kick, sweep or knee, upward an uppercut (or a high kick).
 */
function flickStrike() {
  const P = player, o = P?.opponent;
  if (!o || flickTrail.length < 3) return;
  const a = flickTrail[0], b = flickTrail[flickTrail.length - 1];
  const dtm = (b.t - a.t) / 1000;
  if (dtm < 0.03) return;
  const fx = b.x - a.x, fy = b.y - a.y, len = Math.hypot(fx, fy), speed = len / dtm;
  if (speed < 1300 || len < 18 || len > 220) return;
  const d = flat(o.P[0].clone().sub(P.P[0])).length();
  if (d > 1.9 || (P.state !== 'free' && !(P.state === 'attack' && clock > P.until - 0.25)) || clock < (P.flickAfter ?? 0)) return;
  const r = canvas.getBoundingClientRect();
  const toScreen = v => { const q = v.clone().project(camera); return [(q.x + 1) / 2 * r.width, (1 - q.y) / 2 * r.height]; };
  const ps = toScreen(P.P[I.Chest]), os = toScreen(o.P[I.Chest]);
  let ex = os[0] - ps[0], ey = os[1] - ps[1];
  const el = Math.hypot(ex, ey) || 1; ex /= el; ey /= el;
  const along = (fx * ex + fy * ey) / len, up = -fy / len;
  const fast = speed > 2600;
  let family;
  if (up < -0.6) family = d < 0.95 ? 'knee' : 'low';
  else if (up > 0.6) family = fast ? 'round' : 'uppercut';
  else if (along > 0.55) family = fast ? 'front' : 'straight';
  else family = d < 0.95 ? 'elbow' : fast ? 'round' : 'hook';
  P.flickAfter = clock + 0.35;
  flickTrail.length = 0;
  P.track.request(strikeReq(P, family));
  P.state = 'attack'; P.until = clock + 1.0; P.strikeAfter = clock + 0.6;
}

canvas.addEventListener('pointerup', onButtons);
canvas.addEventListener('pointercancel', () => { buttonsPrev = 0; });
canvas.addEventListener('pointerleave', () => { mouse.seen = false; });
/**
 * A strike comes from the motion: driving at the opponent fast enough, the strike is set off so that it lands
 * where the two bodies will meet (the time to contact from the closing speed); the faster the approach, the
 * heavier the strike (kicks, stepping blows), and the body's momentum goes into it.
 */
function autoStrike(F) {
  const o = F.opponent;
  if (!o || F.state !== 'free' || clock < (F.strikeAfter ?? 0) || o.state === 'down' || o.state === 'fall') return;
  const rel = flat(o.P[0].clone().sub(F.P[0])), d = rel.length();
  if (d > 1.9 || d < 0.3) return;
  // (the body's actual speed, from what is shown: the commanded one runs ahead of it)
  const vSelf = flat(F.P[0].clone().sub(F.prevP[0])).multiplyScalar(60), vOpp = flat(o.P[0].clone().sub(o.prevP[0])).multiplyScalar(60);
  F.vShown = (F.vShown ?? vSelf).lerp(vSelf, 0.2); o.vShown = (o.vShown ?? vOpp).lerp(vOpp, 0.2);
  const dir = rel.clone().normalize(), vr = F.vShown.clone().sub(o.vShown);
  const closing = vr.dot(dir), tang = vr.clone().addScaledVector(dir, -closing).length();
  const swinging = F.isPlayer && d < 1.5 && tang > 1.5 && closing > -0.3; // swinging past, close
  if (!swinging) {
    if (closing < (F.isPlayer ? 1.1 : 0.9)) return;
    const tc = (d - 0.85) / closing; // until the bodies are at striking distance
    if (tc > 0.7 || tc < -0.1) return;
  }
  F.strikeAfter = clock + 0.7;
  strike(F);
}
/**
 * The player's fist or foot homes in on the target over the last stretch (up to 20 cm, more the nearer it is):
 * the generated motion is what is shown, bent a little where the target has moved since it was asked for.
 */
/**
 * The press answered at once: until the strike's generated motion is shown (the model takes a moment), the body
 * loads up - the trunk turns away from the side that will strike (toward the rear hand), the shoulders draw back a
 * little and the hips sink - over 0.12 s, and lets go as the strike's motion takes over.
 */
function anticipate(F) {
  const A = F.antic;
  if (!A) return;
  const t = clock - A.t0;
  if (F.track.tag.kind === 'attack' && !A.until) A.until = clock;
  const w = A.until ? Math.max(0, 1 - (clock - A.until) / 0.12) : Math.min(1, t / 0.12);
  if ((A.until && w <= 0) || t > 1.2) { F.antic = null; return; }
  const p = clonePose(F.shown);
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, A.side * 0.22 * w));
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(UP, A.side * 0.12 * w));
  p.root.y -= 0.03 * w;
  F.shown = rig.fixRigid(p);
}
/**
 * Feet that the motion plants stay where they were put down (the generated motion slides them a little, most of
 * all turning on the spot or being pushed): while a foot's contact holds, the leg is bent to keep it there; it lets
 * go when the contact ends or the body has moved too far from it, easing in and out.
 */
function footLock(F) {
  if (['fall', 'down', 'getup', 'out'].includes(F.state) || F.ragOn) { F.locks = null; return; }
  const fr = F.track.frames[Math.min(F.track.frames.length - 1, Math.floor(F.track.t))].pose;
  const c = fr.contact;
  if (!c) return;
  F.locks ??= { Left: null, Right: null };
  const P = rig.positions(F.shown);
  let p = null;
  for (const [side, k] of [['Left', 0], ['Right', 2]]) {
    const on = Math.max(c[k], c[k + 1]) > 0.55, foot = P[I[side + 'Foot']];
    let L = F.locks[side];
    if (on && !L && foot.y < ANKLE + 0.04) L = F.locks[side] = { at: foot.clone().setY(Math.max(ANKLE, Math.min(foot.y, ANKLE + 0.02))), w: 0 };
    if (L && (!on || foot.distanceTo(L.at) > 0.3)) L.out = true;
    if (!L) continue;
    L.w = L.out ? L.w - 1 / 6 : Math.min(1, L.w + 1 / 4);
    if (L.w <= 0) { F.locks[side] = null; continue; }
    p ??= clonePose(F.shown);
    const goal = foot.clone().lerp(L.at, L.w);
    legTo(p, side, goal);
  }
  if (p) F.shown = rig.fixRigid(p);
}
function homing(F) {
  const A = F.attack, o = F.aim?.enemy ?? F.opponent;
  if (F.state !== 'attack' || !A?.key || A.landed || !o || F.track.tag.kind !== 'attack' || /ForeArm|Shin/.test(A.part)) return;
  const tip = rig.positions(F.shown)[I[A.part]];
  const k = A.key;
  const goal = F.aim && F.aim.part !== 'legs' ? o.P[F.aim.joint].clone().add(F.aim.off) : k.family === 'low' ? o.P[I.LeftShin].clone().lerp(o.P[I.RightShin], 0.5) : k.high ? o.P[I.Head].clone() : o.P[I.Spine2].clone();
  const w = clamp((0.7 - tip.distanceTo(goal)) / 0.45, 0, 1) * 0.7;
  if (w <= 0) return;
  F.shown = lerpPose(F.shown, reachLimb(F.shown, A.part, goal, 0.2), w);
}
function strike(F, kind = null) {
  if (!F.opponent || (F.state !== 'free' && F.state !== 'attack' && F.state !== 'guard')) return;
  // during a strike: kept, and the next one starts from where this one ends (a combination)
  if (F.state === 'attack' && F.attack && !F.attack.followed) { F.attackQueued = clock; F.queuedKind = kind; return; }
  F.attackQueued = null;
  if (!F.track.request(strikeReq(F, null, 'now', kind))) return; // (a hit or a fall on its way: not now)
  if (F.state !== 'attack') F.attack = null; // (an old strike's record no longer counts)
  // the load-up shows at once: turning away from the hand nearer the opponent (that one jabs, the other one winds)
  if (F.isPlayer && F.opponent) {
    const h = rig.heading(F.P), left = new Vector3(Math.cos(h), 0, -Math.sin(h)), to = flat((F.aim?.enemy ?? F.opponent).P[0].clone().sub(F.P[0]));
    F.antic = { t0: clock, side: Math.sign(to.dot(left) || 1) };
  }
  F.state = 'attack'; F.until = clock + 1.0;
}
// the right button pressed while the left one is held gives no pointerdown (a chorded press): take it from
// mousedown / contextmenu too (once per press)
canvas.addEventListener('contextmenu', e => e.preventDefault());
/**
 * What the cursor is on. The ray from the camera through the cursor against each opponent's head (sphere), trunk
 * and legs (capsules): the part it passes nearest (within a margin) is the aim, at the point the ray passes; with
 * none near, the floor point under the cursor at hip height is only where the body goes. The aim is kept as the
 * part and its offset from the part's joint, so the blow is placed on where that part will be.
 */
const aimRay = new THREE.Raycaster();
let aimOverride = null; // (tests: { enemy, part })
function aimFromCursor() {
  if (aimOverride) return aimOverride.enemy?.state !== 'out' ? aimOverride : null;
  if (!mouse.seen) return null;
  const r = canvas.getBoundingClientRect();
  aimRay.setFromCamera(new THREE.Vector2(((mouse.x - r.left) / r.width) * 2 - 1, -((mouse.y - r.top) / r.height) * 2 + 1), camera);
  const o = aimRay.ray.origin, far = o.clone().addScaledVector(aimRay.ray.direction, 60);
  let best = null;
  for (const E of fighters) {
    if (E.isPlayer || E.state === 'out') continue;
    const P = E.P;
    const parts = [
      ['head', P[I.Head], P[I.Head], 0.13, I.Head],
      ['body', P[0], P[I.Neck1], 0.2, I.Spine2],
      ['legs', P[I.LeftLeg], P[I.LeftFoot], 0.11, I.LeftShin],
      ['legs', P[I.RightLeg], P[I.RightFoot], 0.11, I.RightShin],
    ];
    for (const [part, a, b, rad, joint] of parts) {
      const c = segSeg(o, far, a, b);
      const score = c.d - rad;
      if (score < 0.25 && (!best || score < best.score)) best = { enemy: E, part, joint, point: c.b.clone(), off: c.b.clone().sub(P[joint]), score };
    }
  }
  return best;
}
function cursorFloor() { return mouse.seen ? floorPoint(mouse.x, mouse.y) : null; }
// the look line: faint, from the eyes to what is looked at (a dot where it meets an opponent); a ring at the aim
// when a button goes down (punch yellow, kick orange)
const lookGeo = new THREE.BufferGeometry().setFromPoints([new Vector3(), new Vector3(0, 0, 1)]);
const lookLine = new THREE.Line(lookGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, depthTest: false }));
const lookDot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8), new THREE.MeshBasicMaterial({ color: 0xfff2b0, transparent: true, opacity: 0.8, depthTest: false }));
const cueRing = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.1, 32), new THREE.MeshBasicMaterial({ color: 0xffe070, transparent: true, side: THREE.DoubleSide, depthTest: false }));
lookLine.renderOrder = lookDot.renderOrder = cueRing.renderOrder = 15;
scene.add(lookLine, lookDot, cueRing);
let cueT = -9;
function cueStrike(kind) { cueT = performance.now(); cueRing.material.color.set(kind === 'kick' ? 0xff9a3c : 0xffe070); }
function drawLook() {
  const P = player;
  if (!P) return;
  const eye = P.P[I.Head].clone().add(new Vector3(0, 0.04, 0));
  const aim = P.aim ? P.aim.enemy.P[P.aim.joint].clone().add(P.aim.off) : null;
  const to = aim ?? (P.lookAt ? P.lookAt.clone().setY(eye.y - 0.15) : null);
  lookLine.visible = !!to && !['down', 'fall', 'getup'].includes(P.state);
  if (to) {
    const dir = to.clone().sub(eye), len = dir.length();
    const end = aim ? to : eye.clone().addScaledVector(dir.normalize(), Math.min(len, 2.6));
    const a = lookGeo.attributes.position;
    a.setXYZ(0, eye.x, eye.y, eye.z); a.setXYZ(1, end.x, end.y, end.z); a.needsUpdate = true; lookGeo.computeBoundingSphere();
  }
  lookDot.visible = !!aim && lookLine.visible;
  if (aim) lookDot.position.copy(aim);
  const age = (performance.now() - cueT) / 1000;
  cueRing.visible = age < 0.35 && !!(aim ?? to);
  if (cueRing.visible) { cueRing.position.copy(aim ?? to); cueRing.lookAt(camera.position); cueRing.scale.setScalar(1 + (age < 0.15 ? (0.15 - age) * 6 : 0)); cueRing.material.opacity = Math.max(0, 1 - age / 0.35); }
}
/**
 * A body slam: the player's hips driven into an opponent's. Its energy is the body's (about 25 kg behind the
 * shoulder) at the speed they close (the shown speed, or the spring's when the shown motion lags behind it); the
 * one driving loses half its speed in it. A fast one knocks down or sends flying, a slow one only staggers.
 */
function bodySlam(A, B, dt) {
  if (B.state === 'fall' || B.state === 'down' || B.state === 'getup' || A.state === 'hit' || A.state === 'fall' || A.state === 'down' || A.state === 'getup') return;
  if (clock < (A.slamAfter ?? 0) || !A.prevP || !B.prevP) return;
  const d = flat(B.P[0].clone().sub(A.P[0])), len = d.length();
  if (len > 0.8 || len < 1e-3) return;
  const n = d.normalize();
  const vA = flat(A.P[0].clone().sub(A.prevP[0])).divideScalar(dt), vB = flat(B.P[0].clone().sub(B.prevP[0])).divideScalar(dt);
  const closing = Math.max(vA.dot(n), 0.8 * A.vel.dot(n)); // (the player's own drive: an opponent walking into it is no slam)
  if (closing < 1.6) return;
  A.slamAfter = clock + 0.6;
  const at = A.P[I.Chest].clone().lerp(B.P[I.Spine2], 0.5);
  land(A, B, 0.5 * 25 * closing * closing * (A.state === 'attack' ? 0.5 : 1), n, at, false);
  if (lastHit) { lastHit.slam = true; hudFight(); }
  A.vel.multiplyScalar(0.5);
}
/**
 * The body as a person's under a blow: an active ragdoll (src/ragdoll.js) that follows the generated motion with its
 * muscles. A blow loosens the muscles (more for a harder one) and drives the struck parts; the muscles then firm up
 * again over a moment (longer after a harder blow), so the body sways, catches itself and comes back to the motion
 * - or, knocked down, stays loose until it lies still, its limbs falling as they will. What is shown is the
 * ragdoll, blended into the motion as it settles.
 */
function jolt(F, at, J, loose, recover, feetFree = false) {
  const R = F.rag;
  if (!F.ragOn) { R.reset(F.shown); R.last = null; }
  F.ragOn = true; F.ragW = 1; F.ragFade = false; F.ragT = 0; F.ragRecover = recover; F.ragLoose = loose; F.ragFeetFree = feetFree;
  R.k = F.ragK.map(k => k * loose);
  R.rootK = feetFree ? 2000 : 600;
  const order = R.x.map((x, i) => [x.distanceTo(at), i]).sort((a, b) => a[0] - b[0]).slice(0, 5);
  const wsum = order.reduce((a, [d]) => a + 1 / (d + 0.05), 0);
  for (const [d, i] of order) R.v[i].addScaledVector(J, (1 / (d + 0.05)) / wsum / R.mass[i]);
}
function ragStep(F, dt) {
  if (!F.ragOn) return;
  const R = F.rag;
  F.ragT += dt;
  const down = F.state === 'fall';
  // muscles firm up again (not while going down: the body stays loose until it lies)
  const u = down ? 0 : clamp(F.ragT / F.ragRecover, 0, 1);
  const loose = F.ragLoose + (1 - F.ragLoose) * u * u;
  R.k = F.ragK.map(k => k * (down ? Math.max(F.ragLoose, 0.25) : loose));
  R.rootK = down ? 2500 : 600 + 2400 * u;
  const contact = F.ragFeetFree || down ? [0, 0, 0, 0] : null;
  // (the ragdoll keeps its own state: a reset, should it drift far from the motion, is not shown as a jump)
  const before = R.x[0].clone();
  R.step(dt, F.shown, contact, null);
  const ragged = R.pose(F.shown);
  const drift = ragged.root.distanceTo(F.shown.root);
  if (R.x[0].distanceTo(before) > 0.3 || drift > 0.45) F.ragFade = true;
  // back to the motion once firm again; on the floor (lying, stirring) the motion's own; drifting off: at once
  if (F.ragFade) F.ragW = Math.max(0, F.ragW - dt / 0.12);
  else if (F.state === 'down' || F.state === 'getup') F.ragW = Math.max(0, F.ragW - dt / 0.4);
  else if (!down && u >= 1) F.ragW = Math.max(0, F.ragW - dt / 0.2);
  if (F.ragW <= 0) { F.ragOn = false; F.ragFade = false; return; }
  F.shown = lerpPose(F.shown, ragged, F.ragW);
}
/** A right press, taken within 0.3 s when the body can strike (after a hit, getting up: dropped). */
function takeStrikeInput() {
  const P = player, inp = P?.strikeInput;
  if (!inp) return;
  const t = typeof inp === 'number' ? inp : inp.t, kind = typeof inp === 'number' ? null : inp.kind;
  if (clock - t > 0.3 || P.state === 'down' || P.state === 'fall' || P.state === 'getup') { if (clock - t > 0.3) P.strikeInput = null; return; }
  if (P.state === 'hit') return;
  P.strikeInput = null;
  strike(P, kind);
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), clock = 0, fpsAcc = 0, fpsN = 0, fps = 0;
const camTarget = new Vector3(0, 0.9, 0), camSide = new Vector3(1, 0, 0);
function simulate(dt) {
  if (!db) return;
  if (settings.trigger !== 'auto') takeStrikeInput();
  if (hitstop > 0) { hitstop -= dt; scheduler.tick(fighters.filter(f => f.state !== 'out').map(f => f.track)); return; } // a landed blow holds the picture a moment (the model keeps working)
  if (player && fighters.filter(f => !f.isPlayer && f.state !== 'out').length < settings.enemies && clock > spawnAt) { spawn(); spawnAt = clock + 1.5; }
  // the player's leash on the hips
  if (player) {
    const free = player.state === 'free' || player.state === 'attack';
    player.aim = aimFromCursor();
    let goal = scripted ? scripted.point : cursorFloor();
    const hips = flat(player.P[0]);
    if (goal) {
      goal = flat(goal);
      // over an opponent: stop at striking distance, unless the cursor runs on past it (a charge: a body slam)
      const A = player.aim;
      if (A && !scripted) {
        const e = flat(A.enemy.P[0]), to = e.clone().sub(hips), d = to.length();
        if (goal.distanceTo(e) < 0.9 && d > 0.3) goal = e.clone().addScaledVector(to.normalize(), -0.85);
      }
      if (!A && goal.distanceTo(hips) > 0.35) player.lookAt = goal.clone().setY(1.35); // (at eye height ahead)
      if (goal.distanceTo(hips) < 0.2) goal = null; // close enough: stand
    }
    if (player.aim) player.lookAt = player.aim.enemy.P[player.aim.joint].clone().add(player.aim.off); // on what the cursor is on
    player.spring = { target: goal && free ? goal.clone() : null };
    springStep(player.vel, flat(player.P[0]), player.spring.target, dt);
    if (!free) player.vel.multiplyScalar(Math.exp(-dt * 4));
    // striking: the spring's speed follows what the body really does (no speed stored up for a lurch afterwards)
    if (player.state === 'attack' && player.prevP) player.vel.lerp(flat(player.P[0].clone().sub(player.prevP[0])).divideScalar(dt), 1 - Math.exp(-dt * 6));
    player.want.copy(player.vel);
    if (settings.trigger === 'auto') { flickStrike(); autoStrike(player); }
  }
  for (const F of fighters) {
    if (F.isPlayer && F.state === 'free' && clock - (F.hitAt ?? -9) > 3 && F.hp < F.maxHp) { F.hp = Math.min(F.maxHp, F.hp + settings.regen * dt); if (Math.random() < 0.05) hudFight(); }
    if (!F.isPlayer) {
      think(F, dt); approach(F.vel, F.want, dt);
      // only the one whose turn it is strikes (the others only when the player comes right up to them)
      if (F === crew.attacker || flat(player.P[0].clone().sub(F.P[0])).length() < 1.1) autoStrike(F);
    }
    // states end
    if (F.state !== 'free' && F.state !== 'out' && clock > F.until && !F.track.busy) {
      if (F.state === 'down') {
        if (F.ko && !F.isPlayer) { F.state = 'out'; F.outAt = clock; kos++; spawnAt = clock + 2.5; hudFight(); }
        else { F.track.request(getupReq(F)); F.state = 'getup'; F.until = clock + 5; if (F.isPlayer && F.hp <= 0) F.hp = F.maxHp; }
      } else if (F.state === 'attack' && F.attackQueued != null && clock - F.attackQueued < 0.8) { F.attackQueued = null; F.state = 'free'; strike(F, F.queuedKind); }
      else if (F.state !== 'down') { F.state = 'free'; F.guard = null; }
    }
    // a strike carries straight on into its follow-through (asked for as the top comes near)
    const paceNow = F.track.frames[Math.min(F.track.frames.length - 1, Math.floor(F.track.t))].rate;
    if (F.state === 'attack' && F.attack?.spliced && !F.attack.followed && !F.track.busy && F.track.remaining / FPS / paceNow < scheduler.latency + 0.12) {
      F.attack.followed = true;
      if (F.attackQueued != null && clock - F.attackQueued < 0.8) { F.attackQueued = null; F.attack.chained = true; F.track.request(strikeReq(F, null, 'end', F.queuedKind)); F.until = clock + 1.2; }
      else F.track.request(followReq(F));
    }
    // plans while free
    F.planTimer -= dt;
    // the player is not held by its own follow-through or a light blow: pulled, it moves on (close in, nothing
    // should feel stuck)
    if (F.isPlayer && F.spring?.target && ((F.state === 'attack' && F.attack?.followSpliced && clock - F.attack.followSpliced > 0.08) || (F.state === 'hit' && clock - (F.hitAt ?? 0) > 0.3))) {
      F.state = 'free'; F.planTimer = 0;
    }
    if (F.state === 'free' && scheduler.ready && !F.track.busy) {
      const active = F.want.length() > 0.05 || F.vel.length() > 0.05;
      // (the player's speed changes all the time on its spring: only grabbing and letting go count as a change)
      const pulling = !!F.spring?.target;
      const changed = F.isPlayer ? pulling !== !!F.wasPulling : F.want.distanceTo(F.lastWant) > 0.4;
      F.wasPulling = pulling;
      if (active && (F.planTimer <= 0 || changed)) { F.track.request(planReq(F, 'move')); F.planTimer = settings.replan; F.lastWant.copy(F.want); }
      else if (!active && F.track.remaining / FPS < scheduler.latency + 0.25) F.track.request(planReq(F, 'move', 'end'));
    }
    F.track.advance(dt);
  }
  scheduler.tick(fighters.filter(f => f.state !== 'out').map(f => f.track));
  for (const F of fighters) {
    F.prevP = F.P;
    F.shown = clonePose(F.track.display);
    ragStep(F, dt);
    footLock(F);
    if (F.isPlayer) { anticipate(F); homing(F); }
    F.P = rig.positions(F.shown);
    if (F.P[0].distanceTo(F.prevP[0]) > 0.3) jumps.push({ t: +clock.toFixed(2), who: F.isPlayer ? 'P' : 'E', state: F.state, d: +F.P[0].distanceTo(F.prevP[0]).toFixed(2), match: F.lastMatch?.name, kind: F.lastMatch?.kind, splices: F.track.splices, frames: F.track.frames.length, t2: +F.track.t.toFixed(1) });
  }
  // bodies do not pass through each other: hips closer than 0.7 m are pushed apart (the motion is shifted)
  const live = fighters.filter(f => f.state !== 'out');
  // the player driving its body into an opponent: a body slam
  if (player) for (const B of live) if (!B.isPlayer) bodySlam(player, B, dt);
  for (let a = 0; a < live.length; a++) for (let b = a + 1; b < live.length; b++) {
    const A = live[a], B = live[b];
    const d = flat(B.P[0].clone().sub(A.P[0])), len = d.length();
    // the trunks (hips to neck, 17 cm round) do not pass through each other; lying bodies are stepped over
    const c = segSeg(A.P[0], A.P[I.Neck1], B.P[0], B.P[I.Neck1]);
    const sep = flat(c.b.clone().sub(c.a)), gap = 0.36 - c.d;
    if (gap > 0 && len > 1e-4) {
      const push = (sep.lengthSq() > 1e-6 ? sep.normalize() : d.normalize()).multiplyScalar(Math.min(0.05, gap * 0.5));
      A.track.shift(-push.x, -push.z); B.track.shift(push.x, push.z);
      for (const [F, nn] of [[A, push.clone().negate().normalize()], [B, push.clone().normalize()]]) if (F.isPlayer) { const into = -F.vel.dot(nn); if (into > 0) F.vel.addScaledVector(nn, into); }
      // (the shown positions move with it, so the push is not taken for a speed)
      for (const v of A.P) { v.x -= push.x; v.z -= push.z; }
      for (const v of B.P) { v.x += push.x; v.z += push.z; }
    }
  }
  // hits both ways
  for (const A of live) for (const B of live) if (A !== B && A.isPlayer !== B.isPlayer) checkHits(A, B, dt);
  // knocked-out opponents go away after a while
  for (let k = fighters.length - 1; k >= 0; k--) { const F = fighters[k]; if (F.state === 'out' && clock - F.outAt > 2) { F.remove(); fighters.splice(k, 1); } }
  clock += dt;
  window.brawl?.onStep?.(dt);
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000) * (now < slowUntil ? 0.35 : 1);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  for (const F of fighters) F.body.update(F.P, rig.extraPositions(F.shown, F.P), dt);
  drawLook();
  for (let k = sparks.length - 1; k >= 0; k--) { const s = sparks[k]; s.t += dt; s.m.scale.setScalar(s.s * (1 + 4 * s.t)); s.m.material.opacity = Math.max(0, 0.9 - 3 * s.t); if (s.t > 0.3) { scene.remove(s.m); sparks.splice(k, 1); } }
  // camera: from the side of the line between the player and the nearest opponent, far enough to see both
  if (player) {
    const o = player.opponent;
    const mid = o ? player.P[0].clone().lerp(o.P[0], 0.5) : player.P[0].clone();
    const line = o ? flat(o.P[0].clone().sub(player.P[0])) : new Vector3(0, 0, 1);
    const side = line.lengthSq() > 1e-4 ? new Vector3(line.z, 0, -line.x).normalize() : camSide.clone();
    if (side.dot(camSide) < 0) side.negate(); // keep the camera on its side
    camSide.lerp(side, 1 - Math.exp(-dt * 1.2)).normalize();
    const dist = clamp(4.2 + 0.8 * line.length(), 4.5, 9);
    camTarget.lerp(new Vector3(mid.x, 0.95, mid.z), 1 - Math.exp(-dt * 3));
    camera.position.copy(camTarget).addScaledVector(camSide, dist).add(new Vector3(0, 1.2, 0));
    camera.lookAt(camTarget);
    if (shake > 0.001) { camera.position.add(new Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).multiplyScalar(shake)); shake *= Math.exp(-dt * 12); }
    stage.followFloor(player.P[0]);
  }
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hudStats(); }
}
function loop(now) { frame(now); requestAnimationFrame(loop); }

function hudStats() {
  if (!scheduler.ready) return;
  $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms · 描画 ${fps.toFixed(0)} fps${player?.lastMatch ? ` · ${player.lastMatch.name.slice(0, 28)}` : ''}`;
  hudFight();
}
function hudFight() {
  if (!player) return;
  $('hp-player').style.width = `${(player.hp / player.maxHp) * 100}%`;
  const o = player.opponent;
  $('hp-enemy').style.width = o ? `${(o.hp / o.maxHp) * 100}%` : '0%';
  $('kos').textContent = `${kos}`;
  if (lastHit && clock - lastHit.t < 1.5) $('last-hit').textContent = lastHit.blocked ? `${lastHit.by}の一撃はガードされた（${lastHit.energy.toFixed(0)} J）` : lastHit.touch ? `${lastHit.by}の手足がかすった` : lastHit.slam ? `${lastHit.by}の体当たり ${lastHit.energy.toFixed(0)} J` : `${lastHit.by}の${lastHit.isHead ? '顔' : '体'}への一撃 ${lastHit.energy.toFixed(0)} J`;
}

// ---------------------------------------------------------------- model loading

function warmup(id) {
  const t = new Track(rig, { stance: rig.restPose() }); t.reset(new Vector3(), 0);
  const batch = emptyBatch(BATCH);
  const ctx = t.context(t.end);
  for (let b = 0; b < BATCH; b++) encodeRequest(rig, batch, b, ctx, [ctx[3]], [8, 9, 10]);
  const { n, ...inputs } = batch;
  worker.postMessage({ type: 'run', id, n, inputs }, Object.values(inputs).map(x => x.buffer));
}
worker.onmessage = ({ data }) => {
  const bar = $('progress-bar'), label = $('progress-label');
  if (data.type === 'progress') {
    if (data.phase === 'download') {
      const mb = x => (x / 1048576).toFixed(0);
      bar.style.width = data.total ? `${(data.loaded / data.total) * 100}%` : '50%';
      label.textContent = `モデルをダウンロード中… ${mb(data.loaded)}${data.total ? ` / ${mb(data.total)}` : ''} MB`;
    } else if (data.phase === 'cache') { bar.style.width = '100%'; label.textContent = '保存したモデルを読み込み中…'; }
    else { bar.style.width = '100%'; label.textContent = '推論の準備中（初回はシェーダーのコンパイルで数十秒かかることがあります）…'; }
  } else if (data.type === 'ready') {
    scheduler.backend = data.backend;
    label.textContent = `${data.backend} で準備完了。最初の生成を待っています…`;
    warmup(-1);
  } else if (data.type === 'result') {
    if (data.id === -1) warmup(0);
    else if (data.id === 0) { scheduler.latency = Math.max(0.05, data.ms / 1000); scheduler.ready = true; $('overlay').classList.add('hidden'); }
    else scheduler.onResult(data);
  } else if (data.type === 'error') {
    if (data.id) scheduler.onFail(data.id);
    label.textContent = `エラー: ${data.message}`;
    $('start').disabled = false;
  }
};
$('start').addEventListener('click', () => {
  const choice = document.querySelector('input[name=model]:checked').value;
  const url = new URL(params.get('model') ?? MODELS[choice], location.href).href;
  $('start').disabled = true;
  $('progress').classList.add('show');
  worker.postMessage({ type: 'load', url, ep: params.get('ep') ?? 'webgpu', batch: BATCH });
});
// the "how many opponents" links keep the other settings of the address (the model, the trigger)
for (const el of document.querySelectorAll('a[href^="?enemies="]')) {
  const u = new URLSearchParams(location.search);
  u.set('enemies', new URLSearchParams(el.getAttribute('href').slice(1)).get('enemies'));
  el.href = `?${u}`;
  if (Number(u.get('enemies')) === settings.enemies) el.style.fontWeight = 'bold';
}
for (const el of document.querySelectorAll('input[name=trigger]')) {
  if (el.value === settings.trigger) el.checked = true;
  el.addEventListener('change', () => { settings.trigger = el.value; });
}

requestAnimationFrame(loop);
if (params.has('debug')) {
  window.brawl = {
    rig, scheduler, settings, fighters, simulate, strike, land, crew, camera, keys: { fallKeys, getupKeys, guardKeys },
    get player() { return player; }, jumps, get db() { return db; }, get clock() { return clock; }, get kos() { return kos; }, get lastHit() { return lastHit; },
    pull(point) { scripted = { point: point ?? null }; },
    unscript() { scripted = null; },
    aim(a) { aimOverride = a; },
    draw: () => frame(performance.now()),
    onStep: null,
  };
}
