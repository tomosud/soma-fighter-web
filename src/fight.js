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
import { createPlayLog } from './playlog.js';
import { tempoRates, holdFrames } from './tempo.js';
// Tempo (緩急) measured from the Mixamo clips (tools/mixamo_tempo.mjs): how fast the striking limb goes at each moment
// of a strike, a hit reaction, a recovery. SOMA makes the poses and the motion between them; the time is ours.
const TEMPO = await fetch(new URL('../assets/tempo_profiles.json', import.meta.url)).then(r => (r.ok ? r.json() : null)).catch(() => null);
const TEMPO_OF = { straight: 'straight', hook: 'hook', uppercut: 'uppercut', elbow: 'elbow', knee: 'knee', front: 'kick_front', round: 'kick_round', low: 'kick_low' };
const tempoProfile = fam => (TEMPO && !params.has('notempo') ? TEMPO.families[TEMPO_OF[fam] ?? fam] ?? null : null);
let plog = null; // the play log (src/playlog.js)
let started = false; // (nothing is simulated, generated or logged until the start button is pressed)

const MB_REV = '54fc563aeaaf819a753d72cc9fdc6713989cdd43';
const MODELS = {
  fp16: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx`,
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
  damage: { player: 0.45, enemy: 0.35 }, // hp per joule (by the player, by an opponent)
  hipsMass: 12, // the part of the body's weight a moving body puts behind a blow (kg)
  regen: 3, // the player's hp back per second after 3 s without being hit
  knockdown: 95, // J in one blow that puts an opponent on the floor (the player: 1.4 of it, which only a blow from behind reaches); below it the body is knocked but stays up, and the fight stays on its feet
};

// ---------------------------------------------------------------- the reference database

const TAGS = ['move', 'stance', 'attack', 'block', 'hit', 'fall', 'getup'];
const T = Object.fromEntries(TAGS.map((t, k) => [t, k]));
let db = null, entTag = null, change = null, sunk = null, strikes = [], entryAt = new Map(), standPose = null;
PoseDB.load(rig, new URL('../assets/fightdb', import.meta.url).href).then(d => {
  db = d;
  d.search(new Float32Array(64), new Float32Array(64)); // (the search's float features and boxes built now, while loading - not in the first fight frame)
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
        const arc = rig.positions(chamber[1].pose)[I[part]].distanceTo(tip) + 0.5 * Math.max(0, tip.y - 1.0);
        strikeKeys.push({ clip: k, name: c.name, part, kind, family: familyOf(c.name, kind), frames, tip, high: tip.y > 1.25, miss, hit, chamber, travel, size: clamp((arc - 0.35) / 0.6, 0, 1) });
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
    this.id = Fighter.seq = (Fighter.seq ?? 0) + 1;
    this.track = new Track(rig, { stance: standPose ?? rig.restPose() });
    this.track.priority = isPlayer ? 0.3 : 0;
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
function springStep(v, pos, target, dt, feel = null) {
  const sp = settings.spring, k = sp.k * (feel?.k ?? 1), c = sp.c * (feel?.c ?? 1), vmax = settings.vmax * (feel?.vmax ?? 1);
  const a = target ? flat(target.clone().sub(pos)).multiplyScalar(k).addScaledVector(v, -c) : v.clone().multiplyScalar(-sp.coast);
  v.addScaledVector(a, dt);
  v.y = 0;
  if (v.length() > vmax) v.setLength(vmax);
  return v;
}
/** How a fighter's spring feels now: reeling from a blow sluggish (it still steers), stepping out of a blow crisp. */
/** A dash: a burst toward the point clicked (empty space), flowing into normal movement. */
function startDash() {
  const P = player, g = cursorFloor();
  if (!P || !g || clock < (P.dashAfter ?? 0) || !['free', 'attack'].includes(P.state)) return false;
  const dir = flat(g.clone().sub(P.P[0]));
  if (dir.length() < 0.4) return false;
  dir.normalize();
  P.dash = { t0: clock, until: clock + 0.25, dir };
  postureHit(P, 6, 'dash');
  P.vel.addScaledVector(dir, 2.8);
  P.dashAfter = clock + 0.45;
  if (P.state === 'attack' && P.attack?.followSpliced) P.state = 'free'; // (out of a follow-through at once)
  P.planTimer = 0;
  dust(P.P[0]);
  plog?.note('dash', {});
  return true;
}
function feelOf(F) {
  // (the player striking: the feet take the weight - the body brakes hard and does not drift through the blow)
  if (F.isPlayer && F.state === 'attack' && !(F.attack?.followed)) return { k: 0.6, c: 2.6, vmax: 1.0 };
  if (F.dash && clock < F.dash.until) return { k: 1.6, c: 1.2, vmax: 1.45 };
  if (F.charge && F.attack?.charged) return { k: 1.8, c: 1.4, vmax: 1.3 };
  if (F.dodge && clock < F.dodge.until) return { k: 2.4, c: 2.3, vmax: 1.3 };
  if (F.stun && clock < F.stun) return { k: 0.8, c: 2, vmax: 1.4 };
  return null;
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
    // (the dogfight: the player turns quicker than an opponent, which can be got round by a player circling at
    // ~2 m/s; an opponent off balance, or just out of a blow, hardly turns)
    const turn = F.isPlayer ? 4.5 : F.broken && clock < F.broken ? 1.0 : clock < (F.exposed ?? 0) ? 1.1 : F.state === 'free' ? 2.2 : 1.6;
    yaw += clamp(wrap(face - yaw), -turn / FPS, turn / FPS);
    if (TRAJ.includes(k)) out.push({ pos: p.clone(), yaw });
  }
  return out;
}

const qbuf = new Float32Array(DIM);
const ALLOW = { move: [T.move, T.stance, T.block], hit: [T.hit], fall: [T.fall], getup: [T.getup] };
// (the entries each kind of plan may use, worked out once: the search's bias is a typed array filled in a tight loop,
// not a closure called 37 000 times - that alone took 5 ms a plan, and four plans can fall in one frame)
const allowMask = {};
function maskOf(kind) {
  if (allowMask[kind]?.length === db.n) return allowMask[kind];
  const m = new Uint8Array(db.n), al = ALLOW[kind === 'attack' ? 'move' : kind];
  for (let i = 0; i < db.n; i++) m[i] = al.includes(entTag[i]) && !(kind === 'move' && sunk[i]) ? 1 : 0;
  return (allowMask[kind] = m);
}
let biasBuf = null, recentClip = null;
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
        if (biasBuf?.length !== db.n) biasBuf = new Float32Array(db.n);
        if (recentClip?.length !== db.clips.length) recentClip = new Uint8Array(db.clips.length);
        recentClip.fill(0); for (const c of recentSet) recentClip[c] = 1;
        {
          const mask = maskOf(kind), eC = db.entClip, eF = db.entFrame, cc = cont ? F.current.clip : -1, cf = cont ? cont.frame : 0, stance = T.stance;
          for (let i = 0; i < db.n; i++) {
            if (!mask[i]) { biasBuf[i] = Infinity; continue; }
            let b = calm * change[i];
            if (movingOn && entTag[i] === stance) b += 1.5; // pulled along: not a clip that stays put
            const c = eC[i];
            if (c === cc) { if (eF[i] < cf && eF[i] > cf - 40) b += 4; }
            else if (recentClip[c]) b += 2;
            biasBuf[i] = b;
          }
        }
        // (the clip playing on wins unless something is clearly better: that is the search's bound)
        const cc = cont ? costOf(cont.i, q, w) + calm * change[cont.i] : Infinity, stayK = movingOn ? 1 : settings.stay;
        // (an opponent carrying on the way it was going, its clip still fitting: no search this time - at most every
        // 0.5 s; the player always searches)
        const carryOn = cont && kind === 'move' && mode === 'now' && !F.isPlayer && F.searchedAt != null && clock - F.searchedAt < 0.5 && F.want.distanceTo(F.searchWant) < 0.3;
        if (carryOn) best = { i: -1 };
        else { best = db.search(q, w, biasBuf, cont ? stayK * cc : Infinity); F.searchedAt = clock; F.searchWant = F.want.clone(); }
        if (cont && best.i < 0) best = { i: cont.i, cost: cc, clip: F.current.clip, frame: cont.frame, mirrored: F.current.mirror, stayed: true };
        if (best.i < 0) best = db.search(q, w, Float32Array.from(maskOf('move'), v => (v ? 0 : Infinity))); // nothing of that kind
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
      // (the player setting off - pulled on from slow - takes its first steps quicker: the move feels instant)
      const setOff = kind === 'move' && F.isPlayer && !!F.spring?.target && F.vel.length() < 1.4 && flat(F.spring.target.clone().sub(Pn[0])).length() > 0.5;
      return {
        targets: out, tokens: kind === 'attack' ? [t0] : [6, 7], rate, fixEnd: kind === 'move' ? (moving ? 'root' : true) : true,
        tag: { kind: kind === 'move' ? (moving ? 'move' : 'idle') : kind === 'attack' ? 'attack' : 'hit' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          if (setOff) frames.forEach((f, i) => { if (i < 12) f.rate = (f.rate ?? rate) * (1 + 0.6 * (1 - i / 12)); });
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
function warpStrike(F, frames, part, base, ready = 0, family = null, wound = false) {
  const j = I[part], n = frames.length;
  const pos = frames.map(f => rig.positions(f.pose)[j]);
  const prof = tempoProfile(family);
  if (prof && n > 4) {
    const charged = !F.isPlayer && F.attack?.charged;
    const out = {};
    const rates = tempoRates(frames, pos, prof, {
      apex: n - 1, phases: F.isPlayer || charged || wound || ready > 0.6 ? ['accel'] : ['ant', 'accel'],
      durationScale: F.isPlayer ? 0.65 : charged ? 0.8 : wound ? 0.6 : 0.85, peakSpeed: F.isPlayer ? 8.5 : 7.5, maxRate: 6, out,
    });
    rates.forEach((r, i) => { frames[i].rate = r; });
    // a kick, a knee: a beat at full extension (Mixamo holds 4 frames); a punch goes straight on
    const hold = holdFrames(prof);
    if (hold > 0) frames[n - 1].rate = Math.min(frames[n - 1].rate, 1 / (0.6 * hold)); // (a beat, not a stop)
    return frames.reduce((a, f) => a + 1 / (FPS * f.rate), 0);
  }
  const sp = pos.map((q, i) => (i ? q.distanceTo(pos[i - 1]) * FPS : 0));
  let ip = 1;
  for (let i = 1; i < n; i++) if (sp[i] >= sp[ip]) ip = i;
  const charged = !F.isPlayer && F.attack?.charged; // (a charge has shown its wind-up already: the lunge is quick)
  const slow = F.isPlayer ? 1.0 + 0.3 * ready : charged ? 0.9 : 0.5, ramp = F.isPlayer ? Math.round(8 - 4 * ready) : charged ? 4 : 6; // (wound up already: a short, quick wind-up)
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
function warpReact(frames, strength, energy = 30) {
  const prof = tempoProfile(energy < 25 ? 'hit_small' : energy < 60 ? 'hit_medium' : 'hit_large');
  if (prof && frames.length > 4) {
    const pos = frames.map(f => rig.positions(f.pose)[I.Head]);
    const rates = tempoRates(frames, pos, prof, { phases: ['accel', 'hold', 'recover'], maxRate: 5, minRate: 0.45 });
    rates.forEach((r, i) => { frames[i].rate = r; });
    return;
  }
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
function strikeReq(F, family = null, mode = 'now', kindSel = null, pressId = 0, opts = {}) {
  return {
    kind: 'attack', mode, interrupt: mode === 'now',
    // (wound up: the context the model starts from is the wound-up body, held - so the motion it makes swings out of it)
    context(tr, a) {
      const base = tr.context(a), A = F.antic;
      if (!A?.big || mode !== 'now' || A.until) return base;
      const w = windWeight(A, clock - A.t0 + Math.max(0, a - tr.now) / FPS);
      if (w < 0.2) return base;
      A.baked = true; A.a = a;
      const wound = windPose(base[3], A, w);
      return [windPose(base[0], A, 0.85 * w), windPose(base[1], A, 0.95 * w), wound, wound];
    },
    build(tr, a, ctx) {
      const from = ctx[3];
      // the body as it will be when the strike starts (the request's context: now + the wait, or the end of a
      // strike being chained from), and its hands' motion
      const feat = poseFeat(rig.positions(ctx[2]), rig.positions(from));
      const hNow = rig.heading(rig.positions(from));
      const aimed = F.aim ?? null;
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
      else if (opp) { // (an opponent throws at where the player is going: its spring, run on to the contact)
        const ahead = opp.isPlayer ? playerAhead(tLand).sub(flat(opp.P[0])).clampLength(0, 1.4) : flat(opp.vel).multiplyScalar(0.7 * tLand);
        OP = opp.P.map(v => v.clone().add(ahead));
      }
      const head = opp ? OP[I.Head].clone().add(new Vector3(0, -0.02, 0)) : A.clone().add(new Vector3(Math.sin(yaw), 1.5, Math.cos(yaw)));
      const body = opp ? OP[I.Spine2].clone() : head.clone().setY(1.1);
      const speedIn = opp ? F.vel.dot(flat(opp.P[0].clone().sub(A)).normalize()) : 0;
      // where this body will be: the player's hips run on their spring (it may swing back), others glide on
      if (F.isPlayer && F.spring) {
        const v = F.vel.clone(), p = A.clone();
        for (let t = 0; t < tLand; t += 1 / 30) { springStep(v, p, F.spring.target, 1 / 30); p.addScaledVector(v, 1 / 30); }
        A.add(p.sub(A).clampLength(0, 0.8)); // (planted: it steps in, it does not glide in)
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
      // (running in hard and a kick pressed: a jump kick - a front, side or round kick, the body off the floor)
      const jump = F.isPlayer && kindSel === 'kick' && !!F.runIn && clock - F.runIn.t < 0.6 && F.runIn.speed > 1.6;
      if (F.runIn) F.runIn = null;
      let pool = strikeKeys.filter(k => byKind(k) && (!family || k.family === family) && (!jump || (k.kind === 'kick' && k.family !== 'low')));
      if (!jump && pool.some(byPart)) pool = pool.filter(byPart);
      if (!jump && aimPart === 'legs' && pool.some(k => k.family === 'low')) pool = pool.filter(k => k.family === 'low'); // at the legs: a low kick, a sweep
      for (const k of (pool.length ? pool : strikeKeys)) {
        const goal = local(aimOf(k));
        const step = clamp(goal.z - k.tip.z + 0.08, -0.45, WIND[k.kind] + (jump ? 0.7 : F.isPlayer ? 0.25 : opts.lunge ? 0.7 : 0.45)); // (aim assist: it may step further in, the player most; a lunge far; a jump kick flies in)
        let cost = Math.abs(goal.z - k.tip.z - step) * 4 + Math.abs(goal.y - k.tip.y) * 2 + Math.abs(goal.x - k.tip.x) * 1.5;
        // (variety: the last four strikes thrown cost more the more recent, the same family too, and more left to chance)
        { const hi = (F.hist ?? []).indexOf(k.clip); if (hi >= 0) cost += [1.4, 1.0, 0.6, 0.35][hi]; }
        cost += (k.family === F.lastFamily ? 0.7 : 0) + (k.family === F.lastFamily2 ? 0.35 : 0) + Math.random() * 0.7;
        cost -= (F.isPlayer ? 1.3 : 1.4) * k.size; // (a wide swing reads; a short poke does not)
        if (F.isPlayer && k.kind === 'elbow' && kindSel === 'punch') cost += 0.6; // (a press for a punch: a punch, the short elbow only when nothing else fits)
        // wound up: the hand drawn back is the one that throws (a cross, a hook - not the lead hand's short jab)
        const cocked = F.antic?.big && F.antic.signs?.rear;
        if (cocked && (k.kind === 'punch' || k.kind === 'elbow') && !k.part.startsWith(cocked)) cost += 1.5;
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
      F.lastStrike2 = F.lastStrike; F.lastStrike = k.clip; F.lastFamily2 = F.lastFamily; F.lastFamily = k.family;
      (F.hist ??= []).unshift(k.clip); F.hist.length = Math.min(F.hist.length, 4);
      const fwd = new Vector3(Math.sin(yaw), 0, Math.cos(yaw));
      const at = A.clone().addScaledVector(fwd, step);
      const aim = aimOf(k).clone();
      const JH = 0.3; if (jump) aim.y -= 0.8 * JH; // (the body is lifted that much at the top: the foot is placed for it)
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
      const through = F.isPlayer ? [-0.26, -0.06, 0.14, 0.34] : [-0.22, -0.08, 0.06, 0.2]; // (the player's fist goes on well through)
      targets = targets.map(t => bigTop(t, k, fwd, F.isPlayer ? 1.3 : 1));
      if (!/ForeArm|Shin/.test(k.part)) targets = targets.map((t, i) => {
        const g = aim.clone().addScaledVector(fwd, through[i]), tip = rig.positions(t)[I[k.part]];
        const short = g.clone().sub(tip).dot(fwd); if (short < 0) g.addScaledVector(fwd, -short); // (out to full reach)
        return reachLimb(t, k.part, g, F.isPlayer ? 0.5 : 0.35);
      });
      // the one being struck sees it coming: a guard or a dodge, or it takes it
      if (opp) defend(opp, F, k);
      if (opp) targets = targets.map(t => { const r = clonePose(t); gaze(r, head); return rig.fixRigid(r); });
      // (it takes over when its motion is spliced in: until then the strike being played is the one that counts)
      const wound = !F.isPlayer && !!F.antic?.baked && F.antic.a === a; // (generated out of a held wind-up: it snaps out)
      const atk = { part: k.part, id: Math.random(), kind: k.kind, key: k, yaw, at, jump: jump ? JH : 0, spliceAt: 0, oppAt: opp ? flat(OP[0]) : null, followed: false, landed: false, sweep: swing && /hook|round/.test(k.family), t0: clock, pressId, aimPart, charged: !!opts.lunge };
      F.lastMatch = { name: k.name, kind: 'attack', cost: bc, frame: 0, dPose: +dPose.toFixed(2) };
      F.current = null;
      // already wound up that way: out sooner (less wind-up, a little faster)
      const ready = clamp(1 - dPose / 0.8, 0, 1);
      if (F.isPlayer && mode === 'now') F.spec = { pose: targets[0], t0: Math.min(clock, F.windAt ?? clock), atk }; // (shown from the press on: specStrike)
      const rate = RATE[k.kind] * (1 + 0.25 * ready) * (opts.lunge ? 1.25 : 1);
      atk.ready = ready;
      return {
        targets, tokens: [6], rate, fixEnd: true, tag: { kind: 'attack' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          if (!F.isPlayer && F.broken && clock < F.broken) return; // (off balance: no blow)
          F.attack = atk; atk.spliced = true;
          if (F.isPlayer) postureHit(F, 2, 'swing');
          const dur = warpStrike(F, frames, k.part, rate, ready, k.family, wound);
          atk.contactAt = clock + dur; atk.spliceAt = clock; // (when the limb reaches the top: the blow is under way until then)
          if (atk.jump) F.jump = { t0: clock, up: Math.max(0.18, dur), h: atk.jump, side: k.part.startsWith('Left') ? 'Right' : 'Left' };
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
      const rate = A.landed ? 1.5 : F.isPlayer ? 1.3 : A.charged ? 0.7 : 0.9; // (an opponent's miss carries it on, slowly: it is open; a missed charge most)
      return {
        targets: targets.map((t, k) => { if (k === 0) return t; const r = clonePose(t); r.free = true; return r; }),
        tokens: [6], rate, fixEnd: false, tag: { kind: 'move' },
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          const prof = tempoProfile(A.key.family);
          if (prof && gen.length > 4) {
            const pos = gen.map(f => rig.positions(f.pose)[I[A.key.part]]);
            const rates = tempoRates(gen, pos, prof, { apex: 0, phases: ['recover'], durationScale: A.landed ? 0.7 : F.isPlayer ? 0.8 : A.charged ? 1.4 : 1.0, maxRate: 4, minRate: 0.4 });
            rates.forEach((r, i) => { gen[i].rate = r; });
          } else gen.forEach((f, i) => { f.rate = 1.6 - 0.7 * (i / Math.max(1, gen.length - 1)); }); // carried through, easing out
          if (F.attack) F.attack.followSpliced = clock;
          const dur = gen.reduce((s2, f) => s2 + 1 / (FPS * f.rate), 0);
          F.until = clock + dur - 0.15;
          if (!F.isPlayer && !A.landed) { const open = A.charged ? 0.6 : 0.3; F.exposed = clock + dur + open; F.until += open; } // a whiff: open for a counter
          if (!A.landed && !A.dodgedBy) postureHit(F, F.isPlayer ? 4 : A.charged ? 12 : 8, 'whiff'); // (a dodged one has cost it already)
          F.charge = null;
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
    if (!recentlyHit && r < 0.08) kind = 'dodge';
    else if (!recentlyHit && r < 0.22) kind = k.family === 'low' || !k.high ? 'low' : 'high';
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
function fallPlan(F, ctx, energy, dir, a) {
  const A = flat(ctx[3].root);
  const blow = flat(dir).normalize();
  // the way the body has been carried since the blow (shown on top of the motion) is where the fall starts from
  const K = F.knockFx, carried = K && !K.baked && K.d > 0.01 ? K.dir.clone().multiplyScalar(K.d) : null;
  if (carried) A.add(carried);
  const bake = a0 => { if (carried && F.knockFx === K) K.baked = { o: carried, from: a0 }; };
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
  F.lastFall = { name: best.name, fly: !!fly, dist: +dist.toFixed(2), t: clock };
  const hold = 0.5 + clamp(energy / 200, 0, 0.8);
  const settle = rate => (tr2, a0, gen) => {
    groundFrames(gen);
    bake(a0);
    const left = (tr2.frames.length - 1 - tr2.t) / FPS / rate;
    const end = tr2.frames[tr2.frames.length - 1];
    // knocked out: it lies there; alive: it stirs at once (crawling away, or toward where the player points)
    const still = F.ko ? hold + 1 : 0;
    for (let k = 0; k < FPS * still; k++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
    F.state = 'down'; F.until = clock + left + still + 0.1;
    // (alive: an opponent stirs once, then gets up - a fight on its feet, not a floor to pound; the player gets
    // straight up: it is never left lying)
    if (!F.ko) { if (!F.isPlayer) tr2.request(stirReq(F, hold)); jolt(F, F.P[0].clone(), new Vector3(), 0.5, 0.5, true); }
  };
  if (!fly) return { targets: lying, tokens: [8, 9], rate: 1.6, fixEnd: true, tag: { kind: 'down' }, after: settle(1.6) };
  // in the air: the moment on the way down, half way along, lifted
  const lift = clamp((energy - settings.knockdown) / 220, 0.1, 0.45);
  const air = placeAt(best.mid, A.clone().addScaledVector(blow, dist * 0.5)).map(t => { const r = clonePose(t); r.root.y += lift; return r; });
  const landing = { kind: 'hit', mode: 'end', build: () => ({ targets: lying, tokens: [6, 7], rate: 1.5, fixEnd: true, tag: { kind: 'down' }, after: settle(1.5) }) };
  return {
    targets: air, tokens: [6], rate: 1.9, fixEnd: true, tag: { kind: 'down' },
    after: (tr2, a0, gen) => { groundFrames(gen, true); bake(a0); F.state = 'fall'; F.until = clock + 5; tr2.request(landing); },
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
      // which way it crawls: an opponent away from the player; the player toward the cursor
      let away = new Vector3();
      if (!F.isPlayer && player) away = flat(Pn[0].clone().sub(player.P[0])).normalize().multiplyScalar(0.35);
      else if (F.isPlayer) { const c = cursorFloor(); if (c && flat(c).distanceTo(flat(Pn[0])) > 0.5) away = flat(c.clone().sub(Pn[0])).normalize().multiplyScalar(0.4); }
      const targets = fr.map(f => ground(placePose(f, ang, ZERO, at.clone().add(away))));
      const rate = 0.85 + 0.3 * Math.random();
      F.lastMatch = { name: `${g.name}（床で）`, kind: 'down', cost: 0, frame: 0 };
      return {
        targets, tokens: [7, 8], rate, fixEnd: true, tag: { kind: 'down' },
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          F.until = clock + (tr2.frames.length - 1 - tr2.t) / FPS / rate;
          // a second stir (crawling on), then up (two stirs: down about two seconds)
          if (n < 1) tr2.request(stirReq(F, hold, n + 1));
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
      const rate = F.isPlayer ? 1.8 : 1.3;
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
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          if (!F.isPlayer && !F.failedGetup && (F.hp < 0.3 * F.maxHp || Math.random() < 0.25)) { // (the player always gets up first time)
            F.failedGetup = true; F.state = 'down'; F.until = clock + 3; tr2.request(stirReq(F, 1, 1)); return;
          }
          F.failedGetup = false; F.state = 'getup'; F.until = clock + 5; tr2.request(rise);
        },
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
      if (down && fallKeys.length) return fallPlan(F, ctx, energy, dir ?? away, a);
      const pool = down ? fallKeys : hitKeys.filter(h => h.where === (isHead ? 'head' : 'body') || h.where === 'any');
      const list = pool.length ? pool : hitKeys;
      // stronger blows: the bigger reactions
      const wantStrength = clamp(0.25 + energy / 120, 0.2, 1.2);
      let best = null, bc = Infinity;
      for (const h of list) { const c = Math.abs(h.strength - wantStrength) + Math.random() * 0.4 + (h.clip === F.lastReact ? 0.5 : 0); if (c < bc) { bc = c; best = h; } }
      F.lastReact = best?.clip;
      const back = down ? 0.35 + clamp(energy / 90, 0, 1.5) : clamp(0.1 + energy / 200, 0.1, 0.55); // a big blow sends the body flying
      const dest = A.clone().addScaledVector(away, back);
      // reactions face the attacker - but a body struck from the side or behind turns only part way round (it is
      // still caught side-on: the one who got round keeps the advantage)
      const h0 = rig.heading(rig.positions(ctx[3])), sq = (F.recentHits?.filter(t => clock - t < 1.2).length ?? 0) >= 2 ? 1.3 : 0.4, yaw = h0 + clamp(wrap(Math.atan2(-away.x, -away.z) - h0), -sq, sq); // (the second blow in a row: it squares up)
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
          if (!down) { warpReact(frames, clamp(energy / 45, 0, 1), energy); bakeKnock(F, frames, a0); }
          const left = down ? (tr2.frames.length - 1 - tr2.t) / FPS / rate : frames.reduce((s2, f) => s2 + 1 / (FPS * f.rate), 0);
          if (down) {
            const end = tr2.frames[tr2.frames.length - 1];
            for (let k = 0; k < FPS * 0.8; k++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
            F.state = 'down'; F.until = clock + left + 0.8;
          } else { F.state = 'hit'; F.until = clock + left + 0.15; tr2.request(recoverReq(F)); } // (an opponent is back in it soon)
        },
      };
    },
  };
}

// ---------------------------------------------------------------- hits

// striking points and the effective mass behind each (kg)
const STRIKE_MASS = { LeftHand: 1.4, RightHand: 1.4, LeftForeArm: 2.0, RightForeArm: 2.0, LeftFoot: 3.5, RightFoot: 3.5, LeftShin: 4.5, RightShin: 4.5 };
const sparks = [];
const sparkGeo = new THREE.RingGeometry(0.7, 1, 28);
function spark(at, energy, color = 0xffe9a8) {
  // (a ring opening out from the contact, not a ball over it: what met what stays in view inside it)
  const m = new THREE.Mesh(sparkGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false, side: THREE.DoubleSide }));
  m.position.copy(at); m.renderOrder = 20; scene.add(m);
  sparks.push({ m, t: 0, s: 0.07 * (1 + Math.min(2.5, energy / 50)) });
}
/**
 * What struck what: the striking limb (fist and forearm, or foot and shin) and the part it met light up in the same
 * colour for a moment - the player's blows gold, an opponent's on the player red, a blow taken on a guard or turned
 * aside blue - so the impact reads even when the flash is over it.
 */
const LIMB = { Hand: ['Arm', 'ForeArm', 'Hand', 'HandMiddleEnd'], ForeArm: ['Arm', 'ForeArm', 'Hand'], Foot: ['Shin', 'Foot', 'ToeBase'], Shin: ['Leg', 'Shin', 'Foot'] };
const PART_BONES = { head: [['Neck1'], true], body: [['Hips', 'Spine2', 'Chest'], false], armL: [['LeftArm', 'LeftForeArm', 'LeftHand'], false], armR: [['RightArm', 'RightForeArm', 'RightHand'], false], legL: [['LeftShin', 'LeftFoot'], false], legR: [['RightShin', 'RightFoot'], false] };
function impactMark(A, B, at, kind, energy = 30, hitPart = null) {
  const color = kind === 'guard' ? [0.35, 0.65, 1.0] : A.isPlayer ? [1.0, 0.75, 0.2] : [1.0, 0.12, 0.08];
  const k = kind === 'touch' ? 0.45 : clamp(0.7 + energy / 150, 0.7, 1.2), c = color.map(x => x * k), dur = kind === 'touch' ? 0.25 : 0.45;
  const limb = A.strikePart ?? A.attack?.part;
  if (limb) { const side = limb.startsWith('Left') ? 'Left' : 'Right', seg = limb.replace(/^(Left|Right)/, ''); A.body.mark((LIMB[seg] ?? [seg]).map(n => side + n), false, c, dur); }
  let part = hitPart ?? partAt(B, at)[0];
  if (part === 'arm') part = B.P[I.LeftForeArm].distanceTo(at) < B.P[I.RightForeArm].distanceTo(at) ? 'armL' : 'armR';
  if (part === 'legs') part = B.P[I.LeftShin].distanceTo(at) < B.P[I.RightShin].distanceTo(at) ? 'legL' : 'legR';
  const [names, head] = PART_BONES[part] ?? PART_BONES.body;
  B.body.mark(names, head, c, dur);
}
/**
 * Contact of A's striking limbs with B's head (sphere) or trunk (capsule). While a limb is in contact its fastest
 * closing speed is kept; the blow lands when the limb starts to slow down or leaves (so a fist that brushes the
 * body on its way out is not counted at its slowest moment). Energy = 0.5 m v^2 with the limb's effective mass.
 */
function checkHits(A, B, dt) {
  // one contact per target (a swing can go through several bodies)
  A.contacts ??= new Map();
  if (A.state === 'out' || B.state === 'out' || B.state === 'fall') { A.contacts.delete(B); return; } // (not while flying)
  // a strike asked for but not yet under way (the motion still the one before): nothing it touches counts; an
  // opponent's blow does not land in its first 0.3 s (the wind-up is always there to see)
  if (A.state === 'attack' && !A.attack) { A.contacts.delete(B); return; }
  if (!A.isPlayer && A.attack && clock - A.attack.t0 < 0.22) { A.contacts.delete(B); return; }
  if (B.dodge && clock < B.dodge.iframes) {
    if (A.attack && !A.attack.dodgedBy) { A.attack.dodgedBy = B.id; postureHit(A, 10, 'dodged'); if (A.brain) A.brain.combo = 0; A.attackQueued = null; // (a dodged blow has no second half)
      plog?.note('dodged', { by: A.id }); if (B.isPlayer) { dust(B.P[0], 0x9fe0ff); slowUntil = performance.now() + 120; } }
    A.contacts.delete(B); return;
  }
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
    let closing = Math.min(11, v.clone().sub(vB).dot(n)); // (a fist or foot of a person: at most about 11 m/s)
    // an opponent's blow that reaches the player lands as the blow it is (its limb often arrives at the slow end of
    // the motion): touching it means being hit, stepping out of it means a miss
    // (B2: only while the blow is under way, not the follow-through brushing past)
    if (!A.isPlayer && A.attack && part === A.attack.part && !A.attack.followed && clock < (A.attack.contactAt ?? A.attack.t0 + 0.7) + 0.1) closing = Math.max(closing, { punch: 5.5, elbow: 5, knee: 4.5, kick: 5 }[A.attack.kind] ?? 5);
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
  A.strikePart = b.part; // (the limb that met the body: it lights up with the part it struck)
  land(A, B, e, b.n, b.at, b.isHead, b.onArm);
  A.strikePart = null;
}
// readable blows: a red edge on the screen (the player hurt), a camera kick, numbers where a blow lands
const hurtEl = document.createElement('div');
hurtEl.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:5;box-shadow:inset 0 0 160px 40px rgba(220,20,20,0.9);opacity:0;transition:none';
document.body.appendChild(hurtEl);
let hurt = 0;
const camKick = new Vector3();
/** A ring of dust spreading at the feet (a dodge, a dash): a shape, not words. */
const dusts = [];
function dust(at, color = 0xd0d4dc) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.28, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthTest: false }));
  m.rotation.x = -Math.PI / 2; m.position.set(at.x, 0.03, at.z); m.renderOrder = 13; scene.add(m);
  dusts.push({ m, t: 0 });
}
function damageNumber(at, text, color, size) {
  return; // (no text on screen: blows read by the body's motion, sparks and the camera)
  const r = canvas.getBoundingClientRect(), q = at.clone().project(camera);
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = `position:fixed;left:${r.left + (q.x + 1) / 2 * r.width}px;top:${r.top + (1 - q.y) / 2 * r.height}px;color:${color};font:bold ${size}px sans-serif;pointer-events:none;z-index:6;text-shadow:0 0 4px #000;transform:translate(-50%,-50%);transition:transform 0.8s ease-out, opacity 0.8s ease-in`;
  document.body.appendChild(el);
  requestAnimationFrame(() => { el.style.transform = 'translate(-50%,-150%)'; el.style.opacity = '0'; });
  setTimeout(() => el.remove(), 850);
}
/** A blow on a body lying (or getting up): it hurts less, jolts and pushes the body along the floor; a body getting up goes back down. */
function groundHit(A, B, energy, dir, at, isHead) {
  const dmg = energy * 0.25 * (A.isPlayer ? settings.damage.player : settings.damage.enemy); // (a body on the floor: kicked, but it is the standing fight that decides)
  pushPart(B, at, dir, energy);
  B.hp = Math.max(0, B.hp - dmg);
  spark(at, energy * 0.6);
  const n = flat(dir).normalize();
  // (a body on the floor kicked: its ragdoll takes the blow loose and free - it rolls and slides as far as the blow
  // sends it, its limbs flop, and the motion is moved along to where it ended up; see ragStep)
  B.groundJolt = clock;
  jolt(B, at, n.clone().addScaledVector(UP, 0.25).multiplyScalar(Math.sqrt(2 * energy * 6) * 1.15), 0.12, 1.0, true);
  // the player is never kept down by kicks: it hurts, the body is pushed, the getting-up goes on
  if (B.state === 'getup' && !B.isPlayer) { B.state = 'down'; B.until = clock + 2; B.getupKey = null; B.track.request(stirReq(B, 1, 0)); }
  else if (B.state === 'down' && !B.isPlayer) B.until += clamp(energy / 150, 0.1, 0.5);
  if (B.hp <= 0 && !B.isPlayer) B.ko = true;
  hitstop = Math.max(hitstop, 0.03); shake = Math.max(shake, 0.03);
  if (B.isPlayer) hurt = Math.max(hurt, 0.5);
  damageNumber(at, `-${Math.round(dmg)}`, B.isPlayer ? '#ff6060' : '#ffe9a8', 14 + Math.min(14, energy / 6));
  lastHit = { energy, dmg, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, tier: 'ground', victim: B.id, striker: A.id, pressId: A.attack?.pressId ?? 0 };
  if (A.attack) A.attack.landed = true;
  hudFight();
}
/**
 * The dogfight. Where a blow comes from, seen from the one struck: from the front (within 45 degrees of where it
 * faces), the flank (to 120) or the rear. Head-on against a fighter that is ready, a blow is turned aside (and it
 * costs the striker its balance); from the flank or the rear it lands full and harder, and meets vital points.
 */
function aspect(B, A) {
  const h = rig.heading(B.P), to = flat(A.P[0].clone().sub(B.P[0]));
  const ang = Math.abs(wrap(Math.atan2(to.x, to.z) - h));
  return ang < 0.79 ? 'front' : ang < 2.09 ? 'flank' : 'rear';
}
const ASPECT = { front: { dmg: 1.0, posture: 1.0 }, flank: { dmg: 1.2, posture: 1.4 }, rear: { dmg: 1.5, posture: 1.8 } };
// vital points by the part met and the side it is met from: [name, damage x, balance x]
function vitalOf(part, asp, openGuard) {
  if (part === 'head') return asp === 'flank' ? ['jaw', 1.5, 2.2] : asp === 'rear' ? ['occiput', 1.8, 3.0] : null;
  if (part === 'body') return asp === 'flank' ? ['liver', 1.5, 2.2] : asp === 'rear' ? ['kidney', 1.4, 2.0] : openGuard ? ['solar', 1.5, 2.5] : null;
  if (part.startsWith('leg')) return asp === 'rear' ? ['knee', 1.2, 3.0] : null;
  return null;
}
/** Balance: blows, guarded blows, whiffs and dodged blows wear it down; it comes back when left alone; at nothing it breaks. */
function postureHit(F, j, why) {
  if (F.broken && clock < F.broken) return;
  F.posture = (F.posture ?? 1) - j / (F.isPlayer ? 170 : 100);
  F.postureAt = clock;
  if (F.posture <= 0) breakPosture(F, why);
}
function breakPosture(F, why) {
  F.posture = 0; F.broken = clock + (F.isPlayer ? 1.2 : 1.7);
  plog?.note('postureBreak', { who: F.isPlayer ? 'P' : F.id, why });
  if (F.state === 'down' || F.state === 'fall' || F.state === 'getup') return;
  dust(F.P[0], 0xffc030);
  // the player stays in control (a heavy, slow body for a moment, open to harder blows); an opponent staggers
  if (F.isPlayer) { F.stun = Math.max(F.stun ?? 0, clock + 0.45); return; }
  F.attack = null; F.charge = null; F.attackQueued = null; F.guard = null;
  F.state = 'hit'; F.until = clock + 3;
  const from = player?.P[0] ?? F.P[0].clone().add(new Vector3(0, 0, 1));
  F.track.request(reactReq(F, from, 35, false, false, flat(F.P[0].clone().sub(from)).normalize())); // (a stagger: the strike it was in is called off)
}
function land(A, B, energy, dir, at, isHead, onArm = false) {
  if (B.state === 'down' || B.state === 'getup') { groundHit(A, B, energy, dir, at, isHead); return; }
  // ---- the dogfight
  // (the arm is what the contact test says it met - a body blow at an open level is not an arm because a forearm
  // happens to hang near it)
  const asp = aspect(B, A), part = onArm ? 'arm' : isHead ? 'head' : partAt(B, at, true)[0];
  const inWind = B.state === 'attack' && (B.attack ? clock - B.attack.t0 < 0.35 : clock - (B.windAt ?? -9) < 0.5) && !B.charge;
  const broken = B.broken && clock < B.broken;
  // a ready opponent covers one level at a time (its stance shows which): high covers the head, low the body and legs
  const covered = !B.isPlayer && (part === 'arm' || (B.stance === 'low' ? part !== 'head' : part === 'head'));
  const aware = asp === 'front' && (B.state === 'free' || B.state === 'hit' || (B.state === 'guard' && B.guard?.kind !== 'dodge')) && clock > (B.stun ?? 0) && (B.posture ?? 1) > 0.25 && !broken && covered;
  const vital = vitalOf(part, asp, inWind);
  let balance = ASPECT[asp].posture * (vital ? vital[2] : 1) * (inWind ? 2 : 1);
  const mult = (B.isPlayer ? { front: 0.7, flank: 1.25, rear: 1.8 }[asp] : ASPECT[asp].dmg) * (vital ? vital[1] : 1) * (broken && !B.isPlayer ? 2 : 1); // (a player off balance is slowed and open, not flattened)
  const e00 = energy; // (all the multipliers together are capped below)
  energy *= B.isPlayer ? 1 + (mult - 1) * 0.5 : mult; // (the player is spared half the extra: it is warned, not ambushed)
  // a fighter turning a blow aside answers harder for a moment
  let answer = false;
  if (A.counterUntil && clock < A.counterUntil) { energy *= 1.3; answer = true; A.counterUntil = 0; }
  // head-on against a fighter that is ready: turned aside - little gets through, the striker's balance suffers
  if (aware && energy >= 12) {
    const e0 = energy;
    energy *= B.isPlayer ? 0.5 : 0.3; // (the player half-turns a blow head-on: the opponents' blows still tell)
    jolt(B, at, flat(dir).normalize().multiplyScalar(Math.sqrt(2 * e0 * 6) * 0.4), 0.6, 0.25);
    pushPart(B, at, dir, Math.max(energy, 12));
    jolt(A, at, flat(dir).normalize().multiplyScalar(-Math.sqrt(2 * e0 * 6) * 0.3), 0.6, 0.3); // (the blow bounces off)
    shove(B, dir, clamp(0.06 + e0 / 500, 0.08, 0.2)); shove(A, dir.clone().negate(), clamp(0.04 + e0 / 700, 0.06, 0.15));
    if (A.attack) A.attack.landed = true;
    spark(at, e0 * 0.5, 0x9fd3ff);
    impactMark(A, B, at, 'guard', e0, part);
    shake = Math.max(shake, 0.03);
    B.hp = Math.max(0, B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy));
    postureHit(A, 6 + 0.05 * e0, 'guardedInto'); postureHit(B, 0.04 * e0, 'guard');
    B.counterUntil = clock + 0.45;
    lastHit = { energy, dmg: 0, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, blocked: true, tier: 'blocked', victim: B.id, striker: A.id, pressId: A.attack?.pressId ?? 0, aspect: asp, autoGuard: true };
    plog?.note('guardAuto', { by: B.isPlayer ? 'P' : B.id });
    hudFight();
    return;
  }
  const counter_ = !!(broken || answer);
  // a counter: an opponent caught in the recovery of a missed blow, or a blow right after the player stepped out of one
  let counter = counter_;
  if (!B.isPlayer && clock < (B.exposed ?? 0)) { energy *= 1.25; counter = true; B.exposed = 0; } // (one counter per whiff)
  else if (A.isPlayer && A.dodge?.flick && clock - A.dodge.at < 1.3) { energy *= 1.4; counter = true; }
  energy = Math.min(energy, e00 * 2.6);
  // head-on, no vital point, no counter: the player's blow does not floor anyone (only getting round, a vital point or
  // a counter does)
  if (A.isPlayer && asp === 'front' && !vital && !counter) energy = Math.min(energy, 45);
  // a player off balance takes more (it is open), but is not flattened for it
  if (B.isPlayer && broken) energy *= 1.4;
  if (B.isPlayer) energy = Math.min(energy, asp === 'rear' ? 140 : 100); // (no blow throws the player across the floor; only one from behind floors it)
  // a touch (a glancing contact, little in it): a small mark, a small sway of the body; no stop, no shake, no
  // reaction motion - the striker carries on through
  if (energy < 12 && B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy) > 0) {
    B.hp = Math.max(0, B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy));
    spark(at, energy * 0.3, 0xd8d8d8);
    impactMark(A, B, at, 'touch', energy, part);
    const J = flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6) * 0.6);
    jolt(B, at, J, 0.7, 0.25);
    shove(B, dir, 0.05 + energy * 0.004);
    lastHit = { energy, dmg: 0, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, touch: true, tier: 'touch', victim: B.id, striker: A.id, pressId: A.attack?.pressId ?? 0 };
    if (B.isPlayer) { hurt = Math.max(hurt, 0.2); shake = Math.max(shake, 0.02); }
    pushPart(B, at, dir, energy);
    hudFight();
    return;
  }
  // a guard up at the right height takes most of it; the strike is stopped (it comes back as a hit)
  if (B.guard && clock < B.guard.until && B.guard.kind !== 'dodge' && (B.guard.high === isHead || Math.random() < 0.4)) {
    energy *= 0.2;
    B.blocked = true;
    jolt(B, at, flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6)), 0.5, 0.3);
    shove(B, dir, clamp(0.1 + energy / 100, 0.1, 0.22));
    pushPart(B, at, dir, Math.max(energy, 12)); // (the guarding arm is knocked aside)
    if (A.attack) A.attack.landed = true;
    spark(at, energy, 0x9fd3ff);
    impactMark(A, B, at, 'guard', energy, part);
    shake = Math.max(shake, 0.02);
    B.hp = Math.max(0, B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy));
    lastHit = { energy, dmg: 0, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, blocked: true, tier: 'blocked', victim: B.id, striker: A.id, pressId: A.attack?.pressId ?? 0 };
    hudFight();
    return;
  }
  B.hitAt = clock;
  {
    const down0 = B.hp - energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy) <= 0 || energy > settings.knockdown * (B.isPlayer ? 1.4 : 1);
    const J = flat(dir).normalize().multiplyScalar(Math.sqrt(2 * energy * 6)); // the momentum the blow puts in (kg m/s)
    J.y = (isHead ? 0.25 : 0.1) * J.length();
    jolt(B, at, J, clamp(0.35 - energy / 300, down0 ? 0.04 : 0.08, 0.3), clamp(0.3 + energy / 120, 0.3, 0.9), down0);
    jolt(A, at, J.clone().multiplyScalar(-0.12), 0.75, 0.2);
  }
  const dmg = energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy);
  B.hp = Math.max(0, B.hp - dmg);
  spark(at, energy);
  impactMark(A, B, at, 'hit', energy, part);
  pushPart(B, at, dir, energy * (vital ? 1.5 : 1));
  postureHit(B, energy * (B.isPlayer ? 0.5 : 0.09) * balance, vital ? vital[0] : asp); // (an opponent's balance breaks after a few good blows, not every exchange)
  const push = flat(dir).normalize().multiplyScalar(clamp(0.35 * Math.sqrt(energy), 0.6, 2.6));
  B.knock.copy(push);
  const down = B.hp <= 0 || energy > settings.knockdown * (B.isPlayer ? 1.4 : 1) * (broken && !B.isPlayer ? 0.5 : 1);
  // an opponent already throwing its blow takes a light one through it (it sways, the blow goes on): a trade
  const charging = !B.isPlayer && B.charge && clock - B.charge.t0 < 1.4 && asp === 'front'; // (from the side or behind, a charge is no armour)
  const armored = !counter && !B.isPlayer && !down && ((charging && energy < 45) || (B.state === 'attack' && B.attack?.spliced && clock - B.attack.t0 > 0.3 && energy < 25) || (clock < (B.armorUntil ?? 0) && energy < 70));
  if (!armored) B.charge = null;
  if (!armored) {
    B.state = down ? 'fall' : 'hit';
    B.until = clock + 3;
    B.track.request(reactReq(B, A.P[0], energy, isHead, down, dir, at));
  }
  // an opponent that is hit but stays up answers now and then, quickly
  if (!B.isPlayer && !down && Math.random() < 0.7) B.counterAt = clock + rand(0.3, 0.55);
  // an opponent caught twice in quick succession covers up and backs off (mashing alone does not hold it)
  if (!B.isPlayer && !down && !armored) {
    B.recentHits = (B.recentHits ?? []).filter(t => clock - t < 1.2); B.recentHits.push(clock);
    // (half the time it covers up and backs off; half the time it bursts back through the player's blows)
    if (B.recentHits.length >= 2) { if (Math.random() < 0.5) B.coverAt = clock + 0.35; else B.burstAt = clock + 0.2; B.recentHits = []; }
  }
  // the body is carried off at once (until the generated reaction takes over)
  // (every damaging blow: even one taken through a blow of its own sways the body back a little)
  if (energy >= 12) B.knockFx = { t0: clock, dir: flat(dir).normalize(), v0: clamp(Math.sqrt(2 * energy / 70) * (down ? 1.3 : 0.7), 0.75, 3) * (armored ? 0.8 : 1), until: null, fade: 0.15 };
  if (A.attack) {
    if (A.attack.followed && !A.attack.landed && !A.attack.sweep && !A.attack.chained) A.track.request(followReq(A, 'now'));
    A.attack.landed = true;
  }
  B.body.flash = 0.3; // (the struck part itself lights up: impactMark)
  // the stop and the shake grow with the blow (a heavy one is felt, an ordinary one only marked)
  hitstop = down ? 0.12 : energy < 45 ? clamp(0.04 + energy / 1200, 0.04, 0.08) : clamp(0.08 + energy / 2000, 0.08, 0.12);
  shake = Math.max(shake, energy < 45 ? clamp(energy / 600, 0.02, 0.06) : clamp(energy / 500, 0.06, 0.14));
  // the player hurt: felt (a red edge, a stronger shake and stop, the camera knocked along the blow) and reeling a
  // moment (sluggish, presses held), so the blow reads
  if (B.isPlayer) {
    hurt = Math.max(hurt, clamp(energy / 60, 0.35, 1));
    shake = Math.max(shake, clamp(energy / 250, 0.05, 0.18));
    hitstop = Math.max(hitstop, clamp(0.06 + energy / 1300, 0.06, 0.12));
    camKick.addScaledVector(flat(dir).normalize(), clamp(energy / 400, 0.04, 0.14));
    B.stun = clock + clamp(0.2 + energy / 200, 0.2, 0.3); // (never more than 0.3 s out of the player's hands)
    if (asp !== 'front') { hurt = 1; shake = Math.max(shake, 0.2); camKick.addScaledVector(flat(dir).normalize(), 0.12); spark(at, energy * 1.5, 0xff2010); }
  }
  damageNumber(at, counter ? `カウンター -${Math.round(dmg)}` : `-${Math.round(dmg)}`, B.isPlayer ? '#ff5050' : counter ? '#ffd040' : '#ffffff', (B.isPlayer ? 20 : 15) + Math.min(16, energy / 5));
  if (counter) { spark(at, energy * 1.3, 0xffc030); spark(at, energy * 0.6, 0xffffff); hitstop = Math.max(hitstop, 0.1); slowUntil = Math.max(slowUntil, performance.now() + 250); camKick.addScaledVector(flat(dir).normalize(), 0.08); }
  if (down || energy > 60) slowUntil = performance.now() + clamp(energy * 4, 160, 420); // a heavy one: a moment of slow motion
  if (!B.isPlayer && down) crew.watchUntil = clock + 0.25; // the others hold back a moment (a beat, not a pause)
  if (!armored && (B.state === 'attack' || B.attack)) B.attack = null; // a hit cancels the one hit's own strike
  if (B.isPlayer && B.attackQueued != null) plog?.pressResult(B.queuedPress, 'lost:hit');
  if (B.isPlayer && B.strikeInput?.id) plog?.pressResult(B.strikeInput.id, 'lost:hit');
  if (!armored) B.attackQueued = null;
  if (B.hp <= 0 && !B.isPlayer) B.ko = true; // (the player is never left lying like a knocked-out body: it stirs and gets up, hp restored)
  lastHit = { energy, dmg, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock, tier: down ? (energy > settings.knockdown * 1.2 ? 'fly' : 'down') : energy >= 45 ? 'heavy' : 'hit', victim: B.id, striker: A.id, pressId: A.attack?.pressId ?? 0, counter, aspect: asp, vital: vital?.[0] ?? null };
  hudFight();
}
let lastHit = null, hitstop = 0, shake = 0, slowUntil = 0;
const jumps = [];

// ---------------------------------------------------------------- the player and the opponents

let player = null, kos = 0, spawnAt = 0;
// (the start: one opponent alone - a tutorial in all but name; knocked out, three come at once)
let tutorial = true, arrivals = 0;
function start() {
  player = new Fighter(true, 0xd9b26a, new Vector3(0, 0, 0), 0);
  fighters.push(player);
  spawnAt = clock + 0.5;
}
function spawn() {
  // in front of the player, a little to one side when others are already there
  const others = fighters.filter(f => !f.isPlayer && f.state !== 'out').length;
  const h = rig.heading(player.P) + (others ? (others % 2 ? 1.3 : -1.3) : 0.4), d = 3.0;
  const pos = flat(player.P[0]).add(new Vector3(Math.sin(h), 0, Math.cos(h)).multiplyScalar(d));
  const e = new Fighter(false, [0x7f97c9, 0x5fae98, 0x9a7fc9][(Fighter.seq ?? 0) % 3], pos, h + Math.PI);
  e.hp = e.maxHp = tutorial ? 260 : 400 + 20 * Math.min(kos, 6); // (a knock-out every 20-45 s: one opponent is a fight, not a pushover; the first one sooner)
  fighters.push(e);
  plog?.note('spawn', { id: e.id });
  hudFight();
}
/**
 * The opponents fight as the player does. Each moves its hips on the same spring toward a point it chooses (an
 * unseen cursor: the same momentum, the same top speed) and strikes with the same strikes, chosen the same way
 * (from its pose and momentum, at a part it aims at). The two nearest fight at once; a third waits a little apart,
 * circling, and steps in when one of them goes down. A fighter keeps its own range and goes in and out of it, circles,
 * steps in to strike (from a little out it drives in first), throws one to three blows, now and then steps out of the
 * player's wind-up, and answers a blow it took. When one takes a heavy blow the others hold back a moment.
 */
const crew = { attacker: null, watchUntil: 0 };
function enemies() { return fighters.filter(f => !f.isPlayer && f.state !== 'out'); }
const FIGHT = ['free', 'attack', 'guard', 'hit'];
/** Is E within the player's sight (55 degrees either side of where it faces)? */
function inView(P, E) {
  const h = rig.heading(P.P), to = flat(E.P[0].clone().sub(P.P[0]));
  return Math.abs(wrap(Math.atan2(to.x, to.z) - h)) < 55 * Math.PI / 180;
}
function think(E, dt) {
  const P = player;
  if (!P || !FIGHT.includes(E.state)) { E.spring = { target: null }; return; }
  const pp = flat(P.P[0]), ep = flat(E.P[0]);
  const to = pp.clone().sub(ep), d = to.length();
  to.normalize();
  const side = new Vector3(to.z, 0, -to.x);
  const B = E.brain ??= { range: rand(1.1, 1.4), bob: Math.random() * 6, circle: Math.random() < 0.5 ? 1 : -1, nextAttack: clock + rand(0.4, 1.0), combo: 0, lungeUntil: 0, evadeUntil: 0, evDir: 1 };
  // who fights: the two nearest that are on their feet
  const up = enemies().filter(e => FIGHT.includes(e.state)).sort((a, b) => flat(a.P[0]).distanceTo(pp) - flat(b.P[0]).distanceTo(pp));
  const fighting = up.indexOf(E) < 2;
  crew.attacker = up[0] ?? null;
  // keep off each other
  const apart = new Vector3();
  for (const o of enemies()) if (o !== E) { const dd = ep.clone().sub(flat(o.P[0])); const l = dd.length(); if (l < 1.2 && l > 1e-3) apart.addScaledVector(dd.normalize(), (1.2 - l) * 1.2); }
  let goal;
  const playerDown = P.state === 'down' || P.state === 'fall' || P.state === 'getup';
  E.mode = 'foot';
  if (!playerDown) B.groundAttack = null;
  else if (B.groundAttack == null) B.groundAttack = Math.random() < 0.3;
  if (playerDown && fighting && B.groundAttack && clock > crew.watchUntil) {
    E.mode = 'lunge';
    goal = pp.clone().addScaledVector(to, -1.0);
    if (E.state === 'free' && d < 1.4 && clock > B.nextAttack) {
      E.aim = { enemy: P, part: Math.random() < 0.5 ? 'body' : 'legs', joint: I.Spine2, off: new Vector3() };
      if (strike(E, 'kick') === 'started') { B.nextAttack = clock + rand(1.0, 1.6); B.groundAttack = false; } // (once)
    }
  } else if (playerDown || clock < crew.watchUntil) {
    E.mode = 'hold';
    goal = pp.clone().addScaledVector(to, -2.4).addScaledVector(side, 0.5 * Math.sin(clock + B.bob));
  } else if (!fighting) {
    // waiting: a place a little apart, circling slowly
    E.mode = 'wait';
    const at = Math.atan2(-to.x, -to.z) + 0.6 * dt * B.circle;
    goal = pp.clone().add(new Vector3(Math.sin(at), 0, Math.cos(at)).multiplyScalar(2.3));
  } else {
    // footwork. The pincer: the nearer presses from the front (in and out of its range), the other works round to
    // the player's flank or back (an arc off the player's facing)
    if (Math.random() < dt * 0.5) B.circle *= -1;
    const role = up[0] === E ? 'press' : 'angle';
    const r = B.range + 0.3 * Math.sin(clock * 3.6 + B.bob);
    goal = pp.clone().addScaledVector(to, -r).addScaledVector(side, 0.7 * B.circle);
    if (role === 'angle') {
      B.wantRear ??= Math.random() < 0.7;
      const facing = rig.heading(P.P), sideOf = Math.sign(wrap(Math.atan2(ep.x - pp.x, ep.z - pp.z) - facing)) || 1;
      const ang = facing + sideOf * (B.wantRear ? 2.6 : 1.4);
      goal = pp.clone().add(new Vector3(Math.sin(ang), 0, Math.cos(ang)).multiplyScalar(B.range + 0.1));
    }
    // flanked by the player: step off its line and turn
    if (aspect(E, P) !== 'front' && d < 2) { goal.addScaledVector(side, 0.5 * B.circle); E.mode = 'evade'; }
    // the player winds up close: step out of it now and then (back and to a side)
    if (clock > (E.exposed ?? 0) && P.state === 'attack' && P.attack && !P.attack.landed && d < 1.6 && clock > B.evadeUntil + 0.6 && Math.random() < dt * 4) { B.evadeUntil = clock + 0.35; B.evDir = Math.random() < 0.5 ? -1 : 1; }
    if (clock < B.evadeUntil) E.mode = 'evade';
    if (clock < B.evadeUntil) goal = ep.clone().addScaledVector(to, -0.7).addScaledVector(side, 0.6 * B.evDir);
    // driving in to strike
    // (driving in stops short of the body - at punching range, with room for the blow to be seen)
    if (clock < B.lungeUntil) { goal = pp.clone().addScaledVector(to, -0.95); E.mode = 'lunge'; }
    // strike: in range, its turn
    if (clock < (E.exposed ?? 0)) B.nextAttack = Math.max(B.nextAttack, E.exposed + rand(0.3, 0.6));
    // a charge: step back and load, hold, then lunge with the blow (one opponent at a time; every 2-3 s each)
    B.nextCharge ??= clock + rand(1.2, 2.2);
    const fromSide = aspect(P, E) !== 'front';
    // balance broken: reeling back, no blows until it is found again
    if (E.broken && clock < E.broken) { goal = pp.clone().addScaledVector(to, -(d + 0.6)).addScaledVector(side, 0.4 * Math.sin(clock * 4 + B.bob)); E.mode = 'evade'; B.nextAttack = Math.max(B.nextAttack, E.broken + 0.2); B.nextCharge = Math.max(B.nextCharge ?? 0, E.broken + 0.8); }
    const committed = (P.state === 'attack' && P.attack && clock - P.attack.t0 > 0.15) || (P.posture ?? 1) < 0.3 || (P.broken && clock < P.broken) || (P.dodge && clock - P.dodge.at < 0.4 && clock > P.dodge.until);
    if (E.state === 'free' && role === 'press' && !fromSide && !committed && d > 0.9 && d < 2.0 && clock > B.nextCharge && clock > (crew.chargeFreeUntil ?? 0) && clock > B.evadeUntil && clock > (E.exposed ?? 0)) {
      const kind = d > 1.3 ? (Math.random() < 0.6 ? 'kick' : 'punch') : Math.random() < 0.7 ? 'punch' : 'kick';
      const u = Math.random(), part = kind === 'kick' && u < 0.3 ? 'legs' : u < 0.6 ? 'head' : 'body';
      E.aim = { enemy: P, part, joint: part === 'head' ? I.Head : part === 'body' ? I.Spine2 : I.LeftShin, off: new Vector3() };
      // head-on: now and then a feint (the load, then nothing) in place of the real charge
      startCharge(E, kind, Math.random() < 0.15);
      B.nextCharge = clock + rand(1.2, 2.0) + 0.6; B.nextAttack = clock + 1.2;
      crew.chargeFreeUntil = clock + 0.9; crew.lastStrikeAt = clock; crew.lastStriker = E;
    }
    // the load: back the hips go; the lunge: through
    if (E.charge && !E.attack?.charged) { goal = pp.clone().addScaledVector(to, -(d + 0.45)); E.mode = 'evade'; }
    else if (E.charge && E.attack?.charged && !E.attack.followed) { goal = pp.clone().addScaledVector(to, 0.3); E.mode = 'lunge'; }
    // quick blows: from the side or behind, or when the player is committed (head-on otherwise: only a jab now and then)
    const facingP = aspect(E, P) === 'front';
    const turn = E.state === 'free' && facingP && clock > B.nextAttack && clock > B.evadeUntil && (fromSide || committed || clock > (B.nextJab ??= clock + rand(0.45, 0.75)));
    // its turn but a little out of reach (1.3-2 m): it drives in first, so the blow begins within reach and lands
    if (turn && d >= 1.3 && d < 2.0 && clock > B.lungeUntil + 0.5 && !E.charge) { B.lungeUntil = clock + 0.3; goal = pp.clone().addScaledVector(to, -0.95); E.mode = 'lunge'; }
    // the tell: a quick step back, then in with the blow
    if (B.tellAt) {
      { const ah = playerAhead(0.35); goal = ah.clone().addScaledVector(flat(ah.clone().sub(E.P[0])).normalize(), -0.9); E.mode = 'lunge'; } // (in to where the player is going, just short of it: the blow lands there)
      if (clock >= B.tellAt || E.state !== 'free') {
        const k = B.tellKind; B.tellAt = null;
        if (E.state === 'free' && strike(E, k) === 'started') {
          if (B.burst) { E.armorUntil = clock + 0.6; B.burst = false; } // (bursting back: light blows do not stop this one)
          B.announced = false;
          crew.lastStrikeAt = clock; crew.lastStriker = E;
          E.antic ??= { t0: clock, side: Math.random() < 0.5 ? 1 : -1 };
          B.combo = Math.random() < 0.5 ? (Math.random() < 0.4 ? 2 : 1) : 0;
          B.nextAttack = clock + rand(0.15, 0.35);
          if (!fromSide && !committed) B.nextJab = clock + rand(0.45, 0.75);
        } else { B.nextAttack = clock + 0.2; B.burst = false; if (E.antic?.big && !E.antic.baked) E.antic.until ??= clock; }
      }
    }
    const dAhead = flat(playerAhead(0.4)).distanceTo(flat(E.P[0]));
    if (turn && (d < 1.3 || dAhead < 1.4) && !B.tellAt) {
      {
        const kind = d > 1.1 ? (Math.random() < 0.65 ? 'kick' : 'punch') : Math.random() < 0.55 ? 'punch' : 'kick';
        const u = Math.random(), part = kind === 'kick' && u < 0.3 ? 'legs' : u < 0.6 ? 'head' : 'body';
        E.aim = { enemy: P, part, joint: part === 'head' ? I.Head : part === 'body' ? I.Spine2 : I.LeftShin, off: new Vector3() };
        // one at a time: the other waits a moment after one starts (a one-two from two sides, never together)
        // (one after the other: the second starts 0.35 s after the first, so the two glows and blows come apart)
        if (clock - (crew.lastStrikeAt ?? -9) < 0.12 && crew.lastStriker !== E) B.nextAttack = clock + rand(0.08, 0.15);
        else if (!inView(P, E) && !B.announced) {
          // out of the player's sight: it announces itself first (a warning at the player's feet), then strikes
          B.announced = true; E.warnUntil = clock + 0.45; B.nextAttack = clock + 0.45;
        } else { B.tellAt = clock + 0.16; B.tellKind = kind; crew.lastStrikeAt = clock; crew.lastStriker = E; E.antic = { t0: clock, side: Math.random() < 0.5 ? 1 : -1, big: true, kind }; } // (the tell first: a chamber held)
      }
    }
    // the rest of a combination: kept while the blow is out (it is chained from the top)
    if (E.state === 'attack' && B.combo > 0 && E.attack?.spliced && !E.attack.followed && E.attackQueued == null) {
      B.combo--;
      strike(E, Math.random() < 0.7 ? 'punch' : 'kick');
    }
  }
  E.spring = { target: goal.add(apart) };
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
  // on the floor: a click gets the player up at once (the fall itself still plays out)
  if (pressed && player.state === 'down' && !player.ko) player.getupNow = true;
  for (const [bit, kind] of [[1, 'punch'], [2, 'kick']]) {
    if (!(pressed & bit)) continue;
    let aim = aimFromCursor();
    // a left click on empty space (no opponent part under the cursor): a dash that way - unless the cursor is
    // still near the opponent the player is locked on (knocked back by a blow, stepped round in a dodge): that
    // click is a blow at it (high or low by where the cursor is), not a step away from it
    if (bit === 1 && !aim) {
      aim = lockedAim();
      const striking = player.state === 'attack' && player.attack && !player.attack.followed;
      if (!aim && !striking) { startDash(); continue; }
      if (!aim) aim = lockedAim(true);
    }
    // a press not yet taken is replaced by this one (logged as replaced)
    if (player.strikeInput?.id) plog?.pressResult(player.strikeInput.id, 'replaced');
    const id = plog?.press(kind) ?? 0;
    player.strikeInput = { t: clock, kind, id, aim: aim?.locked ? aim : null };
    cueStrike(kind);
  }
}
/** The aim a click means when the cursor is off every body but near the locked-on opponent (within `reach` m of its hips). */
function lockedAim(loose = false) {
  const L = player?.lockOn, c = cursorFloor();
  if (!L || L.state === 'out' || clock >= (player.lockUntil ?? 0) || !c) return null;
  if (flat(c).distanceTo(flat(L.P[0])) > (loose ? 2.0 : 1.1)) return null;
  const r = canvas.getBoundingClientRect(), q = L.P[I.Chest].clone().project(camera), chestY = r.top + (1 - q.y) / 2 * r.height;
  const high = mouse.y < chestY, joint = high ? I.Head : I.Spine2;
  return { enemy: L, part: high ? 'head' : 'body', joint, point: L.P[joint].clone(), off: new Vector3(), score: 0, locked: true };
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
/**
 * The jump kick: the body leaves the floor as the kick goes out (up to 30 cm at its top), the other leg tucked, comes
 * down after it, and the knees take the landing (the hips dip). Shown over the generated kick.
 */
/**
 * The player's body leads its motion without sliding its feet (speculative, then handed over): what the spring wants
 * and what the generated motion has actually moved apart (the "debt", leaking away over 0.35 s) is shown as the hips
 * and the trunk going first - the feet stay where they are on the floor and the legs take the angle, the trunk leans
 * into it (as a body falls into its first step before the foot moves). Up to 10 cm, gone again as MotionBricks'
 * own steps catch up. Nothing is slid: the generated footwork is untouched.
 */
const TUNE = { lead: true, spec: true, track: true };
function leadOn(F, dt, r0) {
  F.debt ??= new Vector3();
  if (!TUNE.lead || F.state !== 'free' || F.ragOn || (F.dash && clock < F.dash.until) || !F.spring?.target) { F.debt.multiplyScalar(Math.exp(-dt / 0.08)); return; }
  const r1 = F.track.display.root;
  F.debt.x += F.vel.x * dt - (r1.x - r0.x); F.debt.z += F.vel.z * dt - (r1.z - r0.z);
  F.debt.multiplyScalar(Math.exp(-dt / 0.35)).clampLength(0, 0.4);
}
function leadLayer(F) {
  F.leadOff = null;
  if (!F.debt || F.state !== 'free' || F.ragOn) { F.leadShown = 0; return; }
  // (eased in and out: the offset never jumps)
  const want = Math.min(0.1, F.debt.length() * 0.6);
  F.leadShown = (F.leadShown ?? 0) + (want - (F.leadShown ?? 0)) * 0.25;
  if (F.leadShown < 0.005) return;
  const dir = flat(F.debt).normalize(), off = dir.clone().multiplyScalar(F.leadShown);
  F.leadOff = off;
  const p = clonePose(F.shown), P0 = rig.positions(p);
  const feet = ['Left', 'Right'].map(sd => P0[I[sd + 'Foot']].clone());
  p.root.x += off.x; p.root.z += off.z; p.root.y -= 0.35 * F.leadShown * F.leadShown / 0.1; // (the hips a little lower: the legs reach)
  const axis = new Vector3().crossVectors(UP, dir).normalize();
  if (F.leanSign == null) { const q = clonePose(p); rotateSubtree(rig, q, I.Spine1, new Quaternion().setFromAxisAngle(axis, 0.2)); F.leanSign = rig.positions(q)[I.Head].dot(dir) > rig.positions(p)[I.Head].dot(dir) ? 1 : -1; }
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(axis, F.leanSign * 1.4 * F.leadShown));
  ['Left', 'Right'].forEach((sd, i) => legTo(p, sd, feet[i]));
  F.shown = rig.fixRigid(p);
}
/**
 * The player's blow starts with the press (speculative, then handed over): from 0.06 s after the press (the wind-up
 * shown first) the body is drawn toward the strike's top pose, accelerating, reaching it 0.22 s later; when the
 * generated strike arrives, the layer gives way to it as fast as the generated limb gets there itself (the shown
 * progress is the larger of the two: it never goes back to the generated wind-up).
 */
function specStrike(F) {
  const S = F.spec;
  if (!S) return;
  const A = S.atk, t = clock - S.t0;
  const pg = A.spliced && A.contactAt > A.spliceAt ? clamp((clock - A.spliceAt) / (A.contactAt - A.spliceAt), 0, 1) : 0;
  if (F.state !== 'attack' || (F.attack && F.attack !== A && F.attack.spliced) || pg >= 0.98 || t > 0.9 || ((A.landed || A.followed) && clock - (S.endAt ??= clock) > 0.15)) { F.spec = null; return; }
  const u = clamp((t - 0.06) / 0.22, 0, 1), ps = u * u;
  let w = ps > pg ? (ps - pg) / (1 - pg) : 0;
  if (A.landed || A.followed) w = (S.wEnd ??= w) * (1 - (clock - S.endAt) / 0.15); else S.wEnd = null;
  if (w <= 0.001 || !TUNE.spec) return;
  S.joints ??= rig.subtree(I.Spine1); // (the upper body only: a kicking leg turned while its foot is still down slid it along the floor)
  const p = clonePose(F.shown);
  for (const j of S.joints) p.rot[j].slerp(S.pose.rot[j], 0.9 * w);
  F.shown = rig.fixRigid(p);
}
/**
 * Where the player will be `T` seconds on: its hips run on their spring toward the cursor (as its own strikes are
 * placed), so an opponent throws at where the player is going, not where its motion has got to.
 */
function playerAhead(T) {
  const P = player, v = P.vel.clone(), p = flat(P.P[0]);
  for (let t = 0; t < T; t += 1 / 30) { springStep(v, p, P.spring?.target ?? null, 1 / 30, feelOf(P)); p.addScaledVector(v, 1 / 30); }
  return p;
}
function jumpLayer(F) {
  const J = F.jump;
  if (!J) return;
  const t = clock - J.t0, FALL = 0.26, LAND = 0.22;
  if (t > J.up + FALL + LAND || ['hit', 'fall', 'down', 'getup', 'out'].includes(F.state)) { F.jump = null; return; }
  const y = t < J.up ? J.h * Math.sin((Math.PI / 2) * (t / J.up)) : t < J.up + FALL ? J.h * Math.cos((Math.PI / 2) * ((t - J.up) / FALL)) : -0.08 * Math.sin(Math.PI * ((t - J.up - FALL) / LAND));
  const p = clonePose(F.shown), P0 = rig.positions(p);
  const feet = ['Left', 'Right'].map(sd => P0[I[sd + 'Foot']].clone());
  p.root.y += y;
  if (y > 0) {
    // airborne: the support leg drawn up under the body
    const i = J.side === 'Left' ? 0 : 1, f = feet[i].clone(); f.y += y + 0.18 * (y / J.h);
    legTo(p, J.side, f);
  } else ['Left', 'Right'].forEach((sd, i) => legTo(p, sd, feet[i])); // (landing: the knees bend, the feet stay down)
  F.shown = rig.fixRigid(p);
}
/**
 * The weight of a blow, after its top: the trunk carries on past the pose and swings back (a damped swing, about
 * 2 Hz), the hips sink into the knees and come up, the turn of the shoulders overshoots - a body with mass, not a
 * pose set down. A blow that met something recoils (the trunk back) rather than carrying through; a kick lands
 * heavier.
 */
function settleLayer(F) {
  const A = F.attack;
  if (A?.contactAt && !A.settled && clock >= A.contactAt && F.state === 'attack') {
    A.settled = true;
    F.settle = { t0: clock, kind: A.kind, landed: !!A.landed, side: A.part?.startsWith('Left') ? 1 : -1, jump: !!A.jump };
  }
  const S = F.settle;
  if (!S) return;
  const t = clock - S.t0;
  if (t > 0.7 || ['hit', 'fall', 'down', 'getup', 'out'].includes(F.state)) { F.settle = null; return; }
  const kick = S.kind === 'kick' || S.kind === 'knee', amp = F.isPlayer ? 1 : 0.8;
  const swing = Math.exp(-t / 0.2) * Math.sin(2 * Math.PI * 2.1 * t);
  const lean = amp * (S.landed ? -0.09 : kick ? 0.07 : 0.13) * swing;
  const twist = amp * (S.landed ? 0.05 : 0.12) * swing * S.side;
  const dip = S.jump ? 0 : amp * (kick ? 0.055 : 0.04) * Math.sin(Math.PI * Math.min(1, t / 0.4)) * Math.exp(-t / 0.35);
  const p = clonePose(F.shown), P0 = rig.positions(p), h = rig.heading(P0), fwd = new Vector3(Math.sin(h), 0, Math.cos(h)), lat = new Vector3().crossVectors(UP, fwd).normalize();
  const feet = ['Left', 'Right'].map(sd => P0[I[sd + 'Foot']].clone());
  if (S.fwdSign == null) { const q = clonePose(p); rotateSubtree(rig, q, I.Spine1, new Quaternion().setFromAxisAngle(lat, 0.2)); S.fwdSign = rig.positions(q)[I.Head].dot(fwd) > P0[I.Head].dot(fwd) ? 1 : -1; }
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lat, S.fwdSign * lean));
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(UP, twist));
  p.root.y -= dip;
  // (the feet stay down; a kicking foot still in the air the first moment is left free)
  ['Left', 'Right'].forEach((sd, i) => { if (!(kick && t < 0.15 && (sd === 'Left') === (S.side === 1))) legTo(p, sd, feet[i]); });
  F.shown = rig.fixRigid(p);
}
/**
 * An opponent's blow follows the player until it commits: from the moment its motion plays until 0.14 s before the
 * contact, the whole body slides after how far the player has gone since it was aimed (up to 2.2 m/s) - walking off
 * does not take you out of it; a dodge (the flick), or stepping away in the last instant, does.
 */
function trackIn(F, dt) {
  const A = F.attack;
  if (F.isPlayer || F.state !== 'attack' || !A?.spliced || !A.oppAt || A.landed || A.followed || !A.contactAt || !player) return;
  if (player.dodge?.flick && clock < player.dodge.until + 0.1) { A.oppAt = null; return; } // (dodged: it goes where it was going)
  if (clock > A.contactAt - 0.12 || ['down', 'fall', 'getup'].includes(player.state)) return;
  if (!TUNE.track || clock > A.spliceAt + 0.6 * (A.contactAt - A.spliceAt)) return;
  A.tracked ??= 0;
  const want = playerAhead(clamp(A.contactAt - clock, 0, 0.35)).sub(A.oppAt), step = want.clampLength(0, Math.min(1.5 * dt, Math.max(0, 0.3 - A.tracked)));
  if (step.lengthSq() < 1e-8) return;
  F.track.shift(step.x, step.z); A.oppAt.add(step); A.tracked += step.length();
}
function anticipate(F) {
  const A = F.antic;
  if (!A) return;
  const t = clock - A.t0;
  if (F.track.tag.kind === 'attack' && !A.until) A.until = clock;
  if (A.big) { chamber(F, A, t); return; }
  const w = A.until ? Math.max(0, 1 - (clock - A.until) / 0.12) : Math.min(1, t / 0.12);
  if ((A.until && w <= 0) || t > 1.2) { F.antic = null; return; }
  const p = clonePose(F.shown);
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, A.side * 0.22 * w));
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(UP, A.side * 0.12 * w));
  p.root.y -= 0.03 * w;
  F.shown = rig.fixRigid(p);
}
/**
 * An opponent's wind-up, plain in the pose (not only in the glow): over a quarter second the trunk turns well away
 * and leans back, the hips sink, and for a punch the rear arm is drawn back and up, folded (for a kick the arms come
 * up and the trunk leans back further, the weight on the back leg); it is held until the strike's motion shows,
 * then let go in 0.08 s - the snap.
 */
/**
 * The player setting off: before the generated step comes (the model takes a moment), the body tips the way it is
 * pulled - over 0.06 s, held while the step is on its way, eased out as it arrives - so a pull answers at once.
 */
function setOffLean(F) {
  const S = F.setOff;
  if (!S) return;
  const t = clock - S.t0;
  if (F.state !== 'free') S.until ??= clock;
  const w = Math.min(1, t / 0.06) * (t < 0.3 ? 1 : Math.max(0, 1 - (t - 0.3) / 0.3)) * (S.until != null ? Math.max(0, 1 - (clock - S.until) / 0.08) : 1);
  if (t > 0.6 || w <= 0) { F.setOff = null; return; }
  const p = clonePose(F.shown), P0 = rig.positions(p);
  const feet = ['Left', 'Right'].map(sd => P0[I[sd + 'Foot']].clone());
  const axis = new Vector3().crossVectors(UP, S.dir).normalize();
  const pivot = feet[0].clone().lerp(feet[1], 0.5).setY(0);
  rotateSubtree(rig, p, 0, new Quaternion().setFromAxisAngle(axis, 0.12 * w), pivot);
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(axis, 0.08 * w));
  p.root.addScaledVector(S.dir, 0.05 * w);
  ['Left', 'Right'].forEach((sd, i) => legTo(p, sd, feet[i]));
  F.shown = rig.fixRigid(p);
}
/**
 * A living body is never still: when the shown pose has all but stopped (a held top of a strike, a body waiting for
 * its next motion, a guard held) a little motion comes in over it - the trunk breathing and swaying, the head
 * settling, the arms drifting - at a few centimetres, never hiding the pose; it fades as the body moves again.
 */
function aliveLayer(F, dt) {
  if (['fall', 'down', 'getup', 'out'].includes(F.state) || F.ragOn) { F.aliveW = 0; F.aliveBase = null; return; }
  const P = rig.positions(F.shown), D = rig.positions(F.track.display), prev = F.aliveBase;
  F.aliveBase = [I.Head, I.LeftHand, I.RightHand, I.Spine2].map(j => D[j].clone());
  if (!prev) return;
  let sp = 0; F.aliveBase.forEach((v, i) => { sp += v.distanceTo(prev[i]); }); sp /= 4 * Math.max(dt, 1e-3);
  const want = clamp(1 - (sp - 0.15) / 0.5, 0, 1);
  F.aliveW = (F.aliveW ?? 0) + (want - (F.aliveW ?? 0)) * (1 - Math.exp(-dt / (want > (F.aliveW ?? 0) ? 0.25 : 0.08)));
  const w = F.aliveW;
  if (w < 0.02) return;
  const ph = F.alivePh ??= Math.random() * 100, t = clock + ph;
  const h = rig.heading(P), fwd = new Vector3(Math.sin(h), 0, Math.cos(h)), lat = new Vector3().crossVectors(UP, fwd).normalize();
  const p = clonePose(F.shown);
  const feet = ['Left', 'Right'].map(sd => P[I[sd + 'Foot']].clone());
  const breath = Math.sin(t * 2 * Math.PI * 0.45), sway = Math.sin(t * 2 * Math.PI * 0.31 + 1.3), jit = Math.sin(t * 2 * Math.PI * 1.7 + 0.4);
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lat, 0.035 * breath * w));
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, 0.04 * sway * w));
  rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(fwd, 0.04 * jit * w));
  for (const [sd, k] of [['Left', 0.7], ['Right', -0.9]]) rotateSubtree(rig, p, I[sd + 'Arm'], new Quaternion().setFromAxisAngle(lat, 0.05 * Math.sin(t * 2 * Math.PI * 0.6 + k) * w));
  p.root.addScaledVector(lat, 0.02 * sway * w); p.root.y += 0.012 * breath * w;
  ['Left', 'Right'].forEach((sd, i) => legTo(p, sd, feet[i]));
  F.shown = rig.fixRigid(p);
}
function windWeight(A, t) { const u = Math.min(1, Math.max(0, t) / (A.fast ? 0.1 : 0.25)); return u * u * (3 - 2 * u); }
function chamber(F, A, t) {
  const win = windWeight(A, t);
  let w;
  if (A.baked && A.a != null) {
    // a strike generated from the wound-up body (windPose in its context): the layer gives way over the frame
    // where that motion begins
    w = win * clamp(A.a + 1 - F.track.now, 0, 1);
    if (w <= 0 || clock - A.t0 > 2) { F.antic = null; return; }
  } else {
    if (F.track.tag.kind === 'attack' && !A.until) A.until = clock;
    // (4) a wind-up for a strike that never came (or held too long): let go over 0.08 s, never a snap
    if (!A.until && (t > 1.4 || !['free', 'attack', 'guard'].includes(F.state))) A.until = clock;
    w = A.until ? win * Math.max(0, 1 - (clock - A.until) / 0.08) : win;
    if (A.until && w <= 0) { F.antic = null; return; }
  }
  F.shown = windPose(F.shown, A, w);
}
/**
 * The wind-up: the trunk turned well away from the strike and leaning back, the hips sunk, for a punch the arm on the
 * turned-back side drawn back and up, folded (for a kick the arms up and the trunk back further, the weight on the
 * back leg). The swing then goes from the other side through to the target - the load that shows the blow coming.
 */
function windPose(pose, A, w) {
  if (w <= 0) return pose;
  const p = clonePose(pose);
  const P0 = rig.positions(p), h = rig.heading(P0), fwd = new Vector3(Math.sin(h), 0, Math.cos(h)), lat = new Vector3().crossVectors(UP, fwd).normalize();
  const kick = A.kind === 'kick';
  const feet = ['Left', 'Right'].map(sd => P0[I[sd + 'Foot']].clone());
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, A.side * 0.34 * w));
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(UP, A.side * 0.22 * w));
  const lean = (kick ? 0.28 : 0.14) * w;
  // (which way each turn goes is tried at full strength once, then kept for this wind-up)
  const S = A.signs ??= {};
  const sgn = (key, j, axis, ang, score) => { if (S[key] == null) { let bs = -Infinity; for (const sg of [1, -1]) { const q = clonePose(p); rotateSubtree(rig, q, j, new Quaternion().setFromAxisAngle(axis, sg * ang)); const v = score(rig.positions(q)); if (v > bs) { bs = v; S[key] = sg; } } } return S[key]; };
  const full = kick ? 0.28 : 0.14;
  rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lat, sgn('lean', I.Spine1, lat, full, Q => -Q[I.Head].dot(fwd)) * lean));
  p.root.y -= 0.07 * w;
  if (!S.rear) { const P1 = rig.positions(p); S.rear = P1[I.LeftArm].dot(fwd) < P1[I.RightArm].dot(fwd) ? 'Left' : 'Right'; }
  // the hands placed, not swung: for a punch the fist drawn back beside the ear on the turned-back side (elbow
  // back, cocked); for a kick both hands up in front of the face (a guard, for balance)
  const P2 = rig.positions(p), up = new Vector3(0, 1, 0);
  for (const side of kick ? ['Left', 'Right'] : [S.rear]) {
    const sh = P2[I[side + 'Arm']], hand = P2[I[side + 'Hand']], out = side === 'Left' ? lat : lat.clone().negate();
    const goal = kick ? sh.clone().addScaledVector(fwd, 0.28).addScaledVector(up, 0.12).addScaledVector(out, -0.06)
                      : sh.clone().addScaledVector(fwd, -0.16).addScaledVector(up, 0.04).addScaledVector(out, 0.1);
    const q = reachLimb(p, side + 'Hand', hand.clone().lerp(goal, w), 1.0);
    for (const j of rig.subtree(I[side + 'Arm'])) p.rot[j].copy(q.rot[j]);
  }
  // the feet stay where they were (the hips sink into bent knees, not the feet into the floor)
  ['Left', 'Right'].forEach((sd, i) => legTo(p, sd, feet[i]));
  return rig.fixRigid(p);
}
/**
 * The top of a strike made bigger: a punch or an elbow turns the trunk on into it (the striking shoulder forward), a
 * kick or a knee leans the trunk back against it - the silhouette reads from across the floor.
 */
function bigTop(pose, k, fwd, gain = 1) {
  const p = clonePose(pose), tipJ = I[k.part];
  if (k.kind === 'punch' || k.kind === 'elbow') {
    const a = 0.26 * gain;
    const fwdOf = sg => { const q = clonePose(p); rotateSubtree(rig, q, I.Spine1, new Quaternion().setFromAxisAngle(UP, sg * a)); return rig.positions(q)[tipJ].dot(fwd); };
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, (fwdOf(1) > fwdOf(-1) ? 1 : -1) * a));
  } else {
    const lat = new Vector3().crossVectors(UP, fwd).normalize(), a = 0.16 * gain;
    const headFwd = sg => { const q = clonePose(p); rotateSubtree(rig, q, I.Spine1, new Quaternion().setFromAxisAngle(lat, sg * a)); return rig.positions(q)[I.Head].dot(fwd); };
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lat, (headFwd(1) < headFwd(-1) ? 1 : -1) * a));
  }
  return rig.fixRigid(p);
}
/** A body knocked a little way along the floor: the motion itself moves (and goes on from there), over a moment. */
function shove(F, dir, dist, dur = 0.12) {
  const d = flat(dir.clone()); if (d.lengthSq() < 1e-6 || !(dist > 0)) return;
  F.shove = { dir: d.normalize(), dist: Math.min(dist + (F.shove ? F.shove.dist - F.shove.done : 0), 0.6), t0: clock, dur, done: 0 };
}
function shoveOn(F) {
  const S = F.shove;
  if (!S) return;
  const u = Math.min(1, (clock - S.t0) / S.dur), want = S.dist * (1 - (1 - u) ** 2), step = want - S.done;
  if (step > 0) { F.track.shift(S.dir.x * step, S.dir.z * step); S.done = want; }
  if (u >= 1) F.shove = null;
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
/**
 * A heavy blow carries the body off at once: until the generated reaction is shown (the model takes a moment), the
 * shown body moves along the blow at the knock's speed; then that shift eases out under the reaction's own motion.
 */
/** On the floor, alive: the chest and head rise and fall, the body never lies like a thing. */
function breathe(F) {
  if (F.state !== 'down' && F.state !== 'fall' || F.ko) return;
  const p = clonePose(F.shown), t = clock * 2.4 + (F.id ?? 0);
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), 0.06 * Math.sin(t)));
  rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), 0.1 * Math.sin(t * 0.7 + 1)));
  rotateSubtree(rig, p, I.LeftArm, new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), 0.12 * Math.sin(t * 0.5)));
  F.shown = rig.fixRigid(p);
}
/** Which part a blow at `at` met: the nearest of head, trunk, forearms, shins. */
function partAt(F, at, noArms = false) {
  const c = [['head', I.Head], ['body', I.Spine2], ['armL', I.LeftForeArm], ['armR', I.RightForeArm], ['legL', I.LeftShin], ['legR', I.RightShin]];
  let best = c[1], bd = 9;
  for (const x of c) { if (noArms && x[0].startsWith('arm')) continue; const d = F.P[x[1]].distanceTo(at); if (d < bd) { bd = d; best = x; } }
  return best;
}
/** A blow pushes the part it met, along the blow, at once (a quick snap, then easing back over ~0.3 s). */
function pushPart(F, at, dir, energy) {
  const [part, joint] = partAt(F, at);
  (F.pushes ??= []).push({ part, dir: flat(dir).normalize(), amp: energy < 12 ? 0.05 : clamp(0.08 + energy / 300, 0.08, 0.34), t0: clock, dur: energy < 12 ? 0.2 : 0.34 });
  F.dispProbe = { joint, p0: F.P[joint].clone(), t0: clock, energy, part };
}
function balanceLayer(F, dtNow = 1 / 30) {
  const broken = F.broken && clock < F.broken;
  // the guard level, as a stance one can read: low - crouched, leaning in, hands down at the body
  if (!F.isPlayer) {
    F.stanceW = (F.stanceW ?? 0) + ((F.stance === 'low' && !broken ? 1 : 0) - (F.stanceW ?? 0)) * Math.min(1, dtNow * 6);
    const sw = F.stanceW;
    // the switch overshoots for a moment (hands thrown down into the low guard, or snapped up into the high one)
    const sinceSwitch = clock - (F.stanceAt ?? -9), pulse = sinceSwitch < 0.35 && !broken ? Math.sin(Math.PI * sinceSwitch / 0.35) * (F.stance === 'low' ? 1 : -1) : 0;
    if ((sw > 0.01 || pulse !== 0) && (F.state === 'free' || F.state === 'guard')) {
      const p = clonePose(F.shown), P0 = rig.positions(p), h = rig.heading(P0), fwd = new Vector3(Math.sin(h), 0, Math.cos(h));
      const axis = new Vector3().crossVectors(UP, fwd).normalize();
      rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(axis, 0.22 * sw));
      for (const s of ['Left', 'Right']) rotateSubtree(rig, p, I[s + 'Arm'], new Quaternion().setFromAxisAngle(axis, 0.55 * sw + 0.3 * pulse));
      p.root.y -= 0.12 * sw;
      for (const s of ['Left', 'Right']) legTo(p, s, P0[I[s + 'Foot']].clone());
      F.shown = rig.fixRigid(p);
    }
  }
  const w = broken ? 1 : clamp((0.7 - (F.posture ?? 1)) / 0.7, 0, 1);
  if (w <= 0.01 || ['down', 'fall', 'getup', 'out'].includes(F.state)) return;
  const p = clonePose(F.shown), h = rig.heading(rig.positions(p)), fwd = new Vector3(Math.sin(h), 0, Math.cos(h));
  const axis = new Vector3().crossVectors(UP, fwd).normalize(), sway = Math.sin(clock * (broken ? 4.5 : 3)) * (broken ? 0.1 : 0.04) * w;
  rotateSubtree(rig, p, I.Spine2, new Quaternion().setFromAxisAngle(axis, (broken ? 0.35 : 0.12) * w));
  rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(axis, (broken ? 0.5 : 0.15) * w));
  for (const s of ['Left', 'Right']) rotateSubtree(rig, p, I[s + 'Arm'], new Quaternion().setFromAxisAngle(fwd, (s === 'Left' ? -1 : 1) * 0.4 * w)); // (the guard drops)
  p.root.addScaledVector(new Vector3(fwd.z, 0, -fwd.x), sway);
  if (broken) {
    const P0 = rig.positions(p); p.root.y -= 0.1;
    for (const s of ['Left', 'Right']) legTo(p, s, P0[I[s + 'Foot']].clone());
  }
  F.shown = rig.fixRigid(p);
}
function hitPush(F) {
  if (!F.pushes?.length) return;
  F.pushes = F.pushes.filter(h => clock - h.t0 < h.dur);
  if (!F.pushes.length) return;
  const p = clonePose(F.shown);
  for (const h of F.pushes) {
    const t = (clock - h.t0) / h.dur, w = t < 0.15 ? t / 0.15 : (1 - (t - 0.15) / 0.85) ** 2;
    const axis = new Vector3().crossVectors(UP, h.dir);
    if (axis.lengthSq() < 1e-6) continue;
    axis.normalize();
    const a = h.amp * w, q = k => new Quaternion().setFromAxisAngle(axis, k * a);
    if (h.part === 'head') { rotateSubtree(rig, p, I.Neck1, q(4)); rotateSubtree(rig, p, I.Spine2, q(1.5)); }
    else if (h.part === 'body') { rotateSubtree(rig, p, I.Spine1, q(3)); p.root.addScaledVector(h.dir, a * 0.6); }
    else if (h.part.startsWith('arm')) rotateSubtree(rig, p, I[(h.part === 'armL' ? 'Left' : 'Right') + 'Arm'], q(5));
    else {
      const side = h.part === 'legL' ? 'Left' : 'Right', P0 = rig.positions(p);
      const foot = P0[I[side + 'Foot']].clone().addScaledVector(h.dir, a * 1.5); foot.y += 0.08 * w;
      legTo(p, side, foot);
    }
  }
  F.shown = rig.fixRigid(p);
}
/**
 * The knock becomes real: the reaction MotionBricks made starts from where the body was before the blow, so the
 * way the body has been carried since (shown on top) is added to the reaction's frames, and held on top until the
 * playhead reaches them (and through the blend into them).
 */
function bakeKnock(F, frames, a0) {
  const K = F.knockFx;
  if (!K || K.baked || !(K.d > 0.01)) return;
  const o = K.dir.clone().multiplyScalar(K.d);
  for (const f of frames) { f.pose = clonePose(f.pose); f.pose.root.add(o); }
  K.baked = { o, from: a0 };
}
function knockOn(F) {
  const K = F.knockFx;
  if (!K) return;
  if (K.baked) {
    const w = F.track.now < K.baked.from ? 1 : F.track.blend ? F.track.blend.w ** 2 : 0;
    if (w <= 0) { F.knockFx = null; return; }
    F.shown.root.x += K.baked.o.x * w; F.shown.root.z += K.baked.o.z * w;
    return;
  }
  // the carry holds until the blow's own reaction takes it over (bakeKnock, from the reaction's frames); a blow
  // that gets no reaction of its own (a sway through a strike, a body freed to move, a body already reeling from the
  // last blow) eases back after half a second
  // (not from the motion's tag: a body struck while still in the last blow's reaction must still be carried off)
  if (!K.until && clock - K.t0 > 0.45 && !(F.track.busy && ['hit', 'fall'].includes(F.state) && clock - K.t0 < 1.4)) { const d0 = K.v0 * 0.35; F.track.shift(K.dir.x * d0, K.dir.z * d0); F.knockFx = null; return; }
  const t = (K.until ?? clock) - K.t0;
  const w = K.until ? Math.max(0, 1 - Math.max(0, clock - K.until) / (K.fade ?? 0.15)) : 1;
  if ((K.until && w <= 0) || clock - K.t0 > 1.5) { F.knockFx = null; return; }
  const d = K.v0 * Math.min(t, 0.35) * w;
  K.d = d;
  F.shown.root.x += K.dir.x * d; F.shown.root.z += K.dir.z * d;
}
/** The body slid along the floor by a blow taken lying: the motion shifted a little each step, slowing down. */
function slideOn(F) {
  const S = F.slide;
  if (!S) return;
  const u = Math.min(1, (clock - S.t0) / S.dur), want = S.dist * (1 - (1 - u) ** 2), step = want - S.done;
  if (step > 0) { F.track.shift(S.dir.x * step, S.dir.z * step); S.done = want; }
  if (u >= 1) F.slide = null;
}
function homing(F) {
  const A = F.attack, o = F.aim?.enemy ?? F.opponent;
  if (F.state !== 'attack' || !A?.key || A.landed || !o || ['down', 'getup', 'fall'].includes(o.state) || F.track.tag.kind !== 'attack' || /ForeArm|Shin/.test(A.part)) return;
  const tip = rig.positions(F.shown)[I[A.part]];
  const k = A.key;
  const goal = F.aim && F.aim.part !== 'legs' ? o.P[F.aim.joint].clone().add(F.aim.off) : k.family === 'low' ? o.P[I.LeftShin].clone().lerp(o.P[I.RightShin], 0.5) : k.high ? o.P[I.Head].clone() : o.P[I.Spine2].clone();
  // the player's limb homes in more; an opponent's a little, and not on a player stepping out of it
  if (!F.isPlayer && o.dodge?.flick && clock < o.dodge.until + 0.1) return;
  const w = clamp(((F.isPlayer ? 0.7 : 0.85) - tip.distanceTo(goal)) / 0.45, 0, 1) * 0.7;
  if (w <= 0) return;
  F.shown = lerpPose(F.shown, reachLimb(F.shown, A.part, goal, F.isPlayer ? 0.2 : 0.32), w);
}
function startCharge(E, kind, feint = false) {
  // (a hit or a fall already on its way outranks the load: then no charge - not a fighter marked as charging that never loads)
  if (!E.track.request(chargeLoadReq(E, kind))) { E.charge = null; return false; }
  E.charge = { t0: clock, kind, phase: 'load', feint };
  E.attack = null; E.state = 'attack'; E.until = clock + 3; E.windAt = clock; E.windKind = kind;
  plog?.note('charge', { who: E.id, kind });
  return true;
}
/**
 * The load of a charge: the strike's own wind-up pose (10 frames before its top), 0.4 m back from where the body is,
 * weight sunk and the trunk turned away; held 0.3 s; then the lunge strike follows (queued at its end).
 */
function chargeLoadReq(E, kind) {
  return {
    kind: 'attack', mode: 'now', interrupt: true,
    build(tr, a, ctx) {
      const A = flat(ctx[3].root), P = player;
      const yaw = Math.atan2(P.P[0].x - A.x, P.P[0].z - A.z), fwd = new Vector3(Math.sin(yaw), 0, Math.cos(yaw));
      const pool = strikeKeys.filter(k => (kind === 'kick' ? k.kind === 'kick' || k.kind === 'knee' : k.kind === 'punch' || k.kind === 'elbow'));
      const k = pool[Math.floor(Math.random() * pool.length)] ?? strikeKeys[0];
      let t = ground(placePose(k.chamber[1].pose, yaw, ZERO, ZERO));
      const want = A.clone().addScaledVector(fwd, -0.4), off = want.sub(flat(t.root));
      t.root.add(off);
      t = clonePose(t);
      rotateSubtree(rig, t, I.Spine1, new Quaternion().setFromAxisAngle(UP, (Math.random() < 0.5 ? 1 : -1) * 0.3));
      t.root.y -= 0.06;
      t = rig.fixRigid(t);
      return {
        targets: [t, t, t, t], tokens: [6], rate: 1.6, fixEnd: true, tag: { kind: 'attack' },
        after: (tr2, a0, gen) => {
          groundFrames(gen);
          const end = tr2.frames[tr2.frames.length - 1];
          for (let i = 0; i < FPS * 0.3; i++) tr2.frames.push({ pose: end.pose, rate: 1, tag: end.tag });
          if (E.charge?.feint) { E.charge = null; E.state = 'free'; E.until = 0; plog?.note('feint', { who: E.id }); } // (a feint: the load, and back to footwork)
          else if (E.charge) { E.charge.phase = 'hold'; tr2.request(strikeReq(E, null, 'end', kind, 0, { lunge: true })); }
        },
      };
    },
  };
}
function strike(F, kind = null, pressId = 0) {
  if (!F.opponent) return 'no-opponent';
  if (F.state !== 'free' && F.state !== 'attack' && F.state !== 'guard') return `refused:${F.state}`;
  // during a strike: kept, and the next one starts from where this one ends (a combination)
  if (F.state === 'attack' && F.attack && !F.attack.followed) {
    if (F.attackQueued != null && F.queuedPress) plog?.pressResult(F.queuedPress, 'merged');
    F.attackQueued = clock; F.queuedKind = kind; F.queuedPress = pressId; return 'queued';
  }
  F.attackQueued = null;
  if (!F.track.request(strikeReq(F, null, 'now', kind, pressId))) return 'refused:busy';
  F.windAt = clock; F.windKind = kind; // (a hit or a fall on its way: not now)
  if (F.state !== 'attack') F.attack = null; // (an old strike's record no longer counts)
  // the load-up shows at once: turning away from the hand nearer the opponent (that one jabs, the other one winds)
  if (F.isPlayer && F.opponent) {
    const h = rig.heading(F.P), left = new Vector3(Math.cos(h), 0, -Math.sin(h)), to = flat((F.aim?.enemy ?? F.opponent).P[0].clone().sub(F.P[0]));
    F.antic = { t0: clock, side: Math.sign(to.dot(left) || 1), big: true, fast: true, kind: kind ?? 'punch' };
  }
  // (out of a run the plant takes most of the momentum; out of a dodge the step carries into the counter)
  if (F.isPlayer) {
    // (the run-in read at the press, before the plant takes the momentum: a kick out of a hard run is a jump kick, which keeps most of it)
    const to = flat((F.aim?.enemy ?? F.opponent).P[0].clone().sub(F.P[0])), dist = to.length();
    F.runIn = { t: clock, speed: dist > 1e-3 ? F.vel.dot(to.normalize()) : 0 };
    const jumping = kind === 'kick' && F.runIn.speed > 1.6;
    if (!(F.dodge && clock - F.dodge.at < 0.6)) F.vel.multiplyScalar(jumping ? 0.85 : F.dash && clock < F.dash.until + 0.2 ? 0.45 : 0.7);
    F.dash = null; F.setOff = null;
  }
  F.state = 'attack'; F.until = clock + 1.0;
  return 'started';
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
const lookLine = new THREE.Line(lookGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45, depthTest: false }));
const lookDot = new THREE.Mesh(new THREE.SphereGeometry(0.04, 10, 8), new THREE.MeshBasicMaterial({ color: 0xfff2b0, transparent: true, opacity: 0.8, depthTest: false }));
const cueRing = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.1, 32), new THREE.MeshBasicMaterial({ color: 0xffe070, transparent: true, side: THREE.DoubleSide, depthTest: false }));
lookLine.renderOrder = lookDot.renderOrder = cueRing.renderOrder = 15;
scene.add(lookLine, lookDot, cueRing);
let cueT = -9;
function cueStrike(kind) { cueT = performance.now(); cueRing.material.color.set(kind === 'kick' ? 0xff9a3c : 0xffe070); }
const wedges = new Map();
function drawWedges() {
  for (const F of fighters) {
    let m = wedges.get(F);
    if (!m) {
      // the fill, and an arc along its edge (the front, where blows are turned aside, reads from above)
      const color = F.isPlayer ? 0xd9b26a : 0x9fb4e0;
      m = new THREE.Mesh(new THREE.CircleGeometry(0.95, 20, -Math.PI / 4, Math.PI / 2), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
      m.edge = new THREE.Mesh(new THREE.RingGeometry(0.89, 0.95, 20, 1, -Math.PI / 4, Math.PI / 2), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }));
      m.renderOrder = m.edge.renderOrder = 2; m.add(m.edge); scene.add(m); wedges.set(F, m);
    }
    m.visible = !['out', 'down', 'fall'].includes(F.state);
    if (!m.visible) continue;
    const h = rig.heading(F.P);
    m.position.set(F.P[0].x, 0.02, F.P[0].z);
    m.quaternion.setFromEuler(new THREE.Euler(-Math.PI / 2, 0, h - Math.PI / 2, 'YXZ'));
    const ready = (F.state === 'free' || F.state === 'guard') && (F.posture ?? 1) > 0.25;
    const busy = fighters.some(E => !E.isPlayer && E.state === 'attack' && !(E.attack?.followed) && flat(E.P[0]).distanceTo(flat(F.P[0])) < 2.5) ? 0.35 : 1; // (a blow coming: the fans step back)
    m.material.opacity = (ready ? 0.35 : 0.08) * busy;
    m.edge.material.opacity = (ready ? 0.7 : 0.2) * busy;
  }
  for (const [F, m] of wedges) if (!fighters.includes(F)) { scene.remove(m); wedges.delete(F); }
}
const windRings = new Map();
function drawWindups() {
  for (const E of fighters) {
    if (E.isPlayer) continue;
    const winding = E.state === 'attack' && !(E.attack?.followed) && E.state !== 'out';
    const t = winding ? clock - (E.attack?.t0 ?? E.windAt ?? clock) : 0;
    const switching = clock - (E.stanceAt ?? -9) < 0.22 && (E.state === 'free' || E.state === 'guard'); // (a stance switch: a short glow)
    E.body.tint += ((winding && E.charge ? 0.35 : switching ? 0.4 : 0) - E.body.tint) * 0.35;
    E.body.limb = null; E.body.limbGlow = 0; // (no glowing hand: the motion shows the blow coming)
    let m = windRings.get(E);
    if (!m) { m = new THREE.Mesh(new THREE.RingGeometry(0.85, 0.95, 40), new THREE.MeshBasicMaterial({ color: 0xffe070, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthTest: false })); m.rotation.x = -Math.PI / 2; m.renderOrder = 14; scene.add(m); windRings.set(E, m); }
    m.visible = winding;
    if (!winding) continue;
    const p = Math.min(1, t / 0.65); // (closing over the wind-up: small at the moment the blow arrives)
    m.position.set(E.P[0].x, 0.03, E.P[0].z);
    m.scale.setScalar(1 - 0.6 * p);
    m.material.color.set((E.attack?.kind ?? E.windKind) === 'kick' || (E.attack?.kind ?? E.windKind) === 'knee' ? 0xff9a3c : 0xffe070);
    m.material.opacity = 0.5 + 0.5 * p;
  }
  for (const [E, m] of windRings) if (!fighters.includes(E)) { scene.remove(m); windRings.delete(E); }
}
/**
 * Where a blow is coming from, readable from a distance: from the moment an opponent commits (driving in, the
 * wind-up, a charge's load) until its blow lands or is over, a narrow wedge on the floor points from it at the
 * player - faint while it closes in, brighter through the wind-up, full as the blow is thrown.
 */
const laneGeo = (() => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.2, 0, 0, 0.2, 0, 0, 0, 0, 1]), 3)); g.setIndex([0, 2, 1]); return g; })();
const lanes = new Map();
function drawLanes() {
  for (const E of fighters) {
    if (E.isPlayer) continue;
    let m = lanes.get(E);
    const B = E.brain, A = E.antic;
    let a = 0;
    if (player && E.state !== 'out' && !['down', 'fall', 'getup'].includes(player.state)) {
      if (E.state === 'attack' && !(E.attack?.followed) && !(E.attack?.landed)) a = 0.85; // the blow under way
      else if (A?.big && !A.until) a = 0.35 + 0.45 * Math.min(1, (clock - A.t0) / 0.3); // the wind-up
      else if (B && clock < (B.lungeUntil ?? 0)) a = 0.3; // driving in
    }
    if (!m && a > 0) { m = new THREE.Mesh(laneGeo, new THREE.MeshBasicMaterial({ color: 0xff5a28, transparent: true, opacity: 0, side: THREE.DoubleSide, depthTest: false })); m.renderOrder = 13; scene.add(m); lanes.set(E, m); }
    if (!m) continue;
    // (eased: no flicker)
    m.userData.a = (m.userData.a ?? 0) + (a - (m.userData.a ?? 0)) * (a > (m.userData.a ?? 0) ? 0.5 : 0.25);
    m.visible = m.userData.a > 0.03;
    if (!m.visible) continue;
    const from = flat(E.P[0]), to = flat(player.P[0]).sub(from), len = Math.max(0.2, to.length() - 0.25);
    a *= clamp((to.length() - 1.0) / 0.5, 0, 1); // (close in, the body and its ring say it: the lane is for blows coming from a distance)
    m.position.set(from.x, 0.025, from.z);
    m.rotation.set(0, Math.atan2(to.x, to.z), 0);
    m.scale.set(1, 1, len);
    m.material.opacity = 0.75 * m.userData.a;
  }
  for (const [E, m] of lanes) if (!fighters.includes(E)) { scene.remove(m); m.material.dispose(); lanes.delete(E); }
}
const warnArcs = new Map();
function drawWarnings() {
  for (const E of fighters) {
    if (E.isPlayer) continue;
    let m = warnArcs.get(E);
    const hot = player && (clock < (E.warnUntil ?? 0) || (E.state === 'attack' && E.attack && !E.attack.followed && !inView(player, E))) && E.state !== 'out';
    const lurk = player && !hot && ['free', 'attack', 'guard'].includes(E.state) && !inView(player, E) && flat(E.P[0]).distanceTo(flat(player.P[0])) < 2.6;
    const on = hot || lurk;
    if (!m && on) {
      m = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.75, 24, 1, -0.5, 1.0), new THREE.MeshBasicMaterial({ color: 0xff3020, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthTest: false }));
      m.rotation.x = -Math.PI / 2; m.renderOrder = 16; scene.add(m); warnArcs.set(E, m);
    }
    if (!m) continue;
    m.visible = !!on;
    if (!on) continue;
    m.material.opacity = hot ? 0.85 : 0.3;
    const to = flat(E.P[0].clone().sub(player.P[0]));
    m.position.set(player.P[0].x, 0.03, player.P[0].z);
    m.rotation.z = Math.atan2(to.x, to.z) - Math.PI / 2; // (the arc's middle toward the opponent)
    m.material.opacity = 0.5 + 0.4 * Math.sin(clock * 30);
  }
  for (const [E, m] of warnArcs) if (!fighters.includes(E)) { scene.remove(m); warnArcs.delete(E); }
}
function drawLook() {
  drawWarnings();
  drawWedges();
  drawWindups();
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
  if (closing < 2.6) return; // (a real charge, not walking up to it)
  const T = A.spring?.target; // and only driving through it: where the body is going lies beyond the one hit
  if (!T || flat(T.clone().sub(A.P[0])).dot(n) < len + 0.4) return;
  A.slamAfter = clock + 1.0;
  const at = A.P[I.Chest].clone().lerp(B.P[I.Spine2], 0.5);
  land(A, B, Math.min(90, 0.5 * 15 * closing * closing * (A.state === 'attack' ? 0.5 : 1)), n, at, false);
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
  const R = F.rag;
  const dead = !!F.ko && !F.isPlayer, lying = F.state === 'down' || F.state === 'out';
  // knocked out: from the blow on, the ragdoll alone - the strings cut (muscles from 12 % down to 5 % over 0.6 s),
  // under gravity, the hips free, the blow's momentum carried (little damping) - it buckles, falls, rolls and slides
  // until the floor's rub stops it (mean speed under 6 cm/s for half a second, after at least 0.8 s); then it lies
  // as it came to rest
  if (dead) {
    if (F.restPose && (F.groundJolt ?? -9) > F.restAt) { F.restPose = null; F.restAt = null; F.stillFor = 0; } // (kicked again: it moves again)
    if (F.restPose) { F.shown = clonePose(F.restPose); return; }
    if (!F.ragOn || !F.limp) { if (!F.ragOn) { R.reset(F.shown); R.last = null; } F.ragOn = true; F.limp = clock; F.ragW = 1; F.ragFade = false; }
    const tl = clock - F.limp;
    R.gravity = true; R.friction = 0.35; R.dampScale = 0.25; R.k = F.ragK.map(k => k * (0.12 - 0.07 * clamp(tl / 0.6, 0, 1))); R.rootK = 0;
    R.last = null; // (no motion velocity to carry: the parts only slow down - air, the floor's rub)
    const anim = clonePose(F.shown); anim.root.copy(R.x[0]); // (the lying shape, carried with the body: only a faint pull toward it)
    R.step(dt, anim, [0, 0, 0, 0], null);
    F.shown = R.pose(anim);
    const sp = R.v.reduce((a, v) => a + v.length(), 0) / R.v.length;
    F.stillFor = sp < 0.06 ? (F.stillFor ?? 0) + dt : 0;
    if (F.stillFor > 0.5 && tl > 0.8) { F.restPose = clonePose(F.shown); F.restAt = clock; }
    return;
  }
  if (!F.ragOn) return;
  F.ragT += dt;
  const down = F.state === 'fall';
  const kicked = lying && clock - (F.groundJolt ?? -9) < 1.0;
  R.gravity = lying || down; R.friction = lying ? 0.2 : 0; R.dampScale = 1;
  // muscles firm up again (not while going down: the body stays loose until it lies)
  const u = down ? 0 : clamp(F.ragT / F.ragRecover, 0, 1);
  const loose = F.ragLoose + (1 - F.ragLoose) * u * u;
  R.k = F.ragK.map(k => k * (down ? Math.max(F.ragLoose, 0.25) : loose));
  R.rootK = kicked ? 60 + 2440 * ((clock - F.groundJolt) / 1.0) ** 2 : down || F.state === 'down' || F.state === 'getup' ? 2500 : 600 + 2400 * u;
  const contact = F.ragFeetFree || down ? [0, 0, 0, 0] : null;
  // (the ragdoll keeps its own state: a reset, should it drift far from the motion, is not shown as a jump)
  const before = R.x[0].clone();
  R.step(dt, F.shown, contact, null);
  const ragged = R.pose(F.shown);
  const drift = ragged.root.distanceTo(F.shown.root);
  if (R.x[0].distanceTo(before) > 0.3 || drift > (kicked ? 0.9 : 0.45)) F.ragFade = true;
  // (kicked on the floor: the motion is moved along to where the body has gone - no pull back to where it lay)
  if (kicked) { const d = flat(R.x[0].clone().sub(F.shown.root)).multiplyScalar(1 - Math.exp(-dt / 0.2)); F.track.shift(d.x, d.z); }
  // back to the motion once firm again; on the floor (lying, stirring) the motion's own; drifting off: at once
  if (F.ragFade) F.ragW = Math.max(0, F.ragW - dt / 0.12);
  else if (kicked) F.ragW = clock - F.groundJolt < 0.5 ? 1 : Math.max(0.35, F.ragW - dt / 0.5); // (all ragdoll while the blow plays out)
  else if (F.state === 'down') F.ragW = Math.max(0.35, F.ragW - dt / 0.4);
  else if (F.state === 'getup') F.ragW = Math.max(0, F.ragW - dt / 0.4);
  else if (!down && u >= 1) F.ragW = Math.max(0, F.ragW - dt / 0.2);
  if (F.ragW <= 0) { F.ragOn = false; F.ragFade = false; return; }
  F.shown = lerpPose(F.shown, ragged, F.ragW);
}
/** A right press, taken within 0.3 s when the body can strike (after a hit, getting up: dropped). */
function takeStrikeInput() {
  const P = player, inp = P?.strikeInput;
  if (!inp) return;
  const t = typeof inp === 'number' ? inp : inp.t, kind = typeof inp === 'number' ? null : inp.kind, id = inp.id;
  const keep = Math.max(0.3, (P.stun ?? 0) - t + 0.15); // (a press while reeling waits for the end of it)
  if (clock - t > keep || P.state === 'down' || P.state === 'fall' || P.state === 'getup') { if (clock - t > keep) { P.strikeInput = null; plog?.pressResult(id, `expired:${P.state}`); } return; }
  // reeling from a light blow: after a quarter of a second the player can hit back (a press is not lost in a trade)
  if (P.state === 'hit') { if (clock - (P.hitAt ?? 0) > 0.2) P.state = 'free'; else return; }
  P.strikeInput = null;
  // (a click near the locked-on opponent, off its body: the blow goes to it, not to whoever is nearest)
  if (inp.aim && inp.aim.enemy.state !== 'out' && !P.aim) { P.aim = inp.aim; P.aimAt = clock; }
  plog?.pressResult(id, strike(P, kind, id));
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), clock = 0, fpsAcc = 0, fpsN = 0, fps = 0;
/**
 * Where the time goes, per second of real time (to the play log as 'perf', and to the debug footer): frames drawn,
 * simulation steps, generation runs and their GPU time, the main thread's milliseconds in the simulation, in
 * drawing, in turning results into motion and in packing requests, and the frame intervals (the longest, the 95th
 * percentile, how many over 25 ms) - so a stutter can be told from the slow motion that is meant.
 */
const perf = { t0: 0, frames: 0, steps: 0, gen: 0, genMs: 0, sim: 0, simMax: 0, draw: 0, drawMax: 0, res: 0, resMax: 0, req: 0, reqMax: 0, gaps: [], slow: 0, last: null };
function perfFlush(now) {
  const sec = (now - perf.t0) / 1000;
  if (!perf.t0 || sec < 1) { if (!perf.t0) perf.t0 = now; return; }
  const g = perf.gaps.sort((a, b) => a - b), n = Math.max(1, perf.frames), r1 = x => +x.toFixed(1);
  perf.last = {
    fps: r1(perf.frames / sec), simHz: r1(perf.steps / sec), genHz: r1(perf.gen / sec), genMs: Math.round(perf.genMs / Math.max(1, perf.gen)),
    sim: r1(perf.sim / n), simMax: r1(perf.simMax), draw: r1(perf.draw / n), drawMax: r1(perf.drawMax),
    res: r1(perf.res / sec), resMax: r1(perf.resMax), req: r1(perf.req / sec), reqMax: r1(perf.reqMax),
    gap95: r1(g[Math.floor(g.length * 0.95)] ?? 0), gapMax: r1(g[g.length - 1] ?? 0), long: g.filter(x => x > 25).length,
    slow: +(perf.slow / sec).toFixed(2), fighters: fighters.filter(f => f.state !== 'out').length,
  };
  if (started && document.visibilityState === 'visible') plog?.note('perf', perf.last);
  Object.assign(perf, { t0: now, frames: 0, steps: 0, gen: 0, genMs: 0, sim: 0, simMax: 0, draw: 0, drawMax: 0, res: 0, resMax: 0, req: 0, reqMax: 0, gaps: [], slow: 0 });
}
const camTarget = new Vector3(0, 0.9, 0), camSide = new Vector3(1, 0, 0), camLean = new Vector3();
let camDist = 4.6, camRaise = 0;
function simulate(dt) {
  if (!db) return;
  if (settings.trigger !== 'auto') takeStrikeInput();
  // (a landed blow: time runs at 15 % for a moment - see frame(); the struck body's ragdoll keeps reeling at more)
  const alive = fighters.filter(f => !f.isPlayer && f.state !== 'out').length;
  // the next ones come before the last is gone: down to one standing, another within a second; below the full
  // number, one more every 4-6 s (two or three at a time, not always three)
  const standing = fighters.filter(f => !f.isPlayer && f.state !== 'out' && !f.ko).length;
  if (tutorial && kos >= 1) { tutorial = false; spawnAt = clock + 0.9; arrivals = settings.enemies - 1; }
  if (player && standing <= 1 && !tutorial) spawnAt = Math.min(spawnAt, clock + 1.0);
  if (player && alive < (tutorial ? 1 : settings.enemies) && clock > spawnAt) { spawn(); spawnAt = clock + (arrivals > 0 ? (arrivals--, 0.35) : standing + 1 <= 1 ? 1.0 : rand(4, 6)); }
  // the player's leash on the hips
  if (player) {
    const free = player.state === 'free' || player.state === 'attack';
    { const a = aimFromCursor(); if (a) { player.aim = a; player.aimAt = clock; } else if (clock - (player.aimAt ?? -9) > 0.25 || player.aim?.enemy?.state === 'out') player.aim = null; }
    let goal = scripted ? scripted.point : player.dodge && clock < player.dodge.until && player.dodge.goal ? player.dodge.goal.clone() : cursorFloor();
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
    if (player.aim) { player.lookAt = player.aim.enemy.P[player.aim.joint].clone().add(player.aim.off); player.lockOn = player.aim.enemy; player.lockUntil = clock + 1.0; } // on what the cursor is on
    else if (player.lockOn && player.lockOn.state !== 'out') {
      // the lock holds while the cursor stays near the locked opponent and it is close (circling it, the cursor on
      // the floor beside it): only a far pull, or the cursor on another opponent, lets go of it
      const L = player.lockOn, c = cursorFloor();
      if (c && flat(c).distanceTo(flat(L.P[0])) < 1.5 && flat(L.P[0]).distanceTo(hips) < 2.5) player.lockUntil = Math.max(player.lockUntil ?? 0, clock + 0.6);
      if (clock < player.lockUntil) player.lookAt = L.P[I.Head].clone(); // (still facing it: circling round it)
    }
    const reeling = player.state === 'hit' && player.stun && clock < player.stun;
    player.spring = { target: goal && (free || reeling) ? goal.clone() : null };
    // a dodge: a quick flick of the cursor to the side (or back) while an opponent close by is winding up
    {
      const c = cursorFloor();
      if (c && player.cPrev) { player.cVel = (player.cVel ?? new Vector3()).lerp(flat(c.clone().sub(player.cPrev)).divideScalar(dt), 0.5); }
      player.cPrev = c ? c.clone() : null;
      // the threat: of the opponents striking close by, the one whose blow is nearest to landing (close, and
      // longest under way) - not the first in the list
      let threat = null, threatScore = Infinity;
      for (const E of enemies()) {
        if (E.state !== 'attack' || E.attack?.followed) continue;
        const dE = flat(E.P[0]).distanceTo(flat(player.P[0]));
        if (dE >= 2.2) continue;
        const s = dE - (clock - (E.attack?.t0 ?? E.windAt ?? clock));
        if (s < threatScore) { threatScore = s; threat = E; }
      }
      if (threat && (player.state === 'free' || player.state === 'attack') && clock > (player.dodge?.at ?? -9) + 0.45) {
        // the pointer flicked on the screen (the last 0.15 s): sideways to the line to the attacker, or back from it
        const r = canvas.getBoundingClientRect(), scr = v => { const q = v.clone().project(camera); return [(q.x + 1) / 2 * r.width, (1 - q.y) / 2 * r.height]; };
        const a = flickTrail[0], b = flickTrail[flickTrail.length - 1];
        let flick = null;
        if (a && b && b.t - a.t > 30) {
          const sec = (b.t - a.t) / 1000, fx = b.x - a.x, fy = b.y - a.y, len = Math.hypot(fx, fy);
          const ps = scr(player.P[0]), ts = scr(threat.P[0]);
          let ax = ts[0] - ps[0], ay = ts[1] - ps[1]; const al = Math.hypot(ax, ay) || 1; ax /= al; ay /= al;
          const par = (fx * ax + fy * ay) / sec, perp = Math.abs(fx * ay - fy * ax) / sec;
          if (len > 25 && len < 420 && ((perp > 900 && Math.abs(par) < 0.9 * perp) || par < -900)) flick = { x: fx, y: fy };
        }
        // or the body itself already moving out across the line fast
        const along = flat(threat.P[0].clone().sub(player.P[0])).normalize(), hv = player.prevP ? flat(player.P[0].clone().sub(player.prevP[0])).divideScalar(dt) : new Vector3();
        const across = hv.clone().addScaledVector(along, -hv.dot(along)).length();
        if (flick || across > 2.0) {
          // (a flick is a dodge: i-frames and a counter after it; the body just moving out fast only spoils the aim)
          player.dodge = { at: clock, until: clock + (flick ? 0.4 : 0.15), iframes: flick ? clock + 0.18 : clock, flick: !!flick, goal: player.spring?.target?.clone() ?? null };
          if (flick) {
            // the burst of the step: the screen flick turned into the floor direction (sideways from the attacker)
            const side = new Vector3(along.z, 0, -along.x), sgn = Math.sign((flick.x * (scr(player.P[0].clone().add(side))[0] - scr(player.P[0])[0])) + (flick.y * (scr(player.P[0].clone().add(side))[1] - scr(player.P[0])[1]))) || 1;
            player.vel.addScaledVector(side.clone().multiplyScalar(sgn), 2.2).addScaledVector(along, 0.2);
            // and round: the step goes to a point beside the attacker (a metre off its side, level with it), so the
            // blow passes and the player comes out on its flank - the dogfight's turn
            player.dodge.goal = flat(threat.P[0]).addScaledVector(side.multiplyScalar(sgn), 0.95).addScaledVector(along, -0.15);
            // and the eyes stay on the one dodged: the next click is a blow at it
            player.lockOn = threat; player.lockUntil = Math.max(player.lockUntil ?? 0, clock + 0.6);
          }
          plog?.note('dodge', { from: threat.id, flick: !!flick });
        }
      }
    }
    {
      const tg = player.spring.target, pull = tg ? flat(tg.clone().sub(player.P[0])) : null;
      if (!pull || pull.length() < 0.3 || player.vel.length() > 1.4) player.setOffArmed = true;
      if (player.setOffArmed && pull && pull.length() > 0.5 && player.vel.length() < 1.0 && player.state === 'free') {
        player.setOffArmed = false; player.setOff = { t0: clock, dir: pull.normalize() }; if (!player.track.busy) player.planTimer = 0;
      }
    }
    // the spring stiffens with the mouse's speed (gain scheduling of a second-order system: natural frequency
    // w = sqrt(k), damping ratio z = c / 2w, settling in about 4 / (z w)). Slow: the soft pendulum as before (w 3.7,
    // z 0.35: it swings back); fast: stiff and nearly critically damped (w up to 8.6, z 0.7: it settles in about
    // 0.7 s instead of 3, a little overshoot left). The speed is taken at once and let go over 0.3 s.
    {
      const sNow = clamp(((player.cVel ? flat(player.cVel).length() : 0) - 0.5) / 2.5, 0, 1);
      player.stiff = (player.stiff ?? 0) + (sNow - (player.stiff ?? 0)) * (1 - Math.exp(-dt / (sNow > (player.stiff ?? 0) ? 0.04 : 0.3)));
    }
    const w0 = Math.sqrt(settings.spring.k), z0 = settings.spring.c / (2 * w0), wS = w0 * (1 + 1.3 * player.stiff), zS = z0 + (0.7 - z0) * player.stiff;
    const fl = feelOf(player) ?? {}, sched = { k: (fl.k ?? 1) * (wS * wS) / settings.spring.k, c: (fl.c ?? 1) * (2 * zS * wS) / settings.spring.c, vmax: fl.vmax ?? 1 };
    // (the spring measured from the hips the generated motion has, not the lead shown on top: no loop through the lead)
    springStep(player.vel, flat(player.P[0]).sub(player.leadOff ?? ZERO), player.spring.target, dt, sched);
    // the cursor's own speed fed straight in (half of it, within 0.08 s) while it moves on ahead of the hips: the body
    // moves the moment the mouse does, linearly, and the spring does the rest (its swing back when the mouse stops)
    if (player.state === 'free' && player.spring.target && player.cVel && !(player.dash && clock < player.dash.until)) {
      const cv = flat(player.cVel).clampLength(0, settings.vmax), ahead = flat(player.spring.target.clone().sub(player.P[0]));
      if (cv.length() > 0.4 && cv.dot(ahead) > 0 && ahead.length() > 0.25) player.vel.lerp(cv, (1 - Math.exp(-dt / 0.08)) * 0.35);
    }
    if (!free) player.vel.multiplyScalar(Math.exp(-dt * 4));
    // striking: the spring's speed follows what the body really does (no speed stored up for a lurch afterwards)
    if (player.state === 'attack' && player.prevP) player.vel.lerp(flat(player.P[0].clone().sub(player.prevP[0])).divideScalar(dt), 1 - Math.exp(-dt * 6));
    player.want.copy(player.vel);
    if (settings.trigger === 'auto') { flickStrike(); autoStrike(player); }
  }
  for (const F of fighters) {
    if (F.broken && clock >= F.broken) { F.broken = 0; F.posture = 0.4; if (!F.isPlayer) F.exposed = clock + 0.3; }
    else if (!F.broken && (F.posture ?? 1) < 1 && clock - (F.postureAt ?? -9) > 1.2) F.posture = Math.min(1, F.posture + (F.isPlayer ? 0.35 : 0.25) * dt * (F.state === 'attack' ? 0.5 : 1));
    if (F.isPlayer && F.state === 'free' && clock - (F.hitAt ?? -9) > 3 && F.hp < F.maxHp) { F.hp = Math.min(F.maxHp, F.hp + settings.regen * dt); if (Math.random() < 0.05) hudFight(); }
    if (!F.isPlayer) {
      // the stance: guard high (hands by the face) or low (crouched, hands at the body), changing now and then
      // (held at least 3 s, so it can be read and used; the switch itself shows: a short glow and an overshoot of the stance)
      if (clock > (F.stanceUntil ?? 0)) { if (F.stance) F.stanceAt = clock; F.stance = F.stance === 'low' ? 'high' : (F.stance ? 'low' : 'high'); F.stanceUntil = clock + rand(3.0, 4.5); }
      // covering up (after two quick blows): a guard, stepping back
      if (F.coverAt && clock > F.coverAt) {
        if (F.state === 'free' || F.state === 'attack') {
          F.coverAt = null; F.counterAt = null;
          F.track.request(guardReq(F, player, Math.random() < 0.6 ? 'high' : 'low'));
          F.state = 'guard'; F.until = clock + 0.9; F.guard = { kind: 'high', high: true, until: clock + 0.9 };
          if (F.brain) { F.brain.evadeUntil = clock + 0.5; F.brain.evDir = Math.random() < 0.5 ? -1 : 1; }
        } else if (clock > F.coverAt + 1) F.coverAt = null;
      }
      // bursting back: a short wind-up, then a strike that light blows do not stop (it cuts into the player's combination)
      if (F.burstAt && clock > F.burstAt) {
        const pd = ['down', 'fall', 'getup'].includes(player.state);
        if (!pd && (F.state === 'free' || (F.state === 'hit' && !F.track.busy)) && F.brain && flat(player.P[0].clone().sub(F.P[0])).length() < 1.6) {
          F.burstAt = null; F.counterAt = null; F.coverAt = null;
          F.state = 'free'; F.brain.burst = true;
          const kind = Math.random() < 0.6 ? 'punch' : 'kick';
          F.brain.tellAt = clock + 0.18; F.brain.tellKind = kind; F.brain.nextAttack = clock; F.brain.evadeUntil = 0;
          F.antic = { t0: clock, side: Math.random() < 0.5 ? 1 : -1, big: true, fast: true, kind };
          plog?.note('burst', { who: F.id });
        } else if (clock > F.burstAt + 0.8) F.burstAt = null;
      }
      if (F.counterAt && clock > F.counterAt) {
        F.counterAt = null;
        const d = flat(player.P[0].clone().sub(F.P[0])).length();
        if (F.brain) { if (d < 1.5) F.brain.nextAttack = clock; else F.brain.lungeUntil = clock + 0.3; }
      }
      think(F, dt);
      springStep(F.vel, flat(F.P[0]), F.spring?.target ?? null, dt, feelOf(F));
      if (!FIGHT.includes(F.state) || F.state === 'hit') F.vel.multiplyScalar(Math.exp(-dt * 4));
      if (F.state === 'attack' && F.prevP) F.vel.lerp(flat(F.P[0].clone().sub(F.prevP[0])).divideScalar(dt), 1 - Math.exp(-dt * 6));
      F.want.copy(F.vel);
    }
    // the player on the floor: a click gets it up at once, and it is never left there more than 1.2 s without one
    if (F.isPlayer && F.state === 'down') { F.downAt ??= clock; if (F.getupNow || clock - F.downAt > 1.2) F.until = Math.min(F.until, clock); }
    else if (F.isPlayer) { F.downAt = null; F.getupNow = false; }
    // states end
    if (F.state !== 'free' && F.state !== 'out' && (clock > F.until || (F.state === 'down' && !F.ko && F.until - clock < scheduler.latency + 0.1)) && !F.track.busy) {
      if (F.state === 'down') {
        if (F.ko && !F.isPlayer) { F.state = 'out'; F.outAt = clock; kos++; hudFight(); }
        else { F.track.request(getupReq(F)); F.state = 'getup'; F.until = clock + 5; if (F.isPlayer && F.hp <= 0) F.hp = F.maxHp; }
      } else if (F.state === 'attack' && F.attackQueued != null && clock - F.attackQueued < 0.8) { F.attackQueued = null; F.state = 'free'; const r = strike(F, F.queuedKind, F.queuedPress); if (F.isPlayer) plog?.pressResult(F.queuedPress, `after:${r}`); }
      else if (F.state !== 'down') { F.state = 'free'; F.guard = null; F.charge = null; }
    }
    // a strike carries straight on into its follow-through (asked for as the top comes near)
    const paceNow = F.track.frames[Math.min(F.track.frames.length - 1, Math.floor(F.track.t))].rate;
    if (F.state === 'attack' && F.attack?.spliced && !F.attack.followed && !F.track.busy && F.track.timeLeft() < scheduler.latency + 0.12) {
      F.attack.followed = true;
      if (F.attackQueued != null && clock - F.attackQueued < 1.2) { F.attackQueued = null; F.attack.chained = true; F.track.request(strikeReq(F, null, 'end', F.queuedKind, F.queuedPress)); if (F.isPlayer) plog?.pressResult(F.queuedPress, 'chained'); F.until = clock + 1.2; }
      else if (F.attackQueued != null && F.isPlayer) { plog?.pressResult(F.queuedPress, 'lost:late'); F.attackQueued = null; }
      else F.track.request(followReq(F));
    }
    // a strike whose motion has run out with nothing asked for after it: free again at once (never a body held still
    // in its last frame, waiting for the strike's time to pass)
    if (F.state === 'attack' && !F.charge && !F.track.busy && F.track.remaining < 1 && clock - (F.windAt ?? -9) > 0.3) { F.state = 'free'; F.planTimer = 0; }
    // plans while free
    F.planTimer -= dt;
    // the player is not held by its own follow-through or a light blow: pulled, it moves on (close in, nothing
    // should feel stuck)
    if (F.isPlayer && F.spring?.target && F.state === 'attack' && F.attack?.followSpliced && clock - F.attack.followSpliced > 0.08) {
      F.state = 'free'; F.planTimer = 0;
    }
    // (and a light blow holds it a third of a second at most, pulling or not: the reaction plays on under whatever
    // it does next)
    if (F.isPlayer && F.state === 'hit' && clock - (F.hitAt ?? 0) > 0.3 && clock > (F.stun ?? 0)) { F.state = 'free'; F.planTimer = 0; }
    if (F.state === 'free' && scheduler.ready && !F.track.busy) {
      const active = F.want.length() > 0.05 || F.vel.length() > 0.05;
      // (the player's speed changes all the time on its spring: only grabbing and letting go count as a change)
      const pulling = !!F.spring?.target;
      const changed = F.isPlayer ? pulling !== !!F.wasPulling : F.want.distanceTo(F.lastWant) > 0.4;
      F.wasPulling = pulling;
      if (active && (F.planTimer <= 0 || changed)) { F.track.request(planReq(F, 'move')); F.planTimer = settings.replan * (F.isPlayer ? 1 : F.mode === 'wait' || F.mode === 'hold' ? 2 : 1.5); F.lastWant.copy(F.want); }
      else if (!active && F.track.remaining / FPS < scheduler.latency + 0.25) F.track.request(planReq(F, 'move', 'end'));
    }
    slideOn(F); shoveOn(F); trackIn(F, dt);
    const r0 = F.isPlayer ? F.track.display.root.clone() : null;
    F.track.advance(dt);
    if (F.isPlayer) leadOn(F, dt, r0);
  }
  { const t = performance.now(); scheduler.tick(fighters.filter(f => f.state !== 'out').map(f => f.track)); const ms = performance.now() - t; perf.req += ms; perf.reqMax = Math.max(perf.reqMax, ms); }
  for (const F of fighters) {
    F.prevP = F.P;
    F.shown = clonePose(F.track.display);
    ragStep(F, hitstop > 0 ? dt * 4 : dt);
    knockOn(F);
    balanceLayer(F, dt);
    hitPush(F);
    breathe(F);
    footLock(F);
    leadLayer(F);
    anticipate(F);
    specStrike(F);
    jumpLayer(F);
    settleLayer(F);
    setOffLean(F);
    aliveLayer(F, dt);
    homing(F);
    F.P = rig.positions(F.shown);
    // (the play log: how far the struck part moved 0.2 s after a blow)
    if (F.dispProbe && clock - F.dispProbe.t0 >= 0.2) { const D = F.dispProbe; plog?.note('partDisp', { who: F.isPlayer ? 'P' : F.id, part: D.part, disp: +F.P[D.joint].distanceTo(D.p0).toFixed(3), energy: +D.energy.toFixed(1) }); F.dispProbe = null; }
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
  for (let k = fighters.length - 1; k >= 0; k--) { const F = fighters[k]; if (F.state === 'out' && ((F.restAt != null && clock - Math.max(F.restAt, F.outAt) > 1.5) || clock - F.outAt > 6)) { F.remove(); fighters.splice(k, 1); } }
  clock += dt;
  plog?.step(dt);
  window.brawl?.onStep?.(dt);
}

function frame(now) {
  if (now - last > 150 && now - last < 10000 && player && document.visibilityState === 'visible') plog?.note('hitch', { ms: Math.round(now - last), fighters: fighters.length });
  const real = Math.min(0.1, (now - last) / 1000);
  if (started) { perf.frames++; perf.gaps.push(now - last); if (hitstop > 0 || now < slowUntil) perf.slow += real; }
  if (hitstop > 0) hitstop -= real;
  hurt = Math.max(0, hurt - real / 0.35);
  const dt = started ? real * (hitstop > 0 ? 0.15 : now < slowUntil ? 0.35 : 1) : 0;
  last = now;
  acc += dt;
  const tS = performance.now();
  while (acc >= STEP) { simulate(STEP); acc -= STEP; perf.steps++; }
  const tD = performance.now(); perf.sim += tD - tS; perf.simMax = Math.max(perf.simMax, tD - tS);
  // (drawn where the bodies are between the last two steps: the steps run at 60 Hz whatever the screen does)
  const alpha = clamp(acc / STEP, 0, 1);
  for (const F of fighters) {
    F.R = F.prevP && F.prevP.length === F.P.length ? F.P.map((v, j) => F.prevP[j].clone().lerp(v, alpha)) : F.P;
    F.body.update(F.R, rig.extraPositions(F.shown, F.R), dt);
  }
  drawLook();
  for (let k = dusts.length - 1; k >= 0; k--) { const D = dusts[k]; D.t += dt; D.m.scale.setScalar(1 + D.t * 9); D.m.material.opacity = Math.max(0, 0.8 - D.t * 3.2); if (D.t > 0.25) { scene.remove(D.m); D.m.geometry.dispose(); D.m.material.dispose(); dusts.splice(k, 1); } }
  for (let k = sparks.length - 1; k >= 0; k--) { const s = sparks[k]; s.t += dt; s.m.quaternion.copy(camera.quaternion); s.m.scale.setScalar(s.s * (1 + 6 * s.t)); s.m.material.opacity = Math.max(0, 0.9 - 3 * s.t); if (s.t > 0.3) { scene.remove(s.m); s.m.material.dispose(); sparks.splice(k, 1); } }
  // camera: from the side of the line between the player and the nearest opponent, far enough to see both
  if (player) {
    // (the camera's opponent is the one the player is locked on - it does not swing to whoever is nearest while circling)
    const o = player.lockOn && player.lockOn.state !== 'out' && clock < (player.lockUntil ?? 0) ? player.lockOn : player.opponent;
    // everyone close in (within 3.5 m) framed, the one striking a little toward the middle
    // (each one eases into the framing as it comes near - over about a second - and out as it goes away or is
    // knocked out: a new opponent never jumps the view)
    let wsum = 1;
    const mid = player.R[0].clone();
    for (const f of fighters) {
      if (f.isPlayer) continue;
      const dd = flat(f.R[0]).distanceTo(flat(player.R[0]));
      const want = f.state === 'out' ? 0 : clamp((4.5 - dd) / 1.5, 0, 1);
      f.camW = (f.camW ?? 0) + (want - (f.camW ?? 0)) * (1 - Math.exp(-dt / 0.9));
      if (f.camW > 1e-3) { mid.addScaledVector(f.R[0], f.camW); wsum += f.camW; }
    }
    mid.divideScalar(wsum);
    const close = fighters.filter(f => !f.isPlayer && (f.camW ?? 0) > 0.05);
    const striking = close.find(f => f.state === 'attack');
    camLean.lerp(striking ? flat(striking.P[0].clone().sub(mid)).multiplyScalar(0.2) : new Vector3(), 1 - Math.exp(-dt / 0.4));
    mid.add(camLean);
    let radius = flat(player.P[0]).distanceTo(flat(mid)); for (const f of close) radius = Math.max(radius, f.camW * flat(f.P[0]).distanceTo(flat(mid)));
    const line = o ? flat(o.P[0].clone().sub(player.P[0])) : new Vector3(0, 0, 1);
    const side = line.lengthSq() > 1e-4 ? new Vector3(line.z, 0, -line.x).normalize() : camSide.clone();
    if (side.dot(camSide) < 0) side.negate(); // keep the camera on its side
    // the camera turns slowly, and never faster than about 50 degrees a second (circling the opponent does not spin the view)
    { const a0 = Math.atan2(camSide.x, camSide.z), da = clamp(wrap(Math.atan2(side.x, side.z) - a0) * (1 - Math.exp(-dt * 0.4)), -0.6 * dt, 0.6 * dt); camSide.set(Math.sin(a0 + da), 0, Math.cos(a0 + da)); }
    camDist += (clamp(3.4 + 1.3 * radius, 4.0, 6.8) - camDist) * (1 - Math.exp(-dt / 1.2));
    const dist = camDist;
    camTarget.lerp(new Vector3(mid.x, 0.75, mid.z), 1 - Math.exp(-dt * 2));
    // from a little above (about 18 degrees down): from the side, so a blow travels across the screen; still enough to read flanks
    // (an opponent between the camera and the player would hide the exchange: then the camera rises a little, over it)
    {
      const eye = camTarget.clone().addScaledVector(camSide, dist + 0.6).setY(1.2), pl = player.P[0].clone().setY(1.1), ray = pl.clone().sub(eye), L = ray.length(); ray.normalize();
      let occ = 0;
      for (const f of fighters) { if (f.isPlayer || f.state === 'out') continue; const v = f.P[0].clone().setY(1.1).sub(eye), along = v.dot(ray); if (along > 0.3 && along < L - 0.3 && v.addScaledVector(ray, -along).length() < 0.45) occ = 1; }
      camRaise += (occ - camRaise) * (1 - Math.exp(-dt / 0.5));
    }
    camera.position.copy(camTarget).addScaledVector(camSide, dist + 0.6 - 0.6 * camRaise).add(new Vector3(0, clamp(0.33 * (dist + 0.6), 1.6, 2.6) + 1.3 * camRaise, 0));
    camera.lookAt(camTarget);
    if (shake > 0.001) { camera.position.add(new Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).multiplyScalar(shake)); shake *= Math.exp(-real * 12); }
    camera.position.add(camKick); camKick.multiplyScalar(Math.exp(-real / 0.08));
    hurtEl.style.opacity = hurt.toFixed(2);
    stage.followFloor(player.P[0]);
  }
  drawBones();
  drawLanes();
  stage.render();
  { const ms = performance.now() - tD; perf.draw += ms; perf.drawMax = Math.max(perf.drawMax, ms); }
  perfFlush(now);
  fpsAcc += real; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hudStats(); }
}
function loop(now) { frame(now); requestAnimationFrame(loop); }

// ---------------------------------------------------------------- the bone view (B key)
// What MotionBricks is asked for, drawn over the bodies as bones: each fighter's last request - its target poses
// (the first bright, the rest fainter: where the motion must pass) and the pose it starts from (grey; for a strike
// the wound-up body). Toggled with the B key; nothing is drawn otherwise.
let boneView = false;
const boneLines = new Map();
const BONE_SETS = 5; // targets (up to 4) + the start
function boneMesh(F) {
  let m = boneLines.get(F);
  if (m) return m;
  const segs = rig.parents.filter(p => p >= 0).length * BONE_SETS;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs * 6), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segs * 6), 3));
  m = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthTest: false }));
  m.renderOrder = 20; m.frustumCulled = false; scene.add(m); boneLines.set(F, m);
  return m;
}
function drawBones() {
  for (const [F, m] of boneLines) if (!boneView || !fighters.includes(F) || F.state === 'out') { m.visible = false; if (!fighters.includes(F)) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); boneLines.delete(F); } }
  if (!boneView) return;
  for (const F of fighters) {
    if (F.state === 'out' || !F.track.asked) continue;
    const m = boneMesh(F), pos = m.geometry.attributes.position.array, col = m.geometry.attributes.color.array;
    // a blow or a reaction (kept 1.2 s): the pose asked for at its top, bright, the pose it starts from (the wind-up)
    // grey, and a line from the limb now to where it is sent; otherwise only the last target of the move (faint):
    // the four targets of a move are a 30th of a second apart and lie on the body, so they told nothing
    const K = F.track.askedKey && performance.now() - F.track.askedKey.at < 1200 ? F.track.askedKey : null;
    const base = new THREE.Color(F.isPlayer ? 0xffd860 : 0x60e0ff);
    const sets = K ? [[K.from, new THREE.Color(0x8a8a8a)], [K.targets[0], base]] : [[F.track.asked.targets.at(-1), base.clone().multiplyScalar(0.45)]];
    let k = 0;
    for (const [pose, c] of sets) {
      const P = rig.positions(pose);
      rig.parents.forEach((p, j) => { if (p < 0) return; pos.set([P[p].x, P[p].y, P[p].z, P[j].x, P[j].y, P[j].z], k * 6); col.set([c.r, c.g, c.b, c.r, c.g, c.b], k * 6); k++; });
    }
    const part = K?.kind === 'attack' ? F.attack?.part : null;
    if (part && I[part] != null) {
      const a = (F.R ?? F.P)[I[part]], b = rig.positions(K.targets[0])[I[part]];
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], k * 6); col.set([1, 1, 1, 1, 1, 1], k * 6); k++;
    }
    m.geometry.setDrawRange(0, k * 2);
    m.geometry.attributes.position.needsUpdate = true; m.geometry.attributes.color.needsUpdate = true;
    m.visible = true;
  }
}
addEventListener('keydown', e => { if ((e.key === 'b' || e.key === 'B') && !e.repeat) { boneView = !boneView; drawBones(); if ($('help-bones')) $('help-bones').textContent = boneView ? 'B: target poses ON - bright: the pose asked of the model, grey: where it starts, white: where the limb is sent' : 'B: show target poses'; } });

function hudStats() {
  if (!scheduler.ready) return;
  const p = perf.last;
  $('stats').textContent = `${scheduler.backend} · draw ${fps.toFixed(0)} fps` + (p ? ` · sim ${p.simHz} Hz ${p.sim}/${p.simMax} ms · render ${p.draw}/${p.drawMax} ms · gen ${p.genHz}/s ${p.genMs} ms · results ${p.res} ms/s · gap95 ${p.gap95} max ${p.gapMax} ms` : '');
  hudFight();
}
function hudFight() {
  if (!player) return;
  $('hp-player').style.width = `${(player.hp / player.maxHp) * 100}%`;
  const o = player.lockOn && player.lockOn.state !== 'out' && clock < (player.lockUntil ?? 0) ? player.lockOn : player.opponent; // (the one looked at)
  $('hp-enemy').style.width = o ? `${(o.hp / o.maxHp) * 100}%` : '0%';
  $('kos').textContent = `${kos}`;
  if ($('kos-pips')) $('kos-pips').textContent = '●'.repeat(Math.min(kos, 30)); // (knock-outs as marks, no words)
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
    else if (data.id === 0) {
      scheduler.latency = Math.max(0.05, data.ms / 1000); scheduler.ready = true;
      if (started) $('overlay').classList.add('hidden');
      else { label.textContent = `${scheduler.backend} で準備完了`; $('go').hidden = false; $('go').focus(); }
    }
    else { const t = performance.now(); scheduler.onResult(data); const ms = performance.now() - t; perf.res += ms; perf.resMax = Math.max(perf.resMax, ms); perf.gen++; perf.genMs += data.ms ?? 0; }
  } else if (data.type === 'error') {
    if (data.id) scheduler.onFail(data.id);
    label.textContent = `エラー: ${data.message}`;
    $('start').disabled = false; $('start').hidden = false; $('start').textContent = 'もう一度読み込む';
  }
};
// (the small model only, loaded as the page opens; the button is there again only to retry after an error)
function loadModel() {
  const url = new URL(params.get('model') ?? MODELS.fp16, location.href).href;
  $('start').disabled = true; $('start').hidden = true;
  $('progress').classList.add('show');
  worker.postMessage({ type: 'load', url, ep: params.get('ep') ?? 'webgpu', batch: BATCH });
}
$('start').addEventListener('click', loadModel);
// the fight starts on the button (the model loads and warms up before; nothing runs or is logged until then)
function begin() {
  if (started) return;
  started = true; last = performance.now(); acc = 0;
  if (scheduler.ready) $('overlay').classList.add('hidden');
  plog = createPlayLog({ get player() { return player; }, get fighters() { return fighters; }, get clock() { return clock; }, mouse, get lastHit() { return lastHit; }, crew, cursorFloor, settings, I });
  if ($('play-log-info')) $('play-log-info').textContent = plog.local ? `記録中（20 秒ごとに eval_out/play_${plog.sid}_* へ）` : '記録中（ボタンで保存）';
}
$('go').addEventListener('click', begin);
loadModel();
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

// the play is recorded all along (saved to eval_out/ when served locally; the button downloads it)
$('play-log')?.addEventListener('click', async () => { if (!plog) return; if (plog.local) { await plog.save(); $('play-log-info').textContent = `保存した：eval_out/play_${plog.sid}_*`; } else await plog.download(); });
requestAnimationFrame(loop);
if (params.has('debug')) {
  if ($('stats-foot')) $('stats-foot').hidden = false;
  window.brawl = {
    rig, scheduler, settings, fighters, simulate, strike, land, crew, camera, get plog() { return plog; }, keys: { fallKeys, getupKeys, guardKeys },
    reset() {
      for (const f of fighters) if (!f.isPlayer) { f.state = 'out'; f.outAt = -9; }
      kos = 0; spawnAt = clock; crew.watchUntil = 0; tutorial = false; arrivals = 0;
      if (player) { player.hp = player.maxHp; player.stun = 0; }
    },
    get player() { return player; }, jumps, get db() { return db; }, get clock() { return clock; }, get kos() { return kos; }, get lastHit() { return lastHit; },
    pull(point) { scripted = { point: point ?? null }; },
    unscript() { scripted = null; },
    aim(a) { aimOverride = a; },
    draw: () => frame(performance.now()),
    tune: TUNE,
    get perf() { return perf.last; },
    start: begin,
    onStep: null,
  };
}
