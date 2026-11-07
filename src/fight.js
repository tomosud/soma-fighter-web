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
import { Rig, placePose, clonePose, emptyBatch, encodeRequest, FPS } from './soma.js';
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
  trigger: params.get('trigger') ?? 'click', // 'click' (right button) or 'ram' (strike when the body drives into the opponent)
  enemies: clamp(Number(params.get('enemies') ?? 1), 1, 3), // how many opponents at once
  w: { feetPos: 1, feetVel: 0.75, hipsVel: 1, hipsH: 0.7, hipsVy: 0.8, hands: 0.3, head: 0.3, handsVel: 0.1, trajPos: 1, trajDir: 1.2, handle: 1.5, trajH: 1 },
  damage: { player: 0.9, enemy: 0.45 }, // hp per joule (the opponents hit softer)
  knockdown: 60, // J in one blow that puts the one hit on the floor
};

// ---------------------------------------------------------------- the reference database

const TAGS = ['move', 'stance', 'attack', 'block', 'hit', 'fall', 'getup'];
const T = Object.fromEntries(TAGS.map((t, k) => [t, k]));
let db = null, entTag = null, change = null, strikes = [], entryAt = new Map(), standPose = null;
PoseDB.load(rig, new URL('../assets/fightdb', import.meta.url).href).then(d => {
  db = d;
  entTag = new Uint8Array(d.n);
  change = new Float32Array(d.n);
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
const strikeKeys = [], hitKeys = [], fallKeys = [];
function keyFrames(d, c, f, mirrored, ref = f) {
  const p0 = d.pose(c, ref, mirrored);
  const h = rig.heading(rig.positions(p0)), at = flat(p0.root);
  return [0, 1, 2, 3].map(k => placePose(d.pose(c, f + k, mirrored), -h, at, at.clone().negate()));
}
function buildKeys(d) {
  d.clips.forEach((c, k) => {
    if (c.tag === 'attack' && c.strike) {
      for (const mirrored of [false, true]) {
        const part = mirrored ? swapSide(c.strike.part) : c.strike.part;
        const frames = keyFrames(d, k, c.strike.apex, mirrored);
        const tip = rig.positions(frames[0])[I[part]];
        const kind = /Foot/.test(part) ? 'kick' : /Shin/.test(part) ? 'knee' : /ForeArm/.test(part) ? 'elbow' : 'punch';
        if (tip.z < 0.25) continue; // a strike that does not reach forward (a spin, a back kick)
        // what comes after the top, in the same frame of reference: the swing carrying on (a miss) and the limb
        // coming back a little sooner (a blow that met something)
        const own = (c.own ?? c.frames) - 4;
        const miss = keyFrames(d, k, Math.min(own, c.strike.apex + 9), mirrored, c.strike.apex);
        const hit = keyFrames(d, k, Math.min(own, c.strike.apex + 5), mirrored, c.strike.apex);
        strikeKeys.push({ clip: k, name: c.name, part, kind, frames, tip, high: tip.y > 1.25, miss, hit });
      }
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
      const where = /head|face|jaw|chin|nose/.test(name) ? 'head' : /body|stomach|rib|kidney|liver|side|gut|chest/.test(name) ? 'body' : 'any';
      for (const mirrored of [false, true]) {
        const key = { clip: k, name: c.name, frames: keyFrames(d, k, best, mirrored), where, strength: bd };
        (c.tag === 'fall' ? fallKeys : hitKeys).push(key);
      }
    }
  });
}
const toLocal = p => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));
function fromLocalPose(root, L) {
  const rot = [];
  L.forEach((q, j) => { rot.push(rig.parents[j] < 0 ? q.clone() : rot[rig.parents[j]].clone().multiply(q)); });
  return rig.fixRigid({ root: root.clone(), rot });
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
function groundFrames(frames) {
  const n = frames.length, off = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const p = frames[k].pose, prev = frames[Math.max(0, k - 1)].pose.root;
    const speed = Math.hypot(p.root.x - prev.x, p.root.z - prev.z) * FPS;
    const low = lowest(rig.positions(p));
    off[k] = speed > 2 ? 0 : Math.abs(low) > 0.015 ? -low : 0;
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
  for (let k = 1; k <= TRAJ.at(-1); k++) {
    if (F.spring) springStep(v, p, F.spring.target, 1 / FPS); else approach(v, F.want, 1 / FPS);
    p.addScaledVector(v, 1 / FPS);
    const face = opp && F.state !== 'fall' && F.state !== 'down' ? Math.atan2(opp.P[0].x - p.x, opp.P[0].z - p.z) : (v.length() > 0.3 ? Math.atan2(v.x, v.z) : yaw);
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
          if (i != null && allow.includes(entTag[i])) cont = { i, frame: fe };
        }
        const recentSet = new Set(F.recent.filter(r => clock - r.t < 1.5).map(r => r.clip));
        const passive = F.vel.length() < 0.05 && F.want.length() < 0.05;
        const calm = kind === 'move' ? (passive ? 2 : 0.4) : 0;
        const bias = i => {
          if (!allow.includes(entTag[i])) return Infinity;
          let b = calm * change[i];
          if (cont && db.entClip[i] === F.current.clip && db.entFrame[i] < cont.frame && db.entFrame[i] > cont.frame - 40) b += 4;
          if (recentSet.has(db.entClip[i]) && !(cont && db.entClip[i] === F.current.clip)) b += 2;
          return b;
        };
        best = db.search(q, w, bias);
        if (best.i < 0) best = db.search(q, w, i => (ALLOW.move.includes(entTag[i]) ? 0 : Infinity)); // nothing of that kind
        if (cont) {
          const c = costOf(cont.i, q, w) + calm * change[cont.i];
          if (best.cost > settings.stay * c) best = { i: cont.i, cost: c, clip: F.current.clip, frame: cont.frame, mirrored: F.current.mirror, stayed: true };
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
      if (opp && kind !== 'fall' && kind !== 'getup') { const r = clonePose(targets[0]); gaze(r, vital(opp)); targets[0] = rig.fixRigid(r); }
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
function strikeReq(F) {
  return {
    kind: 'attack', mode: 'now', interrupt: true,
    build(tr, a, ctx) {
      const from = ctx[3];
      const opp = F.opponent;
      const A = flat(from.root);
      const yaw = opp ? Math.atan2(opp.P[0].x - A.x, opp.P[0].z - A.z) : rig.heading(rig.positions(from));
      const c = Math.cos(-yaw), sn = Math.sin(-yaw);
      const local = v => { const dx = v.x - A.x, dz = v.z - A.z; return new Vector3(dx * c + dz * sn, v.y, -dx * sn + dz * c); };
      const head = opp ? opp.P[I.Head].clone().add(new Vector3(0, -0.02, 0)) : A.clone().add(new Vector3(Math.sin(yaw), 1.5, Math.cos(yaw)));
      const body = opp ? opp.P[I.Spine2].clone() : head.clone().setY(1.1);
      const speedIn = opp ? F.vel.dot(flat(opp.P[0].clone().sub(A)).normalize()) : 0;
      // where the bodies will be when the strike lands (about 0.4 s on): the body carries on with its momentum
      const tLand = scheduler.latency * 0 + 0.38;
      const carry = F.vel.clone().multiplyScalar(tLand).clampLength(0, 1.2);
      const oppAhead = opp ? opp.vel.clone().multiplyScalar(tLand + scheduler.latency).clampLength(0, 0.8) : new Vector3();
      head.add(oppAhead); body.add(oppAhead);
      A.add(carry);
      let best = null, bc = Infinity;
      for (const k of strikeKeys) {
        const goal = local(k.high ? head : body);
        const step = clamp(goal.z - k.tip.z, -0.15, WIND[k.kind]);
        let cost = Math.abs(goal.z - k.tip.z - step) * 4 + Math.abs(goal.y - k.tip.y) * 2 + Math.abs(goal.x - k.tip.x) * 1.5;
        cost += (k.clip === F.lastStrike ? 1.5 : 0) + (k.clip === F.lastStrike2 ? 0.7 : 0) + Math.random() * 0.8;
        if (speedIn > 2.2 && (k.kind === 'kick' || k.kind === 'knee')) cost -= 0.8; // charging in: a kick or a knee
        else if (speedIn > 1.2 && step > 0.3) cost -= 0.4; // coming in: a stepping blow
        if (cost < bc) { bc = cost; best = { k, step, goal }; }
      }
      if (!best) return planReq(F, 'move').build(tr, a, ctx);
      const { k, step } = best;
      F.lastStrike2 = F.lastStrike; F.lastStrike = k.clip;
      const fwd = new Vector3(Math.sin(yaw), 0, Math.cos(yaw));
      const at = A.clone().addScaledVector(fwd, step);
      const aim = (k.high ? head : body).clone().addScaledVector(fwd, 0.25); // aim through the target, not at it
      let targets = k.frames.map(f => ground(placePose(f, yaw, ZERO, at)));
      targets = targets.map((t, i) => reachLimb(t, k.part, aim.clone().addScaledVector(fwd, 0.03 * i), 0.35));
      if (opp) targets = targets.map(t => { const r = clonePose(t); gaze(r, head); return rig.fixRigid(r); });
      F.attack = { part: k.part, id: Math.random(), kind: k.kind, key: k, yaw, at, followed: false, landed: false };
      F.lastMatch = { name: k.name, kind: 'attack', cost: bc, frame: 0 };
      F.current = null;
      const rate = RATE[k.kind];
      return {
        targets, tokens: [6], rate, fixEnd: true, tag: { kind: 'attack' },
        after: (tr2, a0, frames) => { groundFrames(frames); F.state = 'attack'; F.until = clock + frames.length / FPS / rate + 0.6; if (F.attack) F.attack.spliced = true; },
      };
    },
  };
}
/**
 * After the top of a strike, at once (no stop): a miss carries on along the reference's own swing (the follow-
 * through, with the body's momentum); a blow that met something comes back sooner. Soft (not held to the pose):
 * MotionBricks settles it, then normal motion carries on.
 */
function followReq(F) {
  return {
    kind: 'attack', mode: 'end',
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
        after: (tr2, a0, gen) => { groundFrames(gen); F.until = clock + (tr2.frames.length - 1 - tr2.t) / FPS / rate - 0.15; },
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
      return { targets: [t], tokens: [6, 7], rate: 1.2, fixEnd: 'root', tag: { kind: 'idle' }, after: (tr2, a0, gen) => { groundFrames(gen); F.until = clock + 0.2; } };
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
/** Hit: the reaction built from the blow, mixed with a reference reaction at its strongest; then back. */
function reactReq(F, from, energy, isHead, down, dir = null, hitAt = null) {
  return {
    kind: down ? 'down' : 'hit', mode: 'now', interrupt: true,
    build(tr, a, ctx) {
      const A = flat(ctx[3].root);
      const away = A.clone().sub(flat(from));
      if (away.lengthSq() < 1e-6) away.set(1, 0, 0);
      away.normalize();
      const pool = down ? fallKeys : hitKeys.filter(h => h.where === (isHead ? 'head' : 'body') || h.where === 'any');
      const list = pool.length ? pool : hitKeys;
      // stronger blows: the bigger reactions
      const wantStrength = clamp(0.25 + energy / 120, 0.2, 1.2);
      let best = null, bc = Infinity;
      for (const h of list) { const c = Math.abs(h.strength - wantStrength) + Math.random() * 0.4 + (h.clip === F.lastReact ? 0.5 : 0); if (c < bc) { bc = c; best = h; } }
      F.lastReact = best?.clip;
      const back = down ? 0.35 : clamp(0.1 + energy / 250, 0.1, 0.45);
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
        targets, tokens: down ? [9, 10] : [6], rate, fixEnd: true, tag: { kind: down ? 'down' : 'hit' },
        after: (tr2, a0, frames) => {
          groundFrames(frames);
          const left = (tr2.frames.length - 1 - tr2.t) / FPS / rate;
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
function spark(at, energy) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffe9a8, transparent: true, opacity: 0.9, depthTest: false }));
  m.position.copy(at); m.renderOrder = 20; scene.add(m);
  sparks.push({ m, t: 0, s: 1 + Math.min(4, energy / 40) });
}
/**
 * Contact of A's striking limbs with B's head (sphere) or trunk (capsule). While a limb is in contact its fastest
 * closing speed is kept; the blow lands when the limb starts to slow down or leaves (so a fist that brushes the
 * body on its way out is not counted at its slowest moment). Energy = 0.5 m v^2 with the limb's effective mass.
 */
function checkHits(A, B, dt) {
  if (A.state === 'out' || B.state === 'out' || B.state === 'fall' || B.state === 'down') { A.contact = null; return; }
  const striking = A.state === 'attack' || settings.trigger === 'ram';
  if (!striking) { A.contact = null; return; }
  const head = B.P[I.Head].clone().add(new Vector3(0, 0.06, 0)), hips = B.P[0], neck = B.P[I.Neck1];
  const vB = B.P[0].clone().sub(B.prevP[0]).divideScalar(dt);
  let touching = null;
  for (const [part, m] of Object.entries(STRIKE_MASS)) {
    if (A.state === 'attack' && A.attack && part !== A.attack.part && !/ForeArm|Shin/.test(part)) continue;
    const p = A.P[I[part]], v = p.clone().sub(A.prevP[I[part]]).divideScalar(dt);
    A.peak = Math.max(A.peak ?? 0, v.length());
    const seg = neck.clone().sub(hips), tt = clamp(p.clone().sub(hips).dot(seg) / seg.lengthSq(), 0, 1);
    const onTrunk = hips.clone().addScaledVector(seg, tt);
    const dHead = p.distanceTo(head) - 0.12, dTrunk = p.distanceTo(onTrunk) - 0.16;
    const isHead = dHead < dTrunk + 0.03; // a fist near both is a blow to the head
    const d = Math.min(dHead, dTrunk);
    if (d > 0.06) continue;
    const c = isHead ? head : onTrunk;
    const n = c.clone().sub(p).normalize();
    const closing = v.clone().sub(vB).dot(n);
    if (!touching || closing > touching.closing) touching = { part, m, closing, isHead, n, at: p.clone() };
  }
  const id = A.attack?.id ?? `ram${Math.floor(clock * 2)}`;
  const C = A.contact;
  if (touching && touching.closing > 0.5) {
    if (!C || C.id !== id) A.contact = { id, best: touching, done: false };
    else if (!C.done) {
      if (touching.closing > C.best.closing) C.best = touching;
      else if (touching.closing < 0.7 * C.best.closing) settle(A, B, C);
    }
  } else if (C && !C.done) settle(A, B, C);
}
function settle(A, B, C) {
  C.done = true;
  const b = C.best;
  if (b.closing < 1.2 || B.lastHitBy === C.id) return;
  B.lastHitBy = C.id;
  land(A, B, 0.5 * b.m * b.closing * b.closing * (b.isHead ? 1.2 : 1), b.n, b.at, b.isHead);
}
function land(A, B, energy, dir, at, isHead) {
  const dmg = energy * (A.isPlayer ? settings.damage.player : settings.damage.enemy);
  B.hp = Math.max(0, B.hp - dmg);
  spark(at, energy);
  const push = flat(dir).normalize().multiplyScalar(clamp(0.35 * Math.sqrt(energy), 0.6, 2.6));
  B.knock.copy(push);
  const down = B.hp <= 0 || energy > settings.knockdown;
  B.state = down ? 'fall' : 'hit';
  B.until = clock + 3;
  B.track.request(reactReq(B, A.P[0], energy, isHead, down, dir, at));
  if (A.attack) A.attack.landed = true;
  B.body.flash = 1;
  hitstop = clamp(0.04 + energy / 2000, 0.04, 0.12);
  if (B.state === 'attack' || B.attack) B.attack = null; // a hit cancels the one hit's own strike
  if (B.hp <= 0) B.ko = true;
  lastHit = { energy, dmg, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock };
  hudFight();
}
let lastHit = null, hitstop = 0;
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
/** Opponents: keep about arm's length, drift around a little, strike now and then (not too often, not too hard). */
function think(E, dt) {
  const P = player;
  if (!P || E.state !== 'free') { E.want.set(0, 0, 0); return; }
  const to = flat(P.P[0].clone().sub(E.P[0])), d = to.length();
  to.normalize();
  const side = new Vector3(to.z, 0, -to.x);
  const keep = P.state === 'down' || P.state === 'fall' ? 2.2 : 1.15;
  E.want.copy(to).multiplyScalar(clamp((d - keep) * 1.4, -0.9, 1.3)).addScaledVector(side, 0.35 * Math.sin(clock * 0.45 + E.maxHp));
  const nearest = P.opponent === E;
  if (!nearest) E.want.addScaledVector(to, -0.4); // the others wait their turn a little further off
  // now and then the nearest one lunges in (and its strike comes out of the lunge)
  if (nearest && d < 2.4 && clock > E.nextAttack && P.state !== 'down' && P.state !== 'fall') {
    E.lungeUntil = clock + 0.6;
    E.nextAttack = clock + rand(2.4, 4.2);
  }
  if (clock < (E.lungeUntil ?? 0)) E.want.copy(to).multiplyScalar(2.0);
}

// ---------------------------------------------------------------- input (no camera control)

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let drag = null;
function floorPoint(x, y) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const out = new Vector3();
  return ray.ray.intersectPlane(new THREE.Plane(UP, -0.95), out) ? out : null; // the plane at hip height
}
canvas.addEventListener('pointerdown', e => {
  if (!player) return;
  if (e.button !== 0) return;
  drag = { screen: { x: e.clientX, y: e.clientY }, point: null };
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => { if (drag) drag.screen = { x: e.clientX, y: e.clientY }; });
const release = () => { drag = null; if (player) player.planTimer = 0; };
canvas.addEventListener('pointerup', e => { if (e.button === 0) release(); });
canvas.addEventListener('pointercancel', release);
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
  const closing = F.vShown.clone().sub(o.vShown).dot(rel.clone().normalize());
  if (closing < (F.isPlayer ? 1.1 : 0.9)) return;
  const tc = (d - 0.85) / closing; // until the bodies are at striking distance
  if (tc > 0.55 || tc < -0.1) return;
  F.strikeAfter = clock + 0.7;
  strike(F);
}
function strike(F) {
  if (!F.opponent || (F.state !== 'free' && F.state !== 'attack')) return;
  if (F.state === 'attack' && clock < F.until - 0.2) { F.attackQueued = true; return; } // one more right after this one
  F.track.request(strikeReq(F));
  F.state = 'attack'; F.until = clock + 1.0;
}
// the right button pressed while the left one is held gives no pointerdown (a chorded press): take it from
// mousedown / contextmenu too (once per press)
canvas.addEventListener('contextmenu', e => e.preventDefault());

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), clock = 0, fpsAcc = 0, fpsN = 0, fps = 0;
const camTarget = new Vector3(0, 0.9, 0), camSide = new Vector3(1, 0, 0);
function simulate(dt) {
  if (!db) return;
  if (hitstop > 0) { hitstop -= dt; return; } // a landed blow holds the picture a moment
  if (player && fighters.filter(f => !f.isPlayer && f.state !== 'out').length < settings.enemies && clock > spawnAt) { spawn(); spawnAt = clock + 1.5; }
  // the player's leash on the hips
  if (player) {
    if (drag && !drag.scripted) drag.point = floorPoint(drag.screen.x, drag.screen.y) ?? drag.point;
    const free = player.state === 'free' || player.state === 'attack';
    player.spring = { target: drag?.point && free ? drag.point.clone() : null };
    springStep(player.vel, flat(player.P[0]), player.spring.target, dt);
    if (!free) player.vel.multiplyScalar(Math.exp(-dt * 4));
    player.want.copy(player.vel);
    autoStrike(player);
  }
  for (const F of fighters) {
    if (!F.isPlayer) { think(F, dt); approach(F.vel, F.want, dt); autoStrike(F); }
    // states end
    if (F.state !== 'free' && F.state !== 'out' && clock > F.until && !F.track.busy) {
      if (F.state === 'down') {
        if (F.ko && !F.isPlayer) { F.state = 'out'; F.outAt = clock; kos++; spawnAt = clock + 2.5; hudFight(); }
        else { F.track.request(planReq(F, 'getup', 'end')); F.state = 'getup'; F.until = clock + 2.5; if (F.isPlayer && F.hp <= 0) F.hp = F.maxHp; }
      } else if (F.state === 'attack' && F.attackQueued) { F.attackQueued = false; F.state = 'free'; strike(F); }
      else if (F.state !== 'down') F.state = 'free';
    }
    // a strike carries straight on into its follow-through (asked for as the top comes near)
    if (F.state === 'attack' && F.attack?.spliced && !F.attack.followed && !F.track.busy && F.track.remaining / FPS < scheduler.latency + 0.2) {
      F.attack.followed = true;
      F.track.request(followReq(F));
    }
    // plans while free
    F.planTimer -= dt;
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
    F.P = rig.positions(F.shown);
    if (F.P[0].distanceTo(F.prevP[0]) > 0.3) jumps.push({ t: +clock.toFixed(2), who: F.isPlayer ? 'P' : 'E', state: F.state, d: +F.P[0].distanceTo(F.prevP[0]).toFixed(2), match: F.lastMatch?.name, kind: F.lastMatch?.kind, splices: F.track.splices, frames: F.track.frames.length, t2: +F.track.t.toFixed(1) });
  }
  // bodies do not pass through each other: hips closer than 0.7 m are pushed apart (the motion is shifted)
  const live = fighters.filter(f => f.state !== 'out');
  for (let a = 0; a < live.length; a++) for (let b = a + 1; b < live.length; b++) {
    const A = live[a], B = live[b];
    const d = flat(B.P[0].clone().sub(A.P[0])), len = d.length();
    if (len < 0.75 && len > 1e-4) {
      const push = d.normalize().multiplyScalar(Math.min(0.05, (0.75 - len) * 0.4));
      A.track.shift(-push.x, -push.z); B.track.shift(push.x, push.z);
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
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  for (const F of fighters) F.body.update(F.P, rig.extraPositions(F.shown, F.P), dt);
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
  if (lastHit && clock - lastHit.t < 1.5) $('last-hit').textContent = `${lastHit.by}の${lastHit.isHead ? '顔' : '体'}への一撃 ${lastHit.energy.toFixed(0)} J`;
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
    rig, scheduler, settings, fighters, simulate, strike,
    get player() { return player; }, jumps, get db() { return db; }, get clock() { return clock; }, get kos() { return kos; }, get lastHit() { return lastHit; },
    pull(point) { drag = point ? { point, scripted: true } : null; },
    draw: () => frame(performance.now()),
    onStep: null,
  };
}
