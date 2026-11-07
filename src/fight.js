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
  vmax: 2.0, stiff: 2.4, replan: 0.2, stay: 0.85,
  trigger: params.get('trigger') ?? 'click', // 'click' (right button) or 'ram' (strike when the body drives into the opponent)
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
  const idle = d.clips.findIndex(c => /fight_idle|boxing_idle|bouncing_fight_idle|fighting_idle/.test(c.name));
  standPose = d.pose(Math.max(0, idle), 10);
  $('db-info').textContent = `${d.clips.length} クリップ / 攻撃 ${new Set(strikes.map(s => db.entClip[s.i])).size} 種`;
  start();
}).catch(e => { $('db-info').textContent = `読み込めません: ${e.message}`; });
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
function predict(F, pose, heading) {
  let yaw = heading;
  const p = flat(pose.root), v = F.vel.clone();
  const opp = F.opponent;
  const out = [];
  for (let k = 1; k <= TRAJ.at(-1); k++) {
    approach(v, F.want, 1 / FPS);
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
  B.track.request(planReq(B, down ? 'fall' : 'hit'));
  if (B.hp <= 0) B.ko = true;
  lastHit = { energy, dmg, isHead, by: A.isPlayer ? 'あなた' : '相手', t: clock };
  hudFight();
}
let lastHit = null;

// ---------------------------------------------------------------- the player and the opponents

let player = null, kos = 0, spawnAt = 0;
function start() {
  player = new Fighter(true, 0xd9b26a, new Vector3(0, 0, 0), 0);
  fighters.push(player);
  spawnAt = clock + 0.5;
}
function spawn() {
  const h = rig.heading(player.P), d = 5.5;
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
  if (d < 1.45 && clock > E.nextAttack && P.state !== 'down' && P.state !== 'fall') {
    E.track.request(planReq(E, 'attack'));
    E.state = 'attack'; E.until = clock + 1.5;
    E.nextAttack = clock + rand(2.2, 4.0);
  }
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
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => {
  if (!player) return;
  if (e.button === 2) { strike(player); e.preventDefault(); return; }
  if (e.button !== 0) return;
  drag = { screen: { x: e.clientX, y: e.clientY }, point: null };
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => { if (drag) drag.screen = { x: e.clientX, y: e.clientY }; });
const release = () => { drag = null; if (player) player.planTimer = 0; };
canvas.addEventListener('pointerup', e => { if (e.button === 0) release(); });
canvas.addEventListener('pointercancel', release);
function strike(F) {
  if (!F.opponent || (F.state !== 'free' && F.state !== 'attack')) return;
  if (F.state === 'attack' && clock < F.until - 0.35) { F.attackQueued = true; return; } // one more after this one
  F.track.request(planReq(F, 'attack'));
  F.state = 'attack'; F.until = clock + 1.5;
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), clock = 0, fpsAcc = 0, fpsN = 0, fps = 0;
const camTarget = new Vector3(0, 0.9, 0), camSide = new Vector3(1, 0, 0);
function simulate(dt) {
  if (!db) return;
  if (player && !fighters.some(f => !f.isPlayer && f.state !== 'out') && clock > spawnAt) spawn();
  // the player's leash on the hips
  if (player) {
    if (drag && !drag.scripted) drag.point = floorPoint(drag.screen.x, drag.screen.y) ?? drag.point;
    if (drag?.point && (player.state === 'free' || player.state === 'attack')) {
      const d = flat(drag.point.clone().sub(player.P[0]));
      const stretch = d.length() - 0.1;
      player.want.copy(stretch > 0 ? d.normalize().multiplyScalar(settings.vmax * (1 - Math.exp(-stretch * settings.stiff / settings.vmax))) : ZERO);
    } else player.want.set(0, 0, 0);
    if (settings.trigger === 'ram' && player.opponent && player.state === 'free') {
      // drive into the opponent fast enough and the body strikes by itself
      const o = player.opponent, to = flat(o.P[0].clone().sub(player.P[0]));
      if (to.length() < 1.5 && player.vel.dot(to.normalize()) > 0.9) strike(player);
    }
  }
  for (const F of fighters) {
    if (!F.isPlayer) think(F, dt);
    approach(F.vel, F.want, dt);
    // states end
    if (F.state !== 'free' && F.state !== 'out' && clock > F.until && !F.track.busy) {
      if (F.state === 'down') {
        if (F.ko && !F.isPlayer) { F.state = 'out'; F.outAt = clock; kos++; spawnAt = clock + 2.5; hudFight(); }
        else { F.track.request(planReq(F, 'getup', 'end')); F.state = 'getup'; F.until = clock + 3; if (F.isPlayer && F.hp <= 0) F.hp = F.maxHp; }
      } else if (F.state === 'attack' && F.attackQueued) { F.attackQueued = false; F.state = 'free'; strike(F); }
      else if (F.state !== 'down') F.state = 'free';
    }
    // plans while free
    F.planTimer -= dt;
    if (F.state === 'free' && scheduler.ready && !F.track.busy) {
      const active = F.want.length() > 0.05 || F.vel.length() > 0.05;
      const changed = F.want.distanceTo(F.lastWant) > 0.4;
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
for (const el of document.querySelectorAll('input[name=trigger]')) {
  if (el.value === settings.trigger) el.checked = true;
  el.addEventListener('change', () => { settings.trigger = el.value; });
}

requestAnimationFrame(loop);
if (params.has('debug')) {
  window.brawl = {
    rig, scheduler, settings, fighters, simulate, strike,
    get player() { return player; }, get db() { return db; }, get clock() { return clock; }, get kos() { return kos; }, get lastHit() { return lastHit; },
    pull(point) { drag = point ? { point, scripted: true } : null; },
    draw: () => frame(performance.now()),
    onStep: null,
  };
}
