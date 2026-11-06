// SOMA Pull: grab a hand, the hips, the head or a foot and pull; the body follows like a living person.
//
// How it moves:
// 1. A leash: the grabbed point pulls the body only once the rope is taut. The stretch sets a wanted ground velocity
//    (capped, limited acceleration, the hips never go up). How far ahead MotionBricks plans also depends on it: a
//    small tug is answered in about half a second, a long pull is planned up to 1.4 s ahead.
// 2. MotionBricks gets four targets on the predicted path, 1/30 s apart, so they carry speed and facing; it plans
//    its own steps. While walking the targets come from a gait library grown from the model's own steady output.
//    When standing, the targets fix only where the hips go and which way they face: the body is the model's own,
//    and every few seconds the weight shifts a little.
// 3. A full-body cascade (Cascadeur-like): a pull is taken up by the arm, then the collarbone, the spine and the
//    hips (feet stay, knees bend), the other arm balancing. The cascaded pose is the model's target while standing
//    and is laid over the generated motion on screen through per-joint springs, so the body lags and settles.
// 4. Gaze: a knob in front of the face sets where the eyes look; the head, neck and chest turn toward it, and when
//    it is far behind, the body turns on its feet. Breathing and planted feet (from the model's foot contacts) are
//    added on screen. The floor is endless and the camera follows.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, clonePose, emptyBatch, encodeRequest, FPS } from './soma.js';
import { rotateSubtree, twoBoneIK } from './poses.js';
import { Track, Scheduler, BATCH } from './motion.js';
import { Stage, Mannequin } from './render.js';

const MB_REV = '54fc563aeaaf819a753d72cc9fdc6713989cdd43';
const MODELS = {
  fp16: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx`,
  fp32: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx`,
};
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const UP = new Vector3(0, 1, 0);
const ZERO = new Vector3();

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const I = rig.index;
const canvas = $('view');
const stage = new Stage(canvas);
stage.makeEndless();
const scene = stage.scene;
const camera = stage.camera;
const worker = new Worker(new URL('./infer.worker.js', import.meta.url), { type: 'module' });
const scheduler = new Scheduler(rig, worker);

const settings = { vmax: 2.4, stiff: 2.6, footlock: true };
for (const [id, key, fmt] of [['vmax', 'vmax', v => v.toFixed(1)], ['stiff', 'stiff', v => v.toFixed(1)], ['footlock', 'footlock']]) {
  const el = $(id);
  const read = () => { settings[key] = el.type === 'checkbox' ? el.checked : Number(el.value); if ($(`${id}-v`)) $(`${id}-v`).textContent = fmt(settings[key]); };
  el.addEventListener('input', read);
  read();
}

// ---------------------------------------------------------------- math and pose helpers

const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const flat = v => new Vector3(v.x, 0, v.z);
const ANKLE = rig.restPos[I.LeftFoot].y;
const ARM = ['Arm', 'ForeArm', 'Hand'], LEG = ['Leg', 'Shin', 'Foot'];
const chain = (side, parts) => parts.map(p => I[side + p]);
const reachOf = (P, [a, b, c]) => P[b].distanceTo(P[a]) + P[c].distanceTo(P[b]);
const sideOf = part => (part.startsWith('Left') ? 'Left' : 'Right');
const otherSide = side => (side === 'Left' ? 'Right' : 'Left');

function logq(q) {
  let { x, y, z, w } = q;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return new Vector3();
  return new Vector3(x, y, z).multiplyScalar((2 * Math.atan2(s, w)) / s);
}
function expq(v) {
  const a = v.length();
  return a < 1e-9 ? new Quaternion() : new Quaternion().setFromAxisAngle(v.clone().divideScalar(a), a);
}
const qpow = (q, k) => expq(logq(q).multiplyScalar(k));
const qcap = (q, max) => { const v = logq(q); return v.length() > max ? expq(v.setLength(max)) : q.clone(); };
const turn = (from, to) => new Quaternion().setFromUnitVectors(from.clone().normalize(), to.clone().normalize());
const toLocal = p => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));
function fromLocal(root, L) {
  const rot = [];
  L.forEach((q, j) => { rot.push(rig.parents[j] < 0 ? q.clone() : rot[rig.parents[j]].clone().multiply(q)); });
  return rig.fixRigid({ root: root.clone(), rot });
}
const facing = P => { const h = rig.heading(P); return new Vector3(Math.sin(h), 0, Math.cos(h)); };

/** Bend both legs so the ankles reach the given points, keeping each foot's orientation. */
function legsTo(p, feet) {
  ['Left', 'Right'].forEach((side, i) => {
    const c = chain(side, LEG);
    const P = rig.positions(p);
    const keep = p.rot[c[2]].clone();
    const knee = P[c[1]].clone().sub(P[c[0]].clone().lerp(P[c[2]], 0.5));
    twoBoneIK(rig, p, c[0], c[1], c[2], feet[i], knee.lengthSq() > 1e-6 ? knee.add(facing(P).multiplyScalar(0.3)) : facing(P));
    rotateSubtree(rig, p, c[2], keep.multiply(p.rot[c[2]].clone().invert()));
  });
}

/** A relaxed standing pose, facing +Z with the hips over the origin (only used for placement and as a fallback). */
function makeStand() {
  const p = rig.restPose();
  twoBoneIK(rig, p, I.LeftArm, I.LeftForeArm, I.LeftHand, new Vector3(0.24, 0.86, 0.04), new Vector3(0.3, 0, -1));
  twoBoneIK(rig, p, I.RightArm, I.RightForeArm, I.RightHand, new Vector3(-0.24, 0.86, 0.04), new Vector3(-0.3, 0, -1));
  p.root.y -= 0.02;
  legsTo(p, [new Vector3(0.1, ANKLE, 0.02), new Vector3(-0.1, ANKLE, 0.02)]);
  return rig.fixRigid(p);
}
const STAND = makeStand();
const HIPS_Y = STAND.root.y;

function canonicalize(poses, ref = poses[0]) {
  const h = rig.heading(rig.positions(ref));
  const at = new Vector3(ref.root.x, 0, ref.root.z);
  return poses.map(p => {
    const q = placePose(p, -h, at, at.clone().negate());
    if (p.contact) q.contact = p.contact;
    return q;
  });
}

// ---------------------------------------------------------------- gait library (grows from the model's output)

const BINS = [
  { name: 'ゆっくり', lo: 0.3, hi: 0.9 }, { name: '歩き', lo: 0.9, hi: 1.6 },
  { name: '早歩き', lo: 1.6, hi: 2.3 }, { name: '走り', lo: 2.3, hi: 3.3 },
];
const PER_KEY = 8;
const library = new Map();
const binOf = s => BINS.findIndex(b => s >= b.lo && s < b.hi);

function stanceFoot(contact) {
  if (!contact) return 'D';
  const l = Math.max(contact[0], contact[1]) > 0.5, r = Math.max(contact[2], contact[3]) > 0.5;
  return l && r ? 'D' : l ? 'L' : r ? 'R' : 'F';
}

function harvest(frames) {
  const n = frames.length;
  for (let i = 6; i + 4 < n - 6; i += 3) {
    const w = frames.slice(i, i + 4).map(f => f.pose);
    const v = [];
    for (let k = 1; k < 4; k++) v.push(new Vector3(w[k].root.x - w[k - 1].root.x, 0, w[k].root.z - w[k - 1].root.z).multiplyScalar(FPS));
    const speeds = v.map(x => x.length());
    const mean = speeds.reduce((a, b) => a + b, 0) / speeds.length;
    if (Math.max(...speeds) - Math.min(...speeds) > 0.2 + 0.1 * mean) continue;
    const bin = binOf(mean);
    if (bin < 0) continue;
    if (w.some(p => p.root.y < HIPS_Y - 0.14 || p.root.y > HIPS_Y + 0.08)) continue;
    if (w.some(p => { const Q = rig.positions(p); return Math.min(Q[I.LeftFoot].y, Q[I.RightFoot].y) > 0.22; })) continue;
    const fwd = facing(rig.positions(w[0]));
    const dir = v[1].dot(fwd) / Math.max(1e-6, v[1].length()) < -0.3 ? 'b' : 'f';
    const key = `${bin}:${dir}:${stanceFoot(w[0].contact)}`;
    const list = library.get(key) ?? [];
    const entry = { frames: canonicalize(w), speed: mean };
    if (list.length < PER_KEY) list.push(entry); else list[Math.floor(Math.random() * PER_KEY)] = entry;
    library.set(key, list);
  }
  showLibrary();
}

function lookup(speed, dir, phase) {
  const bin = binOf(speed);
  if (bin < 0) return null;
  for (const b of [bin, bin - 1, bin + 1]) {
    if (b < 0 || b >= BINS.length) continue;
    for (const ph of [phase, 'D', phase === 'L' ? 'R' : 'L', 'F']) {
      const list = library.get(`${b}:${dir}:${ph}`);
      if (list?.length) return list[Math.floor(Math.random() * list.length)];
    }
  }
  return null;
}

function showLibrary() {
  const el = $('gait-bins');
  el.innerHTML = '';
  BINS.forEach((b, i) => {
    let n = 0;
    for (const [k, v] of library) if (k.startsWith(`${i}:`)) n += v.length;
    el.insertAdjacentHTML('beforeend', `<span>${b.name}</span><div class="bar"><div class="fill" style="width:${Math.min(100, (n / (PER_KEY * 8)) * 100)}%"></div></div><span class="n">${n}</span>`);
  });
}

// ---------------------------------------------------------------- the leash

const ctrl = { vel: new Vector3(), want: new Vector3(), stretch: 0 };
const SLACK = { LeftHand: 0.5, RightHand: 0.5, Hips: 0.12, Head: 0.3, LeftFoot: 0.9, RightFoot: 0.9 };
const ANCHOR = { LeftHand: 'LeftArm', RightHand: 'RightArm', Hips: 'Hips', Head: 'Head', LeftFoot: 'LeftLeg', RightFoot: 'RightLeg' };

function pullVelocity(P) {
  ctrl.stretch = 0;
  if (!drag) return new Vector3();
  const d = flat(drag.point.clone().sub(P[I[ANCHOR[drag.part]]]));
  const stretch = d.length() - SLACK[drag.part];
  ctrl.stretch = Math.max(0, stretch);
  if (stretch <= 0) return new Vector3();
  return d.normalize().multiplyScalar(Math.min(settings.vmax, stretch * settings.stiff));
}

function approach(v, want, dt) {
  const diff = want.clone().sub(v);
  const acc = (want.length() < v.length() ? 3.5 : 2.5) * dt;
  if (diff.length() > acc) diff.setLength(acc);
  return v.add(diff);
}

/** The hips' ground path from the anchor (velocity easing toward the wanted one, facing turning toward travel). */
function predict(anchorPose, horizon) {
  const P = rig.positions(anchorPose);
  let yaw = rig.heading(P);
  const p = flat(anchorPose.root);
  const v = ctrl.vel.clone();
  const out = [];
  const steps = Math.round(horizon * FPS) + 3;
  for (let k = 1; k <= steps; k++) {
    approach(v, ctrl.want, 1 / FPS);
    p.addScaledVector(v, 1 / FPS);
    const s = v.length();
    if (s > 0.3) {
      const travel = Math.atan2(v.x, v.z);
      const back = Math.cos(travel - yaw) < -0.3 && s < 1.2;
      yaw += clamp(wrap((back ? yaw : travel) - yaw), -3 / FPS, 3 / FPS);
    }
    if (k > steps - 4) out.push({ pos: p.clone(), yaw, vel: v.clone() });
  }
  return out;
}

// ---------------------------------------------------------------- full-body cascade

/**
 * The pose with `part` pulled toward G, the pull taken up joint after joint like a person would: for a hand the
 * arm, then the collarbone, the spine and the hips (feet stay where they are, knees bend), the other arm swinging
 * the other way. `walking` leaves the hips and legs to the gait.
 */
function cascade(src, part, G, walking = false) {
  const p = clonePose(src);
  let P = rig.positions(p);
  const feet = [P[I.LeftFoot].clone(), P[I.RightFoot].clone()];
  const fwd = facing(P);
  if (part.endsWith('Hand')) {
    const side = sideOf(part), arm = chain(side, ARM), sh = I[side + 'Shoulder'];
    const reach = reachOf(P, arm);
    let ex = G.distanceTo(P[arm[0]]) - reach * 0.92;
    const clav = clamp((ex + 0.15) / 0.35, 0, 1); // collarbone first
    if (clav > 0) rotateSubtree(rig, p, sh, qpow(qcap(turn(P[arm[0]].clone().sub(P[sh]), G.clone().sub(P[sh])), 0.35), clav));
    P = rig.positions(p);
    ex = G.distanceTo(P[arm[0]]) - reach * 0.92;
    if (ex > 0) { // then the spine, shared by three joints
      const q = qpow(qcap(turn(P[arm[0]].clone().sub(P[I.Spine1]), G.clone().sub(P[I.Spine1])), 0.8), Math.min(1, ex / 0.5) / 3);
      for (const j of ['Spine1', 'Spine2', 'Chest']) rotateSubtree(rig, p, I[j], q);
    }
    P = rig.positions(p);
    ex = G.distanceTo(P[arm[0]]) - reach * 0.96;
    if (!walking) {
      if (ex > 0) { // then the hips, over the feet
        const d = G.clone().sub(P[arm[0]]).normalize().multiplyScalar(Math.min(ex, 0.22));
        p.root.x += d.x; p.root.z += d.z;
        p.root.y += clamp(d.y * 1.2, -0.3, 0.02);
      }
      if (G.y < 0.8) p.root.y -= Math.min(0.38, (0.8 - G.y) * 0.75); // reaching low: knees bend
      legsTo(p, feet);
    }
    const k2 = clamp(ex + 0.25, 0, 0.6); // the other arm balances
    const pull = flat(G.clone().sub(P[0]));
    if (k2 > 0 && pull.lengthSq() > 1e-6) rotateSubtree(rig, p, I[otherSide(side) + 'Arm'], new Quaternion().setFromAxisAngle(new Vector3().crossVectors(UP, pull.normalize()), 0.6 * k2));
    P = rig.positions(p);
    const to = G.clone().sub(P[arm[0]]);
    const tip = to.length() > reach * 0.995 ? P[arm[0]].clone().addScaledVector(to.normalize(), reach * 0.995) : G.clone();
    const elbow = P[arm[1]].clone().sub(P[arm[0]].clone().lerp(P[arm[2]], 0.5));
    twoBoneIK(rig, p, arm[0], arm[1], arm[2], tip, elbow.lengthSq() > 1e-6 ? elbow : new Vector3(0, -1, -0.3));
  } else if (part.endsWith('Foot')) {
    const side = sideOf(part), leg = chain(side, LEG), stand = side === 'Left' ? 1 : 0;
    const lift = clamp((G.y - ANKLE) / 0.5, 0, 1);
    if (!walking) { // weight over the standing foot, a little lower
      const toStand = flat(feet[stand].clone().sub(p.root));
      p.root.addScaledVector(toStand, 0.6 * Math.min(1, 0.4 + lift));
      p.root.y -= 0.04 * lift;
    }
    const away = flat(P[leg[0]].clone().sub(G));
    if (away.lengthSq() > 1e-6) rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(new Vector3().crossVectors(UP, away.normalize()), 0.3 * lift));
    for (const [s2, sign] of [['Left', 1], ['Right', -1]]) rotateSubtree(rig, p, I[s2 + 'Arm'], new Quaternion().setFromAxisAngle(fwd, sign * 0.45 * lift));
    if (!walking) legsTo(p, side === 'Left' ? [feet[0], feet[1]] : [feet[0], feet[1]]);
    P = rig.positions(p);
    const goal = G.clone();
    goal.y = clamp(goal.y, ANKLE, P[leg[0]].y - 0.12);
    const reach = reachOf(P, leg) * 0.98;
    const to = goal.clone().sub(P[leg[0]]);
    const tip = to.length() > reach ? P[leg[0]].clone().addScaledVector(to.normalize(), reach) : goal;
    twoBoneIK(rig, p, leg[0], leg[1], leg[2], tip, fwd.clone().add(new Vector3(0, 0.6, 0)));
  } else if (part === 'Hips') {
    const d = G.clone().sub(P[0]);
    const h = flat(d);
    if (h.length() > 0.25) h.setLength(0.25);
    if (!walking) {
      p.root.x += h.x; p.root.z += h.z;
      p.root.y += clamp(d.y, -0.42, 0.03);
      legsTo(p, feet);
    }
    if (h.lengthSq() > 1e-6) { // the trunk leans back so the head stays over the feet
      const w = h.clone().normalize().negate();
      rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(new Vector3().crossVectors(UP, w), 1.4 * h.length()));
    }
    const down = clamp(-d.y / 0.4, 0, 1); // squatting leans the chest forward
    if (down > 0) rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(new Vector3().crossVectors(UP, fwd), 0.5 * down));
  } else if (part === 'Head') {
    const ex = clamp((G.distanceTo(P[I.Head]) - 0.04) / 0.6, 0, 1);
    const q = qpow(qcap(turn(P[I.Head].clone().sub(P[I.Spine1]), G.clone().sub(P[I.Spine1])), 0.9), ex / 3);
    for (const j of ['Spine1', 'Spine2', 'Chest']) rotateSubtree(rig, p, I[j], q);
    if (!walking) {
      P = rig.positions(p);
      const lean = flat(P[I.Head].clone().sub(P[0]));
      p.root.addScaledVector(lean, -0.25); // hips go the other way for balance
      legsTo(p, feet);
    }
  }
  return rig.fixRigid(p);
}

// ---------------------------------------------------------------- gaze

const look = { detached: false, point: new Vector3(0, 1.6, 1.2), smooth: new Vector3(0, 1.6, 1.2) };

/** Turn the head (45%), neck (30%) and chest (25%) toward L, as far as a person would. */
function gaze(p, L, weight = 1) {
  const P = rig.positions(p);
  const head = P[I.Head];
  const fwdHead = new Vector3(0, 0, 1).applyQuaternion(p.rot[I.Head].clone().multiply(rig.restRot[I.Head].clone().invert()));
  const want = L.clone().sub(head);
  if (want.lengthSq() < 1e-4) return p;
  // limit to what the neck and chest can do relative to the body
  const body = facing(P);
  const yaw = wrap(Math.atan2(want.x, want.z) - Math.atan2(body.x, body.z));
  const pitch = Math.atan2(want.y, Math.hypot(want.x, want.z));
  const y = clamp(yaw, -1.5, 1.5), pt = clamp(pitch, -0.6, 0.5);
  const dir = new Vector3(Math.sin(Math.atan2(body.x, body.z) + y) * Math.cos(pt), Math.sin(pt), Math.cos(Math.atan2(body.x, body.z) + y) * Math.cos(pt));
  const q = qpow(turn(fwdHead, dir), weight);
  for (const [j, share] of [['Chest', 0.25], ['Neck1', 0.3], ['Head', 0.45]]) rotateSubtree(rig, p, I[j], qpow(q, share));
  return p;
}

// ---------------------------------------------------------------- requests

function horizonFor(stretch, moving) {
  if (!drag) return moving ? 0.9 : 1.2;
  return clamp(0.5 + 0.45 * stretch, 0.5, 1.4);
}
function tokensFor(horizon) {
  const k = Math.round((horizon * FPS + 6) / 4);
  return [k - 1, k, k + 1].filter(t => t >= 6 && t <= 16);
}

/** Stand (or come to a stop) where the path ends: only the hips' place and facing are fixed. */
function freeTarget(pos, yaw) {
  const t = placePose(STAND, yaw, ZERO, flat(pos));
  t.free = true;
  return t;
}

function locoReq(mode = 'now') {
  return {
    kind: 'move', mode,
    build(tr, a) {
      const from = tr.frame(a).pose;
      const horizon = horizonFor(ctrl.stretch, ctrl.vel.length() > 0.3);
      const path = predict(from, horizon);
      const end = path[0];
      const speed = end.vel.length();
      let targets, free = false;
      if (speed >= 0.3) {
        const fwd = new Vector3(Math.sin(end.yaw), 0, Math.cos(end.yaw));
        const dir = end.vel.dot(fwd) / speed < -0.3 ? 'b' : 'f';
        const steps = Math.round(horizon * (1.6 + 0.45 * speed));
        const now = stanceFoot(from.contact);
        const phase = now === 'L' || now === 'R' ? (steps % 2 ? (now === 'L' ? 'R' : 'L') : now) : (Math.random() < 0.5 ? 'L' : 'R');
        const entry = lookup(speed, dir, phase);
        targets = entry ? entry.frames.map(f => placePose(f, end.yaw, ZERO, end.pos)) : path.map(p => placePose(STAND, p.yaw, ZERO, p.pos));
        if (drag) targets = targets.map(t => cascade(t, drag.part, drag.point, true));
      } else if (drag) {
        // standing and being pulled: the body as it is now, moved along the (short) path, with the pull cascaded
        const h0 = rig.heading(rig.positions(from));
        targets = path.map(p => cascade(placePose(from, wrap(p.yaw - h0), flat(from.root), flat(p.pos).sub(flat(from.root))), drag.part, drag.point));
      } else {
        targets = [freeTarget(end.pos, end.yaw)];
        free = true;
      }
      ghostPose = free ? null : targets[0];
      return {
        targets, tokens: tokensFor(horizon), rate: 1, tag: { kind: free ? 'idle' : 'move' }, fixEnd: !free,
        after: (_tr, _a, frames) => harvest(frames),
      };
    },
  };
}

/** Standing: shift the weight a little, or turn on the feet toward where the eyes look. */
function idleReq(mode = 'end', fidget = false) {
  return {
    kind: 'idle', mode,
    build(tr, a) {
      const from = tr.frame(a).pose;
      const h = rig.heading(rig.positions(from));
      const A = flat(from.root);
      let yaw = h;
      const L = look.point;
      const toLook = Math.atan2(L.x - A.x, L.z - A.z);
      const d = wrap(toLook - h);
      if (look.detached && Math.abs(d) > 1.2) yaw = h + d - Math.sign(d) * 0.45; // turn the body, the head does the rest
      if (fidget) { yaw += rand(-0.12, 0.12); const ang = rand(0, Math.PI * 2); A.add(new Vector3(Math.sin(ang), 0, Math.cos(ang)).multiplyScalar(rand(0.02, 0.06))); }
      ghostPose = null;
      return { targets: [freeTarget(A, yaw)], tokens: [10, 11, 12, 13, 14], rate: 1, fixEnd: false, tag: { kind: 'idle' } };
    },
  };
}

// ---------------------------------------------------------------- scene objects

const track = new Track(rig, { stance: STAND });
const body = new Mannequin(scene, rig, 0xd9b26a);
const ghost = new Mannequin(scene, rig, 0x8fb4ff);
ghost.material.transparent = true;
ghost.material.opacity = 0.1;
ghost.material.depthWrite = false;
ghost.bones.forEach(b => { b.mesh.castShadow = false; });
ghost.head.castShadow = false;
let ghostPose = null;
let shown = clonePose(STAND);
let shownP = rig.positions(shown);

const HANDLE_PARTS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head', 'Hips'];
const handleMat = new THREE.MeshBasicMaterial({ color: 0xffe08a, depthTest: false, transparent: true, opacity: 0.9 });
const handles = HANDLE_PARTS.map(part => {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(part === 'Hips' ? 0.075 : 0.06, 16, 12), handleMat.clone());
  mesh.renderOrder = 10;
  scene.add(mesh);
  return { part, mesh };
});
// gaze knob colours: cyan = in front of the face (follows the body), pink = fixed in space
const LOOK_FRONT = 0x7fe3ff, LOOK_FIXED = 0xff7ad9;
const lookKnob = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: LOOK_FRONT, depthTest: false, transparent: true, opacity: 0.85 }));
lookKnob.renderOrder = 10;
const lookLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new Vector3(), new Vector3()]),
  new THREE.LineDashedMaterial({ color: LOOK_FRONT, dashSize: 0.06, gapSize: 0.05, depthTest: false, transparent: true, opacity: 0.6 }));
lookLine.renderOrder = 9;
scene.add(lookKnob, lookLine);
const band = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new Vector3(), new Vector3()]),
  new THREE.LineBasicMaterial({ color: 0xffe08a, depthTest: false, transparent: true, opacity: 0.8 }));
band.renderOrder = 11;
band.visible = false;
const grip = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
grip.renderOrder = 12;
grip.visible = false;
scene.add(band, grip);

// ---------------------------------------------------------------- dragging

const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let drag = null; // { part, point, plane, ground, screen }
let lookDrag = null;

function setRay(x, y) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
}

/** Vertical plane facing the camera through p, or (with Shift) the horizontal plane at p's height. */
function dragPlane(p, ground) {
  const n = ground ? UP.clone() : camera.getWorldDirection(new Vector3()).setY(0).normalize().negate();
  return new THREE.Plane().setFromNormalAndCoplanarPoint(n, p);
}

function pointOn(d) {
  setRay(d.screen.x, d.screen.y);
  // the plane moves along with the body (so the pull never runs out while the camera follows)
  const anchor = d.ground ? new Vector3(0, d.height, 0) : shownP[0].clone().setY(d.height);
  const plane = dragPlane(d.ground ? anchor : anchor.add(d.offset), d.ground);
  const out = new Vector3();
  if (!ray.ray.intersectPlane(plane, out)) return null;
  out.y = clamp(out.y, 0.02, 2.2);
  return out;
}

canvas.addEventListener('pointerdown', e => {
  setRay(e.clientX, e.clientY);
  if (ray.intersectObject(lookKnob).length) {
    lookDrag = { screen: { x: e.clientX, y: e.clientY }, height: lookKnob.position.y, ground: e.shiftKey, offset: flat(lookKnob.position.clone().sub(shownP[0])) };
    look.detached = true;
  } else {
    const hit = ray.intersectObjects(handles.map(h => h.mesh))[0];
    if (!hit) return;
    const h = handles.find(x => x.mesh === hit.object);
    const pos = h.mesh.position.clone();
    drag = { part: h.part, point: pos.clone(), sent: null, screen: { x: e.clientX, y: e.clientY }, height: pos.y, ground: e.shiftKey, offset: new Vector3() };
    // keep the plane's offset from the hips along the camera's view so it starts through the grabbed part
    const view = camera.getWorldDirection(new Vector3()).setY(0).normalize();
    drag.offset = view.multiplyScalar(flat(pos.clone().sub(shownP[0])).dot(view));
  }
  canvas.setPointerCapture(e.pointerId);
  e.stopImmediatePropagation();
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => {
  if (drag) { drag.screen = { x: e.clientX, y: e.clientY }; drag.ground = e.shiftKey; return; }
  if (lookDrag) { lookDrag.screen = { x: e.clientX, y: e.clientY }; return; }
  setRay(e.clientX, e.clientY);
  const hit = ray.intersectObjects([...handles.map(h => h.mesh), lookKnob])[0];
  for (const h of handles) h.mesh.scale.setScalar(hit?.object === h.mesh ? 1.5 : 1);
  lookKnob.scale.setScalar(hit?.object === lookKnob ? 1.4 : 1);
  canvas.style.cursor = hit ? 'grab' : '';
});
const release = () => {
  if (drag) planTimer = 0;
  drag = null;
  lookDrag = null;
  canvas.style.cursor = '';
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);
canvas.addEventListener('dblclick', e => {
  setRay(e.clientX, e.clientY);
  if (ray.intersectObject(lookKnob).length) look.detached = false;
});

const controls = new OrbitControls(camera, canvas);
camera.position.set(0.8, 2.0, 5.5);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 1.5;
controls.maxDistance = 14;
controls.update();

// ---------------------------------------------------------------- on-screen layer

// per-joint springs between the generated motion and the cascaded pose: the body lags, overshoots a little, settles
const springS = rig.names.map(() => new Vector3()), springV = rig.names.map(() => new Vector3());
const rootS = new Vector3(), rootV = new Vector3();
const STIFF = rig.names.map(n => (/Hips/.test(n) ? 90 : /Spine|Chest/.test(n) ? 120 : /Neck|Head|Jaw|Eye/.test(n) ? 150 : /Leg|Shin|Foot|Toe/.test(n) ? 320 : 240));

function springLayer(base, goal, dt) {
  const Lb = toLocal(base), Lg = toLocal(goal);
  const L = Lb.map((q, j) => {
    const t = logq(q.clone().invert().multiply(Lg[j]));
    const k = STIFF[j], c = 2 * Math.sqrt(k) * 0.55;
    springV[j].addScaledVector(t.sub(springS[j]).multiplyScalar(k).sub(springV[j].clone().multiplyScalar(c)), dt);
    springS[j].addScaledVector(springV[j], dt);
    if (springS[j].length() > 1.6) springS[j].setLength(1.6);
    return q.clone().multiply(expq(springS[j]));
  });
  const tr = goal.root.clone().sub(base.root);
  rootV.addScaledVector(tr.sub(rootS).multiplyScalar(90).sub(rootV.clone().multiplyScalar(2 * Math.sqrt(90) * 0.55)), dt);
  rootS.addScaledVector(rootV, dt);
  return fromLocal(base.root.clone().add(rootS), L);
}

const locks = { Left: null, Right: null };
let clock = 0;

function overlay(base, contact, dt) {
  clock += dt;
  const walking = ctrl.vel.length() > 0.3;
  const goal = drag ? cascade(base, drag.part, drag.point, walking) : base;
  let p = springLayer(base, goal, dt);
  let P = rig.positions(p);
  // planted feet stay where they landed
  for (const side of ['Left', 'Right']) {
    const c = contact ? Math.max(...(side === 'Left' ? contact.slice(0, 2) : contact.slice(2, 4))) : 1;
    const foot = I[side + 'Foot'];
    if (!settings.footlock || c < 0.5 || drag?.part === side + 'Foot' || P[foot].y > ANKLE + 0.06) { locks[side] = null; continue; }
    if (!locks[side] || Math.hypot(locks[side].x - P[foot].x, locks[side].z - P[foot].z) > 0.2) { locks[side] = flat(P[foot]); continue; }
    const c3 = chain(side, LEG);
    const keep = p.rot[foot].clone();
    const knee = P[c3[1]].clone().sub(P[c3[0]].clone().lerp(P[c3[2]], 0.5));
    twoBoneIK(rig, p, c3[0], c3[1], c3[2], new Vector3(locks[side].x, P[foot].y, locks[side].z), knee.lengthSq() > 1e-6 ? knee : facing(P));
    rotateSubtree(rig, p, foot, keep.multiply(p.rot[foot].clone().invert()));
    P = rig.positions(p);
  }
  // the grabbed hand or foot is exactly on the pointer (as far as it reaches)
  if (drag && (drag.part.endsWith('Hand') || drag.part.endsWith('Foot'))) {
    const isHand = drag.part.endsWith('Hand');
    const c3 = chain(sideOf(drag.part), isHand ? ARM : LEG);
    const reach = reachOf(P, c3) * 0.995;
    const tip = drag.point.clone();
    if (!isHand) tip.y = clamp(tip.y, ANKLE, P[c3[0]].y - 0.12);
    const to = tip.clone().sub(P[c3[0]]);
    if (to.length() > reach) tip.copy(P[c3[0]]).addScaledVector(to.normalize(), reach);
    const mid = P[c3[1]].clone().sub(P[c3[0]].clone().lerp(P[c3[2]], 0.5));
    twoBoneIK(rig, p, c3[0], c3[1], c3[2], tip, mid.lengthSq() > 1e-6 ? mid : new Vector3(0, -1, 0));
  }
  // breathing
  const breath = Math.sin(clock * Math.PI * 2 / 4.2);
  const left = new Vector3().crossVectors(UP, facing(P));
  rotateSubtree(rig, p, I.Chest, new Quaternion().setFromAxisAngle(left, -0.018 * breath));
  rotateSubtree(rig, p, I.LeftShoulder, new Quaternion().setFromAxisAngle(facing(P), 0.012 * breath));
  rotateSubtree(rig, p, I.RightShoulder, new Quaternion().setFromAxisAngle(facing(P), -0.012 * breath));
  // gaze: the knob, or the point being pulled
  const target = look.detached || !drag ? look.point : drag.point;
  look.smooth.lerp(target, 1 - Math.exp(-dt * 6));
  gaze(p, look.smooth);
  return rig.fixRigid(p);
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
let planTimer = 0, idleTimer = 3, turnTimer = 0;

function simulate(dt) {
  if (drag && !drag.scripted) { const pt = pointOn(drag); if (pt) drag.point.copy(pt); }
  if (lookDrag) { const pt = pointOn(lookDrag); if (pt) look.point.copy(pt); }
  if (!look.detached) look.point.copy(shownP[I.Head]).add(facing(shownP).multiplyScalar(1.2)).setY(shownP[I.Head].y - 0.05);
  ctrl.want.copy(pullVelocity(shownP));
  approach(ctrl.vel, ctrl.want, dt);
  planTimer -= dt; idleTimer -= dt; turnTimer -= dt;
  if (scheduler.ready && !track.busy) {
    const moving = ctrl.vel.length() > 0.05 || ctrl.want.length() > 0.05;
    const pointerMoved = drag && (!drag.sent || drag.sent.distanceTo(drag.point) > 0.04);
    if ((moving || pointerMoved) && planTimer <= 0) {
      if (drag) drag.sent = drag.point.clone();
      track.request(locoReq('now'));
      planTimer = 0.18 + 0.12 * Math.min(1, ctrl.stretch);
    } else if (!moving && !drag) {
      const h = rig.heading(shownP);
      const d = Math.abs(wrap(Math.atan2(look.point.x - shownP[0].x, look.point.z - shownP[0].z) - h));
      if (look.detached && d > 1.3 && turnTimer <= 0) { track.request(idleReq('now')); turnTimer = 1.2; }
      else if (track.remaining / FPS < scheduler.latency + 0.3) track.request(idleReq('end'));
      else if (idleTimer <= 0) { track.request(idleReq('now', true)); idleTimer = rand(2.5, 5); }
    }
  }
  track.advance(dt);
  scheduler.tick([track]);
  const f = track.frames[Math.min(track.frames.length - 1, Math.floor(track.t))].pose;
  shown = overlay(track.display, f.contact, dt);
  shownP = rig.positions(shown);
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  body.update(shownP, rig.extraPositions(shown, shownP), dt);
  if (ghostPose) {
    const P = rig.positions(ghostPose);
    ghost.update(P, rig.extraPositions(ghostPose, P), dt);
  }
  ghost.group.visible = !!ghostPose;
  for (const h of handles) h.mesh.position.copy(shownP[I[h.part]]);
  lookKnob.position.copy(look.point);
  const lookColor = look.detached ? LOOK_FIXED : LOOK_FRONT;
  if (lookKnob.material.color.getHex() !== lookColor) {
    lookKnob.material.color.setHex(lookColor);
    lookLine.material.color.setHex(lookColor);
    const el = $('look-mode');
    el.textContent = look.detached ? '空間に固定' : '正面（体についてくる）';
    el.className = look.detached ? 'fixed' : 'front';
  }
  lookLine.geometry.setFromPoints([shownP[I.Head], look.point]);
  lookLine.computeLineDistances();
  band.visible = grip.visible = !!drag;
  if (drag) {
    band.geometry.setFromPoints([shownP[I[drag.part]], drag.point]);
    grip.position.copy(drag.point);
  }
  // the camera follows the body (faster while it walks), the floor goes on forever
  const hips = shownP[0];
  const k = 1 - Math.exp(-dt * (ctrl.vel.length() > 0.3 ? 4 : 2));
  const move = new Vector3(hips.x, 0.9, hips.z).sub(controls.target).multiplyScalar(k);
  controls.target.add(move);
  camera.position.add(move);
  controls.update();
  stage.followFloor(hips);
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hud(); }
  requestAnimationFrame(frame);
}

function hud() {
  if (!scheduler.ready) return;
  $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms · 速さ ${ctrl.vel.length().toFixed(1)} m/s · 描画 ${fps.toFixed(0)} fps`;
}

// ---------------------------------------------------------------- model loading

function warmup(id) {
  const batch = emptyBatch(BATCH);
  const ctx = track.context(track.end);
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
    } else if (data.phase === 'cache') {
      bar.style.width = '100%'; label.textContent = '保存したモデルを読み込み中…';
    } else {
      bar.style.width = '100%'; label.textContent = '推論の準備中（初回はシェーダーのコンパイルで数十秒かかることがあります）…';
    }
  } else if (data.type === 'ready') {
    scheduler.backend = data.backend;
    label.textContent = `${data.backend} で準備完了。最初の生成を待っています…`;
    warmup(-1);
  } else if (data.type === 'result') {
    if (data.id === -1) warmup(0);
    else if (data.id === 0) {
      scheduler.latency = Math.max(0.05, data.ms / 1000);
      scheduler.ready = true;
      $('overlay').classList.add('hidden');
    } else scheduler.onResult(data);
  } else if (data.type === 'error') {
    if (data.id) scheduler.onFail(data.id);
    label.textContent = `エラー: ${data.message}`;
    $('stats').textContent = `エラー: ${data.message}`;
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
$('home').addEventListener('click', () => {
  drag = null;
  ctrl.vel.set(0, 0, 0);
  track.reset(new Vector3(0, 0, 0), 0);
  look.detached = false;
  locks.Left = locks.Right = null;
  for (const v of [...springS, ...springV, rootS, rootV]) v.set(0, 0, 0);
});
// the guide can be folded away; the choice is remembered on this browser
let guidePref = null;
try { guidePref = localStorage.getItem('soma-pull-guide'); } catch { /* no storage */ }
if (guidePref === 'closed' || (guidePref === null && innerWidth < 760)) $('guide').classList.add('closed'); // narrow screens start folded
$('guide-toggle').addEventListener('click', () => {
  const closed = $('guide').classList.toggle('closed');
  $('guide-toggle').setAttribute('aria-label', closed ? '操作説明を開く' : '操作説明を閉じる');
  try { localStorage.setItem('soma-pull-guide', closed ? 'closed' : 'open'); } catch { /* no storage */ }
});
if (!navigator.gpu) $('gpu-note').textContent = 'このブラウザでは WebGPU が使えません。WASM（CPU）で動きますが、かなり遅くなります。';

showLibrary();
track.reset(new Vector3(0, 0, 0), 0);
requestAnimationFrame(frame);
if (params.has('debug')) {
  window.lab = {
    track, scheduler, simulate, settings, ctrl, library, rig, stage, look, cascade,
    get shownP() { return shownP; }, get drag() { return drag; },
    // a scripted pull: `point` is used as is (no screen ray)
    pull(part, point) { drag = part ? { part, point, sent: null, scripted: true } : null; if (!part) planTimer = 0; },
    draw: () => frame(performance.now()),
  };
}
