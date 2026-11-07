// SOMA Pull MM: motion matching over Mixamo reference poses, with every transition made by MotionBricks.
//
// Loop (about every 0.25 s, at once when the pull changes):
// 1. The pull (a leash on the grabbed part) gives a wanted ground velocity; from it the hips' path 1/3, 2/3 and 1 s
//    ahead is predicted (velocity easing toward the wanted one, facing turning toward travel).
// 2. A query is made from what SOMA is doing at the plan's start (its context frames) plus that path and where the
//    grabbed part should be 0.8 s from now, all in the body frame (hips on the floor, facing forward).
// 3. The best reference moment in the pose database (src/posedb.js) is found; staying on the clip it is already
//    following is preferred unless something clearly better comes up.
// 4. The four frames 0.8 s after that moment, placed so the matched moment lines up with the body now, are the
//    targets; MotionBricks makes the motion there. The screen shows MotionBricks' output only.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, clonePose, emptyBatch, encodeRequest, FPS } from './soma.js';
import { Track, Scheduler, BATCH } from './motion.js';
import { rotateSubtree, twoBoneIK } from './poses.js';
import { Stage, Mannequin } from './render.js';
import { PoseDB, queryFeatures, weightVector, meanHeading, handleSlot, AHEAD, TRAJ, DIM } from './posedb.js';

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

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const I = rig.index;
const canvas = $('view');
const stage = new Stage(canvas);
stage.makeEndless();
const scene = stage.scene, camera = stage.camera;
const worker = new Worker(new URL('./infer.worker.js', import.meta.url), { type: 'module' });
const scheduler = new Scheduler(rig, worker);

// tunable (the evaluator searches over these)
const settings = {
  vmax: 2.2, stiff: 2.6,
  replan: 0.25, // s between plans while something is going on
  stay: 0.85, // keep following the current clip unless the best is cheaper than this fraction of its cost
  calm: { passive: 2.0, active: 0.4 },
  fix: 'auto', // how hard the target holds at its end: 'auto' (whole pose standing, hips only moving), true, 'root', false
  targets: 4, // consecutive target frames given (1 = arrive and stop, 2 = speed, 3-4 = acceleration)
  pose: 'mixed', // what of the reference pose is given: 'full', 'free' (only where the hips go and which way they face),
  // or 'mixed' (the pose on the first target only, the rest carry the hips' path; free while nothing is going on) // cost per metre of body-shape change over the next 0.8 s (not asked for)
  w: { feetPos: 1, feetVel: 0.75, hipsVel: 1, hipsH: 0.7, hipsVy: 0.8, hands: 0.3, head: 0.3, handsVel: 0.1, trajPos: 1, trajDir: 0.7, handle: 1.5, trajH: 1 },
};
for (const [id, key, fmt] of [['vmax', 'vmax', v => v.toFixed(1)], ['stiff', 'stiff', v => v.toFixed(1)]]) {
  const el = $(id);
  const read = () => { settings[key] = Number(el.value); $(`${id}-v`).textContent = fmt(settings[key]); };
  el.addEventListener('input', read);
  read();
}

let db = null, dbInfo = '読み込み中…', change = null;
const entryAt = new Map(); // `${clip}:${mirror}:${frame}` -> entry index
PoseDB.load(rig, new URL('../assets/posedb', import.meta.url).href).then(d => {
  db = d;
  for (let i = 0; i < d.n; i++) entryAt.set(`${d.entClip[i]}:${d.entMirror[i]}:${d.entFrame[i]}`, i);
  dbInfo = `${d.clips.length} クリップ / ${d.n} 件`;
  // how much each entry's body shape changes by its target (hands, feet, head relative to the hips): a reference
  // moment that is about to do something big is only worth taking when the pull asks for it
  change = new Float32Array(d.n);
  for (let i = 0; i < d.n; i++) {
    const x = d.entryFeatures(i);
    const hipsA = [x[58], x[59], x[60]], hipsNow = [0, x[14], 0];
    const cur = { LeftHand: [16, 17, 18], RightHand: [19, 20, 21], LeftFoot: [0, 1, 2], RightFoot: [3, 4, 5], Head: [22, 23, 24] };
    let c = 0;
    ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head'].forEach((n, k) => {
      const a = 43 + 3 * k, b = cur[n];
      c += Math.hypot((x[a] - hipsA[0]) - (x[b[0]] - hipsNow[0]), (x[a + 1] - hipsA[1]) - (x[b[1]] - hipsNow[1]), (x[a + 2] - hipsA[2]) - (x[b[2]] - hipsNow[2]));
    });
    change[i] = c;
  }
  $('db-info').textContent = dbInfo;
  // start from a standing idle of the database (not the rest T pose)
  const idle = d.clips.findIndex(c => /^(standing_idle|breathing_idle|idle)/.test(c.name));
  if (idle >= 0) { track.book.stance = d.pose(idle, 10); current = null; track.reset(flat(shownP[0]), 0); }
}).catch(e => { dbInfo = `読み込めません: ${e.message}`; $('db-info').textContent = dbInfo; });

// ---------------------------------------------------------------- the leash

const ctrl = { vel: new Vector3(), want: new Vector3(), stretch: 0 };
const SLACK = { LeftHand: 0.45, RightHand: 0.45, Hips: 0.1, Head: 0.25, LeftFoot: 0.6, RightFoot: 0.6 };
const ANCHOR = { LeftHand: 'LeftArm', RightHand: 'RightArm', Hips: 'Hips', Head: 'Head', LeftFoot: 'LeftLeg', RightFoot: 'RightLeg' };
function pullVelocity(P) {
  ctrl.stretch = 0;
  if (!drag) return new Vector3();
  const d = flat(drag.point.clone().sub(P[I[ANCHOR[drag.part]]]));
  const stretch = d.length() - SLACK[drag.part];
  ctrl.stretch = Math.max(0, stretch);
  if (stretch <= 0) return new Vector3();
  const cap = settings.vmax;
  return d.normalize().multiplyScalar(cap * (1 - Math.exp(-stretch * settings.stiff / cap)));
}
function approach(v, want, dt) {
  const s = v.length(), w = want.length();
  if (s > 0.2 && w > 0.2) {
    const a = Math.atan2(v.x, v.z), b = Math.atan2(want.x, want.z);
    const heading = a + clamp(wrap(b - a), -4 * dt, 4 * dt);
    const speed = s + clamp(w * Math.cos(clamp(wrap(b - a), -Math.PI / 2, Math.PI / 2)) - s, -3 * dt, 2 * dt);
    return v.set(Math.sin(heading) * speed, 0, Math.cos(heading) * speed);
  }
  const diff = want.clone().sub(v);
  const acc = (w < s ? 3 : 2) * dt;
  if (diff.length() > acc) diff.setLength(acc);
  return v.add(diff);
}
/** Hips ground path from a pose: [{pos, yaw}] at TRAJ frames ahead. */
function predict(pose, heading) {
  let yaw = heading;
  const p = flat(pose.root), v = ctrl.vel.clone();
  const out = [];
  for (let k = 1; k <= TRAJ.at(-1); k++) {
    approach(v, ctrl.want, 1 / FPS);
    p.addScaledVector(v, 1 / FPS);
    const s = v.length();
    if (s > 0.3) {
      const travel = Math.atan2(v.x, v.z), c = Math.cos(travel - yaw);
      // backing up or a slow side step keeps the facing
      if (!(c < -0.3 && s < 1.3) && !(Math.abs(c) < 0.5 && s < 0.9)) yaw += clamp(wrap(travel - yaw), -3.5 / FPS, 3.5 / FPS);
    }
    if (TRAJ.includes(k)) out.push({ pos: p.clone(), yaw });
  }
  return out;
}

// ---------------------------------------------------------------- feet

/** Bend a leg so its ankle reaches `goal`, keeping the foot's world orientation. */
function legTo(p, side, goal) {
  const c = ['Leg', 'Shin', 'Foot'].map(n => I[side + n]);
  const P = rig.positions(p);
  const keep = p.rot[c[2]].clone();
  const knee = P[c[1]].clone().sub(P[c[0]].clone().lerp(P[c[2]], 0.5));
  const fwd = new Vector3(Math.sin(rig.heading(P)), 0, Math.cos(rig.heading(P)));
  twoBoneIK(rig, p, c[0], c[1], c[2], goal, knee.lengthSq() > 1e-6 ? knee.add(fwd.multiplyScalar(0.3)) : fwd);
  rotateSubtree(rig, p, c[2], keep.multiply(p.rot[c[2]].clone().invert()));
}
/**
 * Which feet stay planted in the reference from the matched frame to the target (low and still the whole time):
 * those keep the place they have on the floor now, so SOMA does not shuffle to match where the reference stood.
 */
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
const ANKLE = rig.restPos[I.LeftFoot].y;

/**
 * The grabbed hand or foot onto the pointer, on the posed target only (the reference rarely has exactly that
 * reach; a held-up foot hardly at all). Within the limb's reach, the elbow / knee bending the way it already does.
 */
function reachLimb(p, part, goal) {
  const side = part.startsWith('Left') ? 'Left' : 'Right';
  const names = part.endsWith('Hand') ? ['Arm', 'ForeArm', 'Hand'] : ['Leg', 'Shin', 'Foot'];
  const c = names.map(n => I[side + n]);
  const q = clonePose(p);
  const P = rig.positions(q);
  const len = P[c[1]].distanceTo(P[c[0]]) + P[c[2]].distanceTo(P[c[1]]);
  const to = goal.clone().sub(P[c[0]]);
  const tip = to.length() > len * 0.98 ? P[c[0]].clone().addScaledVector(to.normalize(), len * 0.98) : goal.clone();
  if (part.endsWith('Foot')) tip.y = Math.max(tip.y, ANKLE);
  const bend = P[c[1]].clone().sub(P[c[0]].clone().lerp(P[c[2]], 0.5));
  const keep = q.rot[c[2]].clone();
  twoBoneIK(rig, q, c[0], c[1], c[2], tip, bend.lengthSq() > 1e-6 ? bend : new Vector3(0, part.endsWith('Hand') ? -1 : 0.5, 1));
  if (part.endsWith('Foot')) rotateSubtree(rig, q, c[2], keep.multiply(q.rot[c[2]].clone().invert()));
  return rig.fixRigid(q);
}

// ---------------------------------------------------------------- matching

const recent = []; // clips left recently ({ clip, t })
let current = null; // { clip, mirror, frame, anchor } the clip being followed (frame at the anchor of the last plan)
let lastMatch = null, leadPoint = null;
const qbuf = new Float32Array(DIM);
// While a hand is pulled the other one should stay about where it is (the reference otherwise happily lifts both);
// while a foot is pulled the other one stays planted where it stands.
const OTHER = { LeftHand: ['RightHand', 0.8], RightHand: ['LeftHand', 0.8], LeftFoot: ['RightFoot', 1.5], RightFoot: ['LeftFoot', 1.5] };
function weightsFor(part) {
  const w = weightVector(settings.w);
  w.fill(0, 43, 61);
  if (part) for (let d = 0; d < 3; d++) w[handleSlot(part) + d] = settings.w.handle;
  if (OTHER[part]) for (let d = 0; d < 3; d++) w[handleSlot(OTHER[part][0]) + d] = OTHER[part][1];
  return w;
}
const TOE = rig.restPos[I.LeftToeBase].y;
/**
 * Put a target on the floor: its lowest point (an ankle or toe against its height at rest, any other joint 5 cm
 * above its centre) touches the floor. Lifting is left alone while running (a stride has a flight phase).
 */
function ground(t, running) {
  const P = rig.positions(t);
  let low = Math.min(P[I.LeftFoot].y - ANKLE, P[I.RightFoot].y - ANKLE, P[I.LeftToeBase].y - TOE, P[I.RightToeBase].y - TOE);
  for (let j = 0; j < P.length; j++) if (!/Foot|Toe/.test(rig.names[j])) low = Math.min(low, P[j].y - 0.05);
  if (low < -0.01 || (low > 0.02 && !running)) { const q = clonePose(t); q.root.y -= low; return rig.fixRigid(q); }
  return t;
}
const goalW = weightVector({ trajPos: 1, trajDir: 0.7, trajH: 1 });
function costOf(i, q, w) {
  const F = db.feat, s = db.scale, k = 1 / db.q;
  let c = 0;
  for (let d = 0; d < DIM; d++) { if (!w[d]) continue; const e = F[i * DIM + d] * k - q[d] / s[d]; c += w[d] * e * e; }
  return c;
}

function mmReq(mode = 'now') {
  return {
    kind: 'move', mode,
    build(tr, a, ctx) {
      const from = ctx[3];
      const Pn = rig.positions(from), Pp = rig.positions(ctx[2]);
      const heading = meanHeading(ctx.map(p => rig.heading(rig.positions(p))));
      const traj = predict(from, heading);
      let target = null;
      if (drag) {
        leadPoint = drag.point.clone();
        if (drag.vel) leadPoint.add(drag.vel.clone().multiplyScalar(scheduler.latency + AHEAD / FPS).clampLength(0, 0.45));
        leadPoint.y = Math.max(0.05, leadPoint.y);
        target = leadPoint;
      }
      const extra = drag && OTHER[drag.part] ? [{ part: OTHER[drag.part][0], target: Pn[I[OTHER[drag.part][0]]].clone() }] : [];
      const q = queryFeatures(rig, Pp, Pn, heading, traj, drag?.part ?? null, target, qbuf, extra);
      const w = weightsFor(drag?.part ?? null);
      if (!drag && ctrl.vel.length() < 0.05) for (let d = 25; d < 31; d++) w[d] = 0.6;
      // where the followed clip has got to: a plan made at the end of the motion starts from its target frame, one
      // made in the middle from as far as time has gone
      let cont = null;
      if (current) {
        const f = current.frame + (mode === 'end' ? AHEAD : a - current.anchor);
        const fe = f - (f % 2);
        const i = entryAt.get(`${current.clip}:${current.mirror ? 1 : 0}:${fe}`);
        if (i != null) cont = { i, frame: fe };
      }
      // no going back a little way in the same clip (that replays the same piece over and over), and a cost for
      // a big change of body shape that nothing asks for
      const passive = !drag && ctrl.vel.length() < 0.05;
      const calm = passive ? settings.calm.passive : settings.calm.active;
      // (a clip and its mirror image count as the same; a clip used in the last 1.5 s costs more to jump back into,
      // except carrying on from where it is)
      const back = i => (cont && db.entClip[i] === current.clip && db.entFrame[i] < cont.frame && db.entFrame[i] > cont.frame - 40 ? 4 : 0);
      const recentSet = new Set(recent.filter(r => clock - r.t < 1.5).map(r => r.clip));
      const again = i => (recentSet.has(db.entClip[i]) && !(cont && db.entClip[i] === current.clip && Math.abs(db.entFrame[i] - cont.frame) <= 4) ? 2 : 0);
      const bias = i => back(i) + again(i) + calm * change[i];
      let best = db.search(q, w, bias);
      if (cont) {
        const c = costOf(cont.i, q, w) + calm * change[cont.i];
        // nothing asked for: carry on with the clip while its future still keeps still; otherwise stay unless clearly worse
        if ((passive && costOf(cont.i, q, goalW) < 1.5) || best.cost > settings.stay * c) best = { i: cont.i, cost: c, clip: current.clip, frame: cont.frame, mirrored: current.mirror, stayed: true };
      }
      if (!current || current.clip !== best.clip) recent.push({ clip: current?.clip ?? best.clip, t: clock });
      if (recent.length > 20) recent.shift();
      current = { clip: best.clip, mirror: best.mirrored, frame: best.frame, anchor: a };
      lastMatch = { ...best, name: db.clips[best.clip].name };
      // line the matched moment up with the body now, then take the frames AHEAD later
      const at = db.pose(best.clip, best.frame, best.mirrored);
      const hAt = meanHeading([-2, -1, 0, 1, 2].map(o => rig.heading(rig.positions(db.pose(best.clip, best.frame + o, best.mirrored)))));
      const yaw = wrap(heading - hAt);
      const pivot = flat(at.root), offset = flat(from.root).sub(pivot);
      let targets = [0, 1, 2, 3].map(k => placePose(db.pose(best.clip, best.frame + AHEAD + k, best.mirrored), yaw, pivot, offset));
      // on the floor (reference floors differ a little, and lying or kneeling must not sink or float)
      const running = traj.at(-1).pos.distanceTo(flat(from.root)) / (TRAJ.at(-1) / FPS) > 1.8;
      targets = targets.map(t => ground(t, running));
      // feet that stay planted in the reference keep their place on the floor now
      const keep = plantedThrough(best.clip, best.frame, best.frame + AHEAD + 3, best.mirrored);
      for (const side of ['Left', 'Right']) {
        if (!keep[side]) continue;
        const now = Pn[I[side + 'Foot']].clone();
        targets = targets.map(t => { const q = clonePose(t); legTo(q, side, new Vector3(now.x, Math.max(now.y, ANKLE), now.z)); return rig.fixRigid(q); });
      }
      if (drag && target && drag.part !== 'Hips' && drag.part !== 'Head') targets[0] = reachLimb(targets[0], drag.part, target);
      ghostPose = targets[0];
      const moving = traj.at(-1).pos.distanceTo(flat(from.root)) > 0.25;
      const fixEnd = settings.fix === 'auto' ? (moving ? 'root' : true) : settings.fix;
      let out = targets.slice(0, settings.targets);
      const freeAll = settings.pose === 'free' || (settings.pose === 'mixed' && passive);
      if (freeAll) out = out.map(t => { const q = clonePose(t); q.free = true; return q; });
      else if (settings.pose === 'mixed') out = out.map((t, k) => { if (k === 0) return t; const q = clonePose(t); q.free = true; return q; });
      return { targets: out, tokens: [6, 7], rate: 1, fixEnd: freeAll ? false : fixEnd, tag: { kind: moving ? 'move' : 'idle' } };
    },
  };
}

// ---------------------------------------------------------------- scene

const track = new Track(rig, { stance: rig.restPose() });
const body = new Mannequin(scene, rig, 0xd9b26a);
const ghost = new Mannequin(scene, rig, 0x8fb4ff);
ghost.material.transparent = true; ghost.material.opacity = 0.12; ghost.material.depthWrite = false;
ghost.bones.forEach(b => { b.mesh.castShadow = false; }); ghost.head.castShadow = false;
let ghostPose = null;
let shown = clonePose(track.display), shownP = rig.positions(shown);

const HANDLE_PARTS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head', 'Hips'];
const handleMat = new THREE.MeshBasicMaterial({ color: 0xffe08a, depthTest: false, transparent: true, opacity: 0.9 });
const handles = HANDLE_PARTS.map(part => {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(part === 'Hips' ? 0.075 : 0.06, 16, 12), handleMat.clone());
  mesh.renderOrder = 10;
  scene.add(mesh);
  return { part, mesh };
});
const band = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new Vector3(), new Vector3()]), new THREE.LineBasicMaterial({ color: 0xffe08a, depthTest: false, transparent: true, opacity: 0.8 }));
band.renderOrder = 11; band.visible = false;
const grip = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
grip.renderOrder = 12; grip.visible = false;
scene.add(band, grip);

// ---------------------------------------------------------------- dragging

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let drag = null;
function setRay(x, y) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
}
function pointOn(d) {
  setRay(d.screen.x, d.screen.y);
  const anchor = d.ground ? new Vector3(0, d.height, 0) : shownP[0].clone().setY(d.height).add(d.offset);
  const n = d.ground ? UP.clone() : camera.getWorldDirection(new Vector3()).setY(0).normalize().negate();
  const out = new Vector3();
  if (!ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, anchor), out)) return null;
  out.y = clamp(out.y, 0.02, 2.2);
  return out;
}
canvas.addEventListener('pointerdown', e => {
  setRay(e.clientX, e.clientY);
  const hit = ray.intersectObjects(handles.map(h => h.mesh))[0];
  if (!hit) return;
  const h = handles.find(x => x.mesh === hit.object);
  const pos = h.mesh.position.clone();
  const view = camera.getWorldDirection(new Vector3()).setY(0).normalize();
  drag = { part: h.part, point: pos.clone(), screen: { x: e.clientX, y: e.clientY }, height: pos.y, ground: e.shiftKey, offset: view.multiplyScalar(flat(pos.clone().sub(shownP[0])).dot(view)) };
  planTimer = 0;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
  e.stopImmediatePropagation(); e.preventDefault();
});
canvas.addEventListener('pointermove', e => {
  if (drag) { drag.screen = { x: e.clientX, y: e.clientY }; drag.ground = e.shiftKey; return; }
  setRay(e.clientX, e.clientY);
  const hit = ray.intersectObjects(handles.map(h => h.mesh))[0];
  for (const h of handles) h.mesh.scale.setScalar(hit?.object === h.mesh ? 1.5 : 1);
  canvas.style.cursor = hit ? 'grab' : '';
});
const release = () => { if (drag) planTimer = 0; drag = null; canvas.style.cursor = ''; };
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

const controls = new OrbitControls(camera, canvas);
camera.position.set(0.8, 2.0, 5.5);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 1.5; controls.maxDistance = 14;
controls.update();

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0, clock = 0, planTimer = 0, lastWant = new Vector3(), lastPoint = null;
function simulate(dt) {
  if (drag && !drag.scripted) { const pt = pointOn(drag); if (pt) drag.point.copy(pt); }
  if (drag) {
    drag.vel ??= new Vector3();
    if (drag.prev) drag.vel.lerp(drag.point.clone().sub(drag.prev).divideScalar(dt), 1 - Math.exp(-dt * 10));
    drag.prev = drag.point.clone();
  }
  ctrl.want.copy(pullVelocity(shownP));
  approach(ctrl.vel, ctrl.want, dt);
  planTimer -= dt;
  if (scheduler.ready && db && !track.busy) {
    // the pull changed clearly: plan at once
    // (the pointer relative to the body: a pointer that just keeps leading a walking body is not a change)
    const rel = drag ? drag.point.clone().sub(shownP[I[ANCHOR[drag.part]]]) : null;
    const changed = ctrl.want.distanceTo(lastWant) > 0.5 || (rel && lastPoint && rel.distanceTo(lastPoint) > 0.25) || (!!drag !== !!lastPoint);
    const active = drag || ctrl.vel.length() > 0.05;
    if (active && (planTimer <= 0 || changed)) {
      track.request(mmReq('now'));
      planTimer = settings.replan;
      lastWant.copy(ctrl.want);
      lastPoint = rel;
    } else if (!active && track.remaining / FPS < scheduler.latency + 0.25) {
      // nothing going on: carry on from where the motion ends (the last target)
      track.request(mmReq('end'));
      planTimer = settings.replan;
      lastWant.copy(ctrl.want);
      lastPoint = drag ? drag.point.clone() : null;
    }
  }
  track.advance(dt);
  scheduler.tick([track]);
  const f = track.frames[Math.min(track.frames.length - 1, Math.floor(track.t))].pose;
  shown = clonePose(track.display);
  shown.contact = f.contact;
  shownP = rig.positions(shown);
  clock += dt;
  window.lab?.onStep?.(shown, dt);
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  body.update(shownP, rig.extraPositions(shown, shownP), dt);
  if (ghostPose) { const P = rig.positions(ghostPose); ghost.update(P, rig.extraPositions(ghostPose, P), dt); }
  ghost.group.visible = !!ghostPose && $('ghost').checked;
  for (const h of handles) h.mesh.position.copy(shownP[I[h.part]]);
  band.visible = grip.visible = !!drag;
  if (drag) { band.geometry.setFromPoints([shownP[I[drag.part]], drag.point]); grip.position.copy(drag.point); }
  const hips = shownP[0];
  const k = 1 - Math.exp(-dt * (ctrl.vel.length() > 0.3 ? 4 : 2));
  const move = new Vector3(hips.x, 0.9, hips.z).sub(controls.target).multiplyScalar(k);
  controls.target.add(move); camera.position.add(move);
  controls.update();
  stage.followFloor(hips);
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hud(); }
}
function loop(now) { frame(now); requestAnimationFrame(loop); }

function hud() {
  if (!scheduler.ready) return;
  $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms · 速さ ${ctrl.vel.length().toFixed(1)} m/s · 描画 ${fps.toFixed(0)} fps`;
  if (lastMatch) $('match').textContent = `${lastMatch.name}${lastMatch.mirrored ? '（左右反転）' : ''} f${lastMatch.frame} · コスト ${lastMatch.cost.toFixed(1)}${lastMatch.stayed ? ' · 継続' : ''}`;
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
$('home').addEventListener('click', () => { drag = null; ctrl.vel.set(0, 0, 0); current = null; track.reset(new Vector3(), 0); });

track.reset(new Vector3(), 0);
requestAnimationFrame(loop);
if (params.has('debug')) {
  window.lab = {
    rig, track, scheduler, settings, ctrl, simulate,
    get db() { return db; }, get lastMatch() { return lastMatch; }, get clock() { return clock; },
    get shownP() { return shownP; }, get shown() { return shown; }, get drag() { return drag; }, onStep: null,
    pull(part, point) { drag = part ? { part, point, scripted: true } : null; planTimer = 0; },
    draw: () => frame(performance.now()),
  };
}
