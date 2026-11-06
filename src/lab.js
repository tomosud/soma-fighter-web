// SOMA Pull: grab a hand, the hips, the head or a foot and pull; the body follows with human footwork.
//
// How it moves:
// 1. A leash: the grabbed point pulls the body only once the rope (as long as the arm, for a hand) is taut. The
//    stretch sets a wanted ground velocity, capped (walk/run speed) and reached with limited acceleration; the
//    hips never go up, so there is no flying.
// 2. MotionBricks is given four target poses about one second ahead on the predicted path, 1/30 s apart, so they
//    carry the speed and facing there. It then plans its own steps (walk, jog, run) instead of lunging and stopping.
//    It is re-planned every 0.3 s from where the motion will be when the result arrives, played at real time.
// 3. The target poses come from a gait library that grows from MotionBricks' own output: windows of four frames
//    at a steady speed, sorted by speed, direction and which foot is on the ground (the model's foot contacts).
// 4. On screen, the grabbed hand (or foot) is bent onto the pointer by IK over the generated motion, and feet the
//    model says are on the ground are held in place so they do not slide.
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
const AREA = 7; // the floor the body may walk on (radius)
const HORIZON = 1.0; // seconds ahead of the anchor the targets are placed
const REPLAN = 0.3; // seconds between plans while moving

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const I = rig.index;
const canvas = $('view');
const stage = new Stage(canvas);
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

// ---------------------------------------------------------------- poses

const rand = (a, b) => a + Math.random() * (b - a);
const ANKLE = rig.restPos[I.LeftFoot].y;
const ARM = ['Arm', 'ForeArm', 'Hand'], LEG = ['Leg', 'Shin', 'Foot'];
const chain = (side, parts) => parts.map(p => I[side + p]);
const reachOf = (P, [a, b, c]) => P[b].distanceTo(P[a]) + P[c].distanceTo(P[b]);

/** A relaxed standing pose (arms down, knees soft), facing +Z with the hips over the origin. */
function makeStand() {
  const p = rig.restPose();
  twoBoneIK(rig, p, I.LeftArm, I.LeftForeArm, I.LeftHand, new Vector3(0.24, 0.86, 0.04), new Vector3(0.3, 0, -1));
  twoBoneIK(rig, p, I.RightArm, I.RightForeArm, I.RightHand, new Vector3(-0.24, 0.86, 0.04), new Vector3(-0.3, 0, -1));
  p.root.y -= 0.02;
  twoBoneIK(rig, p, I.LeftLeg, I.LeftShin, I.LeftFoot, new Vector3(0.1, ANKLE, 0.02), new Vector3(0.1, 0, 1));
  twoBoneIK(rig, p, I.RightLeg, I.RightShin, I.RightFoot, new Vector3(-0.1, ANKLE, 0.02), new Vector3(-0.1, 0, 1));
  return rig.fixRigid(p);
}
const STAND = makeStand();
const HIPS_Y = STAND.root.y;

/** Hips over the origin, facing +Z; the same transform applied to a list of poses (keeps their relative motion). */
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
const library = new Map(); // `${bin}:${dir}:${phase}` -> [{frames, speed}]
const binOf = s => BINS.findIndex(b => s >= b.lo && s < b.hi);

function stanceFoot(contact) {
  if (!contact) return 'D';
  const l = Math.max(contact[0], contact[1]) > 0.5, r = Math.max(contact[2], contact[3]) > 0.5;
  return l && r ? 'D' : l ? 'L' : r ? 'R' : 'F';
}

/** Keep steady windows of generated locomotion: four frames, speed roughly constant, feet near the floor. */
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
    const P = rig.positions(w[0]);
    if (w.some(p => p.root.y < HIPS_Y - 0.14 || p.root.y > HIPS_Y + 0.08)) continue;
    if (w.some(p => { const Q = rig.positions(p); return Math.min(Q[I.LeftFoot].y, Q[I.RightFoot].y) > 0.22; })) continue;
    const h = rig.heading(P);
    const fwd = new Vector3(Math.sin(h), 0, Math.cos(h));
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
    const max = PER_KEY * 2 * 4;
    el.insertAdjacentHTML('beforeend', `<span>${b.name}</span><div class="bar"><div class="fill" style="width:${Math.min(100, (n / max) * 100)}%"></div></div><span class="n">${n}</span>`);
  });
}

// ---------------------------------------------------------------- the leash

const ctrl = { vel: new Vector3(), want: new Vector3() };
const SLACK = { LeftHand: 0.5, RightHand: 0.5, Hips: 0.06, Head: 0.25, LeftFoot: 0.9, RightFoot: 0.9 };
const ANCHOR = { LeftHand: 'LeftArm', RightHand: 'RightArm', Hips: 'Hips', Head: 'Head', LeftFoot: 'LeftLeg', RightFoot: 'RightLeg' };

/** Wanted ground velocity from the pull (m/s): zero while the rope is slack, then grows with the stretch. */
function pullVelocity(P) {
  if (!drag) return new Vector3();
  const from = P[I[ANCHOR[drag.part]]];
  const d = new Vector3(drag.point.x - from.x, 0, drag.point.z - from.z);
  const stretch = d.length() - SLACK[drag.part];
  if (stretch <= 0) return new Vector3();
  return d.normalize().multiplyScalar(Math.min(settings.vmax, stretch * settings.stiff));
}

function approach(v, want, dt) {
  const diff = want.clone().sub(v);
  const acc = (want.length() < v.length() ? 3.5 : 2.5) * dt; // m/s^2: start gently, stop a little faster
  if (diff.length() > acc) diff.setLength(acc);
  return v.add(diff);
}

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/**
 * Predict the hips' ground path from the anchor: velocity eases toward the wanted one, the facing turns toward
 * the travel direction (at most 3 rad/s), except that a light pull from behind is walked backward.
 * Returns the four target placements at HORIZON .. HORIZON + 3/30 s.
 */
function predict(anchorPose) {
  const P = rig.positions(anchorPose);
  let yaw = rig.heading(P);
  const p = new Vector3(anchorPose.root.x, 0, anchorPose.root.z);
  const v = ctrl.vel.clone();
  const out = [];
  const steps = Math.round(HORIZON * FPS) + 3;
  for (let k = 1; k <= steps; k++) {
    approach(v, ctrl.want, 1 / FPS);
    p.addScaledVector(v, 1 / FPS);
    const r = Math.hypot(p.x, p.z);
    if (r > AREA) { p.multiplyScalar(AREA / r); v.multiplyScalar(0.5); }
    const s = v.length();
    if (s > 0.3) {
      const travel = Math.atan2(v.x, v.z);
      const back = Math.cos(travel - yaw) < -0.3 && s < 1.2;
      const want = back ? yaw : travel;
      yaw += Math.max(-3 / FPS, Math.min(3 / FPS, wrap(want - yaw)));
    }
    if (k > steps - 4) out.push({ pos: p.clone(), yaw, vel: v.clone() });
  }
  return out;
}

// ---------------------------------------------------------------- requests

/** Edits for the grabbed part on a target pose (the body comes from the gait library, the part is reached for). */
function reachFor(pose) {
  if (!drag) return pose;
  const part = drag.part, G = drag.point;
  let P = rig.positions(pose);
  if (part.endsWith('Hand')) {
    const c = chain(part.startsWith('Left') ? 'Left' : 'Right', ARM);
    const reach = reachOf(P, c) * 0.97;
    const to = G.clone().sub(P[c[0]]);
    const target = to.length() > reach ? P[c[0]].clone().addScaledVector(to.normalize(), reach) : G.clone();
    twoBoneIK(rig, pose, c[0], c[1], c[2], target, new Vector3(0, -1, -0.4));
  } else if (part.endsWith('Foot')) {
    const side = part.startsWith('Left') ? 'Left' : 'Right';
    const c = chain(side, LEG);
    const reach = reachOf(P, c) * 0.97;
    const goal = G.clone();
    goal.y = Math.min(goal.y, 0.95);
    const to = goal.clone().sub(P[c[0]]);
    const target = to.length() > reach ? P[c[0]].clone().addScaledVector(to.normalize(), reach) : goal;
    target.y = Math.max(ANKLE, target.y);
    twoBoneIK(rig, pose, c[0], c[1], c[2], target, new Vector3(0, 0.3, 1));
    // weight onto the other foot
    const other = P[I[(side === 'Left' ? 'Right' : 'Left') + 'Foot']];
    pose.root.x += (other.x - pose.root.x) * 0.35;
    pose.root.z += (other.z - pose.root.z) * 0.35;
    const o = chain(side === 'Left' ? 'Right' : 'Left', LEG);
    twoBoneIK(rig, pose, o[0], o[1], o[2], other.clone(), new Vector3(0, 0, 1));
  } else if (part === 'Head') {
    const s1 = P[I.Spine1];
    const bend = new Quaternion().setFromUnitVectors(P[I.Head].clone().sub(s1).normalize(), G.clone().sub(s1).normalize());
    rotateSubtree(rig, pose, I.Spine1, qcap(bend, 0.45));
  } else if (part === 'Hips' && G.y < HIPS_Y - 0.12) { // pulled down: bend the knees, feet stay
    const feet = [P[I.LeftFoot].clone(), P[I.RightFoot].clone()];
    pose.root.y = Math.max(HIPS_Y - 0.4, G.y);
    twoBoneIK(rig, pose, I.LeftLeg, I.LeftShin, I.LeftFoot, feet[0], new Vector3(0.2, 0, 1));
    twoBoneIK(rig, pose, I.RightLeg, I.RightShin, I.RightFoot, feet[1], new Vector3(-0.2, 0, 1));
  }
  return rig.fixRigid(pose);
}

function qcap(q, max) {
  const a = 2 * Math.acos(Math.min(1, Math.abs(q.w)));
  return a <= max ? q : new Quaternion().slerp(q, max / a);
}

function locoReq(mode = 'now') {
  return {
    kind: 'move', mode,
    build(tr, a) {
      const from = tr.frame(a).pose;
      const path = predict(from);
      const end = path[0];
      const speed = end.vel.length();
      let targets;
      let entry = null;
      if (speed >= 0.3) {
        const fwd = new Vector3(Math.sin(end.yaw), 0, Math.cos(end.yaw));
        const dir = end.vel.dot(fwd) / speed < -0.3 ? 'b' : 'f';
        // which foot will be on the ground: steps over the horizon at a speed-dependent cadence
        const steps = Math.round(HORIZON * (1.6 + 0.45 * speed));
        const now = stanceFoot(from.contact);
        const phase = now === 'L' || now === 'R' ? (steps % 2 ? (now === 'L' ? 'R' : 'L') : now) : (Math.random() < 0.5 ? 'L' : 'R');
        entry = lookup(speed, dir, phase);
      }
      if (entry) {
        // a window the model made itself, at about this speed: keep its own displacement, put it on the path
        targets = entry.frames.map(f => placePose(f, end.yaw, ZERO, end.pos));
      } else {
        // nothing learned yet (or standing): the stand pose on each point of the path carries the speed
        targets = path.map(p => placePose(STAND, p.yaw, ZERO, p.pos));
      }
      targets = targets.map(t => reachFor(t));
      ghostPose = targets[0];
      return {
        targets, tokens: speed >= 0.3 ? [8, 9, 10] : [8, 9, 10, 11, 12], rate: 1, tag: { kind: 'move' },
        after: (_tr, _a, frames) => harvest(frames),
      };
    },
  };
}

// ---------------------------------------------------------------- scene objects

const track = new Track(rig, { stance: STAND });
const body = new Mannequin(scene, rig, 0xd9b26a);
const ghost = new Mannequin(scene, rig, 0x8fb4ff);
ghost.material.transparent = true;
ghost.material.opacity = 0.12;
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
const plane = new THREE.Plane();
let drag = null;

function pointerRay(e) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
}

// registered before OrbitControls so a grab can stop the orbit
canvas.addEventListener('pointerdown', e => {
  pointerRay(e);
  const hit = ray.intersectObjects(handles.map(h => h.mesh))[0];
  if (!hit) return;
  const h = handles.find(x => x.mesh === hit.object);
  // drag on a vertical plane facing the camera through the grabbed part
  const n = camera.getWorldDirection(new Vector3()).setY(0).normalize().negate();
  plane.setFromNormalAndCoplanarPoint(n, h.mesh.position);
  drag = { part: h.part, point: h.mesh.position.clone(), sent: null, w: 0 };
  canvas.setPointerCapture(e.pointerId);
  e.stopImmediatePropagation();
  e.preventDefault();
});
canvas.addEventListener('pointermove', e => {
  pointerRay(e);
  if (!drag) {
    const hit = ray.intersectObjects(handles.map(h => h.mesh))[0];
    for (const h of handles) h.mesh.scale.setScalar(hit?.object === h.mesh ? 1.5 : 1);
    canvas.style.cursor = hit ? 'grab' : '';
    return;
  }
  const out = new Vector3();
  if (ray.ray.intersectPlane(plane, out)) {
    out.y = Math.max(0.02, Math.min(2.2, out.y));
    const r = Math.hypot(out.x, out.z);
    if (r > AREA + 1) out.multiplyScalar((AREA + 1) / r);
    drag.point.copy(out);
  }
});
const release = () => {
  if (!drag) return;
  drag = null;
  canvas.style.cursor = '';
  planTimer = 0;
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

const controls = new OrbitControls(camera, canvas);
camera.position.set(0.8, 2.0, 5.5);
controls.target.set(0, 0.9, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 1.5;
controls.maxDistance = 14;
controls.update();

// ---------------------------------------------------------------- on-screen layer: reach and planted feet

const locks = { Left: null, Right: null };

function overlay(pose, contact, dt) {
  const p = clonePose(pose);
  let P = rig.positions(p);
  // feet the model says are on the ground stay where they landed
  for (const side of ['Left', 'Right']) {
    const c = contact ? Math.max(...(side === 'Left' ? contact.slice(0, 2) : contact.slice(2, 4))) : 0;
    const foot = I[side + 'Foot'];
    if (!settings.footlock || c < 0.5 || drag?.part === side + 'Foot') { locks[side] = null; continue; }
    if (!locks[side] || locks[side].distanceTo(new Vector3(P[foot].x, locks[side].y, P[foot].z)) > 0.2) {
      locks[side] = new Vector3(P[foot].x, 0, P[foot].z);
      continue;
    }
    const want = new Vector3(locks[side].x, P[foot].y, locks[side].z);
    const c3 = chain(side, LEG);
    const knee = P[c3[1]].clone().sub(P[c3[0]].clone().lerp(P[c3[2]], 0.5));
    const footRot = p.rot[foot].clone();
    twoBoneIK(rig, p, c3[0], c3[1], c3[2], want, knee.lengthSq() > 1e-6 ? knee : new Vector3(0, 0, 1));
    // keep the foot's own orientation
    const fix = footRot.multiply(p.rot[foot].clone().invert());
    rotateSubtree(rig, p, foot, fix);
    P = rig.positions(p);
  }
  // the grabbed hand or foot goes to the pointer (as far as it reaches), easing in
  if (drag && (drag.part.endsWith('Hand') || drag.part.endsWith('Foot'))) {
    drag.w = Math.min(1, drag.w + dt / 0.15);
    const side = drag.part.startsWith('Left') ? 'Left' : 'Right';
    const c3 = chain(side, drag.part.endsWith('Hand') ? ARM : LEG);
    const reach = reachOf(P, c3) * 0.995;
    const tip = P[c3[2]].clone().lerp(drag.point, drag.w);
    const to = tip.clone().sub(P[c3[0]]);
    if (to.length() > reach) tip.copy(P[c3[0]]).addScaledVector(to.normalize(), reach);
    if (drag.part.endsWith('Foot')) tip.y = Math.max(ANKLE, tip.y);
    const mid = P[c3[1]].clone().sub(P[c3[0]].clone().lerp(P[c3[2]], 0.5));
    twoBoneIK(rig, p, c3[0], c3[1], c3[2], tip, mid.lengthSq() > 1e-6 ? mid : new Vector3(0, -1, 0));
  }
  return rig.fixRigid(p);
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
let planTimer = 0;

function simulate(dt) {
  ctrl.want.copy(pullVelocity(shownP));
  approach(ctrl.vel, ctrl.want, dt);
  planTimer -= dt;
  if (scheduler.ready && !track.busy) {
    const moving = ctrl.vel.length() > 0.05 || ctrl.want.length() > 0.05;
    const pointerMoved = drag && (!drag.sent || drag.sent.distanceTo(drag.point) > 0.05);
    if ((moving || pointerMoved) && planTimer <= 0) {
      if (drag) drag.sent = drag.point.clone();
      track.request(locoReq('now'));
      planTimer = REPLAN;
    } else if (!moving && !drag && track.remaining / FPS < scheduler.latency + 0.3) {
      track.request(locoReq('end'));
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
  for (const h of handles) h.mesh.position.copy(shownP[I[h.part]]);
  band.visible = grip.visible = !!drag;
  if (drag) {
    band.geometry.setFromPoints([shownP[I[drag.part]], drag.point]);
    grip.position.copy(drag.point);
  }
  // keep the body in view
  const hips = shownP[0];
  const move = new Vector3(hips.x, 0.9, hips.z).sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 2.5));
  controls.target.add(move);
  camera.position.add(move);
  controls.update();
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hud(); }
  requestAnimationFrame(frame);
}

function hud() {
  if (!scheduler.ready) return;
  const v = Math.hypot(ctrl.vel.x, ctrl.vel.z);
  $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms · 速さ ${v.toFixed(1)} m/s · 描画 ${fps.toFixed(0)} fps`;
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
  locks.Left = locks.Right = null;
});
if (!navigator.gpu) $('gpu-note').textContent = 'このブラウザでは WebGPU が使えません。WASM（CPU）で動きますが、かなり遅くなります。';

showLibrary();
track.reset(new Vector3(0, 0, 0), 0);
requestAnimationFrame(frame);
if (params.has('debug')) {
  window.lab = { track, scheduler, simulate, settings, ctrl, library, rig, stage, get shownP() { return shownP; },
    get drag() { return drag; }, set drag(v) { drag = v ? { sent: null, w: 0, ...v } : null; }, draw: () => frame(performance.now()) };
}
