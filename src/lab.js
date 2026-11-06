// SOMA Pose Lab: MotionBricks keeps generating toward poses picked from a stock and mutated; a pose from the middle
// of every generated motion is mutated again and added to the stock, so the poses drift further and further.
// Dragging a hand, foot, the head or the hips bends the body toward the pointer by IK (moving the whole body when
// the limb cannot reach), and MotionBricks keeps generating toward that pulled pose, so the body follows.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, clonePose, emptyBatch, encodeRequest } from './soma.js';
import { KeyBook, rotateSubtree, twoBoneIK } from './poses.js';
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
const AREA = 3; // the body wanders within this radius
const MAX_STOCK = 72;

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const book = new KeyBook(rig, await (await fetch(new URL('../assets/keyposes.json', import.meta.url))).json());
const I = rig.index;
const canvas = $('view');
const stage = new Stage(canvas);
const scene = stage.scene;
const camera = stage.camera;
const worker = new Worker(new URL('./infer.worker.js', import.meta.url), { type: 'module' });
const scheduler = new Scheduler(rig, worker);

const track = new Track(rig, book);
const body = new Mannequin(scene, rig, 0xd9b26a);
const ghost = new Mannequin(scene, rig, 0x8fb4ff);
ghost.material.transparent = true;
ghost.material.opacity = 0.16;
ghost.material.depthWrite = false;
ghost.bones.forEach(b => { b.mesh.castShadow = false; });
ghost.head.castShadow = false;
let ghostPose = null;

const settings = { auto: true, strength: 0.45, rate: 1.2 };
const bind = (id, key, fmt = v => v) => {
  const el = $(id);
  const read = () => { settings[key] = el.type === 'checkbox' ? el.checked : Number(el.value); if ($(`${id}-v`)) $(`${id}-v`).textContent = fmt(settings[key]); };
  el.addEventListener('input', read);
  read();
};
bind('auto', 'auto');
bind('strength', 'strength', v => v.toFixed(2));
bind('rate', 'rate', v => v.toFixed(1));

// ---------------------------------------------------------------- pose helpers

const rand = (a, b) => a + Math.random() * (b - a);
const randUnit = () => new Vector3(rand(-1, 1), rand(-1, 1), rand(-1, 1)).normalize();
const toLocal = p => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));
function fromLocal(root, L) {
  const rot = [];
  L.forEach((q, j) => { rot.push(rig.parents[j] < 0 ? q.clone() : rot[rig.parents[j]].clone().multiply(q)); });
  return rig.fixRigid({ root: root.clone(), rot });
}

function lowest(pose) {
  const P = rig.positions(pose);
  return Math.min(...P.map(v => v.y), ...rig.extraPositions(pose, P).map(v => v.y));
}

/** Rest the pose on the floor (canonical stock poses), or only lift it out of the floor (pulled poses). */
function ground(pose, onlyLift = false) {
  const low = lowest(pose);
  if (!onlyLift || low < 0.02) pose.root.y += 0.02 - low;
  pose.root.y = Math.min(pose.root.y, 2.2);
  return pose;
}

/** Hips over the origin, facing +Z. */
function canonical(pose) {
  const P = rig.positions(pose);
  const at = new Vector3(pose.root.x, 0, pose.root.z);
  return placePose(pose, -rig.heading(P), at, at.clone().negate());
}

const LIMBS = {
  LeftHand: ['LeftArm', 'LeftForeArm', 'LeftHand'], RightHand: ['RightArm', 'RightForeArm', 'RightHand'],
  LeftFoot: ['LeftLeg', 'LeftShin', 'LeftFoot'], RightFoot: ['RightLeg', 'RightShin', 'RightFoot'],
};
const TWIST = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftLeg', 'LeftShin', 'LeftFoot', 'RightLeg', 'RightShin', 'RightFoot'];
const SUBTREES = ['Spine1', 'Neck1', 'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg', 'LeftForeArm', 'RightShin'];
const stanceLocal = toLocal(canonical(book.stance));

/** q^k (rotation angle scaled by k), shortest arc, capped below a half turn. */
function qpow(q, k, cap = Math.PI * 0.95) {
  const d = q.clone();
  if (d.w < 0) d.set(-d.x, -d.y, -d.z, -d.w);
  const angle = 2 * Math.acos(Math.min(1, d.w));
  const sn = Math.sin(angle / 2);
  if (sn < 1e-6) return new Quaternion();
  return new Quaternion().setFromAxisAngle(new Vector3(d.x / sn, d.y / sn, d.z / sn), Math.min(cap, angle * k));
}

const angleOf = q => 2 * Math.acos(Math.min(1, Math.abs(q.w)));

/** How far a (canonical) pose is from the fighting stance: mean joint rotation away from it, in radians. */
function energy(pose) {
  const L = toLocal(pose);
  let sum = 0;
  for (let j = 0; j < L.length; j++) sum += angleOf(stanceLocal[j].clone().invert().multiply(L[j]));
  return sum / L.length;
}

/** Scale every joint's rotation away from the stance by `factor` (keeps the body's own shape, only stronger). */
function scaleFromStance(pose, factor) {
  const L = toLocal(pose);
  for (let j = 0; j < L.length; j++) {
    const k = j === 0 ? 1 + (factor - 1) * 0.5 : factor; // tip the whole body less
    L[j] = stanceLocal[j].clone().multiply(qpow(stanceLocal[j].clone().invert().multiply(L[j]), k));
  }
  return fromLocal(pose.root, L);
}

const exaggerate = (pose, amount) => scaleFromStance(pose, 1 + amount);

/** Joint-angle distance between two canonical poses (for novelty). */
function distance(a, b) {
  const A = toLocal(a), B = toLocal(b);
  let sum = 0;
  for (let j = 0; j < A.length; j++) sum += angleOf(A[j].clone().invert().multiply(B[j]));
  return sum / A.length;
}

/** Change a pose by a few random operations; strength 0..1. Human proportions are kept, human limits are not. */
function mutate(src, strength) {
  let p = clonePose(src);
  const ops = 1 + Math.floor(Math.random() * (1 + strength * 3));
  for (let o = 0; o < ops; o++) {
    const r = Math.random();
    if (r < 0.32) { // twist one joint (and everything below it)
      const j = I[TWIST[Math.floor(Math.random() * TWIST.length)]];
      const angle = rand(-1, 1) * strength * Math.PI * (j === 0 ? 0.35 : 0.75);
      rotateSubtree(rig, p, j, new Quaternion().setFromAxisAngle(randUnit(), angle));
    } else if (r < 0.56) { // fling a hand or a foot somewhere it can reach
      const chain = Object.values(LIMBS)[Math.floor(Math.random() * 4)].map(n => I[n]);
      const P = rig.positions(p);
      const reach = P[chain[1]].distanceTo(P[chain[0]]) + P[chain[2]].distanceTo(P[chain[1]]);
      const target = P[chain[0]].clone().addScaledVector(randUnit(), reach * rand(0.45, 0.98));
      twoBoneIK(rig, p, chain[0], chain[1], chain[2], target, randUnit());
    } else if (r < 0.76 && stock.length > 1) { // borrow a part of another pose in the stock
      const other = stock[Math.floor(Math.random() * stock.length)].pose;
      const L = toLocal(p), Lo = toLocal(other);
      const w = rand(0.5, 1);
      for (const k of rig.subtree(I[SUBTREES[Math.floor(Math.random() * SUBTREES.length)]])) L[k].slerp(Lo[k], w);
      p = fromLocal(p.root, L);
    } else if (r < 0.9) { // push every joint further from the fighting stance
      p = exaggerate(p, strength * 0.7);
    } else { // tip the whole body
      const axis = new Vector3(rand(-1, 1), 0, rand(-1, 1)).normalize();
      rotateSubtree(rig, p, 0, new Quaternion().setFromAxisAngle(axis, rand(-1, 1) * strength * 1.1));
    }
  }
  return ground(rig.fixRigid(p));
}

// ---------------------------------------------------------------- stock

const stock = [];
let nextId = 1, maxGen = 0, playing = null, queued = null;

function addStock(pose, gen, seed = false) {
  const entry = { id: nextId++, pose, gen, seed, el: null, energy: energy(pose) };
  // novelty: how far it is from the closest of (a sample of) the poses already kept
  const sample = stock.length > 24 ? Array.from({ length: 24 }, () => stock[Math.floor(Math.random() * stock.length)]) : stock;
  entry.novelty = sample.length ? Math.min(...sample.map(e => distance(e.pose, pose))) : 1;
  entry.score = entry.energy * 0.6 + entry.novelty * 0.4;
  stock.push(entry);
  maxGen = Math.max(maxGen, gen);
  entry.el = thumb(entry);
  $('stock-grid').prepend(entry.el);
  while (stock.length > MAX_STOCK) {
    let worst = -1;
    stock.forEach((e, i) => {
      if (e.seed || e === playing || e === queued || e === entry) return;
      if (worst < 0 || e.score < stock[worst].score) worst = i;
    });
    if (worst < 0) break;
    stock[worst].el.remove();
    stock.splice(worst, 1);
  }
  $('stock-n').textContent = stock.length;
  $('gen').textContent = maxGen;
  return entry;
}

function resetStock() {
  for (const e of stock) e.el.remove();
  stock.length = 0;
  maxGen = 0;
  for (const name of Object.keys(book.keys)) addStock(ground(canonical(book.keys[name][0])), 0, true);
}

/** Pick by score, newer poses a little more often, so the stock keeps moving on and does not settle. */
function pick() {
  let total = 0;
  const w = stock.map((e, i) => { const v = (0.15 + e.score) * (0.5 + i / stock.length); total += v; return v; });
  let r = Math.random() * total;
  for (let i = 0; i < stock.length; i++) { r -= w[i]; if (r <= 0) return stock[i]; }
  return stock.at(-1);
}

/**
 * Take a pose from the middle of a generated motion without letting the stock fade toward the average:
 * 1. use the strongest frame of the middle part (not a random one),
 * 2. carry it further along its own motion (momentum: each joint keeps turning the way it was turning),
 * 3. if it is still weaker than the pose it came from, scale it away from the stance until it is not.
 */
function harvest(frames, parent) {
  const n = frames.length;
  if (n < 10) return;
  let best = -1, bestE = -1;
  for (let i = Math.floor(n * 0.25); i <= Math.floor(n * 0.75); i++) {
    const e = energy(canonical(frames[i].pose));
    if (e > bestE) { bestE = e; best = i; }
  }
  const mid = frames[best].pose, prev = frames[Math.max(0, best - 3)].pose;
  const Lm = toLocal(mid), Lp = toLocal(prev);
  const carry = 1 + settings.strength * 2;
  const L = Lm.map((q, j) => q.clone().multiply(qpow(Lp[j].clone().invert().multiply(q), carry, Math.PI * 0.4)));
  let child = canonical(fromLocal(mid.root, L));
  child = mutate(child, settings.strength * 0.5);
  const want = Math.min(1.3, Math.max(0.3, (parent?.energy ?? 0.3) * 1.05));
  for (let k = 0; k < 3; k++) {
    const e = energy(child);
    if (e >= want || e < 1e-3) break;
    child = ground(scaleFromStance(child, Math.min(2.5, want / e)));
  }
  addStock(ground(child), (parent?.gen ?? 0) + 1);
}

// thumbnails: the pose seen from the front-right, coloured by generation
const LINES = [['Hips', 'Spine2'], ['Spine2', 'Chest'], ['Chest', 'Neck1'], ['Neck1', 'Head'], ['LeftLeg', 'RightLeg'],
  ['Chest', 'LeftArm'], ['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'], ['Chest', 'RightArm'], ['RightArm', 'RightForeArm'], ['RightForeArm', 'RightHand'],
  ['Hips', 'LeftLeg'], ['LeftLeg', 'LeftShin'], ['LeftShin', 'LeftFoot'], ['LeftFoot', 'LeftToeBase'],
  ['Hips', 'RightLeg'], ['RightLeg', 'RightShin'], ['RightShin', 'RightFoot'], ['RightFoot', 'RightToeBase']];

function thumb(entry) {
  const el = document.createElement('div');
  el.className = 'thumb';
  const c = document.createElement('canvas');
  c.width = 90; c.height = 120;
  const g = c.getContext('2d');
  const P = rig.positions(entry.pose);
  const yaw = 0.5, cs = Math.cos(yaw), sn = Math.sin(yaw);
  const pts = P.map(v => [v.x * cs + v.z * sn, v.y]);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const span = Math.max(1.9, Math.max(...ys) + 0.1, (Math.max(...xs) - Math.min(...xs)) * 1.35);
  const scale = 110 / span, cx = (Math.max(...xs) + Math.min(...xs)) / 2;
  const sx = x => 45 + (x - cx) * scale, sy = y => 114 - y * scale;
  g.strokeStyle = 'rgba(255,255,255,0.12)';
  g.beginPath(); g.moveTo(6, 114.5); g.lineTo(84, 114.5); g.stroke();
  g.strokeStyle = entry.seed ? '#c8a24a' : `hsl(${(200 + entry.gen * 23) % 360} 70% 62%)`;
  g.lineWidth = 2.2;
  g.lineCap = 'round';
  g.beginPath();
  for (const [a, b] of LINES) { const p = pts[I[a]], q = pts[I[b]]; g.moveTo(sx(p[0]), sy(p[1])); g.lineTo(sx(q[0]), sy(q[1])); }
  g.stroke();
  const h = pts[I.Head];
  g.beginPath(); g.arc(sx(h[0]), sy(h[1] + 0.06), 0.1 * scale, 0, Math.PI * 2); g.stroke();
  el.appendChild(c);
  const label = document.createElement('span');
  label.className = 'g';
  label.textContent = `${entry.seed ? 'seed' : `g${entry.gen}`} · ${entry.energy.toFixed(2)}`;
  el.appendChild(label);
  el.addEventListener('click', () => {
    queued = entry;
    if (!drag && scheduler.ready) track.request(evolveReq(entry, 'now', true));
  });
  return el;
}

// ---------------------------------------------------------------- requests

function evolveReq(entry, mode = 'end', exact = false) {
  return {
    kind: 'idle', mode,
    build(tr, a) {
      if (queued === entry) queued = null;
      const from = tr.frame(a).pose;
      const A = new Vector3(from.root.x, 0, from.root.z);
      const s = settings.strength;
      const heading = rig.heading(rig.positions(from));
      const yaw = heading + rand(-1, 1) * (0.25 + s) * 0.9;
      const ang = rand(0, Math.PI * 2), step = rand(0, 0.15 + 0.6 * s);
      const dest = A.clone().add(new Vector3(Math.sin(ang) * step, 0, Math.cos(ang) * step));
      if (dest.length() > AREA) dest.multiplyScalar(-0.3); // wander back toward the middle
      const shape = exact ? clonePose(entry.pose) : mutate(entry.pose, s);
      const target = placePose(shape, yaw, ZERO, dest);
      ghostPose = target;
      return {
        targets: [target], tokens: [8, 9, 10, 11, 12], rate: settings.rate, tag: { kind: 'evolve', entry },
        after: (_tr, _a, frames) => harvest(frames, entry),
      };
    },
  };
}

/** The pose at the anchor with the grabbed part pulled to the pointer; the body comes along when the limb is at full stretch. */
function pulled(base, part, point) {
  const p = clonePose(base);
  let P = rig.positions(p);
  if (LIMBS[part]) {
    const [a, b, c] = LIMBS[part].map(n => I[n]);
    const reach = (P[b].distanceTo(P[a]) + P[c].distanceTo(P[b])) * 0.97;
    const toward = point.clone().sub(P[a]);
    const excess = toward.length() - reach;
    if (excess > 0) {
      const dir = toward.clone().normalize();
      // lean the trunk toward the pull, then let the rest of the stretch drag the whole body
      const lean = Math.min(1, excess / 0.6) * 0.45;
      const side = new Vector3().crossVectors(UP, new Vector3(dir.x, 0, dir.z));
      if (side.lengthSq() > 1e-6) rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(side.normalize(), lean * (part.endsWith('Hand') ? 1 : -0.5)));
      P = rig.positions(p);
      const left = point.clone().sub(P[a]).length() - reach;
      if (left > 0) p.root.addScaledVector(point.clone().sub(P[a]).normalize(), left);
    }
    P = rig.positions(p);
    const bend = P[b].clone().sub(P[a].clone().lerp(P[c], 0.5));
    twoBoneIK(rig, p, a, b, c, point, bend.lengthSq() > 1e-6 ? bend : new Vector3(0, -1, 0));
  } else if (part === 'Head') {
    const s1 = P[I.Spine1];
    const cur = P[I.Head].clone().sub(s1), want = point.clone().sub(s1);
    const bendQ = new Quaternion().setFromUnitVectors(cur.clone().normalize(), want.clone().normalize());
    rotateSubtree(rig, p, I.Spine1, new Quaternion().slerp(bendQ, 0.7));
    P = rig.positions(p);
    p.root.add(point.clone().sub(P[I.Head]).multiplyScalar(0.6));
  } else { // Hips
    p.root.add(point.clone().sub(P[0]));
  }
  return ground(rig.fixRigid(p), true);
}

function dragReq() {
  const part = drag.part, point = drag.point.clone();
  return {
    kind: 'move', mode: 'now',
    build(tr, a) {
      const target = pulled(tr.frame(a).pose, part, point);
      ghostPose = target;
      return {
        targets: [target], tokens: [6], rate: 1.7, tag: { kind: 'drag' },
        after: (_tr, _a, frames) => { if (Math.random() < 0.35) harvest(frames, playing); },
      };
    },
  };
}

// ---------------------------------------------------------------- dragging

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
  plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new Vector3()).negate(), h.mesh.position);
  drag = { part: h.part, point: h.mesh.position.clone() };
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
    out.y = Math.max(0.02, Math.min(2.6, out.y));
    const r = Math.hypot(out.x, out.z);
    if (r > AREA + 0.8) out.multiplyScalar((AREA + 0.8) / r);
    drag.point.copy(out);
  }
});
const release = () => {
  if (!drag) return;
  if (scheduler.ready) track.request(dragReq());
  drag = null;
  canvas.style.cursor = '';
};
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

const controls = new OrbitControls(camera, canvas);
camera.position.set(0.6, 1.6, 4.2);
controls.target.set(0, 1, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 1.5;
controls.maxDistance = 10;
controls.update();

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;

function simulate(dt) {
  if (scheduler.ready) {
    if (drag) {
      // re-aim only when the pointer has moved: otherwise the motion plays out, ends on the pulled pose and holds it
      const moved = !drag.sent || drag.sent.distanceTo(drag.point) > 0.05;
      if (!track.busy && moved) {
        drag.sent = drag.point.clone();
        track.request(dragReq());
      }
    } else if (!track.busy && track.remaining / 30 < scheduler.latency + 0.25) {
      track.request(settings.auto ? evolveReq(pick()) : evolveReq(stock[0], 'end', true));
    }
  }
  track.advance(dt);
  scheduler.tick([track]);
  const tag = track.tag;
  if (tag.kind === 'evolve') playing = tag.entry;
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  body.update(track.P, track.extra, dt);
  if (ghostPose) {
    const P = rig.positions(ghostPose);
    ghost.update(P, rig.extraPositions(ghostPose, P), dt);
  }
  for (const h of handles) h.mesh.position.copy(track.P[I[h.part]]);
  band.visible = grip.visible = !!drag;
  if (drag) {
    const from = handles.find(h => h.part === drag.part).mesh.position;
    band.geometry.setFromPoints([from, drag.point]);
    grip.position.copy(drag.point);
  }
  // keep the body in view
  const hips = track.P[0];
  const want = new Vector3(hips.x, Math.max(0.6, Math.min(1.4, hips.y)), hips.z);
  const move = want.sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 2));
  controls.target.add(move);
  camera.position.add(move);
  controls.update();
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hud(); }
  requestAnimationFrame(frame);
}

function hud() {
  for (const e of stock) {
    e.el.classList.toggle('now', e === playing);
    e.el.classList.toggle('queued', e === queued);
  }
  if (scheduler.ready) $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms · 描画 ${fps.toFixed(0)} fps`;
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
$('reset').addEventListener('click', resetStock);
if (!navigator.gpu) $('gpu-note').textContent = 'このブラウザでは WebGPU が使えません。WASM（CPU）で動きますが、かなり遅くなります。';

resetStock();
track.reset(new Vector3(0, 0, 0), 0);
requestAnimationFrame(frame);
if (params.has('debug')) window.lab = { track, stock, scheduler, simulate, settings, get drag() { return drag; }, set drag(v) { drag = v; }, dragReq, pulled, mutate, stage, rig, draw: () => frame(performance.now()) };
