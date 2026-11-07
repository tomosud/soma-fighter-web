// Frame-by-frame naturalness evaluation of a motion (the displayed SOMA output, or a reference clip).
//
// Normalisation (so that "unusual" means unusual for a human body, not just "moves a lot"):
// - Root motion is taken out: every joint position is expressed in the body frame (hips projected to the floor,
//   x = body left, z = facing). The hips themselves get their own rows (speed, turning rate, height).
// - Rotations are the local ones (each bone relative to its parent), so turning the whole body is not counted.
// - For each signal the residual from its centred moving average (window 0.23 s) is measured: how far the motion
//   leaves its own smooth trend (jitter, pops, sudden flips). Position residuals in metres, rotation residuals as
//   the angle between the rotation and the window's mean rotation.
// - Each residual is turned into a robust z-score with the median / MAD of the same residual measured on the
//   Mixamo reference clips (per bone and per quantity, separately for standing and for moving: walking legs move
//   much more than idle ones), so a hand and the pelvis are judged by their own natural range. The toes' rotation
//   is shown but left out of the totals (SOMA rolls onto its toes, the Mixamo X Bot hardly bends them). Feet:
//   sliding speed while on the floor (cm/s) and going through the floor, in absolute units.
import { Quaternion, Vector3 } from 'three';

export const EVAL_FPS = 30;
const WIN = 7; // frames at 30 fps (0.23 s): a step's swing (about 1 Hz) passes, pops and jitter do not

/** Joints worth judging (the rigid finger ends, eyes and jaw follow their parents). */
export function evalJoints(rig) {
  const skip = new Set(['Jaw', 'LeftEye', 'RightEye', 'LeftHandThumbEnd', 'LeftHandMiddleEnd', 'RightHandThumbEnd', 'RightHandMiddleEnd', 'Neck2']);
  return rig.names.map((n, j) => j).filter(j => !skip.has(rig.names[j]));
}

const toLocal = (rig, p) => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));

/** Per frame: body-frame joint positions, local rotations, hips position / heading, feet heights. */
function measure(rig, poses) {
  const I = rig.index;
  return poses.map(p => {
    const P = rig.positions(p);
    const h = rig.heading(P);
    const c = Math.cos(h), s = Math.sin(h);
    const hip = P[0];
    // body frame: x = left, z = forward (facing h: forward = (sin h, 0, cos h))
    const body = P.map(v => { const dx = v.x - hip.x, dz = v.z - hip.z; return new Vector3(dx * c - dz * s, v.y, dx * s + dz * c); });
    const local = toLocal(rig, p);
    local[0] = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), -h).multiply(local[0]); // the pelvis without the heading
    return { body, local, hip: hip.clone(), heading: h, feet: [P[I.LeftFoot].clone(), P[I.RightFoot].clone()], toes: [P[I.LeftToeBase].clone(), P[I.RightToeBase].clone()], contact: p.contact ?? null };
  });
}

function meanQuat(qs) {
  const ref = qs[Math.floor(qs.length / 2)];
  const m = new Quaternion(0, 0, 0, 0);
  for (const q of qs) {
    const sgn = q.dot(ref) < 0 ? -1 : 1;
    m.x += q.x * sgn; m.y += q.y * sgn; m.z += q.z * sgn; m.w += q.w * sgn;
  }
  return m.normalize();
}
const angle = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a.dot(b))));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/**
 * Raw residuals per frame (no z yet): for each judged joint the position residual (m) and rotation residual
 * (rad), plus whole-body rows: hips speed residual, yaw-rate residual, hips height residual, foot slide L/R,
 * floor penetration. `poses` at EVAL_FPS.
 */
export function residuals(rig, poses, opt = {}) {
  const joints = evalJoints(rig);
  const M = measure(rig, poses);
  const n = M.length, half = Math.floor(WIN / 2);
  const ankle = opt.ankle ?? rig.restPos[rig.index.LeftFoot].y, toeRest = rig.restPos[rig.index.LeftToeBase].y;
  const pos = joints.map(() => new Float32Array(n)), rot = joints.map(() => new Float32Array(n));
  const glob = { hipSpeed: new Float32Array(n), yawRate: new Float32Array(n), hipHeight: new Float32Array(n), slideL: new Float32Array(n), slideR: new Float32Array(n), below: new Float32Array(n) };
  const speed = new Float32Array(n), yaw = new Float32Array(n);
  // each contact point's floor: the low 5 % of its heights in this motion
  const low = arr => { const a = Float32Array.from(arr).sort(); return a[Math.floor(a.length * 0.05)] ?? 0; };
  const floorA = [0, 1].map(sd => low(M.map(m => m.feet[sd].y))), floorT = [0, 1].map(sd => low(M.map(m => m.toes[sd].y)));
  for (let t = 1; t < n; t++) {
    speed[t] = Math.hypot(M[t].hip.x - M[t - 1].hip.x, M[t].hip.z - M[t - 1].hip.z) * EVAL_FPS;
    yaw[t] = wrap(M[t].heading - M[t - 1].heading) * EVAL_FPS;
  }
  speed[0] = speed[1] ?? 0; yaw[0] = yaw[1] ?? 0;
  for (let t = 0; t < n; t++) {
    const a = Math.max(0, t - half), b = Math.min(n - 1, t + half);
    joints.forEach((j, k) => {
      const m = new Vector3();
      for (let u = a; u <= b; u++) m.add(M[u].body[j]);
      m.divideScalar(b - a + 1);
      pos[k][t] = M[t].body[j].distanceTo(m);
      const qs = [];
      for (let u = a; u <= b; u++) qs.push(M[u].local[j]);
      rot[k][t] = angle(M[t].local[j], meanQuat(qs));
    });
    let ms = 0, my = 0, mh = 0;
    for (let u = a; u <= b; u++) { ms += speed[u]; my += yaw[u]; mh += M[u].hip.y; }
    const cnt = b - a + 1;
    glob.hipSpeed[t] = Math.abs(speed[t] - ms / cnt);
    glob.yawRate[t] = Math.abs(yaw[t] - my / cnt);
    glob.hipHeight[t] = Math.abs(M[t].hip.y - mh / cnt);
    // feet: sliding while planted. The foot's contact point is the lower of its ankle and toe (each against the
    // lowest it gets in this motion: proportions differ); it counts as planted when it stays within 1.5 cm of that
    // for three frames running (so
    // landing and lifting off are not counted), and its slide is that point's ground speed.
    for (const [side, key] of [[0, 'slideL'], [1, 'slideR']]) {
      if (t === 0 || t === n - 1) continue;
      const lift = u => Math.min(M[u].feet[side].y - floorA[side], M[u].toes[side].y - floorT[side]);
      if (lift(t - 1) > 0.015 || lift(t) > 0.015 || lift(t + 1) > 0.015) continue;
      const useToe = M[t].toes[side].y - floorT[side] < M[t].feet[side].y - floorA[side];
      const a = useToe ? M[t - 1].toes[side] : M[t - 1].feet[side], b = useToe ? M[t].toes[side] : M[t].feet[side];
      glob[key][t] = Math.hypot(b.x - a.x, b.z - a.z) * EVAL_FPS * 100; // cm/s
    }
    glob.below[t] = Math.max(0, ankle - 0.03 - Math.min(M[t].feet[0].y, M[t].feet[1].y)) * 100; // cm
  }
  // standing or moving (hips speed averaged over 0.5 s): which reference spread applies
  const moving = new Uint8Array(n);
  for (let t = 0; t < n; t++) {
    let m = 0, c = 0;
    for (let u = Math.max(0, t - 7); u <= Math.min(n - 1, t + 7); u++) { m += speed[u]; c++; }
    moving[t] = m / c > 0.3 ? 1 : 0;
  }
  return { joints, pos, rot, glob, n, moving };
}

function medMad(values) {
  const a = Float32Array.from(values).sort();
  if (!a.length) return { med: 0, mad: 1e-3 };
  const med = a[Math.floor(a.length / 2)];
  const dev = a.map(v => Math.abs(v - med)).sort();
  return { med, mad: Math.max(1e-4, 1.4826 * dev[Math.floor(dev.length / 2)]) };
}

/** Reference statistics from natural clips (arrays of poses at EVAL_FPS). */
export function referenceStats(rig, clips) {
  const joints = evalJoints(rig);
  const blank = () => ({ pos: joints.map(() => []), rot: joints.map(() => []), hipSpeed: [], yawRate: [], hipHeight: [] });
  const acc = [blank(), blank()]; // standing, moving
  for (const poses of clips) {
    if (poses.length < WIN + 2) continue;
    const r = residuals(rig, poses);
    // only the middle (the window is cut at the ends)
    for (let t = WIN; t < r.n - WIN; t++) {
      const a = acc[r.moving[t]];
      joints.forEach((j, k) => { a.pos[k].push(r.pos[k][t]); a.rot[k].push(r.rot[k][t]); });
      a.hipSpeed.push(r.glob.hipSpeed[t]); a.yawRate.push(r.glob.yawRate[t]); a.hipHeight.push(r.glob.hipHeight[t]);
    }
  }
  const pack = a => ({ pos: a.pos.map(medMad), rot: a.rot.map(medMad), hipSpeed: medMad(a.hipSpeed), yawRate: medMad(a.yawRate), hipHeight: medMad(a.hipHeight) });
  return { joints: joints.map(j => rig.names[j]), still: pack(acc[0]), moving: pack(acc[1]), samples: [acc[0].hipSpeed.length, acc[1].hipSpeed.length] };
}

// slide and penetration are judged in absolute terms: 2 cm/s of sliding is about what people start to notice
// (Prazak et al. 2011: about 21 mm per footstep), 10 cm/s is clearly wrong
const SLIDE_OK = 2, SLIDE_BAD = 10, BELOW_BAD = 3;

/** z-scores (rows x frames) and summary scores. */
export function score(rig, poses, stats, marks = []) {
  const r = residuals(rig, poses);
  // the reference clips are hand-keyed and very smooth: below these spreads a difference is not visible anyway
  // (3 mm, 0.6 degrees, 3 cm/s, 6 degrees/s, 5 mm)
  const FLOOR = { pos: 0.003, rot: 0.01, hipSpeed: 0.03, yawRate: 0.1, hipHeight: 0.005 };
  const zf = floor => (v, s) => Math.max(0, (v - s.med) / Math.max(s.mad, floor));
  const zp = zf(FLOOR.pos), zr = zf(FLOOR.rot);
  const S = t => (r.moving[t] ? stats.moving : stats.still);
  const rows = [];
  const toe = new Set(['LeftToeBase', 'RightToeBase']);
  r.joints.forEach((j, k) => rows.push({ name: `${rig.names[j]} 位置`, z: Array.from(r.pos[k], (v, t) => zp(v, S(t).pos[k])) }));
  r.joints.forEach((j, k) => rows.push({ name: `${rig.names[j]} 回転`, z: Array.from(r.rot[k], (v, t) => zr(v, S(t).rot[k])), minor: toe.has(rig.names[j]) }));
  rows.push({ name: '腰の速さ', z: Array.from(r.glob.hipSpeed, (v, t) => zf(FLOOR.hipSpeed)(v, S(t).hipSpeed)), whole: true });
  rows.push({ name: '向きの回転', z: Array.from(r.glob.yawRate, (v, t) => zf(FLOOR.yawRate)(v, S(t).yawRate)), whole: true });
  rows.push({ name: '腰の高さ', z: Array.from(r.glob.hipHeight, (v, t) => zf(FLOOR.hipHeight)(v, S(t).hipHeight)), whole: true });
  // feet: 0 below SLIDE_OK, 5 at SLIDE_BAD (same colour scale as the z rows)
  const footZ = v => (v <= SLIDE_OK ? 0 : 5 * Math.min(1, (v - SLIDE_OK) / (SLIDE_BAD - SLIDE_OK)));
  rows.push({ name: '左足すべり', z: Array.from(r.glob.slideL, footZ), whole: true, raw: r.glob.slideL });
  rows.push({ name: '右足すべり', z: Array.from(r.glob.slideR, footZ), whole: true, raw: r.glob.slideR });
  rows.push({ name: '床めり込み', z: Array.from(r.glob.below, v => 5 * Math.min(1, v / BELOW_BAD)), whole: true, raw: r.glob.below });
  // summaries (the first and last half window are unreliable)
  const t0 = Math.min(r.n, WIN), t1 = Math.max(t0, r.n - WIN);
  const body = rows.filter(x => !x.whole && !x.minor);
  let sum = 0, cnt = 0, events = 0;
  const all = [];
  for (const row of body) for (let t = t0; t < t1; t++) { sum += row.z[t]; cnt++; all.push(row.z[t]); if (row.z[t] > 4) events++; }
  all.sort((a, b) => a - b);
  const slide = [...r.glob.slideL.slice(t0, t1), ...r.glob.slideR.slice(t0, t1)];
  const planted = slide.filter(v => v > 0);
  const worstBones = body.map(row => ({ name: row.name, mean: row.z.slice(t0, t1).reduce((a, b) => a + b, 0) / Math.max(1, t1 - t0) }))
    .sort((a, b) => b.mean - a.mean).slice(0, 6).map(x => `${x.name} ${x.mean.toFixed(2)}`);
  const summary = {
    frames: r.n, seconds: +(r.n / EVAL_FPS).toFixed(1),
    meanZ: +(sum / Math.max(1, cnt)).toFixed(3),
    p95Z: +(all[Math.floor(all.length * 0.95)] ?? 0).toFixed(2),
    eventsPerMin: +(events / Math.max(1, (t1 - t0) / EVAL_FPS / 60)).toFixed(1),
    slideCmPerS: +(planted.length ? Float32Array.from(planted).sort()[planted.length >> 1] : 0).toFixed(2), // median while planted
    slideBadPlanted: +(planted.length ? planted.filter(v => v > SLIDE_BAD).length / planted.length : 0).toFixed(3),
    slideBadFrac: +(slide.filter(v => v > SLIDE_BAD).length / Math.max(1, slide.length)).toFixed(3),
    belowMaxCm: +Math.max(0, ...r.glob.below).toFixed(1),
    yawZ95: +((rows.find(x => x.name === '向きの回転').z.slice(t0, t1).sort((a, b) => a - b)[Math.floor((t1 - t0) * 0.95)]) ?? 0).toFixed(2),
    worst: worstBones,
  };
  return { rows, summary, marks, n: r.n };
}

/** Draw a heat map (rows x frames, z 0..5) onto a 2D canvas context; marks = [{t, label, color}]. */
export function drawHeatmap(ctx, result, title = '') {
  const { rows, n, marks } = result;
  const left = 150, top = 40, cell = Math.max(1, Math.min(4, Math.floor(1600 / Math.max(1, n)))), rh = 9;
  const W = left + n * cell + 20, H = top + rows.length * rh + 60;
  ctx.canvas.width = W; ctx.canvas.height = H;
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, W, H);
  ctx.font = '12px sans-serif'; ctx.fillStyle = '#ddd'; ctx.fillText(title, 10, 18);
  const col = v => { // 0 dark blue -> 1.5 teal -> 3 yellow -> 5 red
    const x = Math.min(1, v / 5);
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.8 * x - 0.2)));
    const g = Math.round(255 * Math.max(0, Math.min(1, 1.6 * x + 0.1) - Math.max(0, 2.2 * (x - 0.6))));
    const b = Math.round(255 * Math.max(0, 0.55 - x));
    return `rgb(${r},${g},${b})`;
  };
  rows.forEach((row, i) => {
    const y = top + i * rh;
    ctx.fillStyle = row.whole ? '#fc6' : '#aaa'; ctx.font = '8px sans-serif'; ctx.fillText(row.name, 4, y + rh - 1);
    for (let t = 0; t < n; t++) { ctx.fillStyle = col(row.z[t]); ctx.fillRect(left + t * cell, y, cell, rh - 1); }
  });
  for (const m of marks ?? []) {
    ctx.fillStyle = m.color ?? 'rgba(255,255,255,0.5)';
    ctx.fillRect(left + m.t * cell, top - 6, 1, rows.length * rh + 6);
    if (m.label) { ctx.font = '9px sans-serif'; ctx.fillText(m.label, left + m.t * cell + 2, top - 8); }
  }
  // time axis (seconds)
  ctx.fillStyle = '#888'; ctx.font = '9px sans-serif';
  for (let s = 0; s * EVAL_FPS < n; s++) ctx.fillText(`${s}s`, left + s * EVAL_FPS * cell, top + rows.length * rh + 12);
  // legend
  for (let k = 0; k <= 50; k++) { ctx.fillStyle = col(k / 10); ctx.fillRect(left + k * 4, H - 22, 4, 10); }
  ctx.fillStyle = '#ccc'; ctx.fillText('z 0', left - 22, H - 13); ctx.fillText('5+', left + 206, H - 13);
  ctx.fillText('（位置・回転＝移動平均からのずれを Mixamo の自然なばらつきで割ったもの / 足すべり 2→10 cm/s / めり込み 0→3 cm）', left + 230, H - 13);
}
