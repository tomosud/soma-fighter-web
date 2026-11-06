// SOMA skeleton, poses and the MotionBricks tensor contract.
// Encoding/decoding follows NVlabs/motionbricks v0.2.0 `motionbricks/model/soma/kernels.py` (Apache-2.0).
// World space here is SOMA model space: metres, Y up, ground = XZ.
import { Matrix4, Quaternion, Vector3 } from 'three';

export const FPS = 30;
export const NJ = 30;
export const FEAT = 366;
export const MAX_FRAMES = 64;
const UP = new Vector3(0, 1, 0);
const X_AXIS = new Vector3(1, 0, 0);
const Y_AXIS = new Vector3(0, 1, 0);
const _v = new Vector3(), _w = new Vector3(), _q = new Quaternion(), _m = new Matrix4();

export class Rig {
  constructor(json) {
    this.names = json.joints.map(j => j.name);
    this.index = Object.fromEntries(this.names.map((n, i) => [n, i]));
    this.parents = json.joints.map(j => j.parent);
    this.restPos = json.joints.map(j => new Vector3(...j.position));
    this.restRot = json.joints.map(j => new Quaternion(...j.rotation).normalize());
    this.offset = this.names.map((_, j) => {
      const p = this.parents[j];
      return p < 0 ? new Vector3() : this.restPos[j].clone().sub(this.restPos[p]).applyQuaternion(this.restRot[p].clone().invert());
    });
    this.localRest = this.names.map((_, j) => {
      const p = this.parents[j];
      return p < 0 ? this.restRot[j].clone() : this.restRot[p].clone().invert().multiply(this.restRot[j]);
    });
    // Finger ends hang off the hand through finger joints the model does not drive: keep them rigid (as the official decoder does).
    this.rigid = ['LeftHandThumbEnd', 'LeftHandMiddleEnd', 'RightHandThumbEnd', 'RightHandMiddleEnd'].map(n => this.index[n]);
    this.extras = json.extras.map(e => ({
      name: e.name, parent: e.parent,
      offset: new Vector3(...e.position).sub(this.restPos[e.parent]).applyQuaternion(this.restRot[e.parent].clone().invert()),
    }));
    this.children = this.names.map(() => []);
    this.parents.forEach((p, j) => { if (p >= 0) this.children[p].push(j); });
    this.legs = [this.index.RightLeg, this.index.LeftLeg];
  }

  subtree(j) {
    const out = [j];
    for (let i = 0; i < out.length; i++) out.push(...this.children[out[i]]);
    return out;
  }

  /** Rest (T) pose. */
  restPose() {
    return { root: this.restPos[0].clone(), rot: this.restRot.map(q => q.clone()) };
  }

  /** World joint positions of a pose ({root, rot[30]}), FK with the rest bone offsets. */
  positions(pose, out = null) {
    const P = out ?? this.names.map(() => new Vector3());
    P[0].copy(pose.root);
    for (let j = 1; j < NJ; j++) {
      const p = this.parents[j];
      P[j].copy(this.offset[j]).applyQuaternion(pose.rot[p]).add(P[p]);
    }
    return P;
  }

  extraPositions(pose, P) {
    return this.extras.map(e => e.offset.clone().applyQuaternion(pose.rot[e.parent]).add(P[e.parent]));
  }

  fixRigid(pose) {
    for (const j of this.rigid) {
      const p = this.parents[j];
      pose.rot[j].copy(pose.rot[p]).multiply(this.localRest[j]);
    }
    return pose;
  }

  /** Ground heading (radians, atan2(x, z)) from the leg joints, as in training ("hips_pos"). */
  heading(P) {
    const ax = P[this.legs[0]].x - P[this.legs[1]].x, az = P[this.legs[0]].z - P[this.legs[1]].z;
    const len = Math.hypot(ax, az);
    if (len < 1e-6) return 0;
    // forward = up x across
    return Math.atan2(az / len, -ax / len);
  }
}

export const clonePose = p => ({ root: p.root.clone(), rot: p.rot.map(q => q.clone()) });

export function lerpPose(a, b, t, out = null) {
  const o = out ?? clonePose(a);
  o.root.lerpVectors(a.root, b.root, t);
  for (let j = 0; j < NJ; j++) o.rot[j].slerpQuaternions(a.rot[j], b.rot[j], t);
  return o;
}

/** Rotate and move a pose: rotate by `yaw` about the vertical through `pivot` (xz), then translate by `move`. */
export function placePose(pose, yaw, pivot, move) {
  const q = _q.setFromAxisAngle(UP, yaw);
  const out = clonePose(pose);
  out.root.sub(_v.set(pivot.x, 0, pivot.z)).applyQuaternion(q).add(_v).add(move);
  for (const r of out.rot) r.premultiply(q);
  return out;
}

// ---------------------------------------------------------------- model tensors

export function emptyBatch(n) {
  return {
    n,
    global_root_values: new Float32Array(n * 8 * 5),
    local_root_values: new Float32Array(n * 8 * 4),
    local_poses: new Float32Array(n * 8 * 267),
    has_global_root_values: new Int32Array(n * 8),
    has_local_root_values: new Int32Array(n * 8),
    has_local_poses: new Int32Array(n * 8),
    allowed_pred_num_tokens: new Int32Array(n * 11),
  };
}

/**
 * Encode request `b` of a batch: four context poses, 1-4 target poses, allowed token counts.
 * Returns the alignment ({origin, q}) needed to decode the result.
 */
export function encodeRequest(rig, batch, b, context, targets, tokens) {
  if (context.length !== 4 || targets.length < 1 || targets.length > 4) throw new Error('need 4 context poses and 1-4 targets');
  const nT = targets.length;
  const poses = [...context, ...targets];
  while (poses.length < 8) poses.push(targets[nT - 1]);
  const world = poses.map(p => rig.positions(p));
  const last = world[3];
  const alignQ = new Quaternion().setFromAxisAngle(UP, rig.heading(last) - Math.PI / 2); // last context faces +X
  const origin = new Vector3(last[0].x, 0, last[0].z);
  const inv = alignQ.clone().invert();
  const aligned = world.map(P => P.map(v => v.clone().sub(origin).applyQuaternion(inv)));
  const headings = aligned.map(P => rig.heading(P));

  const g = batch.global_root_values, l = batch.local_root_values, lp = batch.local_poses;
  for (let f = 0; f < 8; f++) {
    const next = f === 3 ? f : Math.min(f + 1, 7);
    const hip = aligned[f][0], nhip = aligned[next][0];
    const dh = Math.atan2(Math.sin(headings[next] - headings[f]), Math.cos(headings[next] - headings[f]));
    g.set([hip.x, hip.y, hip.z, Math.cos(headings[f]), Math.sin(headings[f])], (b * 8 + f) * 5);
    l.set([dh * FPS, (nhip.x - hip.x) * FPS, (nhip.z - hip.z) * FPS, hip.y], (b * 8 + f) * 4);
    const base = (b * 8 + f) * 267;
    for (let j = 0; j < NJ; j++) {
      if (j > 0) {
        const p = aligned[f][j];
        lp.set([p.x - hip.x, p.y, p.z - hip.z], base + (j - 1) * 3);
      }
      _q.copy(inv).multiply(poses[f].rot[j]);
      const xa = _v.copy(X_AXIS).applyQuaternion(_q), ya = _w.copy(Y_AXIS).applyQuaternion(_q);
      lp.set([xa.x, xa.y, xa.z, ya.x, ya.y, ya.z], base + 87 + j * 6);
    }
    const given = f < 4 + nT ? 1 : 0;
    batch.has_global_root_values[b * 8 + f] = given;
    batch.has_local_poses[b * 8 + f] = given;
    batch.has_local_root_values[b * 8 + f] = (f === 3 || f === 4 + nT - 1) ? 0 : given;
  }
  for (let i = 0; i < 11; i++) batch.allowed_pred_num_tokens[b * 11 + i] = tokens.includes(6 + i) ? 1 : 0;
  return { origin, q: alignQ };
}

function rot6d(data, o, out) {
  const x = _v.set(data[o], data[o + 1], data[o + 2]).normalize();
  const yr = _w.set(data[o + 3], data[o + 4], data[o + 5]);
  const z = yr.clone().crossVectors(x, yr).normalize();
  const y = new Vector3().crossVectors(z, x);
  _m.makeBasis(x, y, z);
  return out.setFromRotationMatrix(_m).normalize();
}

/** Decode request `b`: drop three boundary frames at each end and restore world placement. */
export function decodeRequest(rig, motions, tokensOut, b, align) {
  const tokens = tokensOut[b];
  const raw = Math.max(0, Math.min(MAX_FRAMES, tokens * 4));
  const frames = [];
  for (let f = 3; f < raw - 3; f++) {
    const o = (b * MAX_FRAMES + f) * FEAT;
    const root = new Vector3(motions[o], motions[o + 1], motions[o + 2]).applyQuaternion(align.q).add(align.origin);
    const rot = [];
    for (let j = 0; j < NJ; j++) rot.push(rot6d(motions, o + 92 + j * 6, new Quaternion()).premultiply(align.q));
    frames.push(rig.fixRigid({ root, rot }));
  }
  return { frames, tokens };
}

/**
 * Endpoint correction (as the official engine): move the first frame onto the last context pose and the last
 * frame onto the first target pose, fading each correction across the motion. Works on local rotations.
 */
export function correctEndpoints(rig, frames, lastContext, firstTarget, fixEnd = true) {
  const n = frames.length;
  if (n < 2) return frames;
  const toLocal = p => p.rot.map((q, j) => (rig.parents[j] < 0 ? q.clone() : p.rot[rig.parents[j]].clone().invert().multiply(q)));
  const L0 = toLocal(frames[0]), L1 = toLocal(frames[n - 1]);
  const C = toLocal(lastContext), T = toLocal(firstTarget);
  const dStart = C.map((c, j) => c.clone().multiply(L0[j].clone().invert()));
  const dEnd = T.map((t, j) => t.clone().multiply(L1[j].clone().invert()));
  const rStart = lastContext.root.clone().sub(frames[0].root);
  const rEnd = firstTarget.root.clone().sub(frames[n - 1].root);
  const id = new Quaternion();
  return frames.map((fr, i) => {
    const w = i / (n - 1), ws = 1 - w, we = fixEnd ? w : 0;
    const L = toLocal(fr);
    const out = { root: fr.root.clone().addScaledVector(rStart, ws).addScaledVector(rEnd, we), rot: [] };
    for (let j = 0; j < NJ; j++) {
      const ds = id.clone().slerp(dStart[j], ws), de = id.clone().slerp(dEnd[j], we);
      const local = de.multiply(ds).multiply(L[j]);
      out.rot.push(rig.parents[j] < 0 ? local : out.rot[rig.parents[j]].clone().multiply(local));
    }
    return rig.fixRigid(out);
  });
}
