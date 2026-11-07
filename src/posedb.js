// The reference pose database (Mixamo clips on the SOMA rig) and its motion-matching features.
//
// Features are taken in the body frame of the frame they describe: origin = hips projected to the floor, z = facing
// (rig.heading, the same definition MotionBricks aligns with, averaged over a few frames so it does not flicker),
// x = the body's left. Lengths in metres (the clips are already on the SOMA rig), velocities in m/s.
//
// layout (DIM = 64):
//   0-5   feet position L, R          6-11  feet velocity L, R
//   12-13 hips velocity (x, z)        14    hips height        15 hips vertical velocity
//   16-21 hands position L, R         22-24 head position      25-30 hands velocity L, R
//   31-36 hips position +10/+20/+30 frames (x, z)   37-42 facing +10/+20/+30 frames (x, z)
//   43-60 where each handle is at +D frames (LeftHand, RightHand, LeftFoot, RightFoot, Head, Hips; x, y, z)
//   61-63 hips height change +10/+20/+30 frames (so standing still does not pick the moment before a jump or a fall)
import { Quaternion, Vector3 } from 'three';

export const DIM = 64;
export const AHEAD = 24; // frames from the matched frame to the target (what a 6-token MotionBricks plan spans)
export const TRAJ = [10, 20, 30];
export const HANDLES = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head', 'Hips'];
export const GROUPS = {
  feetPos: [0, 6], feetVel: [6, 12], hipsVel: [12, 14], hipsH: [14, 15], hipsVy: [15, 16], hands: [16, 22], head: [22, 25],
  handsVel: [25, 31], trajPos: [31, 37], trajDir: [37, 43], handle: [43, 61], trajH: [61, 64],
};
export const handleSlot = part => 43 + 3 * HANDLES.indexOf(part);

const UP = new Vector3(0, 1, 0);
const clampH = v => Math.max(-0.5, Math.min(0.3, v));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/** Mean of headings (circular). */
export function meanHeading(hs) {
  let s = 0, c = 0;
  for (const h of hs) { s += Math.sin(h); c += Math.cos(h); }
  return Math.atan2(s, c);
}

/** A body frame: world point -> (left, up, forward) coordinates; world direction -> (left, forward). */
export function bodyFrame(hip, h) {
  const c = Math.cos(h), s = Math.sin(h);
  return {
    hip, h,
    point: v => { const dx = v.x - hip.x, dz = v.z - hip.z; return [dx * c - dz * s, v.y, dx * s + dz * c]; },
    dir: v => [v.x * c - v.z * s, v.x * s + v.z * c],
    // back to the world: body (x, y, z) -> world
    world: (x, y, z) => new Vector3(hip.x + x * c + z * s, y, hip.z - x * s + z * c),
  };
}

/** Left-right mirror of a pose (swap the Left / Right joints, reflect across x = 0). */
export function makeMirror(rig) {
  const I = rig.index;
  const M = rig.names.map(n => I[n.startsWith('Left') ? `Right${n.slice(4)}` : n.startsWith('Right') ? `Left${n.slice(5)}` : n]);
  const reflect = q => new Quaternion(q.x, -q.y, -q.z, q.w);
  return pose => {
    const rot = pose.rot.map((_, j) => { const m = M[j]; return reflect(pose.rot[m].clone().multiply(rig.restRot[m].clone().invert())).multiply(rig.restRot[j]); });
    const out = rig.fixRigid({ root: new Vector3(-pose.root.x, pose.root.y, pose.root.z), rot });
    if (pose.contact) out.contact = [pose.contact[2], pose.contact[3], pose.contact[0], pose.contact[1]];
    return out;
  };
}

/** Joint positions and headings of a clip (computed once per clip). */
export function clipCache(rig, poses) {
  const P = poses.map(p => rig.positions(p));
  const head = P.map(Q => rig.heading(Q));
  return { poses, P, head };
}

/** The smoothed heading of frame f of a cached clip. */
const smoothHeading = (C, f) => meanHeading([-2, -1, 0, 1, 2].map(o => C.head[Math.max(0, Math.min(C.head.length - 1, f + o))]));

/** Features of frame f of a clip (needs f-1 and f+30, f+AHEAD). */
export function features(rig, C, f, out = new Float32Array(DIM)) {
  const I = rig.index, P = C.P, fps = 30;
  const B = bodyFrame(P[f][0], smoothHeading(C, f));
  const put = (k, a) => { for (let i = 0; i < a.length; i++) out[k + i] = a[i]; };
  const vel = (j) => { const a = P[f - 1][j], b = P[f][j]; return B.dir(new Vector3((b.x - a.x) * fps, 0, (b.z - a.z) * fps)).concat(); };
  const pv = j => { const a = B.point(P[f - 1][j]), b = B.point(P[f][j]); return [(b[0] - a[0]) * fps, (b[1] - a[1]) * fps, (b[2] - a[2]) * fps]; };
  put(0, B.point(P[f][I.LeftFoot])); put(3, B.point(P[f][I.RightFoot]));
  put(6, pv(I.LeftFoot)); put(9, pv(I.RightFoot));
  put(12, vel(0)); out[14] = P[f][0].y; out[15] = (P[f][0].y - P[f - 1][0].y) * fps;
  put(16, B.point(P[f][I.LeftHand])); put(19, B.point(P[f][I.RightHand])); put(22, B.point(P[f][I.Head]));
  put(25, pv(I.LeftHand)); put(28, pv(I.RightHand));
  TRAJ.forEach((d, k) => {
    const g = P[f + d][0];
    const p = B.point(g);
    out[31 + 2 * k] = p[0]; out[32 + 2 * k] = p[2];
    const h = smoothHeading(C, f + d);
    const dir = B.dir(new Vector3(Math.sin(h), 0, Math.cos(h)));
    out[37 + 2 * k] = dir[0]; out[38 + 2 * k] = dir[1];
    out[61 + k] = g.y - P[f][0].y;
  });
  HANDLES.forEach((n, k) => put(43 + 3 * k, B.point(P[f + AHEAD][I[n]])));
  return out;
}

/**
 * Query features from what SOMA is doing (the last two context frames, positions given) and what is wanted
 * (hips trajectory [{pos, yaw}] at TRAJ frames ahead; the grabbed handle's target in the world, or null).
 */
export function queryFeatures(rig, Pprev, Pnow, heading, traj, part, target, out = new Float32Array(DIM)) {
  const I = rig.index, fps = 30;
  const B = bodyFrame(Pnow[0], heading);
  const put = (k, a) => { for (let i = 0; i < a.length; i++) out[k + i] = a[i]; };
  const pv = j => { const a = B.point(Pprev[j]), b = B.point(Pnow[j]); return [(b[0] - a[0]) * fps, (b[1] - a[1]) * fps, (b[2] - a[2]) * fps]; };
  put(0, B.point(Pnow[I.LeftFoot])); put(3, B.point(Pnow[I.RightFoot]));
  put(6, pv(I.LeftFoot)); put(9, pv(I.RightFoot));
  put(12, B.dir(new Vector3((Pnow[0].x - Pprev[0].x) * fps, 0, (Pnow[0].z - Pprev[0].z) * fps)));
  out[14] = Pnow[0].y; out[15] = (Pnow[0].y - Pprev[0].y) * fps;
  put(16, B.point(Pnow[I.LeftHand])); put(19, B.point(Pnow[I.RightHand])); put(22, B.point(Pnow[I.Head]));
  put(25, pv(I.LeftHand)); put(28, pv(I.RightHand));
  traj.forEach((t, k) => {
    const p = B.point(t.pos);
    out[31 + 2 * k] = p[0]; out[32 + 2 * k] = p[2];
    const dir = B.dir(new Vector3(Math.sin(t.yaw), 0, Math.cos(t.yaw)));
    out[37 + 2 * k] = dir[0]; out[38 + 2 * k] = dir[1];
  });
  out.fill(0, 43, 64);
  if (part && target) put(handleSlot(part), B.point(target));
  // hips height: kept, unless the hips themselves are pulled up or down
  if (part === 'Hips' && target) TRAJ.forEach((d, k) => { out[61 + k] = clampH(target.y - Pnow[0].y) * Math.min(1, d / AHEAD); });
  return out;
}

// ---------------------------------------------------------------- binary database

/**
 * Loaded database: clips (pose arrays, decoded on demand), entries (clip, frame, mirrored) and their quantised
 * features, per-dimension scales.
 */
export class PoseDB {
  constructor(rig, meta, buf) {
    this.rig = rig;
    this.meta = meta;
    this.mirror = makeMirror(rig);
    const nj = rig.names.length;
    this.frameBytes = 12 + nj * 8;
    this.poseView = new DataView(buf, meta.poses.offset, meta.poses.bytes);
    this.clips = meta.clips;
    this.n = meta.entries.count;
    this.entClip = new Uint16Array(buf, meta.entries.clipOffset, this.n);
    this.entFrame = new Uint16Array(buf, meta.entries.frameOffset, this.n);
    this.entMirror = new Uint8Array(buf, meta.entries.mirrorOffset, this.n);
    this.feat = new Int16Array(buf, meta.features.offset, this.n * DIM);
    this.scale = Float32Array.from(meta.features.scale);
    this.q = meta.features.quant;
    this.cache = new Map();
  }

  static async load(rig, url) {
    const meta = await (await fetch(url + '.json')).json();
    const buf = await (await fetch(url + '.bin')).arrayBuffer();
    return new PoseDB(rig, meta, buf);
  }

  /** Pose `f` of clip `c` (canonical: the clip's first frame faces +Z with its hips over the origin). */
  pose(c, f, mirrored = false) {
    const clip = this.clips[c];
    f = Math.max(0, Math.min(clip.frames - 1, f));
    const key = (c << 17) | (f << 1) | (mirrored ? 1 : 0);
    let p = this.cache.get(key);
    if (p) return p;
    const nj = this.rig.names.length, v = this.poseView;
    let o = (clip.start + f) * this.frameBytes;
    const root = new Vector3(v.getFloat32(o, true), v.getFloat32(o + 4, true), v.getFloat32(o + 8, true));
    o += 12;
    const rot = [];
    for (let j = 0; j < nj; j++, o += 8) rot.push(new Quaternion(v.getInt16(o, true) / 32767, v.getInt16(o + 2, true) / 32767, v.getInt16(o + 4, true) / 32767, v.getInt16(o + 6, true) / 32767).normalize());
    p = this.rig.fixRigid({ root, rot });
    if (mirrored) p = this.mirror(p);
    if (this.cache.size > 4000) this.cache.clear();
    this.cache.set(key, p);
    return p;
  }

  /**
   * Best entry for a query: weighted squared distance over the standardised features, plus `bias(i)` (a cost
   * added per entry, e.g. to stay on the current clip). Brute force (a few ms for tens of thousands of entries).
   */
  search(q, weights, bias = null) {
    const n = this.n, F = this.feat, s = this.scale, k = 1 / this.q;
    const qs = new Float32Array(DIM), w = new Float32Array(DIM);
    for (let d = 0; d < DIM; d++) { qs[d] = q[d] / s[d]; w[d] = weights[d]; }
    let best = -1, bc = Infinity;
    for (let i = 0; i < n; i++) {
      let c = bias ? bias(i) : 0, o = i * DIM;
      for (let d = 0; d < DIM && c < bc; d++) { const wd = w[d]; if (wd === 0) continue; const e = F[o + d] * k - qs[d]; c += wd * e * e; }
      if (c < bc) { bc = c; best = i; }
    }
    return { i: best, cost: bc, clip: this.entClip[best], frame: this.entFrame[best], mirrored: !!this.entMirror[best] };
  }

  /** The feature vector of entry i (world units). */
  entryFeatures(i) {
    const out = new Float32Array(DIM), k = 1 / this.q;
    for (let d = 0; d < DIM; d++) out[d] = this.feat[i * DIM + d] * k * this.scale[d];
    return out;
  }
}

/** Feature weights per group (zeros mean "not compared"). */
export function weightVector(g) {
  const w = new Float32Array(DIM);
  for (const [name, [a, b]] of Object.entries(GROUPS)) for (let d = a; d < b; d++) w[d] = g[name] ?? 0;
  return w;
}
export { wrap };
