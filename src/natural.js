// Numeric naturalness of the shown poses (SOMA Brawl). Three measures, cheap enough to sample live:
// - refDist: distance from the shown pose to the nearest reference pose in the fight database (body-frame
//   features: feet, hands, head, hips height - no trajectory, no velocities). A pose no fighter in the
//   reference data ever takes scores high. Meters-ish (sqrt of the weighted squared distance).
// - jerk: third difference of limb positions (hands, feet, head, hips), averaged, m/s^3 / 1000. Pops and
//   snapping layers score high; smooth motion, even fast, stays low.
// - pen: how deep any limb point of one fighter sits inside another's trunk capsule (m); 0 when clear.
import { queryFeatures, weightVector } from './posedb.js';
import { Vector3 } from 'three';

const NATW = weightVector({ feetPos: 1, hipsH: 2, hands: 1, head: 1.5 });
const TRAJ0 = h => [{ pos: h, yaw: 0 }, { pos: h, yaw: 0 }, { pos: h, yaw: 0 }]; // (zero-weighted anyway)
const JOINTS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head']; // and the hips (index 0)

/** Distance of the shown pose (positions now and one step back) to the nearest reference pose. */
export function refDist(rig, db, Pprev, Pnow) {
  const h = rig.heading(Pnow);
  const q = queryFeatures(rig, Pprev, Pnow, h, TRAJ0(Pnow[0]), null, null);
  const r = db.search(q, NATW);
  return { d: Math.sqrt(Math.max(0, r.cost)), clip: db.clips[r.clip]?.name ?? '?', frame: r.frame };
}

/** Per-fighter jerk tracker: push positions each sampled step, read the mean third difference. */
export function makeJerk(rig) {
  const I = rig.index, js = [0, ...JOINTS.map(n => I[n])];
  const hist = new Map(); // F -> [P3, P2, P1]
  return {
    push(F, P, dt) {
      const h = hist.get(F) ?? [];
      h.push({ P: js.map(j => P[j].clone()), dt });
      if (h.length > 4) h.shift();
      hist.set(F, h);
      if (h.length < 4) return null;
      const [a, b, c, d] = h;
      const T = (b.dt + c.dt + d.dt) / 3;
      if (T < 0.004 || T > 0.1) return null;
      let s = 0;
      for (let k = 0; k < js.length; k++) {
        const v = d.P[k].clone().addScaledVector(c.P[k], -3).addScaledVector(b.P[k], 3).addScaledVector(a.P[k], -1);
        s += v.length() / (T * T * T);
      }
      return s / js.length / 1000; // (k-m/s^3)
    },
    drop(F) { hist.delete(F); },
  };
}

/** Deepest penetration of A's limb points into B's trunk capsule (hips->neck, r=0.17 + limb ~0.05). */
export function penetration(rig, A, B) {
  const I = rig.index;
  const a0 = B[0], a1 = B[I.Neck1], seg = a1.clone().sub(a0), L2 = Math.max(1e-6, seg.lengthSq());
  let worst = 0;
  for (const n of ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftShin', 'RightShin', 'Head']) {
    const p = A[I[n]];
    const t = Math.max(0, Math.min(1, p.clone().sub(a0).dot(seg) / L2));
    const d = p.distanceTo(a0.clone().addScaledVector(seg, t));
    const pen = 0.20 - d;
    if (pen > worst) worst = pen;
  }
  return worst;
}
