// Authored key poses. Built from the SOMA rest skeleton with rotations and two-bone IK, in a local frame where the
// fighter stands at the origin facing +Z (toward the opponent). Only these key poses are authored: every frame
// that is shown comes from MotionBricks generating toward them.
import { Quaternion, Vector3 } from 'three';
import { clonePose, NJ } from './soma.js';

const UP = new Vector3(0, 1, 0);
const deg = d => (d * Math.PI) / 180;

export function rotateSubtree(rig, pose, j, delta, pivot = null) {
  if (j === 0 || pivot) {
    const P = rig.positions(pose);
    const c = pivot ?? P[0];
    pose.root.sub(c).applyQuaternion(delta).add(c);
  }
  for (const k of rig.subtree(j)) pose.rot[k].premultiply(delta);
  return pose;
}

export function setWorldRot(rig, pose, j, q) {
  const delta = q.clone().multiply(pose.rot[j].clone().invert());
  return rotateSubtree(rig, pose, j, delta);
}

/** Rotate joint `a` so that the bone a->b points along `dir`. */
export function aimBone(rig, pose, a, b, dir) {
  const P = rig.positions(pose);
  const cur = P[b].clone().sub(P[a]).normalize();
  return rotateSubtree(rig, pose, a, new Quaternion().setFromUnitVectors(cur, dir.clone().normalize()));
}

/** Two-bone IK: a (shoulder/hip) -> b (elbow/knee) -> c (hand/ankle) reaches `target`; `pole` bends the middle joint. */
export function twoBoneIK(rig, pose, a, b, c, target, pole) {
  let P = rig.positions(pose);
  const la = P[b].distanceTo(P[a]), lb = P[c].distanceTo(P[b]);
  const toT = target.clone().sub(P[a]);
  const d = Math.min(Math.max(toT.length(), Math.abs(la - lb) + 1e-3), (la + lb) * 0.999);
  const dir = toT.normalize();
  const cosA = (la * la + d * d - lb * lb) / (2 * la * d);
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const side = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir)));
  if (side.lengthSq() < 1e-8) side.set(0, -1, 0).sub(dir.clone().multiplyScalar(-dir.y));
  side.normalize();
  const elbow = P[a].clone().addScaledVector(dir, la * cosA).addScaledVector(side, la * sinA);
  aimBone(rig, pose, a, b, elbow.clone().sub(P[a]));
  P = rig.positions(pose);
  const hand = P[a].clone().addScaledVector(dir, d);
  aimBone(rig, pose, b, c, hand.sub(P[b]));
  return pose;
}

export function reach(rig, pose, a, c) {
  const P = rig.positions(pose);
  const b = rig.children[a][0];
  return P[b].distanceTo(P[a]) + P[c].distanceTo(P[b]);
}

export class PoseBook {
  constructor(rig) {
    this.rig = rig;
    const I = rig.index;
    this.I = I;
    this.stance = this.makeStance();
    this.hit = this.makeHit();
    this.down = this.makeDown();
  }

  footFlat(pose, foot, yaw) {
    setWorldRot(this.rig, pose, foot, new Quaternion().setFromAxisAngle(UP, yaw).multiply(this.rig.restRot[foot]));
  }

  legs(pose, left, right, leftYaw = deg(-15), rightYaw = deg(-50)) {
    const I = this.I, rig = this.rig;
    twoBoneIK(rig, pose, I.LeftLeg, I.LeftShin, I.LeftFoot, left, new Vector3(0.25, 0, 1));
    twoBoneIK(rig, pose, I.RightLeg, I.RightShin, I.RightFoot, right, new Vector3(-0.1, 0, 1));
    this.footFlat(pose, I.LeftFoot, leftYaw);
    this.footFlat(pose, I.RightFoot, rightYaw);
  }

  guard(pose, lead = true, rear = true) {
    const I = this.I, rig = this.rig;
    if (lead) twoBoneIK(rig, pose, I.LeftArm, I.LeftForeArm, I.LeftHand, new Vector3(0.12, 1.38, 0.30), new Vector3(0.6, -1, -0.2));
    if (rear) twoBoneIK(rig, pose, I.RightArm, I.RightForeArm, I.RightHand, new Vector3(-0.05, 1.40, 0.20), new Vector3(-0.6, -1, -0.3));
  }

  makeStance() {
    const rig = this.rig, I = this.I;
    const p = rig.restPose();
    p.root.y -= 0.07;
    rotateSubtree(rig, p, I.Hips, new Quaternion().setFromAxisAngle(UP, deg(-28)));
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), deg(6)));
    rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(UP, deg(22)));
    this.legs(p, new Vector3(0.13, 0.072, 0.17), new Vector3(-0.12, 0.072, -0.24));
    this.guard(p);
    return rig.fixRigid(p);
  }

  /** Stance with the lead (left) or rear (right) hand extended to `target` (local frame). */
  punch(target, lead) {
    const rig = this.rig, I = this.I;
    const p = clonePose(this.stance);
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(UP, deg(lead ? -10 : 38)));
    if (!lead) {
      rotateSubtree(rig, p, I.Hips, new Quaternion().setFromAxisAngle(UP, deg(14)));
      this.legs(p, new Vector3(0.13, 0.072, 0.17), new Vector3(-0.14, 0.072, -0.22), deg(-15), deg(-15));
    }
    this.guard(p, !lead, lead);
    const [a, b, c] = lead ? [I.LeftArm, I.LeftForeArm, I.LeftHand] : [I.RightArm, I.RightForeArm, I.RightHand];
    twoBoneIK(rig, p, a, b, c, target, new Vector3(lead ? 1 : -1, -0.6, 0));
    return rig.fixRigid(p);
  }

  /** Rear-leg (right) kick with the foot at `target` (local frame). */
  kick(target) {
    const rig = this.rig, I = this.I;
    const p = clonePose(this.stance);
    rotateSubtree(rig, p, I.Hips, new Quaternion().setFromAxisAngle(UP, deg(-10)));
    const lean = new Vector3(1, 0, 0).applyAxisAngle(UP, deg(-38));
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lean, deg(-18)));
    // support leg under the body, kicking leg to the target
    twoBoneIK(rig, p, I.LeftLeg, I.LeftShin, I.LeftFoot, new Vector3(0.08, 0.072, 0.02), new Vector3(0.2, 0, 1));
    this.footFlat(p, I.LeftFoot, deg(-60));
    twoBoneIK(rig, p, I.RightLeg, I.RightShin, I.RightFoot, target, new Vector3(-0.2, 0.8, 1));
    this.guard(p);
    return rig.fixRigid(p);
  }

  makeHit() {
    const rig = this.rig, I = this.I;
    const p = clonePose(this.stance);
    const lateral = new Vector3(1, 0, 0);
    rotateSubtree(rig, p, I.Spine1, new Quaternion().setFromAxisAngle(lateral, deg(-22)));
    rotateSubtree(rig, p, I.Neck1, new Quaternion().setFromAxisAngle(lateral, deg(-25)));
    twoBoneIK(rig, p, I.LeftArm, I.LeftForeArm, I.LeftHand, new Vector3(0.35, 1.25, 0.05), new Vector3(1, -1, 0));
    twoBoneIK(rig, p, I.RightArm, I.RightForeArm, I.RightHand, new Vector3(-0.33, 1.2, 0.0), new Vector3(-1, -1, 0));
    this.legs(p, new Vector3(0.14, 0.072, 0.08), new Vector3(-0.12, 0.072, -0.32));
    return rig.fixRigid(p);
  }

  makeDown() {
    const rig = this.rig, I = this.I;
    const p = rig.restPose();
    twoBoneIK(rig, p, I.LeftArm, I.LeftForeArm, I.LeftHand, new Vector3(0.55, 1.05, 0.05), new Vector3(1, 0, -1));
    twoBoneIK(rig, p, I.RightArm, I.RightForeArm, I.RightHand, new Vector3(-0.55, 1.05, 0.05), new Vector3(-1, 0, -1));
    twoBoneIK(rig, p, I.LeftLeg, I.LeftShin, I.LeftFoot, new Vector3(0.2, 0.15, 0.12), new Vector3(0.2, 0, 1));
    twoBoneIK(rig, p, I.RightLeg, I.RightShin, I.RightFoot, new Vector3(-0.16, 0.08, 0.05), new Vector3(-0.2, 0, 1));
    // lie on the back, head away from the opponent
    rotateSubtree(rig, p, I.Hips, new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), deg(-90)));
    const P = rig.positions(p);
    const low = Math.min(...P.map(v => v.y));
    p.root.y += 0.05 - low;
    p.root.z -= 0.0;
    return rig.fixRigid(p);
  }
}

export { NJ };
