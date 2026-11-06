// An active ragdoll: the body as weighted particles (XPBD, small steps), held to the generated motion by soft
// "muscles", with gravity, a floor with friction, and a pointer that pulls with a spring rather than pinning.
// Pulls therefore spread through the body by weight and stiffness; the result is turned back into joint rotations.
import { Quaternion, Vector3 } from 'three';

const G = new Vector3(0, -9.81, 0);

// particle: [name, parent particle, mass kg, muscle stiffness to the parent N/m, floor radius m]
const SPEC = [
  ['Hips', null, 11, 3000, 0.08],
  ['Spine1', 'Hips', 5, 5000, 0.08], ['Spine2', 'Spine1', 5, 4000, 0.08], ['Chest', 'Spine2', 9, 4000, 0.1],
  ['Neck1', 'Chest', 1.2, 2500, 0.05], ['Neck2', 'Neck1', 0.8, 2000, 0.05], ['Head', 'Neck2', 4.5, 1500, 0.09],
  ['@HeadEnd', 'Head', 0.5, 1000, 0.08],
];
for (const s of ['Left', 'Right']) {
  SPEC.push([`${s}Shoulder`, 'Chest', 1.5, 3000, 0.05], [`${s}Arm`, `${s}Shoulder`, 2.2, 1500, 0.05],
    [`${s}ForeArm`, `${s}Arm`, 1.3, 600, 0.04], [`${s}Hand`, `${s}ForeArm`, 0.8, 800, 0.03], [`${s}HandMiddleEnd`, `${s}Hand`, 0.2, 600, 0.02]);
  SPEC.push([`${s}Leg`, 'Hips', 6, 5000, 0.07], [`${s}Shin`, `${s}Leg`, 4, 2500, 0.05], [`${s}Foot`, `${s}Shin`, 1.2, 2500, 0.055],
    [`${s}ToeBase`, `${s}Foot`, 0.4, 2000, 0.018], [`@${s}ToeEnd`, `${s}ToeBase`, 0.2, 1500, 0.004]);
}
// extra distance constraints that keep the pelvis, the chest and the feet rigid
const SHAPE = [
  ['LeftLeg', 'RightLeg'], ['Spine1', 'LeftLeg'], ['Spine1', 'RightLeg'],
  ['LeftShoulder', 'RightShoulder'], ['Spine2', 'LeftShoulder'], ['Spine2', 'RightShoulder'], ['Chest', 'LeftArm'], ['Chest', 'RightArm'],
  ['LeftFoot', '@LeftToeEnd'], ['RightFoot', '@RightToeEnd'], ['Neck1', 'Head'],
];
const FEET = { Left: ['LeftFoot', 'LeftToeBase', '@LeftToeEnd'], Right: ['RightFoot', 'RightToeBase', '@RightToeEnd'] };

export class Ragdoll {
  constructor(rig) {
    this.rig = rig;
    this.names = SPEC.map(s => s[0]);
    this.index = Object.fromEntries(this.names.map((n, i) => [n, i]));
    this.parent = SPEC.map(s => (s[1] ? this.index[s[1]] : -1));
    this.w = SPEC.map(s => 1 / s[2]);
    this.mass = SPEC.map(s => s[2]);
    this.k = SPEC.map(s => s[3]);
    this.radius = SPEC.map(s => s[4]);
    this.total = this.mass.reduce((a, b) => a + b, 0);
    const rest = this.points(rig.restPose());
    this.links = [];
    this.parent.forEach((p, i) => { if (p >= 0) this.links.push([i, p, rest[i].distanceTo(rest[p])]); });
    for (const [a, b] of SHAPE) this.links.push([this.index[a], this.index[b], rest[this.index[a]].distanceTo(rest[this.index[b]])]);
    this.restY = rest.map(v => v.y);
    this.feet = { Left: FEET.Left.map(n => this.index[n]), Right: FEET.Right.map(n => this.index[n]) };
    this.isFoot = this.names.map(n => FEET.Left.includes(n) || FEET.Right.includes(n));
    this.sideOf = this.names.map(n => (n.replace('@', '').startsWith('Left') ? 'Left' : n.replace('@', '').startsWith('Right') ? 'Right' : null));
    this.dampOf = this.mass.map(m => (m < 1 ? 14 : 5));
    this.x = rest; this.p = rest.map(v => v.clone()); this.v = rest.map(() => new Vector3());
    this.rootK = 3000; this.footK = 30000; this.pullK = 500; this.pullMax = 400; this.damp = 5; this.substeps = 8;
    this.gravity = false;
    this.last = null;
    this.contactAt = this.names.map(() => false);
  }

  /** Particle positions of a pose (model joints by FK, end joints from the rig's extras). */
  points(pose) {
    const P = this.rig.positions(pose);
    const X = this.rig.extraPositions(pose, P);
    return SPEC.map(([n]) => (n[0] === '@' ? X[this.rig.extras.findIndex(e => e.name === n.slice(1))] : P[this.rig.index[n]]).clone());
  }

  reset(pose) {
    this.x = this.points(pose);
    this.p = this.x.map(v => v.clone());
    this.v = this.x.map(() => new Vector3());
    this.last = null;
  }

  /**
   * One frame. `anim`: the motion to follow; `contact`: the model's foot contacts; `grab`: { name, point } or null.
   * Planted feet are held on the floor where the motion puts them; everything else follows its parent's offset.
   */
  step(dt, anim, contact, grab) {
    // very short frames are saved up: velocities come from corrections divided by the step, so a tiny step explodes
    this.acc = (this.acc ?? 0) + dt;
    if (this.acc < 1 / 120 && this.last) return;
    dt = this.acc;
    this.acc = 0;
    const T = this.points(anim);
    // the motion's own velocity, but not across a jump (a splice or a reset) and never faster than a person moves
    const Tv = this.last ? T.map((t, i) => {
      const d = t.clone().sub(this.last[i]);
      return d.length() > 0.08 ? new Vector3() : d.divideScalar(dt).clampLength(0, 5);
    }) : T.map(() => new Vector3());
    this.last = T;
    // failsafe: if the body has been flung far from the motion, put it back on the motion
    if (this.x.some((x, i) => !Number.isFinite(x.x) || x.distanceTo(T[i]) > 1.5)) this.reset(anim);
    const planted = {
      Left: contact ? Math.max(contact[0], contact[1]) > 0.5 : true,
      Right: contact ? Math.max(contact[2], contact[3]) > 0.5 : true,
    };
    const n = this.substeps, h = dt / n, h2 = h * h;
    const x = this.x, p = this.p, v = this.v, w = this.w;
    const gi = grab ? this.index[grab.name] : -1;
    const e = new Vector3();
    for (let s = 0; s < n; s++) {
      for (let i = 0; i < x.length; i++) {
        // gravity is carried by the muscles (a living body holds its own weight); what is felt is inertia
        if (this.gravity) v[i].addScaledVector(G, h);
        v[i].lerp(Tv[i], 1 - Math.exp(-this.dampOf[i] * h)); // muscles also carry the motion's own velocity (light parts closely)
        p[i].copy(x[i]);
        x[i].addScaledVector(v[i], h);
      }
      // root: held to where the motion puts the hips
      this.point(0, T[0], this.rootK, h2);
      // muscles: each particle keeps its offset from its parent as in the motion
      for (let i = 1; i < x.length; i++) {
        const q = this.parent[i];
        e.copy(x[i]).sub(x[q]).sub(T[i]).add(T[q]);
        const a = 1 / (this.k[i] * h2);
        const d = -1 / (w[i] + w[q] + a);
        x[i].addScaledVector(e, d * w[i]);
        x[q].addScaledVector(e, -d * w[q]);
      }
      // planted feet: on the floor, where the motion plants them
      for (const side of ['Left', 'Right']) {
        if (!planted[side] || grab?.name.startsWith(side) && grab.name.endsWith('Foot')) continue;
        for (const i of this.feet[side]) {
          const t = T[i].clone();
          t.y = Math.min(t.y, this.restY[i]);
          this.point(i, t, this.footK, h2);
        }
      }
      // the pointer pulls with a spring (its force is capped), it does not pin
      if (gi >= 0) {
        e.copy(x[gi]).sub(grab.point);
        if (gi === 0) e.y *= 0.2; // the hips are dragged along, not lifted
        const a = 1 / (this.pullK * h2);
        const d = e.multiplyScalar(-w[gi] / (w[gi] + a));
        const cap = (gi === 0 ? this.pullMax * 0.375 : this.pullMax) * h2 * w[gi]; // the hips: 150 N, the walk does the rest
        if (d.length() > cap) d.setLength(cap);
        x[gi].add(d);
      }
      // bones and rigid parts
      for (let it = 0; it < 2; it++) {
        for (const [i, j, len] of this.links) {
          e.copy(x[i]).sub(x[j]);
          const l = e.length();
          if (l < 1e-9) continue;
          const c = (l - len) / l / (w[i] + w[j]);
          x[i].addScaledVector(e, -c * w[i]);
          x[j].addScaledVector(e, c * w[j]);
        }
      }
      // floor: nothing goes through it; feet grip (they keep only part of their sliding), the rest slides
      for (let i = 0; i < x.length; i++) {
        const r = this.radius[i];
        this.contactAt[i] = x[i].y <= r + 0.003;
        if (x[i].y < r) {
          x[i].y = r;
          if (this.isFoot[i] && planted[this.sideOf[i]]) {
            x[i].x = p[i].x + (x[i].x - p[i].x) * 0.5;
            x[i].z = p[i].z + (x[i].z - p[i].z) * 0.5;
          }
        }
      }
      for (let i = 0; i < x.length; i++) v[i].copy(x[i]).sub(p[i]).divideScalar(h);
    }
  }

  point(i, target, k, h2) {
    const e = this.x[i].clone().sub(target);
    const a = 1 / (k * h2);
    this.x[i].addScaledVector(e, -this.w[i] / (this.w[i] + a));
  }

  /** Centre of mass and its velocity. */
  com() {
    const c = new Vector3(), cv = new Vector3();
    this.x.forEach((x, i) => { c.addScaledVector(x, this.mass[i]); cv.addScaledVector(this.v[i], this.mass[i]); });
    return { pos: c.divideScalar(this.total), vel: cv.divideScalar(this.total) };
  }

  /** Floor contact points of the feet (x, z). */
  support() {
    const out = [];
    for (const side of ['Left', 'Right']) for (const i of this.feet[side]) if (this.x[i].y < this.radius[i] + 0.02) out.push([this.x[i].x, this.x[i].z]);
    return out;
  }

  /**
   * The motion's pose turned so each bone points where the particles are: rotations are inherited down the
   * hierarchy and each joint is aimed at its main child (the hips and chest also keep their left-right line).
   */
  pose(anim) {
    const rig = this.rig, ix = this.index, x = this.x;
    const rot = [];
    const P = rig.positions(anim);
    const X = rig.extraPositions(anim, P);
    const animPt = n => (n[0] === '@' ? X[rig.extras.findIndex(e => e.name === n.slice(1))] : P[rig.index[n]]);
    for (let j = 0; j < rig.names.length; j++) {
      const name = rig.names[j], par = rig.parents[j];
      let r = par < 0 ? anim.rot[j].clone() : rot[par].clone().multiply(anim.rot[par].clone().invert()).multiply(anim.rot[j]);
      const child = MAIN[name];
      if (child && ix[name] !== undefined) {
        // the bone as it is after inheriting, and as the particles have it
        const inherit = r.clone().multiply(anim.rot[j].clone().invert());
        const from = animPt(child).clone().sub(animPt(name)).applyQuaternion(inherit);
        const to = x[ix[child]].clone().sub(x[ix[name]]);
        let q = new Quaternion().setFromUnitVectors(from.normalize(), to.clone().normalize());
        const across = ACROSS[name];
        if (across) { // turn about the bone so the left-right line matches too
          const a0 = animPt(across[0]).clone().sub(animPt(across[1])).applyQuaternion(inherit).applyQuaternion(q);
          const a1 = x[ix[across[0]]].clone().sub(x[ix[across[1]]]);
          const axis = to.clone().normalize();
          a0.addScaledVector(axis, -a0.dot(axis));
          a1.addScaledVector(axis, -a1.dot(axis));
          if (a0.lengthSq() > 1e-8 && a1.lengthSq() > 1e-8) q = new Quaternion().setFromUnitVectors(a0.normalize(), a1.normalize()).multiply(q);
        }
        r = q.multiply(r);
      }
      rot.push(r.normalize());
    }
    const out = rig.fixRigid({ root: x[0].clone(), rot });
    if (anim.contact) out.contact = anim.contact;
    return out;
  }
}

const MAIN = { Hips: 'Spine1', Spine1: 'Spine2', Spine2: 'Chest', Chest: 'Neck1', Neck1: 'Neck2', Neck2: 'Head', Head: '@HeadEnd' };
const ACROSS = { Hips: ['LeftLeg', 'RightLeg'], Chest: ['LeftShoulder', 'RightShoulder'] };
for (const s of ['Left', 'Right']) {
  Object.assign(MAIN, { [`${s}Shoulder`]: `${s}Arm`, [`${s}Arm`]: `${s}ForeArm`, [`${s}ForeArm`]: `${s}Hand`, [`${s}Hand`]: `${s}HandMiddleEnd`,
    [`${s}Leg`]: `${s}Shin`, [`${s}Shin`]: `${s}Foot`, [`${s}Foot`]: `${s}ToeBase`, [`${s}ToeBase`]: `@${s}ToeEnd` });
}
