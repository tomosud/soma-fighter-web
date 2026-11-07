// Mixamo FBX clips (X Bot, 30 fps, no skin) -> assets/mixamo_clips.json on the SOMA rig.
//
// Each SOMA joint takes the turn its Mixamo counterpart made from the Mixamo rest (T) pose, applied after first
// lining the SOMA rest bone up with the Mixamo rest bone: R_soma(t) = D(t) * A * R_soma_rest, where
// D(t) = R_mixamo(t) * R_mixamo_rest^-1 and A turns the SOMA rest bone direction onto the Mixamo one. Joints the
// model does not drive from Mixamo (jaw, eyes, finger ends) follow their parent. The hips' path is scaled by the
// ratio of the hips heights. Output: SOMA model space (Y up, metres), world rotations of the 30 model joints. The FBX files themselves are
// not kept in the repository (tools/mixamo_src/ is ignored).
//
//   node tools/mixamo_clips.mjs <dir with mx_*.fbx> [out.json]
import fs from 'fs';
import path from 'path';
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { Rig } from '../src/soma.js';

const { Quaternion, Vector3 } = THREE;
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const rig = new Rig(JSON.parse(fs.readFileSync(path.join(root, 'assets/soma_rig.json'), 'utf8')));
const I = rig.index;

// SOMA joint -> [Mixamo bone, SOMA child for the bone direction, Mixamo child for the bone direction]
const MAP = {
  Hips: ['Hips', 'Spine1', 'Spine'],
  Spine1: ['Spine', 'Spine2', 'Spine1'],
  Spine2: ['Spine1', 'Chest', 'Spine2'],
  Chest: ['Spine2', 'Neck1', 'Neck'],
  Neck1: ['Neck', 'Neck2', 'Head'],
  Neck2: ['Neck', 'Head', 'Head'],
  Head: ['Head', null, 'HeadTop_End'],
};
for (const s of ['Left', 'Right']) {
  Object.assign(MAP, {
    [`${s}Shoulder`]: [`${s}Shoulder`, `${s}Arm`, `${s}Arm`],
    [`${s}Arm`]: [`${s}Arm`, `${s}ForeArm`, `${s}ForeArm`],
    [`${s}ForeArm`]: [`${s}ForeArm`, `${s}Hand`, `${s}Hand`],
    [`${s}Hand`]: [`${s}Hand`, `${s}HandMiddleEnd`, `${s}HandMiddle1`],
    [`${s}Leg`]: [`${s}UpLeg`, `${s}Shin`, `${s}Leg`],
    [`${s}Shin`]: [`${s}Leg`, `${s}Foot`, `${s}Foot`],
    [`${s}Foot`]: [`${s}Foot`, `${s}ToeBase`, `${s}ToeBase`],
    [`${s}ToeBase`]: [`${s}ToeBase`, null, `${s}Toe_End`],
  });
}

function load(file) {
  const buf = fs.readFileSync(file);
  const g = new FBXLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '');
  const bone = n => g.getObjectByName(`mixamorig${n}`);
  return { g, bone, clip: g.animations[0] };
}

export function convert(file) {
  const { g, bone, clip } = load(file);
  g.updateMatrixWorld(true);
  const wq = o => o.getWorldQuaternion(new Quaternion());
  const wp = o => o.getWorldPosition(new Vector3());
  // rest (T pose, before the animation is applied)
  const restQ = {}, restP = {};
  g.traverse(o => { if (o.isBone) { restQ[o.name] = wq(o); restP[o.name] = wp(o); } });
  const mx = n => `mixamorig${n}`;
  const scale = rig.restPos[0].y / restP[mx('Hips')].y;
  // A: SOMA rest bone onto the Mixamo rest bone
  const align = {};
  for (const [j, [mb, sc, mc]] of Object.entries(MAP)) {
    const ji = I[j];
    const dS = sc ? rig.restPos[I[sc]].clone().sub(rig.restPos[ji]) : new Vector3(0, 1, 0).applyQuaternion(rig.restRot[ji].clone().multiply(rig.restRot[ji].clone().invert()));
    const dM = restP[mx(mc)].clone().sub(restP[mx(mb)]);
    align[j] = sc ? new Quaternion().setFromUnitVectors(dS.normalize(), dM.normalize()) : null;
  }
  const mixer = new THREE.AnimationMixer(g);
  const action = mixer.clipAction(clip);
  action.play();
  const n = Math.max(1, Math.round(clip.duration * 30));
  const frames = [];
  for (let f = 0; f <= n; f++) {
    mixer.setTime(Math.min(clip.duration, f / 30));
    g.updateMatrixWorld(true);
    const rot = new Array(rig.names.length);
    const delta = {};
    rig.names.forEach((name, j) => {
      const m = MAP[name];
      if (m) {
        const D = wq(bone(m[0])).multiply(restQ[mx(m[0])].clone().invert());
        delta[name] = D;
        // a bone with no SOMA child to aim (head, toes) takes the alignment of its parent's bone
        const A = align[name] ?? align[rig.names[rig.parents[j]]] ?? new Quaternion();
        rot[j] = D.clone().multiply(A).multiply(rig.restRot[j]);
      } else {
        // jaw, eyes, finger ends: rigid on the parent
        const p = rig.parents[j];
        rot[j] = rot[p].clone().multiply(rig.restRot[p].clone().invert()).multiply(rig.restRot[j]);
      }
    });
    const hp = wp(bone('Hips')).multiplyScalar(scale);
    frames.push({ root: [hp.x, hp.y, hp.z].map(v => +v.toFixed(3)), rot: rot.map(q => q.normalize().toArray().map(v => +v.toFixed(3))) });
  }
  // a looping clip's last sample is its first frame again (the hips jump back): drop it
  if (frames.length > 2) {
    const a = frames.at(-2).root, b2 = frames.at(-1).root;
    if (Math.hypot(a[0] - b2[0], a[2] - b2[2]) > 0.2) frames.pop();
  }
  return { frames, scale };
}

/** How well the converted pose matches the Mixamo one: hands, feet and head (metres, after scaling). */
function check(file, frames, scale) {
  const { g, bone, clip } = load(file);
  const mixer = new THREE.AnimationMixer(g);
  mixer.clipAction(clip).play();
  let worst = 0, sum = 0, cnt = 0;
  for (let f = 0; f < frames.length; f += Math.max(1, Math.floor(frames.length / 12))) {
    mixer.setTime(Math.min(clip.duration, f / 30));
    g.updateMatrixWorld(true);
    const pose = { root: new Vector3(...frames[f].root), rot: frames[f].rot.map(q => new Quaternion(...q)) };
    const P = rig.positions(pose);
    for (const [s, m] of [['LeftHand', 'LeftHand'], ['RightHand', 'RightHand'], ['LeftFoot', 'LeftFoot'], ['RightFoot', 'RightFoot'], ['Head', 'Head']]) {
      const d = P[I[s]].distanceTo(bone(m).getWorldPosition(new Vector3()).multiplyScalar(scale));
      worst = Math.max(worst, d); sum += d; cnt++;
    }
  }
  return { mean: +(sum / cnt).toFixed(3), worst: +worst.toFixed(3) };
}

// run as a script (it is also imported by tools/build_posedb.mjs for convert())
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))) {
  const dir = process.argv[2];
  const out = process.argv[3] ?? path.join(root, 'assets/mixamo_clips.json');
  const files = fs.readdirSync(dir).filter(f => /^mx_.*\.fbx$/i.test(f)).sort();
  const clips = {};
  for (const f of files) {
    const key = f.replace(/^mx_/, '').replace(/\.fbx$/i, '');
    const file = path.join(dir, f);
    const { frames, scale } = convert(file);
    const err = check(file, frames, scale);
    // long clips (idles, getting up) are kept at 15 fps; the short ones (kicks, jumps, steps) at 30
    const step = frames.length > 120 ? 2 : 1;
    clips[key] = { fps: 30 / step, frames: frames.filter((_, k) => k % step === 0) };
    console.log(`${key.padEnd(16)} ${String(frames.length).padStart(4)} frames  error mean ${err.mean} m, worst ${err.worst} m`);
  }
  fs.writeFileSync(out, JSON.stringify({
    about: 'Mixamo animations (X Bot) retargeted to the SOMA rig by tools/mixamo_clips.mjs. SOMA model space, metres, Y up, 30 fps; per frame the hips position and the world rotations of the 30 model joints; `fps` per clip (long clips at 15).',
    joints: rig.names, clips,
  }));
  console.log(`${Object.keys(clips).length} clips -> ${out} (${(fs.statSync(out).size / 1048576).toFixed(1)} MB)`);
}
