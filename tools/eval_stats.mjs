// Reference statistics for the evaluator: the natural spread of each residual, measured on the Mixamo clips.
//   node tools/eval_stats.mjs [clips.json ...] -> assets/eval_stats.json
import fs from 'fs';
import path from 'path';
import { Quaternion, Vector3 } from 'three';
import { Rig } from '../src/soma.js';
import { referenceStats, EVAL_FPS } from '../src/evaluate.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const rig = new Rig(JSON.parse(fs.readFileSync(path.join(root, 'assets/soma_rig.json'), 'utf8')));
const inputs = process.argv.slice(2).length ? process.argv.slice(2) : [path.join(root, 'assets/mixamo_clips.json')];
const clips = [];
// the pose database (its clips' own frames, without the held or repeated padding)
if (inputs.length === 1 && inputs[0].endsWith('posedb')) {
  const { PoseDB } = await import('../src/posedb.js');
  const meta = JSON.parse(fs.readFileSync(inputs[0] + '.json', 'utf8'));
  const b = fs.readFileSync(inputs[0] + '.bin');
  const db = new PoseDB(rig, meta, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  meta.clips.forEach((c, k) => clips.push(Array.from({ length: c.own ?? c.frames }, (_, f) => db.pose(k, f))));
  inputs.length = 0;
}
for (const file of inputs) {
  const j = JSON.parse(fs.readFileSync(file, 'utf8'));
  for (const [name, c] of Object.entries(j.clips)) {
    if ((c.fps ?? 30) !== EVAL_FPS) continue;
    clips.push(c.frames.map(f => rig.fixRigid({ root: new Vector3(...f.root), rot: f.rot.map(q => new Quaternion(...q).normalize()) })));
  }
}
const stats = referenceStats(rig, clips);
fs.writeFileSync(path.join(root, 'assets/eval_stats.json'), JSON.stringify(stats));
const show = (label, S) => `${label}: ${stats.joints.slice(0, 3).map((n, k) => `${n} pos ${S.pos[k].med.toFixed(4)}±${S.pos[k].mad.toFixed(4)} rot ${S.rot[k].med.toFixed(4)}±${S.rot[k].mad.toFixed(4)}`).join(' | ')} | hips speed ${S.hipSpeed.med.toFixed(3)}±${S.hipSpeed.mad.toFixed(3)}`;
console.log(`${clips.length} clips, samples standing ${stats.samples[0]} moving ${stats.samples[1]}`);
console.log(show('standing', stats.still));
console.log(show('moving', stats.moving));
