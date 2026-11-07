// Scripted drag scenarios for the evaluator. Works with any page exposing the debug API (`?debug`): pull(part,
// point), draw(), shown, shownP, rig, track, scheduler. Records every displayed frame, scores it against the
// Mixamo reference statistics, draws a heat map and saves both to the local server (eval_out/).
//   const m = await import('/src/evalrun.js'); await m.runAll(lab, 'baseline');
import { Vector3 } from 'three';
import { EVAL_FPS, score, drawHeatmap } from './evaluate.js';

// a hidden tab throttles its timers to once a second after a few minutes: tick from a worker instead
let ticker = null;
const wait = () => new Promise(r => {
  ticker ??= new Worker(URL.createObjectURL(new Blob(['setInterval(() => postMessage(0), 16)'], { type: 'text/javascript' })));
  ticker.addEventListener('message', () => r(), { once: true });
});

/** Each scenario: duration (s) and where the grabbed point goes (relative to the body at the start). */
export const SCENARIOS = [
  { name: 'idle', sec: 5, part: null },
  { name: 'hand_walk', sec: 9, part: 'RightHand', path: (t, s) => s.at('RightArm').add(s.fwd(t < 7 ? 1.1 : 0.4)).setY(1.25), follow: true, release: 7 },
  { name: 'hips_wind', sec: 12, part: 'Hips', path: (t, s) => { const a = 0.9 * Math.sin(t * 0.5); return s.at('Hips').add(s.dir(a, 0.75)); }, follow: true, release: 10 },
  { name: 'head_side', sec: 6, part: 'Head', path: (t, s) => s.p0('Head').add(s.left(0.35 * Math.min(1, t / 1.2))).add(s.fwd(0.1)), release: 4 },
  { name: 'foot_lift', sec: 6, part: 'RightFoot', path: (t, s) => s.p0('RightFoot').add(s.fwd(0.35 * Math.min(1, t / 1.5))).add(new Vector3(0, 0.45 * Math.min(1, t / 1.5), 0)), release: 4 },
  { name: 'hand_yank', sec: 5, part: 'LeftHand', path: (t, s) => s.p0('LeftHand').add(s.left(0.9 * Math.min(1, t / 0.2))), release: 1.5 },
  { name: 'hips_down', sec: 7, part: 'Hips', path: (t, s) => s.p0('Hips').add(new Vector3(0, -0.35 * Math.sin(Math.min(Math.PI, t / 4 * Math.PI)), 0)), release: 5 },
  { name: 'turn_back', sec: 8, part: 'RightHand', path: (t, s) => s.at('RightArm').add(s.dir(Math.PI * Math.min(1, t / 1.5), 1.0)).setY(1.2), follow: true, release: 6 },
];

/** Run one scenario; returns { poses (30 fps), marks }. */
export async function runScenario(lab, sc) {
  const rig = lab.rig, I = rig.index;
  const P0 = lab.shownP.map(v => v.clone());
  const h0 = rig.heading(P0);
  const F = new Vector3(Math.sin(h0), 0, Math.cos(h0)), L = new Vector3(Math.cos(h0), 0, -Math.sin(h0));
  const s = {
    p0: n => P0[I[n]].clone(),
    at: n => lab.shownP[I[n]].clone(), // follows the body (the point leads it)
    fwd: d => F.clone().multiplyScalar(d),
    left: d => L.clone().multiplyScalar(d),
    dir: (a, d) => F.clone().multiplyScalar(Math.cos(a) * d).add(L.clone().multiplyScalar(Math.sin(a) * d)),
  };
  const frames = [], marks = [], follow = [];
  let step = 0, lastSplices = lab.track.splices ?? 0;
  lab.onStep = shown => {
    if (step++ % 2 === 0) { // 60 Hz simulation -> 30 Hz
      frames.push({ root: shown.root.clone(), rot: shown.rot.map(q => q.clone()), contact: shown.contact ? [...shown.contact] : null });
      // how far the grabbed part is from the pointer (while it is held)
      if (lab.drag && sc.part) follow.push(lab.shownP[I[sc.part]].distanceTo(lab.drag.point));
      const sp = lab.track.splices ?? 0;
      if (sp !== lastSplices) { marks.push({ t: frames.length - 1, color: 'rgba(255,255,255,0.25)' }); lastSplices = sp; }
    }
  };
  if (sc.part) { lab.pull(sc.part, lab.shownP[I[sc.part]].clone()); marks.push({ t: 0, label: `つかむ ${sc.part}`, color: '#6cf' }); }
  const t0 = performance.now();
  let released = !sc.part;
  while (performance.now() - t0 < sc.sec * 1000) {
    const t = (performance.now() - t0) / 1000;
    if (!released && sc.release != null && t >= sc.release) { lab.pull(null); released = true; marks.push({ t: frames.length, label: '離す', color: '#f96' }); }
    if (!released && lab.drag && sc.path) lab.drag.point.copy(sc.path(t, s));
    lab.draw();
    await wait();
  }
  if (!released) lab.pull(null);
  lab.onStep = null;
  const held = follow.slice(Math.min(follow.length, 30)); // after the first second
  return { poses: frames, marks, follow: held.length ? +(held.reduce((a, b) => a + b, 0) / held.length).toFixed(3) : 0 };
}

const packRec = rec => JSON.stringify({ marks: rec.marks, follow: rec.follow, frames: rec.poses.map(p => ({ r: p.root.toArray().map(v => +v.toFixed(4)), q: p.rot.map(q => q.toArray().map(v => +v.toFixed(5))), c: p.contact })) });
const unpackRec = (j, Q, V) => ({ marks: j.marks, follow: j.follow, poses: j.frames.map(f => ({ root: new V(...f.r), rot: f.q.map(q => new Q(...q)), contact: f.c })) });

/** Score a recording, draw its heat map and save both (eval_out/<tag>_<name>.png, scores in the summary). */
async function report(rig, rec, stats, tag, name) {
  const res = score(rig, rec.poses, stats, rec.marks);
  const canvas = document.createElement('canvas');
  drawHeatmap(canvas.getContext('2d'), res, `${tag} / ${name}  meanZ ${res.summary.meanZ}  p95 ${res.summary.p95Z}  events/min ${res.summary.eventsPerMin}  slide ${res.summary.slideCmPerS} cm/s  (${EVAL_FPS} fps)`);
  const png = await new Promise(r => canvas.toBlob(r, 'image/png'));
  await fetch(`/save/${tag}_${name}.png`, { method: 'POST', body: png });
  return res.summary;
}

async function summarize(tag, results) {
  const keys = Object.keys(results);
  const avg = k => +(keys.reduce((a, n) => a + results[n][k], 0) / Math.max(1, keys.length)).toFixed(3);
  const total = { meanZ: avg('meanZ'), p95Z: avg('p95Z'), eventsPerMin: avg('eventsPerMin'), slideCmPerS: avg('slideCmPerS'), slideBadPlanted: avg('slideBadPlanted'), follow: avg('follow') };
  await fetch(`/save/${tag}_scores.json`, { method: 'POST', body: JSON.stringify({ tag, total, results }, null, 1) });
  return { total, results };
}

export async function runAll(lab, tag = 'run', only = null) {
  const stats = await (await fetch(new URL('../assets/eval_stats.json', import.meta.url))).json();
  const results = {};
  for (const sc of SCENARIOS) {
    if (only && !only.includes(sc.name)) continue;
    // settle first
    for (const t0 = performance.now(); performance.now() - t0 < 1500;) { lab.draw(); await wait(); }
    const rec = await runScenario(lab, sc);
    await fetch(`/save/${tag}_${sc.name}.rec.json`, { method: 'POST', body: packRec(rec) });
    results[sc.name] = { ...(await report(lab.rig, rec, stats, tag, sc.name)), follow: rec.follow };
  }
  return summarize(tag, results);
}

/** Score saved recordings again (after the evaluator or its reference statistics changed). */
export async function rescore(rig, tag) {
  const { Quaternion, Vector3: V } = await import('three');
  const stats = await (await fetch(new URL('../assets/eval_stats.json', import.meta.url))).json();
  const results = {};
  for (const sc of SCENARIOS) {
    const r = await fetch(`/eval_out/${tag}_${sc.name}.rec.json`);
    if (!r.ok) continue;
    const rec = unpackRec(await r.json(), Quaternion, V);
    results[sc.name] = { ...(await report(rig, rec, stats, tag, sc.name)), follow: rec.follow ?? 0 };
  }
  return summarize(tag, results);
}
