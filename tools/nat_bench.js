// Naturalness bench for SOMA Brawl, run inside fight.html?debug (served by tools/serve.mjs):
//   const N = await import('/tools/nat_bench.js?v=1'); N.start({ secs: 45, seed: 1, kind: 'skilled', layers: { spec: false } });
//   ... await N.wait(40); N.result()
// Drives frames from a Worker timer (a hidden tab keeps running), plays one autoplay run, and samples the
// shown poses: refDist (distance to the nearest reference pose - src/natural.js), jerk, trunk penetration.
// `layers` switches off display layers (brawl.tune.layers) for the run and restores them after.
import { refDist, makeJerk, penetration } from '/src/natural.js';

const S = { by: {}, worst: [], jw: [], pen: { t: 0, bad: 0, worst: 0 }, res: null, err: null, tk: null, lastClock: null, n: 0, jerkers: null, saved: null };

function agg(key) { return S.by[key] ??= { d: [], j: [] }; }

function tick() {
  const brawl = window.brawl;
  try { brawl.draw(performance.now()); } catch (e) { S.err ??= String(e.stack).slice(0, 300); }
  const dtc = S.lastClock == null ? 0 : brawl.clock - S.lastClock; S.lastClock = brawl.clock;
  if (dtc <= 0.004 || dtc > 0.1) return;
  S.n++;
  const rig = brawl.rig, db = brawl.db;
  const live = brawl.fighters.filter(f => f.state !== 'out');
  for (let i = 0; i < live.length; i++) {
    const F = live[i];
    const key = (F.isPlayer ? 'P' : 'E') + ':' + F.state;
    // jerk: every tick (it needs even sampling)
    const j = S.jerkers.push(F, F.P, dtc);
    if (j != null) { agg(key).j.push(j); if (j > 1 && S.jw.length < 400) S.jw.push({ t: +brawl.clock.toFixed(2), who: F.isPlayer ? 'P' : 'E' + F.id, state: F.state, j: +j.toFixed(2) }); }
    // refDist: staggered, every 4th tick per fighter
    if ((S.n + i) % 4 === 0 && F.prevP && F.prevP.length === F.P.length) {
      const r = refDist(rig, db, F.prevP, F.P);
      agg(key).d.push(r.d);
      S.worst.push({ t: +brawl.clock.toFixed(2), who: F.isPlayer ? 'P' : 'E' + F.id, state: F.state, d: +r.d.toFixed(2), clip: r.clip });
      if (S.worst.length > 4000) { S.worst.sort((a, b) => b.d - a.d); S.worst.length = 200; }
    }
  }
  // trunk penetration between live standing pairs
  for (let a = 0; a < live.length; a++) for (let b = 0; b < live.length; b++) {
    if (a === b) continue;
    const pen = penetration(brawl.rig, live[a].P, live[b].P);
    S.pen.t += dtc / Math.max(1, live.length - 1);
    if (pen > 0.03) { S.pen.bad += dtc / Math.max(1, live.length - 1); S.pen.worst = Math.max(S.pen.worst, pen); }
  }
}

export async function start(opts = {}) {
  const { secs = 45, seed = 1, kind = 'skilled', layers = null } = opts;
  for (let i = 0; i < 60 && !window.brawl?.scheduler?.ready; i++) await new Promise(r => setTimeout(r, 1000));
  const ap = await import('/src/autoplay.js');
  const brawl = window.brawl;
  S.saved = { ...brawl.tune.layers };
  if (layers) Object.assign(brawl.tune.layers, layers);
  S.tk?.terminate();
  S.tk = new Worker(URL.createObjectURL(new Blob(['setInterval(()=>postMessage(0),16)'], { type: 'text/javascript' })));
  S.tk.onmessage = tick;
  Object.assign(S, { by: {}, worst: [], jw: [], pen: { t: 0, bad: 0, worst: 0 }, res: null, err: null, lastClock: null, n: 0, jerkers: makeJerk(brawl.rig) });
  ap.series(brawl, { secs, seeds: [seed], kinds: [kind] }).then(r => { S.res = r; stop(); }).catch(e => { S.err = String(e.stack); stop(); });
  return 'running';
}
export function stop() { S.tk?.terminate(); S.tk = null; if (S.saved) Object.assign(window.brawl.tune.layers, S.saved); }
export const done = () => !!(S.res || S.err);

export function result() {
  if (S.err) return { err: S.err };
  const q = (a, p) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return +s[Math.min(s.length - 1, Math.floor(s.length * p))].toFixed(2); };
  const mean = a => a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2) : 0;
  const by = Object.fromEntries(Object.entries(S.by).map(([k, v]) => [k, { n: v.d.length, d: mean(v.d), d95: q(v.d, 0.95), j: mean(v.j), j95: q(v.j, 0.95) }]));
  S.worst.sort((a, b) => b.d - a.d);
  return {
    done: done(), by,
    pen: { badShare: +(S.pen.bad / Math.max(1e-6, S.pen.t)).toFixed(3), worst: +S.pen.worst.toFixed(2) },
    worstPoses: S.worst.slice(0, 12),
    worstJerk: S.jw.sort((a, b) => b.j - a.j).slice(0, 8),
  };
}
export async function wait(secs = 40) { for (let i = 0; i < secs && !done(); i++) await new Promise(r => setTimeout(r, 1000)); return done(); }
