// SOMA Brawl bench, run inside fight.html?debug (the page served by tools/serve.mjs):
//   const b = await import('/tools/swing_bench.js'); b.start(); ... await b.result()
// Drives the frames from a Worker timer (so a hidden tab keeps running), plays the autoplay series and measures,
// besides the series' own scores:
//   - swing: per strike, the striking limb's tip relative to the hips in the body's facing frame from the strike's
//     start for 1 s: amp (largest distance between two tip positions), reach (how far forward at the top), pull (from
//     the rearmost point before the top to the top: the wind-up and swing together)
//   - pops: frames where a hand jumps more than 0.25 m relative to the hips from one drawn frame to the next
//   - starve: the share of time each fighter's motion has run out (waiting for the model)
const S = { sw: new Map(), by: {}, pops: { n: 0, big: 0 }, prev: new Map(), errs: [], res: null, err: null, tk: null };
const PARTS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'LeftForeArm', 'RightForeArm', 'LeftShin', 'RightShin'];

function tick() {
  const brawl = window.brawl, I = brawl.rig.index;
  try { brawl.draw(performance.now()); } catch (e) { if (S.errs.length < 5) S.errs.push(String(e.stack).slice(0, 300)); }
  for (const F of brawl.fighters) {
    if (F.state === 'out') continue;
    if (F.windAt != null) { const key = F.id + ':' + F.windAt; if (!S.sw.has(key)) S.sw.set(key, { who: F.isPlayer ? 'P' : 'E', kind: F.windKind, t0: brawl.clock, F, pts: [] }); }
    const P = F.P, hands = ['LeftHand', 'RightHand'].map(n => P[I[n]].clone().sub(P[0])), pv = S.prev.get(F);
    if (pv) { S.pops.n++; if (Math.max(hands[0].distanceTo(pv[0]), hands[1].distanceTo(pv[1])) > 0.25) S.pops.big++; }
    S.prev.set(F, hands);
  }
  for (const s of S.sw.values()) {
    if (s.done) continue;
    const F = s.F;
    if (brawl.clock - s.t0 > 1.0) { s.done = true; continue; }
    if (F.attack?.part) s.part ??= F.attack.part;
    const P = F.P, l = P[I.LeftArm], r = P[I.RightArm];
    let fx = -(l.z - r.z), fz = l.x - r.x; const n = Math.hypot(fx, fz) || 1; fx /= n; fz /= n;
    s.pts.push(Object.fromEntries(PARTS.map(pt => { const tp = P[I[pt]], dx = tp.x - P[0].x, dz = tp.z - P[0].z; return [pt, [dx * fx + dz * fz, dx * fz - dz * fx, tp.y - P[0].y]]; })));
  }
  brawl.fighters.forEach((F, i) => { const k = i === 0 ? 'P' : 'E' + i; const s = S.by[k] ??= { n: 0, starve: 0 }; if (F.state === 'out') return; s.n++; if (F.track.remaining < 1) s.starve++; });
}

export function swing() {
  const out = { P: [], E: [] };
  for (const s of S.sw.values()) {
    if (!s.done || !s.part) continue;
    const p = s.pts.map(o => o[s.part]);
    if (p.length < 10) continue;
    let amp = 0; for (const a of p) for (const b of p) amp = Math.max(amp, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    let iMax = 0; p.forEach((v, i) => { if (v[0] > p[iMax][0]) iMax = i; });
    const back = Math.min(...p.slice(0, iMax + 1).map(v => v[0]));
    out[s.who].push({ kind: s.kind, amp, reach: p[iMax][0], pull: p[iMax][0] - back });
  }
  const sum = a => a.length ? { n: a.length, amp: +(a.reduce((x, y) => x + y.amp, 0) / a.length).toFixed(2), reach: +(a.reduce((x, y) => x + y.reach, 0) / a.length).toFixed(2), pull: +(a.reduce((x, y) => x + y.pull, 0) / a.length).toFixed(2) } : null;
  return { P: sum(out.P), Pp: sum(out.P.filter(x => x.kind === 'punch')), Pk: sum(out.P.filter(x => x.kind === 'kick')), E: sum(out.E), Ep: sum(out.E.filter(x => x.kind === 'punch')), Ek: sum(out.E.filter(x => x.kind === 'kick')) };
}

/** Start: waits for the model, then plays the series (default: skilled, novice, person, 60 s each). */
export async function start(opts = { secs: 60, seeds: [1], kinds: ['skilled', 'novice', 'person'] }) {
  for (let i = 0; i < 60 && !window.brawl?.scheduler?.ready; i++) await new Promise(r => setTimeout(r, 1000));
  const ap = await import('/src/autoplay.js');
  S.tk?.terminate();
  S.tk = new Worker(URL.createObjectURL(new Blob(['setInterval(()=>postMessage(0),16)'])));
  S.tk.onmessage = tick;
  await new Promise(r => setTimeout(r, 2000));
  S.sw.clear(); S.by = {}; S.pops = { n: 0, big: 0 }; S.res = null; S.err = null;
  ap.series(window.brawl, opts).then(r => { S.res = r; }).catch(e => { S.err = String(e.stack); });
  return { ready: window.brawl.scheduler.ready };
}
export function stop() { S.tk?.terminate(); S.tk = null; }
export const done = () => !!(S.res || S.err);
/** The result once the series is over (null before). */
export function result() {
  if (S.err) return { err: S.err };
  if (!S.res) return null;
  const pick = /^(enemyStrikes|enemyLanded|takenPerMin|kos|playerUnable|sideShareJ|takenSideShare|blowUnmoved|flatFrac)$/;
  return {
    mean: S.res.mean, worst: S.res.worst, swing: swing(), pops: S.pops,
    starve: Object.fromEntries(Object.entries(S.by).map(([k, s]) => [k, +(s.starve / s.n).toFixed(2)])),
    runs: S.res.runs.map(r => ({ kind: r.kind, total: r.total, bad: Object.fromEntries(Object.entries(r.parts ?? {}).filter(([, v]) => v > 0)), s: Object.fromEntries(Object.entries(r.sum).filter(([k]) => pick.test(k))) })),
    errs: S.errs,
  };
}
/** Waits up to `secs` for the result. */
export async function wait(secs = 40) { for (let i = 0; i < secs && !done(); i++) await new Promise(r => setTimeout(r, 1000)); return result() ?? { clock: +window.brawl.clock.toFixed(0) }; }
