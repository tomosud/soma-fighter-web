// Self-test of src/tempo.js: an evenly moving effector path (as SOMA generates) warped by the Mixamo profiles.
//   node tools/tempo_selftest.mjs
import fs from 'fs';
import path from 'path';
import { tempoRates, profileTimeline, realizedProfile, resampleProfile, correlation, holdFrames, withHold, decayRates, findApex, arcLength } from '../src/tempo.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const profiles = JSON.parse(fs.readFileSync(path.join(root, 'assets/tempo_profiles.json'), 'utf8')).families;
const FPS = 30;
let failed = 0;
const check = (name, ok, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); if (!ok) failed++; };
const r2 = v => +v.toFixed(2);

// a straight, even path: SOMA-like frames in which the fist goes 0.6 m out in 20 frames, after 4 still frames
function evenPath(nStill, nMove, dist, back = 0) {
  const P = [];
  for (let i = 0; i < nStill; i++) P.push({ x: 0, y: 1.4, z: 0.2 });
  for (let i = 1; i <= nMove; i++) P.push({ x: 0, y: 1.4, z: 0.2 + dist * (i / nMove) });
  for (let i = 1; i <= back; i++) P.push({ x: 0, y: 1.4, z: 0.2 + dist * (1 - i / back) });
  return P;
}

// 1. pre-apex only (a strike request's frames: rest -> top), straight punch
{
  const P = evenPath(4, 20, 0.6);
  const frames = P.map(() => ({ rate: 1 }));
  const prof = profiles.straight;
  const opts = { apex: P.length - 1, phases: ['ant', 'accel'] };
  const rates = tempoRates(frames, P, prof, opts);
  const real = realizedProfile(P, rates);
  const tl = profileTimeline(prof, ['ant', 'accel']);
  const target = resampleProfile(tl.t, tl.v, 24), got = resampleProfile(real.t, real.v, 24);
  const c = correlation(target, got);
  check('straight pre-apex: rates finite & clamped', rates.every(r => Number.isFinite(r) && r >= 0.25 && r <= 4), `min ${r2(Math.min(...rates))} max ${r2(Math.max(...rates))}`);
  check('straight pre-apex: duration = ant+accel medians', Math.abs(real.duration - (prof.dur.ant.med + prof.dur.accel.med)) < 0.05, `${r2(real.duration)} s vs ${r2(prof.dur.ant.med + prof.dur.accel.med)}`);
  check('straight pre-apex: speed shape follows the profile', c > 0.85, `corr ${r2(c)}; peak/mean ${r2(real.peakToMean)} (profile ${prof.peakToMean.med}); time to peak ${r2(real.timeToPeakFrac)}`);
  const im = Array.from(rates).indexOf(Math.max(...rates.slice(4, P.length - 1)), 4);
  check('straight pre-apex: still frames take the anticipation time, the fist slows into the top', rates[1] < 1 && im >= 4 + 0.3 * 20 && rates[P.length - 2] < Math.max(...rates), `rate@still ${r2(rates[1])} fastest frame ${im} of ${P.length - 1}; peak→top decel ${r2(rates[P.length - 2])}`);
  console.log('     rates', Array.from(rates, r2).join(' '));
}

// 2. whole event (out and back): hook with its recovery, peak speed pinned at 8 m/s
{
  const P = evenPath(4, 14, 0.5, 14);
  const frames = P.map(() => ({ rate: 1 }));
  const prof = profiles.hook;
  const apex = 17;
  const opts = { apex, peakSpeed: 8, maxRate: 8 };
  const rates = tempoRates(frames, P, prof, opts);
  const real = realizedProfile(P, rates);
  const tl = profileTimeline(prof);
  const c = correlation(resampleProfile(tl.t, tl.v, 32), resampleProfile(real.t, real.v, 32));
  check('hook whole event: apex found where the path turns', findApex(P) === apex, `found ${findApex(P)}`);
  check('hook whole event: played peak ≈ 8 m/s', Math.abs(real.peak - 8) / 8 < 0.08, `${r2(real.peak)} m/s`);
  check('hook whole event: shape follows the profile', c > 0.8, `corr ${r2(c)}; peak/mean ${r2(real.peakToMean)} (profile ${prof.peakToMean.med}); duration ${r2(real.duration)} s (apex at ${r2(opts.out.apexT)} s)`);
  check('hook whole event: recovery slower than the strike', rates.slice(apex + 1, apex + 8).reduce((a, b) => a + b, 0) / 7 < rates.slice(apex - 4, apex).reduce((a, b) => a + b, 0) / 4);
  console.log('     rates', Array.from(rates, r2).join(' '));
}

// 3. a kick with its hold at the top (frames repeated), then the recovery eased out
{
  const prof = profiles.kick_round;
  const h = holdFrames(prof);
  const P0 = evenPath(3, 12, 0.9, 12);
  const frames0 = P0.map(() => ({ rate: 1 }));
  const frames = withHold(frames0, 15, h), P = withHold(P0, 15, h);
  const o3 = { apex: 15, phases: ['ant', 'accel', 'hold', 'recover'] };
  const rates = tempoRates(frames, P, prof, o3);
  const real = realizedProfile(P, rates);
  check('kick: hold frames from the profile', h >= 3 && frames.length === frames0.length + h, `${h} frames (${prof.dur.hold.med} s)`);
  check('kick: held frames play slowly, not collapsed', rates.slice(16, 16 + h).every(r => r < 1.2), `hold rates ${Array.from(rates.slice(16, 16 + h), r2).join(' ')}`);
  check('kick: total duration = the profile total (the clamp trims the still tail)', Math.abs(o3.out.duration - opts3Total(prof)) < 0.01 && Math.abs(real.duration - opts3Total(prof)) < 0.2, `${r2(o3.out.duration)} s planned, ${r2(real.duration)} s played vs ${r2(opts3Total(prof))}`);
  function opts3Total(p) { return p.dur.ant.med + p.dur.accel.med + p.dur.hold.med + p.dur.recover.med; }
}

// 4. hit reaction: head path to the impact pose then back, with the medium-hit profile; and a plain decay
{
  const prof = profiles.hit_medium;
  const P = evenPath(0, 6, 0.28, 18);
  const frames = P.map(() => ({ rate: 1 }));
  const rates = tempoRates(frames, P, prof, { apex: 6, phases: ['accel', 'hold', 'recover'] });
  const real = realizedProfile(P, rates);
  check('hit: the head reaches its full displacement in ~0.33 s (speed peak at 0.13 s)', Math.abs(real.t[6] - prof.dur.accel.med) < 0.06, `${r2(real.t[6])} s (profile onset→top ${prof.dur.accel.med} s)`);
  check('hit: settles over ~0.67 s', Math.abs(real.duration - real.t[6] - prof.dur.recover.med) < 0.08, `${r2(real.duration - real.t[6])} s`);
  const d = decayRates(24, { r0: 2.6, floor: 0.6, tau: prof.decayTau.med });
  check('decayRates: monotone, from r0 toward the floor', Math.abs(d[0] - 2.6) < 1e-6 && d.every((r, i) => i === 0 || r <= d[i - 1] + 1e-9) && d[23] > 0.6, `last ${r2(d[23])}`);
}

// 5. degenerate inputs
{
  const P = evenPath(8, 0, 0); // the effector never moves
  const rates = tempoRates(P.map(() => ({})), P, profiles.straight, { apex: 7 });
  check('still path: finite rates (uniform progression)', rates.every(r => Number.isFinite(r) && r > 0), Array.from(rates, r2).join(' '));
  const one = tempoRates([{}], [{ x: 0, y: 0, z: 0 }], profiles.straight);
  check('single frame: rate 1', one.length === 1 && one[0] === 1);
  const flat = new Float32Array([0, 1, 0, 0, 1, 0.3, 0, 1, 0.6]);
  check('flat xyz input accepted', Math.abs(arcLength(flat)[2] - 0.6) < 1e-6);
}

console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
