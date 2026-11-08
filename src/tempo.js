// Tempo (緩急) for generated motion: per-frame playback rates that make a SOMA frame sequence play with the speed
// layout of a Mixamo clip (assets/tempo_profiles.json, by tools/mixamo_tempo.mjs): a slow wind-up, the limb
// whipping in, a beat at full extension, a recovery that eases out; a hit that jolts then decays.
//
// SOMA's frames move the end effector evenly; the profile says how fast it should be going at each fraction of
// the way. The warp matches the cumulative arc length of the effector's path in the generated frames to the
// cumulative distance under the profile's speed curve, so that at playback the effector covers the right
// distance in the right time. Pure functions, no rig: the caller hands in the effector positions per frame.
//
//   import { tempoRates, holdFrames, decayRates } from './tempo.js';
//   const rates = tempoRates(frames, frames.map(f => rig.positions(f.pose)[I.RightHand]), profiles.families.straight,
//                            { apex, phases: ['ant', 'accel'], peakSpeed: 8 });
//   frames.forEach((f, i) => { f.rate = rates[i]; });

export const PHASES = ['ant', 'accel', 'hold', 'recover'];
const DEFAULT_PHASE_AT = [0, 0.3, 0.5, 0.58, 1];

const px = (P, i) => (P.x !== undefined ? P.x : P[3 * i]);
/** Position i of an effector path given as an array of {x,y,z} or a flat Float32Array xyz. */
export function posAt(P, i) {
  if (Array.isArray(P) || (P.length !== undefined && typeof P[0] === 'object')) { const q = P[i]; return [q.x, q.y, q.z]; }
  return [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
}
export const pathLength = P => (Array.isArray(P) || typeof P[0] === 'object' ? P.length : P.length / 3);

/** Cumulative arc length (metres) of an effector path over frames [start..end]. */
export function arcLength(P, start = 0, end = pathLength(P) - 1) {
  const s = new Float64Array(end - start + 1);
  let a = posAt(P, start);
  for (let i = start + 1; i <= end; i++) {
    const b = posAt(P, i);
    s[i - start] = s[i - start - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    a = b;
  }
  return s;
}

/** Per-frame speed (m/s) of an effector path (speed of the step into frame i; frame 0 repeats frame 1). */
export function speeds(P, fps = 30) {
  const n = pathLength(P), v = new Float64Array(n);
  for (let i = 1; i < n; i++) { const a = posAt(P, i - 1), b = posAt(P, i); v[i] = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) * fps; }
  if (n > 1) v[0] = v[1];
  return v;
}

/** The frame where the effector is at its speed peak (the contact is a few frames after it). */
export function findPeak(P, start = 0, end = pathLength(P) - 1, fps = 30) {
  const v = speeds(P, fps);
  let b = start + 1;
  for (let i = start + 1; i <= end; i++) if (v[i] > v[b]) b = i;
  return Math.min(end, b);
}
/** The frame where the effector stops going the way it was at the peak (the top of the strike). */
export function findApex(P, start = 0, end = pathLength(P) - 1, fps = 30) {
  const p = findPeak(P, start, end, fps);
  const d = (i, j) => { const a = posAt(P, i), b = posAt(P, j); return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; };
  const vp = d(p - 1, p);
  let a = p;
  while (a + 1 <= end) { const w = d(a, a + 1); if (w[0] * vp[0] + w[1] * vp[1] + w[2] * vp[2] <= 0) break; a++; }
  return a;
}

/**
 * The profile's speed curve laid out on real time: the normalized curve (phases on fixed spans `phaseAt`) is
 * stretched phase by phase onto the median phase durations, keeping only `phases`. Returns { t, v, D, apexT,
 * total } with t in seconds from the start of the first kept phase, v the speed over its peak, D the cumulative
 * distance (peak-speed × seconds) and apexT the time of the top (end of the acceleration phase).
 */
export function profileTimeline(profile, phases = PHASES, scale = 1) {
  const curve = profile.curve, n = curve.length;
  const at = profile.phaseAt ?? DEFAULT_PHASE_AT;
  const durs = PHASES.map(k => Math.max(0, (profile.dur?.[k]?.med ?? 0) * scale));
  const keep = PHASES.map(k => phases.includes(k));
  // a kept phase of zero median length (a punch has no hold) still gets one frame's worth so the peak is reached
  const dur = durs.map((d, k) => (keep[k] ? Math.max(d, k === 2 ? 0 : 1 / 60) : 0));
  const t = [], v = [];
  let t0 = 0;
  for (let k = 0; k < 4; k++) {
    if (!keep[k]) continue;
    const a = at[k], b = at[k + 1];
    const i0 = Math.ceil(a * (n - 1) - 1e-9), i1 = Math.floor(b * (n - 1) + 1e-9);
    for (let i = i0; i <= i1; i++) {
      const tau = i / (n - 1);
      const tt = t0 + dur[k] * ((tau - a) / (b - a));
      if (t.length && tt <= t[t.length - 1] + 1e-9) { v[v.length - 1] = Math.max(v[v.length - 1], curve[i]); continue; }
      t.push(tt); v.push(curve[i]);
    }
    t0 += dur[k];
  }
  const vmax = Math.max(...v) || 1;
  for (let i = 0; i < v.length; i++) v[i] = Math.max(0.02, v[i] / vmax); // (a floor: the effector never quite stops in the profile either)
  const D = new Float64Array(t.length);
  for (let i = 1; i < t.length; i++) D[i] = D[i - 1] + 0.5 * (v[i] + v[i - 1]) * (t[i] - t[i - 1]);
  const apexT = (keep[0] ? dur[0] : 0) + (keep[1] ? dur[1] : 0);
  return { t, v, D, apexT, total: t0, phases: phases.filter(p => PHASES.includes(p)) };
}

/** Inverse of a non-decreasing sampled function: the x where f(x) = y (linear between samples). */
function invert(xs, fs, y) {
  const n = xs.length;
  if (y <= fs[0]) return xs[0];
  if (y >= fs[n - 1]) return xs[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (fs[m] <= y) lo = m; else hi = m; }
  const df = fs[hi] - fs[lo];
  return df > 1e-12 ? xs[lo] + (xs[hi] - xs[lo]) * ((y - fs[lo]) / df) : xs[lo];
}
function interp(xs, fs, x) {
  const n = xs.length;
  if (x <= xs[0]) return fs[0];
  if (x >= xs[n - 1]) return fs[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const dx = xs[hi] - xs[lo];
  return dx > 1e-12 ? fs[lo] + (fs[hi] - fs[lo]) * ((x - xs[lo]) / dx) : fs[lo];
}

/**
 * Per-frame playback rates for `frames` (any array; only its length is used) so that the effector, whose world
 * position per frame is `P` (array of {x,y,z} or flat xyz), plays with the profile's speed layout.
 *
 * The span is cut at the apex. Before it, frames in which the effector has not yet started (its step under
 * `onsetFrac` of the largest step) are the anticipation and share the profile's anticipation time evenly (the
 * fist is still: its path says nothing; the body is winding up); from the onset to the apex the cumulative arc
 * length is matched to the profile's acceleration phase. After the apex, frames in which the effector stays
 * (step under `holdFrac`) are the hold and share the hold time; the rest is matched to the recovery. A phase
 * with no frames takes no time (a punch generated up to its top has no hold unless the caller adds one with
 * withHold()).
 *
 * opts:
 *   apex        frame index of the top of the strike (contact); default: found from the path (findApex)
 *   start, end  the span of frames to shape (default all); frames outside keep rate `outside` (default 1)
 *   phases      which profile phases the span may cover: default ['ant','accel'] when the span ends at the
 *               apex, ['hold','recover'] when it starts there, all four otherwise
 *   onset       frame where the effector starts out (default: found with onsetFrac = 0.25)
 *   duration    seconds the span should take (default: the covered phases' median durations × durationScale)
 *   durationScale  (default 1) e.g. 0.8 for a quicker fighter
 *   peakSpeed   if given, the whole warp is scaled so the fastest played frame shows the effector at this speed (m/s)
 *   uniformMix  0..1 (default 0.15): share of plain frame-count progression blended into the arc-length
 *               progression inside the acceleration and recovery phases (keeps slow frames from collapsing)
 *   minRate, maxRate  clamps (default 0.25, 4)
 *   fps         (default 30)
 * Returns a Float32Array of rates (frames.length), and sets opts.out = { t (playback seconds per frame of the
 * span), duration, apexT, realizedPeak, phases (those that got frames), onset, holdEnd, apex }.
 */
export function tempoRates(frames, P, profile, opts = {}) {
  const n = frames.length, fps = opts.fps ?? 30;
  const start = Math.max(0, opts.start ?? 0), end = Math.min(n - 1, opts.end ?? n - 1);
  const rates = new Float32Array(n).fill(opts.outside ?? 1);
  if (end - start < 1) return rates;
  const apex = Math.max(start, Math.min(end, opts.apex ?? findApex(P, start, end, fps)));
  const phases = opts.phases ?? (apex >= end - 1 ? ['ant', 'accel'] : apex <= start + 1 ? ['hold', 'recover'] : PHASES);
  const scaleD = opts.durationScale ?? 1;
  const dur = Object.fromEntries(PHASES.map(k => [k, phases.includes(k) ? Math.max(0, (profile.dur?.[k]?.med ?? 0) * scaleD) : 0]));
  const s = arcLength(P, start, end);
  const step = i => (i > start ? s[i - start] - s[i - 1 - start] : 0);
  let dmax = 0; for (let i = start + 1; i <= end; i++) dmax = Math.max(dmax, step(i));
  // onset: the first step that counts, before the apex
  let onset = start;
  if (opts.onset != null) onset = Math.max(start, Math.min(apex, opts.onset));
  else if (dur.ant > 0) while (onset < apex - 1 && step(onset + 1) < (opts.onsetFrac ?? 0.25) * dmax) onset++;
  // hold: frames after the apex in which the effector stays
  let holdEnd = apex;
  if (dur.hold > 0) while (holdEnd < end - 1 && step(holdEnd + 1) < (opts.holdFrac ?? 0.1) * dmax) holdEnd++;
  const mix = opts.uniformMix ?? 0.15;
  // the profile's timeline (all phases) and its cumulative distance across one phase as a function of that
  // phase's own time fraction, inverted: distance fraction -> time fraction
  const tl = profileTimeline(profile, PHASES, 1);
  const dur0 = p => Math.max(0, (profile.dur?.[p]?.med ?? 0));
  const tAt = k => PHASES.slice(0, k).reduce((a, p) => a + Math.max(dur0(p), p === 'hold' ? 0 : 1 / 60), 0);
  const invDfrac = (k, y) => {
    const a = tAt(k), b = tAt(k + 1);
    const Da = interp(tl.t, tl.D, a), Db = interp(tl.t, tl.D, b);
    if (!(Db > Da) || !(b > a)) return y;
    return (invert(tl.t, tl.D, Da + (Db - Da) * y) - a) / (b - a);
  };
  // playback time of each frame of the span
  const t = new Float64Array(end - start + 1);
  const used = [];
  let t0 = 0;
  // anticipation: even
  if (onset > start) { const d = dur.ant; if (d > 0) used.push('ant'); for (let i = start; i <= onset; i++) t[i - start] = t0 + d * ((i - start) / (onset - start)); t0 += d; }
  // acceleration: arc length onto the profile's cumulative distance
  if (apex > onset) {
    const d = dur.accel; if (d > 0) used.push('accel');
    const sa = s[onset - start], sb = s[apex - start], span = apex - onset;
    for (let i = onset; i <= apex; i++) {
      const lin = (i - onset) / span, arc = sb > sa + 1e-9 ? (s[i - start] - sa) / (sb - sa) : lin;
      t[i - start] = t0 + d * ((1 - mix) * invDfrac(1, arc) + mix * lin);
    }
    t0 += d;
  }
  const apexT = t0;
  // hold: even
  if (holdEnd > apex) { const d = dur.hold; if (d > 0) used.push('hold'); for (let i = apex + 1; i <= holdEnd; i++) t[i - start] = t0 + d * ((i - apex) / (holdEnd - apex)); t0 += d; }
  // recovery: arc length onto the profile's recovery distance
  if (end > holdEnd) {
    const d = dur.recover; if (d > 0) used.push('recover');
    const sa = s[holdEnd - start], sb = s[end - start], span = end - holdEnd;
    for (let i = holdEnd + 1; i <= end; i++) {
      const lin = (i - holdEnd) / span, arc = sb > sa + 1e-9 ? (s[i - start] - sa) / (sb - sa) : lin;
      t[i - start] = t0 + d * ((1 - mix) * invDfrac(3, arc) + mix * lin);
    }
    t0 += d;
  }
  // the wanted duration, or the shown peak speed, scale the whole
  let scale = 1;
  if (opts.duration && t0 > 1e-9) scale = opts.duration / t0;
  if (opts.peakSpeed) {
    let best = 0;
    for (let i = start; i < end; i++) { const dt = (t[i + 1 - start] - t[i - start]) * scale; if (dt > 1e-9) best = Math.max(best, step(i + 1) / dt); }
    if (best > 1e-9) scale *= best / opts.peakSpeed;
  }
  const minR = opts.minRate ?? 0.25, maxR = opts.maxRate ?? 4;
  let realized = 0;
  for (let i = start; i < end; i++) {
    const dt = (t[i + 1 - start] - t[i - start]) * scale;
    const r = dt > 1e-9 ? 1 / (fps * dt) : maxR;
    rates[i] = Math.min(maxR, Math.max(minR, r));
    realized = Math.max(realized, step(i + 1) * fps * rates[i]);
  }
  rates[end] = rates[end - 1];
  opts.out = { t: Array.from(t, x => x * scale), duration: t0 * scale, apexT: apexT * scale, realizedPeak: realized, phases: used, onset, holdEnd, apex };
  return rates;
}

/** How many frames (at fps) the top of a strike is held in the profile (a punch: 0; a kick or knee: ~4). */
export const holdFrames = (profile, fps = 30, scale = 1) => Math.round((profile.dur?.hold?.med ?? 0) * fps * scale);

/** `frames` with frame `apex` repeated `count` times after itself (the same object: a held pose). */
export function withHold(frames, apex, count) {
  if (count <= 0) return frames.slice();
  const out = frames.slice(0, apex + 1);
  for (let i = 0; i < count; i++) out.push(frames[apex]);
  return out.concat(frames.slice(apex + 1));
}

/**
 * Rates for a jolt that decays (a hit reaction, a stagger): frame `from` plays at `r0` and the rate falls as
 * e^(-t/tau) toward `floor`, in playback time. Mixamo: tau ≈ 0.47 s (small hit), 0.63 (medium), 0.72 (large).
 */
export function decayRates(n, { from = 0, r0 = 2.5, floor = 0.6, tau = 0.6, fps = 30 } = {}) {
  const rates = new Float32Array(n).fill(1);
  let t = 0;
  for (let i = from; i < n; i++) {
    const r = floor + (r0 - floor) * Math.exp(-t / tau);
    rates[i] = r;
    t += 1 / (fps * r);
  }
  return rates;
}

/**
 * The speed the effector will be shown at when the frames play with `rates`: samples of (playback seconds,
 * m/s) and summary numbers, for checking a warp against the profile.
 */
export function realizedProfile(P, rates, fps = 30, start = 0, end = pathLength(P) - 1) {
  const t = [0], v = [];
  for (let i = start; i < end; i++) {
    const a = posAt(P, i), b = posAt(P, i + 1);
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const dt = 1 / (fps * rates[i]);
    v.push(d / dt);
    t.push(t[t.length - 1] + dt);
  }
  const peak = Math.max(...v), mean = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
  let ip = 0; for (let i = 0; i < v.length; i++) if (v[i] > v[ip]) ip = i;
  return { t, v, peak, mean, peakToMean: peak / (mean || 1), timeToPeakFrac: t[ip] / (t[t.length - 1] || 1), duration: t[t.length - 1] };
}

/** Resample a (t, v) sampling onto m even points over [0, t_end] (linear), for comparing two profiles. */
export function resampleProfile(t, v, m = 24) {
  const out = new Float64Array(m), T = t[t.length - 1];
  for (let k = 0; k < m; k++) {
    const x = (T * k) / (m - 1);
    let i = 0; while (i + 1 < t.length - 1 && t[i + 1] <= x) i++;
    const a = t[i], b = t[Math.min(t.length - 1, i + 1)], f = b > a ? (x - a) / (b - a) : 0;
    const va = v[Math.min(v.length - 1, i)], vb = v[Math.min(v.length - 1, i + 1)];
    out[k] = va + (vb - va) * Math.min(1, Math.max(0, f));
  }
  return out;
}

/** Pearson correlation of two equal-length samplings. */
export function correlation(a, b) {
  const n = Math.min(a.length, b.length);
  let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}
