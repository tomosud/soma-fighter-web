// Tempo (緩急) of Mixamo fighting clips: how the speed of a strike, a dodge, a hit reaction, a fall, a getup or a
// step is laid out in time, measured straight from the FBX files (X Bot, 30 fps) and summarised per motion family
// as a compact "tempo model" the game can shape SOMA's generated frames with (src/tempo.js).
//
//   node tools/mixamo_tempo.mjs [tools/mixamo_src] [--out assets/tempo_profiles.json] [--events reports/tempo_events.json]
//
// Per clip: Mixamo bone world positions (scaled to the SOMA rig's hips height, metres) at 30 fps; per-frame speed
// of each candidate end effector (hands, elbows, feet, knees, head) and a whole-body kinetic proxy (sum of joint
// speeds). Strikes are cut into rest / anticipation / acceleration / contact / hold / recovery around each speed
// peak of the fastest effector; reactions, dodges, falls, getups and steps around the peak of the body proxy.
import fs from 'fs';
import path from 'path';
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { Rig } from '../src/soma.js';

const { Vector3 } = THREE;
const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const root = path.resolve(here, '..');
const rig = new Rig(JSON.parse(fs.readFileSync(path.join(root, 'assets/soma_rig.json'), 'utf8')));
const FPS = 30, DT = 1 / FPS;

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const srcDir = args.find(a => !a.startsWith('--') && args[args.indexOf(a) - 1] !== '--out' && args[args.indexOf(a) - 1] !== '--events') ?? path.join(root, 'tools/mixamo_src');
const outFile = opt('--out', path.join(root, 'assets/tempo_profiles.json'));
const eventsFile = opt('--events', path.join(root, 'reports/tempo_events.json'));

// ---------------------------------------------------------------- classification (from the file name)
const RECEIVE = /receiv|getting_hit|getting_rocked|getting_shoved|getting_knocked|knocked|hit_reaction|reaction_hit|_react|reaction_to|stumble|shove|clipped|victim|surprised|dying|death|die_|falling|slip_and_fall|tripping|hitting_steering|cpr|catch_on/;
/** { cat, fam } of a clip, or null when it is not one of the motions studied. */
export function classify(file) {
  const n = file.toLowerCase().replace(/\.fbx$/, '');
  const base = n.replace(/^(mf|mm|mx)_/, '');
  const pre = n.slice(0, 3);
  if (pre === 'mf_') {
    const cat = base.split('_')[0], body = base.slice(cat.length + 1);
    if (cat === 'punch' || cat === 'kick') {
      if (RECEIVE.test(body)) return hitClass(body);
      if (/restrain|grab_attacker|hand_stand|stretching|speedbag|chase_down|hadouken/.test(body)) return null;
      if (/sweep|low_kick|to_the_knee/.test(body)) return { cat: 'strike', fam: 'kick_low' };
      if (/elbow/.test(body)) return { cat: 'strike', fam: 'elbow' };
      if (/knee/.test(body)) return { cat: 'strike', fam: 'knee' };
      if (/uppercut/.test(body)) return { cat: 'strike', fam: 'uppercut' };
      if (/hook/.test(body)) return { cat: 'strike', fam: 'hook' };
      if (cat === 'kick') {
        if (/snap|front|thrust|lead_foot|kicking_with/.test(body)) return { cat: 'strike', fam: 'kick_front' };
        return { cat: 'strike', fam: 'kick_round' };
      }
      return { cat: 'strike', fam: 'straight' };
    }
    if (cat === 'block') {
      if (/transition|restrain/.test(body)) return null;
      if (/dodge/.test(body)) return { cat: 'dodge', fam: 'dodge' };
      return { cat: 'block', fam: 'block' };
    }
    if (cat === 'hit') {
      if (/cpr|steering|surprised|catch_on/.test(body)) return null;
      return hitClass(body);
    }
    if (cat === 'fall') {
      if (/in_the_air_idle|land_from|landing_from/.test(body)) return null;
      return { cat: 'fall', fam: 'fall' };
    }
    if (cat === 'getup') return /sliding|looking_around/.test(body) ? null : { cat: 'getup', fam: 'getup' };
    if (cat === 'stance') {
      if (/step|pivot|advancing|side_step|rotate_from/.test(body)) return { cat: 'step', fam: 'step' };
      if (/idle|stance|fight|clown|distance/.test(body) && !/transition|into|ready_pose/.test(body)) return { cat: 'idle', fam: 'idle' };
      return null;
    }
    return null;
  }
  if (pre === 'mm_' && /jumping_backwards_dodge/.test(base)) return { cat: 'dodge', fam: 'dodge' };
  return null; // mm_/mx_ duplicates of mf_ clips, locomotion, gestures: not studied here
}
function hitClass(body) {
  if (/knocked|dying|death|falling|slip|tripping|knee|unconscious|falls_to/.test(body)) return { cat: 'fall', fam: 'fall' };
  if (/small|light/.test(body)) return { cat: 'hit', fam: 'hit_small' };
  if (/large|big|rocked|shove/.test(body)) return { cat: 'hit', fam: 'hit_large' };
  return { cat: 'hit', fam: 'hit_medium' };
}

// ---------------------------------------------------------------- loading
const BONES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'];
/** Bone world positions per frame (metres, SOMA scale), 30 fps. */
export function loadClip(file) {
  const buf = fs.readFileSync(file);
  const g = new FBXLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '');
  const clip = g.animations[0];
  g.updateMatrixWorld(true);
  const bone = Object.fromEntries(BONES.map(b => [b, g.getObjectByName(`mixamorig${b}`)]));
  const scale = rig.restPos[0].y / bone.Hips.getWorldPosition(new Vector3()).y;
  const mixer = new THREE.AnimationMixer(g);
  mixer.clipAction(clip).play();
  const n = Math.max(2, Math.round(clip.duration * FPS));
  const pos = Object.fromEntries(BONES.map(b => [b, []]));
  for (let f = 0; f <= n; f++) {
    mixer.setTime(Math.min(clip.duration, f * DT));
    g.updateMatrixWorld(true);
    for (const b of BONES) pos[b].push(bone[b].getWorldPosition(new Vector3()).multiplyScalar(scale));
  }
  // a looping clip's last sample jumps back to its first: drop it
  if (pos.Hips.length > 2 && pos.Hips.at(-1).distanceTo(pos.Hips.at(-2)) > 0.2) for (const b of BONES) pos[b].pop();
  return { pos, frames: pos.Hips.length, duration: clip.duration };
}

// ---------------------------------------------------------------- signals
const speedOf = P => P.map((q, i) => (i === 0 ? 0 : q.distanceTo(P[i - 1]) * FPS));
const velOf = P => P.map((q, i) => (i === 0 ? new Vector3() : q.clone().sub(P[i - 1]).multiplyScalar(FPS)));
const smooth = (a, w = 1) => a.map((_, i) => { let s = 0, c = 0; for (let k = -w; k <= w; k++) { const j = i + k; if (j >= 0 && j < a.length) { s += a[j]; c++; } } return s / c; });
const yawOf = (L, R) => L.map((l, i) => Math.atan2(l.x - R[i].x, l.z - R[i].z));
const unwrap = a => { const o = [a[0]]; for (let i = 1; i < a.length; i++) { let d = a[i] - a[i - 1]; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; o.push(o[i - 1] + d); } return o; };
const rateOf = a => a.map((v, i) => (i ? (v - a[i - 1]) * FPS : 0));
const argmax = (a, lo = 0, hi = a.length) => { let b = lo; for (let i = lo; i < hi; i++) if (a[i] > a[b]) b = i; return b; };
const argmin = (a, lo = 0, hi = a.length) => { let b = lo; for (let i = lo; i < hi; i++) if (a[i] < a[b]) b = i; return b; };
const pct = (arr, p) => { const a = [...arr].filter(Number.isFinite).sort((x, y) => x - y); if (!a.length) return NaN; const k = (a.length - 1) * p, i = Math.floor(k); return a[i] + (a[Math.min(a.length - 1, i + 1)] - a[i]) * (k - i); };
const r3 = v => (Number.isFinite(v) ? +v.toFixed(3) : null);

/** Whole-body kinetic proxy: sum of the speeds of the main joints (m/s), the hips' own travel subtracted (pose change, not locomotion). */
function bodyEnergy(pos) {
  const K = new Array(pos.Hips.length).fill(0);
  const joints = ['Head', 'Spine2', 'LeftHand', 'RightHand', 'LeftForeArm', 'RightForeArm', 'LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'];
  for (const j of joints) {
    const rel = pos[j].map((q, i) => q.clone().sub(pos.Hips[i]));
    const s = speedOf(rel);
    for (let i = 0; i < K.length; i++) K[i] += s[i];
  }
  const hs = speedOf(pos.Hips);
  for (let i = 0; i < K.length; i++) K[i] += hs[i];
  return K;
}

/** Exponential decay fit of a (positive) signal from index a to b: v(t) = v(a) e^(-t/tau); returns tau (s) and the fit's r². */
function decayFit(v, a, b) {
  const xs = [], ys = [];
  for (let i = a; i <= b; i++) if (v[i] > 1e-4) { xs.push((i - a) * DT); ys.push(Math.log(v[i])); }
  if (xs.length < 3) return { tau: NaN, r2: NaN };
  const mx = xs.reduce((s, x) => s + x, 0) / xs.length, my = ys.reduce((s, y) => s + y, 0) / ys.length;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  const slope = sxy / sxx;
  return { tau: slope < 0 ? -1 / slope : Infinity, r2: syy > 0 ? (sxy * sxy) / (sxx * syy) : 0 };
}

/** Resample a[lo..hi] (inclusive) to n points, linear. */
function resample(a, lo, hi, n) {
  const out = [];
  for (let k = 0; k < n; k++) {
    const t = lo + (hi - lo) * (n === 1 ? 0 : k / (n - 1));
    const i = Math.floor(t), f = t - i;
    out.push(i >= hi ? a[hi] : a[i] * (1 - f) + a[i + 1] * f);
  }
  return out;
}

// ---------------------------------------------------------------- strikes
const HANDS = ['LeftHand', 'RightHand', 'LeftForeArm', 'RightForeArm'];
const FEET = ['LeftFoot', 'RightFoot', 'LeftLeg', 'RightLeg'];
const sideOf = j => (j.startsWith('Left') ? 'Left' : 'Right');
const CURVE_N = 24;
// where the phases sit on the normalized curve (anticipation / acceleration / hold / recovery); fixed so the peak
// of every event lands on the same sample and the per-family curves can be averaged
const PHASE_AT = [0, 0.30, 0.50, 0.58, 1.0];

/** Strike events in a clip: each speed peak of the fastest limb, cut into phases. */
function strikeEvents(clip, fam, name) {
  const { pos, frames: n } = clip;
  const cands = /^kick|knee/.test(fam) ? FEET : HANDS;
  const sp = Object.fromEntries(cands.map(j => [j, smooth(speedOf(pos[j]), 1)]));
  const spRaw = Object.fromEntries(cands.map(j => [j, speedOf(pos[j])]));
  const vel = Object.fromEntries(cands.map(j => [j, velOf(pos[j])]));
  const K = smooth(bodyEnergy(pos), 1);
  const Krest = Math.max(pct(K, 0.15), 0.4);
  const S = sp[cands[0]].map((_, i) => Math.max(...cands.map(j => sp[j][i])));
  const gmax = Math.max(...S);
  // peaks: local maxima above 45% of the clip's fastest, at least 10 frames apart (the faster one kept)
  let peaks = [];
  for (let i = 2; i < n - 2; i++) if (S[i] >= S[i - 1] && S[i] >= S[i + 1] && S[i] > S[i - 2] && S[i] > S[i + 2] && S[i] > 0.45 * gmax && S[i] > 2.0) peaks.push(i);
  peaks = peaks.filter((p, k) => !peaks.some((q, k2) => k2 !== k && Math.abs(q - p) < 10 && (S[q] > S[p] || (S[q] === S[p] && k2 < k))));
  const events = [];
  const shYaw = unwrap(yawOf(pos.LeftArm, pos.RightArm)), hipYaw = unwrap(yawOf(pos.LeftUpLeg, pos.RightUpLeg));
  const shRate = smooth(rateOf(shYaw).map(Math.abs), 1), hipRate = smooth(rateOf(hipYaw).map(Math.abs), 1);
  const headSp = smooth(speedOf(pos.Head), 1), hipsSp = smooth(speedOf(pos.Hips), 1);
  const chestSp = smooth(speedOf(pos.Spine2), 1);
  for (let k = 0; k < peaks.length; k++) {
    const p = peaks[k];
    const eff = cands.reduce((b, j) => (sp[j][p] > sp[b][p] ? j : b), cands[0]);
    // the hand (or the foot) itself, when the elbow/knee led only marginally: the tip is what lands
    const tip = eff.replace('ForeArm', 'Hand').replace(/Leg$/, 'Foot');
    const e = sp[tip][p] > 0.8 * sp[eff][p] ? tip : eff;
    const v = vel[e], s = sp[e], peak = s[p];
    const lo = k > 0 ? peaks[k - 1] + 1 : 0, hi = k + 1 < peaks.length ? peaks[k + 1] - 1 : n - 1;
    // contact/apex: the limb stops going the way it was (or nearly stops)
    let apex = p;
    while (apex + 1 <= hi && v[apex + 1].dot(v[p]) > 0 && s[apex + 1] > 0.25 * peak) apex++;
    const dir = pos[e][apex].clone().sub(pos[e][Math.max(lo, p - 6)]);
    if (dir.length() < 0.05) continue;
    dir.normalize();
    // onset of the acceleration: back from the peak while the limb is going that way fast enough
    let onset = p;
    while (onset - 1 >= lo && s[onset - 1] > 0.12 * peak && v[onset - 1].dot(dir) > 0 && s[onset - 1] <= s[onset] + 0.3) onset--;
    // anticipation: before the onset, while the body or the limb is still doing something (wind-up), up to 1 s
    let ant = onset;
    while (ant - 1 >= lo && ant - 1 > onset - FPS && (s[ant - 1] > Math.max(0.08 * peak, 0.35) || K[ant - 1] > 1.3 * Krest)) ant--;
    // hold at full extension: the limb nearly still, still out there
    const extOf = i => pos[e][i].distanceTo(pos.Spine2[i]);
    const extApex = extOf(apex);
    let holdEnd = apex;
    while (holdEnd + 1 <= hi && s[holdEnd + 1] < 0.3 * peak && extOf(holdEnd + 1) > 0.85 * extApex) holdEnd++;
    // recovery: until the limb is quiet again for three frames and the hips have stopped (or the next strike begins)
    let rec = holdEnd;
    const recCap = Math.min(hi, holdEnd + Math.round(1.0 * FPS));
    const quiet = i => s[i] <= Math.max(0.12 * peak, 0.4) && hipsSp[i] < 0.5;
    while (rec + 1 <= recCap && !(quiet(rec + 1) && quiet(Math.min(recCap, rec + 2)) && quiet(Math.min(recCap, rec + 3)))) rec++;
    if (rec - holdEnd < 2) rec = Math.min(recCap, holdEnd + 2);
    const antCut = ant === lo && lo === 0 && onset <= 2; // the clip starts inside the strike: no anticipation to measure
    const recCut = rec >= recCap && recCap === hi; // runs into the clip's end or the next strike
    const recPeakI = argmax(s, holdEnd, rec + 1);
    const fit = decayFit(s, recPeakI, rec);
    // acceleration phase shape: peak acceleration and how early the limb is at half speed
    const acc = []; for (let i = onset; i <= p; i++) acc.push(i ? (s[i] - s[i - 1]) * FPS : 0);
    const accMax = acc.length ? Math.max(...acc) : NaN;
    let half = onset; while (half < p && s[half] < 0.5 * peak) half++;
    const accDur = (p - onset) * DT;
    // wind-up: how far the limb goes against the strike direction before the onset, and how far the hips travel
    let wind = 0; for (let i = ant; i < onset; i++) wind = Math.max(wind, -pos[e][i].clone().sub(pos[e][onset]).dot(dir) + 0);
    const windBack = Math.max(0, ...Array.from({ length: onset - ant + 1 }, (_, q) => pos[e][onset].clone().sub(pos[e][ant + q]).dot(dir)));
    const stepIn = pos.Hips[apex].clone().sub(pos.Hips[ant]).setY(0);
    const fwd = dir.clone().setY(0).normalize();
    const hipsFwd = stepIn.dot(fwd);
    // which foot moves (a step in), how far along the strike
    const footStep = Math.max(...['LeftFoot', 'RightFoot'].map(f => pos[f][apex].clone().sub(pos[f][ant]).setY(0).dot(fwd)));
    // trunk and head timing against the limb's peak (ms; negative = before the limb)
    const win = [Math.max(lo, p - 12), Math.min(hi, p + 12) + 1];
    const shLead = (argmax(shRate, ...win) - p) * DT * 1000;
    const hipLead = (argmax(hipRate, ...win) - p) * DT * 1000;
    const headLag = (argmax(headSp, ...win) - p) * DT * 1000;
    const hipsPk = (argmax(hipsSp, ...win) - p) * DT * 1000;
    const hipsDrop = pos.Hips[ant].y - Math.min(...Array.from({ length: onset - ant + 1 }, (_, q) => pos.Hips[ant + q].y));
    const hipsAtApex = pos.Hips[apex].y - pos.Hips[ant].y;
    // weight shift before a kick: the hips moving over the standing foot (the one that is not the effector's)
    const supportFoot = /Foot|Leg/.test(e) ? (sideOf(e) === 'Left' ? 'RightFoot' : 'LeftFoot') : null;
    let weightShift = null;
    if (supportFoot) {
      const toward = pos[supportFoot][ant].clone().sub(pos.Hips[ant]).setY(0).normalize();
      weightShift = pos.Hips[onset].clone().sub(pos.Hips[ant]).setY(0).dot(toward);
    }
    // overshoot / settle: the limb's extension after the recovery against where it rests before (negative = pulled in past the guard)
    const extRest = extOf(ant), extEnd = extOf(rec);
    let extMinRec = Infinity; for (let i = holdEnd; i <= rec; i++) extMinRec = Math.min(extMinRec, extOf(i));
    const retPeak = s[recPeakI];
    // normalized speed curve: each phase resampled onto its fixed span of the curve
    const curve = phaseCurve(s, [ant, onset, apex, holdEnd, rec], peak);
    const Kcurve = phaseCurve(K, [ant, onset, apex, holdEnd, rec], Math.max(...K.slice(ant, rec + 1)));
    const mean = s.slice(ant, rec + 1).reduce((a, b) => a + b, 0) / (rec - ant + 1);
    events.push({
      clip: name, fam, effector: e, frames: { ant, onset, peak: p, apex, holdEnd, rec },
      dur: { ant: antCut ? NaN : (onset - ant) * DT, accel: (apex - onset) * DT, hold: (holdEnd - apex) * DT, recover: recCut ? NaN : (rec - holdEnd) * DT, total: (rec - ant) * DT },
      antCut, recCut,
      peak, mean, peakToMean: peak / mean, timeToPeakFrac: (p - ant) / Math.max(1, rec - ant), peakToApex: (apex - p) * DT, onsetToPeak: (p - onset) * DT,
      headPeak: Math.max(...headSp.slice(ant, rec + 1)), hipsPeak: Math.max(...hipsSp.slice(ant, rec + 1)),
      accMax, halfAt: (half - onset) / Math.max(1, p - onset), accDur,
      windBack, hipsFwd, footStep, shLead, hipLead, headLag, hipsPk, hipsDrop, hipsAtApex, weightShift,
      extApex, extRest, extEnd, extMinRec, retRatio: retPeak / peak, recTau: fit.tau, recR2: fit.r2,
      bodyPeak: Math.max(...K.slice(ant, rec + 1)), bodyRest: Krest,
      curve, Kcurve,
    });
  }
  return events;
}

/** The signal's phases ([a0,a1,a2,a3,a4] frame bounds) resampled onto the fixed spans of PHASE_AT, divided by `norm`. */
function phaseCurve(sig, bounds, norm) {
  const out = new Array(CURVE_N).fill(0);
  for (let k = 0; k < CURVE_N; k++) {
    const t = k / (CURVE_N - 1);
    let ph = 0; while (ph < 3 && t > PHASE_AT[ph + 1]) ph++;
    const u = (t - PHASE_AT[ph]) / (PHASE_AT[ph + 1] - PHASE_AT[ph]);
    const f = bounds[ph] + (bounds[ph + 1] - bounds[ph]) * u;
    const i = Math.floor(f), fr = f - i;
    const v = i + 1 < sig.length ? sig[i] * (1 - fr) + sig[i + 1] * fr : sig[sig.length - 1];
    out[k] = norm > 0 ? v / norm : 0;
  }
  return out;
}

// ---------------------------------------------------------------- reactions, dodges, blocks, falls, getups, steps
/** One burst event on the body proxy (or a given signal): onset, peak, settle, decay. */
function burstEvent(clip, fam, name, cat) {
  const { pos, frames: n } = clip;
  const K = smooth(bodyEnergy(pos), 1);
  const headSp = smooth(speedOf(pos.Head), 1), hipsSp = smooth(speedOf(pos.Hips), 1), chestSp = smooth(speedOf(pos.Spine2), 1);
  const handSp = smooth(pos.LeftHand.map((_, i) => Math.max(speedOf(pos.LeftHand)[i], speedOf(pos.RightHand)[i])), 1);
  const lead = cat === 'block' ? handSp : cat === 'step' ? hipsSp : cat === 'dodge' ? headSp : K;
  const Krest = Math.max(pct(K, 0.15), 0.4);
  const p = argmax(lead, 1, n);
  const peak = lead[p];
  if (!(peak > 0)) return null;
  const restLvl = Math.max(0.12 * peak, pct(lead, 0.15));
  let onset = p; while (onset - 1 >= 0 && lead[onset - 1] > restLvl && lead[onset - 1] <= lead[onset] + 0.2) onset--;
  let ant = onset; while (ant - 1 >= 0 && ant - 1 > onset - FPS && (lead[ant - 1] > restLvl || K[ant - 1] > 1.3 * Krest)) ant--;
  // the extreme (full displacement) of the leading part: where its velocity turns
  const lp = cat === 'block' ? (speedOf(pos.LeftHand)[p] > speedOf(pos.RightHand)[p] ? 'LeftHand' : 'RightHand') : cat === 'step' ? 'Hips' : cat === 'dodge' ? 'Head' : 'Head';
  const v = velOf(pos[lp]);
  let apex = p;
  if (cat === 'hit' || cat === 'dodge' || cat === 'block') while (apex + 1 < n && v[apex + 1].dot(v[p]) > 0 && lead[apex + 1] > 0.2 * peak) apex++;
  let holdEnd = apex; while (holdEnd + 1 < n && lead[holdEnd + 1] < 0.25 * peak && K[holdEnd + 1] < 0.5 * Math.max(...K.slice(ant, apex + 1))) holdEnd++;
  let settle = holdEnd;
  const cap = Math.min(n - 1, holdEnd + Math.round(2.0 * FPS));
  while (settle + 1 <= cap && (lead[settle + 1] > restLvl || K[settle + 1] > 1.4 * Krest)) settle++;
  const fit = decayFit(K, argmax(K, ant, settle + 1), settle);
  const dir = pos[lp][apex].clone().sub(pos[lp][onset]);
  const disp = dir.length();
  dir.normalize();
  // rebound: does the part come back past where it started (along the blow), and by how much of the displacement
  let rebound = 0; for (let i = apex; i <= settle; i++) rebound = Math.max(rebound, -pos[lp][i].clone().sub(pos[lp][onset]).dot(dir));
  // for reactions: head vs chest timing (head whipped later = positive ms), the hips' push
  const win = [Math.max(0, p - 10), Math.min(n - 1, p + 10) + 1];
  const headVsChest = (argmax(headSp, ...win) - argmax(chestSp, ...win)) * DT * 1000;
  const hipsPush = pos.Hips[apex].clone().sub(pos.Hips[onset]).setY(0).length();
  const footMove = Math.max(...['LeftFoot', 'RightFoot'].map(f => pos[f][settle].clone().sub(pos[f][onset]).setY(0).length()));
  const hipsY = pos.Hips.map(q => q.y);
  const hipsMin = Math.min(...hipsY), hipsStart = hipsY[0], hipsEnd = hipsY[n - 1];
  // falls: when the hips reach the floor; getups: when they are back up
  let floorAt = null, upAt = null;
  if (cat === 'fall') { for (let i = 0; i < n; i++) if (hipsY[i] < 0.35) { floorAt = i * DT; break; } }
  if (cat === 'getup') { const top = Math.max(...hipsY); for (let i = n - 1; i > 0; i--) if (hipsY[i - 1] < 0.9 * top) { upAt = i * DT; break; } }
  const curve = phaseCurve(lead, [ant, onset, apex, holdEnd, settle], peak);
  const Kcurve = phaseCurve(K, [ant, onset, apex, holdEnd, settle], Math.max(...K.slice(ant, settle + 1)));
  const mean = lead.slice(ant, settle + 1).reduce((a, b) => a + b, 0) / (settle - ant + 1);
  const acc = []; for (let i = onset; i <= p; i++) acc.push(i ? (lead[i] - lead[i - 1]) * FPS : 0);
  const K2 = (() => { const t = K.map((x, i) => x); const seg = []; for (let i = ant; i <= settle; i++) seg.push(t[i]); return seg; })();
  // settle time: from the peak until K is under 1.4 × rest for good
  const settleT = (settle - p) * DT;
  return {
    clip: name, fam, cat, lead: lp, frames: { ant, onset, peak: p, apex, holdEnd, settle },
    dur: { ant: (onset - ant) * DT, accel: (apex - onset) * DT, hold: (holdEnd - apex) * DT, recover: (settle - holdEnd) * DT, total: (settle - ant) * DT, clip: (n - 1) * DT },
    peak, mean, peakToMean: peak / mean, timeToPeakFrac: (p - ant) / Math.max(1, settle - ant), accMax: acc.length ? Math.max(...acc) : NaN, onsetToPeak: (p - onset) * DT,
    disp, rebound: disp > 0 ? rebound / disp : 0, headVsChest, hipsPush, footMove, settleT, decayTau: fit.tau, decayR2: fit.r2,
    headPeak: Math.max(...headSp.slice(ant, settle + 1)), chestPeak: Math.max(...chestSp.slice(ant, settle + 1)), hipsPeak: Math.max(...hipsSp.slice(ant, settle + 1)),
    headPeakMs: (argmax(headSp, ant, settle + 1) - p) * DT * 1000, hipsPeakMs: (argmax(hipsSp, ant, settle + 1) - p) * DT * 1000,
    antCut: ant === 0 && onset <= 2, recCut: settle >= cap,
    bodyPeak: Math.max(...K2), bodyRest: Krest, hipsMin, hipsDrop: hipsStart - hipsMin, hipsEnd: hipsEnd - hipsStart, floorAt, upAt,
    curve, Kcurve,
  };
}

// ---------------------------------------------------------------- summary
const stat = (rows, key, f = r => r[key]) => {
  const v = rows.map(f).filter(x => Number.isFinite(x));
  if (!v.length) return null;
  return { med: r3(pct(v, 0.5)), q1: r3(pct(v, 0.25)), q3: r3(pct(v, 0.75)), n: v.length };
};
const medCurve = (rows, key) => Array.from({ length: CURVE_N }, (_, k) => r3(pct(rows.map(r => r[key][k]), 0.5)));

function summarise(events) {
  const byFam = {};
  for (const e of events) (byFam[e.fam] ??= []).push(e);
  const out = {};
  for (const [fam, rows] of Object.entries(byFam)) {
    const strike = rows[0].cat === undefined;
    const base = {
      n: rows.length, clips: new Set(rows.map(r => r.clip)).size,
      phaseAt: PHASE_AT,
      dur: { ant: stat(rows, 'dur', r => r.dur.ant), accel: stat(rows, 'dur', r => r.dur.accel), hold: stat(rows, 'dur', r => r.dur.hold), recover: stat(rows, 'dur', r => r.dur.recover), total: stat(rows, 'dur', r => r.dur.total) },
      peakSpeed: stat(rows, 'peak'), peakToMean: stat(rows, 'peakToMean'), timeToPeakFrac: stat(rows, 'timeToPeakFrac'), accMax: stat(rows, 'accMax'),
      bodyPeak: stat(rows, 'bodyPeak'), bodyRest: stat(rows, 'bodyRest'),
      curve: medCurve(rows, 'curve'), bodyCurve: medCurve(rows, 'Kcurve'),
    };
    if (strike) Object.assign(base, {
      kind: 'strike',
      accDur: stat(rows, 'accDur'), halfAt: stat(rows, 'halfAt'), peakToApex: stat(rows, 'peakToApex'), onsetToPeak: stat(rows, 'onsetToPeak'),
      headPeak: stat(rows, 'headPeak'), hipsPeak: stat(rows, 'hipsPeak'),
      recTau: stat(rows, 'recTau', r => (r.recR2 > 0.5 ? r.recTau : NaN)), retRatio: stat(rows, 'retRatio'),
      windBack: stat(rows, 'windBack'), hipsFwd: stat(rows, 'hipsFwd'), footStep: stat(rows, 'footStep'),
      shLeadMs: stat(rows, 'shLead'), hipLeadMs: stat(rows, 'hipLead'), headLagMs: stat(rows, 'headLag'), hipsPeakMs: stat(rows, 'hipsPk'),
      hipsDrop: stat(rows, 'hipsDrop'), hipsAtApex: stat(rows, 'hipsAtApex'), weightShift: stat(rows, 'weightShift'),
      extApex: stat(rows, 'extApex'), extOvershoot: stat(rows, 'ext', r => r.extRest - r.extMinRec), extEndVsRest: stat(rows, 'ext', r => r.extEnd - r.extRest),
      effectors: Object.fromEntries(Object.entries(rows.reduce((m, r) => { m[r.effector] = (m[r.effector] ?? 0) + 1; return m; }, {}))),
    });
    else Object.assign(base, {
      kind: rows[0].cat,
      onsetToPeak: stat(rows, 'onsetToPeak'), settleT: stat(rows, 'settleT'), decayTau: stat(rows, 'decayTau', r => (r.decayR2 > 0.4 && Number.isFinite(r.decayTau) ? r.decayTau : NaN)),
      disp: stat(rows, 'disp'), rebound: stat(rows, 'rebound'), headVsChestMs: stat(rows, 'headVsChest'), hipsPush: stat(rows, 'hipsPush'), footMove: stat(rows, 'footMove'),
      headPeak: stat(rows, 'headPeak'), chestPeak: stat(rows, 'chestPeak'), hipsPeak: stat(rows, 'hipsPeak'), headPeakMs: stat(rows, 'headPeakMs'), hipsPeakMs: stat(rows, 'hipsPeakMs'),
      hipsDrop: stat(rows, 'hipsDrop'), hipsEnd: stat(rows, 'hipsEnd'), floorAt: stat(rows, 'floorAt'), upAt: stat(rows, 'upAt'), clipDur: stat(rows, 'dur', r => r.dur.clip),
    });
    out[fam] = base;
  }
  return out;
}

// ---------------------------------------------------------------- main
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'))) {
  const files = fs.readdirSync(srcDir).filter(f => /\.fbx$/i.test(f)).sort();
  const events = [], counts = {}, skipped = [];
  const t0 = Date.now();
  for (const f of files) {
    const c = classify(f);
    if (!c) { skipped.push(f); continue; }
    const name = f.replace(/\.fbx$/i, '').replace(/_[0-9a-f]{6}$/, '');
    let clip;
    try { clip = loadClip(path.join(srcDir, f)); } catch (e) { console.error(`skip ${f}: ${e.message}`); continue; }
    counts[c.cat] = (counts[c.cat] ?? 0) + 1;
    if (c.cat === 'strike') {
      const ev = strikeEvents(clip, c.fam, name);
      events.push(...ev);
      console.log(`${c.fam.padEnd(11)} ${name.slice(0, 48).padEnd(48)} ${String(clip.frames).padStart(4)}f  ${ev.length} strike(s)  ${ev.map(e => `${e.effector.replace(/Left|Right/, m => m[0])}@${e.frames.peak} ${e.peak.toFixed(1)}m/s ant${e.dur.ant.toFixed(2)} acc${e.dur.accel.toFixed(2)} hold${e.dur.hold.toFixed(2)} rec${e.dur.recover.toFixed(2)}`).join(' | ')}`);
    } else if (c.cat === 'idle') {
      const K = bodyEnergy(clip.pos);
      events.push({ clip: name, fam: 'idle', cat: 'idle', peak: Math.max(...K), mean: K.reduce((a, b) => a + b, 0) / K.length, bodyRest: pct(K, 0.15), bodyPeak: pct(K, 0.95), peakToMean: Math.max(...K) / (K.reduce((a, b) => a + b, 0) / K.length), dur: { total: (clip.frames - 1) * DT, clip: (clip.frames - 1) * DT, ant: 0, accel: 0, hold: 0, recover: 0 }, timeToPeakFrac: 0, accMax: 0, curve: new Array(CURVE_N).fill(1), Kcurve: new Array(CURVE_N).fill(1), frames: {} });
    } else {
      const ev = burstEvent(clip, c.fam, name, c.cat);
      if (ev) { events.push(ev); console.log(`${c.fam.padEnd(11)} ${name.slice(0, 48).padEnd(48)} ${String(clip.frames).padStart(4)}f  ${ev.lead}@${ev.frames.peak} ${ev.peak.toFixed(1)}m/s onset→peak ${ev.onsetToPeak.toFixed(2)} settle ${ev.settleT.toFixed(2)} tau ${Number.isFinite(ev.decayTau) ? ev.decayTau.toFixed(2) : '-'}`); }
    }
  }
  const summary = summarise(events);
  const profiles = {
    about: 'Tempo profiles of Mixamo fighting clips (X Bot, 30 fps), by tools/mixamo_tempo.mjs: per motion family the median (q1/q3) phase durations in seconds, peak speeds (m/s, SOMA scale), and a normalized speed-vs-normalized-time curve (24 samples; the phases anticipation/acceleration/hold/recovery sit on the fixed spans `phaseAt` so the speed peak of every event lines up). `curve` is the end effector speed over its peak (strikes: the striking hand/foot; dodges: head; blocks: hand; steps: hips; reactions/falls/getups: the whole-body proxy); `bodyCurve` is the whole-body proxy (sum of joint speeds relative to the hips, plus the hips, m/s) over its own peak. Timing columns in ms are relative to the effector\'s speed peak (negative = before it).',
    fps: FPS, curveSamples: CURVE_N, phaseAt: PHASE_AT, source: { clips: Object.values(counts).reduce((a, b) => a + b, 0), byCategory: counts, events: events.length },
    families: summary,
  };
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, JSON.stringify(profiles, null, 1));
  fs.mkdirSync(path.dirname(eventsFile), { recursive: true });
  fs.writeFileSync(eventsFile, JSON.stringify(events.map(e => ({ ...e, curve: e.curve.map(r3), Kcurve: e.Kcurve.map(r3) })), (k, v) => (typeof v === 'number' ? r3(v) : v)));
  console.log(`\n${files.length} files, ${Object.values(counts).reduce((a, b) => a + b, 0)} studied (${JSON.stringify(counts)}), ${events.length} events, ${skipped.length} not studied; ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  console.log(`-> ${outFile}\n-> ${eventsFile}`);
  for (const [fam, s] of Object.entries(summary)) {
    const d = s.dur;
    console.log(`${fam.padEnd(11)} n=${String(s.n).padStart(3)} ant ${d.ant?.med} acc ${d.accel?.med} hold ${d.hold?.med} rec ${d.recover?.med} | peak ${s.peakSpeed?.med} m/s  peak/mean ${s.peakToMean?.med}  t2peak ${s.timeToPeakFrac?.med}` + (s.kind === 'strike' ? `  tau ${s.recTau?.med}  shLead ${s.shLeadMs?.med}ms headLag ${s.headLagMs?.med}ms hipsFwd ${s.hipsFwd?.med}m` : `  tau ${s.decayTau?.med} settle ${s.settleT?.med}`));
  }
}
