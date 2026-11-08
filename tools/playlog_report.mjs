#!/usr/bin/env node
// SOMA Brawl: compare play logs (eval_out/play_*_log.json, written by src/playlog.js) - a person's plays against the
// bots' (src/autoplay.js) - as one compact table of measures, one column per log (or per bot kind), with how far each
// bot column is from the person's reference column. Node only, no dependencies.
//
//   node tools/playlog_report.mjs eval_out/play_20261108004434_log.json eval_out/play_20261108003159_log.json
//   node tools/playlog_report.mjs --label eval_out/play_X_log.json=bot --ref eval_out/play_Y_log.json  files...
//   node tools/playlog_report.mjs --runs files...      (a bot log: one column per run instead of per kind)
//   node tools/playlog_report.mjs --json out.json files... (the measures as JSON as well)
//   node tools/playlog_report.mjs --only presses,cursor files...  (only these groups of rows)
//   node tools/playlog_report.mjs --score person files...  (each column against score()'s bands; 'own': a bot's own kind)
//
// Who played: a bot run leaves 'autoplay' notes in the log (src/autoplay.js, with its kind and seed); an older bot
// series is told by its resets (the player's health back to full at once while not down) and by a cursor that never
// rests (a bot's cursor moves every 33 ms; a person's hand stops). `--label file=human|bot` overrides the guess.
import fs from 'node:fs';
import path from 'node:path';

const FPS = 30;
const args = process.argv.slice(2);
const labels = new Map();
let ref = null, perRun = false, jsonOut = null, only = null, scoreKind = null;
const files = [];
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--label') { const [f, l] = args[++i].split('='); labels.set(path.resolve(f), l); }
  else if (a === '--ref') ref = path.resolve(args[++i]);
  else if (a === '--runs') perRun = true;
  else if (a === '--json') jsonOut = args[++i];
  else if (a === '--only') only = args[++i].split(',');
  else if (a === '--score') scoreKind = args[++i];
  else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(new URL(import.meta.url), 'utf8').split('\n').filter(l => l.startsWith('//')).map(l => l.slice(3)).join('\n')); process.exit(0); }
  else files.push(a);
}
if (!files.length) { console.error('usage: node tools/playlog_report.mjs [--label f=human|bot] [--ref f] [--runs] [--json out] [--only groups] logs...'); process.exit(1); }

// ------------------------------------------------------------------ reading and telling who played
function load(file) {
  const L = JSON.parse(fs.readFileSync(file, 'utf8'));
  const name = path.basename(file).replace(/^play_/, '').replace(/_log\.json$/, '');
  const F = L.frames, E = L.events;
  const marks = E.filter(e => e.type === 'autoplay');
  // resets: health back to full at once while not getting up (a getup after a knock-out restores it too: from 0)
  const resets = [];
  for (let i = 1; i < F.length; i++) if (F[i].hp - F[i - 1].hp > 20 && F[i - 1].hp > 0) resets.push(F[i].f ?? i);
  const restFrac = F.filter(fr => fr.ms < 20).length / Math.max(1, F.length);
  let label = labels.get(path.resolve(file)), why;
  if (label) why = 'given';
  else if (marks.length) { label = 'bot'; why = `${marks.filter(m => m.phase === 'start').length} autoplay runs`; }
  else if (resets.length >= 2) { label = 'bot'; why = `${resets.length} resets`; }
  else if (restFrac < 0.05) { label = 'bot'; why = `cursor rests ${(restFrac * 100).toFixed(1)}%`; }
  else { label = 'human'; why = `cursor rests ${(restFrac * 100).toFixed(0)}%, no resets`; }
  return { file, name, L, label, why, marks };
}

// a piece of a log: frames [f0, f1) with their events and presses
// (clicks: the presses, and the left clicks on empty space that the game turns into a dash - not presses since the
// dash came in, but the same hand's rhythm)
const piece = (L, f0, f1) => {
  const events = L.events.filter(e => e.f >= f0 && e.f < f1), presses = L.presses.filter(p => p.f >= f0 && p.f < f1);
  const clicks = [...presses.map(p => ({ t: p.t, f: p.f, kind: p.kind })), ...events.filter(e => e.type === 'dash').map(e => ({ t: e.t, f: e.f, kind: 'dash' }))].sort((a, b) => a.t - b.t);
  return { frames: L.frames.slice(f0, f1), events, presses, clicks };
};

/** The columns of one log: the whole of a person's log; a bot log by kind (or by run with --runs). */
function columns(g) {
  const { L, marks } = g;
  if (g.label !== 'bot' || !marks.length) return [{ name: g.name, label: g.label, why: g.why, pieces: [piece(L, 0, L.frames.length)] }];
  const runs = [];
  for (const m of marks) if (m.phase === 'start') {
    const end = marks.find(x => x.f > m.f && (x.phase === 'start' || x.run === m.run)); // (its end, or the next run's start)
    runs.push({ kind: m.kind, seed: m.seed, p: piece(L, m.f, end ? end.f : L.frames.length), done: end?.phase === 'end' });
  }
  if (perRun) return runs.map(r => ({ name: `${g.name}:${r.kind}#${r.seed}`, label: 'bot', kind: r.kind, why: g.why, pieces: [r.p] }));
  const kinds = [...new Set(runs.map(r => r.kind))];
  return kinds.map(k => ({ name: `${g.name}:${k}`, label: 'bot', kind: k, why: `${runs.filter(r => r.kind === k).length} runs`, pieces: runs.filter(r => r.kind === k).map(r => r.p) }));
}

// ------------------------------------------------------------------ the measures
const q = (a, p) => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? v[Math.min(v.length - 1, Math.floor(p * (v.length - 1) + 0.5))] : null; };
const mean = a => { const v = a.filter(Number.isFinite); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null; };
const frac = (a, f) => (a.length ? a.filter(f).length / a.length : null);

function measure(pieces) {
  const F = pieces.flatMap(p => p.frames), E = pieces.flatMap(p => p.events), P = pieces.flatMap(p => p.presses);
  const sec = F.length / FPS, per = n => (sec > 0 ? (n / sec) * 60 : null);
  const ty = t => E.filter(e => e.type === t);
  const M = {};
  M.sec = sec;
  // pressing: rhythm (gaps within a piece; a gap over 2 s is a pause, not rhythm)
  const C = pieces.flatMap(p => p.clicks);
  const gaps = pieces.flatMap(p => p.clicks.slice(1).map((x, i) => x.t - p.clicks[i].t));
  const rhythm = gaps.filter(g => g <= 2);
  M.pressPerMin = per(C.length); M.strikePressPerMin = per(P.length); M.dashPerMin = per(C.filter(c => c.kind === 'dash').length);
  M.kickShare = C.length ? C.filter(p => p.kind === 'kick').length / C.length : null;
  M.gapP10 = q(rhythm, 0.1); M.gapP25 = q(rhythm, 0.25); M.gapP50 = q(rhythm, 0.5); M.gapP75 = q(rhythm, 0.75); M.gapP90 = q(rhythm, 0.9);
  M.gapUnder025 = frac(rhythm, g => g < 0.25);
  M.pauses = per(gaps.filter(g => g > 2).length);
  // bursts of mashing: presses with gaps under 0.6 s
  const bursts = pieces.flatMap(p => { const b = [], c = p.clicks; let n = c.length ? 1 : 0; for (let i = 1; i < c.length; i++) { if (c[i].t - c[i - 1].t < 0.6) n++; else { b.push(n); n = 1; } } if (n) b.push(n); return b; });
  M.burstP50 = q(bursts, 0.5); M.burstP75 = q(bursts, 0.75); M.burstP90 = q(bursts, 0.9);
  M.pressDistP50 = q(pieces.flatMap(p => p.clicks.map(x => p.frames[Math.round((x.t - (p.frames[0]?.t ?? 0)) * FPS)]?.d)), 0.5);
  const oc = o => (P.length ? P.filter(p => String(p.outcome).startsWith(o)).length / P.length : null);
  M.outStarted = oc('started'); M.outChained = oc('chained'); M.outMerged = oc('merged');
  M.outLost = P.length ? P.filter(p => !/^(started|chained|merged|queued)/.test(String(p.outcome))).length / P.length : null;
  M.pressShown = frac(P, p => p.tStrike != null);
  M.pressToStrike = q(P.map(p => p.tStrike), 0.5); M.pressToContact = q(P.map(p => p.tContact), 0.5);
  // the cursor (screen px/s, smoothed as the log does) and the body
  const ms = F.map(fr => fr.ms);
  M.cursorP25 = q(ms, 0.25); M.cursorP50 = q(ms, 0.5); M.cursorP75 = q(ms, 0.75); M.cursorP90 = q(ms, 0.9); M.cursorP99 = q(ms, 0.99);
  M.cursorRest = frac(ms, v => v < 20); M.cursorFast = frac(ms, v => v > 1500);
  let jumps = 0, sweeps = 0;
  for (const p of pieces) {
    const fr = p.frames;
    for (let i = 1; i < fr.length; i++) if (fr[i].mx != null && fr[i - 1].mx != null && Math.hypot(fr[i].mx - fr[i - 1].mx, fr[i].my - fr[i - 1].my) > 120) jumps++;
    // a sweep: the cursor travels over 250 px within 0.3 s (counted once per 0.3 s)
    for (let i = 9; i < fr.length; i += 3) {
      let len = 0; for (let k = i - 8; k <= i; k++) if (fr[k].mx != null && fr[k - 1].mx != null) len += Math.hypot(fr[k].mx - fr[k - 1].mx, fr[k].my - fr[k - 1].my);
      if (len > 250) { sweeps++; i += 6; }
    }
  }
  M.cursorJumps = per(jumps); M.sweeps = per(sweeps);
  M.cursorOff = frac(F, fr => fr.mx == null);
  // the cursor on the floor (m/s, smoothed as the log does: the same whatever the window's size)
  const fs2 = F.map(fr => Math.hypot(fr.cvx, fr.cvz));
  M.floorP50 = q(fs2, 0.5); M.floorP90 = q(fs2, 0.9);
  // where the cursor is: on which part of an opponent (none, head, body, legs), and how far its floor point is from the hips
  for (const [k, v] of [['aimNone', 0], ['aimHead', 1], ['aimBody', 2], ['aimLegs', 3]]) M[k] = frac(F, fr => fr.aim === v);
  M.cursorReach = q(F.filter(fr => fr.cx != null).map(fr => Math.hypot(fr.cx - fr.x, fr.cz - fr.z)), 0.5);
  const hv = F.map(fr => fr.hv);
  M.hipP50 = q(hv, 0.5); M.hipP90 = q(hv, 0.9); M.hipStill = frac(hv, v => v < 0.1);
  M.stretch = q(F.map(fr => fr.st), 0.5);
  const d = F.map(fr => fr.d);
  M.distP10 = q(d, 0.1); M.distP50 = q(d, 0.5); M.distP90 = q(d, 0.9);
  M.nearFrac = frac(F, fr => fr.near); M.closeFrac = frac(d, v => v < 0.7);
  // follow: lag that lines the hips' velocity up with the cursor's, and the speed taken on (as the log does)
  const fol = follow(pieces);
  M.followLag = fol.lag; M.followGain = fol.gain;
  // the opponents' attacks: how many, from how far, what came of them
  const es = ty('enemyStrike'), dodged = ty('dodged'), taken = ty('taken');
  M.enemyAttPerMin = per(es.length);
  M.enemyReachPerMin = per(es.filter(e => e.d <= 1.3).length); // (started close enough to land without a step)
  M.enemyStrikeDist = q(es.map(e => e.d), 0.5);
  M.enemyFront = es.some(e => e.angle != null) ? frac(es, e => e.angle != null && e.angle <= 55) : null;
  const used = new Set(), res = { landed: 0, dodged: 0, missed: 0 };
  for (const e of es) {
    const dg = dodged.find(x => x.by === e.who && x.t >= e.t - 0.05 && x.t <= e.t + 1.6 && !used.has(x));
    const tk = !dg && taken.find(x => x.t >= e.t && x.t <= e.t + 1.6 && !used.has(x));
    if (dg) { used.add(dg); res.dodged++; } else if (tk) { used.add(tk); res.landed++; } else res.missed++;
  }
  M.enemyLanded = es.length ? taken.length / es.length : null;
  M.enemyDodgedShare = es.length ? res.dodged / es.length : null;
  M.enemyMissedShare = es.length ? res.missed / es.length : null;
  M.takenPerMin = per(taken.length);
  M.takenDmgPerMin = per(taken.filter(e => !e.touch && !e.blocked).length);
  M.takenHeavyPerMin = per(taken.filter(e => ['heavy', 'down', 'fly'].includes(e.tier)).length);
  M.playerDownsPerMin = per(ty('down').filter(e => e.who === 'P').length);
  M.hitsUnseen = (() => { const t = taken.filter(e => !e.touch && e.tier !== 'ground'); return t.length ? t.filter(e => e.angle == null || (e.wind ?? 0) < 0.3).length / t.length : null; })();
  // dodging
  const dg = ty('dodge');
  M.dodgePerMin = per(dg.length); M.dodgeFlick = frac(dg, e => e.flick);
  M.dodgedPerMin = per(dodged.length);
  M.dodgedPerAtt = es.length ? dodged.length / es.length : null;
  // what the player gives
  const hits = ty('hit'), dmg = hits.filter(e => !e.touch && !e.blocked);
  M.hitsPerMin = per(dmg.length); M.touchPerMin = per(hits.filter(e => e.touch).length); M.blockedPerMin = per(hits.filter(e => e.blocked).length);
  for (const t of ['hit', 'heavy', 'down', 'fly', 'ground']) M[`tier_${t}`] = per(dmg.filter(e => e.tier === t).length);
  M.slamsPerMin = per(hits.filter(e => e.slam).length);
  M.countersPerMin = per(hits.filter(e => e.counter).length);
  M.kosPerMin = per(ty('ko').length);
  M.fallsPerMin = per(dmg.filter(e => e.fall).length);
  // blows seen: does the struck body move? (0.3 s after a damaging blow, how far the hips went; barely = under 0.1 m)
  const moved = [...dmg, ...taken.filter(e => !e.touch && !e.blocked)];
  M.blowDisp03 = q(moved.map(e => e.disp03), 0.5);
  M.blowUnmoved = (() => { const v = moved.filter(e => e.disp03 != null); return v.length ? v.filter(e => e.disp03 < 0.1).length / v.length : null; })();
  M.reactDelay = q(moved.map(e => e.react), 0.5); M.knockDelay = q(moved.filter(e => ['heavy', 'down', 'fly'].includes(e.tier)).map(e => e.knock), 0.5);
  // dull or stuck time
  M.flatFrac = flat(pieces);
  M.longestApart = Math.max(0, ...pieces.map(p => { let best = 0, run = 0; for (const fr of p.frames) { if (fr.d > 3) run++; else { best = Math.max(best, run); run = 0; } } return Math.max(best, run) / FPS; }));
  M.playerUnable = frac(F, fr => fr.ps >= 3);
  M.playerStill = F.length && F[0].pl !== undefined ? frac(F, fr => fr.ps >= 3 && (fr.pl ?? fr.hv) < 0.08) : null;
  const downEnds = ty('downEnd').filter(e => e.next !== 'out');
  M.downStill = mean(downEnds.map(e => e.still));
  M.enemyStunned = (() => { const n = F.filter(fr => fr.near && fr.act !== undefined); return n.length ? n.filter(fr => !fr.act).length / n.length : null; })();
  return M;
}

function follow(pieces) {
  let best = { L: 0, c: -1 };
  const all = pieces.map(p => p.frames).filter(f => f.length > 20);
  if (!all.length) return { lag: null, gain: null };
  for (let L = 0; L <= 24; L++) {
    let su = 0, sv = 0, uv = 0;
    for (const F of all) for (let t = 0; t + L < F.length; t++) { const a = F[t], b = F[t + L]; su += a.cvx ** 2 + a.cvz ** 2; sv += b.vx ** 2 + b.vz ** 2; uv += a.cvx * b.vx + a.cvz * b.vz; }
    const c = uv / Math.sqrt(Math.max(1e-9, su * sv));
    if (c > best.c) best = { L, c };
  }
  let num = 0, den = 0;
  for (const F of all) for (let t = 0; t + best.L < F.length; t++) { const a = F[t], b = F[t + best.L]; const u = Math.hypot(a.cvx, a.cvz); if (u > 0.3) { num += Math.min(Math.hypot(b.vx, b.vz), 6); den += Math.min(u, 6); } }
  return { lag: best.L / FPS, gain: den ? num / den : null };
}

function flat(pieces) {
  // close in, yet nothing for 0.6 s (no press, strike, blow, attack, fall; the hips not running) - as the log counts
  let near = 0, dull = 0;
  for (const p of pieces) {
    const n = p.frames.length, busy = new Uint8Array(n);
    // (frames carry no number: an event's frame within the piece goes by time)
    const t0 = p.frames[0]?.t ?? 0;
    for (const e of p.events) if (['press', 'strike', 'hit', 'taken', 'enemyStrike', 'down', 'ko'].includes(e.type)) {
      const i0 = Math.round((e.t - t0) * FPS);
      for (let i = Math.max(0, i0 - 3); i < Math.min(n, i0 + 18); i++) busy[i] = 1;
    }
    p.frames.forEach((fr, i) => { if (fr.hv > 0.8 || fr.ps >= 2) busy[i] = 1; if (fr.near) { near++; if (!busy[i]) dull++; } });
  }
  return near ? dull / near : null;
}

// ------------------------------------------------------------------ the rows: [key, label, digits, group]
const ROWS = [
  ['sec', '秒', 0, 'base'],
  ['pressPerMin', 'クリック/分(押+ダッシュ)', 0, 'presses'], ['strikePressPerMin', '押した(打撃)/分', 0, 'presses'], ['dashPerMin', 'ダッシュ/分', 1, 'presses'], ['kickShare', 'キックの割合', 2, 'presses'],
  ['gapP10', 'クリック間隔 p10 s', 2, 'presses'], ['gapP25', 'クリック間隔 p25 s', 2, 'presses'], ['gapP50', 'クリック間隔 p50 s', 2, 'presses'], ['gapP75', 'クリック間隔 p75 s', 2, 'presses'], ['gapP90', 'クリック間隔 p90 s', 2, 'presses'],
  ['gapUnder025', '間隔<0.25s の割合', 2, 'presses'], ['pauses', '2秒超の休み/分', 1, 'presses'],
  ['burstP50', '連打の長さ p50 回', 0, 'presses'], ['burstP75', '連打の長さ p75 回', 0, 'presses'], ['burstP90', '連打の長さ p90 回', 0, 'presses'], ['pressDistP50', '押した時の距離 p50', 2, 'presses'],
  ['outStarted', '押→すぐ出た', 2, 'presses'], ['outChained', '押→つないだ', 2, 'presses'], ['outMerged', '押→まとめられた', 2, 'presses'], ['outLost', '押→消えた', 2, 'presses'],
  ['pressShown', '押→打撃が出た率', 2, 'presses'], ['pressToStrike', '押→打撃 s', 2, 'presses'], ['pressToContact', '押→当たり s', 2, 'presses'],
  ['cursorP25', 'カーソル速さ p25', 0, 'cursor'], ['cursorP50', 'カーソル速さ p50', 0, 'cursor'], ['cursorP75', 'カーソル速さ p75', 0, 'cursor'], ['cursorP90', 'カーソル速さ p90', 0, 'cursor'], ['cursorP99', 'カーソル速さ p99', 0, 'cursor'],
  ['cursorRest', 'カーソル静止の割合', 2, 'cursor'], ['cursorFast', 'カーソル>1500の割合', 2, 'cursor'], ['cursorJumps', 'カーソル跳び/分', 1, 'cursor'], ['sweeps', '大振り(250px/0.3s)/分', 1, 'cursor'], ['cursorOff', 'カーソル画面外', 2, 'cursor'],
  ['floorP50', 'カーソル速さ(床) p50 m/s', 2, 'cursor'], ['floorP90', 'カーソル速さ(床) p90 m/s', 2, 'cursor'], ['cursorReach', 'カーソル-腰 p50 m', 2, 'cursor'],
  ['aimNone', '狙い: どこでもない', 2, 'cursor'], ['aimHead', '狙い: 頭', 2, 'cursor'], ['aimBody', '狙い: 胴', 2, 'cursor'], ['aimLegs', '狙い: 脚', 2, 'cursor'],
  ['hipP50', '腰の速さ p50', 2, 'body'], ['hipP90', '腰の速さ p90', 2, 'body'], ['hipStill', '腰が止まっている', 2, 'body'], ['stretch', 'ばねの伸び p50 m', 2, 'body'],
  ['followLag', '追従の遅れ s', 2, 'body'], ['followGain', '追従の強さ', 2, 'body'],
  ['distP10', '相手との距離 p10', 2, 'body'], ['distP50', '相手との距離 p50', 2, 'body'], ['distP90', '相手との距離 p90', 2, 'body'], ['nearFrac', '交戦中の割合', 2, 'body'], ['closeFrac', '距離<0.7 の割合', 2, 'body'],
  ['enemyAttPerMin', '相手の攻め/分', 1, 'enemy'], ['enemyReachPerMin', '届く距離からの攻め/分', 1, 'enemy'], ['enemyStrikeDist', '攻め始めの距離 p50', 2, 'enemy'], ['enemyFront', '前からの攻め', 2, 'enemy'],
  ['enemyLanded', '攻め→当たり', 2, 'enemy'], ['enemyDodgedShare', '攻め→かわされた', 2, 'enemy'], ['enemyMissedShare', '攻め→空振り', 2, 'enemy'],
  ['takenPerMin', '受けた/分', 1, 'enemy'], ['takenDmgPerMin', '受けた(かすり以外)/分', 1, 'enemy'], ['takenHeavyPerMin', '受けた重い/分', 1, 'enemy'], ['playerDownsPerMin', '自分が倒れた/分', 2, 'enemy'], ['hitsUnseen', '見えない被弾', 2, 'enemy'],
  ['dodgePerMin', 'よけ(dodge)/分', 1, 'dodge'], ['dodgeFlick', 'よけ: フリック', 2, 'dodge'], ['dodgedPerMin', 'かわした(dodged)/分', 1, 'dodge'], ['dodgedPerAtt', 'かわした/攻め', 2, 'dodge'], ['countersPerMin', 'カウンター/分', 1, 'dodge'],
  ['hitsPerMin', '当てた/分', 1, 'hits'], ['touchPerMin', 'かすり/分', 1, 'hits'], ['blockedPerMin', '防がれた/分', 1, 'hits'],
  ['tier_hit', '当て: 普通/分', 1, 'hits'], ['tier_heavy', '当て: 重い/分', 1, 'hits'], ['tier_down', '当て: 倒す/分', 1, 'hits'], ['tier_fly', '当て: 飛ぶ/分', 1, 'hits'], ['tier_ground', '当て: 倒れた相手/分', 1, 'hits'],
  ['slamsPerMin', '体当たり/分', 1, 'hits'], ['kosPerMin', 'KO/分', 2, 'hits'], ['fallsPerMin', '倒した/分', 1, 'hits'],
  ['blowDisp03', '打撃0.3s後の移動 p50 m', 2, 'blows'], ['blowUnmoved', '打っても動かない(<0.1m)', 2, 'blows'], ['reactDelay', '揺れるまで s', 2, 'blows'], ['knockDelay', '飛ぶまで s', 2, 'blows'],
  ['flatFrac', 'メリハリなし', 2, 'dull'], ['longestApart', '最長の離れ s', 1, 'dull'], ['playerUnable', '自分が動けない', 2, 'dull'], ['playerStill', '動けず止まる', 2, 'dull'], ['downStill', '倒れて止まる', 2, 'dull'], ['enemyStunned', '相手が動けない', 2, 'dull'],
];

// ------------------------------------------------------------------ the table
const logs = files.map(f => load(f));
const cols = logs.flatMap(columns).map(c => ({ ...c, M: measure(c.pieces) }));
const humans = cols.filter(c => c.label === 'human');
let refCol = ref ? cols.find(c => path.resolve(logs.find(g => g.name === c.name.split(':')[0])?.file ?? '') === ref) : null;
if (!refCol && humans.length) refCol = humans.reduce((a, b) => (a.name > b.name ? a : b)); // (the latest person's play)

const fmt = (v, dg) => (v == null || !Number.isFinite(v) ? '-' : v.toFixed(dg));
const W0 = 24, W = 11;
const pad = (s, w) => { const len = [...s].reduce((a, ch) => a + (ch.charCodeAt(0) > 0xff ? 2 : 1), 0); return s + ' '.repeat(Math.max(1, w - len)); };
const lpad = (s, w) => ' '.repeat(Math.max(1, w - s.length)) + s;
console.log('ログ:');
for (const c of cols) console.log(`  ${c === refCol ? '*' : ' '} ${c.name.padEnd(28)} ${c.label.padEnd(6)} ${c.why}  (${(c.M.sec).toFixed(0)} s)`);
console.log(refCol ? `  (* = 基準: 人のプレイ。bot の列の後ろの「Δ」は基準からのずれ、! は大きくずれている)` : '  (人のプレイが無いので比較なし)');
const bots = cols.filter(c => c.label === 'bot');
const short = c => (c.label === 'human' ? 'H' : 'B') + ':' + c.name.replace(/^\d{8}/, '').slice(0, W - 3);
let head = pad('', W0) + cols.map(c => lpad(short(c), W)).join('');
if (refCol) head += '   ' + bots.map(c => lpad('Δ' + short(c).slice(2, 9), 9)).join('');
let group = null;
for (const [key, label, dg, g] of ROWS) {
  if (only && !only.includes(g)) continue;
  if (g !== group) { group = g; console.log(`\n[${g}]` + (group === ROWS[0][3] ? '' : '')); console.log(head); }
  let line = pad(label, W0) + cols.map(c => lpad(fmt(c.M[key], dg), W)).join('');
  if (refCol) line += '   ' + bots.map(c => lpad(delta(refCol.M[key], c.M[key], dg), 9)).join('');
  console.log(line);
}
function delta(h, b, dg) {
  if (h == null || b == null || !Number.isFinite(h) || !Number.isFinite(b)) return '-';
  const diff = b - h, rel = Math.abs(h) > 1e-9 ? diff / Math.abs(h) : (Math.abs(diff) > 0 ? Infinity : 0);
  const big = Math.abs(h) < 1 && Math.abs(h) > 0 && dg >= 2 ? Math.abs(diff) > 0.1 && Math.abs(rel) > 0.3 : Math.abs(rel) > 0.3;
  const s = Math.abs(h) >= 1 || dg < 2 ? (Number.isFinite(rel) ? `${rel >= 0 ? '+' : ''}${Math.round(rel * 100)}%` : '+∞') : `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}`;
  return (big ? '!' : '') + s;
}

if (refCol && bots.length) {
  // the bot columns' fit to the person on the rows that describe how a person plays (not the game's answer to it)
  const PLAY = ['pressPerMin', 'gapP25', 'gapP50', 'gapP75', 'gapUnder025', 'burstP50', 'cursorP50', 'cursorP90', 'cursorRest', 'cursorFast', 'cursorJumps', 'sweeps', 'floorP50', 'cursorReach', 'aimNone', 'aimHead', 'distP10', 'distP50', 'distP90', 'hipP50', 'kickShare', 'slamsPerMin', 'dodgePerMin', 'takenPerMin', 'enemyLanded'];
  console.log(`\n人らしさ（遊び方の ${PLAY.length} 項目、基準からのずれの平均。0 = 同じ。|Δ| が 30% を超える項目）:`);
  for (const c of bots) {
    const ds = PLAY.map(k => { const h = refCol.M[k], b = c.M[k]; if (h == null || b == null) return null; return { k, r: Math.min(2, Math.abs(b - h) / Math.max(Math.abs(h), /^(cursorRest|kickShare|gapUnder025|enemyLanded|aim)/.test(k) ? 0.1 : 1e-6)) }; }).filter(Boolean);
    const m = mean(ds.map(x => x.r));
    console.log(`  ${short(c).padEnd(14)} ${m.toFixed(2)}   ${ds.filter(x => x.r > 0.3).map(x => `${x.k}(${Math.round(x.r * 100)}%)`).join(' ')}`);
  }
}
if (scoreKind) {
  // every column against the bands of src/autoplay.js score() for one kind of player (a bot column: its own kind) -
  // a person's play too, from the same measures (no tally: its counter rate is not counted)
  const { score } = await import(new URL('../src/autoplay.js', import.meta.url));
  console.log(`
score() の帯で（${scoreKind === 'own' ? 'bot は各自の種類、人は person' : scoreKind}）: 合計 / 帯の外の項目`);
  for (const c of cols) {
    const M = c.M, n = x => (x == null ? null : Math.round((x * M.sec) / 60));
    const sum = { enemyStrikes: n(M.enemyAttPerMin), enemyLanded: M.enemyLanded, enemyStunned: M.enemyStunned, flatFrac: M.flatFrac, counters: n(M.countersPerMin), downStill: M.downStill,
      playerStill: M.playerStill, hitsUnseen: M.hitsUnseen, longestApart: M.longestApart, kos: n(M.kosPerMin), dodged: n(M.dodgedPerMin),
      enemyReachPerMin: M.enemyReachPerMin, takenDmgPerMin: M.takenDmgPerMin, blowUnmoved: M.blowUnmoved, playerUnable: M.playerUnable, pressLostFrac: M.outLost, dodgePerMin: M.dodgePerMin, slamsPerMin: M.slamsPerMin };
    const kind = scoreKind === 'own' ? (c.kind ?? 'person') : scoreKind;
    const sc = score(sum, { counters: 0 }, M.sec, kind);
    console.log(`  ${short(c).padEnd(14)} ${kind.padEnd(8)} ${sc.total.toFixed(2).padStart(6)}  (legacy ${sc.legacy.toFixed(2)})  ${Object.entries(sc.parts).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(', ')}`);
  }
}
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(cols.map(c => ({ name: c.name, label: c.label, kind: c.kind ?? null, why: c.why, ref: c === refCol, M: c.M })), null, 1));
