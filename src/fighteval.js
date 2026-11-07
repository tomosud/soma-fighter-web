// SOMA Brawl test bench: scripted player intents (what a person would try) run in the page through the debug API
// (fight.html?debug), each recorded frame by frame: the input (where the pointer pulls, right presses) and what came
// of it (how soon the body answered, where it went, whether the blow landed and how hard, whether the body got
// stuck, sat down or jumped, and how natural the motion is by the evaluator's z-scores against Mixamo). Saved as
// heat maps and JSON in eval_out/ (tools/serve.mjs), so runs can be compared change after change.
//   const ev = await import('./src/fighteval.js'); ev.runAll(brawl, 'tag')   (a worker ticker must drive brawl.draw)
import { Vector3 } from 'three';
import { score, drawHeatmap, EVAL_FPS } from './evaluate.js';

const flat = v => new Vector3(v.x, 0, v.z);
const sleep = ms => new Promise(r => setTimeout(r, ms));

/**
 * Intents. step(S, t) is called every simulation step: S.pull(point | null), S.press(), S.e (the opponent),
 * S.P (the player), S.around(angle, r) (a point at distance r from the opponent; angle 0 = on the player's side).
 */
export const SCENARIOS = [
  { name: 'idle', label: '何もしない', sec: 6, step() {} },
  {
    name: 'step_jab', label: '踏み込んで打つ', sec: 9,
    step(S, t) {
      const c = t % 3;
      if (c < 0.9) S.pull(S.around(0, 0.85));
      else if (c < 1.8) S.pull(S.around(0, 2.4));
      else S.pull(null);
      if (S.once(`p${Math.floor(t / 3)}`, c > 0.55)) S.press('punch', 'head');
    },
  },
  {
    name: 'circle_hook', label: '回り込んで打つ', sec: 9,
    step(S, t) {
      S.pull(S.around(0.9 + t * 1.6, 1.1));
      if (S.once(`p${Math.floor(t / 1.5)}`, t % 1.5 > 0.7)) S.press(Math.floor(t / 1.5) % 2 ? 'kick' : 'punch');
    },
  },
  {
    name: 'combo', label: '連打', sec: 9,
    step(S, t) {
      const c = t % 3;
      S.pull(c < 1.4 ? S.around(0, 0.95) : S.around(0, 2.3));
      for (const [k, kind] of [[0.5, 'punch'], [0.68, 'punch'], [0.86, 'kick']]) if (S.once(`p${Math.floor(t / 3)}_${k}`, c > k)) S.press(kind);
    },
  },
  {
    name: 'back_kick', label: '下がりながら迎え撃つ', sec: 9,
    step(S, t) {
      S.pull(S.around(0, 2.6));
      const d = S.dist();
      if (d < 1.5 && S.once(`p${Math.floor(t / 1.2)}`, true)) S.press('kick', 'body');
    },
  },
  {
    name: 'slam', label: '体当たり', sec: 9,
    step(S, t) {
      const c = t % 3;
      S.pull(c < 1.3 ? S.around(0, 3.2) : S.around(Math.PI, 1.8)); // back off, then drive straight through
    },
  },
  {
    name: 'close_dance', label: '近くで細かく動く', sec: 8,
    step(S, t) {
      // small quick moves around the opponent at arm's length: is the body with the hand, or stuck?
      S.pull(S.around(0.9 * Math.sin(t * 3.1), 1.0 + 0.25 * Math.sin(t * 5.3)));
    },
  },
  {
    name: 'sidestep_counter', label: '横に避けて返す', sec: 10,
    step(S, t) {
      const e = S.e;
      if (e.state === 'attack' && !S.mem.dodgeAt) S.mem.dodgeAt = t;
      if (S.mem.dodgeAt != null) {
        const u = t - S.mem.dodgeAt;
        S.pull(S.around(0.9, 1.3));
        if (u > 0.4 && S.once(`c${S.mem.dodgeAt.toFixed(2)}`, true)) S.press('punch', 'head');
        if (u > 1.4) S.mem.dodgeAt = null;
      } else S.pull(S.around(0, 1.5));
    },
  },
  {
    name: 'trade', label: '立ち止まって打ち合う', sec: 8,
    step(S, t) {
      S.pull(null);
      if (S.once(`p${Math.floor(t / 0.9)}`, true) && S.dist() < 1.7) S.press(Math.floor(t / 0.9) % 3 === 2 ? 'kick' : 'punch', Math.floor(t / 0.9) % 3 === 2 ? 'legs' : 'head');
    },
  },
];

const STATE_CODE = { free: 0, attack: 2, guard: 1, hit: 3.5, fall: 5, down: 4.5, getup: 4, out: 5 };

/** Run one intent; returns a recording. */
async function runScenario(brawl, sc) {
  const P = brawl.player;
  // one opponent, placed 2.2 m in front; strong (the run goes on), the player full
  for (const f of brawl.fighters) if (!f.isPlayer && f.state !== 'out' && f !== brawl.fighters.find(g => !g.isPlayer && g.state !== 'out')) { f.state = 'out'; f.outAt = -9; }
  brawl.settings.enemies = 1;
  for (let k = 0; k < 120 && !brawl.fighters.some(f => !f.isPlayer && f.state === 'free'); k++) await sleep(50);
  const e = brawl.fighters.find(f => !f.isPlayer && f.state !== 'out');
  P.hp = P.maxHp; e.hp = e.maxHp = 400;
  const h = Math.atan2(e.P[0].x - P.P[0].x, e.P[0].z - P.P[0].z);
  e.track.reset(flat(P.P[0]).add(new Vector3(Math.sin(h), 0, Math.cos(h)).multiplyScalar(2.2)), h + Math.PI);
  brawl.pull(null);
  await sleep(800);
  const rec = { name: sc.name, label: sc.label, frames: [], poses: [], ePoses: [], presses: [], hits: [], marks: [], jumps0: brawl.jumps.length };
  let pointer = null, step = 0, lastHit = brawl.lastHit;
  const fired = new Set();
  const S = {
    P, e, mem: {},
    pull(p) { pointer = p ? p.clone().setY(0.95) : null; brawl.pull(pointer); },
    // (the cursor's aim follows the opponent while moving about: as a person's would)
    press(kind = 'punch', part = kind === 'kick' ? 'body' : 'head') {
      const joint = brawl.rig.index[part === 'head' ? 'Head' : part === 'body' ? 'Spine2' : 'LeftShin'];
      brawl.aim({ enemy: e, part, joint, off: new Vector3() }); S.mem.aimSet = brawl.clock;
      P.strikeInput = { t: brawl.clock, kind };
      rec.presses.push({ t: brawl.clock - t0, f: rec.frames.length, motion: null, contact: null, kind });
    },
    once(key, cond) { if (!cond || fired.has(key)) return false; fired.add(key); return true; },
    dist() { return flat(e.P[0].clone().sub(P.P[0])).length(); },
    around(angle, r) {
      const toP = flat(P.P[0].clone().sub(e.P[0])); const a = Math.atan2(toP.x, toP.z) + angle;
      return flat(e.P[0]).add(new Vector3(Math.sin(a), 0, Math.cos(a)).multiplyScalar(r));
    },
  };
  const t0 = brawl.clock;
  await new Promise(resolve => {
    brawl.onStep = dt => {
      const t = brawl.clock - t0;
      if (t > sc.sec) { brawl.onStep = null; resolve(); return; }
      // the cursor stays on the opponent (its head), as a person's would; a press moves it to the part it strikes
      if (!S.mem.aimSet || brawl.clock - S.mem.aimSet > 0.6) brawl.aim({ enemy: e, part: 'head', joint: brawl.rig.index.Head, off: new Vector3() });
      sc.step(S, t);
      // the press answered: the first frame of the strike's motion shown; its contact
      for (const pr of rec.presses) if (pr.motion == null && P.track.tag.kind === 'attack' && t > pr.t) pr.motion = t - pr.t;
      // the first visible answer: the hands (or the chest) moving clearly faster than before the press
      {
        const j = ['LeftHand', 'RightHand', 'Chest'].map(n => brawl.rig.index[n]);
        const sp = Math.max(...j.map(i => P.P[i].distanceTo(P.prevP[i]) / dt));
        for (const pr of rec.presses) {
          if (pr.base == null) pr.base = S.mem.sp ?? sp;
          if (pr.move == null && t > pr.t && sp > pr.base + 0.5) pr.move = t - pr.t;
        }
        S.mem.sp = S.mem.sp == null ? sp : 0.9 * S.mem.sp + 0.1 * sp;
      }
      const lh = brawl.lastHit;
      if (lh && lh !== lastHit) {
        lastHit = lh;
        rec.hits.push({ t, f: rec.frames.length, by: lh.by === 'あなた' ? 'P' : 'E', energy: lh.energy, blocked: !!lh.blocked, slam: !!lh.slam });
        if (lh.by === 'あなた') for (const pr of rec.presses) if (pr.contact == null && t - pr.t < 1.2 && t > pr.t) { pr.contact = t - pr.t; pr.energy = lh.energy; break; }
      }
      if (++step % 2) return; // 30 fps
      const hv = flat(P.P[0].clone().sub(P.prevP[0])).length() / dt;
      const lag = pointer ? flat(pointer.clone().sub(P.P[0])).length() : 0;
      let limb = 0;
      for (const part of ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']) {
        const j = brawl.rig.index[part];
        limb = Math.max(limb, P.P[j].distanceTo(P.prevP[j]) / dt);
      }
      const downish = ['fall', 'down', 'getup', 'out'].includes(P.state);
      rec.frames.push({
        t, lag, hv, limb, d: S.dist(),
        stuck: pointer && lag > 0.6 && hv < 0.35 && !downish ? 1 : 0,
        sit: P.P[0].y < 0.6 && !downish ? 1 : 0,
        ps: STATE_CODE[P.state] ?? 0, es: STATE_CODE[e.state] ?? 0,
      });
      rec.poses.push({ root: P.shown.root.clone(), rot: P.shown.rot.map(q => q.clone()) });
    };
  });
  rec.jumps = brawl.jumps.length - rec.jumps0;
  brawl.pull(null); brawl.aim(null);
  return rec;
}

const med = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

/** The evaluator's summary over the frames where the player is free or striking (blows, falls, getting up are
 * movements the everyday reference does not cover). */
function maskedNat(rec, nat) {
  const n = Math.min(nat.n, rec.frames.length), ok = [];
  for (let t = 8; t < n - 8; t++) if (rec.frames[t].ps < 3) ok.push(t);
  const body = nat.rows.filter(r => !r.whole && !r.minor);
  const all = [];
  for (const r of body) for (const t of ok) all.push(r.z[t]);
  all.sort((a, b) => a - b);
  const slide = [];
  for (const nm of ['左足すべり', '右足すべり']) { const r = nat.rows.find(x => x.name === nm); for (const t of ok) if (r.raw[t] > 0) slide.push(r.raw[t]); }
  slide.sort((a, b) => a - b);
  return { meanZ: all.length ? all.reduce((a, b) => a + b, 0) / all.length : 0, p95Z: all[Math.floor(all.length * 0.95)] ?? 0, slide: slide[slide.length >> 1] ?? 0, frames: ok.length };
}
function metrics(rec, nat) {
  const nm = maskedNat(rec, nat);
  const F = rec.frames, mine = rec.hits.filter(h => h.by === 'P' && !h.blocked), theirs = rec.hits.filter(h => h.by === 'E');
  return {
    presses: rec.presses.length,
    latMove: med(rec.presses.map(p => p.move)),
    latMotion: med(rec.presses.map(p => p.motion)),
    latContact: med(rec.presses.map(p => p.contact)),
    hitRate: rec.presses.length ? rec.presses.filter(p => p.contact != null).length / rec.presses.length : null,
    hits: mine.length,
    energy: mine.length ? mean(mine.map(h => h.energy)) : 0,
    slams: mine.filter(h => h.slam).length,
    taken: theirs.reduce((a, h) => a + h.energy, 0),
    stuck: mean(F.map(f => f.stuck)),
    lag: mean(F.filter(f => f.lag > 0).map(f => f.lag)),
    sits: F.reduce((a, f, i) => a + (f.sit && !(F[i - 1]?.sit) ? 1 : 0), 0),
    jumps: rec.jumps,
    meanZ: nm.meanZ, p95Z: nm.p95Z, slide: nm.slide,
  };
}

// how bad a value is, 0 (good) .. 5 (bad), for the matrix colours
const BAD = {
  latMove: v => (v == null ? null : 5 * clamp01((v - 0.08) / 0.3)),
  latMotion: v => (v == null ? null : 5 * clamp01((v - 0.15) / 0.35)),
  latContact: v => (v == null ? null : 5 * clamp01((v - 0.4) / 0.5)),
  hitRate: v => (v == null ? null : 5 * clamp01((0.7 - v) / 0.5)),
  energy: (v, m) => (m.hits ? 5 * clamp01((60 - v) / 50) : null),
  taken: v => 5 * clamp01((v - 60) / 200),
  stuck: v => 5 * clamp01((v - 0.03) / 0.3),
  lag: v => 5 * clamp01((v - 0.5) / 0.9),
  sits: v => 5 * clamp01(v / 2),
  jumps: v => 5 * clamp01(v),
  meanZ: v => 5 * clamp01((v - 0.8) / 1.7),
  p95Z: v => 5 * clamp01((v - 3) / 3),
  slide: v => 5 * clamp01((v - 2) / 6),
};
const clamp01 = x => Math.max(0, Math.min(1, x));
const LABEL = { latMove: '押→腕が動く s', latMotion: '押→打撃 s', latContact: '押→当たり s', hitRate: '命中率', hits: '当てた数', energy: '平均 J', slams: '体当たり', taken: '受けた J', stuck: '止まり率', lag: '指との距離 m', sits: 'しりもち', jumps: '瞬間移動', meanZ: '自然さ z', p95Z: 'z 95%', slide: '足すべり cm/s' };
const heat = v => { // same scale as the evaluator: 0 blue -> 3 yellow -> 5 red
  const x = Math.min(1, v / 5);
  return `rgb(${Math.round(255 * Math.min(1, Math.max(0, 1.8 * x - 0.2)))},${Math.round(255 * Math.max(0, Math.min(1, 1.6 * x + 0.1) - Math.max(0, 2.2 * (x - 0.6))))},${Math.round(255 * Math.max(0, 0.55 - x))})`;
};

function drawMatrix(results, tag) {
  const keys = Object.keys(LABEL), names = Object.keys(results);
  const left = 170, top = 70, cw = 86, rh = 26;
  const c = document.createElement('canvas');
  c.width = left + keys.length * cw + 20; c.height = top + names.length * rh + 30;
  const g = c.getContext('2d');
  g.fillStyle = '#111'; g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = '#ddd'; g.font = '13px sans-serif'; g.fillText(`SOMA Brawl 試験 ${tag}（意図 × 指標、青=良い 赤=悪い、灰=該当なし）`, 10, 20);
  g.font = '11px sans-serif';
  keys.forEach((k, i) => { g.save(); g.translate(left + i * cw + 6, top - 8); g.fillStyle = '#ccc'; g.fillText(LABEL[k], 0, 0); g.restore(); });
  names.forEach((n, r) => {
    const m = results[n].metrics, y = top + r * rh;
    g.fillStyle = '#ddd'; g.fillText(results[n].label, 8, y + 17);
    keys.forEach((k, i) => {
      const v = m[k], b = BAD[k]?.(v, m);
      g.fillStyle = b == null ? '#333' : heat(b); g.fillRect(left + i * cw, y, cw - 2, rh - 2);
      g.fillStyle = b != null && b > 2.2 && b < 4.2 ? '#000' : '#fff';
      g.fillText(v == null ? '-' : typeof v === 'number' ? (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)) : String(v), left + i * cw + 6, y + 17);
    });
  });
  return c;
}

/** Per-intent timeline: input and result channels over the evaluator's naturalness rows. */
function drawTimeline(rec, nat, tag) {
  const F = rec.frames, n = Math.min(F.length, nat.n);
  const row = (name, f, whole = true) => ({ name, z: Float32Array.from({ length: n }, (_, t) => Math.max(0, Math.min(5, f(F[t], t)))), whole });
  const hitAt = new Map(rec.hits.map(h => [Math.min(n - 1, Math.floor(h.f)), h]));
  const rows = [
    row('入力: 指との距離', f => f.lag / 0.4),
    row('入力: 右クリック', (f, t) => (rec.presses.some(p => p.f === t) ? 5 : 0)),
    row('結果: 腰の速さ', f => f.hv / 0.8),
    row('結果: 止まり', f => f.stuck * 5),
    row('結果: 手足の速さ', f => f.limb / 2.5),
    row('結果: 自分の状態', f => f.ps),
    row('結果: 相手の状態', f => f.es),
    row('結果: 相手との距離', f => Math.max(0, 2.5 - f.d) * 2),
    row('結果: 当てた J', (f, t) => { const h = hitAt.get(t); return h && h.by === 'P' ? h.energy / 25 : 0; }),
    row('結果: 受けた J', (f, t) => { const h = hitAt.get(t); return h && h.by === 'E' ? h.energy / 25 : 0; }),
    row('結果: しりもち', f => f.sit * 5),
    ...nat.rows.filter(r => r.whole).map(r => ({ ...r, z: r.z.slice(0, n) })),
    { name: '体の自然さ（関節平均）', z: Float32Array.from({ length: n }, (_, t) => { const b = nat.rows.filter(r => !r.whole && !r.minor); return b.reduce((a, r) => a + r.z[t], 0) / b.length; }), whole: true },
  ];
  const marks = [...rec.presses.map(p => ({ t: p.f, label: '押', color: 'rgba(255,255,255,0.7)' })), ...rec.hits.map(h => ({ t: Math.floor(h.f), label: h.by === 'P' ? `${h.slam ? '体' : ''}${h.energy.toFixed(0)}` : `受${h.energy.toFixed(0)}`, color: h.by === 'P' ? 'rgba(255,170,60,0.9)' : 'rgba(255,80,80,0.9)' }))];
  const c = document.createElement('canvas');
  drawHeatmap(c.getContext('2d'), { rows, n, marks }, `${tag} / ${rec.label}（${rec.name}）  ${EVAL_FPS} fps`);
  return c;
}

const save = async (name, body) => fetch(`/save/${name}`, { method: 'POST', body });
const png = c => new Promise(r => c.toBlob(r, 'image/png'));

export async function runAll(brawl, tag = 'run', only = null) {
  const stats = await (await fetch(new URL('../assets/eval_stats.json', import.meta.url))).json();
  const results = {};
  const keepEnemies = brawl.settings.enemies;
  for (const sc of SCENARIOS) {
    if (only && !only.includes(sc.name)) continue;
    const rec = await runScenario(brawl, sc);
    const nat = score(brawl.rig, rec.poses, stats, []);
    results[sc.name] = { label: sc.label, metrics: metrics(rec, nat) };
    await save(`fight_${tag}_${sc.name}.png`, await png(drawTimeline(rec, nat, tag)));
    window.fightEvalProgress = sc.name;
  }
  await save(`fight_${tag}_matrix.png`, await png(drawMatrix(results, tag)));
  await save(`fight_${tag}_scores.json`, JSON.stringify({ tag, date: new Date().toISOString(), results }, null, 1));
  brawl.settings.enemies = keepEnemies;
  brawl.unscript(); // (the mouse again)
  return results;
}
