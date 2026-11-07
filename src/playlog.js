// SOMA Brawl play log: what a person does and what comes of it, recorded all through play (30 fps), cut into
// stretches (fighting at close range / apart, and cut again past 40 s), each summed up - so the play can be judged as
// it is felt and compared change after change:
//   - moving: how the hips follow where the cursor is going (delay, how much of the cursor's speed they take on)
//   - pressing: what became of every press (started, kept and chained, merged, dropped), how soon its strike showed
//     and how soon it landed
//   - the fight: how often the opponents attack and land, how much of the time they cannot act (hit, down)
//   - blows: how soon the struck body moves, how far, which fall, how varied
//   - on the floor: how much the body moves while down (or lies still like a thing)
//   - flat time: close in, yet nothing happening (no press, strike, blow, attack, fall, no real movement)
// Saved while playing to eval_out/play_<session>_* (tools/serve.mjs, local only), or downloaded with the button:
//   play_<s>_timeline.png (the whole play as rows over time), play_<s>_segments.png (one row per stretch),
//   play_<s>_log.json (every frame, press and event, and the stretches' summaries)
const FPS = 30;
const ENGAGE_IN = 2.2, ENGAGE_OUT = 3.0; // m between the hips: in close range below, apart above
const STATE = { free: 0, guard: 1, attack: 2, hit: 3, getup: 4, down: 4.5, fall: 5, out: 5 };
const PART = { head: 1, body: 2, legs: 3 };
const PHASE = { approach: 1, circle: 2, lunge: 3, feint: 3.5, retreat: 4 };

export function createPlayLog(ctx) {
  const sid = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const local = /^(127\.0\.0\.1|localhost)$/.test(location.hostname);
  const frames = [], events = [], segments = [], presses = [], pressById = new Map();
  let step = 0, seg = null, lastHit = null, lastAttack = null, lastSave = 0, engaged = false, pressSeq = 0;
  const prevState = new WeakMap(), downs = new WeakMap(), watches = [];
  let mousePrev = null, mouseSpeed = 0, cursorPrev = null;
  const cursorV = { x: 0, z: 0 };

  const ev = (type, data = {}) => { const e = { t: +ctx.clock.toFixed(3), f: frames.length, type, ...data }; events.push(e); if (seg) seg.events.push(e); return e; };
  function newSegment(kind) {
    if (seg) closeSegment();
    seg = { k: segments.length, kind, t0: ctx.clock, f0: frames.length, events: [] };
  }
  function closeSegment() {
    const s = seg; seg = null;
    s.t1 = ctx.clock; s.f1 = frames.length;
    Object.assign(s, summarize(s));
    delete s.events;
    segments.push(s);
  }

  // ---------------------------------------------------------------- the measures of a stretch
  const med = a => { const v = a.filter(x => x != null && Number.isFinite(x)).sort((x, y) => x - y); return v.length ? +v[v.length >> 1].toFixed(2) : null; };
  const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  function follow(F) {
    // the hips' velocity against the cursor's (on the floor): the lag that lines them up best, how well, and how
    // much of the cursor's speed the hips take on
    const n = F.length;
    if (n < 20) return { lag: null, corr: null, gain: null };
    let best = { L: 0, c: -1 };
    for (let L = 0; L <= 24; L++) {
      let su = 0, sv = 0, uv = 0;
      for (let t = 0; t + L < n; t++) { const a = F[t], b = F[t + L]; su += a.cvx ** 2 + a.cvz ** 2; sv += b.vx ** 2 + b.vz ** 2; uv += a.cvx * b.vx + a.cvz * b.vz; }
      const c = uv / Math.sqrt(Math.max(1e-9, su * sv));
      if (c > best.c) best = { L, c };
    }
    let num = 0, den = 0;
    for (let t = 0; t + best.L < n; t++) { const a = F[t], b = F[t + best.L]; const u = Math.hypot(a.cvx, a.cvz); if (u > 0.3) { num += Math.min(Math.hypot(b.vx, b.vz), 6); den += Math.min(u, 6); } }
    return { lag: +(best.L / FPS).toFixed(2), corr: +best.c.toFixed(2), gain: den ? +(num / den).toFixed(2) : null };
  }
  function summarize(s) {
    const F = frames.slice(s.f0, s.f1), E = s.events, n = F.length;
    const busy = new Uint8Array(n);
    const mark = (f, w = Math.round(0.6 * FPS)) => { for (let i = Math.max(0, f - s.f0 - 3); i < Math.min(n, f - s.f0 + w); i++) busy[i] = 1; };
    for (const e of E) if (['press', 'strike', 'hit', 'taken', 'enemyStrike', 'down', 'ko'].includes(e.type)) mark(e.f);
    F.forEach((fr, i) => { if (fr.hv > 0.8 || fr.ps >= 2) busy[i] = 1; });
    let flat = 0; const lulls = []; let run = 0;
    F.forEach((fr, i) => {
      const dull = fr.near && !busy[i];
      fr.dull = dull ? 1 : 0;
      if (dull) { flat++; run++; } else { if (run >= FPS) lulls.push(+(run / FPS).toFixed(2)); run = 0; }
    });
    if (run >= FPS) lulls.push(+(run / FPS).toFixed(2));
    const near = F.filter(fr => fr.near).length;
    const sec = n / FPS, given = E.filter(e => e.type === 'hit'), taken = E.filter(e => e.type === 'taken');
    const P = presses.filter(p => p.f >= s.f0 && p.f < s.f1);
    const out = o => P.filter(p => (p.outcome ?? '').startsWith(o)).length;
    const lost = P.filter(p => /^(expired|lost|refused|replaced|merged|no-)/.test(p.outcome ?? 'none') || p.outcome == null).length;
    const fol = follow(F);
    const reacts = [...given, ...taken].filter(e => !e.touch && !e.blocked);
    const fallsHere = reacts.filter(e => e.fall);
    const downEnds = E.filter(e => e.type === 'downEnd');
    return {
      sec: +sec.toFixed(1), nearSec: +(near / FPS).toFixed(1),
      flatFrac: near ? +(flat / near).toFixed(2) : null, longestLull: lulls.length ? Math.max(...lulls) : 0, lulls,
      eventsPerMin: +(E.filter(e => e.type !== 'press').length / Math.max(sec, 1e-3) * 60).toFixed(1),
      // moving
      followLag: fol.lag, followCorr: fol.corr, followGain: fol.gain,
      stretch: n ? +mean(F.map(fr => fr.st)).toFixed(2) : null,
      // pressing
      punch: P.filter(p => p.kind === 'punch').length, kick: P.filter(p => p.kind === 'kick').length,
      pressShown: P.length ? +(P.filter(p => p.tStrike != null).length / P.length).toFixed(2) : null,
      pressLost: lost, pressQueued: out('queued') + out('chained'),
      pressToStrike: med(P.map(p => p.tStrike)), pressToContact: med(P.map(p => p.tContact)),
      strikes: E.filter(e => e.type === 'strike').length,
      // the fight
      hits: given.filter(e => !e.touch && !e.blocked).length, touches: given.filter(e => e.touch).length, blocked: given.filter(e => e.blocked).length,
      slams: given.filter(e => e.slam).length, givenJ: Math.round(given.reduce((a, e) => a + e.energy, 0)),
      taken: taken.length, takenJ: Math.round(taken.reduce((a, e) => a + e.energy, 0)),
      enemyStrikes: E.filter(e => e.type === 'enemyStrike').length,
      enemyLanded: E.filter(e => e.type === 'enemyStrike').length ? +(taken.length / E.filter(e => e.type === 'enemyStrike').length).toFixed(2) : null,
      enemyStunned: n ? +(F.filter(fr => fr.es >= 3).length / n).toFixed(2) : null,
      enemyAttackerNear: n ? +(F.filter(fr => fr.att && fr.near).length / Math.max(1, near)).toFixed(2) : null,
      // blows and falls
      reactDelay: med(reacts.map(e => e.react)), knockDelay: med(reacts.map(e => e.knock)),
      reactDisp: med(reacts.map(e => e.disp06)),
      reactSpread: reacts.length > 2 ? +Math.sqrt(mean(reacts.map(e => (e.disp06 ?? 0) ** 2)) - (mean(reacts.map(e => e.disp06 ?? 0)) ** 2)).toFixed(2) : null,
      falls: fallsHere.length, fallKinds: new Set(fallsHere.map(e => e.fall.name)).size,
      downStill: downEnds.length ? +mean(downEnds.map(e => e.still)).toFixed(2) : null,
      downMotion: downEnds.length ? +mean(downEnds.map(e => e.motion)).toFixed(2) : null,
      meanDist: n ? +(F.reduce((a, fr) => a + fr.d, 0) / n).toFixed(2) : null,
      cursorSpeed: n ? Math.round(F.reduce((a, fr) => a + fr.ms, 0) / n) : 0,
    };
  }

  // ---------------------------------------------------------------- every step
  const limbSpeed = (F, dt) => {
    if (!F.prevP) return 0;
    let s = 0;
    for (const n of ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head']) { const j = ctx.I[n]; s += F.P[j].distanceTo(F.prevP[j]) / dt; }
    return s / 5;
  };
  function stepLog(dt) {
    const P = ctx.player;
    if (!P) return;
    const clock = ctx.clock;
    const m = ctx.mouse;
    if (m.seen && mousePrev) mouseSpeed = 0.8 * mouseSpeed + 0.2 * Math.hypot(m.x - mousePrev.x, m.y - mousePrev.y) / dt;
    mousePrev = m.seen ? { x: m.x, y: m.y } : null;
    // the cursor's point on the floor and its velocity (smoothed)
    const c = ctx.cursorFloor?.();
    if (c && cursorPrev) { cursorV.x = 0.75 * cursorV.x + 0.25 * (c.x - cursorPrev.x) / dt; cursorV.z = 0.75 * cursorV.z + 0.25 * (c.z - cursorPrev.z) / dt; }
    else { cursorV.x *= 0.8; cursorV.z *= 0.8; }
    cursorPrev = c ? c.clone() : null;
    // blows, strikes, attacks, falls
    const lh = ctx.lastHit;
    if (lh && lh !== lastHit) {
      lastHit = lh;
      const e = ev(lh.by === 'あなた' ? 'hit' : 'taken', { energy: +lh.energy.toFixed(1), tier: lh.tier, touch: !!lh.touch, slam: !!lh.slam, blocked: !!lh.blocked, head: !!lh.isHead, victim: lh.victim, pressId: lh.pressId || null });
      const pr = lh.pressId && pressById.get(lh.pressId);
      if (pr && lh.by === 'あなた' && pr.tContact == null) { pr.tContact = +(clock - pr.t).toFixed(2); pr.energy = e.energy; pr.tier = lh.tier; }
      const V = ctx.fighters.find(f => f.id === lh.victim);
      if (V && !lh.touch && !lh.blocked) watches.push({ e, V, h0: V.P[0].clone(), t0: clock });
    }
    for (let k = watches.length - 1; k >= 0; k--) {
      const w = watches[k], t = clock - w.t0, d = Math.hypot(w.V.P[0].x - w.h0.x, w.V.P[0].z - w.h0.z) + Math.abs(w.V.P[0].y - w.h0.y);
      if (w.e.react == null && d > 0.04) w.e.react = +t.toFixed(2);
      if (w.e.knock == null && d > 0.25) w.e.knock = +t.toFixed(2); // (the body really carried off)
      if (w.e.disp03 == null && t >= 0.3) w.e.disp03 = +d.toFixed(2);
      if (w.e.disp06 == null && t >= 0.6) w.e.disp06 = +d.toFixed(2);
      if (!w.e.fall && w.V.lastFall && w.V.lastFall.t >= w.t0 - 0.05) w.e.fall = { ...w.V.lastFall };
      if (t > 1.5) watches.splice(k, 1);
    }
    if (P.attack && P.attack !== lastAttack) {
      lastAttack = P.attack;
      ev('strike', { kind: P.attack.kind, family: P.attack.key?.family, name: P.attack.key?.name, ready: P.attack.ready != null ? +P.attack.ready.toFixed(2) : null, aim: P.attack.aimPart ?? null, pressId: P.attack.pressId || null });
      const pr = P.attack.pressId && pressById.get(P.attack.pressId);
      if (pr && pr.tStrike == null) { pr.tStrike = +(clock - pr.t).toFixed(2); pr.family = P.attack.key?.family; }
    }
    for (const F of ctx.fighters) {
      const was = prevState.get(F);
      if (was !== F.state) {
        if (!F.isPlayer && F.state === 'attack') ev('enemyStrike', { d: +F.P[0].distanceTo(P.P[0]).toFixed(2), who: F.id });
        if (F.state === 'fall') ev('down', { who: F.isPlayer ? 'P' : F.id });
        if (F.state === 'out') ev('ko', { who: F.id });
        if (F.state === 'down') downs.set(F, { t0: clock, n: 0, sum: 0, still: 0 });
        if (was === 'down' && downs.get(F)) {
          const D = downs.get(F);
          ev('downEnd', { who: F.isPlayer ? 'P' : F.id, sec: +(clock - D.t0).toFixed(2), motion: +(D.sum / Math.max(1, D.n)).toFixed(3), still: +(D.still / Math.max(1, D.n)).toFixed(2), next: F.state });
          downs.delete(F);
        }
        prevState.set(F, F.state);
      }
      const D = downs.get(F);
      if (D && F.state === 'down') { const sp = limbSpeed(F, dt); D.n++; D.sum += sp; if (sp < 0.05) D.still++; }
    }
    if (++step % 2) return; // 30 fps
    const es = ctx.fighters.filter(f => !f.isPlayer && f.state !== 'out');
    let near = null, d = 9;
    for (const e of es) { const x = Math.hypot(e.P[0].x - P.P[0].x, e.P[0].z - P.P[0].z); if (x < d) { d = x; near = e; } }
    const vx = P.prevP ? (P.P[0].x - P.prevP[0].x) / dt : 0, vz = P.prevP ? (P.P[0].z - P.prevP[0].z) / dt : 0;
    const tgt = P.spring?.target;
    // stretches: close range in / out (with some hysteresis); long ones cut at 40 s
    const nowNear = engaged ? d < ENGAGE_OUT : d < ENGAGE_IN;
    if (!seg) newSegment(nowNear ? 'near' : 'apart');
    else if (nowNear !== engaged) newSegment(nowNear ? 'near' : 'apart');
    else if (clock - seg.t0 > 40) newSegment(seg.kind);
    engaged = nowNear;
    frames.push({
      t: +clock.toFixed(3), mx: m.seen ? Math.round(m.x) : null, my: m.seen ? Math.round(m.y) : null, ms: Math.round(mouseSpeed),
      cx: c ? +c.x.toFixed(2) : null, cz: c ? +c.z.toFixed(2) : null, cvx: +cursorV.x.toFixed(2), cvz: +cursorV.z.toFixed(2),
      tx: tgt ? +tgt.x.toFixed(2) : null, tz: tgt ? +tgt.z.toFixed(2) : null, st: tgt ? +Math.hypot(tgt.x - P.P[0].x, tgt.z - P.P[0].z).toFixed(2) : 0,
      aim: P.aim ? PART[P.aim.part] : 0, x: +P.P[0].x.toFixed(2), z: +P.P[0].z.toFixed(2), vx: +vx.toFixed(2), vz: +vz.toFixed(2), hv: +Math.hypot(vx, vz).toFixed(2),
      d: +d.toFixed(2), near: nowNear ? 1 : 0, ps: STATE[P.state] ?? 0, es: near ? STATE[near.state] ?? 0 : 0,
      hp: Math.round(P.hp), ehp: near ? Math.round(near.hp) : 0, att: ctx.crew.attacker === near ? 1 : 0, eph: near && ctx.crew.attacker === near ? PHASE[near.phase] ?? 0 : 0,
      el: near ? +limbSpeed(near, dt).toFixed(2) : 0,
    });
    if (local && clock - lastSave > 20) { lastSave = clock; save(); }
  }

  // ---------------------------------------------------------------- pictures
  const heat = v => { const x = Math.min(1, Math.max(0, v) / 5); return `rgb(${Math.round(255 * Math.min(1, Math.max(0, 1.8 * x - 0.2)))},${Math.round(255 * Math.max(0, Math.min(1, 1.6 * x + 0.1) - Math.max(0, 2.2 * (x - 0.6))))},${Math.round(255 * Math.max(0, 0.55 - x))})`; };
  const OUTCOME = o => (o == null ? 5 : o === 'started' ? 1 : o === 'chained' ? 1.8 : o === 'queued' ? 2.6 : /^after:started/.test(o) ? 2.2 : 5);
  function timeline() {
    const all = segments.concat(seg ? [{ ...seg, ...summarize(seg), f1: frames.length }] : []);
    const bin = 3; // 0.1 s per column
    const n = Math.ceil(frames.length / bin);
    const col = f => { const out = new Float32Array(n); for (let c = 0; c < n; c++) { let v = 0; for (let i = c * bin; i < Math.min(frames.length, c * bin + bin); i++) v = Math.max(v, f(frames[i], i)); out[c] = v; } return out; };
    const evRow = (type, val = () => 5, cond = () => true) => { const out = new Float32Array(n); for (const e of events) if (e.type === type && cond(e)) { const c = Math.min(n - 1, Math.floor(e.f / bin)); out[c] = Math.max(out[c], val(e)); } return out; };
    const pressRow = kind => { const out = new Float32Array(n); for (const p of presses) if (p.kind === kind) { const c = Math.min(n - 1, Math.floor(p.f / bin)); out[c] = Math.max(out[c], OUTCOME(p.outcome)); } return out; };
    const rows = [
      ['入力: カーソルの速さ（画面）', col(fr => fr.ms / 400)],
      ['入力: カーソルの速さ（床）', col(fr => Math.hypot(fr.cvx, fr.cvz) / 0.8)],
      ['入力: 狙い（頭/胴/脚）', col(fr => [0, 5, 3, 1.5][fr.aim])],
      ['入力: パンチ（青=出た 黄=予約 赤=消えた）', pressRow('punch')],
      ['入力: キック（同）', pressRow('kick')],
      ['腰の速さ', col(fr => fr.hv / 0.7)],
      ['腰の遅れ（ばねの伸び）', col(fr => fr.st / 0.5)],
      ['相手との近さ', col(fr => Math.max(0, 3 - fr.d) * 1.8)],
      ['近い（交戦中）', col(fr => fr.near * 2.5)],
      ['自分の状態', col(fr => fr.ps)],
      ['相手の状態', col(fr => fr.es)],
      ['相手が動けない（被弾・ダウン）', col(fr => (fr.es >= 3 ? 5 : 0))],
      ['攻め役の段階（近づく→回る→突っ込む→下がる）', col(fr => fr.eph * 1.2)],
      ['相手が攻める', evRow('enemyStrike')],
      ['自分の打撃', evRow('strike')],
      ['当てた J', evRow('hit', e => (e.touch ? 0.8 : e.energy / 25), e => !e.blocked)],
      ['受けた J', evRow('taken', e => e.energy / 25)],
      ['当たってから飛ぶまで（遅いほど赤）', evRow('hit', e => (e.knock == null ? 0 : (e.knock - 0.1) * 12), e => !e.touch && !e.blocked)],
      ['体当たり', evRow('hit', () => 5, e => e.slam)],
      ['倒れている相手の動き（止まるほど赤）', col(fr => (fr.es === 4.5 ? Math.max(0, 5 - fr.el * 12) : 0))],
      ['メリハリなし（近いのに何も起きない）', col(fr => (fr.dull ? 5 : 0))],
      ['自分の体力（減るほど赤）', col(fr => (100 - fr.hp) / 20)],
    ];
    const left = 290, top = 46, rh = 14, cell = Math.max(1, Math.min(4, Math.floor(2400 / Math.max(1, n))));
    const cv = document.createElement('canvas');
    cv.width = left + n * cell + 20; cv.height = top + rows.length * rh + 40;
    const g = cv.getContext('2d');
    g.fillStyle = '#111'; g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#ddd'; g.font = '13px sans-serif';
    const tot = summarize({ f0: 0, f1: frames.length, events, kind: 'all' });
    g.fillText(`プレイ ${sid}  ${(frames.length / FPS).toFixed(0)} 秒  交戦 ${tot.nearSec} 秒（メリハリなし ${Math.round((tot.flatFrac ?? 0) * 100)}%）  押した ${tot.punch + tot.kick}→出た ${Math.round((tot.pressShown ?? 0) * 100)}%（押→打撃 ${tot.pressToStrike}s・当たり ${tot.pressToContact}s）  腰の追従 遅れ ${tot.followLag}s 強さ ${tot.followGain}  相手の攻め ${tot.enemyStrikes} 回→当たり ${tot.taken}  相手が動けない ${Math.round((tot.enemyStunned ?? 0) * 100)}%  揺れ ${tot.reactDelay}s・飛ぶ ${tot.knockDelay}s  倒れて止まる ${tot.downStill ?? '-'}`, 10, 18);
    rows.forEach(([name, z], r) => {
      const y = top + r * rh;
      g.fillStyle = '#bbb'; g.font = '10px sans-serif'; g.fillText(name, 6, y + rh - 3);
      for (let x = 0; x < n; x++) { g.fillStyle = heat(z[x]); g.fillRect(left + x * cell, y, cell, rh - 1); }
    });
    for (const s of all) {
      const x = left + Math.floor(s.f0 / bin) * cell;
      g.fillStyle = 'rgba(255,255,255,0.55)'; g.fillRect(x, top - 8, 1, rows.length * rh + 8);
      g.fillStyle = s.kind === 'near' ? '#ffcf70' : '#8ab4ff'; g.font = '9px sans-serif'; g.fillText(`${s.k}${s.kind === 'near' ? '近' : '離'}`, x + 2, top - 10);
    }
    g.fillStyle = '#888'; g.font = '9px sans-serif';
    for (let sec = 0; sec * FPS < frames.length; sec += 5) g.fillText(`${sec}s`, left + Math.floor(sec * FPS / bin) * cell, top + rows.length * rh + 12);
    return cv;
  }
  // columns: key, label, badness 0..5 (only for close-range stretches) or null
  const SEG_COLS = [
    ['sec', '長さ s', null], ['nearSec', '交戦 s', null],
    ['flatFrac', 'メリハリなし', v => v * 8], ['longestLull', '最長の間 s', v => (v - 1) * 1.6], ['eventsPerMin', '出来事/分', v => (40 - v) / 7],
    ['followLag', '腰の遅れ s', v => (v - 0.1) * 12], ['followGain', '勢いの乗り', v => (1 - v) * 6], ['stretch', 'ばねの伸び m', v => (v - 0.4) * 8],
    ['punch', 'パンチ', null], ['kick', 'キック', null], ['pressShown', '押→出た率', v => (1 - v) * 8], ['pressLost', '消えた押し', v => v * 0.8],
    ['pressToStrike', '押→打撃 s', v => (v - 0.15) * 14], ['pressToContact', '押→当たり s', v => (v - 0.35) * 10],
    ['hits', '当てた', null], ['touches', 'かすり', null], ['slams', '体当たり', null], ['givenJ', '当てた J', null],
    ['enemyStrikes', '相手の攻め', v => (3 - v) * 1.2], ['taken', '受けた', null], ['enemyLanded', '攻め→当たり', v => (0.4 - v) * 10],
    ['enemyStunned', '相手が動けない', v => (v - 0.25) * 10],
    ['reactDelay', '揺れるまで s', v => (v - 0.05) * 30], ['knockDelay', '飛ぶまで s', v => (v - 0.15) * 12], ['reactSpread', '反応のばらつき m', v => (0.25 - v) * 15], ['fallKinds', '倒れ方の種類', null],
    ['downStill', '倒れて止まる率', v => v * 8], ['meanDist', '平均距離 m', null],
  ];
  function segmentTable() {
    const all = segments.concat(seg ? [{ ...seg, ...summarize(seg) }] : []);
    const total = { k: '計', kind: 'near', t0: 0, ...summarize({ f0: 0, f1: frames.length, events }) };
    const rowsT = [...all, total];
    const left = 120, top = 50, cw = 66, rh = 20;
    const cv = document.createElement('canvas');
    cv.width = left + SEG_COLS.length * cw + 10; cv.height = top + rowsT.length * rh + 20;
    const g = cv.getContext('2d');
    g.fillStyle = '#111'; g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#ddd'; g.font = '13px sans-serif'; g.fillText(`プレイ ${sid} の区切りごと（近＝交戦、離＝離れている。最後の行は全体。色は悪いほど赤）`, 10, 18);
    g.font = '9px sans-serif';
    SEG_COLS.forEach(([, label], i) => { g.fillStyle = '#ccc'; g.fillText(label, left + i * cw + 3, top - 8); });
    g.font = '10px sans-serif';
    rowsT.forEach((s, r) => {
      const y = top + r * rh;
      g.fillStyle = s.k === '計' ? '#fff' : s.kind === 'near' ? '#ffcf70' : '#8ab4ff';
      g.fillText(s.k === '計' ? '全体' : `${s.k} ${s.kind === 'near' ? '近' : '離'} ${s.t0.toFixed(0)}s`, 8, y + 14);
      SEG_COLS.forEach(([key, , bad], i) => {
        const v = s[key];
        const b = bad && v != null && s.kind === 'near' ? Math.max(0, Math.min(5, bad(v))) : null;
        g.fillStyle = b == null ? '#2a2a2a' : heat(b); g.fillRect(left + i * cw, y, cw - 2, rh - 2);
        g.fillStyle = b != null && b > 2.2 && b < 4.2 ? '#000' : '#fff';
        g.fillText(v == null ? '-' : String(v), left + i * cw + 4, y + 14);
      });
    });
    return cv;
  }
  const png = cv => new Promise(r => cv.toBlob(r, 'image/png'));
  function json() { return JSON.stringify({ sid, date: new Date().toISOString(), fps: FPS, total: summarize({ f0: 0, f1: frames.length, events }), segments: segments.concat(seg ? [{ ...seg, ...summarize(seg), events: undefined }] : []), presses, events, frames }); }
  async function save() {
    if (!local) return;
    try {
      await fetch(`/save/play_${sid}_log.json`, { method: 'POST', body: json() });
      await fetch(`/save/play_${sid}_timeline.png`, { method: 'POST', body: await png(timeline()) });
      await fetch(`/save/play_${sid}_segments.png`, { method: 'POST', body: await png(segmentTable()) });
    } catch { /* the page works on without the local server */ }
  }
  async function download() {
    const a = document.createElement('a');
    for (const [name, blob] of [[`play_${sid}_log.json`, new Blob([json()], { type: 'application/json' })], [`play_${sid}_timeline.png`, await png(timeline())], [`play_${sid}_segments.png`, await png(segmentTable())]]) {
      a.href = URL.createObjectURL(blob); a.download = name; a.click(); URL.revokeObjectURL(a.href);
    }
  }
  function press(kind) {
    const p = { id: ++pressSeq, kind, t: +ctx.clock.toFixed(3), f: frames.length, outcome: null };
    presses.push(p); pressById.set(p.id, p);
    ev('press', { kind, id: p.id });
    return p.id;
  }
  function pressResult(id, outcome) {
    const p = pressById.get(id);
    if (!p) return;
    // (the first outcome is what the press got; a kept press's later fate is noted after it)
    if (p.outcome == null || p.outcome === 'queued') { p.outcome = outcome; p.tOutcome = +(ctx.clock - p.t).toFixed(2); }
    else p.later = outcome;
  }
  addEventListener('pagehide', () => { if (local) navigator.sendBeacon?.(`/save/play_${sid}_log.json`, json()); });
  return { step: stepLog, press, pressResult, save, download, get sid() { return sid; }, get local() { return local; }, summary: () => summarize({ f0: 0, f1: frames.length, events }), segments, frames, events, presses };
}
