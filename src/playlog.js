// SOMA Brawl play log: what a person does and what comes of it, recorded all through play (30 fps), cut into
// stretches (closing in / fighting at close range / apart, and at falls and knock-outs), each summed up. The point
// is to see the play as it is felt: above all the "flat" time at close range - near an opponent, yet nothing
// happening (no press, no strike, no hit either way, no opponent's attack, the body hardly moving).
// Saved while playing to eval_out/play_<session>_* (tools/serve.mjs, local only), or downloaded with the button:
//   play_<s>_timeline.png   the whole play as rows over time (inputs, distances, states, blows, flat time)
//   play_<s>_segments.png   one row per stretch: its measures as colours (blue good .. red bad)
//   play_<s>_log.json       every frame and event, and the stretches' summaries
const FPS = 30;
const ENGAGE_IN = 2.2, ENGAGE_OUT = 3.0; // m between the hips: in close range below, apart above
const STATE = { free: 0, guard: 1, attack: 2, hit: 3, getup: 4, down: 4.5, fall: 5, out: 5 };
const PART = { head: 1, body: 2, legs: 3 };

export function createPlayLog(ctx) {
  const sid = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const local = /^(127\.0\.0\.1|localhost)$/.test(location.hostname);
  const frames = [], events = [], segments = [];
  let step = 0, seg = null, lastHit = null, lastAttack = null, lastSave = 0, engaged = false;
  const prevState = new WeakMap();
  let mousePrev = null, mouseSpeed = 0;

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
  // a frame is "lively" within 0.6 s of anything happening, or while the body is really moving
  function summarize(s) {
    const F = frames.slice(s.f0, s.f1), E = s.events, n = F.length;
    const busy = new Uint8Array(n);
    const mark = (f, w = Math.round(0.6 * FPS)) => { for (let i = Math.max(0, f - s.f0 - 3); i < Math.min(n, f - s.f0 + w); i++) busy[i] = 1; };
    for (const e of E) if (['press', 'strike', 'hit', 'taken', 'enemyStrike', 'slam', 'down', 'ko'].includes(e.type)) mark(e.f);
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
    const presses = E.filter(e => e.type === 'press');
    return {
      sec: +sec.toFixed(1), nearSec: +(near / FPS).toFixed(1),
      flatFrac: near ? +(flat / near).toFixed(2) : null, longestLull: lulls.length ? Math.max(...lulls) : 0, lulls,
      punch: presses.filter(e => e.kind === 'punch').length, kick: presses.filter(e => e.kind === 'kick').length,
      strikes: E.filter(e => e.type === 'strike').length,
      hits: given.filter(e => !e.touch && !e.blocked).length, touches: given.filter(e => e.touch).length, blocked: given.filter(e => e.blocked).length,
      slams: given.filter(e => e.slam).length, givenJ: Math.round(given.reduce((a, e) => a + e.energy, 0)),
      taken: taken.length, takenJ: Math.round(taken.reduce((a, e) => a + e.energy, 0)),
      enemyStrikes: E.filter(e => e.type === 'enemyStrike').length,
      eventsPerMin: +(E.filter(e => e.type !== 'press').length / Math.max(sec, 1e-3) * 60).toFixed(1),
      meanDist: n ? +(F.reduce((a, fr) => a + fr.d, 0) / n).toFixed(2) : null,
      cursorSpeed: n ? Math.round(F.reduce((a, fr) => a + fr.ms, 0) / n) : 0,
    };
  }

  function stepLog(dt) {
    const P = ctx.player;
    if (!P) return;
    const clock = ctx.clock;
    // the cursor's speed on screen (px/s)
    const m = ctx.mouse;
    if (m.seen && mousePrev) mouseSpeed = 0.8 * mouseSpeed + 0.2 * Math.hypot(m.x - mousePrev.x, m.y - mousePrev.y) / dt;
    mousePrev = m.seen ? { x: m.x, y: m.y } : null;
    // events: blows (from the last hit record), the player's strikes, the opponents' strikes, falls, knock-outs
    const lh = ctx.lastHit;
    if (lh && lh !== lastHit) {
      lastHit = lh;
      ev(lh.by === 'あなた' ? 'hit' : 'taken', { energy: +lh.energy.toFixed(1), touch: !!lh.touch, slam: !!lh.slam, blocked: !!lh.blocked, head: !!lh.isHead });
    }
    if (P.attack && P.attack !== lastAttack) { lastAttack = P.attack; ev('strike', { kind: P.attack.kind, family: P.attack.key?.family, name: P.attack.key?.name, ready: P.attack.ready != null ? +P.attack.ready.toFixed(2) : null }); }
    for (const F of ctx.fighters) {
      const was = prevState.get(F);
      if (was !== F.state) {
        if (!F.isPlayer && F.state === 'attack') ev('enemyStrike', { d: +F.P[0].distanceTo(P.P[0]).toFixed(2) });
        if (F.state === 'fall') ev('down', { who: F.isPlayer ? 'P' : 'E' });
        if (F.state === 'out') ev('ko');
        prevState.set(F, F.state);
      }
    }
    if (++step % 2) return; // 30 fps
    const es = ctx.fighters.filter(f => !f.isPlayer && f.state !== 'out');
    let near = null, d = 9;
    for (const e of es) { const x = Math.hypot(e.P[0].x - P.P[0].x, e.P[0].z - P.P[0].z); if (x < d) { d = x; near = e; } }
    const hv = P.prevP ? Math.hypot(P.P[0].x - P.prevP[0].x, P.P[0].z - P.prevP[0].z) / dt : 0;
    // stretches: close range in / out (with some hysteresis), a fall, a knock-out
    const nowNear = engaged ? d < ENGAGE_OUT : d < ENGAGE_IN;
    if (!seg) newSegment(nowNear ? 'near' : 'apart');
    else if (nowNear !== engaged) newSegment(nowNear ? 'near' : 'apart');
    else if (clock - seg.t0 > 40) newSegment(seg.kind); // (long ones are cut too)
    engaged = nowNear;
    frames.push({
      t: +clock.toFixed(3), mx: m.seen ? Math.round(m.x) : null, my: m.seen ? Math.round(m.y) : null, ms: Math.round(mouseSpeed),
      aim: P.aim ? PART[P.aim.part] : 0, x: +P.P[0].x.toFixed(2), z: +P.P[0].z.toFixed(2), hv: +hv.toFixed(2),
      d: +d.toFixed(2), near: nowNear ? 1 : 0, ps: STATE[P.state] ?? 0, es: near ? STATE[near.state] ?? 0 : 0,
      hp: Math.round(P.hp), ehp: near ? Math.round(near.hp) : 0, att: ctx.crew.attacker === near ? 1 : 0,
    });
    if (local && clock - lastSave > 20) { lastSave = clock; save(); }
  }

  // ---------------------------------------------------------------- pictures
  const heat = v => { const x = Math.min(1, Math.max(0, v) / 5); return `rgb(${Math.round(255 * Math.min(1, Math.max(0, 1.8 * x - 0.2)))},${Math.round(255 * Math.max(0, Math.min(1, 1.6 * x + 0.1) - Math.max(0, 2.2 * (x - 0.6))))},${Math.round(255 * Math.max(0, 0.55 - x))})`; };
  function timeline() {
    const all = segments.concat(seg ? [{ ...seg, ...summarize(seg), f1: frames.length }] : []);
    // (flat-time marks for the frames of every stretch)
    const bin = 3; // 0.1 s per column
    const n = Math.ceil(frames.length / bin);
    const col = (f) => { const out = new Float32Array(n); for (let c = 0; c < n; c++) { let v = 0; for (let i = c * bin; i < Math.min(frames.length, c * bin + bin); i++) v = Math.max(v, f(frames[i], i)); out[c] = v; } return out; };
    const evRow = (type, val = () => 5, cond = () => true) => { const out = new Float32Array(n); for (const e of events) if (e.type === type && cond(e)) { const c = Math.min(n - 1, Math.floor(e.f / bin)); out[c] = Math.max(out[c], val(e)); } return out; };
    const rows = [
      ['入力: カーソルの速さ', col(fr => fr.ms / 400)],
      ['入力: 狙い（頭/胴/脚）', col(fr => [0, 5, 3, 1.5][fr.aim])],
      ['入力: パンチ', evRow('press', () => 5, e => e.kind === 'punch')],
      ['入力: キック', evRow('press', () => 5, e => e.kind === 'kick')],
      ['相手との近さ', col(fr => Math.max(0, 3 - fr.d) * 1.8)],
      ['近い（交戦中）', col(fr => fr.near * 2.5)],
      ['腰の速さ', col(fr => fr.hv / 0.7)],
      ['自分の状態', col(fr => fr.ps)],
      ['相手の状態', col(fr => fr.es)],
      ['相手が攻める', evRow('enemyStrike')],
      ['自分の打撃', evRow('strike')],
      ['当てた J', evRow('hit', e => (e.touch ? 0.8 : e.energy / 25), e => !e.blocked)],
      ['受けた J', evRow('taken', e => e.energy / 25)],
      ['体当たり', evRow('hit', () => 5, e => e.slam)],
      ['メリハリなし（近いのに何も起きない）', col(fr => (fr.dull ? 5 : 0))],
      ['自分の体力（減るほど赤）', col(fr => (100 - fr.hp) / 20)],
    ];
    const left = 230, top = 46, rh = 14, cell = Math.max(1, Math.min(4, Math.floor(2400 / Math.max(1, n))));
    const c = document.createElement('canvas');
    c.width = left + n * cell + 20; c.height = top + rows.length * rh + 40;
    const g = c.getContext('2d');
    g.fillStyle = '#111'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#ddd'; g.font = '13px sans-serif';
    const tot = summarize({ f0: 0, f1: frames.length, events, kind: 'all' });
    g.fillText(`プレイ ${sid}  ${(frames.length / FPS).toFixed(0)} 秒  交戦 ${tot.nearSec} 秒・うちメリハリなし ${Math.round((tot.flatFrac ?? 0) * 100)}%（最長 ${tot.longestLull} 秒）  当てた ${tot.hits}（かすり ${tot.touches}）・受けた ${tot.taken}  出来事 ${tot.eventsPerMin}/分`, 10, 18);
    rows.forEach(([name, z], r) => {
      const y = top + r * rh;
      g.fillStyle = '#bbb'; g.font = '10px sans-serif'; g.fillText(name, 6, y + rh - 3);
      for (let x = 0; x < n; x++) { g.fillStyle = heat(z[x]); g.fillRect(left + x * cell, y, cell, rh - 1); }
    });
    // stretches: boundaries and their kind
    for (const s of all) {
      const x = left + Math.floor(s.f0 / bin) * cell;
      g.fillStyle = 'rgba(255,255,255,0.55)'; g.fillRect(x, top - 8, 1, rows.length * rh + 8);
      g.fillStyle = s.kind === 'near' ? '#ffcf70' : '#8ab4ff'; g.font = '9px sans-serif'; g.fillText(`${s.k}${s.kind === 'near' ? '近' : '離'}`, x + 2, top - 10);
    }
    g.fillStyle = '#888'; g.font = '9px sans-serif';
    for (let sec = 0; sec * FPS < frames.length; sec += 5) g.fillText(`${sec}s`, left + Math.floor(sec * FPS / bin) * cell, top + rows.length * rh + 12);
    return c;
  }
  const SEG_COLS = [
    ['sec', '長さ s', null], ['nearSec', '交戦 s', null], ['flatFrac', 'メリハリなし', v => v * 8], ['longestLull', '最長の間 s', v => (v - 1) * 1.6],
    ['eventsPerMin', '出来事/分', v => (40 - v) / 7], ['punch', 'パンチ', null], ['kick', 'キック', null], ['strikes', '打撃', null],
    ['hits', '当てた', null], ['touches', 'かすり', null], ['slams', '体当たり', null], ['givenJ', '当てた J', null],
    ['taken', '受けた', null], ['takenJ', '受けた J', v => v / 60], ['enemyStrikes', '相手の攻め', null], ['meanDist', '平均距離 m', null], ['cursorSpeed', 'カーソル px/s', null],
  ];
  function segmentTable() {
    const all = segments.concat(seg ? [{ ...seg, ...summarize(seg) }] : []);
    const left = 120, top = 50, cw = 74, rh = 20;
    const c = document.createElement('canvas');
    c.width = left + SEG_COLS.length * cw + 10; c.height = top + all.length * rh + 20;
    const g = c.getContext('2d');
    g.fillStyle = '#111'; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#ddd'; g.font = '13px sans-serif'; g.fillText(`プレイ ${sid} の区切りごと（近＝交戦、離＝離れている。色は悪いほど赤）`, 10, 18);
    g.font = '10px sans-serif';
    SEG_COLS.forEach(([, label], i) => { g.fillStyle = '#ccc'; g.fillText(label, left + i * cw + 4, top - 8); });
    all.forEach((s, r) => {
      const y = top + r * rh;
      g.fillStyle = s.kind === 'near' ? '#ffcf70' : '#8ab4ff';
      g.fillText(`${s.k} ${s.kind === 'near' ? '近' : '離'} ${s.t0.toFixed(0)}s`, 8, y + 14);
      SEG_COLS.forEach(([key, , bad], i) => {
        const v = s[key];
        const b = bad && v != null && s.kind === 'near' ? bad(v) : null;
        g.fillStyle = b == null ? '#2a2a2a' : heat(b); g.fillRect(left + i * cw, y, cw - 2, rh - 2);
        g.fillStyle = b != null && b > 2.2 && b < 4.2 ? '#000' : '#fff';
        g.fillText(v == null ? '-' : String(v), left + i * cw + 4, y + 14);
      });
    });
    return c;
  }
  const png = c => new Promise(r => c.toBlob(r, 'image/png'));
  function json() { return JSON.stringify({ sid, date: new Date().toISOString(), fps: FPS, segments: segments.concat(seg ? [{ ...seg, ...summarize(seg), events: undefined }] : []), events, frames }); }
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
  function press(kind) { ev('press', { kind }); }
  addEventListener('pagehide', () => { if (local) navigator.sendBeacon?.(`/save/play_${sid}_log.json`, json()); });
  return { step: stepLog, press, save, download, get sid() { return sid; }, get local() { return local; }, summary: () => summarize({ f0: 0, f1: frames.length, events }), segments, frames, events };
}
