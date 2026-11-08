// A person-like player for tuning SOMA Brawl automatically: it plays through the real input (cursor moves and
// clicks dispatched on the canvas), the way a person tries to: closes in on the nearest opponent with the cursor on
// its head, punches when in range (a kick from a little out), and when an opponent it can SEE (in front of it,
// within 55 degrees) winds up, it steps out of the blow after a person's reaction time (a cursor flick to the side),
// then answers. Two kinds: a novice (slow, dodges seldom, presses slowly, aims loosely) and a skilled player.
// Its play is recorded by the play log like a person's; it adds its own tally of dodges and counters.
//   const ap = await import('./src/autoplay.js'); const r = await ap.series(brawl, { secs: 90, seeds: [1, 2, 3] })
const sleep = ms => new Promise(r => setTimeout(r, ms));

export const KINDS = {
  novice: { react: [0.3, 0.45], dodge: 0.3, pressGap: 0.8, jitter: 8 },
  skilled: { react: [0.2, 0.35], dodge: 0.6, pressGap: 0.55, jitter: 2 },
};

export async function run(brawl, seconds = 60, opt = {}) {
  const kind = KINDS[opt.kind ?? 'skilled'];
  const cv = document.querySelector('canvas'), R = cv.getBoundingClientRect();
  const V = brawl.player.P[0].constructor;
  const toScreen = v => { const q = v.clone().project(brawl.camera); return [R.left + (q.x + 1) / 2 * R.width, R.top + (1 - q.y) / 2 * R.height]; };
  const send = (type, button, buttons, x, y) => cv.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 1, button, buttons, clientX: x, clientY: y }));
  const rng = (() => { let s = (opt.seed ?? 1) * 2654435761 >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const I = brawl.rig.index;
  const facing = P => { const l = P.P[I.LeftArm], r = P.P[I.RightArm]; return new V(-(l.z - r.z), 0, l.x - r.x).normalize(); };
  const tally = { dodges: 0, hitInDodge: 0, counters: 0, counterHits: 0, presses: 0, seen: 0, unseen: 0 };
  const t0 = brawl.clock, f0 = brawl.plog.frameCount;
  let mode = 'press', modeUntil = 0, dodgeSide = 1, nextPress = 0, watching = null, lastHit = brawl.lastHit;
  const seen = new WeakMap();
  while (brawl.clock - t0 < seconds) {
    await sleep(33);
    const P = brawl.player, now = brawl.clock;
    const es = brawl.fighters.filter(f => !f.isPlayer && f.state !== 'out');
    if (!es.length) continue;
    const near = es.reduce((a, b) => (a.P[0].distanceTo(P.P[0]) < b.P[0].distanceTo(P.P[0]) ? a : b));
    const d = Math.hypot(near.P[0].x - P.P[0].x, near.P[0].z - P.P[0].z);
    // a blow starting: noticed only if in front (as a person sees), after a reaction time
    const fw = facing(P);
    for (const e of es) {
      // a warning at the feet (an opponent behind about to strike) is seen like a wind-up
      if ((e.warnUntil ?? -9) > now && seen.get(e) !== 'warn' + e.warnUntil) {
        seen.set(e, 'warn' + e.warnUntil);
        const react = kind.react[0] + (kind.react[1] - kind.react[0]) * rng();
        watching = { e, at: now + react + 0.15, will: kind.dodge > rng() }; tally.seen++;
        continue;
      }
      if (e.state !== 'attack' || e.attack === seen.get(e)) continue;
      seen.set(e, e.attack);
      const to = new V(e.P[0].x - P.P[0].x, 0, e.P[0].z - P.P[0].z), dd = to.length();
      if (dd > 1.9) continue;
      const inFront = fw.dot(to.normalize()) > Math.cos(55 * Math.PI / 180);
      if (inFront) tally.seen++; else { tally.unseen++; continue; }
      const react = kind.react[0] + (kind.react[1] - kind.react[0]) * rng();
      watching = { e, at: now + react, will: kind.dodge > rng() };
    }
    const lh = brawl.lastHit;
    if (lh && lh !== lastHit) { lastHit = lh; if (lh.by !== 'あなた' && mode === 'dodge') tally.hitInDodge++; if (lh.by === 'あなた' && mode === 'counter' && !lh.touch) tally.counterHits++; }
    if (watching && watching.will && now >= watching.at && mode !== 'dodge') {
      watching.will = false; mode = 'dodge'; modeUntil = now + 0.4; dodgeSide = rng() < 0.5 ? -1 : 1; tally.dodges++;
    }
    if (watching && now > watching.at + 0.9) watching = null;
    if (mode === 'dodge' && now > modeUntil) { mode = 'counter'; modeUntil = now + 0.9; tally.counters++; nextPress = now + 0.05; }
    if (mode === 'counter' && now > modeUntil) mode = 'press';
    // the cursor
    const hips = P.P[0];
    let [x, y] = toScreen(near.P[I.Head]);
    if (mode === 'dodge') {
      // a person's dodge: a quick flick of the hand sideways on the screen (across the line to the attacker), about 140 px
      const [px, py] = toScreen(hips), [ex, ey] = toScreen(watching?.e?.P[0] ?? near.P[0]);
      let ax = ex - px, ay = ey - py; const al = Math.hypot(ax, ay) || 1; ax /= al; ay /= al;
      x += -ay * 140 * dodgeSide - ax * 30; y += ax * 140 * dodgeSide - ay * 30;
    }
    x += (rng() - 0.5) * 2 * kind.jitter; y += (rng() - 0.5) * 2 * kind.jitter;
    send('pointermove', -1, 0, x, y);
    // presses
    if (mode !== 'dodge' && now > nextPress && d < (mode === 'counter' ? 1.8 : 1.5) && !['down', 'fall'].includes(P.state)) {
      const kick = d > 1.25 ? rng() < 0.6 : rng() < 0.15;
      send('pointerdown', kick ? 2 : 0, kick ? 2 : 1, x, y); send('pointerup', kick ? 2 : 0, 0, x, y);
      tally.presses++;
      nextPress = now + (mode === 'counter' ? 0.3 : kind.pressGap * (0.7 + 0.6 * rng()));
    }
  }
  const sum = brawl.plog.summaryFrom(f0);
  tally.dodgeOk = tally.dodges ? +((tally.dodges - tally.hitInDodge) / tally.dodges).toFixed(2) : null;
  tally.counterRate = tally.counters ? +(tally.counterHits / tally.counters).toFixed(2) : null;
  return { tally, sum };
}

/** The measures against their target bands: one number each (0 = in band, more = further out) and the sum. */
export function score(sum, tally, sec, kind = 'skilled') {
  const per = x => (x / sec) * 60;
  const band = (v, lo, hi, scale) => (v == null ? 1 : v < lo ? (lo - v) / scale : v > hi ? (v - hi) / scale : 0);
  const landed = kind === 'novice' ? [0.35, 0.55] : [0.25, 0.4];
  const parts = {
    enemyAttacks: band(per(sum.enemyStrikes), 28, 45, 8),
    enemyLanded: band(sum.enemyLanded, landed[0], landed[1], 0.1),
    stunned: band(sum.enemyStunned, 0, 0.32, 0.08),
    flat: band(sum.flatFrac, 0, 0.08, 0.05),
    counterRate: tally.counters >= 6 ? band(tally.counterRate, 0.4, 0.75, 0.15) : 0, // (fewer than 6 tries: noise)
    counters: kind === 'skilled' ? band(per(sum.counters ?? 0), 4, 12, 2) : 0,
    downStill: band(sum.downStill ?? 0, 0, 0.15, 0.08),
    playerStill: band(sum.playerStill, 0, 0.03, 0.02),
    unseen: band(sum.hitsUnseen ?? 0, 0, 0.2, 0.1),
    lull: band(sum.longestApart, 0, 4, 2),
    ko: band(per(sum.kos ?? 0), 1.5, 4, 1), // (a knock-out every 15-40 s: long enough to be a fight)
  };
  return { parts, total: +Object.values(parts).reduce((a, b) => a + b, 0).toFixed(2) };
}

/** A series: each kind of player x each seed, from a fresh fight each time; mean and worst. */
export async function series(brawl, { secs = 90, seeds = [1, 2, 3], kinds = ['novice', 'skilled'], onRun } = {}) {
  const runs = [];
  for (const kind of kinds) for (const seed of seeds) {
    brawl.reset();
    await sleep(2500); // (the fight forms again)
    const r = await run(brawl, secs, { kind, seed });
    const sc = score(r.sum, r.tally, secs, kind);
    const one = { kind, seed, total: sc.total, parts: sc.parts, tally: r.tally, sum: r.sum };
    runs.push(one);
    onRun?.(one);
  }
  const totals = runs.map(r => r.total);
  return { runs, mean: +(totals.reduce((a, b) => a + b, 0) / totals.length).toFixed(2), worst: Math.max(...totals) };
}
