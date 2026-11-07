// A person-like player for tuning SOMA Brawl automatically: it plays through the real input (cursor moves and
// clicks dispatched on the canvas), the way a person tries to: closes in on the nearest opponent with the cursor on
// its head, punches when in range (a kick from a little out), and when an opponent near it winds up it steps out of
// the blow (the cursor flicked to the side and back), then steps round and answers (a counter). Its play is recorded
// by the play log like a person's; it adds its own tally of dodges and counters.
//   const ap = await import('./src/autoplay.js'); const r = await ap.run(brawl, 60, { seed: 1 })
const sleep = ms => new Promise(r => setTimeout(r, ms));

export async function run(brawl, seconds = 60, opt = {}) {
  const cv = document.querySelector('canvas'), R = cv.getBoundingClientRect();
  const V = brawl.player.P[0].constructor;
  const toScreen = v => { const q = v.clone().project(brawl.camera); return [R.left + (q.x + 1) / 2 * R.width, R.top + (1 - q.y) / 2 * R.height]; };
  const send = (type, button, buttons, x, y) => cv.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 1, button, buttons, clientX: x, clientY: y }));
  const rng = (() => { let s = (opt.seed ?? 1) >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const tally = { dodges: 0, dodged: 0, hitInDodge: 0, counters: 0, counterHits: 0, presses: 0 };
  const t0 = brawl.clock;
  let mode = 'press', modeUntil = 0, dodgeSide = 1, nextPress = 0, watching = null, lastTaken = brawl.lastHit;
  const reactMs = opt.react ?? 240; // a person's reaction to a wind-up
  const seen = new WeakMap();
  while (brawl.clock - t0 < seconds) {
    await sleep(33);
    const P = brawl.player, now = brawl.clock;
    const es = brawl.fighters.filter(f => !f.isPlayer && !['out'].includes(f.state));
    if (!es.length) continue;
    const near = es.reduce((a, b) => (a.P[0].distanceTo(P.P[0]) < b.P[0].distanceTo(P.P[0]) ? a : b));
    const d = Math.hypot(near.P[0].x - P.P[0].x, near.P[0].z - P.P[0].z);
    // an opponent close by starting a blow: noticed after a person's reaction time
    for (const e of es) {
      const dd = Math.hypot(e.P[0].x - P.P[0].x, e.P[0].z - P.P[0].z);
      if (e.state === 'attack' && e.attack !== seen.get(e)) { seen.set(e, e.attack); if (dd < 1.9) watching = { e, at: now + reactMs / 1000, will: (opt.dodge ?? 0.55) > rng() }; }
    }
    const lh = brawl.lastHit;
    if (lh && lh !== lastTaken) { lastTaken = lh; if (lh.by !== 'あなた' && mode === 'dodge') tally.hitInDodge++; if (lh.by === 'あなた' && mode === 'counter' && !lh.touch) tally.counterHits++; }
    if (watching && watching.will && now >= watching.at && mode !== 'dodge') {
      watching.will = false;
      mode = 'dodge'; modeUntil = now + 0.45; dodgeSide = rng() < 0.5 ? -1 : 1; tally.dodges++; watching.dodging = true;
    }
    if (watching && now > watching.at + 0.9) watching = null;
    if (mode === 'dodge' && now > modeUntil) { mode = 'counter'; modeUntil = now + 0.9; tally.counters++; nextPress = now + 0.05; }
    if (mode === 'counter' && now > modeUntil) mode = 'press';
    // the cursor
    const hips = P.P[0], eh = near.P[brawl.rig.index.Head];
    let aimPt = eh.clone();
    if (mode === 'dodge') {
      // out to the side and a little back (the cursor on the floor there)
      const to = new V(near.P[0].x - hips.x, 0, near.P[0].z - hips.z).normalize(), side = new V(to.z, 0, -to.x).multiplyScalar(dodgeSide);
      aimPt = hips.clone().addScaledVector(side, 1.1).addScaledVector(to, -0.3); aimPt.y = 0.95;
    } else if (mode === 'counter') {
      // round to its side: aim at its head from a little to the side
      aimPt = eh.clone();
    } else if (d > 2.2 && rng() < 0.02) {
      aimPt = eh.clone(); // keep closing in
    }
    let [x, y] = toScreen(aimPt);
    send('pointermove', -1, 0, x, y);
    // presses
    if (mode !== 'dodge' && now > nextPress && d < (mode === 'counter' ? 1.8 : 1.5) && P.state !== 'down' && P.state !== 'fall') {
      const kick = d > 1.25 ? rng() < 0.6 : rng() < 0.15;
      send('pointerdown', kick ? 2 : 0, kick ? 2 : 1, x, y); send('pointerup', kick ? 2 : 0, 0, x, y);
      tally.presses++;
      nextPress = now + (mode === 'counter' ? 0.28 : (opt.pressGap ?? 0.55) * (0.7 + 0.6 * rng()));
    }
  }
  const L = brawl.plog, sum = L.summary();
  tally.dodgeOk = tally.dodges ? +((tally.dodges - tally.hitInDodge) / tally.dodges).toFixed(2) : null;
  tally.counterRate = tally.counters ? +(tally.counterHits / tally.counters).toFixed(2) : null;
  return { tally, sum };
}

/** The measures to tune toward, as one number per measure (0 = on target, more = worse) and their sum. */
export function score(sum, tally, sec) {
  const per = x => (x / sec) * 60;
  const band = (v, lo, hi, scale) => (v == null ? 1 : v < lo ? (lo - v) / scale : v > hi ? (v - hi) / scale : 0);
  const parts = {
    enemyAttacks: band(per(sum.enemyStrikes), 25, 45, 10),     // attacks per minute: pressure
    enemyLanded: band(sum.enemyLanded, 0.25, 0.45, 0.1),       // dodging should matter, but they should land
    stunned: band(sum.enemyStunned, 0.15, 0.32, 0.1),          // opponents mostly able to act
    flat: band(sum.flatFrac, 0, 0.08, 0.05),
    dodge: band(tally.dodgeOk, 0.55, 0.95, 0.15),              // a dodge done in time works
    counter: band(tally.counterRate, 0.35, 1, 0.15),           // and opens a counter
    counters: band(per(sum.counters ?? 0), 3, 16, 2),         // counters landed per minute
    ko: band(per(sum.kos ?? 0), 1.5, 4, 1),                    // a KO every 15-40 s
    downStill: band(sum.downStill, 0, 0.25, 0.1),               // never lying like a thing
    knock: band(sum.knockDelay, 0, 0.25, 0.05),
    playerStill: band(sum.playerStill, 0, 0.03, 0.02),        // the player never stuck motionless
  };
  return { parts, total: +Object.values(parts).reduce((a, b) => a + b, 0).toFixed(2) };
}
