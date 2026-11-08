// A person-like player for tuning SOMA Brawl automatically: it plays through the real input (cursor moves and
// clicks dispatched on the canvas), the way a person tries to: closes in on the nearest opponent with the cursor on
// its head, punches when in range (a kick from a little out), and when an opponent it can SEE (in front of it,
// within 55 degrees) winds up, it steps out of the blow after a person's reaction time (a cursor flick to the side),
// then answers. Three kinds: a novice (slow, dodges seldom, presses slowly, aims loosely), a skilled player, and
// `person` - measured from the person's own plays (eval_out/play_20261108004434 and earlier; tools/playlog_report.mjs
// compares them): mashes the buttons in long bursts (gaps ~0.2-0.4 s, ~150 presses a minute), throws the cursor about
// in quick sweeps with short rests between (on the body, the legs, past the opponent, beside it; seldom the head),
// so the body charges in close and slams into opponents, and flicks aside when it sees a wind-up.
// Its play is recorded by the play log like a person's; it adds its own tally of dodges and counters, and marks each
// run in the log ('autoplay' notes: start/end, kind, seed) so the report can tell the runs apart.
//   const ap = await import('./src/autoplay.js'); const r = await ap.series(brawl, { secs: 90, seeds: [1, 2, 3] })
//   ... kinds: ['person'] for the person-like one alone. A large page (the person plays ~1900 px wide) keeps the
//   cursor's screen speeds and the game's flick thresholds (px/s) as the person meets them.
const sleep = ms => new Promise(r => setTimeout(r, ms));

// A ticker that keeps going when the page is hidden (timers there are slowed to once a second; a worker's are not).
function makeTicker() {
  let w = null, waiters = [];
  try {
    w = new Worker(URL.createObjectURL(new Blob(['setInterval(() => postMessage(0), 8)'], { type: 'text/javascript' })));
    w.onmessage = () => { const ws = waiters; waiters = []; for (const f of ws) f(); };
  } catch { w = null; }
  return {
    wait: async ms => { if (!w) return sleep(ms); const end = performance.now() + ms - 4; while (performance.now() < end) await new Promise(r => waiters.push(r)); },
    stop: () => w?.terminate(),
  };
}

export const KINDS = {
  novice: { react: [0.3, 0.45], dodge: 0.3, pressGap: 0.8, jitter: 8, flank: 0 },
  // (flank: how often, facing an opponent that is ready, it works round to the side before striking - the dogfight)
  skilled: { react: [0.2, 0.35], dodge: 0.6, pressGap: 0.55, jitter: 2, flank: 0.7 },
  // (the numbers from the person's play log: see the head of this file)
  person: {
    style: 'person', react: [0.22, 0.4], dodge: 0.12, jitter: 0,
    // presses: bursts of mashing while close; within a burst, gaps lognormal (median, spread) clipped; the burst's
    // length lognormal too (the person's: median 6, a long tail of 20-30 - dodges and swallowed clicks cut a burst,
    // so the bot's median is set higher), then a pause
    gap: [0.27, 0.38, 0.13, 1.1], burst: [10, 1.0, 1, 50], pause: [0.75, 0.5], pressRange: 2.0, kick: 0.08, kickFar: 0.15,
    // the cursor: quick moves (minimum jerk, longer the further: 0.08 s + 0.6 ms a px, with a spread) between
    // targets, holds between them (most with the hand quite still); about half the moves small corrections (6-90 px)
    move: [0.08, 0.0006, 0.3, 0.05, 0.5], hold: [0.15, 0.8, 0.03, 0.9], still: 0.65, snap: 0.04, tremor: 0.6, nudge: [28, 0.6, 6, 90],
    // where the cursor goes, by weight: on the body, the legs, past the opponent (charging through), beside it, the head
    targets: { nudge: 1.0, body: 0.18, legs: 0.08, beyond: 0.38, side: 0.31, head: 0.05 },
    beyond: [0.4, 2.2], side: [0.5, 1.3],
  },
};

export async function run(brawl, seconds = 60, opt = {}) {
  const kindName = opt.kind ?? 'skilled', kind = KINDS[kindName];
  const cv = document.querySelector('canvas'), R = cv.getBoundingClientRect();
  const V = brawl.player.P[0].constructor;
  const toScreen = v => { const q = v.clone().project(brawl.camera); return [R.left + (q.x + 1) / 2 * R.width, R.top + (1 - q.y) / 2 * R.height]; };
  const send = (type, button, buttons, x, y) => cv.dispatchEvent(new PointerEvent(type, { bubbles: true, pointerId: 1, button, buttons, clientX: x, clientY: y }));
  const rng = (() => { let s = (opt.seed ?? 1) * 2654435761 >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const gauss = () => { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); };
  const logn = ([med, sd, lo = 0, hi = Infinity]) => Math.min(hi, Math.max(lo, med * Math.exp(sd * gauss())));
  const I = brawl.rig.index;
  const facing = P => { const l = P.P[I.LeftArm], r = P.P[I.RightArm]; return new V(-(l.z - r.z), 0, l.x - r.x).normalize(); };
  const tally = { dodges: 0, hitInDodge: 0, counters: 0, counterHits: 0, presses: 0, seen: 0, unseen: 0 };
  brawl.plog.pin = true; // (no new log part in the middle of a run: its frame numbers hold)
  const t0 = brawl.clock, f0 = brawl.plog.frameCount, e0 = brawl.plog.events.length;
  const runId = opt.run ?? Math.round(Math.random() * 1e6);
  brawl.plog.note?.('autoplay', { phase: 'start', kind: kindName, seed: opt.seed ?? 1, run: runId, w: Math.round(R.width), h: Math.round(R.height) });
  const ticker = makeTicker();
  const person = kind.style === 'person' ? personPolicy() : null;
  let mode = 'press', modeUntil = 0, dodgeSide = 1, nextPress = 0, watching = null, lastHit = brawl.lastHit, flankAfter = 0, flankSide = 1, counterWin = -1, aimStance = { s: null, read: false };
  const seen = new WeakMap();
  try {
    while (brawl.clock - t0 < seconds) {
      await ticker.wait(person ? 16 : 33);
      const P = brawl.player, now = brawl.clock;
      const es = brawl.fighters.filter(f => !f.isPlayer && f.state !== 'out');
      if (!es.length) continue;
      // (a skilled player fights the one on its feet; a novice or the person pounds whatever is nearest)
      const standing = es.filter(e => !['down', 'fall', 'getup'].includes(e.state) && e.P[0].distanceTo(P.P[0]) < 3.2);
      const pool = (kind.flank ?? 0) >= 0.5 && standing.length ? standing : es;
      const near = pool.reduce((a, b) => (a.P[0].distanceTo(P.P[0]) < b.P[0].distanceTo(P.P[0]) ? a : b));
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
      if (lh && lh !== lastHit) { lastHit = lh; if (lh.by !== 'あなた' && mode === 'dodge') tally.hitInDodge++; if (lh.by === 'あなた' && mode === 'counter' && !lh.touch && counterWin !== tally.counters) { tally.counterHits++; counterWin = tally.counters; } } // (one success per counter try)
      if (watching && watching.will && now >= watching.at && mode !== 'dodge') {
        watching.will = false; mode = 'dodge'; modeUntil = now + (person ? 0.25 : 0.4); dodgeSide = rng() < 0.5 ? -1 : 1; tally.dodges++;
        person?.dodge(watching.e);
      }
      if (watching && now > watching.at + 0.9) watching = null;
      if (mode === 'dodge' && now > modeUntil) { mode = 'counter'; modeUntil = now + 0.9; tally.counters++; nextPress = now + 0.05; person?.counter(); }
      if (mode === 'counter' && now > modeUntil) mode = 'press';
      // off its front (past 50 degrees of where it faces): the flank is won - strike now, from there
      const offFront = () => { const nf = facing(near), toP = new V(P.P[0].x - near.P[0].x, 0, P.P[0].z - near.P[0].z).normalize(); return nf.dot(toP) < Math.cos(50 * Math.PI / 180); };
      if (mode === 'flank' && (now > modeUntil || near.state === 'attack' || (offFront() && now > modeUntil - 0.8))) { if (offFront()) tally.flanksWon = (tally.flanksWon ?? 0) + 1; mode = 'counter'; modeUntil = now + 0.7; nextPress = now; }
      // head-on against a ready opponent: work round to its side first (the cursor on the floor beside and a little
      // behind it; the body keeps facing it), until it is off the opponent's front (up to 1.1 s), then strike
      if (!person && mode === 'press' && d < 1.9 && ['free', 'guard'].includes(near.state) && now > (flankAfter ?? 0)) {
        const nf = facing(near), toP = new V(P.P[0].x - near.P[0].x, 0, P.P[0].z - near.P[0].z).normalize();
        if (nf.dot(toP) > Math.cos(50 * Math.PI / 180)) {
          flankAfter = now + 0.8;
          if (rng() < (kind.flank ?? 0)) { mode = 'flank'; modeUntil = now + 1.1; flankSide = (new V(nf.z, 0, -nf.x).dot(toP) >= 0 ? 1 : -1); tally.flanks = (tally.flanks ?? 0) + 1; }
        }
      }
      if (person) { person.step(P, near, d, now, mode); continue; }
      // the cursor
      const hips = P.P[0];
      // reading the guard: a high guard leaves the body open, a low one the head (a novice reads it seldom)
      const readGuard = rng() < (kind.flank ?? 0) + 0.2;
      if (near.stance !== aimStance.s && rng() < 0.1) aimStance = { s: near.stance, read: readGuard };
      let [x, y] = toScreen(near.P[aimStance.read && aimStance.s === 'high' ? I.Chest : I.Head]);
      if (mode === 'flank') {
        const nf = facing(near), sd = new V(nf.z, 0, -nf.x).multiplyScalar(flankSide);
        [x, y] = toScreen(new V(near.P[0].x + sd.x * 1.1 - nf.x * 0.3, 0, near.P[0].z + sd.z * 1.1 - nf.z * 0.3));
      }
      if (mode === 'dodge') {
        // a person's dodge: a quick flick of the hand sideways on the screen (across the line to the attacker), about 140 px
        const [px, py] = toScreen(hips), [ex, ey] = toScreen(watching?.e?.P[0] ?? near.P[0]);
        let ax = ex - px, ay = ey - py; const al = Math.hypot(ax, ay) || 1; ax /= al; ay /= al;
        x += -ay * 140 * dodgeSide - ax * 30; y += ax * 140 * dodgeSide - ay * 30;
      }
      x += (rng() - 0.5) * 2 * kind.jitter; y += (rng() - 0.5) * 2 * kind.jitter;
      send('pointermove', -1, 0, x, y);
      // presses
      if (mode !== 'dodge' && mode !== 'flank' && now > nextPress && d < (mode === 'counter' ? 1.8 : 1.5) && !['down', 'fall'].includes(P.state)) {
        const kick = d > 1.25 ? rng() < 0.6 : rng() < 0.15;
        send('pointerdown', kick ? 2 : 0, kick ? 2 : 1, x, y); send('pointerup', kick ? 2 : 0, 0, x, y);
        tally.presses++;
        nextPress = now + (mode === 'counter' ? 0.3 : kind.pressGap * (0.7 + 0.6 * rng()));
      }
    }
  } finally { ticker.stop(); }
  brawl.plog.note?.('autoplay', { phase: 'end', kind: kindName, seed: opt.seed ?? 1, run: runId });
  const sum = brawl.plog.summaryFrom(f0);
  Object.assign(sum, extraMeasures(brawl.plog, f0, e0, (brawl.plog.frameCount - f0) / 30));
  brawl.plog.pin = false;
  tally.dodgeOk = tally.dodges ? +((tally.dodges - tally.hitInDodge) / tally.dodges).toFixed(2) : null;
  tally.counterRate = tally.counters ? +(tally.counterHits / tally.counters).toFixed(2) : null;
  if (person) Object.assign(tally, person.tally);
  return { tally, sum };

  // ---------------------------------------------------------------- the person-like hand
  function personPolicy() {
    const K = kind, scale = R.width / 1900; // (px sizes as on the person's screen)
    const pt = { tally: { bursts: 0, sweeps: 0, snaps: 0 } };
    let cur = null; // the cursor on screen
    let mv = null, holdUntil = 0, holdStill = false, anchor = null;
    let burstLeft = 0, nextP = 0, pauseUntil = 0;
    const pick = w => { let r = rng() * Object.values(w).reduce((a, b) => a + b, 0); for (const [k, v] of Object.entries(w)) { if ((r -= v) <= 0) return k; } return Object.keys(w)[0]; };
    // a target tied to the opponent (followed as it moves): a joint, or a point on the floor (at hip height) off it
    function newAnchor(P, e, d) {
      let k = pick(K.targets);
      if (d > 2.6 && k !== 'beyond') k = rng() < 0.7 ? 'beyond' : 'body'; // (far off: charge in)
      if (k === 'nudge' && cur) { const a = rng() * 2 * Math.PI, r = logn(K.nudge) * scale; return { k, e, fixed: [cur[0] + r * Math.cos(a), cur[1] + r * Math.sin(a)] }; }
      if (k === 'nudge') k = 'body';
      const dir = new V(e.P[0].x - P.P[0].x, 0, e.P[0].z - P.P[0].z).normalize(), side = new V(dir.z, 0, -dir.x).multiplyScalar(rng() < 0.5 ? -1 : 1);
      const off = k === 'beyond' ? dir.clone().multiplyScalar(K.beyond[0] + (K.beyond[1] - K.beyond[0]) * rng()).addScaledVector(side, (rng() - 0.5) * 0.8)
        : k === 'side' ? side.clone().multiplyScalar(K.side[0] + (K.side[1] - K.side[0]) * rng()).addScaledVector(dir, (rng() - 0.3) * 0.6) : null;
      const joint = k === 'head' ? I.Head : k === 'body' ? (rng() < 0.5 ? I.Chest : I.Spine2) : k === 'legs' ? (rng() < 0.5 ? I.LeftShin : I.RightShin) : null;
      if (k === 'beyond' || k === 'side') pt.tally.sweeps++;
      return { k, e, at: () => (joint != null ? e.P[joint].clone() : new V(e.P[0].x + off.x, 0.95, e.P[0].z + off.z)) };
    }
    const scr = a => { const [x, y] = toScreen(a.at()); return [x, y]; };
    const ease = s => s * s * s * (10 - 15 * s + 6 * s * s);
    function goTo(P, e, d, now, a, dur) {
      anchor = a ?? newAnchor(P, e, d);
      mv = { from: cur ? cur.slice() : scr(anchor), t0: now, dur };
    }
    pt.dodge = e => {
      // a flick of the hand across the line to the attacker (and a little back): fast, 150-260 px on the person's screen
      const P = brawl.player, [px, py] = toScreen(P.P[0]), [ex, ey] = toScreen(e.P[0]);
      let ax = ex - px, ay = ey - py; const al = Math.hypot(ax, ay) || 1; ax /= al; ay /= al;
      const sd = rng() < 0.5 ? -1 : 1, len = (150 + 110 * rng()) * scale, from = cur ? cur.slice() : [px, py];
      const to = [from[0] - ay * len * sd - ax * 40 * scale, from[1] + ax * len * sd - ay * 40 * scale];
      anchor = { k: 'flick', e, at: null, fixed: to };
      mv = { from, t0: brawl.clock, dur: 0.07 + 0.05 * rng() };
      holdUntil = 0; nextP = Math.max(nextP, brawl.clock + 0.15); // (the hand is busy: no presses during the flick)
    };
    pt.counter = () => { burstLeft = Math.max(burstLeft, 2 + Math.floor(rng() * 4)); nextP = brawl.clock + 0.05; pauseUntil = 0; };
    pt.step = (P, e, d, now, mode) => {
      // the cursor
      if (!cur) { anchor = newAnchor(P, e, d); cur = scr(anchor); }
      if (anchor?.e && anchor.e !== e && anchor.e.state === 'out') { mv = null; holdUntil = 0; }
      if (mv) {
        const s = Math.min(1, (now - mv.t0) / mv.dur), to = anchor.fixed ?? scr(anchor), w = ease(s);
        cur = [mv.from[0] + (to[0] - mv.from[0]) * w, mv.from[1] + (to[1] - mv.from[1]) * w];
        send('pointermove', -1, 0, cur[0], cur[1]);
        if (s >= 1) { mv = null; holdUntil = now + logn(K.hold); holdStill = rng() < K.still; }
      } else if (now < holdUntil) {
        // holding: quite still (no events at all), or keeping on the target as it moves (with a tremor)
        if (!holdStill && anchor && !anchor.fixed) { const to = scr(anchor); cur = [to[0] + gauss() * K.tremor * scale, to[1] + gauss() * K.tremor * scale]; send('pointermove', -1, 0, cur[0], cur[1]); }
      } else if (mode !== 'dodge') {
        // the next move: now and then a sudden jump (one or two frames)
        const snap = rng() < K.snap;
        if (snap) pt.tally.snaps++;
        const a = newAnchor(P, e, d), to = a.fixed ?? scr(a), amp = Math.hypot(to[0] - cur[0], to[1] - cur[1]) / scale;
        const [m0, m1, sd, lo, hi] = K.move;
        goTo(P, e, d, now, a, snap ? 0.03 : Math.min(hi, Math.max(lo, (m0 + m1 * amp) * Math.exp(sd * gauss()))));
      }
      // presses: bursts of mashing while close
      if (['down', 'fall'].includes(P.state)) return; // (presses go on through a dodge: the hand flicks and clicks)
      if (d > K.pressRange + 1.2) burstLeft = 0; // (a long way off: the burst is over)
      if (d > K.pressRange && mode !== 'counter') return;
      if (burstLeft <= 0) {
        if (now < pauseUntil) return;
        burstLeft = Math.round(logn(K.burst)); pt.tally.bursts++; nextP = now;
      }
      if (now < nextP) return;
      const kick = rng() < (d > 1.2 ? K.kickFar : K.kick);
      const [x, y] = cur;
      send('pointerdown', kick ? 2 : 0, kick ? 2 : 1, x, y); send('pointerup', kick ? 2 : 0, 0, x, y);
      tally.presses++;
      if (--burstLeft <= 0) pauseUntil = now + 0.6 + logn(K.pause);
      nextP = now + logn(K.gap);
    };
    return pt;
  }
}

/** Measures the play log has the makings of but does not sum: blows that barely move the struck body, attacks begun
 * within reach, damaging blows taken, the time the player cannot act, presses lost. */
export function extraMeasures(plog, f0, e0, sec) {
  const E = plog.events.slice(e0), F = plog.frames.slice(f0), P = plog.presses.filter(p => p.f >= f0);
  const per = n => (sec > 0 ? +((n / sec) * 60).toFixed(1) : null);
  const dmg = E.filter(e => (e.type === 'hit' || e.type === 'taken') && !e.touch && !e.blocked && e.disp03 != null && !(e.tier === 'ground' && e.energy < 12)); // (a tap on a body lying: no blow to move it)
  const taken = E.filter(e => e.type === 'taken');
  return {
    blowUnmoved: dmg.length ? +(dmg.filter(e => e.disp03 < 0.1).length / dmg.length).toFixed(2) : null,
    blowDisp03: dmg.length ? +dmg.map(e => e.disp03).sort((a, b) => a - b)[dmg.length >> 1].toFixed(2) : null,
    enemyReachPerMin: per(E.filter(e => e.type === 'enemyStrike' && e.d <= 1.3).length),
    takenPerMin: per(taken.length), takenDmgPerMin: per(taken.filter(e => !e.touch && !e.blocked).length),
    dodgePerMin: per(E.filter(e => e.type === 'dodge').length), dodgedPerMin: per(E.filter(e => e.type === 'dodged').length),
    slamsPerMin: per(E.filter(e => e.type === 'hit' && e.slam).length),
    pressPerMin: per(P.length),
    pressLostFrac: P.length ? +(P.filter(p => !/^(started|chained|merged|queued)/.test(String(p.outcome))).length / P.length).toFixed(2) : null,
    playerUnable: F.length ? +(F.filter(fr => fr.ps >= 3).length / F.length).toFixed(3) : null,
  };
}

/**
 * The target bands per kind of player: [lo, hi, scale] (0 when inside; outside, the distance in scales). From what
 * the person asks for: the opponents attack actively and clearly (many attacks, begun within reach, and some land:
 * a fight one has to dodge); a dodging game (dodges register and some attacks are dodged); every damaging blow
 * visibly moves the struck body; no text; no time motionless or out of control. Rates are per minute.
 */
export const BANDS = {
  common: {
    enemyAttacks: [28, 45, 8], enemyReach: [18, 38, 6],
    stunned: [0, 0.32, 0.08], flat: [0, 0.08, 0.05], downStill: [0, 0.15, 0.08], playerStill: [0, 0.03, 0.02],
    unseen: [0, 0.2, 0.1], lull: [0, 4, 2], ko: [1.5, 4, 1],
    blowUnmoved: [0, 0.05, 0.05], playerUnable: [0, 0.1, 0.05], pressLost: [0, 0.08, 0.05],
  },
  novice: { enemyLanded: [0.35, 0.55, 0.1], takenDmg: [4, 12, 3], dodgedPerAtt: [0.03, 0.3, 0.1] },
  skilled: { enemyLanded: [0.25, 0.4, 0.1], takenDmg: [3, 9, 3], dodgedPerAtt: [0.2, 0.55, 0.1], counters: [4, 12, 2],
    // the dogfight: a player who goes round wins on the flank, breaks balance, meets vital points
    sideShare: [0.5, 0.85, 0.15], breaks: [0.8, 3, 0.5], vitalShare: [0.15, 0.45, 0.1] },
  // (the person was hit 3.8 times a minute, 0.18 of the attacks, and felt the opponents' blows did not land: more of
  // them should land and read as landing; slams were a large part of the fun - 7.8 a minute)
  person: { enemyLanded: [0.25, 0.42, 0.08], takenDmg: [4, 10, 2], dodgedPerAtt: [0.15, 0.45, 0.1], dodges: [8, 30, 6], slams: [3, 12, 3], counters: [2, 12, 2] },
};

/** The measures against their target bands: one number each (0 = in band, more = further out) and the sum. */
export function score(sum, tally, sec, kind = 'skilled') {
  const per = x => (x / sec) * 60;
  const B = { ...BANDS.common, ...(BANDS[kind] ?? BANDS.skilled) };
  const band = (v, [lo, hi, scale]) => (v == null || !Number.isFinite(v) ? 1 : v < lo ? (lo - v) / scale : v > hi ? (v - hi) / scale : 0);
  const opt = (v, b) => (v === undefined ? 0 : band(v, b)); // (a measure an older log summary lacks: not counted)
  const parts = {
    enemyAttacks: band(per(sum.enemyStrikes), B.enemyAttacks),
    enemyLanded: band(sum.enemyLanded, B.enemyLanded),
    stunned: band(sum.enemyStunned, B.stunned),
    flat: band(sum.flatFrac, B.flat),
    counterRate: tally.counters >= 6 ? band(tally.counterRate, [0.4, 0.75, 0.15]) : 0, // (fewer than 6 tries: noise)
    counters: B.counters ? band(per(sum.counters ?? 0), B.counters) : 0,
    downStill: band(sum.downStill ?? 0, B.downStill),
    playerStill: band(sum.playerStill, B.playerStill),
    unseen: band(sum.hitsUnseen ?? 0, B.unseen),
    lull: band(sum.longestApart, B.lull),
    ko: band(per(sum.kos ?? 0), B.ko), // (a knock-out every 15-40 s: long enough to be a fight)
    // added: what the person asked for directly
    enemyReach: opt(sum.enemyReachPerMin, B.enemyReach),
    takenDmg: opt(sum.takenDmgPerMin, B.takenDmg),
    dodgedPerAtt: sum.enemyStrikes ? band((sum.dodged ?? 0) / sum.enemyStrikes, B.dodgedPerAtt) : 1,
    blowUnmoved: sum.blowUnmoved === undefined ? 0 : sum.blowUnmoved == null ? 0 : band(sum.blowUnmoved, B.blowUnmoved),
    playerUnable: opt(sum.playerUnable, B.playerUnable),
    pressLost: opt(sum.pressLostFrac, B.pressLost),
    dodges: B.dodges ? opt(sum.dodgePerMin, B.dodges) : 0,
    slams: B.slams ? opt(sum.slamsPerMin, B.slams) : 0,
    sideShare: B.sideShare ? opt(sum.sideShareJ, B.sideShare) : 0,
    breaks: B.breaks ? opt(sum.breaksPerMin, B.breaks) : 0,
    vitalShare: B.vitalShare && sum.vitalShare != null ? opt(sum.vitalShare, B.vitalShare) : 0,
  };
  for (const k in parts) parts[k] = +parts[k].toFixed(2);
  // (legacy: the sum over the measures there were before the added ones, to compare with earlier gates)
  const LEGACY = ['enemyAttacks', 'enemyLanded', 'stunned', 'flat', 'counterRate', 'counters', 'downStill', 'playerStill', 'unseen', 'lull', 'ko'];
  return { parts, total: +Object.values(parts).reduce((a, b) => a + b, 0).toFixed(2), legacy: +LEGACY.reduce((a, k) => a + parts[k], 0).toFixed(2) };
}

/** A series: each kind of player x each seed, from a fresh fight each time; mean and worst (overall and per kind). */
export async function series(brawl, { secs = 90, seeds = [1, 2, 3], kinds = ['novice', 'skilled'], onRun } = {}) {
  const runs = [];
  for (const kind of kinds) for (const seed of seeds) {
    brawl.reset();
    await sleep(2500); // (the fight forms again)
    const r = await run(brawl, secs, { kind, seed, run: runs.length + 1 });
    const sc = score(r.sum, r.tally, secs, kind);
    const one = { kind, seed, total: sc.total, legacy: sc.legacy, parts: sc.parts, tally: r.tally, sum: r.sum };
    runs.push(one);
    onRun?.(one);
  }
  const stat = rs => { const t = rs.map(r => r.total); return { mean: +(t.reduce((a, b) => a + b, 0) / t.length).toFixed(2), worst: Math.max(...t) }; };
  const byKind = Object.fromEntries(kinds.map(k => [k, stat(runs.filter(r => r.kind === k))]));
  return { runs, ...stat(runs), byKind };
}
