// Hit-reaction bench for SOMA Brawl, run inside fight.html?debug (served by tools/serve.mjs):
//   const H = await import('/tools/hit_bench.js?v=1'); H.start({ secs: 60, seed: 1, kind: 'skilled' });
//   ... await H.wait(40); H.result()
// Measures, for every landed blow, how fast and how far the struck body ANSWERS it - the contact feel:
//   - latMs / latRootMs: wall-clock ms from the contact to the struck part (root) having moved 5 (8) cm
//   - d10 / d20: how far the struck part has moved 0.1 / 0.2 s (sim time) after the contact (m)
//   - rootD20 for knockdown blows: how far the body has been carried 0.2 s in (the launch)
// and keeps a rolling ring of frames so the biggest blows save a contact sheet each (eval_out/hit_<n>.png):
// the picture for the number, so "the launch is late" is a measurable, visible thing.
const S = { hits: [], live: [], ring: [], sheets: 0, res: null, err: null, tk: null, lastHitT: null, n: 0, wall0: 0, strikes: [], liveStrikes: new Map(), lastClk: null };

function tick() {
  const brawl = window.brawl;
  try { brawl.draw(performance.now()); } catch (e) { S.err ??= String(e.stack).slice(0, 300); }
  const now = performance.now();
  // ring of small frames (every other tick ~ 30 fps, ~2 s kept)
  if (S.n++ % 2 === 0) {
    const src = document.querySelector('canvas');
    const c = document.createElement('canvas'); c.width = 320; c.height = Math.round(320 * src.height / src.width);
    const g = c.getContext('2d');
    g.drawImage(src, 0, 0, c.width, c.height);
    // the victim of the newest live hit is MARKED on the frame (reviews of crowded scenes kept tracking the wrong
    // body), with how far its root has been carried so far
    const h = S.live[S.live.length - 1];
    if (h) {
      const q = h.F.P[0].clone().project(brawl.camera);
      const x = (q.x + 1) / 2 * c.width, y = (1 - q.y) / 2 * c.height;
      if (x > -20 && x < c.width + 20 && y > -20 && y < c.height + 20) {
        const off = h.F.P[0].clone().sub(h.r0);
        const dr = h.dir ? Math.max(0, off.x * h.dir.x + off.z * h.dir.z) : Math.hypot(off.x, off.z);
        g.strokeStyle = '#0f0'; g.lineWidth = 2;
        g.beginPath(); g.arc(x, y, 9, 0, Math.PI * 2); g.stroke();
        g.fillStyle = '#0f0'; g.font = 'bold 11px monospace'; g.fillText(dr.toFixed(2) + 'm', x + 11, y + 4);
      }
    }
    S.ring.push({ wall: now, clk: brawl.clock, c });
    if (S.ring.length > 64) S.ring.shift();
  }
  // a new landed blow?
  const LH = brawl.lastHit;
  if (LH && LH.t !== S.lastHitT && LH.tier !== 'touch' && LH.tier !== 'blocked' && !LH.armored) {
    S.lastHitT = LH.t;
    const F = brawl.fighters.find(f => f.id === LH.victim);
    if (F) {
      const probe = F.dispProbe && Math.abs(F.dispProbe.t0 - brawl.clock) < 0.05 ? F.dispProbe : null;
      const joint = probe ? probe.joint : 0;
      S.live.push({
        t0: brawl.clock, wall0: now, F, joint, energy: LH.energy, tier: LH.tier ?? 'hit',
        p0: F.P[joint].clone(), r0: F.P[0].clone(), dir: F.knockFx?.dir?.clone() ?? null, latMs: null, latRootMs: null, d10: null, d20: null, rootD20: null,
        down: false, done: false, curve: [],
      });
    }
  }
  // follow the live ones for 0.6 s of sim time
  for (const h of S.live) {
    if (h.done) continue;
    const t = brawl.clock - h.t0;
    if (h.F.state === 'fall' || h.F.state === 'down') h.down = true;
    const dp = h.F.P[h.joint] ? h.F.P[h.joint].distanceTo(h.p0) : 0;
    // along the blow (the victim's own walking does not count as being carried)
    const off = h.F.P[0].clone().sub(h.r0);
    const dr = h.dir ? Math.max(0, off.x * h.dir.x + off.z * h.dir.z) : Math.hypot(off.x, off.z);
    h.curve.push({ t: +t.toFixed(3), dr: +dr.toFixed(3), dp: +dp.toFixed(3) });
    if (h.latMs == null && dp > 0.05) h.latMs = Math.round(now - h.wall0);
    if (h.latRootMs == null && dr > 0.08) h.latRootMs = Math.round(now - h.wall0);
    if (h.d10 == null && t >= 0.1) h.d10 = +dp.toFixed(3);
    if (h.d20 == null && t >= 0.2) { h.d20 = +dp.toFixed(3); h.rootD20 = +dr.toFixed(3); }
    if (t >= 0.6) {
      h.done = true;
      // does the reaction CONNECT? the carried speed along the blow, step to step: a body that stalls (speed
      // near zero, then moving again) is a reaction in two pieces - and that holds for every blow, not only knockdowns
      if (h.curve.length > 6) {
        let peak = 0, stall = 0, run = 0;
        const v = [];
        for (let i = 1; i < h.curve.length; i++) { const dv = (h.curve[i].dr - h.curve[i - 1].dr) / Math.max(1e-3, h.curve[i].t - h.curve[i - 1].t); v.push(dv); peak = Math.max(peak, dv); }
        for (let i = 0; i < v.length; i++) {
          const t2 = h.curve[i + 1].t;
          if (t2 > 0.45) break;
          if (v[i] < Math.max(0.25, peak * 0.15)) { run += (h.curve[i + 1].t - h.curve[i].t); stall = Math.max(stall, run); } else run = 0;
        }
        // a stall only counts after the body really got going, and if it MOVES AGAIN after the lull (settling to
        // rest is the natural end of a light blow, not a seam)
        let movedAfter = false;
        { let lull = false; for (let i = 0; i < v.length && h.curve[i + 1].t <= 0.55; i++) { if (v[i] < Math.max(0.25, peak * 0.15)) lull = true; else if (lull && v[i] > Math.max(0.5, peak * 0.4)) movedAfter = true; } }
        if (peak > 0.8 && (movedAfter || h.down)) { h.stallMs = Math.round(stall * 1000); h.vPeak = +peak.toFixed(2); }
      }
      const { F, ...keep } = h;
      S.hits.push(keep);
      // the biggest blows save their film strip
      if (S.sheets < 4 && (h.energy >= 60 || h.down)) { S.sheets++; saveSheet(h, S.sheets); }
    }
  }
  S.live = S.live.filter(h => !h.done);
  // the striking limb's speed, strike by strike (the curve of the blow: wind-up, burst, follow-through)
  const dtc = S.lastClk == null ? 0 : brawl.clock - S.lastClk; S.lastClk = brawl.clock;
  if (dtc > 0.004 && dtc < 0.1) for (const F of brawl.fighters) {
    const A = F.state === 'attack' ? F.attack : null;
    let st = S.liveStrikes.get(F);
    if (A && A.part && (!st || st.id !== A.id)) { st = { id: A.id, kind: A.kind, who: F.isPlayer ? 'P' : 'E', t0: brawl.clock, pts: [], prev: null }; S.liveStrikes.set(F, st); }
    if (!st) continue;
    if (!A || st.id !== A.id || brawl.clock - st.t0 > 1.0) {
      if (st.pts.length > 10) S.strikes.push(st);
      S.liveStrikes.delete(F);
      continue;
    }
    const I = brawl.rig.index, tip = F.P[I[A.part]];
    if (st.prev) st.pts.push({ t: +(brawl.clock - st.t0).toFixed(3), v: +(tip.distanceTo(st.prev) / dtc).toFixed(2) });
    st.prev = tip.clone();
  }
}

/**
 * The curves as one picture (eval_out/hit_curves.png):
 * - top: every strike's limb-tip speed over its first second, one row each, grouped by kind - a good blow reads as
 *   a dark wind-up, one bright burst, a fading tail; mush reads as grey all along
 * - bottom: every landed blow's root displacement over 0.6 s, one row each, SORTED BY ENERGY - if the effect fits
 *   the energy, the picture brightens steadily from top to bottom, with no bright light rows or dark heavy ones
 */
async function saveCurves() {
  const CW = 4, RH = 5, X0 = 120, strikes = [...S.strikes].sort((a, b) => (a.who + a.kind).localeCompare(b.who + b.kind));
  const hits = [...S.hits].sort((a, b) => a.energy - b.energy);
  const W = X0 + 62 * CW + 20, H1 = strikes.length * RH, H2 = hits.length * RH;
  const c = document.createElement('canvas'); c.width = W; c.height = H1 + H2 + 90;
  const g = c.getContext('2d'); g.fillStyle = '#181818'; g.fillRect(0, 0, c.width, c.height);
  const col = u => { u = Math.max(0, Math.min(1, u)); return `rgb(${Math.round(255 * u ** 0.7)},${Math.round(235 * u ** 1.6)},${Math.round(90 * (1 - u) + 30)})`; };
  g.font = '10px monospace';
  g.fillStyle = '#ddd'; g.fillText('strike limb speed 0-1s (bright = fast, full = 11 m/s)', 4, 12);
  strikes.forEach((st, r) => {
    const y = 18 + r * RH;
    g.fillStyle = '#999'; g.fillText(`${st.who} ${st.kind}`, 4, y + RH - 1);
    for (const p of st.pts) { const x = X0 + Math.floor(p.t * 60) * CW; g.fillStyle = col(p.v / 11); g.fillRect(x, y, CW, RH - 1); }
  });
  const yB = 18 + H1 + 30;
  g.fillStyle = '#ddd'; g.fillText('hit -> root carried 0-0.6s, rows sorted by energy (bright = far, full = 0.9 m)', 4, yB - 8);
  hits.forEach((h, r) => {
    const y = yB + r * RH;
    g.fillStyle = h.down ? '#fa6' : '#999'; g.fillText(`${Math.round(h.energy)}J${h.down ? ' D' : ''}`, 4, y + RH - 1);
    for (const p of h.curve) { const x = X0 + Math.floor(p.t * 100) * CW; g.fillStyle = col(p.dr / 0.9); g.fillRect(x, y, CW, RH - 1); }
  });
  const blob = await new Promise(r => c.toBlob(r, 'image/png'));
  await fetch('/save/hit_curves.png', { method: 'POST', body: blob });
}
export { saveCurves };

async function saveSheet(h, k) {
  const frames = S.ring.filter(f => f.wall >= h.wall0 - 200 && f.wall <= h.wall0 + 650);
  if (frames.length < 4) return;
  const W = 320, H = frames[0].c.height, cols = Math.min(8, frames.length), rows = Math.ceil(frames.length / cols);
  const sheet = document.createElement('canvas'); sheet.width = cols * W; sheet.height = rows * H + 20;
  const g = sheet.getContext('2d'); g.fillStyle = '#222'; g.fillRect(0, 0, sheet.width, sheet.height);
  frames.forEach((f, i) => {
    g.drawImage(f.c, (i % cols) * W, Math.floor(i / cols) * H);
    g.fillStyle = f.wall < h.wall0 ? '#999' : '#ff0';
    g.font = '11px monospace';
    g.fillText(`${Math.round(f.wall - h.wall0)}ms`, (i % cols) * W + 3, Math.floor(i / cols) * H + 12);
  });
  g.fillStyle = '#ff0'; g.font = '13px monospace';
  g.fillText(`hit ${k}: ${Math.round(h.energy)}J ${h.tier}${h.down ? ' DOWN' : ''}  0ms = contact`, 4, sheet.height - 6);
  const blob = await new Promise(r => sheet.toBlob(r, 'image/png'));
  await fetch('/save/hit_' + k + '.png', { method: 'POST', body: blob });
}

export async function start(opts = {}) {
  const { secs = 60, seed = 1, kind = 'skilled' } = opts;
  for (let i = 0; i < 60 && !window.brawl?.scheduler?.ready; i++) await new Promise(r => setTimeout(r, 1000));
  const ap = await import('/src/autoplay.js');
  S.tk?.terminate();
  S.tk = new Worker(URL.createObjectURL(new Blob(['setInterval(()=>postMessage(0),16)'], { type: 'text/javascript' })));
  S.tk.onmessage = tick;
  Object.assign(S, { hits: [], live: [], ring: [], sheets: 0, res: null, err: null, lastHitT: window.brawl.lastHit?.t ?? null, n: 0, strikes: [], liveStrikes: new Map(), lastClk: null });
  ap.series(window.brawl, { secs, seeds: [seed], kinds: [kind] }).then(r => { S.res = r; stop(); saveCurves(); }).catch(e => { S.err = String(e.stack); stop(); });
  return 'running';
}
export function stop() { S.tk?.terminate(); S.tk = null; }
export const done = () => !!(S.res || S.err);

export function result() {
  if (S.err) return { err: S.err };
  const med = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const mean = a => a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(3) : null;
  const groups = { light: h => h.energy < 30 && !h.down, mid: h => h.energy >= 30 && !h.down, down: h => h.down };
  const out = {};
  for (const [g, f] of Object.entries(groups)) {
    const hs = S.hits.filter(f);
    out[g] = {
      n: hs.length,
      latMs: med(hs.map(h => h.latMs).filter(v => v != null)),
      latNone: hs.filter(h => h.latMs == null).length, // (never moved 5 cm: no visible answer)
      d10: mean(hs.map(h => h.d10).filter(v => v != null)),
      d20: mean(hs.map(h => h.d20).filter(v => v != null)),
      latRootMs: med(hs.map(h => h.latRootMs).filter(v => v != null)),
      rootD20: mean(hs.map(h => h.rootD20).filter(v => v != null)),
      stallMs: med(hs.map(h => h.stallMs).filter(v => v != null)),
      stallBad: hs.filter(h => (h.stallMs ?? 0) > 80).length, // flights in two pieces
    };
  }
  // does the effect fit the energy? displacement by energy band, and the rank correlation (1 = strictly: more J,
  // further carried)
  const bins = [[12, 30], [30, 60], [60, 100], [100, 999]];
  out.byEnergy = bins.map(([a, b]) => {
    const hs = S.hits.filter(h => h.energy >= a && h.energy < b && h.rootD20 != null);
    return { band: `${a}-${b}J`, n: hs.length, rootD20: mean(hs.map(h => h.rootD20)), d20: mean(hs.map(h => h.d20).filter(v => v != null)) };
  });
  {
    const hs = S.hits.filter(h => h.rootD20 != null);
    const rank = arr => { const idx = arr.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r = []; idx.forEach(([, i], k) => r[i] = k); return r; };
    const ra = rank(hs.map(h => h.energy)), rb = rank(hs.map(h => h.rootD20));
    const n = hs.length, ma = (n - 1) / 2;
    let num = 0, da = 0, db2 = 0;
    for (let i = 0; i < n; i++) { num += (ra[i] - ma) * (rb[i] - ma); da += (ra[i] - ma) ** 2; db2 += (rb[i] - ma) ** 2; }
    out.energyEffectCorr = n > 5 ? +(num / Math.sqrt(da * db2)).toFixed(2) : null;
  }
  // the strike curves: one burst is a blow, two bursts split by a lull is a seam (the speculative swing and the
  // generated one not handing over cleanly)
  {
    const sts = S.strikes.filter(st => st.pts.length > 12);
    let dbl = 0, n = 0;
    for (const st of sts) {
      const M = Math.max(...st.pts.map(p => p.v));
      if (M < 3) continue;
      n++;
      // a second burst only counts after a real LULL (a fast return right after the top is how a punch moves;
      // a dead pause and then a second swing is a seam)
      let regions = 0, inR = false, lull = 0, valley = true;
      for (let i = 1; i < st.pts.length; i++) {
        const p = st.pts[i], dt2 = p.t - st.pts[i - 1].t;
        if (p.v > 0.55 * M) { if (!inR && valley) { regions++; valley = false; } inR = true; lull = 0; }
        else { inR = false; if (p.v < 0.25 * M) { lull += dt2; if (lull > 0.12) valley = true; } }
      }
      if (regions >= 2) dbl++;
    }
    out.strikes = { n, doubleBurst: dbl, doubleShare: n ? +(dbl / n).toFixed(2) : null };
  }
  return { done: done(), ...out, sheets: S.sheets };
}
export async function wait(secs = 40) { for (let i = 0; i < secs && !done(); i++) await new Promise(r => setTimeout(r, 1000)); return done(); }
