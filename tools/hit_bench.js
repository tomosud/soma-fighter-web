// Hit-reaction bench for SOMA Brawl, run inside fight.html?debug (served by tools/serve.mjs):
//   const H = await import('/tools/hit_bench.js?v=1'); H.start({ secs: 60, seed: 1, kind: 'skilled' });
//   ... await H.wait(40); H.result()
// Measures, for every landed blow, how fast and how far the struck body ANSWERS it - the contact feel:
//   - latMs / latRootMs: wall-clock ms from the contact to the struck part (root) having moved 5 (8) cm
//   - d10 / d20: how far the struck part has moved 0.1 / 0.2 s (sim time) after the contact (m)
//   - rootD20 for knockdown blows: how far the body has been carried 0.2 s in (the launch)
// and keeps a rolling ring of frames so the biggest blows save a contact sheet each (eval_out/hit_<n>.png):
// the picture for the number, so "the launch is late" is a measurable, visible thing.
const S = { hits: [], live: [], ring: [], sheets: 0, res: null, err: null, tk: null, lastHitT: null, n: 0, wall0: 0 };

function tick() {
  const brawl = window.brawl;
  try { brawl.draw(performance.now()); } catch (e) { S.err ??= String(e.stack).slice(0, 300); }
  const now = performance.now();
  // ring of small frames (every other tick ~ 30 fps, ~2 s kept)
  if (S.n++ % 2 === 0) {
    const src = document.querySelector('canvas');
    const c = document.createElement('canvas'); c.width = 320; c.height = Math.round(320 * src.height / src.width);
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    S.ring.push({ wall: now, clk: brawl.clock, c });
    if (S.ring.length > 64) S.ring.shift();
  }
  // a new landed blow?
  const LH = brawl.lastHit;
  if (LH && LH.t !== S.lastHitT && LH.tier !== 'touch' && LH.tier !== 'blocked') {
    S.lastHitT = LH.t;
    const F = brawl.fighters.find(f => f.id === LH.victim);
    if (F) {
      const probe = F.dispProbe && Math.abs(F.dispProbe.t0 - brawl.clock) < 0.05 ? F.dispProbe : null;
      const joint = probe ? probe.joint : 0;
      S.live.push({
        t0: brawl.clock, wall0: now, F, joint, energy: LH.energy, tier: LH.tier ?? 'hit',
        p0: F.P[joint].clone(), r0: F.P[0].clone(), latMs: null, latRootMs: null, d10: null, d20: null, rootD20: null,
        down: false, done: false,
      });
    }
  }
  // follow the live ones for 0.6 s of sim time
  for (const h of S.live) {
    if (h.done) continue;
    const t = brawl.clock - h.t0;
    if (h.F.state === 'fall' || h.F.state === 'down') h.down = true;
    const dp = h.F.P[h.joint] ? h.F.P[h.joint].distanceTo(h.p0) : 0;
    const dr = h.F.P[0].distanceTo(h.r0);
    if (h.latMs == null && dp > 0.05) h.latMs = Math.round(now - h.wall0);
    if (h.latRootMs == null && dr > 0.08) h.latRootMs = Math.round(now - h.wall0);
    if (h.d10 == null && t >= 0.1) h.d10 = +dp.toFixed(3);
    if (h.d20 == null && t >= 0.2) { h.d20 = +dp.toFixed(3); h.rootD20 = +dr.toFixed(3); }
    if (t >= 0.6) {
      h.done = true;
      const { F, ...keep } = h;
      S.hits.push(keep);
      // the biggest blows save their film strip
      if (S.sheets < 4 && (h.energy >= 60 || h.down)) { S.sheets++; saveSheet(h, S.sheets); }
    }
  }
  S.live = S.live.filter(h => !h.done);
}

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
  Object.assign(S, { hits: [], live: [], ring: [], sheets: 0, res: null, err: null, lastHitT: window.brawl.lastHit?.t ?? null, n: 0 });
  ap.series(window.brawl, { secs, seeds: [seed], kinds: [kind] }).then(r => { S.res = r; stop(); }).catch(e => { S.err = String(e.stack); stop(); });
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
    };
  }
  return { done: done(), ...out, sheets: S.sheets };
}
export async function wait(secs = 40) { for (let i = 0; i < secs && !done(); i++) await new Promise(r => setTimeout(r, 1000)); return done(); }
