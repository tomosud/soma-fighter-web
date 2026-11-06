// SOMA Fighter Web: a one-on-one fighting game whose every frame of motion is generated in the browser by
// NVIDIA MotionBricks (SOMA). Only key poses are given (from Mixamo-derived SOMA clips, placed against the
// opponent and bent by IK); MotionBricks generates the motion toward them.
import { Quaternion, Vector3 } from 'three';
import { Rig, placePose, emptyBatch, encodeRequest } from './soma.js';
import { KeyBook } from './poses.js';
import { Track, Scheduler, BATCH } from './motion.js';
import { Stage, Mannequin } from './render.js';

const MB_REV = '54fc563aeaaf819a753d72cc9fdc6713989cdd43';
const MODELS = {
  fp16: { label: 'Mixed FP16（380 MB）', url: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx` },
  fp32: { label: 'FP32（758 MB）', url: `https://media.githubusercontent.com/media/NVlabs/motionbricks/${MB_REV}/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx` },
};
const params = new URLSearchParams(location.search);
const UP = new Vector3(0, 1, 0);
const ZERO = new Vector3();
const RING = 3.7;
const HP = 100;
const $ = id => document.getElementById(id);

// key: key pose, chain: IK chain to the striking end, tip: the tip is between the chain's end and this joint
const ARM_L = ['LeftArm', 'LeftForeArm', 'LeftHand'], ARM_R = ['RightArm', 'RightForeArm', 'RightHand'];
const LEG_R = ['RightLeg', 'RightShin', 'RightFoot'];
const ATTACKS = {
  jab: { name: 'ジャブ', key: 'jab', kind: 'punch', chain: ARM_L, tip: 'LeftHandMiddleEnd', aim: 'head', power: 7, rate: 2.4, tokens: [6], step: 0.45 },
  cross: { name: 'ストレート', key: 'cross', kind: 'punch', chain: ARM_R, tip: 'RightHandMiddleEnd', aim: 'head', power: 12, rate: 1.8, tokens: [6], step: 0.55 },
  hook: { name: 'フック', key: 'hook', kind: 'punch', chain: ARM_R, tip: 'RightHandMiddleEnd', aim: 'head', power: 14, rate: 1.6, tokens: [6], step: 0.5 },
  uppercut: { name: 'アッパー', key: 'uppercut', kind: 'punch', chain: ARM_L, tip: 'LeftHandMiddleEnd', aim: 'chin', power: 13, rate: 1.7, tokens: [6], step: 0.45 },
  mid_kick: { name: 'ミドルキック', key: 'mid_kick', kind: 'kick', chain: LEG_R, tip: 'RightToeBase', aim: 'body', power: 15, rate: 1.7, tokens: [6], step: 0.5 },
  high_kick: { name: 'ハイキック', key: 'high_kick', kind: 'kick', chain: LEG_R, tip: 'RightToeBase', aim: 'head', power: 20, rate: 1.6, tokens: [7], step: 0.6 },
};
// the defender's key pose by where and how hard it was hit; back = metres pushed away
const REACTIONS = {
  head_light: { key: 'hit_head_light', back: 0.15, tokens: [6], rate: 2.2 },
  head: { key: 'hit_head', back: 0.25, tokens: [6], rate: 2.0 },
  head_big: { key: 'hit_head_big', back: 0.35, tokens: [6, 7], rate: 1.8 },
  body: { key: 'hit_body', back: 0.3, tokens: [6], rate: 2.0 },
  stomach: { key: 'hit_stomach', back: 0.15, tokens: [6], rate: 2.0 },
  down: { key: 'down_back', back: 0.4, tokens: [10], rate: 1.5, down: true },
  down_ko: { key: 'down_ko', back: 0.3, tokens: [12, 13, 14], rate: 1.3, down: true },
};

const rig = new Rig(await (await fetch(new URL('../assets/soma_rig.json', import.meta.url))).json());
const book = new KeyBook(rig, await (await fetch(new URL('../assets/keyposes.json', import.meta.url))).json());
const stage = new Stage($('view'));
const worker = new Worker(new URL('./infer.worker.js', import.meta.url), { type: 'module' });
const scheduler = new Scheduler(rig, worker);

class Fighter {
  constructor(name, color, cpu) {
    this.name = name;
    this.cpu = cpu;
    this.track = new Track(rig, book);
    this.body = new Mannequin(stage.scene, rig, color);
    this.hp = HP;
    this.wins = 0;
    this.prevTip = null;
    this.think = 0;
    this.stopSent = false;
  }
  get root() { return this.track.P[0]; }
  get P() { return this.track.P; }
}

const p1 = new Fighter('YOU', 0x4f8cff, false);
const p2 = new Fighter('CPU', 0xff5a4f, true);
p1.opp = p2; p2.opp = p1;
const fighters = [p1, p2];

// ---------------------------------------------------------------- requests (built when the batch is sent)

const yawTo = (from, to) => Math.atan2(to.x - from.x, to.z - from.z);
const ground = v => new Vector3(v.x, 0, v.z);

function clampRing(v) {
  const r = Math.hypot(v.x, v.z);
  if (r > RING) v.multiplyScalar(RING / r);
  return v;
}

function idleReq(f, mode = 'end') {
  return {
    kind: 'idle', mode,
    build(track, a) {
      const A = ground(track.frame(a).pose.root);
      const yaw = yawTo(A, f.opp.root);
      return { targets: book.get(Math.random() < 0.5 ? 'stance' : 'stance_b').map(p => placePose(p, yaw, ZERO, A)), tokens: [8, 9, 10, 11, 12], rate: 1, tag: { kind: 'idle' } };
    },
  };
}

function moveReq(f, dir, mode = 'now') {
  return {
    kind: 'move', mode, dir: dir.clone(),
    build(track, a) {
      const A = ground(track.frame(a).pose.root);
      const dest = clampRing(A.clone().addScaledVector(dir, 0.6));
      const yaw = yawTo(dest, f.opp.root);
      return { targets: book.get('stance').map(p => placePose(p, yaw, ZERO, dest)), tokens: [6, 7, 8], rate: 1.25, tag: { kind: 'move', dir: dir.clone() } };
    },
  };
}

function aimPoint(f, aim) {
  const P = f.P;
  if (aim === 'head') return P[rig.index.Head].clone().lerp(f.track.extra[0], 0.45);
  if (aim === 'chin') return P[rig.index.Jaw].clone().lerp(P[rig.index.Head], 0.3);
  return P[rig.index.Spine2].clone();
}

function attackReq(f, name) {
  const atk = ATTACKS[name];
  const natural = book.tip(atk.key, atk.chain[2], atk.tip); // where the key pose's limb tip is
  return {
    kind: 'attack', mode: 'now', key: name,
    build(track, a) {
      const A = ground(track.frame(a).pose.root);
      const yaw = yawTo(A, f.opp.root);
      const fwd = new Vector3(Math.sin(yaw), 0, Math.cos(yaw));
      const local = aimPoint(f.opp, atk.aim).sub(A).applyQuaternion(new Quaternion().setFromAxisAngle(UP, -yaw));
      // step in (or back a little) so the key's own reach lands near the point, then bend the limb onto it
      const step = Math.max(-0.15, Math.min(atk.step, local.z - natural.z));
      local.z -= step;
      local.z += 0.08; // swing through the point
      const poses = book.strike(atk.key, atk.chain, atk.tip, local);
      const tag = { kind: 'attack', atk: { ...atk, limb: atk.chain[2], done: false, start: 0, end: 0 } };
      const at = clampRing(A.clone().addScaledVector(fwd, step));
      return {
        targets: poses.map(p => placePose(p, yaw, ZERO, at)),
        tokens: atk.tokens, rate: atk.rate, tag,
        after(tr, a0, frames) {
          tag.atk.start = a0;
          tag.atk.end = a0 + frames.length - 1;
          tr.request(idleReq(f, 'end'));
        },
      };
    },
  };
}

function hitReq(f, from, how) {
  const r = REACTIONS[how];
  return {
    kind: r.down ? 'down' : 'hit', mode: 'now', interrupt: true,
    build(track, a) {
      const A = ground(track.frame(a).pose.root);
      const away = A.clone().sub(ground(from));
      if (away.lengthSq() < 1e-6) away.set(1, 0, 0);
      away.normalize();
      const dest = clampRing(A.clone().addScaledVector(away, r.back));
      const yaw = Math.atan2(-away.x, -away.z); // keys face the attacker
      return {
        targets: book.get(r.key).map(p => placePose(p, yaw, ZERO, dest)),
        tokens: r.tokens, rate: r.rate, tag: { kind: r.down ? 'down' : 'hit' },
        after: r.down ? null : tr => tr.request(idleReq(f, 'end')),
      };
    },
  };
}

function victoryReq(f) {
  return {
    kind: 'down', mode: 'now', interrupt: true,
    build(track, a) {
      const A = ground(track.frame(a).pose.root);
      return { targets: book.get('victory').map(p => placePose(p, yawTo(A, f.opp.root), ZERO, A)), tokens: [10, 11, 12], rate: 1, tag: { kind: 'down' } };
    },
  };
}

// ---------------------------------------------------------------- input

const keys = new Set();
let buffered = null; // attack pressed while the player cannot act yet
const ATTACK_KEYS = { KeyJ: 'jab', KeyL: 'cross', KeyU: 'hook', KeyI: 'uppercut', KeyK: 'mid_kick', KeyO: 'high_kick' };
addEventListener('keydown', e => {
  if (e.repeat) return;
  keys.add(e.code);
  if (ATTACK_KEYS[e.code]) buffered = { key: ATTACK_KEYS[e.code], time: performance.now() };
});
addEventListener('keyup', e => keys.delete(e.code));
addEventListener('blur', () => keys.clear());
$('view').addEventListener('mousedown', e => { buffered = { key: e.button === 2 ? 'mid_kick' : e.button === 1 ? 'cross' : 'jab', time: performance.now() }; });
$('view').addEventListener('contextmenu', e => e.preventDefault());

let cam = { right: new Vector3(1, 0, 0), toward: new Vector3(0, 0, 1) };

function playerControl(f) {
  const tr = f.track;
  if (buffered && performance.now() - buffered.time > 450) buffered = null;
  if (buffered && tr.tag.kind !== 'down' && tr.tag.kind !== 'hit') {
    if (tr.request(attackReq(f, buffered.key))) buffered = null;
  }
  if (tr.locked || (tr.pending && tr.pending.kind === 'attack') || tr.inflight?.kind === 'attack') return;
  const x = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0);
  const y = (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0) - (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0);
  if (x || y) {
    f.stopSent = false;
    const dir = cam.right.clone().multiplyScalar(x).addScaledVector(cam.toward, y).normalize();
    const cur = tr.inflight?.kind === 'move' ? tr.inflight.dir : tr.pending?.kind === 'move' ? tr.pending.dir : tr.tag.kind === 'move' ? tr.tag.dir : null;
    if (!cur || cur.dot(dir) < 0.8) tr.request(moveReq(f, dir, 'now'));
    else if (!tr.busy && tr.remaining / 30 < scheduler.latency + 0.2) tr.request(moveReq(f, dir, 'end'));
  } else if (!f.stopSent && (tr.tag.kind === 'move' || tr.pending?.kind === 'move' || tr.inflight?.kind === 'move')) {
    f.stopSent = true;
    tr.request(idleReq(f, 'now'));
  }
}

function cpuControl(f, dt) {
  const tr = f.track;
  f.think -= dt;
  if (f.think > 0 || tr.locked || tr.busy) return;
  f.think = 0.3 + Math.random() * 0.45;
  const d = Math.hypot(f.opp.root.x - f.root.x, f.opp.root.z - f.root.z);
  const toOpp = new Vector3(f.opp.root.x - f.root.x, 0, f.opp.root.z - f.root.z).normalize();
  const side = new Vector3(-toOpp.z, 0, toOpp.x).multiplyScalar(Math.random() < 0.5 ? 1 : -1);
  const r = Math.random();
  if (f.opp.track.tag.kind === 'attack' && d < 1.3 && r < 0.3) return tr.request(moveReq(f, toOpp.clone().negate()));
  if (d > 1.4) return tr.request(moveReq(f, toOpp));
  if (d < 0.65) return tr.request(r < 0.5 ? moveReq(f, toOpp.clone().negate()) : attackReq(f, 'jab'));
  if (r < 0.62) {
    const pick = ['jab', 'jab', 'cross', 'hook', 'uppercut', 'mid_kick', 'mid_kick', 'high_kick'];
    return tr.request(attackReq(f, pick[Math.floor(Math.random() * pick.length)]));
  }
  if (r < 0.74) return tr.request(moveReq(f, toOpp.clone().negate()));
  if (r < 0.86) return tr.request(moveReq(f, side));
}

// ---------------------------------------------------------------- hits

function segDist(p, a, b) {
  const ab = b.clone().sub(a), t = Math.max(0, Math.min(1, p.clone().sub(a).dot(ab) / Math.max(1e-9, ab.lengthSq())));
  return p.distanceTo(a.clone().addScaledVector(ab, t));
}

let hitstop = 0;
const traces = [];
function trace(att, atk, result) {
  if (!params.has('debug')) return;
  const d = Math.hypot(att.opp.root.x - att.root.x, att.opp.root.z - att.root.z);
  traces.push({ who: att.name, atk: atk.name, result, closest: +atk.closest.toFixed(3), dist: +d.toFixed(2), frames: atk.end - atk.start });
}

function checkHits(att, def, dt) {
  const tag = att.track.tag;
  const atk = tag.kind === 'attack' ? tag.atk : null;
  if (!atk) { att.prevTip = null; return; }
  const P = att.P;
  const tip = P[rig.index[atk.limb]].clone().lerp(P[rig.index[atk.tip]], 0.5);
  const speed = att.prevTip && dt > 0 ? tip.distanceTo(att.prevTip) / dt : 0;
  att.prevTip = tip;
  if (atk.done || !atk.end) return;
  const now = att.track.now;
  const D = def.P;
  const head = D[rig.index.Head].clone().lerp(def.track.extra[0], 0.5);
  const dHead = tip.distanceTo(head), dBody = segDist(tip, D[rig.index.Hips], D[rig.index.Neck1]);
  atk.closest = Math.min(atk.closest ?? 9, dHead - 0.125, dBody - 0.17);
  if (now >= atk.end - 0.01) { atk.done = true; trace(att, atk, 'miss'); return; } // whiffed
  if ((now - atk.start) / Math.max(1, atk.end - atk.start) < 0.25) return;
  const dk = def.track.tag.kind;
  if (dk === 'down' || state !== 'fight') return;
  const r = atk.kind === 'kick' ? 0.07 : 0.055;
  const onHead = dHead < 0.125 + r;
  const onBody = dBody < 0.17 + r;
  if (!onHead && !onBody) return;
  atk.done = true;
  trace(att, atk, onHead ? 'head' : 'body');
  const dmg = Math.round(atk.power * Math.max(0.5, Math.min(1.4, 0.45 + speed / 7)) * (onHead ? 1.25 : 1));
  def.hp = Math.max(0, def.hp - dmg);
  def.body.flash = 1;
  hitstop = 0.07;
  if (dk === 'attack') def.track.tag.atk.done = true; // a hit cancels the defender's own strike
  const how = def.hp <= 0 ? (atk.kind === 'kick' || dmg >= 18 ? 'down_ko' : 'down')
    : onHead ? (dmg < 8 ? 'head_light' : dmg >= 18 ? 'head_big' : 'head')
      : atk.kind === 'kick' ? 'body' : 'stomach';
  def.track.request(hitReq(def, att.root, how));
  popText(`${atk.name} ${dmg}${onHead ? '（頭）' : ''}`, att === p1 ? 'p1' : 'p2');
  if (def.hp <= 0) knockout(att);
}

function separate() {
  if (p1.track.tag.kind === 'down' || p2.track.tag.kind === 'down') return;
  const dx = p2.root.x - p1.root.x, dz = p2.root.z - p1.root.z, d = Math.hypot(dx, dz);
  const min = 0.5;
  if (d < min && d > 1e-6) {
    const push = (min - d) / 2;
    p1.track.shift((-dx / d) * push, (-dz / d) * push);
    p2.track.shift((dx / d) * push, (dz / d) * push);
  }
  for (const f of fighters) {
    const r = Math.hypot(f.root.x, f.root.z);
    if (r > RING + 0.2) f.track.shift((-f.root.x / r) * (r - RING - 0.2), (-f.root.z / r) * (r - RING - 0.2));
  }
}

// ---------------------------------------------------------------- rounds

let state = 'wait'; // wait (model not ready) | intro | fight | ko
let stateTime = 0;
let round = 1;

function resetRound() {
  p1.track.reset(new Vector3(-1.1, 0, 0), Math.PI / 2);
  p2.track.reset(new Vector3(1.1, 0, 0), -Math.PI / 2);
  for (const f of fighters) { f.hp = HP; f.prevTip = null; f.stopSent = false; f.think = 0.8; }
  buffered = null;
  state = 'intro';
  stateTime = 0;
  banner(`ROUND ${round}`);
}

function knockout(winner) {
  state = 'ko';
  stateTime = 0;
  winner.wins++;
  banner('K.O.');
  setTimeout(() => { if (state === 'ko') winner.track.request(victoryReq(winner)); }, 700);
}

function banner(text, ms = 1300) {
  const el = $('banner');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(banner.timer);
  if (ms) banner.timer = setTimeout(() => el.classList.remove('show'), ms);
}

function popText(text, who) {
  const el = document.createElement('div');
  el.className = `pop ${who}`;
  el.textContent = text;
  $('pops').appendChild(el);
  setTimeout(() => el.remove(), 1200);
}

function updateRounds(dt) {
  stateTime += dt;
  if (state === 'intro' && stateTime > 1.4) { state = 'fight'; banner('FIGHT', 700); }
  if (state === 'ko' && stateTime > 3.2) {
    if (p1.wins >= 2 || p2.wins >= 2) {
      banner(p1.wins >= 2 ? 'YOU WIN' : 'YOU LOSE', 2200);
      p1.wins = p2.wins = 0;
      round = 1;
      state = 'wait';
      setTimeout(resetRound, 2400);
    } else {
      round++;
      resetRound();
    }
  }
}

// ---------------------------------------------------------------- loop

const STEP = 1 / 60;
let acc = 0, last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;

function simulate(dt) {
  updateRounds(dt);
  if (state === 'fight') {
    playerControl(p1);
    cpuControl(p2, dt);
  }
  for (const f of fighters) {
    const tr = f.track;
    if (scheduler.ready && !tr.busy && tr.tag.kind !== 'down' && tr.remaining / 30 < scheduler.latency + 0.3) tr.request(idleReq(f, 'end'));
  }
  const step = hitstop > 0 ? 0 : dt;
  hitstop = Math.max(0, hitstop - dt);
  for (const f of fighters) f.track.advance(step);
  if (step > 0) { checkHits(p1, p2, step); checkHits(p2, p1, step); }
  separate();
  scheduler.tick([p1.track, p2.track]);
}

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  acc += dt;
  while (acc >= STEP) { simulate(STEP); acc -= STEP; }
  for (const f of fighters) f.body.update(f.P, f.track.extra, dt);
  cam = stage.follow(p1.root, p2.root, dt);
  stage.render();
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; hud(); }
  requestAnimationFrame(frame);
}

function hud() {
  $('hp1').style.width = `${(p1.hp / HP) * 100}%`;
  $('hp2').style.width = `${(p2.hp / HP) * 100}%`;
  $('wins1').textContent = '●'.repeat(p1.wins) + '○'.repeat(2 - p1.wins);
  $('wins2').textContent = '●'.repeat(p2.wins) + '○'.repeat(2 - p2.wins);
  if (scheduler.ready) {
    $('stats').textContent = `${scheduler.backend} · 生成 ${scheduler.lastMs.toFixed(0)} ms（${scheduler.lastBatch} 人分）· 遅延 ${(scheduler.latency * 1000).toFixed(0)} ms · 描画 ${fps.toFixed(0)} fps`;
  }
}

// ---------------------------------------------------------------- model loading

worker.onmessage = ({ data }) => {
  if (data.type === 'progress') {
    const bar = $('progress-bar'), label = $('progress-label');
    if (data.phase === 'download') {
      const mb = x => (x / 1048576).toFixed(0);
      bar.style.width = data.total ? `${(data.loaded / data.total) * 100}%` : '50%';
      label.textContent = `モデルをダウンロード中… ${mb(data.loaded)}${data.total ? ` / ${mb(data.total)}` : ''} MB`;
    } else if (data.phase === 'cache') {
      bar.style.width = '100%';
      label.textContent = 'キャッシュからモデルを読み込み中…';
    } else if (data.phase === 'session') {
      bar.style.width = '100%';
      label.textContent = '推論の準備中（初回はシェーダーのコンパイルで数十秒かかることがあります）…';
    }
  } else if (data.type === 'ready') {
    scheduler.backend = data.backend;
    for (const w of data.warnings) console.warn(w);
    $('progress-label').textContent = `${data.backend} で準備完了。最初の生成を待っています…`;
    warmup();
  } else if (data.type === 'result') {
    if (data.id === -1) {
      $('progress-label').textContent = `${scheduler.backend} · 1 回目 ${data.ms.toFixed(0)} ms。速さを測っています…`;
      warmup(0);
    } else if (data.id === 0) {
      scheduler.latency = Math.max(0.05, data.ms / 1000);
      scheduler.ready = true;
      $('overlay').classList.add('hidden');
      $('stats').textContent = `${scheduler.backend} · 生成 ${data.ms.toFixed(0)} ms`;
      if (data.warn) console.warn(data.warn);
      resetRound();
    } else scheduler.onResult(data);
  } else if (data.type === 'error') {
    if (data.id) scheduler.onFail(data.id);
    $('progress-label').textContent = `エラー: ${data.message}`;
    $('stats').textContent = `エラー: ${data.message}`;
    $('start').disabled = false;
    console.error(data.message);
  }
};

/**
 * Throw-away generations before the first round: the first compiles the shaders, the second (id 0) measures the
 * speed. Both are real requests (both fighters stepping), batched like the game.
 */
function warmup(id = -1) {
  const batch = emptyBatch(BATCH);
  fighters.forEach((f, b) => {
    const ctx = f.track.context(f.track.end);
    const A = ground(ctx[3].root);
    encodeRequest(rig, batch, b, ctx, [placePose(book.stance, yawTo(A, f.opp.root), ZERO, A.clone().multiplyScalar(0.7))], [8, 9, 10]);
  });
  const { n, ...inputs } = batch;
  worker.postMessage({ type: 'run', id, n, inputs }, Object.values(inputs).map(a => a.buffer));
}

$('start').addEventListener('click', () => {
  const choice = document.querySelector('input[name=model]:checked').value;
  const url = new URL(params.get('model') ?? MODELS[choice].url, location.href).href;
  $('start').disabled = true;
  $('progress').classList.add('show');
  worker.postMessage({ type: 'load', url, ep: params.get('ep') ?? 'webgpu', batch: BATCH });
});

if (!navigator.gpu) $('gpu-note').textContent = 'このブラウザでは WebGPU が使えません。WASM（CPU）で動きますが、かなり遅くなります。Chrome / Edge の最新版をおすすめします。';

p1.track.reset(new Vector3(-1.1, 0, 0), Math.PI / 2);
p2.track.reset(new Vector3(1.1, 0, 0), -Math.PI / 2);
requestAnimationFrame(frame);
if (params.has('debug')) window.game = { p1, p2, scheduler, rig, book, traces, stage, simulate, get state() { return state; } };
