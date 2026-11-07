// A fighter's motion track (generated frames being played) and the batched MotionBricks scheduler.
import { Vector3 } from 'three';
import { FPS, clonePose, lerpPose, placePose, emptyBatch, encodeRequest, decodeRequest, correctEndpoints } from './soma.js';

const LOCK = new Set(['attack', 'hit', 'down']);
const PRIORITY = { idle: 0, move: 1, attack: 2, hit: 3, down: 4 };
const ZERO = new Vector3();
export const BATCH = 2; // the model session is built for this fixed batch size

export class Track {
  constructor(rig, book) {
    this.rig = rig;
    this.book = book;
    this.frames = [];
    this.base = 0; // absolute index of frames[0]
    this.t = 0; // playhead, index into frames (float)
    this.pending = null; // request waiting for the scheduler
    this.inflight = null; // request being generated
    this.serial = 0; // bumped to discard the in-flight result
    this.blend = null;
    this.display = clonePose(book.stance);
    this.P = rig.positions(this.display);
    this.extra = rig.extraPositions(this.display, this.P);
  }

  reset(pos, yaw) {
    const pose = placePose(this.book.stance, yaw, ZERO, new Vector3(pos.x, 0, pos.z));
    this.frames = [0, 1, 2, 3].map(() => ({ pose: clonePose(pose), rate: 1, tag: { kind: 'idle' } }));
    this.base = 0;
    this.t = 3;
    this.pending = null;
    this.inflight = null;
    this.serial++;
    this.blend = null;
    this.advance(0);
  }

  get end() { return this.base + this.frames.length - 1; }
  get now() { return this.base + this.t; }
  frame(abs) { return this.frames[Math.max(0, Math.min(this.frames.length - 1, Math.round(abs) - this.base))]; }
  get tag() { return this.frames[Math.min(this.frames.length - 1, Math.floor(this.t))].tag; }
  get remaining() { return this.frames.length - 1 - this.t; }
  get locked() { return LOCK.has(this.tag.kind) && (this.remaining > 0.01 || this.tag.kind === 'down'); }
  get busy() { return !!(this.pending || this.inflight); }

  advance(dt) {
    const n = this.frames.length;
    const i = Math.min(n - 1, Math.floor(this.t));
    this.t = Math.min(n - 1, this.t + dt * FPS * this.frames[i].rate);
    const i0 = Math.floor(this.t), i1 = Math.min(n - 1, i0 + 1);
    lerpPose(this.frames[i0].pose, this.frames[i1].pose, this.t - i0, this.display);
    if (this.blend) {
      this.blend.w -= dt / 0.18;
      if (this.blend.w <= 0) this.blend = null;
      else lerpPose(this.display, this.blend.from, this.blend.w * this.blend.w, this.display);
    }
    this.rig.positions(this.display, this.P);
    this.extra = this.rig.extraPositions(this.display, this.P);
    if (this.t > 160 && !this.inflight) { // forget old frames
      const drop = Math.floor(this.t) - 8;
      this.frames.splice(0, drop);
      this.base += drop;
      this.t -= drop;
    }
  }

  /** Shift the playing and future frames (push apart, arena walls). */
  shift(dx, dz) {
    const from = Math.max(0, Math.floor(this.t) - 4);
    for (let i = from; i < this.frames.length; i++) { this.frames[i].pose.root.x += dx; this.frames[i].pose.root.z += dz; }
    this.display.root.x += dx; this.display.root.z += dz;
    for (const v of this.P) { v.x += dx; v.z += dz; }
    for (const v of this.extra) { v.x += dx; v.z += dz; }
  }

  request(req) {
    if (this.pending && PRIORITY[this.pending.kind] > PRIORITY[req.kind]) return false;
    if (req.interrupt) this.serial++;
    this.pending = req;
    return true;
  }

  /** Absolute frame where the new motion will start: far enough ahead to cover the inference time. */
  anchor(req, latSec) {
    if (req.mode === 'end') return this.end;
    const rate = this.frames[Math.min(this.frames.length - 1, Math.floor(this.t))].rate;
    let a = Math.ceil(this.now + latSec * FPS * rate) + 1;
    if (!req.interrupt) while (a < this.end && LOCK.has(this.frame(a).tag.kind)) a++;
    return Math.min(a, this.end);
  }

  context(a) { return [a - 3, a - 2, a - 1, a].map(i => this.frame(i).pose); }

  splice(a, newFrames) {
    const idx = a - this.base;
    if (idx < 0) return false;
    const before = clonePose(this.display);
    this.frames = this.frames.slice(0, idx + 1).concat(newFrames.slice(1));
    if (this.t > idx + 0.01) this.blend = { from: before, w: 1 };
    this.splices = (this.splices ?? 0) + 1; // (counted for the evaluator's marks)
    this.t = Math.min(this.t, this.frames.length - 1);
    return true;
  }
}

/** Sends every waiting request in one batch to the worker, then splices the generated motion into each track. */
export class Scheduler {
  constructor(rig, worker) {
    this.rig = rig;
    this.worker = worker;
    this.busy = false;
    this.ready = false;
    this.latency = 0.3; // seconds, smoothed
    this.lastMs = 0;
    this.lastBatch = 0;
    this.runs = 0;
    this.nextId = 1;
    this.jobs = new Map();
    this.onError = null;
  }

  tick(tracks) {
    if (!this.ready || this.busy) return;
    const ready = tracks.filter(t => t.pending);
    if (!ready.length) return;
    const lat = this.latency * 1.15 + 0.02;
    const batch = emptyBatch(BATCH);
    const metas = ready.slice(0, BATCH).map((track, b) => {
      const req = track.pending;
      track.pending = null;
      track.inflight = req;
      const a = track.anchor(req, lat);
      // a request may start from something other than the recorded motion (e.g. the pose physics pushed it into)
      const ctx = req.context ? req.context(track, a) : track.context(a);
      const spec = req.build(track, a, ctx);
      const align = encodeRequest(this.rig, batch, b, ctx, spec.targets, spec.tokens);
      return { track, req, serial: track.serial, a, ctx, spec, align };
    });
    for (let b = metas.length; b < BATCH; b++) encodeRequest(this.rig, batch, b, metas[0].ctx, metas[0].spec.targets, metas[0].spec.tokens); // padding
    const id = this.nextId++;
    this.jobs.set(id, { metas, t0: performance.now() });
    this.busy = true;
    const { n, ...inputs } = batch;
    this.worker.postMessage({ type: 'run', id, n, inputs }, Object.values(inputs).map(a => a.buffer));
  }

  onResult({ id, motions, tokens, ms }) {
    const job = this.jobs.get(id);
    this.jobs.delete(id);
    this.busy = false;
    if (!job) return;
    const wall = (performance.now() - job.t0) / 1000;
    this.latency = this.runs++ < 2 ? wall : this.latency * 0.8 + wall * 0.2;
    this.lastMs = ms;
    this.lastBatch = job.metas.length;
    job.metas.forEach((m, b) => {
      if (m.track.inflight === m.req) m.track.inflight = null;
      if (m.track.serial !== m.serial) return; // superseded (hit, reset)
      const dec = decodeRequest(this.rig, motions, tokens, b, m.align);
      if (!dec.frames.length || !dec.frames.every(f => Number.isFinite(f.root.x) && f.rot.every(q => Number.isFinite(q.w)))) return; // degenerate output
      const fixed = correctEndpoints(this.rig, dec.frames, m.ctx[3], m.spec.targets[0], m.spec.fixEnd ?? true);
      const frames = fixed.map(pose => ({ pose, rate: m.spec.rate, tag: m.spec.tag }));
      if (m.track.splice(m.a, frames)) m.spec.after?.(m.track, m.a, frames);
    });
  }

  onFail(id) {
    const job = this.jobs.get(id);
    this.jobs.delete(id);
    this.busy = false;
    for (const m of job?.metas ?? []) if (m.track.inflight === m.req) m.track.inflight = null;
  }
}
