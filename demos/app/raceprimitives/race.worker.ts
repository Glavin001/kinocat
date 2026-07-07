/// <reference lib="webworker" />

// Race plan worker — runs the offloadable plan compute
// (`computeReplanArtifactPure`) off the main thread for ONE car. The host
// (`WorkerReplanDispatcher`) spawns one of these per car and drives it with the
// message protocol defined in `workerReplanDispatcher.ts`. Kept small and
// stateless beyond the per-car compute context built at init; mirrors the shape
// of `carchase.worker.ts`.

import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { computeReplanArtifactPure, type ReplanComputeCtx } from '../lib/race-scenario';
import type { RaceWorkerInbound, RaceWorkerOutbound } from './workerReplanDispatcher';

declare const self: DedicatedWorkerGlobalScope;

// Per-car compute context, built on `init`. The race path is multi-goal, so
// there is no `course.goal` (only polygons + obstacles).
let ctx: ReplanComputeCtx | null = null;

self.onmessage = (e: MessageEvent<RaceWorkerInbound>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    const world = new InMemoryNavWorld(msg.polygons, msg.obstacles);
    const lib = MotionPrimitiveLibrary.fromJSON(msg.libJSON);
    ctx = {
      lib,
      agent: msg.agent,
      world,
      tuning: msg.tuning,
      course: { polygons: msg.polygons, obstacles: msg.obstacles },
    };
    const ack: RaceWorkerOutbound = { type: 'init-ack' };
    self.postMessage(ack);
    return;
  }

  if (msg.type === 'plan') {
    let out: RaceWorkerOutbound;
    try {
      // `ctx` is set by `init`, which the host always sends (and awaits nothing
      // between) before any plan. If a plan somehow races ahead of init, the
      // null deref is caught below and the host falls back to an inline compute.
      const artifact = computeReplanArtifactPure(msg.req, ctx!);
      out = { type: 'plan-result', reqId: msg.reqId, carId: msg.carId, artifact };
    } catch (err) {
      out = {
        type: 'plan-result',
        reqId: msg.reqId,
        carId: msg.carId,
        error: err instanceof Error ? err.message : String(err),
      };
    }
    self.postMessage(out);
  }
};
