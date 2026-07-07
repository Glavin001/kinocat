// Worker-backed replan dispatcher for the racing demo — offloads the per-car
// plan compute (`computeReplanArtifactPure`) onto one Web Worker per car.
//
// Shape mirrors core's `PlannerPool` (one worker per agent, a single request
// in flight per car, latest-wins via stale-reqId rejection), but with
// race-specific messages and no world-broadcast (the course geometry is static
// for a race, sent once at init). Browser-only: `createRaceScenario` builds
// this ONLY when the caller passes `spawnPlanWorker`; headless/tests never do,
// so they stay on the deterministic inline dispatcher.

import {
  computeReplanArtifactPure,
  type ReplanDispatcher,
  type ReplanRequest,
  type ReplanArtifact,
  type ReplanComputeCtx,
  type RaceTuning,
} from '../lib/race-scenario';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavPolygon } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import type { VehicleAgent } from 'kinocat/agent';

// ---------------------------------------------------------------------------
// Worker message protocol (also imported by `race.worker.ts`).

/** Per-car init payload the host serialises once and the worker rebuilds its
 *  compute context from. Everything here structure-clones cleanly. */
export interface RaceWorkerCarInit {
  id: string;
  libJSON: string;
  agent: VehicleAgent | undefined;
  tuning: RaceTuning;
  polygons: NavPolygon[];
  obstacles: [number, number][][];
}

export interface RaceWorkerInitMsg {
  type: 'init';
  polygons: NavPolygon[];
  obstacles: [number, number][][];
  agent: VehicleAgent | undefined;
  libJSON: string;
  tuning: RaceTuning;
}

export interface RaceWorkerPlanMsg {
  type: 'plan';
  reqId: number;
  carId: string;
  /** `ReplanRequest` with the non-serializable fields (`rootRollout`,
   *  `singleGoalParams`) stripped by the host before posting. */
  req: ReplanRequest;
}

export type RaceWorkerInbound = RaceWorkerInitMsg | RaceWorkerPlanMsg;

export interface RaceWorkerInitAck {
  type: 'init-ack';
}

export interface RaceWorkerPlanResult {
  type: 'plan-result';
  reqId: number;
  carId: string;
  /** Present on success. */
  artifact?: ReplanArtifact;
  /** Present when the worker compute threw — the host falls back to an inline
   *  compute so a replan is never silently dropped. */
  error?: string;
}

export type RaceWorkerOutbound = RaceWorkerInitAck | RaceWorkerPlanResult;

// ---------------------------------------------------------------------------
// Dispatcher

export interface WorkerReplanDispatcherOptions {
  /** Spawns one fresh module worker. Called once per car. */
  spawnWorker: () => Worker;
  /** Per-car serialised init data (keyed by `id`, which is the car's entry
   *  name — the field `CarInternal` is identified by). */
  cars: RaceWorkerCarInit[];
}

/** The slice of a `CarInternal` this dispatcher touches. Kept structural so the
 *  dispatcher doesn't depend on the (unexported) internal car shape; method
 *  parameters are bivariant, so this satisfies `ReplanDispatcher` whose methods
 *  are typed against the full `CarInternal`. */
interface DispatchCar {
  entry: { name: string };
  pendingArtifact: ReplanArtifact | null;
}

export class WorkerReplanDispatcher implements ReplanDispatcher {
  private readonly workers = new Map<string, Worker>();
  /** Host-side compute context per car — mirrors the worker's. Used for the
   *  non-offloadable inline paths (dynamic root rollout / parking) and as the
   *  fallback when a worker compute throws. */
  private readonly hostCtx = new Map<string, ReplanComputeCtx>();
  /** carId -> reqId currently in flight (single request per car). */
  private readonly inflight = new Map<string, number>();
  /** carId -> ready artifact awaiting a `poll`. */
  private readonly ready = new Map<string, ReplanArtifact>();
  /** carId -> the request last posted (for the worker-error inline fallback). */
  private readonly pendingReq = new Map<string, ReplanRequest>();
  private nextReqId = 0;

  constructor(opts: WorkerReplanDispatcherOptions) {
    for (const car of opts.cars) {
      const world = new InMemoryNavWorld(car.polygons, car.obstacles);
      const lib = MotionPrimitiveLibrary.fromJSON(car.libJSON);
      this.hostCtx.set(car.id, {
        lib,
        agent: car.agent,
        world,
        tuning: car.tuning,
        course: { polygons: car.polygons, obstacles: car.obstacles },
      });
      const w = opts.spawnWorker();
      w.onmessage = (ev: MessageEvent<RaceWorkerOutbound>) =>
        this.onMessage(car.id, ev.data);
      const init: RaceWorkerInitMsg = {
        type: 'init',
        polygons: car.polygons,
        obstacles: car.obstacles,
        agent: car.agent,
        libJSON: car.libJSON,
        tuning: car.tuning,
      };
      w.postMessage(init);
      this.workers.set(car.id, w);
    }
  }

  private onMessage(carId: string, msg: RaceWorkerOutbound): void {
    if (msg.type !== 'plan-result') return; // init-ack: nothing to await here
    // Stale / superseded result — the in-flight id moved on. Drop it.
    if (this.inflight.get(carId) !== msg.reqId) return;
    this.inflight.delete(carId);
    const req = this.pendingReq.get(carId);
    this.pendingReq.delete(carId);
    if (msg.artifact) {
      this.ready.set(carId, msg.artifact);
      return;
    }
    // Worker threw — recover with a host-side inline compute so the car still
    // gets its plan. If even that throws, skip this cycle; the next cadence
    // replan retries from fresher state.
    const ctx = this.hostCtx.get(carId);
    if (ctx && req) {
      try {
        this.ready.set(carId, computeReplanArtifactPure(req, ctx));
      } catch {
        /* give up this cycle */
      }
    }
  }

  dispatch(c: DispatchCar, req: ReplanRequest): void {
    const id = c.entry.name;
    // Not offloadable: dynamic root rollout carries a live closure, and parking
    // uses non-serializable single-goal params (lib/agent/world instances).
    // Compute inline on the host and stash on the car — same contract as
    // InlineDispatcher — so these rare cases still work.
    if (req.rootRollout || req.isParking) {
      const ctx = this.hostCtx.get(id);
      if (ctx) c.pendingArtifact = computeReplanArtifactPure(req, ctx);
      return;
    }
    const w = this.workers.get(id);
    if (!w) return;
    // Strip the non-serializable fields (undefined on the race path anyway) and
    // post to this car's worker. A fresh reqId marks the single in-flight slot;
    // any earlier result is rejected by the stale check (latest-wins).
    const serReq: ReplanRequest = {
      ...req,
      rootRollout: undefined,
      singleGoalParams: undefined,
    };
    const reqId = this.nextReqId++;
    this.inflight.set(id, reqId);
    this.pendingReq.set(id, serReq);
    const plan: RaceWorkerPlanMsg = { type: 'plan', reqId, carId: id, req: serReq };
    w.postMessage(plan);
  }

  poll(c: DispatchCar): ReplanArtifact | null {
    const id = c.entry.name;
    const slot = this.ready.get(id);
    if (slot) {
      this.ready.delete(id);
      return slot;
    }
    // Inline-fallback artifact (dynamic rollout / parking) stashed by dispatch.
    const inline = c.pendingArtifact;
    c.pendingArtifact = null;
    return inline ?? null;
  }

  hasInflight(c: DispatchCar): boolean {
    return this.inflight.has(c.entry.name);
  }

  dispose(): void {
    for (const w of this.workers.values()) w.terminate();
    this.workers.clear();
    this.inflight.clear();
    this.ready.clear();
    this.pendingReq.clear();
    this.hostCtx.clear();
  }
}
