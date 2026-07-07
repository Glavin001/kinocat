// Worker-backed MPPI tracker dispatcher for the racing demo — offloads the
// per-car MPPI SOLVE (`mpcTrack`, ~50 ms/tick) onto one Web Worker per car.
//
// Shape mirrors `workerReplanDispatcher.ts`: one worker per car, a single
// solve in flight per car, latest-wins via stale-reqId rejection, and a
// host-side inline fallback if the worker solve throws so a command is never
// silently dropped. The scenario keeps its "hold the latest command between
// solves" cadence unchanged — this only moves WHERE the solve runs, not WHEN
// a fresh command is committed. Browser-only: `createRaceScenario` builds this
// ONLY when the caller passes `spawnTrackerWorker`; headless/tests never do,
// so they stay on the deterministic inline dispatcher.

import {
  mpcTrack,
  type MPCTrackerConfig,
  type MPCTrackerState,
} from 'kinocat/execute';
import type { CarKinematicState } from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import type {
  TrackerDispatcher,
  TrackSolveRequest,
  TrackSolveResult,
  TrackerForwardModelSpec,
} from '../lib/race-scenario';

// ---------------------------------------------------------------------------
// Worker message protocol (also imported by `race.tracker.worker.ts`).

/** Per-car init payload — the worker rebuilds its forward model + persistent
 *  MPPI state from this. Everything structure-clones cleanly (the config
 *  objects are plain; `forwardModelSpec` carries serialised model JSON). */
export interface TrackerWorkerCarInit {
  id: string;
  config: MPCTrackerConfig;
  cuspConfig: MPCTrackerConfig;
  horizonSteps: number;
  forwardModelSpec?: TrackerForwardModelSpec;
}

export interface TrackerWorkerInitMsg {
  type: 'init';
  carId: string;
  config: MPCTrackerConfig;
  cuspConfig: MPCTrackerConfig;
  horizonSteps: number;
  forwardModelSpec?: TrackerForwardModelSpec;
}

export interface TrackerWorkerSolveMsg {
  type: 'solve';
  reqId: number;
  carId: string;
  state: CarKinematicState;
  plan: CarKinematicState[];
  cusp: boolean;
}

export type TrackerWorkerInbound = TrackerWorkerInitMsg | TrackerWorkerSolveMsg;

export interface TrackerWorkerInitAck {
  type: 'init-ack';
}

export interface TrackerWorkerSolveResult {
  type: 'solve-result';
  reqId: number;
  carId: string;
  /** Present on success. */
  result?: TrackSolveResult;
  /** Wall-clock ms the worker spent in `mpcTrack` (for compute accounting). */
  solveMs?: number;
  /** Present when the worker solve threw — the host falls back to an inline
   *  solve so the car still gets a fresh command. */
  error?: string;
}

export type TrackerWorkerOutbound = TrackerWorkerInitAck | TrackerWorkerSolveResult;

// ---------------------------------------------------------------------------
// Dispatcher

export interface WorkerTrackerDispatcherOptions {
  /** Spawns one fresh module worker. Called once per car. */
  spawnWorker: () => Worker;
  /** Per-car init data (keyed by `id`, the car's entry name). */
  cars: TrackerWorkerCarInit[];
}

/** The slice of a `CarInternal` this dispatcher touches. Kept structural so
 *  the dispatcher doesn't depend on the (unexported) internal car shape.
 *  Method params are bivariant, so this satisfies `TrackerDispatcher` whose
 *  methods are typed against the full `CarInternal`. The `mpc*` fields back
 *  the host-side inline fallback when a worker solve throws. */
interface DispatchCar {
  entry: { name: string };
  pendingTrackResult: TrackSolveResult | null;
  mpcState: MPCTrackerState | null;
  mpcForwardSim: ForwardSim<CarKinematicState>;
  mpcConfig: MPCTrackerConfig;
  mpcCuspConfig: MPCTrackerConfig;
  mpcSolveMsTotal: number;
  mpcSolveCount: number;
}

export class WorkerTrackerDispatcher implements TrackerDispatcher {
  private readonly workers = new Map<string, Worker>();
  /** carId -> reqId currently in flight (single solve per car). */
  private readonly inflight = new Map<string, number>();
  /** carId -> ready result awaiting a `poll`. */
  private readonly ready = new Map<string, TrackSolveResult>();
  /** carId -> the request last posted (for the worker-error inline fallback). */
  private readonly pendingReq = new Map<string, TrackSolveRequest>();
  /** carId -> the live car object (for the fallback solve + accounting). */
  private readonly carRef = new Map<string, DispatchCar>();
  private nextReqId = 0;

  constructor(opts: WorkerTrackerDispatcherOptions) {
    for (const car of opts.cars) {
      const w = opts.spawnWorker();
      w.onmessage = (ev: MessageEvent<TrackerWorkerOutbound>) =>
        this.onMessage(car.id, ev.data);
      const init: TrackerWorkerInitMsg = {
        type: 'init',
        carId: car.id,
        config: car.config,
        cuspConfig: car.cuspConfig,
        horizonSteps: car.horizonSteps,
        forwardModelSpec: car.forwardModelSpec,
      };
      w.postMessage(init);
      this.workers.set(car.id, w);
    }
  }

  private onMessage(carId: string, msg: TrackerWorkerOutbound): void {
    if (msg.type !== 'solve-result') return; // init-ack: nothing to await here
    // Stale / superseded result — the in-flight id moved on. Drop it.
    if (this.inflight.get(carId) !== msg.reqId) return;
    this.inflight.delete(carId);
    const req = this.pendingReq.get(carId);
    this.pendingReq.delete(carId);
    const c = this.carRef.get(carId);
    if (msg.result) {
      if (c) {
        c.mpcSolveMsTotal += msg.solveMs ?? 0;
        c.mpcSolveCount += 1;
      }
      this.ready.set(carId, msg.result);
      return;
    }
    // Worker threw — recover with a host-side inline solve so the car still
    // gets a fresh command. Uses the car's OWN forward model + warm-start
    // state (the same ones the inline dispatcher would use). If even that
    // throws, skip this cycle; the car holds its previous command.
    if (c && req && c.mpcState) {
      try {
        const cfg = req.cusp ? c.mpcCuspConfig : c.mpcConfig;
        const t = performance.now();
        const cmdRaw = mpcTrack(req.state, req.plan, c.mpcForwardSim, c.mpcState, cfg);
        c.mpcSolveMsTotal += performance.now() - t;
        c.mpcSolveCount += 1;
        this.ready.set(carId, {
          cmd: {
            steer: cmdRaw.steer,
            driveForce: cmdRaw.driveForce,
            brakeForce: cmdRaw.brakeForce,
          },
          targetSpeed: cmdRaw.targetSpeed,
        });
      } catch {
        /* give up this cycle */
      }
    }
  }

  solve(c: DispatchCar, req: TrackSolveRequest): void {
    const id = c.entry.name;
    const w = this.workers.get(id);
    if (!w) return;
    this.carRef.set(id, c);
    // A fresh reqId marks the single in-flight slot; any earlier result is
    // rejected by the stale check (latest-wins).
    const reqId = this.nextReqId++;
    this.inflight.set(id, reqId);
    this.pendingReq.set(id, req);
    const msg: TrackerWorkerSolveMsg = {
      type: 'solve',
      reqId,
      carId: id,
      state: req.state,
      plan: req.plan,
      cusp: req.cusp,
    };
    w.postMessage(msg);
  }

  poll(c: DispatchCar): TrackSolveResult | null {
    const id = c.entry.name;
    const slot = this.ready.get(id);
    if (slot) {
      this.ready.delete(id);
      return slot;
    }
    return null;
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
    this.carRef.clear();
  }
}
