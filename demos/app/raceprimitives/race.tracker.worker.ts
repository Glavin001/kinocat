/// <reference lib="webworker" />

// Race MPPI tracker worker — runs the per-car MPPI SOLVE (`mpcTrack`, the
// ~50 ms/tick cost) off the main thread for ONE car. The host
// (`WorkerTrackerDispatcher`) spawns one of these per car and drives it with
// the message protocol in `trackerWorkerHost.ts`. Small and stateful only in
// the per-car forward model + configs + a persistent `MPCTrackerState` (so the
// MPPI warm-start survives across solves, exactly like the inline path).

import {
  mpcTrack,
  createMPCTrackerState,
  type MPCTrackerConfig,
  type MPCTrackerState,
} from 'kinocat/execute';
import {
  parametricForwardV2,
  learnedForwardSimV2,
  forwardSimV3Rollout,
  v3FromJson,
  KINEMATIC_NATIVE_PARAMS,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  type CarKinematicState,
} from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import { modelFromJson } from '../lib/v2-model-file';
import type {
  TrackerWorkerInbound,
  TrackerWorkerOutbound,
} from './trackerWorkerHost';
import type { TrackerForwardModelSpec } from '../lib/race-scenario';

declare const self: DedicatedWorkerGlobalScope;

// Per-car state, built on `init`.
let forwardSim: ForwardSim<CarKinematicState> | null = null;
let config: MPCTrackerConfig | null = null;
let cuspConfig: MPCTrackerConfig | null = null;
let mpcState: MPCTrackerState | null = null;

/** Rebuild the forward dynamics model from its spec — mirrors `resolveLib`'s
 *  model → forward-sim mapping in RacePrimitives so the worker rolls the SAME
 *  model the car would inline. Missing spec ⇒ the scenario's shared v2-default
 *  parametric backbone (the inline default when an entry ships no model). */
function buildForwardSim(
  spec: TrackerForwardModelSpec | undefined,
): ForwardSim<CarKinematicState> {
  if (!spec) {
    return parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
  }
  if (spec.kind === 'v3' && spec.modelJson) {
    return forwardSimV3Rollout(v3FromJson(JSON.parse(spec.modelJson)));
  }
  if (spec.kind === 'v2' && spec.modelJson) {
    return learnedForwardSimV2(modelFromJson(JSON.parse(spec.modelJson)));
  }
  // kinematic (and any spec missing its JSON) → the naive idealised-bicycle
  // parametric model.
  return parametricForwardV2(KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG);
}

self.onmessage = (e: MessageEvent<TrackerWorkerInbound>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    forwardSim = buildForwardSim(msg.forwardModelSpec);
    config = msg.config;
    cuspConfig = msg.cuspConfig;
    // Persistent MPPI state — warm-start sequence + deterministic RNG seed —
    // held across solves so warm-start continuity matches the inline path.
    mpcState = createMPCTrackerState(msg.horizonSteps);
    const ack: TrackerWorkerOutbound = { type: 'init-ack' };
    self.postMessage(ack);
    return;
  }

  if (msg.type === 'solve') {
    let out: TrackerWorkerOutbound;
    try {
      // `init` always precedes any solve (the host awaits nothing between, but
      // posts init first), so these are set. A null deref is caught below and
      // the host falls back to an inline solve.
      const cfg = msg.cusp ? cuspConfig! : config!;
      const t = performance.now();
      const cmdRaw = mpcTrack(msg.state, msg.plan, forwardSim!, mpcState!, cfg);
      const solveMs = performance.now() - t;
      out = {
        type: 'solve-result',
        reqId: msg.reqId,
        carId: msg.carId,
        result: {
          cmd: {
            steer: cmdRaw.steer,
            driveForce: cmdRaw.driveForce,
            brakeForce: cmdRaw.brakeForce,
          },
          targetSpeed: cmdRaw.targetSpeed,
        },
        solveMs,
      };
    } catch (err) {
      out = {
        type: 'solve-result',
        reqId: msg.reqId,
        carId: msg.carId,
        error: err instanceof Error ? err.message : String(err),
      };
    }
    self.postMessage(out);
  }
};
