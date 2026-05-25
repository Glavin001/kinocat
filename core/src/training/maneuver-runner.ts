// Generic "run a driver against a body for N seconds, emit a Trial" runner.
//
// Sits on top of `runTrial` from `kinocat/scene`. The car-specific maneuver
// factories (in `kinocat/vehicle/car/maneuvers.ts`) produce
// `Driver<CarKinematicState, WheeledCarControls>` instances; this module
// wraps the run-trial output into the `Trial<S, C, Cfg>` shape the
// training pipeline + coverage meter consume.

import type { Body } from '../scene/body';
import type { Driver } from '../scene/driver';
import { runTrial } from '../scene/run-trial';
import type { Trial, TrialSplit } from '../learning/trial-store';
import { assignSplit } from '../learning/trial-store';

export interface ManeuverRunOptions<S, C, Cfg> {
  /** Initial state for the body (teleported before the run). */
  initialState: S;
  /** Per-tick physics dt. */
  dt: number;
  /** Total number of physics ticks. */
  steps: number;
  /** Record every Nth state into `samples`. Minimum 1. */
  sampleEveryNTicks: number;
  /** Trial id (must be stable across re-runs for split assignment). */
  id: string;
  /** Vehicle config to attach to the trial. */
  config: Cfg;
  /** Stable config key for grouping. */
  configKey: string;
  /** Optional metadata for coverage / split. */
  maneuverId?: string;
  maneuverParams?: Record<string, number>;
  scenarioId?: string;
  terrainKind?: string;
  /** Optional explicit split override. When omitted, hash-based assignment
   *  via `assignSplit` is applied. */
  split?: TrialSplit;
}

/** Run a `Driver` against a `Body` for the requested duration and emit a
 *  `Trial<S, C, Cfg>` ready to add to a `TrialStore`. */
export function runManeuver<S, C, Cfg>(
  body: Body<S, C>,
  driver: Driver<S, C>,
  opts: ManeuverRunOptions<S, C, Cfg>,
): Trial<S, C, Cfg> {
  const sampleEvery = Math.max(1, opts.sampleEveryNTicks);
  const result = runTrial(body, driver, {
    dt: opts.dt,
    steps: opts.steps,
    initialState: opts.initialState,
  });
  // states[0] is initial; states[i] is BEFORE controls[i] applied.
  // Final state after all controls is states[opts.steps].
  const samples: { t: number; state: S }[] = [];
  for (let i = 0; i <= opts.steps; i += sampleEvery) {
    if (i < result.states.length) {
      samples.push({ t: i * opts.dt, state: result.states[i]! });
    }
  }
  const trial: Trial<S, C, Cfg> = {
    id: opts.id,
    initialState: result.states[0]!,
    controlsTrace: result.controls,
    dt: opts.dt,
    samples,
    config: opts.config,
    configKey: opts.configKey,
    maneuverId: opts.maneuverId,
    maneuverParams: opts.maneuverParams,
    scenarioId: opts.scenarioId,
    terrainKind: opts.terrainKind,
  };
  trial.split = opts.split ?? assignSplit(trial);
  return trial;
}
