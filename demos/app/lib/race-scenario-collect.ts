// Closed-loop scenario collection for training — Phase 3 of the training-
// dataset plan. Runs the shared `createRaceScenario` runner with a single
// car (using a caller-supplied primitive library), records every tick's
// (state, controls, next_state), and emits 1-second `Trial`s tagged with
// `scenarioId: 'race-primitives'`.
//
// Workflow: load (or train) a v2 model → race the model on the track,
// recording trials → mix the trials into the next training round →
// retrain. This is the DAgger loop the plan calls the "headline win for
// lap time" — the training distribution becomes the planner's actual
// deployment distribution.
//
// Generic across primitive libraries: pass `buildKinematicLibrary()` to
// collect from the kinematic baseline, pass `buildLearnedRaceLibraryV2(m)`
// to collect from the v2 model driving itself.

import { createRaceScenario } from './race-scenario';
import {
  createScenarioCollector,
  type ScenarioCollector,
} from 'kinocat/training';
import type { Trial } from 'kinocat/learning';
import type {
  CarKinematicState,
  WheeledCarControls,
} from 'kinocat/vehicle/car';
import type { LearnableVehicleConfig } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { deriveLearnableConfig } from 'kinocat/adapters/rapier';
import { DEFAULT_VEHICLE_OPTS } from './training-driver';

export interface CollectFromRaceOptions {
  /** Primitive library the planner uses while collecting. */
  lib: MotionPrimitiveLibrary;
  /** Stop after this many laps. */
  targetLaps?: number;
  /** Safety budget — stop after this much sim time even if laps unmet. */
  maxSimTime?: number;
  /** Window size (s) per emitted trial. Default 1.0. */
  windowSec?: number;
  /** Sample every Nth tick into the emitted samples. Default 6 (matches
   *  the offline-training pipeline's sampleEveryNTicks). */
  sampleEveryNTicks?: number;
  /** Scenario id stamped on every emitted trial. Default
   *  `'race-primitives'`. */
  scenarioId?: string;
  /** Optional progress callback — called every progressEverySec sim
   *  seconds. */
  onProgress?: (msg: string) => void;
  progressEverySec?: number;
}

export interface CollectFromRaceResult {
  trials: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[];
  /** Lap times completed during collection (s). */
  lapTimes: number[];
  /** Total sim time spent collecting (s). */
  simTime: number;
}

/** Run the race scenario with a single car and emit per-window trials. */
export async function collectFromRaceScenario(
  opts: CollectFromRaceOptions,
): Promise<CollectFromRaceResult> {
  const targetLaps = opts.targetLaps ?? 3;
  const maxSimTime = opts.maxSimTime ?? 300;
  const sampleEveryNTicks = opts.sampleEveryNTicks ?? 6;
  const windowSec = opts.windowSec ?? 1.0;
  const scenarioId = opts.scenarioId ?? 'race-primitives';
  const progressEvery = opts.progressEverySec ?? 10;
  const config = deriveLearnableConfig({
    id: 'collect', position: { x: 0, z: 0 }, heading: 0, ...DEFAULT_VEHICLE_OPTS,
  });
  const scenario = await createRaceScenario({
    entries: [{ name: 'collect', lib: opts.lib }],
    targetLaps,
    syncHold: false,
    offTrackRecovery: 'waypoint',
  });
  const collector: ScenarioCollector<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> =
    createScenarioCollector({
      scenarioId,
      dt: 1 / 60,
      sampleEveryNTicks,
      windowSec,
      config,
      configKey: 'rwd-default',
    });
  const trials: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[] = [];
  let nextProgressAt = progressEvery;
  let prevState = scenario.status()[0]!.state;
  while (scenario.simTime() < maxSimTime) {
    const r = scenario.tick();
    const car = r.cars[0]!;
    const t = collector.record(r.simTime, prevState, car.controls, car.state);
    if (t) trials.push(t);
    prevState = car.state;
    if (r.allFinished) break;
    if (r.simTime >= nextProgressAt) {
      opts.onProgress?.(
        `t=${r.simTime.toFixed(1)}s laps=${car.laps.length}/${targetLaps} trials=${trials.length}`,
      );
      nextProgressAt += progressEvery;
    }
  }
  const final = scenario.status()[0]!;
  const flushed = collector.flush();
  if (flushed) trials.push(flushed);
  scenario.dispose();
  return {
    trials,
    lapTimes: final.laps.map((l) => l.duration),
    simTime: scenario.simTime(),
  };
}
