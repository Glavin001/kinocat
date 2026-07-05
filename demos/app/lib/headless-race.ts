// Headless race benchmark — thin Node-friendly wrapper around the shared
// `createRaceScenario` runner in `race-scenario.ts`. Used by:
//   - `pnpm run race` CLI to compare v2 / kinematic / arbitrary-model
//     against each other deterministically.
//   - Phase 3 acceptance gate ("v2 beats kinematic on lap time") as a
//     pass/fail CI signal.
//
// CRITICAL: this file deliberately delegates ALL simulation logic
// (planner call, pure-pursuit, lap detection, stall guard, off-track
// recovery) to `createRaceScenario`. The React `/raceprimitives` page
// is the OTHER consumer of the same module, so CLI lap times match
// what the page produces on the same seed up to physics determinism
// — the single source of truth for "how the race runs".

import {
  createRaceScenario,
  type RaceEntry,
  type RaceLap,
  type RaceTuning,
} from './race-scenario';
import {
  buildLearnedRaceLibraryV2,
  buildKinematicLibrary,
  type RaceCourse,
} from './race-primitives-scenarios';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  type LearnedVehicleModel,
} from 'kinocat/agent';

export type { RaceEntry, RaceLap, RaceTuning } from './race-scenario';
export { DEFAULT_TUNING, LEGACY_TUNING } from './race-scenario';

export interface RaceResult {
  name: string;
  laps: RaceLap[];
  /** Best lap duration (s) or NaN. */
  best: number;
  /** Mean lap duration (s) or NaN. */
  avg: number;
  /** Sample standard deviation of lap durations (s) or NaN. */
  stddev: number;
  /** Total sim time consumed (s). */
  totalSimTime: number;
  /** Did the car complete `targetLaps` within the time budget? */
  finished: boolean;
  /** How many times the chassis left the arena / rolled. */
  offTrackEvents: number;
  /** How many distinct times the chassis struck a course wall (0 on the
   *  open course; the key model-fidelity signal on the technical course). */
  wallStrikes: number;
  /** RMS prediction error at primitive boundary (m). The honest
   *  "how accurate is the model the planner uses" metric. */
  predErrorRms: number;
  /** Total successful + failed replans (planner-quality proxy). */
  totalReplans: number;
  successfulReplans: number;
}

export interface RunRaceOptions {
  entries: RaceEntry[];
  targetLaps?: number;
  /** Max sim seconds before DNF. */
  maxSimTime?: number;
  /** Whether the leader waits at the lap line for the trailer (web demo
   *  default). The CLI default is `false` so a single slow entry doesn't
   *  hold up everyone else. */
  syncHold?: boolean;
  /** Called every `progressEverySec` simulated seconds with a small
   *  status update string (for the CLI progress bar). */
  onProgress?: (msg: string) => void;
  progressEverySec?: number;
  /** Per-feature toggles for ablation studies. Defaults to all-on. */
  tuning?: Partial<RaceTuning>;
  /** Course to race on. Defaults to the open course (`buildRaceCourse()`).
   *  Pass `buildRaceCourse('technical')` for the walled variant. */
  course?: RaceCourse;
}

/** Race every entry against each other in independent Rapier worlds
 *  (one per entry — matches the React demo's split-viewport setup so
 *  cars never physically interact, only the lap timer compares them). */
export async function runHeadlessRace(
  opts: RunRaceOptions,
): Promise<RaceResult[]> {
  const targetLaps = opts.targetLaps ?? 3;
  const maxSimTime = opts.maxSimTime ?? 240;
  const progressEvery = opts.progressEverySec ?? 5;
  const scenario = await createRaceScenario({
    entries: opts.entries,
    targetLaps,
    syncHold: opts.syncHold ?? false,
    offTrackRecovery: 'spawn',
    tuning: opts.tuning,
    course: opts.course,
  });
  let nextProgressAt = progressEvery;
  while (scenario.simTime() < maxSimTime) {
    const r = scenario.tick();
    if (r.allFinished) break;
    if (r.simTime >= nextProgressAt) {
      const progress = r.cars.map((c) =>
        `${c.name}:lap${c.laps.length}/${targetLaps}@wp${c.loopIndex},pos=(${c.state.x.toFixed(1)},${c.state.z.toFixed(1)}),spd=${c.state.speed.toFixed(1)}`,
      ).join(' | ');
      opts.onProgress?.(`t=${r.simTime.toFixed(1)}s ${progress}`);
      nextProgressAt += progressEvery;
    }
  }
  const final = scenario.status();
  const finalSimTime = scenario.simTime();
  scenario.dispose();
  return final.map((c): RaceResult => {
    const durations = c.laps.map((l) => l.duration);
    const best = durations.length > 0 ? Math.min(...durations) : NaN;
    const avg = durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : NaN;
    let stddev = NaN;
    if (durations.length >= 2 && Number.isFinite(avg)) {
      const variance =
        durations.reduce((s, d) => s + (d - avg) * (d - avg), 0) / (durations.length - 1);
      stddev = Math.sqrt(variance);
    }
    return {
      name: c.name,
      laps: c.laps,
      best,
      avg,
      stddev,
      totalSimTime: finalSimTime,
      finished: c.laps.length >= targetLaps,
      offTrackEvents: c.offTrackEvents,
      wallStrikes: c.wallStrikes,
      predErrorRms: c.diagnostics.predErrorRms,
      totalReplans: c.diagnostics.totalReplans,
      successfulReplans: c.diagnostics.successfulReplans,
    };
  });
}

/** Build a kinematic-baseline `RaceEntry`. */
export function kinematicEntry(name = 'kinematic'): RaceEntry {
  return { name, lib: buildKinematicLibrary() };
}

/** Build a v2 `RaceEntry` from a `LearnedVehicleModel`. */
export function v2Entry(name: string, model: LearnedVehicleModel): RaceEntry {
  return { name, lib: buildLearnedRaceLibraryV2(model) };
}

/** Build a parametric-only baseline (no residual ensemble) from the
 *  default params + config. */
export function parametricOnlyEntry(name = 'parametric-only'): RaceEntry {
  const m = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2);
  return { name, lib: buildLearnedRaceLibraryV2(m) };
}
