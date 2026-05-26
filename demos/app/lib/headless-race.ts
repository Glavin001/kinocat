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
  type ReplanReason,
  type ReplanSnapshot,
} from './race-scenario';
import {
  buildLearnedRaceLibraryV2,
  buildKinematicLibrary,
} from './race-primitives-scenarios';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  type LearnedVehicleModel,
} from 'kinocat/agent';

export type { RaceEntry, RaceLap, RaceTuning, ReplanReason, ReplanSnapshot } from './race-scenario';
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
  /** RMS prediction error at primitive boundary (m). The honest
   *  "how accurate is the model the planner uses" metric. */
  predErrorRms: number;
  /** Total successful + failed replans (planner-quality proxy). */
  totalReplans: number;
  successfulReplans: number;
  /** Per-trigger replan counts (`cadence` | `lateral-error` | `waypoint-advance`
   *  | `failure-retry` | `manual`). Sum equals the size of the per-car
   *  replanHistory ring buffer (capped at 30 — older replans are dropped). */
  replanReasonCounts: Record<ReplanReason, number>;
  /** Mean planner search time across the captured replanHistory window (ms). */
  plannerMsMean: number;
  /** Max planner search time across the captured replanHistory window (ms). */
  plannerMsMax: number;
  /** Number of replans where the planner hit its deadline (cumulative). */
  plannerDeadlineHits: number;
  /** Ticks where commanded steering exceeded 75% of `minTurnRadius`
   *  curvature — useful to ask "did the controller wrench the wheel?" */
  sharpSteerTicks: number;
  /** Most recent replan snapshots (ring buffer; newest last; max 30). Lets
   *  downstream tooling drill into HOW the planner reacted at each point —
   *  reason, expansions, deadline-hit, plan vs prev-plan drift. */
  replanHistory: ReplanSnapshot[];
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
  });
  let nextProgressAt = progressEvery;
  while (scenario.simTime() < maxSimTime) {
    const r = scenario.tick();
    if (r.allFinished) break;
    if (r.simTime >= nextProgressAt) {
      const progress = r.cars.map((c) => {
        const ctrl = c.metrics.liveControls;
        const thr = ctrl ? `thr=${(ctrl.throttle * 100).toFixed(0)}%` : '';
        const brk = ctrl ? `brk=${(ctrl.brake * 100).toFixed(0)}%` : '';
        return `${c.name}:lap${c.laps.length}/${targetLaps}@wp${c.loopIndex},spd=${c.state.speed.toFixed(1)},peak=${c.metrics.peakSpeed.toFixed(1)},${thr},${brk}`;
      }).join(' | ');
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
    // Pull TOTAL counts from the per-car diagnostics (not the
    // replanHistory ring buffer — that one is capped at 30 entries and
    // would under-report on long runs). Ring buffer is still surfaced
    // separately for replan-by-replan drill-down.
    const replanReasonCounts = { ...c.diagnostics.replanReasonTotals };
    const plannerMsMean = c.diagnostics.totalReplans > 0
      ? c.diagnostics.plannerMsTotal / c.diagnostics.totalReplans
      : 0;
    const plannerMsMax = c.diagnostics.plannerMsMax;
    const plannerDeadlineHits = c.diagnostics.plannerDeadlineHitsTotal;
    const sharpSteerTicks = c.diagnostics.sharpSteerTicks;
    return {
      name: c.name,
      laps: c.laps,
      best,
      avg,
      stddev,
      totalSimTime: finalSimTime,
      finished: c.laps.length >= targetLaps,
      offTrackEvents: c.offTrackEvents,
      predErrorRms: c.diagnostics.predErrorRms,
      totalReplans: c.diagnostics.totalReplans,
      successfulReplans: c.diagnostics.successfulReplans,
      replanReasonCounts,
      plannerMsMean,
      plannerMsMax,
      plannerDeadlineHits,
      sharpSteerTicks,
      replanHistory: c.replanHistory,
    };
  });
}

/** Build a kinematic-baseline `RaceEntry`. */
export function kinematicEntry(name = 'kinematic'): RaceEntry {
  return { name, lib: buildKinematicLibrary() };
}

/** Build a v2 `RaceEntry` from a `LearnedVehicleModel`. The model is
 *  stored on the entry so the MPC tracker can use its dynamics for
 *  plan-following (aligning execution with planning). */
export function v2Entry(name: string, model: LearnedVehicleModel): RaceEntry {
  return { name, lib: buildLearnedRaceLibraryV2(model), model };
}

/** Build a parametric-only baseline (no residual ensemble) from the
 *  default params + config. */
export function parametricOnlyEntry(name = 'parametric-only'): RaceEntry {
  const m = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2);
  return { name, lib: buildLearnedRaceLibraryV2(m), model: m };
}
