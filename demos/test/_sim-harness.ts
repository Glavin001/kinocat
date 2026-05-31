// Shared headless-run helper for the scenario invariant + determinism tests.
// Drives the SAME `createRaceScenario` engine the web pages use, attaches the
// telemetry monitor, and returns the per-car status + RunReport + trajectory.
// Not a test file (no `.test.ts` suffix) so vitest ignores it.

import {
  createRaceScenario,
  PHYSICS_DT,
  type RaceScenarioOptions,
  type RaceCarStatus,
} from '../app/lib/race-scenario';
import {
  createSimMonitor,
  type RunReport,
  type TelemetryRow,
  type MonitorGoal,
  type SuccessTolerances,
  type Pt,
} from '../app/lib/sim-monitor';

export interface MonitoredRunOpts {
  /** Options passed straight to `createRaceScenario`. */
  scenario: RaceScenarioOptions;
  /** Body-local footprint polygon (e.g. PARKING_AGENT.footprint). */
  footprint: ReadonlyArray<Pt>;
  /** Static obstacle polygons (course.obstacles). Empty ⇒ clearance Infinity. */
  obstacles: ReadonlyArray<ReadonlyArray<readonly [number, number]>>;
  /** Optional goal pose for progress + success diagnostics. */
  goal?: MonitorGoal;
  /** Tolerances for the `parkedOk` flag. */
  success?: SuccessTolerances;
  /** Hard cap on physics ticks. */
  maxTicks: number;
  /** Optional early-stop predicate evaluated on each tick's status. */
  done?: (s: RaceCarStatus) => boolean;
  /** Which car to monitor (default 0). */
  carIndex?: number;
}

export interface MonitoredRun {
  status: RaceCarStatus;
  report: RunReport;
  trajectory: readonly TelemetryRow[];
  ticks: number;
}

export async function runMonitored(opts: MonitoredRunOpts): Promise<MonitoredRun> {
  const scenario = await createRaceScenario(opts.scenario);
  const monitor = createSimMonitor({
    footprint: opts.footprint,
    obstacles: opts.obstacles,
    dt: PHYSICS_DT,
    goal: opts.goal,
    success: opts.success,
  });
  const idx = opts.carIndex ?? 0;
  let ticks = 0;
  for (let i = 0; i < opts.maxTicks; i++) {
    scenario.tick();
    const st = scenario.status()[idx]!;
    monitor.sample(st);
    ticks++;
    if (opts.done?.(st)) break;
  }
  const status = scenario.status()[idx]!;
  const report = monitor.summary();
  const trajectory = monitor.trajectory();
  scenario.dispose();
  return { status, report, trajectory, ticks };
}

export { PHYSICS_DT };
