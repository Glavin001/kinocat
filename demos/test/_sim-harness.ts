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
import { createSettleLatch, type SettleState } from 'kinocat/execute';

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
  /**
   * Closed-loop settle semantics (mutually exclusive with `done`): run until
   * `predicate` holds continuously at rest for `holdSeconds`, then keep
   * simulating `postHoldSeconds` more (creep-out after success = violations),
   * then stop. If it never settles, runs the full `maxTicks` budget.
   */
  settle?: {
    predicate: (s: RaceCarStatus) => boolean;
    holdSeconds: number;
    speedTol: number;
    postHoldSeconds: number;
  };
  /** Which car to monitor (default 0). */
  carIndex?: number;
}

export interface MonitoredRun {
  status: RaceCarStatus;
  report: RunReport;
  trajectory: readonly TelemetryRow[];
  ticks: number;
  /** Final settle-latch state (present iff `settle` was requested). */
  settle?: SettleState;
  /** Total replans at the moment the settled hold began (iff settled). */
  replansAtSettle?: number;
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
  const latch = opts.settle
    ? createSettleLatch({
        holdSeconds: opts.settle.holdSeconds,
        speedTol: opts.settle.speedTol,
      })
    : null;
  let ticks = 0;
  let postTicks = 0;
  let replansAtSettle: number | undefined;
  const postTickBudget = opts.settle
    ? Math.round(opts.settle.postHoldSeconds / PHYSICS_DT)
    : 0;
  for (let i = 0; i < opts.maxTicks; i++) {
    scenario.tick();
    const st = scenario.status()[idx]!;
    monitor.sample(st);
    ticks++;
    if (latch && opts.settle) {
      const wasSettled = latch.state.settled;
      latch.update({ ok: opts.settle.predicate(st), speed: st.state.speed }, PHYSICS_DT);
      if (latch.state.settled && !wasSettled) {
        replansAtSettle = st.diagnostics.totalReplans;
      }
      if (latch.state.settled && ++postTicks >= postTickBudget) break;
    }
    if (opts.done?.(st)) break;
  }
  const status = scenario.status()[idx]!;
  const report = monitor.summary();
  const trajectory = monitor.trajectory();
  scenario.dispose();
  return {
    status,
    report,
    trajectory,
    ticks,
    ...(latch ? { settle: latch.state, replansAtSettle } : {}),
  };
}

export { PHYSICS_DT };
