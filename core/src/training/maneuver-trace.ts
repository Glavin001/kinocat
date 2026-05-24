// Pre-roll a `Driver<S, C>` against time-only (no body) to produce a
// controls trace consumable by the headless trial harness. Most maneuvers
// in `kinocat/vehicle/car/maneuvers` are state-independent (OU random walk,
// transition probes, identification maneuvers, …) so this trick yields the
// same trace whether driven open-loop or closed-loop and lets the existing
// trial harness consume them without changes.
//
// Closed-loop maneuvers (PlanFollower) require a real body and should not
// be pre-rolled; use `runManeuver(body, driver, opts)` for those.

import type { Driver } from '../scene/driver';

export interface BuildControlsTraceOptions<S> {
  /** Initial state passed to every `driver.sample` call. */
  state: S;
  /** Tick interval (s). */
  dt: number;
  /** Number of ticks to generate. */
  steps: number;
}

export function buildControlsTrace<S, C>(
  driver: Driver<S, C>,
  opts: BuildControlsTraceOptions<S>,
): C[] {
  const trace: C[] = [];
  driver.reset?.();
  for (let i = 0; i < opts.steps; i++) {
    trace.push(driver.sample(opts.state, i * opts.dt, opts.dt));
  }
  return trace;
}
