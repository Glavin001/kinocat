// Controller-isolation metrics (evaluation guide §4.1): given a known-good,
// known-feasible reference and the trajectory the controller actually drove,
// quantify the GAP between them — this IS the controller tracking error.
//
// Cross-track and heading error are computed by geometric projection onto the
// reference (projection.ts), not by snapping to stored vertices. Velocity
// error compares the executed speed to the reference speed at the foot point.

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import { angleDiff } from '../internal/math';
import { toReferenceTrajectory, type ReferenceTrajectory } from './reference-trajectory';
import { projectOntoPath } from './projection';
import type { TerminalAccuracy } from './plan-quality';

export interface ErrorStats {
  rmse: number;
  max: number;
}

export interface CrossTrackStats extends ErrorStats {
  /** 95th percentile of |cross-track error| (m). */
  p95: number;
}

export interface TrackingReport {
  /** Perpendicular distance from car to reference path (m). */
  crossTrack: CrossTrackStats;
  /** Car heading vs path heading (rad). */
  heading: ErrorStats;
  /** Executed speed vs reference target speed (m/s). */
  velocity: ErrorStats;
  /** RMS steering rate (rad/s) — 0 when no steer series supplied. */
  steerRateRms: number;
  /** Sign changes of steering rate (deadbanded) — "chatter". */
  steerReversals: number;
  /** Number of executed samples scored. */
  samples: number;
}

function rms(sumSq: number, count: number): number {
  return count > 0 ? Math.sqrt(sumSq / count) : 0;
}

/** Count genuine steering reversals (left→right→left "chatter") in a steer
 *  series, robust to stepwise commands. The old approach counted sign changes of
 *  the per-tick steer RATE, which SATURATES on piecewise-constant commands: a
 *  monotone staircase (steer ramping one way in held steps) flip-flops the rate
 *  sign on every step and every step is miscounted as a reversal. This instead
 *  tracks turning points of the steer VALUE with a magnitude deadband
 *  (hysteresis): a reversal is credited only when the steer swings back from its
 *  last extremum by more than `deadband` radians. A monotone staircase yields 0;
 *  only true oscillations larger than the deadband count. */
export function countSteerReversals(
  steer: ReadonlyArray<number>,
  deadband: number,
): number {
  if (steer.length < 2) return 0;
  let count = 0;
  let dir = 0; // 0 = not yet moving, +1 = rising, −1 = falling
  let pivot = steer[0]!; // value at the last confirmed turning point
  for (let i = 1; i < steer.length; i++) {
    const s = steer[i]!;
    if (dir === 0) {
      if (s > pivot + deadband) { dir = 1; pivot = s; }
      else if (s < pivot - deadband) { dir = -1; pivot = s; }
    } else if (dir === 1) {
      if (s > pivot) pivot = s; // extend the rising run
      else if (s <= pivot - deadband) { count++; dir = -1; pivot = s; } // turned down
    } else {
      if (s < pivot) pivot = s; // extend the falling run
      else if (s >= pivot + deadband) { count++; dir = 1; pivot = s; } // turned up
    }
  }
  return count;
}

/** Default steer-swing deadband (rad ≈ 1.7°) for reversal counting. */
export const DEFAULT_STEER_REVERSAL_DEADBAND = 0.03;

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((x, y) => x - y);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx]!;
}

export interface TrackingMetricsOptions {
  /** Fixed tick dt (s) — used for the steering-rate finite difference. */
  dt: number;
  /** Commanded steering (curvature or angle) per executed sample, for
   *  smoothness. Omit to skip the steer-rate / reversal metrics. */
  steer?: ReadonlyArray<number>;
  /** Steer-swing deadband (rad) for reversal counting — a reversal counts only
   *  when the steer swings back past its last extremum by more than this.
   *  Default `DEFAULT_STEER_REVERSAL_DEADBAND` (0.03 rad). */
  steerReversalDeadband?: number;
}

/** Score an executed trajectory against a reference trajectory. */
export function trackingMetrics(
  executed: ReadonlyArray<CarKinematicState>,
  ref: ReferenceTrajectory,
  opts: TrackingMetricsOptions,
): TrackingReport {
  let ctSumSq = 0;
  let ctMax = 0;
  const ctAbs: number[] = [];
  let hdgSumSq = 0;
  let hdgMax = 0;
  let velSumSq = 0;
  let velMax = 0;

  for (const st of executed) {
    const proj = projectOntoPath(ref, st.x, st.z);
    const ct = Math.abs(proj.crossTrack);
    ctSumSq += ct * ct;
    if (ct > ctMax) ctMax = ct;
    ctAbs.push(ct);

    const hdg = Math.abs(angleDiff(st.heading, proj.psiAtFoot));
    hdgSumSq += hdg * hdg;
    if (hdg > hdgMax) hdgMax = hdg;

    const vel = Math.abs(st.speed - proj.vAtFoot);
    velSumSq += vel * vel;
    if (vel > velMax) velMax = vel;
  }

  // Steering smoothness from the commanded-steer series. steerRateRms is the
  // RMS magnitude of the per-tick steer rate; reversals use the stepwise-robust
  // hysteresis counter (not rate-sign flips, which saturate on held commands).
  let steerRateSumSq = 0;
  let steerReversals = 0;
  let diffCount = 0;
  if (opts.steer && opts.steer.length >= 2) {
    for (let i = 1; i < opts.steer.length; i++) {
      const rate = (opts.steer[i]! - opts.steer[i - 1]!) / opts.dt;
      steerRateSumSq += rate * rate;
      diffCount++;
    }
    steerReversals = countSteerReversals(
      opts.steer,
      opts.steerReversalDeadband ?? DEFAULT_STEER_REVERSAL_DEADBAND,
    );
  }

  const n = executed.length;
  return {
    crossTrack: { rmse: rms(ctSumSq, n), max: ctMax, p95: percentile(ctAbs, 95) },
    heading: { rmse: rms(hdgSumSq, n), max: hdgMax },
    velocity: { rmse: rms(velSumSq, n), max: velMax },
    steerRateRms: rms(steerRateSumSq, diffCount),
    steerReversals,
    samples: n,
  };
}

/** A controller reduced to its essential contract for the isolation test: given
 *  the current state and the reference path, emit a `ForwardSim` control vector
 *  and whether the goal is reached. (Pure-pursuit and MPPI both adapt to this.) */
export type RefController = (
  state: CarKinematicState,
  path: ReadonlyArray<CarKinematicState>,
) => { controls: number[]; steer: number; atGoal: boolean };

export interface ControllerIsolationResult {
  executed: CarKinematicState[];
  steer: number[];
  report: TrackingReport;
  reachedGoal: boolean;
  /** Terminal-state gap to the reference goal (final executed vs reference end).
   *  Distinct from `report.crossTrack` (perpendicular gap ALONG the path): a
   *  controller can hug the corridor yet stop short of the end (`posError`) or
   *  settle at the right spot but the wrong heading (`headingError`, e.g. 16°),
   *  and `speed` is the final |speed| — nonzero at a parking goal means it never
   *  actually settled. These catch the stop-short / wrong-terminal-heading bug
   *  classes cross-track cannot see. Reuses `plan-quality`'s `TerminalAccuracy`
   *  rather than a parallel type. */
  terminal: TerminalAccuracy;
}

/** Run ONLY the controller against a known-good reference (the planner is
 *  removed). Anything that goes wrong is the controller's responsibility. */
export function runControllerIsolation(
  reference: ReadonlyArray<CarKinematicState>,
  controller: RefController,
  sim: ForwardSim<CarKinematicState>,
  dt: number,
  opts?: { maxSteps?: number; start?: CarKinematicState },
): ControllerIsolationResult {
  const ref = toReferenceTrajectory(reference);
  const maxSteps = opts?.maxSteps ?? Math.ceil(reference.length * 4 + 200);
  let state: CarKinematicState = opts?.start ?? { ...reference[0]! };
  const executed: CarKinematicState[] = [{ ...state }];
  const steer: number[] = [];
  let reachedGoal = false;

  for (let step = 0; step < maxSteps; step++) {
    const cmd = controller(state, reference);
    steer.push(cmd.steer);
    if (cmd.atGoal) {
      reachedGoal = true;
      break;
    }
    state = sim(state, cmd.controls, dt);
    executed.push({ ...state });
  }

  const report = trackingMetrics(executed, ref, { dt, steer });
  // Terminal accuracy: final executed pose/speed vs the reference's END state
  // (its intended goal), NOT the projection foot point — so a controller that
  // stops short still reports a large terminal posError even though its
  // cross-track (perpendicular) error is tiny.
  const goal = reference[reference.length - 1]!;
  const finalState = executed[executed.length - 1]!;
  const terminal: TerminalAccuracy = {
    posError: Math.hypot(finalState.x - goal.x, finalState.z - goal.z),
    headingError: Math.abs(angleDiff(finalState.heading, goal.heading)),
    // Final |speed|. References here end at rest (parking cusp / stop), so this
    // is the terminal speed error directly — a car still rolling never settled.
    speed: Math.abs(finalState.speed),
  };
  return { executed, steer, report, reachedGoal, terminal };
}
