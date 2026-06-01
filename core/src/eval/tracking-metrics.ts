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
  /** Deadband (rad/s) below which a steer-rate change is noise. Default 0.05. */
  steerRateDeadband?: number;
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

  // Steering smoothness from the commanded-steer series.
  let steerRateSumSq = 0;
  let steerReversals = 0;
  let diffCount = 0;
  const deadband = opts.steerRateDeadband ?? 0.05;
  if (opts.steer && opts.steer.length >= 2) {
    let prevSign = 0;
    for (let i = 1; i < opts.steer.length; i++) {
      const rate = (opts.steer[i]! - opts.steer[i - 1]!) / opts.dt;
      steerRateSumSq += rate * rate;
      diffCount++;
      const sign = rate > deadband ? 1 : rate < -deadband ? -1 : 0;
      if (sign !== 0) {
        if (prevSign !== 0 && sign !== prevSign) steerReversals++;
        prevSign = sign;
      }
    }
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
  return { executed, steer, report, reachedGoal };
}
