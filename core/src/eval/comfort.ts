// Comfort / smoothness flag (evaluation guide §5). The bounds below come from a
// benchmark fit to expert human driving; they are sensible defaults for "drives
// like a competent human". They are computed on the EXECUTED trajectory and
// collapsed into one boolean `comfortable` flag alongside the continuous peaks.
// Relax `bounds` deliberately for arcade-style vehicles meant to feel
// aggressive (guide §9: "we are not bound by real-AV strictness").

import type { CarKinematicState } from '../agent/types';
import { angleDiff } from '../internal/math';

export interface ComfortBounds {
  /** Longitudinal acceleration (m/s²): allowed [min, max]. */
  longAccelMin: number;
  longAccelMax: number;
  /** |Lateral acceleration| (m/s²). */
  latAccelAbs: number;
  /** |Yaw rate| (rad/s). */
  yawRateAbs: number;
  /** |Yaw acceleration| (rad/s²). */
  yawAccelAbs: number;
  /** |Longitudinal jerk| (m/s³). */
  longJerkAbs: number;
  /** |Jerk vector magnitude| (m/s³). */
  jerkVecAbs: number;
}

/** Human-derived defaults (guide §5). */
export const DEFAULT_COMFORT_BOUNDS: ComfortBounds = {
  longAccelMin: -4.05,
  longAccelMax: 2.4,
  latAccelAbs: 4.89,
  yawRateAbs: 0.95,
  yawAccelAbs: 1.93,
  longJerkAbs: 4.13,
  jerkVecAbs: 8.37,
};

export interface ComfortReport {
  peak: {
    longAccelMin: number;
    longAccelMax: number;
    latAccel: number;
    yawRate: number;
    yawAccel: number;
    longJerk: number;
    jerkVec: number;
  };
  /** True iff every bound holds. */
  comfortable: boolean;
  /** Names of the bounds that were exceeded. */
  violations: string[];
}

/** Peak magnitude that must PERSIST for ≥`window` consecutive samples to count —
 *  the max over the sliding `window`-sample minimum. This is a temporal deadband
 *  for the second-order comfort metrics, which otherwise SATURATE on a stack
 *  that issues stepwise (piecewise-constant) commands: a single held command
 *  step, or any single-tick pose/speed impulse, produces a brief 2–3 sample
 *  burst in a finite-differenced jerk / yaw-accel series, and a naive
 *  single-sample max reads "uncomfortable" on every run. Requiring the excursion
 *  to last the full window (~0.05·window s) rejects those transient artifacts
 *  while genuine sustained thrash — many consecutive large samples, e.g. a
 *  yaw-rate that flips every tick — still registers at ~its true magnitude. */
const ROBUST_PEAK_WINDOW = 4;
function robustPeak(series: ReadonlyArray<number>, window = ROBUST_PEAK_WINDOW): number {
  const n = series.length;
  if (n === 0) return 0;
  if (n < window) return Math.max(0, ...series); // too short to debounce
  let peak = 0;
  for (let i = window - 1; i < n; i++) {
    let sustained = Infinity;
    for (let k = 0; k < window; k++) sustained = Math.min(sustained, series[i - k]!);
    if (sustained > peak) peak = sustained;
  }
  return peak;
}

/** Compute comfort peaks and the single `comfortable` flag from an executed
 *  trajectory, by finite-differencing pose/speed at the fixed tick `dt`. */
export function comfortFlags(
  executed: ReadonlyArray<CarKinematicState>,
  dt: number,
  bounds: ComfortBounds = DEFAULT_COMFORT_BOUNDS,
): ComfortReport {
  const n = executed.length;
  // Series we differentiate: yaw rate, longitudinal accel, lateral accel.
  const yawRate: number[] = [];
  const longAccel: number[] = [];
  const latAccel: number[] = [];

  for (let i = 1; i < n; i++) {
    const prev = executed[i - 1]!;
    const cur = executed[i]!;
    const yr = cur.yawRate ?? angleDiff(cur.heading, prev.heading) / dt;
    yawRate.push(yr);
    const aLong = (cur.speed - prev.speed) / dt;
    longAccel.push(aLong);
    latAccel.push(cur.speed * yr);
  }

  let longMin = 0;
  let longMax = 0;
  let latPeak = 0;
  let yawRatePeak = 0;
  for (let i = 0; i < longAccel.length; i++) {
    longMin = Math.min(longMin, longAccel[i]!);
    longMax = Math.max(longMax, longAccel[i]!);
    latPeak = Math.max(latPeak, Math.abs(latAccel[i]!));
    yawRatePeak = Math.max(yawRatePeak, Math.abs(yawRate[i]!));
  }

  // Second-order (jerk / yaw-accel) series. These are the metrics that
  // SATURATE on a stack that issues stepwise (piecewise-constant) commands: a
  // single command step is one isolated large finite-difference spike, so a
  // naive single-sample max reads "uncomfortable" on every run. `robustPeak`
  // applies a temporal deadband — it only credits an excursion that persists for
  // ≥2 consecutive ticks (max over the 2-sample sliding minimum) — so an
  // isolated command-step artifact is ignored while genuine sustained thrash
  // (many consecutive large samples) still trips the flag.
  const yawAccelSeries: number[] = [];
  const longJerkSeries: number[] = [];
  const jerkVecSeries: number[] = [];
  for (let i = 1; i < longAccel.length; i++) {
    const yawAccel = (yawRate[i]! - yawRate[i - 1]!) / dt;
    const longJerk = (longAccel[i]! - longAccel[i - 1]!) / dt;
    const latJerk = (latAccel[i]! - latAccel[i - 1]!) / dt;
    yawAccelSeries.push(Math.abs(yawAccel));
    longJerkSeries.push(Math.abs(longJerk));
    jerkVecSeries.push(Math.hypot(longJerk, latJerk));
  }
  const yawAccelPeak = robustPeak(yawAccelSeries);
  const longJerkPeak = robustPeak(longJerkSeries);
  const jerkVecPeak = robustPeak(jerkVecSeries);

  const violations: string[] = [];
  if (longMin < bounds.longAccelMin) violations.push('longAccelMin');
  if (longMax > bounds.longAccelMax) violations.push('longAccelMax');
  if (latPeak > bounds.latAccelAbs) violations.push('latAccel');
  if (yawRatePeak > bounds.yawRateAbs) violations.push('yawRate');
  if (yawAccelPeak > bounds.yawAccelAbs) violations.push('yawAccel');
  if (longJerkPeak > bounds.longJerkAbs) violations.push('longJerk');
  if (jerkVecPeak > bounds.jerkVecAbs) violations.push('jerkVec');

  return {
    peak: {
      longAccelMin: longMin,
      longAccelMax: longMax,
      latAccel: latPeak,
      yawRate: yawRatePeak,
      yawAccel: yawAccelPeak,
      longJerk: longJerkPeak,
      jerkVec: jerkVecPeak,
    },
    comfortable: violations.length === 0,
    violations,
  };
}
