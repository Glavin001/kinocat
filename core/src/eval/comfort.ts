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

  let yawAccelPeak = 0;
  let longJerkPeak = 0;
  let jerkVecPeak = 0;
  for (let i = 1; i < longAccel.length; i++) {
    const yawAccel = (yawRate[i]! - yawRate[i - 1]!) / dt;
    yawAccelPeak = Math.max(yawAccelPeak, Math.abs(yawAccel));
    const longJerk = (longAccel[i]! - longAccel[i - 1]!) / dt;
    const latJerk = (latAccel[i]! - latAccel[i - 1]!) / dt;
    longJerkPeak = Math.max(longJerkPeak, Math.abs(longJerk));
    jerkVecPeak = Math.max(jerkVecPeak, Math.hypot(longJerk, latJerk));
  }

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
