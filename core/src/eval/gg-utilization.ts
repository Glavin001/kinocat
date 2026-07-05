// Capability utilization via the friction-circle / g-g diagram (evaluation
// guide §4.2b). The tires deliver combined acceleration only up to a limit:
//
//     sqrt(a_long² + a_lat²) ≤ μ·g
//
// A timid planner clusters its (a_long, a_lat) points near the origin; a
// planner that uses the car pushes them toward the boundary. Reporting mean and
// peak utilization quantifies exactly how much performance the planner leaves
// unused — directly answering "it could go faster / turn sharper".

import type { ReferenceTrajectory } from './reference-trajectory';

export interface GgPoint {
  /** Longitudinal acceleration (m/s²). */
  aLong: number;
  /** Lateral acceleration (m/s²) = v²·κ. */
  aLat: number;
  /** Combined utilization as a fraction of the friction limit. */
  util: number;
}

export interface GgReport {
  /** Mean utilization fraction over all points. */
  meanUtil: number;
  /** Peak utilization fraction. */
  peakUtil: number;
  /** The (a_long, a_lat) cloud, for plotting / regression. */
  cloud: GgPoint[];
}

/** Compute friction-circle utilization for a reference trajectory. */
export function ggUtilization(
  ref: ReferenceTrajectory,
  frictionLimit: number,
): GgReport {
  const limit = Math.max(frictionLimit, 1e-6);
  const cloud: GgPoint[] = new Array(ref.length);
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < ref.length; i++) {
    const p = ref[i]!;
    const aLat = p.v * p.v * Math.abs(p.kappa);
    const aLong = p.a;
    const util = Math.hypot(aLong, aLat) / limit;
    cloud[i] = { aLong, aLat, util };
    sum += util;
    if (util > peak) peak = util;
  }
  return {
    meanUtil: ref.length > 0 ? sum / ref.length : 0,
    peakUtil: peak,
    cloud,
  };
}
