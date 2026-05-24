// Plan polyline utilities for the car domain.
//
// `trimPlan` discards plan samples whose `t` is already in the past (relative
// to `elapsed`), keeping only the tail the pure-pursuit tracker still cares
// about. Used by every demo that follows a kinocat plan with a real chassis.

import type { CarKinematicState } from './types';

/** Drop plan samples whose `t` is <= `elapsed`. Keeps at least one sample
 *  (the goal pose) so downstream code can always read a non-empty path. */
export function trimPlan<S extends { t: number }>(
  plan: ReadonlyArray<S>,
  elapsed: number,
): S[] {
  if (plan.length === 0) return [];
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
}

/** Convenience alias when the elements are `CarKinematicState`. */
export function trimCarPlan(plan: ReadonlyArray<CarKinematicState>, elapsed: number): CarKinematicState[] {
  return trimPlan(plan, elapsed);
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/** Linearly interpolate a CarKinematicState plan at relative time `t`
 *  (seconds from the plan's start sample). Used by commit-window plan
 *  stitching to compute the predicted future state to replan from.
 *  Clamps to the first/last sample if `t` is out of range. */
export function samplePlanAt(
  plan: ReadonlyArray<CarKinematicState>,
  t: number,
): CarKinematicState | null {
  if (plan.length === 0) return null;
  if (plan.length === 1 || t <= plan[0]!.t) return { ...plan[0]! };
  const last = plan[plan.length - 1]!;
  if (t >= last.t) return { ...last };
  // Binary-search for the bracket [i, i+1] containing t.
  let lo = 0;
  let hi = plan.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (plan[mid]!.t <= t) lo = mid;
    else hi = mid;
  }
  const a = plan[lo]!;
  const b = plan[hi]!;
  const dt = b.t - a.t;
  const u = dt > 1e-9 ? (t - a.t) / dt : 0;
  // Heading is interpolated on the shorter arc.
  const dh = wrapPi(b.heading - a.heading);
  return {
    x: a.x + (b.x - a.x) * u,
    z: a.z + (b.z - a.z) * u,
    heading: wrapPi(a.heading + dh * u),
    speed: a.speed + (b.speed - a.speed) * u,
    t,
  };
}
