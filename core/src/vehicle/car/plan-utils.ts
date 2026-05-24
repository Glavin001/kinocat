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
