// Shared guard evaluation — the SINGLE definition of "did this motion satisfy
// this guard", used by both the planner bridge (`ScenarioEnvironment`) and the
// deterministic progress evaluator (`progress.ts`), so the visualizer shows
// exactly what the planner acted on.

import type { Acceptance, ScenarioState } from './types';
import type { GuardPredicate } from './automaton';
import { angleDiff } from '../internal/math';

/** Do the acceptance conjuncts (speed / heading / window / by) hold at `s`? */
export function checkAcceptance(accept: Acceptance | undefined, s: ScenarioState): boolean {
  if (!accept) return true;
  if (accept.speed) {
    if (accept.speed.min !== undefined && s.speed < accept.speed.min) return false;
    if (accept.speed.max !== undefined && s.speed > accept.speed.max) return false;
  }
  if (accept.heading) {
    // Heading band as an arc [min, max] of absolute angles; wrap-aware via the
    // midpoint + half-width formulation.
    const lo = accept.heading.min;
    const hi = accept.heading.max;
    if (lo !== undefined && hi !== undefined) {
      const mid = (lo + hi) / 2;
      const half = Math.abs(hi - lo) / 2;
      if (Math.abs(angleDiff(mid, s.heading)) > half) return false;
    } else if (lo !== undefined) {
      if (s.heading < lo) return false;
    } else if (hi !== undefined) {
      if (s.heading > hi) return false;
    }
  }
  if (accept.window) {
    if (s.t < accept.window[0] || s.t > accept.window[1]) return false;
  }
  if (accept.by !== undefined && s.t > accept.by) return false;
  return true;
}

/** Did the motion from->to satisfy `guard` (spatial membership/crossing AND
 *  acceptance)? Dynamic regions with a `crossed` test use it (catches a fast car
 *  tunneling through a thin region between nodes); otherwise endpoint
 *  membership is evaluated at `to`. */
export function guardSatisfied(
  guard: GuardPredicate,
  from: ScenarioState,
  to: ScenarioState,
): boolean {
  const positional = guard.region.crossed
    ? guard.region.crossed(from, to, from.t) || guard.region.contains(to, to.t)
    : guard.region.contains(to, to.t);
  return positional && checkAcceptance(guard.accept, to);
}
