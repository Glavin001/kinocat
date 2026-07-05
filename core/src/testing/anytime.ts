// Anytime contract: giving the planner MORE budget must never make the
// answer worse. Runs each scenario at an ascending ladder of expansion
// budgets and asserts (a) once a plan is found it stays found at every
// larger budget, and (b) plan cost is non-increasing as the budget grows.

import { plan } from '../planner/ighastar';
import { tol, type ConformanceFailure, type DomainHarness } from './types';

export function checkAnytimeMonotonic<State>(
  h: DomainHarness<State>,
  opts: {
    /** Budget ladder as fractions of each scenario's maxExpansions
     *  (default [1/8, 1/4, 1/2, 1]). */
    fractions?: number[];
  } = {},
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const eps = tol(h);
  const fractions = opts.fractions ?? [1 / 8, 1 / 4, 1 / 2, 1];
  for (const sc of h.scenarios) {
    let prevFound = false;
    let prevCost = Infinity;
    let prevBudget = 0;
    for (const f of fractions) {
      const budget = Math.max(1, Math.ceil(sc.maxExpansions * f));
      const r = plan(
        {
          start: sc.start,
          goal: sc.goal,
          environment: h.makeEnv(),
          options: { maxExpansions: budget },
        },
        Infinity,
      );
      const found = r.found && !r.partial;
      if (prevFound && !found) {
        failures.push({
          check: 'anytime-monotonic',
          message:
            `scenario '${sc.name}': found at budget ${prevBudget} but NOT at ` +
            `larger budget ${budget}`,
        });
      }
      if (prevFound && found && r.cost > prevCost + eps) {
        failures.push({
          check: 'anytime-monotonic',
          message:
            `scenario '${sc.name}': cost worsened ${prevCost} → ${r.cost} as ` +
            `budget grew ${prevBudget} → ${budget}`,
        });
      }
      if (found) {
        prevFound = true;
        prevCost = r.cost;
      }
      prevBudget = budget;
    }
  }
  return failures;
}
