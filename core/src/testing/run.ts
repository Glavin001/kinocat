// The whole battery in one call. `runConformance(harness)` is what "this
// domain works" means: a green report says the environment satisfies every
// contract the planner assumes. Test-runner integration is one line:
//
//   const report = runConformance(harness);
//   expect(report.failures).toEqual([]);   // prints the failures on red
//
// Individual checks are exported for à-la-carte use (e.g. skip the
// monotone-time check for a domain with legitimate zero-duration edges).

import { checkAnytimeMonotonic } from './anytime';
import { checkDeterminism } from './determinism';
import { checkSuccessorFidelity } from './fidelity';
import { checkHeuristicAdmissible, checkHeuristicConsistency } from './heuristic';
import { runScenarioBudget } from './scenario-budget';
import { checkNodeStability, checkSuccessorInvariants } from './successors';
import type {
  CheckOptions,
  ConformanceReport,
  DomainHarness,
} from './types';

export function runConformance<State>(
  h: DomainHarness<State>,
  opts: CheckOptions = {},
): ConformanceReport {
  const checks: Array<[string, () => ReturnType<typeof runScenarioBudget>]> = [
    ['heuristic-consistency', () => checkHeuristicConsistency(h, opts)],
    ['heuristic-admissible', () => checkHeuristicAdmissible(h)],
    ['successor-invariants', () => checkSuccessorInvariants(h, opts)],
    ['node-stability', () => checkNodeStability(h, opts)],
    ['determinism', () => checkDeterminism(h)],
    ['anytime-monotonic', () => checkAnytimeMonotonic(h)],
    ['scenario-budget', () => runScenarioBudget(h)],
  ];
  // Opt-in: runs only when the harness supplies resimulation hooks.
  if (h.fidelity) {
    checks.push(['successor-fidelity', () => checkSuccessorFidelity(h, opts)]);
  }
  const report: ConformanceReport = { ok: true, checks: [], failures: [] };
  for (const [name, fn] of checks) {
    report.checks.push(name);
    report.failures.push(...fn());
  }
  report.ok = report.failures.length === 0;
  return report;
}
