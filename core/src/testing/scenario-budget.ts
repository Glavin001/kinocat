// Budgeted solvability: every harness scenario must produce a full (not
// partial) plan within its expansion budget. This is the packaged form of
// the demo-regression pattern — a scenario that stops solving in budget is
// a planner or environment regression, not noise.

import { plan } from '../planner/ighastar';
import type { ConformanceFailure, DomainHarness } from './types';

export function runScenarioBudget<State>(
  h: DomainHarness<State>,
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  for (const sc of h.scenarios) {
    const env = h.makeEnv();
    const [startValid, goalValid] = env.checkValidity(sc.start, sc.goal);
    if (!startValid || !goalValid) {
      failures.push({
        check: 'scenario-budget',
        message:
          `scenario '${sc.name}': checkValidity rejected ` +
          `${!startValid ? 'start' : 'goal'} — fix the scenario or the env`,
        sample: { scenario: sc.name, startValid, goalValid },
      });
      continue;
    }
    const r = plan(
      {
        start: sc.start,
        goal: sc.goal,
        environment: env,
        options: { maxExpansions: sc.maxExpansions },
      },
      Infinity,
    );
    if (!r.found || r.partial) {
      failures.push({
        check: 'scenario-budget',
        message:
          `scenario '${sc.name}': no full plan within ${sc.maxExpansions} ` +
          `expansions (found=${r.found}, partial=${r.partial === true}, ` +
          `expansions=${r.stats.expansions})`,
        sample: { scenario: sc.name, stats: { expansions: r.stats.expansions } },
      });
    }
  }
  return failures;
}
