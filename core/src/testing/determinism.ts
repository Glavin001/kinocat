// Cross-instance determinism: two independently-constructed environments
// given the identical request must produce bit-identical plans. This is the
// property that makes replanning stable, recorded scenarios replayable, and
// CI regressions attributable. Uses an expansion budget (not wall-clock) so
// the check itself is deterministic.

import { plan } from '../planner/ighastar';
import {
  statesClose,
  tol,
  type ConformanceFailure,
  type DomainHarness,
} from './types';

export function checkDeterminism<State>(
  h: DomainHarness<State>,
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const eps = tol(h);
  for (const sc of h.scenarios) {
    const run = () =>
      plan(
        {
          start: sc.start,
          goal: sc.goal,
          environment: h.makeEnv(),
          options: { maxExpansions: sc.maxExpansions },
        },
        Infinity,
      );
    const a = run();
    const b = run();
    if (a.found !== b.found) {
      failures.push({
        check: 'determinism',
        message: `scenario '${sc.name}': found=${a.found} vs ${b.found} across instances`,
      });
      continue;
    }
    if (!a.found) continue;
    if (Math.abs(a.cost - b.cost) > eps) {
      failures.push({
        check: 'determinism',
        message: `scenario '${sc.name}': cost ${a.cost} vs ${b.cost} across instances`,
      });
    }
    if (a.path.length !== b.path.length) {
      failures.push({
        check: 'determinism',
        message: `scenario '${sc.name}': path length ${a.path.length} vs ${b.path.length}`,
      });
      continue;
    }
    for (let i = 0; i < a.path.length; i++) {
      if (!statesClose(a.path[i], b.path[i], eps)) {
        failures.push({
          check: 'determinism',
          message: `scenario '${sc.name}': path diverges at step ${i}`,
          sample: { a: a.path[i], b: b.path[i] },
        });
        break;
      }
    }
  }
  return failures;
}
