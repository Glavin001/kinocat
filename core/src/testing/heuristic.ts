// Heuristic contract checks. IGHA*'s optimality argument needs an admissible
// heuristic; its dedup-on-hash argument prefers a consistent one. Consistency
// (h(a) ≤ c(a,b) + h(b) along every edge, h ≥ 0, h(a,a) ≈ 0) implies
// admissibility along expanded paths, so `checkHeuristicConsistency` is the
// primary check; `checkHeuristicAdmissible` is a plan-level backstop for
// domains whose heuristic is admissible but not provably consistent.

import { plan } from '../planner/ighastar';
import { rng } from './rng';
import {
  tol,
  type CheckOptions,
  type ConformanceFailure,
  type DomainHarness,
} from './types';

export function checkHeuristicConsistency<State>(
  h: DomainHarness<State>,
  opts: CheckOptions = {},
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const env = h.makeEnv();
  const eps = tol(h);
  const rand = rng(opts.seed ?? 0xc0ffee);
  const samples = opts.samples ?? 200;
  const goals = h.scenarios.map((s) => s.goal);
  if (goals.length === 0) {
    return [
      {
        check: 'heuristic-consistency',
        message: 'harness has no scenarios — need at least one goal state',
      },
    ];
  }

  let tested = 0;
  for (let i = 0; i < samples; i++) {
    const s = h.sampleState(rand);
    if (!env.checkValidity(s, s)[0]) continue;
    const goalState = goals[i % goals.length]!;
    const goalNode = env.createNode(goalState, null, null);
    const node = env.createNode(s, null, null);

    const hSelf = env.heuristic(s, s);
    if (!(Math.abs(hSelf) <= eps)) {
      failures.push({
        check: 'heuristic-consistency',
        message: `h(a, a) = ${hSelf}, expected ≈ 0`,
        sample: s,
      });
    }
    const hA = env.heuristic(s, goalState);
    if (!Number.isFinite(hA) || hA < -eps) {
      failures.push({
        check: 'heuristic-consistency',
        message: `h(a, goal) = ${hA}, expected finite and ≥ 0`,
        sample: s,
      });
      continue;
    }

    for (const c of env.succ(node, goalNode, env.levels - 1)) {
      const cost = c.edge ? c.edge.cost : c.g - node.g;
      const hB = env.heuristic(c.state, goalState);
      if (hA > cost + hB + eps) {
        failures.push({
          check: 'heuristic-consistency',
          message:
            `inconsistent along '${c.edge?.kind ?? '?'}' edge: ` +
            `h(a)=${hA} > cost=${cost} + h(b)=${hB} (violation ${hA - cost - hB})`,
          sample: { from: s, to: c.state, edge: c.edge },
        });
      }
    }
    tested++;
  }

  if (tested === 0) {
    failures.push({
      check: 'heuristic-consistency',
      message: `sampler produced 0 valid states out of ${samples} — fix sampleState`,
    });
  }
  return failures;
}

export function checkHeuristicAdmissible<State>(
  h: DomainHarness<State>,
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const eps = tol(h);
  for (const sc of h.scenarios) {
    const env = h.makeEnv();
    const r = plan(
      {
        start: sc.start,
        goal: sc.goal,
        environment: env,
        options: { maxExpansions: sc.maxExpansions },
      },
      Infinity,
    );
    if (!r.found || r.partial) continue; // solvability is scenario-budget's job
    const hStart = env.heuristic(sc.start, sc.goal);
    if (hStart > r.cost + eps) {
      failures.push({
        check: 'heuristic-admissible',
        message:
          `scenario '${sc.name}': h(start, goal)=${hStart} exceeds an actually ` +
          `achieved plan cost ${r.cost} — heuristic is inadmissible`,
        sample: { scenario: sc.name, h: hStart, cost: r.cost },
      });
    }
  }
  return failures;
}
