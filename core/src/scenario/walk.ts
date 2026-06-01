// AST walkers for tooling (the visualizer, rubric, generators). Extract the
// regions referenced by a goal / scenario so they can be rendered or scored.

import type { Goal, Region, Scenario, Invariant } from './types';

/** All `reach` regions in a goal, in left-to-right AST order. */
export function goalRegions(goal: Goal): Region[] {
  const out: Region[] = [];
  const walk = (g: Goal): void => {
    switch (g.kind) {
      case 'reach':
        out.push(g.region);
        return;
      case 'repeat':
        walk(g.goal);
        return;
      case 'seq':
      case 'all':
      case 'any':
        for (const c of g.goals) walk(c);
        return;
    }
  };
  walk(goal);
  return out;
}

/** The `avoid` regions in an invariant list. */
export function avoidRegions(invariants: Invariant[] | undefined): Region[] {
  return (invariants ?? []).filter((i) => i.kind === 'avoid').map((i) => i.region);
}

/** The `maintain` regions in an invariant list (incl. scoped). */
export function maintainRegions(invariants: Invariant[] | undefined): Region[] {
  return (invariants ?? []).filter((i) => i.kind === 'maintain').map((i) => i.region);
}

export interface ScenarioRegions {
  objective: Region[];
  avoid: Region[];
  maintain: Region[];
}

/** Bucket every region in a scenario by the plane it belongs to. */
export function collectScenarioRegions(scenario: Scenario): ScenarioRegions {
  return {
    objective: goalRegions(scenario.goal),
    avoid: avoidRegions(scenario.invariants),
    maintain: maintainRegions(scenario.invariants),
  };
}
