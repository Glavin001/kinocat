// Surface syntax (sugar). Fluent builders emit canonical AST nodes, so two
// spellings that mean the same thing produce identical ASTs. This is the
// "concise" layer; the discriminated-union AST in `types.ts` is the official,
// serializable one.

import type {
  Goal,
  Region,
  Acceptance,
  Invariant,
  MaintainInvariant,
  AvoidInvariant,
  Scenario,
  ScenarioState,
  CostTerm,
} from './types';
import { inside, type Pose } from './regions';
import type { Pt } from '../internal/geom';

// --- Objective plane -------------------------------------------------------

export function reach(region: Region, accept?: Acceptance): Goal {
  return accept ? { kind: 'reach', region, accept } : { kind: 'reach', region };
}
export function seq(...goals: Goal[]): Goal {
  return { kind: 'seq', goals };
}
export function all(...goals: Goal[]): Goal {
  return { kind: 'all', goals };
}
export function any(...goals: Goal[]): Goal {
  return { kind: 'any', goals };
}
export function repeat(goal: Goal): Goal {
  return { kind: 'repeat', goal };
}

// --- Constraint plane ------------------------------------------------------

export function avoid(region: Region): AvoidInvariant {
  return { kind: 'avoid', region };
}

/** A `maintain` invariant with a fluent `.while(region)` scope. The returned
 *  object is itself a valid `Invariant` (unscoped); `.while(r)` yields a new,
 *  scoped one. */
export type ScopedMaintain = MaintainInvariant & {
  while(region: Region): MaintainInvariant;
};

export function maintain(region: Region): ScopedMaintain {
  const base: MaintainInvariant = { kind: 'maintain', region };
  return {
    ...base,
    while(scope: Region): MaintainInvariant {
      return { kind: 'maintain', region, scope };
    },
  };
}

/** Convenience: must always remain inside the given area (polygon or region). */
export function stayInside(area: Region | ReadonlyArray<Pt>): MaintainInvariant {
  const region = Array.isArray(area) ? inside(area as ReadonlyArray<Pt>) : (area as Region);
  return { kind: 'maintain', region };
}

// --- Whole scenario --------------------------------------------------------

export interface DefineScenarioInput {
  start: ScenarioState;
  goal: Goal;
  invariants?: Invariant[];
  prefer?: CostTerm[];
  agents?: Scenario['agents'];
}

export function defineScenario(name: string, input: DefineScenarioInput): Scenario {
  return {
    name,
    start: input.start,
    goal: input.goal,
    invariants: input.invariants ?? [],
    prefer: input.prefer ?? [],
    agents: input.agents ?? [],
  };
}

/** Degrees -> radians, for readable heading tolerances. */
export function deg(d: number): number {
  return (d * Math.PI) / 180;
}

// Re-export pose type used by `at`.
export type { Pose };
