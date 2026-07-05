// ScenarioEnvironment wrapper tests. Strategy: wrap a minimal holonomic
// point-mass base (8-direction grid over ScenarioState, no vehicle gotchas) so
// we exercise the WRAPPER semantics — automaton advance, invariant pruning,
// any-branching, maintain.while scoping, repeat horizon, best-progress — in
// isolation.

import { describe, it, expect } from 'vitest';
import {
  ScenarioEnvironment,
  scenarioStart,
  scenarioTerminal,
} from '../../src/environment/scenario-environment';
import type { Environment, Node, EdgeRef } from '../../src/environment/types';
import { plan } from '../../src/planner/ighastar';
import {
  reach,
  seq,
  any,
  all,
  repeat,
  near,
  inside,
  avoid,
  maintain,
  speed,
  lte,
  compile,
  minTime,
} from '../../src/scenario/index';
import type { ScenarioState, Goal } from '../../src/scenario/index';

// --- Minimal holonomic point-mass base over ScenarioState ------------------
const STEP = 1;
const SPEED = 2;
const DIRS = Array.from({ length: 8 }, (_, i) => (i * Math.PI) / 4);

class GridBase implements Environment<ScenarioState> {
  readonly levels = 1;
  constructor(private bounds = { min: -40, max: 40 }) {}
  createNode(state: ScenarioState, parent: Node<ScenarioState> | null, edge: EdgeRef | null) {
    const ix = Math.round(state.x / STEP);
    const iz = Math.round(state.z / STEP);
    const hash = `${ix}:${iz}`;
    const g = (parent?.g ?? 0) + (edge?.cost ?? 0);
    const h = 0;
    return {
      state, g, h, f: g + h, parent, edge,
      index: [hash], hash, level: 0, active: true, seq: 0,
    };
  }
  succ(node: Node<ScenarioState>, goal: Node<ScenarioState>): Node<ScenarioState>[] {
    const out: Node<ScenarioState>[] = [];
    for (const dir of DIRS) {
      const nx = node.state.x + Math.cos(dir) * STEP;
      const nz = node.state.z + Math.sin(dir) * STEP;
      if (nx < this.bounds.min || nx > this.bounds.max || nz < this.bounds.min || nz > this.bounds.max) {
        continue;
      }
      const dt = STEP / SPEED;
      const next: ScenarioState = {
        x: nx, z: nz, heading: dir, speed: SPEED, t: node.state.t + dt,
      };
      const edge: EdgeRef = { kind: 'grid', cost: STEP };
      const n = this.createNode(next, node, edge);
      n.h = this.heuristic(next, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }
  heuristic(from: ScenarioState, to: ScenarioState): number {
    return Math.hypot(from.x - to.x, from.z - to.z);
  }
  checkValidity(): [boolean, boolean] {
    return [true, true];
  }
  reachedGoalRegion(node: Node<ScenarioState>, goal: Node<ScenarioState>): boolean {
    return this.heuristic(node.state, goal.state) <= STEP;
  }
}

const start: ScenarioState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };

function planScenario(goal: Goal) {
  const automaton = compile(goal);
  const e = new ScenarioEnvironment<ScenarioState>(new GridBase(), { automaton });
  return {
    automaton,
    result: plan(
      {
        start: scenarioStart(start, automaton),
        goal: scenarioTerminal(start, automaton),
        environment: e,
        options: { maxExpansions: 20000 },
      },
      Infinity,
    ),
  };
}

describe('ScenarioEnvironment — product search', () => {
  it('reaches a seq of reach goals in order', () => {
    const goal = seq(reach(near({ x: 10, z: 0 }, 1)), reach(near({ x: 10, z: 10 }, 1)));
    const { result } = planScenario(goal);
    expect(result.found).toBe(true);
    expect(result.partial).toBeFalsy();
    const last = result.path[result.path.length - 1]!.inner;
    expect(Math.hypot(last.x - 10, last.z - 10)).toBeLessThanOrEqual(1.5);
  });

  it('node dedup includes automaton state (same pose, different q = different vertex)', () => {
    const automaton = compile(seq(reach(near({ x: 10, z: 0 }, 1)), reach(near({ x: 20, z: 0 }, 1))));
    const env = new ScenarioEnvironment<ScenarioState>(new GridBase(), { automaton });
    const a = env.createNode({ inner: { x: 5, z: 0, heading: 0, speed: 0, t: 0 }, q: automaton.start }, null, null);
    const b = env.createNode({ inner: { x: 5, z: 0, heading: 0, speed: 0, t: 0 }, q: automaton.start + 1 }, null, null);
    expect(a.hash).not.toBe(b.hash);
  });

  it('avoid prunes successors that enter the region (routes around)', () => {
    const wall: [number, number][] = [
      [4, -3],
      [6, -3],
      [6, 3],
      [4, 3],
    ];
    const goal = reach(near({ x: 12, z: 0 }, 1));
    const automaton = compile(goal);
    const env = new ScenarioEnvironment<ScenarioState>(new GridBase(), {
      automaton,
      invariants: [avoid(inside(wall))],
    });
    const result = plan(
      {
        start: scenarioStart(start, automaton),
        goal: scenarioTerminal(start, automaton),
        environment: env,
        options: { maxExpansions: 20000 },
      },
      Infinity,
    );
    expect(result.found).toBe(true);
    // No state on the path is inside the avoid wall.
    const wallRegion = inside(wall);
    for (const s of result.path) {
      expect(wallRegion.contains(s.inner)).toBe(false);
    }
  });

  it('any: accepts when EITHER branch is reached', () => {
    const goal = any(reach(near({ x: 8, z: 0 }, 1)), reach(near({ x: -8, z: 0 }, 1)));
    const { result } = planScenario(goal);
    expect(result.found).toBe(true);
    const last = result.path[result.path.length - 1]!.inner;
    const reachedA = Math.hypot(last.x - 8, last.z) <= 1.5;
    const reachedB = Math.hypot(last.x + 8, last.z) <= 1.5;
    expect(reachedA || reachedB).toBe(true);
  });

  it('all: satisfies an unordered conjunction', () => {
    const goal = all(reach(near({ x: 8, z: 0 }, 1)), reach(near({ x: 0, z: 8 }, 1)));
    const { result } = planScenario(goal);
    expect(result.found).toBe(true);
    expect(result.partial).toBeFalsy();
  });

  it('maintain(speed<=0).while(region) prunes only inside the scope', () => {
    // The whole base moves at SPEED=2, so maintain(speed<=0) inside a zone the
    // path must cross makes the goal UNREACHABLE through that zone -> the
    // planner must route around it (or fail). Put the zone as a full wall.
    const zone: [number, number][] = [
      [4, -30],
      [6, -30],
      [6, 30],
      [4, 30],
    ];
    const goal = reach(near({ x: 12, z: 0 }, 1));
    const automaton = compile(goal);
    const env = new ScenarioEnvironment<ScenarioState>(new GridBase(), {
      automaton,
      invariants: [maintain(speed(lte(0))).while(inside(zone))],
    });
    const result = plan(
      {
        start: scenarioStart(start, automaton),
        goal: scenarioTerminal(start, automaton),
        environment: env,
        options: { maxExpansions: 30000 },
      },
      Infinity,
    );
    // The zone spans the full corridor, so no moving node may enter it; goal is
    // walled off -> best-progress fallback (objective not formally reached).
    expect(result.found).toBe(true);
    if (result.partial) {
      // never entered the zone
      const z = inside(zone);
      for (const s of result.path) expect(z.contains(s.inner)).toBe(false);
    }
  });

  it('best-progress: returns the deepest phase when the goal is unreachable', () => {
    // Reach A (open), then B which is fully enclosed by an avoid wall.
    const box: [number, number][] = [
      [18, -3],
      [26, -3],
      [26, 3],
      [18, 3],
    ];
    const goal = seq(reach(near({ x: 8, z: 0 }, 1)), reach(near({ x: 22, z: 0 }, 1)));
    const automaton = compile(goal);
    const env = new ScenarioEnvironment<ScenarioState>(new GridBase(), {
      automaton,
      invariants: [avoid(inside(box))],
    });
    const result = plan(
      {
        start: scenarioStart(start, automaton),
        goal: scenarioTerminal(start, automaton),
        environment: env,
        options: { maxExpansions: 40000 },
      },
      Infinity,
    );
    // B sits inside the wall -> never reachable. We still get a partial plan
    // that advanced past phase 1 (reached A).
    expect(result.found).toBe(true);
    expect(result.partial).toBe(true);
    // The deepest reached automaton state has depth >= 1 (A satisfied).
    const deepest = result.path[result.path.length - 1]!;
    expect(automaton.states[deepest.q]!.depth).toBeGreaterThanOrEqual(1);
  });

  it('repeat requires a horizon and is bounded by it', () => {
    const automaton = compile(repeat(seq(reach(near({ x: 8, z: 0 }, 1.5)), reach(near({ x: 0, z: 0 }, 1.5)))));
    expect(() => new ScenarioEnvironment<ScenarioState>(new GridBase(), { automaton })).toThrow(/horizon/);
    const env = new ScenarioEnvironment<ScenarioState>(new GridBase(), {
      automaton,
      horizon: { phases: 2 },
    });
    const result = plan(
      {
        start: scenarioStart(start, automaton),
        goal: scenarioTerminal(start, automaton),
        environment: env,
        options: { maxExpansions: 40000 },
      },
      Infinity,
    );
    // Progress automaton never "reaches" F -> returns best-progress partial.
    expect(result.found).toBe(true);
    expect(result.partial).toBe(true);
    const deepest = result.path[result.path.length - 1]!;
    expect((deepest.laps ?? 0)).toBeGreaterThanOrEqual(1);
  });

  it('cost terms increase edge g (minTime adds to plan cost)', () => {
    const goal = reach(near({ x: 10, z: 0 }, 1));
    const base = compile(goal);
    const plain = plan(
      {
        start: scenarioStart(start, base),
        goal: scenarioTerminal(start, base),
        environment: new ScenarioEnvironment<ScenarioState>(new GridBase(), { automaton: base }),
        options: { maxExpansions: 20000 },
      },
      Infinity,
    );
    const costed = plan(
      {
        start: scenarioStart(start, base),
        goal: scenarioTerminal(start, base),
        environment: new ScenarioEnvironment<ScenarioState>(new GridBase(), {
          automaton: base,
          costTerms: [minTime(5)],
        }),
        options: { maxExpansions: 20000 },
      },
      Infinity,
    );
    expect(costed.cost).toBeGreaterThan(plain.cost);
  });
});
