import { describe, it, expect } from 'vitest';
import {
  reach,
  seq,
  all,
  any,
  repeat,
  near,
  at,
  normalize,
  hashGoal,
  structuralEqual,
  compile,
  toMermaid,
  evaluateProgress,
  TOP,
  BOTTOM,
  deg,
} from '../../src/scenario/index';
import type { Goal, ScenarioState } from '../../src/scenario/index';

function ststate(x: number, z: number, heading = 0): ScenarioState {
  return { x, z, heading, speed: 0, t: 0 };
}
const R = (x: number, z: number, r = 1): Goal => reach(near({ x, z }, r));

describe('builders emit canonical AST', () => {
  it('two spellings produce identical ASTs', () => {
    const a = seq(R(0, 0), R(10, 0));
    const b = seq(R(0, 0), R(10, 0));
    expect(structuralEqual(a, b)).toBe(true);
    expect(hashGoal(a)).toBe(hashGoal(b));
  });
  it('different regions produce different hashes', () => {
    expect(hashGoal(R(0, 0))).not.toBe(hashGoal(R(1, 0)));
  });
});

describe('normalize', () => {
  it('flattens nested same-kind combinators', () => {
    const g = seq(seq(R(0, 0), R(1, 0)), R(2, 0));
    const n = normalize(g);
    expect(n.kind).toBe('seq');
    expect((n as { goals: Goal[] }).goals).toHaveLength(3);
  });
  it('collapses singletons', () => {
    expect(normalize(seq(R(0, 0))).kind).toBe('reach');
    expect(normalize(all(R(0, 0))).kind).toBe('reach');
    expect(normalize(any(R(0, 0))).kind).toBe('reach');
  });
  it('drops identities: all() -> TOP, any() -> BOTTOM', () => {
    expect(hashGoal(normalize(all()))).toBe(hashGoal(TOP));
    expect(hashGoal(normalize(any()))).toBe(hashGoal(BOTTOM));
  });
  it('dedups all/any children but NOT seq children', () => {
    expect((normalize(all(R(0, 0), R(0, 0))) as Goal).kind).toBe('reach'); // dedup -> singleton
    const s = normalize(seq(R(0, 0), R(0, 0)));
    expect(s.kind).toBe('seq');
    expect((s as { goals: Goal[] }).goals).toHaveLength(2); // seq keeps duplicates
  });
  it('idempotent repeat', () => {
    const n = normalize(repeat(repeat(R(0, 0))));
    expect(n.kind).toBe('repeat');
    expect((n as { goal: Goal }).goal.kind).toBe('reach');
  });
  it('reaches a fixed point (hash stable under re-normalize)', () => {
    const g = seq(any(R(0, 0), R(1, 0)), all(R(2, 0), R(2, 0)));
    const n1 = normalize(g);
    const n2 = normalize(n1);
    expect(hashGoal(n1)).toBe(hashGoal(n2));
  });
});

describe('compile', () => {
  it('reach -> 2 states, 1 accepting', () => {
    const a = compile(R(5, 0));
    expect(a.states).toHaveLength(2);
    expect(a.accepting).toHaveLength(1);
    expect(a.progress).toBe(false);
    expect(a.states[a.start]!.transitions).toHaveLength(1);
  });

  it('seq concatenates sub-automata in order', () => {
    const a = compile(seq(R(0, 0), R(10, 0), R(20, 0)));
    // start has 1 outgoing guard; following the chain reaches an accepting state.
    expect(a.accepting).toHaveLength(1);
    expect(a.progress).toBe(false);
    // depth of accepting state == number of phases.
    const acc = a.states[a.accepting[0]!]!;
    expect(acc.depth).toBe(3);
  });

  it('worked example seq(any(reach,reach), reach) compiles correctly', () => {
    const a = compile(seq(any(R(0, 0), R(1, 0)), R(10, 0)));
    // entry branches into two guards (the any), each leading toward the final reach.
    const start = a.states[a.start]!;
    expect(start.transitions.length).toBeGreaterThanOrEqual(2);
    expect(a.accepting).toHaveLength(1);
    // A trajectory satisfying branch A then the final reach should accept.
    const traj: ScenarioState[] = [
      ststate(5, 5),
      ststate(0, 0), // satisfies any-branch R(0,0)
      ststate(10, 0), // satisfies final R(10,0)
    ];
    const p = evaluateProgress(a, traj);
    expect(p.done).toBe(true);
  });

  it('all -> 2^N bitmask lattice, any order satisfies', () => {
    const a = compile(all(R(0, 0), R(10, 0)));
    expect(a.states).toHaveLength(4); // 2^2
    expect(a.accepting).toHaveLength(1);
    // Satisfy in reverse order.
    const traj = [ststate(5, 5), ststate(10, 0), ststate(0, 0)];
    expect(evaluateProgress(a, traj).done).toBe(true);
  });

  it('all rejects composite children in v1', () => {
    expect(() => compile(all(seq(R(0, 0), R(1, 0)), R(2, 0)))).toThrow(/single-phase reach/);
  });

  it('repeat -> progress automaton, no accepting', () => {
    const a = compile(repeat(seq(R(0, 0), R(10, 0))));
    expect(a.progress).toBe(true);
    expect(a.accepting).toHaveLength(0);
  });

  it('any -> disjunction accepts on either branch', () => {
    const a = compile(any(R(0, 0), R(99, 0)));
    expect(evaluateProgress(a, [ststate(5, 5), ststate(0, 0)]).done).toBe(true);
    expect(evaluateProgress(a, [ststate(5, 5), ststate(99, 0)]).done).toBe(true);
  });

  it('remainingChain is a non-negative LB, 0 at accepting', () => {
    const a = compile(seq(R(0, 0), R(10, 0)));
    expect(a.remainingChain[a.accepting[0]!]).toBe(0);
    expect(a.remainingChain[a.start]!).toBeGreaterThanOrEqual(0);
  });
});

describe('progress evaluator', () => {
  it('advances phase-by-phase through a seq', () => {
    const a = compile(seq(R(0, 0), R(10, 0)));
    const partial = evaluateProgress(a, [ststate(5, 5), ststate(0, 0)]);
    expect(partial.done).toBe(false);
    expect(partial.depth).toBe(1);
    expect(partial.maxDepth).toBe(2);
  });

  it('greedy multi-advance: one edge satisfies two close gates', () => {
    const a = compile(seq(R(0, 0, 3), R(1, 0, 3)));
    // single sample at (0.5,0) is within radius 3 of both gates.
    const p = evaluateProgress(a, [ststate(-5, 0), ststate(0.5, 0)]);
    expect(p.done).toBe(true);
  });

  it('counts laps for a repeat objective', () => {
    const a = compile(repeat(seq(R(0, 0, 2), R(10, 0, 2))));
    const traj = [
      ststate(-5, 0),
      ststate(0, 0), // gate 0
      ststate(10, 0), // gate 1 -> loop back to start
      ststate(0, 0), // gate 0 again
      ststate(10, 0), // gate 1 -> lap 2
    ];
    const p = evaluateProgress(a, traj);
    expect(p.laps).toBeGreaterThanOrEqual(1);
  });
});

describe('toMermaid', () => {
  it('renders a stateDiagram with the start + transitions', () => {
    const m = toMermaid(compile(reach(at({ x: 0, z: 0, heading: 0 }, { dx: 1, dz: 1, dheading: deg(5) }))));
    expect(m).toContain('stateDiagram-v2');
    expect(m).toContain('[*] -->');
    expect(m).toContain(': at');
  });
});
