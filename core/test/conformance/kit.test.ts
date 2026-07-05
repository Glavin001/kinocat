// The kit's own validation: a minimal correct environment passes every
// check, and deliberately-broken variants fail exactly the check that
// polices their defect. If these stop failing, the kit has gone blind.

import { describe, it, expect } from 'vitest';
import type { Environment, EdgeRef, Node } from '../../src/environment/types';
import { makeNode } from '../../src/planner/node';
import {
  runConformance,
  checkHeuristicConsistency,
  checkHeuristicAdmissible,
  checkSuccessorInvariants,
  checkNodeStability,
  type DomainHarness,
} from '../../src/testing';

interface LineState {
  x: number;
  z: number;
  t: number;
}

interface LineEnvOptions {
  /** Multiply the heuristic (values > 1 make it inadmissible). */
  hScale?: number;
  /** Advance time by this much per step (≤ 0 breaks monotone time). */
  dt?: number;
  /** Edge cost per step (0 breaks termination guarantees). */
  stepCost?: number;
  /** Salt the hash with a call counter (breaks hash determinism). */
  unstableHash?: boolean;
}

/** 1-D line world: step ±1 in x, unit cost, unit time. Correct by default;
 *  each option knob introduces exactly one contract violation. */
class LineEnv implements Environment<LineState> {
  readonly levels = 2;
  private counter = 0;
  constructor(private readonly o: LineEnvOptions = {}) {}

  createNode(
    state: LineState,
    parent: Node<LineState> | null,
    edge: EdgeRef | null,
  ): Node<LineState> {
    const ix = Math.round(state.x);
    const it = Math.round(state.t / 0.25);
    const salt = this.o.unstableHash ? `#${this.counter++}` : '';
    return makeNode(
      state,
      parent,
      edge,
      [`${Math.floor(ix / 2)}`, `${ix}`],
      `${ix},${it}${salt}`,
    );
  }

  succ(node: Node<LineState>, goal: Node<LineState>): Node<LineState>[] {
    const st = node.state;
    const dt = this.o.dt ?? 1;
    const cost = this.o.stepCost ?? 1;
    const out: Node<LineState>[] = [];
    for (const dx of [-1, 1]) {
      const next: LineState = { x: st.x + dx, z: st.z, t: st.t + dt };
      const edge: EdgeRef = { cost, kind: 'step' };
      const n = this.createNode(next, node, edge);
      n.g = node.g + cost;
      n.h = this.heuristic(next, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  heuristic(from: LineState, to: LineState): number {
    return (this.o.hScale ?? 1) * Math.abs(from.x - to.x);
  }

  checkValidity(): [boolean, boolean] {
    return [true, true];
  }

  reachedGoalRegion(node: Node<LineState>, goal: Node<LineState>): boolean {
    return Math.abs(node.state.x - goal.state.x) <= 0.5;
  }
}

function harness(o: LineEnvOptions = {}): DomainHarness<LineState> {
  return {
    makeEnv: () => new LineEnv(o),
    sampleState: (rand) => ({
      x: (rand() - 0.5) * 100,
      z: 0,
      t: rand() * 50,
    }),
    scenarios: [
      {
        name: 'line',
        start: { x: 0, z: 0, t: 0 },
        goal: { x: 6, z: 0, t: 0 },
        maxExpansions: 2_000,
      },
    ],
  };
}

describe('conformance kit self-test', () => {
  it('a correct environment passes the full battery', () => {
    const report = runConformance(harness());
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
    expect(report.checks).toContain('heuristic-consistency');
    expect(report.checks).toContain('scenario-budget');
  });

  it('an inflated heuristic fails consistency and admissibility', () => {
    const h = harness({ hScale: 10 });
    expect(checkHeuristicConsistency(h).length).toBeGreaterThan(0);
    expect(checkHeuristicAdmissible(h).length).toBeGreaterThan(0);
  });

  it('non-advancing time fails successor invariants', () => {
    const fails = checkSuccessorInvariants(harness({ dt: -1 }));
    expect(fails.some((f) => f.message.includes('time did not advance'))).toBe(true);
  });

  it('zero-cost edges fail successor invariants', () => {
    const fails = checkSuccessorInvariants(harness({ stepCost: 0 }));
    expect(fails.some((f) => f.message.includes('non-positive cost'))).toBe(true);
  });

  it('an unstable hash fails node stability', () => {
    const fails = checkNodeStability(harness({ unstableHash: true }));
    expect(fails.some((f) => f.message.includes('not deterministic'))).toBe(true);
  });

  it('the aggregate report flags a broken env as not ok', () => {
    const report = runConformance(harness({ hScale: 10 }));
    expect(report.ok).toBe(false);
    expect(report.failures.length).toBeGreaterThan(0);
  });
});
