// TimeAwareEnvironment is a composing wrapper, so it must forward the full
// optional Environment surface: the `level` argument to succ() (per-level
// primitive sets — aircraft levelControls — would otherwise silently use one
// set on every pass) and the `progress` hook (the planner's best-progress
// fallback would otherwise vanish when a base env is wrapped). The contract
// is stated on Environment in environment/types.ts.

import { describe, it, expect } from 'vitest';
import type { Environment, EdgeRef, Node } from '../../src/environment/types';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { makeNode } from '../../src/planner/node';

interface S {
  x: number;
  z: number;
  t: number;
}

/** Minimal base env that records the `level` values succ() receives. */
class RecordingEnv implements Environment<S> {
  readonly levels = 3;
  readonly seenLevels: Array<number | undefined> = [];
  progressCalls = 0;
  constructor(private readonly withProgress: boolean) {
    if (withProgress) {
      this.progress = (node: Node<S>) => {
        this.progressCalls++;
        return -node.state.x;
      };
    }
  }
  progress?: (node: Node<S>) => number;

  createNode(state: S, parent: Node<S> | null, edge: EdgeRef | null): Node<S> {
    const ix = Math.round(state.x);
    return makeNode(state, parent, edge, [`${Math.floor(ix / 4)}`, `${Math.floor(ix / 2)}`, `${ix}`], `${ix}`);
  }
  succ(node: Node<S>, goal: Node<S>, level?: number): Node<S>[] {
    this.seenLevels.push(level);
    const next: S = { x: node.state.x + 1, z: 0, t: node.state.t + 1 };
    const edge: EdgeRef = { cost: 1, kind: 'step' };
    const n = this.createNode(next, node, edge);
    n.g = node.g + 1;
    n.h = this.heuristic(next, goal.state);
    n.f = n.g + n.h;
    return [n];
  }
  heuristic(from: S, to: S): number {
    return Math.abs(from.x - to.x);
  }
  checkValidity(): [boolean, boolean] {
    return [true, true];
  }
  reachedGoalRegion(node: Node<S>, goal: Node<S>): boolean {
    return Math.abs(node.state.x - goal.state.x) <= 0.5;
  }
}

describe('TimeAwareEnvironment forwarding', () => {
  it('forwards the level argument to the base succ()', () => {
    const base = new RecordingEnv(false);
    const env = new TimeAwareEnvironment(base);
    const start = env.createNode({ x: 0, z: 0, t: 0 }, null, null);
    const goal = env.createNode({ x: 10, z: 0, t: 0 }, null, null);
    env.succ(start, goal, 0);
    env.succ(start, goal, 2);
    env.succ(start, goal);
    expect(base.seenLevels).toEqual([0, 2, undefined]);
  });

  it('exposes progress exactly when the base has it, and forwards calls', () => {
    const withHook = new TimeAwareEnvironment(new RecordingEnv(true));
    const without = new TimeAwareEnvironment(new RecordingEnv(false));
    expect(typeof withHook.progress).toBe('function');
    expect(without.progress).toBeUndefined();

    const base = new RecordingEnv(true);
    const env = new TimeAwareEnvironment(base);
    const n = env.createNode({ x: 3, z: 0, t: 0 }, null, null);
    expect(env.progress!(n)).toBe(-3);
    expect(base.progressCalls).toBe(1);
  });
});
