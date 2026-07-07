// Generic multi-goal Environment wrapper. Composes any `Environment<S>` to
// produce an `Environment<MultiGoalState<S>>` whose search state is augmented
// with "next gate index". A single A* over this wrapped environment then
// optimizes a path through a SEQUENCE of intermediate goals (gates) instead
// of one terminal goal — globally, not as a chain of independent segments.
//
// The wrapper is domain-agnostic: works for vehicle / humanoid / aircraft
// states, given a caller-supplied gate-crossing predicate.
//
// Architecture:
//   - State = { inner: S, gateIndex: number }
//   - On expansion, the base env is called with the CURRENT gate as goal.
//     For each base successor, we check whether the successor's state has
//     crossed gate[gateIndex] (caller predicate), and if so advance the
//     index. The successor is wrapped with the (possibly new) index.
//   - Dedup key includes gateIndex so two paths to "same chassis pose,
//     different gates left" stay distinct.
//   - Heuristic = base.heuristic(inner, gates[gateIndex])
//                 + Σ legHeuristic(gates[i], gates[i+1]) for remaining legs.
//     Admissible iff the leg heuristic is admissible (caller default uses
//     base.heuristic).
//   - reachedGoalRegion: gateIndex >= gates.length.

import type { EdgeRef, Environment, Node } from './types';
import type { PerfRecorder } from '../planner/perf';

export interface MultiGoalState<S> {
  readonly inner: S;
  /** Index of the NEXT gate to reach. 0 = chasing gates[0]; gates.length
   *  = all gates passed (terminal). */
  readonly gateIndex: number;
}

export interface MultiGoalOptions<S> {
  /** Ordered gate states. Non-empty. */
  gates: S[];
  /** Caller-supplied "has `state` reached `gate`?" predicate. Typically a
   *  Euclidean disk check at some radius. */
  reachedGate: (state: S, gate: S) => boolean;
  /** Optional admissible per-leg cost lower bound. Defaults to
   *  `base.heuristic(from, to)` if not supplied. Useful when the caller
   *  has a tighter / cheaper bound (e.g. straight-line / max-speed). */
  legHeuristic?: (from: S, to: S) => number;
}

export class MultiGoalEnvironment<S> implements Environment<MultiGoalState<S>> {
  readonly levels: number;
  private readonly gates: S[];
  private readonly reachedGate: (s: S, g: S) => boolean;
  private readonly legHeuristic: (from: S, to: S) => number;
  /** tailLowerBound[i] = Σ legHeuristic(gates[k], gates[k+1]) for k=i..N-2.
   *  Precomputed once per planning request. */
  private readonly tailLowerBound: number[];

  constructor(
    private readonly base: Environment<S>,
    opts: MultiGoalOptions<S>,
  ) {
    if (opts.gates.length === 0) {
      throw new Error('MultiGoalEnvironment requires at least one gate');
    }
    this.levels = base.levels;
    this.gates = opts.gates;
    this.reachedGate = opts.reachedGate;
    this.legHeuristic = opts.legHeuristic ?? ((from, to) => base.heuristic(from, to));
    const N = opts.gates.length;
    this.tailLowerBound = new Array(N + 1).fill(0);
    let sum = 0;
    for (let i = N - 2; i >= 0; i--) {
      sum += this.legHeuristic(opts.gates[i]!, opts.gates[i + 1]!);
      this.tailLowerBound[i] = sum;
    }
    // tailLowerBound[N-1] = 0 (no leg after the last gate)
    // tailLowerBound[N]   = 0 (done — no remaining gates)
  }

  attachRecorder(rec: PerfRecorder): void {
    this.base.attachRecorder?.(rec);
  }

  createNode(
    state: MultiGoalState<S>,
    parent: Node<MultiGoalState<S>> | null,
    edge: EdgeRef | null,
  ): Node<MultiGoalState<S>> {
    const innerNode = this.base.createNode(state.inner, null, null);
    // Include gateIndex in both the multi-level coarse index AND the exact
    // hash so dedup/dominance treat two states with same inner but
    // different gateIndex as DIFFERENT vertices.
    const tag = `g${state.gateIndex}`;
    const index = innerNode.index.map((k) => `${k}|${tag}`);
    const hash = `${innerNode.hash}|${tag}`;
    const g = (parent?.g ?? 0) + (edge?.cost ?? 0);
    const h = this.heuristic(state, state);
    return {
      state,
      g,
      h,
      f: g + h,
      parent,
      edge,
      index,
      hash,
      level: 0,
      active: true,
      seq: 0,
    };
  }

  succ(
    node: Node<MultiGoalState<S>>,
    _goal: Node<MultiGoalState<S>>,
    level?: number,
  ): Node<MultiGoalState<S>>[] {
    if (node.state.gateIndex >= this.gates.length) return [];
    // Build an inner-state pseudo-node and an inner-state goal node so the
    // base env's heuristic / analytic-shot can focus on the CURRENT gate.
    // Preserve the producing edge so the base env's gear-history pricing
    // (direction-change penalty) survives the wrapper (see scenario-environment).
    const innerNode = this.base.createNode(node.state.inner, null, node.edge);
    innerNode.g = node.g;
    const currentGate = this.gates[node.state.gateIndex]!;
    const innerGoal = this.base.createNode(currentGate, null, null);
    const innerSuccs = level !== undefined
      ? this.base.succ(innerNode, innerGoal, level)
      : this.base.succ(innerNode, innerGoal);
    const out: Node<MultiGoalState<S>>[] = [];
    const gateIndex = node.state.gateIndex;
    const tail = this.tailLowerBound[gateIndex] ?? 0;
    for (const inner of innerSuccs) {
      // Greedy gate advance: a single primitive can sweep past multiple
      // gates on a tight loop, so chase them all in one step.
      let newIdx = gateIndex;
      while (
        newIdx < this.gates.length &&
        this.reachedGate(inner.state, this.gates[newIdx]!)
      ) {
        newIdx++;
      }
      // Fast wrap: reuse the inner successor node the base env just built —
      // its `index`, `hash`, and (crucially) its already-computed heuristic
      // `h = base.heuristic(inner, gates[gateIndex])`. Rebuilding a fresh inner
      // node via `createNode` here would recompute that Reeds-Shepp solve (the
      // dominant per-successor cost) a SECOND time and rebuild the index/hash
      // strings — pure waste. Only when a gate was crossed (newIdx advanced)
      // does the head gate change, so inner.h no longer matches and we fall
      // back to the full heuristic.
      const tag = `g${newIdx}`;
      const h =
        newIdx === gateIndex
          ? inner.h + tail
          : this.heuristic({ inner: inner.state, gateIndex: newIdx }, node.state);
      out.push({
        state: { inner: inner.state, gateIndex: newIdx },
        g: inner.g,
        h,
        f: inner.g + h,
        parent: node,
        edge: inner.edge,
        index: inner.index.map((k) => `${k}|${tag}`),
        hash: `${inner.hash}|${tag}`,
        level: 0,
        active: true,
        seq: 0,
      });
    }
    return out;
  }

  heuristic(from: MultiGoalState<S>, _to: MultiGoalState<S>): number {
    if (from.gateIndex >= this.gates.length) return 0;
    const headGate = this.gates[from.gateIndex]!;
    const hHead = this.legHeuristic(from.inner, headGate);
    const hTail = this.tailLowerBound[from.gateIndex] ?? 0;
    return hHead + hTail;
  }

  checkValidity(
    start: MultiGoalState<S>,
    _goal: MultiGoalState<S>,
  ): [boolean, boolean] {
    // Validate the start against the FINAL gate (the most distant goal
    // the base env will ever be asked about). The "goal" in multi-goal
    // frame is just gateIndex == N — always valid.
    const finalGate = this.gates[this.gates.length - 1]!;
    const [startValid] = this.base.checkValidity(start.inner, finalGate);
    return [startValid, true];
  }

  reachedGoalRegion(
    node: Node<MultiGoalState<S>>,
    _goal: Node<MultiGoalState<S>>,
  ): boolean {
    return node.state.gateIndex >= this.gates.length;
  }
}

/** Convenience builder for the canonical "terminal" multi-goal node — the
 *  goal you pass to the planner. Its only meaningful field is gateIndex
 *  set to the gate count. The `inner` field is filled with the final gate
 *  pose so `checkValidity` can still examine it if needed. */
export function multiGoalTerminal<S>(gates: S[]): MultiGoalState<S> {
  return {
    inner: gates[gates.length - 1]!,
    gateIndex: gates.length,
  };
}

/** Convenience builder for the start node. */
export function multiGoalStart<S>(start: S): MultiGoalState<S> {
  return { inner: start, gateIndex: 0 };
}
