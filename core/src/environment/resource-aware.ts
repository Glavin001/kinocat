// Generic resource-aware environment wrapper. Composes any base
// `Environment<State>` and layers a per-state resource value of type `R`
// that:
//   - participates in node hash/index (so A* treats (state, resource) as
//     the search key),
//   - evolves deterministically per edge via `step`,
//   - gates which successors are admissible via `allow`,
//   - can be mutated by world affordances when a primitive's spatial trace
//     intersects them via `affordance`.
//
// The wrapper has zero domain coupling — `R` and the hooks are entirely
// user-supplied. Sample use cases:
//   - aircraft fuel: gate BOOST primitives by fuel, refill on ring crossings.
//   - rover battery: gate fast primitives by charge, refill at recharge pads.
//   - puzzle keys: gate locked doors by key set, pickup keys as affordances.
//   - resource economies: edges consume inputs, affordances produce outputs.
//
// Composes alongside `TimeAwareEnvironment`: wrap base → time-aware →
// resource-aware (or any other order; the augmentations stack).

import type { Environment, EdgeRef, Node } from './types';
import type { PerfRecorder } from '../planner/perf';

export interface ResourceAwareOptions<State, R> {
  /** Initial resource value attached to the start node (and to any node
   *  reached via a path the wrapper hasn't seen — defensive fallback). */
  initial: R;

  /** Stable string segment representing the quantized resource. Coarser
   *  bucketing collapses more (state, resource) pairs into one search node
   *  and shrinks the search space. */
  bucket(r: R): string;

  /** Edge precondition. Return false to drop this successor entirely (the
   *  planner never sees it). Called BEFORE `step` — use the current
   *  resource value, not the post-edge one. */
  allow(r: R, from: State, edge: EdgeRef, to: State): boolean;

  /** Resource transition along an edge. Pure function — no side effects.
   *  `dt` is the time span of the edge (`to.t - from.t` when the state
   *  carries time; 0 otherwise). */
  step(r: R, from: State, edge: EdgeRef, to: State, dt: number): R;

  /** Optional: world features that mutate the resource when a primitive's
   *  spatial segment intersects them (e.g. fuel pickups, charge pads).
   *  Called AFTER `step` so a single primitive can both consume and refill.
   *  Return the updated R, or `null` for no effect. */
  affordance?(r: R, from: State, to: State): R | null;
}

type HasT = { t: number };

export class ResourceAwareEnvironment<State, R>
  implements Environment<State>
{
  readonly levels: number;
  /** Sidecar map keyed by augmented node hash → resource at that node. */
  private readonly res = new Map<string, R>();

  constructor(
    private readonly base: Environment<State>,
    private readonly opts: ResourceAwareOptions<State, R>,
  ) {
    this.levels = base.levels;
  }

  attachRecorder(rec: PerfRecorder): void {
    this.base.attachRecorder?.(rec);
  }

  private augment(node: Node<State>, r: R): Node<State> {
    const bucket = this.opts.bucket(r);
    const idx = node.index;
    for (let L = 0; L < idx.length; L++) {
      idx[L] = `${idx[L]}|r${bucket}`;
    }
    node.hash = `${node.hash}|r${bucket}`;
    this.res.set(node.hash, r);
    return node;
  }

  createNode(
    state: State,
    parent: Node<State> | null,
    edge: EdgeRef | null,
  ): Node<State> {
    const n = this.base.createNode(state, parent, edge);
    // Root (parent === null) → seed with opts.initial. Non-root createNode
    // calls from outside succ() are unusual; if the parent's resource isn't
    // tracked, fall back to initial to keep the env total.
    const r = parent ? (this.res.get(parent.hash) ?? this.opts.initial) : this.opts.initial;
    return this.augment(n, r);
  }

  succ(node: Node<State>, goal: Node<State>, level?: number): Node<State>[] {
    const rFrom = this.res.get(node.hash) ?? this.opts.initial;
    const baseSuccs = this.base.succ(node, goal, level);
    const out: Node<State>[] = [];
    for (const c of baseSuccs) {
      const edge = c.edge;
      if (!edge) continue; // shouldn't happen — succ always sets edge
      if (!this.opts.allow(rFrom, node.state, edge, c.state)) continue;
      const dt =
        (node.state as unknown as HasT)?.t !== undefined &&
        (c.state as unknown as HasT)?.t !== undefined
          ? (c.state as unknown as HasT).t - (node.state as unknown as HasT).t
          : 0;
      let rTo = this.opts.step(rFrom, node.state, edge, c.state, dt);
      const afford = this.opts.affordance?.(rTo, node.state, c.state);
      if (afford != null) rTo = afford;
      // `c.hash` / `c.index` were already set by the base env. We augment
      // them with the resource bucket so A*'s open/closed sets treat
      // (state, resource) as the key.
      this.augment(c, rTo);
      out.push(c);
    }
    return out;
  }

  heuristic(from: State, to: State): number {
    return this.base.heuristic(from, to);
  }

  checkValidity(start: State, goal: State): [boolean, boolean] {
    return this.base.checkValidity(start, goal);
  }

  reachedGoalRegion(node: Node<State>, goal: Node<State>): boolean {
    return this.base.reachedGoalRegion(node, goal);
  }

  /** Read back the resource attached to a node (by augmented `node.hash`).
   *  Useful for tests and for callers reconstructing a plan's resource
   *  trajectory after the search returns. */
  resourceOf(node: Node<State>): R | undefined {
    return this.res.get(node.hash);
  }
}
