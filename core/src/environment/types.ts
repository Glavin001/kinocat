// The planner's only coupling to a domain. IGHA* operates purely on these.

import type { PerfRecorder } from '../planner/perf';

/** An edge taken to reach a node (motion primitive, curve, affordance, …).
 *  `cost` is the additive g-cost of traversing it. JSON-serializable. */
export interface EdgeRef {
  cost: number;
  kind: string;
  data?: unknown;
}

/** A search vertex. `index` holds one packed cell key per resolution level,
 *  ordered coarse (0) → fine (levels-1). `hash` uniquely keys the exact state
 *  class (used for optimal dedup). `level`/`active`/`seq` are planner-managed
 *  — environments should treat them as opaque. */
export interface Node<State> {
  state: State;
  g: number;
  h: number;
  f: number;
  parent: Node<State> | null;
  edge: EdgeRef | null;
  index: string[];
  hash: string;
  level: number;
  active: boolean;
  /** Insertion sequence, set by the planner when the node enters the open
   *  list; used as the priority-queue tiebreaker (FIFO within equal-f). */
  seq: number;
}

/** The five-method domain interface, plus a `levels` count for the
 *  multi-resolution machinery. Everything kinocat does on top of IGHA*
 *  (time, affordances, navcat collision) lives in implementations of this. */
export interface Environment<State> {
  /** Number of resolution levels; `Node.index` has this length. */
  readonly levels: number;

  /** Factory from raw state. Fills `index`/`hash`; g/h/f default to 0. */
  createNode(
    state: State,
    parent: Node<State> | null,
    edge: EdgeRef | null,
  ): Node<State>;

  /** Valid successors of `node` toward `goal`, each with g/h/f set.
   *  Optional `level` argument: 0 = coarsest pass, `levels - 1` = finest.
   *  Envs may use it to widen the primitive set on finer passes (coarse
   *  passes plan a low-branching skeleton; finest pass refines). Envs
   *  that ignore the parameter behave identically across passes — the
   *  trailing `?` keeps existing implementations type-compatible. */
  succ(node: Node<State>, goal: Node<State>, level?: number): Node<State>[];

  /** Admissible (ideally consistent) cost-to-go estimate. */
  heuristic(from: State, to: State): number;

  /** Pre-search validity of start and goal: `[startValid, goalValid]`. */
  checkValidity(start: State, goal: State): [boolean, boolean];

  /** Goal predicate. */
  reachedGoalRegion(node: Node<State>, goal: Node<State>): boolean;

  /** Optional: install a performance recorder so the env can increment
   *  per-search counters (collisions, predicts, …) at zero cost when the
   *  recorder is the shared `NULL_RECORDER`. Implementations that do not
   *  contribute counters can omit this method. Composing wrappers (e.g.
   *  TimeAwareEnvironment) must forward to their base environment. */
  attachRecorder?(rec: PerfRecorder): void;

  /** Optional BEST-PROGRESS score (higher = closer to the objective). When the
   *  goal region is never reached (e.g. an unbounded `repeat`, or an
   *  infeasible course), the planner keeps the highest-progress node it saw and
   *  returns it as an "anytime" incumbent (`PlanResult.partial = true`).
   *  Environments that omit this behave exactly as before — the planner only
   *  pays for the hook when the method is present. Composing wrappers must
   *  forward to (or augment) the base environment's score. */
  progress?(node: Node<State>): number;
}
