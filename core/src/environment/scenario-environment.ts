// ScenarioEnvironment — the planner bridge for the Scenario & Goal Spec Layer.
// Generalizes `MultiGoalEnvironment` (an ordered gate list is the special case
// of a `seq`-of-`reach` automaton): instead of a flat `gateIndex`, the search
// state carries the compiled automaton state `q`, and transitions are driven by
// the automaton's guards. Invariants compile to successor pruning; `prefer`
// cost terms add to the edge g; `repeat` (progress) objectives are bounded by an
// explicit horizon and surfaced via the planner's best-progress hook.
//
// Compose as `ScenarioEnvironment(TimeAwareEnvironment(VehicleEnvironment))` —
// Scenario is OUTERMOST (same as MultiGoal), so the inner state keeps x/z/t and
// the time wrapper handles the clock + moving-obstacle pruning underneath.

import type { EdgeRef, Environment, Node } from './types';
import type { PerfRecorder } from '../planner/perf';
import type {
  CompiledAutomaton,
  ScenarioState,
  Invariant,
  CostTerm,
  Region,
} from '../scenario/index';
import { guardSatisfied } from '../scenario/index';

export interface ScenarioAugState<S> {
  readonly inner: S;
  /** Automaton state id. */
  readonly q: number;
  /** Laps completed — present only for progress (`repeat`) objectives. */
  readonly laps?: number;
}

export interface ScenarioEnvOptions {
  automaton: CompiledAutomaton;
  invariants?: Invariant[];
  costTerms?: CostTerm[];
  /** Bound for progress (`repeat`) objectives. Required when the automaton is a
   *  progress automaton; otherwise ignored. */
  horizon?: { phases?: number; seconds?: number };
}

interface ScopedInvariant {
  region: Region;
  scope: Region;
}

export class ScenarioEnvironment<S extends ScenarioState>
  implements Environment<ScenarioAugState<S>>
{
  readonly levels: number;
  private readonly automaton: CompiledAutomaton;
  private readonly acceptSet: Set<number>;
  private readonly costTerms: CostTerm[];
  private readonly horizon?: { phases?: number; seconds?: number };
  private readonly progressMode: boolean;
  private readonly maxDepth: number;
  /** Representative pose of each outgoing guard, precomputed per automaton
   *  state — these are deterministic and never change, so we avoid re-calling
   *  `region.representative()` (an allocation) on every expanded node. */
  private readonly repByState: ScenarioState[][];

  /** avoid (+ unscoped maintain) — active in every automaton state. */
  private readonly alwaysActive: Invariant[] = [];
  /** maintain(.while(scope)) — active only where the state is inside `scope`. */
  private readonly scopedActive: ScopedInvariant[] = [];

  private startT = 0;
  private startTSet = false;

  constructor(
    private readonly base: Environment<S>,
    opts: ScenarioEnvOptions,
  ) {
    this.levels = base.levels;
    this.automaton = opts.automaton;
    this.acceptSet = new Set(opts.automaton.accepting);
    this.costTerms = opts.costTerms ?? [];
    this.horizon = opts.horizon;
    this.progressMode = opts.automaton.progress;
    this.maxDepth = opts.automaton.states.reduce((m, s) => Math.max(m, s.depth), 0);
    this.repByState = opts.automaton.states.map((s) =>
      s.transitions.map((tr) => tr.guard.region.representative()),
    );

    if (this.progressMode && !this.horizon) {
      throw new Error(
        'ScenarioEnvironment: a progress (repeat) objective requires an explicit horizon { phases | seconds }',
      );
    }
    for (const inv of opts.invariants ?? []) {
      if (inv.kind === 'maintain' && inv.scope) {
        this.scopedActive.push({ region: inv.region, scope: inv.scope });
      } else {
        this.alwaysActive.push(inv);
      }
    }
  }

  attachRecorder(rec: PerfRecorder): void {
    this.base.attachRecorder?.(rec);
  }

  /** Build an inner-state goal pose from a representative ScenarioState,
   *  inheriting the template's extra fields so the result is a valid `S`. */
  private asInner(template: S, rep: ScenarioState): S {
    return { ...template, x: rep.x, z: rep.z, heading: rep.heading };
  }

  createNode(
    state: ScenarioAugState<S>,
    parent: Node<ScenarioAugState<S>> | null,
    edge: EdgeRef | null,
  ): Node<ScenarioAugState<S>> {
    if (parent === null && !this.startTSet) {
      this.startT = state.inner.t;
      this.startTSet = true;
    }
    const innerNode = this.base.createNode(state.inner, null, null);
    const tag = state.laps !== undefined ? `q${state.q}l${state.laps}` : `q${state.q}`;
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

  private overHorizon(state: ScenarioAugState<S>): boolean {
    if (!this.horizon) return false;
    if (
      this.horizon.seconds !== undefined &&
      state.inner.t - this.startT >= this.horizon.seconds
    ) {
      return true;
    }
    if (this.horizon.phases !== undefined && (state.laps ?? 0) >= this.horizon.phases) {
      return true;
    }
    return false;
  }

  private violatesInvariant(prev: S, next: S): boolean {
    for (const inv of this.alwaysActive) {
      if (this.violates(inv.kind, inv.region, prev, next)) return true;
    }
    for (const s of this.scopedActive) {
      if (s.scope.contains(next, next.t) && this.violates('maintain', s.region, prev, next)) {
        return true;
      }
    }
    return false;
  }

  private violates(kind: 'avoid' | 'maintain', region: Region, prev: S, next: S): boolean {
    if (kind === 'avoid') {
      if (region.contains(next, next.t)) return true;
      if (region.crossed && region.crossed(prev, next, prev.t)) return true;
      return false;
    }
    // maintain: must always be inside.
    return !region.contains(next, next.t);
  }

  private costPenalty(prev: S, next: S): number {
    if (this.costTerms.length === 0) return 0;
    const dt = next.t - prev.t;
    let extra = 0;
    for (const term of this.costTerms) extra += term.edgeCost(prev, next, dt);
    return extra;
  }

  succ(
    node: Node<ScenarioAugState<S>>,
    _goal: Node<ScenarioAugState<S>>,
    level?: number,
  ): Node<ScenarioAugState<S>>[] {
    const q = node.state.q;
    if (this.acceptSet.has(q) && !this.progressMode) return [];
    if (this.progressMode && this.overHorizon(node.state)) return [];
    const st = this.automaton.states[q];
    if (!st || st.transitions.length === 0) return [];

    const rep = this.repByState[q]![0] ?? null;
    const innerNode = this.base.createNode(node.state.inner, null, null);
    innerNode.g = node.g;
    const innerGoal = this.base.createNode(
      rep ? this.asInner(node.state.inner, rep) : node.state.inner,
      null,
      null,
    );
    const innerSuccs =
      level !== undefined
        ? this.base.succ(innerNode, innerGoal, level)
        : this.base.succ(innerNode, innerGoal);

    const out: Node<ScenarioAugState<S>>[] = [];
    for (const inner of innerSuccs) {
      const prev = node.state.inner;
      const next = inner.state;
      if (this.violatesInvariant(prev, next)) continue;

      // Greedy multi-phase advance: one edge may satisfy several sequential
      // guards; advance as many as fire, in phase order.
      let qCur = q;
      let laps = node.state.laps;
      for (let steps = 0; steps < this.automaton.states.length + 1; steps++) {
        const cur = this.automaton.states[qCur]!;
        let advanced = false;
        for (const tr of cur.transitions) {
          if (guardSatisfied(tr.guard, prev, next)) {
            // A depth decrease means we crossed a `repeat` back-edge -> one lap.
            if (
              this.progressMode &&
              this.automaton.states[tr.target]!.depth < cur.depth
            ) {
              laps = (laps ?? 0) + 1;
            }
            qCur = tr.target;
            advanced = true;
            break;
          }
        }
        if (!advanced) break;
      }

      const extra = this.costPenalty(prev, next);
      const baseCost = inner.edge?.cost ?? 0;
      const edge: EdgeRef = inner.edge
        ? { ...inner.edge, cost: baseCost + extra }
        : { kind: 'scenario', cost: extra };
      const succState: ScenarioAugState<S> =
        laps !== undefined ? { inner: next, q: qCur, laps } : { inner: next, q: qCur };
      out.push(this.createNode(succState, node, edge));
    }
    return out;
  }

  heuristic(from: ScenarioAugState<S>, _to: ScenarioAugState<S>): number {
    const q = from.q;
    if (this.acceptSet.has(q) && !this.progressMode) return 0;
    const st = this.automaton.states[q];
    if (!st || st.transitions.length === 0) return this.progressMode ? 0 : Infinity;
    let head = Infinity;
    for (const rep of this.repByState[q]!) {
      head = Math.min(head, this.base.heuristic(from.inner, this.asInner(from.inner, rep)));
    }
    const chain = this.automaton.remainingChain[q] ?? 0;
    return (Number.isFinite(head) ? head : 0) + chain;
  }

  checkValidity(start: ScenarioAugState<S>, _goal: ScenarioAugState<S>): [boolean, boolean] {
    const [startValid] = this.base.checkValidity(start.inner, start.inner);
    return [startValid, true];
  }

  reachedGoalRegion(node: Node<ScenarioAugState<S>>, _goal: Node<ScenarioAugState<S>>): boolean {
    return this.acceptSet.has(node.state.q);
  }

  /** Best-progress score: automaton depth (+ laps for progress objectives),
   *  with a tiny g penalty so cheaper plans win at equal depth. */
  progress(node: Node<ScenarioAugState<S>>): number {
    const depth = this.automaton.states[node.state.q]?.depth ?? 0;
    const laps = node.state.laps ?? 0;
    return depth + laps * (this.maxDepth + 1) - node.g * 1e-6;
  }
}

/** The start node for a scenario search. */
export function scenarioStart<S extends ScenarioState>(
  start: S,
  automaton: CompiledAutomaton,
): ScenarioAugState<S> {
  return automaton.progress
    ? { inner: start, q: automaton.start, laps: 0 }
    : { inner: start, q: automaton.start };
}

/** A terminal node for the planner's `goal` argument. ScenarioEnvironment's
 *  `reachedGoalRegion` only reads `q`, so the inner pose is just a placeholder. */
export function scenarioTerminal<S extends ScenarioState>(
  start: S,
  automaton: CompiledAutomaton,
): ScenarioAugState<S> {
  const q = automaton.accepting[0] ?? automaton.start;
  return automaton.progress ? { inner: start, q, laps: 0 } : { inner: start, q };
}
