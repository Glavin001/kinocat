import type { Environment, Node } from '../environment/types';
import type { HysteresisOptions } from './hysteresis';
import type { PassStats, PerfMode, PlanCounters, PlanTimings } from './perf';

export interface PlannerOptions {
  /** Resolution levels to use; defaults to `environment.levels`. */
  levels?: number;
  /** Hard cap on total expansions across all passes. */
  maxExpansions?: number;
  /** Wall-clock check granularity (expansions between `now()` checks). */
  deadlineCheckEvery?: number;
  hysteresis?: HysteresisOptions;
  /**
   * Performance-logging mode. `'counts'` (default) updates integer event
   * counters on the hot path at ~1% overhead; surfaces them through
   * `result.stats.counters`. `'timings'` additionally samples
   * `performance.now()` at top-level brackets (env.succ, reconstruct) and
   * populates `result.stats.timings`. `'off'` opts the counters out entirely
   * (use only when every cycle matters and you've already optimized).
   */
  profile?: PerfMode;
  /**
   * Weighted-A* multiplier on the heuristic (Pohl 1970): `f = g + weight·h`.
   * `weight = 1` (default) is pure admissible A* — returned plan is optimal.
   * `weight > 1` is ε-suboptimal: returned plan cost is bounded by
   * `weight × cost*` but expansion count drops dramatically (typically
   * 2-10× fewer expansions for `weight = 1.5`, more for larger). Useful in
   * anytime mode — the planner's anytime loop still improves the incumbent
   * over time, so a heavy initial `weight` plus a generous deadline gives
   * both a fast first plan and continued tightening. The environment's
   * `h` is unchanged (kept admissible); only the planner's f-ordering is
   * inflated.
   */
  weight?: number;
}

export interface PlanRequest<State> {
  start: State;
  goal: State;
  environment: Environment<State>;
  options?: PlannerOptions;
}

export interface PlanStats {
  expansions: number;
  generated: number;
  deadlineHit: boolean;
  budgetHit: boolean;
  passesRun: number;
  improvements: number;
  /** Always populated (zeroed when `profile: 'off'`). */
  counters: PlanCounters;
  /** Populated only when `profile: 'timings'`. */
  timings?: PlanTimings;
  /** One entry per resolution-level pass that actually ran. */
  perPass: PassStats[];
}

export interface PlanResult<State> {
  found: boolean;
  /** True when `found` is satisfied by a BEST-PROGRESS fallback (the env's
   *  `progress` hook) rather than by reaching an accepting/goal region — i.e.
   *  the objective was NOT formally satisfied but this is the furthest the
   *  search advanced. Always absent/false for ordinary goal-reaching plans. */
  partial?: boolean;
  cost: number;
  /** State sequence start → goal (JSON-serializable). */
  path: State[];
  /** Node sequence for the best path (not JSON-serializable; debug/tests). */
  nodes: Node<State>[];
  stats: PlanStats;
  /** Every improving solution found during the anytime search, best last. */
  solutionHistory: State[][];
}
