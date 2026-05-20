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
  cost: number;
  /** State sequence start → goal (JSON-serializable). */
  path: State[];
  /** Node sequence for the best path (not JSON-serializable; debug/tests). */
  nodes: Node<State>[];
  stats: PlanStats;
  /** Every improving solution found during the anytime search, best last. */
  solutionHistory: State[][];
}
