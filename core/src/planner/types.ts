import type { Environment, Node } from '../environment/types';
import type { HysteresisOptions } from './hysteresis';

export interface PlannerOptions {
  /** Resolution levels to use; defaults to `environment.levels`. */
  levels?: number;
  /** Hard cap on total expansions across all passes. */
  maxExpansions?: number;
  /** Wall-clock check granularity (expansions between `now()` checks). */
  deadlineCheckEvery?: number;
  hysteresis?: HysteresisOptions;
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
