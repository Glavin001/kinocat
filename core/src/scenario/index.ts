// kinocat/scenario — canonical, serializable Scenario & Goal Specification
// Layer. A scenario is (Objective, Invariants, Cost) over a shared set of
// Regions: objectives (reach/repeat composed by seq/all/any) compile to an
// AUTOMATON; invariants (maintain/avoid) compile to PRUNING; preferences
// (prefer) compile to COST. The canonical form is a serializable AST; the
// fluent builders are sugar over it.

export type {
  ScenarioState,
  RegionAgent,
  Region,
  Bound,
  Acceptance,
  Goal,
  ReachGoal,
  SeqGoal,
  AllGoal,
  AnyGoal,
  RepeatGoal,
  Invariant,
  AvoidInvariant,
  MaintainInvariant,
  CostTerm,
  Scenario,
} from './types';

// Regions
export {
  at,
  near,
  inside,
  gate,
  corridor,
  halfPlane,
  FORWARD,
  BACKWARD,
} from './regions';
export type { Pose, PoseMargins, GateDir } from './regions';
export {
  within,
  ahead,
  behind,
  beside,
  cone,
  LEFT,
  RIGHT,
} from './regions-dynamic';
export type { Side } from './regions-dynamic';

// Conditions
export { lte, gte, inRange, speed, distanceFrom } from './conditions';

// Builders (sugar)
export {
  reach,
  seq,
  all,
  any,
  repeat,
  avoid,
  maintain,
  stayInside,
  defineScenario,
  deg,
} from './builders';
export type { ScopedMaintain, DefineScenarioInput } from './builders';

// Cost terms
export { minTime, smooth, keepClear, racingLine, maxProgress } from './cost';

// Normalization / hashing
export { normalize, hashGoal, structuralEqual, TOP, BOTTOM } from './normalize';

// Compilation
export { compile, nextGuardPose } from './automaton';
export type {
  CompiledAutomaton,
  AutomatonState,
  Transition,
  GuardPredicate,
} from './automaton';

// Guard evaluation (shared with the planner bridge + visualizer)
export { guardSatisfied, checkAcceptance } from './guard';

// Progress evaluation (deterministic, for the visualizer)
export { evaluateProgress, stepAutomaton } from './progress';
export type { ProgressSnapshot } from './progress';

// Validation
export { validate } from './validate';
export type { Diagnostic, Severity, ValidateOptions } from './validate';

// Diagram
export { toMermaid } from './diagram';

// AST walkers (visualizer / rubric / generators)
export {
  goalRegions,
  avoidRegions,
  maintainRegions,
  collectScenarioRegions,
} from './walk';
export type { ScenarioRegions } from './walk';
