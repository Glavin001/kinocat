// kinocat/eval — component-level evaluation harness for the planner &
// controller. Decomposes every run into (1) the plan as a static artifact,
// (2) the execution, and (3) the gap between them (= controller tracking
// error), so problems localize to the responsible component (evaluation guide
// §1). All pieces are pure and framework-free; scenario orchestration lives in
// the demos package.

export {
  toReferenceTrajectory,
  referencePoseAt,
  referenceLength,
  type ReferencePoint,
  type ReferenceTrajectory,
} from './reference-trajectory';

export { projectOntoPath, type Projection } from './projection';

export {
  trackingMetrics,
  runControllerIsolation,
  type TrackingReport,
  type ErrorStats,
  type CrossTrackStats,
  type TrackingMetricsOptions,
  type RefController,
  type ControllerIsolationResult,
} from './tracking-metrics';

export {
  checkFeasibility,
  limitsFromAgent,
  type DynamicLimits,
  type FeasibilityReport,
  type FeasibilityViolation,
  type ViolationKind,
} from './feasibility';

export { ggUtilization, type GgReport, type GgPoint } from './gg-utilization';

export {
  comfortFlags,
  DEFAULT_COMFORT_BOUNDS,
  type ComfortBounds,
  type ComfortReport,
} from './comfort';

export {
  scorePlan,
  rolloutTeleportFollow,
  type PlanQualityReport,
  type TerminalAccuracy,
  type ScorePlanOptions,
} from './plan-quality';

export {
  diagnose,
  DEFAULT_DIAGNOSIS_THRESHOLDS,
  type Verdict,
  type Diagnosis,
  type DiagnosisThresholds,
} from './diagnosis';

export {
  straightLine,
  arcPath,
  laneChange,
  slalom,
} from './reference-shapes';

export {
  buildDogLegCorridor,
  assessPassability,
  sweptClearance,
  runGauntlet,
  type CorridorWorld,
  type BuildCorridorOptions,
  type PassabilityReport,
  type GauntletReport,
  type RunGauntletOptions,
} from './gauntlet';
