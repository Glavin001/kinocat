// kinocat/vehicle/car — car-domain helpers + drivers + encoders.
//
// Picks concrete types `<CarKinematicState, WheeledCarControls>` for the
// generic scene runtime in `kinocat/scene`. Pair with a `Body<...>` impl
// (see `kinocat/adapters/rapier`'s `RapierCarBody`) for the full car stack.

export type { CarKinematicState, WheeledCarControls } from './types';
export { trimPlan, trimCarPlan, samplePlanAt } from './plan-utils';
export {
  keyboardAckermann,
  keysFromSet,
  type KeyState,
  type AckermannKeyboardCommand,
  type KeyboardOpts,
} from './keyboard';
export { followPlan, type FollowPlanOpts } from './follow-plan';
export {
  encodeForParametricV2,
  encodeForKinematic,
  encodeWheeledRaw,
} from './encoders';
export {
  wheeledFromNormalized,
  ZERO_WHEELED,
  type CarForceTuning,
  type NormalizedCarCommand,
} from './wheeled';
export {
  KeyboardCarDriver,
  type KeyboardCarDriverOpts,
  PlanFollowerCarDriver,
  type PlanFollowerCarDriverOpts,
  PlaybackPatternCarDriver,
  type PlaybackPatternSegment,
} from './drivers';
export { carRecorderFormatters } from './recorder-formatters';

// Maneuver library — Phase 1 of the training-dataset plan.
export type {
  ManeuverLimits,
  ManeuverSpec,
  OuParams,
  MixtureMode,
} from './maneuvers';
export {
  seededRng,
  ouControls,
  mixtureRandomWalk,
  throttleRelease,
  throttleToBrake,
  brakeToThrottle,
  steerToZero,
  steerReversal,
  panicTurn,
  liftOffOversteer,
  reverseWithSteer,
  stepSteer,
  sinSweepSteer,
  slalom,
  trailBrake,
  throttleOnApex,
  jTurn,
  fishhook,
  scandinavianFlick,
  doubleLaneChange,
  donut,
  defaultManeuverBundle,
  passiveCoast,
  wheelspin,
  stuckState,
  multiCuspParkingScript,
  threePointTurnScript,
  universalManeuverBundle,
} from './maneuvers';

// Coverage projection — Phase 0 of the training-dataset plan.
export {
  CAR_COVERAGE_AXES,
  carCoverageProjection,
  wheeledControlsToVec,
} from './coverage-projection';
