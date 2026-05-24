// kinocat/vehicle/car — car-domain helpers + drivers + encoders.
//
// Picks concrete types `<CarKinematicState, WheeledCarControls>` for the
// generic scene runtime in `kinocat/scene`. Pair with a `Body<...>` impl
// (see `kinocat/adapters/rapier`'s `RapierCarBody`) for the full car stack.

export type { CarKinematicState, WheeledCarControls } from './types';
export { trimPlan, trimCarPlan } from './plan-utils';
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
