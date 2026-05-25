// kinocat/primitives — motion-primitive library + characterization harness.
export type {
  ForwardSim,
  LocalPose,
  MotionPrimitive,
  SerializedLibrary,
} from './types';
export { MotionPrimitiveLibrary } from './library';
export {
  characterizeVehicle,
  type CharacterizeVehicleOptions,
} from './characterize';

export {
  coarseWheeledControls,
  fineWheeledControls,
  DEFAULT_WHEELED_START_SPEEDS,
  FINE_WHEELED_START_SPEEDS,
  type WheeledControlTierOptions,
} from './control-sets-wheeled';
