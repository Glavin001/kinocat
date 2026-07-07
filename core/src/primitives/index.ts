// kinocat/primitives — motion-primitive library + characterization harness.
export type {
  ForwardSim,
  LocalPose,
  MotionPrimitive,
  SerializedLibrary,
} from './types';
export { MotionPrimitiveLibrary } from './library';
export {
  characterize,
  crossRuns,
  characterizeVehicle,
  characterizeVehicleFromState,
  type CharacterizeOptions,
  type CharacterizeRun,
  type CharacterizedPrimitive,
  type CharacterizeVehicleOptions,
} from './characterize';

export {
  coarseWheeledControls,
  fineWheeledControls,
  DEFAULT_WHEELED_START_SPEEDS,
  FINE_WHEELED_START_SPEEDS,
  type WheeledControlTierOptions,
} from './control-sets-wheeled';

export {
  designControlSet,
  coverageReport,
  rollEndpoint,
  endpointDistance,
  type ControlSetDesignOptions,
  type CoverageReport,
  type Endpoint,
} from './control-set-design';
