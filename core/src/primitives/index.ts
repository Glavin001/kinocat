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
