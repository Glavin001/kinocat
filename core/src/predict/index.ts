// kinocat/predict — Predict<T> factories + dynamic-object abstractions.
export type { Predict, MovingObstacle, AffordanceState } from './types';
export {
  constantVelocity,
  constantAcceleration,
  fromPhysicsRollout,
  fromObservations,
  linearObstacle,
  asObstacle,
} from './factories';
