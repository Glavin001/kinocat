// kinocat/predict — Predict<T> factories + dynamic-object abstractions,
// multi-agent plan sharing, and affordances.
export type { Predict, MovingObstacle, AffordanceState } from './types';
export {
  constantVelocity,
  constantAcceleration,
  fromPhysicsRollout,
  fromObservations,
  linearObstacle,
  asObstacle,
} from './factories';
export {
  PlanRegistry,
  fromPublishedPlan,
  type PublishedPlan,
} from './plan-registry';
export {
  AffordanceRegistry,
  AffordanceType,
  createJumpAffordance,
  createBoostAffordance,
  createMisdirectAffordance,
  type Affordance,
  type AffordanceUseResult,
} from './affordance-registry';
export {
  createEtaOracle,
  type EtaOracle,
  type EtaOracleOptions,
  type EtaResult,
} from './eta-oracle';
