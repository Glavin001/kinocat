// kinocat/agent — vehicle / humanoid agent metadata + default forward models.
export type {
  VehicleState,
  HumanoidState,
  AircraftState,
  AgentState,
  VehicleAgent,
  HumanoidAgent,
  AircraftAgent,
  AgentModel,
} from './types';
export {
  defaultVehicleAgent,
  kinematicForwardSim,
  learnedForwardSim,
  DEFAULT_LEARNED_PARAMS,
} from './vehicle';
export type { LearnedVehicleParams } from './vehicle';
export { defaultHumanoidAgent } from './humanoid';
export { defaultAircraftAgent, aircraftForwardSim } from './aircraft';

// Generic wheeled-vehicle action shape + clamp helpers.
export type { WheeledControls } from './controls';
export {
  WHEELED_CONTROL_DIM,
  encodeWheeled,
  decodeWheeled,
  clampWheeled,
} from './controls';

// Generic wheeled-vehicle physical configuration (a learned model can be
// conditioned on, identified from, or generalize over).
export type { LearnableVehicleConfig, DrivenWheels } from './vehicle-config';
export {
  DEFAULT_LEARNABLE_CONFIG,
  LEARNABLE_CONFIG_VEC_DIM_ORDINAL,
  LEARNABLE_CONFIG_VEC_DIM_ONEHOT,
  encodeConfigOrdinal,
  encodeConfigOneHot,
  CONFIG_SCALES_ORDINAL,
} from './vehicle-config';

// V2 learned dynamics: extended parametric backbone (config-aware,
// friction-circle aware, yaw-rate inertia, asymmetric understeer/oversteer)
// + optional MLP residual ensemble for uncertainty-aware prediction.
export type {
  LearnedVehicleParamsV2,
  LearnedVehicleModel,
  PredictionWithUncertainty,
} from './vehicle-model';
export {
  DEFAULT_LEARNED_PARAMS_V2,
  parametricForwardV2,
  learnedForwardSimV2,
  predictWithUncertainty,
  buildParametricOnlyModel,
} from './vehicle-model';
