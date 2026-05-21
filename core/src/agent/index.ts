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
