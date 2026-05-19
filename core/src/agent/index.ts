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
export { defaultVehicleAgent, kinematicForwardSim } from './vehicle';
export { defaultHumanoidAgent } from './humanoid';
export { defaultAircraftAgent, aircraftForwardSim } from './aircraft';
