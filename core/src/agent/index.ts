// kinocat/agent — vehicle / humanoid agent metadata + default forward models.
export type {
  VehicleState,
  HumanoidState,
  AgentState,
  VehicleAgent,
  HumanoidAgent,
  AgentModel,
} from './types';
export { defaultVehicleAgent, kinematicForwardSim } from './vehicle';
export { defaultHumanoidAgent } from './humanoid';
