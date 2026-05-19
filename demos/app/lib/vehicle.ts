import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import type { VehicleAgent } from 'kinocat/agent';

/** Shared demo vehicle + a small motion-primitive library. */
export function buildVehicle(overrides: Partial<VehicleAgent> = {}) {
  const agent = defaultVehicleAgent({
    minTurnRadius: 3,
    maxSpeed: 8,
    maxReverseSpeed: 4,
    footprint: [
      [1.2, 0.6],
      [-1.2, 0.6],
      [-1.2, -0.6],
      [1.2, -0.6],
    ],
    ...overrides,
  });
  const k = 1 / agent.minTurnRadius;
  const lib = characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 6],
      [k, 6],
      [-k, 6],
      [k / 2, 6],
      [-k / 2, 6],
      [0, -4],
      [k, -4],
      [-k, -4],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
  return { agent, lib };
}

export const PALETTE = {
  bg: '#0b0b0f',
  floor: '#161a22',
  obstacle: '#5a2230',
  path: '#44ddff',
  start: '#55ff88',
  goal: '#ffcc33',
  agent: '#7fd6ff',
  ghost: '#9b6cff',
};
