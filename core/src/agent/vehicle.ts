import type { VehicleAgent, VehicleState } from './types';
import type { ForwardSim } from '../primitives/types';
import { clamp, wrapAngle } from '../internal/math';

const DEFAULT_FOOTPRINT: [number, number][] = [
  [1.6, 0.9],
  [-1.6, 0.9],
  [-1.6, -0.9],
  [1.6, -0.9],
];

export function defaultVehicleAgent(overrides: Partial<VehicleAgent> = {}): VehicleAgent {
  return {
    kind: 'vehicle',
    minTurnRadius: 4,
    maxSpeed: 12,
    maxReverseSpeed: 4,
    footprint: DEFAULT_FOOTPRINT,
    reverseCostMultiplier: 2,
    directionChangePenalty: 0.5,
    ...overrides,
  };
}

/**
 * Kinematic forward model for an agent without a host physics engine.
 * `controls = [curvature, targetSpeed]`. Curvature is clamped to the agent's
 * minimum turning radius; speed is clamped to its forward/reverse limits.
 * Used by the characterization harness and tests; real games supply a
 * physics-backed ForwardSim instead.
 */
export function kinematicForwardSim(agent: VehicleAgent): ForwardSim<VehicleState> {
  const kMax = 1 / agent.minTurnRadius;
  return (s: VehicleState, controls: number[], dt: number): VehicleState => {
    const curvature = clamp(controls[0] ?? 0, -kMax, kMax);
    const target = clamp(controls[1] ?? 0, -agent.maxReverseSpeed, agent.maxSpeed);
    // first-order speed tracking toward target
    const speed = target;
    const heading = wrapAngle(s.heading + speed * curvature * dt);
    return {
      x: s.x + speed * Math.cos(s.heading) * dt,
      z: s.z + speed * Math.sin(s.heading) * dt,
      heading,
      speed,
      t: s.t + dt,
    };
  };
}
