import type { AircraftAgent, AircraftState } from './types';
import type { ForwardSim } from '../primitives/types';
import { clamp, wrapAngle } from '../internal/math';

export function defaultAircraftAgent(
  overrides: Partial<AircraftAgent> = {},
): AircraftAgent {
  return {
    kind: 'aircraft',
    minTurnRadius: 14,
    minSpeed: 6,
    maxSpeed: 18,
    maxClimbAngle: Math.PI / 6,
    radius: 1.4,
    ...overrides,
  };
}

/**
 * Kinematic flight model for an agent without a host flight-dynamics engine.
 * `controls = [curvature, climbAngle, targetSpeed]`:
 *   - curvature is clamped to ±1/minTurnRadius (a coordinated bank-to-turn),
 *   - climbAngle (the commanded flight-path angle) clamped to ±maxClimbAngle,
 *   - targetSpeed clamped to [minSpeed, maxSpeed].
 * Pitch tracks the commanded climb angle within the step (a quasi-static
 * airframe); airspeed is constant along the 3D path, so its ground projection
 * shrinks as the climb angle steepens. Used by the characterization/tests and
 * the planner's successor integration; a real game supplies a physics-backed
 * ForwardSim instead.
 */
export function aircraftForwardSim(
  agent: AircraftAgent,
): ForwardSim<AircraftState> {
  const kMax = 1 / agent.minTurnRadius;
  return (s: AircraftState, controls: number[], dt: number): AircraftState => {
    const curvature = clamp(controls[0] ?? 0, -kMax, kMax);
    const pitch = clamp(
      controls[1] ?? 0,
      -agent.maxClimbAngle,
      agent.maxClimbAngle,
    );
    const speed = clamp(
      controls[2] ?? agent.maxSpeed,
      agent.minSpeed,
      agent.maxSpeed,
    );
    const heading = wrapAngle(s.heading + speed * curvature * dt);
    const hs = speed * Math.cos(pitch);
    return {
      x: s.x + hs * Math.cos(s.heading) * dt,
      z: s.z + hs * Math.sin(s.heading) * dt,
      y: s.y + speed * Math.sin(pitch) * dt,
      heading,
      pitch,
      speed,
      t: s.t + dt,
    };
  };
}
