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
    maxBank: Math.PI / 2,
    halfLength: 2,
    halfSpan: 1.5,
    halfHeight: 0.3,
    ...overrides,
  };
}

/**
 * Kinematic flight model for an agent without a host flight-dynamics engine.
 * `controls = [curvature, climbAngle, rollTarget, targetSpeed]`:
 *   - curvature is clamped to ±1/minTurnRadius (horizontal turn rate),
 *   - climbAngle (the commanded flight-path angle) clamped to ±maxClimbAngle,
 *   - rollTarget (the commanded bank angle) clamped to ±maxBank — purely a
 *     footprint-orientation control in this kinematic model; the planner uses
 *     it to slip the OBB through tight slots,
 *   - targetSpeed clamped to [minSpeed, maxSpeed].
 * Pitch and roll track their commanded targets within the step (a
 * quasi-static airframe). Used by tests and the planner's successor
 * integration; a real game supplies a physics-backed ForwardSim instead.
 */
export function aircraftForwardSim(
  agent: AircraftAgent,
): ForwardSim<AircraftState> {
  const baseKMax = 1 / agent.minTurnRadius;
  return (s: AircraftState, controls: number[], dt: number): AircraftState => {
    const pitch = clamp(
      controls[1] ?? 0,
      -agent.maxClimbAngle,
      agent.maxClimbAngle,
    );
    const roll = clamp(controls[2] ?? 0, -agent.maxBank, agent.maxBank);
    const speed = clamp(
      controls[3] ?? agent.maxSpeed,
      agent.minSpeed,
      agent.maxSpeed,
    );
    // Speed-dependent turn cap: agility shrinks as speed grows when the
    // agent provides a `turnRadiusAt` profile (real airframes obey ≈v²/g).
    const kMax = agent.turnRadiusAt
      ? 1 / agent.turnRadiusAt(speed)
      : baseKMax;
    const curvature = clamp(controls[0] ?? 0, -kMax, kMax);
    const heading = wrapAngle(s.heading + speed * curvature * dt);
    const hs = speed * Math.cos(pitch);
    return {
      x: s.x + hs * Math.cos(s.heading) * dt,
      z: s.z + hs * Math.sin(s.heading) * dt,
      y: s.y + speed * Math.sin(pitch) * dt,
      heading,
      pitch,
      roll,
      speed,
      t: s.t + dt,
    };
  };
}
