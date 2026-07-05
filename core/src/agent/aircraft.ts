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
    maxRollRate: Math.PI, // 180°/s — a full knife-edge takes 0.5 s
    maxPitchRate: 1,
    halfLength: 2,
    halfSpan: 1.5,
    halfHeight: 0.3,
    ...overrides,
  };
}

/** Move `current` toward `target` by at most `rate·dt`. */
function moveToward(
  current: number,
  target: number,
  rate: number,
  dt: number,
): number {
  const d = target - current;
  const step = rate * dt;
  if (d > step) return current + step;
  if (d < -step) return current - step;
  return target;
}

/**
 * Kinematic flight model for an agent without a host flight-dynamics engine.
 * Controls are SETPOINTS; state evolves continuously from its current value —
 * a primitive's effect on roll is whatever this sim integrates, exactly like
 * its effect on position. `controls = [curvature, climbTarget, rollTarget,
 * targetSpeed]`:
 *   - curvature is clamped to ±1/minTurnRadius (horizontal turn rate),
 *   - climbTarget (commanded flight-path angle) clamped to ±maxClimbAngle;
 *     pitch moves toward it at ≤ maxPitchRate,
 *   - rollTarget (commanded bank) clamped to ±maxBank; roll moves toward it
 *     at ≤ maxRollRate — in this kinematic model bank is footprint
 *     orientation (the planner uses it to slip the OBB through tight slots),
 *     and the rate limit is what makes knife-edge maneuvers a matter of
 *     TIMING rather than a free instant snap,
 *   - targetSpeed clamped to [minSpeed, maxSpeed] (a deliberate remaining
 *     setpoint: constant speed per primitive; give the agent an accel model
 *     via a custom ForwardSim if you need speed to evolve).
 * `maxRollRate: Infinity` / `maxPitchRate: Infinity` recover the legacy
 * quasi-static snap model. Used by tests and the planner's successor
 * integration; a real game supplies a physics-backed ForwardSim instead.
 */
export function aircraftForwardSim(
  agent: AircraftAgent,
): ForwardSim<AircraftState> {
  const kMax = 1 / agent.minTurnRadius;
  return (s: AircraftState, controls: number[], dt: number): AircraftState => {
    const curvature = clamp(controls[0] ?? 0, -kMax, kMax);
    const pitchTarget = clamp(
      controls[1] ?? 0,
      -agent.maxClimbAngle,
      agent.maxClimbAngle,
    );
    const rollTarget = clamp(controls[2] ?? 0, -agent.maxBank, agent.maxBank);
    const speed = clamp(
      controls[3] ?? agent.maxSpeed,
      agent.minSpeed,
      agent.maxSpeed,
    );
    const pitch = moveToward(s.pitch, pitchTarget, agent.maxPitchRate, dt);
    const roll = moveToward(s.roll, rollTarget, agent.maxRollRate, dt);
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
