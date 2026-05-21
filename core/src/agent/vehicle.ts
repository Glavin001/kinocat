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

/**
 * Five-coefficient parametric dynamics model the learner fits against Rapier
 * trial data. Captures what `kinematicForwardSim` misses: first-order speed
 * lag, finite accel/decel, understeer (effective curvature drops with v²) and
 * lateral drag (turning at speed costs forward velocity).
 */
export interface LearnedVehicleParams {
  /** Max forward acceleration (m/s²) — roughly engineForce / mass. */
  maxAccel: number;
  /** Max braking deceleration magnitude (m/s²). */
  maxDecel: number;
  /** First-order speed-tracking time constant (s). */
  accelTau: number;
  /** Curvature degradation with speed²: κ_eff = κ / (1 + g·v²). */
  understeerGain: number;
  /** Forward speed loss while turning: dv -= drag·κ²·v·|v|·dt. */
  lateralDrag: number;
}

/** Sensible starting point for the learner's optimisation. Roughly matches the
 *  default Rapier chassis (4000N engine, ~580kg). */
export const DEFAULT_LEARNED_PARAMS: LearnedVehicleParams = {
  maxAccel: 6.5,
  maxDecel: 8,
  accelTau: 0.2,
  understeerGain: 0.01,
  lateralDrag: 0.05,
};

/**
 * Same `ForwardSim<VehicleState>` shape as `kinematicForwardSim`, but with a
 * dynamics model rich enough to reproduce a Rapier raycast vehicle inside
 * ~0.3m over 0.55s once `params` have been fit. Drop-in for
 * `characterizeVehicle()` → no other planner change needed.
 */
export function learnedForwardSim(
  params: LearnedVehicleParams,
  agent: VehicleAgent,
): ForwardSim<VehicleState> {
  const kMax = 1 / agent.minTurnRadius;
  return (s: VehicleState, controls: number[], dt: number): VehicleState => {
    const curvature = clamp(controls[0] ?? 0, -kMax, kMax);
    const target = clamp(controls[1] ?? 0, -agent.maxReverseSpeed, agent.maxSpeed);
    const tau = Math.max(params.accelTau, 1e-3);
    const speedErr = target - s.speed;
    const accel = clamp(speedErr / tau, -params.maxDecel, params.maxAccel);
    const v = s.speed;
    const latDrag = params.lateralDrag * curvature * curvature * v * Math.abs(v) * dt;
    const speed = v + accel * dt - latDrag;
    const effK = curvature / (1 + params.understeerGain * v * v);
    const heading = wrapAngle(s.heading + speed * effK * dt);
    return {
      x: s.x + speed * Math.cos(s.heading) * dt,
      z: s.z + speed * Math.sin(s.heading) * dt,
      heading,
      speed,
      t: s.t + dt,
    };
  };
}
