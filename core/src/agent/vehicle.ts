import type { VehicleAgent, VehicleState } from './types';
import type { ForwardSim } from '../primitives/types';
import { clamp, wrapAngle } from '../internal/math';

// Default footprint sized to encompass the default Rapier raycast-vehicle
// chassis (half-extents 2.4 × 1.0, see adapters/rapier/raycast-vehicle.ts)
// plus a 0.15 m baseline clearance buffer all around. Sizing the planning
// polygon below the physics chassis lets paths clip walls during execution;
// keeping a small buffer here means any default-built agent plans with
// real-world wall margin out of the box.
const DEFAULT_FOOTPRINT: [number, number][] = [
  [2.55, 1.15],
  [-2.55, 1.15],
  [-2.55, -1.15],
  [2.55, -1.15],
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
    // Smooth saturating accel: `dir * tanh(speedErr / (dir * tau))`
    //   - At small |speedErr|: accel ≈ speedErr / tau (linear / proportional,
    //     so `tau` actually matters for cruise samples)
    //   - At large |speedErr|: accel saturates at ±dir (= ±maxAccel for
    //     accelerating, ±maxDecel for braking)
    //   - C¹ smooth across the transition
    // The OLD `clamp(speedErr/tau, ...)` form let the fit park `tau` at the
    // lower bound for free because — at small tau — the clamp was active
    // for nearly every sample, making `tau` a degenerate parameter (only
    // affected the few cruise samples). With tanh, `tau` controls the
    // proportional gain everywhere and has gradient signal in every sample.
    const dir = speedErr >= 0 ? params.maxAccel : params.maxDecel;
    const denom = Math.max(dir * tau, 1e-6);
    const accel = dir * Math.tanh(speedErr / denom);
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
