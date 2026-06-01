// Plan feasibility (evaluation guide §4.2a): walk a reference trajectory and
// check every point against the car's TRUE dynamic limits. An infeasible plan
// cannot be executed by ANY controller, so its missed turns / understeer are a
// PLANNER fault, not a controller fault — this check must run before diagnosing
// the controller (guide §6).

import type { VehicleAgent } from '../agent/types';
import type { ReferenceTrajectory } from './reference-trajectory';

/** The car's true dynamic limits (SI). */
export interface DynamicLimits {
  /** Friction-circle radius — max combined accel μ·g (m/s²). Also the lateral
   *  accel a single corner may demand. */
  frictionLimit: number;
  /** Minimum turning radius (m). */
  minTurnRadius: number;
  /** Max longitudinal acceleration (m/s²). */
  maxAccel: number;
  /** Max longitudinal deceleration (m/s², positive magnitude). */
  maxDecel: number;
  /** Optional cap on curvature rate of change |dκ/ds|·v (1/(m·s)·... ) used as
   *  a proxy for "implied steering rate must be achievable". Omit to skip. */
  maxCurvatureRate?: number;
}

/** Build limits from a `VehicleAgent` and a friction-circle / accel budget. The
 *  `frictionLimit` mirrors the `maxLateralAccel` pure-pursuit and the
 *  speed-profile smoother already use, so feasibility is checked against the
 *  same envelope the planner/controller are tuned to. */
export function limitsFromAgent(
  agent: VehicleAgent,
  budget: { frictionLimit: number; maxAccel: number; maxDecel: number; maxCurvatureRate?: number },
): DynamicLimits {
  return {
    frictionLimit: budget.frictionLimit,
    minTurnRadius: agent.minTurnRadius,
    maxAccel: budget.maxAccel,
    maxDecel: budget.maxDecel,
    maxCurvatureRate: budget.maxCurvatureRate,
  };
}

export type ViolationKind =
  | 'lateral-accel'
  | 'turn-radius'
  | 'longitudinal-accel'
  | 'curvature-rate';

export interface FeasibilityViolation {
  index: number;
  s: number;
  kind: ViolationKind;
  /** The demanded value at this point. */
  value: number;
  /** The limit it exceeded. */
  limit: number;
}

export interface FeasibilityReport {
  feasible: boolean;
  violations: FeasibilityViolation[];
  counts: Record<ViolationKind, number>;
  /** Worst (largest) demand/limit ratio across all checks; ≤ 1 ⇒ feasible. */
  worstRatio: number;
}

/** Check a plan's feasibility against the car's dynamic limits. */
export function checkFeasibility(
  ref: ReferenceTrajectory,
  limits: DynamicLimits,
  opts?: { tolerance?: number },
): FeasibilityReport {
  const tol = 1 + (opts?.tolerance ?? 1e-3);
  const violations: FeasibilityViolation[] = [];
  const counts: Record<ViolationKind, number> = {
    'lateral-accel': 0,
    'turn-radius': 0,
    'longitudinal-accel': 0,
    'curvature-rate': 0,
  };
  let worstRatio = 0;

  const record = (
    index: number,
    s: number,
    kind: ViolationKind,
    value: number,
    limit: number,
  ): void => {
    const ratio = limit > 1e-9 ? value / limit : Infinity;
    if (ratio > worstRatio) worstRatio = ratio;
    if (ratio > tol) {
      violations.push({ index, s, kind, value, limit });
      counts[kind]++;
    }
  };

  for (let i = 0; i < ref.length; i++) {
    const p = ref[i]!;
    const v = Math.abs(p.v);
    const k = Math.abs(p.kappa);

    // Lateral acceleration demanded by driving the planned speed through the
    // planned curvature: a_lat = v²·|κ|.
    record(i, p.s, 'lateral-accel', v * v * k, limits.frictionLimit);

    // Turning radius vs the car's Ackermann minimum (radius = 1/κ).
    if (k > 1e-6) {
      const radius = 1 / k;
      // Demand is the curvature; limit is the max curvature 1/minTurnRadius.
      record(i, p.s, 'turn-radius', k, 1 / Math.max(limits.minTurnRadius, 1e-6));
      void radius;
    }

    // Longitudinal accel within the accel/decel envelope.
    if (p.a >= 0) record(i, p.s, 'longitudinal-accel', p.a, limits.maxAccel);
    else record(i, p.s, 'longitudinal-accel', -p.a, limits.maxDecel);

    // Optional curvature-rate proxy for steering rate.
    if (limits.maxCurvatureRate !== undefined && i > 0) {
      const prev = ref[i - 1]!;
      const ds = p.s - prev.s;
      if (ds > 1e-6) {
        const rate = (Math.abs(Math.abs(p.kappa) - Math.abs(prev.kappa)) / ds) * v;
        record(i, p.s, 'curvature-rate', rate, limits.maxCurvatureRate);
      }
    }
  }

  return { feasible: violations.length === 0, violations, counts, worstRatio };
}
