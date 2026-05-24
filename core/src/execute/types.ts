import type { CarKinematicState } from '../agent/types';

export interface PurePursuitConfig {
  /** Minimum lookahead distance (world units). */
  lookaheadMin: number;
  /** Lookahead added per unit current speed. */
  lookaheadGain: number;
  /** Maximum lookahead distance. */
  lookaheadMax: number;
  /** Curvature-aware speed cap: v = sqrt(maxLateralAccel / |κ|). */
  maxLateralAccel: number;
  maxAccel: number;
  maxDecel: number;
  /** Free-running cruise speed magnitude. */
  cruiseSpeed: number;
  /** Distance at which the goal is considered reached. */
  goalTolerance: number;
  /** Optional min turning radius to clamp commanded curvature. */
  minTurnRadius?: number;
}

export interface TrackingCommand {
  /** Commanded path curvature (1/radius), signed. */
  steering: number;
  /** Throttle ∈ [0,1] (toward target speed magnitude). */
  throttle: number;
  /** Brake ∈ [0,1]. */
  brake: number;
  /** Signed target speed (negative ⇒ reverse). */
  targetSpeed: number;
  /** Lookahead point used this tick. */
  lookahead: { x: number; z: number };
  atGoal: boolean;
}

export interface ReplanTrigger {
  divergenceThresholdMeters: number;
  refreshIntervalMs: number;
  /**
   * Plan-switch hysteresis. On a routine *periodic* replan (agent on-track,
   * current plan still valid) a freshly-planned route is adopted only if it
   * is cheaper than the committed plan by at least this fraction (default
   * 0.15) plus `switchCostMargin`. Prevents flip-flopping between two
   * equal-cost routes around a symmetric obstacle. Divergence / dirty
   * replans always adopt (the current plan is no longer valid).
   */
  switchCostImprovement?: number;
  switchCostMargin?: number;
}

export type ReplanReason = 'no-plan' | 'dirty' | 'divergence' | 'periodic' | 'none';

export type PlanPath = CarKinematicState[];
