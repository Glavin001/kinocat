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
  /**
   * When true, the tracker also caps target speed by the brake-distance-
   * adjusted minimum planned `speed` over a forward window of the path
   * (default ~lookaheadMax worth of arc-length). This is how the
   * controller actually consumes a friction-circle / speed-profile
   * smoother's output — without this the curvature-aware `vCurve` is
   * the only forward-looking term and the smoothed profile is ignored.
   * Default false (legacy behaviour).
   */
  respectPathSpeed?: boolean;
  /**
   * Floor on per-sample plan speeds when `respectPathSpeed` is on.
   * Samples with `|speed| < minPathSpeed` are skipped in the
   * brake-distance pass. Without this, an interior 0-speed primitive
   * (e.g., the planner included a [0,0,brake] primitive at a slow
   * start, or the plan begins from a stationary chassis) forces the
   * controller to brake to 0 and stall. Racing scenarios set this
   * non-zero so honest cruise-speed primitives dominate; parking
   * scenarios leave it 0 so stop intents are honoured. Default 0.
   */
  minPathSpeed?: number;
  /**
   * Compute `vCurve` from the max curvature in a forward window of
   * the plan polyline (with brake-distance backward sweep) instead
   * of the single reactive lookahead-point curvature. Lets the
   * controller anticipate corners and brake before entry instead of
   * over-shooting because vCurve only fired when the chassis was
   * already in the corner. Racing scenarios benefit; parking
   * scenarios should leave this off (the planner's tight in-stall
   * curvature would crawl the chassis to a halt). Default false.
   */
  lookaheadCurvature?: boolean;
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
