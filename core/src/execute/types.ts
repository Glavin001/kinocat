import type { CarKinematicState } from '../agent/types';

export interface PurePursuitConfig {
  /** Minimum lookahead distance (world units). */
  lookaheadMin: number;
  /** Lookahead added per unit current speed. */
  lookaheadGain: number;
  /** Maximum lookahead distance. */
  lookaheadMax: number;
  /**
   * Floor for the commanded approach speed toward a stop terminal (m/s).
   * Keeps the brake-to-goal ramp from asymptoting to zero far from the goal.
   * When unset, falls back to `lookaheadMin` — a historic unit bug (a DISTANCE
   * used as a speed) that races happened to tune around; parking must set a
   * real value (~0.3-0.5 m/s) or the ramp never engages below cruise and the
   * terminal approach is bang-bang with a 0.1-0.5 m brake skid.
   */
  minApproachSpeed?: number;
  /**
   * Cruise cap while in REVERSE gear (m/s). Without it `cruiseSpeed` applies
   * to both gears, so a chassis whose reverse envelope is lower (agents
   * typically back up at 60-75% of forward speed) is commanded to reverse at
   * forward cruise — it over-speeds the planned reverse arcs, saturates its
   * curvature authority, and drifts wide. Defaults to `cruiseSpeed`.
   */
  reverseCruiseSpeed?: number;
  /**
   * Feed the reference path's local curvature forward into the steering
   * command (kappa = kappa_ff + pursuit feedback). Without it, pure pursuit
   * must ACCUMULATE cross-track error to generate the curvature of an arc —
   * on max-curvature parking swings that steady-state error is 0.1-0.3 m,
   * exactly the clearance margin the plan reserved. With feedforward the
   * feedback term only corrects disturbances. Off by default (race tuning
   * predates it); parking enables it.
   */
  curvatureFeedforward?: boolean;
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
   * When true, the tracker also caps target speed by the minimum planned
   * `speed` over a forward window of the path (default ~lookaheadMax
   * worth of arc-length). This is how the controller actually consumes
   * a friction-circle / speed-profile smoother's output — without this
   * the curvature-aware `vCurve` is the only forward-looking term and
   * the smoothed profile is ignored. Default false (legacy behaviour).
   */
  respectPathSpeed?: boolean;
  /** Optional Stanley-style heading-alignment gain. When > 0, a curvature term
   *  proportional to the heading error against the local path tangent is added
   *  (forward gear only), so the tracker drives the chassis onto the planned
   *  heading rather than only chasing the lookahead point. Default 0 (off) —
   *  used by parking for terminal-pose precision; racing leaves it off. */
  headingGain?: number;
  /** Only apply `headingGain` within this distance of the goal (m). Confines the
   *  heading correction to the clear terminal zone so it doesn't perturb the
   *  chassis off a tight, clearance-critical earlier part of the path. Default
   *  Infinity (apply whenever headingGain > 0). */
  headingRadius?: number;
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
