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
  /** Cruise cap when tracking in REVERSE gear (m/s, positive magnitude).
   *  Defaults to `cruiseSpeed` for backward compatibility — but racing
   *  configs MUST set this to the chassis's reverse limit: without it a
   *  reverse-gear plan segment executes at forward cruise speed
   *  (measured: −24 m/s down a 100 m straight, far outside both the
   *  chassis envelope and the model's training distribution). */
  reverseCruiseSpeed?: number;
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
  /**
   * When true, the tracker caps target speed by UPCOMING path curvature
   * through the braking envelope: for each sample at arc distance d with
   * local (Menger) curvature κ, allowed = sqrt(maxLateralAccel/|κ| +
   * 2·maxDecel·d). Without this, `vCurve` sees only the instantaneous
   * chord to the lookahead point, so a plan that runs straight into a
   * tight corner (e.g. a dynamics-honest brake-then-turn plan) is
   * entered at full speed and overshot. Pure geometry — no trust in the
   * plan's per-sample speeds required. Default false (legacy behaviour).
   */
  previewCurvature?: boolean;
  /** Lateral-accel budget used by the curvature preview's corner-speed
   *  term (m/s²). Defaults to `maxLateralAccel`. Set this to the plant's
   *  physical grip ceiling (μ·g, see deriveVehicleCapabilities) when
   *  `maxLateralAccel` is a comfort cap — braking preview should trigger
   *  at the physical limit, not the comfort limit, or clean laps slow
   *  down across the board. */
  previewLateralAccel?: number;
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
