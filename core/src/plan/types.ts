// The rich Plan — the contract handed from the planner to the controller.
//
// Design principle: a field earns its place only if it carries information
// NOT recoverable from the plotted geometry (x, z, heading, t) + speed —
// equivalently, only if a motion primitive could naturally emit it. The
// plan's job is to hand over what the polyline can't show.
//
// That test organizes the per-point fields into three tiers:
//
//   Tier 1 — trajectory: what you plot. Derivable from geometry + speed.
//            `kappa`/`aRef`/`s` are convenience caches today; a future
//            primitive supplies kappa/aRef at higher fidelity than
//            finite-differencing a smoothed, discretized polyline.
//   Tier 2 — controls: the planner's actual work. NOT recoverable from
//            geometry (atan(L·κ) is only a kinematic approximation of the
//            steer that produced the arc). Highest value. Approximated
//            today; reserved for the real primitive controls.
//   Tier 3 — dynamic state & free space: model-only, zero geometric
//            redundancy. Reserved for LQR / MPC.
//
// Sign conventions: `vRef` is signed (negative = reverse) to match
// `CarKinematicState.speed`, so there is one source of truth for gear and
// `toStatePath` is a trivial copy. `kappa` is signed, left-positive in the
// world XZ plane, so `steerFf = atan(L·kappa)` steers the correct way.

/** +1 forward, -1 reverse. */
export type Direction = 1 | -1;

/** One densely time-sampled reference point along a plan. Geometry uses
 *  kinocat's world-XZ + `heading` (yaw) convention. Every field below
 *  Tier 1 is optional: a reserved slot a future primitive-upgrade fills,
 *  omitted by producers that can't supply it. */
export interface ReferencePoint {
  // --- Tier 1: trajectory (what you plot).
  /** Absolute time (s). */
  t: number;
  /** Cumulative arc length from plan start (m); monotonically non-decreasing. */
  s: number;
  x: number;
  z: number;
  /** Yaw (rad). */
  heading: number;
  /** Signed target speed (m/s); negative = reverse. */
  vRef: number;
  /** Signed path curvature (1/m), left-positive in XZ. Convenience cache;
   *  a future primitive emits this at higher fidelity. */
  kappa: number;
  /** Longitudinal accel d(vRef)/dt (m/s²). Convenience cache. */
  aRef: number;

  // --- Tier 2: controls (the planner's work; not recoverable from geometry).
  /** Feedforward steer angle (rad) = atan(L·kappa). Present only when a
   *  wheelbase was supplied to the builder. Approximated today; reserved
   *  for the true primitive steer command. */
  steerFf?: number;
  /** Feedforward longitudinal accel command (m/s²). Mirrors `aRef` today;
   *  reserved for the true primitive accel command. */
  accelFf?: number;

  // --- Tier 3: dynamic state & free space (model-only; reserved for MPC/LQR).
  /** Sideslip angle (rad). Present only when the source state carried
   *  `lateralVelocity`. */
  betaRef?: number;
  /** Yaw rate (rad/s). Present only when the source state carried `yawRate`. */
  rRef?: number;
  /** Free-space corridor half-widths (m). RESERVED — never populated yet. */
  dLeft?: number;
  dRight?: number;
}

/** A single-gear span of the plan, with inclusive `[startIdx, endIdx]`
 *  indices into `Plan.points`. Encodes the one thing a single point can't:
 *  the cusp topology — where the chassis comes to rest (v→0) and flips gear.
 *  Adjacent segments meet at the shared cusp index. */
export interface Segment {
  startIdx: number;
  endIdx: number;
  direction: Direction;
}

export interface Plan {
  points: ReferencePoint[];
  segments: Segment[];
}
