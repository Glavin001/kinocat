// Model Predictive Path Integral (MPPI) tracker for Ackermann vehicles.
//
// The single principled controller for the entire driving spectrum —
// from racing cruise to tight-space parking and every intermediate
// motion. Replaces the random-shooting MPC variant that preceded it
// (which was the "toy MPC" of pick-the-best-sample), and is the
// established production default in ROS Navigation 2 for ground
// vehicles. Reference: Williams et al., "Aggressive driving with
// model predictive path integral control" (ICRA 2016).
//
// The single-controller-for-everything property comes from how MPPI
// integrates samples: instead of selecting the best, every candidate
// is weighted by `exp(-(cost - min_cost) / λ)` (Boltzmann / Gibbs
// distribution) and the emitted control is the importance-weighted
// average. λ (the temperature) governs the regime:
//
//   λ → 0    Boltzmann collapses to pick-the-best → aggressive but
//            noisy, matches random-shooting MPC.
//   λ → ∞    Uniform average over samples → very smooth, very
//            cautious.
//   λ ~ cost-scale  Smooth weighted blend — the bias-variance sweet
//                   spot that handles racing AND parking with the
//                   same hyperparameter.
//
// The cost function does its own scenario detection: if the plan's
// terminal speed is near zero AND the goal is reachable within the
// MPC horizon, the terminal-pose weights kick in (precision mode).
// Otherwise the controller cruises with cost-weighted plan tracking
// (racing / general driving mode). No mode switching, no developer-
// visible knob — the cost shape derives from the plan content.

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import type { PlanPath } from './types';
import { lerpAngle } from '../internal/math';

/** Native wheeled-vehicle control set the tracker emits each tick. */
export interface MPCCommand {
  steer: number;       // rad (front-wheel angle)
  driveForce: number;  // N (signed; negative = reverse)
  brakeForce: number;  // N (>= 0)
  /** Reference speed at MPC step 1 — for telemetry / logging. */
  targetSpeed: number;
  /** Reference point at MPC step 1 — for visualisation. */
  lookahead: { x: number; z: number };
  /** Within the goal tolerance (terminal cost dominates). */
  atGoal: boolean;
  /** Min cost across sampled rollouts — useful for ablation. */
  bestCost: number;
}

export interface MPCTrackerConfig {
  /** Number of MPC steps in the rolling horizon. Default 10
   *  (~0.5 s at the default 0.05 s step). */
  horizonSteps?: number;
  /** Length of each MPC step (s). Default 0.05 (matches 60 Hz physics
   *  for cost-integration accuracy). */
  stepDt?: number;
  /** Number of control sequences sampled per tick. Default 64.
   *  Compute cost: K × H × (forward-sim cost) per tick. */
  samples?: number;
  /** Sampling stddev around the warm-started prior, in actuator units. */
  steerStd?: number;
  driveStd?: number;
  brakeStd?: number;
  /** Actuator limits — clamp every sample within these. */
  maxSteer: number;
  maxDriveForce: number;
  maxBrakeForce: number;
  /** Allow sampling negative `driveForce` (reverse). Default true. */
  allowReverse?: boolean;
  /** Chassis reverse-speed envelope (m/s, positive). Caps the allowed speed
   *  on reverse plan segments in progress mode. Default 6. */
  maxReverseSpeed?: number;
  /**
   * MPPI temperature. Controls the importance-weighted average's
   * concentration: smaller λ → controller commits to the best-looking
   * sample (aggressive, noisy); larger λ → controller averages many
   * samples (smooth, cautious). Default 1.0 — works for both racing
   * cruise and parking with the same value because cost scaling is
   * normalised by `min(cost)`.
   */
  lambda?: number;
  /** Per-stage cost weights. */
  wLateral?: number;
  wHeading?: number;
  wSpeed?: number;
  wControlRate?: number;
  /** Separate steer-rate weight (much larger than wControlRate at high
   *  speed — see notes in the random-shooting precursor). Default 3. */
  wSteerRate?: number;
  /**
   * Terminal-pose costs. These are auto-activated when the plan's
   * terminal speed is near zero AND the goal is reachable within the
   * MPC horizon — i.e. when the plan is asking the chassis to STOP at
   * a pose (parking, multi-step back-and-forth target). For pure
   * cruise plans (terminal speed ≈ cruise speed) these don't fire and
   * the controller behaves like a standard tracking MPC. The config
   * values cap how strong the terminal cost can get; default to
   * sensible levels for parking.
   */
  wTerminalPosition?: number;
  wTerminalSpeed?: number;
  /** Distance below which `atGoal` is reported true. Default 0.5 m. */
  goalTolerance?: number;
  /**
   * Cruise speed (m/s) the reference advances at where the plan itself
   * doesn't dictate a speed (samples with |speed| ≤ 0.5, e.g. the rest state
   * at the start of a drive-through plan). Without this the tracker derives a
   * cruise from the plan's TERMINAL speed — which the planner leaves ≈ 0 even
   * for race gates — so the reference barely extends ahead and the chassis
   * crawls/stalls instead of accelerating to racing speed. Pass the scenario's
   * cruise (race: agent max speed; parking: the slow parking cap). Defaults to
   * `max(|goal.speed|, 1)` for backward compatibility. */
  cruiseSpeed?: number;
  /**
   * Cost mode. `'track'` (default) — classic reference tracking: build a
   * per-step reference along the plan and penalise deviation from it. The
   * right shape for parking / terminal-pose work, but it DNF'd racing: the
   * reference advances at a speed the samples must CHASE, so the cost
   * gradient points at a moving target instead of "go as fast as physics
   * allows along the plan".
   *
   * `'progress'` — racing (Williams 2016 shape): reward arc-length progress
   * along the plan polyline, penalise leaving a lateral corridor around it,
   * and penalise exceeding a braking-envelope allowed speed derived from the
   * plan's own geometry (and, when `usePlanSpeeds`, its profiled speeds).
   * No reference to chase — the model decides how fast the chassis can go,
   * which is exactly where model fidelity becomes lap time.
   */
  costMode?: 'track' | 'progress';
  /** Progress reward per metre of arc advanced (progress mode). Default 6. */
  wProgress?: number;
  /** Quadratic penalty per step for lateral offset BEYOND the corridor
   *  half-width (progress mode). Default 20. */
  wCorridor?: number;
  /** Lateral corridor half-width (m) around the plan within which rollouts
   *  travel penalty-free (progress mode). Default 2.5. */
  corridorHalfWidth?: number;
  /** Small quadratic centering pull INSIDE the corridor so straights
   *  re-centre onto the plan instead of hugging the corridor edge
   *  (progress mode). Default 0.08. */
  wCenterline?: number;
  /** Quadratic penalty per step for exceeding the braking-envelope allowed
   *  speed at the rollout state's plan projection (progress mode).
   *  Default 4. */
  wOverspeed?: number;
  /** Quadratic penalty per step for chassis-heading error against the plan
   *  tangent at the projection (progress mode). Progress + corridor alone
   *  leave heading unconstrained until position error accrues — by then a
   *  gate overshoot has the chassis 60°+ off the (replanned) plan and every
   *  forward sample looks bad (measured: stall-and-recover at gates).
   *  Default 1.5. */
  wHeadingAlign?: number;
  /** Longitudinal decel (m/s², positive) assumed by the allowed-speed
   *  backward pass — how hard the chassis may be assumed to brake for an
   *  upcoming corner. Default 8. */
  envelopeDecel?: number;
  /** Lateral-accel cap (m/s²) for the curvature-derived allowed speed
   *  (v ≤ √(aLat/κ) at every genuine corner, κ > 0.08). Default 12. */
  envelopeLateralAccel?: number;
  /** Consume the plan's per-sample speeds as additional allowed-speed caps.
   *  Enable when the plan went through a friction-circle speed-profile pass
   *  (technical course); leave off when plan speeds are raw primitive
   *  endpoint speeds (open course) which would pin the launch. Default
   *  false. */
  usePlanSpeeds?: boolean;
  /** Model substeps per MPC step: each `stepDt` rollout step calls the
   *  forward model `substeps` times at stepDt/substeps under a zero-order
   *  hold. Closes the 0.05 s step vs 1/60 s plant mismatch for single-step
   *  Euler models. Default 1. */
  substeps?: number;
  /** Metres to extrapolate the plan past its final sample along the last
   *  tangent — applied to drive-through plans only (terminal speed > 0.5
   *  or `noStopAtEnd`). Racing horizons end at the replanning window, not a
   *  wall: without the extension the plan end acts as an attractor/stop and
   *  the chassis lifts early every time the horizon outruns the plan.
   *  Default 0 (off). */
  referenceExtension?: number;
  /** Treat the plan terminal as a drive-through even if its stored speed is
   *  ≈ 0 (race planners leave gate poses at speed 0). Gates the reference
   *  extension and disables terminal-cost activation. Default false. */
  noStopAtEnd?: boolean;
  /**
   * Progress-mode longitudinal sampling std, in units of the single
   * accel channel a ∈ [−1, 1] (a ≥ 0 → driveForce = a·maxDriveForce;
   * a < 0 → brakeForce = −a·maxBrakeForce). Progress mode samples ONE
   * longitudinal channel instead of independent drive/brake noise:
   * independent channels fight each other (the near-binary raycast brake
   * out-muscles mean drive noise and no sample ever launches). Default 0.5.
   */
  accelStd?: number;
  /**
   * AR(1) correlation of the sampling noise across horizon steps
   * (progress mode). White per-step noise cannot represent "hold full
   * throttle down the straight" — its mean is washed out over the horizon;
   * correlated (coloured) noise makes each sample a coherent maneuver.
   * 0 = white. Default 0.8.
   */
  noiseCorrelation?: number;
  /** Diagnostics hook — called once per solve with the full sample-level
   *  picture (costs, controls, the emitted first command, anchor). Zero
   *  cost when unset. For offline debugging only; not part of the control
   *  contract. */
  onDebug?: (info: MPCDebugInfo) => void;
}

/** Sample-level solve diagnostics passed to `MPCTrackerConfig.onDebug`. */
export interface MPCDebugInfo {
  costs: Float64Array;
  /** Flattened [steer, drive, brake] × H × K sample controls. */
  samples: Float64Array;
  horizon: number;
  emitted: { steer: number; driveForce: number; brakeForce: number };
  minCost: number;
  gear: number;
  /** Progress-mode anchor (arc metres + sample index), if applicable. */
  anchor: { s: number; idx: number } | null;
  /** Softmax weight mass on the best sample (concentration signal). */
  bestWeightShare: number;
  /** Score an arbitrary flattened control sequence ([steer, drive, brake]
   *  × horizon) under THIS solve's exact cost (same geometry, anchor,
   *  weights, substeps, model). Lets an offline probe compare hand-built
   *  maneuvers against the sampled population. */
  scoreSequence: (controls: Float64Array) => number;
}

export interface MPCTrackerState {
  /** Best control sequence from the previous call, kept for warm-start.
   *  Flattened `[s0, d0, b0, s1, d1, b1, ...]`. Length = 3 × horizon. */
  prev: Float64Array;
  /** RNG state (linear congruential) so the tracker is deterministic
   *  given a seed — required for ablation reproducibility. */
  rngState: number;
  /** Progress-mode anchor continuity: the plan array identity + anchor
   *  sample index from the previous call. A plan whose later legs pass
   *  near the chassis (any hairpin / return leg) would otherwise let the
   *  nearest-distance anchor TELEPORT forward across the loop — free arc
   *  progress for standing still, which the progress cost then defends by
   *  parking the car (measured: stall-and-recover cycles at every gate).
   *  Same-plan calls only search a bounded arc window around the previous
   *  anchor; a new plan array (fresh replan, starts at the chassis) resets
   *  the anchor by full search. */
  lastPlan: PlanPath | null;
  lastAnchorIdx: number;
}

/** Create a fresh MPPI tracker state. Deterministic on `seed`. */
export function createMPCTrackerState(horizonSteps: number, seed = 0x1337): MPCTrackerState {
  return {
    prev: new Float64Array(horizonSteps * 3),
    rngState: seed >>> 0 || 1,
    lastPlan: null,
    lastAnchorIdx: 0,
  };
}

// 32-bit LCG (constants from Numerical Recipes). Returns [0, 1).
function lcg(state: MPCTrackerState): number {
  state.rngState = (Math.imul(state.rngState, 1664525) + 1013904223) >>> 0;
  return state.rngState / 0x100000000;
}

// Box–Muller — one standard normal sample per call.
function gauss(state: MPCTrackerState): number {
  const u1 = Math.max(lcg(state), 1e-9);
  const u2 = lcg(state);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

// ---------------------------------------------------------------------------
// Reference extension (drive-through plans)

/**
 * Extrapolate a plan past its final sample along the last segment's tangent.
 * Racing plans end at the replanning window (2 gates ahead), not at a wall —
 * without the extension the plan end acts as a stop target: reference
 * bunches there in track mode, and in progress mode arc-length progress
 * saturates, so every rollout that would sail past the end scores the same
 * as one that parks on it and the chassis lifts early on every horizon.
 *
 * Extension samples carry the last sample's heading and speed (or
 * `fallbackSpeed` when the terminal speed is ≈ 0, e.g. planner gate poses),
 * spaced ~1 m apart. Returns the original array when there is nothing to do.
 */
export function extendPlanForTracking(
  plan: PlanPath,
  metres: number,
  fallbackSpeed = 5,
): PlanPath {
  if (metres <= 0 || plan.length < 2) return plan as PlanPath;
  const last = plan[plan.length - 1]!;
  // Tangent from the last segment with usable length (smoothed plans can
  // end in near-duplicate samples).
  let ax = 0;
  let az = 0;
  for (let i = plan.length - 2; i >= 0; i--) {
    const dx = last.x - plan[i]!.x;
    const dz = last.z - plan[i]!.z;
    const len = Math.hypot(dx, dz);
    if (len > 1e-6) {
      ax = dx / len;
      az = dz / len;
      break;
    }
  }
  if (ax === 0 && az === 0) {
    ax = Math.cos(last.heading);
    az = Math.sin(last.heading);
  }
  const speed = Math.abs(last.speed) > 0.5 ? last.speed : fallbackSpeed;
  const n = Math.max(2, Math.ceil(metres));
  const out = plan.slice();
  for (let i = 1; i <= n; i++) {
    const d = (metres * i) / n;
    out.push({
      x: last.x + ax * d,
      z: last.z + az * d,
      heading: Math.atan2(az, ax),
      speed,
      t: last.t + d / Math.max(Math.abs(speed), 0.5),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Progress-mode geometry (racing cost — Williams 2016 shape)

/** Precomputed plan geometry for the progress cost: polyline cumulative
 *  arc length + a braking-envelope allowed-speed profile at every sample. */
export interface ProgressGeometry {
  pts: PlanPath;
  /** Cumulative arc length at each sample (m). */
  cum: number[];
  /** Allowed speed at each sample after the curvature cap + (optional)
   *  plan-speed cap + backward braking-envelope pass (m/s). */
  vAllow: number[];
}

/** Curvature threshold below which a sample is not a "genuine corner"
 *  (matches pure-pursuit's preview threshold — replanned chord paths carry
 *  curvature noise on straights that must not cap speed). R ≥ 12.5 m. */
const PROGRESS_CURVATURE_MIN = 0.08;

/** Arc distance (m) between the three points of the Menger curvature
 *  estimate. Adjacent-sample triples on a dense (~0.4 m) smoothed plan are
 *  hypersensitive to lateral jitter — measured: phantom κ ≈ 5 (R ≈ 0.2 m!)
 *  spikes on ordinary plans, whose √(aLat/κ) ≈ 1.5 m/s "allowed speed"
 *  wedged the whole field into stop-and-go. A 1.2 m baseline averages the
 *  jitter out while still resolving every corner a 3.4 m-radius primitive
 *  library can produce. */
const CURVATURE_ARC_BASELINE = 1.2;

/**
 * Build the progress-cost geometry for a (possibly extended) plan.
 *
 * The allowed-speed profile is the SAME anticipatory-braking logic
 * pure-pursuit's `previewCurvature` uses, precomputed per sample:
 *   1. curvature cap  v ≤ √(aLat/κ) at every genuine corner,
 *   2. optional plan-speed cap (post-speed-profile plans),
 *   3. backward pass v[i] ≤ √(v[i+1]² + 2·decel·ds) so a slow corner
 *      constrains the approach exactly as far ahead as braking physics
 *      requires.
 * Purely geometric and identical for every entry — model fidelity enters
 * through the ROLLOUTS (a truthful model predicts the real arc; a
 * delusional one holds any line), never through per-car tuning.
 */
export function buildProgressGeometry(
  plan: PlanPath,
  opts: {
    envelopeDecel: number;
    envelopeLateralAccel: number;
    usePlanSpeeds: boolean;
    /** Ignore the terminal sample's stored speed (drive-through horizon). */
    ignoreTerminalSpeed: boolean;
    /** Lateral slack (m) the rollouts may use to cut corners inside the
     *  corridor. The allowed-speed cap uses the EFFECTIVE corner radius
     *  R + 2·slack, not the raw polyline radius: a rollout is free to round
     *  a plan kink with the corridor's width, so capping speed at the raw
     *  kink radius (R ≈ 1–1.5 m at gate corners) throttled the whole field
     *  to ~40% of achievable pace (measured: the cars rode vAllow
     *  everywhere at part throttle). Default 0 (raw geometry). */
    corridorSlack?: number;
  },
): ProgressGeometry {
  const n = plan.length;
  const cum = new Array<number>(n).fill(0);
  for (let i = 1; i < n; i++) {
    cum[i] = cum[i - 1]! + Math.hypot(
      plan[i]!.x - plan[i - 1]!.x,
      plan[i]!.z - plan[i - 1]!.z,
    );
  }
  const vAllow = new Array<number>(n).fill(Infinity);
  // Fixed-arc-baseline curvature: for sample i, find the samples ~one
  // CURVATURE_ARC_BASELINE behind and ahead along the polyline and run
  // Menger through that triple. Adjacent-triple curvature on dense plans
  // amplifies sub-centimetre lateral jitter into phantom hairpins.
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < n; i++) {
    let cap = Infinity;
    while (lo < i && cum[i]! - cum[lo + 1]! >= CURVATURE_ARC_BASELINE) lo++;
    if (hi < i) hi = i;
    while (hi < n - 1 && cum[hi]! - cum[i]! < CURVATURE_ARC_BASELINE) hi++;
    if (lo < i && hi > i) {
      const ax = plan[lo]!.x, az = plan[lo]!.z;
      const bx = plan[i]!.x, bz = plan[i]!.z;
      const cx = plan[hi]!.x, cz = plan[hi]!.z;
      const ab = Math.hypot(bx - ax, bz - az);
      const bc = Math.hypot(cx - bx, cz - bz);
      const ca = Math.hypot(cx - ax, cz - az);
      const denom = ab * bc * ca;
      if (denom > 1e-9) {
        const kappa = (2 * Math.abs((bx - ax) * (cz - az) - (bz - az) * (cx - ax))) / denom;
        // Effective corner radius: raw polyline radius + corridor slack
        // (the rollout may legally round the kink inside the corridor).
        const slack = opts.corridorSlack ?? 0;
        const kappaEff = kappa > 1e-9 ? 1 / (1 / kappa + slack) : 0;
        if (kappaEff >= PROGRESS_CURVATURE_MIN) {
          cap = Math.sqrt(opts.envelopeLateralAccel / kappaEff);
        }
      }
    }
    if (opts.usePlanSpeeds && !(opts.ignoreTerminalSpeed && i === n - 1)) {
      const sp = Math.abs(plan[i]!.speed);
      // A near-zero stored speed on an interior sample is the plan's echo
      // of the current (rest) state or a smoothing artifact, not a stop
      // command — never let it pin the profile to zero mid-plan.
      if (sp > 0.5) cap = Math.min(cap, sp);
    }
    vAllow[i] = cap;
  }
  // Terminal stop (parking-style plans in progress mode): honour it.
  if (!opts.ignoreTerminalSpeed && n > 0 && Math.abs(plan[n - 1]!.speed) < 0.5) {
    vAllow[n - 1] = 0;
  }
  // Backward braking-envelope pass.
  for (let i = n - 2; i >= 0; i--) {
    const ds = cum[i + 1]! - cum[i]!;
    const vBrake = Math.sqrt(vAllow[i + 1]! * vAllow[i + 1]! + 2 * opts.envelopeDecel * ds);
    if (vBrake < vAllow[i]!) vAllow[i] = vBrake;
  }
  return { pts: plan, cum, vAllow };
}

/** Project a point onto the plan polyline, searching only segments within
 *  `arcBehind` metres behind and `arcAhead` metres ahead of the cursor
 *  sample `fromIdx` (arc length along the plan). Returns arc length `s`,
 *  squared lateral distance `d2`, and the winning segment index (the
 *  monotonic cursor for the next projection).
 *
 *  The ARC bound is the load-bearing part: an index- or unbounded search
 *  lets a state near a LATER plan leg (hairpin return legs are metres away
 *  in space, tens of metres away in arc) teleport the projection across the
 *  loop — free progress for cutting the course, or for standing still where
 *  legs cross. Physically a chassis can only advance along the plan about
 *  as fast as it moves, so the projection window must honour that. */
function projectOnto(
  geom: ProgressGeometry,
  x: number,
  z: number,
  fromIdx: number,
  arcAhead: number,
  arcBehind = 1.0,
): { s: number; d2: number; idx: number } {
  const pts = geom.pts;
  const cum = geom.cum;
  const sFrom = cum[Math.min(fromIdx, pts.length - 1)]!;
  let first = fromIdx;
  while (first > 0 && sFrom - cum[first - 1]! <= arcBehind) first--;
  let last = fromIdx;
  while (last < pts.length - 2 && cum[last + 1]! - sFrom <= arcAhead) last++;
  let bestD2 = Infinity;
  let bestS = sFrom;
  let bestIdx = fromIdx;
  for (let i = first; i <= last; i++) {
    const ax = pts[i]!.x;
    const az = pts[i]!.z;
    const bx = pts[i + 1]!.x;
    const bz = pts[i + 1]!.z;
    const dx = bx - ax;
    const dz = bz - az;
    const lenSq = dx * dx + dz * dz;
    let u = 0;
    if (lenSq > 1e-12) {
      u = ((x - ax) * dx + (z - az) * dz) / lenSq;
      if (u < 0) u = 0;
      else if (u > 1) u = 1;
    }
    const px = ax + dx * u;
    const pz = az + dz * u;
    const ddx = x - px;
    const ddz = z - pz;
    const d2 = ddx * ddx + ddz * ddz;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestS = geom.cum[i]! + Math.sqrt(lenSq) * u;
      bestIdx = i;
    }
  }
  return { s: bestS, d2: bestD2, idx: bestIdx };
}

/** Interpolated point on the plan polyline at arc position `s`, searching
 *  forward from sample `fromIdx`. */
function pointAtArc(
  geom: ProgressGeometry,
  s: number,
  fromIdx: number,
): { x: number; z: number } {
  const cum = geom.cum;
  const pts = geom.pts;
  let i = Math.max(0, Math.min(fromIdx, pts.length - 2));
  while (i < pts.length - 2 && cum[i + 1]! < s) i++;
  const a = pts[i]!;
  const b = pts[Math.min(i + 1, pts.length - 1)]!;
  const seg = cum[Math.min(i + 1, pts.length - 1)]! - cum[i]!;
  const u = seg > 1e-9 ? clamp((s - cum[i]!) / seg, 0, 1) : 0;
  return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u };
}

/** Allowed speed at arc position `s` inside segment `idx` — the next
 *  sample's allowed speed relaxed backwards through the braking envelope
 *  (continuous interpolation of the backward pass). */
function allowedSpeedAt(
  geom: ProgressGeometry,
  s: number,
  idx: number,
  envelopeDecel: number,
): number {
  const j = Math.min(idx + 1, geom.vAllow.length - 1);
  const vNext = geom.vAllow[j]!;
  if (!Number.isFinite(vNext)) return Infinity;
  const dsAhead = Math.max(0, geom.cum[j]! - s);
  return Math.sqrt(vNext * vNext + 2 * envelopeDecel * dsAhead);
}

export interface ProgressWeights {
  wProgress: number;
  wCorridor: number;
  corridorHalfWidth: number;
  wCenterline: number;
  wOverspeed: number;
  envelopeDecel: number;
  wControlRate: number;
  wSteerRate: number;
  wHeadingAlign: number;
  /** Hard speed ceiling (m/s) applied on top of the geometric envelope —
   *  the chassis gear envelope. Forward legs pass the scenario cruise cap;
   *  reverse legs pass the chassis's reverse limit (a reverse recovery leg
   *  has straight-line geometry, so without this the progress reward backs
   *  the car up at highway speed — measured −12.6 m/s). */
  speedCap?: number;
}

/**
 * Progress-mode rollout cost (racing). Lower is better; can be negative
 * (progress is a REWARD). Three physical terms plus the shared smoothness
 * terms:
 *   - progress:  −wProgress · (arc metres advanced along the plan)
 *   - corridor:  +wCorridor · max(0, |lateral| − halfWidth)² per step,
 *                plus a small always-on centering pull (wCenterline · lat²)
 *   - overspeed: +wOverspeed · max(0, |v| − vAllow(s))² per step, where
 *                vAllow is the braking-envelope profile from the plan's own
 *                geometry — anticipatory corner braking for corners BEYOND
 *                the MPC horizon.
 * Exported for direct unit testing of the cost ordering (A3.1).
 */
export function scoreRolloutProgress(
  rollout: CarKinematicState[],
  geom: ProgressGeometry,
  startProj: { s: number; idx: number },
  controls: Float64Array,
  prevSeed: Float64Array,
  w: ProgressWeights,
  /** Segment gear: +1 forward, −1 reverse. On a reverse leg the plan's
   *  stored heading is the TRAVEL tangent (smoothed plans recompute heading
   *  from the polyline), which points opposite the chassis nose — the
   *  alignment target flips by π. */
  gear = 1,
): number {
  let cost = 0;
  let cursor = startProj.idx;
  let sPrev = startProj.s;
  const H = rollout.length;
  // Arc window per MPC step: the chassis advances ≤ vMax·stepDt ≈ 1.5 m
  // per 0.05 s step — 4 m gives 2.5× margin without opening the cross-leg
  // teleport hole (see projectOnto).
  const ARC_AHEAD_PER_STEP = 4.0;
  for (let i = 0; i < H; i++) {
    const st = rollout[i]!;
    const proj = projectOnto(geom, st.x, st.z, cursor, ARC_AHEAD_PER_STEP, 1.5);
    cursor = proj.idx;
    cost -= w.wProgress * (proj.s - sPrev);
    sPrev = proj.s;
    const lat2 = proj.d2;
    cost += w.wCenterline * lat2;
    const lat = Math.sqrt(lat2);
    const excess = lat - w.corridorHalfWidth;
    if (excess > 0) cost += w.wCorridor * excess * excess;
    if (w.wHeadingAlign > 0 && i === H - 1) {
      // TERMINAL heading alignment: the rollout must END aligned with the
      // plan tangent. Charged once, not per step — a per-step version taxed
      // the honest models' true yaw-rate lag (30 steps of transient error)
      // harder than the progress reward paid, so their optimum was to PARK
      // at any large heading error, while the kinematic delusion (instant
      // yaw response, near-zero transient) sailed through the same cost.
      // Terminal-only keeps the anti-wrong-way purpose without pricing
      // honesty (measured: v3 wedge → clean under this shape).
      const target =
        gear < 0 ? geom.pts[proj.idx]!.heading + Math.PI : geom.pts[proj.idx]!.heading;
      const dh = wrapPi(st.heading - target);
      cost += w.wHeadingAlign * dh * dh;
    }
    let vAllowHere = allowedSpeedAt(geom, proj.s, proj.idx, w.envelopeDecel);
    if (w.speedCap !== undefined && w.speedCap < vAllowHere) vAllowHere = w.speedCap;
    if (Number.isFinite(vAllowHere)) {
      // Half-m/s deadband: tracking AT the allowed speed must be free, or
      // the geometric envelope (identical for every model) out-prices the
      // progress reward and parks the honest models — the model that
      // truthfully predicts "full throttle = 18 m/s in 1.5 s" gets charged
      // for its honesty while the delusional one sails (measured).
      const over = Math.abs(st.speed) - vAllowHere - 0.5;
      if (over > 0) cost += w.wOverspeed * over * over;
    }
    if (i > 0) {
      const ds = controls[i * 3]! - controls[(i - 1) * 3]!;
      const dd = controls[i * 3 + 1]! - controls[(i - 1) * 3 + 1]!;
      const db = controls[i * 3 + 2]! - controls[(i - 1) * 3 + 2]!;
      cost += w.wSteerRate * ds * ds;
      cost += w.wControlRate * (dd * dd * 1e-6 + db * db * 1e-6);
    } else {
      const ds0 = controls[0]! - prevSeed[0]!;
      cost += w.wSteerRate * 0.5 * ds0 * ds0;
    }
  }
  return cost;
}

/** Per-step reference state from the plan polyline by walking
 *  arc-length forward from the projection of `current`. */
function buildReference(
  current: CarKinematicState,
  plan: PlanPath,
  horizon: number,
  cruiseSpeed: number,
  stepDt: number,
): { ref: CarKinematicState[]; bestI: number } {
  if (plan.length === 0) {
    const ref: CarKinematicState[] = [];
    for (let i = 0; i < horizon; i++) ref.push({ ...current });
    return { ref, bestI: 0 };
  }
  let bestI = 0;
  let bestD = Infinity;
  for (let i = 0; i < plan.length; i++) {
    const dx = plan[i]!.x - current.x;
    const dz = plan[i]!.z - current.z;
    const d = dx * dx + dz * dz;
    if (d < bestD) { bestD = d; bestI = i; }
  }
  const cum: number[] = [0];
  for (let i = bestI + 1; i < plan.length; i++) {
    const a = plan[i - 1]!;
    const b = plan[i]!;
    cum.push(cum[cum.length - 1]! + Math.hypot(b.x - a.x, b.z - a.z));
  }
  // Speed the reference advances along the plan. Use the plan's local speed
  // where it dictates one; otherwise (rest state, or a plan that left speed
  // unset) fall back to the scenario cruise so the reference reaches a full
  // horizon ahead rather than bunching up at the chassis.
  const localSpeed =
    Math.abs(plan[bestI]!.speed) > 0.5 ? Math.abs(plan[bestI]!.speed) : cruiseSpeed;
  const ref: CarKinematicState[] = [];
  for (let k = 1; k <= horizon; k++) {
    const targetArc = k * stepDt * localSpeed;
    let j = 0;
    while (j < cum.length - 1 && cum[j + 1]! < targetArc) j++;
    const segLen = (j < cum.length - 1) ? (cum[j + 1]! - cum[j]!) : 0;
    const u = segLen > 1e-9 ? (targetArc - cum[j]!) / segLen : 0;
    const a = plan[bestI + j]!;
    const b = plan[Math.min(bestI + j + 1, plan.length - 1)]!;
    ref.push({
      x: a.x + (b.x - a.x) * u,
      z: a.z + (b.z - a.z) * u,
      // Wrap-aware: a plan crossing the +-pi seam must not interpolate "the
      // long way" (a reference heading near 0 would poison the wHeading cost).
      heading: lerpAngle(a.heading, b.heading, u),
      // Reference speed = the advance speed, NOT the plan's stored per-sample
      // speed. The two must agree or the speed-tracking cost fights the
      // position-tracking cost: a plan that starts at rest (speed ≈ 0) but
      // whose positions are advanced at cruise would otherwise ask the chassis
      // to be both 15 m ahead AND stopped, so MPPI blends full throttle with
      // full brake and the car never leaves the line (the race DNF).
      speed: localSpeed,
      t: 0,
    });
  }
  return { ref, bestI };
}

function scoreRollout(
  rollout: CarKinematicState[],
  reference: CarKinematicState[],
  controls: Float64Array,
  prevSeed: Float64Array,
  goal: CarKinematicState | undefined,
  terminalActive: boolean,
  w: Required<Pick<MPCTrackerConfig,
    'wLateral' | 'wHeading' | 'wSpeed' | 'wControlRate' | 'wSteerRate' | 'wTerminalPosition' | 'wTerminalSpeed'
  >>,
): number {
  let cost = 0;
  const H = rollout.length;
  for (let i = 0; i < H; i++) {
    const r = reference[i]!;
    const s = rollout[i]!;
    const dx = s.x - r.x;
    const dz = s.z - r.z;
    cost += w.wLateral * (dx * dx + dz * dz);
    const dh = wrapPi(s.heading - r.heading);
    cost += w.wHeading * dh * dh;
    const dv = Math.abs(s.speed) - Math.abs(r.speed);
    cost += w.wSpeed * dv * dv;
    if (i > 0) {
      const ds = controls[i * 3]! - controls[(i - 1) * 3]!;
      const dd = controls[i * 3 + 1]! - controls[(i - 1) * 3 + 1]!;
      const db = controls[i * 3 + 2]! - controls[(i - 1) * 3 + 2]!;
      cost += w.wSteerRate * ds * ds;
      cost += w.wControlRate * (dd * dd * 1e-6 + db * db * 1e-6);
    } else {
      // Inter-tick steer rate (vs the previous tick's first command).
      const ds0 = controls[0]! - prevSeed[0]!;
      cost += w.wSteerRate * 0.5 * ds0 * ds0;
    }
  }
  // Terminal-pose cost. Auto-activated by `terminalActive` (set by
  // the caller when the plan asks the chassis to stop near a pose
  // and the goal is reachable within the horizon).
  if (terminalActive && goal !== undefined) {
    const last = rollout[rollout.length - 1]!;
    const dx = last.x - goal.x;
    const dz = last.z - goal.z;
    cost += w.wTerminalPosition * (dx * dx + dz * dz);
    cost += w.wTerminalSpeed * last.speed * last.speed;
  }
  return cost;
}

/**
 * One MPPI tracker step. Pure modulo the deterministic RNG state
 * mutation inside `state`. Returns the actuator command for THIS
 * tick; the importance-weighted optimal sequence is kept in `state`
 * for next-tick warm-start.
 */
export function mpcTrack(
  current: CarKinematicState,
  planRaw: PlanPath,
  forwardSim: ForwardSim<CarKinematicState>,
  state: MPCTrackerState,
  config: MPCTrackerConfig,
): MPCCommand {
  const H = config.horizonSteps ?? 10;
  const dt = config.stepDt ?? 0.05;
  const K = config.samples ?? 64;
  const substeps = Math.max(1, Math.round(config.substeps ?? 1));
  const subDt = dt / substeps;
  const costMode = config.costMode ?? 'track';
  // Gear of the tracked segment (progress mode). The scenario runner feeds
  // MPPI single-gear plan SEGMENTS (split at forward↔reverse cusps); the
  // race planner legally plans reverse legs (recovery escapes after a gate
  // overshoot — reverseCostMultiplier makes them rare but they exist).
  // A forward-only progress sampler cannot execute them: every sample makes
  // negative progress, "hold still" wins the softmax, and the car wedges
  // until the blind stuck-recovery fires (measured: 10+ recoveries/run for
  // the learned models). Detect the segment's gear from its stored speeds
  // and sample the longitudinal channel in that gear instead.
  let gear = 1;
  if (costMode === 'progress') {
    let sum = 0;
    for (const p of planRaw) sum += p.speed;
    if (sum < -1e-6) gear = -1;
  }
  // Reverse legs end at a cusp (a genuine stop-and-change-gear pose), never
  // at a drive-through gate — their terminal must stay a stop target.
  const noStopAtEnd = gear > 0 && (config.noStopAtEnd ?? false);
  // Reference extension: drive-through plans (terminal speed > 0.5, or the
  // caller declared the terminal a gate via noStopAtEnd) get the tail
  // extrapolated so the horizon end never acts as a stop target.
  const lastRaw = planRaw.length > 0 ? planRaw[planRaw.length - 1]! : undefined;
  const driveThrough =
    lastRaw !== undefined && (noStopAtEnd || Math.abs(lastRaw.speed) > 0.5);
  const ext = config.referenceExtension ?? 0;
  const plan =
    driveThrough && ext > 0
      ? extendPlanForTracking(planRaw, ext, config.cruiseSpeed ?? 5)
      : planRaw;
  const sStd = config.steerStd ?? 0.10;
  const dStd = config.driveStd ?? 0.4 * config.maxDriveForce;
  // The raycast-vehicle brake is near-binary — even a small brake force locks
  // the wheels and produces grip-limited deceleration (measured; see the
  // grip-saturating brake in vehicle-model.ts). Exploring large brake
  // perturbations therefore mostly generates catastrophic-stop rollouts that
  // swamp the importance-weighted average and stall acceleration from rest.
  // Perturb brake gently by default (MPPI modulates speed mainly through
  // drive force); callers that need decisive braking pass an explicit
  // `brakeStd`.
  const bStd = config.brakeStd ?? 0.03 * config.maxBrakeForce;
  const allowReverse = config.allowReverse ?? true;
  const lambda = Math.max(config.lambda ?? 1.0, 1e-6);
  const weights = {
    wLateral: config.wLateral ?? 5,
    wHeading: config.wHeading ?? 2,
    wSpeed: config.wSpeed ?? 3,
    wControlRate: config.wControlRate ?? 0.5,
    wSteerRate: config.wSteerRate ?? 8,
    wTerminalPosition: config.wTerminalPosition ?? 30,
    wTerminalSpeed: config.wTerminalSpeed ?? 20,
  };
  const tol = config.goalTolerance ?? 0.5;

  if (state.prev.length !== H * 3) state.prev = new Float64Array(H * 3);

  // Build the reference + decide whether to activate terminal cost.
  const goal = plan.length > 0 ? plan[plan.length - 1]! : undefined;
  const cruiseSpeed =
    config.cruiseSpeed ?? (goal ? Math.max(Math.abs(goal.speed), 1) : 5);
  const { ref } = buildReference(current, plan, H, cruiseSpeed, dt);

  // Terminal-cost activation. Requires:
  //   (a) caller opted in via non-zero `wTerminalPosition` or
  //       `wTerminalSpeed` weights (intent signal from the scenario:
  //       "this plan asks the chassis to come to rest at a pose"),
  //   (b) the plan's terminal speed is near zero (so the plan agrees
  //       this is a stop, not a drive-through gate), AND
  //   (c) the goal is reachable within the MPC horizon (so terminal
  //       cost meaningfully fires inside the rollout — telling MPPI
  //       to optimise for a goal 100 m away is just noise).
  // The race scenario sets `wTerminalPosition=0` so this never
  // triggers there even when individual gate poses happen to have
  // `speed=0` from the planner's pose() helper.
  let terminalActive = false;
  const wantsTerminal =
    !noStopAtEnd && (weights.wTerminalPosition > 0 || weights.wTerminalSpeed > 0);
  if (wantsTerminal && goal && Math.abs(goal.speed) < 0.5) {
    const distToGoal = Math.hypot(current.x - goal.x, current.z - goal.z);
    const maxReachInHorizon = Math.max(Math.abs(current.speed), 1) * H * dt + 2.0;
    if (distToGoal <= maxReachInHorizon) terminalActive = true;
  }

  // Warm-start prior — previous solution shifted by one step.
  const prior = new Float64Array(H * 3);
  for (let i = 0; i < H - 1; i++) {
    prior[i * 3]! = state.prev[(i + 1) * 3]!;
    prior[i * 3 + 1]! = state.prev[(i + 1) * 3 + 1]!;
    prior[i * 3 + 2]! = state.prev[(i + 1) * 3 + 2]!;
  }
  prior[(H - 1) * 3]! = state.prev[(H - 1) * 3]!;
  prior[(H - 1) * 3 + 1]! = state.prev[(H - 1) * 3 + 1]!;
  prior[(H - 1) * 3 + 2]! = state.prev[(H - 1) * 3 + 2]!;

  // Progress-mode geometry: build once per tick (shared by all K samples),
  // and project the CURRENT state to anchor each rollout's progress origin.
  let progressGeom: ProgressGeometry | null = null;
  let progressStart: { s: number; idx: number } | null = null;
  let progressWeights: ProgressWeights | null = null;
  if (costMode === 'progress' && plan.length >= 2) {
    progressGeom = buildProgressGeometry(plan, {
      envelopeDecel: config.envelopeDecel ?? 8,
      envelopeLateralAccel: config.envelopeLateralAccel ?? 12,
      usePlanSpeeds: config.usePlanSpeeds ?? false,
      ignoreTerminalSpeed: driveThrough,
      corridorSlack: config.corridorHalfWidth ?? 2.5,
    });
    // Anchor with CONTINUITY, not nearest-distance over the whole plan. A
    // plan whose later leg passes near the chassis (hairpins, return legs —
    // routine on a lap course) would let a full-plan nearest search teleport
    // the anchor forward across the loop: free arc progress for standing
    // still, which the cost then defends by parking the car. Same-plan
    // solves search a bounded window around the previous anchor (the
    // chassis moves ≤ a few samples between solves); a NEW plan starts at
    // the chassis, so its anchor lives in the first few metres by
    // construction.
    const sameplan = state.lastPlan === planRaw;
    const anchorFrom = sameplan ? state.lastAnchorIdx : 0;
    // Fresh plans start at the chassis → the true anchor is within the
    // first couple of metres BY CONSTRUCTION. A wider window (was 8 m) let
    // the anchor grab a RETURN leg passing nearer the chassis than the
    // plan's own start (slalom replans loop back within metres), and the
    // whole solve then optimised against the wrong leg — heading "error"
    // of ~π, every forward sample bad, hold-still wins (measured wedge).
    // Same-plan solves move ≤ vMax·0.05 s ≈ 1.5 m.
    const anchor = projectOnto(
      progressGeom,
      current.x,
      current.z,
      anchorFrom,
      sameplan ? 4.0 : 2.0,
      sameplan ? 2.0 : 0,
    );
    state.lastPlan = planRaw;
    state.lastAnchorIdx = anchor.idx;
    progressStart = { s: anchor.s, idx: anchor.idx };
    progressWeights = {
      wProgress: config.wProgress ?? 6,
      wCorridor: config.wCorridor ?? 20,
      corridorHalfWidth: config.corridorHalfWidth ?? 2.5,
      wCenterline: config.wCenterline ?? 0.08,
      wOverspeed: config.wOverspeed ?? 4,
      envelopeDecel: config.envelopeDecel ?? 8,
      wControlRate: weights.wControlRate,
      wSteerRate: weights.wSteerRate,
      wHeadingAlign: config.wHeadingAlign ?? 1.5,
      // Gear envelope: forward legs are capped by the scenario cruise,
      // reverse legs by the chassis reverse limit.
      speedCap: gear < 0 ? config.maxReverseSpeed ?? 6 : config.cruiseSpeed,
    };
  }

  // Allocate sample storage. We need every sample's controls AND its
  // cost — MPPI's emit step is the importance-weighted average of
  // them all (NOT pick-the-best).
  const samples = new Float64Array(K * H * 3);
  const costs = new Float64Array(K);
  let minCost = Infinity;
  const work = new Float64Array(H * 3);

  const isProgress = progressGeom !== null;
  const accelStd = config.accelStd ?? 0.5;
  const rho = clamp(config.noiseCorrelation ?? 0.8, 0, 0.999);
  const rhoComp = Math.sqrt(1 - rho * rho);

  for (let k = 0; k < K; k++) {
    // First sample (k=0) is the unperturbed prior — guarantees the
    // warm-start is in the candidate set.
    const noiseFactor = k === 0 ? 0 : 1;
    if (isProgress) {
      // Racing sampler. Two structural changes vs the legacy one:
      //  1. ONE longitudinal channel a ∈ [−1, 1] (a ≥ 0 → throttle,
      //     a < 0 → brake — a human pedal). Independent drive/brake noise
      //     fights itself: the near-binary raycast brake (locks at ~25 %
      //     of full force) out-muscles mean drive noise, so no sample ever
      //     launches, no sample discovers the progress reward, and the
      //     warm start can never bootstrap (measured: the field sat at the
      //     spawn for 180 s).
      //  2. AR(1)-correlated noise across the horizon (coloured, ρ ≈ 0.8).
      //     White per-step noise averages itself out over 30 steps — it
      //     cannot represent "hold full throttle down this straight" or
      //     "brake hard NOW for two steps". Correlated noise makes every
      //     sample a coherent maneuver candidate.
      let nSteer = 0;
      let nAccel = 0;
      for (let i = 0; i < H; i++) {
        nSteer = i === 0 ? gauss(state) : rho * nSteer + rhoComp * gauss(state);
        nAccel = i === 0 ? gauss(state) : rho * nAccel + rhoComp * gauss(state);
        // Pedal channel in the SEGMENT's gear: a ≥ 0 → drive (forward or
        // reverse per `gear`), a < 0 → brake.
        const priorA =
          (gear * prior[i * 3 + 1]!) / config.maxDriveForce -
          prior[i * 3 + 2]! / config.maxBrakeForce;
        const a = clamp(priorA + accelStd * nAccel * noiseFactor, -1, 1);
        const steer = clamp(
          prior[i * 3]! + sStd * nSteer * noiseFactor,
          -config.maxSteer,
          config.maxSteer,
        );
        work[i * 3]! = steer;
        work[i * 3 + 1]! = a >= 0 ? gear * a * config.maxDriveForce : 0;
        work[i * 3 + 2]! = a >= 0 ? 0 : -a * config.maxBrakeForce;
      }
    } else {
      for (let i = 0; i < H; i++) {
        let steer = prior[i * 3]! + sStd * gauss(state) * noiseFactor;
        let drive = prior[i * 3 + 1]! + dStd * gauss(state) * noiseFactor;
        let brake = prior[i * 3 + 2]! + bStd * gauss(state) * noiseFactor;
        steer = clamp(steer, -config.maxSteer, config.maxSteer);
        const minDrive = allowReverse ? -config.maxDriveForce : 0;
        drive = clamp(drive, minDrive, config.maxDriveForce);
        brake = clamp(brake, 0, config.maxBrakeForce);
        work[i * 3]! = steer;
        work[i * 3 + 1]! = drive;
        work[i * 3 + 2]! = brake;
      }
    }
    let s: CarKinematicState = { ...current };
    const traj: CarKinematicState[] = [];
    for (let i = 0; i < H; i++) {
      const u = [work[i * 3]!, work[i * 3 + 1]!, work[i * 3 + 2]!];
      // Zero-order hold over `substeps` model calls at dt/substeps: the
      // model integrates at (or near) its native training resolution
      // instead of one coarse Euler step per MPC step.
      for (let sub = 0; sub < substeps; sub++) s = forwardSim(s, u, subDt);
      traj.push(s);
    }
    const cost =
      progressGeom && progressStart && progressWeights
        ? scoreRolloutProgress(traj, progressGeom, progressStart, work, state.prev, progressWeights, gear)
        : scoreRollout(traj, ref, work, state.prev, goal, terminalActive, weights);
    costs[k] = cost;
    if (cost < minCost) minCost = cost;
    // Copy work → samples[k]
    for (let i = 0; i < H * 3; i++) samples[k * H * 3 + i] = work[i]!;
  }

  // MPPI importance-weighted average. Subtract minCost for numerical
  // stability before the softmax (a classic log-sum-exp trick).
  const weightsArr = new Float64Array(K);
  let weightSum = 0;
  for (let k = 0; k < K; k++) {
    const w = Math.exp(-(costs[k]! - minCost) / lambda);
    weightsArr[k] = w;
    weightSum += w;
  }
  // Compose the optimal sequence as the weighted average.
  const optimal = new Float64Array(H * 3);
  if (weightSum > 1e-9) {
    for (let k = 0; k < K; k++) {
      const w = weightsArr[k]! / weightSum;
      for (let i = 0; i < H * 3; i++) {
        optimal[i]! += w * samples[k * H * 3 + i]!;
      }
    }
  } else {
    // Degenerate case (all weights underflow to 0); fall back to
    // the lowest-cost sample.
    let bestK = 0;
    for (let k = 1; k < K; k++) if (costs[k]! < costs[bestK]!) bestK = k;
    for (let i = 0; i < H * 3; i++) optimal[i] = samples[bestK * H * 3 + i]!;
  }
  // Re-clamp the weighted-average controls (the average of valid
  // controls is always valid for box constraints, but float roundoff
  // can put us microscopically out).
  for (let i = 0; i < H; i++) {
    optimal[i * 3]! = clamp(optimal[i * 3]!, -config.maxSteer, config.maxSteer);
    const minDrive =
      allowReverse || (isProgress && gear < 0) ? -config.maxDriveForce : 0;
    const maxDrive = isProgress && gear < 0 ? 0 : config.maxDriveForce;
    optimal[i * 3 + 1]! = clamp(optimal[i * 3 + 1]!, minDrive, maxDrive);
    optimal[i * 3 + 2]! = clamp(optimal[i * 3 + 2]!, 0, config.maxBrakeForce);
  }
  // Progress mode: re-project the averaged longitudinal onto the single
  // pedal channel. Averaging drive and brake INDEPENDENTLY across samples
  // emits both at once (throttle-drag with the near-binary brake); the
  // sampler explored one a ∈ [−1, 1] channel, so the average must live
  // there too.
  if (isProgress) {
    for (let i = 0; i < H; i++) {
      const a =
        (gear * optimal[i * 3 + 1]!) / config.maxDriveForce -
        optimal[i * 3 + 2]! / config.maxBrakeForce;
      optimal[i * 3 + 1]! = a >= 0 ? gear * a * config.maxDriveForce : 0;
      optimal[i * 3 + 2]! = a >= 0 ? 0 : -a * config.maxBrakeForce;
    }
  }

  state.prev = optimal;

  if (config.onDebug) {
    let bestK = 0;
    for (let k = 1; k < K; k++) if (costs[k]! < costs[bestK]!) bestK = k;
    config.onDebug({
      costs,
      samples,
      horizon: H,
      emitted: { steer: optimal[0]!, driveForce: optimal[1]!, brakeForce: optimal[2]! },
      minCost,
      gear,
      anchor: progressStart,
      bestWeightShare: weightSum > 1e-9 ? weightsArr[bestK]! / weightSum : 1,
      scoreSequence: (controls: Float64Array): number => {
        let s: CarKinematicState = { ...current };
        const traj: CarKinematicState[] = [];
        for (let i = 0; i < H; i++) {
          const u = [controls[i * 3]!, controls[i * 3 + 1]!, controls[i * 3 + 2]!];
          for (let sub = 0; sub < substeps; sub++) s = forwardSim(s, u, subDt);
          traj.push(s);
        }
        return progressGeom && progressStart && progressWeights
          ? scoreRolloutProgress(traj, progressGeom, progressStart, controls, state.prev, progressWeights, gear)
          : scoreRollout(traj, ref, controls, state.prev, goal, terminalActive, weights);
      },
    });
  }

  const atGoal =
    goal !== undefined &&
    Math.hypot(current.x - goal.x, current.z - goal.z) <= tol;
  return {
    steer: optimal[0]!,
    driveForce: optimal[1]!,
    brakeForce: optimal[2]!,
    targetSpeed: ref[0]?.speed ?? 0,
    lookahead: { x: ref[0]?.x ?? current.x, z: ref[0]?.z ?? current.z },
    atGoal,
    bestCost: minCost,
  };
}
