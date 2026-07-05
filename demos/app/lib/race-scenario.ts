// Shared race-scenario runner — single source of truth for the per-tick
// simulation logic used by BOTH the React `/raceprimitives` page and the
// CLI `pnpm run race`. Implements the plan's prescription:
//
// > `RaceScenario` interface: `setup`, `tick(dt) → {laps, status}`, `dispose`.
// > Both the CLI and the React page consume this. Same single-path
// > discipline as the action-space unification.
//
// The scenario owns: one Rapier world per car (matches the web's split-
// viewport setup so cars never physically interact), the per-car
// `CarHandle` + per-car planner + lookahead replan + lap detection +
// optional sync-hold + stall guard + per-tick prediction-error metric.
//
// It does NOT own: rendering (Three.js scenes/meshes are a React-side
// concern; the React component reads `tickResult.cars[i].state` each
// frame and updates its meshes), recording (DebugRecorder), online
// learning (the project moved to offline-trained models).
//
// Determinism: the scenario advances physics in fixed PHYSICS_DT ticks
// driven by the caller. Given the same entries + seed + course, both
// the CLI loop and the React RAF loop produce the same lap times to
// physics-determinism limits.

import RAPIER from '@dimforge/rapier3d-compat';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  stepRaycastVehicle,
  type CarHandle,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import {
  wheeledFromNormalized,
  trimPlan,
  samplePlanAt,
  type CarForceTuning,
  type CarKinematicState,
  type WheeledCarControls,
} from 'kinocat/vehicle/car';
import {
  purePursuit,
  smoothSpeedProfile,
  smoothTrajectory,
  curvaturePerSample,
  resampleScalarByArcLength,
  mpcTrack,
  createMPCTrackerState,
  type MPCTrackerState,
  createSettleLatch,
  type SettleLatch,
} from 'kinocat/execute';
import {
  parametricForwardV2,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  deriveVehicleCapabilities,
  type VehicleAgent,
} from 'kinocat/agent';
import { buildPlan, segmentByGear, type Plan } from 'kinocat/plan';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavWorld, AnalyticEdgeData } from 'kinocat/environment';
import type { PlanResult } from 'kinocat/planner';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import {
  buildRaceCourse,
  planRaceMultiGoal,
  planRace,
  planRaceScenario,
  pickNextWaypoint,
  RACE_AGENT,
  RACE_REPLAN_BUDGET_MS,
  RACE_ARRIVE_RADIUS,
  RACE_PLANNER_GATE_RADIUS,
  RACE_MAX_EXPANSIONS,
  emptyMetrics,
  type RaceMetrics,
} from './race-primitives-scenarios';

// ---------------------------------------------------------------------------
// Configuration — single source of truth for the racing tunables. Anywhere
// these numbers used to live (RacePrimitives.tsx, headless-race.ts) now
// reads them from here.

export const PHYSICS_DT = 1 / 60;
export const VEHICLE_SUBSTEPS = 4;
export const REPLAN_INTERVAL_MS = 300;
export const PLAN_LOOKAHEAD_COUNT = 2;
export const TRACKER_MAX_LATERAL_ACCEL = 12;
export const STALL_TIMEOUT_MS = 4000;
export const ENGINE_FORCE_N = 4000;
export const BRAKE_FORCE_N = 2000;
export const WHEEL_BASE = 1.6;
/**
 * Plan-stitching commit window. The controller is guaranteed to follow the
 * currently-committed plan for this many ms before the next plan takes
 * over — eliminates mid-flight steering jumps caused by purely periodic
 * replanning, which is the dominant source of cyan's visible instability
 * on /raceprimitives. The new plan is computed from the predicted future
 * state at `simTime + COMMIT_WINDOW_MS`, so it's positionally consistent
 * with the segment the controller is still executing.
 */
export const COMMIT_WINDOW_MS = 200;
/**
 * Adaptive (event-driven) replan triggers — fired *in addition to* the
 * fixed `REPLAN_INTERVAL_MS` cadence. Real AV stacks all do this: the
 * cadence covers normal driving, the events cover the moments where the
 * cadence-only loop visibly lags reality.
 *
 * `LATERAL_ERROR_REPLAN_M`: when the chassis drifts farther than this
 * from the plan, the current plan is no longer the right reference and
 * we should replan immediately rather than waiting up to 300 ms.
 *
 * `MIN_TIME_BETWEEN_REPLANS_MS`: rate-limit so a slipping chassis can't
 * thrash the planner. Half the cadence is a reasonable floor.
 */
export const LATERAL_ERROR_REPLAN_M = 2.0;
export const MIN_TIME_BETWEEN_REPLANS_MS = 150;

const FORCE_TUNING: CarForceTuning = {
  engineForceN: ENGINE_FORCE_N,
  brakeForceN: BRAKE_FORCE_N,
};

const VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  suspensionMaxTravel: 0.2,
  engineForce: ENGINE_FORCE_N,
  brakeForce: BRAKE_FORCE_N,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd',
};

const PURE_PURSUIT_CONFIG = {
  lookaheadMin: 3,
  lookaheadGain: 0.45,
  lookaheadMax: 14,
  maxLateralAccel: TRACKER_MAX_LATERAL_ACCEL,
  maxAccel: 6,
  maxDecel: 8,
  cruiseSpeed: RACE_AGENT.maxSpeed,
  goalTolerance: 2,
  minTurnRadius: RACE_AGENT.minTurnRadius,
  // Consume the speed-profile-smoothed `speed` on every plan sample so
  // the friction-circle pass actually influences throttle/brake. Without
  // this the smoother has no effect and the controller would still
  // arrive hot at corner entries. (Tuning override below toggles this.)
  respectPathSpeed: true,
  // Anticipatory curvature braking: brake for UPCOMING plan-geometry
  // corners via the braking envelope, not just the instantaneous chord.
  // Measured on the feedforward executor (deterministic 2-lap
  // benchmark): costs both cars ~4 s of pace but eliminates the v2
  // car's corner-overshoot replan storms (11 → 0 failed replans) and
  // improves BOTH cars' closed-loop prediction error (kin 0.93 → 0.82,
  // v2 0.99 → 0.92 m). Reliability + predictability over peak pace.
  previewCurvature: true,
  previewLateralAccel:
    0.8 * deriveVehicleCapabilities(DEFAULT_LEARNABLE_CONFIG).maxLateralAccel,
  // Reverse gear executes at the chassis's reverse limit, NOT forward
  // cruise (measured without this: −24 m/s reverse down a 100 m
  // straight — outside the chassis envelope AND the model's training
  // distribution).
  reverseCruiseSpeed: RACE_AGENT.maxReverseSpeed,
};

// ---------------------------------------------------------------------------
// Multi-cusp plan segmentation.

/**
 * Split a plan polyline at every forward↔reverse cusp (sample where
 * the sign of `speed` flips). Within each returned segment the chassis
 * is in a single gear, so pure-pursuit's geometric tracking is well-
 * defined and its brake-to-goal logic naturally stops the chassis at
 * the cusp pose before the next-segment gear change. Plans with no
 * cusps return `[plan]` and downstream code is unchanged.
 *
 * Delegates to `segmentByGear` from `kinocat/plan` — the single source of
 * truth for gear splitting, shared with the rich `Plan` builder. Unlike the
 * previous local logic, it splits correctly across an exact rest sample
 * (`speed ≈ 0`, the stop pose the chassis reverses from): the boundary lands
 * ON that rest sample, so adjacent segments share it (the forward segment
 * ends at the stop; the reverse segment starts there). Segments shorter than
 * two samples are dropped, matching the prior contract.
 */
export function splitAtGearCusps(plan: CarKinematicState[]): CarKinematicState[][] {
  if (plan.length < 2) return [plan.slice()];
  const segs = segmentByGear(plan.map((p) => ({ vRef: p.speed })));
  const out: CarKinematicState[][] = [];
  for (const seg of segs) {
    const slice = plan.slice(seg.startIdx, seg.endIdx + 1);
    if (slice.length >= 2) out.push(slice);
  }
  return out.length > 0 ? out : [plan.slice()];
}

/**
 * Lift a plan result's node sequence into a dense, gear-correct trajectory,
 * expanding every Reeds-Shepp analytic "shot to goal" edge into its sampled
 * curve poses.
 *
 * Why this exists: the planner's `result.path` is just `nodes.map(n =>
 * n.state)`. For a motion-primitive edge that's fine (endpoints ~0.75 m apart;
 * the smoother densifies). But the final analytic shot collapses a multi-metre
 * CURVED Reeds-Shepp maneuver — the actual back-in / parallel-park swing — into
 * a single STRAIGHT chord from the last grid node to the goal pose, discarding
 * both the curve geometry and the forward/reverse gear along it. The tracker
 * then chases a straight diagonal that the chassis can't follow at the planned
 * heading, so it clips a neighbour or stalls short with a large heading error.
 *
 * Expanding the stored `poses` restores the true geometry and the per-sample
 * gear sign, so pure-pursuit reads the correct gear and the smoother rounds the
 * real curve instead of a chord. Race plans rarely include an analytic shot and
 * never a reverse one, so this is a no-op for racing.
 */
export function liftAnalyticPath(
  res: PlanResult<CarKinematicState>,
  speedMag: number,
  reverseSpeedMag: number = speedMag,
): CarKinematicState[] {
  const nodes = res.nodes;
  if (!nodes || nodes.length === 0) return res.path;
  const out: CarKinematicState[] = [];
  for (let i = 0; i < nodes.length; i++) {
    const st = nodes[i]!.state;
    if (i === 0) {
      out.push({ ...st });
      continue;
    }
    const edge = nodes[i]!.edge;
    const data = edge?.kind === 'reeds-shepp'
      ? (edge.data as AnalyticEdgeData | undefined)
      : undefined;
    if (data?.poses && data.poses.length >= 2) {
      // poses[0] is the shot's start pose (== previous node state, already
      // emitted); interpolate t linearly across the curve so downstream
      // time-based sampling (commit window, predicted-end) stays monotonic.
      const startT = out[out.length - 1]!.t;
      const endT = st.t;
      const m = data.poses.length;
      for (let j = 1; j < m; j++) {
        const p = data.poses[j]!;
        const frac = j / (m - 1);
        out.push({
          x: p.x,
          z: p.z,
          heading: p.heading,
          // Last sample lands on the goal pose — adopt its (zero) speed so the
          // tracker brakes to rest there; interior samples carry the gear sign
          // so reverse segments are driven in reverse.
          // Reverse legs are capped at the agent's reverse envelope — tagging
          // them with the forward maxSpeed made the plan's timeline (and the
          // commit-window predicted start states, and the tracker's reverse
          // target speed) ~33% faster than the plant can actually back up,
          // injecting ~0.4 m of lateral error at every mid-swing replan.
          speed: j === m - 1 ? st.speed : (p.reverse ? -reverseSpeedMag : speedMag),
          t: startT + (endT - startT) * frac,
        });
      }
    } else {
      out.push({ ...st });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Feature flags — for ablation studies.
//
// Every "best-in-class" planner improvement landed on this branch is gated
// behind a flag here. The default is ALL ON (everything we've measured to
// help). The headless-race CLI (`pnpm run race --tuning=...`) and a new
// `pnpm run ablation` script flip these to isolate the contribution of
// each improvement. Without this discipline, accidental regressions hide
// behind the aggregate "lap time went up by 2s" signal.
//
// Naming: a feature ON is "improvement enabled". Setting to its disabled
// value reverts to the legacy behaviour from before the improvement
// landed. `LEGACY_TUNING` is the all-off baseline; `DEFAULT_TUNING` is
// the all-on current state.

export interface RaceTuning {
  /** Plan-stitching commit window (ms). 0 = replan from live chassis state. */
  commitWindowMs: number;
  /** Trajectory-consistency hysteresis weight passed to the planner
   *  (s/m of deviation from the previously-committed plan). 0 disables. */
  consistencyWeight: number;
  /** Friction-circle forward/backward speed-profile post-pass on the plan. */
  enableSpeedProfile: boolean;
  /** Geometric trajectory smoother — dense (~0.4m) C¹-continuous polyline. */
  enableTrajectorySmoother: boolean;
  /** Pure-pursuit folds the smoothed plan's per-sample speeds into the
   *  target-speed clamp (the smoother has no effect without this). */
  respectPathSpeed: boolean;
  /** Override for the curvature preview's lateral-accel budget (m/s²).
   *  Undefined → PURE_PURSUIT_CONFIG default (plant grip ceiling). */
  previewLateralAccel?: number;
  /** Event-driven replan triggers in addition to the fixed cadence. */
  enableAdaptiveReplan: boolean;
  /** Trigger an extra replan on waypoint advance (subset of adaptive). */
  enableWaypointAdvanceReplan: boolean;
  /** Reeds-Shepp heuristic lookup table inside `VehicleEnvironment`
   *  (faster heuristic evaluation; admissible). */
  enableHeuristicTable: boolean;
  /**
   * Path-following tracker. `'pure-pursuit'` is the classic geometric
   * tracker — fast, reactive, no dynamics model. `'mpc'` is the
   * MPPI sampling MPC over the v2 parametric model — slower per tick
   * but accurate enough for high-fidelity execution (parking, multi-
   * step back-and-forth corrections, terminal-pose precision).
   * Defaults to `'pure-pursuit'`.
   */
  tracker: 'pure-pursuit' | 'mpc';
  /**
   * MPC terminal-pose cost weights. Scenario intent signal:
   * non-zero values mean "this plan asks the chassis to come to rest
   * at a pose" (parking, terminal-pose precision). Zero means "drive
   * through any goal you see" (racing, cruise). The bench's parking
   * entries set these explicitly; the race entry leaves them at 0.
   */
  mpcWTerminalPosition: number;
  mpcWTerminalSpeed: number;
  /**
   * Scenario-level tracker / waypoint knobs. These belong in the
   * tuning bundle (not in MPC config alone) because they govern
   * pure-pursuit AND MPPI alike — both controllers need to know
   * the chassis's cruise speed cap and how close it must get to the
   * goal pose for the scenario to consider it "done". Setting them
   * lets a scenario reuse the same `createRaceScenario` runner for
   * racing (fast cruise, generous gate radius) and parking (slow
   * cruise, tight terminal tolerance) without forking the runner.
   *
   * `cruiseSpeed`: max forward speed pure-pursuit will request.
   *                Defaults to `RACE_AGENT.maxSpeed`. Parking
   *                scenarios set this to the parking library's
   *                top speed (~2 m/s) so the controller doesn't
   *                blow past the planner's slow-maneuver primitives.
   * `goalTolerance`: distance at which `atGoal` triggers pure-
   *                pursuit's terminal brake. Defaults to 2.
   *                Parking sets this to ~0.4 m so the chassis
   *                actually stops in the stall.
   * `arriveRadius`: waypoint-advance radius in `pickNextWaypoint`.
   *                Defaults to RACE_ARRIVE_RADIUS (2.5 m). Parking
   *                needs sub-meter precision so this drops to ~0.5 m.
   */
  cruiseSpeed?: number;
  goalTolerance?: number;
  arriveRadius?: number;
  /**
   * Pure-pursuit lookahead overrides. The chassis defaults (3/0.45/14) are
   * race-scale; parking segments are 0.5-3 m, where a 3 m minimum lookahead
   * degenerates the tracker to "aim at the path endpoint" (it can never
   * remove lateral error). Parking supplies ~0.8/0.5/3.
   */
  lookaheadMin?: number;
  lookaheadGain?: number;
  lookaheadMax?: number;
  /** Approach-speed floor toward stop terminals (m/s) — see PurePursuitConfig. */
  minApproachSpeed?: number;
  /** Tracker curvature clamp (m). Align with the PLANT's real turn envelope. */
  trackerMinTurnRadius?: number;
  /** Chassis steering envelope (rad). Plant turn radius = axleSpacing/tan(this). */
  maxSteerAngle?: number;
  /** Reverse-gear cruise cap (m/s) — see PurePursuitConfig.reverseCruiseSpeed. */
  reverseCruiseSpeed?: number;
  /** Adaptive-replan lateral threshold (m). Race default 2.0; precision
   *  maneuvers should correct early (~0.35) — mid-swing drift toward an
   *  obstacle corner consumes the plan's clearance margin. */
  lateralErrorReplanM?: number;
  /** Curvature feedforward in pure-pursuit — see PurePursuitConfig. */
  curvatureFeedforward?: boolean;
  /**
   * Stanley-style heading-alignment gain for the pure-pursuit tracker (forward
   * gear only). 0/undefined ⇒ classic position-only pursuit (racing). Parking
   * sets this so the chassis drives onto the plan's terminal HEADING instead of
   * cutting the short final straightening curve and resting at its approach
   * angle — the parallel-park "ends ~16° off the curb" failure. See
   * `purePursuit`'s `headingGain`.
   */
  terminalHeadingGain?: number;
  /** Confine the heading-alignment term to within this distance of the goal (m)
   *  — the clear terminal zone — so it doesn't perturb the chassis off the
   *  tight, clearance-critical approach. See `purePursuit`'s `headingRadius`. */
  terminalHeadingRadius?: number;
  /**
   * Planner pose discretisation. Race uses defaults (1.5 m grid,
   * 16 heading buckets, 4 m goal radius, ignore terminal heading);
   * parking sets tight values (0.3 m / 36 / 0.35 / 0.15) so the
   * planner finds sub-meter precision plans with terminal-heading
   * constraints. Plumbed to `planVehicleOnce` / `planRaceMultiGoal`
   * via their `envOptions` override.
   */
  plannerPosCell?: number;
  plannerHeadingBuckets?: number;
  plannerGoalRadius?: number;
  plannerGoalHeadingTol?: number;
  /** Planner replan budget (ms). Race=120 ms; tight parking maneuvers
   *  need ~500 ms to find a maneuver through sub-meter clearances. */
  plannerBudgetMs?: number;
  /** Planner expansion cap. Race=30k; parking=80k. */
  plannerMaxExpansions?: number;
  /**
   * Fixed replan cadence (ms). Defaults to `REPLAN_INTERVAL_MS` (300).
   * Racing wants a brisk cadence so the line stays fresh against the
   * moving horizon. A multi-cusp PARKING maneuver is the opposite: it must
   * be executed as a COMMITTED sequence (reverse segment → forward segment →
   * …). Replanning from scratch mid-maneuver resets the segment cursor and
   * leaves the chassis oscillating at a forward↔reverse cusp (each replan
   * re-picks "reverse more" vs "pull in" from near-equal cost). A long
   * cadence lets the segment-advance logic carry the chassis through the
   * cusps; the adaptive lateral-drift trigger still forces a replan if the
   * chassis genuinely diverges from the plan. */
  replanIntervalMs?: number;
}

/**
 * Ablation-proven best configuration as measured by the headless race
 * harness (`pnpm run ablation`). On the kinematic library this beats
 * the LEGACY baseline by -45% best lap, -48% avg, -60% stddev, with
 * zero off-track events.
 *
 * What's enabled (each individually proven helpful by --mode=only-x):
 *  - consistency cost     (-52% avg alone, the largest single win)
 *  - adaptive replan      (-44% avg alone)
 *  - heuristic table      (-38% avg alone)
 *  - trajectory smoother  (-25% avg alone, also fixes the sharp-lines
 *                         visualisation; combines additively with the
 *                         three planner improvements above for the
 *                         best single-lap times we have measured)
 *
 * What's DISABLED and why:
 *  - commitWindowMs:    in combination with the smoother + adaptive
 *                       replan triggers, the predicted-future start
 *                       state diverges from where the chassis really
 *                       ends up, producing pure-pursuit overcorrections
 *                       and 17+ off-track events per 180 s in the
 *                       ablation harness. Needs a forward-simulation
 *                       prediction (not linear sampling) to be safe.
 *  - speed profile +    in combination with the smoother, the
 *    respectPathSpeed:  curvature-cap speeds are too conservative for
 *                       this physics + tyre model — cars complete only
 *                       1 lap per 180 s. The smoother's
 *                       distance-distributed curvature plus the 75%
 *                       aLat safety margin compound. Re-enable when
 *                       paired with a curvature-aware controller (MPC)
 *                       that can handle aggressive profiles.
 *
 * Re-enable any of these only after the ablation confirms they help.
 */
export const DEFAULT_TUNING: RaceTuning = {
  commitWindowMs: 0,
  consistencyWeight: 0.08,
  enableSpeedProfile: false,
  enableTrajectorySmoother: true,
  respectPathSpeed: false,
  enableAdaptiveReplan: true,
  enableWaypointAdvanceReplan: true,
  enableHeuristicTable: true,
  tracker: 'pure-pursuit',
  mpcWTerminalPosition: 0,
  mpcWTerminalSpeed: 0,
  // cruiseSpeed / goalTolerance / arriveRadius left undefined — fall
  // through to PURE_PURSUIT_CONFIG / RACE_ARRIVE_RADIUS chassis defaults.
};

/** All improvements disabled — reverts to the pre-improvement baseline.
 *  Useful as the reference point in ablation studies. */
export const LEGACY_TUNING: RaceTuning = {
  commitWindowMs: 0,
  consistencyWeight: 0,
  enableSpeedProfile: false,
  enableTrajectorySmoother: false,
  respectPathSpeed: false,
  enableAdaptiveReplan: false,
  enableWaypointAdvanceReplan: false,
  enableHeuristicTable: false,
  tracker: 'pure-pursuit',
  mpcWTerminalPosition: 0,
  mpcWTerminalSpeed: 0,
};

// ---------------------------------------------------------------------------
// Public types

export interface RaceEntry {
  /** Display name (unique among entries). */
  name: string;
  /** Motion-primitive library this entry's planner uses. */
  lib: MotionPrimitiveLibrary;
  /** Agent the planner reasons about for this entry — footprint, turn
   *  radius, maxSpeed, reverse/direction-change costs. MUST match the agent
   *  `lib` was characterised from; a mismatch rescales the planner's
   *  time-cost heuristic away from the primitives' real progress and
   *  collapses A* into near-breadth-first search (the parking replan storm).
   *  Defaults to `RACE_AGENT` (race course). Parking entries set
   *  `PARKING_AGENT`. */
  agent?: VehicleAgent;
}

export interface RaceLap {
  /** 1-based lap number. */
  lap: number;
  /** Simulation time when the lap completed (s). */
  simTime: number;
  /** Duration of THIS lap (s). */
  duration: number;
  /** Per-sector durations (one per waypoint cleared this lap). */
  sectors: number[];
}

export interface RaceCarDiagnostics {
  lastReplanMs: number;
  lastReplanFound: boolean;
  consecutiveFailedReplans: number;
  planAgeMs: number;
  successfulReplans: number;
  totalReplans: number;
  /** RMS prediction error at primitive boundary (planned vs actual pose). */
  predErrorRms: number;
}

export interface RaceCarStatus {
  name: string;
  state: CarKinematicState;
  controls: WheeledCarControls;
  loopIndex: number;
  laps: RaceLap[];
  finished: boolean;
  holdingForSync: boolean;
  offTrackEvents: number;
  diagnostics: RaceCarDiagnostics;
  metrics: RaceMetrics;
  /** Latest plan (lifted to a sequence of state samples for visualization). */
  plan: CarKinematicState[] | null;
  /** Rich Plan built from `plan` (kinocat/plan): per-point curvature,
   *  feedforward, dynamic-state slots, and single-gear segment/cusp
   *  structure — for the debug overlay and future controllers. */
  richPlan: Plan | null;
  /** Sim time at which the current plan was committed. */
  planStartSimTime: number;
  /** Most recent predicted state at the first primitive's boundary. */
  predictedEnd: { state: CarKinematicState; dueSimTime: number } | null;
  /** Sim time the most recent waypoint advance happened. */
  lastAdvanceSimTime: number;
}

export interface RaceTickResult {
  simTime: number;
  cars: RaceCarStatus[];
  /** True iff every car is `finished` OR no targetLaps was specified. */
  allFinished: boolean;
}

export interface RaceScenarioOptions {
  entries: RaceEntry[];
  /** Stop ticking + mark each car `finished` after this many laps. When
   *  omitted the scenario runs indefinitely (web demo). */
  targetLaps?: number;
  /**
   * TRUE goal-completion semantics (parking-class scenarios). When present,
   * `finished` latches ONLY when this predicate has held continuously at
   * rest for `holdSeconds` (a settle latch) — waypoint arrival keeps its
   * bookkeeping role but can no longer terminate the goal loop. Without
   * this, the runner's position-only 0.25 m arrival disk declared courses
   * complete while the actual goal (in-stall, square, centered, stopped)
   * was unsatisfied — so the corrective replans that would have fixed a
   * crooked park (shunt, or pull out and re-enter) were never attempted:
   * the car just held the brake, visibly mis-parked, forever.
   */
  goalSettle?: {
    predicate: (state: CarKinematicState) => boolean;
    holdSeconds: number;
    speedTol: number;
  };
  /** Hold the leader at the lap line until every car finishes the lap, so
   *  every car starts the next lap head-to-head. Web demo: true. CLI race:
   *  false (cars race independently). */
  syncHold?: boolean;
  /** When the chassis leaves the arena, what to do. 'spawn' teleports to
   *  the spawn point (CLI default); 'waypoint' teleports to the current
   *  target waypoint (web behavior); 'none' lets the car stay off-track. */
  offTrackRecovery?: 'spawn' | 'waypoint' | 'none';
  /** If the chassis hasn't moved >0.5m in this many ms, teleport it back
   *  to the current waypoint. Default = STALL_TIMEOUT_MS. */
  stallTimeoutMs?: number;
  /** Optional spawn offset along z per car (default: spread cars 3m apart). */
  spawnSpacingZ?: number;
  /** Feature-flag bundle controlling every best-in-class improvement
   *  layered onto the planner+controller. Defaults to `DEFAULT_TUNING`
   *  (everything ON). Used by ablation tooling and the CLI to A/B test
   *  individual improvements. */
  tuning?: Partial<RaceTuning>;
  /** Override the default race course. Used by the controller-bench
   *  to drive the same `createRaceScenario` runner through arbitrary
   *  test scenarios (parking, cruise, custom obstacle layouts...) —
   *  one scenario runner, many courses. Shape matches
   *  `buildRaceCourse()`'s return type. */
  course?: ReturnType<typeof buildRaceCourse>;
}

export interface RaceScenario {
  /** Advance the simulation by `dt` seconds. */
  tick(dt?: number): RaceTickResult;
  /** Read the current per-car status without advancing. */
  status(): RaceCarStatus[];
  /** Underlying Rapier world for the named entry (web-only — used to mount
   *  visual colliders alongside the chassis). */
  getWorld(name: string): RAPIER.World | null;
  /** Underlying CarHandle for the named entry (web-only — used to attach
   *  Three.js meshes to the chassis pose). */
  getCarHandle(name: string): CarHandle | null;
  /** Current sim time (s). */
  simTime(): number;
  /** Reset all cars to spawn + clear lap history. */
  reset(): void;
  /** Release any Rapier resources. */
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Internal state

interface CarInternal {
  entry: RaceEntry;
  world: RAPIER.World;
  car: CarHandle;
  navWorld: NavWorld;
  // Mutable per-tick state.
  loopIndex: number;
  laps: RaceLap[];
  currentLapSectors: number[];
  raceTime: number;
  lapStartSimTime: number;
  /** Settle latch driving `finished` when opts.goalSettle is present. */
  goalLatch: SettleLatch | null;
  waypointsCleared: number;
  plan: CarKinematicState[] | null;
  planStartSimTime: number;
  /**
   * Pending plan from the most recent replan. Computed from the predicted
   * future state at `simTime + COMMIT_WINDOW_MS`; promoted to `plan` once
   * we cross that wall, so the controller never sees a steering jump
   * caused by replanning alone.
   */
  pendingPlan: CarKinematicState[] | null;
  pendingPlanStartSimTime: number;
  /**
   * Plan split into single-gear segments at forward↔reverse cusps.
   * Pure-pursuit's geometric tracking assumes monotonic forward (or
   * reverse) motion within a segment; without this split, when the
   * chassis crosses a cusp the lookahead point can land in the
   * opposite-gear segment and the controller commands a nonsensical
   * steer. Each segment is executed end-to-end (brake to cusp pose,
   * advance to next segment which starts the gear change). For plans
   * with no cusps (race plans), `segments = [plan]` and behaviour is
   * unchanged.
   */
  segments: CarKinematicState[][];
  /**
   * Rich Plan (kinocat/plan) built from the committed `plan` for
   * visualization / future controllers. Produce-but-don't-consume:
   * pure-pursuit / MPPI still track `plan`/`segments`; this carries the
   * per-point curvature, feedforward, and cusp structure the bare state
   * array discards, so the debug overlay can plot it. Null until the first
   * plan commits.
   */
  richPlan: Plan | null;
  activeSegIdx: number;
  /** Sim time of the chassis state when the active segment was
   *  entered — used to gate the "segment done, advance" check on
   *  ≥0.2 s elapsed so the controller doesn't insta-skip a segment
   *  whose first sample happens to be within arrive radius. */
  activeSegStartSimTime: number;
  predictedEnd: { state: CarKinematicState; dueSimTime: number } | null;
  predErrSumSq: number;
  predErrCount: number;
  diagnostics: RaceCarDiagnostics;
  metrics: RaceMetrics;
  // Last replan wall (sim-time) for replan cadence.
  lastReplanSimTime: number;
  // Latest control vector applied (for status surface).
  lastControls: WheeledCarControls;
  // Sync hold / stall + off-track.
  holdingForSync: boolean;
  finished: boolean;
  offTrackEvents: number;
  // Stall guard (diagnostics only — no teleport rescue).
  lastMoveSimTime: number;
  lastPos: { x: number; z: number };
  // Whether the chassis was within arena bounds on the previous tick (used to
  // count off-track excursions on the in→out edge — diagnostics only).
  lastInBounds: boolean;
  // Spawn pose (for reset / off-track).
  spawn: { x: number; z: number; heading: number };
  // Persistent MPC tracker state (warm-start sequence + RNG seed) when
  // the tracker is `'mpc'`. Lazily initialised on first MPC tick.
  mpcState: MPCTrackerState | null;
}

// ---------------------------------------------------------------------------
// Public factory

export async function createRaceScenario(
  opts: RaceScenarioOptions,
): Promise<RaceScenario> {
  const rapier = await ensureRapier();
  const course = opts.course ?? buildRaceCourse();
  const targetLaps = opts.targetLaps;
  const syncHold = opts.syncHold ?? false;
  // NOTE: `offTrackRecovery` and `stallTimeoutMs` are retained on the options
  // for backward compatibility but are now INERT — the runner performs no
  // teleportation under any setting. A stuck or off-track chassis fails
  // honestly (see the stall/off-track tracking in stepOne). They are read here
  // only to avoid "unknown option" surprises and may be removed later.
  void opts.offTrackRecovery;
  void opts.stallTimeoutMs;
  const spacing = opts.spawnSpacingZ ?? 3;
  const tuning: RaceTuning = { ...DEFAULT_TUNING, ...(opts.tuning ?? {}) };
  // Tracker config derives from the tuning bundle so a single
  // `createRaceScenario` instance handles racing or parking based on
  // the scenario's per-tuning overrides. Anything not set falls
  // through to the chassis-level race defaults.
  const trackerConfig = {
    ...PURE_PURSUIT_CONFIG,
    cruiseSpeed: tuning.cruiseSpeed ?? PURE_PURSUIT_CONFIG.cruiseSpeed,
    goalTolerance: tuning.goalTolerance ?? PURE_PURSUIT_CONFIG.goalTolerance,
    lookaheadMin: tuning.lookaheadMin ?? PURE_PURSUIT_CONFIG.lookaheadMin,
    lookaheadGain: tuning.lookaheadGain ?? PURE_PURSUIT_CONFIG.lookaheadGain,
    lookaheadMax: tuning.lookaheadMax ?? PURE_PURSUIT_CONFIG.lookaheadMax,
    minApproachSpeed: tuning.minApproachSpeed,
    reverseCruiseSpeed: tuning.reverseCruiseSpeed,
    minTurnRadius: tuning.trackerMinTurnRadius ?? PURE_PURSUIT_CONFIG.minTurnRadius,
    curvatureFeedforward: tuning.curvatureFeedforward ?? false,
    respectPathSpeed: tuning.respectPathSpeed,
    previewLateralAccel:
      tuning.previewLateralAccel ?? PURE_PURSUIT_CONFIG.previewLateralAccel,
    headingGain: tuning.terminalHeadingGain ?? 0,
    // The runner gates the heading term on distance to the TRUE goal (see the
    // controller loop), so pure-pursuit's own per-path-end radius stays open.
    headingRadius: Infinity,
  };
  const arriveRadius = tuning.arriveRadius ?? RACE_ARRIVE_RADIUS;

  // MPC tracker shared config + forward simulator (constant per scenario).
  // The dynamics model is the v2 parametric model with default coefficients
  // — captures the Rapier chassis well enough for the controller to plan
  // accurate short-horizon rollouts even on cars that don't ship their own
  // learned model. Per-car: a `MPCTrackerState` holds the warm-start
  // sequence + deterministic RNG seed.
  const MPC_HORIZON = 10;
  const mpcForwardSim = parametricForwardV2(
    DEFAULT_LEARNED_PARAMS_V2,
    DEFAULT_LEARNABLE_CONFIG,
  );
  // MPPI tracker config. The cost weights work across the entire
  // racing-to-parking spectrum because the tracker auto-activates the
  // terminal-pose cost when the plan asks the chassis to stop near a
  // pose AND the goal is reachable within the horizon (i.e. parking
  // automatically when the plan structure says so, racing otherwise).
  //
  // λ (lambda) is THE key MPPI knob: smaller → more aggressive
  // (concentrates weight on the lowest-cost samples); larger → more
  // averaging (smoother but less decisive). 0.5 strikes the bias-
  // variance balance that handles both modes without retuning.
  //
  // Reverse stays ON (the parking workflow needs back-and-forth);
  // race plans simply won't have samples with negative drive selected
  // by the softmax because positive-drive samples track the plan
  // better. Letting reverse stay in the sample distribution costs
  // virtually nothing.
  const MPC_CONFIG = {
    horizonSteps: MPC_HORIZON,
    stepDt: 0.05,
    samples: 64,
    maxSteer: VEHICLE_TUNING.maxSteerAngle ?? 0.6,
    maxDriveForce: ENGINE_FORCE_N,
    maxBrakeForce: BRAKE_FORCE_N,
    allowReverse: true,
    lambda: 0.5,
    steerStd: 0.10,
    driveStd: 0.5 * ENGINE_FORCE_N,
    brakeStd: 0.10 * BRAKE_FORCE_N,
    wLateral: 2,
    wHeading: 3,
    wSpeed: 10,
    wControlRate: 0.15,
    wSteerRate: 25,
    wTerminalPosition: tuning.mpcWTerminalPosition,
    wTerminalSpeed: tuning.mpcWTerminalSpeed,
    goalTolerance: 0.5,
    // Cruise the reference at the SAME speed pure-pursuit uses, so the MPC
    // reference extends a full horizon ahead and the chassis accelerates to
    // racing speed. Without this the tracker inferred cruise from the plan's
    // terminal speed (≈ 0 even on race gates) and crawled to a stall (DNF).
    cruiseSpeed: trackerConfig.cruiseSpeed,
  };

  const cars: CarInternal[] = opts.entries.map((entry, i) => {
    // One world per car (matches web demo's split-viewport setup; cars
    // never physically interact). The headless CLI used to share a world
    // — switching to one world per car eliminates cross-car collisions
    // as a possible source of nondeterminism.
    const world = new rapier.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(world, {
      bounds: course.bounds,
      pad: 20,
      friction: 1.5,
    });
    const offset = (i - (opts.entries.length - 1) / 2) * spacing;
    const spawn = {
      x: course.spawn.x,
      z: course.spawn.z + offset,
      heading: course.spawn.heading,
    };
    const car = createRaycastVehicle(world, {
      id: entry.name,
      position: { x: spawn.x, z: spawn.z },
      heading: spawn.heading,
      ...VEHICLE_TUNING,
      // Steering envelope is scenario-tunable: parking chassis steer sharper
      // (0.75 rad -> R ~ 3.44 m) so the PLANT can actually drive the tight
      // arcs the parking planner needs in stall-sized geometry; racing keeps
      // the stable 0.6 rad envelope.
      maxSteerAngle: tuning.maxSteerAngle ?? VEHICLE_TUNING.maxSteerAngle,
    });
    const navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
    return {
      entry,
      world,
      car,
      navWorld,
      loopIndex: 0,
      laps: [],
      currentLapSectors: [],
      raceTime: 0,
      lapStartSimTime: 0,
      goalLatch: opts.goalSettle
        ? createSettleLatch({
            holdSeconds: opts.goalSettle.holdSeconds,
            speedTol: opts.goalSettle.speedTol,
          })
        : null,
      waypointsCleared: 0,
      plan: null,
      planStartSimTime: 0,
      pendingPlan: null,
      pendingPlanStartSimTime: 0,
      segments: [],
      richPlan: null,
      activeSegIdx: 0,
      activeSegStartSimTime: 0,
      predictedEnd: null,
      predErrSumSq: 0,
      predErrCount: 0,
      diagnostics: {
        lastReplanMs: 0,
        lastReplanFound: false,
        consecutiveFailedReplans: 0,
        planAgeMs: 0,
        successfulReplans: 0,
        totalReplans: 0,
        predErrorRms: 0,
      },
      metrics: emptyMetrics(),
      lastReplanSimTime: -Infinity,
      lastControls: { steer: 0, driveForce: 0, brakeForce: 0 },
      holdingForSync: false,
      finished: false,
      offTrackEvents: 0,
      lastMoveSimTime: 0,
      lastPos: { x: spawn.x, z: spawn.z },
      lastInBounds: true,
      spawn,
      mpcState: null,
    };
  });

  // Settle suspension briefly — fixed budget so it doesn't pollute
  // raceTime accumulation (which only ticks while `running` and never
  // before the first tick() call returns).
  for (let i = 0; i < 20; i++) {
    for (const c of cars) {
      c.car.applyWheeledControls({ steer: 0, driveForce: 0, brakeForce: 0 });
      stepRaycastVehicle(c.world, [c.car], { dt: PHYSICS_DT, substeps: 1 });
    }
  }
  // Snap each car back to spawn after settle.
  for (const c of cars) c.car.teleport(c.spawn);

  let simTime = 0;
  const replanIntervalSec = (tuning.replanIntervalMs ?? REPLAN_INTERVAL_MS) / 1000;

  function replanCar(c: CarInternal): void {
    if (c.holdingForSync || c.finished) return;
    // Commit-window plan stitching. If a plan is already committed, we
    // plan from the predicted state at `simTime + COMMIT_WINDOW_MS`, not
    // from the current state — guaranteeing the controller can keep
    // following the existing plan through the commit window without
    // discontinuity. First replan (no plan yet) starts from `now`.
    const commitWindowSec = tuning.commitWindowMs / 1000;
    let planStartSimTime = simTime;
    let startState: CarKinematicState;
    if (commitWindowSec > 0 && c.plan && c.plan.length > 1) {
      const elapsed = simTime - c.planStartSimTime + commitWindowSec;
      const sampled = samplePlanAt(c.plan, elapsed);
      if (sampled) {
        startState = { ...sampled, t: 0 };
        planStartSimTime = simTime + commitWindowSec;
      } else {
        startState = { ...c.car.readState(simTime), t: 0 };
      }
    } else {
      startState = { ...c.car.readState(simTime), t: 0 };
    }
    // Pick the planning loopIndex by walking forward through the
    // waypoint list as long as the predicted START state is already
    // within the arrive radius of the next gate — otherwise the commit
    // window can land the search past a gate that the planner is still
    // told to reach (the path would loop back). The actual
    // `c.loopIndex` is only advanced at 60Hz from real state in
    // `pickNextWaypoint`, so this is a planning-only correction.
    let planLoopIndex = c.loopIndex;
    {
      const advRadiusSq = arriveRadius * arriveRadius;
      for (let i = 0; i < PLAN_LOOKAHEAD_COUNT; i++) {
        const wp = course.waypoints[planLoopIndex % course.waypoints.length]!;
        const dx = startState.x - wp.x;
        const dz = startState.z - wp.z;
        if (dx * dx + dz * dz <= advRadiusSq) {
          planLoopIndex = (planLoopIndex + 1) % course.waypoints.length;
        } else {
          break;
        }
      }
    }
    const gates: CarKinematicState[] = [];
    for (let i = 0; i < PLAN_LOOKAHEAD_COUNT; i++) {
      const idx = (planLoopIndex + i) % course.waypoints.length;
      gates.push({ ...course.waypoints[idx]!, t: 0 });
    }
    // Trajectory-consistency reference: feed the freshest plan we have
    // (pending if available — it's strictly newer than the committed one)
    // to the planner as a soft hysteresis term, so a freshly-searched
    // plan only wins when meaningfully better than the one the chassis
    // is already tracking. Trim to the part of the plan we have NOT yet
    // executed (relative to the new search's start time) so reference
    // geometry is only about the future, not the past.
    let referencePath: ReadonlyArray<{ x: number; z: number }> | undefined;
    if (tuning.consistencyWeight > 0) {
      const refPlan = c.pendingPlan ?? c.plan;
      const refStart = c.pendingPlan ? c.pendingPlanStartSimTime : c.planStartSimTime;
      if (refPlan && refPlan.length > 1) {
        const elapsedForRef = planStartSimTime - refStart;
        const tail = trimPlan(refPlan, elapsedForRef);
        if (tail.length >= 2) {
          referencePath = tail.map((p) => ({ x: p.x, z: p.z }));
        }
      }
    }
    const tStart = performance.now();
    // Single-waypoint courses (parking) use `planRace` (=
    // `planVehicleOnce` with terminal-heading constraint + tight
    // discretisation). Multi-waypoint courses (race loops) use the
    // multi-goal planner that trades off entries/exits across
    // gates. The bench's parking entries always supply exactly one
    // waypoint (the goal pose), so this branch is the natural
    // discriminator — no extra config flag needed.
    const isParking = course.waypoints.length === 1;
    const plannerBudget = tuning.plannerBudgetMs ?? RACE_REPLAN_BUDGET_MS;
    const plannerMaxExp = tuning.plannerMaxExpansions ?? RACE_MAX_EXPANSIONS;
    // Shared single-goal planner params (parking tuning).
    const singleGoalParams = {
      state: startState,
      lib: c.entry.lib,
      agent: c.entry.agent,
      polygons: course.polygons,
      obstacles: course.obstacles,
      world: c.navWorld,
      deadlineMs: plannerBudget,
      maxExpansions: plannerMaxExp,
      posCell: tuning.plannerPosCell,
      headingBuckets: tuning.plannerHeadingBuckets,
      goalRadius: tuning.plannerGoalRadius,
      goalHeadingTol: tuning.plannerGoalHeadingTol,
      enableHeuristicTable: tuning.enableHeuristicTable,
      // Soft hysteresis toward the last committed plan so the multi-cusp
      // parking maneuver stays stable across replans instead of flipping
      // between near-equal-cost back-in alternatives every 300 ms.
      referencePath,
      referenceWeight: tuning.consistencyWeight,
    };
    const res = isParking
      ? course.goal
        ? // NEW: plan toward the canonical Scenario goal through the
          // ScenarioEnvironment bridge (the goal is described in the
          // kinocat/scenario layer and read by both planner + visualizer).
          planRaceScenario({
            ...singleGoalParams,
            goal: course.goal,
            invariants: course.invariants,
            prefer: course.prefer,
          })
        : // Legacy fallback: single goal pose.
          planRace({ ...singleGoalParams, goal: gates[0]! })
      : planRaceMultiGoal({
          state: startState,
          gates,
          lib: c.entry.lib,
          agent: c.entry.agent,
          polygons: course.polygons,
          obstacles: course.obstacles,
          world: c.navWorld,
          deadlineMs: plannerBudget,
          gateRadius: RACE_PLANNER_GATE_RADIUS,
          referencePath,
          referenceWeight: tuning.consistencyWeight,
          disableHeuristicTable: !tuning.enableHeuristicTable,
        });
    const replanMs = performance.now() - tStart;
    c.diagnostics.lastReplanMs = replanMs;
    // A found 1-point path is the planner saying "the start already satisfies
    // the goal" (start-state acceptance) — a SUCCESS with nothing to drive,
    // not a planner failure. Count it as found (so failedReplanRatio stays
    // honest) but skip the commit: the degenerate-plan brake-hold keeps the
    // chassis at rest and the settle latch finishes the course.
    const trivial = res.found && res.path.length === 1 && res.cost === 0;
    c.diagnostics.lastReplanFound = res.found && (res.path.length > 1 || trivial);
    c.diagnostics.totalReplans += 1;
    if (trivial) {
      c.diagnostics.successfulReplans += 1;
      c.diagnostics.consecutiveFailedReplans = 0;
      c.lastReplanSimTime = simTime;
      return;
    }
    if (c.diagnostics.lastReplanFound) {
      // Two-stage post-process pipeline (Apollo / Autoware shape), each
      // stage individually toggleable for ablation:
      //  (a) Geometric trajectory smoother — turns the sparse, sharp-
      //      seamed motion-primitive polyline into a dense (~0.4m
      //      spacing), C¹-continuous reference. Without this the
      //      prediction/visualisation/lookahead all sample on a
      //      piecewise-linear interpolation of primitive endpoints.
      //  (b) Friction-circle speed-profile pass — assigns a
      //      curvature- and brake-distance-aware speed at every sample.
      // Two-stage plan post-process: smoother (sparse → dense C¹) +
      // friction-circle speed pass (curvature-aware speeds the chassis
      // can physically achieve). Speed profile uses ORIGINAL primitive
      // curvature, resampled onto the smoothed samples by arc-length,
      // so the smoother's geometric rounding doesn't fool the speed
      // pass into commanding impossible speeds.
      // Expand any analytic Reeds-Shepp shot-to-goal into its real curved,
      // gear-tagged samples BEFORE smoothing — otherwise the multi-cusp
      // back-in / parallel-park swing is a straight chord the chassis can't
      // track at the planned heading. Only the terminal-pose (parking) branch
      // needs this: race plans are driven through gates forward-only, and an
      // analytic shot there may include a reverse sub-segment that the
      // forward-only race controller can't execute — collapsing it to the
      // straight chord (res.path) is what keeps racing flowing.
      const liftedPath = isParking
        ? liftAnalyticPath(
            res,
            (c.entry.agent ?? RACE_AGENT).maxSpeed,
            (c.entry.agent ?? RACE_AGENT).maxReverseSpeed ?? (c.entry.agent ?? RACE_AGENT).maxSpeed,
          )
        : res.path;
      let smoothed = liftedPath;
      const kRaw = tuning.enableSpeedProfile ? curvaturePerSample(liftedPath) : null;
      if (tuning.enableTrajectorySmoother) {
        smoothed = smoothTrajectory(smoothed, {
          sampleSpacing: 0.4,
          iterations: 4,
          dataWeight: 0.7,
          smoothWeight: 0.15,
          anchorEndpoints: true,
        });
      }
      if (tuning.enableSpeedProfile && kRaw) {
        const kForProfile = resampleScalarByArcLength(liftedPath, kRaw, smoothed);
        smoothed = smoothSpeedProfile(smoothed, {
          aLatMax: PURE_PURSUIT_CONFIG.maxLateralAccel * 0.85,
          aLonMaxAccel: PURE_PURSUIT_CONFIG.maxAccel,
          aLonMaxDecel: PURE_PURSUIT_CONFIG.maxDecel,
          maxSpeed: PURE_PURSUIT_CONFIG.cruiseSpeed,
          minSpeed: 0.5,
          honorEntrySpeed: true,
          curvatureOverride: kForProfile,
        });
      }
      c.diagnostics.successfulReplans += 1;
      c.diagnostics.consecutiveFailedReplans = 0;
      if (commitWindowSec > 0 && c.plan && c.plan.length > 1) {
        // Promote later — keep the current plan active through the
        // commit window. `stepOne` will swap when simTime crosses
        // pendingPlanStartSimTime. Skipped when commit window is 0 so
        // the legacy "replace immediately" behaviour is preserved.
        c.pendingPlan = smoothed;
        c.pendingPlanStartSimTime = planStartSimTime;
      } else {
        c.plan = smoothed;
        c.planStartSimTime = planStartSimTime;
        c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
        c.segments = splitAtGearCusps(smoothed);
        c.richPlan = buildPlan(smoothed, { wheelBase: 2 * WHEEL_BASE });
        c.activeSegIdx = 0;
        c.activeSegStartSimTime = simTime;
      }
      const firstEnd = smoothed.find((p) => p.t > 0.05) ?? smoothed[smoothed.length - 1]!;
      c.predictedEnd = { state: firstEnd, dueSimTime: planStartSimTime + firstEnd.t };
    } else {
      c.diagnostics.consecutiveFailedReplans += 1;
    }
    c.lastReplanSimTime = simTime;
  }

  /** Promote the pending plan if its commit window has elapsed. */
  function maybePromotePlan(c: CarInternal): void {
    if (c.pendingPlan && simTime >= c.pendingPlanStartSimTime) {
      c.plan = c.pendingPlan;
      c.planStartSimTime = c.pendingPlanStartSimTime;
      c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
      c.segments = splitAtGearCusps(c.plan);
      c.richPlan = buildPlan(c.plan, { wheelBase: 2 * WHEEL_BASE });
      c.activeSegIdx = 0;
      c.activeSegStartSimTime = simTime;
    }
  }

  /** Perpendicular distance from a point to the unexecuted future of the
   *  current plan polyline. Returns Infinity if the plan is missing or
   *  fully past. Used by the adaptive replan trigger to detect when the
   *  chassis has drifted from the reference too far for the controller
   *  alone to recover comfortably. */
  function lateralFromPlan(c: CarInternal, x: number, z: number): number {
    // GEOMETRIC divergence, measured against the ACTIVE SEGMENT — the exact
    // polyline the tracker is following. Two prior designs both misfired:
    //  - time-trimmed tails read plan-time lag as "divergence" (segments now
    //    genuinely DWELL at cusp rest samples, so the chassis is always
    //    behind plan-time near a cusp — that is not an error);
    //  - whole-plan minimum distance under-reports on multi-cusp plans whose
    //    legs pass near each other (the pose matches a PAST leg).
    // Plan exhaustion is a separate, progress-based condition in the cadence
    // gate — pretending it is infinite lateral error caused replan storms.
    const tail = c.segments[c.activeSegIdx] ?? c.plan;
    if (!tail || tail.length < 2) return Infinity;
    let best = Infinity;
    for (let i = 0; i < tail.length - 1; i++) {
      const ax = tail[i]!.x;
      const az = tail[i]!.z;
      const bx = tail[i + 1]!.x;
      const bz = tail[i + 1]!.z;
      const dx = bx - ax;
      const dz = bz - az;
      const lenSq = dx * dx + dz * dz;
      let u = 0;
      if (lenSq > 1e-9) {
        u = ((x - ax) * dx + (z - az) * dz) / lenSq;
        if (u < 0) u = 0;
        else if (u > 1) u = 1;
      }
      const px = ax + dx * u;
      const pz = az + dz * u;
      const d = Math.hypot(x - px, z - pz);
      if (d < best) best = d;
    }
    return best;
  }

  /** Returns true if any adaptive trigger fires AND the rate-limit allows
   *  it. The trigger set follows the Apollo / Autoware shape: lateral
   *  divergence from the plan, mid-cadence waypoint advance (so the
   *  planner can re-aim at the new horizon immediately), and
   *  consecutive failures (back off — keep retrying until something
   *  takes). */
  function shouldEarlyReplan(c: CarInternal, state: CarKinematicState): boolean {
    if (!tuning.enableAdaptiveReplan) return false;
    if (c.holdingForSync || c.finished) return false;
    const sinceLastMs = (simTime - c.lastReplanSimTime) * 1000;
    if (sinceLastMs < MIN_TIME_BETWEEN_REPLANS_MS) return false;
    // Trigger: large lateral error from the plan.
    const dLat = lateralFromPlan(c, state.x, state.z);
    if (dLat > (tuning.lateralErrorReplanM ?? LATERAL_ERROR_REPLAN_M)) return true;
    // Trigger: consecutive failures — keep trying. Cadence already
    // does this every 300ms but events let us retry earlier.
    if (c.diagnostics.consecutiveFailedReplans >= 2) return true;
    return false;
  }

  function stepOne(c: CarInternal, dt: number): void {
    if (c.finished) {
      c.car.applyWheeledControls({ steer: 0, driveForce: 0, brakeForce: BRAKE_FORCE_N });
      stepRaycastVehicle(c.world, [c.car], { dt, substeps: VEHICLE_SUBSTEPS });
      return;
    }
    const stateBefore = c.car.readState(simTime);
    // Promote any pending plan whose commit window has elapsed BEFORE
    // we decide whether to replan, so the next replan sees the most
    // recently-committed plan as its baseline.
    maybePromotePlan(c);
    // Replan if the fixed cadence elapsed OR an adaptive trigger fires
    // (lateral divergence, consecutive planner failures, etc.). The
    // trigger checks rate-limit themselves so a degenerate state can't
    // thrash the planner.
    const cadenceDue = simTime - c.lastReplanSimTime >= replanIntervalSec;
    // A cadence replan is only worth taking when the committed plan is no
    // longer serving: the chassis has drifted off it, it is near exhaustion,
    // or planning has been failing. Unconditionally re-planning every cycle
    // re-rolls the maneuver from mid-execution poses — on parking approaches
    // each re-roll re-threads the SAME gap from a slightly worse pose and
    // progressively commits thinner-clearance plans (0.55 m -> 0.39 m
    // observed) until execution noise turns margin into contact. A healthy,
    // on-track plan is left alone. (Multi-waypoint racing keeps the horizon
    // fresh: its plans chase moving gates, so staleness there IS drift/
    // exhaustion and the same conditions fire naturally.)
    let cadenceUseful = true;
    // While the goal predicate holds at rest, the settle hold is in progress:
    // planning is pure churn (it returns the trivial plan) and any commit
    // could only disturb the hold. Let the latch finish.
    if (c.goalLatch?.state.holding) cadenceUseful = false;
    if (cadenceUseful && cadenceDue && c.plan && c.plan.length > 1) {
      const dLat = lateralFromPlan(c, stateBefore.x, stateBefore.z);
      // Exhaustion by PROGRESS, not by clock: cusp dwells and decel ramps put
      // the chassis behind the plan's timeline by design, so elapsed-time
      // exhaustion re-planned mid-maneuver on every cusp (the post-merge
      // 102-replan churn). The plan is spent when the chassis is tracking
      // its FINAL segment and has consumed most of it.
      let nearExhaustion = false;
      const seg = c.segments[c.activeSegIdx];
      if (!seg || c.activeSegIdx >= c.segments.length - 1) {
        const tail = seg ?? c.plan;
        let ni = 0;
        let bd = Infinity;
        for (let i = 0; i < tail.length; i++) {
          const d = Math.hypot(stateBefore.x - tail[i]!.x, stateBefore.z - tail[i]!.z);
          if (d < bd) {
            bd = d;
            ni = i;
          }
        }
        nearExhaustion = ni >= (tail.length - 1) * 0.8;
      }
      const failing = c.diagnostics.consecutiveFailedReplans > 0;
      cadenceUseful = dLat > 0.25 || nearExhaustion || failing;
    }
    if ((cadenceDue && cadenceUseful) || shouldEarlyReplan(c, stateBefore)) {
      replanCar(c);
    }
    // Waypoint advance + lap detection (60Hz so lap times aren't quantized
    // to the replan cadence). Single-waypoint goal-settle courses (parking)
    // skip this entirely: the lone waypoint can only wrap onto itself, so
    // every crossing of the arrive disk read as an "advance" — firing the
    // waypoint-advance replan trigger every rate-limit window (5+/s) while
    // the car shunted near the goal, and stamping phantom laps. Completion
    // is the settle latch's job there.
    if (!c.holdingForSync && !(opts.goalSettle && course.waypoints.length === 1)) {
      const pick = pickNextWaypoint(
        { ...stateBefore, t: 0 },
        course.waypoints,
        c.loopIndex,
        arriveRadius,
      );
      if (pick.advanced) {
        c.waypointsCleared++;
        c.loopIndex = pick.nextIndex;
        const sectorTime = c.raceTime - c.lapStartSimTime;
        c.currentLapSectors.push(sectorTime);
        // Adaptive trigger: a gate has just been cleared, so the planner's
        // horizon has shifted. Re-plan immediately (subject to the
        // MIN_TIME_BETWEEN_REPLANS rate-limit) instead of waiting up to
        // ~300 ms of cadence — otherwise the controller can chase a plan
        // whose first gate the car has already passed.
        if (tuning.enableWaypointAdvanceReplan) {
          const sinceLastMs = (simTime - c.lastReplanSimTime) * 1000;
          if (sinceLastMs >= MIN_TIME_BETWEEN_REPLANS_MS) {
            replanCar(c);
          }
        }
        if (c.waypointsCleared % course.waypoints.length === 0) {
          const lapEnd = c.raceTime;
          const dur = lapEnd - c.lapStartSimTime;
          c.laps.push({
            lap: c.laps.length + 1,
            simTime: lapEnd,
            duration: dur,
            sectors: c.currentLapSectors.slice(),
          });
          c.metrics.laps = c.laps.length;
          c.metrics.lastLapTime = dur;
          c.metrics.bestLapTime = Number.isFinite(c.metrics.bestLapTime)
            ? Math.min(c.metrics.bestLapTime, dur)
            : dur;
          c.lapStartSimTime = lapEnd;
          c.currentLapSectors = [];
          if (targetLaps !== undefined && c.laps.length >= targetLaps && !opts.goalSettle) {
            // Lap-count completion is a RACE concept. Goal-settle courses
            // (parking) latch `finished` from the settle oracle below —
            // a position-only arrival must not stop the goal loop while the
            // true goal (square, centered, at rest) is unsatisfied.
            c.finished = true;
          } else if (syncHold) {
            c.holdingForSync = true;
          }
        }
      }
    }
    // Apply controls.
    if (c.holdingForSync) {
      const cmd = wheeledFromNormalized({ steer: 0, throttle: 0, brake: 1 }, FORCE_TUNING);
      c.car.applyWheeledControls(cmd);
      c.lastControls = cmd;
      c.metrics.liveControls = { steer: 0, throttle: 0, brake: 1, targetSpeed: 0 };
    } else if (c.plan && c.plan.length > 1) {
      // Multi-cusp segment advancement: when the chassis reaches the
      // end of the active segment (within arrive radius AND nearly
      // stopped) AND ≥0.2 s has elapsed in the segment (avoid insta-
      // skipping segments whose first sample is already inside the
      // arrive radius), move to the next segment. The brake-to-end
      // logic in pure-pursuit naturally drives the chassis to a
      // near-stop at the cusp pose; the gear change happens at the
      // segment boundary by the new segment's plan-speed sign. For
      // plans with no cusps (race), `segments = [plan]` and this
      // never advances past 0.
      if (c.segments.length > 1 && c.activeSegIdx < c.segments.length - 1) {
        const seg = c.segments[c.activeSegIdx]!;
        const segEnd = seg[seg.length - 1]!;
        const segElapsed = simTime - c.activeSegStartSimTime;
        const dist = Math.hypot(stateBefore.x - segEnd.x, stateBefore.z - segEnd.z);
        if (
          segElapsed >= 0.2 &&
          dist <= (tuning.arriveRadius ?? RACE_ARRIVE_RADIUS) &&
          Math.abs(stateBefore.speed) < 0.5
        ) {
          c.activeSegIdx++;
          c.activeSegStartSimTime = simTime;
        }
      }
      // Source of truth for the controller: the current single-gear
      // segment, NOT the full plan. Pure-pursuit's geometric formula
      // assumes monotonic motion within the path it sees; feeding it
      // a multi-cusp plan makes its lookahead land in the wrong gear.
      // Pure-pursuit's own `nearestIndex` walk handles "where am I on
      // this segment" so we don't need to trim by elapsed time — the
      // last sample of the segment is naturally the brake-to-goal
      // target (the cusp pose or the final goal).
      const live = c.segments[c.activeSegIdx] ?? c.plan;
      if (live.length >= 2) {
        if (tuning.tracker === 'mpc') {
          // Sampling MPC over the v2 parametric model.
          if (!c.mpcState) c.mpcState = createMPCTrackerState(MPC_HORIZON);
          const cmdRaw = mpcTrack(stateBefore, live, mpcForwardSim, c.mpcState, MPC_CONFIG);
          const cmd: WheeledCarControls = {
            steer: cmdRaw.steer,
            driveForce: cmdRaw.driveForce,
            brakeForce: cmdRaw.brakeForce,
          };
          c.car.applyWheeledControls(cmd);
          c.lastControls = cmd;
          c.metrics.liveControls = {
            steer: cmdRaw.steer,
            throttle: cmd.driveForce >= 0
              ? cmd.driveForce / ENGINE_FORCE_N
              : -cmd.driveForce / ENGINE_FORCE_N,
            brake: cmd.brakeForce / BRAKE_FORCE_N,
            targetSpeed: cmdRaw.targetSpeed,
          };
        } else {
          // Gate the terminal heading-alignment term on distance to the TRUE
          // goal (the final waypoint), not the per-segment path end. The executor
          // feeds `purePursuit` per-SEGMENT paths, so its built-in `headingRadius`
          // (keyed on distance-to-path-end) would also fire within range of every
          // forward↔reverse cusp, rotating the chassis toward a cusp tangent
          // mid-maneuver rather than only straightening onto the goal heading.
          // Keying on the real goal engages the term on the terminal approach
          // (whichever segment is near the goal) and nowhere else. `purePursuit`'s
          // own radius is left open (Infinity, above) so this is the sole gate.
          let trkCfg = trackerConfig;
          if (trackerConfig.headingGain) {
            const wp = course.waypoints[c.loopIndex % course.waypoints.length]!;
            const dGoal = Math.hypot(stateBefore.x - wp.x, stateBefore.z - wp.z);
            if (dGoal > (tuning.terminalHeadingRadius ?? Infinity)) {
              trkCfg = { ...trackerConfig, headingGain: 0 };
            }
          }
          const trk = purePursuit(stateBefore, live, trkCfg);
          // pure-pursuit returns `throttle` as a non-negative MAGNITUDE and
          // encodes the drive direction in the SIGN of `targetSpeed` (it also
          // flips the steering body-frame for reverse, so `trk.steering` is the
          // curvature in the direction of travel). Two corrections are needed
          // before this drives a raycast vehicle:
          //
          //  1. SIGNED THROTTLE. `wheeledFromNormalized` takes throttle in
          //     [-1,1] (negative = reverse drive force). Without re-applying
          //     the gear sign the chassis always drives forward and silently
          //     ignores any planned reverse maneuver (reverse-perp back-in,
          //     parallel-park shunts) — it just coasts forward off the plan.
          //
          //  2. STEER SIGN FLIP IN REVERSE. For an Ackermann/bicycle chassis a
          //     given front-wheel angle produces OPPOSITE world-frame curvature
          //     depending on travel direction. pure-pursuit's curvature is for
          //     the travel direction, so when backing up the applied steer
          //     angle must be negated — otherwise the car steers the wrong way
          //     while reversing and curls away from the plan (it backed out of
          //     the stall and stalled at the wrong heading).
          //
          // Race plans are forward-only (targetSpeed ≥ 0 ⇒ gear = +1), so both
          // corrections are no-ops for racing.
          const gear = trk.targetSpeed < 0 ? -1 : 1;
          const signedThrottle = gear * trk.throttle;
          const steer = -gear * Math.atan(trk.steering * (2 * WHEEL_BASE));
          const cmd = wheeledFromNormalized(
            { steer, throttle: signedThrottle, brake: trk.brake },
            FORCE_TUNING,
          );
          c.car.applyWheeledControls(cmd);
          c.lastControls = cmd;
          c.metrics.liveControls = {
            steer: trk.steering, throttle: signedThrottle, brake: trk.brake,
            targetSpeed: trk.targetSpeed,
          };
        }
      } else {
        // Degenerate live segment (a trivial already-satisfied plan, or a
        // 1-point stub): HOLD — brake to rest. This is the natural terminal
        // state after settling: replans from a satisfied pose return the
        // trivial plan and the chassis simply keeps holding the brake. The
        // old behavior (creep forward at 0.2 throttle) silently drove a
        // planless car into whatever was ahead of it.
        const cmd = wheeledFromNormalized({ steer: 0, throttle: 0, brake: 0.6 }, FORCE_TUNING);
        c.car.applyWheeledControls(cmd);
        c.lastControls = cmd;
        c.metrics.liveControls = { steer: 0, throttle: 0, brake: 0.6, targetSpeed: 0 };
      }
    } else {
      // No plan at all (planner failed / first tick): hold still and wait for
      // the next replan rather than creeping blind.
      const cmd = wheeledFromNormalized({ steer: 0, throttle: 0, brake: 0.6 }, FORCE_TUNING);
      c.car.applyWheeledControls(cmd);
      c.lastControls = cmd;
      c.metrics.liveControls = { steer: 0, throttle: 0, brake: 0.6, targetSpeed: 0 };
    }
    stepRaycastVehicle(c.world, [c.car], { dt, substeps: VEHICLE_SUBSTEPS });
    const after = c.car.readState(simTime + dt);
    // TRUE goal completion (goal-settle courses): `finished` latches only
    // when the goal predicate has held continuously at rest — the same
    // settle semantics the bench/tests/HUD measure. Until then the goal
    // loop stays live: a mis-parked car keeps replanning (shunt, or pull
    // out and re-enter) instead of holding the brake on a crooked pose.
    if (c.goalLatch && opts.goalSettle) {
      c.goalLatch.update(
        { ok: opts.goalSettle.predicate(after), speed: after.speed },
        dt,
      );
      if (c.goalLatch.state.settled && !c.finished) {
        c.finished = true;
        if (c.laps.length === 0) {
          // Record the completion for HUD/bench timing parity: duration is
          // the honest time-to-settled, not the transient arrival instant.
          const t = c.goalLatch.state.timeToSettled ?? simTime + dt;
          c.laps.push({ lap: 1, simTime: simTime + dt, duration: t, sectors: [] });
          c.metrics.laps = 1;
        }
      }
    }
    // Prediction-error metric.
    if (c.predictedEnd && simTime + dt >= c.predictedEnd.dueSimTime) {
      const p = c.predictedEnd.state;
      const dx = after.x - p.x;
      const dz = after.z - p.z;
      c.predErrSumSq += dx * dx + dz * dz;
      c.predErrCount++;
      c.diagnostics.predErrorRms = Math.sqrt(c.predErrSumSq / c.predErrCount);
      c.metrics.trackingErrorRms = c.diagnostics.predErrorRms;
      c.predictedEnd = null;
    }
    // Metrics.
    c.metrics.peakSpeed = Math.max(c.metrics.peakSpeed, Math.abs(after.speed));
    c.diagnostics.planAgeMs = (simTime - c.planStartSimTime) * 1000;
    c.metrics.planDiagnostics.lastReplanMs = c.diagnostics.lastReplanMs;
    c.metrics.planDiagnostics.lastReplanFound = c.diagnostics.lastReplanFound;
    c.metrics.planDiagnostics.consecutiveFailedReplans = c.diagnostics.consecutiveFailedReplans;
    c.metrics.planDiagnostics.planAgeMs = c.diagnostics.planAgeMs;
    c.metrics.planDiagnostics.successfulReplans = c.diagnostics.successfulReplans;
    c.metrics.planDiagnostics.totalReplans = c.diagnostics.totalReplans;
    if (!c.holdingForSync) c.raceTime += dt;
    c.metrics.raceTime = c.raceTime;
    c.metrics.waypointsCleared = c.waypointsCleared;
    // Stall tracking — diagnostics ONLY. There is NO teleport rescue: real
    // vehicles don't teleport, and snapping a stuck chassis back onto a
    // waypoint masks the failure (it's how the reverse-perp parking bug stayed
    // hidden). A stuck car simply stays stuck and the run fails honestly.
    if (!c.holdingForSync) {
      const moved = Math.hypot(after.x - c.lastPos.x, after.z - c.lastPos.z) > 0.5;
      if (moved) {
        c.lastMoveSimTime = simTime;
        c.lastPos = { x: after.x, z: after.z };
      }
    }
    // Off-track tracking — diagnostics ONLY, no teleport rescue. We still count
    // excursions (and NaN blow-ups) so they surface as an honest failure
    // signal in `offTrackEvents`; the chassis is never snapped back.
    {
      const x0 = course.bounds.x0 - 15;
      const x1 = course.bounds.x1 + 15;
      const z0 = course.bounds.z0 - 15;
      const z1 = course.bounds.z1 + 15;
      const wasInBounds = c.lastInBounds;
      const inBounds =
        Number.isFinite(after.x) &&
        after.x >= x0 && after.x <= x1 && after.z >= z0 && after.z <= z1;
      // Count a new excursion on the in→out transition (edge, not per-tick).
      if (wasInBounds && !inBounds) c.offTrackEvents++;
      c.lastInBounds = inBounds;
    }
  }

  function buildStatus(c: CarInternal): RaceCarStatus {
    return {
      name: c.entry.name,
      state: c.car.readState(simTime),
      controls: c.lastControls,
      loopIndex: c.loopIndex,
      laps: c.laps,
      finished: c.finished,
      holdingForSync: c.holdingForSync,
      offTrackEvents: c.offTrackEvents,
      diagnostics: c.diagnostics,
      metrics: c.metrics,
      plan: c.plan,
      richPlan: c.richPlan,
      planStartSimTime: c.planStartSimTime,
      predictedEnd: c.predictedEnd,
      lastAdvanceSimTime: c.lastMoveSimTime,
    };
  }

  function reset(): void {
    simTime = 0;
    for (const c of cars) {
      c.car.teleport(c.spawn);
      c.loopIndex = 0;
      c.laps = [];
      c.currentLapSectors = [];
      c.raceTime = 0;
      c.lapStartSimTime = 0;
      c.waypointsCleared = 0;
      c.plan = null;
      c.planStartSimTime = 0;
      c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
      c.richPlan = null;
      c.pendingPlanStartSimTime = 0;
      c.predictedEnd = null;
      c.predErrSumSq = 0;
      c.predErrCount = 0;
      c.diagnostics = {
        lastReplanMs: 0, lastReplanFound: false, consecutiveFailedReplans: 0,
        planAgeMs: 0, successfulReplans: 0, totalReplans: 0, predErrorRms: 0,
      };
      c.metrics = emptyMetrics();
      c.lastReplanSimTime = -Infinity;
      c.lastControls = { steer: 0, driveForce: 0, brakeForce: 0 };
      c.holdingForSync = false;
      c.finished = false;
      c.offTrackEvents = 0;
      c.lastMoveSimTime = 0;
      c.lastPos = { x: c.spawn.x, z: c.spawn.z };
      c.lastInBounds = true;
    }
  }

  return {
    tick(dtOverride?: number) {
      const dt = dtOverride ?? PHYSICS_DT;
      for (const c of cars) stepOne(c, dt);
      // Sync-hold release: when EVERY car is holding, release them all.
      if (syncHold && cars.length > 1 && cars.every((c) => c.holdingForSync || c.finished)) {
        for (const c of cars) {
          if (!c.finished) c.holdingForSync = false;
          c.plan = null;
          c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
          c.richPlan = null;
        }
        // Force an immediate replan so neither car coasts on a stale plan.
        for (const c of cars) replanCar(c);
      }
      simTime += dt;
      const allFinished = targetLaps !== undefined && cars.every((c) => c.finished);
      return {
        simTime,
        cars: cars.map(buildStatus),
        allFinished,
      };
    },
    status() {
      return cars.map(buildStatus);
    },
    getWorld(name) {
      return cars.find((c) => c.entry.name === name)?.world ?? null;
    },
    getCarHandle(name) {
      return cars.find((c) => c.entry.name === name)?.car ?? null;
    },
    simTime() {
      return simTime;
    },
    reset,
    dispose() {
      // Rapier worlds are GC-ed; nothing to release.
    },
  };
}
