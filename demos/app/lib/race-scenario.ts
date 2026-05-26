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
  expandPlanSweeps,
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
} from 'kinocat/execute';
import {
  parametricForwardV2,
  learnedForwardSimV2,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  type LearnedVehicleModel,
} from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavWorld } from 'kinocat/environment';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import {
  buildRaceCourse,
  planRaceMultiGoal,
  planRace,
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
export const PLAN_LOOKAHEAD_COUNT = 3;
export const TRACKER_MAX_LATERAL_ACCEL = 12;
export const STALL_TIMEOUT_MS = 2000;
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
 * Borrowed verbatim from the WIP parking branch
 * (claude/fervent-cori-KXMEy → `splitAtGearCusps` in Parking.tsx).
 */
export function splitAtGearCusps(plan: CarKinematicState[]): CarKinematicState[][] {
  if (plan.length < 2) return [plan.slice()];
  const out: CarKinematicState[][] = [];
  let cur: CarKinematicState[] = [plan[0]!];
  for (let i = 1; i < plan.length; i++) {
    const s = plan[i]!;
    const prev = cur[cur.length - 1]!;
    const flipped =
      (prev.speed > 1e-3 && s.speed < -1e-3) ||
      (prev.speed < -1e-3 && s.speed > 1e-3);
    if (flipped) {
      if (cur.length >= 2) out.push(cur);
      cur = [s];
    } else {
      cur.push(s);
    }
  }
  if (cur.length >= 2) out.push(cur);
  return out.length > 0 ? out : [plan.slice()];
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
  /** Optional learned dynamics model. When present AND the tracker is
   *  `'mpc'`, the MPC controller uses this model's forward sim instead of
   *  the default parametric model — aligning the tracker's dynamics with
   *  the planner's primitives so there is no planning-execution mismatch. */
  model?: LearnedVehicleModel;
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

/** Per-replan snapshot for diagnosing plan instability. */
export interface ReplanSnapshot {
  /** Sim time when replanCar() was called. */
  simTime: number;
  /** Wall-clock ms the planner search took. */
  searchMs: number;
  /** Whether a valid path was found. */
  found: boolean;
  /** A* expansion count. */
  expansions: number;
  /** Total nodes generated. */
  generated: number;
  /** Whether the deadline was hit. */
  deadlineHit: boolean;
  /** Plan cost (Infinity if not found). */
  cost: number;
  /** Number of anytime improvements found. */
  improvements: number;
  /** Start state the planner used (may differ from chassis if commit-window). */
  startState: CarKinematicState;
  /** Actual chassis state at replan time. */
  chassisState: CarKinematicState;
  /** Gate sequence the planner targeted. */
  gates: Array<{ x: number; z: number }>;
  /** loopIndex at replan time. */
  loopIndex: number;
  /** Planned path endpoint count (after expansion/smoothing). */
  planLength: number;
  /** First 3 + last 3 plan waypoints (compact summary). */
  planEndpoints: CarKinematicState[];
  /** Path displacement vs previous plan: mean + max distance (m) between
   *  corresponding arc-length-sampled points. -1 if no previous plan. */
  vsLastPlan: { meanDist: number; maxDist: number };
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
  /** Sim time at which the current plan was committed. */
  planStartSimTime: number;
  /** Most recent predicted state at the first primitive's boundary. */
  predictedEnd: { state: CarKinematicState; dueSimTime: number } | null;
  /** Sim time the most recent waypoint advance happened. */
  lastAdvanceSimTime: number;
  /** Index of the currently executing gear-segment (0-based). */
  activeSegmentIndex: number;
  /** Total number of gear-segments in the current plan. */
  totalSegments: number;
  /** Gear of the active segment: 'fwd' | 'rev' | 'unknown'. */
  activeSegmentGear: 'fwd' | 'rev' | 'unknown';
  /** Recent replan snapshots (ring buffer, newest last, max 30). */
  replanHistory: ReplanSnapshot[];
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
  // Stall guard.
  lastMoveSimTime: number;
  lastPos: { x: number; z: number };
  // Spawn pose (for reset / off-track).
  spawn: { x: number; z: number; heading: number };
  // Persistent MPC tracker state (warm-start sequence + RNG seed) when
  // the tracker is `'mpc'`. Lazily initialised on first MPC tick.
  mpcState: MPCTrackerState | null;
  /** Recent replan snapshots (ring buffer, newest last). */
  replanHistory: ReplanSnapshot[];
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
  const offTrackRecovery = opts.offTrackRecovery ?? 'spawn';
  const stallTimeoutMs = opts.stallTimeoutMs ?? STALL_TIMEOUT_MS;
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
    respectPathSpeed: tuning.respectPathSpeed,
  };
  const arriveRadius = tuning.arriveRadius ?? RACE_ARRIVE_RADIUS;

  // MPC tracker config + forward simulator. When an entry supplies a
  // `model`, the MPC controller uses that model's forward sim — aligning
  // the tracker's dynamics with the planner's primitives. Without a model,
  // falls back to the default parametric model (still reasonable for
  // kinematic-library entries). Per-car forward sims are built lazily in
  // `mpcForwardSimFor` so the shared config block stays constant.
  const MPC_HORIZON = 10;
  const defaultMpcForwardSim = parametricForwardV2(
    DEFAULT_LEARNED_PARAMS_V2,
    DEFAULT_LEARNABLE_CONFIG,
  );
  const perCarMpcForwardSim = new Map<string, ForwardSim<CarKinematicState>>();
  function mpcForwardSimFor(entry: RaceEntry): ForwardSim<CarKinematicState> {
    let sim = perCarMpcForwardSim.get(entry.name);
    if (sim) return sim;
    if (entry.model) {
      sim = parametricForwardV2(entry.model.params, entry.model.config);
    } else {
      sim = defaultMpcForwardSim;
    }
    perCarMpcForwardSim.set(entry.name, sim);
    return sim;
  }
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
    allowReverse: tuning.mpcWTerminalPosition > 0,
    lambda: 0.5,
    samples: 128,
    steerStd: 0.08,
    driveStd: 0.3 * ENGINE_FORCE_N,
    brakeStd: 0.15 * BRAKE_FORCE_N,
    wLateral: 8,
    wHeading: 3,
    wSpeed: 0.5,
    wControlRate: 0.1,
    wSteerRate: 5,
    wTerminalPosition: tuning.mpcWTerminalPosition,
    wTerminalSpeed: tuning.mpcWTerminalSpeed,
    goalTolerance: 0.5,
    // Racing: the reference trajectory should advance at the car's
    // actual speed (or faster), not the plan's endpoint speed. Without
    // this, the MPC brakes to match low plan speeds near waypoint
    // endpoints. The min floor of 5 prevents near-zero advance rates
    // when the car is stopped. Parking scenarios leave this at 0.
    minReferenceSpeed: tuning.mpcWTerminalPosition > 0 ? 0 : 5,
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
      waypointsCleared: 0,
      plan: null,
      planStartSimTime: 0,
      pendingPlan: null,
      pendingPlanStartSimTime: 0,
      segments: [],
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
      spawn,
      mpcState: null,
      replanHistory: [],
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
  const replanIntervalSec = REPLAN_INTERVAL_MS / 1000;

  function replanCar(c: CarInternal): void {
    if (c.holdingForSync || c.finished) return;
    // Snapshot the previous plan before it's replaced (for drift comparison).
    const prevPlan = c.plan;
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
    const res = isParking
      ? planRace({
          state: startState,
          goal: gates[0]!,
          lib: c.entry.lib,
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
        })
      : planRaceMultiGoal({
          state: startState,
          gates,
          lib: c.entry.lib,
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
    c.diagnostics.lastReplanFound = res.found && res.path.length > 1;
    c.diagnostics.totalReplans += 1;
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
      // Expand sparse primitive endpoints → dense sweep-sampled path.
      // This gives the smoother and visualisation a much more faithful
      // representation of the actual curved trajectories the planner chose.
      const densePath = res.nodes.length > 1
        ? expandPlanSweeps(res.nodes, c.entry.lib.primitives)
        : res.path;
      let smoothed = densePath;
      const kRaw = tuning.enableSpeedProfile ? curvaturePerSample(densePath) : null;
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
        const kForProfile = resampleScalarByArcLength(densePath, kRaw, smoothed);
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
        c.activeSegIdx = 0;
        c.activeSegStartSimTime = simTime;
      }
      const firstEnd = smoothed.find((p) => p.t > 0.05) ?? smoothed[smoothed.length - 1]!;
      c.predictedEnd = { state: firstEnd, dueSimTime: planStartSimTime + firstEnd.t };
    } else {
      c.diagnostics.consecutiveFailedReplans += 1;
    }
    // --- Capture replan snapshot for debugging plan instability ---
    const MAX_REPLAN_HISTORY = 30;
    const chassisNow = c.car.readState(simTime);
    const finalPlan = c.diagnostics.lastReplanFound ? (c.plan ?? []) : [];
    // Compact plan summary: first 3 + last 3 samples.
    const planEndpoints: CarKinematicState[] = [];
    if (finalPlan.length > 0) {
      for (let i = 0; i < Math.min(3, finalPlan.length); i++) planEndpoints.push(finalPlan[i]!);
      for (let i = Math.max(finalPlan.length - 3, 3); i < finalPlan.length; i++) planEndpoints.push(finalPlan[i]!);
    }
    // Compare vs previous plan: sample both at evenly-spaced arc lengths
    // and compute displacement stats.
    let vsLastPlan = { meanDist: -1, maxDist: -1 };
    if (prevPlan && prevPlan.length >= 2 && finalPlan.length >= 2) {
      // Sample 10 evenly-spaced points along each plan by index-fraction
      // and measure positional difference between old and new plan.
      {
        const N = 10;
        let sumDist = 0;
        let maxDist = 0;
        for (let i = 0; i < N; i++) {
          const u = i / (N - 1);
          const ai = Math.min(Math.floor(u * (finalPlan.length - 1)), finalPlan.length - 1);
          const bi = Math.min(Math.floor(u * (prevPlan.length - 1)), prevPlan.length - 1);
          const a = finalPlan[ai]!;
          const b = prevPlan[bi]!;
          const d = Math.hypot(a.x - b.x, a.z - b.z);
          sumDist += d;
          if (d > maxDist) maxDist = d;
        }
        vsLastPlan = { meanDist: sumDist / N, maxDist };
      }
    }
    c.replanHistory.push({
      simTime,
      searchMs: replanMs,
      found: c.diagnostics.lastReplanFound,
      expansions: res.stats.expansions,
      generated: res.stats.generated,
      deadlineHit: res.stats.deadlineHit,
      cost: res.cost,
      improvements: res.stats.improvements,
      startState: { ...startState },
      chassisState: { ...chassisNow, t: 0 },
      gates: gates.map((g) => ({ x: g.x, z: g.z })),
      loopIndex: planLoopIndex,
      planLength: finalPlan.length,
      planEndpoints,
      vsLastPlan,
    });
    if (c.replanHistory.length > MAX_REPLAN_HISTORY) {
      c.replanHistory.shift();
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
    if (!c.plan || c.plan.length < 2) return Infinity;
    const elapsed = Math.max(0, simTime - c.planStartSimTime);
    const tail = trimPlan(c.plan, elapsed);
    if (tail.length < 2) return Infinity;
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
    if (dLat > LATERAL_ERROR_REPLAN_M) return true;
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
    if (cadenceDue || shouldEarlyReplan(c, stateBefore)) {
      replanCar(c);
    }
    // Waypoint advance + lap detection (60Hz so lap times aren't quantized
    // to the replan cadence).
    if (!c.holdingForSync) {
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
          if (targetLaps !== undefined && c.laps.length >= targetLaps) {
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
          const cmdRaw = mpcTrack(stateBefore, live, mpcForwardSimFor(c.entry), c.mpcState, MPC_CONFIG);
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
          const trk = purePursuit(stateBefore, live, trackerConfig);
          const steer = -Math.atan(trk.steering * (2 * WHEEL_BASE));
          const cmd = wheeledFromNormalized(
            { steer, throttle: trk.throttle, brake: trk.brake },
            FORCE_TUNING,
          );
          c.car.applyWheeledControls(cmd);
          c.lastControls = cmd;
          c.metrics.liveControls = {
            steer: trk.steering, throttle: trk.throttle, brake: trk.brake,
            targetSpeed: trk.targetSpeed,
          };
        }
      } else {
        // Short plan — apply stronger throttle if stalled to escape zero-speed deadzone.
        const stalled = Math.abs(stateBefore.speed) < 0.5;
        const thr = stalled ? 0.6 : 0.2;
        const cmd = wheeledFromNormalized({ steer: 0, throttle: thr, brake: 0 }, FORCE_TUNING);
        c.car.applyWheeledControls(cmd);
        c.lastControls = cmd;
        c.metrics.liveControls = { steer: 0, throttle: thr, brake: 0, targetSpeed: 5 };
      }
    } else {
      // No plan yet — apply stronger throttle if stalled.
      const stalled = Math.abs(stateBefore.speed) < 0.5;
      const thr = stalled ? 0.6 : 0.2;
      const cmd = wheeledFromNormalized({ steer: 0, throttle: thr, brake: 0 }, FORCE_TUNING);
      c.car.applyWheeledControls(cmd);
      c.lastControls = cmd;
      c.metrics.liveControls = { steer: 0, throttle: thr, brake: 0, targetSpeed: 5 };
    }
    stepRaycastVehicle(c.world, [c.car], { dt, substeps: VEHICLE_SUBSTEPS });
    const after = c.car.readState(simTime + dt);
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
    // Stall guard.
    if (!c.holdingForSync) {
      const moved = Math.hypot(after.x - c.lastPos.x, after.z - c.lastPos.z) > 0.5;
      if (moved) {
        c.lastMoveSimTime = simTime;
        c.lastPos = { x: after.x, z: after.z };
      } else if ((simTime - c.lastMoveSimTime) * 1000 > stallTimeoutMs) {
        const wp = course.waypoints[c.loopIndex]!;
        c.car.teleport({ x: wp.x, z: wp.z, heading: wp.heading });
        c.plan = null;
        c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
        c.lastMoveSimTime = simTime;
        c.lastPos = { x: wp.x, z: wp.z };
      }
    }
    // Off-track recovery.
    if (offTrackRecovery !== 'none') {
      const x0 = course.bounds.x0 - 15;
      const x1 = course.bounds.x1 + 15;
      const z0 = course.bounds.z0 - 15;
      const z1 = course.bounds.z1 + 15;
      if (after.x < x0 || after.x > x1 || after.z < z0 || after.z > z1 || !Number.isFinite(after.x)) {
        c.offTrackEvents++;
        const target = offTrackRecovery === 'waypoint' ? course.waypoints[c.loopIndex]! : c.spawn;
        c.car.teleport({ x: target.x, z: target.z, heading: target.heading });
        c.plan = null;
        c.pendingPlan = null; c.segments = []; c.activeSegIdx = 0;
      }
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
      planStartSimTime: c.planStartSimTime,
      predictedEnd: c.predictedEnd,
      lastAdvanceSimTime: c.lastMoveSimTime,
      activeSegmentIndex: c.activeSegIdx,
      totalSegments: c.segments.length,
      activeSegmentGear: (() => {
        const seg = c.segments[c.activeSegIdx];
        if (!seg || seg.length < 2) return 'unknown' as const;
        const sample = seg[1];
        return sample && sample.speed >= 0 ? 'fwd' as const : 'rev' as const;
      })(),
      replanHistory: c.replanHistory,
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
      c.replanHistory = [];
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
