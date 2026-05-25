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
} from 'kinocat/execute';
import {
  parametricForwardV2,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
} from 'kinocat/agent';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavWorld } from 'kinocat/environment';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import {
  buildRaceCourse,
  planRaceMultiGoal,
  pickNextWaypoint,
  RACE_AGENT,
  RACE_REPLAN_BUDGET_MS,
  RACE_ARRIVE_RADIUS,
  RACE_PLANNER_GATE_RADIUS,
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
};

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
   * tracker — fast, reactive, no dynamics model. `'mpc'` is a
   * short-horizon sampling MPC over the v2 parametric model — slower
   * per tick but accurate enough for high-fidelity execution
   * (parking, multi-step back-and-forth corrections, terminal-pose
   * precision). Defaults to `'pure-pursuit'`.
   */
  tracker: 'pure-pursuit' | 'mpc';
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
};

// ---------------------------------------------------------------------------
// Public types

export interface RaceEntry {
  /** Display name (unique among entries). */
  name: string;
  /** Motion-primitive library this entry's planner uses. */
  lib: MotionPrimitiveLibrary;
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
}

// ---------------------------------------------------------------------------
// Public factory

export async function createRaceScenario(
  opts: RaceScenarioOptions,
): Promise<RaceScenario> {
  const rapier = await ensureRapier();
  const course = buildRaceCourse();
  const targetLaps = opts.targetLaps;
  const syncHold = opts.syncHold ?? false;
  const offTrackRecovery = opts.offTrackRecovery ?? 'spawn';
  const stallTimeoutMs = opts.stallTimeoutMs ?? STALL_TIMEOUT_MS;
  const spacing = opts.spawnSpacingZ ?? 3;
  const tuning: RaceTuning = { ...DEFAULT_TUNING, ...(opts.tuning ?? {}) };
  const trackerConfig = {
    ...PURE_PURSUIT_CONFIG,
    respectPathSpeed: tuning.respectPathSpeed,
  };

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
    wTerminalPosition: 50,
    wTerminalSpeed: 30,
    goalTolerance: 0.5,
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
      const advRadiusSq = RACE_ARRIVE_RADIUS * RACE_ARRIVE_RADIUS;
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
    const res = planRaceMultiGoal({
      state: startState,
      gates,
      lib: c.entry.lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      world: c.navWorld,
      deadlineMs: RACE_REPLAN_BUDGET_MS,
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
      let smoothed = res.path;
      const kRaw = tuning.enableSpeedProfile ? curvaturePerSample(res.path) : null;
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
        const kForProfile = resampleScalarByArcLength(res.path, kRaw, smoothed);
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
        c.pendingPlan = null;
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
      c.pendingPlan = null;
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
        RACE_ARRIVE_RADIUS,
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
      const elapsed = simTime - c.planStartSimTime;
      const live = trimPlan(c.plan, elapsed);
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
        const cmd = wheeledFromNormalized({ steer: 0, throttle: 0.2, brake: 0 }, FORCE_TUNING);
        c.car.applyWheeledControls(cmd);
        c.lastControls = cmd;
        c.metrics.liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 5 };
      }
    } else {
      const cmd = wheeledFromNormalized({ steer: 0, throttle: 0.2, brake: 0 }, FORCE_TUNING);
      c.car.applyWheeledControls(cmd);
      c.lastControls = cmd;
      c.metrics.liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 5 };
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
        c.pendingPlan = null;
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
        c.pendingPlan = null;
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
      c.pendingPlan = null;
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
          c.pendingPlan = null;
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
