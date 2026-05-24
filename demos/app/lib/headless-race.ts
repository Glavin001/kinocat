// Headless race benchmark — runs the `/raceprimitives` scenario without
// any renderer or React, returning per-car lap times. Used by:
//   - `pnpm run race` CLI to compare v2 / kinematic / arbitrary-model
//     against each other deterministically.
//   - Phase 3 acceptance gate ("v2 beats kinematic on lap time") as a
//     pass/fail CI signal.
//
// The simulation core (Rapier world + per-car planner + pure-pursuit) is
// the same as the React page, factored out here so both consumers share
// one truth and CLI lap times match what the page shows on the same seed.

import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  RapierCarBody,
  stepRaycastVehicle,
  type CarHandle,
} from 'kinocat/adapters/rapier';
import {
  PlanFollowerCarDriver,
  type CarKinematicState,
  type WheeledCarControls,
} from 'kinocat/vehicle/car';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import {
  buildRaceCourse,
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  planThroughWaypoints,
  pickNextWaypoint,
  RACE_ARRIVE_RADIUS,
  RACE_BOUNDS,
} from './race-primitives-scenarios';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  type LearnedVehicleModel,
} from 'kinocat/agent';

const PHYSICS_DT = 1 / 60;
const PLAN_LOOKAHEAD_COUNT = 2;
const REPLAN_EVERY_SEC = 0.3;

const PURE_PURSUIT_CONFIG = {
  lookaheadMin: 4.0,
  lookaheadGain: 0.7,
  lookaheadMax: 12,
  maxLateralAccel: 12,
  maxAccel: 8,
  maxDecel: 14,
  cruiseSpeed: 28,
  goalTolerance: 1.6,
  minTurnRadius: 4.5,
};

export interface RaceConfig {
  /** Physics body engine force (N). */
  engineForce?: number;
  brakeForce?: number;
  maxSteerAngle?: number;
  wheelBase?: number;
  wheelTrack?: number;
  wheelRadius?: number;
}

const DEFAULT_CFG: Required<RaceConfig> = {
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
};

const SUSPENSION_REST_LENGTH = 0.3;
const SUSPENSION_MAX_TRAVEL = 0.2;

export interface RaceEntry {
  /** Display name. */
  name: string;
  /** Primitive library this entry's planner uses. */
  lib: MotionPrimitiveLibrary;
}

export interface RaceLap {
  /** 1-based lap number. */
  lap: number;
  /** Wall-clock simulation time when the lap completed (s). */
  time: number;
  /** Duration of THIS lap (= time - prev lap time, or - 0 for lap 1). */
  duration: number;
}

export interface RaceResult {
  name: string;
  laps: RaceLap[];
  /** Best lap duration (s) or NaN. */
  best: number;
  /** Mean lap duration (s) or NaN. */
  avg: number;
  /** Total sim time consumed (s). */
  totalSimTime: number;
  /** Did the car complete `targetLaps` within the time budget? */
  finished: boolean;
  /** How many times the chassis left the arena / rolled. */
  offTrackEvents: number;
}

export interface RunRaceOptions {
  entries: RaceEntry[];
  targetLaps?: number;
  /** Max sim seconds before DNF. */
  maxSimTime?: number;
  /** Per-replan budget (ms). Default 80 to keep the CLI snappy. */
  replanBudgetMs?: number;
  cfg?: RaceConfig;
  /** Called every `progressEverySec` simulated seconds with a small
   *  status update string (for the CLI progress bar). */
  onProgress?: (msg: string) => void;
  progressEverySec?: number;
}

interface CarState {
  entry: RaceEntry;
  body: RapierCarBody;
  car: CarHandle;
  driver: PlanFollowerCarDriver;
  loopIndex: number;
  laps: RaceLap[];
  offTrackEvents: number;
  lastPlanTime: number;
  finished: boolean;
  planMisses?: number;
  lastSegments?: number;
}

/** Run the race scenario for each entry. All entries race in parallel in
 *  the same Rapier world (separate chassis bodies). Returns per-entry
 *  lap results. */
export async function runHeadlessRace(
  opts: RunRaceOptions,
): Promise<RaceResult[]> {
  const rapier = await ensureRapier();
  const world = new rapier.World({ x: 0, y: -9.81, z: 0 });
  createGroundCollider(world, {
    bounds: { x0: RACE_BOUNDS.x0 - 20, x1: RACE_BOUNDS.x1 + 20, z0: RACE_BOUNDS.z0 - 20, z1: RACE_BOUNDS.z1 + 20 },
    pad: 20,
    friction: 1.5,
  });
  const course = buildRaceCourse();
  const cfg = { ...DEFAULT_CFG, ...opts.cfg };
  const targetLaps = opts.targetLaps ?? 3;
  const maxSimTime = opts.maxSimTime ?? 240;
  // Lay out entries along the spawn line so they don't collide.
  const cars: CarState[] = opts.entries.map((entry, i) => {
    const spawnX = course.spawn.x;
    const spawnZ = course.spawn.z + (i - (opts.entries.length - 1) / 2) * 3;
    const car = createRaycastVehicle(world, {
      id: entry.name,
      position: { x: spawnX, z: spawnZ },
      heading: course.spawn.heading,
      chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
      chassisDensity: 60,
      wheelBase: cfg.wheelBase,
      wheelTrack: cfg.wheelTrack,
      wheelRadius: cfg.wheelRadius,
      suspensionRestLength: SUSPENSION_REST_LENGTH,
      suspensionMaxTravel: SUSPENSION_MAX_TRAVEL,
      engineForce: cfg.engineForce,
      brakeForce: cfg.brakeForce,
      maxSteerAngle: cfg.maxSteerAngle,
      driveTrain: 'rwd',
    });
    const body = new RapierCarBody({ world, car, stepPolicy: 'external' });
    const driver = new PlanFollowerCarDriver({
      config: PURE_PURSUIT_CONFIG,
      wheelBase: cfg.wheelBase,
      engineForceN: cfg.engineForce,
      brakeForceN: cfg.brakeForce,
      maxSteerAngle: cfg.maxSteerAngle,
    });
    return {
      entry,
      body,
      car,
      driver,
      loopIndex: 0,
      laps: [],
      offTrackEvents: 0,
      lastPlanTime: -Infinity,
      finished: false,
    };
  });

  // Settle suspension briefly.
  for (let i = 0; i < 20; i++) {
    stepRaycastVehicle(world, cars.map((c) => c.car), { dt: PHYSICS_DT, substeps: 1 });
  }
  // After settle, snap each car to its spawn pose to remove any drift.
  for (const [i, c] of cars.entries()) {
    const spawnZ = course.spawn.z + (i - (cars.length - 1) / 2) * 3;
    c.car.teleport({ x: course.spawn.x, z: spawnZ, heading: course.spawn.heading });
  }

  let simTime = 0;
  const progressEvery = opts.progressEverySec ?? 5;
  let nextProgressAt = progressEvery;
  while (simTime < maxSimTime) {
    // Per-car planning at the replan cadence.
    for (const c of cars) {
      if (c.finished) continue;
      if (simTime - c.lastPlanTime < REPLAN_EVERY_SEC) continue;
      const state = c.body.readState();
      const pick = pickNextWaypoint(state, course.waypoints, c.loopIndex);
      if (pick.advanced) {
        if (pick.nextIndex === 0 && c.loopIndex !== 0) {
          // Completed a lap.
          const prevTime = c.laps.length > 0 ? c.laps[c.laps.length - 1]!.time : 0;
          c.laps.push({ lap: c.laps.length + 1, time: simTime, duration: simTime - prevTime });
          if (c.laps.length >= targetLaps) {
            c.finished = true;
          }
        }
        c.loopIndex = pick.nextIndex;
      }
      const { path, segments } = planThroughWaypoints({
        state,
        waypoints: course.waypoints,
        fromIdx: c.loopIndex,
        count: PLAN_LOOKAHEAD_COUNT,
        lib: c.entry.lib,
        polygons: course.polygons,
        obstacles: course.obstacles,
        totalBudgetMs: opts.replanBudgetMs ?? 80,
      });
      if (path.length >= 2) {
        c.driver.setPlan(path, simTime);
        c.planMisses = 0;
      } else {
        c.planMisses = (c.planMisses ?? 0) + 1;
      }
      c.lastSegments = segments;
      c.lastPlanTime = simTime;
    }
    // Per-car control sampling.
    for (const c of cars) {
      if (c.finished) {
        c.car.applyWheeledControls({ steer: 0, driveForce: 0, brakeForce: cfg.brakeForce });
        continue;
      }
      const state = c.body.readState();
      const cmd: WheeledCarControls = c.driver.sample(state, simTime, PHYSICS_DT);
      c.car.applyWheeledControls(cmd);
    }
    // Advance the world for all cars.
    stepRaycastVehicle(world, cars.map((c) => c.car), { dt: PHYSICS_DT, substeps: 1 });
    simTime += PHYSICS_DT;
    // Off-track watcher — count + reset only on serious failure (out of
    // ground bounds + 20m margin, or NaN). The ground extends ±20 past
    // RACE_BOUNDS, so this only fires when the chassis falls off.
    for (const c of cars) {
      if (c.finished) continue;
      const s = c.body.readState();
      const x0Margin = RACE_BOUNDS.x0 - 15;
      const x1Margin = RACE_BOUNDS.x1 + 15;
      const z0Margin = RACE_BOUNDS.z0 - 15;
      const z1Margin = RACE_BOUNDS.z1 + 15;
      const offTrack = s.x < x0Margin || s.x > x1Margin
        || s.z < z0Margin || s.z > z1Margin
        || !Number.isFinite(s.x);
      if (offTrack) {
        c.offTrackEvents++;
        c.car.teleport({
          x: course.waypoints[c.loopIndex]!.x,
          z: course.waypoints[c.loopIndex]!.z,
          heading: course.waypoints[c.loopIndex]!.heading,
        });
        c.driver.clearPlan();
      }
    }
    if (cars.every((c) => c.finished)) break;
    if (simTime >= nextProgressAt) {
      const progress = cars.map((c) => {
        const s = c.body.readState();
        return `${c.entry.name}:lap${c.laps.length}/${targetLaps}@wp${c.loopIndex},pos=(${s.x.toFixed(1)},${s.z.toFixed(1)},h=${s.heading.toFixed(2)}),spd=${s.speed.toFixed(1)},pm=${c.planMisses ?? 0}`;
      }).join(' | ');
      opts.onProgress?.(`t=${simTime.toFixed(1)}s ${progress}`);
      nextProgressAt += progressEvery;
    }
  }

  // Build results.
  const results: RaceResult[] = cars.map((c) => {
    const durations = c.laps.map((l) => l.duration);
    const best = durations.length > 0 ? Math.min(...durations) : NaN;
    const avg = durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : NaN;
    return {
      name: c.entry.name,
      laps: c.laps,
      best,
      avg,
      totalSimTime: simTime,
      finished: c.laps.length >= targetLaps,
      offTrackEvents: c.offTrackEvents,
    };
  });
  return results;
}

/** Build a kinematic-baseline `RaceEntry`. */
export function kinematicEntry(name = 'kinematic'): RaceEntry {
  return { name, lib: buildKinematicLibrary() };
}

/** Build a v2 `RaceEntry` from a `LearnedVehicleModel`. */
export function v2Entry(name: string, model: LearnedVehicleModel): RaceEntry {
  return { name, lib: buildLearnedRaceLibraryV2(model) };
}

/** Build a parametric-only baseline (no residual ensemble) from the
 *  default params + config. */
export function parametricOnlyEntry(name = 'parametric-only'): RaceEntry {
  const m = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2);
  return { name, lib: buildLearnedRaceLibraryV2(m) };
}
