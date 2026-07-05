// Headless obstacle-course runtime — the single source of truth for how the
// obstacle-course drives, shared by the /obstaclecourse web page and the
// Vitest tests. The React component becomes a thin renderer that builds this,
// calls tick()/status() each frame, and draws the result.
//
// Two things this fixes versus the old in-component loop:
//   1. Determinism. The page replanned on a wall-clock setInterval and timed
//      the planner by wall-clock deadlineMs, and trimmed the plan by
//      performance.now() — all non-deterministic. Here replanning is driven by
//      SIM time, the plan is trimmed by SIM time, and the planner is bounded by
//      maxExpansions (deadline Infinity), so the same inputs reproduce exactly.
//   2. NO teleportation. There is no stall-guard / off-track rescue. A car that
//      gets stuck or leaves the course just fails — the run is judged by
//      whether it actually made progress, never by a snap back onto the track.

import RAPIER from '@dimforge/rapier3d-compat';
import {
  InMemoryNavWorld,
  rampHeightSampler,
  combineHeightSamplers,
} from 'kinocat/environment';
import type { HeightSampler, NavWorld } from 'kinocat/environment';
import type { CarKinematicState } from 'kinocat/agent';
import {
  ensureRapier,
  createRaycastVehicle,
  createGroundCollider,
  createBoxCollider,
  createHeightfieldCollider,
  planToAckermannControls,
  stepRaycastVehicle,
  type CarHandle,
} from 'kinocat/adapters/rapier';
import {
  trimPlan,
  wheeledFromNormalized,
  ZERO_WHEELED,
  type WheeledCarControls,
  type CarForceTuning,
} from 'kinocat/vehicle/car';
import {
  OBS_AGENT,
  OBS_BOUNDS,
  OBS_BLOCKS_ALL,
  OBS_MAX_EXPANSIONS,
  buildObstacleCourse,
  obsPickWaypoint,
  obsSpawn,
  planObstacleCourse,
  type ObstacleCourse,
  type ObsBlocks,
} from './obstaclecourse-scenarios';

export const PHYSICS_DT = 1 / 60;
export const VEHICLE_SUBSTEPS = 4;
export const OBS_REPLAN_INTERVAL_MS = 120;
const WHEEL_BASE = 1.6;

export const OBSTACLE_FORCE_TUNING: CarForceTuning = {
  engineForceN: 4000,
  brakeForceN: 2000,
};

/** Gentle terrain so a flat-ground vehicle still copes — the SAME sampler the
 *  page uses to build its heightfield mesh, so physics and visuals agree. */
export function obstacleTerrainSampler(x: number, z: number): number {
  return 0.6 * Math.sin(x / 18) + 0.6 * Math.cos(z / 14);
}

/** Combined drivable-surface sampler for a course (terrain ⊕ ramps), or null
 *  when the course is flat ground. Exposed so the page's mesh matches physics. */
export function obstacleHeightSampler(course: ObstacleCourse): HeightSampler | null {
  const samplers: HeightSampler[] = [];
  if (course.blocks.heightfield) samplers.push(obstacleTerrainSampler);
  if (course.ramps.length > 0) samplers.push(rampHeightSampler(course.ramps));
  return samplers.length > 0 ? combineHeightSamplers(...samplers) : null;
}

/** MonitorSample-compatible per-car status (plus goal, for rendering). */
export interface ObsCarStatus {
  state: CarKinematicState;
  metrics: {
    liveControls: { steer: number; throttle: number; brake: number; targetSpeed: number };
  };
  diagnostics: {
    totalReplans: number;
    successfulReplans: number;
    consecutiveFailedReplans: number;
    lastExpansions: number;
  };
  plan: CarKinematicState[] | null;
  loopIndex: number;
  goal: CarKinematicState | null;
}

export interface ObstacleCourseScenario {
  /** Advance one fixed tick. Pass `controls` to drive manually (player mode);
   *  omit to let the planner+controller drive. */
  tick(controls?: WheeledCarControls | null): void;
  status(): ObsCarStatus;
  simTime(): number;
  /** Current course (geometry) — for the page's visuals. */
  course: ObstacleCourse;
  /** Drivable-surface sampler matching the physics heightfield (or null). */
  heightSampler: HeightSampler | null;
  navWorld: NavWorld;
  getWorld(): RAPIER.World;
  getCar(): CarHandle;
  /** Rebuild the course + physics for new block toggles (page HUD). */
  rebuild(blocks: ObsBlocks): void;
  /** Reset car to spawn and clear the plan (start-of-run reset, not a rescue). */
  reset(): void;
  dispose(): void;
}

export interface ObstacleCourseScenarioOptions {
  blocks?: ObsBlocks;
}

export async function createObstacleCourseScenario(
  opts: ObstacleCourseScenarioOptions = {},
): Promise<ObstacleCourseScenario> {
  await ensureRapier();
  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  let blocks = opts.blocks ?? OBS_BLOCKS_ALL;
  let course = buildObstacleCourse(blocks);
  let navWorld: NavWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
  let heightSampler = obstacleHeightSampler(course);
  let coursePhysics: RAPIER.Collider[] = [];

  function buildPhysics(): void {
    for (const col of coursePhysics) {
      const body = col.parent();
      world.removeCollider(col, true);
      if (body) world.removeRigidBody(body);
    }
    coursePhysics = [];
    if (heightSampler) {
      coursePhysics.push(
        createHeightfieldCollider(world, { sampler: heightSampler, bounds: OBS_BOUNDS, cellSize: 2 }),
      );
    } else {
      coursePhysics.push(createGroundCollider(world, { bounds: OBS_BOUNDS, pad: 20 }));
    }
    for (const b of course.buildings) {
      coursePhysics.push(
        createBoxCollider(world, {
          x: b.x,
          y: b.height / 2,
          z: b.z,
          hx: b.hx,
          hy: b.height / 2,
          hz: b.hz,
        }),
      );
    }
  }
  buildPhysics();

  const car = createRaycastVehicle(world, {
    id: 'obs-car',
    position: { x: obsSpawn().x, z: obsSpawn().z },
    heading: obsSpawn().heading,
  });

  let simTime = 0;
  let plan: CarKinematicState[] | null = null;
  let planStartSimTime = 0;
  let loopIndex = 0;
  let goal: CarKinematicState | null = null;
  let lastReplanSimTime = -Infinity;
  let totalReplans = 0;
  let successfulReplans = 0;
  let consecutiveFailedReplans = 0;
  let lastExpansions = 0;
  let liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };

  function replan(): void {
    const state = car.readState(simTime);
    const pick = obsPickWaypoint(state, course, loopIndex, navWorld);
    loopIndex = pick.nextIndex;
    goal = pick.goal;
    // deadline Infinity ⇒ the planner is bounded only by expansions, so the
    // result is independent of machine speed (deterministic).
    const res = planObstacleCourse({
      state: { ...state, t: 0 },
      goal: { ...pick.goal, t: 0 },
      course,
      world: navWorld,
      deadlineMs: Number.POSITIVE_INFINITY,
      maxExpansions: OBS_MAX_EXPANSIONS,
    });
    totalReplans++;
    lastExpansions = res.stats.expansions;
    if (res.found && res.path.length > 1) {
      plan = res.path;
      planStartSimTime = simTime;
      successfulReplans++;
      consecutiveFailedReplans = 0;
    } else {
      consecutiveFailedReplans++;
    }
  }

  function tick(controls?: WheeledCarControls | null): void {
    if (controls) {
      // Manual / player drive: no planner.
      car.applyWheeledControls(controls);
      liveControls = {
        steer: 0,
        throttle: controls.driveForce / OBSTACLE_FORCE_TUNING.engineForceN,
        brake: controls.brakeForce / OBSTACLE_FORCE_TUNING.brakeForceN,
        targetSpeed: 0,
      };
    } else {
      // Replan on a SIM-time cadence (not wall clock).
      if ((simTime - lastReplanSimTime) * 1000 >= OBS_REPLAN_INTERVAL_MS) {
        replan();
        lastReplanSimTime = simTime;
      }
      const state = car.readState(simTime);
      if (plan && plan.length > 1) {
        const live = trimPlan(plan, simTime - planStartSimTime);
        if (live.length >= 2) {
          const cmd = planToAckermannControls(state, live, {
            wheelBase: 2 * WHEEL_BASE,
            lookaheadMin: 3,
            lookaheadGain: 0.45,
            lookaheadMax: 14,
            maxLateralAccel: 8,
            maxAccel: 6,
            maxDecel: 8,
            cruiseSpeed: OBS_AGENT.maxSpeed,
            goalTolerance: 2,
            minTurnRadius: OBS_AGENT.minTurnRadius,
          });
          car.applyWheeledControls(wheeledFromNormalized(cmd, OBSTACLE_FORCE_TUNING));
          liveControls = {
            steer: cmd.steer,
            throttle: cmd.throttle,
            brake: cmd.brake,
            targetSpeed: OBS_AGENT.maxSpeed,
          };
        } else {
          car.applyWheeledControls(
            wheeledFromNormalized({ steer: 0, throttle: 0.2, brake: 0 }, OBSTACLE_FORCE_TUNING),
          );
          liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 0 };
        }
      } else {
        // No plan — coast in neutral. NO teleport rescue.
        car.applyWheeledControls(ZERO_WHEELED);
        liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };
      }
    }
    stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: VEHICLE_SUBSTEPS });
    simTime += PHYSICS_DT;
  }

  function status(): ObsCarStatus {
    return {
      state: car.readState(simTime),
      metrics: { liveControls },
      diagnostics: {
        totalReplans,
        successfulReplans,
        consecutiveFailedReplans,
        lastExpansions,
      },
      plan,
      loopIndex,
      goal,
    };
  }

  function rebuild(nextBlocks: ObsBlocks): void {
    blocks = nextBlocks;
    course = buildObstacleCourse(blocks);
    navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
    heightSampler = obstacleHeightSampler(course);
    buildPhysics();
    plan = null;
    goal = null;
  }

  function reset(): void {
    const sp = obsSpawn();
    car.teleport({ x: sp.x, z: sp.z, heading: sp.heading });
    simTime = 0;
    plan = null;
    goal = null;
    loopIndex = 0;
    lastReplanSimTime = -Infinity;
    totalReplans = 0;
    successfulReplans = 0;
    consecutiveFailedReplans = 0;
    lastExpansions = 0;
    liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };
  }

  return {
    tick,
    status,
    simTime: () => simTime,
    get course() {
      return course;
    },
    get heightSampler() {
      return heightSampler;
    },
    get navWorld() {
      return navWorld;
    },
    getWorld: () => world,
    getCar: () => car,
    rebuild,
    reset,
    dispose() {
      car.dispose();
      world.free();
    },
  };
}
