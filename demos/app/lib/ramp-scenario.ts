// Headless ramp runtime — the single source of truth for the /ramp demo,
// shared by the web page and the Vitest tests. Same deterministic, teleport-
// free contract as the obstacle-course runtime: sim-time replanning, sim-time
// plan trimming, expansion-bounded planning (deadline Infinity), no rescue.
//
// The ramp scenario has a single goal (reach the pad past the ramp/gap) and a
// BallisticJump affordance the planner may pick over a planner-only "gap"
// obstacle when affordances are enabled.

import RAPIER from '@dimforge/rapier3d-compat';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavWorld } from 'kinocat/environment';
import type { CarKinematicState } from 'kinocat/agent';
import {
  ensureRapier,
  createRaycastVehicle,
  createHeightfieldCollider,
  planToAckermannControls,
  stepRaycastVehicle,
  type CarHandle,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import {
  trimPlan,
  wheeledFromNormalized,
  ZERO_WHEELED,
  type WheeledCarControls,
  type CarForceTuning,
} from 'kinocat/vehicle/car';
import {
  RAMP_AGENT,
  RAMP_BOUNDS,
  RAMP_MAX_EXPANSIONS,
  buildRampCourse,
  planRampDemo,
  rampHeightSampler,
  type RampCourse,
} from './ramp-scenarios';

export const PHYSICS_DT = 1 / 60;
export const VEHICLE_SUBSTEPS = 4;
export const RAMP_REPLAN_INTERVAL_MS = 120;
const WHEEL_BASE = 1.6;
const ENGINE_FORCE_N = 4500;
const BRAKE_FORCE_N = 2000;
const MAX_STEER_RAD = 0.6;

export const RAMP_FORCE_TUNING: CarForceTuning = {
  engineForceN: ENGINE_FORCE_N,
  brakeForceN: BRAKE_FORCE_N,
};

/** Softer chassis than the obstacle course — the ramp lip + landing benefit
 *  from more suspension travel. Shared so page physics == test physics. */
export const RAMP_VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.4,
  suspensionMaxTravel: 0.3,
  engineForce: ENGINE_FORCE_N,
  brakeForce: BRAKE_FORCE_N,
  maxSteerAngle: MAX_STEER_RAD,
  driveTrain: 'rwd',
};

export interface RampCarStatus {
  state: CarKinematicState;
  metrics: {
    liveControls: { steer: number; throttle: number; brake: number; targetSpeed: number };
  };
  diagnostics: {
    totalReplans: number;
    successfulReplans: number;
    consecutiveFailedReplans: number;
    lastExpansions: number;
    /** True when the latest plan took the ballistic-jump affordance. */
    usedAffordance: boolean;
  };
  plan: CarKinematicState[] | null;
  goal: CarKinematicState;
}

export interface RampScenario {
  tick(controls?: WheeledCarControls | null): void;
  status(): RampCarStatus;
  simTime(): number;
  course: RampCourse;
  navWorld: NavWorld;
  getWorld(): RAPIER.World;
  getCar(): CarHandle;
  /** Toggle the jump affordance (forces a replan so the path swaps). */
  setAffordance(on: boolean): void;
  reset(): void;
  dispose(): void;
}

export interface RampScenarioOptions {
  affordance?: boolean;
}

export async function createRampScenario(
  opts: RampScenarioOptions = {},
): Promise<RampScenario> {
  await ensureRapier();
  const course = buildRampCourse();
  const navWorld: NavWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
  const sampler = rampHeightSampler(course.ramps);

  const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
  createHeightfieldCollider(world, { sampler, bounds: RAMP_BOUNDS, cellSize: 2, friction: 1.5 });

  const car = createRaycastVehicle(world, {
    id: 'ramp-car',
    position: { x: course.spawn.x, z: course.spawn.z },
    heading: course.spawn.heading,
    ...RAMP_VEHICLE_TUNING,
  });

  let affordanceOn = opts.affordance ?? true;
  let simTime = 0;
  let plan: CarKinematicState[] | null = null;
  let planStartSimTime = 0;
  let lastReplanSimTime = -Infinity;
  let totalReplans = 0;
  let successfulReplans = 0;
  let consecutiveFailedReplans = 0;
  let lastExpansions = 0;
  let usedAffordance = false;
  let liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };

  function replan(): void {
    const state = car.readState(simTime);
    const res = planRampDemo({
      state: { ...state, t: 0 },
      goal: { ...course.goal, t: 0 },
      course,
      world: navWorld,
      withoutAffordances: !affordanceOn,
      // deadline Infinity ⇒ bounded only by expansions ⇒ machine-independent.
      deadlineMs: Number.POSITIVE_INFINITY,
      maxExpansions: RAMP_MAX_EXPANSIONS,
    });
    totalReplans++;
    lastExpansions = res.stats.expansions;
    if (res.found && res.path.length > 1) {
      plan = res.path;
      planStartSimTime = simTime;
      successfulReplans++;
      consecutiveFailedReplans = 0;
      // An affordance step jumps the planner state by more than any primitive
      // could in a single tick (same heuristic the page used).
      usedAffordance = res.path.some(
        (_, i) =>
          i > 0 &&
          Math.hypot(res.path[i]!.x - res.path[i - 1]!.x, res.path[i]!.z - res.path[i - 1]!.z) > 10,
      );
    } else {
      consecutiveFailedReplans++;
    }
  }

  function tick(controls?: WheeledCarControls | null): void {
    if (controls) {
      car.applyWheeledControls(controls);
      liveControls = {
        steer: 0,
        throttle: controls.driveForce / ENGINE_FORCE_N,
        brake: controls.brakeForce / BRAKE_FORCE_N,
        targetSpeed: 0,
      };
    } else {
      if ((simTime - lastReplanSimTime) * 1000 >= RAMP_REPLAN_INTERVAL_MS) {
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
            cruiseSpeed: RAMP_AGENT.maxSpeed,
            goalTolerance: 2,
            minTurnRadius: RAMP_AGENT.minTurnRadius,
          });
          car.applyWheeledControls(wheeledFromNormalized(cmd, RAMP_FORCE_TUNING));
          liveControls = { steer: cmd.steer, throttle: cmd.throttle, brake: cmd.brake, targetSpeed: RAMP_AGENT.maxSpeed };
        } else {
          car.applyWheeledControls(
            wheeledFromNormalized({ steer: 0, throttle: 0.2, brake: 0 }, RAMP_FORCE_TUNING),
          );
          liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 0 };
        }
      } else {
        car.applyWheeledControls(ZERO_WHEELED);
        liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };
      }
    }
    stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: VEHICLE_SUBSTEPS });
    simTime += PHYSICS_DT;
  }

  function status(): RampCarStatus {
    return {
      state: car.readState(simTime),
      metrics: { liveControls },
      diagnostics: {
        totalReplans,
        successfulReplans,
        consecutiveFailedReplans,
        lastExpansions,
        usedAffordance,
      },
      plan,
      goal: course.goal,
    };
  }

  function setAffordance(on: boolean): void {
    affordanceOn = on;
    plan = null; // force a fresh plan so the path swaps detour ↔ jump
    lastReplanSimTime = -Infinity;
  }

  function reset(): void {
    car.teleport({ x: course.spawn.x, z: course.spawn.z, heading: course.spawn.heading });
    simTime = 0;
    plan = null;
    lastReplanSimTime = -Infinity;
    totalReplans = 0;
    successfulReplans = 0;
    consecutiveFailedReplans = 0;
    lastExpansions = 0;
    usedAffordance = false;
    liveControls = { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 };
  }

  return {
    tick,
    status,
    simTime: () => simTime,
    course,
    navWorld,
    getWorld: () => world,
    getCar: () => car,
    setAffordance,
    reset,
    dispose() {
      car.dispose();
      world.free();
    },
  };
}
