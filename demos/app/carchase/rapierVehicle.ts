// Car-chase Rapier wiring — thin wrapper around `kinocat/adapters/rapier`
// that pins the chassis tuning to the car-chase course (sport-sedan-ish:
// ~580 kg, 4 kN engine, 60 Hz raycast vehicle stable under sub-stepping).
// Everything else (world creation, raycast vehicle, pure-pursuit conversion)
// lives in core.
import RAPIER from '@dimforge/rapier3d-compat';
import type { VehicleState } from 'kinocat/agent';
import {
  createRaycastVehicle,
  createBoxCollider,
  createGroundCollider,
  ensureRapier as coreEnsureRapier,
  planToAckermannControls,
  type CarHandle as CoreCarHandle,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import type {
  CarChaseCourse,
  BuildingSpec,
  JumpSpec,
} from '../lib/carchase-scenarios';
import { CARCHASE_AGENT, CARCHASE_BOUNDS } from '../lib/carchase-scenarios';

export const ensureRapier = coreEnsureRapier;

// Wheel-base used by the car-chase tuning; pure-pursuit needs `2*WHEEL_BASE`.
const WHEEL_BASE = 1.6;

const CARCHASE_VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  suspensionMaxTravel: 0.2,
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd',
};

export type CarHandle = CoreCarHandle;

export interface CarChaseWorld {
  world: RAPIER.World;
  staticHandles: number[];
  rampBodies: Array<{ spec: JumpSpec; position: { x: number; y: number; z: number } }>;
}

const GRAVITY = { x: 0, y: -9.81, z: 0 };

/** Build the static physics world for the car-chase course (flat ground,
 *  building + ramp cuboids). Pure delegation to the core collider helpers. */
export function createCarChaseWorld(course: CarChaseCourse): CarChaseWorld {
  const world = new RAPIER.World(GRAVITY);
  const staticHandles: number[] = [];

  staticHandles.push(
    createGroundCollider(world, { bounds: CARCHASE_BOUNDS, pad: 20, friction: 1.5 }).handle,
  );

  for (const b of course.buildings) {
    staticHandles.push(
      createBoxCollider(world, {
        x: b.x,
        y: b.height / 2,
        z: b.z,
        hx: b.hx,
        hy: b.height / 2,
        hz: b.hz,
      }).handle,
    );
  }

  const rampBodies: CarChaseWorld['rampBodies'] = [];
  for (const j of course.jumps) {
    staticHandles.push(
      createBoxCollider(world, {
        x: j.launch.x,
        y: j.height / 2,
        z: j.launch.z,
        hx: j.hx,
        hy: j.height / 2,
        hz: j.hz,
      }).handle,
    );
    rampBodies.push({
      spec: j,
      position: { x: j.launch.x, y: j.height / 2, z: j.launch.z },
    });
  }

  return { world, staticHandles, rampBodies };
}

export interface SpawnCarOptions {
  position: { x: number; y?: number; z: number };
  heading: number;
  id: string;
}

/** Spawn a car-chase-tuned raycast vehicle. */
export function spawnCar(world: RAPIER.World, opts: SpawnCarOptions): CarHandle {
  return createRaycastVehicle(world, {
    id: opts.id,
    position: opts.position,
    heading: opts.heading,
    ...CARCHASE_VEHICLE_TUNING,
  });
}

/** Convert a kinocat plan tail to (steer, throttle, brake) for the car-chase
 *  tuning. Wraps the core helper with the chase's pure-pursuit config. */
export function planToControls(
  state: VehicleState,
  path: VehicleState[],
): { steer: number; throttle: number; brake: number; atGoal: boolean; lookahead: { x: number; z: number } } {
  return planToAckermannControls(state, path, {
    wheelBase: 2 * WHEEL_BASE,
    lookaheadMin: 3,
    lookaheadGain: 0.45,
    lookaheadMax: 14,
    maxLateralAccel: 8,
    maxAccel: 6,
    maxDecel: 8,
    cruiseSpeed: CARCHASE_AGENT.maxSpeed,
    goalTolerance: 2,
    minTurnRadius: CARCHASE_AGENT.minTurnRadius,
  });
}

export type { BuildingSpec, JumpSpec, CarChaseCourse };
