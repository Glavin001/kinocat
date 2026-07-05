// Car-chase Rapier wiring — thin wrapper around `kinocat/adapters/rapier`
// that pins the chassis tuning to the car-chase course (sport-sedan-ish:
// ~580 kg, 4 kN engine, 60 Hz raycast vehicle stable under sub-stepping).
// Everything else (world creation, raycast vehicle, pure-pursuit conversion)
// lives in core.
import RAPIER from '@dimforge/rapier3d-compat';
import type { CarKinematicState } from 'kinocat/agent';
import {
  createRaycastVehicle,
  createBoxCollider,
  createGroundCollider,
  createHeightfieldCollider,
  ensureRapier as coreEnsureRapier,
  planToAckermannControls,
  type CarHandle as CoreCarHandle,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import { rampHeightSampler } from 'kinocat/environment';
import type {
  CarChaseCourse,
  BuildingSpec,
  JumpSpec,
} from '../lib/carchase-scenarios';
import { CARCHASE_AGENT, CARCHASE_BOUNDS } from '../lib/carchase-scenarios';

export const ensureRapier = coreEnsureRapier;

// Wheel-base used by the car-chase tuning; pure-pursuit needs `2*WHEEL_BASE`.
const WHEEL_BASE = 1.6;

// Force constants exported so call sites can build canonical
// `WheeledCarControls` without hard-coding magic numbers.
export const CARCHASE_ENGINE_FORCE_N = 4000;
export const CARCHASE_BRAKE_FORCE_N = 2000;
export const CARCHASE_MAX_STEER_RAD = 0.6;

const CARCHASE_VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  suspensionMaxTravel: 0.2,
  engineForce: CARCHASE_ENGINE_FORCE_N,
  brakeForce: CARCHASE_BRAKE_FORCE_N,
  maxSteerAngle: CARCHASE_MAX_STEER_RAD,
  driveTrain: 'rwd',
};

export type CarHandle = CoreCarHandle;

export interface CarChaseWorld {
  world: RAPIER.World;
  staticHandles: number[];
}

const GRAVITY = { x: 0, y: -9.81, z: 0 };

/** Build the static physics world for the car-chase course: flat ground
 *  slab everywhere + a small per-ramp heightfield collider in the ramp's
 *  immediate footprint + cuboid building colliders.
 *
 *  Why per-ramp heightfields instead of one over the whole course: a
 *  120x90 heightfield (cellSize 2) feeding four sub-stepped raycast
 *  vehicles slows the chassis acceleration to a crawl on this map —
 *  every wheel raycast walks a much bigger BVH each substep. Keeping
 *  the heightfield local (≈20x20 m per ramp) lets the rest of the map
 *  stay on the cheap flat slab the carchase tuning was designed for. */
export function createCarChaseWorld(course: CarChaseCourse): CarChaseWorld {
  const world = new RAPIER.World(GRAVITY);
  const staticHandles: number[] = [];

  staticHandles.push(
    createGroundCollider(world, { bounds: CARCHASE_BOUNDS, pad: 20, friction: 1.5 }).handle,
  );

  // Perimeter walls flush with CARCHASE_BOUNDS. The ground slab extends 20 m
  // past the bounds (pad) with nothing to stop a car, so a vehicle that drifts
  // to the edge under execution used to roll straight off and fall (triggering
  // an auto-reset teleport). These invisible walls sit just OUTSIDE the bounds
  // — the planner already keeps every plan's footprint inside the bounds with
  // clearance, so valid plans never touch them; they only catch overshoot.
  {
    const b = CARCHASE_BOUNDS;
    const t = 1; // wall half-thickness
    const h = 3; // wall half-height
    const midX = (b.x0 + b.x1) / 2;
    const midZ = (b.z0 + b.z1) / 2;
    const halfX = (b.x1 - b.x0) / 2;
    const halfZ = (b.z1 - b.z0) / 2;
    const walls: Array<{ x: number; z: number; hx: number; hz: number }> = [
      { x: b.x0 - t, z: midZ, hx: t, hz: halfZ + 2 * t }, // west
      { x: b.x1 + t, z: midZ, hx: t, hz: halfZ + 2 * t }, // east
      { x: midX, z: b.z0 - t, hx: halfX + 2 * t, hz: t }, // south
      { x: midX, z: b.z1 + t, hx: halfX + 2 * t, hz: t }, // north
    ];
    for (const w of walls) {
      staticHandles.push(
        createBoxCollider(world, { x: w.x, y: h, z: w.z, hx: w.hx, hy: h, hz: w.hz }).handle,
      );
    }
  }

  for (const r of course.ramps) {
    // Pad the heightfield ~6 m past the ramp footprint so the lateral
    // skirt / back-slope are fully captured. The pad keeps the
    // sampled-mesh tile cheap (≈15x15 m) so wheel raycasts stay fast.
    const pad = 6;
    const halfL = r.length / 2 + (r.backSkirt ?? 2.5) + pad;
    const halfW = r.width / 2 + (r.lateralSkirt ?? 1.5) + pad;
    const c = Math.cos(r.heading);
    const s = Math.sin(r.heading);
    const cx = r.base.x;
    const cz = r.base.z;
    // World-axis AABB of the rotated ramp footprint + pad.
    const ex = Math.abs(halfL * c) + Math.abs(halfW * s);
    const ez = Math.abs(halfL * s) + Math.abs(halfW * c);
    staticHandles.push(
      createHeightfieldCollider(world, {
        sampler: rampHeightSampler([r]),
        bounds: { x0: cx - ex, x1: cx + ex, z0: cz - ez, z1: cz + ez },
        cellSize: 2,
        friction: 1.5,
      }).handle,
    );
  }

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

  return { world, staticHandles };
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
  state: CarKinematicState,
  path: CarKinematicState[],
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
