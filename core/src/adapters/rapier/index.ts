// kinocat/adapters/rapier — default ForwardSim wrapper backed by a Rapier
// physics world. Rapier is an OPTIONAL peer; only consumers importing this
// subpath need it. Games with bespoke vehicle physics supply their own
// ForwardSim instead — this is the batteries-included default for
// characterization.

import type { ForwardSim } from '../../primitives/types';
import type { VehicleState } from '../../agent/types';
import { wrapAngle } from '../../internal/math';
import type { RapierBodyLike, RapierQuatLike, RapierWorldLike } from './types';

export type { RapierBodyLike, RapierWorldLike, RapierVec3Like, RapierQuatLike } from './types';

// Raycast vehicle wrapper + collider sugar — RAPIER WASM is required only at
// runtime; the import is direct but tsup keeps `@dimforge/rapier3d-compat`
// external so this subpath stays peer-dep-only.
export {
  ensureRapier,
  createRaycastVehicle,
  deriveLearnableConfig,
  planToAckermannControls,
  createBoxCollider,
  createGroundCollider,
  createHeightfieldCollider,
} from './raycast-vehicle';
export type {
  CarHandle,
  DriveTrain,
  RaycastVehicleOptions,
  AckermannConfig,
  AckermannCommand,
  BoxColliderOptions,
  GroundColliderOptions,
  HeightfieldColliderOptions,
  HeightSampler,
  WheelTelemetry,
} from './raycast-vehicle';

export { stepRaycastVehicle } from './step';
export type { StepRaycastVehicleOptions } from './step';
export { RapierCarBody } from './car-body';
export type { RapierCarBodyOptions } from './car-body';

export { createHeadlessTrialHarness } from './headless-trial';
export type {
  HeadlessTrialHarness,
  HeadlessTrialOptions,
  TrialSpec,
  TrialOutcome,
} from './headless-trial';

export interface RapierForwardSimOptions {
  world: RapierWorldLike;
  body: RapierBodyLike;
  /** Body Y on the planning plane (planning is XZ; Y derived elsewhere). */
  groundY?: number;
}

function yawQuat(h: number): RapierQuatLike {
  return { x: 0, y: Math.sin(h / 2), z: 0, w: Math.cos(h / 2) };
}

function yawFromQuat(q: RapierQuatLike): number {
  return wrapAngle(
    Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z)),
  );
}

/**
 * Build a ForwardSim that drives a Rapier rigid body. `controls` is
 * `[curvature, targetSpeed]`: the body is teleported to the input state, given
 * the corresponding planar linear/angular velocity, stepped for `dt`, and read
 * back. Deterministic for a fixed Rapier build.
 */
export function rapierForwardSim(
  opts: RapierForwardSimOptions,
): ForwardSim<VehicleState> {
  const y = opts.groundY ?? 0;
  const { world, body } = opts;
  return (s: VehicleState, controls: number[], dt: number): VehicleState => {
    const curvature = controls[0] ?? 0;
    const speed = controls[1] ?? 0;
    body.setTranslation({ x: s.x, y, z: s.z }, true);
    body.setRotation(yawQuat(s.heading), true);
    body.setLinvel(
      { x: speed * Math.cos(s.heading), y: 0, z: speed * Math.sin(s.heading) },
      true,
    );
    body.setAngvel({ x: 0, y: speed * curvature, z: 0 }, true);
    world.timestep = dt;
    world.step();
    const t = body.translation();
    return {
      x: t.x,
      z: t.z,
      heading: yawFromQuat(body.rotation()),
      speed,
      t: s.t + dt,
    };
  };
}
