// Rapier raycast-vehicle wrapper + Ackermann pure-pursuit conversion. Rapier
// is an OPTIONAL peer; only consumers importing this subpath need it. The
// wrapper handles the kinocat <-> Rapier heading-sign convention so callers
// can stay in planning coordinates (heading 0 = +X, +heading rotates +X
// toward +Z).
//
// Why a dedicated wrapper? `RAPIER.DynamicRayCastVehicleController` is twitchy
// at 60 Hz on its own and exposes a low-level "set wheel inputs, call
// updateVehicle, then world.step" tick order that's easy to get wrong. This
// module bakes the canonical order into `applyControls` and exposes a
// `CarHandle` with the same shape every kinocat demo wants:
//   { id, chassis, vehicle, readState(now), applyControls, teleport, dispose }
// Game-specific tuning (chassis mass, engine force, suspension constants) is
// fully overridable via `RaycastVehicleOptions`.

import RAPIER from '@dimforge/rapier3d-compat';
import type { VehicleState } from '../../agent/types';
import type { PurePursuitConfig } from '../../execute/types';
import { purePursuit } from '../../execute/pure-pursuit';
import { wrapAngle } from '../../internal/math';

let rapierReady: Promise<typeof RAPIER> | null = null;

/** Initialize the Rapier WASM module. Safe to call from concurrent mounts;
 *  the underlying `RAPIER.init()` is idempotent. */
export function ensureRapier(): Promise<typeof RAPIER> {
  if (!rapierReady) rapierReady = RAPIER.init().then(() => RAPIER);
  return rapierReady;
}

// ---------------------------------------------------------------------------
// CarHandle — uniform shape every demo wants from a raycast vehicle.

export interface CarHandle {
  id: string;
  chassis: RAPIER.RigidBody;
  vehicle: RAPIER.DynamicRayCastVehicleController;
  /** Read the chassis pose/speed back as a planner-shaped state. */
  readState(now: number): VehicleState;
  /** Drive the raycast vehicle for the upcoming physics tick. Caller is
   *  responsible for calling `vehicle.updateVehicle(dt)` then `world.step()`. */
  applyControls(c: { steer: number; throttle: number; brake: number }): void;
  /** Hard reset to a pose (e.g. on /R). */
  teleport(pose: { x: number; z: number; heading: number }): void;
  /** Like {@link teleport} but with a forward speed (m/s) imparted along the
   *  new heading. Angular velocity is zeroed. Used by the autonomous
   *  motion-primitive learner to start trials at non-zero speed. */
  teleportWithSpeed(
    pose: { x: number; z: number; heading: number },
    speed: number,
  ): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Options. Defaults match a sport-car-ish chassis (~580 kg). All knobs are
// overridable; pass `{}` to take every default.

export type DriveTrain = 'rwd' | 'fwd' | 'awd';

export interface RaycastVehicleOptions {
  /** Diagnostic id; not used by Rapier. */
  id: string;
  position: { x: number; y?: number; z: number };
  heading: number;
  /** Chassis cuboid half-extents (world units). Default = sport-car ballpark. */
  chassisHalf?: { x: number; y: number; z: number };
  chassisDensity?: number;
  chassisFriction?: number;
  chassisRestitution?: number;
  linearDamping?: number;
  angularDamping?: number;
  /** Wheel layout: distance from chassis centre along the forward axis. */
  wheelBase?: number;
  /** Wheel layout: distance from chassis centre along the lateral axis. */
  wheelTrack?: number;
  wheelRadius?: number;
  suspensionRestLength?: number;
  suspensionMaxTravel?: number;
  suspensionStiffness?: number;
  suspensionCompression?: number;
  suspensionRelaxation?: number;
  suspensionMaxForce?: number;
  frictionSlip?: number;
  sideFrictionStiffness?: number;
  /** Max engine force on the driven wheels (N). */
  engineForce?: number;
  brakeForce?: number;
  /** Max |steer angle| at the front wheels (radians). */
  maxSteerAngle?: number;
  driveTrain?: DriveTrain;
}

const DEFAULTS = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  chassisFriction: 0.4,
  chassisRestitution: 0.1,
  linearDamping: 0.1,
  angularDamping: 0.5,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  suspensionMaxTravel: 0.2,
  suspensionStiffness: 80,
  suspensionCompression: 0.83,
  suspensionRelaxation: 20,
  suspensionMaxForce: 12000,
  frictionSlip: 1.8,
  sideFrictionStiffness: 1.0,
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd' as DriveTrain,
};

function withDefaults(opts: RaycastVehicleOptions) {
  return {
    chassisHalf: opts.chassisHalf ?? DEFAULTS.chassisHalf,
    chassisDensity: opts.chassisDensity ?? DEFAULTS.chassisDensity,
    chassisFriction: opts.chassisFriction ?? DEFAULTS.chassisFriction,
    chassisRestitution: opts.chassisRestitution ?? DEFAULTS.chassisRestitution,
    linearDamping: opts.linearDamping ?? DEFAULTS.linearDamping,
    angularDamping: opts.angularDamping ?? DEFAULTS.angularDamping,
    wheelBase: opts.wheelBase ?? DEFAULTS.wheelBase,
    wheelTrack: opts.wheelTrack ?? DEFAULTS.wheelTrack,
    wheelRadius: opts.wheelRadius ?? DEFAULTS.wheelRadius,
    suspensionRestLength: opts.suspensionRestLength ?? DEFAULTS.suspensionRestLength,
    suspensionMaxTravel: opts.suspensionMaxTravel ?? DEFAULTS.suspensionMaxTravel,
    suspensionStiffness: opts.suspensionStiffness ?? DEFAULTS.suspensionStiffness,
    suspensionCompression: opts.suspensionCompression ?? DEFAULTS.suspensionCompression,
    suspensionRelaxation: opts.suspensionRelaxation ?? DEFAULTS.suspensionRelaxation,
    suspensionMaxForce: opts.suspensionMaxForce ?? DEFAULTS.suspensionMaxForce,
    frictionSlip: opts.frictionSlip ?? DEFAULTS.frictionSlip,
    sideFrictionStiffness: opts.sideFrictionStiffness ?? DEFAULTS.sideFrictionStiffness,
    engineForce: opts.engineForce ?? DEFAULTS.engineForce,
    brakeForce: opts.brakeForce ?? DEFAULTS.brakeForce,
    maxSteerAngle: opts.maxSteerAngle ?? DEFAULTS.maxSteerAngle,
    driveTrain: opts.driveTrain ?? DEFAULTS.driveTrain,
  };
}

// ---------------------------------------------------------------------------
// Vehicle factory.

/** Spawn a chassis + raycast vehicle in `world`. Wheels are attached in a
 *  standard 4-corner layout; the two front wheels (indices 0,1) steer. The
 *  rear wheels (indices 2,3) take engine torque by default (RWD), which is
 *  the most numerically stable choice for the raycast solver. */
export function createRaycastVehicle(
  world: RAPIER.World,
  opts: RaycastVehicleOptions,
): CarHandle {
  const D = withDefaults(opts);

  // Spawn the chassis high enough that a fully-extended suspension places the
  // wheels at the ground plane (y=0). Hub is at chassis local y=0, wheels
  // hang `wheelRadius + suspensionRestLength` below.
  const cy = opts.position.y ?? D.wheelRadius + D.suspensionRestLength + 0.05;
  const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(opts.position.x, cy, opts.position.z)
    .setRotation(yawQuat(opts.heading))
    .setLinearDamping(D.linearDamping)
    .setAngularDamping(D.angularDamping)
    .setCanSleep(false);
  const chassis = world.createRigidBody(bodyDesc);
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(D.chassisHalf.x, D.chassisHalf.y, D.chassisHalf.z)
      .setDensity(D.chassisDensity)
      .setFriction(D.chassisFriction)
      .setRestitution(D.chassisRestitution),
    chassis,
  );

  const vehicle = world.createVehicleController(chassis);
  vehicle.indexUpAxis = 1; // Y up
  // Rapier.js exposes the forward-axis as a (mis-named) setter:
  //   set setIndexForwardAxis(axis: number)
  // — so assignment is the only path. See
  // https://github.com/dimforge/rapier.js/blob/master/src.ts/control/ray_cast_vehicle_controller.ts
  (vehicle as unknown as { setIndexForwardAxis: number }).setIndexForwardAxis = 0; // +X forward

  const dir = { x: 0, y: -1, z: 0 };
  const axle = { x: 0, y: 0, z: 1 };
  const wheels: Array<[number, number]> = [
    [+D.wheelBase, -D.wheelTrack], // 0: front-right (steered)
    [+D.wheelBase, +D.wheelTrack], // 1: front-left  (steered)
    [-D.wheelBase, -D.wheelTrack], // 2: rear-right
    [-D.wheelBase, +D.wheelTrack], // 3: rear-left
  ];
  for (const [fx, fz] of wheels) {
    vehicle.addWheel(
      { x: fx, y: 0, z: fz },
      dir,
      axle,
      D.suspensionRestLength,
      D.wheelRadius,
    );
  }
  for (let i = 0; i < 4; i++) {
    vehicle.setWheelSuspensionStiffness(i, D.suspensionStiffness);
    vehicle.setWheelMaxSuspensionForce(i, D.suspensionMaxForce);
    vehicle.setWheelSuspensionCompression(i, D.suspensionCompression);
    vehicle.setWheelSuspensionRelaxation(i, D.suspensionRelaxation);
    vehicle.setWheelMaxSuspensionTravel(i, D.suspensionMaxTravel);
    vehicle.setWheelFrictionSlip(i, D.frictionSlip);
    vehicle.setWheelSideFrictionStiffness(i, D.sideFrictionStiffness);
  }

  function readState(now: number): VehicleState {
    const t = chassis.translation();
    const q = chassis.rotation();
    const lin = chassis.linvel();
    const heading = yawFromQuat(q);
    // Signed forward speed: project linear velocity onto the chassis forward.
    const fwd = { x: Math.cos(heading), z: Math.sin(heading) };
    const speed = lin.x * fwd.x + lin.z * fwd.z;
    return { x: t.x, z: t.z, heading, speed, t: now };
  }

  // Which wheel indices take engine torque, per drive-train choice.
  const driveIdx = D.driveTrain === 'fwd' ? [0, 1] : D.driveTrain === 'awd' ? [0, 1, 2, 3] : [2, 3];

  function applyControls(c: { steer: number; throttle: number; brake: number }) {
    const steer = clamp(c.steer, -D.maxSteerAngle, D.maxSteerAngle);
    const throttle = clamp(c.throttle, -1, 1);
    const brake = clamp(c.brake, 0, 1);
    // Front-wheel steering (indices 0,1).
    vehicle.setWheelSteering(0, steer);
    vehicle.setWheelSteering(1, steer);
    const engineForce = throttle * D.engineForce;
    for (let i = 0; i < 4; i++) {
      vehicle.setWheelEngineForce(i, driveIdx.includes(i) ? engineForce : 0);
    }
    const brakeForce = brake * D.brakeForce;
    for (let i = 0; i < 4; i++) vehicle.setWheelBrake(i, brakeForce);
  }

  function teleport(pose: { x: number; z: number; heading: number }) {
    chassis.setTranslation({ x: pose.x, y: cy, z: pose.z }, true);
    chassis.setRotation(yawQuat(pose.heading), true);
    chassis.setLinvel({ x: 0, y: 0, z: 0 }, true);
    chassis.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  function teleportWithSpeed(
    pose: { x: number; z: number; heading: number },
    speed: number,
  ) {
    chassis.setTranslation({ x: pose.x, y: cy, z: pose.z }, true);
    chassis.setRotation(yawQuat(pose.heading), true);
    chassis.setLinvel(
      { x: speed * Math.cos(pose.heading), y: 0, z: speed * Math.sin(pose.heading) },
      true,
    );
    chassis.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  function dispose() {
    world.removeVehicleController(vehicle);
    world.removeRigidBody(chassis);
  }

  return {
    id: opts.id,
    chassis,
    vehicle,
    readState,
    applyControls,
    teleport,
    teleportWithSpeed,
    dispose,
  };
}

// ---------------------------------------------------------------------------
// Pure-pursuit → Rapier controls.

export interface AckermannConfig extends PurePursuitConfig {
  /** Distance between front and rear axles (world units). Equals
   *  `2 * wheelBase` for a symmetric chassis. */
  wheelBase: number;
}

export interface AckermannCommand {
  steer: number;
  throttle: number;
  brake: number;
  atGoal: boolean;
  lookahead: { x: number; z: number };
}

/** Convert a kinocat plan tail to (steer, throttle, brake) for Rapier.
 *  `state` is the current chassis state read back from physics. */
export function planToAckermannControls(
  state: VehicleState,
  path: VehicleState[],
  cfg: AckermannConfig,
): AckermannCommand {
  const cmd = purePursuit(state, path, cfg);
  // purePursuit returns curvature (1/radius). Ackermann steering:
  //   steer = atan(curvature * wheelBase)
  const steer = Math.atan(cmd.steering * cfg.wheelBase);
  return {
    // kinocat curvature → Rapier wheel angle. kinocat sign convention has
    // +curvature rotate +X toward +Z; Rapier's +yaw (right-hand about +Y)
    // rotates +X toward -Z. Negate at the boundary so positive plan curvature
    // produces the matching physical yaw rate.
    steer: -steer,
    throttle: cmd.throttle,
    brake: cmd.brake,
    atGoal: cmd.atGoal,
    lookahead: cmd.lookahead,
  };
}

// ---------------------------------------------------------------------------
// Static collider sugar — keeps callers free of RAPIER.ColliderDesc verbosity.

export interface BoxColliderOptions {
  x: number;
  y: number;
  z: number;
  /** Half-extents (world units). */
  hx: number;
  hy: number;
  hz: number;
  friction?: number;
}

/** Add a fixed cuboid collider to `world`. Returns the Rapier collider so
 *  callers can keep handles for cleanup or queries. */
export function createBoxCollider(
  world: RAPIER.World,
  opts: BoxColliderOptions,
): RAPIER.Collider {
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed().setTranslation(opts.x, opts.y, opts.z),
  );
  return world.createCollider(
    RAPIER.ColliderDesc.cuboid(opts.hx, opts.hy, opts.hz).setFriction(
      opts.friction ?? 1.0,
    ),
    body,
  );
}

export interface GroundColliderOptions {
  /** Drivable rectangle on the XZ plane. The collider extends `pad` past it. */
  bounds: { x0: number; x1: number; z0: number; z1: number };
  /** Slab thickness centred at y=-thickness/2 so its top sits at y=0. */
  thickness?: number;
  /** Extra horizontal padding so the agent can't drive off the slab edge. */
  pad?: number;
  friction?: number;
}

/** A flat ground slab whose top face is the y=0 planning plane. Use this for
 *  arenas; for terrain pass a heightfield collider instead. */
export function createGroundCollider(
  world: RAPIER.World,
  opts: GroundColliderOptions,
): RAPIER.Collider {
  const thickness = opts.thickness ?? 1.0;
  const pad = opts.pad ?? 20;
  const halfX = (opts.bounds.x1 - opts.bounds.x0) / 2 + pad;
  const halfZ = (opts.bounds.z1 - opts.bounds.z0) / 2 + pad;
  const cx = (opts.bounds.x0 + opts.bounds.x1) / 2;
  const cz = (opts.bounds.z0 + opts.bounds.z1) / 2;
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed().setTranslation(cx, -thickness / 2, cz),
  );
  return world.createCollider(
    RAPIER.ColliderDesc.cuboid(halfX, thickness / 2, halfZ).setFriction(
      opts.friction ?? 1.5,
    ),
    body,
  );
}

// ---------------------------------------------------------------------------
// Heightfield collider — same sampler can feed nav generation downstream.

export type HeightSampler = (x: number, z: number) => number;

export interface HeightfieldColliderOptions {
  sampler: HeightSampler;
  /** Drivable rectangle on the XZ plane. */
  bounds: { x0: number; x1: number; z0: number; z1: number };
  /** Sample spacing in world units; finer = closer to the analytic surface. */
  cellSize?: number;
  friction?: number;
}

/** Build a Rapier heightfield collider from a continuous `sampler(x, z)`. The
 *  same sampler can feed `kinocat/adapters/navcat` for slope-aware navmesh
 *  generation so physics and planner see the same terrain. */
export function createHeightfieldCollider(
  world: RAPIER.World,
  opts: HeightfieldColliderOptions,
): RAPIER.Collider {
  const cellSize = opts.cellSize ?? 2;
  const widthW = opts.bounds.x1 - opts.bounds.x0;
  const depthW = opts.bounds.z1 - opts.bounds.z0;
  const nrows = Math.max(1, Math.round(widthW / cellSize));
  const ncols = Math.max(1, Math.round(depthW / cellSize));
  // Rapier heightfield: heights buffer is (nrows+1)*(ncols+1) row-major,
  // scales (sx, sy, sz) are the world extents along each axis.
  const heights = new Float32Array((nrows + 1) * (ncols + 1));
  for (let j = 0; j <= ncols; j++) {
    for (let i = 0; i <= nrows; i++) {
      const u = i / nrows;
      const v = j / ncols;
      const x = opts.bounds.x0 + u * widthW;
      const z = opts.bounds.z0 + v * depthW;
      heights[j * (nrows + 1) + i] = opts.sampler(x, z);
    }
  }
  const cx = (opts.bounds.x0 + opts.bounds.x1) / 2;
  const cz = (opts.bounds.z0 + opts.bounds.z1) / 2;
  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed().setTranslation(cx, 0, cz),
  );
  return world.createCollider(
    RAPIER.ColliderDesc.heightfield(
      nrows,
      ncols,
      heights,
      { x: widthW, y: 1, z: depthW },
    ).setFriction(opts.friction ?? 1.5),
    body,
  );
}

// ---------------------------------------------------------------------------
// Internals — the kinocat<->Rapier heading sign flip.

// kinocat heading convention: heading 0 = +X, +heading rotates +X toward +Z
// (i.e. vx = cos h, vz = sin h). Rapier yaw is the standard right-hand
// rotation about +Y, which sends +X toward -Z. The two are sign-flipped, so
// every kinocat→Rapier orientation conversion goes through `-h`.
function yawQuat(h: number): { x: number; y: number; z: number; w: number } {
  const half = -h / 2;
  return { x: 0, y: Math.sin(half), z: 0, w: Math.cos(half) };
}

function yawFromQuat(q: { x: number; y: number; z: number; w: number }): number {
  const rapierYaw = Math.atan2(
    2 * (q.w * q.y + q.x * q.z),
    1 - 2 * (q.y * q.y + q.z * q.z),
  );
  return wrapAngle(-rapierYaw);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
