// Rapier physics + DynamicRayCastVehicleController wiring for the car-chase
// demo. Everything that touches Rapier lives here so the rest of the demo
// (and the headless scenario module) stays Rapier-agnostic.
//
// The kinocat planner produces a `path: VehicleState[]` from the kinematic
// forward model; this module translates that to per-tick Rapier controls
// (steer / engine force / brake) via `purePursuit`, applies the controls to
// the raycast vehicle, and reads the chassis back as a `VehicleState` so the
// next plan picks up where physics left off.
import RAPIER from '@dimforge/rapier3d-compat';
import type { VehicleState } from 'kinocat/agent';
import type { PurePursuitConfig } from 'kinocat/execute';
import { purePursuit } from 'kinocat/execute';
import type {
  CarChaseCourse,
  BuildingSpec,
  JumpSpec,
} from '../lib/carchase-scenarios';
import { CARCHASE_AGENT, CARCHASE_BOUNDS } from '../lib/carchase-scenarios';

let rapierReady: Promise<typeof RAPIER> | null = null;

/** Initialize the Rapier WASM module. Safe to call from concurrent mounts;
 *  the underlying `RAPIER.init()` is idempotent. */
export function ensureRapier(): Promise<typeof RAPIER> {
  if (!rapierReady) {
    rapierReady = RAPIER.init().then(() => RAPIER);
  }
  return rapierReady;
}

const GRAVITY = { x: 0, y: -9.81, z: 0 };

export interface CarChaseWorld {
  world: RAPIER.World;
  /** Ground / building / ramp collider handles (so we can remove on reset). */
  staticHandles: number[];
  /** Per-jump ramp meshes — used by the renderer to align visuals. */
  rampBodies: Array<{ spec: JumpSpec; position: { x: number; y: number; z: number } }>;
}

/** Build the static physics world for the car-chase course. Ground is a flat
 *  cuboid (a heightfield would be overkill for this stunt-arena scope);
 *  buildings + jump ramps are cuboid colliders sized from the same specs the
 *  renderer uses. */
export function createCarChaseWorld(course: CarChaseCourse): CarChaseWorld {
  const world = new RAPIER.World(GRAVITY);
  const staticHandles: number[] = [];

  // Ground plane: a very wide thin slab centred at y=-0.5 so its top is y=0
  // (the planning plane). Friction high enough to keep cars stuck.
  {
    const b = CARCHASE_BOUNDS;
    const halfX = (b.x1 - b.x0) / 2 + 20;
    const halfZ = (b.z1 - b.z0) / 2 + 20;
    const cx = (b.x0 + b.x1) / 2;
    const cz = (b.z0 + b.z1) / 2;
    const groundBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(cx, -0.5, cz),
    );
    const groundCollider = world.createCollider(
      RAPIER.ColliderDesc.cuboid(halfX, 0.5, halfZ).setFriction(1.5),
      groundBody,
    );
    staticHandles.push(groundCollider.handle);
  }

  // Building colliders.
  for (const b of course.buildings) {
    const bodyDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(
      b.x,
      b.height / 2,
      b.z,
    );
    const body = world.createRigidBody(bodyDesc);
    const col = world.createCollider(
      RAPIER.ColliderDesc.cuboid(b.hx, b.height / 2, b.hz).setFriction(1.0),
      body,
    );
    staticHandles.push(col.handle);
  }

  // Jump ramps. Modeled as a low cuboid at the launch position — physical
  // launch is mostly via the AI's `applyControls` (the jump affordance edge
  // in the plan teleports the agent on the planner side; the chassis takes
  // some real airtime crossing the ramp + a small Y impulse).
  const rampBodies: CarChaseWorld['rampBodies'] = [];
  for (const j of course.jumps) {
    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(
        j.launch.x,
        j.height / 2,
        j.launch.z,
      ),
    );
    const col = world.createCollider(
      RAPIER.ColliderDesc.cuboid(j.hx, j.height / 2, j.hz).setFriction(1.0),
      body,
    );
    staticHandles.push(col.handle);
    rampBodies.push({
      spec: j,
      position: { x: j.launch.x, y: j.height / 2, z: j.launch.z },
    });
  }

  return { world, staticHandles, rampBodies };
}

// Chassis dimensions match the planner footprint (5 m × 2 m); height 1.0 m.
const CHASSIS_HALF = { x: 2.4, y: 0.5, z: 1.0 };
const WHEEL_RADIUS = 0.4;
const WHEEL_SUSPENSION_REST = 0.6;
const WHEEL_BASE = 1.6; // distance from chassis centre forward/back
const WHEEL_TRACK = 0.85; // distance from chassis centre left/right
// Engine + brake force tuned for the ~10 kg Rapier chassis so it actually
// accelerates from rest under AI throttle ≈ 0.2 (low) up to 1.0 (full).
const ENGINE_FORCE = 600;
const BRAKE_FORCE = 80;

export interface SpawnCarOptions {
  position: { x: number; y?: number; z: number };
  heading: number;
  /** Diagnostic id; not used by Rapier. */
  id: string;
}

export interface CarHandle {
  id: string;
  chassis: RAPIER.RigidBody;
  vehicle: RAPIER.DynamicRayCastVehicleController;
  /** Read the chassis back as a planner-shaped state. */
  readState(now: number): VehicleState;
  /** Drive the raycast vehicle for one physics tick. */
  applyControls(c: { steer: number; throttle: number; brake: number }): void;
  /** Hard reset to a pose (e.g. on /R). */
  teleport(pose: { x: number; z: number; heading: number }): void;
  dispose(): void;
}

/** Spawn a chassis + raycast vehicle in `world`. Wheels are attached in a
 *  standard 4-corner layout; the two front wheels (indices 0,1) steer. */
export function spawnCar(
  world: RAPIER.World,
  opts: SpawnCarOptions,
): CarHandle {
  const cy = opts.position.y ?? CHASSIS_HALF.y + WHEEL_RADIUS + WHEEL_SUSPENSION_REST;
  const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(opts.position.x, cy, opts.position.z)
    .setRotation(yawQuat(opts.heading))
    .setLinearDamping(0.05)
    .setAngularDamping(0.5)
    .setCanSleep(false);
  const chassis = world.createRigidBody(bodyDesc);
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(CHASSIS_HALF.x, CHASSIS_HALF.y, CHASSIS_HALF.z)
      .setDensity(1.2)
      .setFriction(0.6),
    chassis,
  );

  const vehicle = world.createVehicleController(chassis);
  vehicle.indexUpAxis = 1; // Y up
  // setIndexForwardAxis setter name is `setIndexForwardAxis` per Rapier d.ts.
  (vehicle as unknown as { setIndexForwardAxis: number }).setIndexForwardAxis = 0; // +X forward
  const dir = { x: 0, y: -1, z: 0 };
  const axle = { x: 0, y: 0, z: 1 };
  const wheels: Array<[number, number]> = [
    [+WHEEL_BASE, -WHEEL_TRACK], // front-right
    [+WHEEL_BASE, +WHEEL_TRACK], // front-left
    [-WHEEL_BASE, -WHEEL_TRACK], // rear-right
    [-WHEEL_BASE, +WHEEL_TRACK], // rear-left
  ];
  for (const [fx, fz] of wheels) {
    vehicle.addWheel(
      { x: fx, y: -CHASSIS_HALF.y + 0.1, z: fz },
      dir,
      axle,
      WHEEL_SUSPENSION_REST,
      WHEEL_RADIUS,
    );
  }
  for (let i = 0; i < 4; i++) {
    vehicle.setWheelSuspensionStiffness(i, 30);
    vehicle.setWheelMaxSuspensionForce(i, 6000);
    vehicle.setWheelSuspensionCompression(i, 4);
    vehicle.setWheelSuspensionRelaxation(i, 2.5);
    vehicle.setWheelFrictionSlip(i, 1.3);
    vehicle.setWheelSideFrictionStiffness(i, 0.7);
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

  function applyControls(c: { steer: number; throttle: number; brake: number }) {
    const steer = clamp(c.steer, -0.6, 0.6);
    const throttle = clamp(c.throttle, -1, 1);
    const brake = clamp(c.brake, 0, 1);
    // Front-wheel steering.
    vehicle.setWheelSteering(0, steer);
    vehicle.setWheelSteering(1, steer);
    // AWD: engine force on every wheel so the chassis (~10 kg in Rapier
    // units) actually accelerates from rest. Reverse uses the same wheels.
    const engineForce = throttle * ENGINE_FORCE;
    for (let i = 0; i < 4; i++) vehicle.setWheelEngineForce(i, engineForce);
    // Brake on all wheels.
    const brakeForce = brake * BRAKE_FORCE;
    for (let i = 0; i < 4; i++) vehicle.setWheelBrake(i, brakeForce);
  }

  function teleport(pose: { x: number; z: number; heading: number }) {
    chassis.setTranslation({ x: pose.x, y: cy, z: pose.z }, true);
    chassis.setRotation(yawQuat(pose.heading), true);
    chassis.setLinvel({ x: 0, y: 0, z: 0 }, true);
    chassis.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  function dispose() {
    world.removeVehicleController(vehicle);
    world.removeRigidBody(chassis);
  }

  return { id: opts.id, chassis, vehicle, readState, applyControls, teleport, dispose };
}

// ---------------------------------------------------------------------------
// Pure-pursuit → Rapier controls.

const PURE_PURSUIT_CFG: PurePursuitConfig = {
  lookaheadMin: 3,
  lookaheadGain: 0.45,
  lookaheadMax: 14,
  maxLateralAccel: 8,
  maxAccel: 6,
  maxDecel: 8,
  cruiseSpeed: CARCHASE_AGENT.maxSpeed,
  goalTolerance: 2,
  minTurnRadius: CARCHASE_AGENT.minTurnRadius,
};

/** Convert a kinocat plan tail to (steer, throttle, brake) for Rapier.
 *  `state` is the current chassis state read back from physics. */
export function planToControls(
  state: VehicleState,
  path: VehicleState[],
): { steer: number; throttle: number; brake: number; atGoal: boolean; lookahead: { x: number; z: number } } {
  const cmd = purePursuit(state, path, PURE_PURSUIT_CFG);
  // purePursuit returns curvature (1/radius). Ackermann steering for a car
  // with wheel-base 2*WHEEL_BASE: steer = atan(curvature * wheelBase).
  const wheelBase = 2 * WHEEL_BASE;
  const steer = Math.atan(cmd.steering * wheelBase);
  return {
    steer,
    throttle: cmd.throttle,
    brake: cmd.brake,
    atGoal: cmd.atGoal,
    lookahead: cmd.lookahead,
  };
}

// ---------------------------------------------------------------------------
// Three.js render helpers — building / boost-pad / drift-gate meshes are
// authored here so the geometry source-of-truth stays in carchase-scenarios.
// (We don't import three at the top level — let callers do that — but we
// expose accessor types from BuildingSpec / JumpSpec that the renderer
// consumes.)

export type { BuildingSpec, JumpSpec, CarChaseCourse };

function yawQuat(h: number): { x: number; y: number; z: number; w: number } {
  return { x: 0, y: Math.sin(h / 2), z: 0, w: Math.cos(h / 2) };
}

function yawFromQuat(q: { x: number; y: number; z: number; w: number }): number {
  return Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
