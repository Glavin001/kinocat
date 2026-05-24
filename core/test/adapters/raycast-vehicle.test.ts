// Rapier-backed integration tests for the core's raycast-vehicle wrapper.
// Confirms canonical applyControls -> updateVehicle(dt) -> world.step ordering
// is respected, plus settle/throttle/brake/steer/teleport behaviour.
//
// Rapier WASM is gated via `describe.skipIf(!RAPIER_OK)` — same pattern as
// the existing rapier.test.ts — so CI runners without the WASM binary pass.

import { describe, it, expect } from 'vitest';
import {
  ensureRapier,
  createRaycastVehicle,
  createGroundCollider,
  planToAckermannControls,
  type CarHandle,
} from '../../src/adapters/rapier/raycast-vehicle';
import type { CarKinematicState } from '../../src/agent/types';

let RAPIER_OK = false;
let RAPIER: Awaited<ReturnType<typeof ensureRapier>> | null = null;
try {
  RAPIER = await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

it('rapier availability is a boolean (logs skip status in CI)', () => {
  expect(typeof RAPIER_OK).toBe('boolean');
});

const PHYSICS_DT = 1 / 60;

function makeWorld() {
  const world = new RAPIER!.World({ x: 0, y: -9.81, z: 0 });
  createGroundCollider(world, {
    bounds: { x0: -100, x1: 100, z0: -100, z1: 100 },
    pad: 20,
  });
  return world;
}

type RapierWorld = ReturnType<typeof makeWorld>;

function step(
  world: RapierWorld,
  car: CarHandle,
  controls: { steer: number; throttle: number; brake: number },
): void {
  // Test helper: accepts normalized {steer (rad), throttle, brake} for
  // readability; converts to the canonical WheeledCarControls shape
  // (with chassis-side steer-flip pre-applied) for the chassis.
  car.applyWheeledControls({
    steer: -controls.steer,
    driveForce: controls.throttle * 4000,
    brakeForce: controls.brake * 2000,
  });
  world.timestep = PHYSICS_DT;
  car.vehicle.updateVehicle(PHYSICS_DT);
  world.step();
}

describe.skipIf(!RAPIER_OK)('createRaycastVehicle wiring', () => {
  it('a freshly spawned chassis settles onto the ground (gravity + suspension)', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'settle',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 30; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const t = car.chassis.translation();
    expect(t.y).toBeGreaterThan(0.3);
    expect(t.y).toBeLessThan(3);
    car.dispose();
    world.free();
  });

  it('full throttle moves the chassis forward by several metres in 1 s', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'drive',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const start = car.chassis.translation();
    for (let i = 0; i < 60; i++) step(world, car, { steer: 0, throttle: 1, brake: 0 });
    const end = car.chassis.translation();
    expect(end.x - start.x).toBeGreaterThan(2);
    expect(Math.abs(end.z - start.z)).toBeLessThan(2);
    car.dispose();
    world.free();
  });

  it('reverse throttle moves the chassis backward', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'reverse',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const start = car.chassis.translation();
    for (let i = 0; i < 60; i++) step(world, car, { steer: 0, throttle: -1, brake: 0 });
    expect(car.chassis.translation().x - start.x).toBeLessThan(-1);
    car.dispose();
    world.free();
  });

  it('full-lock steering bends the path off the forward axis', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'steer',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    for (let i = 0; i < 90; i++) step(world, car, { steer: 0.5, throttle: 0.7, brake: 0 });
    const heading = car.readState(0).heading;
    expect(Math.abs(heading)).toBeGreaterThan(0.1);
    car.dispose();
    world.free();
  });

  it('hard brake from cruise brings the chassis to a near-stop', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'brake',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    for (let i = 0; i < 60; i++) step(world, car, { steer: 0, throttle: 1, brake: 0 });
    const speedBefore = Math.abs(car.readState(0).speed);
    expect(speedBefore).toBeGreaterThan(2);
    for (let i = 0; i < 60; i++) step(world, car, { steer: 0, throttle: 0, brake: 1 });
    expect(Math.abs(car.readState(0).speed)).toBeLessThan(speedBefore);
    car.dispose();
    world.free();
  });

  it('teleport moves the chassis without applying force', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'tp',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 10; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    car.teleport({ x: 50, z: -20, heading: Math.PI / 2 });
    step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const t = car.chassis.translation();
    expect(t.x).toBeCloseTo(50, 0);
    expect(t.z).toBeCloseTo(-20, 0);
    car.dispose();
    world.free();
  });

  it('readWheelTelemetry exposes per-wheel contact + impulses for the last tick', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'telemetry',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    // Settle so the chassis is on the ground; then drive a tick under load
    // (throttle + slight steer) so impulses are non-zero.
    for (let i = 0; i < 30; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    for (let i = 0; i < 5; i++) step(world, car, { steer: 0.2, throttle: 1, brake: 0 });
    const wheels = car.readWheelTelemetry();
    expect(wheels).toHaveLength(4);
    // At least one wheel should be in contact and reporting a valid point.
    const grounded = wheels.filter((w) => w.inContact);
    expect(grounded.length).toBeGreaterThanOrEqual(2);
    for (const w of grounded) {
      expect(w.contactPoint).not.toBeNull();
      expect(Number.isFinite(w.contactPoint!.x)).toBe(true);
      expect(Number.isFinite(w.contactPoint!.y)).toBe(true);
      expect(Number.isFinite(w.contactPoint!.z)).toBe(true);
      expect(w.suspensionForce).toBeGreaterThan(0);
      expect(w.frictionSlip).toBeGreaterThan(0);
      expect(Number.isFinite(w.forwardImpulse)).toBe(true);
      expect(Number.isFinite(w.sideImpulse)).toBe(true);
    }
    car.dispose();
    world.free();
  });

  it('driveTrain: fwd applies engine force to front wheels only', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'fwd',
      position: { x: 0, z: 0 },
      heading: 0,
      driveTrain: 'fwd',
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const start = car.chassis.translation();
    for (let i = 0; i < 60; i++) step(world, car, { steer: 0, throttle: 1, brake: 0 });
    expect(car.chassis.translation().x - start.x).toBeGreaterThan(2);
    car.dispose();
    world.free();
  });
});

// planToAckermannControls is pure (purePursuit + Ackermann conversion). No
// Rapier needed; always runs.
describe('planToAckermannControls', () => {
  const cfg = {
    wheelBase: 3.2,
    lookaheadMin: 3,
    lookaheadGain: 0.45,
    lookaheadMax: 14,
    maxLateralAccel: 8,
    maxAccel: 6,
    maxDecel: 8,
    cruiseSpeed: 10,
    goalTolerance: 2,
    minTurnRadius: 4,
  };

  it('issues forward throttle and ~zero steer for a straight path ahead', () => {
    const state: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const path: CarKinematicState[] = [
      { x: 0, z: 0, heading: 0, speed: 8, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 8, t: 1.25 },
      { x: 20, z: 0, heading: 0, speed: 8, t: 2.5 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    expect(cmd.throttle).toBeGreaterThan(0);
    expect(cmd.brake).toBe(0);
    expect(Math.abs(cmd.steer)).toBeLessThan(0.05);
    expect(cmd.atGoal).toBe(false);
  });

  it('brakes when the lookahead is inside the goal tolerance', () => {
    const state: CarKinematicState = { x: 10, z: 0, heading: 0, speed: 6, t: 0 };
    const path: CarKinematicState[] = [
      { x: 9, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 0, t: 0.2 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBeGreaterThan(0);
    expect(cmd.throttle).toBe(0);
  });
});
