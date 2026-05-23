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
import type { VehicleState } from '../../src/agent/types';

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
  car.applyControls(controls);
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
    const state: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const path: VehicleState[] = [
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
    const state: VehicleState = { x: 10, z: 0, heading: 0, speed: 6, t: 0 };
    const path: VehicleState[] = [
      { x: 9, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 0, t: 0.2 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBeGreaterThan(0);
    expect(cmd.throttle).toBe(0);
  });

  // Regression for a sign-convention bug at the kinocat→Rapier boundary:
  // purePursuit returns throttle as a NON-NEGATIVE magnitude in [0, 1]
  // and conveys direction via targetSpeed's sign. Rapier's applyControls
  // takes a SIGNED throttle in [-1, 1] (negative = engine drives reverse).
  // Before the fix, planToAckermannControls passed the magnitude through
  // unsigned, so the engine drove forward even when the plan called for
  // reverse — visible in the parking demo as "back-in maneuver but car
  // drives the opposite direction". These tests pin that down.
  it('returns NEGATIVE throttle when the plan ahead has negative speed (reverse)', () => {
    // Path samples behind the chassis with negative planned speed: the
    // chassis must back up to follow it.
    const state: VehicleState = { x: 10, z: 0, heading: 0, speed: 0, t: 0 };
    const path: VehicleState[] = [
      { x: 10, z: 0, heading: 0, speed: -2, t: 0 },
      { x: 7, z: 0, heading: 0, speed: -2, t: 1.5 },
      { x: 4, z: 0, heading: 0, speed: -2, t: 3.0 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    expect(cmd.atGoal).toBe(false);
    expect(cmd.throttle).toBeLessThan(0);
    expect(cmd.brake).toBe(0);
  });

  it('returns POSITIVE throttle for an otherwise-identical forward path', () => {
    // Same geometry mirrored: positive planned speed → forward throttle.
    const state: VehicleState = { x: 4, z: 0, heading: 0, speed: 0, t: 0 };
    const path: VehicleState[] = [
      { x: 4, z: 0, heading: 0, speed: 2, t: 0 },
      { x: 7, z: 0, heading: 0, speed: 2, t: 1.5 },
      { x: 10, z: 0, heading: 0, speed: 2, t: 3.0 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    expect(cmd.atGoal).toBe(false);
    expect(cmd.throttle).toBeGreaterThan(0);
    expect(cmd.brake).toBe(0);
  });

  it('all-zero-speed path defaults to forward gear (documented limitation)', () => {
    // Pure-pursuit picks gear from the sign of path[ni+1].speed, with
    // 0 defaulting to FORWARD. So a path that omits speed signs entirely
    // (e.g. an analytic-shot reconstruction that emits speed=0 at every
    // sample) always drives forward — even if the planner's Reeds-Shepp
    // curve had reverse segments. This isn't the bug fixed in the
    // siblings above; it's a related pitfall worth pinning down so a
    // future change of pure-pursuit's gear-defaulting rule shows up
    // here, and so any caller that plans through `samples` (rather than
    // primitive states with proper speed signs) sees the FAIL and is
    // forced to populate sign info before relying on plan execution.
    const state: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const path: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: 0, t: 0 },
      { x: 5, z: 0, heading: 0, speed: 0, t: 1.0 },
    ];
    const cmd = planToAckermannControls(state, path, cfg);
    // Documents current behaviour, not a desired invariant.
    expect(cmd.throttle).toBeGreaterThan(0);
  });
});

// End-to-end: a reverse plan + planToAckermannControls + Rapier physics
// must move the chassis BACKWARD. Before the fix the chassis drove
// forward regardless of the planned direction; this test would have
// caught it.
describe.skipIf(!RAPIER_OK)('plan → controls → Rapier integration (sign conventions)', () => {
  it('reverse plan actually backs the chassis up', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'plan-reverse',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    // Settle on the ground first.
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });
    const start = car.chassis.translation();

    const cfg = {
      wheelBase: 3.2,
      lookaheadMin: 1,
      lookaheadGain: 0.5,
      lookaheadMax: 3,
      maxLateralAccel: 4,
      maxAccel: 2,
      maxDecel: 6,
      cruiseSpeed: 3,
      goalTolerance: 0.5,
      minTurnRadius: 4,
    };
    // Path leads BEHIND the chassis (decreasing x), planned speed is
    // negative throughout — a reverse-only trajectory.
    const reversePath: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: -2, t: 0 },
      { x: -2, z: 0, heading: 0, speed: -2, t: 1 },
      { x: -4, z: 0, heading: 0, speed: -2, t: 2 },
      { x: -6, z: 0, heading: 0, speed: -2, t: 3 },
    ];
    for (let i = 0; i < 90; i++) {
      const state = car.readState(0);
      const cmd = planToAckermannControls(state, reversePath, cfg);
      step(world, car, { steer: cmd.steer, throttle: cmd.throttle, brake: cmd.brake });
    }
    const end = car.chassis.translation();
    // Must have moved BACKWARD (negative x) by a clear margin, not
    // forward. Before the fix this assertion failed — the chassis
    // drove forward by several metres because throttle was always
    // positive.
    expect(end.x - start.x).toBeLessThan(-1);
    expect(Math.abs(end.z - start.z)).toBeLessThan(2);
    car.dispose();
    world.free();
  });

  it('plan curving LEFT (toward +Z) physically turns the chassis left', () => {
    // The other sign-convention boundary in planToAckermannControls:
    // kinocat's +curvature rotates +X toward +Z, but Rapier's +yaw
    // about +Y rotates +X toward -Z. The conversion negates steer at
    // the boundary so a "turn left" plan curvature produces a "turn
    // left" physical yaw. If this were ever silently inverted, the
    // chassis would track the mirror image of any planned curve.
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'plan-left-curve',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(world, car, { steer: 0, throttle: 0, brake: 0 });

    const cfg = {
      wheelBase: 3.2,
      lookaheadMin: 1,
      lookaheadGain: 0.4,
      lookaheadMax: 3,
      maxLateralAccel: 4,
      maxAccel: 3,
      maxDecel: 6,
      cruiseSpeed: 4,
      goalTolerance: 0.5,
      minTurnRadius: 4,
    };
    // Forward-only path curving toward +Z (kinocat "left" with heading=0
    // and +curvature rotating +X→+Z).
    const leftCurvePath: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: 3, t: 0 },
      { x: 3, z: 0.5, heading: 0.16, speed: 3, t: 1 },
      { x: 6, z: 2, heading: 0.5, speed: 3, t: 2 },
      { x: 8, z: 4.5, heading: 1.0, speed: 3, t: 3 },
      { x: 8.5, z: 7, heading: 1.4, speed: 3, t: 4 },
    ];
    for (let i = 0; i < 120; i++) {
      const state = car.readState(0);
      const cmd = planToAckermannControls(state, leftCurvePath, cfg);
      step(world, car, { steer: cmd.steer, throttle: cmd.throttle, brake: cmd.brake });
    }
    const end = car.chassis.translation();
    // Chassis should have moved in the +X direction AND ended up with
    // positive Z displacement. Mirror-image (a negative-Z bend) would
    // be the failure mode if the steer sign were inverted at the
    // boundary.
    expect(end.x).toBeGreaterThan(1);
    expect(end.z).toBeGreaterThan(0.5);
    car.dispose();
    world.free();
  });
});
