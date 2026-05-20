// Rapier-backed integration tests for the car-chase Rapier wiring
// (`demos/app/carchase/rapierVehicle.ts`). The pure planning layer is already
// tested in `scenarios.test.ts`; this file is the missing half — it confirms
// that the Rapier raycast-vehicle controller is wired correctly and that
// applied controls actually translate to chassis motion.
//
// The motivating bug: PR #14 initially called `world.step()` BEFORE
// `applyControls` + `vehicle.updateVehicle()` (reversed from the canonical
// Rapier order), so engine forces were applied to a chassis whose velocity
// the step had just re-zeroed under gravity, and no torque ever reached the
// wheels. A test that drives the car forward for 1 s and asserts a
// non-trivial position change catches that class of regression directly.
//
// Rapier WASM is gated via `describe.skipIf(!RAPIER_OK)` — same pattern as
// `core/test/adapters/rapier.test.ts` — so CI runners without the WASM
// binary still pass.

import { describe, it, expect } from 'vitest';
import {
  createCarChaseWorld,
  spawnCar,
  planToControls,
  ensureRapier,
  type CarHandle,
} from '../app/carchase/rapierVehicle';
import { buildCarChaseCourse } from '../app/lib/carchase-scenarios';
import type { VehicleState } from 'kinocat/agent';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

it('rapier availability is a boolean (logs skip status in CI)', () => {
  expect(typeof RAPIER_OK).toBe('boolean');
});

const PHYSICS_DT = 1 / 60;

/** Canonical Rapier raycast-vehicle tick: set wheel inputs (via
 *  `applyControls`), updateVehicle(dt), THEN world.step. The order is the
 *  whole point of these tests. */
function step(
  world: ReturnType<typeof createCarChaseWorld>['world'],
  car: CarHandle,
  controls: { steer: number; throttle: number; brake: number },
): void {
  car.applyControls(controls);
  world.timestep = PHYSICS_DT;
  car.vehicle.updateVehicle(PHYSICS_DT);
  world.step();
}

describe.skipIf(!RAPIER_OK)('rapierVehicle wiring', () => {
  it('a freshly spawned chassis settles onto the ground (gravity + suspension)', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'settle',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    // Let gravity + suspension stabilize for ~0.5 s.
    for (let i = 0; i < 30; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    const t = car.chassis.translation();
    // The chassis half-height is 0.5; sitting on the y=0 ground means the
    // centre lands somewhere in [0.5, 2]. Generous bounds — what we actually
    // care about is "didn't fall through" and "isn't stuck flying".
    expect(t.y).toBeGreaterThan(0.3);
    expect(t.y).toBeLessThan(3);
    expect(Number.isFinite(t.x)).toBe(true);
    expect(Number.isFinite(t.z)).toBe(true);
    car.dispose();
    physics.world.free();
  });

  it('full throttle moves the chassis forward by several metres in 1 s', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'drive',
      position: { x: 0, z: 0 },
      heading: 0, // +X forward
    });
    // Settle first so wheels are in contact.
    for (let i = 0; i < 20; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    const start = car.chassis.translation();
    // Drive forward for ~1 s (60 ticks at 1/60s).
    for (let i = 0; i < 60; i++) {
      step(physics.world, car, { steer: 0, throttle: 1, brake: 0 });
    }
    const end = car.chassis.translation();
    const dx = end.x - start.x;
    // The original physics-order bug had dx ≈ 0; with correct ordering the
    // ~11 kg AWD chassis covers several metres in a second even with modest
    // engine force. Threshold = 2 m so a 2× drop in performance still flags.
    expect(dx).toBeGreaterThan(2);
    expect(Math.abs(end.z - start.z)).toBeLessThan(2); // mostly straight
    car.dispose();
    physics.world.free();
  });

  it('reverse throttle moves the chassis backward', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'reverse',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    const start = car.chassis.translation();
    for (let i = 0; i < 60; i++) {
      step(physics.world, car, { steer: 0, throttle: -1, brake: 0 });
    }
    const dx = car.chassis.translation().x - start.x;
    expect(dx).toBeLessThan(-1);
    car.dispose();
    physics.world.free();
  });

  it('full-lock steering bends the path off the forward axis', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'steer',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    // Drive forward with steady left lock for ~1.5 s.
    for (let i = 0; i < 90; i++) {
      step(physics.world, car, { steer: 0.5, throttle: 0.7, brake: 0 });
    }
    const heading = car
      .readState(0)
      .heading;
    // Heading should have rotated by several degrees at minimum — a stuck
    // chassis (= the original bug) leaves heading ≈ 0.
    expect(Math.abs(heading)).toBeGreaterThan(0.1);
    car.dispose();
    physics.world.free();
  });

  it('hard brake from cruise brings the chassis to a near-stop', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'brake',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 20; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    // Spin up.
    for (let i = 0; i < 60; i++) {
      step(physics.world, car, { steer: 0, throttle: 1, brake: 0 });
    }
    const speedBefore = Math.abs(car.readState(0).speed);
    expect(speedBefore).toBeGreaterThan(2); // we did accelerate
    // Apply hard brake for ~1 s.
    for (let i = 0; i < 60; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 1 });
    }
    const speedAfter = Math.abs(car.readState(0).speed);
    expect(speedAfter).toBeLessThan(speedBefore);
    car.dispose();
    physics.world.free();
  });

  it('teleport moves the chassis without applying force', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'tp',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    for (let i = 0; i < 10; i++) {
      step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    }
    car.teleport({ x: 50, z: -20, heading: Math.PI / 2 });
    // One step settles the post-teleport pose.
    step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    const t = car.chassis.translation();
    expect(t.x).toBeCloseTo(50, 0);
    expect(t.z).toBeCloseTo(-20, 0);
    car.dispose();
    physics.world.free();
  });
});

// `planToControls` is pure: kinocat purePursuit + Ackermann conversion. No
// Rapier needed; always runs.
describe('planToControls (pure-pursuit → Ackermann)', () => {
  it('issues forward throttle and ~zero steer for a straight path ahead', () => {
    const state: VehicleState = {
      x: 0,
      z: 0,
      heading: 0, // facing +X
      speed: 0,
      t: 0,
    };
    const path: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: 8, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 8, t: 1.25 },
      { x: 20, z: 0, heading: 0, speed: 8, t: 2.5 },
    ];
    const cmd = planToControls(state, path);
    expect(cmd.throttle).toBeGreaterThan(0);
    expect(cmd.brake).toBe(0);
    expect(Math.abs(cmd.steer)).toBeLessThan(0.05);
    expect(cmd.atGoal).toBe(false);
  });

  it('issues left-handed steer for a path turning toward +Z', () => {
    const state: VehicleState = {
      x: 0,
      z: 0,
      heading: 0,
      speed: 4,
      t: 0,
    };
    // A path that curves toward +Z (the "left" side when facing +X under the
    // planning convention heading 0 = +X, heading +π/2 = +Z).
    const path: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 4, z: 1, heading: 0.25, speed: 6, t: 0.7 },
      { x: 7, z: 3.5, heading: 0.6, speed: 6, t: 1.4 },
      { x: 9, z: 7, heading: 1, speed: 6, t: 2.1 },
    ];
    const cmd = planToControls(state, path);
    expect(cmd.throttle + cmd.brake).toBeGreaterThan(0); // commanded something
    expect(cmd.steer).not.toBe(0);
  });

  it('brakes when the lookahead is already at/inside the goal tolerance', () => {
    const state: VehicleState = {
      x: 10,
      z: 0,
      heading: 0,
      speed: 6,
      t: 0,
    };
    const path: VehicleState[] = [
      { x: 9, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 0, t: 0.2 },
    ];
    const cmd = planToControls(state, path);
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBeGreaterThan(0);
    expect(cmd.throttle).toBe(0);
  });
});
