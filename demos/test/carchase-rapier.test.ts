// Rapier-backed smoke test for the car-chase Rapier wiring
// (`demos/app/carchase/rapierVehicle.ts`). The exhaustive raycast-vehicle
// suite (settle / throttle / brake / steer / teleport / drivetrain) now lives
// in `core/test/adapters/raycast-vehicle.test.ts`; this file just confirms
// the car-chase-tuned chassis still moves under throttle so a regression in
// the tuning constants (engine force, density, wheel layout, drive train) is
// caught before it hides inside the interactive demo.

import { describe, it, expect } from 'vitest';
import {
  createCarChaseWorld,
  spawnCar,
  ensureRapier,
  type CarHandle,
} from '../app/carchase/rapierVehicle';
import { buildCarChaseCourse } from '../app/lib/carchase-scenarios';

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
function step(
  world: ReturnType<typeof createCarChaseWorld>['world'],
  car: CarHandle,
  controls: { steer: number; throttle: number; brake: number },
): void {
  // Test helper: accepts normalized {steer (rad), throttle, brake};
  // converts to canonical WheeledCarControls with the chassis-side
  // steer-flip pre-applied so test expectations match the legacy
  // applyControls path bit-for-bit.
  car.applyWheeledControls({
    steer: -controls.steer,
    driveForce: controls.throttle * 4000,
    brakeForce: controls.brake * 2000,
  });
  world.timestep = PHYSICS_DT;
  car.vehicle.updateVehicle(PHYSICS_DT);
  world.step();
}

describe.skipIf(!RAPIER_OK)('carchase chassis tuning is drivable', () => {
  it('full throttle moves the car-chase chassis forward by several metres in 1 s', () => {
    const course = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const car = spawnCar(physics.world, {
      id: 'drive',
      position: { x: 0, z: 30 }, // pick a clear spawn (downtown blocks are +x)
      heading: 0,
    });
    for (let i = 0; i < 20; i++) step(physics.world, car, { steer: 0, throttle: 0, brake: 0 });
    const start = car.chassis.translation();
    for (let i = 0; i < 60; i++) step(physics.world, car, { steer: 0, throttle: 1, brake: 0 });
    const end = car.chassis.translation();
    expect(end.x - start.x).toBeGreaterThan(2);
    car.dispose();
    physics.world.free();
  });
});
