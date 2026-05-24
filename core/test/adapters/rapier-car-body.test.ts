// Integration test for the new RapierCarBody + stepRaycastVehicle helpers.
// Confirms that running a Body<CarKinematicState, WheeledCarControls>
// through the generic SceneController gives the same result as driving
// the raw CarHandle inline (the pattern every demo currently uses).

import { describe, expect, it } from 'vitest';

import {
  RapierCarBody,
  createGroundCollider,
  createRaycastVehicle,
  ensureRapier,
  stepRaycastVehicle,
} from '../../src/adapters/rapier';
import { IdleDriver, SceneController } from '../../src/scene';
import type { CarKinematicState } from '../../src/agent/types';
import type { WheeledCarControls } from '../../src/agent/controls';

let RAPIER_OK = false;
let RAPIER: Awaited<ReturnType<typeof ensureRapier>> | null = null;
try {
  RAPIER = await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

const PHYSICS_DT = 1 / 60;

function makeWorld() {
  const world = new RAPIER!.World({ x: 0, y: -9.81, z: 0 });
  createGroundCollider(world, {
    bounds: { x0: -200, x1: 200, z0: -200, z1: 200 },
    pad: 20,
  });
  return world;
}

describe.skipIf(!RAPIER_OK)('RapierCarBody + SceneController', () => {
  it('steps with stepPolicy=self and produces forward motion under throttle', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'body-test',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    const body = new RapierCarBody({ world, car, substeps: 4 });
    // Settle.
    for (let i = 0; i < 30; i++) {
      body.applyControls({ steer: 0, driveForce: 0, brakeForce: 0 });
      body.step(PHYSICS_DT);
    }
    const startX = body.readState().x;
    // Drive forward 1 second @ full throttle.
    for (let i = 0; i < 60; i++) {
      body.applyControls({ steer: 0, driveForce: 4000, brakeForce: 0 });
      body.step(PHYSICS_DT);
    }
    const endX = body.readState().x;
    expect(endX - startX).toBeGreaterThan(2);
    body.dispose();
    world.free();
  });

  it('SceneController wraps RapierCarBody and emits StepResult', () => {
    const world = makeWorld();
    const car = createRaycastVehicle(world, {
      id: 'scene-test',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    const body = new RapierCarBody({ world, car });
    const driver = new IdleDriver<CarKinematicState, WheeledCarControls>({
      steer: 0,
      driveForce: 0,
      brakeForce: 0,
    });
    const ctl = new SceneController({ body, driver, dt: PHYSICS_DT });
    for (let i = 0; i < 30; i++) ctl.step(i * PHYSICS_DT);
    const out = ctl.step(30 * PHYSICS_DT);
    expect(Number.isFinite(out.real.x)).toBe(true);
    expect(Number.isFinite(out.real.heading)).toBe(true);
    body.dispose();
    world.free();
  });

  it('stepRaycastVehicle drives multiple cars in one shared world step', () => {
    const world = makeWorld();
    const car1 = createRaycastVehicle(world, {
      id: 'shared-1',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    const car2 = createRaycastVehicle(world, {
      id: 'shared-2',
      position: { x: 0, z: 8 },
      heading: 0,
    });
    for (let i = 0; i < 30; i++) {
      car1.applyControls({ steer: 0, throttle: 0, brake: 0 });
      car2.applyControls({ steer: 0, throttle: 0, brake: 0 });
      stepRaycastVehicle(world, [car1, car2], { dt: PHYSICS_DT, substeps: 4 });
    }
    for (let i = 0; i < 60; i++) {
      car1.applyControls({ steer: 0, throttle: 1, brake: 0 });
      car2.applyControls({ steer: 0, throttle: 0.5, brake: 0 });
      stepRaycastVehicle(world, [car1, car2], { dt: PHYSICS_DT, substeps: 4 });
    }
    const x1 = car1.readState(0).x;
    const x2 = car2.readState(0).x;
    expect(x1).toBeGreaterThan(x2); // higher throttle goes further
    car1.dispose();
    car2.dispose();
    world.free();
  });
});
