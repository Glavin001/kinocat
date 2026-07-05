// Verify the v2 extensions to the Rapier raycast-vehicle wrapper:
//   - readState populates yawRate + lateralVelocity
//   - teleportFull seeds the full kinematic state (lateral + yaw)
//   - applyWheeledControls writes the native action shape to the right wheels

import { describe, it, expect } from 'vitest';
import {
  ensureRapier,
  createRaycastVehicle,
  createGroundCollider,
  deriveLearnableConfig,
} from '../../src/adapters/rapier/raycast-vehicle';

let RAPIER_OK = false;
let RAPIER: Awaited<ReturnType<typeof ensureRapier>> | null = null;
try {
  RAPIER = await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

const PHYSICS_DT = 1 / 60;

function makeCar() {
  const world = new RAPIER!.World({ x: 0, y: -9.81, z: 0 });
  createGroundCollider(world, {
    bounds: { x0: -200, x1: 200, z0: -200, z1: 200 },
    pad: 20,
  });
  const car = createRaycastVehicle(world, {
    id: 'v2-test',
    position: { x: 0, z: 0 },
    heading: 0,
  });
  // Settle.
  for (let i = 0; i < 30; i++) {
    car.applyWheeledControls({ steer: 0, driveForce: 0, brakeForce: 0 });
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    world.step();
  }
  return { world, car };
}

describe.skipIf(!RAPIER_OK)('readState — extended kinematic fields', () => {
  it('yawRate is approximately what we set via setAngvel (with kinocat sign flip)', () => {
    const { car } = makeCar();
    // Set angvel.y = -1.0 (rapier convention) → planning-frame yawRate = +1.0.
    car.chassis.setAngvel({ x: 0, y: -1.0, z: 0 }, true);
    const st = car.readState(0);
    expect(st.yawRate).toBeDefined();
    expect(Math.abs((st.yawRate ?? 0) - 1.0)).toBeLessThan(1e-3);
    car.dispose();
  });

  it('lateralVelocity matches linvel projected onto chassis-right', () => {
    const { car } = makeCar();
    // Heading 0 → forward = +X, right = (sin 0, -cos 0) = (0, -1).
    // So linvel.z = -2.0 → lateralVelocity = 0*0 - (-2)*1 = 2.0
    car.chassis.setLinvel({ x: 5, y: 0, z: -2 }, true);
    const st = car.readState(0);
    expect(st.speed).toBeCloseTo(5, 3);
    expect(st.lateralVelocity).toBeCloseTo(2, 3);
    car.dispose();
  });
});

describe.skipIf(!RAPIER_OK)('teleportFull — seeds full kinematic state', () => {
  it('seeds linvel including lateral component and angvel', () => {
    const { car } = makeCar();
    car.teleportFull(
      { x: 10, z: 5, heading: 0 },
      { forwardSpeed: 8, lateralVelocity: 1.5, yawRate: 0.4 },
    );
    const v = car.chassis.linvel();
    const a = car.chassis.angvel();
    // forward=(cos h, sin h) at h=0 → forward=(1,0,0); right=(sin h, -cos h)=(0,0,-1).
    // worldVel = forward * 8 + right * 1.5 = (8, 0, -1.5)
    expect(v.x).toBeCloseTo(8, 3);
    expect(v.z).toBeCloseTo(-1.5, 3);
    // angvel.y = -yawRate (rapier convention sign-flip)
    expect(a.y).toBeCloseTo(-0.4, 3);
    car.dispose();
  });
});

describe.skipIf(!RAPIER_OK)('applyWheeledControls — native-action shape', () => {
  it('writes engineForce to driven wheels only (RWD default)', () => {
    const { world, car } = makeCar();
    car.applyWheeledControls({ steer: 0, driveForce: 1500, brakeForce: 0 });
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    // For RWD (driveTrain default 'rwd'): wheels 0,1 = 0, wheels 2,3 = 1500.
    expect(car.vehicle.wheelEngineForce(0)).toBe(0);
    expect(car.vehicle.wheelEngineForce(1)).toBe(0);
    expect(car.vehicle.wheelEngineForce(2)).toBe(1500);
    expect(car.vehicle.wheelEngineForce(3)).toBe(1500);
    car.dispose();
  });

  it('clamps steer, sign-flips at the Rapier boundary, and applies Ackermann per wheel', () => {
    const { car } = makeCar();
    // The wrapper's default maxSteerAngle is 0.6. Pass a kinocat-frame steer
    // of +0.4 -> Rapier bicycle steer is -0.4 after the sign-flip, and each
    // front wheel gets its Ackermann angle about the shared turn centre:
    // L = 2*wheelBase = 3.2, w = wheelTrack = 0.85, R = L/tan(0.4).
    car.applyWheeledControls({ steer: 0.4, driveForce: 0, brakeForce: 0 });
    const L = 3.2;
    const w = 0.85;
    const R = L / Math.tan(0.4);
    const inner = Math.atan(L / (R - w));
    const outer = Math.atan(L / (R + w));
    // Rapier steer -0.4 turns toward +z => inner is wheel 1 (front-left, +z).
    expect(car.vehicle.wheelSteering(1)).toBeCloseTo(-inner, 5);
    expect(car.vehicle.wheelSteering(0)).toBeCloseTo(-outer, 5);
    // The inner wheel turns harder than the bicycle angle, the outer softer.
    expect(inner).toBeGreaterThan(0.4);
    expect(outer).toBeLessThan(0.4);
    // Clamp on the upper bound applies to the BICYCLE command (the per-wheel
    // inner angle may legitimately exceed it).
    car.applyWheeledControls({ steer: 99, driveForce: 0, brakeForce: 0 });
    const Rc = L / Math.tan(0.6);
    expect(car.vehicle.wheelSteering(1)).toBeCloseTo(-Math.atan(L / (Rc - w)), 5);
    expect(car.vehicle.wheelSteering(0)).toBeCloseTo(-Math.atan(L / (Rc + w)), 5);
    car.dispose();
  });
});

describe.skipIf(!RAPIER_OK)('deriveLearnableConfig — pulls from options', () => {
  it('chassis mass = density × volume', () => {
    const cfg = deriveLearnableConfig({
      id: 'x', position: { x: 0, z: 0 }, heading: 0,
      chassisHalf: { x: 2, y: 0.5, z: 1 },
      chassisDensity: 100,
      engineForce: 3000,
      brakeForce: 1500,
    });
    expect(cfg.chassisMass).toBe(8 * 2 * 0.5 * 1 * 100);
    expect(cfg.maxDriveForce).toBe(3000);
    expect(cfg.maxBrakeForce).toBe(1500);
  });

  it('drivenWheels propagates', () => {
    const cfg = deriveLearnableConfig({
      id: 'x', position: { x: 0, z: 0 }, heading: 0,
      driveTrain: 'awd',
    });
    expect(cfg.drivenWheels).toBe('awd');
  });
});
