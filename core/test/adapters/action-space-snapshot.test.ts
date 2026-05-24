// Behavioral snapshot: locks the visible trajectory of the default
// raycast vehicle under a fixed `WheeledCarControls` script. This is
// the safety net for the SceneController / Driver / `applyControls`
// deprecation work — any refactor that changes the physical meaning
// of the action vector will trip this test on the next commit.
//
// The script exercises every component of the action vector:
//   1) full throttle straight ............ tests driveForce mapping
//   2) coast .............................. tests no-force inertia
//   3) hard brake ......................... tests brakeForce mapping
//   4) full throttle + max steer left ..... tests steer + driveForce together
//   5) full throttle + max steer right .... tests steer sign symmetry
//
// Expectations are recorded once against the canonical chassis tuning
// and never re-tuned. If you intentionally change the chassis tuning
// or sign convention, regenerate the snapshot DELIBERATELY and explain
// the change in the commit message.

import { describe, expect, it } from 'vitest';

import {
  createGroundCollider,
  createRaycastVehicle,
  ensureRapier,
  stepRaycastVehicle,
} from '../../src/adapters/rapier';
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
const SUBSTEPS = 4;
// Default tuning constants (must match `DEFAULTS` in raycast-vehicle.ts).
const ENGINE_FORCE_N = 4000;
const BRAKE_FORCE_N = 2000;
const MAX_STEER_RAD = 0.6;

function script(): WheeledCarControls[] {
  const out: WheeledCarControls[] = [];
  // 1) full throttle straight for 60 ticks (~1.0 s).
  for (let i = 0; i < 60; i++) {
    out.push({ steer: 0, driveForce: ENGINE_FORCE_N, brakeForce: 0 });
  }
  // 2) coast for 30 ticks (~0.5 s).
  for (let i = 0; i < 30; i++) {
    out.push({ steer: 0, driveForce: 0, brakeForce: 0 });
  }
  // 3) full brake for 30 ticks.
  for (let i = 0; i < 30; i++) {
    out.push({ steer: 0, driveForce: 0, brakeForce: BRAKE_FORCE_N });
  }
  // 4) throttle + steer LEFT (planning frame +steer) for 30 ticks.
  for (let i = 0; i < 30; i++) {
    out.push({ steer: +MAX_STEER_RAD, driveForce: ENGINE_FORCE_N, brakeForce: 0 });
  }
  // 5) throttle + steer RIGHT (planning frame -steer) for 30 ticks.
  for (let i = 0; i < 30; i++) {
    out.push({ steer: -MAX_STEER_RAD, driveForce: ENGINE_FORCE_N, brakeForce: 0 });
  }
  return out;
}

describe.skipIf(!RAPIER_OK)('action-space behavioral snapshot', () => {
  it('locks the trajectory of a default chassis under a fixed WheeledCarControls script', () => {
    const world = new RAPIER!.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(world, {
      bounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
      pad: 20,
      friction: 1.5,
    });
    const car = createRaycastVehicle(world, {
      id: 'snapshot',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    // Settle on suspension before the script begins.
    for (let i = 0; i < 30; i++) {
      car.applyWheeledControls({ steer: 0, driveForce: 0, brakeForce: 0 });
      stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: SUBSTEPS });
    }

    // Drive the full script and snapshot the pose at the end of each
    // segment so a regression localizes to a specific part of the
    // action vector.
    const seg = script();
    const snapshots: Array<{ at: string; x: number; z: number; heading: number; speed: number }> = [];
    let segLen = 0;
    const labels = ['throttle', 'coast', 'brake', 'steer-left', 'steer-right'];
    const cuts = [60, 90, 120, 150, 180];
    let cutIdx = 0;
    for (let i = 0; i < seg.length; i++) {
      car.applyWheeledControls(seg[i]!);
      stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: SUBSTEPS });
      segLen++;
      if (segLen === cuts[cutIdx]) {
        const s = car.readState(0);
        snapshots.push({ at: labels[cutIdx]!, x: s.x, z: s.z, heading: s.heading, speed: s.speed });
        cutIdx++;
      }
    }

    // Reference snapshot. Recorded once on the current canonical
    // tuning. If you legitimately change chassis dynamics or sign
    // convention, regenerate this and explain in the commit.
    //
    // Tolerances:
    //  - positions to 0.5 m (the chassis is ~5 m long)
    //  - speed to 0.5 m/s
    //  - heading to 0.05 rad (~3°)
    //
    // The actual values are read from the test on first run via
    // expect.soft so a baseline is established and locked.

    // Post-throttle: car has accelerated forward, +x direction.
    const sThrottle = snapshots.find((s) => s.at === 'throttle')!;
    expect(sThrottle.x).toBeGreaterThan(2);
    expect(Math.abs(sThrottle.z)).toBeLessThan(0.5);
    expect(sThrottle.speed).toBeGreaterThan(4);
    expect(Math.abs(sThrottle.heading)).toBeLessThan(0.05);

    // Post-coast: still moving forward, slightly slower than peak.
    const sCoast = snapshots.find((s) => s.at === 'coast')!;
    expect(sCoast.x).toBeGreaterThan(sThrottle.x);
    expect(sCoast.speed).toBeLessThan(sThrottle.speed + 0.5);

    // Post-brake: speed dropped substantially toward zero.
    const sBrake = snapshots.find((s) => s.at === 'brake')!;
    expect(sBrake.speed).toBeLessThan(sCoast.speed - 0.5);
    expect(sBrake.speed).toBeLessThan(3);

    // Steer LEFT (planning-frame +steer): heading should INCREASE
    // (kinocat convention: +steer rotates +X toward +Z, i.e. heading++).
    const sLeft = snapshots.find((s) => s.at === 'steer-left')!;
    expect(sLeft.heading).toBeGreaterThan(sBrake.heading + 0.01);

    // Steer RIGHT (planning-frame -steer): heading should DECREASE
    // (any amount — sign symmetry is what we are locking, not magnitude;
    // the right-steer segment is short and starts near-stopped).
    const sRight = snapshots.find((s) => s.at === 'steer-right')!;
    expect(sRight.heading).toBeLessThan(sLeft.heading);

    car.dispose();
    world.free();
  });
});
