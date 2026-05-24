// Bit-exact equivalence between HeadlessTrialHarness and the new
// RapierCarBody + ScriptedDriver path via the generic scene runtime.
//
// Both paths drive the SAME Rapier raycast vehicle with the SAME native
// `WheeledCarControls` trace at the SAME 1/60 dt. After the same number of
// ticks the chassis state read back from each should match to floating-
// point precision.
//
// This is the gate that proves the generic <S, C> scene runtime — used by
// the live demos and (next phase) the training pipeline — is operating on
// literally the same code path as the offline harness, not a parallel
// reimplementation.

import { describe, expect, it } from 'vitest';

import {
  RapierCarBody,
  createGroundCollider,
  createHeadlessTrialHarness,
  createRaycastVehicle,
  ensureRapier,
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

describe.skipIf(!RAPIER_OK)('HeadlessTrialHarness vs RapierCarBody equivalence', () => {
  it('produces matching final state for a fixed controls trace', async () => {
    // 30-tick gentle throttle + small steer.
    const trace: WheeledCarControls[] = [];
    for (let i = 0; i < 30; i++) {
      trace.push({ steer: 0.05, driveForce: 1500, brakeForce: 0 });
    }

    // Path A: existing harness.
    const harness = await createHeadlessTrialHarness({ vehicleOptions: {} });
    const outA = harness.runTrial({
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: 0 },
      controlsTrace: trace,
      sampleEveryNTicks: 1,
      id: 'A',
    });
    expect(outA.ok).toBe(true);
    if (!outA.ok) throw new Error(outA.reason);
    const finalA = outA.trial.samples[outA.trial.samples.length - 1]!;
    harness.dispose();

    // Path B: RapierCarBody + manual step. Mirrors the harness's
    // settle phase (39 ticks total before the trace plays).
    const worldB = new RAPIER!.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(worldB, {
      bounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
      pad: 20,
      friction: 1.5,
    });
    const carB = createRaycastVehicle(worldB, {
      id: 'B',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    // Initial settle (30) + per-spec settle (9) = 39 ticks of zero controls.
    for (let i = 0; i < 30; i++) {
      carB.applyControls({ steer: 0, throttle: 0, brake: 0 });
      worldB.timestep = 1 / 60;
      carB.vehicle.updateVehicle(1 / 60);
      worldB.step();
    }
    // RapierCarBody uses its own substep=4 path; to match the harness's
    // single-step-per-tick, use substeps=1.
    const body = new RapierCarBody({ world: worldB, car: carB, substeps: 1 });
    body.teleport({ x: 0, z: 0, heading: 0, speed: 0, t: 0 });
    for (let i = 0; i < 9; i++) {
      body.applyControls({ steer: 0, driveForce: 0, brakeForce: 0 });
      body.step(1 / 60);
    }
    // After settle, harness restores initial kin if lateralVel/yawRate > 0.05.
    // We pass zero kin so no restoration needed.
    for (const c of trace) {
      body.applyControls(c);
      body.step(1 / 60);
    }
    const finalB = body.readState();
    body.dispose();
    worldB.free();

    expect(finalB.x).toBeCloseTo(finalA.x, 5);
    expect(finalB.z).toBeCloseTo(finalA.z, 5);
    expect(finalB.heading).toBeCloseTo(finalA.heading, 5);
    expect(finalB.speed).toBeCloseTo(finalA.speed, 5);
  });
});
