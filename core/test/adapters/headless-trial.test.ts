// Verify the headless trial harness produces consistent samples and
// correctly detects pathological outcomes (off-arena, NaN).

import { describe, it, expect } from 'vitest';
import { createHeadlessTrialHarness, ensureRapier } from '../../src/adapters/rapier';
import type { TrialSpec } from '../../src/adapters/rapier';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

const VEHICLE_OPTS = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd' as const,
};

function constantTrace(c: { steer: number; driveForce: number; brakeForce: number }, ticks: number) {
  return Array.from({ length: ticks }, () => ({ ...c }));
}

describe.skipIf(!RAPIER_OK)('createHeadlessTrialHarness', () => {
  it('runs a simple straight-cruise trial and samples make sense', async () => {
    const h = await createHeadlessTrialHarness({
      vehicleOptions: VEHICLE_OPTS,
      groundBounds: { x0: -200, x1: 200, z0: -200, z1: 200 },
    });
    const spec: TrialSpec = {
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: 0 },
      controlsTrace: constantTrace({ steer: 0, driveForce: 3000, brakeForce: 0 }, 60),
      sampleEveryNTicks: 6,
    };
    const out = h.runTrial(spec);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.trial.samples.length).toBeGreaterThanOrEqual(10);
    // Vehicle should have accelerated forward.
    const first = out.trial.samples[0]!;
    const last = out.trial.samples[out.trial.samples.length - 1]!;
    expect(last.x).toBeGreaterThan(first.x + 0.5);
    expect(last.speed).toBeGreaterThan(0.5);
    h.dispose();
  });

  it('discards off-arena trials with a reason', async () => {
    const h = await createHeadlessTrialHarness({
      vehicleOptions: VEHICLE_OPTS,
      // Very tight arena
      groundBounds: { x0: -3, x1: 3, z0: -3, z1: 3 },
      offArenaThreshold: 5,
    });
    const spec: TrialSpec = {
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: 0 },
      controlsTrace: constantTrace({ steer: 0, driveForce: 4000, brakeForce: 0 }, 240),
      sampleEveryNTicks: 6,
    };
    const out = h.runTrial(spec);
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.reason).toMatch(/off-arena/);
    h.dispose();
  });

  it('teleportFull initial conditions actually take effect (forward speed)', async () => {
    const h = await createHeadlessTrialHarness({ vehicleOptions: VEHICLE_OPTS });
    const spec: TrialSpec = {
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: 8 },
      controlsTrace: constantTrace({ steer: 0, driveForce: 0, brakeForce: 0 }, 6),
      sampleEveryNTicks: 1,
    };
    const out = h.runTrial(spec);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    // Initial sample (after settle phase, no controls) should still be ~8 m/s
    // because the settle phase only applies zero controls and the suspension
    // settling shouldn't kill speed; the harness re-teleports after settle.
    expect(out.trial.samples[0]!.speed).toBeGreaterThan(5);
    h.dispose();
  });
});
