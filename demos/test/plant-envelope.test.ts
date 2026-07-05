// WS-0 — Plant-envelope regression test.
//
// Re-measures the Rapier raycast-vehicle plant on the headless harness and
// compares to the committed `plant-envelope.json`. Catches plant-tuning drift
// (the `capability-drift.test.ts` pattern, but for MEASURED rather than
// DERIVED quantities). Also documents the key WS-0 finding: the plant has no
// intrinsic top speed (no aero drag), so `RACE_AGENT.maxSpeed = 30` is a
// policy choice, not a physical ceiling.

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { measurePlantEnvelope, type PlantEnvelope } from '../scripts/plant-envelope';

const committedPath = resolve(
  fileURLToPath(new URL('..', import.meta.url)),
  'public/models/plant-envelope.json',
);
const committed = JSON.parse(readFileSync(committedPath, 'utf8')) as PlantEnvelope;

describe('plant envelope (measured, race tuning)', () => {
  it('re-measures within tolerance of the committed artifact', async () => {
    const env = await measurePlantEnvelope();

    // vMax: terminal speed in the 12 s launch — pin tightly.
    expect(Math.abs(env.vMax - committed.vMax)).toBeLessThan(0.5);

    // Brake decel: ±5% per entry speed.
    for (const c of committed.brakeDecel) {
      const m = env.brakeDecel.find((e) => e.entrySpeed === c.entrySpeed)!;
      expect(m).toBeDefined();
      expect(Math.abs(m.decel - c.decel) / c.decel).toBeLessThan(0.05);
    }

    // Cornering lateral-accel boundary: ±5% per (steer, entry).
    for (const c of committed.corneringBoundary) {
      const m = env.corneringBoundary.find(
        (e) => e.steer === c.steer && e.entrySpeed === c.entrySpeed,
      )!;
      expect(m).toBeDefined();
      expect(Math.abs(m.latAccel - c.latAccel) / c.latAccel).toBeLessThan(0.05);
    }

    expect(
      Math.abs(env.maxLateralAccel - committed.maxLateralAccel) / committed.maxLateralAccel,
    ).toBeLessThan(0.05);
  }, 30_000);

  it('confirms the plant has no intrinsic top speed (no aero drag)', () => {
    // Accel stays strongly positive even at the top of the tested range —
    // proof that 30 m/s (RACE_AGENT.maxSpeed) is a POLICY cap, not a plant
    // ceiling. If a future chassis gains drag this flips and the policy cap
    // can be justified physically.
    const aTop = committed.launchCurve.find((p) => p.v === 28)!;
    expect(aTop.a).toBeGreaterThan(5);
    expect(committed.vMax).toBeGreaterThan(40);
  });

  it('measures longitudinal limits well above the analytic derivation', () => {
    // The derived traction accel is ~8.83 m/s² and derived brake ~13.89;
    // the raycast plant measures HIGHER on both (launch ~13.8, brake ≥15),
    // which is exactly why WS-0 measures instead of derives.
    const aLaunch = committed.launchCurve.find((p) => p.v === 0)!;
    expect(aLaunch.a).toBeGreaterThan(10);
    const brake24 = committed.brakeDecel.find((b) => b.entrySpeed === 24)!;
    expect(brake24.decel).toBeGreaterThan(13.89);
  });
});
