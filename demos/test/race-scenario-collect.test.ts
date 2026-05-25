// Smoke test for the Phase 3 closed-loop collector. Drives the race
// scenario for a short fixed budget and asserts the collector emits
// trials with the expected schema (scenarioId, split, sample shape).
// Lap completion is NOT asserted — the budget here is too small to
// finish a lap; the test only validates the data-collection path.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  buildKinematicLibrary,
} from '../app/lib/race-primitives-scenarios';
import { collectFromRaceScenario } from '../app/lib/race-scenario-collect';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('collectFromRaceScenario', () => {
  it('emits scenario-tagged trials with the expected schema', { timeout: 60000 }, async () => {
    const lib = buildKinematicLibrary();
    const result = await collectFromRaceScenario({
      lib,
      targetLaps: 99, // unreachable; bounded by maxSimTime instead
      maxSimTime: 4, // 4s sim — produces ~4 trials at windowSec=1
      windowSec: 1.0,
      sampleEveryNTicks: 6,
      scenarioId: 'test-race',
    });
    expect(result.trials.length).toBeGreaterThan(0);
    for (const t of result.trials) {
      expect(t.scenarioId).toBe('test-race');
      expect(t.maneuverId).toBe('scenario');
      expect(t.split).toBeDefined();
      expect(t.controlsTrace.length).toBeGreaterThan(0);
      expect(t.samples.length).toBeGreaterThanOrEqual(2);
      // First sample is at t=0.
      expect(t.samples[0]!.t).toBeCloseTo(0, 6);
    }
  });
});
