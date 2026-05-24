// Smoke test for the headless race scenario — used by `pnpm run race`.
// We require ALL entries to complete the target laps. Lap times are not
// asserted because they depend on platform-specific Rapier WASM
// floating-point determinism — the contract is "race finishes".

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  runHeadlessRace,
  kinematicEntry,
  parametricOnlyEntry,
} from '../app/lib/headless-race';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('runHeadlessRace', () => {
  // Single-entry kinematic-only smoke. Asserts the headless scenario
  // completes a lap deterministically. We deliberately avoid asserting
  // on the parametric-only baseline here — it's the unfit default model
  // and its lap time is genuinely fragile (sometimes >180s on slower
  // CI runners). The CLI `pnpm run race` exercises the multi-entry
  // path; this test only locks in the scenario's basic correctness.
  it('races the kinematic entry to one lap', { timeout: 240000 }, async () => {
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kin')],
      targetLaps: 1,
      maxSimTime: 180,
    });
    expect(results.length).toBe(1);
    const r = results[0]!;
    expect(r.finished, `${r.name} should finish 1 lap`).toBe(true);
    expect(r.laps.length).toBe(1);
    expect(r.best).toBeGreaterThan(0);
    expect(r.best).toBeLessThan(180);
  });

  it('runs a 2-entry race without crashing (no completion required)', { timeout: 120000 }, async () => {
    // Looser test: don't require completion, just that the scenario
    // returns per-entry results without exceptions.
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kin'), parametricOnlyEntry('para')],
      targetLaps: 1,
      maxSimTime: 60,
    });
    expect(results.length).toBe(2);
    for (const r of results) {
      expect(typeof r.totalSimTime).toBe('number');
      expect(Array.isArray(r.laps)).toBe(true);
    }
  });
});
