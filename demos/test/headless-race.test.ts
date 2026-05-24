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
  it('races kinematic + parametric-only entries to one lap each', { timeout: 180000 }, async () => {
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kin'), parametricOnlyEntry('para')],
      targetLaps: 1,
      maxSimTime: 120,
    });
    expect(results.length).toBe(2);
    for (const r of results) {
      expect(r.finished, `${r.name} should finish 1 lap`).toBe(true);
      expect(r.laps.length).toBe(1);
      expect(r.best).toBeGreaterThan(0);
      expect(r.best).toBeLessThan(120);
    }
  });
});
