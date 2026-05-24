// Smoke test for the headless race scenario — used by `pnpm run race`.
// We require the KINEMATIC entry to complete a lap; other entries
// (parametric-only, trained v2) are exercised by the CLI itself.
//
// Lap times are not asserted because they depend on platform-specific
// Rapier WASM floating-point determinism — the contract is "race
// finishes within the budget".

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runHeadlessRace, kinematicEntry } from '../app/lib/headless-race';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('runHeadlessRace', () => {
  // Single-entry kinematic test. The kinematic entry uses the simplest
  // primitive library (no learned model) so its lap time is the most
  // deterministic baseline — locally ~40s sim, ~1.5s wall. On a slower
  // CI runner the wall time scales roughly with the sim/wall ratio but
  // the 240 s vitest budget gives plenty of margin.
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
});
