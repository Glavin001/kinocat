// Smoke test for the headless race scenario — used by `pnpm run race`.
//
// We DON'T assert lap completion here because a full lap takes ~40 s sim
// (2400 physics ticks for 2 cars) which is reliably fast locally but
// genuinely too slow for a free GitHub-hosted CI runner under load.
// Instead we drive the scenario for a short fixed budget and assert the
// invariants: physics steps without throwing, cars move, planner emits
// plans, no NaN in the state stream. Full lap-time validation lives in
// the CLI smoke (`pnpm run race:quick`) where the runtime budget is
// per-step (entire job has a long timeout).

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry } from '../app/lib/headless-race';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('createRaceScenario', () => {
  it('ticks for 3 s sim without throwing and the chassis moves', { timeout: 60000 }, async () => {
    const scenario = await createRaceScenario({
      entries: [kinematicEntry('kin')],
      syncHold: false,
      offTrackRecovery: 'spawn',
    });
    const initial = scenario.status()[0]!;
    // Tick 3 s sim = 180 physics ticks. Bounded both ways: enough for the
    // first replan to land (300 ms cadence × multiple) AND fast enough for
    // a slow CI runner.
    for (let i = 0; i < 180; i++) scenario.tick();
    const final = scenario.status()[0]!;
    expect(Number.isFinite(final.state.x)).toBe(true);
    expect(Number.isFinite(final.state.z)).toBe(true);
    expect(Number.isFinite(final.state.speed)).toBe(true);
    // Chassis should have moved at least a metre under throttle.
    const moved = Math.hypot(final.state.x - initial.state.x, final.state.z - initial.state.z);
    expect(moved).toBeGreaterThan(1);
    // Planner should have produced at least one plan in 3 s.
    expect(final.diagnostics.successfulReplans).toBeGreaterThan(0);
    scenario.dispose();
  });

  it('reports per-car status with consistent fields', { timeout: 60000 }, async () => {
    const scenario = await createRaceScenario({
      entries: [kinematicEntry('a'), kinematicEntry('b')],
      syncHold: false,
    });
    for (let i = 0; i < 30; i++) scenario.tick();
    const status = scenario.status();
    expect(status.length).toBe(2);
    for (const s of status) {
      expect(typeof s.name).toBe('string');
      expect(typeof s.loopIndex).toBe('number');
      expect(Array.isArray(s.laps)).toBe(true);
      expect(typeof s.finished).toBe('boolean');
      expect(s.diagnostics.totalReplans).toBeGreaterThanOrEqual(0);
    }
    scenario.dispose();
  });
});
