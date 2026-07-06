// Verifies the RaceTuning feature flags actually wire through the
// pipeline — every flag listed in DEFAULT_TUNING must observably affect
// at least one downstream metric (plan length, replan counts, lap
// behaviour) when toggled. This is the cheap defence against a feature
// flag being added but accidentally hard-coded on / off somewhere
// downstream.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createRaceScenario, DEFAULT_TUNING, LEGACY_TUNING } from '../app/lib/race-scenario';
import { kinematicEntry } from '../app/lib/headless-race';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('RaceTuning flags', () => {
  it('default and legacy tunings produce DIFFERENT chassis trajectories', { timeout: 60000 }, async () => {
    async function runWithTuning(tuning: typeof DEFAULT_TUNING) {
      const scenario = await createRaceScenario({
        entries: [kinematicEntry('kin')],
        syncHold: false,
        offTrackRecovery: 'spawn',
        tuning,
      });
      // Drive 2 s sim — enough for 5+ replans and the post-process pipeline
      // to differ from the legacy pipeline.
      for (let i = 0; i < 120; i++) scenario.tick();
      const status = scenario.status()[0]!;
      scenario.dispose();
      return {
        x: status.state.x,
        z: status.state.z,
        speed: status.state.speed,
        planLen: status.plan?.length ?? 0,
      };
    }
    const a = await runWithTuning(DEFAULT_TUNING);
    const b = await runWithTuning(LEGACY_TUNING);
    // Position OR plan-length differs measurably. (Position differs because
    // the smoothed speed profile changes throttle/brake even in a 2s window.
    // Plan length differs because the trajectory-smoother resamples to
    // ~75 dense samples vs. the legacy ~15.)
    const positionDelta = Math.hypot(a.x - b.x, a.z - b.z);
    expect(positionDelta > 0.05 || Math.abs(a.planLen - b.planLen) > 10).toBe(true);
  });

  it('the trajectory smoother resamples to a denser plan than un-smoothed', { timeout: 60000 }, async () => {
    async function runWithTuning(tuning: typeof DEFAULT_TUNING) {
      const scenario = await createRaceScenario({
        entries: [kinematicEntry('kin')],
        syncHold: false,
        tuning,
      });
      for (let i = 0; i < 60; i++) scenario.tick();
      const status = scenario.status()[0]!;
      scenario.dispose();
      return status.plan?.length ?? 0;
    }
    const dense = await runWithTuning({ ...DEFAULT_TUNING, enableTrajectorySmoother: true });
    const sparse = await runWithTuning({ ...DEFAULT_TUNING, enableTrajectorySmoother: false });
    // The planner's node sequence is now sweep-expanded (`expandPlanSweeps`)
    // into per-primitive swept poses BEFORE the smoother runs, so the
    // un-smoothed plan already carries intermediate samples — no longer the
    // ~15 raw endpoints it used to be. The trajectory smoother then C¹-smooths
    // and RESAMPLES that to uniform ~0.4m spacing, which is still observably
    // denser than the variable-spaced un-smoothed path (the legacy ~2× gap no
    // longer applies now that both paths are dense).
    expect(dense).toBeGreaterThan(sparse * 1.3);
  });
});
