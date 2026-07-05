// Closed-loop head-to-head race benchmark — the END-TO-END measurement:
// planner + primitive library + tracker + real Rapier plant, two laps
// of the /raceprimitives course per library, IDENTICAL system for every
// entry (same planner, tracker config, replan cadence, chassis). The
// only variable is the primitive library — i.e. the model. This is the
// harness for "is the learned model actually better", and where all
// model-fidelity work must ultimately show up.
//
// Measured history (2026-07, this branch):
//   before fixes:  kinematic 32.6 s/lap, v2 49.3 s (+51%), 18 failed replans
//   after fixes:   parity on average pace, 0 failed replans both,
//                  v2 prediction error strictly better
// The fixes that closed the gap (all symmetric — applied to BOTH cars):
//   - anticipatory curvature braking in pure-pursuit (previewCurvature,
//     budget 0.8·μg from the derived capability envelope) — eliminated
//     the v2 car's 90°-corner overshoot + 3 s failed-replan U-turn
//   - v2 library decision-cadence parity (0.55 s primitives, matching
//     the kinematic library) and a full-throttle action in the
//     top-speed control set (the planner previously could not plan
//     sustained acceleration above ~23 m/s)
//
// RATCHET: tighten `V2_VS_KIN_AVG_RATIO` toward < 1.0 as feedforward /
// MPPI / on-demand primitive rollout land. Do NOT loosen it to make a
// regression pass.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runHeadlessRace, kinematicEntry, v2Entry } from '../app/lib/headless-race';
import { loadTrainedV2FromDisk } from './helpers/trained-model';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// v2 average lap must be within this factor of kinematic's. History:
// measured 1.51 → 1.28 (executor fixes) → 1.188 after the grip-saturating
// brake model landed (the model's biggest open-loop error — a ~10×
// longitudinal under-brake on the brake-in-turn channel — was fixed, and
// the regenerated in-bounds artifact's honest braking narrowed the
// closed-loop lap-time gap on the OPEN course from ~28% to ~19%). Pinned at
// 1.25 (measured 1.188 + margin for the closed loop's chaotic ±10%
// sensitivity to 1% parameter nudges). The STABLE model-quality signals are
// the failed-replan and prediction-error assertions below. Tighten toward
// < 1.0 as the executor learns to consume the model's speeds (feedforward /
// MPPI / on-demand primitive rollout); NEVER loosen it to pass a regression.
const V2_VS_KIN_AVG_RATIO = 1.25;

describe.skipIf(!RAPIER_OK)('closed-loop head-to-head race benchmark (real Rapier plant)', () => {
  it('trained-v2 library races the kinematic library within the pace ratchet, clean', { timeout: 480_000, retry: 0 }, async () => {
    const model = loadTrainedV2FromDisk();
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kinematic'), v2Entry('v2-trained', model)],
      targetLaps: 2,
      maxSimTime: 150,
      // DETERMINISM: the default 120 ms replan budget is WALL-CLOCK, so
      // machine load changes expansion counts → plans → lap times (B13
      // in the production-readiness review). A large time budget makes
      // the 30k-expansion cap the binding limit, so this benchmark is
      // bit-reproducible on any machine (retry: 0 above enforces it —
      // a flaky pass via vitest's global retry would hide exactly the
      // nondeterminism this guards against).
      tuning: { plannerBudgetMs: 10_000 },
    });
    for (const r of results) {
      console.log(
        `${r.name}: laps=${r.laps.length} best=${r.best.toFixed(1)}s avg=${r.avg.toFixed(1)}s ` +
        `offTrack=${r.offTrackEvents} failedReplans=${r.totalReplans - r.successfulReplans} ` +
        `predErrRms=${r.predErrorRms.toFixed(2)}m`,
      );
    }
    const kin = results.find((r) => r.name === 'kinematic')!;
    const v2 = results.find((r) => r.name === 'v2-trained')!;

    // Both libraries complete 2 laps, clean, healthy planner.
    expect(kin.finished).toBe(true);
    expect(v2.finished).toBe(true);
    expect(kin.offTrackEvents).toBe(0);
    expect(v2.offTrackEvents).toBe(0);
    expect(kin.totalReplans - kin.successfulReplans).toBeLessThanOrEqual(2);
    expect(v2.totalReplans - v2.successfulReplans).toBeLessThanOrEqual(2);

    // Absolute pace health for BOTH cars — the ratio ratchet below must
    // never be satisfied by slowing the kinematic yardstick down.
    // (Measured on the merged executor: kin 33.0 s avg, v2 41.8 s avg.)
    expect(kin.avg).toBeLessThan(40);
    expect(v2.avg).toBeLessThan(50);

    // Head-to-head pace ratchet (measured 1.27) — tighten toward < 1.0,
    // never loosen.
    expect(v2.avg).toBeLessThan(kin.avg * V2_VS_KIN_AVG_RATIO);

    // Closed-loop prediction error sanity for both. NOTE: with curvature
    // feedforward in the executor this metric conflates tracking quality
    // with model quality (tight tracking flatters the kinematic model's
    // closed-loop error), so the strict "v2 < kinematic" model-accuracy
    // claim lives in the OPEN-LOOP harness
    // (model-vs-plant-fidelity.test.ts: 1.08 m vs 6.15 m).
    expect(kin.predErrorRms).toBeLessThan(1.2);
    expect(v2.predErrorRms).toBeLessThan(1.2);
  });
});
