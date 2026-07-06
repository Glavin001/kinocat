// Closed-loop head-to-head on the TECHNICAL race course (walls, guard
// blocks, thread-the-gate chicane). Same end-to-end system for both cars —
// planner + primitive library + tracker + real Rapier plant — the ONLY
// variable is the primitive library (i.e. the model). The technical course
// adds physical consequence to corner overshoot (a wall strike), which is
// where model fidelity is supposed to pay off.
//
// The technical course auto-enables the friction-circle speed profile +
// curvature feedforward (see createRaceScenario). After WS-1 (faithful speed
// execution: no phantom horizon braking, bang-bang throttle + coast band,
// envelope-raised brake authority) the two cars SEPARATE on this walled
// course — which is the whole point of the technical variant:
//
//   - the v2 car threads the gates clean and fast (its honest model plans
//     corner speeds the chassis can actually hold), and
//   - the kinematic delusion over-drives corners into the walls (strikes +
//     a failed-replan lap), paying a real physical cost.
//
// So this test now pins the D5 result: v2 completes clean and STRICTLY beats
// kinematic on lap time (ratio < 1.0). The kinematic car is allowed wall
// strikes — that consequence is the experiment. (Pre-WS-1 this test pinned
// "both clean, v2 within 1.25×"; the executor no longer conceals the
// fidelity gap, so the assertion follows the physics.)

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runHeadlessRace, kinematicEntry, v2Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { loadTrainedV2FromDisk } from './helpers/trained-model';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// Measured after WS-1 (deterministic, expansion-capped): v2 35.9 s avg,
// kin 44.4 s avg → ratio 0.81. v2 now STRICTLY beats kinematic on the walled
// course. Pinned at 0.90 (measured 0.81 + margin for the closed loop's
// chaotic ±10% sensitivity). Tighten toward the measured value, never loosen.
// History: 1.12 (pre-WS-1) → 0.81 (WS-1).
const V2_VS_KIN_TECH_RATIO = 0.9;

describe.skipIf(!RAPIER_OK)('technical-course closed-loop benchmark (walls, real Rapier plant)', () => {
  it('both libraries complete the walled course cleanly; v2 within the pace ratchet', { timeout: 480_000, retry: 0 }, async () => {
    const model = loadTrainedV2FromDisk();
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kinematic'), v2Entry('v2-trained', model)],
      targetLaps: 2,
      maxSimTime: 180,
      course: buildRaceCourse('technical'),
      // DETERMINISM: large budget so the 30k-expansion cap binds first, not
      // wall-clock load. retry:0 (above) forbids a flaky vitest auto-pass.
      tuning: { plannerBudgetMs: 10_000 },
    });
    for (const r of results) {
      const q = r.quality;
      console.log(
        `${r.name}: laps=${r.laps.length} best=${r.best.toFixed(1)}s avg=${r.avg.toFixed(1)}s ` +
        `wallStrikes=${r.wallStrikes} offTrack=${r.offTrackEvents} ` +
        `failedReplans=${r.totalReplans - r.successfulReplans} predErrRms=${r.predErrorRms.toFixed(2)}m ` +
        `| dist/lap=${(q.distanceTravelled / Math.max(1, r.laps.length)).toFixed(0)}m ` +
        `meanSpd=${q.meanSpeed.toFixed(1)} ggMean=${q.ggMeanUtil.toFixed(3)} ` +
        `stopped=${q.timeStopped.toFixed(1)}s recov=${q.recoveryCount}`,
      );
    }
    const kin = results.find((r) => r.name === 'kinematic')!;
    const v2 = results.find((r) => r.name === 'v2-trained')!;

    // Both complete the walled course — it is a fair, drivable circuit for
    // both libraries, not a DNF trap.
    expect(kin.finished).toBe(true);
    expect(v2.finished).toBe(true);

    // v2 is CLEAN: its honest model plans corner speeds the chassis holds, so
    // it threads the gates without striking them and without a replan storm.
    expect(v2.wallStrikes).toBeLessThanOrEqual(1);
    expect(v2.offTrackEvents).toBe(0);
    expect(v2.totalReplans - v2.successfulReplans).toBeLessThanOrEqual(3);

    // The kinematic delusion pays a physical cost on this course: it is
    // ALLOWED to strike walls (that consequence is the experiment). We only
    // assert it strikes AT LEAST as many as v2 — the fidelity gap made
    // physical — and does not silently go clean (which would mean the course
    // stopped separating the models). Pinned loosely for chaotic sensitivity.
    expect(kin.wallStrikes).toBeGreaterThanOrEqual(v2.wallStrikes);

    // Absolute pace health (the ratio ratchet must never be satisfied by
    // slowing the kinematic yardstick down).
    expect(kin.avg).toBeLessThan(55);
    expect(v2.avg).toBeLessThan(45);

    // Head-to-head: v2 STRICTLY beats kinematic on the walled course (D5).
    expect(v2.avg).toBeLessThan(kin.avg * V2_VS_KIN_TECH_RATIO);

    // Driving-quality sanity: the accumulators are populated and physical.
    // (These are the "how well is it driving" measurements beyond lap time:
    // line efficiency, tire utilization, hesitation.)
    for (const r of [kin, v2]) {
      expect(r.quality.distanceTravelled).toBeGreaterThan(300); // ≥ ~1 lap of course
      expect(r.quality.meanSpeed).toBeGreaterThan(3);
      expect(r.quality.ggMeanUtil).toBeGreaterThan(0.05);
      expect(r.quality.ggMeanUtil).toBeLessThan(1.0);
      expect(r.quality.ggPeakUtil).toBeLessThanOrEqual(1.5); // clamp bound
    }
  });
});
