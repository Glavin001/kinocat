// Closed-loop head-to-head on the TECHNICAL race course (walls, guard
// blocks, thread-the-gate chicane). Same end-to-end system for both cars —
// planner + primitive library + tracker + real Rapier plant — the ONLY
// variable is the primitive library (i.e. the model). The technical course
// adds physical consequence to corner overshoot (a wall strike), which is
// where model fidelity is supposed to pay off.
//
// The technical course auto-enables the friction-circle speed profile +
// curvature feedforward (see createRaceScenario): geometry-only pursuit
// wedges the chassis against a wall on overshoot (a failed-replan storm),
// while the speed profile brakes into corners so BOTH cars thread the walls
// cleanly. This test pins the CLEAN, both-complete, deterministic result and
// the pace ratchet; tighten the ratchet toward < 1.0 as the executor learns
// to consume the model's speeds (feedforward / MPPI), never loosen it.

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

// Measured 2026-07 (deterministic, expansion-capped): kin 38.4 s avg,
// v2 43.1 s avg → ratio 1.12. Pinned at 1.25 with margin for the closed
// loop's chaotic sensitivity. Tighten toward < 1.0, never loosen.
const V2_VS_KIN_TECH_RATIO = 1.25;

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
      console.log(
        `${r.name}: laps=${r.laps.length} best=${r.best.toFixed(1)}s avg=${r.avg.toFixed(1)}s ` +
        `wallStrikes=${r.wallStrikes} offTrack=${r.offTrackEvents} ` +
        `failedReplans=${r.totalReplans - r.successfulReplans} predErrRms=${r.predErrorRms.toFixed(2)}m`,
      );
    }
    const kin = results.find((r) => r.name === 'kinematic')!;
    const v2 = results.find((r) => r.name === 'v2-trained')!;

    // Both complete the walled course — it is a fair, drivable circuit for
    // both libraries, not a DNF trap.
    expect(kin.finished).toBe(true);
    expect(v2.finished).toBe(true);

    // Clean: with the technical course's speed profile both cars thread the
    // walls without striking them and without a replan storm.
    expect(kin.wallStrikes).toBeLessThanOrEqual(1);
    expect(v2.wallStrikes).toBeLessThanOrEqual(1);
    expect(kin.offTrackEvents).toBe(0);
    expect(v2.offTrackEvents).toBe(0);
    expect(kin.totalReplans - kin.successfulReplans).toBeLessThanOrEqual(2);
    expect(v2.totalReplans - v2.successfulReplans).toBeLessThanOrEqual(2);

    // Absolute pace health (the ratio ratchet must never be satisfied by
    // slowing the kinematic yardstick down).
    expect(kin.avg).toBeLessThan(50);
    expect(v2.avg).toBeLessThan(55);

    // Head-to-head pace ratchet on the walled course.
    expect(v2.avg).toBeLessThan(kin.avg * V2_VS_KIN_TECH_RATIO);
  });
});
