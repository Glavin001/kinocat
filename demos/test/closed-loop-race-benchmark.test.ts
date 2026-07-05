// Closed-loop race benchmark — the END-TO-END measurement: planner +
// primitive library + tracker + real Rapier plant, one full lap, per
// library. This is where model-fidelity work must ultimately show up;
// open-loop model accuracy (model-vs-plant-fidelity.test.ts) is a
// necessary but not sufficient condition.
//
// Measured baselines (2026-07, shipping DEFAULT_TUNING):
//   kinematic: 1 lap in ~32.6 s, 0 off-track, 0 failed replans
//   v2 trained: 1 lap in ~49.3 s, 0 off-track, ~8% failed replans
//
// Honest status the budgets encode: the v2 library CLOSES a lap cleanly
// but is currently SLOWER closed-loop than the kinematic library,
// because the default tracker re-derives speed itself (plan speeds are
// not consumed for racing — measured: enabling respectPathSpeed dropped
// BOTH cars to sub-5 m/s crawls, since race plan node speeds are not
// yet trustworthy targets). Closing that gap (feedforward, trustworthy
// per-node speeds, MPPI launch fix) is the remaining roadmap work; when
// it lands, TIGHTEN the v2 budget below toward the kinematic one — that
// ratchet is the point of this file.

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

describe.skipIf(!RAPIER_OK)('closed-loop race benchmark (real Rapier plant)', () => {
  it('kinematic and trained-v2 libraries each complete a clean lap within budget', { timeout: 300_000 }, async () => {
    const model = loadTrainedV2FromDisk();
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kinematic'), v2Entry('v2-trained', model)],
      targetLaps: 1,
      maxSimTime: 90,
    });
    for (const r of results) {
      console.log(
        `${r.name}: lap=${r.best.toFixed(1)}s offTrack=${r.offTrackEvents} ` +
        `replans=${r.successfulReplans}/${r.totalReplans} predErrRms=${r.predErrorRms.toFixed(2)}m`,
      );
    }
    const kin = results.find((r) => r.name === 'kinematic')!;
    const v2 = results.find((r) => r.name === 'v2-trained')!;

    // Both libraries must finish a lap, clean.
    expect(kin.finished).toBe(true);
    expect(v2.finished).toBe(true);
    expect(kin.offTrackEvents).toBe(0);
    expect(v2.offTrackEvents).toBe(0);

    // Lap-time budgets (measured + headroom; tighten as fidelity lands).
    expect(kin.best).toBeLessThan(45);
    expect(v2.best).toBeLessThan(70);

    // Planner health: failed replans are the fallback-creep engine.
    const v2FailRatio = 1 - v2.successfulReplans / Math.max(1, v2.totalReplans);
    expect(v2FailRatio).toBeLessThan(0.15);
    expect(kin.successfulReplans).toBeGreaterThan(0);
  });
});
