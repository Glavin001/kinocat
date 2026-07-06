// Closed-loop race benchmark for the PURELY LEARNED v3 dynamics model
// (docs/v3-purely-learned-model.md): same planner + pure-pursuit tracker +
// real Rapier plant as the kinematic baseline; the only variable is the
// primitive library (i.e. the model).
//
// What this pins (measured 2026-07, deterministic expansion-capped config):
//   - v3 COMPLETES the open course CLEAN (no wall strikes, no off-track,
//     no recovery teleports) — the honest model's plans are drivable.
//   - v3 stays within a pace ratchet of the kinematic baseline
//     (measured 44.0 s vs 37.6 s avg → ratio 1.17).
//
// Why v3 does not yet WIN on lap time despite 1.7× better open-loop
// fidelity than v2 (0.63 m vs 1.07 m endpoint error) and 6.6× better than
// kinematic: the pure-pursuit executor tracks plan GEOMETRY and re-derives
// speed from curvature — the model's honest dynamics knowledge (planned
// controls, transient-aware speeds) never reaches the actuators, and the
// kinematic library's delusionally tight arcs still produce shorter lines
// that pure-pursuit patches over at this course's consequence-free corner
// speeds. That executor gap is exactly the pending WS-1½ (control
// feedforward) / WS-3 (MPPI) work in docs/max-pace-roadmap.md. This
// benchmark exists so that when the executor starts consuming model
// knowledge, v3's fidelity advantage becomes measurable here — tighten the
// ratchet as it lands; never loosen it.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runHeadlessRace, kinematicEntry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { loadTrainedV3FromDisk } from './helpers/trained-model';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// Measured 1.17 (v3 44.0 s / kin 37.6 s avg over 2 laps). Headroom for the
// closed loop's chaotic ±10% sensitivity. History: 1.35 (v3 introduction).
const V3_VS_KIN_RATIO = 1.35;

describe.skipIf(!RAPIER_OK)('v3 closed-loop race benchmark (open course, real Rapier plant)', () => {
  it('v3 completes the open course clean and within the pace ratchet', { timeout: 480_000, retry: 0 }, async () => {
    const model = loadTrainedV3FromDisk();
    const results = await runHeadlessRace({
      entries: [kinematicEntry('kinematic'), v3Entry('v3-learned', model)],
      targetLaps: 2,
      maxSimTime: 180,
      course: buildRaceCourse(),
      // DETERMINISM: large budget so the 30k-expansion cap binds first,
      // not wall-clock load. retry:0 forbids a flaky vitest auto-pass.
      tuning: { plannerBudgetMs: 10_000 },
    });
    for (const r of results) {
      const q = r.quality;
      console.log(
        `${r.name}: laps=${r.laps.length} best=${r.best.toFixed(1)}s avg=${r.avg.toFixed(1)}s ` +
        `walls=${r.wallStrikes} offTrack=${r.offTrackEvents} ` +
        `failedReplans=${r.totalReplans - r.successfulReplans} predErrRms=${r.predErrorRms.toFixed(2)}m ` +
        `| dist/lap=${(q.distanceTravelled / Math.max(1, r.laps.length)).toFixed(0)}m ` +
        `meanSpd=${q.meanSpeed.toFixed(1)} recov=${q.recoveryCount}`,
      );
    }
    const kin = results.find((r) => r.name === 'kinematic')!;
    const v3 = results.find((r) => r.name === 'v3-learned')!;

    expect(kin.finished).toBe(true);
    expect(v3.finished).toBe(true);

    // v3 drives CLEAN: honest plans are physically drivable end-to-end.
    expect(v3.wallStrikes).toBe(0);
    expect(v3.offTrackEvents).toBe(0);
    expect(v3.quality.recoveryCount).toBe(0);

    // Pace ratchet vs the kinematic baseline.
    expect(v3.avg / kin.avg).toBeLessThan(V3_VS_KIN_RATIO);
  });
});
