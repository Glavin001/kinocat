// Pins the contract of the streaming `TrainingEvent` sequence emitted by
// the offline + maneuver training drivers.
//
// What this guards against:
//   * The `phase` event sequence is what drives the Model Lab ProgressCard.
//     A regression that drops the early `initializing` event would put the
//     UI back into the "frozen on STARTING" failure mode the merged work
//     fixed.
//   * `collectManeuverBatch` streams incremental `trial-batch` deltas via
//     its callback; the surrounding driver MUST NOT also emit a redundant
//     summary `trial-batch` for the same round, or the UI accumulator
//     would double-count. This regression was hit during the merge and is
//     exactly what this test pins down.
//   * DAgger is gated behind `daggerStartRound` (default `Infinity`). It
//     dynamic-imports an expensive race scenario module and triggers a
//     full closed-loop run; turning it on accidentally would massively
//     slow the default training path.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  runOfflineTraining,
  runManeuverTraining,
  type TrainingEvent,
  type TrainingPhase,
} from '../app/lib/training-driver';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// Helpers --------------------------------------------------------------------

function phasesFor(events: TrainingEvent[], round: number): TrainingPhase[] {
  return events
    .filter((e): e is Extract<TrainingEvent, { type: 'phase' }> => e.type === 'phase')
    .filter((e) => e.round === round || e.phase === 'initializing')
    .map((e) => e.phase);
}

function trialBatchesFor(events: TrainingEvent[], round: number) {
  return events.filter(
    (e): e is Extract<TrainingEvent, { type: 'trial-batch' }> =>
      e.type === 'trial-batch' && e.round === round,
  );
}

// ----------------------------------------------------------------------------

describe.skipIf(!RAPIER_OK)('TrainingEvent stream — runOfflineTraining', () => {
  it('emits `phase` events in the documented order (initializing → collecting → parametric → residual → evaluating) per round', { timeout: 120000 }, async () => {
    const events: TrainingEvent[] = [];
    await runOfflineTraining({
      rounds: 2,
      trialsPerActiveRound: 8,
      trialTicks: 30, // ~0.5s/trial
      sampleEveryNTicks: 6,
      seed: 99,
      onEvent: (e) => events.push(e),
    });

    // The early `initializing` phase fires BEFORE any round-start, so the
    // ProgressCard can paint a non-frozen state while Rapier WASM warms up.
    const firstPhase = events.find((e): e is Extract<TrainingEvent, { type: 'phase' }> => e.type === 'phase');
    expect(firstPhase?.phase).toBe('initializing');
    const firstPhaseIdx = events.indexOf(firstPhase!);
    const firstRoundStartIdx = events.findIndex((e) => e.type === 'round-start');
    expect(firstPhaseIdx).toBeLessThan(firstRoundStartIdx);

    // Per-round phase sequence: collecting, parametric, residual, evaluating.
    for (const round of [0, 1]) {
      const ph = phasesFor(events, round).filter((p) => p !== 'initializing');
      expect(ph).toEqual(['collecting', 'parametric', 'residual', 'evaluating']);
    }
  });
});

// ----------------------------------------------------------------------------

describe.skipIf(!RAPIER_OK)('TrainingEvent stream — runManeuverTraining', () => {
  it('does not double-count trial-batch events (collectManeuverBatch deltas only)', { timeout: 90000 }, async () => {
    const events: TrainingEvent[] = [];
    const result = await runManeuverTraining({
      rounds: 1,
      trialsPerRound: 16,
      trialTicks: 30,
      sampleEveryNTicks: 6,
      seed: 5,
      onEvent: (e) => events.push(e),
    });

    const batches = trialBatchesFor(events, 0);
    expect(batches.length).toBeGreaterThan(0);

    // The sum of `delta.collected` across all trial-batch events for the
    // round MUST equal the final number of trials in the store. If a
    // regression re-introduced a "summary" event after `collectManeuverBatch`
    // (in addition to its incremental deltas), this sum would be ~2× the
    // actual count.
    const sumCollected = batches.reduce((a, b) => a + b.collected, 0);
    expect(sumCollected).toBe(result.trials.size());

    // `runSoFar` is monotonically non-decreasing and matches the last
    // delta's `totalSoFar`.
    let prev = -1;
    for (const b of batches) {
      expect(b.runSoFar ?? 0).toBeGreaterThanOrEqual(prev);
      prev = b.runSoFar ?? prev;
    }
    const last = batches[batches.length - 1]!;
    expect(last.runTarget).toBe(16);
    expect(last.runSoFar).toBe(16);
  });

  it('emits a `coverage` event once per round, after `evaluation`', { timeout: 90000 }, async () => {
    const events: TrainingEvent[] = [];
    await runManeuverTraining({
      rounds: 1,
      trialsPerRound: 12,
      trialTicks: 30,
      sampleEveryNTicks: 6,
      seed: 11,
      onEvent: (e) => events.push(e),
    });
    const coverageEvents = events.filter((e) => e.type === 'coverage');
    expect(coverageEvents.length).toBe(1);
    const evalIdx = events.findIndex((e) => e.type === 'evaluation');
    const covIdx = events.findIndex((e) => e.type === 'coverage');
    expect(covIdx).toBeGreaterThan(evalIdx);
  });

  it('does NOT load or run DAgger by default (daggerStartRound omitted ⇒ Infinity)', { timeout: 90000 }, async () => {
    // The DAgger code path dynamically imports `race-scenario-collect` and
    // emits a trial-batch event with `runTarget === undefined` (the race
    // collector has no incremental progress). If gating breaks, that DAgger
    // trial-batch will appear in the stream.
    const events: TrainingEvent[] = [];
    await runManeuverTraining({
      rounds: 1,
      trialsPerRound: 12,
      trialTicks: 30,
      sampleEveryNTicks: 6,
      seed: 17,
      onEvent: (e) => events.push(e),
      // daggerStartRound intentionally omitted — must default to Infinity.
    });
    const batches = trialBatchesFor(events, 0);
    // Every batch must come from collectManeuverBatch's incremental
    // deltas, which always set `runTarget`.
    for (const b of batches) {
      expect(b.runTarget).toBeDefined();
    }
  });
});
