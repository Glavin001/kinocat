// Smoke test for the maneuver-based training pipeline used by
// `pnpm run train`. Verifies that:
//   - one round completes without errors
//   - trials carry maneuverId + maneuverParams + split assignments
//   - the model can round-trip through `modelToJson` / `modelFromJson`

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runManeuverTraining } from '../app/lib/training-driver';
import { modelToJson, modelFromJson } from '../app/lib/v2-model-file';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('runManeuverTraining', () => {
  it('runs one round + emits maneuver-tagged trials', { timeout: 90000 }, async () => {
    const result = await runManeuverTraining({
      rounds: 1,
      trialsPerRound: 16,
      trialTicks: 60,
      sampleEveryNTicks: 6,
      seed: 1,
    });
    expect(result.trials.size()).toBeGreaterThanOrEqual(8);
    const sample = result.trials.all()[0]!;
    expect(sample.maneuverId).toBeDefined();
    expect(sample.maneuverParams).toBeDefined();
    expect(sample.split).toBeDefined();
    // Roughly equal split between train/val/test partitions (within 50% of
    // the 70/15/15 expectation given small N).
    const trainCount = result.trials.all('train').length;
    const total = result.trials.size();
    expect(trainCount / total).toBeGreaterThan(0.3);
  });

  it('round-trips the trained model through file persistence', { timeout: 90000 }, async () => {
    const result = await runManeuverTraining({
      rounds: 1,
      trialsPerRound: 12,
      trialTicks: 30,
      sampleEveryNTicks: 6,
      seed: 7,
    });
    const json = modelToJson(result.model, {
      trialsUsed: result.trials.size(),
      openLoopRmsAt1s: 0,
      createdAt: 0,
    });
    const back = modelFromJson(json);
    expect(back.params).toEqual(result.model.params);
    expect(back.residualEnsemble?.length ?? 0).toBe(result.model.residualEnsemble?.length ?? 0);
  });
});
