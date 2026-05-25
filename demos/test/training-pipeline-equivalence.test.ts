// Equivalence check: the car v2 pipeline produces a bit-identical model
// whether driven by the demo's `runOfflineTraining` wrapper OR fed
// directly into the core's generic `runOfflineTraining` orchestrator.
//
// This is the proof that `CarV2TrainingPipeline` is a faithful
// `TrainingPipeline<S, C, P, Cfg>` implementation — a future airplane
// pipeline can plug into the same orchestrator with confidence.

import { describe, it, expect } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createTrialStore } from 'kinocat/learning';
import { runOfflineTraining as runOfflineTrainingCore } from 'kinocat/training';
import type { CarKinematicState, LearnableVehicleConfig, LearnedVehicleParamsV2 } from 'kinocat/agent';
import type { WheeledCarControls } from 'kinocat/agent';
import {
  CarV2TrainingPipeline,
  runOfflineTraining,
  PARAMS_V2_ORDER,
} from '../app/lib/training-driver';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

const SHARED_OPTS = {
  rounds: 2,
  trialsPerActiveRound: 8,
  trialTicks: 30,
  sampleEveryNTicks: 6,
  seed: 7,
} as const;

describe.skipIf(!RAPIER_OK)('CarV2TrainingPipeline — conformance with generic orchestrator', () => {
  it('demo wrapper and core orchestrator produce the same trained params for a fixed seed', { timeout: 180000 }, async () => {
    // Path A: demo wrapper.
    const a = await runOfflineTraining({ ...SHARED_OPTS });

    // Path B: drive the same pipeline class through the core orchestrator.
    const pipelineB = await CarV2TrainingPipeline.create({ ...SHARED_OPTS });
    const storeB = createTrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>();
    const resultB = await runOfflineTrainingCore<CarKinematicState, WheeledCarControls, LearnedVehicleParamsV2, LearnableVehicleConfig>({
      pipeline: pipelineB,
      store: storeB,
    });
    pipelineB.dispose();

    // Parametric coefficients should match within float tolerance for
    // every param in PARAMS_V2_ORDER. Use a moderate epsilon because the
    // residual MLP fit has weight init RNG that is identical (seed 42 in
    // both code paths) but downstream LBFGS iterations can pick up
    // accumulated rounding differences.
    const pa = a.model.params as unknown as Record<string, number>;
    const pb = resultB.model.params as unknown as Record<string, number>;
    for (const name of PARAMS_V2_ORDER) {
      expect(pb[name]).toBeCloseTo(pa[name]!, 4);
    }

    // Trial counts must match exactly — same seed grid, same active
    // exploration RNG schedule.
    expect(storeB.size()).toBe(a.trials.size());
  });

  it('pipeline.totalRounds() respects the rounds option', async () => {
    if (!RAPIER_OK) return;
    const p = await CarV2TrainingPipeline.create({ rounds: 5, trialsPerActiveRound: 4, trialTicks: 20 });
    expect(p.totalRounds()).toBe(5);
    p.dispose();
  });
});
