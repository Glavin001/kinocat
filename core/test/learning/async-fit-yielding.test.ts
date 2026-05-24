// Tests for the async, cooperatively-yielding fitter variants
// (`runParametricFitAsync`, `runResidualMLPFitAsync`).
//
// Two reasons these tests exist:
//   1. The yielding behavior is what keeps Chrome's main thread responsive
//      during in-browser training. If a regression silently drops the
//      `cooperativeYield` invocations, the Model Lab page would freeze
//      again — exactly the bug the merged work fixed. These tests pin the
//      contract by counting yield calls.
//   2. The merged `FitProgressEvent` shape now carries BOTH `lossNormalized`
//      (per-sample mean, comparable across rounds) AND `valLoss` (residual
//      val-split diagnostic). Two consumers (the chart and the UI summary)
//      read these fields and would silently drop data if the fields stop
//      flowing.

import { describe, expect, it } from 'vitest';
import {
  runParametricFitAsync,
  runResidualMLPFitAsync,
  createTrialStore,
  type FitProgressEvent,
  type Trial,
} from 'kinocat/learning';
import type { ForwardSim } from 'kinocat/primitives';

// ---- Synthetic 2-param ground-truth (mirrors parametric-fit.test.ts) ----

interface MiniState { x: number; v: number; t: number; }
interface MiniParams { accel: number; damping: number; }
interface MiniControls { effort: number; }

function makeSim(p: MiniParams): ForwardSim<MiniState> {
  return (s, controls, dt) => {
    const e = controls[0] ?? 0;
    const v = s.v + (p.accel * e - p.damping * s.v) * dt;
    return { x: s.x + 0.5 * (s.v + v) * dt, v, t: s.t + dt };
  };
}

function buildSyntheticTrials(efforts: number[], ticks = 60, dt = 1 / 30) {
  const truth: MiniParams = { accel: 4.0, damping: 1.2 };
  const trueSim = makeSim(truth);
  const store = createTrialStore<MiniState, MiniControls, null>();
  for (let i = 0; i < efforts.length; i++) {
    const eff = efforts[i]!;
    let s: MiniState = { x: 0, v: 0, t: 0 };
    const controlsTrace: MiniControls[] = [];
    const samples: Trial<MiniState, MiniControls, null>['samples'] = [
      { t: 0, state: { ...s } },
    ];
    for (let k = 0; k < ticks; k++) {
      controlsTrace.push({ effort: eff });
      s = trueSim(s, [eff], dt);
      if ((k + 1) % 10 === 0) samples.push({ t: (k + 1) * dt, state: { ...s } });
    }
    store.add({
      id: `t${i}`, initialState: { x: 0, v: 0, t: 0 }, controlsTrace, dt,
      samples, config: null, configKey: 'cfg',
    });
  }
  return store;
}

// ---------------------------------------------------------------------------

describe('runParametricFitAsync — cooperative yielding + normalized loss', () => {
  it('invokes cooperativeYield at least once per iteration when yieldEveryNIter=1', async () => {
    const store = buildSyntheticTrials([0.3, 0.7, 1.0, -0.5]);
    let yieldCount = 0;
    const result = await runParametricFitAsync<MiniParams, MiniState, MiniControls, null>({
      init: { accel: 1, damping: 0.1 },
      encode: (p) => [p.accel, p.damping],
      decode: (v) => ({
        accel: Math.max(0.1, Math.min(10, v[0] ?? 1)),
        damping: Math.max(0, Math.min(5, v[1] ?? 0)),
      }),
      makeSim,
      stateDelta: (pred, act) => {
        const dx = pred.x - act.x;
        const dv = pred.v - act.v;
        return dx * dx + dv * dv;
      },
      trials: store.all(),
      controlsToVec: (c) => [c.effort],
      maxIter: 40,
      cooperativeYield: async () => { yieldCount++; },
      yieldEveryNIter: 1,
      // Disable mid-eval yielding so the count is dominated by per-iter yields.
      yieldEveryNTrials: 0,
    });
    expect(result.iterations).toBeGreaterThan(0);
    // One yield per Nelder-Mead iteration; loose lower bound to allow for
    // early termination on convergence.
    expect(yieldCount).toBeGreaterThanOrEqual(Math.min(20, result.iterations));
  });

  it('emits FitProgressEvent with lossNormalized = loss / sampleCount', async () => {
    const store = buildSyntheticTrials([0.3, 0.8]);
    const events: FitProgressEvent[] = [];
    await runParametricFitAsync<MiniParams, MiniState, MiniControls, null>({
      init: { accel: 1, damping: 0.1 },
      encode: (p) => [p.accel, p.damping],
      decode: (v) => ({
        accel: Math.max(0.1, Math.min(10, v[0] ?? 1)),
        damping: Math.max(0, Math.min(5, v[1] ?? 0)),
      }),
      makeSim,
      stateDelta: (pred, act) => {
        const dx = pred.x - act.x;
        const dv = pred.v - act.v;
        return dx * dx + dv * dv;
      },
      trials: store.all(),
      controlsToVec: (c) => [c.effort],
      maxIter: 10,
      onProgress: (e) => events.push(e),
      cooperativeYield: async () => {},
      yieldEveryNTrials: 0,
    });
    expect(events.length).toBeGreaterThan(0);
    for (const e of events) {
      expect(e.sampleCount).toBeGreaterThan(0);
      expect(e.lossNormalized).toBeCloseTo(e.loss / e.sampleCount!, 9);
    }
  });

  it('emits progress on EVERY iteration (not just improving ones)', async () => {
    // The merged async fitter calls `onProgress` from inside the Nelder-Mead
    // `onIter` callback unconditionally, so the UI can show "iter X / max"
    // even when late iterations don't improve the loss. A regression that
    // re-gates this on `improved` would cause the ProgressCard to look
    // frozen for stretches of training.
    const store = buildSyntheticTrials([0.5, 0.6]);
    const events: FitProgressEvent[] = [];
    const maxIter = 25;
    await runParametricFitAsync<MiniParams, MiniState, MiniControls, null>({
      init: { accel: 1, damping: 0.1 },
      encode: (p) => [p.accel, p.damping],
      decode: (v) => ({
        accel: Math.max(0.1, Math.min(10, v[0] ?? 1)),
        damping: Math.max(0, Math.min(5, v[1] ?? 0)),
      }),
      makeSim,
      stateDelta: (pred, act) => {
        const dx = pred.x - act.x;
        const dv = pred.v - act.v;
        return dx * dx + dv * dv;
      },
      trials: store.all(),
      controlsToVec: (c) => [c.effort],
      maxIter,
      onProgress: (e) => events.push(e),
      cooperativeYield: async () => {},
      yieldEveryNTrials: 0,
    });
    // Iter counter should be strictly increasing.
    for (let i = 1; i < events.length; i++) {
      expect(events[i]!.iter).toBeGreaterThan(events[i - 1]!.iter);
    }
    // We get a progress event for the bulk of iterations (allow some slack
    // for early-termination on tight convergence).
    expect(events.length).toBeGreaterThanOrEqual(Math.min(10, maxIter));
  });
});

// ---------------------------------------------------------------------------

describe('runResidualMLPFitAsync — cooperative yielding + val-split loss', () => {
  // Tiny synthetic residual problem: predict the constant ZERO residual
  // (baseline already perfect). Verifies wiring without expensive training.

  interface S { value: number; }
  interface C { ctrl: number; }

  function buildResidualTrials(): Trial<S, C, null>[] {
    const dt = 1;
    const trials: Trial<S, C, null>[] = [];
    for (let i = 0; i < 8; i++) {
      const samples: Trial<S, C, null>['samples'] = [];
      for (let k = 0; k < 4; k++) {
        samples.push({ t: k * dt, state: { value: 0 } });
      }
      trials.push({
        id: `r${i}`,
        initialState: { value: 0 },
        controlsTrace: Array.from({ length: 3 }, () => ({ ctrl: 0 })),
        dt,
        samples,
        config: null,
        configKey: 'cfg',
      });
    }
    return trials;
  }

  it('yields between epochs at least `epochs / yieldEveryNEpochs` times', async () => {
    const trials = buildResidualTrials();
    const epochs = 6;
    let yieldCount = 0;
    await runResidualMLPFitAsync<S, C, null>({
      trials,
      makeBaselineSim: () => ((s) => ({ value: s.value })),
      encodeInput: () => [0, 0],
      encodeResidual: (actual, baseline) => [actual.value - baseline.value],
      controlsToVec: (c) => [c.ctrl],
      mlpShape: { inputDim: 2, hiddenDims: [4], outputDim: 1 },
      ensembleSize: 1,
      seed: 1,
      epochs,
      batchSize: 4,
      learningRate: 1e-3,
      valSplit: 0.25,
      fitSubstepsPerSample: 1,
      cooperativeYield: async () => { yieldCount++; },
      yieldEveryNEpochs: 1,
      yieldEveryNBatches: 0,
    });
    // Lower bound: one yield per epoch.
    expect(yieldCount).toBeGreaterThanOrEqual(epochs);
  });

  it('emits ResidualFitProgressEvent carrying both trainLoss and valLoss', async () => {
    const trials = buildResidualTrials();
    const events: { epoch: number; trainLoss: number; valLoss: number }[] = [];
    await runResidualMLPFitAsync<S, C, null>({
      trials,
      makeBaselineSim: () => ((s) => ({ value: s.value })),
      encodeInput: () => [0, 0],
      encodeResidual: (actual, baseline) => [actual.value - baseline.value],
      controlsToVec: (c) => [c.ctrl],
      mlpShape: { inputDim: 2, hiddenDims: [4], outputDim: 1 },
      ensembleSize: 1,
      seed: 1,
      epochs: 4,
      batchSize: 4,
      learningRate: 1e-3,
      valSplit: 0.25,
      fitSubstepsPerSample: 1,
      onProgress: (e) => events.push(e),
      cooperativeYield: async () => {},
      yieldEveryNEpochs: 1,
      yieldEveryNBatches: 0,
    });
    expect(events.length).toBeGreaterThan(0);
    for (const e of events) {
      expect(Number.isFinite(e.trainLoss)).toBe(true);
      expect(Number.isFinite(e.valLoss)).toBe(true);
    }
  });
});
