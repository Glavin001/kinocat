// Generic SGD/Adam fit for a residual MLP that corrects a parametric
// baseline's prediction. Domain-agnostic: caller supplies the trial data,
// an input encoder, a target encoder (residual = actual - baseline), and the
// MLP shape.

import {
  type MLP,
  createMLP,
  createAdam,
  type AdamState,
  forward,
  backward,
  adamStep,
  zeroGradients,
  accumulateGradients,
} from '../internal/mlp';
import type { Trial } from './trial-store';
import type { ForwardSim } from '../primitives/types';

export interface ResidualMLPFitOptions<S, C, Cfg> {
  /** Trial data to fit against. */
  trials: ReadonlyArray<Trial<S, C, Cfg>>;
  /** Baseline forward-sim factory (e.g. the fitted parametric model). */
  makeBaselineSim: (cfg: Cfg) => ForwardSim<S>;
  /** Encode `(state, controls, config) → numeric input vector` for the MLP. */
  encodeInput: (state: S, controls: ReadonlyArray<number>, config: Cfg) => number[];
  /** Encode the residual `(actual - baseline) → numeric target vector`.
   *  Must match the MLP output dim. */
  encodeResidual: (actual: S, baseline: S) => number[];
  /** Convert `C` → opaque `number[]` controls for the baseline sim. */
  controlsToVec: (c: C) => number[];
  /** MLP shape: number of input dims, hidden layer sizes, output dims. */
  mlpShape: { inputDim: number; hiddenDims: number[]; outputDim: number };
  /** Ensemble size — independent random-seed MLPs. */
  ensembleSize: number;
  /** Base random seed (members use `seed`, `seed+1`, ...). */
  seed?: number;
  /** Epoch count. */
  epochs?: number;
  /** Mini-batch size. */
  batchSize?: number;
  /** Learning rate. */
  learningRate?: number;
  /** Train/validation split by trial id (deterministic). */
  valSplit?: number; // fraction (0..1) of trials reserved for validation
  /** Progress callback per epoch (averages across the ensemble). */
  onProgress?: (e: ResidualFitProgressEvent) => void;
  /** Substeps per recorded-sample interval used for the baseline rollout
   *  inside the residual-target computation. Default 6 (matches
   *  parametric-fit default). */
  fitSubstepsPerSample?: number;
}

export interface ResidualFitProgressEvent {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface ResidualMLPFitResult {
  ensemble: MLP[];
  finalTrainLoss: number;
  finalValLoss: number;
  history: ResidualFitProgressEvent[];
}

/** Pre-compute (input, residualTarget) sample pairs from the trial set. The
 *  baseline sim is rolled from each trial's initial state and at each
 *  recorded sample we record the input + the residual between baseline
 *  prediction and recorded actual state. */
function buildSamples<S, C, Cfg>(
  opts: ResidualMLPFitOptions<S, C, Cfg>,
  trials: ReadonlyArray<Trial<S, C, Cfg>>,
): { input: number[]; target: number[] }[] {
  const out: { input: number[]; target: number[] }[] = [];
  const fitSubs = opts.fitSubstepsPerSample ?? 6;
  for (const trial of trials) {
    const sim = opts.makeBaselineSim(trial.config);
    let s: S = trial.initialState;
    for (let k = 1; k < trial.samples.length; k++) {
      const a = trial.samples[k - 1]!;
      const b = trial.samples[k]!;
      const windowDt = (b.t - a.t) / fitSubs;
      // Encode input from the state we're predicting FROM (sample a's state).
      // The controls applied are those at the midpoint of the window — use
      // the same lookup as parametric-fit.
      const tMid = a.t + 0.5 * (b.t - a.t);
      const ctrlIdx = Math.min(trial.controlsTrace.length - 1, Math.floor(tMid / trial.dt));
      const ctrl = opts.controlsToVec(trial.controlsTrace[ctrlIdx]!);
      const input = opts.encodeInput(s, ctrl, trial.config);
      // Roll baseline through the window.
      for (let j = 0; j < fitSubs; j++) {
        s = sim(s, ctrl, windowDt);
      }
      const target = opts.encodeResidual(b.state, s);
      out.push({ input, target });
      // Reseat baseline to the actual state for the next window so error
      // doesn't compound during target collection.
      s = b.state;
    }
  }
  return out;
}

function meanLoss(mlp: MLP, samples: { input: number[]; target: number[] }[]): number {
  let total = 0;
  for (const s of samples) {
    const cache = forward(mlp, s.input);
    let l = 0;
    for (let o = 0; o < cache.output.length; o++) {
      const d = cache.output[o]! - (s.target[o] ?? 0);
      l += 0.5 * d * d;
    }
    total += l;
  }
  return total / Math.max(1, samples.length);
}

export function runResidualMLPFit<S, C, Cfg>(
  opts: ResidualMLPFitOptions<S, C, Cfg>,
): ResidualMLPFitResult {
  const baseSeed = opts.seed ?? 1;
  const trials = [...opts.trials];
  const valFrac = opts.valSplit ?? 0.2;
  // Deterministic split by trial index.
  const valCount = Math.max(0, Math.floor(trials.length * valFrac));
  const trainTrials = trials.slice(valCount);
  const valTrials = trials.slice(0, valCount);
  const trainSamples = buildSamples(opts, trainTrials);
  const valSamples = buildSamples(opts, valTrials);
  const epochs = opts.epochs ?? 200;
  const batchSize = opts.batchSize ?? 64;
  const lr = opts.learningRate ?? 1e-3;

  const ensemble: MLP[] = [];
  const adams: AdamState[] = [];
  for (let m = 0; m < opts.ensembleSize; m++) {
    const mlp = createMLP(opts.mlpShape, baseSeed + m);
    ensemble.push(mlp);
    adams.push(createAdam(mlp, lr));
  }

  const history: ResidualFitProgressEvent[] = [];
  for (let ep = 0; ep < epochs; ep++) {
    // Train each ensemble member through the train set in different shuffled order
    // (the shuffling is keyed by ensemble idx, deterministic given seed).
    for (let m = 0; m < ensemble.length; m++) {
      const mlp = ensemble[m]!;
      const adam = adams[m]!;
      const order = shuffleIndices(trainSamples.length, baseSeed + m * 7919 + ep * 31);
      for (let b = 0; b < order.length; b += batchSize) {
        const grads = zeroGradients(mlp);
        const end = Math.min(order.length, b + batchSize);
        const inv = 1 / Math.max(1, end - b);
        for (let j = b; j < end; j++) {
          const s = trainSamples[order[j]!]!;
          const cache = forward(mlp, s.input);
          const g = backward(mlp, cache, s.target);
          accumulateGradients(grads, g, inv);
        }
        adamStep(mlp, grads, adam);
      }
    }
    if (ep % Math.max(1, Math.floor(epochs / 50)) === 0 || ep === epochs - 1) {
      const trainL = ensemble.reduce((a, m) => a + meanLoss(m, trainSamples), 0) / ensemble.length;
      const valL = ensemble.reduce((a, m) => a + meanLoss(m, valSamples), 0) / Math.max(1, ensemble.length);
      const evt: ResidualFitProgressEvent = { epoch: ep, trainLoss: trainL, valLoss: valL };
      history.push(evt);
      opts.onProgress?.(evt);
    }
  }
  const finalTrain = ensemble.reduce((a, m) => a + meanLoss(m, trainSamples), 0) / ensemble.length;
  const finalVal = ensemble.reduce((a, m) => a + meanLoss(m, valSamples), 0) / Math.max(1, ensemble.length);
  return { ensemble, finalTrainLoss: finalTrain, finalValLoss: finalVal, history };
}

// ---------------------------------------------------------------------------
// Async, cooperatively-yielding variant for browser callers — same logic as
// `runResidualMLPFit` but yields to the event loop periodically so a long
// SGD training run does not lock up the Chrome main thread.

export interface ResidualMLPFitAsyncOptions<S, C, Cfg>
  extends ResidualMLPFitOptions<S, C, Cfg>
{
  /** Yield to the event loop every N epochs. Default 1. */
  yieldEveryNEpochs?: number;
  /** Yield to the event loop every N mini-batches within an epoch.
   *  Default 8. Set to 0 to disable mid-epoch yielding. */
  yieldEveryNBatches?: number;
  /** How to yield. Default: `setTimeout(resolve, 0)`. */
  cooperativeYield?: () => Promise<void>;
}

const defaultMlpYield = (): Promise<void> =>
  new Promise<void>((resolve) => setTimeout(resolve, 0));

export async function runResidualMLPFitAsync<S, C, Cfg>(
  opts: ResidualMLPFitAsyncOptions<S, C, Cfg>,
): Promise<ResidualMLPFitResult> {
  const baseSeed = opts.seed ?? 1;
  const trials = [...opts.trials];
  const valFrac = opts.valSplit ?? 0.2;
  const valCount = Math.max(0, Math.floor(trials.length * valFrac));
  const trainTrials = trials.slice(valCount);
  const valTrials = trials.slice(0, valCount);
  const trainSamples = buildSamples(opts, trainTrials);
  const valSamples = buildSamples(opts, valTrials);
  const epochs = opts.epochs ?? 200;
  const batchSize = opts.batchSize ?? 64;
  const lr = opts.learningRate ?? 1e-3;
  const coop = opts.cooperativeYield ?? defaultMlpYield;
  const yieldEveryNEpochs = Math.max(1, opts.yieldEveryNEpochs ?? 1);
  const yieldEveryNBatches = Math.max(0, opts.yieldEveryNBatches ?? 8);

  const ensemble: MLP[] = [];
  const adams: AdamState[] = [];
  for (let m = 0; m < opts.ensembleSize; m++) {
    const mlp = createMLP(opts.mlpShape, baseSeed + m);
    ensemble.push(mlp);
    adams.push(createAdam(mlp, lr));
  }

  const history: ResidualFitProgressEvent[] = [];
  for (let ep = 0; ep < epochs; ep++) {
    for (let m = 0; m < ensemble.length; m++) {
      const mlp = ensemble[m]!;
      const adam = adams[m]!;
      const order = shuffleIndices(trainSamples.length, baseSeed + m * 7919 + ep * 31);
      let batchIdx = 0;
      for (let b = 0; b < order.length; b += batchSize) {
        const grads = zeroGradients(mlp);
        const end = Math.min(order.length, b + batchSize);
        const inv = 1 / Math.max(1, end - b);
        for (let j = b; j < end; j++) {
          const s = trainSamples[order[j]!]!;
          const cache = forward(mlp, s.input);
          const g = backward(mlp, cache, s.target);
          accumulateGradients(grads, g, inv);
        }
        adamStep(mlp, grads, adam);
        batchIdx++;
        if (yieldEveryNBatches > 0 && batchIdx % yieldEveryNBatches === 0) {
          await coop();
        }
      }
    }
    if (ep % Math.max(1, Math.floor(epochs / 50)) === 0 || ep === epochs - 1) {
      const trainL = ensemble.reduce((a, m) => a + meanLoss(m, trainSamples), 0) / ensemble.length;
      const valL = ensemble.reduce((a, m) => a + meanLoss(m, valSamples), 0) / Math.max(1, ensemble.length);
      const evt: ResidualFitProgressEvent = { epoch: ep, trainLoss: trainL, valLoss: valL };
      history.push(evt);
      opts.onProgress?.(evt);
    }
    if ((ep + 1) % yieldEveryNEpochs === 0) {
      await coop();
    }
  }
  const finalTrain = ensemble.reduce((a, m) => a + meanLoss(m, trainSamples), 0) / ensemble.length;
  const finalVal = ensemble.reduce((a, m) => a + meanLoss(m, valSamples), 0) / Math.max(1, ensemble.length);
  return { ensemble, finalTrainLoss: finalTrain, finalValLoss: finalVal, history };
}

function shuffleIndices(n: number, seed: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  // Mulberry32-style PRNG for reproducibility.
  let state = (seed | 0) || 1;
  const rand = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i]!;
    arr[i] = arr[j]!;
    arr[j] = tmp;
  }
  return arr;
}
