// V3 dynamics fit — train a neural transition model DIRECTLY on recorded
// plant transitions. No parametric backbone, no hand-set parameter bounds:
// the network IS the model, and every constant it uses (input/output
// normalization, noise magnitudes) is a statistic of the training data.
//
// Data contract: trials recorded at the plant's physics tick with
// `sampleEveryNTicks = 1`, so every consecutive sample pair is one exact
// plant transition under a single constant control vector. That makes the
// learning target the plant's own one-tick transition function — nothing is
// smoothed, averaged, or re-derived.
//
// Two purely data-level techniques give rollout stability without touching
// the model class:
//   - MIRROR AUGMENTATION: the chassis is geometrically left/right
//     symmetric (symmetric wheel placement, identical friction), so every
//     recorded transition implies its mirror image. Training on both halves
//     doubles the data and removes spurious asymmetry.
//   - NOISE INJECTION (à la graph-network simulators): each epoch adds
//     small Gaussian noise to the dynamic-state inputs while keeping the
//     recorded next state as the target, teaching the network to pull
//     drifting rollouts back toward the data manifold. The noise magnitude
//     is a fraction of the per-channel data std — data-derived, not
//     hand-set in physical units.

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
import type { CarKinematicState } from '../agent/types';
import type { WheeledCarControls } from '../agent/controls';
import type { LearnableVehicleConfig } from '../agent/vehicle-config';
import {
  V3_INPUT_DIM,
  V3_OUTPUT_DIM,
  buildV3Input,
  type V3Normalization,
  type LearnedVehicleModelV3,
} from '../agent/vehicle-model-v3';
import { wrapAngle } from '../internal/math';

export type CarTrial = Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>;

export interface DynamicsV3FitOptions {
  trials: ReadonlyArray<CarTrial>;
  hiddenDims?: number[];
  ensembleSize?: number;
  seed?: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  /** Fraction of trials (by index, deterministic) reserved for validation. */
  valSplit?: number;
  /** Mirror every transition through the chassis's left/right symmetry. */
  mirrorAugment?: boolean;
  /** Rollout-stabilizing input noise: perturb the dynamic-state inputs
   *  (speed, yawRate, latVel) with Gaussian noise whose std is this
   *  fraction of the corresponding one-step OUTPUT delta std — i.e. noise
   *  the size of the drift a prediction step can introduce, which is what
   *  the model sees mid-rollout. Targets are left untouched (pure
   *  denoising regularization): correcting targets was measured to inject
   *  a conflicting derivative constraint that wrecked the longitudinal
   *  channel. 0 disables. */
  noiseScale?: number;
  onProgress?: (e: { epoch: number; trainLoss: number; valLoss: number }) => void;
}

export interface DynamicsV3FitResult {
  model: LearnedVehicleModelV3;
  finalTrainLoss: number;
  finalValLoss: number;
  /** Validation RMS per output channel in RAW units
   *  [dFwd m, dRight m, dHeading rad, dSpeed m/s, dYawRate rad/s,
   *  dLateralVelocity m/s] — per reference step. */
  valRmsRaw: number[];
  trainPairs: number;
  valPairs: number;
  history: { epoch: number; trainLoss: number; valLoss: number }[];
}

interface Pair {
  input: number[];  // raw units
  target: number[]; // raw units
}

/** Extract one-tick body-frame transition pairs from a trial. */
function pairsFromTrial(trial: CarTrial): Pair[] {
  const out: Pair[] = [];
  const dt = trial.dt;
  for (let k = 1; k < trial.samples.length; k++) {
    const a = trial.samples[k - 1]!;
    const b = trial.samples[k]!;
    // Only exact one-tick transitions qualify — anything longer spans
    // multiple control vectors and is not a plant transition sample.
    const span = Math.round((b.t - a.t) / dt);
    if (span !== 1) continue;
    const ctrlIdx = Math.min(trial.controlsTrace.length - 1, Math.round(a.t / dt));
    const c = trial.controlsTrace[ctrlIdx]!;
    const sa = a.state;
    const sb = b.state;
    const cosH = Math.cos(sa.heading);
    const sinH = Math.sin(sa.heading);
    const dxW = sb.x - sa.x;
    const dzW = sb.z - sa.z;
    out.push({
      input: buildV3Input(sa, [c.steer, c.driveForce, c.brakeForce]),
      target: [
        dxW * cosH + dzW * sinH,   // dFwd  (body forward)
        dxW * sinH - dzW * cosH,   // dRight (body right)
        wrapAngle(sb.heading - sa.heading),
        sb.speed - sa.speed,
        (sb.yawRate ?? 0) - (sa.yawRate ?? 0),
        (sb.lateralVelocity ?? 0) - (sa.lateralVelocity ?? 0),
      ],
    });
  }
  return out;
}

/** The chassis's left/right mirror: negate every laterally-signed quantity. */
function mirrorPair(p: Pair): Pair {
  const [v, yr, lv, st, dr, br] = p.input as [number, number, number, number, number, number];
  const [dF, dR, dH, dS, dYr, dLv] = p.target as [number, number, number, number, number, number];
  return {
    input: [v, -yr, -lv, -st, dr, br],
    target: [dF, -dR, -dH, dS, -dYr, -dLv],
  };
}

function computeNorm(pairs: Pair[]): V3Normalization {
  const n = Math.max(1, pairs.length);
  const inputMean = new Array<number>(V3_INPUT_DIM).fill(0);
  const outputMean = new Array<number>(V3_OUTPUT_DIM).fill(0);
  for (const p of pairs) {
    for (let i = 0; i < V3_INPUT_DIM; i++) inputMean[i]! += p.input[i]! / n;
    for (let i = 0; i < V3_OUTPUT_DIM; i++) outputMean[i]! += p.target[i]! / n;
  }
  const inputStd = new Array<number>(V3_INPUT_DIM).fill(0);
  const outputStd = new Array<number>(V3_OUTPUT_DIM).fill(0);
  for (const p of pairs) {
    for (let i = 0; i < V3_INPUT_DIM; i++) {
      const d = p.input[i]! - inputMean[i]!;
      inputStd[i]! += (d * d) / n;
    }
    for (let i = 0; i < V3_OUTPUT_DIM; i++) {
      const d = p.target[i]! - outputMean[i]!;
      outputStd[i]! += (d * d) / n;
    }
  }
  // sqrt + tiny floor purely to avoid division by zero on a degenerate
  // (constant) channel — not a physical assumption.
  for (let i = 0; i < V3_INPUT_DIM; i++) inputStd[i] = Math.max(1e-9, Math.sqrt(inputStd[i]!));
  for (let i = 0; i < V3_OUTPUT_DIM; i++) outputStd[i] = Math.max(1e-9, Math.sqrt(outputStd[i]!));
  return { inputMean, inputStd, outputMean, outputStd };
}

function normalizePairs(pairs: Pair[], norm: V3Normalization): Pair[] {
  return pairs.map((p) => ({
    input: p.input.map((v, i) => (v - norm.inputMean[i]!) / norm.inputStd[i]!),
    target: p.target.map((v, i) => (v - norm.outputMean[i]!) / norm.outputStd[i]!),
  }));
}

function meanLoss(mlp: MLP, samples: Pair[]): number {
  let total = 0;
  for (const s of samples) {
    const cache = forward(mlp, s.input);
    let l = 0;
    for (let o = 0; o < cache.output.length; o++) {
      const d = cache.output[o]! - s.target[o]!;
      l += 0.5 * d * d;
    }
    total += l;
  }
  return total / Math.max(1, samples.length);
}

function shuffleIndices(n: number, seed: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
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

/** Seeded standard-normal sampler (Box-Muller over Mulberry32). */
function makeNormal(seed: number): () => number {
  let state = (seed | 0) || 1;
  const u01 = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (): number => {
    const u = Math.max(1e-12, u01());
    const v = u01();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// Input dynamic-state channel i corresponds to output delta channel
// STATE_TO_DELTA[i]: speed→dSpeed, yawRate→dYawRate, latVel→dLatVel.
// Used by noise injection to keep input and target consistent.
const STATE_TO_DELTA = [3, 4, 5];

export function runDynamicsV3Fit(opts: DynamicsV3FitOptions): DynamicsV3FitResult {
  const seed = opts.seed ?? 1;
  const hiddenDims = opts.hiddenDims ?? [64, 64];
  const ensembleSize = opts.ensembleSize ?? 3;
  const epochs = opts.epochs ?? 80;
  const batchSize = opts.batchSize ?? 64;
  const lr = opts.learningRate ?? 1e-3;
  const valFrac = opts.valSplit ?? 0.2;
  const mirror = opts.mirrorAugment ?? true;
  const noiseScale = opts.noiseScale ?? 0.5;

  const supplied = [...opts.trials];
  if (supplied.length === 0) throw new Error('dynamics-v3-fit: no trials supplied');
  const referenceDt = supplied[0]!.dt;
  const config = supplied[0]!.config;

  // Deterministic shuffle before the split — collection orders trials by
  // maneuver category, so a positional split would hold out a whole
  // category instead of a representative sample.
  const trialOrder = shuffleIndices(supplied.length, seed ^ 0x5f3759df);
  const trials = trialOrder.map((i) => supplied[i]!);
  const valCount = Math.max(0, Math.floor(trials.length * valFrac));
  const valTrials = trials.slice(0, valCount);
  const trainTrials = trials.slice(valCount);

  const collect = (ts: CarTrial[]): Pair[] => {
    const out: Pair[] = [];
    for (const t of ts) {
      for (const p of pairsFromTrial(t)) {
        out.push(p);
        if (mirror) out.push(mirrorPair(p));
      }
    }
    return out;
  };
  const trainRaw = collect(trainTrials);
  const valRaw = collect(valTrials);
  if (trainRaw.length === 0) {
    throw new Error(
      'dynamics-v3-fit: no one-tick transition pairs found — trials must be ' +
      'recorded with sampleEveryNTicks = 1',
    );
  }

  const norm = computeNorm(trainRaw);
  const train = normalizePairs(trainRaw, norm);
  const val = normalizePairs(valRaw, norm);

  // Noise magnitude per dynamic-state channel, in NORMALIZED input units:
  // raw std = noiseScale × (one-step delta std), converted through the
  // input normalization. Only state channels are perturbed — controls are
  // exact commands, not estimates.
  const noiseNormStd = STATE_TO_DELTA.map(
    (dIdx, sIdx) => (noiseScale * norm.outputStd[dIdx]!) / norm.inputStd[sIdx]!,
  );

  const ensemble: MLP[] = [];
  const adams: AdamState[] = [];
  for (let m = 0; m < ensembleSize; m++) {
    const mlp = createMLP(
      { inputDim: V3_INPUT_DIM, hiddenDims, outputDim: V3_OUTPUT_DIM },
      seed + m,
    );
    ensemble.push(mlp);
    adams.push(createAdam(mlp, lr));
  }

  const history: { epoch: number; trainLoss: number; valLoss: number }[] = [];
  const input = new Array<number>(V3_INPUT_DIM);
  const target = new Array<number>(V3_OUTPUT_DIM);
  for (let ep = 0; ep < epochs; ep++) {
    for (let m = 0; m < ensemble.length; m++) {
      const mlp = ensemble[m]!;
      const adam = adams[m]!;
      const order = shuffleIndices(train.length, seed + m * 7919 + ep * 31);
      const noise = makeNormal(seed + m * 104729 + ep * 613);
      for (let b = 0; b < order.length; b += batchSize) {
        const grads = zeroGradients(mlp);
        const end = Math.min(order.length, b + batchSize);
        const inv = 1 / Math.max(1, end - b);
        for (let j = b; j < end; j++) {
          const s = train[order[j]!]!;
          for (let i = 0; i < V3_INPUT_DIM; i++) input[i] = s.input[i]!;
          for (let i = 0; i < V3_OUTPUT_DIM; i++) target[i] = s.target[i]!;
          if (noiseScale > 0) {
            for (let i = 0; i < STATE_TO_DELTA.length; i++) {
              input[i]! += noise() * noiseNormStd[i]!;
            }
          }
          const cache = forward(mlp, input);
          const g = backward(mlp, cache, target);
          accumulateGradients(grads, g, inv);
        }
        adamStep(mlp, grads, adam);
      }
    }
    if (ep % Math.max(1, Math.floor(epochs / 25)) === 0 || ep === epochs - 1) {
      const trainL = ensemble.reduce((a, m) => a + meanLoss(m, train), 0) / ensemble.length;
      const valL = ensemble.reduce((a, m) => a + meanLoss(m, val), 0) / Math.max(1, ensemble.length);
      const evt = { epoch: ep, trainLoss: trainL, valLoss: valL };
      history.push(evt);
      opts.onProgress?.(evt);
    }
  }

  // Per-channel validation RMS of the ENSEMBLE MEAN, in raw units.
  const sqErr = new Array<number>(V3_OUTPUT_DIM).fill(0);
  for (const s of val) {
    const mean = new Float64Array(V3_OUTPUT_DIM);
    for (const mlp of ensemble) {
      const out = forward(mlp, s.input).output;
      for (let i = 0; i < V3_OUTPUT_DIM; i++) mean[i]! += out[i]! / ensemble.length;
    }
    for (let i = 0; i < V3_OUTPUT_DIM; i++) {
      const d = (mean[i]! - s.target[i]!) * norm.outputStd[i]!;
      sqErr[i]! += d * d;
    }
  }
  const valRmsRaw = sqErr.map((e) => Math.sqrt(e / Math.max(1, val.length)));

  const finalTrain = ensemble.reduce((a, m) => a + meanLoss(m, train), 0) / ensemble.length;
  const finalVal = ensemble.reduce((a, m) => a + meanLoss(m, val), 0) / Math.max(1, ensemble.length);
  return {
    model: { config, ensemble, norm, referenceDt },
    finalTrainLoss: finalTrain,
    finalValLoss: finalVal,
    valRmsRaw,
    trainPairs: train.length,
    valPairs: val.length,
    history,
  };
}
