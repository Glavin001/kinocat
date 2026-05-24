// Generic Nelder-Mead simplex fitter for any parametric ForwardSim against
// recorded trial data. Domain-agnostic: caller supplies the parameter
// encoder/decoder, a `makeSim(params, config) → ForwardSim<S>` factory, a
// state-difference function that computes the weighted loss between predicted
// and actual states, and the trial set.
//
// Lifts the implementation that lived in `demos/app/lib/learn-primitives.ts`
// into core so it can be reused for the v2 model and future agent kinds.

import type { ForwardSim } from '../primitives/types';
import type { Trial } from './trial-store';

export interface ParametricFitOptions<P, S, C, Cfg> {
  /** Initial parameter values (used as the simplex starting vertex). */
  init: P;
  /** Encode parameters as a numeric vector. */
  encode: (p: P) => number[];
  /** Decode a numeric vector back into parameters (with bounds clamping). */
  decode: (v: ReadonlyArray<number>) => P;
  /** Build a ForwardSim from parameters + config. */
  makeSim: (p: P, cfg: Cfg) => ForwardSim<S>;
  /** Weighted L2 difference between predicted and actual state. Higher =
   *  worse fit. The fitter sums these across all (trial, sample) pairs. */
  stateDelta: (predicted: S, actual: S) => number;
  /** Per-state breakdown of the same comparison (for diagnostics charts). */
  decomposeDelta?: (predicted: S, actual: S) => LossDecomposition;
  /** Trial data. Each trial is rolled out forward through the sim and
   *  predictions are compared to the recorded samples. */
  trials: ReadonlyArray<Trial<S, C, Cfg>>;
  /** L2 regularization toward `priorVec` (in encoded-vector space). Default
   *  0 (no regularization). */
  regularization?: { strength: number; priorVec: number[]; scales: number[] };
  /** Max Nelder-Mead iterations. Default 400. */
  maxIter?: number;
  /** Per-iteration progress callback. Fires on every accepted simplex move
   *  with the current best loss. */
  onProgress?: (e: FitProgressEvent) => void;
  /** Optional initial simplex step (fractional). Default 0.1. */
  simplexStep?: number;
  /** Convergence tolerance on simplex range. Default 1e-6. */
  tol?: number;
  /** Sub-steps per recorded-sample interval used for sim integration. Higher
   *  = finer integration, slower fit. Default 6. */
  fitSubstepsPerSample?: number;
  /** Map a single tick's controls. If `Trial.controlsTrace` has one entry per
   *  *physics* tick (60 Hz) but samples are at e.g. 10 Hz, the fitter needs
   *  to know which control to apply at each sub-step. Default: use the
   *  control corresponding to the integer tick under the current sub-step
   *  time, i.e. `controlsTrace[floor(tickIdx)]`. */
  controlAt?: (controlsTrace: ReadonlyArray<C>, dtElapsed: number, trialDt: number) => C;
  /** How to convert a `C` controls value into the opaque `number[]` array
   *  consumed by the ForwardSim. */
  controlsToVec: (c: C) => number[];
}

export interface LossDecomposition {
  pos: number;
  heading: number;
  speed: number;
  yawRate: number;
  lateralVelocity: number;
}

export interface FitProgressEvent {
  iter: number;
  loss: number;
  /** Optional per-component breakdown (averaged across samples). */
  perComponent?: LossDecomposition;
  /** Optional held-out (validation-split) loss. Populated by the residual
   *  MLP fitter (which has an explicit val split); the parametric fitter
   *  omits this field (no val split). */
  valLoss?: number;
}

export interface ParametricFitResult<P> {
  params: P;
  finalLoss: number;
  iterations: number;
  /** Final per-component loss decomposition (if `decomposeDelta` supplied). */
  perComponent?: LossDecomposition;
  /** Full loss history (one entry per iteration where loss improved). */
  history: FitProgressEvent[];
}

/** Roll the supplied `forwardSim` along a trial from `initialState`,
 *  comparing the predicted state at each recorded sample to the recorded
 *  actual state. Returns the total weighted L2 loss (no regularization).
 *  Exposed so consumers can re-use the same rollout for diagnostics. */
export function rolloutAndScore<S, C, Cfg>(
  sim: ForwardSim<S>,
  trial: Trial<S, C, Cfg>,
  stateDelta: (predicted: S, actual: S) => number,
  controlsToVec: (c: C) => number[],
  fitSubstepsPerSample = 6,
  decompose?: (predicted: S, actual: S) => LossDecomposition,
): { loss: number; decomposition?: LossDecomposition; sampleCount: number } {
  let s: S = trial.initialState;
  let loss = 0;
  let count = 0;
  const decomp: LossDecomposition | null = decompose
    ? { pos: 0, heading: 0, speed: 0, yawRate: 0, lateralVelocity: 0 }
    : null;
  for (let k = 1; k < trial.samples.length; k++) {
    const a = trial.samples[k - 1]!;
    const b = trial.samples[k]!;
    const windowDt = (b.t - a.t) / fitSubstepsPerSample;
    for (let j = 0; j < fitSubstepsPerSample; j++) {
      // The control to apply: pick the control corresponding to the current
      // sub-step time, mapping back to the trial-tick index. controlsTrace
      // is at trial.dt resolution; current elapsed time = a.t + (j+0.5)*windowDt.
      const tNow = a.t + (j + 0.5) * windowDt;
      const idx = Math.min(trial.controlsTrace.length - 1, Math.floor(tNow / trial.dt));
      const c = trial.controlsTrace[idx]!;
      s = sim(s, controlsToVec(c), windowDt);
    }
    loss += stateDelta(s, b.state);
    if (decomp) {
      const d = decompose!(s, b.state);
      decomp.pos += d.pos;
      decomp.heading += d.heading;
      decomp.speed += d.speed;
      decomp.yawRate += d.yawRate;
      decomp.lateralVelocity += d.lateralVelocity;
    }
    count++;
  }
  if (decomp && count > 0) {
    decomp.pos /= count;
    decomp.heading /= count;
    decomp.speed /= count;
    decomp.yawRate /= count;
    decomp.lateralVelocity /= count;
  }
  return { loss, decomposition: decomp ?? undefined, sampleCount: count };
}

function fullLoss<P, S, C, Cfg>(
  encoded: ReadonlyArray<number>,
  opts: ParametricFitOptions<P, S, C, Cfg>,
): number {
  const params = opts.decode(encoded);
  let total = 0;
  let totalCount = 0;
  for (const trial of opts.trials) {
    const sim = opts.makeSim(params, trial.config);
    const r = rolloutAndScore(
      sim,
      trial,
      opts.stateDelta,
      opts.controlsToVec,
      opts.fitSubstepsPerSample ?? 6,
    );
    total += r.loss;
    totalCount += r.sampleCount;
  }
  // L2 regularization toward priorVec (encoded-space).
  const reg = opts.regularization;
  if (reg && reg.strength > 0 && totalCount > 0) {
    const scale = reg.strength * totalCount;
    for (let i = 0; i < encoded.length; i++) {
      const s = reg.scales[i] || 1;
      const d = (encoded[i]! - (reg.priorVec[i] ?? 0)) / s;
      total += scale * d * d;
    }
  }
  return total;
}

function nelderMead(
  x0: number[],
  loss: (v: number[]) => number,
  opts: { maxIter: number; tol: number; step: number; onIter?: (iter: number, best: number) => void },
): { x: number[]; iters: number; best: number } {
  const n = x0.length;
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const v = x0.slice();
    v[i] = (v[i] ?? 0) * (1 + opts.step) + ((v[i] ?? 0) === 0 ? opts.step : 0);
    simplex.push(v);
  }
  let scores = simplex.map(loss);
  let bestEverScore = Math.min(...scores);
  for (let iter = 0; iter < opts.maxIter; iter++) {
    const order = scores
      .map((s, i) => [s, i] as const)
      .sort((a, b) => a[0] - b[0])
      .map((p) => p[1]);
    const sortedSim = order.map((i) => simplex[i]!);
    const sortedScores = order.map((i) => scores[i]!);
    for (let i = 0; i < simplex.length; i++) {
      simplex[i] = sortedSim[i]!;
      scores[i] = sortedScores[i]!;
    }
    const best = scores[0]!;
    const worst = scores[n]!;
    if (best < bestEverScore) {
      bestEverScore = best;
      opts.onIter?.(iter, best);
    }
    if (worst - best < opts.tol) return { x: simplex[0]!, iters: iter, best };
    const centroid = new Array(n).fill(0) as number[];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]!;
    }
    for (let j = 0; j < n; j++) centroid[j]! /= n;
    const xr = centroid.map((c, j) => c + (c - simplex[n]![j]!));
    const fr = loss(xr);
    if (fr < scores[n - 1]! && fr >= scores[0]!) {
      simplex[n] = xr;
      scores[n] = fr;
      continue;
    }
    if (fr < scores[0]!) {
      const xe = centroid.map((c, j) => c + 2 * (c - simplex[n]![j]!));
      const fe = loss(xe);
      if (fe < fr) {
        simplex[n] = xe;
        scores[n] = fe;
      } else {
        simplex[n] = xr;
        scores[n] = fr;
      }
      continue;
    }
    const xc = centroid.map((c, j) => c + 0.5 * (simplex[n]![j]! - c));
    const fc = loss(xc);
    if (fc < scores[n]!) {
      simplex[n] = xc;
      scores[n] = fc;
      continue;
    }
    for (let i = 1; i <= n; i++) {
      simplex[i] = simplex[0]!.map((b, j) => b + 0.5 * (simplex[i]![j]! - b));
      scores[i] = loss(simplex[i]!);
    }
  }
  return { x: simplex[0]!, iters: opts.maxIter, best: scores[0]! };
}

export function runParametricFit<P, S, C, Cfg>(
  opts: ParametricFitOptions<P, S, C, Cfg>,
): ParametricFitResult<P> {
  const x0 = opts.encode(opts.init);
  const history: FitProgressEvent[] = [];
  const result = nelderMead(
    x0,
    (v) => fullLoss(v, opts),
    {
      maxIter: opts.maxIter ?? 400,
      tol: opts.tol ?? 1e-6,
      step: opts.simplexStep ?? 0.1,
      onIter: (iter, best) => {
        const e: FitProgressEvent = { iter, loss: best };
        history.push(e);
        opts.onProgress?.(e);
      },
    },
  );
  const params = opts.decode(result.x);
  // Final decomposition (no regularization).
  let perComp: LossDecomposition | undefined;
  if (opts.decomposeDelta) {
    let agg: LossDecomposition = { pos: 0, heading: 0, speed: 0, yawRate: 0, lateralVelocity: 0 };
    let nTrials = 0;
    for (const trial of opts.trials) {
      const sim = opts.makeSim(params, trial.config);
      const r = rolloutAndScore(
        sim,
        trial,
        opts.stateDelta,
        opts.controlsToVec,
        opts.fitSubstepsPerSample ?? 6,
        opts.decomposeDelta,
      );
      if (r.decomposition) {
        agg.pos += r.decomposition.pos;
        agg.heading += r.decomposition.heading;
        agg.speed += r.decomposition.speed;
        agg.yawRate += r.decomposition.yawRate;
        agg.lateralVelocity += r.decomposition.lateralVelocity;
        nTrials++;
      }
    }
    if (nTrials > 0) {
      perComp = {
        pos: agg.pos / nTrials,
        heading: agg.heading / nTrials,
        speed: agg.speed / nTrials,
        yawRate: agg.yawRate / nTrials,
        lateralVelocity: agg.lateralVelocity / nTrials,
      };
    }
  }
  return {
    params,
    finalLoss: result.best,
    iterations: result.iters,
    perComponent: perComp,
    history,
  };
}
