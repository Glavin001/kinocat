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
  /**
   * Trajectory horizon — reseat the simulator's state to the actual
   * recorded state every `N` samples during the open-loop rollout.
   * `N=Infinity` (default) is the classic "roll all the way through
   * the trial and accumulate every sample's error" behaviour, which
   * heavily weights long-horizon trajectory shape but lets long-trial
   * drift dominate the loss. `N=1` is single-step (no compounding).
   *
   * For racing the planner uses 0.55–1.0 s primitives (5–10 samples at
   * sampleDt=0.1 s), so a horizon of 5–10 puts the loss focus where
   * the controller actually consumes the model's predictions. Smaller
   * horizons also make the loss easier to optimise because there's
   * less compounding error to fit.
   */
  trajectoryHorizon?: number;
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
  /** Raw objective the optimizer minimizes. For the parametric fit this
   *  is a SUM across all (trial, sample) pairs, so its absolute value
   *  scales with the dataset size and is NOT directly comparable across
   *  active-learning rounds. Prefer `lossNormalized` for display. */
  loss: number;
  /** `loss / sampleCount` — per-sample mean. Comparable across rounds
   *  even as the trial set grows. Populated by the async fitter when
   *  the sample count is known. */
  lossNormalized?: number;
  /** Total `(trial, sample)` pairs the loss was summed over. */
  sampleCount?: number;
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
 *  Exposed so consumers can re-use the same rollout for diagnostics.
 *
 *  `trajectoryHorizon` (default `Infinity`) lets the caller reseat the
 *  simulator's state to the actual recorded state every `N` samples,
 *  bounding how far prediction errors can compound before the loss is
 *  re-anchored. Default behaviour (no reseat) matches the original
 *  implementation. */
export function rolloutAndScore<S, C, Cfg>(
  sim: ForwardSim<S>,
  trial: Trial<S, C, Cfg>,
  stateDelta: (predicted: S, actual: S) => number,
  controlsToVec: (c: C) => number[],
  fitSubstepsPerSample = 6,
  decompose?: (predicted: S, actual: S) => LossDecomposition,
  trajectoryHorizon = Infinity,
): { loss: number; decomposition?: LossDecomposition; sampleCount: number } {
  let s: S = trial.initialState;
  let loss = 0;
  let count = 0;
  let stepsInChunk = 0;
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
    stepsInChunk++;
    if (stepsInChunk >= trajectoryHorizon) {
      // Reseat to the actual state so error doesn't keep compounding
      // past the chosen horizon. This bounds the multi-step trajectory
      // loss to N-sample chunks instead of letting drift over a
      // 30-sample trial dominate the loss.
      s = b.state;
      stepsInChunk = 0;
    }
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
      undefined,
      opts.trajectoryHorizon ?? Infinity,
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
  return finalizeFit(opts, result, history);
}

// ---------------------------------------------------------------------------
// Async, cooperatively-yielding variant for browser callers.
//
// The synchronous `runParametricFit` is fine for tests and Node, but in a
// browser tab a 200-iteration Nelder-Mead over ~100 trials can monopolize
// the main thread for tens of seconds and trigger "page unresponsive".
// This variant yields to the event loop:
//   - once every `yieldEveryNIter` simplex iterations (default 1),
//   - and once every `yieldEveryNTrials` trials inside each loss eval
//     (default 24), since a single loss eval is itself heavy.
//
// `cooperativeYield` defaults to a setTimeout(0) macrotask, which is the
// most portable way to let the browser paint and process input. Pass a
// faster `scheduler.yield()`-based yielder when targeting modern Chrome.

export interface ParametricFitAsyncOptions<P, S, C, Cfg>
  extends ParametricFitOptions<P, S, C, Cfg>
{
  /** Yield to the event loop every N Nelder-Mead iterations. Default 1. */
  yieldEveryNIter?: number;
  /** Yield to the event loop every N trials inside a single loss eval.
   *  Default 24. Set to 0 to disable mid-loss-eval yielding. */
  yieldEveryNTrials?: number;
  /** How to yield. Default: `setTimeout(resolve, 0)`. */
  cooperativeYield?: () => Promise<void>;
}

const defaultYield = (): Promise<void> =>
  new Promise<void>((resolve) => setTimeout(resolve, 0));

export async function runParametricFitAsync<P, S, C, Cfg>(
  opts: ParametricFitAsyncOptions<P, S, C, Cfg>,
): Promise<ParametricFitResult<P>> {
  const x0 = opts.encode(opts.init);
  const history: FitProgressEvent[] = [];
  const coop = opts.cooperativeYield ?? defaultYield;
  const yieldEveryNIter = Math.max(1, opts.yieldEveryNIter ?? 1);
  const yieldEveryNTrials = Math.max(0, opts.yieldEveryNTrials ?? 24);
  const maxIter = opts.maxIter ?? 400;
  // Precompute the total sample count so every progress event can
  // expose a normalized per-sample loss. Each trial contributes
  // (samples.length - 1) comparisons (the rollout starts at sample 1
  // and predicts forward to each subsequent sample). Matches
  // `rolloutAndScore`'s counting.
  let sampleCount = 0;
  for (const t of opts.trials) {
    sampleCount += Math.max(0, t.samples.length - 1);
  }
  const sampleCountSafe = Math.max(1, sampleCount);
  const loss = (v: ReadonlyArray<number>): Promise<number> =>
    fullLossAsync(v, opts, coop, yieldEveryNTrials);
  const result = await nelderMeadAsync(
    x0,
    loss,
    {
      maxIter,
      tol: opts.tol ?? 1e-6,
      step: opts.simplexStep ?? 0.1,
      onIter: (iter, best, improved) => {
        // Always notify so the UI can show "iter X / maxIter" even when
        // late iterations aren't improving the loss. Only append to the
        // returned history when the loss actually moved (keeps the
        // serialized fit curve clean).
        const e: FitProgressEvent = {
          iter,
          loss: best,
          lossNormalized: best / sampleCountSafe,
          sampleCount,
        };
        if (improved) history.push(e);
        opts.onProgress?.(e);
      },
      yieldEveryNIter,
      cooperativeYield: coop,
    },
  );
  return finalizeFit(opts, result, history);
}

function finalizeFit<P, S, C, Cfg>(
  opts: ParametricFitOptions<P, S, C, Cfg>,
  result: { x: number[]; iters: number; best: number },
  history: FitProgressEvent[],
): ParametricFitResult<P> {
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
        opts.trajectoryHorizon ?? Infinity,
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

// ---------------------------------------------------------------------------
// Async loss + Nelder-Mead helpers (used by `runParametricFitAsync`).

async function fullLossAsync<P, S, C, Cfg>(
  encoded: ReadonlyArray<number>,
  opts: ParametricFitOptions<P, S, C, Cfg>,
  coop: () => Promise<void>,
  yieldEveryNTrials: number,
): Promise<number> {
  const params = opts.decode(encoded);
  let total = 0;
  let totalCount = 0;
  let trialIdx = 0;
  for (const trial of opts.trials) {
    const sim = opts.makeSim(params, trial.config);
    const r = rolloutAndScore(
      sim,
      trial,
      opts.stateDelta,
      opts.controlsToVec,
      opts.fitSubstepsPerSample ?? 6,
      undefined,
      opts.trajectoryHorizon ?? Infinity,
    );
    total += r.loss;
    totalCount += r.sampleCount;
    trialIdx++;
    if (yieldEveryNTrials > 0 && trialIdx % yieldEveryNTrials === 0) {
      await coop();
    }
  }
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

async function nelderMeadAsync(
  x0: number[],
  loss: (v: number[]) => Promise<number>,
  opts: {
    maxIter: number;
    tol: number;
    step: number;
    /** Fired every iteration; `improved=true` when the iter beat the
     *  previous best (so callers can keep the official history clean
     *  while still updating a live "iter X / Y" UI). */
    onIter?: (iter: number, best: number, improved: boolean) => void;
    yieldEveryNIter: number;
    cooperativeYield: () => Promise<void>;
  },
): Promise<{ x: number[]; iters: number; best: number }> {
  const n = x0.length;
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const v = x0.slice();
    v[i] = (v[i] ?? 0) * (1 + opts.step) + ((v[i] ?? 0) === 0 ? opts.step : 0);
    simplex.push(v);
  }
  const scores: number[] = [];
  for (const v of simplex) scores.push(await loss(v));
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
    const improved = best < bestEverScore;
    if (improved) bestEverScore = best;
    opts.onIter?.(iter, best, improved);
    if (worst - best < opts.tol) return { x: simplex[0]!, iters: iter, best };
    const centroid = new Array(n).fill(0) as number[];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]!;
    }
    for (let j = 0; j < n; j++) centroid[j]! /= n;
    const xr = centroid.map((c, j) => c + (c - simplex[n]![j]!));
    const fr = await loss(xr);
    if (fr < scores[n - 1]! && fr >= scores[0]!) {
      simplex[n] = xr;
      scores[n] = fr;
    } else if (fr < scores[0]!) {
      const xe = centroid.map((c, j) => c + 2 * (c - simplex[n]![j]!));
      const fe = await loss(xe);
      if (fe < fr) {
        simplex[n] = xe;
        scores[n] = fe;
      } else {
        simplex[n] = xr;
        scores[n] = fr;
      }
    } else {
      const xc = centroid.map((c, j) => c + 0.5 * (simplex[n]![j]! - c));
      const fc = await loss(xc);
      if (fc < scores[n]!) {
        simplex[n] = xc;
        scores[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[0]!.map((b, j) => b + 0.5 * (simplex[i]![j]! - b));
          scores[i] = await loss(simplex[i]!);
        }
      }
    }
    if ((iter + 1) % opts.yieldEveryNIter === 0) {
      await opts.cooperativeYield();
    }
  }
  return { x: simplex[0]!, iters: opts.maxIter, best: scores[0]! };
}
