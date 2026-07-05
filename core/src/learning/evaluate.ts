// Generic evaluation harness for a learned dynamics model.
//
// The PRIMARY success metric is `openLoopDivergence`: starting from each
// trial's initial state, roll the model forward through the trial's
// recorded controls trace and measure RMS divergence from the recorded
// ground-truth state at each requested horizon T ∈ {0.1s, 0.5s, 1s, 2s, ...}.
// This is what the IGHA* planner actually depends on for plan quality.
//
// Also returns:
//   - per-state-component RMS (caller-supplied state-difference functions)
//   - coverage / density grids (caller supplies the cell-binning function)
//   - baseline comparisons (caller passes additional ForwardSims by name)

import type { ForwardSim } from '../primitives/types';
import type { Trial } from './trial-store';

export interface OpenLoopRow {
  /** Horizon in seconds. */
  tSec: number;
  /** Position RMS (meters) across trials at this horizon. */
  posRms: number;
  /** Heading RMS (radians). */
  headingRms: number;
  /** Speed RMS (m/s). */
  speedRms: number;
}

export interface CoverageCell {
  /** Bin index (caller-supplied; usually flat-encoded from N-D bins). */
  binId: string;
  /** Trial samples that fell into this cell. */
  count: number;
  /** RMS prediction error in this cell (model.predict vs. actual). */
  errorRms: number;
}

export interface ModelDiagnostics {
  /** Open-loop divergence at each requested horizon — the headline metric. */
  openLoopDivergence: OpenLoopRow[];
  /** Per-state-component RMS over all samples. */
  perStateRms: { name: string; rms: number }[];
  /** Coverage grid (one cell per bin). */
  coverage: CoverageCell[];
  /** Baseline `ForwardSim`s' open-loop divergence at the same horizons,
   *  keyed by name. Empty if no baselines were supplied. */
  baselines: Record<string, OpenLoopRow[]>;
  /** Optional per-split breakdown — populated by callers that re-evaluate
   *  the model on each of `train` / `val` / `test` partitions of the trial
   *  store. Absent when the caller didn't supply a split-aware trial set.
   *  Phase 0 of the training-dataset plan asserts this is the honest
   *  surface for cross-phase progress reports. */
  perSplit?: {
    train?: OpenLoopRow[];
    val?: OpenLoopRow[];
    test?: OpenLoopRow[];
  };
}

export interface ForwardSimUnderTest<S, C, Cfg> {
  /** Build a ForwardSim from the trial's config (so config-aware models
   *  evaluate at the right config per trial). */
  make: (cfg: Cfg) => ForwardSim<S>;
}

export interface EvaluateOptions<S, C, Cfg> {
  /** Held-out trial set (do not overlap with training). */
  trials: ReadonlyArray<Trial<S, C, Cfg>>;
  /** Model under test. */
  model: ForwardSimUnderTest<S, C, Cfg>;
  /** Horizons (seconds) to report open-loop divergence at. */
  horizons: number[];
  /** Convert `C` → opaque `number[]` controls for the sim. */
  controlsToVec: (c: C) => number[];
  /** Extract (x, z, heading, speed) from a state for RMS computation.
   *  Returns NaN for any missing component (and that sample is dropped). */
  extractMetricFields: (s: S) => {
    x: number;
    z: number;
    heading: number;
    speed: number;
  };
  /** Per-state-component RMS rows (consumer-supplied: e.g. yawRate,
   *  lateralVelocity, ...). Each returns the per-sample squared error. */
  perStateRmsFields?: Array<{
    name: string;
    sqError: (predicted: S, actual: S) => number;
  }>;
  /** Coverage-cell binning. Return a stable string id per sample to bin. */
  binSample?: (state: S, controls: ReadonlyArray<number>, cfg: Cfg) => string;
  /** Baselines to compare against (keyed by display name). */
  baselines?: Record<string, ForwardSimUnderTest<S, C, Cfg>>;
  /** Integration sub-steps per control tick used during the open-loop
   *  rollout. Default 1 (use the trial's controlsTrace as-is). */
  rolloutSubsteps?: number;
}

interface RolloutResult<S> {
  /** Predicted state at each tick of the trial's controlsTrace. */
  states: S[];
  /** Time per tick. */
  tickTimes: number[];
}

function rolloutTrial<S, C, Cfg>(
  sim: ForwardSim<S>,
  trial: Trial<S, C, Cfg>,
  controlsToVec: (c: C) => number[],
  substeps: number,
): RolloutResult<S> {
  const dt = trial.dt / substeps;
  let s: S = trial.initialState;
  const states: S[] = [s];
  const tickTimes: number[] = [0];
  let curT = 0;
  for (let i = 0; i < trial.controlsTrace.length; i++) {
    const cv = controlsToVec(trial.controlsTrace[i]!);
    for (let j = 0; j < substeps; j++) {
      s = sim(s, cv, dt);
      curT += dt;
    }
    states.push(s);
    tickTimes.push(curT);
  }
  return { states, tickTimes };
}

function rmsAtHorizons<S, C, Cfg>(
  trials: ReadonlyArray<Trial<S, C, Cfg>>,
  modelSimMaker: (cfg: Cfg) => ForwardSim<S>,
  controlsToVec: (c: C) => number[],
  extract: EvaluateOptions<S, C, Cfg>['extractMetricFields'],
  horizons: number[],
  substeps: number,
): OpenLoopRow[] {
  // For each horizon, accumulate squared error across trials. We bin trial
  // samples by their `t` (closest sample within ±dt/2 of the horizon).
  const accum = horizons.map(() => ({ n: 0, posSq: 0, headSq: 0, speedSq: 0 }));
  for (const trial of trials) {
    const sim = modelSimMaker(trial.config);
    const result = rolloutTrial(sim, trial, controlsToVec, substeps);
    for (let h = 0; h < horizons.length; h++) {
      const tTgt = horizons[h]!;
      // Find recorded sample closest to tTgt.
      let bestIdx = -1;
      let bestDt = Infinity;
      for (let i = 0; i < trial.samples.length; i++) {
        const dt = Math.abs(trial.samples[i]!.t - tTgt);
        if (dt < bestDt) { bestDt = dt; bestIdx = i; }
      }
      if (bestIdx < 0 || bestDt > trial.dt) continue;
      const actualSample = trial.samples[bestIdx]!;
      // Find the predicted state closest in time.
      let bestPredIdx = -1;
      let bestPredDt = Infinity;
      for (let i = 0; i < result.tickTimes.length; i++) {
        const dt = Math.abs(result.tickTimes[i]! - actualSample.t);
        if (dt < bestPredDt) { bestPredDt = dt; bestPredIdx = i; }
      }
      if (bestPredIdx < 0) continue;
      const pred = extract(result.states[bestPredIdx]!);
      const act = extract(actualSample.state);
      if (![pred.x, pred.z, pred.heading, pred.speed].every(Number.isFinite)) continue;
      const dx = pred.x - act.x;
      const dz = pred.z - act.z;
      let dh = pred.heading - act.heading;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const ds = pred.speed - act.speed;
      const a = accum[h]!;
      a.posSq += dx * dx + dz * dz;
      a.headSq += dh * dh;
      a.speedSq += ds * ds;
      a.n++;
    }
  }
  return horizons.map((t, i) => {
    const a = accum[i]!;
    const n = Math.max(1, a.n);
    return {
      tSec: t,
      posRms: Math.sqrt(a.posSq / n),
      headingRms: Math.sqrt(a.headSq / n),
      speedRms: Math.sqrt(a.speedSq / n),
    };
  });
}

export function evaluateModel<S, C, Cfg>(
  opts: EvaluateOptions<S, C, Cfg>,
): ModelDiagnostics {
  const substeps = opts.rolloutSubsteps ?? 1;
  const openLoop = rmsAtHorizons(
    opts.trials, opts.model.make, opts.controlsToVec, opts.extractMetricFields,
    opts.horizons, substeps,
  );
  const baselines: Record<string, OpenLoopRow[]> = {};
  if (opts.baselines) {
    for (const [name, bl] of Object.entries(opts.baselines)) {
      baselines[name] = rmsAtHorizons(
        opts.trials, bl.make, opts.controlsToVec, opts.extractMetricFields,
        opts.horizons, substeps,
      );
    }
  }
  // Per-state RMS.
  const perStateRms: { name: string; rms: number }[] = [];
  if (opts.perStateRmsFields) {
    for (const field of opts.perStateRmsFields) {
      let sq = 0;
      let n = 0;
      for (const trial of opts.trials) {
        const sim = opts.model.make(trial.config);
        let s: S = trial.initialState;
        for (let k = 1; k < trial.samples.length; k++) {
          const a = trial.samples[k - 1]!;
          const b = trial.samples[k]!;
          const dt = (b.t - a.t);
          const ctrlIdx = Math.min(trial.controlsTrace.length - 1, Math.floor(b.t / trial.dt));
          const cv = opts.controlsToVec(trial.controlsTrace[ctrlIdx]!);
          s = sim(s, cv, dt);
          sq += field.sqError(s, b.state);
          n++;
        }
      }
      perStateRms.push({ name: field.name, rms: Math.sqrt(sq / Math.max(1, n)) });
    }
  }
  // Coverage.
  const coverage: CoverageCell[] = [];
  if (opts.binSample) {
    const cells = new Map<string, { count: number; sq: number }>();
    for (const trial of opts.trials) {
      const sim = opts.model.make(trial.config);
      let s: S = trial.initialState;
      for (let k = 1; k < trial.samples.length; k++) {
        const a = trial.samples[k - 1]!;
        const b = trial.samples[k]!;
        const dt = b.t - a.t;
        const ctrlIdx = Math.min(trial.controlsTrace.length - 1, Math.floor(b.t / trial.dt));
        const cv = opts.controlsToVec(trial.controlsTrace[ctrlIdx]!);
        s = sim(s, cv, dt);
        const pred = opts.extractMetricFields(s);
        const act = opts.extractMetricFields(b.state);
        const sq = (pred.x - act.x) ** 2 + (pred.z - act.z) ** 2;
        const binId = opts.binSample(a.state, cv, trial.config);
        const cur = cells.get(binId) ?? { count: 0, sq: 0 };
        cur.count++;
        cur.sq += sq;
        cells.set(binId, cur);
      }
    }
    for (const [id, c] of cells) {
      coverage.push({ binId: id, count: c.count, errorRms: Math.sqrt(c.sq / c.count) });
    }
  }
  return { openLoopDivergence: openLoop, perStateRms, coverage, baselines };
}
