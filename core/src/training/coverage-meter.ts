// Generic N-dimensional coverage histogram over trial samples.
//
// Used by the dataset-coverage observatory (Phase 0 of the training-dataset
// plan): the caller supplies a projection from `(state, controls, config)`
// into a low-dimensional bin vector, and the meter accumulates per-bin
// counts + per-bin held-out prediction-error stats. The downstream UI
// (Model Lab) renders a heatmap; the CI gate asserts minimum trial counts
// per bin to prevent silent coverage regressions.
//
// Domain-agnostic — `S`, `C`, `Cfg` are opaque. The car projection lives in
// `kinocat/vehicle/car/coverage-projection.ts`; airplanes / other vehicles
// supply their own.

import type { Trial, TrialSplit } from '../learning/trial-store';

/** One dimension of the histogram. Linear binning between `lo` and `hi`. */
export interface CoverageAxis {
  /** Stable name (e.g. "speed", "yawRate"). */
  name: string;
  /** Lower edge (inclusive). Values below clamp into bin 0. */
  lo: number;
  /** Upper edge (exclusive). Values >= hi clamp into the last bin. */
  hi: number;
  /** Number of bins along this axis. */
  bins: number;
}

/** Projection from one observed sample to an N-dim point in axis space. */
export type CoverageProjection<S, C, Cfg> = (
  state: S,
  controls: ReadonlyArray<number>,
  cfg: Cfg,
) => number[];

export interface CoverageCellSummary {
  /** Flat-encoded bin id (e.g. "2,3,1,0,2"). */
  binId: string;
  /** Per-axis bin indices. */
  binIndex: number[];
  /** Per-axis bin midpoints (for axis-aware UI). */
  binMid: number[];
  /** Trial-sample count in this cell across `all`. */
  count: number;
  /** Count restricted to `train` split. */
  trainCount: number;
  /** Count restricted to `val` split. */
  valCount: number;
  /** Count restricted to `test` split (frozen reference). */
  testCount: number;
  /** Held-out (test split) position-error RMS in this cell. NaN when the
   *  cell has no test samples. */
  testErrorRms: number;
}

export interface CoverageMeterOptions<S, C, Cfg> {
  axes: CoverageAxis[];
  project: CoverageProjection<S, C, Cfg>;
  /** Convert opaque `C` to a flat number vector for the projection. */
  controlsToVec: (c: C) => number[];
}

export interface CoverageMeter<S, C, Cfg> {
  /** Add every per-tick sample of `trial` into the histogram. The trial's
   *  `split` is used to populate the per-split counters; the test split
   *  also drives `testErrorRms` when `errorPerSample` is supplied. */
  record(
    trial: Trial<S, C, Cfg>,
    errorPerSample?: ReadonlyArray<number>,
  ): void;
  /** Per-cell summary, only cells with count > 0. */
  summary(): CoverageCellSummary[];
  /** Number of cells with at least one sample across any split. */
  occupiedCells(): number;
  /** Total number of bins (Πaxes.bins). */
  totalCells(): number;
  /** Reset everything. */
  clear(): void;
}

interface CellAcc {
  binIndex: number[];
  count: number;
  trainCount: number;
  valCount: number;
  testCount: number;
  testSq: number;
  testN: number;
}

function clampedBin(x: number, axis: CoverageAxis): number {
  if (!Number.isFinite(x)) return 0;
  const u = (x - axis.lo) / Math.max(1e-12, axis.hi - axis.lo);
  const idx = Math.floor(u * axis.bins);
  if (idx < 0) return 0;
  if (idx >= axis.bins) return axis.bins - 1;
  return idx;
}

function midOf(axis: CoverageAxis, bin: number): number {
  const w = (axis.hi - axis.lo) / axis.bins;
  return axis.lo + (bin + 0.5) * w;
}

export function createCoverageMeter<S, C, Cfg>(
  opts: CoverageMeterOptions<S, C, Cfg>,
): CoverageMeter<S, C, Cfg> {
  const cells = new Map<string, CellAcc>();
  const totalCells = opts.axes.reduce((acc, a) => acc * a.bins, 1);

  function bucketize(
    state: S,
    controls: ReadonlyArray<number>,
    cfg: Cfg,
  ): { binId: string; binIndex: number[] } | null {
    const pt = opts.project(state, controls, cfg);
    if (pt.length !== opts.axes.length) return null;
    const idx = pt.map((v, i) => clampedBin(v, opts.axes[i]!));
    return { binId: idx.join(','), binIndex: idx };
  }

  return {
    record(trial, errorPerSample) {
      const split: TrialSplit = trial.split ?? 'train';
      for (let i = 0; i < trial.samples.length; i++) {
        const s = trial.samples[i]!;
        const ctrlIdx = Math.min(
          trial.controlsTrace.length - 1,
          Math.max(0, Math.floor(s.t / trial.dt)),
        );
        const cv = opts.controlsToVec(trial.controlsTrace[ctrlIdx]!);
        const b = bucketize(s.state, cv, trial.config);
        if (!b) continue;
        let acc = cells.get(b.binId);
        if (!acc) {
          acc = {
            binIndex: b.binIndex,
            count: 0,
            trainCount: 0,
            valCount: 0,
            testCount: 0,
            testSq: 0,
            testN: 0,
          };
          cells.set(b.binId, acc);
        }
        acc.count++;
        if (split === 'train') acc.trainCount++;
        else if (split === 'val') acc.valCount++;
        else acc.testCount++;
        if (split === 'test' && errorPerSample && errorPerSample.length > i) {
          const e = errorPerSample[i]!;
          if (Number.isFinite(e)) {
            acc.testSq += e * e;
            acc.testN++;
          }
        }
      }
    },
    summary() {
      const out: CoverageCellSummary[] = [];
      for (const [id, acc] of cells) {
        out.push({
          binId: id,
          binIndex: acc.binIndex,
          binMid: acc.binIndex.map((bi, i) => midOf(opts.axes[i]!, bi)),
          count: acc.count,
          trainCount: acc.trainCount,
          valCount: acc.valCount,
          testCount: acc.testCount,
          testErrorRms: acc.testN > 0 ? Math.sqrt(acc.testSq / acc.testN) : NaN,
        });
      }
      out.sort((a, b) => a.binId.localeCompare(b.binId));
      return out;
    },
    occupiedCells() {
      return cells.size;
    },
    totalCells() {
      return totalCells;
    },
    clear() {
      cells.clear();
    },
  };
}
