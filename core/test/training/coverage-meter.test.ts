import { describe, expect, it } from 'vitest';
import {
  createCoverageMeter,
  type CoverageAxis,
} from 'kinocat/training';
import type { Trial } from 'kinocat/learning';

interface S { v: number }
interface C { u: number }
interface Cfg { id: string }

const axes: CoverageAxis[] = [
  { name: 'v', lo: 0, hi: 10, bins: 5 },
  { name: 'u', lo: -1, hi: 1, bins: 4 },
];

function mk(id: string, samples: { t: number; state: S }[], controls: C[], split: 'train' | 'val' | 'test' = 'train'): Trial<S, C, Cfg> {
  return {
    id,
    initialState: samples[0]!.state,
    controlsTrace: controls,
    dt: 0.1,
    samples,
    config: { id: 'A' },
    configKey: 'A',
    split,
  };
}

describe('createCoverageMeter', () => {
  const meter = createCoverageMeter<S, C, Cfg>({
    axes,
    project: (s, c) => [s.v, c[0] ?? 0],
    controlsToVec: (c) => [c.u],
  });

  it('records counts per cell and totals', () => {
    meter.clear();
    meter.record(mk('a', [
      { t: 0, state: { v: 1 } },
      { t: 0.1, state: { v: 2 } },
    ], [{ u: 0.5 }, { u: 0.5 }]));
    const cells = meter.summary();
    expect(cells.length).toBeGreaterThan(0);
    const totalCount = cells.reduce((acc, c) => acc + c.count, 0);
    expect(totalCount).toBe(2);
    expect(meter.totalCells()).toBe(5 * 4);
  });

  it('separates counts per split', () => {
    meter.clear();
    meter.record(mk('a', [{ t: 0, state: { v: 1 } }], [{ u: 0 }], 'train'));
    meter.record(mk('b', [{ t: 0, state: { v: 1 } }], [{ u: 0 }], 'val'));
    meter.record(mk('c', [{ t: 0, state: { v: 1 } }], [{ u: 0 }], 'test'));
    const cells = meter.summary();
    const trainTotal = cells.reduce((acc, c) => acc + c.trainCount, 0);
    const valTotal = cells.reduce((acc, c) => acc + c.valCount, 0);
    const testTotal = cells.reduce((acc, c) => acc + c.testCount, 0);
    expect(trainTotal).toBe(1);
    expect(valTotal).toBe(1);
    expect(testTotal).toBe(1);
  });

  it('accumulates test RMS from errorPerSample', () => {
    meter.clear();
    const trial = mk('a', [
      { t: 0, state: { v: 1 } },
      { t: 0.1, state: { v: 1 } },
    ], [{ u: 0 }, { u: 0 }], 'test');
    meter.record(trial, [3, 4]); // sqrt((9+16)/2) = ~3.54
    const cells = meter.summary();
    const rms = cells.find((c) => c.testCount > 0)?.testErrorRms;
    expect(rms).toBeCloseTo(Math.sqrt(25 / 2), 5);
  });

  it('out-of-range projections clamp into edge bins', () => {
    meter.clear();
    meter.record(mk('a', [
      { t: 0, state: { v: -100 } },
      { t: 0.1, state: { v: 9999 } },
    ], [{ u: -99 }, { u: 99 }]));
    const cells = meter.summary();
    expect(cells.length).toBe(2);
    const bins = new Set(cells.flatMap((c) => c.binIndex));
    expect(bins.has(0)).toBe(true);
    expect(bins.has(4)).toBe(true); // last bin index of axis 0 (5 bins)
  });
});
