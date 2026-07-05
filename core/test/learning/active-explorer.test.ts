import { describe, expect, it } from 'vitest';
import { proposeNextBatch, type ExplorationCell } from 'kinocat/learning';

interface MockSpec { cellId: string; jitter: number; }

describe('proposeNextBatch — scores prefer high-error / high-uncertainty / low-count cells', () => {
  it('top picks include the seeded high-error cells', () => {
    const cells: ExplorationCell<MockSpec>[] = [
      { id: 'a', errorRms: 0.01, uncertaintyStd: 0.001, count: 100, sample: (r) => ({ cellId: 'a', jitter: r() }) },
      { id: 'b', errorRms: 0.5,  uncertaintyStd: 0.2,  count: 5,   sample: (r) => ({ cellId: 'b', jitter: r() }) },
      { id: 'c', errorRms: 0.3,  uncertaintyStd: 0.1,  count: 3,   sample: (r) => ({ cellId: 'c', jitter: r() }) },
      { id: 'd', errorRms: 0.05, uncertaintyStd: 0.01, count: 50,  sample: (r) => ({ cellId: 'd', jitter: r() }) },
      { id: 'e', errorRms: 0.4,  uncertaintyStd: 0.15, count: 4,   sample: (r) => ({ cellId: 'e', jitter: r() }) },
    ];
    const proposed = proposeNextBatch({ cells, budget: 60, seed: 42 });
    const counts: Record<string, number> = {};
    for (const p of proposed) counts[p.cellId] = (counts[p.cellId] ?? 0) + 1;
    // High-leverage cells b/c/e should dominate the count vs a/d.
    const highSum = (counts['b'] ?? 0) + (counts['c'] ?? 0) + (counts['e'] ?? 0);
    const lowSum = (counts['a'] ?? 0) + (counts['d'] ?? 0);
    expect(highSum).toBeGreaterThan(lowSum * 2);
  });

  it('alwaysInclude is prepended verbatim', () => {
    const cells: ExplorationCell<MockSpec>[] = [
      { id: 'a', errorRms: 0.1, count: 0, sample: (r) => ({ cellId: 'a', jitter: r() }) },
    ];
    const probes: MockSpec[] = [
      { cellId: '__probe__1', jitter: 0 },
      { cellId: '__probe__2', jitter: 0 },
    ];
    const proposed = proposeNextBatch({ cells, budget: 5, alwaysInclude: probes, seed: 1 });
    expect(proposed.slice(0, 2).map((p) => p.spec.cellId)).toEqual(['__probe__1', '__probe__2']);
    expect(proposed).toHaveLength(5);
  });

  it('uniform fallback when all scores are zero', () => {
    const cells: ExplorationCell<MockSpec>[] = [
      { id: 'a', errorRms: 0, uncertaintyStd: 0, count: 0, sample: (r) => ({ cellId: 'a', jitter: r() }) },
      { id: 'b', errorRms: 0, uncertaintyStd: 0, count: 0, sample: (r) => ({ cellId: 'b', jitter: r() }) },
    ];
    const proposed = proposeNextBatch({ cells, budget: 10, seed: 1, uncertaintyFloor: 0 });
    expect(proposed.length).toBe(10);
  });
});
