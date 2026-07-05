import { describe, it, expect } from 'vitest';
import {
  planCrossesRegion,
  markAffectedAgents,
  footprintCircumradius,
  type ChangedRegion,
} from '../../src/execute/invalidation';
import { ReplanState } from '../../src/execute/replan';
import { segmentIntersectsAABB } from '../../src/internal/geom';
import type { PlanPath } from '../../src/execute/types';

const TRIGGER = { divergenceThresholdMeters: 1.5, refreshIntervalMs: 500 };

function straightPath(x0: number, x1: number, z: number): PlanPath {
  return [
    { x: x0, z, heading: 0, speed: 4, t: 0 },
    { x: x1, z, heading: 0, speed: 4, t: (x1 - x0) / 4 },
  ];
}

describe('segmentIntersectsAABB', () => {
  it('detects endpoint-inside, crossing, and miss', () => {
    // Endpoint inside.
    expect(segmentIntersectsAABB(1, 1, 10, 10, 0, 0, 2, 2)).toBe(true);
    // Straight crossing (both endpoints outside).
    expect(segmentIntersectsAABB(-5, 1, 5, 1, -1, 0, 1, 2)).toBe(true);
    // Diagonal crossing.
    expect(segmentIntersectsAABB(-2, -2, 2, 2, -1, -1, 1, 1)).toBe(true);
    // Clear miss.
    expect(segmentIntersectsAABB(-5, 5, 5, 5, -1, 0, 1, 2)).toBe(false);
    // Corner near-miss: passes outside the box corner.
    expect(segmentIntersectsAABB(2, 0, 0, 2, 0, 0, 0.5, 0.5)).toBe(false);
    // Axis-parallel degenerate direction handled by the slab guards.
    expect(segmentIntersectsAABB(0.5, -5, 0.5, 5, 0, 0, 1, 1)).toBe(true);
    expect(segmentIntersectsAABB(3, -5, 3, 5, 0, 0, 1, 1)).toBe(false);
  });
});

describe('planCrossesRegion', () => {
  const region: ChangedRegion = { x0: 4, z0: -1, x1: 6, z1: 1 };

  it('detects a plan passing through the region', () => {
    expect(planCrossesRegion(straightPath(0, 10, 0), region)).toBe(true);
  });

  it('rejects a plan far from the region', () => {
    expect(planCrossesRegion(straightPath(0, 10, 5), region)).toBe(false);
  });

  it('inflation flips a near-miss', () => {
    const nearMiss = straightPath(0, 10, 1.5); // 0.5m outside the box
    expect(planCrossesRegion(nearMiss, region)).toBe(false);
    expect(planCrossesRegion(nearMiss, region, 1.0)).toBe(true);
  });

  it('handles empty and single-pose paths', () => {
    expect(planCrossesRegion([], region)).toBe(false);
    expect(planCrossesRegion([{ x: 5, z: 0, heading: 0, speed: 0, t: 0 }], region)).toBe(true);
    expect(planCrossesRegion([{ x: 5, z: 9, heading: 0, speed: 0, t: 0 }], region)).toBe(false);
  });

  it('polygon refinement rejects AABB-only hits', () => {
    // Triangle occupying the lower-left half of the box [4,6]×[-1,1]; a path
    // clipping the upper-right AABB corner misses the triangle itself. At
    // z=0.8 the triangle interior spans only x∈[4,4.2].
    const tri: ChangedRegion = {
      x0: 4, z0: -1, x1: 6, z1: 1,
      polygon: [[4, -1], [6, -1], [4, 1]],
    };
    expect(planCrossesRegion(straightPath(5.5, 10, 0.8), tri)).toBe(false);
    expect(planCrossesRegion(straightPath(0, 10, -0.9), tri)).toBe(true);
  });
});

describe('footprintCircumradius', () => {
  it('returns the farthest vertex distance', () => {
    expect(footprintCircumradius([[1, 0], [-1, 0], [0, 2]])).toBeCloseTo(2, 9);
    expect(footprintCircumradius([])).toBe(0);
  });
});

describe('markAffectedAgents', () => {
  it('marks only agents whose committed plan crosses the region', () => {
    const region: ChangedRegion = { x0: 4, z0: -1, x1: 6, z1: 1 };
    const crossing = new ReplanState(TRIGGER);
    crossing.setPlan(straightPath(0, 10, 0), 0);
    const far = new ReplanState(TRIGGER);
    far.setPlan(straightPath(0, 10, 8), 0);
    const planless = new ReplanState(TRIGGER);

    const marked = markAffectedAgents(region, [
      { replan: crossing },
      { replan: far },
      { replan: planless },
    ]);

    expect(marked).toEqual([crossing]);
    const at = { x: 1, z: 0, heading: 0, speed: 4, t: 0.25 };
    expect(crossing.shouldReplan(at, 1)).toBe(true);
    expect(far.shouldReplan({ ...at, z: 8 }, 1)).toBe(false);
  });

  it('inflation is honored per agent', () => {
    const region: ChangedRegion = { x0: 4, z0: -1, x1: 6, z1: 1 };
    const nearMiss = new ReplanState(TRIGGER);
    nearMiss.setPlan(straightPath(0, 10, 1.5), 0);
    expect(markAffectedAgents(region, [{ replan: nearMiss }])).toEqual([]);
    expect(markAffectedAgents(region, [{ replan: nearMiss, inflate: 1 }])).toEqual([
      nearMiss,
    ]);
  });
});
