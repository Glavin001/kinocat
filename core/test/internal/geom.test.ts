// Unit tests for the pure 2D geometry helpers used by the headless sim
// monitor's clearance/collision tracking. No Rapier, fast — these count
// toward the core 80% coverage gate.

import { describe, it, expect } from 'vitest';
import {
  pointSegmentDistance,
  polygonDistance,
  placeFootprint,
  type Pt,
} from '../../src/internal/geom';

/** Axis-aligned box centred at (cx,cz) with half-extents (hx,hz). */
function box(cx: number, cz: number, hx: number, hz: number): Pt[] {
  return [
    [cx - hx, cz - hz],
    [cx + hx, cz - hz],
    [cx + hx, cz + hz],
    [cx - hx, cz + hz],
  ];
}

describe('pointSegmentDistance', () => {
  it('measures perpendicular distance when the foot is inside the segment', () => {
    // Segment along x-axis from (0,0) to (10,0); point above its midpoint.
    expect(pointSegmentDistance(5, 3, 0, 0, 10, 0)).toBeCloseTo(3, 12);
  });

  it('clamps to the near endpoint when the foot falls before the segment', () => {
    expect(pointSegmentDistance(-4, 0, 0, 0, 10, 0)).toBeCloseTo(4, 12);
  });

  it('clamps to the far endpoint when the foot falls after the segment', () => {
    // (14,3) projects past b=(10,0): distance to b = hypot(4,3) = 5.
    expect(pointSegmentDistance(14, 3, 0, 0, 10, 0)).toBeCloseTo(5, 12);
  });

  it('reduces to point-to-point for a degenerate zero-length segment', () => {
    // (4,5) to the degenerate point (1,1): hypot(3,4) = 5.
    expect(pointSegmentDistance(4, 5, 1, 1, 1, 1)).toBeCloseTo(5, 12);
  });
});

describe('polygonDistance', () => {
  it('returns the gap between two separated boxes', () => {
    // Box A spans x in [-1,1]; box B spans x in [4,6]. Gap along x = 3.
    const a = box(0, 0, 1, 1);
    const b = box(5, 0, 1, 1);
    expect(polygonDistance(a, b)).toBeCloseTo(3, 12);
  });

  it('returns 0 for overlapping boxes', () => {
    const a = box(0, 0, 2, 2);
    const b = box(1, 1, 2, 2);
    expect(polygonDistance(a, b)).toBe(0);
  });

  it('returns ~0 for touching boxes (shared edge)', () => {
    const a = box(0, 0, 1, 1); // right edge at x=1
    const b = box(2, 0, 1, 1); // left edge at x=1
    expect(polygonDistance(a, b)).toBeCloseTo(0, 9);
  });

  it('is symmetric', () => {
    const a = box(0, 0, 1, 1);
    const b = box(5, 0, 1, 1);
    expect(polygonDistance(a, b)).toBeCloseTo(polygonDistance(b, a), 12);
  });

  it('measures clearance for a rotated footprint against an axis-aligned box', () => {
    // A 1x1 footprint (half-extents 0.5) centred at the origin, rotated 45°.
    // Its rightmost vertex reaches x = sqrt(0.5^2+0.5^2) = ~0.7071.
    const local: Pt[] = [
      [0.5, 0.5],
      [-0.5, 0.5],
      [-0.5, -0.5],
      [0.5, -0.5],
    ];
    const fp = placeFootprint(local, 0, 0, Math.PI / 4);
    // Obstacle box whose left edge sits at x = 3.
    const obs = box(4, 0, 1, 1);
    const expected = 3 - Math.SQRT1_2; // ~2.2929
    expect(polygonDistance(fp, obs)).toBeCloseTo(expected, 9);
  });
});
