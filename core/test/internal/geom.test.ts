// Unit tests for the pure 2D geometry helpers used by the headless sim
// monitor's clearance/collision tracking. No Rapier, fast — these count
// toward the core 80% coverage gate.

import { describe, it, expect } from 'vitest';
import {
  pointSegmentDistance,
  polygonDistance,
  placeFootprint,
  polygonArea,
  convexPolygonIntersectionArea,
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

describe('polygonArea', () => {
  it('computes the area of an axis-aligned box (winding-agnostic)', () => {
    expect(polygonArea(box(0, 0, 2, 1))).toBeCloseTo(8, 12); // 4 x 2
    // Reverse winding gives the same (absolute) area.
    expect(polygonArea(box(0, 0, 2, 1).slice().reverse())).toBeCloseTo(8, 12);
  });

  it('returns 0 for a degenerate (<3 vertex) polygon', () => {
    expect(polygonArea([[0, 0], [1, 1]])).toBe(0);
  });

  it('computes the area of a rotated square (rotation-invariant)', () => {
    const local: Pt[] = [
      [1, 1],
      [-1, 1],
      [-1, -1],
      [1, -1],
    ];
    expect(polygonArea(placeFootprint(local, 3, -2, 0.7))).toBeCloseTo(4, 9);
  });
});

describe('convexPolygonIntersectionArea', () => {
  it('returns the full overlap when one box sits inside another', () => {
    // 1x1 box fully inside a 4x4 box ⇒ intersection is the small box (area 4).
    expect(convexPolygonIntersectionArea(box(0, 0, 1, 1), box(0, 0, 2, 2)))
      .toBeCloseTo(4, 9);
  });

  it('returns 0 for disjoint boxes', () => {
    expect(convexPolygonIntersectionArea(box(0, 0, 1, 1), box(10, 0, 1, 1)))
      .toBeCloseTo(0, 12);
  });

  it('computes the partial overlap of two offset boxes', () => {
    // Two 2x2 boxes (half=1) offset by 1 in x ⇒ overlap is 1 (x) by 2 (z) = 2.
    expect(convexPolygonIntersectionArea(box(0, 0, 1, 1), box(1, 0, 1, 1)))
      .toBeCloseTo(2, 9);
  });

  it('is symmetric in its arguments', () => {
    const a = box(0, 0, 1, 1);
    const b = box(0.5, 0.3, 1.5, 0.8);
    expect(convexPolygonIntersectionArea(a, b))
      .toBeCloseTo(convexPolygonIntersectionArea(b, a), 9);
  });

  it('drops coverage when the inner box is rotated out of the outer one', () => {
    // A 4.8 x 2.0 car-sized box exactly covering an identical stall box: full
    // overlap when aligned, strictly less once rotated (corners poke out).
    const car: Pt[] = [
      [2.4, 1.0],
      [-2.4, 1.0],
      [-2.4, -1.0],
      [2.4, -1.0],
    ];
    const stall = box(0, 0, 2.4, 1.0);
    const aligned = convexPolygonIntersectionArea(placeFootprint(car, 0, 0, 0), stall);
    const rotated = convexPolygonIntersectionArea(
      placeFootprint(car, 0, 0, 0.28), // ~16°
      stall,
    );
    expect(aligned).toBeCloseTo(polygonArea(stall), 6);
    expect(rotated).toBeLessThan(aligned * 0.95);
  });
});
