import { describe, it, expect } from 'vitest';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { projectOntoPath } from '../../src/eval/projection';
import { straightLine } from '../../src/eval/reference-shapes';

describe('projectOntoPath', () => {
  const ref = toReferenceTrajectory(straightLine({ length: 10, speed: 5, ds: 0.5 }));

  it('cross-track of a point offset d from a straight line equals d', () => {
    const proj = projectOntoPath(ref, 5, 0.7);
    expect(Math.abs(proj.crossTrack)).toBeCloseTo(0.7, 5);
    expect(proj.s).toBeCloseTo(5, 5);
  });

  it('signs cross-track by side of the path', () => {
    // Path runs +x; +z is to the left of the travel direction.
    const left = projectOntoPath(ref, 5, 1.0);
    const right = projectOntoPath(ref, 5, -1.0);
    expect(Math.sign(left.crossTrack)).toBe(1);
    expect(Math.sign(right.crossTrack)).toBe(-1);
  });

  it('interpolates the foot point mid-segment (does not snap to a vertex)', () => {
    // ds=0.5 ⇒ vertices at 0, 0.5, 1.0...; a point at x=0.73 should project to
    // s≈0.73, not to the nearest stored vertex 0.5 or 1.0.
    const proj = projectOntoPath(ref, 0.73, 0.2);
    expect(proj.s).toBeCloseTo(0.73, 2);
  });

  it('a point on the path has ~0 cross-track', () => {
    const proj = projectOntoPath(ref, 3.3, 0);
    expect(Math.abs(proj.crossTrack)).toBeLessThan(1e-6);
  });
});
