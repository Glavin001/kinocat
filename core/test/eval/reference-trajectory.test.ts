import { describe, it, expect } from 'vitest';
import {
  toReferenceTrajectory,
  referencePoseAt,
  referenceLength,
} from '../../src/eval/reference-trajectory';
import { arcPath, straightLine } from '../../src/eval/reference-shapes';

describe('toReferenceTrajectory', () => {
  it('accumulates arc-length monotonically', () => {
    const ref = toReferenceTrajectory(straightLine({ length: 10, speed: 5, ds: 1 }));
    expect(ref[0]!.s).toBe(0);
    for (let i = 1; i < ref.length; i++) {
      expect(ref[i]!.s).toBeGreaterThan(ref[i - 1]!.s);
    }
    expect(referenceLength(ref)).toBeCloseTo(10, 5);
  });

  it('recovers curvature ≈ 1/R for a circular arc', () => {
    const R = 8;
    const ref = toReferenceTrajectory(
      arcPath({ radius: R, sweep: Math.PI / 2, speed: 4, ds: 0.25 }),
    );
    // Interior samples should have curvature near 1/R.
    const interior = ref.slice(2, ref.length - 2);
    const meanKappa = interior.reduce((a, p) => a + p.kappa, 0) / interior.length;
    expect(meanKappa).toBeCloseTo(1 / R, 1);
  });

  it('derives longitudinal acceleration from the speed profile', () => {
    // Hand-built accelerating straight: speed ramps 0→10 over the path.
    const n = 11;
    const path = Array.from({ length: n }, (_, i) => ({
      x: i,
      z: 0,
      heading: 0,
      speed: i, // 0,1,...,10 m/s
      t: i === 0 ? 0 : undefined as unknown as number,
    }));
    // Fill t from constant-accel-ish spacing (use dt from ds/v average).
    let t = 0;
    for (let i = 1; i < n; i++) {
      const vAvg = 0.5 * (path[i]!.speed + path[i - 1]!.speed);
      t += 1 / Math.max(vAvg, 0.5);
      path[i]!.t = t;
    }
    const ref = toReferenceTrajectory(path);
    // Acceleration is positive everywhere (speeding up).
    for (let i = 1; i < ref.length - 1; i++) {
      expect(ref[i]!.a).toBeGreaterThan(0);
    }
  });
});

describe('referencePoseAt', () => {
  it('interpolates between samples by arc-length', () => {
    const ref = toReferenceTrajectory(straightLine({ length: 10, speed: 5, ds: 2 }));
    const mid = referencePoseAt(ref, 5)!;
    expect(mid.x).toBeCloseTo(5, 5);
    expect(mid.z).toBeCloseTo(0, 5);
  });

  it('clamps to the ends', () => {
    const ref = toReferenceTrajectory(straightLine({ length: 10, speed: 5, ds: 2 }));
    expect(referencePoseAt(ref, -1)!.s).toBe(0);
    expect(referencePoseAt(ref, 999)!.x).toBeCloseTo(10, 5);
  });
});
