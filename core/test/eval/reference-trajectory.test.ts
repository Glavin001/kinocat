import { describe, it, expect } from 'vitest';
import {
  toReferenceTrajectory,
  referencePoseAt,
  referenceLength,
} from '../../src/eval/reference-trajectory';
import { arcPath, straightLine, reversePark } from '../../src/eval/reference-shapes';
import { curvaturePerSample } from '../../src/execute/speed-profile';
import { checkFeasibility } from '../../src/eval/feasibility';
import type { CarKinematicState } from '../../src/agent/types';

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

describe('gear-cusp curvature', () => {
  it('does NOT emit a phantom curvature spike at a Reeds-Shepp cusp', () => {
    // Drive forward to (2,0), then reverse backing up-and-left. The Menger
    // triple straddling the reversal has a collapsed far side |C−A| ≈ 0.05 m,
    // so the raw formula reports a huge curvature there.
    const path: CarKinematicState[] = [
      { x: 0, z: 0, heading: 0, speed: 1, t: 0 },
      { x: 1, z: 0, heading: 0, speed: 1, t: 1 },
      { x: 2, z: 0, heading: 0, speed: 1, t: 2 }, // cusp: forward ends here
      { x: 1, z: 0.05, heading: Math.PI, speed: -1, t: 3 }, // reverse begins
      { x: 0, z: 0.1, heading: Math.PI, speed: -1, t: 4 },
      { x: -1, z: 0.15, heading: Math.PI, speed: -1, t: 5 },
    ];
    // Raw Menger curvature spikes at the cusp (this is the bug)…
    const naive = curvaturePerSample(path);
    expect(naive[2]!).toBeGreaterThan(1); // ~2 m⁻¹ phantom
    // …but the gear-split reference-trajectory curvature is ~0 there.
    const ref = toReferenceTrajectory(path);
    expect(ref[2]!.kappa).toBeLessThan(0.05);
  });

  it('keeps a real reverse-park plan dynamically feasible', () => {
    // Without the cusp split, the phantom κ trips the turn-radius check and the
    // whole (perfectly drivable) parking plan is misdiagnosed as infeasible.
    const park = reversePark();
    // The plan really does reverse (a signed-speed sign flip exists).
    expect(park.some((p) => p.speed < 0)).toBe(true);
    expect(park.some((p) => p.speed > 0)).toBe(true);
    const ref = toReferenceTrajectory(park);
    const report = checkFeasibility(ref, {
      frictionLimit: 4,
      minTurnRadius: 4,
      maxAccel: 6.5,
      maxDecel: 8,
    });
    expect(report.feasible).toBe(true);
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
