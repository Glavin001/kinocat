import { describe, it, expect } from 'vitest';
import { buildPlan, toStatePath } from '../../src/plan/build';
import type { CarKinematicState } from '../../src/agent/types';

function straightLine(n: number, ds: number, v: number): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ x: i * ds, z: 0, heading: 0, speed: v, t: i * (ds / Math.abs(v)) });
  }
  return out;
}

/** A left-hand circle of radius R, swept at constant speed. Heading is the
 *  tangent (theta + pi/2 for a CCW sweep). */
function fullCircle(n: number, R: number, v: number): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  const ds = (2 * Math.PI * R) / (n - 1);
  for (let i = 0; i < n; i++) {
    const theta = (i / (n - 1)) * 2 * Math.PI;
    out.push({
      x: R * Math.sin(theta),
      z: R - R * Math.cos(theta), // turns left
      heading: theta,
      speed: v,
      t: i * (ds / Math.abs(v)),
    });
  }
  return out;
}

describe('buildPlan', () => {
  it('produces constant signed kappa and steerFf on a circle', () => {
    const R = 5; // |kappa| = 1/R = 0.2
    const L = 3;
    const plan = buildPlan(fullCircle(40, R, 6), { wheelBase: L });
    expect(plan.points.length).toBe(40);
    // Interior points: |kappa| ~ 0.2, consistent (left → positive) sign.
    for (let i = 2; i < plan.points.length - 2; i++) {
      const k = plan.points[i]!.kappa;
      expect(Math.abs(k)).toBeCloseTo(0.2, 1);
      expect(k).toBeGreaterThan(0); // left-hand circle ⇒ positive curvature
      expect(plan.points[i]!.steerFf).toBeCloseTo(Math.atan(L * k), 6);
    }
  });

  it('leaves steerFf undefined when no wheelbase is supplied', () => {
    const plan = buildPlan(fullCircle(20, 5, 6));
    for (const p of plan.points) expect(p.steerFf).toBeUndefined();
  });

  it('has near-zero curvature and a single forward segment on a straight line', () => {
    const plan = buildPlan(straightLine(10, 1, 8));
    for (let i = 1; i < plan.points.length - 1; i++) {
      expect(plan.points[i]!.kappa).toBeCloseTo(0, 5);
    }
    expect(plan.segments.length).toBe(1);
    expect(plan.segments[0]!.direction).toBe(1);
    expect(plan.segments[0]!.startIdx).toBe(0);
    expect(plan.segments[0]!.endIdx).toBe(9);
  });

  it('accumulates monotonic arc length matching the polyline length', () => {
    const plan = buildPlan(straightLine(10, 2, 8));
    expect(plan.points[0]!.s).toBe(0);
    for (let i = 1; i < plan.points.length; i++) {
      expect(plan.points[i]!.s).toBeGreaterThanOrEqual(plan.points[i - 1]!.s);
    }
    expect(plan.points[plan.points.length - 1]!.s).toBeCloseTo(18, 6); // 9 steps × 2 m
  });

  it('computes aRef from the speed ramp', () => {
    // Straight path, speed ramps 0 → 9 over 10 samples at 1 m spacing.
    const path: CarKinematicState[] = [];
    let t = 0;
    for (let i = 0; i < 10; i++) {
      const v = 1 + i; // 1..10 m/s
      path.push({ x: i, z: 0, heading: 0, speed: v, t });
      // advance time by ds / v_avg to the next sample
      if (i < 9) t += 1 / ((v + (v + 1)) / 2);
    }
    const plan = buildPlan(path);
    // dv/dt should be positive and finite at interior samples.
    for (let i = 1; i < plan.points.length - 1; i++) {
      expect(plan.points[i]!.aRef).toBeGreaterThan(0);
      expect(plan.points[i]!.accelFf).toBe(plan.points[i]!.aRef);
    }
  });

  it('splits at a forward→reverse cusp into two single-gear segments', () => {
    // Forward 0..5, then reverse 5..9 (speed sign flips at index 5).
    const path: CarKinematicState[] = [];
    for (let i = 0; i < 5; i++) path.push({ x: i, z: 0, heading: 0, speed: 4, t: i });
    for (let i = 5; i < 10; i++) path.push({ x: 9 - i, z: 0, heading: 0, speed: -4, t: i });
    const plan = buildPlan(path);
    expect(plan.segments.length).toBe(2);
    expect(plan.segments[0]!.direction).toBe(1);
    expect(plan.segments[1]!.direction).toBe(-1);
    // Contiguous: the first segment ends where the second begins (shared cusp).
    expect(plan.segments[0]!.endIdx).toBe(plan.segments[1]!.startIdx);
  });

  it('gates tier-3 fields on the source state carrying them', () => {
    const withDyn: CarKinematicState[] = [
      { x: 0, z: 0, heading: 0, speed: 4, t: 0, yawRate: 0.3, lateralVelocity: 1 },
      { x: 1, z: 0, heading: 0, speed: 4, t: 0.25, yawRate: 0.3, lateralVelocity: 1 },
      { x: 2, z: 0, heading: 0, speed: 4, t: 0.5, yawRate: 0.3, lateralVelocity: 1 },
    ];
    const planA = buildPlan(withDyn);
    expect(planA.points[0]!.rRef).toBe(0.3);
    expect(planA.points[0]!.betaRef).toBeCloseTo(Math.atan2(1, 4), 6);
    expect(planA.points[0]!.dLeft).toBeUndefined();
    expect(planA.points[0]!.dRight).toBeUndefined();

    const planB = buildPlan(straightLine(5, 1, 4));
    for (const p of planB.points) {
      expect(p.rRef).toBeUndefined();
      expect(p.betaRef).toBeUndefined();
    }
  });

  it('round-trips through toStatePath', () => {
    const path = fullCircle(20, 5, 6);
    const back = toStatePath(buildPlan(path));
    expect(back.length).toBe(path.length);
    for (let i = 0; i < path.length; i++) {
      expect(back[i]!.x).toBeCloseTo(path[i]!.x, 6);
      expect(back[i]!.z).toBeCloseTo(path[i]!.z, 6);
      expect(back[i]!.heading).toBeCloseTo(path[i]!.heading, 6);
      expect(back[i]!.speed).toBeCloseTo(path[i]!.speed, 6);
      expect(back[i]!.t).toBeCloseTo(path[i]!.t, 6);
    }
  });

  it('does not mutate the input path', () => {
    const path = straightLine(6, 1, 5);
    const snapshot = JSON.stringify(path);
    buildPlan(path, { wheelBase: 3 });
    expect(JSON.stringify(path)).toBe(snapshot);
  });
});
