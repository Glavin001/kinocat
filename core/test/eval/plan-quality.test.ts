import { describe, it, expect } from 'vitest';
import { scorePlan, rolloutTeleportFollow } from '../../src/eval/plan-quality';
import { trackingMetrics } from '../../src/eval/tracking-metrics';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { arcPath } from '../../src/eval/reference-shapes';
import type { DynamicLimits } from '../../src/eval/feasibility';

const limits: DynamicLimits = {
  frictionLimit: 4,
  minTurnRadius: 4,
  maxAccel: 6.5,
  maxDecel: 8,
};

describe('scorePlan', () => {
  it('scores a feasible arc with terminal accuracy', () => {
    const plan = arcPath({ radius: 20, sweep: Math.PI / 2, speed: 6, ds: 0.5 });
    const goalState = plan[plan.length - 1]!;
    const report = scorePlan(plan, limits, {
      goal: { x: goalState.x, z: goalState.z, heading: goalState.heading, speed: goalState.speed },
    });
    expect(report.feasibility.feasible).toBe(true);
    expect(report.terminal!.posError).toBeLessThan(1e-6);
    expect(report.terminal!.headingError).toBeLessThan(1e-6);
    expect(report.pathLength).toBeCloseTo(20 * (Math.PI / 2), 0);
    expect(report.gg.meanUtil).toBeGreaterThan(0);
  });
});

describe('rolloutTeleportFollow', () => {
  it('reproduces the plan with ~zero tracking error (near-perfect tracker)', () => {
    const plan = arcPath({ radius: 15, sweep: Math.PI / 2, speed: 5, ds: 0.5 });
    const executed = rolloutTeleportFollow(plan, 0.05);
    const ref = toReferenceTrajectory(plan);
    const report = trackingMetrics(executed, ref, { dt: 0.05 });
    // Teleport-follow rides the reference, so cross-track error is tiny.
    expect(report.crossTrack.rmse).toBeLessThan(0.05);
  });
});
