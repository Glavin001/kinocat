// Targeted tests for branches the primary suites don't exercise: the analytic
// shape generators, the curvature-rate feasibility check, and a few edge cases.
import { describe, it, expect } from 'vitest';
import { laneChange, slalom, straightLine, arcPath } from '../../src/eval/reference-shapes';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { projectOntoPath } from '../../src/eval/projection';
import { checkFeasibility, type DynamicLimits } from '../../src/eval/feasibility';
import { scorePlan, rolloutTeleportFollow } from '../../src/eval/plan-quality';

describe('reference-shapes generators', () => {
  it('laneChange offsets by the requested width with monotonic x', () => {
    const path = laneChange({ width: 3.5, length: 30, speed: 6, ds: 0.5 });
    expect(path[0]!.z).toBeCloseTo(0, 3);
    expect(path[path.length - 1]!.z).toBeCloseTo(3.5, 3);
    for (let i = 1; i < path.length; i++) expect(path[i]!.x).toBeGreaterThan(path[i - 1]!.x);
  });

  it('slalom weaves around zero within the amplitude band', () => {
    const path = slalom({ spacing: 12, amplitude: 2, cones: 4, speed: 5, ds: 0.5 });
    const maxAbsZ = Math.max(...path.map((p) => Math.abs(p.z)));
    expect(maxAbsZ).toBeLessThanOrEqual(2 + 1e-6);
    expect(maxAbsZ).toBeGreaterThan(1.5);
  });

  it('straightLine has zero curvature and slalom has nonzero curvature', () => {
    const straight = toReferenceTrajectory(straightLine({ length: 10, speed: 5, ds: 1 }));
    expect(Math.max(...straight.map((p) => p.kappa))).toBeLessThan(1e-6);
    const weave = toReferenceTrajectory(slalom({ spacing: 12, amplitude: 2, cones: 4, speed: 5 }));
    expect(Math.max(...weave.map((p) => p.kappa))).toBeGreaterThan(0.05);
  });
});

describe('feasibility curvature-rate gate', () => {
  const base: DynamicLimits = { frictionLimit: 100, minTurnRadius: 0.1, maxAccel: 100, maxDecel: 100 };
  it('flags rapid curvature change when maxCurvatureRate is set', () => {
    // A slalom forces fast curvature reversals; a tiny rate limit must trip.
    const ref = toReferenceTrajectory(slalom({ spacing: 12, amplitude: 2, cones: 4, speed: 8 }));
    const report = checkFeasibility(ref, { ...base, maxCurvatureRate: 0.001 });
    expect(report.counts['curvature-rate']).toBeGreaterThan(0);
    expect(report.feasible).toBe(false);
  });
  it('omitting maxCurvatureRate skips the check', () => {
    const ref = toReferenceTrajectory(slalom({ spacing: 12, amplitude: 2, cones: 4, speed: 8 }));
    const report = checkFeasibility(ref, base);
    expect(report.counts['curvature-rate']).toBe(0);
  });
});

describe('edge cases', () => {
  it('projectOntoPath handles empty and single-point references', () => {
    expect(projectOntoPath([], 1, 1).crossTrack).toBe(0);
    const single = toReferenceTrajectory([{ x: 2, z: 3, heading: 0, speed: 1, t: 0 }]);
    const proj = projectOntoPath(single, 5, 3);
    expect(proj.crossTrack).toBeCloseTo(3, 5); // distance from (5,3) to (2,3)
  });

  it('rolloutTeleportFollow returns [] for an empty plan', () => {
    expect(rolloutTeleportFollow([], 0.05)).toEqual([]);
  });

  it('scorePlan without a goal reports null terminal', () => {
    const plan = arcPath({ radius: 20, sweep: Math.PI / 2, speed: 6, ds: 0.5 });
    const report = scorePlan(plan, { frictionLimit: 4, minTurnRadius: 4, maxAccel: 6.5, maxDecel: 8 });
    expect(report.terminal).toBeNull();
    expect(report.pathLength).toBeGreaterThan(0);
  });
});
