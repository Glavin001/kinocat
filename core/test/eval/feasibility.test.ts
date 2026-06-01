import { describe, it, expect } from 'vitest';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { checkFeasibility, limitsFromAgent, type DynamicLimits } from '../../src/eval/feasibility';
import { arcPath } from '../../src/eval/reference-shapes';
import { defaultVehicleAgent } from '../../src/agent/vehicle';

const limits: DynamicLimits = {
  frictionLimit: 4.0, // μ·g ≈ 4 m/s²
  minTurnRadius: 4,
  maxAccel: 6.5,
  maxDecel: 8,
};

describe('checkFeasibility', () => {
  it('passes a gentle, within-limits arc', () => {
    // radius 20 m at 6 m/s ⇒ a_lat = 36/20 = 1.8 m/s² < 4.
    const ref = toReferenceTrajectory(arcPath({ radius: 20, sweep: Math.PI / 2, speed: 6, ds: 0.5 }));
    const report = checkFeasibility(ref, limits);
    expect(report.feasible).toBe(true);
    expect(report.counts['lateral-accel']).toBe(0);
  });

  it('flags an over-speed corner as infeasible (a_lat exceeds friction)', () => {
    // radius 6 m at 10 m/s ⇒ a_lat = 100/6 ≈ 16.7 m/s² ≫ 4.
    const ref = toReferenceTrajectory(arcPath({ radius: 6, sweep: Math.PI / 2, speed: 10, ds: 0.3 }));
    const report = checkFeasibility(ref, limits);
    expect(report.feasible).toBe(false);
    expect(report.counts['lateral-accel']).toBeGreaterThan(0);
    expect(report.worstRatio).toBeGreaterThan(1);
  });

  it('flags a sub-minimum turning radius', () => {
    // radius 2 m < minTurnRadius 4 m, slow enough that lateral accel is OK.
    const ref = toReferenceTrajectory(arcPath({ radius: 2, sweep: Math.PI / 2, speed: 1, ds: 0.1 }));
    const report = checkFeasibility(ref, limits);
    expect(report.counts['turn-radius']).toBeGreaterThan(0);
  });

  it('limitsFromAgent wires the agent min turn radius', () => {
    const agent = defaultVehicleAgent();
    const l = limitsFromAgent(agent, { frictionLimit: 4, maxAccel: 6.5, maxDecel: 8 });
    expect(l.minTurnRadius).toBe(agent.minTurnRadius);
    expect(l.frictionLimit).toBe(4);
  });
});
