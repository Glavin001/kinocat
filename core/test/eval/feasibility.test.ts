import { describe, it, expect } from 'vitest';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { checkFeasibility, limitsFromAgent, type DynamicLimits } from '../../src/eval/feasibility';
import { arcPath } from '../../src/eval/reference-shapes';
import { defaultVehicleAgent } from '../../src/agent/vehicle';
import type { CarKinematicState } from '../../src/agent/types';

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

  // Reverse-gear accel/decel classification. In reverse the car SPEEDS UP as
  // speed becomes more negative (dv/dt < 0) and BRAKES as speed rises toward 0
  // (dv/dt > 0) — the opposite of forward. Classifying by sign(a) alone silently
  // swaps the maxAccel/maxDecel budgets for every reverse sample. Here maxAccel
  // (6.5) < |a| (7) ≤ maxDecel (8), so the two cases land on opposite verdicts
  // and pin the fix. A straight (κ=0) reverse plan isolates the longitudinal
  // check from lateral/turn-radius.
  const reverseLimits: DynamicLimits = { frictionLimit: 4, minTurnRadius: 4, maxAccel: 6.5, maxDecel: 8 };
  const reverseStraight = (speeds: number[], dt: number): CarKinematicState[] => {
    let x = 0;
    return speeds.map((v, i) => {
      if (i > 0) x -= Math.abs(0.5 * (v + speeds[i - 1]!)) * dt; // back up in −x
      return { x, z: 0, heading: 0, speed: v, t: i * dt };
    });
  };

  it('treats braking in reverse as DECELeration (feasible within maxDecel)', () => {
    // |speed| shrinking 7.2 → 0.2 ⇒ a ≈ +7. That is braking (decel), so |a|=7 ≤
    // maxDecel 8 ⇒ feasible. The buggy sign(a)-only check charged it to maxAccel
    // (6.5) and wrongly flagged it.
    const ref = toReferenceTrajectory(
      reverseStraight([-7.2, -5.8, -4.4, -3.0, -1.6, -0.2], 0.2),
    );
    const report = checkFeasibility(ref, reverseLimits);
    expect(report.counts['longitudinal-accel']).toBe(0);
    expect(report.feasible).toBe(true);
  });

  it('treats speeding up in reverse as ACCELeration (infeasible past maxAccel)', () => {
    // |speed| growing 0.2 → 7.2 ⇒ a ≈ −7. That is accelerating, so |a|=7 >
    // maxAccel 6.5 ⇒ infeasible. The buggy check charged it to maxDecel (8) and
    // wrongly passed it.
    const ref = toReferenceTrajectory(
      reverseStraight([-0.2, -1.6, -3.0, -4.4, -5.8, -7.2], 0.2),
    );
    const report = checkFeasibility(ref, reverseLimits);
    expect(report.counts['longitudinal-accel']).toBeGreaterThan(0);
    expect(report.feasible).toBe(false);
  });

  it('limitsFromAgent wires the agent min turn radius', () => {
    const agent = defaultVehicleAgent();
    const l = limitsFromAgent(agent, { frictionLimit: 4, maxAccel: 6.5, maxDecel: 8 });
    expect(l.minTurnRadius).toBe(agent.minTurnRadius);
    expect(l.frictionLimit).toBe(4);
  });
});
