import { describe, it, expect } from 'vitest';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import {
  trackingMetrics,
  runControllerIsolation,
  countSteerReversals,
  type RefController,
} from '../../src/eval/tracking-metrics';
import { straightLine, arcPath } from '../../src/eval/reference-shapes';
import { purePursuit } from '../../src/execute/pure-pursuit';
import { kinematicForwardSim, defaultVehicleAgent } from '../../src/agent/vehicle';
import type { PurePursuitConfig } from '../../src/execute/types';

describe('trackingMetrics', () => {
  it('reports ~0 error for a perfect follow', () => {
    const path = straightLine({ length: 10, speed: 5, ds: 0.5 });
    const ref = toReferenceTrajectory(path);
    const report = trackingMetrics(path, ref, { dt: 0.05 });
    expect(report.crossTrack.rmse).toBeLessThan(1e-6);
    expect(report.heading.rmse).toBeLessThan(1e-6);
    expect(report.velocity.rmse).toBeLessThan(1e-6);
  });

  it('reports a constant lateral offset as the cross-track RMSE', () => {
    const path = straightLine({ length: 10, speed: 5, ds: 0.5 });
    const ref = toReferenceTrajectory(path);
    const offset = path.map((p) => ({ ...p, z: p.z + 0.4 }));
    const report = trackingMetrics(offset, ref, { dt: 0.05 });
    expect(report.crossTrack.rmse).toBeCloseTo(0.4, 2);
    expect(report.crossTrack.max).toBeCloseTo(0.4, 2);
    expect(report.crossTrack.p95).toBeCloseTo(0.4, 2);
  });
});

describe('runControllerIsolation', () => {
  const cfg: PurePursuitConfig = {
    lookaheadMin: 2,
    lookaheadGain: 0.3,
    lookaheadMax: 6,
    maxLateralAccel: 4,
    maxAccel: 8,
    maxDecel: 8,
    cruiseSpeed: 6,
    goalTolerance: 0.5,
    minTurnRadius: 2,
  };
  const agent = defaultVehicleAgent();
  const sim = kinematicForwardSim(agent);
  const controller: RefController = (state, path) => {
    const cmd = purePursuit(state, path as never, cfg);
    return { controls: [cmd.steering, cmd.targetSpeed], steer: cmd.steering, atGoal: cmd.atGoal };
  };

  it('tracks a gentle arc with small cross-track error', () => {
    const ref = arcPath({ radius: 12, sweep: Math.PI / 2, speed: 5, ds: 0.5 });
    const res = runControllerIsolation(ref, controller, sim, 0.05, { maxSteps: 2000 });
    expect(res.executed.length).toBeGreaterThan(10);
    // Pure-pursuit on a feasible arc should hold a tight line.
    expect(res.report.crossTrack.rmse).toBeLessThan(1.0);
  });

  it('is deterministic — identical reruns', () => {
    const ref = arcPath({ radius: 12, sweep: Math.PI / 2, speed: 5, ds: 0.5 });
    const a = runControllerIsolation(ref, controller, sim, 0.05, { maxSteps: 2000 });
    const b = runControllerIsolation(ref, controller, sim, 0.05, { maxSteps: 2000 });
    expect(a.report.crossTrack.rmse).toBe(b.report.crossTrack.rmse);
    expect(a.report.heading.rmse).toBe(b.report.heading.rmse);
    expect(a.executed.length).toBe(b.executed.length);
  });

  it('reports terminal accuracy that catches a stop-short cross-track misses', () => {
    const ref = arcPath({ radius: 12, sweep: Math.PI / 2, speed: 5, ds: 0.5 });
    // Cut the run off long before the goal: the car has been tracking the line
    // well (small cross-track) but is nowhere near the END pose.
    const short = runControllerIsolation(ref, controller, sim, 0.05, { maxSteps: 15 });
    expect(short.report.crossTrack.rmse).toBeLessThan(1.0); // corridor tracked fine
    expect(short.terminal.posError).toBeGreaterThan(3); // …yet stopped far short
    // A full run parks essentially on the goal — terminal accuracy confirms it.
    const full = runControllerIsolation(ref, controller, sim, 0.05, { maxSteps: 2000 });
    expect(full.terminal.posError).toBeLessThan(1.0);
    expect(full.terminal.headingError).toBeLessThan(0.3);
  });
});

describe('countSteerReversals', () => {
  it('returns 0 for a monotone staircase (stepwise command, no real reversal)', () => {
    // Piecewise-constant steer ramping one way in held steps — the rate-sign
    // counter miscounted this as a reversal on every step.
    const staircase = [0, 0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3];
    expect(countSteerReversals(staircase, 0.03)).toBe(0);
  });

  it('counts genuine left→right→left oscillations above the deadband', () => {
    // +0.2, back to −0.2, back to +0.2 ⇒ two turning points.
    const osc = [0, 0.2, 0.2, 0.0, -0.2, -0.2, 0.0, 0.2, 0.2];
    expect(countSteerReversals(osc, 0.03)).toBe(2);
  });

  it('ignores sub-deadband chatter', () => {
    const jitter = [0, 0.01, -0.01, 0.01, -0.01, 0.01]; // ±0.01 < 0.03 deadband
    expect(countSteerReversals(jitter, 0.03)).toBe(0);
  });
});
