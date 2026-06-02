import { describe, it, expect } from 'vitest';
import { createEvalProbe, type EvalProbeSample } from '../app/lib/eval-probe';
import type { CarKinematicState } from 'kinocat/agent';
import type { DynamicLimits } from 'kinocat/eval';

const limits: DynamicLimits = { frictionLimit: 12, minTurnRadius: 4.5, maxAccel: 6, maxDecel: 8 };

// A gentle straight plan along +x at constant speed.
const plan: CarKinematicState[] = Array.from({ length: 40 }, (_, i) => ({
  x: i * 0.5,
  z: 0,
  heading: 0,
  speed: 6,
  t: (i * 0.5) / 6,
}));

function sampleAt(state: CarKinematicState): EvalProbeSample {
  return {
    state,
    metrics: { liveControls: { steer: 0, throttle: 0.5, brake: 0, targetSpeed: 6 } },
    diagnostics: { totalReplans: 1, successfulReplans: 1, consecutiveFailedReplans: 0 },
    plan,
  };
}

describe('createEvalProbe', () => {
  it('reports ~0 cross-track when the car rides the plan exactly', () => {
    const probe = createEvalProbe({ footprint: [[2, 1], [-2, 1], [-2, -1], [2, -1]], dt: 1 / 60, limits });
    for (const p of plan) probe.sample(sampleAt(p));
    const snap = probe.snapshot();
    expect(snap.crossTrackRmse).toBeLessThan(0.05);
    expect(snap.planFeasible).toBe(true);
  });

  it('reports a constant lateral offset as cross-track error', () => {
    const probe = createEvalProbe({ footprint: [[2, 1], [-2, 1], [-2, -1], [2, -1]], dt: 1 / 60, limits });
    for (const p of plan) probe.sample(sampleAt({ ...p, z: p.z + 0.6 }));
    const snap = probe.snapshot();
    expect(snap.crossTrackNow).toBeCloseTo(0.6, 1);
    expect(snap.crossTrackRmse).toBeGreaterThan(0.4);
  });

  it('is read-only: the SimMonitor it wraps records the stream', () => {
    const probe = createEvalProbe({ footprint: [[2, 1], [-2, 1], [-2, -1], [2, -1]], dt: 1 / 60, limits });
    for (const p of plan) probe.sample(sampleAt(p));
    expect(probe.monitor.trajectory().length).toBe(plan.length);
    expect(probe.snapshot().report.peakSpeed).toBeCloseTo(6, 1);
  });
});
