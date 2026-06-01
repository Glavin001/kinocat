import { describe, it, expect } from 'vitest';
import { diagnose } from '../../src/eval/diagnosis';
import type { PlanQualityReport } from '../../src/eval/plan-quality';
import type { TrackingReport } from '../../src/eval/tracking-metrics';
import type { FeasibilityReport } from '../../src/eval/feasibility';

function plan(opts: { feasible: boolean; meanUtil: number }): PlanQualityReport {
  const feasibility: FeasibilityReport = {
    feasible: opts.feasible,
    violations: opts.feasible ? [] : [{ index: 0, s: 0, kind: 'lateral-accel', value: 10, limit: 4 }],
    counts: {
      'lateral-accel': opts.feasible ? 0 : 1,
      'turn-radius': 0,
      'longitudinal-accel': 0,
      'curvature-rate': 0,
    },
    worstRatio: opts.feasible ? 0.5 : 2.5,
  };
  return {
    feasibility,
    gg: { meanUtil: opts.meanUtil, peakUtil: opts.meanUtil, cloud: [] },
    terminal: null,
    pathLength: 100,
    timeToGoal: 20,
  };
}

function track(crossTrackRmse: number): TrackingReport {
  return {
    crossTrack: { rmse: crossTrackRmse, max: crossTrackRmse, p95: crossTrackRmse },
    heading: { rmse: 0.05, max: 0.1 },
    velocity: { rmse: 0.2, max: 0.5 },
    steerRateRms: 0.1,
    steerReversals: 2,
    samples: 100,
  };
}

describe('diagnose (the 2×2)', () => {
  it('infeasible plan ⇒ planner-infeasible regardless of tracking', () => {
    expect(diagnose(plan({ feasible: false, meanUtil: 0.8 }), track(0.05)).verdict).toBe(
      'planner-infeasible',
    );
    expect(diagnose(plan({ feasible: false, meanUtil: 0.2 }), track(2.0)).verdict).toBe(
      'planner-infeasible',
    );
  });

  it('feasible + fills envelope + poor tracking ⇒ controller', () => {
    expect(diagnose(plan({ feasible: true, meanUtil: 0.8 }), track(1.2)).verdict).toBe('controller');
  });

  it('feasible + timid + good tracking ⇒ planner-timid', () => {
    expect(diagnose(plan({ feasible: true, meanUtil: 0.3 }), track(0.1)).verdict).toBe(
      'planner-timid',
    );
  });

  it('both bad ⇒ both', () => {
    expect(diagnose(plan({ feasible: true, meanUtil: 0.3 }), track(1.2)).verdict).toBe('both');
  });

  it('feasible + fills + good tracking ⇒ ok', () => {
    expect(diagnose(plan({ feasible: true, meanUtil: 0.8 }), track(0.1)).verdict).toBe('ok');
  });
});
