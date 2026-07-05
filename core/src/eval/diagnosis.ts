// The diagnosis 2×2 (evaluation guide §6): combine plan quality (§4.2) and
// controller tracking (§4.1) to point at the responsible component. The
// ordering matters — plan FEASIBILITY is checked first: a controller tuned
// against an infeasible reference is tuned against the wrong target.
//
//  | Plan quality      | Controller tracking | Conclusion          |
//  | feasible+fills    | poor                | controller          |
//  | feasible+timid    | good                | planner (timid)     |
//  | infeasible        | (any)               | planner (infeasible)|
//  | both bad          | both bad            | both — fix planner  |

import type { PlanQualityReport } from './plan-quality';
import type { TrackingReport } from './tracking-metrics';

export type Verdict =
  | 'ok'
  | 'controller'
  | 'planner-timid'
  | 'planner-infeasible'
  | 'both';

export interface DiagnosisThresholds {
  /** Cross-track RMSE (m) above which tracking is judged "poor". */
  crossTrackRmse: number;
  /** Mean g-g utilization below which the planner is judged "timid". */
  minUtil: number;
}

export const DEFAULT_DIAGNOSIS_THRESHOLDS: DiagnosisThresholds = {
  crossTrackRmse: 0.5,
  minUtil: 0.5,
};

export interface Diagnosis {
  verdict: Verdict;
  rationale: string;
}

/** Read off the 2×2 and name the component to fix. */
export function diagnose(
  plan: PlanQualityReport,
  track: TrackingReport,
  thresholds: DiagnosisThresholds = DEFAULT_DIAGNOSIS_THRESHOLDS,
): Diagnosis {
  // Feasibility first — downstream tracking failures are not the controller's
  // fault when the plan cannot be executed by anyone.
  if (!plan.feasibility.feasible) {
    const c = plan.feasibility.counts;
    return {
      verdict: 'planner-infeasible',
      rationale:
        `plan violates dynamic limits (worstRatio=${plan.feasibility.worstRatio.toFixed(2)}; ` +
        `latAccel=${c['lateral-accel']}, turnRadius=${c['turn-radius']}, ` +
        `longAccel=${c['longitudinal-accel']}, curvRate=${c['curvature-rate']}); ` +
        `fix the planner before tuning the controller`,
    };
  }

  const trackingPoor = track.crossTrack.rmse > thresholds.crossTrackRmse;
  const timid = plan.gg.meanUtil < thresholds.minUtil;

  if (trackingPoor && timid) {
    return {
      verdict: 'both',
      rationale:
        `tracking poor (xtrack RMSE ${track.crossTrack.rmse.toFixed(2)}m > ` +
        `${thresholds.crossTrackRmse}m) AND plan timid (util ` +
        `${(plan.gg.meanUtil * 100).toFixed(0)}% < ${(thresholds.minUtil * 100).toFixed(0)}%); ` +
        `fix the planner first`,
    };
  }
  if (trackingPoor) {
    return {
      verdict: 'controller',
      rationale:
        `plan is feasible and uses ${(plan.gg.meanUtil * 100).toFixed(0)}% of the envelope, ` +
        `but tracking is poor (xtrack RMSE ${track.crossTrack.rmse.toFixed(2)}m); ` +
        `tune lookahead/gains/speed-dependence`,
    };
  }
  if (timid) {
    return {
      verdict: 'planner-timid',
      rationale:
        `tracking is good (xtrack RMSE ${track.crossTrack.rmse.toFixed(2)}m) but the plan ` +
        `only uses ${(plan.gg.meanUtil * 100).toFixed(0)}% of the envelope; ` +
        `the planner leaves performance unused`,
    };
  }
  return {
    verdict: 'ok',
    rationale:
      `feasible, uses ${(plan.gg.meanUtil * 100).toFixed(0)}% of the envelope, ` +
      `tracks to ${track.crossTrack.rmse.toFixed(2)}m RMSE`,
  };
}
