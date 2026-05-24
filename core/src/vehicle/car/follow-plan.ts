// Plan-follower for the car domain. Wraps `trimPlan` + the pure-pursuit
// tracker into a single call so demos stop hand-rolling the boilerplate.
//
// Note: the curvature -> Ackermann steer-angle conversion + Rapier sign-flip
// already lives in `adapters/rapier`'s `planToAckermannControls`. This helper
// returns the curvature-form command; demos that target Rapier raycast
// vehicles should pipe the result through `planToAckermannControls` (or use
// the `PlanFollowerCarDriver` in `./drivers`, which does this for you).

import { purePursuit } from '../../execute/pure-pursuit';
import type { PurePursuitConfig, TrackingCommand } from '../../execute/types';
import { trimPlan } from './plan-utils';
import type { CarKinematicState } from './types';

export interface FollowPlanOpts {
  /** Pure-pursuit configuration. */
  config: PurePursuitConfig;
  /** Elapsed seconds since the plan was committed. Used to trim the
   *  already-consumed lead samples. */
  elapsed: number;
}

/** Compute the curvature-form tracking command for `state` following `plan`. */
export function followPlan(
  state: CarKinematicState,
  plan: ReadonlyArray<CarKinematicState>,
  opts: FollowPlanOpts,
): TrackingCommand {
  const trimmed = trimPlan(plan, opts.elapsed);
  if (trimmed.length === 0) {
    return {
      steering: 0,
      throttle: 0,
      brake: 1,
      targetSpeed: 0,
      lookahead: { x: state.x, z: state.z },
      atGoal: true,
    };
  }
  return purePursuit(state, trimmed, opts.config);
}
