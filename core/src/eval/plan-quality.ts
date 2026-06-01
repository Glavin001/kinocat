// Planner isolation (evaluation guide §4.2). Replace the controller with a
// near-perfect "teleport-follow" tracker so execution error is effectively
// zero; any badness that remains is a property of the PLAN itself. The plan is
// then scored as a static artifact: feasibility (4.2a), capability utilization
// (4.2b), and terminal accuracy including heading and velocity (4.2d).

import type { CarKinematicState } from '../agent/types';
import { angleDiff } from '../internal/math';
import {
  toReferenceTrajectory,
  referencePoseAt,
  referenceLength,
  type ReferenceTrajectory,
} from './reference-trajectory';
import { checkFeasibility, type DynamicLimits, type FeasibilityReport } from './feasibility';
import { ggUtilization, type GgReport } from './gg-utilization';

/** A near-perfect tracker: advance the car directly along the planned states,
 *  resampled at a fixed `dt` by the plan's own time stamps. The resulting
 *  executed trajectory equals the plan (up to interpolation), which is exactly
 *  the point — it removes the controller from the equation. */
export function rolloutTeleportFollow(
  plan: ReadonlyArray<CarKinematicState>,
  dt: number,
): CarKinematicState[] {
  const ref = toReferenceTrajectory(plan);
  const n = ref.length;
  if (n === 0) return [];
  const out: CarKinematicState[] = [];
  // Resample by arc-length so the rollout is independent of the plan's timing,
  // stepping a fixed distance per tick at the local planned speed.
  let s = 0;
  const total = referenceLength(ref);
  let t = plan[0]!.t ?? 0;
  let guard = 0;
  const maxSteps = Math.ceil(total / Math.max(dt * 0.1, 1e-3)) + n + 100;
  while (s <= total && guard++ < maxSteps) {
    const p = referencePoseAt(ref, s)!;
    out.push({ x: p.x, z: p.z, heading: p.psi, speed: p.v, t });
    const ds = Math.max(Math.abs(p.v) * dt, 1e-3);
    s += ds;
    t += dt;
  }
  // Always include the exact terminal state.
  const last = plan[plan.length - 1]!;
  out.push({ ...last, t });
  return out;
}

export interface TerminalAccuracy {
  posError: number;
  headingError: number;
  speed: number;
}

export interface PlanQualityReport {
  feasibility: FeasibilityReport;
  gg: GgReport;
  /** Terminal pose error vs a target pose (null when no goal supplied). */
  terminal: TerminalAccuracy | null;
  /** Planned path length (m). */
  pathLength: number;
  /** Planned time-to-goal (s) — last minus first time stamp. */
  timeToGoal: number;
}

export interface ScorePlanOptions {
  /** Target pose for terminal accuracy (position + heading), and optional
   *  target speed (defaults to 0 — i.e. "come to a stop"). */
  goal?: { x: number; z: number; heading: number; speed?: number };
}

/** Score a plan as a static artifact against the car's dynamic limits. */
export function scorePlan(
  plan: ReadonlyArray<CarKinematicState>,
  limits: DynamicLimits,
  opts?: ScorePlanOptions,
): PlanQualityReport {
  const ref = toReferenceTrajectory(plan);
  const feasibility = checkFeasibility(ref, limits);
  const gg = ggUtilization(ref, limits.frictionLimit);

  let terminal: TerminalAccuracy | null = null;
  if (opts?.goal && plan.length > 0) {
    const last = plan[plan.length - 1]!;
    terminal = {
      posError: Math.hypot(last.x - opts.goal.x, last.z - opts.goal.z),
      headingError: Math.abs(angleDiff(last.heading, opts.goal.heading)),
      speed: Math.abs(last.speed - (opts.goal.speed ?? 0)),
    };
  }

  const pathLength = referenceLength(ref);
  const timeToGoal =
    plan.length > 1 ? (plan[plan.length - 1]!.t ?? 0) - (plan[0]!.t ?? 0) : 0;

  return { feasibility, gg, terminal, pathLength, timeToGoal };
}

export type { ReferenceTrajectory };
