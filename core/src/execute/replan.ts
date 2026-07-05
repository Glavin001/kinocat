// Replanning is the universal correction mechanism (no FSM). Triggers:
// divergence from the planned pose, periodic refresh, or an event-driven
// dirty mark (tile rebuild, new affordance, prediction change).

import type { CarKinematicState } from '../agent/types';
import type { PlanPath, ReplanReason, ReplanTrigger } from './types';
import { dist, lerp, lerpAngle } from '../internal/math';

/** Interpolate the planned pose at absolute time `t` (clamped to the ends). */
export function planPoseAt(path: PlanPath, t: number): CarKinematicState | null {
  if (path.length === 0) return null;
  const first = path[0]!;
  const last = path[path.length - 1]!;
  if (t <= first.t) return first;
  if (t >= last.t) return last;
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const u = span > 1e-12 ? (t - a.t) / span : 0;
      return {
        x: lerp(a.x, b.x, u),
        z: lerp(a.z, b.z, u),
        heading: lerpAngle(a.heading, b.heading, u),
        speed: lerp(a.speed, b.speed, u),
        t,
      };
    }
  }
  return last;
}

export class ReplanState {
  private plan: PlanPath | null = null;
  private committedCost = Infinity;
  private dirtyReason: string | null = null;
  private lastReason: ReplanReason = 'none';
  lastReplanMs = -Infinity;

  constructor(private readonly trigger: ReplanTrigger) {}

  /** Commit a plan unconditionally (initial plan / forced). */
  setPlan(path: PlanPath, nowMs: number, cost = 0): void {
    this.plan = path;
    this.committedCost = cost;
    this.lastReplanMs = nowMs;
    this.dirtyReason = null;
  }

  get currentPlan(): PlanPath | null {
    return this.plan;
  }

  /** Force a replan on the next check (tile rebuild, new affordance, …). */
  markDirty(reason: string): void {
    this.dirtyReason = reason;
  }

  /** Divergence between actual state and the plan at the actual state's time. */
  divergence(current: CarKinematicState): number {
    if (!this.plan) return Infinity;
    const expected = planPoseAt(this.plan, current.t);
    if (!expected) return Infinity;
    return dist(current.x, current.z, expected.x, expected.z);
  }

  shouldReplan(current: CarKinematicState, nowMs: number): boolean {
    if (!this.plan) {
      this.lastReason = 'no-plan';
      return true;
    }
    if (this.dirtyReason !== null) {
      this.lastReason = 'dirty';
      return true;
    }
    if (this.divergence(current) > this.trigger.divergenceThresholdMeters) {
      this.lastReason = 'divergence';
      return true;
    }
    if (nowMs - this.lastReplanMs >= this.trigger.refreshIntervalMs) {
      this.lastReason = 'periodic';
      return true;
    }
    this.lastReason = 'none';
    return false;
  }

  /**
   * Decide whether to adopt a freshly-planned route. Always adopts when the
   * current plan is invalid (no-plan / dirty / divergence); on a routine
   * periodic replan it keeps the committed plan unless the candidate is
   * meaningfully cheaper — this is the plan-switch hysteresis that stops the
   * agent oscillating between two equal-cost routes. Returns true if adopted.
   */
  consider(path: PlanPath, cost: number, nowMs: number): boolean {
    const frac = this.trigger.switchCostImprovement ?? 0.15;
    const margin = this.trigger.switchCostMargin ?? 0.5;
    const forced =
      this.plan === null ||
      this.lastReason === 'no-plan' ||
      this.lastReason === 'dirty' ||
      this.lastReason === 'divergence';
    if (forced || cost < this.committedCost * (1 - frac) - margin) {
      this.setPlan(path, nowMs, cost);
      return true;
    }
    // keep the committed plan, but don't re-trigger a periodic replan at once
    this.lastReplanMs = nowMs;
    return false;
  }
}
