// Replanning is the universal correction mechanism (no FSM). Triggers:
// divergence from the planned pose, periodic refresh, or an event-driven
// dirty mark (tile rebuild, new affordance, prediction change).

import type { VehicleState } from '../agent/types';
import type { PlanPath, ReplanTrigger } from './types';
import { dist, lerp, lerpAngle } from '../internal/math';

/** Interpolate the planned pose at absolute time `t` (clamped to the ends). */
export function planPoseAt(path: PlanPath, t: number): VehicleState | null {
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
  private dirtyReason: string | null = null;
  lastReplanMs = -Infinity;

  constructor(private readonly trigger: ReplanTrigger) {}

  setPlan(path: PlanPath, nowMs: number): void {
    this.plan = path;
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
  divergence(current: VehicleState): number {
    if (!this.plan) return Infinity;
    const expected = planPoseAt(this.plan, current.t);
    if (!expected) return Infinity;
    return dist(current.x, current.z, expected.x, expected.z);
  }

  shouldReplan(current: VehicleState, nowMs: number): boolean {
    if (!this.plan) return true;
    if (this.dirtyReason !== null) return true;
    if (nowMs - this.lastReplanMs >= this.trigger.refreshIntervalMs) return true;
    return this.divergence(current) > this.trigger.divergenceThresholdMeters;
  }
}
