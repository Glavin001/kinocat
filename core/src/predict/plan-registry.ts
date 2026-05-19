// Multi-agent coordination by plan sharing. Each NPC publishes its current
// planned path (timestamped states); other NPCs read it back as a Predict via
// `predictNPC`. Cooperation emerges from cost alignment + frequent replanning
// — no negotiation protocol (a simplified MAPF approach).

import type { Predict } from './types';
import type { AgentState } from '../agent/types';
import { lerp, lerpAngle } from '../internal/math';

export interface PublishedPlan {
  npcId: string;
  /** Path states, each carrying an absolute time `t` (ascending). */
  states: AgentState[];
  publishedAt: number;
}

function interp(a: AgentState, b: AgentState, u: number): AgentState {
  const base = {
    x: lerp(a.x, b.x, u),
    z: lerp(a.z, b.z, u),
    heading: lerpAngle(a.heading, b.heading, u),
    t: lerp(a.t, b.t, u),
  };
  if ('speed' in a && 'speed' in b) {
    return { ...base, speed: lerp(a.speed, b.speed, u) };
  }
  return base;
}

export class PlanRegistry {
  private readonly plans = new Map<string, PublishedPlan>();

  publish(npcId: string, states: AgentState[], publishedAt = 0): void {
    if (states.length === 0) {
      this.plans.delete(npcId);
      return;
    }
    this.plans.set(npcId, { npcId, states, publishedAt });
  }

  get(npcId: string): PublishedPlan | null {
    return this.plans.get(npcId) ?? null;
  }

  remove(npcId: string): void {
    this.plans.delete(npcId);
  }

  all(): PublishedPlan[] {
    return [...this.plans.values()];
  }

  /** Predict NPC `npcId` by interpolating its published plan at time `t`.
   *  Before the plan starts ⇒ null; after it ends ⇒ the final pose (the NPC
   *  is assumed to hold position). */
  predictNPC(npcId: string): Predict<AgentState> {
    return (t: number) => {
      const pub = this.plans.get(npcId);
      if (!pub) return null;
      const s = pub.states;
      const first = s[0]!;
      const last = s[s.length - 1]!;
      if (t < first.t) return null;
      if (t >= last.t) return { ...last, t };
      for (let i = 0; i < s.length - 1; i++) {
        const a = s[i]!;
        const b = s[i + 1]!;
        if (t >= a.t && t <= b.t) {
          const span = b.t - a.t;
          return interp(a, b, span > 1e-12 ? (t - a.t) / span : 0);
        }
      }
      return { ...last, t };
    };
  }
}

/** Predictor factory: NPC `npcId` is predicted to follow its published plan. */
export function fromPublishedPlan(
  npcId: string,
  registry: PlanRegistry,
): Predict<AgentState> {
  return registry.predictNPC(npcId);
}
