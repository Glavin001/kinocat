// Dynamic, agent-relative region constructors. Each carries a `RegionAgent`
// predictor and sets `dynamic = true`, which (per the spec's section 8) pulls a
// clock dimension into the search: `contains(s, t)` is evaluated against
// `agent.predict(t)`, so the planner reaches toward where the target WILL be
// (interception), not where it is now (tail-chasing).
//
// `costToGo(s)` with `t` omitted returns a time-AGNOSTIC infimum (the bound at
// the predictor's reference pose), which keeps the precomputed heuristic chain
// admissible even though the true target time is unknown at precompute time.

import type { Region, ScenarioState, RegionAgent } from './types';
import { angleDiff } from '../internal/math';

function ref(agent: RegionAgent): ScenarioState {
  return agent.predict(0) ?? { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
}

/** A ball of radius `r` around an agent's predicted pose (intercept / contact). */
export function within(agent: RegionAgent, r: number): Region {
  return {
    kind: 'within',
    key: `within:${agent.id},${r}`,
    dynamic: true,
    contains(s, t) {
      const a = agent.predict(t ?? s.t);
      if (!a) return false;
      return Math.hypot(s.x - a.x, s.z - a.z) <= r;
    },
    costToGo(s, t) {
      const a = agent.predict(t ?? 0) ?? ref(agent);
      return Math.max(0, Math.hypot(s.x - a.x, s.z - a.z) - r);
    },
    representative() {
      return ref(agent);
    },
  };
}

/** A ball of radius `tol` at distance `d` AHEAD of the agent (overtake clear). */
export function ahead(agent: RegionAgent, d: number, tol = 1.5): Region {
  const target = (a: ScenarioState) => ({
    x: a.x + Math.cos(a.heading) * d,
    z: a.z + Math.sin(a.heading) * d,
  });
  return {
    kind: 'ahead',
    key: `ahead:${agent.id},${d},${tol}`,
    dynamic: true,
    contains(s, t) {
      const a = agent.predict(t ?? s.t);
      if (!a) return false;
      const p = target(a);
      return Math.hypot(s.x - p.x, s.z - p.z) <= tol;
    },
    costToGo(s, t) {
      const a = agent.predict(t ?? 0) ?? ref(agent);
      const p = target(a);
      return Math.max(0, Math.hypot(s.x - p.x, s.z - p.z) - tol);
    },
    representative() {
      const a = ref(agent);
      const p = target(a);
      return { x: p.x, z: p.z, heading: a.heading, speed: 0, t: 0 };
    },
  };
}

/** A ball of radius `tol` at distance `d` BEHIND the agent (convoy / tail). */
export function behind(agent: RegionAgent, d: number, tol = 1.5): Region {
  const flipped = ahead(
    {
      id: `${agent.id}#behind`,
      predict: (t) => {
        const a = agent.predict(t);
        return a ? { ...a, heading: a.heading + Math.PI } : null;
      },
    },
    d,
    tol,
  );
  return { ...flipped, kind: 'behind', key: `behind:${agent.id},${d},${tol}` };
}

export type Side = 'left' | 'right';
export const LEFT: Side = 'left';
export const RIGHT: Side = 'right';

/** A ball of radius `tol` BESIDE the agent on `side`, lateral gap `gap`. */
export function beside(
  agent: RegionAgent,
  side: Side,
  gap: number,
  tol = 1.5,
): Region {
  // +90deg of heading is the agent's left in a right-handed XZ frame.
  const sign = side === 'left' ? 1 : -1;
  const target = (a: ScenarioState) => ({
    x: a.x + Math.cos(a.heading + (Math.PI / 2) * sign) * gap,
    z: a.z + Math.sin(a.heading + (Math.PI / 2) * sign) * gap,
  });
  return {
    kind: 'beside',
    key: `beside:${agent.id},${side},${gap},${tol}`,
    dynamic: true,
    contains(s, t) {
      const a = agent.predict(t ?? s.t);
      if (!a) return false;
      const p = target(a);
      return Math.hypot(s.x - p.x, s.z - p.z) <= tol;
    },
    costToGo(s, t) {
      const a = agent.predict(t ?? 0) ?? ref(agent);
      const p = target(a);
      return Math.max(0, Math.hypot(s.x - p.x, s.z - p.z) - tol);
    },
    representative() {
      const a = ref(agent);
      const p = target(a);
      return { x: p.x, z: p.z, heading: a.heading, speed: 0, t: 0 };
    },
  };
}

/** A vision / keep-out CONE: the wedge of half-angle `fov` and `range` in front
 *  of the agent's heading. Used as `avoid(cone(guard))` for stealth / evasion. */
export function cone(agent: RegionAgent, fov: number, range: number): Region {
  return {
    kind: 'cone',
    key: `cone:${agent.id},${fov},${range}`,
    dynamic: true,
    contains(s, t) {
      const a = agent.predict(t ?? s.t);
      if (!a) return false;
      const dx = s.x - a.x;
      const dz = s.z - a.z;
      const dist = Math.hypot(dx, dz);
      if (dist > range || dist < 1e-9) return dist < 1e-9; // at the apex = seen
      const bearing = Math.atan2(dz, dx);
      return Math.abs(angleDiff(a.heading, bearing)) <= fov;
    },
    costToGo() {
      // A keep-out cone is an invariant target, not a goal; 0 is a safe LB.
      return 0;
    },
    representative() {
      const a = ref(agent);
      return {
        x: a.x + Math.cos(a.heading) * range * 0.5,
        z: a.z + Math.sin(a.heading) * range * 0.5,
        heading: a.heading,
        speed: 0,
        t: 0,
      };
    },
  };
}
