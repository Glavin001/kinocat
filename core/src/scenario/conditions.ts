// Condition-regions. The spec writes scoped invariants like
// `maintain(speed(lte(5)))` and `maintain(distanceFrom(lead, gte(1)))`. Rather
// than introduce a second "Condition" type, we model conditions AS regions
// whose `contains(s)` tests a predicate on the state (or on an agent-relative
// distance). This unifies the constraint plane: every `maintain(R)` means
// "must always be inside R", and the planner bridge needs only one rule
// (`violated when !R.contains`). These regions are non-spatial, so `costToGo`
// is 0 (a safe lower bound) and they are never used as `reach` targets.

import type { Region, ScenarioState, RegionAgent, Bound } from './types';

/** `value <= x`. */
export function lte(x: number): Bound {
  return { max: x };
}
/** `value >= x`. */
export function gte(x: number): Bound {
  return { min: x };
}
/** `lo <= value <= hi`. */
export function inRange(lo: number, hi: number): Bound {
  return { min: lo, max: hi };
}

function satisfies(value: number, b: Bound): boolean {
  if (b.min !== undefined && value < b.min) return false;
  if (b.max !== undefined && value > b.max) return false;
  return true;
}

const ORIGIN: ScenarioState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };

/** A condition-region on signed speed (m/s). */
export function speed(b: Bound): Region {
  return {
    kind: 'cond-speed',
    key: `cond-speed:${b.min ?? ''},${b.max ?? ''}`,
    dynamic: false,
    contains(s) {
      return satisfies(s.speed, b);
    },
    costToGo() {
      return 0;
    },
    representative() {
      return ORIGIN;
    },
  };
}

/** A condition-region: distance from `agent`'s predicted pose is within `b`
 *  (e.g. `gte(1)` = "stay at least 1 m from the lead"). Dynamic. */
export function distanceFrom(agent: RegionAgent, b: Bound): Region {
  return {
    kind: 'cond-distance',
    key: `cond-distance:${agent.id},${b.min ?? ''},${b.max ?? ''}`,
    dynamic: true,
    contains(s, t) {
      const a = agent.predict(t ?? s.t);
      if (!a) return true; // unknown predictor horizon -> not violated
      return satisfies(Math.hypot(s.x - a.x, s.z - a.z), b);
    },
    costToGo() {
      return 0;
    },
    representative() {
      return agent.predict(0) ?? ORIGIN;
    },
  };
}
