// Affordances are no longer vehicle-typed: Affordance<S> / AffordanceRegistry<S>
// are generic over the agent state, and TimeAwareEnvironment generates
// affordance edges for whatever state its base env plans. A humanoid
// teleporter bridging two disconnected navmesh islands is the proof: without
// the affordance the goal is unreachable; with it the plan takes the edge.

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { HumanoidEnvironment } from '../../src/environment/humanoid-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { defaultHumanoidAgent } from '../../src/agent';
import {
  AffordanceRegistry,
  AffordanceType,
  type Affordance,
} from '../../src/predict/affordance-registry';
import type { HumanoidState } from '../../src/agent/types';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const human = defaultHumanoidAgent({ radius: 0.3, maxSpeed: 4 });

/** Two islands with a 20 m void between them — unwalkable. */
function islands() {
  return new InMemoryNavWorld([rect(1, 0, 0, 10, 10), rect(2, 30, 0, 40, 10)]);
}

function teleporter(): Affordance<HumanoidState> {
  return {
    id: 'tp-1',
    type: AffordanceType.Teleporter,
    validFrom: -Infinity,
    validTo: Infinity,
    spatialBound: { x: 8, z: 5, radius: 3 },
    predict: () => ({ position: { x: 8, y: 0, z: 5 } }),
    tryUse(agentState, useTime) {
      const dx = agentState.x - 8;
      const dz = agentState.z - 5;
      if (dx * dx + dz * dz > 9) return null;
      const land: HumanoidState = { x: 32, z: 5, heading: 0, t: useTime + 2 };
      return {
        resultState: land,
        duration: 2,
        cost: 2,
        trajectory: [
          { x: agentState.x, y: 0, z: agentState.z, t: useTime },
          { x: 32, y: 0, z: 5, t: useTime + 2 },
        ],
      };
    },
  };
}

const start: HumanoidState = { x: 2, z: 5, heading: 0, t: 0 };
const goal: HumanoidState = { x: 35, z: 5, heading: 0, t: 0 };

describe('generic (non-vehicle) affordances', () => {
  it('without the teleporter the far island is unreachable', () => {
    const env = new TimeAwareEnvironment(new HumanoidEnvironment(islands(), human));
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 60_000 } },
      Infinity,
    );
    expect(r.found).toBe(false);
  });

  it('a humanoid-typed teleporter affordance bridges the islands', () => {
    const reg = new AffordanceRegistry<HumanoidState>();
    reg.add(teleporter());
    const env = new TimeAwareEnvironment(new HumanoidEnvironment(islands(), human), {
      affordances: reg,
      affordanceRadius: 5,
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 120_000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // The path must actually take the teleport: one step crosses the void.
    const jump = r.path.some(
      (s, i) => i > 0 && s.x - r.path[i - 1]!.x > 15,
    );
    expect(jump).toBe(true);
    // Landing time reflects the affordance duration.
    const landIdx = r.path.findIndex((s) => s.x > 20);
    expect(landIdx).toBeGreaterThan(0);
    expect(r.path[landIdx]!.t).toBeGreaterThanOrEqual(2);
  });
});
