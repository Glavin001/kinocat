import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import {
  AffordanceRegistry,
  createJumpAffordance,
} from '../../src/predict/affordance-registry';
import type { VehicleAgent, VehicleState } from '../../src/agent/types';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}
const agent: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  footprint: [
    [1.0, 0.5],
    [-1.0, 0.5],
    [-1.0, -0.5],
    [1.0, -0.5],
  ],
});
function buildLib(a: VehicleAgent) {
  const k = 1 / a.minTurnRadius;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(a),
    controlSets: [[0, 6], [k, 6], [-k, 6], [k / 2, 6], [-k / 2, 6]],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
}
const lib = buildLib(agent);

describe('lazy affordance edges', () => {
  // Two islands separated by a non-walkable gap; drive primitives can't cross.
  const world = new InMemoryNavWorld([
    rect(1, 0, -6, 14, 6),
    rect(2, 22, -6, 40, 6),
  ]);
  const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 34, z: 0, heading: 0, speed: 0, t: 0 };

  function makeEnv(reg?: AffordanceRegistry) {
    const base = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    return new TimeAwareEnvironment(base, {
      affordances: reg,
      affordanceRadius: 12,
    });
  }

  it('without an affordance the gap is uncrossable', () => {
    const r = plan(
      { start, goal, environment: makeEnv(), options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(r.found).toBe(false);
  });

  it('uses a registered jump affordance to cross the gap', () => {
    const reg = new AffordanceRegistry();
    reg.add(
      createJumpAffordance({
        id: 'gap-jump',
        launch: { x: 12, z: 0 },
        entryRadius: 3,
        land: { x: 25, z: 0, heading: 0, speed: 0, t: 0 },
        duration: 1,
        cost: 1.5,
      }),
    );
    const r = plan(
      {
        start,
        goal,
        environment: makeEnv(reg),
        options: { maxExpansions: 400000 },
      },
      Infinity,
    );
    expect(r.found).toBe(true);
    const used = r.nodes.filter((n) => n.edge?.kind === 'affordance');
    expect(used.length).toBe(1);
    // the affordance lands the agent on the far island
    const landIdx = r.nodes.findIndex((n) => n.edge?.kind === 'affordance');
    expect(r.path[landIdx]!.x).toBeCloseTo(25, 6);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(1.5);
  });
});
