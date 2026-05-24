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
import type { VehicleAgent, CarKinematicState } from '../../src/agent/types';

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
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 34, z: 0, heading: 0, speed: 0, t: 0 };

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

import {
  AffordanceType,
  createBoostAffordance,
  createMisdirectAffordance,
} from '../../src/predict/affordance-registry';

describe('shortcut vs misdirect affordances (emergent rejection)', () => {
  // Open corridor: a normal drive plan exists, so the planner can choose to
  // ignore an affordance entirely based on cost.
  const world = new InMemoryNavWorld([rect(1, 0, -10, 60, 10)]);
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 56, z: 0, heading: 0, speed: 0, t: 0 };
  const makeEnv = (reg?: AffordanceRegistry) =>
    new TimeAwareEnvironment(
      new VehicleEnvironment(world, agent, lib, {
        goalRadius: 1.5,
        goalHeadingTol: Infinity,
      }),
      { affordances: reg, affordanceRadius: 12 },
    );
  const run = (reg?: AffordanceRegistry) =>
    plan(
      { start, goal, environment: makeEnv(reg), options: { maxExpansions: 300000 } },
      Infinity,
    );
  const usedIds = (r: ReturnType<typeof run>) =>
    r.nodes
      .filter((n) => n.edge?.kind === 'affordance')
      .map((n) => (n.edge!.data as { affordanceId: string }).affordanceId);

  const boost = () =>
    createBoostAffordance({
      id: 'boost',
      pad: { x: 6, z: 0 },
      entryRadius: 4,
      exit: { x: 50, z: 0, heading: 0, speed: 0, t: 0 },
      duration: 1,
      cost: 1,
    });
  const misdirect = () =>
    createMisdirectAffordance({
      id: 'decoy',
      launch: { x: 18, z: 0 }, // right on the straight route — tempting
      entryRadius: 4,
      land: { x: 8, z: 0, heading: 0, speed: 0, t: 0 }, // a trap (behind)
      duration: 1,
      cost: 50, // honest, high — makes the honest route cheaper
    });

  it('exports the new affordance types', () => {
    expect(AffordanceType.BoostPad).toBe('boost_pad');
    expect(AffordanceType.Decoy).toBe('decoy');
  });

  it('adopts a genuine cheap boost shortcut (lower cost than driving)', () => {
    const base = run();
    expect(base.found).toBe(true);
    const reg = new AffordanceRegistry();
    reg.add(boost());
    const r = run(reg);
    expect(r.found).toBe(true);
    expect(usedIds(r)).toContain('boost');
    expect(r.cost).toBeLessThan(base.cost - 1e-6);
  });

  it('rejects a misdirect on its own — no special-case logic', () => {
    const base = run();
    const reg = new AffordanceRegistry();
    reg.add(misdirect());
    const r = run(reg);
    expect(r.found).toBe(true);
    expect(usedIds(r)).not.toContain('decoy');
    // identical optimum to the no-affordance plan ⇒ it was truly ignored
    expect(r.cost).toBeCloseTo(base.cost, 6);
  });

  it('with both: takes the boost, ignores the misdirect', () => {
    const reg = new AffordanceRegistry();
    reg.add(boost());
    reg.add(misdirect());
    const r = run(reg);
    expect(r.found).toBe(true);
    const ids = usedIds(r);
    expect(ids).toContain('boost');
    expect(ids).not.toContain('decoy');
  });
});
