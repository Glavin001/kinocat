import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import { PlanRegistry, fromPublishedPlan } from '../../src/predict/plan-registry';
import { asObstacle } from '../../src/predict/factories';
import type { VehicleAgent, VehicleState } from '../../src/agent/types';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}
const agent: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  footprint: [
    [1.2, 0.6],
    [-1.2, 0.6],
    [-1.2, -0.6],
    [1.2, -0.6],
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

describe('PlanRegistry', () => {
  it('interpolates a published plan; clamps before/after', () => {
    const reg = new PlanRegistry();
    const path: VehicleState[] = [
      { x: 0, z: 0, heading: 0, speed: 4, t: 0 },
      { x: 10, z: 0, heading: 0, speed: 4, t: 2 },
    ];
    reg.publish('A', path, 0);
    const p = fromPublishedPlan('A', reg);
    expect(p(-1)).toBeNull();
    expect(p(1)!.x).toBeCloseTo(5, 9);
    const after = p(5)!;
    expect(after.x).toBeCloseTo(10, 9);
    expect(after.t).toBe(5);
    expect(reg.all().length).toBe(1);
    reg.publish('A', []); // empty ⇒ removed
    expect(reg.get('A')).toBeNull();
  });

  it('predictNPC of an unknown NPC is always null', () => {
    const reg = new PlanRegistry();
    expect(reg.predictNPC('ghost')(3)).toBeNull();
  });
});

describe('emergent coordination via shared plans', () => {
  const world = new InMemoryNavWorld([rect(1, 0, -14, 32, 14)]);
  const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };

  it('NPC B routes around NPC A read from the registry', () => {
    const reg = new PlanRegistry();
    // A holds position straddling B's straight line for the whole horizon.
    reg.publish('A', [
      { x: 15, z: 0, heading: 0, speed: 0, t: 0 },
      { x: 15, z: 0, heading: 0, speed: 0, t: 1000 },
    ]);

    const base = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const env = new TimeAwareEnvironment(base, {
      obstacles: [asObstacle(reg.predictNPC('A'), 2.5)],
      agentRadius: 1.4,
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 500000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    const aPred = reg.predictNPC('A');
    for (const s of r.path) {
      const a = aPred(s.t)!;
      expect(Math.hypot(s.x - a.x, s.z - a.z)).toBeGreaterThan(2.5 + 1.4 - 1e-6);
    }

    // Control: with no A published, B passes close to (15,0).
    const reg2 = new PlanRegistry();
    const env2 = new TimeAwareEnvironment(
      new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity }),
      { obstacles: [asObstacle(reg2.predictNPC('A'), 2.5)], agentRadius: 1.4 },
    );
    const r2 = plan(
      { start, goal, environment: env2, options: { maxExpansions: 500000 } },
      Infinity,
    );
    const minD = Math.min(...r2.path.map((s) => Math.hypot(s.x - 15, s.z - 0)));
    expect(minD).toBeLessThan(2.5 + 1.4);
  });
});
