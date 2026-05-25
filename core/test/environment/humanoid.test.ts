import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { HumanoidEnvironment } from '../../src/environment/humanoid-environment';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import {
  defaultHumanoidAgent,
  defaultVehicleAgent,
  kinematicForwardSim,
} from '../../src/agent';
import type { HumanoidState, CarKinematicState } from '../../src/agent/types';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const human = defaultHumanoidAgent({ radius: 0.3, maxSpeed: 4 });

describe('HumanoidEnvironment', () => {
  it('walks across open ground to the goal', () => {
    const world = new InMemoryNavWorld([rect(1, 0, 0, 30, 20)]);
    const env = new HumanoidEnvironment(world, human, { goalRadius: 0.6 });
    const start: HumanoidState = { x: 2, z: 2, heading: 0, t: 0 };
    const goal: HumanoidState = { x: 26, z: 16, heading: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(0.6);
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t - 1e-9);
    }
  });

  it('traverses an L-corridor a car cannot (omnidirectional vs. turn radius)', () => {
    // 1.5-wide right-angle corridor: < 2*minTurnRadius, infeasible for a car.
    const world = new InMemoryNavWorld([
      rect(1, 0, 0, 20, 1.5),
      rect(2, 18.5, 0, 20, 20),
    ]);
    const start = { x: 2, z: 0.75, heading: 0, t: 0 };
    const goalXZ = { x: 19.25, z: 18 };

    const hEnv = new HumanoidEnvironment(world, human, { goalRadius: 0.7 });
    const hr = plan(
      {
        start,
        goal: { ...goalXZ, heading: Math.PI / 2, t: 0 } as HumanoidState,
        environment: hEnv,
        options: { maxExpansions: 300000 },
      },
      Infinity,
    );
    expect(hr.found).toBe(true);

    const car = defaultVehicleAgent({
      minTurnRadius: 3,
      maxSpeed: 8,
      footprint: [
        [1.0, 0.5],
        [-1.0, 0.5],
        [-1.0, -0.5],
        [1.0, -0.5],
      ],
    });
    const k = 1 / car.minTurnRadius;
    const lib = characterizeVehicle({
      forwardSim: kinematicForwardSim(car),
      controlSets: [[0, 6], [k, 6], [-k, 6], [0, -4], [k, -4], [-k, -4]],
      duration: 0.5,
      substeps: 6,
      startSpeeds: [0],
    });
    const vEnv = new VehicleEnvironment(world, car, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const vr = plan(
      {
        start: { ...start, speed: 0 } as CarKinematicState,
        goal: { ...goalXZ, heading: Math.PI / 2, speed: 0, t: 0 } as CarKinematicState,
        environment: vEnv,
        options: { maxExpansions: 40000 },
      },
      Infinity,
    );
    expect(vr.found).toBe(false);
  });

  it('uses an off-mesh jump link to cross an otherwise impassable gap', () => {
    const world = new InMemoryNavWorld([
      rect(1, 0, 0, 8, 6),
      rect(2, 14, 0, 22, 6),
    ]);
    const start: HumanoidState = { x: 2, z: 3, heading: 0, t: 0 };
    const goal: HumanoidState = { x: 18, z: 3, heading: 0, t: 0 };
    const env = new HumanoidEnvironment(world, human, { goalRadius: 0.6 });

    const without = plan(
      { start, goal, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(without.found).toBe(false);

    world.addOffMeshLink({
      from: world.polygonAt(2, 3)!,
      to: world.polygonAt(18, 3)!,
      start: [7, 0, 3],
      end: [15, 0, 3],
      kind: 'jump',
      cost: 1,
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.nodes.some((n) => n.edge?.kind === 'jump')).toBe(true);
  });
});
