import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { VehicleAgent, VehicleState } from '../../src/agent/types';
import { placeFootprint } from '../../src/internal/geom';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

function buildLib(agent: VehicleAgent) {
  const k = 1 / agent.minTurnRadius;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 6],
      [k, 6],
      [-k, 6],
      [k / 2, 6],
      [-k / 2, 6],
      [0, -4],
      [k, -4],
      [-k, -4],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
}

const agent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  maxReverseSpeed: 4,
  footprint: [
    [1.2, 0.6],
    [-1.2, 0.6],
    [-1.2, -0.6],
    [1.2, -0.6],
  ],
});
const lib = buildLib(agent);

function footprintsClear(world: InMemoryNavWorld, path: VehicleState[]): boolean {
  return path.every((s) =>
    world.footprintClear(placeFootprint(agent.footprint, s.x, s.z, s.heading)),
  );
}

describe('VehicleEnvironment', () => {
  it('plans a curvature-feasible path across open ground', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 40, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 24, z: 6, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(footprintsClear(world, r.path)).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(1.5);
  });

  it('detours around an obstacle that blocks the straight line', () => {
    const obstacle = [
      [12, -3],
      [18, -3],
      [18, 3],
      [12, 3],
    ] as [number, number][];
    const world = new InMemoryNavWorld([rect(1, 0, -10, 30, 10)], [obstacle]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    // straight line is blocked, so a detour is mandatory
    expect(world.segmentClear(start.x, start.z, goal.x, goal.z)).toBe(false);
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 400000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(footprintsClear(world, r.path)).toBe(true);
  });

  it('produces a reverse maneuver in a corridor too narrow to turn around', () => {
    // corridor width 2 < 2*minTurnRadius: turning around is impossible
    const world = new InMemoryNavWorld([rect(1, 0, -1, 30, 1)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 20, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 5, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    const reversed = r.nodes.some(
      (n) =>
        n.edge?.kind === 'drive-reverse' ||
        (n.edge?.kind === 'reeds-shepp' &&
          (n.edge.data as { reverse?: boolean }).reverse === true),
    );
    expect(reversed).toBe(true);
    for (const s of r.path) expect(Math.abs(s.heading)).toBeLessThan(0.35);
    expect(footprintsClear(world, r.path)).toBe(true);
  });
});

describe('VehicleEnvironment Reeds-Shepp analytic expansion', () => {
  it('solves a long open straight in a handful of expansions', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 300, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: {},
    });
    const start: VehicleState = { x: 5, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 280, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 100000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(20); // analytic shot, not 90+ prims
    expect(r.nodes.some((n) => n.edge?.kind === 'reeds-shepp')).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(1.5);
  });

  it('never shoots through an obstacle (static collision-checked)', () => {
    const obstacle = [
      [12, -3],
      [18, -3],
      [18, 3],
      [12, 3],
    ] as [number, number][];
    const world = new InMemoryNavWorld([rect(1, 0, -10, 30, 10)], [obstacle]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: {},
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 400000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // every analytic edge's sampled curve stays out of the obstacle
    for (const n of r.nodes) {
      if (n.edge?.kind !== 'reeds-shepp') continue;
      const samples = (n.edge.data as { samples: [number, number][] }).samples;
      for (const [x, z] of samples) {
        const inObstacle = x > 12 && x < 18 && z > -3 && z < 3;
        expect(inObstacle).toBe(false);
      }
    }
  });

  it('can be disabled (no analytic edges, still solves)', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 60, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: false,
    });
    const r = plan(
      {
        start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
        goal: { x: 50, z: 0, heading: 0, speed: 0, t: 0 },
        environment: env,
        options: { maxExpansions: 300000 },
      },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.nodes.every((n) => n.edge?.kind !== 'reeds-shepp')).toBe(true);
  });
});
