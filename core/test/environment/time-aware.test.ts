import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import { linearObstacle } from '../../src/predict/factories';
import type { VehicleAgent, CarKinematicState } from '../../src/agent/types';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const agent: VehicleAgent = defaultVehicleAgent({
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

function buildLib(a: VehicleAgent) {
  const k = 1 / a.minTurnRadius;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(a),
    controlSets: [
      [0, 6],
      [k, 6],
      [-k, 6],
      [k / 2, 6],
      [-k / 2, 6],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
}
const lib = buildLib(agent);
const AGENT_R = 1.4;

describe('TimeAwareEnvironment: time as a dominance dimension', () => {
  it('coarse level collapses nearby times; fine level separates them', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -10, 10, 10)]);
    const base = new VehicleEnvironment(world, agent, lib);
    // Time participates in hash/dominance only when something is actually
    // time-varying — a static wrap skips the tags entirely (dedup). Give the
    // env one (far-away) predicted obstacle so the tagging machinery runs.
    const env = new TimeAwareEnvironment(base, {
      timeQuantum: 0.2,
      obstacles: [{ radius: 0.5, predict: () => ({ x: 999, z: 999, radius: 0.5 }) }],
    });
    const at = (t: number) =>
      env.createNode({ x: 1, z: 1, heading: 0, speed: 0, t }, null, null);

    const a = at(0);
    const b = at(0.05); // same 0.2s fine bucket as t=0
    const c = at(0.5); // different fine bucket

    expect(a.hash).toBe(b.hash);
    expect(a.hash).not.toBe(c.hash);

    const fine = env.levels - 1;
    expect(a.index[fine]).not.toBe(c.index[fine]); // fine separates
    expect(a.index[0]).toBe(c.index[0]); // coarse collapses
  });

  it('a STATIC wrap keeps time out of hash and dominance (dedup preserved)', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -10, 10, 10)]);
    // Two-layer contract: the base env drops its own time bucket via
    // `timeInHash: false` (the plan-vehicle wrappers set this automatically
    // for requests with no moving obstacles/affordances), and the static
    // TimeAware wrap adds no @t tag of its own.
    const base = new VehicleEnvironment(world, agent, lib, { timeInHash: false });
    const env = new TimeAwareEnvironment(base, { timeQuantum: 0.2 });
    const at = (t: number) =>
      env.createNode({ x: 1, z: 1, heading: 0, speed: 0, t }, null, null);
    // Same pose reached at different times is the SAME search state when the
    // world has no dynamics — this was a measured 3.8x expansion inflation.
    expect(at(0).hash).toBe(at(0.5).hash);
    expect(at(0).hash).toBe(at(3.7).hash);
  });

  it('rejects a start that collides with a predicted obstacle', () => {
    const world = new InMemoryNavWorld([rect(1, -5, -5, 35, 5)]);
    const base = new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5 });
    const env = new TimeAwareEnvironment(base, {
      obstacles: [linearObstacle(2, 0, 0, 0, 3, 0, 60)],
      agentRadius: AGENT_R,
    });
    const r = plan(
      {
        start: { x: 2, z: 0, heading: 0, speed: 0, t: 0 },
        goal: { x: 30, z: 0, heading: 0, speed: 0, t: 0 },
        environment: env,
      },
      Infinity,
    );
    expect(r.found).toBe(false);
  });
});

describe('TimeAwareEnvironment: predicted-obstacle avoidance', () => {
  const world = new InMemoryNavWorld([rect(1, 0, -14, 32, 14)]);
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };

  it('detours around a (predicted) stationary obstacle on the straight line', () => {
    const base = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const blockR = 2.5;
    const env = new TimeAwareEnvironment(base, {
      obstacles: [linearObstacle(15, 0, 0, 0, blockR, 0, 1e6)],
      agentRadius: AGENT_R,
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 500000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // hard constraint: no path state overlaps the obstacle at its time
    for (const s of r.path) {
      expect(Math.hypot(s.x - 15, s.z - 0)).toBeGreaterThan(blockR + AGENT_R - 1e-6);
    }
  });

  it('avoids a linearly moving obstacle (no collision at any arrival time)', () => {
    const base = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const obstacle = linearObstacle(15, -12, 0, 4, 2.5, 0, 60); // crosses y=0 ~t=3
    const env = new TimeAwareEnvironment(base, {
      obstacles: [obstacle],
      agentRadius: AGENT_R,
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 500000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    for (const s of r.path) {
      const p = obstacle.predict(s.t);
      if (p) {
        expect(Math.hypot(s.x - p.x, s.z - p.z)).toBeGreaterThan(
          2.5 + AGENT_R - 1e-6,
        );
      }
    }
    // path timestamps strictly increase
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t - 1e-9);
    }
  });
});

describe('TimeAwareEnvironment: moving-obstacle broadphase is a pure accelerator', () => {
  const world = new InMemoryNavWorld([rect(1, 0, -14, 32, 14)]);
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
  const obstacles = [
    linearObstacle(15, -12, 0, 4, 2.5, 0, 60),
    linearObstacle(20, 12, 0, -3, 2.0, 0, 60),
    linearObstacle(10, 8, 1, -2, 1.8, 0, 60),
  ];
  const mk = (broadphase: false | Record<string, never>) => {
    const base = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    return new TimeAwareEnvironment(base, {
      obstacles,
      agentRadius: AGENT_R,
      broadphase,
    });
  };

  it('produces an identical plan with broadphase on vs off', () => {
    const opts = { maxExpansions: 500000 };
    const off = plan({ start, goal, environment: mk(false), options: opts }, Infinity);
    const on = plan({ start, goal, environment: mk({}), options: opts }, Infinity);
    expect(on.found).toBe(off.found);
    expect(on.cost).toBeCloseTo(off.cost, 9);
    expect(on.path.length).toBe(off.path.length);
    for (let i = 0; i < off.path.length; i++) {
      expect(on.path[i]!.x).toBeCloseTo(off.path[i]!.x, 9);
      expect(on.path[i]!.z).toBeCloseTo(off.path[i]!.z, 9);
      expect(on.path[i]!.t).toBeCloseTo(off.path[i]!.t, 9);
    }
    expect(on.stats.expansions).toBe(off.stats.expansions);
  });

  it('still enforces the hard collision constraint', () => {
    const r = plan(
      { start, goal, environment: mk({}), options: { maxExpansions: 500000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    for (const s of r.path) {
      for (const obs of obstacles) {
        const p = obs.predict(s.t);
        if (p) {
          expect(Math.hypot(s.x - p.x, s.z - p.z)).toBeGreaterThan(
            obs.radius + AGENT_R - 1e-6,
          );
        }
      }
    }
  });
});
