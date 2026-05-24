import { describe, it, expect } from 'vitest';
import { planVehicleOnce } from '../../src/planner/plan-vehicle';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { VehicleAgent, CarKinematicState } from '../../src/agent/types';

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
      [0, -3],
    ],
    duration: 0.5,
    substeps: 5,
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

describe('planVehicleOnce', () => {
  const floor = rect(1, -50, -50, 50, 50);

  it('finds a plan on a clear field (matches manual env stack)', () => {
    const world = new InMemoryNavWorld([floor], []);
    const start: CarKinematicState = { x: -20, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: CarKinematicState = { x: 20, z: 0, heading: 0, speed: 0, t: 0 };

    const sugar = planVehicleOnce({
      start,
      goal,
      world,
      agent,
      lib,
      deadlineMs: Infinity,
      maxExpansions: 5000,
    });
    expect(sugar.found).toBe(true);
    expect(sugar.path.length).toBeGreaterThanOrEqual(2);

    // Parity check: identical request through the manual env stack must
    // produce the same final cost (planner is deterministic given identical
    // env construction order + tuning).
    const baseEnv = new VehicleEnvironment(world, agent, lib, {
      posCell: 1.5,
      headingBuckets: 16,
      speedQuant: 4,
      levelDivisors: [4, 2, 1],
      goalRadius: 4,
      goalHeadingTol: Infinity,
      sweepSegmentCheck: false,
      analyticExpansion: { everyN: 6, step: 0.6 },
    });
    // agentRadius derived the same way as planVehicleOnce does internally.
    let rCirc = 0;
    for (const [vx, vz] of agent.footprint) {
      const r = Math.hypot(vx, vz);
      if (r > rCirc) rCirc = r;
    }
    const env = new TimeAwareEnvironment(baseEnv, {
      obstacles: [],
      agentRadius: rCirc,
      broadphase: { sampleStep: 0.5, maxSamples: 24 },
    });
    const manual = plan(
      { start, goal, environment: env, options: { maxExpansions: 5000 } },
      Infinity,
    );
    expect(manual.found).toBe(true);
    expect(sugar.cost).toBeCloseTo(manual.cost, 6);
    expect(sugar.path.length).toBe(manual.path.length);
  });

  it('respects an explicit envOptions override', () => {
    const world = new InMemoryNavWorld([floor], []);
    const start: CarKinematicState = { x: -20, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: CarKinematicState = { x: 20, z: 0, heading: 0, speed: 0, t: 0 };
    // A very tight goal radius — the trivial straight-shot only matches at
    // the goal pose, so analytic Reeds-Shepp shot is the dominant path.
    const r = planVehicleOnce({
      start,
      goal,
      world,
      agent,
      lib,
      envOptions: { goalRadius: 0.5 },
      deadlineMs: Infinity,
      maxExpansions: 10000,
    });
    expect(r.found).toBe(true);
  });
});
