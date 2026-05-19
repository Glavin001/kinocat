import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { AircraftEnvironment } from '../../src/environment/aircraft-environment';
import {
  InMemoryAirspace,
  type AABB,
} from '../../src/environment/airspace-world';
import {
  defaultAircraftAgent,
  aircraftForwardSim,
} from '../../src/agent/aircraft';
import type { AircraftState } from '../../src/agent/types';

const agent = defaultAircraftAgent({
  minTurnRadius: 12,
  minSpeed: 6,
  maxSpeed: 18,
  maxClimbAngle: Math.PI / 6,
  radius: 1.4,
});

function start(over: Partial<AircraftState> = {}): AircraftState {
  return { x: 0, y: 20, z: 0, heading: 0, pitch: 0, speed: 18, t: 0, ...over };
}

describe('aircraftForwardSim', () => {
  it('climbs when commanded a positive flight-path angle', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [0, agent.maxClimbAngle, agent.maxSpeed], 1);
    expect(s.y).toBeGreaterThan(20);
    expect(s.pitch).toBeCloseTo(agent.maxClimbAngle, 6);
    // airspeed is conserved: 3D step length ≈ speed * dt
    expect(Math.hypot(s.x, s.y - 20, s.z)).toBeCloseTo(18, 1);
  });

  it('defaults to straight level flight at cruise when controls are absent', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start({ heading: 0 }), [], 1);
    expect(s.heading).toBeCloseTo(0, 9); // curvature → 0
    expect(s.pitch).toBeCloseTo(0, 9); // climb → 0
    expect(s.y).toBeCloseTo(20, 9); // level
    expect(s.speed).toBeCloseTo(agent.maxSpeed, 9); // speed → maxSpeed
    expect(s.x).toBeCloseTo(agent.maxSpeed, 6);
  });

  it('clamps climb angle and turn curvature to the agent limits', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [99, 99, 999], 1);
    expect(s.pitch).toBeCloseTo(agent.maxClimbAngle, 6);
    expect(Math.abs(s.heading)).toBeLessThanOrEqual(
      (agent.maxSpeed / agent.minTurnRadius) * 1 + 1e-6,
    );
    expect(s.speed).toBeLessThanOrEqual(agent.maxSpeed + 1e-9);
  });
});

describe('InMemoryAirspace', () => {
  const box: AABB = { min: [40, 0, -10], max: [50, 30, 10] };
  const air = new InMemoryAirspace({
    floor: 0,
    ceiling: 60,
    boxes: [box],
    // a no-fly zone drifting along +z: centre = (100, 20, t * 2)
    zones: [{ radius: 5, predict: (t) => ({ x: 100, y: 20, z: t * 2 }) }],
  });

  it('rejects points inside a box, below the floor, above the ceiling', () => {
    expect(air.clear(45, 15, 0, 0, 1.4)).toBe(false); // inside box
    expect(air.clear(0, 0.5, 0, 0, 1.4)).toBe(false); // below floor
    expect(air.clear(0, 59.5, 0, 0, 1.4)).toBe(false); // above ceiling
    expect(air.clear(0, 20, 0, 0, 1.4)).toBe(true); // open air
  });

  it('rejects a point inside a moving zone only while the zone is there', () => {
    expect(air.clear(100, 20, 0, 0, 1.4)).toBe(false); // zone at z=0 at t=0
    expect(air.clear(100, 20, 0, 50, 1.4)).toBe(true); // zone moved to z=100
  });
});

describe('AircraftEnvironment + IGHA*', () => {
  it('plans a straight cruise across open airspace', () => {
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
    });
    const s = start();
    const g = start({ x: 120, z: 0, speed: 0 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    const end = r.path[r.path.length - 1]!;
    expect(Math.hypot(end.x - g.x, end.y - g.y, end.z - g.z)).toBeLessThanOrEqual(8);
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t - 1e-9);
    }
  });

  it('searches altitude: climbs over a wall that blocks level flight', () => {
    // A wall from the floor up to y=30 across the direct route. The only
    // feasible plan gains altitude — altitude is genuinely searched.
    const wall: AABB = { min: [50, 0, -40], max: [58, 30, 40] };
    const air = new InMemoryAirspace({ floor: 0, ceiling: 90, boxes: [wall] });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
      levelDivisors: [4, 2, 1],
    });
    const s = start({ y: 8 });
    const g = start({ x: 110, y: 8, z: 0, speed: 0 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 80000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(Math.max(...r.path.map((p) => p.y))).toBeGreaterThan(30);
    for (const p of r.path) {
      expect(air.clear(p.x, p.y, p.z, p.t, agent.radius)).toBe(true);
    }
  });

  it('heuristic is an admissible lower bound on the plan cost', () => {
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
    });
    const s = start();
    const g = start({ x: 120, z: 0, speed: 0 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(env.heuristic(s, g)).toBeLessThanOrEqual(r.cost + 1e-6);
  });

  it('reports an invalid start when it begins inside an obstacle', () => {
    const box: AABB = { min: [-5, 0, -5], max: [5, 40, 5] };
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80, boxes: [box] });
    const env = new AircraftEnvironment(air, agent);
    const r = plan(
      { start: start({ y: 20 }), goal: start({ x: 120 }), environment: env },
      Infinity,
    );
    expect(r.found).toBe(false);
  });
});
