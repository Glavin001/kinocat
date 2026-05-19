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
import {
  poseToOBB,
  obbHitsAABB,
  obbHitsSphere,
} from '../../src/internal/obb';

const agent = defaultAircraftAgent({
  minTurnRadius: 12,
  minSpeed: 6,
  maxSpeed: 18,
  maxClimbAngle: Math.PI / 6,
  maxBank: Math.PI / 2,
  halfLength: 2,
  halfSpan: 1.5,
  halfHeight: 0.3,
});

const HALF: [number, number, number] = [
  agent.halfLength,
  agent.halfSpan,
  agent.halfHeight,
];

function start(over: Partial<AircraftState> = {}): AircraftState {
  return {
    x: 0,
    y: 20,
    z: 0,
    heading: 0,
    pitch: 0,
    roll: 0,
    speed: 18,
    t: 0,
    ...over,
  };
}

describe('aircraftForwardSim', () => {
  it('climbs when commanded a positive flight-path angle', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [0, agent.maxClimbAngle, 0, agent.maxSpeed], 1);
    expect(s.y).toBeGreaterThan(20);
    expect(s.pitch).toBeCloseTo(agent.maxClimbAngle, 6);
    expect(Math.hypot(s.x, s.y - 20, s.z)).toBeCloseTo(18, 1);
  });

  it('tracks the commanded bank angle in one step (quasi-static airframe)', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [0, 0, agent.maxBank, agent.maxSpeed], 1);
    expect(s.roll).toBeCloseTo(agent.maxBank, 6);
  });

  it('defaults to straight level wings-level flight when controls are absent', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [], 1);
    expect(s.heading).toBeCloseTo(0, 9);
    expect(s.pitch).toBeCloseTo(0, 9);
    expect(s.roll).toBeCloseTo(0, 9);
    expect(s.speed).toBeCloseTo(agent.maxSpeed, 9);
    expect(s.x).toBeCloseTo(agent.maxSpeed, 6);
  });

  it('clamps climb, turn, bank, and speed to agent limits', () => {
    const sim = aircraftForwardSim(agent);
    const s = sim(start(), [99, 99, 99, 999], 1);
    expect(s.pitch).toBeCloseTo(agent.maxClimbAngle, 6);
    expect(s.roll).toBeCloseTo(agent.maxBank, 6);
    expect(Math.abs(s.heading)).toBeLessThanOrEqual(
      (agent.maxSpeed / agent.minTurnRadius) * 1 + 1e-6,
    );
    expect(s.speed).toBeLessThanOrEqual(agent.maxSpeed + 1e-9);
  });
});

describe('OBB primitives', () => {
  it('hits an AABB it overlaps and misses one it does not', () => {
    const pose = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0 };
    const obb = poseToOBB(pose, HALF);
    expect(obbHitsAABB(obb, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])).toBe(true);
    expect(obbHitsAABB(obb, [10, 10, 10], [11, 11, 11])).toBe(false);
  });

  it('lets a banked plane fit a tall slot a level plane cannot', () => {
    // Slot z ∈ [-0.7, 0.7] (1.4 wide), full ceiling tall. Wings level
    // (halfSpan=1.5) cannot fit; banked 90° (wings vertical, halfHeight=0.3
    // becomes the lateral extent) does fit.
    const slot: AABB = { min: [-1, -50, -50], max: [1, 50, -0.7] };
    const slot2: AABB = { min: [-1, -50, 0.7], max: [1, 50, 50] };
    const level = poseToOBB({ x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0 }, HALF);
    expect(
      obbHitsAABB(level, slot.min, slot.max) ||
        obbHitsAABB(level, slot2.min, slot2.max),
    ).toBe(true);
    const banked = poseToOBB(
      { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: Math.PI / 2 },
      HALF,
    );
    expect(obbHitsAABB(banked, slot.min, slot.max)).toBe(false);
    expect(obbHitsAABB(banked, slot2.min, slot2.max)).toBe(false);
  });

  it('detects a sphere hugging the OBB and misses one outside', () => {
    const obb = poseToOBB(
      { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0 },
      HALF,
    );
    expect(obbHitsSphere(obb, [agent.halfLength + 0.4, 0, 0], 0.5)).toBe(true);
    expect(obbHitsSphere(obb, [10, 10, 10], 0.5)).toBe(false);
  });
});

describe('InMemoryAirspace (OBB collision)', () => {
  const box: AABB = { min: [40, 0, -10], max: [50, 30, 10] };
  const air = new InMemoryAirspace({
    floor: 0,
    ceiling: 60,
    boxes: [box],
    zones: [{ radius: 5, predict: (t) => ({ x: 100, y: 20, z: t * 2 }) }],
  });

  const pose = (over: Partial<AircraftState>) => {
    const s = start(over);
    return { x: s.x, y: s.y, z: s.z, yaw: s.heading, pitch: s.pitch, roll: s.roll };
  };

  it('rejects an OBB inside a box, below floor, above ceiling', () => {
    expect(air.clear(pose({ x: 45, y: 15 }), HALF, 0)).toBe(false);
    expect(air.clear(pose({ x: 0, y: 0.1 }), HALF, 0)).toBe(false);
    expect(air.clear(pose({ x: 0, y: 59.9 }), HALF, 0)).toBe(false);
    expect(air.clear(pose({ x: 0, y: 20 }), HALF, 0)).toBe(true);
  });

  it('rejects only while a moving zone is at the queried time', () => {
    expect(air.clear(pose({ x: 100, y: 20, z: 0 }), HALF, 0)).toBe(false);
    expect(air.clear(pose({ x: 100, y: 20, z: 0 }), HALF, 50)).toBe(true);
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
    const g = start({ x: 120 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 40000 } },
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
    const wall: AABB = { min: [50, 0, -40], max: [58, 30, 40] };
    const air = new InMemoryAirspace({ floor: 0, ceiling: 90, boxes: [wall] });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
      levelDivisors: [4, 2, 1],
    });
    const s = start({ y: 8 });
    const g = start({ x: 110, y: 8 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(Math.max(...r.path.map((p) => p.y))).toBeGreaterThan(30);
    for (const p of r.path) {
      const pose = { x: p.x, y: p.y, z: p.z, yaw: p.heading, pitch: p.pitch, roll: p.roll };
      expect(air.clear(pose, HALF, p.t)).toBe(true);
    }
  });

  it('heuristic is an admissible lower bound on plan cost', () => {
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
    });
    const s = start();
    const g = start({ x: 120 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(env.heuristic(s, g)).toBeLessThanOrEqual(r.cost + 1e-6);
  });

  it('reports an invalid start when its OBB intersects an obstacle', () => {
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
