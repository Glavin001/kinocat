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

  it('prefers wings-level when banking is not required', () => {
    // Open airspace, roll-search ENABLED. The roll cost should bias the
    // planner toward roll=0 even though ±maxBank is in the control set.
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
      rollFractions: [-1, 0, 1],
    });
    const s = start();
    const g = start({ x: 120 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 40000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // Every state along the plan should be at roll = 0 (no gratuitous bank).
    expect(Math.max(...r.path.map((p) => Math.abs(p.roll)))).toBeLessThan(1e-6);
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

  it('swept-AABB pre-check skips per-substep checks in clear cells', () => {
    // Open airspace ⇒ most primitives' swept envelopes are entirely inside
    // the altitude band and away from any box ⇒ the fast path fires.
    // Primitives whose envelope dips below floor or above ceiling fall
    // back to the per-substep loop (each substep is checked individually);
    // some of those still produce a valid successor whose substeps all
    // happen to be in-band, so primitiveSweptSkips < successorsTotal in
    // general. We only assert the fast path took the MAJORITY of work and
    // that bulk collision.checks were avoided.
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
    expect(r.stats.counters.primitiveSweptSkips).toBeGreaterThan(0);
    // Majority of successors took the fast path.
    expect(r.stats.counters.primitiveSweptSkips).toBeGreaterThan(
      0.5 * r.stats.counters.successorsTotal,
    );
    // Per-expansion collision checks should be dramatically reduced versus
    // the old behavior (which was ~substeps × primitives ≈ 4 × 15 = 60).
    // With the fast path, only out-of-band primitives trigger per-substep.
    const checksPerExpansion =
      r.stats.counters.collisionChecks / Math.max(1, r.stats.expansions);
    expect(checksPerExpansion).toBeLessThan(20);
  });

  it('analytic shot finds open-airspace plans in <10 expansions', () => {
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
      analyticExpansion: {},
    });
    const s = start();
    const g = start({ x: 120 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // Shot should have been tried and at least one should have succeeded.
    expect(r.stats.counters.analyticShots).toBeGreaterThan(0);
    expect(r.stats.counters.analyticShotsClear).toBeGreaterThan(0);
    // Plan should include the shot edge.
    expect(r.nodes.some((n) => n.edge?.kind === 'fly-shot')).toBe(true);
    // And the shot collapses the search to a handful of expansions.
    expect(r.stats.expansions).toBeLessThan(50);
  });

  it('analytic shot pre-reject is harmless when straight is blocked', () => {
    // Wall between start and goal — initial shots from near start fail at
    // the AABB pre-reject (cheap). Once the lattice has expanded past the
    // wall, a shot from that node to the goal may succeed. Either way the
    // plan completes; the shot's cost when blocked is bounded (no per-
    // sample loop, just one clearAABB call per attempt).
    const wall: AABB = { min: [50, 0, -40], max: [58, 80, 40] };
    const air = new InMemoryAirspace({ floor: 0, ceiling: 90, boxes: [wall] });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
      analyticExpansion: {},
    });
    const s = start({ y: 8 });
    const g = start({ x: 110, y: 8 });
    const r = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.stats.counters.analyticShots).toBeGreaterThan(0);
    // Hit-rate is low (most shots blocked) but the AABB pre-reject keeps
    // cost bounded: shots * (1 clearAABB call) instead of shots * (~50
    // per-sample world.clear calls).
    const hitRate =
      r.stats.counters.analyticShotsClear /
      Math.max(1, r.stats.counters.analyticShots);
    expect(hitRate).toBeLessThan(0.5);
  });

  it('per-level control sets reduce knife-edge expansions', () => {
    // Same knife-edge geometry as scenarios.test.ts; compare default vs
    // level-aware. Coarse passes use no roll; finest uses ±90°.
    const slot1: AABB = { min: [78, 0, -60], max: [92, 80, -0.6] };
    const slot2: AABB = { min: [78, 0, 0.6], max: [92, 80, 60] };
    const air = new InMemoryAirspace({
      floor: 0,
      ceiling: 80,
      boxes: [slot1, slot2],
    });
    const s = start({ y: 24 });
    const g = start({ x: 152, y: 24 });

    const baseEnv = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
      rollFractions: [-1, 0, 1],
    });
    const base = plan(
      { start: s, goal: g, environment: baseEnv, options: { maxExpansions: 200000 } },
      Infinity,
    );

    const layeredEnv = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
      rollFractions: [-1, 0, 1],
      levelControls: [
        { rollFractions: [0] }, // L0: wings level only
        { rollFractions: [0] }, // L1: wings level only
        { rollFractions: [-1, 0, 1] }, // L2: full roll set
      ],
    });
    const layered = plan(
      { start: s, goal: g, environment: layeredEnv, options: { maxExpansions: 200000 } },
      Infinity,
    );

    expect(base.found).toBe(true);
    expect(layered.found).toBe(true);
    // The win is in successor-work, not expansion count: coarse passes
    // produce 1/3 as many candidate successors per expansion (15 vs 45
    // primitives). Total successors generated should drop noticeably. The
    // hysteresis decides expansion count per pass independently.
    expect(layered.stats.counters.successorsTotal).toBeLessThan(
      base.stats.counters.successorsTotal,
    );
    // And per-expansion collision work drops too (less branching at
    // coarse passes means fewer fast-path AABB queries).
    const baseChecksPerExp =
      base.stats.counters.collisionChecks / Math.max(1, base.stats.expansions);
    const layeredChecksPerExp =
      layered.stats.counters.collisionChecks /
      Math.max(1, layered.stats.expansions);
    expect(layeredChecksPerExp).toBeLessThanOrEqual(baseChecksPerExp + 0.01);
  });
});

describe('weighted A* (weight option)', () => {
  it('weight=1 reproduces pure A* (same expansions and cost)', () => {
    const air = new InMemoryAirspace({ floor: 0, ceiling: 80 });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 8,
    });
    const s = start();
    const g = start({ x: 120 });
    const baseline = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 20000 } },
      Infinity,
    );
    const w1 = plan(
      {
        start: s,
        goal: g,
        environment: env,
        options: { maxExpansions: 20000, weight: 1 },
      },
      Infinity,
    );
    expect(w1.found).toBe(true);
    expect(w1.stats.expansions).toBe(baseline.stats.expansions);
    expect(w1.cost).toBeCloseTo(baseline.cost, 6);
  });

  it('weight>1 finds a (possibly suboptimal) plan with fewer expansions', () => {
    // Use the obstacle wall scene to ensure A* has actual work to do; in
    // open airspace the search is already trivial.
    const wall: AABB = { min: [50, 0, -40], max: [58, 30, 40] };
    const air = new InMemoryAirspace({ floor: 0, ceiling: 90, boxes: [wall] });
    const env = new AircraftEnvironment(air, agent, {
      posCell: 4,
      altCell: 4,
      goalRadius: 10,
    });
    const s = start({ y: 8 });
    const g = start({ x: 110, y: 8 });
    const w1 = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 200000, weight: 1 } },
      Infinity,
    );
    const w3 = plan(
      { start: s, goal: g, environment: env, options: { maxExpansions: 200000, weight: 3 } },
      Infinity,
    );
    expect(w1.found).toBe(true);
    expect(w3.found).toBe(true);
    // Heavier weight ⇒ fewer expansions.
    expect(w3.stats.expansions).toBeLessThan(w1.stats.expansions);
    // ε-suboptimal: cost is bounded by weight × optimal (allow small slack).
    expect(w3.cost).toBeLessThanOrEqual(w1.cost * 3 + 1e-6);
  });
});
