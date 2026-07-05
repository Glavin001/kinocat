import { describe, it, expect } from 'vitest';
import { createEtaOracle } from '../../src/predict/eta-oracle';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { near, inside } from '../../src/scenario/regions';
import type { ScenarioState } from '../../src/scenario/types';
import type { Pt } from '../../src/internal/geom';

const AGENT = { maxSpeed: 10 };

function st(x: number, z: number): ScenarioState {
  return { x, z, heading: 0, speed: 0, t: 0 };
}

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const FLOOR = rect(1, 0, 0, 40, 20);
// Wall at x∈[18,22], z∈[0,14] — passable only through the z∈[14,20] gap.
const WALL_WITH_GAP: Pt[] = [
  [18, 0],
  [22, 0],
  [22, 14],
  [18, 14],
];
// Wall spanning the full floor depth — splits the floor into two rooms.
const FULL_WALL: Pt[] = [
  [18, -1],
  [22, -1],
  [22, 21],
  [18, 21],
];

describe('createEtaOracle over InMemoryNavWorld', () => {
  it('reflects the obstacle detour, not the straight line', () => {
    const world = new InMemoryNavWorld([FLOOR], [WALL_WITH_GAP]);
    const oracle = createEtaOracle(world, AGENT);
    const region = near({ x: 35, z: 10 }, 2);
    const r = oracle.eta(st(5, 10), region);
    expect(r.reachable).toBe(true);
    // Straight-line bound: (30 − 2)/10 = 2.8 s. The wall forces a detour
    // through the z>14 gap, so the grid bound must exceed the straight line.
    expect(r.seconds).toBeGreaterThan(2.8);
    // …but stays an admissible lower bound on any real path (sanity ceiling).
    expect(r.seconds).toBeLessThan(10);
  });

  it('is ~0 from inside the region', () => {
    const world = new InMemoryNavWorld([FLOOR], [WALL_WITH_GAP]);
    const oracle = createEtaOracle(world, AGENT);
    const region = inside([
      [30, 5],
      [40, 5],
      [40, 15],
      [30, 15],
    ]);
    const r = oracle.eta(st(35, 10), region);
    expect(r.reachable).toBe(true);
    expect(r.seconds).toBeLessThan(0.1);
  });

  it('reports unreachable across a sealed wall', () => {
    const world = new InMemoryNavWorld([FLOOR], [FULL_WALL]);
    const oracle = createEtaOracle(world, AGENT);
    const region = near({ x: 35, z: 10 }, 2);
    const sealed = oracle.eta(st(5, 10), region);
    expect(sealed.reachable).toBe(false);
    expect(sealed.seconds).toBe(Infinity);
    // Same-room start stays reachable.
    const sameRoom = oracle.eta(st(30, 10), region);
    expect(sameRoom.reachable).toBe(true);
    expect(sameRoom.seconds).toBeGreaterThanOrEqual(0);
  });

  it('LRU: repeat queries reuse the field; revision bump rebuilds it', () => {
    const world = new InMemoryNavWorld([FLOOR], [WALL_WITH_GAP]);
    let builds = 0;
    const orig = world.buildRegionLowerBound.bind(world);
    world.buildRegionLowerBound = (c) => {
      builds++;
      return orig(c);
    };
    const oracle = createEtaOracle(world, AGENT);
    const region = near({ x: 35, z: 10 }, 2);
    oracle.eta(st(5, 10), region);
    oracle.eta(st(6, 10), region);
    oracle.eta(st(7, 10), region);
    expect(builds).toBe(1);
    // Geometry change → revision moves → the cached field must not survive.
    world.setObstacles([WALL_WITH_GAP]);
    oracle.eta(st(5, 10), region);
    expect(builds).toBe(2);
  });

  it('LRU evicts the oldest region past capacity', () => {
    const world = new InMemoryNavWorld([FLOOR], []);
    let builds = 0;
    const orig = world.buildRegionLowerBound.bind(world);
    world.buildRegionLowerBound = (c) => {
      builds++;
      return orig(c);
    };
    const oracle = createEtaOracle(world, AGENT, { lruSize: 2 });
    const a = near({ x: 10, z: 10 }, 1);
    const b = near({ x: 20, z: 10 }, 1);
    const c = near({ x: 30, z: 10 }, 1);
    oracle.eta(st(5, 10), a); // build a
    oracle.eta(st(5, 10), b); // build b
    oracle.eta(st(5, 10), c); // build c, evicts a
    expect(builds).toBe(3);
    oracle.eta(st(5, 10), c); // hit
    oracle.eta(st(5, 10), b); // hit
    expect(builds).toBe(3);
    oracle.eta(st(5, 10), a); // rebuilt after eviction
    expect(builds).toBe(4);
  });

  it('falls back to the kinematic bound when the world lacks the seam method', () => {
    const world = new InMemoryNavWorld([FLOOR], []);
    (world as { buildRegionLowerBound?: unknown }).buildRegionLowerBound = undefined;
    const oracle = createEtaOracle(world, AGENT);
    const r = oracle.eta(st(5, 10), near({ x: 35, z: 10 }, 2));
    expect(r.reachable).toBe(true);
    expect(r.seconds).toBeCloseTo(2.8, 5);
  });

  it('prebuild populates the cache off the query path', () => {
    const world = new InMemoryNavWorld([FLOOR], []);
    let builds = 0;
    const orig = world.buildRegionLowerBound.bind(world);
    world.buildRegionLowerBound = (c) => {
      builds++;
      return orig(c);
    };
    const oracle = createEtaOracle(world, AGENT);
    const region = near({ x: 35, z: 10 }, 2);
    expect(oracle.prebuild(region)).toBe(true);
    expect(builds).toBe(1);
    oracle.eta(st(5, 10), region);
    expect(builds).toBe(1);
  });

  it('meets the latency gate: warm queries well under 2 ms median', () => {
    const world = new InMemoryNavWorld([FLOOR], [WALL_WITH_GAP]);
    const oracle = createEtaOracle(world, AGENT);
    const region = near({ x: 35, z: 10 }, 2);
    oracle.prebuild(region);
    const N = 1000;
    const times: number[] = [];
    for (let i = 0; i < N; i++) {
      const x = 2 + (i % 30);
      const t0 = performance.now();
      oracle.eta(st(x, 10), region);
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const median = times[N / 2]!;
    console.info(
      `eta-oracle: median=${(median * 1000).toFixed(1)}µs ` +
        `p95=${(times[Math.ceil(N * 0.95) - 1]! * 1000).toFixed(1)}µs`,
    );
    expect(median).toBeLessThan(2);
  });
});
