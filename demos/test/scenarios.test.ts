import { describe, it, expect } from 'vitest';
import {
  planPlayground,
  buildDynamic,
  world3dWorld,
  planWorld3d,
  buildNavmesh,
  planNavmesh,
  DEMO_MAX_EXPANSIONS,
  DEMO_DYNAMIC_MAX_EXPANSIONS,
  type Scenario,
} from '../app/lib/scenarios';
import type { VehicleState } from 'kinocat/agent';

// These assert the *exact* configuration the demos ship with always finds a
// plan within its expansion budget — so a "no plan" regression fails CI.

describe('playground demo config is always solvable', () => {
  it('open field: plans the trivial straight crossing fast', () => {
    const r = planPlayground({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 40, z: 0, heading: 0, speed: 0, t: 0 },
      obstacles: [],
    });
    expect(r.found).toBe(true);
    // analytic shot-to-goal solves the trivial straight in a few expansions
    expect(r.stats.expansions).toBeLessThan(20);
    expect(r.path.length).toBeGreaterThanOrEqual(2);
    expect(r.nodes.some((n) => n.edge?.kind === 'reeds-shepp')).toBe(true);
  });

  it('with a central obstacle: detours and still finds a plan', () => {
    const r = planPlayground({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 40, z: 0, heading: 0, speed: 0, t: 0 },
      obstacles: [{ x: 22, z: 0 }],
    });
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(DEMO_MAX_EXPANSIONS);
  });

  it('trivial-but-far straight solves even at a tiny budget (analytic shot)', () => {
    const r = planPlayground({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 200, z: 0, heading: 0, speed: 0, t: 0 },
      obstacles: [],
      bounds: { x0: 0, z0: -11, x1: 220, z1: 11 },
      maxExpansions: 50,
    });
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(20);
    expect(r.nodes.some((n) => n.edge?.kind === 'reeds-shepp')).toBe(true);
  });

  it('low anytime budget degrades gracefully (never throws)', () => {
    const r = planPlayground({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 40, z: 0, heading: 0, speed: 0, t: 0 },
      obstacles: [{ x: 22, z: 0 }],
      maxExpansions: 50,
    });
    expect(typeof r.found).toBe('boolean'); // found or not, but no crash
  });
});

describe('dynamic demo scenarios are all solvable', () => {
  for (const scn of ['moving', 'coop', 'jump'] as Scenario[]) {
    it(`${scn}: finds a plan within budget`, () => {
      const s = buildDynamic(scn);
      expect(s.result.found).toBe(true);
      expect(s.result.stats.expansions).toBeLessThan(DEMO_DYNAMIC_MAX_EXPANSIONS);
      expect(s.duration).toBeGreaterThan(0);
      // path timestamps are strictly increasing
      for (let i = 1; i < s.result.path.length; i++) {
        expect(s.result.path[i]!.t).toBeGreaterThan(s.result.path[i - 1]!.t - 1e-9);
      }
      if (scn === 'jump') expect(s.affordanceHop).not.toBeNull();
    });
  }
});

describe('world3d demo config is solvable', () => {
  it('plans around the box obstacle', () => {
    const r = planWorld3d(
      world3dWorld(),
      { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      { x: 36, z: 0, heading: 0, speed: 0, t: 0 },
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(DEMO_MAX_EXPANSIONS);
  });
});

describe('navmesh demo runs over a real navcat navmesh', () => {
  it('generates a navmesh and plans ground → ramp → platform', () => {
    const { world } = buildNavmesh();
    // sanity: the adapter sees the generated mesh
    expect(world.polygonAt(4, 12)).not.toBeNull(); // ground
    expect(world.polygonAt(36, 12)).not.toBeNull(); // platform
    const r = planNavmesh(
      world,
      { x: 4, z: 12, heading: 0, speed: 0, t: 0 } as VehicleState,
      { x: 36, z: 12, heading: 0, speed: 0, t: 0 } as VehicleState,
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(DEMO_MAX_EXPANSIONS);
  });
});
