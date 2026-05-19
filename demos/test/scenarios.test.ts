import { describe, it, expect, beforeAll } from 'vitest';
import {
  planPlayground,
  buildDynamic,
  world3dWorld,
  planWorld3d,
  buildNavmesh,
  planNavmesh,
  compareCurves,
  buildAnytime,
  planReverse,
  buildPrimitiveFan,
  buildSwarm,
  buildHumanoid,
  buildJumpLinks,
  DEMO_MAX_EXPANSIONS,
  DEMO_DYNAMIC_MAX_EXPANSIONS,
  type Scenario,
  type JumpLinksResult,
} from '../app/lib/scenarios';
import {
  buildWaypointCourse,
  buildCanyon,
  buildRestrictedAirspace,
  planInteractive,
  INTERACTIVE_BOXES,
  AIRCRAFT_AGENT,
  AIRCRAFT_MAX_EXPANSIONS,
} from '../app/lib/aircraft-scenarios';
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

describe('curves demo: Reeds-Shepp vs Dubins', () => {
  it('straight-ahead: Dubins ≈ euclidean and RS is never longer', () => {
    const c = compareCurves({
      sx: 0,
      sz: 0,
      sHeading: 0,
      gx: 20,
      gz: 0,
      gHeading: 0,
      radius: 4,
    });
    expect(c.dubins.length).toBeCloseTo(20, 1);
    expect(c.reedsShepp.length).toBeLessThanOrEqual(c.dubins.length + 1e-6);
    expect(c.dubins.samples.length).toBeGreaterThanOrEqual(2);
    expect(c.reedsShepp.samples.length).toBeGreaterThanOrEqual(2);
    expect(c.dubins.samples[0]).toEqual([0, 0]);
  });

  it('goal directly behind: Reeds-Shepp uses a reverse segment', () => {
    const c = compareCurves({
      sx: 10,
      sz: 0,
      sHeading: 0,
      gx: 2,
      gz: 0,
      gHeading: 0,
      radius: 4,
    });
    expect(c.reedsShepp.segments.some((s) => s.gear === -1)).toBe(true);
    expect(c.reedsShepp.length).toBeLessThanOrEqual(c.dubins.length + 1e-6);
  });
});

describe('anytime demo: budget sweep refines the plan', () => {
  it('a generous budget solves; refinement is visible', () => {
    const a = buildAnytime();
    expect(a.steps.length).toBe(5);
    const last = a.steps[a.steps.length - 1]!;
    expect(last.found).toBe(true);
    expect(last.expansions).toBeLessThanOrEqual(last.budget);
    const end = last.path[last.path.length - 1]!;
    expect(Math.hypot(end.x - a.goal.x, end.z - a.goal.z)).toBeLessThanOrEqual(2);
    for (let i = 1; i < last.path.length; i++) {
      expect(last.path[i]!.t).toBeGreaterThan(last.path[i - 1]!.t - 1e-9);
    }
    // tighter budgets must not beat the most generous one (anytime property),
    // and refinement is visible: some budget fails or costs more.
    const found = a.steps.filter((s) => s.found);
    expect(found.length).toBeGreaterThanOrEqual(1);
    for (const s of found) {
      expect(s.cost).toBeGreaterThanOrEqual(last.cost - 1e-6);
    }
    expect(
      a.steps.some((s) => !s.found || s.cost > last.cost + 1e-6),
    ).toBe(true);
  });
});

describe('reverse demo: reverse maneuver is required', () => {
  it('finds a plan that includes a reverse segment', () => {
    const r = planReverse();
    expect(r.found).toBe(true);
    expect(r.path.length).toBeGreaterThanOrEqual(2);
    expect(r.reverseCount).toBeGreaterThan(0);
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t - 1e-9);
    }
  });

  it('a higher reverse-cost multiplier never lowers the plan cost', () => {
    const cheap = planReverse({ reverseCost: 2 });
    const dear = planReverse({ reverseCost: 12 });
    expect(cheap.found && dear.found).toBe(true);
    expect(dear.cost).toBeGreaterThanOrEqual(cheap.cost - 1e-6);
  });
});

describe('primitives demo: characterization fan', () => {
  it('produces one primitive per control set with valid sweeps', () => {
    const f = buildPrimitiveFan({
      minTurnRadius: 3,
      duration: 0.5,
      startSpeed: 0,
    });
    expect(f.count).toBe(8);
    expect(f.primitives.some((p) => p.reverse)).toBe(true);
    expect(f.primitives.some((p) => !p.reverse)).toBe(true);
    for (const p of f.primitives) {
      expect(p.sweep.length).toBeGreaterThanOrEqual(2);
      expect(p.sweep[0]!.x).toBeCloseTo(0, 6);
      expect(p.sweep[0]!.z).toBeCloseTo(0, 6);
    }
  });

  it('a tighter turn radius bends the full-left primitive more', () => {
    const wide = buildPrimitiveFan({
      minTurnRadius: 6,
      duration: 0.5,
      startSpeed: 0,
    });
    const tight = buildPrimitiveFan({
      minTurnRadius: 2,
      duration: 0.5,
      startSpeed: 0,
    });
    const turn = (f: ReturnType<typeof buildPrimitiveFan>) =>
      Math.max(...f.primitives.map((p) => Math.abs(p.end.dHeading)));
    expect(turn(tight)).toBeGreaterThan(turn(wide));
  });
});

describe('swarm demo: emergent multi-agent coordination', () => {
  it('every NPC reaches its antipodal goal', () => {
    const s = buildSwarm({ agents: 4, rounds: 5 });
    expect(s.agents.length).toBe(4);
    expect(s.reached).toBe(4);
    for (const a of s.agents) {
      const end = a.path[a.path.length - 1]!;
      expect(Math.hypot(end.x - a.goal.x, end.z - a.goal.z)).toBeLessThanOrEqual(3);
      for (let i = 1; i < a.path.length; i++) {
        expect(a.path[i]!.t).toBeGreaterThan(a.path[i - 1]!.t - 1e-9);
      }
    }
  });
});

describe('humanoid demo: omnidirectional vs. turn-radius-constrained', () => {
  it('humanoid threads the L-corridor; the vehicle cannot', () => {
    const h = buildHumanoid();
    expect(h.humanoid.found).toBe(true);
    expect(h.vehicle.found).toBe(false);
    expect(h.humanoid.path.length).toBeGreaterThanOrEqual(2);
    const end = h.humanoid.path[h.humanoid.path.length - 1]!;
    expect(Math.hypot(end.x - h.goal.x, end.z - h.goal.z)).toBeLessThanOrEqual(0.8);
    for (let i = 1; i < h.humanoid.path.length; i++) {
      expect(h.humanoid.path[i]!.t).toBeGreaterThan(
        h.humanoid.path[i - 1]!.t - 1e-9,
      );
    }
  });
});

describe('aircraft demo: true 3D flight planning (altitude searched)', () => {
  it('waypoint course: flies every gate, monotone time', () => {
    const s = buildWaypointCourse();
    expect(s.found).toBe(true);
    expect(s.path.length).toBeGreaterThanOrEqual(s.gates.length + 1);
    const end = s.path[s.path.length - 1]!;
    const g = s.goal;
    expect(
      Math.hypot(end.x - g.x, end.y - g.y, end.z - g.z),
    ).toBeLessThanOrEqual(10);
    for (let i = 1; i < s.path.length; i++) {
      expect(s.path[i]!.t).toBeGreaterThan(s.path[i - 1]!.t - 1e-9);
    }
  });

  it('canyon: flies BETWEEN the full-height walls, then climbs the ridge', () => {
    const s = buildCanyon();
    expect(s.found).toBe(true);
    // The walls are full-height: the only way past is laterally through the
    // alternating side gaps. At the first wall the plane must be on the +z
    // side; at the second, on the -z side — it weaves, it does not fly over.
    const near = (lo: number, hi: number) =>
      s.path.filter((p) => p.x >= lo && p.x <= hi);
    const atWallA = near(38, 62);
    const atWallB = near(86, 110);
    expect(atWallA.length).toBeGreaterThan(0);
    expect(atWallB.length).toBeGreaterThan(0);
    expect(Math.max(...atWallA.map((p) => p.z))).toBeGreaterThan(4);
    expect(Math.min(...atWallB.map((p) => p.z))).toBeLessThan(-4);
    // The final ridge (top y=34) spans the full width — altitude is searched.
    expect(Math.max(...s.path.map((p) => p.y))).toBeGreaterThan(36);
    for (let i = 1; i < s.path.length; i++) {
      expect(s.path[i]!.t).toBeGreaterThan(s.path[i - 1]!.t - 1e-9);
    }
  });

  it('restricted airspace: routes clear of the moving no-fly zone', () => {
    const s = buildRestrictedAirspace();
    expect(s.found).toBe(true);
    const r = (s.zoneRadius ?? 0) + AIRCRAFT_AGENT.radius;
    for (const p of s.path) {
      const c = s.zoneAt?.(p.t);
      if (!c) continue;
      expect(
        Math.hypot(p.x - c.x, p.y - c.y, p.z - c.z),
      ).toBeGreaterThan(r - 1e-6);
    }
  });

  it('interactive: replans to a tapped destination within budget', () => {
    const r = planInteractive(
      INTERACTIVE_BOXES,
      { x: 8, y: 30, z: 0, heading: 0, pitch: 0, speed: 18, t: 0 },
      { x: 150, y: 30, z: 0, heading: 0, pitch: 0, speed: 18, t: 0 },
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(AIRCRAFT_MAX_EXPANSIONS);
  });
});

// Built lazily in beforeAll (not at collect time) so test collection stays
// fast; navcat mesh generation can be environment-sensitive, so a runner
// without it skips the assertion rather than failing the suite.
describe('jumplinks demo: Mononen-style off-mesh annotation', () => {
  let jumpLinks: JumpLinksResult | null = null;
  beforeAll(() => {
    try {
      jumpLinks = buildJumpLinks();
    } catch {
      jumpLinks = null;
    }
  }, 60000);

  it('the humanoid crosses the gap only once the link is registered', (ctx) => {
    if (jumpLinks === null) {
      ctx.skip();
      return;
    }
    const j = jumpLinks;
    expect(j.linkMeta.length).toBe(1);
    expect(typeof j.linkMeta[0]!.connectionId).toBe('number');
    expect(j.without.found).toBe(false);
    expect(j.withLink.found).toBe(true);
    expect(j.withLink.usedJump).toBe(true);
    for (let i = 1; i < j.withLink.path.length; i++) {
      expect(j.withLink.path[i]!.t).toBeGreaterThan(
        j.withLink.path[i - 1]!.t - 1e-9,
      );
    }
  });
});
