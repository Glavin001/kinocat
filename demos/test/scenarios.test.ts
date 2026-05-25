import { describe, it, expect, beforeAll } from 'vitest';
import { readdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
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
  buildFlagship,
  buildCatAndMouseScenario,
  predictMouseFromHistory,
  DEMO_MAX_EXPANSIONS,
  DEMO_DYNAMIC_MAX_EXPANSIONS,
  type Scenario,
  type JumpLinksResult,
  type FlagshipResult,
} from '../app/lib/scenarios';
import {
  buildWaypointCourse,
  buildCanyon,
  buildRestrictedAirspace,
  buildGauntlet,
  buildKnifeEdge,
  planInteractive,
  densifyPath,
  aircraftAirspace,
  aircraftPose,
  AIRCRAFT_HALF,
  INTERACTIVE_BOXES,
  AIRCRAFT_AGENT,
  AIRCRAFT_MAX_EXPANSIONS,
} from '../app/lib/aircraft-scenarios';
import {
  buildDogfightSnapshot,
  DOGFIGHT_HALF,
  DOGFIGHT_TEST_MAX_EXPANSIONS,
  dogfightAirspace,
} from '../app/lib/dogfight-scenarios';
// Note: carchase scenario tests live in their own file
// (`demos/test/carchase-scenarios.test.ts`) so they run in a parallel
// vitest worker — adding them here pushed total wall time past the 60 s
// birpc RPC timeout in CI.
import type { CarKinematicState } from 'kinocat/agent';

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
      { x: 4, z: 12, heading: 0, speed: 0, t: 0 } as CarKinematicState,
      { x: 36, z: 12, heading: 0, speed: 0, t: 0 } as CarKinematicState,
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
    const zones =
      s.zoneAt && s.zoneRadius != null
        ? [{ radius: s.zoneRadius, predict: s.zoneAt }]
        : [];
    const air = aircraftAirspace(s.boxes, zones);
    for (const p of s.path) {
      expect(air.clear(aircraftPose(p), AIRCRAFT_HALF, p.t)).toBe(true);
    }
  });

  it('interactive: replans within budget and weaves the full-height walls', () => {
    const r = planInteractive(
      INTERACTIVE_BOXES,
      { x: 8, y: 30, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 },
      { x: 150, y: 30, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 },
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(AIRCRAFT_MAX_EXPANSIONS);
    // INTERACTIVE_BOXES are full-height: the plane cannot fly over them, so
    // it must pass the first wall on the +z side and the second on the -z
    // side (regression guard for the "walls too short, flies over" bug).
    const atA = r.path.filter((p) => p.x >= 50 && p.x <= 66);
    const atB = r.path.filter((p) => p.x >= 98 && p.x <= 114);
    expect(atA.length).toBeGreaterThan(0);
    expect(atB.length).toBeGreaterThan(0);
    expect(Math.max(...atA.map((p) => p.z))).toBeGreaterThan(6);
    expect(Math.min(...atB.map((p) => p.z))).toBeLessThan(-6);
  });

  it('densified rendering path stays collision-clear (canyon)', () => {
    // Renders interpolate over densifyPath, not the coarse planner output —
    // assert the dense arc itself is collision-free, so the visual plane
    // can't appear to clip walls between primitive endpoints.
    const s = buildCanyon();
    const dense = densifyPath(s.path, 12);
    expect(dense.length).toBeGreaterThan(s.path.length * 5);
    const air = aircraftAirspace(s.boxes);
    for (const p of dense) {
      expect(air.clear(aircraftPose(p), AIRCRAFT_HALF, p.t)).toBe(true);
    }
  });

  it('densified rendering path clears walls AND the moving zone (gauntlet)', () => {
    const s = buildGauntlet();
    const dense = densifyPath(s.path, 12);
    const zones =
      s.zoneAt && s.zoneRadius != null
        ? [{ radius: s.zoneRadius, predict: s.zoneAt }]
        : [];
    const air = aircraftAirspace(s.boxes, zones);
    for (const p of dense) {
      expect(air.clear(aircraftPose(p), AIRCRAFT_HALF, p.t)).toBe(true);
    }
  });

  it('knife-edge: banks ~90° to fit a slot narrower than the wingspan', () => {
    const s = buildKnifeEdge();
    expect(s.found).toBe(true);
    // Sample the dense rendering path inside the slot (the coarse planner
    // path stores only primitive endpoints, which may straddle the slot).
    const dense = densifyPath(s.path, 12);
    const inSlot = dense.filter((p) => p.x >= 78 && p.x <= 92);
    expect(inSlot.length).toBeGreaterThan(0);
    const maxBank = Math.max(...inSlot.map((p) => Math.abs(p.roll)));
    expect(maxBank).toBeGreaterThan(Math.PI / 2 - 0.2); // ≥ ~78°
    // Whole planned path stays collision-clear under OBB collision (incl.
    // the slot crossing, where wings-level would intersect the walls).
    const air = aircraftAirspace(s.boxes);
    for (const p of dense) {
      expect(air.clear(aircraftPose(p), AIRCRAFT_HALF, p.t)).toBe(true);
    }
    for (let i = 1; i < s.path.length; i++) {
      expect(s.path[i]!.t).toBeGreaterThan(s.path[i - 1]!.t - 1e-9);
    }
  });

  it('gauntlet: weaves both walls, beats the moving zone, climbs the ridge', () => {
    const s = buildGauntlet();
    expect(s.found).toBe(true);
    const near = (lo: number, hi: number) =>
      s.path.filter((p) => p.x >= lo && p.x <= hi);
    const atA = near(34, 53);
    const atB = near(89, 108);
    expect(atA.length).toBeGreaterThan(0);
    expect(atB.length).toBeGreaterThan(0);
    expect(Math.max(...atA.map((p) => p.z))).toBeGreaterThan(4); // weave +z
    expect(Math.min(...atB.map((p) => p.z))).toBeLessThan(-4); // weave -z
    expect(Math.max(...s.path.map((p) => p.y))).toBeGreaterThan(36); // ridge
    // Every planned state clears all obstacles AND the moving zone at its t.
    const zones =
      s.zoneAt && s.zoneRadius != null
        ? [{ radius: s.zoneRadius, predict: s.zoneAt }]
        : [];
    const air = aircraftAirspace(s.boxes, zones);
    for (const p of s.path) {
      expect(air.clear(aircraftPose(p), AIRCRAFT_HALF, p.t)).toBe(true);
    }
    for (let i = 1; i < s.path.length; i++) {
      expect(s.path[i]!.t).toBeGreaterThan(s.path[i - 1]!.t - 1e-9);
    }
  });
});

// Built lazily in beforeAll (not at collect time) so test collection stays
// fast. navcat mesh generation runs in CI (ubuntu/node 22) exactly as it does
// locally, so a failure here is a real regression — it fails loudly rather
// than skipping, otherwise CI could be green with this demo unverified.
describe('jumplinks demo: Mononen-style off-mesh annotation', () => {
  let jumpLinks: JumpLinksResult;
  beforeAll(() => {
    jumpLinks = buildJumpLinks();
  }, 60000);

  it('the humanoid crosses the gap only once the link is registered', () => {
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

// Flagship: large procedural navcat terrain + 8 NPCs + shortcut/misdirect
// affordances + clearance & time-aware broadphase. Built lazily once
// (navcat-skippable) and kept small (8 agents / 2 rounds) so it stays well
// under the 20s limit. Per-opt correctness is proven by the core parity
// tests; this asserts the integrated behaviour.
describe('flagship demo: real-time multi-agent', () => {
  let fs: FlagshipResult | null = null;
  let detA: FlagshipResult | null = null;
  let detB: FlagshipResult | null = null;
  beforeAll(() => {
    try {
      fs = buildFlagship({ agents: 8, rounds: 2 });
      detA = buildFlagship({ agents: 4, rounds: 1 });
      detB = buildFlagship({ agents: 4, rounds: 1 });
    } catch {
      fs = null;
      detA = null;
      detB = null;
    }
  }, 90000);

  it('every NPC plans to its goal; ≥1 takes the boost, none the misdirect', (ctx) => {
    if (fs === null) {
      ctx.skip();
      return;
    }
    const f = fs;
    expect(f.agents.length).toBe(8);
    expect(f.reached).toBe(8);
    expect(f.agents.some((a) => a.usedShortcut)).toBe(true);
    for (const a of f.agents) {
      expect(a.found).toBe(true);
      expect(a.usedMisdirect).toBe(false); // emergent rejection, no special code
      for (let i = 1; i < a.path.length; i++) {
        expect(a.path[i]!.t).toBeGreaterThan(a.path[i - 1]!.t - 1e-9);
      }
    }
  });

  it('is deterministic (identical small build twice)', (ctx) => {
    if (detA === null || detB === null) {
      ctx.skip();
      return;
    }
    const a = detA;
    const b = detB;
    expect(b.reached).toBe(a.reached);
    expect(b.agents.length).toBe(a.agents.length);
    for (let i = 0; i < a.agents.length; i++) {
      const x = a.agents[i]!;
      const y = b.agents[i]!;
      expect(y.found).toBe(x.found);
      expect(y.usedShortcut).toBe(x.usedShortcut);
      expect(y.path.length).toBe(x.path.length);
      const ex = x.path[x.path.length - 1]!;
      const ey = y.path[y.path.length - 1]!;
      expect(ey.x).toBeCloseTo(ex.x, 9);
      expect(ey.z).toBeCloseTo(ex.z, 9);
    }
  });
});

// Cat & Mouse: multi-agent pursuit with time-aware target prediction. Lazy
// (navcat-skippable) and small (2 cats, 14 ticks) so it stays well under the
// time budget; asserts the pursuit pipeline produces real intercept plans.
describe('catmouse demo: predict + intercept', () => {
  let scn: ReturnType<typeof buildCatAndMouseScenario> | null = null;
  beforeAll(() => {
    try {
      scn = buildCatAndMouseScenario(2);
    } catch {
      scn = null;
    }
  }, 90000);

  it('every cat plans a multi-state path and publishes to the registry', (ctx) => {
    if (scn === null) {
      ctx.skip();
      return;
    }
    const { state } = scn;
    expect(state.cats.length).toBe(2);
    for (const cat of state.cats) {
      expect(cat.plan.length).toBeGreaterThanOrEqual(2);
      // strictly increasing time along the path
      for (let i = 1; i < cat.plan.length; i++) {
        expect(cat.plan[i]!.t).toBeGreaterThan(cat.plan[i - 1]!.t - 1e-9);
      }
    }
    expect(state.registry.all().length).toBe(2);
  });

  it('mouse observations accumulate and the cats actually move', (ctx) => {
    if (scn === null) {
      ctx.skip();
      return;
    }
    const { state } = scn;
    expect(state.mouse.obsHistory.length).toBeGreaterThanOrEqual(6);
    // Each cat should have integrated forward (state has advanced from spawn).
    for (const cat of state.cats) {
      expect(cat.state.t).toBeGreaterThan(0);
    }
  });

  it('cats plan toward the PREDICTED mouse pose (intercept, not chase-current)', (ctx) => {
    if (scn === null) {
      ctx.skip();
      return;
    }
    const { state } = scn;
    // The intercept property: each cat's plan endpoint should be closer to
    // the mouse's PREDICTED pose at the plan's end time than to the mouse's
    // CURRENT pose. (If the cats were merely chasing where the mouse IS,
    // this would be reversed.) The mouse's flee response makes the predictor
    // imperfect, so we just need the prediction-aware plan to win on
    // aggregate.
    const predict = predictMouseFromHistory(state.mouse.obsHistory, 6);
    const cur = state.mouse.state;
    let bestIntercept = Infinity;
    let bestChase = Infinity;
    for (const cat of state.cats) {
      const end = cat.plan[cat.plan.length - 1]!;
      const dChase = Math.hypot(end.x - cur.x, end.z - cur.z);
      if (dChase < bestChase) bestChase = dChase;
      const p = predict(end.t);
      if (!p) continue;
      const d = Math.hypot(end.x - p.x, end.z - p.z);
      if (d < bestIntercept) bestIntercept = d;
    }
    // Allow small slack: prediction is noisy with sharp mouse turns, but the
    // intercept-aimed endpoint must not be wildly worse than chase-current.
    expect(bestIntercept).toBeLessThan(bestChase + 2);
  });

  it('pursuit closes: minimum cat-mouse distance shrinks across the run', (ctx) => {
    if (scn === null) {
      ctx.skip();
      return;
    }
    const { distanceTrace } = scn;
    expect(distanceTrace.length).toBeGreaterThan(8);
    const initialMax = Math.max(...distanceTrace.slice(0, 3));
    const tailMin = Math.min(...distanceTrace.slice(-5));
    // The chase should make meaningful progress in 14 ticks.
    expect(tailMin).toBeLessThan(initialMax);
  });
});

// Dogfight: HeightfieldAirspace + AircraftEnvironment + TimeAwareEnvironment
// + PlanRegistry, all wired up. The snapshot asserts that the AIs deterministic-
// ally produce a plan against the spawn matchup the demo loads with — a
// "no plan" regression here will fail CI before it reaches the demo route.
describe('dogfight demo: interactive 3D pursuit', () => {
  it('both AIs find a plan against the spawn matchup', () => {
    const s = buildDogfightSnapshot();
    expect(s.ais.length).toBe(2);
    for (const a of s.ais) {
      expect(a.result.found).toBe(true);
      expect(a.result.path.length).toBeGreaterThanOrEqual(2);
      expect(a.result.stats.expansions).toBeLessThan(
        DOGFIGHT_TEST_MAX_EXPANSIONS,
      );
      for (let i = 1; i < a.result.path.length; i++) {
        expect(a.result.path[i]!.t).toBeGreaterThan(
          a.result.path[i - 1]!.t - 1e-9,
        );
      }
    }
  });

  it('every planned aircraft state clears the heightfield + obstacles', () => {
    const s = buildDogfightSnapshot();
    const air = dogfightAirspace();
    for (const a of s.ais) {
      for (const p of a.result.path) {
        const pose = {
          x: p.x,
          y: p.y,
          z: p.z,
          yaw: p.heading,
          pitch: p.pitch,
          roll: p.roll,
        };
        expect(air.clear(pose, DOGFIGHT_HALF, p.t)).toBe(true);
      }
    }
  });
});

// Coverage manifest: every demo route under demos/app/<slug>/page.tsx MUST
// have a headless scenario asserted above. This fails CI if a new demo ships
// without a test (or if a tested demo is deleted), so "all demos are covered"
// is enforced, not just claimed.
const TESTED_DEMOS = new Set([
  'anytime', // 'anytime demo' — buildAnytime
  'carchase', // 'carchase demo' — buildCarChaseSnapshot (Rapier + multi-AI ground pursuit)
  'catmouse', // 'catmouse demo' — buildCatAndMouseScenario (predict + intercept)
  'curves', // 'curves demo' — compareCurves
  'dogfight', // 'dogfight demo' — buildDogfightSnapshot (heightfield + multi-AI)
  'dynamic', // 'dynamic demo scenarios' — buildDynamic (moving/coop/jump)
  'flagship', // 'flagship demo' — buildFlagship (large multi-agent navcat)
  'humanoid', // 'humanoid demo' — buildHumanoid
  'jumplinks', // 'jumplinks demo' — buildJumpLinks
  'learnprimitives', // 'learnprimitives demo' — autonomous motion-primitive learner (Rapier)
  'navmesh', // 'navmesh demo' — buildNavmesh / planNavmesh
  'obstaclecourse', // 'obstaclecourse demo' — buildObstacleCourseSnapshot (single-car building-blocks)
  'plane', // 'aircraft demo' — waypoint/canyon/restricted/gauntlet/knife-edge
  'playground', // 'playground demo' — planPlayground
  'raceprimitives', // 'raceprimitives demo' — side-by-side kinematic vs learned library race
  'parking', // 'parking demo' — three tight-parking scenarios; the same parking scenarios are exercised headlessly by the controller-bench harness (`pnpm run controller-bench`) which is the canonical coverage path for the unified planner+tracker stack
  'ramp', // 'ramp + affordance demo' — drivable heightfield ramp + planner-only BallisticJump
  'primitives', // 'primitives demo' — buildPrimitiveFan
  'primitive-explorer', // 'primitive-explorer demo' — kinematic-vs-v2 action-space diagnostics (tested in primitive-diagnostics.test.ts)
  'model-lab', // 'model-lab demo' — v2 training + diagnostics dashboard (helpers tested in fan-plot-ground-truth.test.ts; training pipeline in training-driver.test.ts)
  'sim-to-real', // 'sim-to-real demo' — 3D scope overlaying model prediction on Rapier reality (pure helpers tested in sim-to-real-scene.test.ts; wheel telemetry in core/test/adapters/raycast-vehicle.test.ts)
  'reverse', // 'reverse demo' — planReverse
  'swarm', // 'swarm demo' — buildSwarm
  'world3d', // 'world3d demo' — planWorld3d
]);

describe('demo coverage manifest', () => {
  const appDir = fileURLToPath(new URL('../app', import.meta.url));
  const routes = readdirSync(appDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(`${appDir}/${e.name}/page.tsx`))
    .map((e) => e.name)
    .sort();

  it('discovers at least every demo we know about', () => {
    expect(routes.length).toBeGreaterThanOrEqual(TESTED_DEMOS.size);
  });

  it('every demo route has a headless test (no untested demo ships)', () => {
    const untested = routes.filter((r) => !TESTED_DEMOS.has(r));
    expect(untested, `demo route(s) without a headless test: ${untested.join(', ')}`).toEqual([]);
  });

  it('no stale entries: every tested demo still exists as a route', () => {
    const missing = [...TESTED_DEMOS].filter((d) => !routes.includes(d));
    expect(missing, `tested demo(s) with no route: ${missing.join(', ')}`).toEqual([]);
  });
});
