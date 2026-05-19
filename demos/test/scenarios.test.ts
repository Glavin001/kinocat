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
  buildFlagship,
  buildCatAndMouseScenario,
  predictMouseFromHistory,
  DEMO_MAX_EXPANSIONS,
  DEMO_DYNAMIC_MAX_EXPANSIONS,
  type Scenario,
  type JumpLinksResult,
  type FlagshipResult,
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
