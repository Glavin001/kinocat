import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { VehicleAgent, VehicleState } from '../../src/agent/types';
import { placeFootprint } from '../../src/internal/geom';

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
      [k / 2, 6],
      [-k / 2, 6],
      [0, -4],
      [k, -4],
      [-k, -4],
    ],
    duration: 0.5,
    substeps: 6,
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

function footprintsClear(world: InMemoryNavWorld, path: VehicleState[]): boolean {
  return path.every((s) =>
    world.footprintClear(placeFootprint(agent.footprint, s.x, s.z, s.heading)),
  );
}

describe('VehicleEnvironment', () => {
  it('plans a curvature-feasible path across open ground', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 40, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 24, z: 6, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(footprintsClear(world, r.path)).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(1.5);
  });

  it('detours around an obstacle that blocks the straight line', () => {
    const obstacle = [
      [12, -3],
      [18, -3],
      [18, 3],
      [12, 3],
    ] as [number, number][];
    const world = new InMemoryNavWorld([rect(1, 0, -10, 30, 10)], [obstacle]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    // straight line is blocked, so a detour is mandatory
    expect(world.segmentClear(start.x, start.z, goal.x, goal.z)).toBe(false);
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 400000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(footprintsClear(world, r.path)).toBe(true);
  });

  it('produces a reverse maneuver in a corridor too narrow to turn around', () => {
    // corridor width 2 < 2*minTurnRadius: turning around is impossible
    const world = new InMemoryNavWorld([rect(1, 0, -1, 30, 1)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 20, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 5, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    const reversed = r.nodes.some(
      (n) =>
        n.edge?.kind === 'drive-reverse' ||
        (n.edge?.kind === 'reeds-shepp' &&
          (n.edge.data as { reverse?: boolean }).reverse === true),
    );
    expect(reversed).toBe(true);
    for (const s of r.path) expect(Math.abs(s.heading)).toBeLessThan(0.35);
    expect(footprintsClear(world, r.path)).toBe(true);
  });
});

describe('VehicleEnvironment Reeds-Shepp analytic expansion', () => {
  it('solves a long open straight in a handful of expansions', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 300, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: {},
    });
    const start: VehicleState = { x: 5, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 280, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 100000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.stats.expansions).toBeLessThan(20); // analytic shot, not 90+ prims
    expect(r.nodes.some((n) => n.edge?.kind === 'reeds-shepp')).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - goal.x, last.z - goal.z)).toBeLessThanOrEqual(1.5);
  });

  it('never shoots through an obstacle (static collision-checked)', () => {
    const obstacle = [
      [12, -3],
      [18, -3],
      [18, 3],
      [12, 3],
    ] as [number, number][];
    const world = new InMemoryNavWorld([rect(1, 0, -10, 30, 10)], [obstacle]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: {},
    });
    const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 400000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // every analytic edge's sampled curve stays out of the obstacle
    for (const n of r.nodes) {
      if (n.edge?.kind !== 'reeds-shepp') continue;
      const samples = (n.edge.data as { samples: [number, number][] }).samples;
      for (const [x, z] of samples) {
        const inObstacle = x > 12 && x < 18 && z > -3 && z < 3;
        expect(inObstacle).toBe(false);
      }
    }
  });

  it('can be disabled (no analytic edges, still solves)', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -20, 60, 20)]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      analyticExpansion: false,
    });
    const r = plan(
      {
        start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
        goal: { x: 50, z: 0, heading: 0, speed: 0, t: 0 },
        environment: env,
        options: { maxExpansions: 300000 },
      },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.nodes.every((n) => n.edge?.kind !== 'reeds-shepp')).toBe(true);
  });
});

// --- Opt 1: clearance broadphase is a pure accelerator -------------------
import type {
  NavWorld,
  PolygonRef,
  OffMeshLink,
} from '../../src/environment/nav-world';

interface Rect {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
}

/** Open world rect minus an optional obstacle rect. `clearanceAt` returns
 *  the EXACT clearance (distance to the world border / obstacle) — a valid
 *  lower bound, so the early-accept is provably sound and the plan must be
 *  byte-identical with the broadphase on or off. */
class StubWorld implements NavWorld {
  readonly revision = 0;
  constructor(
    private readonly world: Rect,
    private readonly obstacle?: Rect,
  ) {}
  private free(x: number, z: number): boolean {
    const w = this.world;
    if (x < w.x0 || x > w.x1 || z < w.z0 || z > w.z1) return false;
    const o = this.obstacle;
    return !(o && x >= o.x0 && x <= o.x1 && z >= o.z0 && z <= o.z1);
  }
  private static distOutside(x: number, z: number, r: Rect): number {
    const dx = Math.max(r.x0 - x, 0, x - r.x1);
    const dz = Math.max(r.z0 - z, 0, z - r.z1);
    return Math.hypot(dx, dz);
  }
  polygonAt(x: number, z: number): PolygonRef | null {
    return this.free(x, z) ? { id: 1, cx: x, cz: z, y: 0 } : null;
  }
  heightAt(): number {
    return 0;
  }
  footprintClear(fp: ReadonlyArray<readonly [number, number]>): boolean {
    for (const [x, z] of fp) if (!this.free(x, z)) return false;
    return true;
  }
  segmentClear(x0: number, z0: number, x1: number, z1: number): boolean {
    for (let i = 0; i <= 8; i++) {
      const u = i / 8;
      if (!this.free(x0 + (x1 - x0) * u, z0 + (z1 - z0) * u)) return false;
    }
    return true;
  }
  offMeshFrom(): ReadonlyArray<OffMeshLink> {
    return [];
  }
  clearanceAt(x: number, z: number): number | null {
    const w = this.world;
    let c = Math.min(x - w.x0, w.x1 - x, z - w.z0, w.z1 - z);
    if (this.obstacle) c = Math.min(c, StubWorld.distOutside(x, z, this.obstacle));
    return c > 0 ? c : 0;
  }
}

describe('VehicleEnvironment clearance broadphase', () => {
  const start: VehicleState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 37, z: 0, heading: 0, speed: 0, t: 0 };
  const WORLD: Rect = { x0: 0, z0: -16, x1: 40, z1: 16 };
  const envOpts = {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: false as const,
  };
  const runPlan = (world: NavWorld, clearanceBroadphase: boolean) =>
    plan(
      {
        start,
        goal,
        environment: new VehicleEnvironment(world, agent, lib, {
          ...envOpts,
          clearanceBroadphase,
        }),
        options: { maxExpansions: 40000 },
      },
      Infinity,
    );

  it('open field: byte-identical plan + expansions on vs off', () => {
    const world = new StubWorld(WORLD);
    const off = runPlan(world, false);
    const on = runPlan(world, true);
    expect(on.found).toBe(true);
    expect(on.found).toBe(off.found);
    expect(on.cost).toBeCloseTo(off.cost, 9);
    expect(on.stats.expansions).toBe(off.stats.expansions);
    expect(on.path.length).toBe(off.path.length);
    for (let i = 0; i < off.path.length; i++) {
      expect(on.path[i]!.x).toBeCloseTo(off.path[i]!.x, 9);
      expect(on.path[i]!.z).toBeCloseTo(off.path[i]!.z, 9);
    }
  });

  it('with an obstacle (gap above): identical & exact check runs near it', () => {
    // Blocks z ≤ 6 at x∈[18,22]; a real gap at z∈(6,16] so a plan exists.
    const obstacle: Rect = { x0: 18, z0: -16, x1: 22, z1: 6 };
    const world = new StubWorld(WORLD, obstacle);
    const off = runPlan(world, false);
    const on = runPlan(world, true);
    expect(on.found).toBe(true);
    expect(on.cost).toBeCloseTo(off.cost, 9);
    expect(on.stats.expansions).toBe(off.stats.expansions);
    expect(on.path.length).toBe(off.path.length);
    // hard constraint preserved: no path state sits inside the obstacle
    for (const s of on.path) {
      const inside =
        s.x >= obstacle.x0 &&
        s.x <= obstacle.x1 &&
        s.z >= obstacle.z0 &&
        s.z <= obstacle.z1;
      expect(inside).toBe(false);
    }
  });

  it('no-ops on a world without clearanceAt (InMemoryNavWorld)', () => {
    const world = new InMemoryNavWorld([rect(1, 0, -16, 40, 16)]);
    const r = plan(
      {
        start,
        goal,
        environment: new VehicleEnvironment(world, agent, lib, {
          ...envOpts,
          clearanceBroadphase: true,
        }),
        options: { maxExpansions: 40000 },
      },
      Infinity,
    );
    expect(r.found).toBe(true);
  });
});
