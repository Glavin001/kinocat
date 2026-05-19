// Pure, headless-testable demo planning. No React / three imports here so the
// exact configuration each demo runs can be asserted by automated tests
// (demos/test/scenarios.test.ts). Tuned so every scenario reliably finds a
// plan quickly: coarse position cells + light primitive set + no per-sweep
// segment check + an expansion budget (deterministic, not wall-clock).

import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  VehicleEnvironment,
  TimeAwareEnvironment,
} from 'kinocat/environment';
import {
  linearObstacle,
  asObstacle,
  PlanRegistry,
  AffordanceRegistry,
  createJumpAffordance,
} from 'kinocat/predict';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import { navWorldFromTriangleMesh } from 'kinocat/adapters/navcat';
import type { VehicleAgent, VehicleState } from 'kinocat/agent';

export const PALETTE = {
  bg: '#0b0b0f',
  floor: '#161a22',
  obstacle: '#5a2230',
  path: '#44ddff',
  start: '#55ff88',
  goal: '#ffcc33',
  agent: '#7fd6ff',
  ghost: '#9b6cff',
};

/** Interactive budget (playground/world3d/navmesh): replanned on input, so
 *  kept modest. The demo worlds solve well within it. */
export const DEMO_MAX_EXPANSIONS = 40000;

/** Time-aware scenarios add a time dimension and are computed once per
 *  scenario switch (not interactively), so they get a larger budget. */
export const DEMO_DYNAMIC_MAX_EXPANSIONS = 500000;

export function demoVehicle(overrides: Partial<VehicleAgent> = {}) {
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
    ...overrides,
  });
  const k = 1 / agent.minTurnRadius;
  const lib = characterizeVehicle({
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
    substeps: 4,
    startSpeeds: [0],
  });
  return { agent, lib };
}

const { agent: AGENT, lib: LIB } = demoVehicle();
export const DEMO_AGENT = AGENT;

/** Env tuned for speed: coarse cells, no per-sweep segment check, soft goal. */
function vehicleEnv(world: InMemoryNavWorld, reverseCost = 2) {
  return new VehicleEnvironment(
    world,
    { ...AGENT, reverseCostMultiplier: reverseCost },
    LIB,
    {
      posCell: 1,
      headingBuckets: 12,
      speedQuant: 4,
      levelDivisors: [4, 2, 1],
      goalRadius: 2,
      goalHeadingTol: Infinity,
      sweepSegmentCheck: false,
      // RS shot-to-goal: trivial/far queries terminate immediately.
      analyticExpansion: {},
    },
  );
}

function box(x: number, z: number, hx: number, hz: number): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

// ---------------------------------------------------------------------------
// Playground

export interface PlaygroundInput {
  start: VehicleState;
  goal: VehicleState;
  obstacles: { x: number; z: number }[];
  obstacleHalf?: number;
  reverseCost?: number;
  maxExpansions?: number;
  bounds?: { x0: number; z0: number; x1: number; z1: number };
}

export function planPlayground(inp: PlaygroundInput): PlanResult<VehicleState> {
  const b = inp.bounds ?? { x0: 0, z0: -11, x1: 44, z1: 11 };
  const oh = inp.obstacleHalf ?? 2.4;
  const world = new InMemoryNavWorld(
    [
      {
        id: 1,
        y: 0,
        ring: [
          [b.x0, b.z0],
          [b.x1, b.z0],
          [b.x1, b.z1],
          [b.x0, b.z1],
        ],
      },
    ],
    inp.obstacles.map((o) => box(o.x, o.z, oh, oh)),
  );
  const env = vehicleEnv(world, inp.reverseCost ?? 2);
  return plan(
    {
      start: inp.start,
      goal: inp.goal,
      environment: env,
      options: { maxExpansions: inp.maxExpansions ?? DEMO_MAX_EXPANSIONS },
    },
    Infinity,
  );
}

// ---------------------------------------------------------------------------
// Dynamic (time-aware + multi-agent)

export type Scenario = 'moving' | 'coop' | 'jump';

export interface DynamicScene {
  scenario: Scenario;
  bounds: { x0: number; z0: number; x1: number; z1: number };
  islands: [number, number, number, number][];
  result: PlanResult<VehicleState>;
  duration: number;
  start: VehicleState;
  goal: VehicleState;
  ghostAt?: (t: number) => { x: number; z: number } | null;
  affordanceHop?: [VehicleState, VehicleState] | null;
  info: string;
}

function rectPoly(id: number, x0: number, z0: number, x1: number, z1: number) {
  return {
    id,
    y: 0,
    ring: [
      [x0, z0],
      [x1, z0],
      [x1, z1],
      [x0, z1],
    ] as [number, number][],
  };
}

// The dynamic scenarios use the FINER env + geometry proven by the core M5
// tests (which reliably solve coop/jump), since they are computed once per
// scenario switch rather than on every interaction.
function dynVehicle() {
  const agent = defaultVehicleAgent({
    minTurnRadius: 3,
    maxSpeed: 8,
    maxReverseSpeed: 4,
    footprint: [
      [1.0, 0.5],
      [-1.0, 0.5],
      [-1.0, -0.5],
      [1.0, -0.5],
    ],
  });
  const k = 1 / agent.minTurnRadius;
  const lib = characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 6],
      [k, 6],
      [-k, 6],
      [k / 2, 6],
      [-k / 2, 6],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
  return { agent, lib };
}
const { agent: DYN_AGENT, lib: DYN_LIB } = dynVehicle();

function dynEnv(world: InMemoryNavWorld) {
  return new VehicleEnvironment(world, DYN_AGENT, DYN_LIB, {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
    // The time-aware wrapper only collision-checks successor endpoints, so a
    // statically-clear analytic curve could clip a moving obstacle mid-shot.
    analyticExpansion: false,
  });
}

export function buildDynamic(scn: Scenario): DynamicScene {
  const start: VehicleState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const opts = { maxExpansions: DEMO_DYNAMIC_MAX_EXPANSIONS };

  if (scn === 'moving') {
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    const world = new InMemoryNavWorld([rectPoly(1, 0, -14, 32, 14)]);
    const obstacle = linearObstacle(15, -12, 0, 4, 2.5, 0, 60);
    const env = new TimeAwareEnvironment(dynEnv(world), {
      obstacles: [obstacle],
      agentRadius: 1.4,
    });
    const r = plan({ start, goal, environment: env, options: opts }, Infinity);
    const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
    return {
      scenario: scn,
      bounds: { x0: 0, z0: -14, x1: 32, z1: 14 },
      islands: [[0, -14, 32, 14]],
      result: r,
      duration: dur,
      start,
      goal,
      ghostAt: (t) => obstacle.predict(t),
      info: r.found
        ? `avoids a linearly-moving obstacle; ${dur.toFixed(1)} s, ${r.path.length} states`
        : 'no plan',
    };
  }

  if (scn === 'coop') {
    const goal: VehicleState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
    const world = new InMemoryNavWorld([rectPoly(1, 0, -14, 32, 14)]);
    const reg = new PlanRegistry();
    // NPC A holds the centre of the corridor; NPC B must route around it.
    reg.publish('A', [
      { x: 15, z: 0, heading: 0, speed: 0, t: 0 },
      { x: 15, z: 0, heading: 0, speed: 0, t: 1000 },
    ]);
    const env = new TimeAwareEnvironment(dynEnv(world), {
      obstacles: [asObstacle(reg.predictNPC('A'), 2.5)],
      agentRadius: 1.4,
    });
    const r = plan({ start, goal, environment: env, options: opts }, Infinity);
    const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
    return {
      scenario: scn,
      bounds: { x0: 0, z0: -14, x1: 32, z1: 14 },
      islands: [[0, -14, 32, 14]],
      result: r,
      duration: Math.max(dur, 0.1),
      start,
      goal,
      ghostAt: (t) => {
        const p = reg.predictNPC('A')(t);
        return p ? { x: p.x, z: p.z } : null;
      },
      info: r.found
        ? 'NPC B reads NPC A from the plan registry and routes around it'
        : 'no plan',
    };
  }

  // jump
  const goal: VehicleState = { x: 34, z: 0, heading: 0, speed: 0, t: 0 };
  const world = new InMemoryNavWorld(
    [rectPoly(1, 0, -6, 14, 6), rectPoly(2, 22, -6, 40, 6)],
    [],
  );
  const reg = new AffordanceRegistry();
  reg.add(
    createJumpAffordance({
      id: 'gap',
      launch: { x: 12, z: 0 },
      entryRadius: 3,
      land: { x: 25, z: 0, heading: 0, speed: 0, t: 0 },
      duration: 1,
      cost: 1.5,
    }),
  );
  const env = new TimeAwareEnvironment(dynEnv(world), {
    affordances: reg,
    affordanceRadius: 12,
  });
  const r = plan({ start, goal, environment: env, options: opts }, Infinity);
  const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
  let hop: [VehicleState, VehicleState] | null = null;
  const hi = r.nodes.findIndex((n) => n.edge?.kind === 'affordance');
  if (hi > 0) hop = [r.path[hi - 1]!, r.path[hi]!];
  return {
    scenario: scn,
    bounds: { x0: 0, z0: -6, x1: 40, z1: 6 },
    islands: [
      [0, -6, 14, 6],
      [22, -6, 40, 6],
    ],
    result: r,
    duration: dur,
    start,
    goal,
    affordanceHop: hop,
    info: r.found
      ? 'drive primitives cannot cross the gap — a jump affordance is used'
      : 'no plan',
  };
}

// ---------------------------------------------------------------------------
// World3D (in-memory box obstacle)

export const WORLD3D_OBSTACLE: [number, number, number, number] = [17, -3, 23, 3];
export const WORLD3D_BOUNDS = { x0: 0, z0: -12, x1: 40, z1: 12 };

export interface BoxObstacle {
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export const WORLD3D_DEFAULT_OBSTACLES: BoxObstacle[] = [
  { x: 20, z: 0, hx: 3, hz: 3 },
];

/** World3D world from an arbitrary list of axis-aligned box obstacles. */
export function world3dWorldFrom(obstacles: BoxObstacle[]): InMemoryNavWorld {
  return new InMemoryNavWorld(
    [{ id: 1, y: 0, ring: [[0, -12], [40, -12], [40, 12], [0, 12]] }],
    obstacles.map((o) => box(o.x, o.z, o.hx, o.hz)),
  );
}

export function world3dWorld(): InMemoryNavWorld {
  return world3dWorldFrom(WORLD3D_DEFAULT_OBSTACLES);
}

export function planWorld3d(
  world: InMemoryNavWorld,
  start: VehicleState,
  goal: VehicleState,
): PlanResult<VehicleState> {
  return plan(
    {
      start: { ...start, t: 0 },
      goal,
      environment: vehicleEnv(world),
      options: { maxExpansions: DEMO_MAX_EXPANSIONS },
    },
    Infinity,
  );
}

// ---------------------------------------------------------------------------
// NavMesh (real navcat navmesh: ground -> ramp -> raised platform)

/** Ground (y=0, x∈[0,20]) -> ramp (x∈[20,26], y 0→4) -> platform (y=4). */
export function navmeshTerrain(): { positions: number[]; indices: number[] } {
  const positions: number[] = [];
  const indices: number[] = [];
  const quad = (
    c0: [number, number, number],
    c1: [number, number, number],
    c2: [number, number, number],
    c3: [number, number, number],
  ) => {
    const b = positions.length / 3;
    positions.push(...c0, ...c1, ...c2, ...c3);
    indices.push(b, b + 3, b + 2, b, b + 2, b + 1);
  };
  quad([0, 0, 0], [20, 0, 0], [20, 0, 24], [0, 0, 24]);
  quad([20, 0, 0], [26, 4, 0], [26, 4, 24], [20, 0, 24]);
  quad([26, 4, 0], [40, 4, 0], [40, 4, 24], [26, 4, 24]);
  return { positions, indices };
}

export function buildNavmesh() {
  const { positions, indices } = navmeshTerrain();
  const { world, navMesh } = navWorldFromTriangleMesh(
    positions,
    indices,
    { cellSize: 0.3, walkableSlopeAngleDegrees: 50 },
    { horizontalTolerance: 0.7, verticalExtent: 1e4, queryHeight: 0 },
  );
  return { world, navMesh, positions, indices };
}

export function planNavmesh(
  world: ReturnType<typeof buildNavmesh>['world'],
  start: VehicleState,
  goal: VehicleState,
): PlanResult<VehicleState> {
  const env = new VehicleEnvironment(world, AGENT, LIB, {
    posCell: 1,
    headingBuckets: 12,
    speedQuant: 4,
    goalRadius: 2,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: {},
  });
  return plan(
    { start: { ...start, t: 0 }, goal, environment: env, options: { maxExpansions: DEMO_MAX_EXPANSIONS } },
    Infinity,
  );
}
