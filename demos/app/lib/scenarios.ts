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
  HumanoidEnvironment,
  TimeAwareEnvironment,
} from 'kinocat/environment';
import {
  linearObstacle,
  asObstacle,
  PlanRegistry,
  AffordanceRegistry,
  createJumpAffordance,
} from 'kinocat/predict';
import {
  defaultVehicleAgent,
  defaultHumanoidAgent,
  kinematicForwardSim,
} from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import {
  navWorldFromTriangleMesh,
  annotateJumpLinks,
} from 'kinocat/adapters/navcat';
import type { StaticAffordanceMetadata } from 'kinocat/adapters/navcat';
import {
  reedsSheppShortestPath,
  dubinsShortestPath,
  sampleCurve,
} from 'kinocat/curves';
import type { CurvePath } from 'kinocat/curves';
import type {
  VehicleAgent,
  VehicleState,
  HumanoidState,
} from 'kinocat/agent';

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
    heuristicTable: {},
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

// ---------------------------------------------------------------------------
// Curves (Reeds-Shepp vs Dubins) — pure kinocat/curves, no planner

export interface CurveQuery {
  sx: number;
  sz: number;
  sHeading: number;
  gx: number;
  gz: number;
  gHeading: number;
  radius: number;
}

export interface CurveResult {
  word: string;
  length: number;
  segments: { steer: string; gear: 1 | -1; length: number }[];
  /** Sampled world (x,z) polyline. */
  samples: [number, number][];
}

export interface CurveCompare {
  dubins: CurveResult;
  reedsShepp: CurveResult;
}

/** Compare the forward-only Dubins curve with the forward+reverse Reeds-Shepp
 *  curve between two poses. The curves library is plane-agnostic — kinocat
 *  maps world XZ onto the curve's (x, y): y == world z. */
export function compareCurves(q: CurveQuery): CurveCompare {
  const start = { x: q.sx, y: q.sz, theta: q.sHeading };
  const goal = { x: q.gx, y: q.gz, theta: q.gHeading };
  const mk = (path: CurvePath): CurveResult => {
    const poses =
      path.segments.length > 0 ? sampleCurve(start, path, q.radius, 0.2) : [start];
    return {
      word: path.word,
      length: path.length,
      segments: path.segments.map((s) => ({
        steer: s.steer,
        gear: s.gear,
        length: s.length,
      })),
      samples: poses.map((p) => [p.x, p.y] as [number, number]),
    };
  };
  return {
    dubins: mk(dubinsShortestPath(start, goal, q.radius)),
    reedsShepp: mk(reedsSheppShortestPath(start, goal, q.radius)),
  };
}

// ---------------------------------------------------------------------------
// Anytime (IGHA* multi-resolution solution history)

export interface AnytimeStep {
  budget: number;
  found: boolean;
  cost: number;
  expansions: number;
  path: VehicleState[];
}

export interface AnytimeResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  obstacles: { x: number; z: number; hx: number; hz: number }[];
  start: VehicleState;
  goal: VehicleState;
  /** Best plan at each anytime budget, smallest → largest. */
  steps: AnytimeStep[];
}

/** The anytime contract: a serpentine box-maze (analytic shot-to-goal
 *  disabled, so the planner must genuinely search) solved at a sweep of
 *  expansion budgets. A tight budget yields no / a rough plan; a generous one
 *  yields a tighter route. `plan()` returns the best incumbent within budget,
 *  so the NPC always has a usable plan. */
export function buildAnytime(): AnytimeResult {
  const bounds = { x0: 0, z0: -11, x1: 44, z1: 11 };
  const walls = [{ x: 22, z: 2, hx: 3.5, hz: 4.5 }];
  const ring: [number, number][] = [
    [bounds.x0, bounds.z0],
    [bounds.x1, bounds.z0],
    [bounds.x1, bounds.z1],
    [bounds.x0, bounds.z1],
  ];
  const start: VehicleState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 41, z: 0, heading: 0, speed: 0, t: 0 };
  const budgets = [250, 700, 2000, 7000, 25000];
  const steps: AnytimeStep[] = budgets.map((budget) => {
    const world = new InMemoryNavWorld(
      [{ id: 1, y: 0, ring }],
      walls.map((w) => box(w.x, w.z, w.hx, w.hz)),
    );
    const env = new VehicleEnvironment(world, AGENT, LIB, {
      posCell: 0.8,
      headingBuckets: 16,
      speedQuant: 4,
      levelDivisors: [4, 2, 1],
      goalRadius: 2,
      goalHeadingTol: Infinity,
      sweepSegmentCheck: false,
      analyticExpansion: false,
      heuristicTable: {},
    });
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: budget } },
      Infinity,
    );
    return {
      budget,
      found: r.found,
      cost: r.cost,
      expansions: r.stats.expansions,
      path: r.path,
    };
  });
  return { bounds, obstacles: walls, start, goal, steps };
}

// ---------------------------------------------------------------------------
// Reverse maneuvers

export interface ReverseInput {
  reverseCost?: number;
  dirChangePenalty?: number;
}

export interface ReverseSegment {
  from: VehicleState;
  to: VehicleState;
  reverse: boolean;
}

export interface ReverseResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  start: VehicleState;
  goal: VehicleState;
  path: VehicleState[];
  segments: ReverseSegment[];
  found: boolean;
  cost: number;
  reverseCount: number;
}

/** A narrow corridor (too tight for a U-turn at the agent's turn radius) with
 *  the goal *behind* the start at the same heading: the only feasible plan is
 *  a reverse maneuver. IGHA* produces it with no special-case logic. */
export function planReverse(inp: ReverseInput = {}): ReverseResult {
  const bounds = { x0: 0, z0: -2.2, x1: 40, z1: 2.2 };
  const world = new InMemoryNavWorld([
    {
      id: 1,
      y: 0,
      ring: [
        [bounds.x0, bounds.z0],
        [bounds.x1, bounds.z0],
        [bounds.x1, bounds.z1],
        [bounds.x0, bounds.z1],
      ],
    },
  ]);
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
    reverseCostMultiplier: inp.reverseCost ?? 2,
    directionChangePenalty: inp.dirChangePenalty ?? 0.5,
  });
  const env = new VehicleEnvironment(world, agent, LIB, {
    posCell: 0.6,
    headingBuckets: 16,
    speedQuant: 4,
    goalRadius: 1.5,
    goalHeadingTol: Math.PI / 4,
    sweepSegmentCheck: false,
    analyticExpansion: false,
    heuristicTable: {},
  });
  const start: VehicleState = { x: 30, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 8, z: 0, heading: 0, speed: 0, t: 0 };
  const r = plan(
    {
      start,
      goal,
      environment: env,
      options: { maxExpansions: DEMO_DYNAMIC_MAX_EXPANSIONS },
    },
    Infinity,
  );
  const segments: ReverseSegment[] = [];
  for (let i = 1; i < r.nodes.length; i++) {
    const kind = r.nodes[i]!.edge?.kind;
    segments.push({
      from: r.path[i - 1]!,
      to: r.path[i]!,
      reverse: kind === 'drive-reverse',
    });
  }
  return {
    bounds,
    start,
    goal,
    path: r.path,
    segments,
    found: r.found,
    cost: r.cost,
    reverseCount: segments.filter((s) => s.reverse).length,
  };
}

// ---------------------------------------------------------------------------
// Motion-primitive characterization fan

export interface PrimitiveInput {
  minTurnRadius: number;
  duration: number;
  startSpeed: number;
}

export interface PrimitiveFan {
  count: number;
  primitives: {
    id: number;
    reverse: boolean;
    sweep: { x: number; z: number; heading: number }[];
    end: { dx: number; dz: number; dHeading: number; speed: number };
  }[];
}

/** Re-run the characterization harness live: the planner's entire action set
 *  is exactly this swept set of primitives. */
export function buildPrimitiveFan(inp: PrimitiveInput): PrimitiveFan {
  const agent = defaultVehicleAgent({
    minTurnRadius: inp.minTurnRadius,
    maxSpeed: 8,
    maxReverseSpeed: 4,
    footprint: [
      [1.2, 0.6],
      [-1.2, 0.6],
      [-1.2, -0.6],
      [1.2, -0.6],
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
      [0, -4],
      [k, -4],
      [-k, -4],
    ],
    duration: inp.duration,
    substeps: 10,
    startSpeeds: [inp.startSpeed],
  });
  const prims = lib.lookup(inp.startSpeed);
  return {
    count: prims.length,
    primitives: prims.map((p) => ({
      id: p.id,
      reverse: p.reverse,
      sweep: p.sweep.map((s) => ({ x: s.x, z: s.z, heading: s.heading })),
      end: { ...p.end },
    })),
  };
}

// ---------------------------------------------------------------------------
// Swarm — live multi-agent coordination via the plan registry

export interface SwarmInput {
  agents: number;
  rounds?: number;
}

export interface SwarmAgent {
  id: string;
  start: VehicleState;
  goal: VehicleState;
  path: VehicleState[];
  found: boolean;
}

export interface SwarmResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  agents: SwarmAgent[];
  registry: PlanRegistry;
  duration: number;
  rounds: number;
  reached: number;
}

/** N NPCs on a ring, each driving to the antipodal point. Each round every
 *  agent replans treating the others' published plans as moving obstacles,
 *  then republishes. Cooperative avoidance is emergent — no negotiation
 *  protocol. Deterministic (fixed order, expansion-budget search). */
export function buildSwarm(inp: SwarmInput): SwarmResult {
  const n = Math.max(2, Math.min(8, inp.agents));
  const rounds = inp.rounds ?? 5;
  const R = 16;
  const bounds = { x0: -22, z0: -22, x1: 22, z1: 22 };
  const world = new InMemoryNavWorld([
    {
      id: 1,
      y: 0,
      ring: [
        [bounds.x0, bounds.z0],
        [bounds.x1, bounds.z0],
        [bounds.x1, bounds.z1],
        [bounds.x0, bounds.z1],
      ],
    },
  ]);
  const reg = new PlanRegistry();
  const agents: SwarmAgent[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    const sx = R * Math.cos(a);
    const sz = R * Math.sin(a);
    const start: VehicleState = {
      x: sx,
      z: sz,
      heading: Math.atan2(-sz, -sx),
      speed: 0,
      t: 0,
    };
    const goal: VehicleState = {
      x: -sx,
      z: -sz,
      heading: 0,
      speed: 0,
      t: 0,
    };
    agents.push({ id: `A${i}`, start, goal, path: [start], found: false });
  }
  const mkEnv = (others: string[]) =>
    new TimeAwareEnvironment(
      new VehicleEnvironment(world, AGENT, LIB, {
        posCell: 1,
        headingBuckets: 12,
        speedQuant: 4,
        levelDivisors: [4, 2, 1],
        goalRadius: 2,
        goalHeadingTol: Infinity,
        sweepSegmentCheck: false,
        analyticExpansion: false,
        heuristicTable: {},
      }),
      {
        obstacles: others.map((id) => asObstacle(reg.predictNPC(id), 1.8)),
        agentRadius: 1.4,
      },
    );
  for (let r = 0; r < rounds; r++) {
    for (const ag of agents) {
      const others = agents.filter((o) => o.id !== ag.id).map((o) => o.id);
      const res = plan(
        {
          start: ag.start,
          goal: ag.goal,
          environment: mkEnv(others),
          options: { maxExpansions: DEMO_MAX_EXPANSIONS },
        },
        Infinity,
      );
      if (res.found) {
        ag.path = res.path;
        ag.found = true;
        reg.publish(ag.id, res.path);
      }
    }
  }
  const duration = agents.reduce(
    (m, a) => Math.max(m, a.path[a.path.length - 1]?.t ?? 0),
    0.1,
  );
  return {
    bounds,
    agents,
    registry: reg,
    duration,
    rounds,
    reached: agents.filter((a) => a.found).length,
  };
}

// ---------------------------------------------------------------------------
// Humanoid (omnidirectional) vs. vehicle (turn-radius-constrained)

export interface HumanoidResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  corridor: [number, number][][];
  start: { x: number; z: number };
  goal: { x: number; z: number };
  humanoid: { found: boolean; path: HumanoidState[] };
  vehicle: { found: boolean; path: VehicleState[] };
}

/** A tight L-corridor. The omnidirectional humanoid threads it; the same
 *  query for a turn-radius-constrained vehicle has no feasible plan. Both run
 *  the identical IGHA* core — only the Environment differs. */
export function buildHumanoid(): HumanoidResult {
  const bounds = { x0: 0, z0: 0, x1: 20, z1: 20 };
  const horiz: [number, number][] = [
    [0, 0],
    [20, 0],
    [20, 1.5],
    [0, 1.5],
  ];
  const vert: [number, number][] = [
    [18.5, 0],
    [20, 0],
    [20, 20],
    [18.5, 20],
  ];
  const world = new InMemoryNavWorld([
    { id: 1, y: 0, ring: horiz },
    { id: 2, y: 0, ring: vert },
  ]);
  const start = { x: 1, z: 0.75 };
  const goal = { x: 19.25, z: 18 };

  const human = defaultHumanoidAgent({ radius: 0.3, maxSpeed: 4 });
  const hEnv = new HumanoidEnvironment(world, human, {
    posCell: 0.4,
    goalRadius: 0.7,
  });
  const hr = plan(
    {
      start: { ...start, heading: 0, t: 0 },
      goal: { ...goal, heading: 0, t: 0 },
      environment: hEnv,
      options: { maxExpansions: DEMO_DYNAMIC_MAX_EXPANSIONS },
    },
    Infinity,
  );

  const vEnv = new VehicleEnvironment(world, AGENT, LIB, {
    posCell: 0.5,
    headingBuckets: 16,
    speedQuant: 4,
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: false,
    heuristicTable: {},
  });
  const vr = plan(
    {
      start: { ...start, heading: 0, speed: 0, t: 0 },
      goal: { ...goal, heading: 0, speed: 0, t: 0 },
      environment: vEnv,
      options: { maxExpansions: DEMO_MAX_EXPANSIONS },
    },
    Infinity,
  );

  return {
    bounds,
    corridor: [horiz, vert],
    start,
    goal,
    humanoid: { found: hr.found, path: hr.path },
    vehicle: { found: vr.found, path: vr.path },
  };
}

// ---------------------------------------------------------------------------
// Static jump links (Mononen-style navcat off-mesh annotation, humanoid)

/** Two coplanar islands separated by a real (un-meshed) gap. */
export function jumplinksTerrain(): { positions: number[]; indices: number[] } {
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
  quad([0, 0, 0], [9, 0, 0], [9, 0, 9], [0, 0, 9]); // island A
  quad([15, 0, 0], [24, 0, 0], [24, 0, 9], [15, 0, 9]); // island B (gap 9→15)
  return { positions, indices };
}

export interface JumpLinksResult {
  world: ReturnType<typeof navWorldFromTriangleMesh>['world'];
  navMesh: ReturnType<typeof navWorldFromTriangleMesh>['navMesh'];
  positions: number[];
  indices: number[];
  linkMeta: StaticAffordanceMetadata[];
  start: HumanoidState;
  goal: HumanoidState;
  without: { found: boolean };
  withLink: { found: boolean; path: HumanoidState[]; usedJump: boolean };
}

/** Build a navcat navmesh with a gap, register a Mononen-style off-mesh jump
 *  across it with `annotateJumpLinks`, and show the humanoid planner cross the
 *  gap only once the link exists. (VehicleEnvironment does not consume
 *  off-mesh links — HumanoidEnvironment does, so this uses a humanoid.) */
export function buildJumpLinks(): JumpLinksResult {
  const { positions, indices } = jumplinksTerrain();
  const { world, navMesh } = navWorldFromTriangleMesh(
    positions,
    indices,
    { cellSize: 0.3, walkableSlopeAngleDegrees: 45 },
    { horizontalTolerance: 0.6, verticalExtent: 1e4, queryHeight: 0 },
  );
  const human = defaultHumanoidAgent({ radius: 0.3, maxSpeed: 4 });
  const mkEnv = () =>
    new HumanoidEnvironment(world, human, {
      posCell: 0.5,
      goalRadius: 0.9,
      directions: 12,
      stepDuration: 0.3,
      footprintSegments: 6,
    });
  const start: HumanoidState = { x: 2, z: 4.5, heading: 0, t: 0 };
  const goal: HumanoidState = { x: 21, z: 4.5, heading: 0, t: 0 };

  // Without the link the gap is uncrossable; a small budget is enough to show
  // "no plan" (the reachable set on island A is fully explored well within it).
  const before = plan(
    { start, goal, environment: mkEnv(), options: { maxExpansions: 4000 } },
    Infinity,
  );

  const linkMeta = annotateJumpLinks(
    world,
    navMesh,
    [{ from: [7, 4.5], to: [17, 4.5], cost: 2.5, kind: 'jump' }],
    { radius: 0.6 },
  );

  const after = plan(
    { start, goal, environment: mkEnv(), options: { maxExpansions: 30000 } },
    Infinity,
  );
  const usedJump = after.nodes.some((n) => n.edge?.kind === 'jump');

  return {
    world,
    navMesh,
    positions,
    indices,
    linkMeta,
    start,
    goal,
    without: { found: before.found },
    withLink: { found: after.found, path: after.path, usedJump },
  };
}
