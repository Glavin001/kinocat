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
  createBoostAffordance,
  createMisdirectAffordance,
} from 'kinocat/predict';
import type { Predict } from 'kinocat/predict';
import { planPoseAt, purePursuit, ReplanState } from 'kinocat/execute';
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
  CarKinematicState,
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
  start: CarKinematicState;
  goal: CarKinematicState;
  obstacles: { x: number; z: number }[];
  obstacleHalf?: number;
  reverseCost?: number;
  maxExpansions?: number;
  bounds?: { x0: number; z0: number; x1: number; z1: number };
}

export function planPlayground(inp: PlaygroundInput): PlanResult<CarKinematicState> {
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
  result: PlanResult<CarKinematicState>;
  duration: number;
  start: CarKinematicState;
  goal: CarKinematicState;
  ghostAt?: (t: number) => { x: number; z: number } | null;
  affordanceHop?: [CarKinematicState, CarKinematicState] | null;
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
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const opts = { maxExpansions: DEMO_DYNAMIC_MAX_EXPANSIONS };

  if (scn === 'moving') {
    const goal: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
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
    const goal: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
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
  const goal: CarKinematicState = { x: 34, z: 0, heading: 0, speed: 0, t: 0 };
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
  let hop: [CarKinematicState, CarKinematicState] | null = null;
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
  start: CarKinematicState,
  goal: CarKinematicState,
): PlanResult<CarKinematicState> {
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
  start: CarKinematicState,
  goal: CarKinematicState,
): PlanResult<CarKinematicState> {
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
  path: CarKinematicState[];
}

export interface AnytimeResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  obstacles: { x: number; z: number; hx: number; hz: number }[];
  start: CarKinematicState;
  goal: CarKinematicState;
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
  const start: CarKinematicState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 41, z: 0, heading: 0, speed: 0, t: 0 };
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
  from: CarKinematicState;
  to: CarKinematicState;
  reverse: boolean;
}

export interface ReverseResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  start: CarKinematicState;
  goal: CarKinematicState;
  path: CarKinematicState[];
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
  const start: CarKinematicState = { x: 30, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 8, z: 0, heading: 0, speed: 0, t: 0 };
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
  start: CarKinematicState;
  goal: CarKinematicState;
  path: CarKinematicState[];
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
    const start: CarKinematicState = {
      x: sx,
      z: sz,
      heading: Math.atan2(-sz, -sx),
      speed: 0,
      t: 0,
    };
    const goal: CarKinematicState = {
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
  vehicle: { found: boolean; path: CarKinematicState[] };
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

// ---------------------------------------------------------------------------
// Flagship — large procedural terrain, many NPCs, shortcut + misdirect
// affordances, staggered round-robin (main-thread) replanning. Exercises all
// three opt-in perf optimizations together.

const FLAGSHIP_W = 60;
const FLAGSHIP_D = 36;
// Canyon wall x∈[30,36] for z≥11 (a real obstacle); open corridor at z<11 is
// the long way round. A couple of pillars make the terrain non-trivial.
const FLAGSHIP_PILLARS: [number, number, number, number][] = [
  [44, 13, 48, 17],
  [16, 24, 20, 28],
];
function flagshipBlocked(x: number, z: number): boolean {
  if (x >= 30 && x < 36 && z >= 11) return true; // canyon wall
  for (const [x0, z0, x1, z1] of FLAGSHIP_PILLARS) {
    if (x >= x0 && x < x1 && z >= z0 && z < z1) return true;
  }
  return false;
}

/** Deterministic procedural terrain: a gently-undulating ground tessellated
 *  into 2-unit quads, with the canyon wall and pillars carved out (omitted
 *  quads ⇒ real navmesh holes). */
export function flagshipTerrain(): {
  positions: number[];
  indices: number[];
  bounds: { x0: number; z0: number; x1: number; z1: number };
} {
  const positions: number[] = [];
  const indices: number[] = [];
  const q = 2;
  const h = (x: number, z: number) =>
    0.5 * Math.sin(x * 0.07) + 0.4 * Math.cos(z * 0.09);
  for (let z = 0; z < FLAGSHIP_D; z += q) {
    for (let x = 0; x < FLAGSHIP_W; x += q) {
      if (flagshipBlocked(x + q / 2, z + q / 2)) continue;
      const b = positions.length / 3;
      positions.push(
        x, h(x, z), z,
        x + q, h(x + q, z), z,
        x + q, h(x + q, z + q), z + q,
        x, h(x, z + q), z + q,
      );
      indices.push(b, b + 3, b + 2, b, b + 2, b + 1);
    }
  }
  return { positions, indices, bounds: { x0: 0, z0: 0, x1: FLAGSHIP_W, z1: FLAGSHIP_D } };
}

export interface FlagshipHazard {
  x: number;
  z: number;
  r: number;
}

export interface FlagshipInput {
  agents?: number;
  rounds?: number;
  clearanceBroadphase?: boolean;
  /**
   * Off by default and intentionally so: the obstacle-aware grid heuristic
   * (Opt 2) is admissible only for the pure navmesh-geodesic problem. A
   * cost-reducing affordance shortcut makes it OVER-estimate true
   * cost-to-go, which is inadmissible; the flagship mixes affordances, so it
   * does not combine the two. Opt 2 stays first-class & benchmarked for its
   * valid domain (obstacle planning without cost-reducing shortcuts).
   */
  gridHeuristic?: boolean;
  timeBroadphase?: boolean;
  /**
   * Richer interactive scenario: opposing cross-traffic start/goal pairs plus
   * two extra affordances (a second boost + a canyon jump). Off by default so
   * the headless scenario test exercises the exact stable lane config.
   */
  crossTraffic?: boolean;
  /** Per-agent goal replacement (click-to-retarget in the UI). */
  goalOverrides?: Record<string, { x: number; z: number }>;
  /** Stationary circular danger zones the planner must route around (a
   *  click-placed obstacle in the UI). */
  hazards?: FlagshipHazard[];
}

export interface FlagshipAgent {
  id: string;
  start: CarKinematicState;
  goal: CarKinematicState;
  path: CarKinematicState[];
  found: boolean;
  usedShortcut: boolean;
  usedMisdirect: boolean;
  usedJump: boolean;
}

export interface FlagshipAffordanceView {
  id: string;
  launch: { x: number; z: number };
  land: { x: number; z: number };
}

export interface FlagshipWorld {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  positions: number[];
  indices: number[];
  world: ReturnType<typeof navWorldFromTriangleMesh>['world'];
  affordances: AffordanceRegistry;
  shortcuts: FlagshipAffordanceView[];
  misdirects: FlagshipAffordanceView[];
  jumps: FlagshipAffordanceView[];
  boostIds: Set<string>;
  decoyIds: Set<string>;
  jumpIds: Set<string>;
}

export interface FlagshipResult {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  positions: number[];
  indices: number[];
  shortcuts: FlagshipAffordanceView[];
  misdirects: FlagshipAffordanceView[];
  jumps: FlagshipAffordanceView[];
  hazards: FlagshipHazard[];
  agents: FlagshipAgent[];
  registry: PlanRegistry;
  affordances: AffordanceRegistry;
  rounds: number;
  reached: number;
  duration: number;
}

const FLAGSHIP_ROUND_DT = 2.5;

/**
 * Build the (expensive) flagship navmesh + affordance registry ONCE. The
 * terrain/affordances never change across interactive re-solves — only the
 * agents' goals and the dynamic hazards do — so the UI builds this a single
 * time and calls solveFlagship() per interaction (no navmesh regen).
 *
 * `rich` adds the cross-traffic-only extra affordances; the default
 * (rich=false) is byte-identical to the original stable lane scenario so the
 * headless scenario test is unaffected.
 */
export function buildFlagshipWorld(rich = false): FlagshipWorld {
  const { positions, indices, bounds } = flagshipTerrain();
  const { world } = navWorldFromTriangleMesh(
    positions,
    indices,
    { cellSize: 0.5, walkableSlopeAngleDegrees: 55, walkableClimbWorld: 1 },
    { clearanceField: true, horizontalTolerance: 0.6 },
  );

  const aff = new AffordanceRegistry();
  const shortcuts: FlagshipAffordanceView[] = [];
  const misdirects: FlagshipAffordanceView[] = [];
  const jumps: FlagshipAffordanceView[] = [];
  const boostIds = new Set<string>();
  const decoyIds = new Set<string>();
  const jumpIds = new Set<string>();

  const boost = createBoostAffordance({
    id: 'boost-canyon',
    pad: { x: 28, z: 24 },
    entryRadius: 5,
    exit: { x: 40, z: 24, heading: 0, speed: 0, t: 0 },
    duration: 1,
    cost: 1.2, // genuine: far cheaper than the long bottom detour
  });
  aff.add(boost);
  boostIds.add(boost.id);
  shortcuts.push({ id: boost.id, launch: { x: 28, z: 24 }, land: { x: 40, z: 24 } });

  const decoy = createMisdirectAffordance({
    id: 'decoy-pocket',
    launch: { x: 28, z: 16 }, // also near the wall — tempting
    entryRadius: 5,
    land: { x: 20, z: 30, heading: 0, speed: 0, t: 0 }, // back-side pocket
    duration: 1,
    cost: 60, // honest & high ⇒ planner rejects it on its own
  });
  aff.add(decoy);
  decoyIds.add(decoy.id);
  misdirects.push({ id: decoy.id, launch: { x: 28, z: 16 }, land: { x: 20, z: 30 } });

  if (rich) {
    const boost2 = createBoostAffordance({
      id: 'boost-return',
      pad: { x: 36, z: 6 },
      entryRadius: 5,
      exit: { x: 20, z: 6, heading: Math.PI, speed: 0, t: 0 },
      duration: 1,
      cost: 1.2, // genuine shortcut for the right→left cross-traffic
    });
    aff.add(boost2);
    boostIds.add(boost2.id);
    shortcuts.push({ id: boost2.id, launch: { x: 36, z: 6 }, land: { x: 20, z: 6 } });

    const jump = createJumpAffordance({
      id: 'jump-wall',
      launch: { x: 33, z: 8 },
      entryRadius: 4,
      land: { x: 33, z: 22, heading: Math.PI / 2, speed: 0, t: 0 },
      apexY: 3,
      duration: 1.1,
      cost: 4, // genuine: hops the canyon wall instead of the long way
    });
    aff.add(jump);
    jumpIds.add(jump.id);
    jumps.push({ id: jump.id, launch: { x: 33, z: 8 }, land: { x: 33, z: 22 } });
  }

  return {
    bounds,
    positions,
    indices,
    world,
    affordances: aff,
    shortcuts,
    misdirects,
    jumps,
    boostIds,
    decoyIds,
    jumpIds,
  };
}

/** Default start/goal pairs. `crossTraffic` spreads opposing diagonal
 *  traffic (rich interactive scene); otherwise the stable left→right lanes
 *  the headless test pins. Deterministic — no RNG, no clock. */
function flagshipAgentPairs(
  n: number,
  crossTraffic: boolean,
): { id: string; start: CarKinematicState; goal: CarKinematicState }[] {
  const span = FLAGSHIP_D - 12;
  const pairs: { id: string; start: CarKinematicState; goal: CarKinematicState }[] = [];
  for (let i = 0; i < n; i++) {
    const z = 6 + (i / (n - 1)) * span;
    if (!crossTraffic) {
      pairs.push({
        id: `V${i}`,
        start: { x: 4, z, heading: 0, speed: 0, t: 0 },
        goal: { x: FLAGSHIP_W - 4, z, heading: 0, speed: 0, t: 0 },
      });
      continue;
    }
    const zMirror = 6 + ((n - 1 - i) / (n - 1)) * span;
    const eastbound = i % 2 === 0;
    pairs.push({
      id: `V${i}`,
      start: eastbound
        ? { x: 4, z, heading: 0, speed: 0, t: 0 }
        : { x: FLAGSHIP_W - 4, z, heading: Math.PI, speed: 0, t: 0 },
      goal: eastbound
        ? { x: FLAGSHIP_W - 4, z: zMirror, heading: 0, speed: 0, t: 0 }
        : { x: 4, z: zMirror, heading: Math.PI, speed: 0, t: 0 },
    });
  }
  return pairs;
}

/**
 * Run the staggered round-robin multi-agent solve over a prebuilt world.
 * Fast (no navmesh regen) so the UI can call it on every interaction.
 */
export function solveFlagship(
  fw: FlagshipWorld,
  inp: FlagshipInput = {},
): FlagshipResult {
  const n = Math.max(8, Math.min(12, inp.agents ?? 8));
  const rounds = inp.rounds ?? 6;
  const cb = inp.clearanceBroadphase ?? true;
  const gh = inp.gridHeuristic ?? false; // see FlagshipInput.gridHeuristic
  const tb = inp.timeBroadphase ?? true;
  const crossTraffic = inp.crossTraffic ?? false;
  const overrides = inp.goalOverrides ?? {};
  const hazards = inp.hazards ?? [];
  const { world } = fw;

  const reg = new PlanRegistry();
  const hazardObstacles = hazards.map((h) =>
    linearObstacle(h.x, h.z, 0, 0, h.r, 0, 1e6),
  );

  const agents: FlagshipAgent[] = flagshipAgentPairs(n, crossTraffic).map(
    (p) => {
      const ov = overrides[p.id];
      const goal: CarKinematicState = ov
        ? { x: ov.x, z: ov.z, heading: 0, speed: 0, t: 0 }
        : p.goal;
      return {
        id: p.id,
        start: p.start,
        goal,
        path: [p.start],
        found: false,
        usedShortcut: false,
        usedMisdirect: false,
        usedJump: false,
      };
    },
  );

  const mkEnv = (selfId: string) =>
    new TimeAwareEnvironment(
      new VehicleEnvironment(world, AGENT, LIB, {
        posCell: 0.8,
        headingBuckets: 16,
        speedQuant: 4,
        levelDivisors: [4, 2, 1],
        goalRadius: 2,
        goalHeadingTol: Infinity,
        sweepSegmentCheck: false,
        // Real-time game-NPC profile (spec §4.1): the Reeds-Shepp shot-to-goal
        // keeps each replan to a handful of expansions; best-effort + frequent
        // replanning corrects the static-only analytic curve vs other agents.
        analyticExpansion: {},
        heuristicTable: {},
        clearanceBroadphase: cb,
        gridHeuristic: gh ? {} : false,
      }),
      {
        obstacles: [
          ...agents
            .filter((o) => o.id !== selfId)
            .map((o) => asObstacle(reg.predictNPC(o.id), 1.8)),
          ...hazardObstacles,
        ],
        agentRadius: 1.4,
        affordances: fw.affordances,
        affordanceRadius: 14,
        broadphase: tb ? {} : false,
      },
    );

  const cur: CarKinematicState[] = agents.map((a) => ({ ...a.start }));
  const arrived: boolean[] = agents.map(() => false);
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < agents.length; i++) {
      if (arrived[i]) continue;
      const ag = agents[i]!;
      const res = plan(
        {
          start: cur[i]!,
          goal: ag.goal,
          environment: mkEnv(ag.id),
          options: { maxExpansions: DEMO_MAX_EXPANSIONS },
        },
        Infinity,
      );
      if (!res.found) continue;
      ag.found = true;
      ag.path = res.path;
      reg.publish(ag.id, res.path);
      for (const node of res.nodes) {
        if (node.edge?.kind !== 'affordance') continue;
        const id = (node.edge.data as { affordanceId?: string }).affordanceId;
        if (id && fw.boostIds.has(id)) ag.usedShortcut = true;
        if (id && fw.decoyIds.has(id)) ag.usedMisdirect = true;
        if (id && fw.jumpIds.has(id)) ag.usedJump = true;
      }
      const next = planPoseAt(res.path, (cur[i]!.t ?? 0) + FLAGSHIP_ROUND_DT);
      if (next) cur[i] = next;
      const last = res.path[res.path.length - 1]!;
      if (Math.hypot(cur[i]!.x - last.x, cur[i]!.z - last.z) <= 2.5) {
        arrived[i] = true;
      }
    }
  }

  const reached = agents.filter((a) => {
    if (!a.found) return false;
    const e = a.path[a.path.length - 1]!;
    return Math.hypot(e.x - a.goal.x, e.z - a.goal.z) <= 2.5;
  }).length;
  const duration = agents.reduce(
    (m, a) => Math.max(m, a.path[a.path.length - 1]?.t ?? 0),
    0.1,
  );

  return {
    bounds: fw.bounds,
    positions: fw.positions,
    indices: fw.indices,
    shortcuts: fw.shortcuts,
    misdirects: fw.misdirects,
    jumps: fw.jumps,
    hazards,
    agents,
    registry: reg,
    affordances: fw.affordances,
    rounds,
    reached,
    duration,
  };
}

/** Headless one-shot: build the world (rich iff crossTraffic) then solve.
 *  The scenario test calls this with the default (stable) config. */
export function buildFlagship(inp: FlagshipInput = {}): FlagshipResult {
  return solveFlagship(buildFlagshipWorld(inp.crossTraffic ?? false), inp);
}

// ---------------------------------------------------------------------------
// Cat & Mouse pursuit — time-aware target prediction + multi-cat coordination
// + opportunistic affordances. Cats observe the mouse, build a Predict<{x,z}>
// from its motion history, and plan to the INTERCEPTION pose (mouse position
// at predicted arrival time), not the mouse's current position. Cats share
// plans via PlanRegistry → emergent flanking. Mouse is a non-cooperative
// humanoid darting around with simple wander+flee AI; it is NOT in the
// registry. Visually compelling pursuit you can leave at defaults.

const CATMOUSE_W = 56;
const CATMOUSE_D = 56;

/** Vertical canyon down the upper half of the arena (un-meshed strip ⇒ real
 *  navmesh holes). Forces detour OR jump affordance when chase crosses it. */
function catMouseBlocked(x: number, z: number): boolean {
  if (x >= 27 && x < 31 && z >= 20) return true;
  return false;
}

/** Procedural terrain: undulating ground tessellated into 2-unit quads with
 *  the canyon strip carved out. Mirrors flagshipTerrain's pattern. */
export function catMouseTerrain(): {
  positions: number[];
  indices: number[];
  bounds: { x0: number; z0: number; x1: number; z1: number };
} {
  const positions: number[] = [];
  const indices: number[] = [];
  const q = 2;
  const h = (x: number, z: number) =>
    0.35 * Math.sin(x * 0.08) + 0.3 * Math.cos(z * 0.1);
  for (let z = 0; z < CATMOUSE_D; z += q) {
    for (let x = 0; x < CATMOUSE_W; x += q) {
      if (catMouseBlocked(x + q / 2, z + q / 2)) continue;
      const b = positions.length / 3;
      positions.push(
        x, h(x, z), z,
        x + q, h(x + q, z), z,
        x + q, h(x + q, z + q), z + q,
        x, h(x, z + q), z + q,
      );
      indices.push(b, b + 3, b + 2, b, b + 2, b + 1);
    }
  }
  return {
    positions,
    indices,
    bounds: { x0: 0, z0: 0, x1: CATMOUSE_W, z1: CATMOUSE_D },
  };
}

// Cats are vehicles — turning-radius-constrained Reeds-Shepp curves look
// visually cat-like. Slightly faster than the mouse so capture is achievable
// but not trivial; the chase is winnable.
const CAT_MAX_SPEED = 9;
const CAT_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 2.5,
  maxSpeed: CAT_MAX_SPEED,
  maxReverseSpeed: 4,
  footprint: [
    [1.0, 0.5],
    [-1.0, 0.5],
    [-1.0, -0.5],
    [1.0, -0.5],
  ],
});
const CAT_K = 1 / CAT_AGENT.minTurnRadius;
const CAT_LIB = characterizeVehicle({
  forwardSim: kinematicForwardSim(CAT_AGENT),
  controlSets: [
    [0, 6],
    [CAT_K, 6],
    [-CAT_K, 6],
    [CAT_K / 2, 6],
    [-CAT_K / 2, 6],
    [0, -3],
    [CAT_K, -3],
    [-CAT_K, -3],
  ],
  duration: 0.5,
  substeps: 6,
  startSpeeds: [0],
});
const CAT_FORWARD_SIM = kinematicForwardSim(CAT_AGENT);

const MOUSE_MAX_SPEED = 5;

/** Pure-pursuit config tuned for the cat (small turn radius, brisk speeds). */
const CAT_PP_CFG = {
  lookaheadMin: 1.5,
  lookaheadGain: 0.35,
  lookaheadMax: 4.5,
  maxLateralAccel: 8,
  maxAccel: 8,
  maxDecel: 8,
  cruiseSpeed: CAT_MAX_SPEED,
  goalTolerance: 1.2,
  minTurnRadius: CAT_AGENT.minTurnRadius,
};

function catEnv(world: ReturnType<typeof navWorldFromTriangleMesh>['world']) {
  return new VehicleEnvironment(world, CAT_AGENT, CAT_LIB, {
    posCell: 0.8,
    headingBuckets: 16,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: {},
    heuristicTable: {},
    clearanceBroadphase: true,
  });
}

export interface CatMouseBoostView {
  id: string;
  pad: { x: number; z: number };
  exit: { x: number; z: number };
  r: number;
}

export interface CatMouseJumpView {
  id: string;
  launch: { x: number; z: number };
  land: { x: number; z: number };
  r: number;
}

export interface CatMouseWorld {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  positions: number[];
  indices: number[];
  world: ReturnType<typeof navWorldFromTriangleMesh>['world'];
  affordances: AffordanceRegistry;
  boosts: CatMouseBoostView[];
  jumps: CatMouseJumpView[];
  boostIds: Set<string>;
  jumpIds: Set<string>;
}

/** Build the (expensive) navmesh + affordance set ONCE. The UI holds this in
 *  a ref across re-inits (cat count slider re-inits only the sim state). */
export function buildCatAndMouseWorld(): CatMouseWorld {
  const { positions, indices, bounds } = catMouseTerrain();
  const { world } = navWorldFromTriangleMesh(
    positions,
    indices,
    { cellSize: 0.5, walkableSlopeAngleDegrees: 55, walkableClimbWorld: 1 },
    { clearanceField: true, horizontalTolerance: 0.6 },
  );

  const aff = new AffordanceRegistry();
  const boosts: CatMouseBoostView[] = [];
  const jumps: CatMouseJumpView[] = [];
  const boostIds = new Set<string>();
  const jumpIds = new Set<string>();

  // Boost SOUTH: east→west fling along the open southern corridor (the
  // natural chase route for cats going from the SE spawn toward an NW prey).
  const boostS = createBoostAffordance({
    id: 'boost-south',
    pad: { x: 40, z: 14 },
    entryRadius: 4,
    exit: { x: 16, z: 14, heading: Math.PI, speed: 0, t: 0 },
    duration: 0.7,
    cost: 1.5,
  });
  aff.add(boostS);
  boostIds.add(boostS.id);
  boosts.push({
    id: boostS.id,
    pad: { x: 40, z: 14 },
    exit: { x: 16, z: 14 },
    r: 4,
  });

  // Boost NORTH: west→east fling along the northern corridor (useful for the
  // mirror chase after a capture when the mouse respawns south-east).
  const boostN = createBoostAffordance({
    id: 'boost-north',
    pad: { x: 16, z: 44 },
    entryRadius: 4,
    exit: { x: 40, z: 44, heading: 0, speed: 0, t: 0 },
    duration: 0.7,
    cost: 1.5,
  });
  aff.add(boostN);
  boostIds.add(boostN.id);
  boosts.push({
    id: boostN.id,
    pad: { x: 16, z: 44 },
    exit: { x: 40, z: 44 },
    r: 4,
  });

  // Jump across the canyon east→west (the only way to cross at z >= 20).
  const jumpCanyon = createJumpAffordance({
    id: 'jump-canyon',
    launch: { x: 33, z: 32 },
    entryRadius: 3.5,
    land: { x: 25, z: 32, heading: Math.PI, speed: 0, t: 0 },
    apexY: 3,
    duration: 0.9,
    cost: 3,
  });
  aff.add(jumpCanyon);
  jumpIds.add(jumpCanyon.id);
  jumps.push({
    id: jumpCanyon.id,
    launch: { x: 33, z: 32 },
    land: { x: 25, z: 32 },
    r: 3.5,
  });

  return {
    bounds,
    positions,
    indices,
    world,
    affordances: aff,
    boosts,
    jumps,
    boostIds,
    jumpIds,
  };
}

export interface CatMouseAgent {
  id: string;
  state: CarKinematicState;
  plan: CarKinematicState[];
  replan: ReplanState;
  usedBoost: boolean;
  usedJump: boolean;
  /** Latest plan cost (for HUD/tests). */
  cost: number;
}

export interface CatMouseSimState {
  cats: CatMouseAgent[];
  mouse: {
    state: HumanoidState;
    obsHistory: HumanoidState[];
    waypoint: { x: number; z: number };
    waypointAge: number;
  };
  registry: PlanRegistry;
  captures: number;
  simTime: number;
  capturedFlashUntil: number;
  /** Min cat-mouse distance observed since last reset (for HUD/tests). */
  minDistanceEver: number;
}

export interface CatMouseKnobs {
  catCount: number;
  /** Wall-clock budget per plan() call, milliseconds. */
  deadlineMs: number;
  /** Seconds; how far ahead the cats predict the mouse. */
  predictionHorizon: number;
  /** When false, cats plan to the mouse's CURRENT position (A/B comparison). */
  predictionEnabled?: boolean;
}

// Small deterministic LCG so the headless scenario test (and the demo's
// reset button) produce reproducible mouse wanders. The UI seeds it from
// the wall clock so successive sessions feel fresh.
function makeRng(seed: number) {
  let s = (seed | 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) % 1000000) / 1000000;
  };
}

/** Spawn the mouse and N cats on opposite corners. Plans are empty until the
 *  first step triggers `shouldReplan` (no-plan). */
export function initCatAndMouseState(
  world: CatMouseWorld,
  catCount: number,
  seed = 1,
): CatMouseSimState {
  const rng = makeRng(seed);
  // Mouse starts NW (across the canyon from the cats).
  const mouseStart: HumanoidState = { x: 8, z: 48, heading: 0, t: 0 };
  const cats: CatMouseAgent[] = [];
  for (let i = 0; i < catCount; i++) {
    const startX = 46 + (i % 2 === 0 ? -2 : 2);
    const startZ = 6 + i * 3;
    const state: CarKinematicState = {
      x: startX,
      z: startZ,
      heading: Math.PI,
      speed: 0,
      t: 0,
    };
    cats.push({
      id: `cat${i}`,
      state,
      plan: [state],
      replan: new ReplanState({
        // Stagger refresh per cat so all N don't replan on the same frame.
        divergenceThresholdMeters: 1.5,
        refreshIntervalMs: 320 + i * 80,
        switchCostImprovement: 0.1,
      }),
      usedBoost: false,
      usedJump: false,
      cost: 0,
    });
  }
  return {
    cats,
    mouse: {
      state: mouseStart,
      obsHistory: [mouseStart],
      waypoint: pickMouseWaypoint(world, mouseStart, rng),
      waypointAge: 0,
    },
    registry: new PlanRegistry(),
    captures: 0,
    simTime: 0,
    capturedFlashUntil: 0,
    minDistanceEver: Infinity,
  };
}

function pickMouseWaypoint(
  world: CatMouseWorld,
  from: { x: number; z: number },
  rng: () => number,
): { x: number; z: number } {
  for (let tries = 0; tries < 30; tries++) {
    const wx = 4 + rng() * (CATMOUSE_W - 8);
    const wz = 4 + rng() * (CATMOUSE_D - 8);
    if (world.world.polygonAt(wx, wz) === null) continue;
    // Require a meaningful displacement so the mouse doesn't dither in place.
    if (Math.hypot(wx - from.x, wz - from.z) < 8) continue;
    return { x: wx, z: wz };
  }
  return from;
}

/** EMA-smoothed constant-velocity predictor over the last few observations.
 *  Humanoid has no signed forward speed; velocity comes from positional
 *  deltas. Rebuilt every tick (cheap closure). */
function buildMousePredictor(
  obs: HumanoidState[],
  horizon: number,
): Predict<{
  x: number;
  z: number;
  heading: number;
  speed: number;
  t: number;
}> {
  const last = obs[obs.length - 1]!;
  if (obs.length < 2) {
    return (t) => {
      const dt = t - last.t;
      if (dt < 0 || dt > horizon) return null;
      return { x: last.x, z: last.z, heading: last.heading, speed: 0, t };
    };
  }
  const alpha = 0.45;
  let vx = 0;
  let vz = 0;
  let init = false;
  const start = Math.max(0, obs.length - 5);
  for (let i = start; i < obs.length - 1; i++) {
    const a = obs[i]!;
    const b = obs[i + 1]!;
    const dt = Math.max(0.01, b.t - a.t);
    const ivx = (b.x - a.x) / dt;
    const ivz = (b.z - a.z) / dt;
    if (!init) {
      vx = ivx;
      vz = ivz;
      init = true;
    } else {
      vx = alpha * ivx + (1 - alpha) * vx;
      vz = alpha * ivz + (1 - alpha) * vz;
    }
  }
  return (t) => {
    const dt = t - last.t;
    if (dt < 0 || dt > horizon) return null;
    return {
      x: last.x + vx * dt,
      z: last.z + vz * dt,
      heading: Math.atan2(vz, vx),
      speed: Math.hypot(vx, vz),
      t,
    };
  };
}

/** Predictor as a public utility — the headless test asserts the cat plans
 *  to where the mouse WILL be, so it rebuilds the predictor to score. */
export function predictMouseFromHistory(
  obs: HumanoidState[],
  horizon: number,
) {
  return buildMousePredictor(obs, horizon);
}

/** Step the simulation forward by `dt` seconds. Pure (mutates `state` and
 *  returns it). Visual layers read directly from the returned state. */
export function stepCatAndMouse(
  world: CatMouseWorld,
  state: CatMouseSimState,
  dt: number,
  knobs: CatMouseKnobs & { nowMs: number; rng?: () => number },
): CatMouseSimState {
  const tStart = state.simTime;
  const tEnd = tStart + dt;
  const nowMs = knobs.nowMs;
  const rng = knobs.rng ?? Math.random;
  const predictionEnabled = knobs.predictionEnabled ?? true;

  // 1) Mouse AI: pick a new waypoint occasionally or on reach. Drive toward
  //    it with a flee blend when a cat gets close.
  const m = state.mouse;
  m.waypointAge += dt;
  const wpDist = Math.hypot(m.state.x - m.waypoint.x, m.state.z - m.waypoint.z);
  if (wpDist < 2.0 || m.waypointAge > 5) {
    m.waypoint = pickMouseWaypoint(world, m.state, rng);
    m.waypointAge = 0;
  }
  let dx = m.waypoint.x - m.state.x;
  let dz = m.waypoint.z - m.state.z;
  const dmag = Math.hypot(dx, dz) || 1;
  let hx = dx / dmag;
  let hz = dz / dmag;
  let nearestD = Infinity;
  let nearestCat: CatMouseAgent | null = null;
  for (const c of state.cats) {
    const d = Math.hypot(c.state.x - m.state.x, c.state.z - m.state.z);
    if (d < nearestD) {
      nearestD = d;
      nearestCat = c;
    }
  }
  if (nearestCat && nearestD < 8) {
    // Cap the flee blend so the mouse can't permanently outmanoeuvre the
    // turn-radius-constrained cats; otherwise captures never happen.
    const fx = m.state.x - nearestCat.state.x;
    const fz = m.state.z - nearestCat.state.z;
    const fm = Math.hypot(fx, fz) || 1;
    const w = Math.min(0.55, 1 - nearestD / 8);
    hx = (1 - w) * hx + w * (fx / fm);
    hz = (1 - w) * hz + w * (fz / fm);
    const nm = Math.hypot(hx, hz) || 1;
    hx /= nm;
    hz /= nm;
  }

  // 2) Build mouse predictor from observation history (BEFORE integrating
  //    this tick — the cats see the past and plan for the future).
  const mousePredict = buildMousePredictor(m.obsHistory, knobs.predictionHorizon);

  // 3) Replan cats (decisions made at tStart). Each cat publishes its plan
  //    so sibling cats can read it back via PlanRegistry → emergent flanking.
  //    Cats target FLANKING OFFSETS around the predicted mouse pose rather
  //    than the exact same point — without this every cat plans to the same
  //    spot, and since `predictNPC` returns the published plan's final pose
  //    forever past its end, the second cat's goal would be permanently
  //    blocked by the first cat's plan endpoint and the planner would
  //    return no-plan.
  for (let ci = 0; ci < state.cats.length; ci++) {
    const cat = state.cats[ci]!;
    if (!cat.replan.shouldReplan(cat.state, nowMs)) continue;

    // Spread N cats around the target on a small circle so they approach
    // from different angles. Radius is intentionally bigger than the
    // per-cat asObstacle radius (1.0) so each cat's intercept point sits
    // outside any sibling's "tail" obstacle. Decay the offset to zero as
    // the cat closes in (≤ 4 m) so the final pounce aims at the mouse
    // itself instead of orbiting around it forever.
    const n = state.cats.length;
    const dToMouseNow = Math.hypot(
      cat.state.x - m.state.x,
      cat.state.z - m.state.z,
    );
    const flankWeight = n > 1 ? Math.min(1, Math.max(0, (dToMouseNow - 3) / 4)) : 0;
    const flankAngle = n > 1 ? (ci / n) * Math.PI * 2 : 0;
    const flankR = 2.5 * flankWeight;
    const flankDx = flankR * Math.cos(flankAngle);
    const flankDz = flankR * Math.sin(flankAngle);

    let goal: CarKinematicState;
    if (predictionEnabled) {
      const dToMouse = Math.hypot(
        cat.state.x - m.state.x,
        cat.state.z - m.state.z,
      );
      const eta = (dToMouse / CAT_MAX_SPEED) * 1.1;
      const interceptT = tStart + Math.min(eta, knobs.predictionHorizon);
      const ip = mousePredict(interceptT);
      goal = {
        x: (ip?.x ?? m.state.x) + flankDx,
        z: (ip?.z ?? m.state.z) + flankDz,
        heading: cat.state.heading,
        speed: 0,
        t: 0,
      };
    } else {
      // Naïve mode: plan to where the mouse IS right now. The A/B story.
      goal = {
        x: m.state.x + flankDx,
        z: m.state.z + flankDz,
        heading: cat.state.heading,
        speed: 0,
        t: 0,
      };
    }

    const others = state.cats.filter((c) => c.id !== cat.id);
    const env = new TimeAwareEnvironment(catEnv(world.world), {
      // Lightweight collision avoidance only — cats are 2×1, so 1.0 radius
      // + 1.0 agentRadius = 2.0 m clearance is enough to prevent overlap
      // without making the search infeasible.
      obstacles: [
        ...others.map((o) => asObstacle(state.registry.predictNPC(o.id), 1.0)),
      ],
      agentRadius: 1.0,
      affordances: world.affordances,
      affordanceRadius: 14,
      broadphase: {},
    });

    const res = plan(
      {
        start: cat.state,
        goal,
        environment: env,
        options: { maxExpansions: 100000 },
      },
      knobs.deadlineMs,
    );
    if (res.found) {
      // Always adopt: hysteresis is designed for fixed-goal navigation
      // (preventing flip-flops around symmetric obstacles), but in a pursuit
      // the goal moves every replan — comparing the new plan's cost against
      // the now-exhausted committed plan would lock the cat onto a stale
      // path. setPlan() updates lastReplanMs and clears the dirty flag.
      // Trivial length-1 plans (cat already at goal radius) are still
      // adopted; atGoal will mark dirty and the next tick re-plans against
      // the mouse's new position.
      cat.replan.setPlan(res.path, nowMs, res.cost);
      cat.plan = res.path;
      cat.cost = res.cost;
      state.registry.publish(cat.id, res.path);
      cat.usedBoost = res.nodes.some((n) => {
        const data = n.edge?.data as { affordanceId?: string } | undefined;
        return data?.affordanceId ? world.boostIds.has(data.affordanceId) : false;
      });
      cat.usedJump = res.nodes.some((n) => {
        const data = n.edge?.data as { affordanceId?: string } | undefined;
        return data?.affordanceId ? world.jumpIds.has(data.affordanceId) : false;
      });
    }
  }

  // 4) Integrate everyone forward by dt.
  const mSpeed = MOUSE_MAX_SPEED * 0.92;
  const nx = m.state.x + hx * mSpeed * dt;
  const nz = m.state.z + hz * mSpeed * dt;
  if (world.world.polygonAt(nx, nz) !== null) {
    m.state = { x: nx, z: nz, heading: Math.atan2(hz, hx), t: tEnd };
  } else {
    // Blocked — hold position but advance time and pick a fresh waypoint.
    m.state = { ...m.state, t: tEnd };
    m.waypoint = pickMouseWaypoint(world, m.state, rng);
    m.waypointAge = 0;
  }
  m.obsHistory.push(m.state);
  if (m.obsHistory.length > 20) m.obsHistory.shift();

  for (const cat of state.cats) {
    if (cat.plan.length < 2) {
      cat.state = { ...cat.state, t: tEnd };
      continue;
    }
    const cmd = purePursuit(cat.state, cat.plan, CAT_PP_CFG);
    if (cmd.atGoal) {
      // The committed plan is exhausted but the mouse keeps moving — request
      // an immediate replan so the cat doesn't sit braked until the next
      // periodic refresh.
      cat.replan.markDirty('plan-end');
    }
    const next = CAT_FORWARD_SIM(
      cat.state,
      [cmd.steering, cmd.targetSpeed],
      dt,
    );
    if (world.world.polygonAt(next.x, next.z) !== null) {
      cat.state = next;
    } else {
      // Stuck against a wall (pure-pursuit drift) — mark dirty so the next
      // tick replans from scratch.
      cat.state = { ...cat.state, t: tEnd, speed: 0 };
      cat.replan.markDirty('off-mesh');
    }
  }

  state.simTime = tEnd;

  // 5) Capture test + min-distance tracking.
  let curMin = Infinity;
  for (const cat of state.cats) {
    const d = Math.hypot(cat.state.x - m.state.x, cat.state.z - m.state.z);
    if (d < curMin) curMin = d;
  }
  if (curMin < state.minDistanceEver) state.minDistanceEver = curMin;
  if (curMin < 2.2) {
    state.captures++;
    const respawn = farthestSpawnFromCats(world, state.cats);
    m.state = { x: respawn.x, z: respawn.z, heading: 0, t: state.simTime };
    m.obsHistory = [m.state];
    m.waypoint = pickMouseWaypoint(world, m.state, rng);
    m.waypointAge = 0;
    state.capturedFlashUntil = state.simTime + 0.7;
    state.minDistanceEver = Infinity;
    // Force every cat to replan to the new target.
    for (const cat of state.cats) cat.replan.markDirty('capture');
  }

  return state;
}

function farthestSpawnFromCats(
  world: CatMouseWorld,
  cats: CatMouseAgent[],
): { x: number; z: number } {
  let best = { x: 8, z: 48 };
  let bestMinDist = -1;
  for (let z = 4; z < CATMOUSE_D - 4; z += 4) {
    for (let x = 4; x < CATMOUSE_W - 4; x += 4) {
      if (world.world.polygonAt(x, z) === null) continue;
      let minCatDist = Infinity;
      for (const c of cats) {
        const d = Math.hypot(c.state.x - x, c.state.z - z);
        if (d < minCatDist) minCatDist = d;
      }
      if (minCatDist > bestMinDist) {
        bestMinDist = minCatDist;
        best = { x, z };
      }
    }
  }
  return best;
}

/** Headless one-shot for the scenario test (and for sanity-checking in
 *  isolation). Builds the world, inits state, steps N times at fixed dt with
 *  a deterministic RNG, returns the final state. */
export function buildCatAndMouseScenario(
  catCount = 2,
  ticks = 14,
  dt = 0.4,
): {
  world: CatMouseWorld;
  state: CatMouseSimState;
  distanceTrace: number[];
} {
  const world = buildCatAndMouseWorld();
  const state = initCatAndMouseState(world, catCount, 42);
  const rng = makeRng(7);
  const distanceTrace: number[] = [];
  const knobs: CatMouseKnobs = {
    catCount,
    deadlineMs: 200,
    predictionHorizon: 3,
  };
  for (let i = 0; i < ticks; i++) {
    stepCatAndMouse(world, state, dt, {
      ...knobs,
      nowMs: i * 400,
      rng,
    });
    let minD = Infinity;
    for (const c of state.cats) {
      const d = Math.hypot(c.state.x - state.mouse.state.x, c.state.z - state.mouse.state.z);
      if (d < minD) minD = d;
    }
    distanceTrace.push(minD);
  }
  return { world, state, distanceTrace };
}
