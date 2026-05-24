// Pure, headless-testable obstacle-course scenario. Single car, single AI,
// every building block toggleable so a regression in any one block is
// reproducible in isolation. The interactive demo (`ObstacleCourse.tsx`)
// imports the course builder + per-tick planning helper from here.
//
// This module is the smaller cousin of `carchase-scenarios.ts`: it exists to
// (a) exercise each kinocat building block (heightfield, buildings, ramps,
// boost pads, drift gates, waypoints) in the smallest possible
// configuration, and (b) prove the new core APIs (`planVehicleOnce`,
// `nudgeGoalClear`, `kinocat/adapters/*`) compose cleanly.
import { planVehicleOnce } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  nudgeGoalClear,
  jumpSpecFromRamp,
} from 'kinocat/environment';
import type {
  NavPolygon,
  NavWorld,
  RampSpec,
  RampJumpSpec,
} from 'kinocat/environment';
import {
  AffordanceRegistry,
  createBoostAffordance,
  createJumpAffordance,
} from 'kinocat/predict';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import { characterizeVehicle, MotionPrimitiveLibrary } from 'kinocat/primitives';

export const OBS_BOUNDS = { x0: -60, x1: 60, z0: -40, z1: 40 } as const;

export const OBS_PALETTE = {
  bg: '#0a0d14',
  fog: '#0e1320',
  ground: '#1a2233',
  building: '#3a4458',
  buildingEdge: '#6c7a94',
  ramp: '#915b3a',
  gate: '#ffd070',
  car: 0x55dcff,
  carPath: 0x7fe9ff,
  goal: 0xffcc33,
  boostPad: '#ffe066',
  boostRing: '#ffa030',
} as const;

export interface ObsBuildingSpec {
  x: number;
  z: number;
  hx: number;
  hz: number;
  height: number;
}

export interface ObsBoostSpec {
  id: string;
  x: number;
  z: number;
  exitHeading: number;
  exitDistance: number;
}

export interface ObsGateSpec {
  x: number;
  z: number;
  heading: number;
}

/** Which blocks are enabled. Flipping any of these and rebuilding the course
 *  is a clean A/B for "does the planner get along with X?". */
export interface ObsBlocks {
  heightfield: boolean;
  buildings: boolean;
  ramp: boolean;
  boost: boolean;
  driftGates: boolean;
}

export const OBS_BLOCKS_ALL: ObsBlocks = {
  heightfield: true,
  buildings: true,
  ramp: true,
  boost: true,
  driftGates: true,
};

export interface ObstacleCourse {
  bounds: typeof OBS_BOUNDS;
  blocks: ObsBlocks;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  buildings: ObsBuildingSpec[];
  /** Drivable heightfield ramps (built via `rampHeightSampler`). Each ramp
   *  also has a `BallisticJump` Affordance entry so the planner *can* leap
   *  it instead of climbing — but climbing is the default. */
  ramps: RampSpec[];
  /** Affordance launch/land for each ramp, derived from `ramps`. */
  jumps: RampJumpSpec[];
  boosts: ObsBoostSpec[];
  driftGates: ObsGateSpec[];
  waypoints: CarKinematicState[];
}

function box(x: number, z: number, hx: number, hz: number): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

const BUILDING_INFLATE = 0.5;

/** Build the obstacle course. `blocks` toggles each building block; the
 *  function is pure (deterministic for the same inputs). */
export function buildObstacleCourse(
  blocks: ObsBlocks = OBS_BLOCKS_ALL,
): ObstacleCourse {
  const b = OBS_BOUNDS;
  const polygons: NavPolygon[] = [
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
  ];

  // Three buildings: one wall-like, one in the middle, one near the goal.
  const buildings: ObsBuildingSpec[] = blocks.buildings
    ? [
        { x: -20, z: 10, hx: 8, hz: 3, height: 12 },
        { x: 0, z: -10, hx: 4, hz: 4, height: 8 },
        { x: 35, z: 12, hx: 5, hz: 5, height: 14 },
      ]
    : [];

  // Drivable heightfield ramp. The same `RampSpec` + `rampHeightSampler`
  // pipeline as the /ramp demo; the affordance is a planner-only shortcut,
  // physics is always real raycast-vehicle (the car climbs the ramp).
  const ramps: RampSpec[] = blocks.ramp
    ? [
        {
          id: 'obs-ramp',
          base: { x: 18, z: 0 },
          length: 10,
          width: 6,
          height: 2,
          heading: 0,
        },
      ]
    : [];
  const jumps: RampJumpSpec[] = ramps.map((r) =>
    jumpSpecFromRamp(r, { launchDist: 8 }),
  );

  const boosts: ObsBoostSpec[] = blocks.boost
    ? [
        {
          id: 'obs-boost',
          x: -40,
          z: -20,
          exitHeading: 0,
          exitDistance: 18,
        },
      ]
    : [];

  // Two drift-slalom pillars (also block planning).
  const driftPillars: ObsBuildingSpec[] = blocks.driftGates
    ? [
        { x: -50, z: 15, hx: 1.2, hz: 1.2, height: 6 },
        { x: -45, z: 22, hx: 1.2, hz: 1.2, height: 6 },
      ]
    : [];
  const driftGates: ObsGateSpec[] = blocks.driftGates
    ? [
        { x: -50, z: 18, heading: 0 },
        { x: -45, z: 25, heading: 0 },
      ]
    : [];

  // Collision obstacles passed to the planner: every cuboid the car must go
  // around. Ramps are NOT in this list — the car drives over them.
  const allBlockers = [...buildings, ...driftPillars];
  const obstacles: Array<[number, number][]> = allBlockers.map((s) =>
    box(s.x, s.z, s.hx + BUILDING_INFLATE, s.hz + BUILDING_INFLATE),
  );

  // Waypoint loop tracing the full course — through the boost pad on the
  // west side, between the buildings, over the ramp, around the goal box,
  // back. Picked so each waypoint is in the open and the next is reachable.
  const waypoints: CarKinematicState[] = [
    pose(-45, -25, 0),
    pose(-15, -25, Math.PI / 4),
    pose(10, -5, 0),     // approach the ramp
    pose(28, 0, 0),      // post-jump
    pose(45, 5, Math.PI / 2),
    pose(20, 22, Math.PI),
    pose(-30, 18, Math.PI),
    pose(-48, 0, -Math.PI / 2),
  ];

  return {
    bounds: b,
    blocks,
    polygons,
    obstacles,
    buildings: [...buildings, ...driftPillars],
    ramps,
    jumps,
    boosts,
    driftGates,
    waypoints,
  };
}

function pose(x: number, z: number, heading: number): CarKinematicState {
  return { x, z, heading, speed: 0, t: 0 };
}

// ---------------------------------------------------------------------------
// Agent + primitive library — modest car, generous turn radius, reverse
// allowed so the AI can back out of a corner without ramming a wall.

export const OBS_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 4.5,
  maxSpeed: 12,
  maxReverseSpeed: 4,
  footprint: [
    [2.2, 0.95],
    [-2.2, 0.95],
    [-2.2, -0.95],
    [2.2, -0.95],
  ],
  reverseCostMultiplier: 1.4,
  directionChangePenalty: 0.4,
});

function buildPrimitiveLibrary(agent: VehicleAgent): MotionPrimitiveLibrary {
  const k = 1 / agent.minTurnRadius;
  const kHalf = k / 2;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 12],
      [0, 8],
      [kHalf, 10],
      [-kHalf, 10],
      [k, 6],
      [-k, 6],
      [0, 4],
      [0, -3],
      [kHalf, -3],
      [-kHalf, -3],
    ],
    duration: 0.55,
    substeps: 5,
    startSpeeds: [0],
  });
}

export const OBS_LIB = buildPrimitiveLibrary(OBS_AGENT);

// ---------------------------------------------------------------------------
// Affordance registry mirroring the course's ramps + pads.

export function obsAffordances(course: ObstacleCourse): AffordanceRegistry {
  const reg = new AffordanceRegistry();
  for (const j of course.jumps) {
    reg.add(
      createJumpAffordance({
        id: j.id,
        launch: j.launch,
        entryRadius: 3.5,
        land: { x: j.land.x, z: j.land.z, heading: j.land.heading, speed: 8, t: 0 },
        apexY: j.height + 2,
        duration: 1.0,
        cost: 1.5,
      }),
    );
  }
  for (const p of course.boosts) {
    const c = Math.cos(p.exitHeading);
    const s = Math.sin(p.exitHeading);
    reg.add(
      createBoostAffordance({
        id: p.id,
        pad: { x: p.x, z: p.z },
        entryRadius: 3.5,
        exit: {
          x: p.x + p.exitDistance * c,
          z: p.z + p.exitDistance * s,
          heading: p.exitHeading,
          speed: 11,
          t: 0,
        },
        duration: 0.7,
        cost: 0.5,
      }),
    );
  }
  return reg;
}

// ---------------------------------------------------------------------------
// Per-tick planning helper. Pure wrapper around `planVehicleOnce` with the
// course's affordances and the canonical obstacle-course budgets.

export const OBS_REPLAN_BUDGET_MS = 100;
export const OBS_MAX_EXPANSIONS = 20000;
export const OBS_TEST_MAX_EXPANSIONS = 60000;

export interface ObsPlanRequest {
  state: CarKinematicState;
  goal: CarKinematicState;
  course: ObstacleCourse;
  world?: NavWorld;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planObstacleCourse(
  req: ObsPlanRequest,
): PlanResult<CarKinematicState> {
  const world =
    req.world ?? new InMemoryNavWorld(req.course.polygons, req.course.obstacles);
  return planVehicleOnce({
    start: req.state,
    goal: req.goal,
    world,
    agent: OBS_AGENT,
    lib: OBS_LIB,
    affordances: obsAffordances(req.course),
    deadlineMs: req.deadlineMs ?? OBS_REPLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? OBS_MAX_EXPANSIONS,
  });
}

// ---------------------------------------------------------------------------
// Waypoint AI — pick the next waypoint, nudge it clear, plan to it.

export interface ObsWaypointPick {
  goal: CarKinematicState;
  nextIndex: number;
}

/** Advance to the next waypoint once within `arriveRadius`; nudge the chosen
 *  pose clear of obstacles using the shared core helper. */
export function obsPickWaypoint(
  state: CarKinematicState,
  course: ObstacleCourse,
  loopIndex: number,
  world: NavWorld,
  arriveRadius = 6,
): ObsWaypointPick {
  const cur = course.waypoints[loopIndex]!;
  const d = Math.hypot(state.x - cur.x, state.z - cur.z);
  const useIdx = d < arriveRadius ? (loopIndex + 1) % course.waypoints.length : loopIndex;
  const target = course.waypoints[useIdx]!;
  const goal = nudgeGoalClear(target, state, world, OBS_AGENT);
  return { goal, nextIndex: useIdx };
}

// ---------------------------------------------------------------------------
// Headless snapshot for the test runner.

export interface ObstacleCourseSnapshot {
  course: ObstacleCourse;
  start: CarKinematicState;
  goal: CarKinematicState;
  result: PlanResult<CarKinematicState>;
}

const SPAWN: CarKinematicState = { x: -50, z: -25, heading: 0, speed: 0, t: 0 };

export function buildObstacleCourseSnapshot(
  blocks: ObsBlocks = OBS_BLOCKS_ALL,
): ObstacleCourseSnapshot {
  const course = buildObstacleCourse(blocks);
  const world = new InMemoryNavWorld(course.polygons, course.obstacles);
  const pick = obsPickWaypoint(SPAWN, course, 0, world);
  const result = planObstacleCourse({
    state: SPAWN,
    goal: pick.goal,
    course,
    world,
    deadlineMs: Infinity,
    maxExpansions: OBS_TEST_MAX_EXPANSIONS,
  });
  return { course, start: SPAWN, goal: pick.goal, result };
}

export function obsSpawn(): CarKinematicState {
  return { ...SPAWN };
}
