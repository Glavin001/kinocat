// Tight-parking demo scenarios. Three sub-scenarios in progression of
// difficulty:
//   - 'forward-pullin' : drive forward into an empty stall.
//   - 'reverse-perp'   : back into an empty stall from a tight aisle.
//   - 'parallel'       : parallel-park between two cars at a curb.
//
// Each scenario exercises the new accuracy knobs (footprintInflate, sweep
// segment check, denser analytic-shot sampling) under sub-meter clearances
// so a regression there shows up visibly: the planner can't find a path,
// or pure-pursuit clips one of the surrounding "parked" cars.
//
// Pure module — no React, no Rapier. Both the interactive demo and the
// headless test import the same `buildParkingScenario()` so they reason
// about the same geometry.

import { planVehicleOnce } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavPolygon, NavWorld } from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import {
  characterizeVehicle,
  type MotionPrimitiveLibrary,
} from 'kinocat/primitives';
// Type-only: the runner's tuning + course-shape types. Erased at runtime, so
// this introduces no module cycle (race-scenario does not import this file).
import type { RaceTuning, RaceScenarioOptions, RaceEntry } from './race-scenario';

export type ParkingScenarioId =
  | 'forward-pullin'
  | 'reverse-perp'
  | 'parallel';

export const PARKING_SCENARIOS: ParkingScenarioId[] = [
  'forward-pullin',
  'reverse-perp',
  'parallel',
];

export const PARKING_LABELS: Record<ParkingScenarioId, string> = {
  'forward-pullin': '1 · forward pull-in (easy)',
  'reverse-perp': '2 · reverse perpendicular (medium)',
  'parallel': '3 · parallel parking (hard)',
};

export const PARKING_BOUNDS = { x0: -25, x1: 25, z0: -20, z1: 20 } as const;

export const PARKING_PALETTE = {
  bg: '#0a0d14',
  fog: '#0e1320',
  ground: '#1a2233',
  curb: '#2a3346',
  curbEdge: '#445064',
  stallLine: '#ffd070',
  stallEmpty: '#1f2a3d',
  parkedCar: '#3a4458',
  parkedCarEdge: '#6c7a94',
  ego: 0x55dcff,
  egoPath: 0x7fe9ff,
  goal: 0xffcc33,
} as const;

/** Rectangular static body (axis-aligned). One `ParkedCar` represents
 *  another car already in the lot — both physics collider and planner
 *  obstacle. `hx`/`hz` are half-extents. */
export interface ParkedCar {
  id: string;
  x: number;
  z: number;
  hx: number;
  hz: number;
  /** Visual yaw in radians; physics is axis-aligned. */
  heading: number;
}

/** Axis-aligned wall (curb, building edge, etc.). */
export interface ParkingWall {
  id: string;
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export interface ParkingStallMark {
  x: number;
  z: number;
  hx: number;
  hz: number;
  heading: number;
}

export interface ParkingScenario {
  id: ParkingScenarioId;
  bounds: typeof PARKING_BOUNDS;
  polygons: NavPolygon[];
  /** Combined obstacles passed to the planner. */
  obstacles: Array<[number, number][]>;
  parkedCars: ParkedCar[];
  walls: ParkingWall[];
  /** Visual outline of the target stall — pure render data. */
  targetStall: ParkingStallMark;
  spawn: CarKinematicState;
  goal: CarKinematicState;
}

function box(
  x: number,
  z: number,
  hx: number,
  hz: number,
): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

function pose(x: number, z: number, heading: number): CarKinematicState {
  return { x, z, heading, speed: 0, t: 0 };
}

// ---------------------------------------------------------------------------
// Agent + motion primitives. Slow, small turn radius, cheap reverse — the
// planner needs the freedom to back-and-fill several times to fit into a
// sub-meter-clearance stall.

export const PARKING_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 3.5,
  // Cap at 2 m/s. Pure-pursuit drift on a Rapier raycast vehicle scales
  // roughly with speed; at 3-4 m/s the chassis cuts the inside of a
  // tight Reeds-Shepp arc by 20-30 cm and clips a parked car, even
  // though the plan itself was collision-free.
  maxSpeed: 2,
  maxReverseSpeed: 1.5,
  // Footprint matches the physical Rapier chassis the runner drives and the
  // three.js car mesh renders (both 4.8 × 2.0 m ⇒ 2.4 × 1.0 half-extents).
  // `defaultVehicleAgent` ships a much smaller 3.2 × 1.8 m box (1.6 × 0.9
  // half); planning with that under-sized footprint reserved no clearance for
  // the car's 0.8 m front/rear overhangs, so collision-free plans drove the
  // real bumpers straight into the parked cars (the visible clipping). With the
  // true footprint the planner reserves clearance for the car that actually
  // exists, and the widened scenario geometry below keeps the maneuvers
  // findable. (Footprint feeds only the planner/monitor collision checks, not
  // the kinematic primitive library.)
  footprint: [
    [2.4, 1.0],
    [-2.4, 1.0],
    [-2.4, -1.0],
    [2.4, -1.0],
  ],
  reverseCostMultiplier: 1.05,
  directionChangePenalty: 0.15,
});

export function buildParkingPrimitives(
  agent: VehicleAgent = PARKING_AGENT,
): MotionPrimitiveLibrary {
  const k = 1 / agent.minTurnRadius; // tightest turn
  const kHalf = k / 2;
  // Slow, varied control set: forward + reverse at multiple curvatures,
  // including hold-still options. Each primitive is 0.4 s — short enough
  // that a parallel-park shunt resolves into a handful of edges.
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    // Top speed 2 m/s anywhere in the library — matches PARKING_AGENT.maxSpeed
    // so the planner can't request a faster primitive than pure-pursuit can
    // physically track on the Rapier chassis.
    controlSets: [
      [0, 2],
      [0, 1.2],
      [kHalf, 1.5],
      [-kHalf, 1.5],
      [k, 1.0],
      [-k, 1.0],
      [0, -1.5],
      [kHalf, -1.0],
      [-kHalf, -1.0],
      [k, -0.8],
      [-k, -0.8],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [-1.0, 0, 1.0],
  });
}

// Lazily-cached so building three scenarios + replans doesn't repeat the
// characterisation pass.
let _libCache: MotionPrimitiveLibrary | null = null;
export function parkingLibrary(): MotionPrimitiveLibrary {
  if (_libCache === null) _libCache = buildParkingPrimitives();
  return _libCache;
}

// ---------------------------------------------------------------------------
// Scenario geometry.

// Standard parked-car footprint — same half-extents as the default Rapier
// raycast-vehicle chassis. Tight on purpose: the goal of this demo is to
// stress the planner's spatial accuracy.
const PARKED_HX = 2.4;
const PARKED_HZ = 1.0;

function lotPolygon(id = 1): NavPolygon {
  const b = PARKING_BOUNDS;
  return {
    id,
    y: 0,
    ring: [
      [b.x0, b.z0],
      [b.x1, b.z0],
      [b.x1, b.z1],
      [b.x0, b.z1],
    ],
  };
}

function forwardPullin(): ParkingScenario {
  // Open lot. Row of parked cars at z = 6 facing north (heading = π/2),
  // with one empty stall at x = 0. Stall spacing 2.7 m centre-to-centre
  // gives ~0.7 m lateral clearance between cars (tight). Ego approaches
  // from the south along the +z axis.
  // Stall-to-stall spacing along the row. 3.0 m centre-to-centre gives
  // the ego (2.3 m wide) ~0.7 m of lateral clearance on each side once
  // it's nosed into the empty stall — tight but enough to absorb the
  // 0.25 m planning inflation plus pure-pursuit drift.
  const stallSpacing = 3.0;
  const parkedCars: ParkedCar[] = [];
  for (const i of [-2, -1, 1, 2]) {
    parkedCars.push({
      id: `pk-${i}`,
      x: i * stallSpacing,
      z: 6,
      hx: PARKED_HX,
      hz: PARKED_HZ,
      heading: Math.PI / 2,
    });
  }
  const targetStall: ParkingStallMark = {
    x: 0,
    z: 6,
    hx: PARKED_HX,
    hz: PARKED_HZ,
    heading: Math.PI / 2,
  };
  const obstacles = parkedCars.map((c) => box(c.x, c.z, c.hz, c.hx));
  return {
    id: 'forward-pullin',
    bounds: PARKING_BOUNDS,
    polygons: [lotPolygon()],
    obstacles,
    parkedCars,
    walls: [],
    targetStall,
    spawn: pose(0, -8, Math.PI / 2),
    goal: pose(0, 6, Math.PI / 2),
  };
}

function reversePerp(): ParkingScenario {
  // A tight aisle 6 m wide running east-west. Row of parked cars
  // along z = 6 facing north; opposing curb wall at z = -1. Ego enters
  // from the west driving east; to park nose-north in the empty stall
  // it must drive past and reverse in.
  // Stall-to-stall spacing along the row. 4.0 m centre-to-centre leaves the
  // 2.0 m-wide ego ~2 m of lateral room on each side of the empty stall —
  // enough for the true-size footprint to swing in slightly off-heading on a
  // reverse-S without a corner clipping a neighbour. (3.0 m, tuned for the old
  // under-sized footprint, had the real car grazing on the angled entry.)
  const stallSpacing = 4.0;
  const parkedCars: ParkedCar[] = [];
  for (const i of [-2, -1, 1, 2]) {
    parkedCars.push({
      id: `pk-${i}`,
      x: i * stallSpacing,
      z: 6,
      hx: PARKED_HX,
      hz: PARKED_HZ,
      heading: Math.PI / 2,
    });
  }
  const targetStall: ParkingStallMark = {
    x: 0,
    z: 6,
    hx: PARKED_HX,
    hz: PARKED_HZ,
    heading: Math.PI / 2,
  };
  // Drivable aisle: parked-car row front at z = 3.6, south curb front
  // at z = -6.6, so the aisle is ~10.2 m wide. The real 4.8 m chassis needs
  // room to swing the reverse-S without its bumpers crossing the curb or the
  // parked-car row; the earlier 5.7 m aisle (tuned for a smaller footprint)
  // left the true car grazing on every back-in. The spawn east of the empty
  // stall (below) still makes a forward 90° turn overshoot, so a back-in
  // remains the natural solution.
  const walls: ParkingWall[] = [
    { id: 'south-curb', x: 0, z: -7.0, hx: 24, hz: 0.4 },
    { id: 'north-back', x: 0, z: 9.4, hx: 24, hz: 0.4 },
  ];
  const obstacles = [
    ...parkedCars.map((c) => box(c.x, c.z, c.hz, c.hx)),
    ...walls.map((w) => box(w.x, w.z, w.hx, w.hz)),
  ];
  return {
    id: 'reverse-perp',
    bounds: PARKING_BOUNDS,
    polygons: [lotPolygon()],
    obstacles,
    parkedCars,
    walls,
    targetStall,
    // Spawn EAST of the empty stall, heading east, in the lane just south of
    // the parked row. A forward arc into the stall from here would require the
    // car to circle back round — the natural plan is a reverse-S back into the
    // stall, which is exactly the maneuver this scenario is here to demonstrate.
    spawn: pose(8, 1.5, 0),
    goal: pose(0, 6, Math.PI / 2),
  };
}

function parallel(): ParkingScenario {
  // Two cars parked parallel to curb at z = 0, with a 14 m gap between their
  // inner edges. The ego is 4.8 m → ratio ≈ 2.9×: a roomy parallel park. The
  // gap has to clear the true-size footprint on the angled forward entry
  // (its diving arc passes the rear car's inner edge ~6.6 m left of centre)
  // AND absorb the residual heading misalignment pure-pursuit leaves at the
  // end. The earlier 8.6 m gap (tuned for a smaller box) had the real bumpers
  // clipping the rear neighbour on the way in.
  const gap = 14;
  const aheadX = gap / 2 + PARKED_HX;
  const behindX = -gap / 2 - PARKED_HX;
  const parkedCars: ParkedCar[] = [
    {
      id: 'pk-front',
      x: aheadX,
      z: 0,
      hx: PARKED_HX,
      hz: PARKED_HZ,
      heading: 0,
    },
    {
      id: 'pk-rear',
      x: behindX,
      z: 0,
      hx: PARKED_HX,
      hz: PARKED_HZ,
      heading: 0,
    },
  ];
  const walls: ParkingWall[] = [
    { id: 'curb', x: 0, z: -2.6, hx: 24, hz: 0.4 },
  ];
  const targetStall: ParkingStallMark = {
    x: 0,
    z: 0,
    hx: gap / 2,
    hz: PARKED_HZ,
    heading: 0,
  };
  const obstacles = [
    ...parkedCars.map((c) => box(c.x, c.z, c.hx, c.hz)),
    ...walls.map((w) => box(w.x, w.z, w.hx, w.hz)),
  ];
  // Spawn approaches from behind the rear car, in the next lane out.
  return {
    id: 'parallel',
    bounds: PARKING_BOUNDS,
    polygons: [lotPolygon()],
    obstacles,
    parkedCars,
    walls,
    targetStall,
    spawn: pose(behindX - 4, 3.0, 0),
    goal: pose(0, 0, 0),
  };
}

export function buildParkingScenario(
  id: ParkingScenarioId,
): ParkingScenario {
  switch (id) {
    case 'forward-pullin':
      return forwardPullin();
    case 'reverse-perp':
      return reversePerp();
    case 'parallel':
      return parallel();
  }
}

// ---------------------------------------------------------------------------
// Planning. One-shot plans with tight discretisation: parking is not a
// per-tick replan, it's a single carefully-budgeted search.

/** The planner's static-obstacle clearance margin. Exported so the demo's
 *  debug overlay can draw matching inflation rings around the parked
 *  cars and walls — that's the band the planner refuses to put any
 *  point of the ego's footprint into. */
export const PARKING_FOOTPRINT_INFLATE = 0.35;

export const PARKING_PLAN_BUDGET_MS = 500;
export const PARKING_MAX_EXPANSIONS = 80_000;
// Same expansion budget as the interactive demo; the deadline is what
// differs in the test (we let it run for a few seconds rather than the
// 500 ms per-frame budget the interactive demo uses).
export const PARKING_TEST_MAX_EXPANSIONS = 80_000;
export const PARKING_TEST_DEADLINE_MS = 8_000;

export interface ParkingPlanRequest {
  scenario: ParkingScenario;
  state?: CarKinematicState;
  world?: NavWorld;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planParking(req: ParkingPlanRequest): PlanResult<CarKinematicState> {
  const world =
    req.world ??
    new InMemoryNavWorld(req.scenario.polygons, req.scenario.obstacles);
  const start = req.state ?? req.scenario.spawn;
  return planVehicleOnce({
    start,
    goal: req.scenario.goal,
    world,
    agent: PARKING_AGENT,
    lib: parkingLibrary(),
    // Sub-meter discretisation: planning grid 0.3 m, heading buckets 10°,
    // 0.15 m clearance margin over the (already-buffered) default
    // footprint, and analytic shots at 0.15 m sampling for tight RS
    // curves. Search is heading-aware at the goal so the car ends up
    // squarely aligned with the stall.
    envOptions: {
      posCell: 0.3,
      headingBuckets: 36,
      goalRadius: 0.35,
      goalHeadingTol: 0.15,
      sweepSegmentCheck: true,
      // 0.35 m planning clearance on top of the default 0.15 m baseline
      // gives ~0.5 m total chassis-to-obstacle margin. The previous
      // 0.25 m wasn't enough cushion for the swing of a parallel-parking
      // approach — the front of the chassis was within ~5 cm of the
      // front parked car at the steepest entry angle, and any
      // pure-pursuit drift closed the gap. 0.5 m lets the chassis brush
      // through without scraping.
      // `footprintInflate` planner option not yet supported on this
      // branch — bench uses default footprint with sweep segment check.
      // footprintInflate: PARKING_FOOTPRINT_INFLATE,
      analyticExpansion: { everyN: 3, step: 0.15 },
    },
    deadlineMs: req.deadlineMs ?? PARKING_PLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? PARKING_MAX_EXPANSIONS,
  });
}

// ---------------------------------------------------------------------------
// Headless snapshot used by the test runner.

export interface ParkingSnapshot {
  scenario: ParkingScenario;
  result: PlanResult<CarKinematicState>;
}

export function buildParkingSnapshot(id: ParkingScenarioId): ParkingSnapshot {
  const scenario = buildParkingScenario(id);
  const result = planParking({
    scenario,
    deadlineMs: PARKING_TEST_DEADLINE_MS,
    maxExpansions: PARKING_TEST_MAX_EXPANSIONS,
  });
  return { scenario, result };
}

// ---------------------------------------------------------------------------
// Shared parking driving config — the SINGLE source of truth for how parking
// runs on the `createRaceScenario` engine. The web page (`Parking.tsx`), the
// `controller-bench` CLI, and the Vitest invariant tests all import these so
// the page renders exactly what the headless runs test — no duplicated tuning
// to drift. (Type-only import of the runner types keeps this module free of any
// runtime dependency on `race-scenario`.)

/** Parking-specific `RaceTuning` overrides: low cruise speed, tight goal +
 *  terminal-heading tolerance, sub-meter planner discretisation, and MPC
 *  terminal-pose weights. Everything else falls through to race defaults so
 *  the same controller code drives both racing and parking. */
export const PARKING_RACE_TUNING: Partial<RaceTuning> = {
  cruiseSpeed: 2,
  goalTolerance: 0.4,
  arriveRadius: 0.6,
  plannerPosCell: 0.3,
  plannerHeadingBuckets: 36,
  plannerGoalRadius: 0.35,
  plannerGoalHeadingTol: 0.2,
  plannerBudgetMs: 500,
  plannerMaxExpansions: 80_000,
  mpcWTerminalPosition: 50,
  mpcWTerminalSpeed: 30,
  // Parking maneuvers are committed multi-cusp sequences (reverse → forward
  // → …), not a per-tick chase. Replan slowly so the segment-advance logic
  // carries the chassis through each forward↔reverse cusp instead of
  // re-deciding the whole maneuver every 300 ms (which leaves it oscillating
  // at the cusp). The adaptive lateral-drift trigger still forces an early
  // replan if the chassis genuinely diverges from the committed plan.
  replanIntervalMs: 800,
  // Don't re-smooth the plan. The runner expands the planner's analytic
  // Reeds-Shepp shot-to-goal into its dense (0.15 m) curve samples already;
  // running the Laplacian trajectory smoother over that re-rounds the tight
  // back-in curve and cuts its corner INTO the adjacent parked car. Parking
  // needs the planner's collision-checked geometry tracked faithfully, not
  // geometrically prettified.
  enableTrajectorySmoother: false,
};

/** Convert a parking scenario to the `createRaceScenario` course shape: the
 *  single goal pose (speed 0 ⇒ "terminal pose intent" to the planner) becomes
 *  the sole waypoint; obstacles + bounds carry over directly. */
export function parkingCourse(id: ParkingScenarioId): NonNullable<RaceScenarioOptions['course']> {
  const s = buildParkingScenario(id);
  return {
    bounds: { x0: s.bounds.x0, x1: s.bounds.x1, z0: s.bounds.z0, z1: s.bounds.z1 },
    polygons: s.polygons,
    obstacles: s.obstacles,
    waypoints: [{ ...s.goal, speed: 0, t: 0 }],
    spawn: { ...s.spawn, speed: 0, t: 0 },
  };
}

/** The COMPLETE, canonical `createRaceScenario` options for a parking scenario
 *  — the one definition the web page, the controller-bench CLI, and the Vitest
 *  tests all build from, so they cannot diverge.
 *
 *  Critically this bakes in ZERO teleportation: `offTrackRecovery: 'none'` and
 *  an infinite stall timeout. Real vehicles don't teleport; a maneuver that
 *  fails to reach the goal must run out the clock and fail honestly rather than
 *  be snapped onto the goal/waypoint (which masks the failure — exactly how the
 *  reverse-perp planner bug stayed hidden). */
export function parkingScenarioOptions(
  id: ParkingScenarioId,
  entries: RaceEntry[],
  tuningOverride?: Partial<RaceTuning>,
): RaceScenarioOptions {
  return {
    // Pin every parking entry to PARKING_AGENT so the planner's heuristic +
    // footprint + turn radius match the parking primitive library. Without
    // this the runner planned parking with RACE_AGENT (30 m/s, 4.5 m turn
    // radius, larger footprint), whose 15×-faster maxSpeed rescaled the
    // time-cost heuristic so far below the 2 m/s primitives' real progress
    // that A* degenerated into near-breadth-first search — the reverse-perp /
    // parallel replan-failure storm. Callers don't need to know the agent;
    // they just pass `{ name, lib: parkingLibrary() }`.
    entries: entries.map((e) => ({ ...e, agent: PARKING_AGENT })),
    targetLaps: 1,
    syncHold: false,
    offTrackRecovery: 'none',
    stallTimeoutMs: Number.POSITIVE_INFINITY,
    tuning: { ...PARKING_RACE_TUNING, ...tuningOverride },
    course: parkingCourse(id),
  };
}
