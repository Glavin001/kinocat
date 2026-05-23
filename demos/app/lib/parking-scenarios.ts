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
import type { VehicleAgent, VehicleState } from 'kinocat/agent';
import {
  characterizeVehicle,
  type MotionPrimitiveLibrary,
} from 'kinocat/primitives';

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
  spawn: VehicleState;
  goal: VehicleState;
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

function pose(x: number, z: number, heading: number): VehicleState {
  return { x, z, heading, speed: 0, t: 0 };
}

// ---------------------------------------------------------------------------
// Agent + motion primitives. Slow, small turn radius, cheap reverse — the
// planner needs the freedom to back-and-fill several times to fit into a
// sub-meter-clearance stall.

export const PARKING_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 3.5,
  maxSpeed: 4,
  maxReverseSpeed: 3,
  // The default footprint (2.55 × 1.15) already encloses the Rapier
  // chassis with a 0.15 m baseline buffer; the demo additionally passes
  // `footprintInflate: 0.15` to the planner for an extra 0.15 m around
  // parked cars during execution.
  reverseCostMultiplier: 1.1,
  directionChangePenalty: 0.2,
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
    controlSets: [
      [0, 3],
      [0, 2],
      [kHalf, 2],
      [-kHalf, 2],
      [k, 1.5],
      [-k, 1.5],
      [0, -2],
      [kHalf, -1.5],
      [-kHalf, -1.5],
      [k, -1.2],
      [-k, -1.2],
    ],
    duration: 0.4,
    substeps: 6,
    startSpeeds: [-1.5, 0, 1.5],
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
  const stallSpacing = 2.7;
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
  const stallSpacing = 2.7;
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
  const walls: ParkingWall[] = [
    { id: 'south-curb', x: 0, z: -2.5, hx: 24, hz: 0.4 },
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
    spawn: pose(-15, 1.4, 0),
    goal: pose(0, 6, Math.PI / 2),
  };
}

function parallel(): ParkingScenario {
  // Two cars parked parallel to curb at z = 0, with a 7.4 m gap between
  // their inner edges. Ego must squeeze in parallel to the curb
  // (heading = 0) between them. Real-world rule of thumb: ~1.5× car
  // length for a comfortable parallel park; 7.4 m / 5.1 m ego ≈ 1.45×,
  // tight but solvable with one reverse shunt.
  const gap = 7.4;
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

export const PARKING_PLAN_BUDGET_MS = 500;
export const PARKING_MAX_EXPANSIONS = 80_000;
// Same expansion budget as the interactive demo; the deadline is what
// differs in the test (we let it run for a few seconds rather than the
// 500 ms per-frame budget the interactive demo uses).
export const PARKING_TEST_MAX_EXPANSIONS = 80_000;
export const PARKING_TEST_DEADLINE_MS = 8_000;

export interface ParkingPlanRequest {
  scenario: ParkingScenario;
  state?: VehicleState;
  world?: NavWorld;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planParking(req: ParkingPlanRequest): PlanResult<VehicleState> {
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
      goalRadius: 0.4,
      goalHeadingTol: 0.18,
      sweepSegmentCheck: true,
      // 0.15 m planning clearance on top of the default 0.15 m baseline
      // gives ~0.3 m total margin from the Rapier chassis edge to any
      // parked-car / curb edge — tight on purpose (this is the demo's
      // whole point) but enough that pure-pursuit drift during execution
      // doesn't scrape a fender.
      footprintInflate: 0.15,
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
  result: PlanResult<VehicleState>;
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
