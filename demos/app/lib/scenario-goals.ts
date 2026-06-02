// Canonical goal expression for the core ground-vehicle scenarios. Each
// scenario's objective/constraints/preferences are authored ONCE here in the
// `kinocat/scenario` AST (the canonical, serializable form) and planned through
// `planVehicleScenario` (the ScenarioEnvironment product search) — the same
// goals the GoalLab visualizer renders. Pure module (no React/Rapier) so the
// authored ASTs + their plans can be asserted headlessly.
//
// The existing tuned runtimes (race-scenario.ts etc.) still drive the live
// pages; this module is the canonical *expression* layer + a headless planner
// that exercises the real bridge, with `validate()` as a load-time gate.

import { InMemoryNavWorld } from 'kinocat/environment';
import { planVehicleScenario } from 'kinocat/planner';
import type { ScenarioPlanResult } from 'kinocat/planner';
import type { CarKinematicState } from 'kinocat/agent';
import {
  defineScenario,
  reach,
  seq,
  repeat,
  near,
  at,
  inside,
  behind,
  avoid,
  maintain,
  stayInside,
  distanceFrom,
  closingSpeed,
  gte,
  inRange,
  minTime,
  smooth,
  maxProgress,
  deg,
} from 'kinocat/scenario';
import type { Scenario, RegionAgent } from 'kinocat/scenario';
import { demoVehicle } from './scenarios';

import {
  buildParkingScenario,
  parkingLibrary,
  PARKING_AGENT,
  type ParkingScenarioId,
} from './parking-scenarios';
import {
  buildRaceCourse,
  buildKinematicLibrary,
  RACE_AGENT,
  type RaceCourse,
} from './race-primitives-scenarios';

// Bounds rectangle -> polygon for `inside` / `stayInside`.
function boundsPoly(b: { x0: number; x1: number; z0: number; z1: number }): [number, number][] {
  return [
    [b.x0, b.z0],
    [b.x1, b.z0],
    [b.x1, b.z1],
    [b.x0, b.z1],
  ];
}

// ---------------------------------------------------------------------------
// Parking: "park at this pose, stopped, aligned, inside the lot, clear of cars."

export function authorParkingScenario(id: ParkingScenarioId): Scenario {
  const s = buildParkingScenario(id);
  return defineScenario(`park-${id}`, {
    start: s.spawn,
    goal: reach(
      at({ x: s.goal.x, z: s.goal.z, heading: s.goal.heading }, { dx: 0.35, dz: 0.35, dheading: deg(12) }),
      { speed: { max: 0 } },
    ),
    invariants: [stayInside(boundsPoly(s.bounds)), ...s.obstacles.map((o) => avoid(inside(o)))],
    prefer: [minTime(1), smooth(0.4)],
  });
}

export function planParkingScenario(
  id: ParkingScenarioId,
  opts: { deadlineMs?: number; maxExpansions?: number } = {},
): ScenarioPlanResult {
  const s = buildParkingScenario(id);
  const scenario = authorParkingScenario(id);
  const world = new InMemoryNavWorld(s.polygons, s.obstacles);
  return planVehicleScenario({
    start: s.spawn,
    goal: scenario.goal,
    invariants: [stayInside(boundsPoly(s.bounds))], // obstacle collision handled by `world`
    prefer: scenario.prefer,
    world,
    agent: PARKING_AGENT,
    lib: parkingLibrary(),
    envOptions: {
      posCell: 0.3,
      headingBuckets: 36,
      goalRadius: 0.35,
      goalHeadingTol: 0.25,
      sweepSegmentCheck: true,
      analyticExpansion: { everyN: 3, step: 0.15 },
    },
    deadlineMs: opts.deadlineMs ?? Infinity,
    maxExpansions: opts.maxExpansions ?? 80_000,
  });
}

// ---------------------------------------------------------------------------
// Race: a single lap is seq(reach(gate)...); the circuit is repeat(seq(...)).

/** A single drive-through lap, ordered. */
export function authorRaceLap(course: RaceCourse = buildRaceCourse()): Scenario {
  return defineScenario('race-lap', {
    start: course.spawn,
    goal: seq(...course.waypoints.map((w) => reach(near({ x: w.x, z: w.z }, 3)))),
    invariants: [stayInside(boundsPoly(course.bounds))],
    prefer: [minTime(1)],
  });
}

/** The full circuit — a repeat objective (progress maximization), the form the
 *  visualizer animates as laps. */
export function authorRaceCircuit(course: RaceCourse = buildRaceCourse()): Scenario {
  return defineScenario('race-circuit', {
    start: course.spawn,
    goal: repeat(seq(...course.waypoints.map((w) => reach(near({ x: w.x, z: w.z }, 3))))),
    invariants: [stayInside(boundsPoly(course.bounds))],
    prefer: [maxProgress(1)],
  });
}

export function planRaceLap(
  course: RaceCourse = buildRaceCourse(),
  opts: { deadlineMs?: number; maxExpansions?: number } = {},
): ScenarioPlanResult {
  const scenario = authorRaceLap(course);
  const world = new InMemoryNavWorld(course.polygons, course.obstacles);
  return planVehicleScenario({
    start: course.spawn,
    goal: scenario.goal,
    invariants: scenario.invariants,
    prefer: scenario.prefer,
    world,
    agent: RACE_AGENT,
    lib: buildKinematicLibrary(),
    envOptions: { posCell: 1.5, headingBuckets: 16, goalRadius: 3, goalHeadingTol: Infinity },
    deadlineMs: opts.deadlineMs ?? Infinity,
    maxExpansions: opts.maxExpansions ?? 200_000,
  });
}

// ---------------------------------------------------------------------------
// Point-to-point (playground): reach a point, avoid boxes, stay in bounds.

export interface PointToPointInput {
  start: CarKinematicState;
  goal: { x: number; z: number };
  obstacles: { x: number; z: number }[];
  obstacleHalf?: number;
  bounds?: { x0: number; z0: number; x1: number; z1: number };
}

function boxPoly(x: number, z: number, h: number): [number, number][] {
  return [
    [x - h, z - h],
    [x + h, z - h],
    [x + h, z + h],
    [x - h, z + h],
  ];
}

export function authorPointToPoint(inp: PointToPointInput): Scenario {
  const oh = inp.obstacleHalf ?? 2.4;
  const b = inp.bounds ?? { x0: 0, z0: -11, x1: 44, z1: 11 };
  return defineScenario('point-to-point', {
    start: inp.start,
    goal: reach(near(inp.goal, 2)),
    invariants: [
      stayInside(boundsPoly(b)),
      ...inp.obstacles.map((o) => avoid(inside(boxPoly(o.x, o.z, oh)))),
    ],
    prefer: [minTime(1)],
  });
}

// ---------------------------------------------------------------------------
// Drafting / close-follow behind a MOVING car. The lead is a dynamic agent, so
// the `behind` region is time-parameterized — the planner aims for where the
// slipstream slot WILL be (interception), not where it is now (tail-chasing).

export interface DraftingInput {
  start: CarKinematicState;
  /** The car being followed (its predicted future poses). */
  lead: RegionAgent;
  /** Slot distance behind the lead (m). */
  gap?: number;
  /** Slot tolerance radius (m). */
  tol?: number;
  /** Minimum safe gap to never breach (m). */
  safe?: number;
  bounds?: { x0: number; x1: number; z0: number; z1: number };
}

const DRAFT_FIELD = { x0: -40, x1: 70, z0: -25, z1: 25 };

/** Get INTO the slipstream slot once (a one-shot reach — interception). */
export function authorDrafting(inp: DraftingInput): Scenario {
  const gap = inp.gap ?? 6;
  const tol = inp.tol ?? 2;
  const safe = inp.safe ?? 2;
  return defineScenario('drafting', {
    start: inp.start,
    agents: [inp.lead],
    goal: reach(behind(inp.lead, gap, tol)),
    invariants: [
      maintain(distanceFrom(inp.lead, gte(safe))), // never rear-end the lead
      stayInside(boundsPoly(inp.bounds ?? DRAFT_FIELD)),
    ],
    prefer: [minTime(1), smooth(0.3)],
  });
}

/** SUSTAINED drafting: keep re-entering the slot (a bounded `repeat`) while
 *  holding station (closing speed ~0) and a safe gap — the real "draft behind
 *  it" objective, which has no terminal so it's progress-over-a-horizon. */
export function authorDraftingHold(inp: DraftingInput): Scenario {
  const gap = inp.gap ?? 6;
  const tol = inp.tol ?? 2.5;
  const safe = inp.safe ?? 2;
  return defineScenario('drafting-hold', {
    start: inp.start,
    agents: [inp.lead],
    goal: repeat(reach(behind(inp.lead, gap, tol))),
    invariants: [
      maintain(distanceFrom(inp.lead, gte(safe))), // always: never rear-end
      // Hold station (match the lead's pace) ONLY while actually in the
      // slipstream — during the approach the ego must be free to close, so the
      // match-pace constraint is region-scoped with `.while`.
      maintain(closingSpeed(inp.lead, inRange(-2, 2))).while(behind(inp.lead, gap, tol + 1.5)),
      stayInside(boundsPoly(inp.bounds ?? DRAFT_FIELD)),
    ],
    prefer: [minTime(1), smooth(0.3)],
  });
}

export function planDrafting(
  inp: DraftingInput,
  opts: { hold?: boolean; horizonSeconds?: number; maxExpansions?: number } = {},
): ScenarioPlanResult {
  const scenario = opts.hold ? authorDraftingHold(inp) : authorDrafting(inp);
  const bounds = inp.bounds ?? DRAFT_FIELD;
  const world = new InMemoryNavWorld(
    [{ id: 1, y: 0, ring: boundsPoly(bounds) }],
    [],
  );
  // Ego is faster than a typical lead so it can actually close + hold the slot.
  const { agent, lib } = demoVehicle({ maxSpeed: 12 });
  return planVehicleScenario({
    start: inp.start,
    goal: scenario.goal,
    invariants: scenario.invariants,
    prefer: scenario.prefer,
    horizon: opts.hold ? { seconds: opts.horizonSeconds ?? 8 } : undefined,
    world,
    agent,
    lib,
    envOptions: { posCell: 1, headingBuckets: 16, goalRadius: 2 },
    deadlineMs: Infinity, // deterministic
    maxExpansions: opts.maxExpansions ?? 120_000,
  });
}

export { boundsPoly };
