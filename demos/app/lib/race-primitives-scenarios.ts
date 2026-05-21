// Side-by-side primitive-library race scenario. Two identical Rapier vehicles
// drive an identical waypoint loop; the ONLY difference is the motion-
// primitive library their planner uses (kinematic-derived vs the learned
// library fit to Rapier ground truth in /learnprimitives). The course is
// deliberately tuned to surface what the kinematic model misses:
//
//   - A tight high-speed slalom — exposes "understeer" + lateral-drag (the
//     kinematic planner assumes you can take the gate at curvature κ at
//     speed v; the real chassis can't, pure-pursuit has to brake hard inside
//     the corner, and the actual trajectory overshoots the plan).
//   - A hard 90° turn after a long straight — exposes finite deceleration
//     (the kinematic planner assumes speed instantly tracks target, so it
//     stays at top speed all the way to the corner).
//
// Pure module — no React, no three.js, no localStorage. The interactive demo
// (`RacePrimitives.tsx`) and the headless test import the course + AI helpers
// from here.

import { planVehicleOnce } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavPolygon, NavWorld } from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type {
  LearnedVehicleParams,
  VehicleAgent,
  VehicleState,
} from 'kinocat/agent';
import {
  characterizeVehicle,
  MotionPrimitiveLibrary,
} from 'kinocat/primitives';
import {
  buildLearnedLibrary,
  defaultControlSets,
  DEFAULT_START_SPEEDS,
} from './learn-primitives';

// ---------------------------------------------------------------------------
// Course geometry. Deliberately empty of buildings — we want to isolate the
// dynamic-model gap, not collision avoidance.

export const RACE_BOUNDS = { x0: -65, x1: 65, z0: -35, z1: 35 } as const;

export const RACE_PALETTE = {
  bg: '#0a0d14',
  ground: '#141a26',
  gridMajor: 0x223044,
  gridMinor: 0x141a26,
  kinematic: 0xff8aa0,   // pink — the "before"
  kinematicPath: 0xffc5d0,
  learned: 0x55dcff,     // cyan — the "after"
  learnedPath: 0xa6e9ff,
  gate: 0xffd070,
  gateMissed: 0xff5566,
  startMarker: 0x55ff88,
} as const;

function pose(x: number, z: number, heading: number): VehicleState {
  return { x, z, heading, speed: 0, t: 0 };
}

/** A challenging waypoint loop. Coordinates picked so the kinematic library
 *  plan goes wrong (overshoot at the slalom, late braking into the 90°) but
 *  remains FEASIBLE — both cars can complete it, the question is who finishes
 *  faster with less tracking error. */
export function buildRaceCourse(): {
  bounds: typeof RACE_BOUNDS;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  waypoints: VehicleState[];
  spawn: VehicleState;
} {
  const b = RACE_BOUNDS;
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
  // No physical obstacles — the course is pure dynamics + waypoint chase.
  const obstacles: Array<[number, number][]> = [];

  // Loop: accelerate → TIGHT slalom (alternating ±z gates only 8m apart,
  // demanding tight curvature at speed) → short straight → hard 90° turn →
  // return leg → start. The slalom gates are deliberately spaced just inside
  // the agent's minimum turn radius at cruise — kinematic plans will say
  // "take it at 12 m/s", real chassis can't, the kinematic car overshoots
  // and has to recover. The learned planner predicts the understeer and
  // plans entry-speed accordingly.
  const waypoints: VehicleState[] = [
    pose(-35, 0, 0),    // 0: accel into slalom
    pose(-22, 8, 0),    // 1: slalom L
    pose(-12, -8, 0),   // 2: slalom R  (8m gates, ±8m throw)
    pose(-2, 8, 0),     // 3: slalom L
    pose(8, -8, 0),     // 4: slalom R
    pose(18, 8, 0),     // 5: slalom L
    pose(35, 0, 0),     // 6: recover to centerline
    pose(55, 22, Math.PI / 2),    // 7: hard 90° turn (north)
    pose(-50, 25, Math.PI),       // 8: long straight to far corner
    pose(-55, -10, -Math.PI / 2), // 9: south leg
    pose(-50, -25, 0),  // 10: back to start
  ];

  return {
    bounds: b,
    polygons,
    obstacles,
    waypoints,
    spawn: pose(-55, -20, 0),
  };
}

// ---------------------------------------------------------------------------
// Agent + primitive libraries. Same agent for both cars — the variable under
// test is the library.

export const RACE_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 4.5,
  maxSpeed: 14,
  maxReverseSpeed: 5,
  footprint: [
    [2.4, 1.0],
    [-2.4, 1.0],
    [-2.4, -1.0],
    [2.4, -1.0],
  ],
  reverseCostMultiplier: 1.4,
  directionChangePenalty: 0.4,
});

/** Same control space as `defaultControlSets` so the kinematic and learned
 *  libraries cover identical (curvature, targetSpeed) combinations and the
 *  ONLY difference is the forward model. */
export function buildKinematicLibrary(): MotionPrimitiveLibrary {
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(RACE_AGENT),
    controlSets: defaultControlSets(RACE_AGENT),
    duration: 0.55,
    substeps: 6,
    startSpeeds: DEFAULT_START_SPEEDS,
  });
}

export function buildLearnedRaceLibrary(
  params: LearnedVehicleParams,
): MotionPrimitiveLibrary {
  return buildLearnedLibrary(params, { agent: RACE_AGENT });
}

// ---------------------------------------------------------------------------
// Per-tick planning helper.

export const RACE_REPLAN_BUDGET_MS = 100;
export const RACE_MAX_EXPANSIONS = 20000;
export const RACE_TEST_MAX_EXPANSIONS = 60000;

export interface RacePlanRequest {
  state: VehicleState;
  goal: VehicleState;
  lib: MotionPrimitiveLibrary;
  world?: NavWorld;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planRace(req: RacePlanRequest): PlanResult<VehicleState> {
  const world = req.world ?? new InMemoryNavWorld(req.polygons, req.obstacles);
  return planVehicleOnce({
    start: req.state,
    goal: req.goal,
    world,
    agent: RACE_AGENT,
    lib: req.lib,
    deadlineMs: req.deadlineMs ?? RACE_REPLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? RACE_MAX_EXPANSIONS,
  });
}

// ---------------------------------------------------------------------------
// Waypoint AI — pick the next waypoint, advance when reached.

export interface WaypointPick {
  goal: VehicleState;
  nextIndex: number;
  /** Did we advance to a new waypoint on this tick? */
  advanced: boolean;
}

/** Advance to the next waypoint once within `arriveRadius`. Loops forever.
 *  Default radius is tight (3m) so cars must actually hit each gate, not
 *  just brush past — exposes the kinematic planner's tendency to overshoot
 *  tight gates at speed. */
export function pickNextWaypoint(
  state: VehicleState,
  waypoints: VehicleState[],
  loopIndex: number,
  arriveRadius = 3,
): WaypointPick {
  const cur = waypoints[loopIndex]!;
  const d = Math.hypot(state.x - cur.x, state.z - cur.z);
  if (d < arriveRadius) {
    const next = (loopIndex + 1) % waypoints.length;
    return { goal: waypoints[next]!, nextIndex: next, advanced: true };
  }
  return { goal: cur, nextIndex: loopIndex, advanced: false };
}

// ---------------------------------------------------------------------------
// Per-car race state. Shared between the interactive demo and the headless
// test so both reason about the race in the same units.

export interface RaceMetrics {
  /** Total laps completed (one full pass through the waypoint loop). */
  laps: number;
  /** How many waypoints crossed in total (incl. partial laps). */
  waypointsCleared: number;
  /** Wall time (s) since the race began, accumulated only while moving. */
  raceTime: number;
  /** Time (s) at which the current lap started (set on each lap completion). */
  lapStartTime: number;
  /** Best lap time recorded (s). NaN until first lap completes. */
  bestLapTime: number;
  /** Last completed lap time (s). NaN until first lap completes. */
  lastLapTime: number;
  /** RMS error (m) between the planned trajectory and the actual chassis
   *  trajectory over the entire race. */
  trackingErrorRms: number;
  /** Peak speed observed (m/s). */
  peakSpeed: number;
}

export function emptyMetrics(): RaceMetrics {
  return {
    laps: 0,
    waypointsCleared: 0,
    raceTime: 0,
    lapStartTime: 0,
    bestLapTime: Number.NaN,
    lastLapTime: Number.NaN,
    trackingErrorRms: 0,
    peakSpeed: 0,
  };
}

// ---------------------------------------------------------------------------
// Headless snapshot for the test runner.

export interface RaceSnapshot {
  spawn: VehicleState;
  goal: VehicleState;
  kinematicResult: PlanResult<VehicleState>;
  learnedResult: PlanResult<VehicleState>;
}

/** Tiny smoke check that BOTH libraries can plan from spawn → first
 *  waypoint. The fully-learned library needs `LearnedVehicleParams`; tests
 *  pass `DEFAULT_LEARNED_PARAMS` so they don't need to run the full Rapier
 *  sweep. */
export function buildRaceSnapshot(
  params: LearnedVehicleParams,
): RaceSnapshot {
  const course = buildRaceCourse();
  const goal = course.waypoints[0]!;
  const kinematicLib = buildKinematicLibrary();
  const learnedLib = buildLearnedRaceLibrary(params);
  const kinematicResult = planRace({
    state: course.spawn,
    goal,
    lib: kinematicLib,
    polygons: course.polygons,
    obstacles: course.obstacles,
    deadlineMs: Infinity,
    maxExpansions: RACE_TEST_MAX_EXPANSIONS,
  });
  const learnedResult = planRace({
    state: course.spawn,
    goal,
    lib: learnedLib,
    polygons: course.polygons,
    obstacles: course.obstacles,
    deadlineMs: Infinity,
    maxExpansions: RACE_TEST_MAX_EXPANSIONS,
  });
  return { spawn: course.spawn, goal, kinematicResult, learnedResult };
}
