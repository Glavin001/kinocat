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
import { buildLearnedLibrary } from './learn-primitives';

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

/** Race waypoint pose. Speed is left at 0 — the planner's goal-region check
 *  only considers position + heading (see vehicle-environment.ts'
 *  `reachedGoalRegion`), so the planner is free to choose whatever speed
 *  is most efficient for the trajectory. Flow-through behaviour is
 *  achieved by the lookahead in `pickNextWaypoint`, not by goal speed. */
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
  // 16 m/s gives the planner real headroom on straights (kinematic library
  // will plan a 14-16 m/s cruise) without making the slalom physically
  // impossible (corner centripetal at 4.5m radius + 16 m/s = ~57 m/s²,
  // well above tire grip — both libraries' plans will get clipped by the
  // tracker's lateral-accel cap, but the LEARNED library plans more
  // honest entry speeds so it gets clipped less).
  maxSpeed: 16,
  maxReverseSpeed: 6,
  footprint: [
    [2.4, 1.0],
    [-2.4, 1.0],
    [-2.4, -1.0],
    [2.4, -1.0],
  ],
  reverseCostMultiplier: 1.4,
  directionChangePenalty: 0.4,
});

/** Race-tuned control set with HIGH-SPEED cruise primitives so the planner
 *  can be ambitious about straights and pay for it (or not) in the corners.
 *  Same shape across kinematic + learned libraries — ONLY the forward model
 *  differs. */
export function raceControlSets(agent: VehicleAgent = RACE_AGENT): number[][] {
  const k = 1 / agent.minTurnRadius;
  const kHalf = k / 2;
  return [
    // Cruise at successive speeds — kinematic plan happily takes the
    // 16 m/s straight bucket; learned plan picks up that it can sustain it.
    [0, 16],
    [0, 12],
    [0, 8],
    [0, 4],
    // Gentle turn at high speed — the most informative test:
    // kinematic library says "take it at 14", real car loses speed and
    // understeers, learned library plans 10-12.
    [kHalf, 14],
    [-kHalf, 14],
    [kHalf, 10],
    [-kHalf, 10],
    // Full-lock turn at moderate speed — kinematic believes 10 m/s is fine
    // at min turn radius (centripetal 22 m/s²); real chassis tops out at
    // ~6.5 m/s; learned library plans accordingly.
    [k, 10],
    [-k, 10],
    [k, 6],
    [-k, 6],
    // Reverse straight + gentle/tight turns.
    [0, -6],
    [kHalf, -5],
    [-kHalf, -5],
    [k, -3],
    [-k, -3],
  ];
}

/** Start-speed buckets covering the full envelope from rest to cruise. */
export const RACE_START_SPEEDS = [0, 6, 12];

export function buildKinematicLibrary(): MotionPrimitiveLibrary {
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(RACE_AGENT),
    controlSets: raceControlSets(RACE_AGENT),
    duration: 0.55,
    substeps: 6,
    startSpeeds: RACE_START_SPEEDS,
  });
}

export function buildLearnedRaceLibrary(
  params: LearnedVehicleParams,
): MotionPrimitiveLibrary {
  return buildLearnedLibrary(params, {
    agent: RACE_AGENT,
    controlSets: raceControlSets(RACE_AGENT),
    startSpeeds: RACE_START_SPEEDS,
  });
}

// ---------------------------------------------------------------------------
// Per-tick planning helper.

export const RACE_REPLAN_BUDGET_MS = 200;
export const RACE_MAX_EXPANSIONS = 30000;
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

/** Plan a single continuous trajectory that PASSES THROUGH `count`
 *  consecutive waypoints starting at `fromIdx`. Internally runs N
 *  planVehicleOnce calls in series (current → wp[k], wp[k] → wp[k+1], …)
 *  and concatenates the resulting paths with monotonically increasing
 *  time stamps so pure-pursuit sees one smooth path.
 *
 *  This is how the planner "knows" about the sequence of gates — a single
 *  planVehicleOnce only supports one goal pose and would happily diagonal
 *  THROUGH the gates if asked for a far-away one. Chaining short segments
 *  forces each intermediate waypoint to be reached, with heading
 *  continuity provided by the planner's `reachedGoalRegion` heading
 *  tolerance.
 *
 *  Returns the merged path plus a count of segments that successfully
 *  planned (so the caller can fall back gracefully if a later segment
 *  fails). */
export function planThroughWaypoints(args: {
  state: VehicleState;
  waypoints: VehicleState[];
  fromIdx: number;
  count: number;
  lib: MotionPrimitiveLibrary;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  world?: NavWorld;
  totalBudgetMs?: number;
}): { path: VehicleState[]; segments: number } {
  const {
    state,
    waypoints,
    fromIdx,
    count,
    lib,
    polygons,
    obstacles,
    world,
  } = args;
  const navWorld = world ?? new InMemoryNavWorld(polygons, obstacles);
  const totalBudget = args.totalBudgetMs ?? RACE_REPLAN_BUDGET_MS;
  const perSegment = Math.max(40, totalBudget / Math.max(1, count));
  const path: VehicleState[] = [];
  let from: VehicleState = { ...state, t: 0 };
  let tOffset = 0;
  let segments = 0;
  for (let i = 0; i < count; i++) {
    const goalIdx = (fromIdx + i) % waypoints.length;
    const res = planRace({
      state: from,
      goal: { ...waypoints[goalIdx]!, t: 0 },
      lib,
      polygons,
      obstacles,
      world: navWorld,
      deadlineMs: perSegment,
    });
    if (!res.found || res.path.length < 2) break;
    // Skip the first state of subsequent segments to avoid duplicating
    // the previous segment's end point.
    const segment = i === 0 ? res.path : res.path.slice(1);
    for (const s of segment) {
      path.push({ ...s, t: s.t + tOffset });
    }
    tOffset = path[path.length - 1]!.t;
    from = { ...res.path[res.path.length - 1]!, t: 0 };
    segments++;
  }
  return { path, segments };
}

// ---------------------------------------------------------------------------
// Waypoint AI — pick the next waypoint, advance when reached.

export interface WaypointPick {
  goal: VehicleState;
  nextIndex: number;
  /** Did we advance to a new waypoint on this tick? */
  advanced: boolean;
}

/** Advance the loop index once within `arriveRadius` of the next-uncleared
 *  waypoint, and return THAT waypoint as the planner's goal.
 *
 *  Subtle: we deliberately do NOT lookahead-bypass to a later waypoint as
 *  the goal — for an alternating slalom course, the planner's shortest
 *  path to a far gate would diagonal STRAIGHT THROUGH the intermediate
 *  gates (bypassing them entirely), since the planner only knows about
 *  the goal pose, not the sequence of waypoints in between. We hit each
 *  gate by aiming for it directly; flow-through (no braking at the gate)
 *  is achieved by EXTENDING the plan past the goal in the caller
 *  (`RacePrimitives.tsx`) when the waypoint is cleared.
 *
 *  Both cars MUST consume the same output of this function so the
 *  comparison stays fair — only the motion-primitive library varies. */
export function pickNextWaypoint(
  state: VehicleState,
  waypoints: VehicleState[],
  loopIndex: number,
  arriveRadius = 12,
): WaypointPick {
  const cur = waypoints[loopIndex]!;
  const d = Math.hypot(state.x - cur.x, state.z - cur.z);
  let nextIndex = loopIndex;
  let advanced = false;
  if (d < arriveRadius) {
    nextIndex = (loopIndex + 1) % waypoints.length;
    advanced = true;
  }
  return { goal: waypoints[nextIndex]!, nextIndex, advanced };
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
