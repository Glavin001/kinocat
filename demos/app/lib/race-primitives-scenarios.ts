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

import { planVehicleOnce, planVehicleMultiGoal } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { NavPolygon, NavWorld } from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim, learnedForwardSimV2 } from 'kinocat/agent';
import type {
  LearnedVehicleParams,
  VehicleAgent,
  CarKinematicState,
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
function pose(x: number, z: number, heading: number): CarKinematicState {
  // Default waypoint speed > 0 so the MPC tracker treats race gates as
  // drive-through targets, not "stop-here" terminals. The parking
  // bench adapter explicitly sets `speed: 0` on its single goal pose
  // to signal terminal-pose intent.
  return { x, z, heading, speed: 5, t: 0 };
}

/** A challenging waypoint loop. Coordinates picked so the kinematic library
 *  plan goes wrong (overshoot at the slalom, late braking into the 90°) but
 *  remains FEASIBLE — both cars can complete it, the question is who finishes
 *  faster with less tracking error. */
export interface RaceCourse {
  bounds: { x0: number; x1: number; z0: number; z1: number };
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  waypoints: CarKinematicState[];
  spawn: CarKinematicState;
}

export function buildRaceCourse(): RaceCourse {
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
  const waypoints: CarKinematicState[] = [
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
  // 30 m/s = the Rapier chassis's physical ceiling on flat ground with the
  // race tuning (~4kN engine on ~580kg, no air drag in Rapier). The planner's
  // action space is `(curvature, targetSpeed)` per primitive — that IS
  // up/down/maintain — and we want it free to choose any speed up to the
  // physical limit. The kinematic library will plan 30 m/s through gentle
  // turns the chassis CAN'T physically take; pure-pursuit will clip to
  // ~10 m/s in tight corners; the learned library will plan honest entry
  // speeds and execute cleanly.
  maxSpeed: 30,
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

/** Race-tuned control set spanning the full speed envelope to chassis
 *  ceiling. The planner picks one of these per 0.55s primitive — that's
 *  its action space (curvature × targetSpeed = up/down/maintain). Same
 *  list across kinematic + learned libraries; only the forward model
 *  differs. */
export function raceControlSets(agent: VehicleAgent = RACE_AGENT): number[][] {
  const k = 1 / agent.minTurnRadius;
  const kHalf = k / 2;
  return [
    // Straight at successive speeds. The 30 m/s primitive lets the planner
    // commit to true top speed on the long straight; the 0 m/s primitive
    // gives it a coast-to-stop option for braking zones.
    [0, 30],
    [0, 24],
    [0, 18],
    [0, 12],
    [0, 6],
    [0, 0],
    // Gentle turn at high speed — the most informative test: kinematic
    // library says "take it at 24", real car loses speed and understeers,
    // learned library plans 14-18.
    [kHalf, 24],
    [-kHalf, 24],
    [kHalf, 16],
    [-kHalf, 16],
    // Full-lock turn — kinematic believes 12 m/s is fine at min turn
    // radius (centripetal 32 m/s²); real chassis tops out at ~7 m/s;
    // learned library plans 6-7 m/s entries.
    [k, 12],
    [-k, 12],
    [k, 6],
    [-k, 6],
    // Reverse straight + gentle/tight turns.
    [0, -8],
    [kHalf, -5],
    [-kHalf, -5],
    [k, -3],
    [-k, -3],
  ];
}

/** Start-speed buckets covering the full envelope from rest to top speed.
 *  The planner picks the bucket nearest the car's current speed when
 *  expanding a node. */
export const RACE_START_SPEEDS = [0, 10, 20, 28];

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

/** v2 race library: drives the v2 learned forward sim with its NATIVE
 *  action vocabulary `(steer, driveForce, brakeForce)` — no curvature /
 *  target-speed adapter. The legacy adapter hid a 1-step P-controller
 *  under every primitive, washing out the distinct intent each control
 *  was supposed to express and collapsing the v2 reachable-area hull to
 *  near-zero at speed extremes (see /primitive-explorer).
 *
 *  Three architectural choices make this work:
 *
 *  1. PER-BUCKET DURATION. A 0.55 s primitive at v=0 covers ~0.9 m (the
 *     acceleration phase dominates and all controls produce nearly the
 *     same endpoint). At v=28 a 0.55 s primitive covers ~15 m (plenty
 *     of room to discriminate; the friction-circle saturates tight
 *     turns regardless). Per-bucket: 1.5 / 0.8 / 0.55 / 0.4 s.
 *
 *  2. SPEED-AWARE CONTROL SETS. Each bucket only includes controls that
 *     produce DISTINGUISHABLE endpoints at that speed. At v=0 we add
 *     accelerate-and-turn primitives (turn-while-spooling-up). At v=28
 *     we drop full-lock turns (friction circle clamps them to the same
 *     near-straight outcome) and add gentle-turn + brake-into-corner
 *     variants instead.
 *
 *  3. NATIVE DYNAMICS. The forward sim is `learnedForwardSimV2` driven
 *     directly with the wheeled-controls vector — no adapter. The v2
 *     model was trained on these inputs; this is the action space it
 *     understands. */
export function buildLearnedRaceLibraryV2(
  model: import('kinocat/agent').LearnedVehicleModel,
): MotionPrimitiveLibrary {
  const inner = learnedForwardSimV2(model);
  const cfg = model.config;
  const buckets: Array<{
    startSpeed: number;
    duration: number;
    controls: number[][];
  }> = [
    // Duration tuned per-bucket: enough time for control differences to
    // produce DISTINGUISHABLE endpoints (so the planner has real choices)
    // but not so long that the planner can't react to nearby gates.
    // At v=0 the chassis needs ~1.5 s to accelerate to a speed where turn
    // dynamics differentiate. At v=28 brake-into-corner trajectories need
    // ~0.8 s to develop (5.5 m/s² brake decel × 0.8 s = 4.4 m/s of speed
    // change, enough to discriminate plans).
    { startSpeed: 0,  duration: 1.5,  controls: lowSpeedV2Controls(cfg) },
    { startSpeed: 10, duration: 0.8,  controls: midSpeedV2Controls(cfg) },
    { startSpeed: 20, duration: 0.8,  controls: highSpeedV2Controls(cfg) },
    { startSpeed: 28, duration: 0.8,  controls: topSpeedV2Controls(cfg) },
  ];
  const all: import('kinocat/primitives').MotionPrimitive[] = [];
  let id = 0;
  for (const b of buckets) {
    const lib = characterizeVehicle({
      forwardSim: inner,
      controlSets: b.controls,
      duration: b.duration,
      substeps: 6,
      startSpeeds: [b.startSpeed],
    });
    for (const p of lib.primitives) {
      all.push({ ...p, id: id++ });
    }
  }
  return new MotionPrimitiveLibrary(all, RACE_START_SPEEDS);
}

type CfgLike = import('kinocat/agent').LearnableVehicleConfig;

/** v=0 controls. From rest, controls only differentiate based on how the
 *  chassis accelerates over the 1.5 s primitive — straight vs turning,
 *  full vs partial throttle, reverse. */
function lowSpeedV2Controls(cfg: CfgLike): number[][] {
  const drv = cfg.maxDriveForce;
  const brk = cfg.maxBrakeForce;
  const st = cfg.maxSteerAngle;
  return [
    [0, drv, 0],                  // 0: full-throttle straight
    [0, 0.5 * drv, 0],            // 1: half-throttle straight
    [+st * 0.5, drv, 0],          // 2: half-left + full throttle
    [-st * 0.5, drv, 0],          // 3: half-right + full throttle
    [+st, 0.7 * drv, 0],          // 4: full-left + ¾ throttle
    [-st, 0.7 * drv, 0],          // 5: full-right + ¾ throttle
    [0, -0.4 * drv, 0],           // 6: reverse
    [0, 0, brk * 0.5],            // 7: brake (no-op at v=0; valid for grid coherence)
  ];
}

/** v=10 controls. Mid-range: tight turns become physically possible;
 *  brake becomes meaningful; trail-brake variants for cornering. */
function midSpeedV2Controls(cfg: CfgLike): number[][] {
  const drv = cfg.maxDriveForce;
  const brk = cfg.maxBrakeForce;
  const st = cfg.maxSteerAngle;
  return [
    [0, drv, 0],                  // 0: accelerate
    [0, 0.5 * drv, 0],            // 1: maintain
    [0, 0, 0],                    // 2: coast
    [0, 0, 0.4 * brk],            // 3: brake gently
    [0, 0, brk],                  // 4: brake hard
    [+st * 0.5, 0.5 * drv, 0],    // 5: moderate-left at speed
    [-st * 0.5, 0.5 * drv, 0],    // 6: moderate-right at speed
    [+st, 0.2 * drv, 0],          // 7: tight-left, low throttle
    [-st, 0.2 * drv, 0],          // 8: tight-right, low throttle
    [+st * 0.5, 0, 0.3 * brk],    // 9: trail-brake left
    [-st * 0.5, 0, 0.3 * brk],    // 10: trail-brake right
  ];
}

/** v=20 controls. High-speed regime where friction circle starts to
 *  bite on tight turns. The widest action spread comes from BRAKE-INTO-
 *  CORNER trajectories that drop speed to a turn-friendly regime within
 *  the primitive. */
function highSpeedV2Controls(cfg: CfgLike): number[][] {
  const drv = cfg.maxDriveForce;
  const brk = cfg.maxBrakeForce;
  const st = cfg.maxSteerAngle;
  return [
    [0, drv, 0],                  // accelerate (still possible at 20)
    [0, 0.4 * drv, 0],            // maintain
    [0, 0, 0],                    // coast
    [0, 0, 0.4 * brk],            // brake light
    [0, 0, brk],                  // brake hard
    [+st * 0.3, 0.4 * drv, 0],    // gentle-right + power
    [-st * 0.3, 0.4 * drv, 0],
    [+st * 0.5, 0, 0.3 * brk],    // brake + moderate-right
    [-st * 0.5, 0, 0.3 * brk],
    [+st * 0.8, 0, 0.6 * brk],    // hard brake + sharp-right (decel-into-corner)
    [-st * 0.8, 0, 0.6 * brk],
    [+st, 0, brk],                // full lock + hard brake (extreme racing entry)
    [-st, 0, brk],
  ];
}

/** v=28 controls (top speed). Friction circle limits gentle-radius
 *  turns to a tight band; meaningful spread comes from BRAKE-INTO-CORNER
 *  trajectories (decelerating to a regime where wider turns become
 *  feasible). Includes hard-brake-with-steer combinations that
 *  effectively transition the chassis to mid-speed by the end of the
 *  primitive. */
function topSpeedV2Controls(cfg: CfgLike): number[][] {
  const drv = cfg.maxDriveForce;
  const brk = cfg.maxBrakeForce;
  const st = cfg.maxSteerAngle;
  return [
    [0, 0.4 * drv, 0],            // accelerate-cruise (small accel)
    [0, 0, 0],                    // coast straight
    [0, 0, 0.3 * brk],            // light brake straight
    [0, 0, 0.7 * brk],            // mid brake straight
    [0, 0, brk],                  // hard brake straight
    [+st * 0.15, 0, 0],           // gentle right coast
    [-st * 0.15, 0, 0],           // gentle left coast
    [+st * 0.3, 0, 0.5 * brk],    // moderate right + brake (decel through turn)
    [-st * 0.3, 0, 0.5 * brk],    // moderate left + brake
    [+st * 0.5, 0, brk],          // sharp right + hard brake (race-line entry)
    [-st * 0.5, 0, brk],          // sharp left + hard brake
    [+st, 0, brk],                // full lock + hard brake (extreme entry)
    [-st, 0, brk],
  ];
}

// ---------------------------------------------------------------------------
// Per-tick planning helper.

// Planning happens for BOTH cars sequentially in the same setInterval
// callback (`RacePrimitives.tsx` :: `replanTimer`) so they always plan at
// the same wall time with the same per-car budget — fairness is structural.
// Per-tick CPU = 2 × RACE_REPLAN_BUDGET_MS / REPLAN_INTERVAL_MS. We aim for
// ~80% busy so the animation still runs: 2 × 120 / 300 = 80%.
// 120ms / 3 segments = 40ms per planVehicleOnce — plenty for a 15-25m goal
// at IGHA*'s ~1500 expansions/ms throughput.
export const RACE_REPLAN_BUDGET_MS = 120;
export const RACE_MAX_EXPANSIONS = 30000;
export const RACE_TEST_MAX_EXPANSIONS = 60000;

/** Radius (m) within which `pickNextWaypoint` advances loopIndex to the
 *  next gate. The visual cone is ~0.8 m radius with a 2 m ring; 2.5 m
 *  feels "hit" without being pixel-precise. */
export const RACE_ARRIVE_RADIUS = 2.5;

/** Radius (m) the multi-goal planner uses for its "gate reached" check.
 *  Strictly less than RACE_ARRIVE_RADIUS so EVERY valid plan brings the
 *  chassis close enough that pickNextWaypoint will advance — prevents the
 *  "plan says I clipped the gate, real chassis overshot by ε, loopIndex
 *  stays, U-turn back" failure mode. */
export const RACE_PLANNER_GATE_RADIUS = 1.8;

export interface RacePlanRequest {
  state: CarKinematicState;
  goal: CarKinematicState;
  lib: MotionPrimitiveLibrary;
  world?: NavWorld;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  deadlineMs?: number;
  maxExpansions?: number;
  /** Override the planner's pose discretisation. Tight values (e.g.
   *  posCell=0.3, headingBuckets=36, goalRadius=0.35, goalHeadingTol=0.15)
   *  let the planner find sub-meter parking maneuvers; race defaults
   *  (1.5 / 16 / 4 / ∞) trade precision for speed. */
  posCell?: number;
  headingBuckets?: number;
  goalRadius?: number;
  goalHeadingTol?: number;
  /** Enable Reeds-Shepp heuristic LUT (default on). */
  enableHeuristicTable?: boolean;
}

export function planRace(req: RacePlanRequest): PlanResult<CarKinematicState> {
  const world = req.world ?? new InMemoryNavWorld(req.polygons, req.obstacles);
  const envOpts: import('kinocat/environment').VehicleEnvOptions = {
    ...(req.posCell !== undefined && { posCell: req.posCell }),
    ...(req.headingBuckets !== undefined && { headingBuckets: req.headingBuckets }),
    ...(req.goalRadius !== undefined && { goalRadius: req.goalRadius }),
    ...(req.goalHeadingTol !== undefined && { goalHeadingTol: req.goalHeadingTol }),
    ...(req.enableHeuristicTable === false ? { heuristicTable: false } : { heuristicTable: {} }),
    // Tight scenarios benefit from sweep-segment collision checks +
    // denser analytic-shot sampling — the parking branch's tuning.
    ...(req.posCell !== undefined && req.posCell < 1.0
      ? { sweepSegmentCheck: true, analyticExpansion: { everyN: 3, step: 0.15 } }
      : {}),
  };
  return planVehicleOnce({
    start: req.state,
    goal: req.goal,
    world,
    agent: RACE_AGENT,
    lib: req.lib,
    deadlineMs: req.deadlineMs ?? RACE_REPLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? RACE_MAX_EXPANSIONS,
    envOptions: Object.keys(envOpts).length > 0 ? envOpts : undefined,
  });
}

export interface RaceMultiGoalRequest {
  state: CarKinematicState;
  /** Ordered sequence of gates the chassis must pass through. */
  gates: CarKinematicState[];
  lib: MotionPrimitiveLibrary;
  world?: NavWorld;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  deadlineMs?: number;
  maxExpansions?: number;
  /** Position radius for "gate reached" check. Default 4 m. */
  gateRadius?: number;
  /**
   * Optional reference polyline (XZ points) to stay close to. When set,
   * the planner adds a small `referenceWeight * perpDist` term to every
   * successor's cost — a cheap hysteresis that prevents the plan from
   * flipping between near-equal-cost alternatives on noise. Typically
   * the previously-committed plan.
   */
  referencePath?: ReadonlyArray<{ x: number; z: number }>;
  /** Weight per metre of deviation. Default 0.1 (s/m). */
  referenceWeight?: number;
  /** Opt out of the Reeds-Shepp heuristic lookup table (which is enabled
   *  by default in `planVehicleMultiGoal`). Used by ablation harnesses
   *  to measure the cache's contribution. */
  disableHeuristicTable?: boolean;
}

/** Single A* through an ordered SEQUENCE of gates. Unlike `planRace`
 *  (one goal pose) or chained `planRace` calls (N independent goals),
 *  this lets the planner GLOBALLY trade off entries to gate i against
 *  exits toward gate i+1, i+2, ... — the racing-line problem.
 *
 *  Same time-cost as `planRace`; same goal radius. The only constraint
 *  is "pass within `gateRadius` of each gate in order" — no heading,
 *  no speed, exactly what the user asked for. */
export function planRaceMultiGoal(req: RaceMultiGoalRequest): PlanResult<CarKinematicState> {
  const world = req.world ?? new InMemoryNavWorld(req.polygons, req.obstacles);
  // Build envOptions only when we have a non-default flag to set. The
  // planner merges this with its own defaults — leaving the others
  // (posCell, headingBuckets, analyticExpansion, heuristicTable) alone.
  const envOptions: import('kinocat/environment').VehicleEnvOptions = {};
  let usedEnvOptions = false;
  if (req.referencePath && req.referencePath.length >= 2) {
    envOptions.referencePath = req.referencePath;
    envOptions.referenceWeight = req.referenceWeight;
    usedEnvOptions = true;
  }
  if (req.disableHeuristicTable) {
    envOptions.heuristicTable = false;
    usedEnvOptions = true;
  }
  return planVehicleMultiGoal({
    start: req.state,
    gates: req.gates,
    world,
    agent: RACE_AGENT,
    lib: req.lib,
    deadlineMs: req.deadlineMs ?? RACE_REPLAN_BUDGET_MS,
    // Larger budget than single-goal because the search space is N× larger
    // (chassis pose × gate index).
    maxExpansions: req.maxExpansions ?? RACE_MAX_EXPANSIONS * 2,
    gateRadius: req.gateRadius,
    envOptions: usedEnvOptions ? envOptions : undefined,
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
  state: CarKinematicState;
  waypoints: CarKinematicState[];
  fromIdx: number;
  count: number;
  lib: MotionPrimitiveLibrary;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  world?: NavWorld;
  totalBudgetMs?: number;
}): { path: CarKinematicState[]; segments: number } {
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
  const path: CarKinematicState[] = [];
  let from: CarKinematicState = { ...state, t: 0 };
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
  goal: CarKinematicState;
  nextIndex: number;
  /** Did we advance to a new waypoint on this tick? */
  advanced: boolean;
}

/** Advance the loop index once within `arriveRadius` of the next-uncleared
 *  waypoint, and return THAT waypoint as the planner's goal.
 *
 *  arriveRadius is tight (2.5m, ≈ the cone's visual ring) so each gate
 *  feels HIT, not brushed past. Multi-segment planning in the caller
 *  (`planThroughWaypoints` with count=3) ensures pure-pursuit's `atGoal`
 *  brake zone only fires for the gate ~3-ahead, never the gate we're
 *  approaching — so we can use a precise arrive radius without the car
 *  braking at each waypoint.
 *
 *  Both cars MUST consume the same output of this function so the
 *  comparison stays fair — only the motion-primitive library varies. */
export function pickNextWaypoint(
  state: CarKinematicState,
  waypoints: CarKinematicState[],
  loopIndex: number,
  arriveRadius = RACE_ARRIVE_RADIUS,
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
  /** Most recent control values applied to the wheels — surfaced in the
   *  HUD so the user can see what the planner is actually asking the car
   *  to do. `targetSpeed` is the pure-pursuit cruise-target setpoint. */
  liveControls: {
    steer: number;        // rad (kinocat sign; positive = curve +X→+Z)
    throttle: number;     // [-1, 1]
    brake: number;        // [0, 1]
    targetSpeed: number;  // m/s the pure-pursuit tracker is aiming at
  };
  /** Per-replan diagnostics — surfaced in the HUD so the user can see
   *  WHY a car is slower (e.g. replans timing out → stale plans → poor
   *  racing line). */
  planDiagnostics: {
    /** Wall-clock ms the most recent replan took. */
    lastReplanMs: number;
    /** True when the most recent replan returned `found: true`. False
     *  means the previous plan is still running (graceful fallback). */
    lastReplanFound: boolean;
    /** Number of consecutive failed replans (resets on a success). High
     *  values mean the planner has been failing repeatedly — the car
     *  is on a stale plan. */
    consecutiveFailedReplans: number;
    /** Wall-clock ms since the currently-executing plan was installed.
     *  Plans older than the replan interval mean failed replans aren't
     *  refreshing the plan in time. */
    planAgeMs: number;
    /** Total successful replans this race (for sanity). */
    successfulReplans: number;
    /** Total replan attempts this race. */
    totalReplans: number;
  };
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
    liveControls: { steer: 0, throttle: 0, brake: 0, targetSpeed: 0 },
    planDiagnostics: {
      lastReplanMs: 0,
      lastReplanFound: false,
      consecutiveFailedReplans: 0,
      planAgeMs: 0,
      successfulReplans: 0,
      totalReplans: 0,
    },
  };
}

// ---------------------------------------------------------------------------
// Headless snapshot for the test runner.

export interface RaceSnapshot {
  spawn: CarKinematicState;
  goal: CarKinematicState;
  kinematicResult: PlanResult<CarKinematicState>;
  learnedResult: PlanResult<CarKinematicState>;
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
