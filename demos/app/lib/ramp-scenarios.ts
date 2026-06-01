// Ramp + Affordance demo scenario. The "ramp" is a real drivable heightfield
// mound (rises from ground to crest, then drops off vertically on the back
// side via the heightfield sample-step) so the car can physically drive up
// and launch off the lip. A `BallisticJump` Affordance is registered at the
// ramp crest so the planner can OPTIONALLY take the arc as a shortcut edge,
// but execution is always real Rapier physics — pure pursuit on the path,
// the car flies under gravity, no kinematic posing.
//
// A small planner-only obstacle ("the gap") sits between the ramp crest and
// the landing zone so:
//   - with affordance: the planner picks the jump edge, skipping the gap
//   - without affordance: the planner must detour around the gap
// This guarantees the affordance plan is strictly cheaper.

import { planVehicleOnce } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  nudgeGoalClear,
  rampHeightSampler,
  jumpSpecFromRamp,
} from 'kinocat/environment';
import type {
  NavPolygon,
  NavWorld,
  RampSpec,
  RampJumpSpec,
  HeightSampler,
} from 'kinocat/environment';
import {
  AffordanceRegistry,
  createJumpAffordance,
} from 'kinocat/predict';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import { characterizeVehicle, MotionPrimitiveLibrary } from 'kinocat/primitives';
import { buildCourseObstacles } from './course-obstacles';

export { rampHeightSampler };
export type { RampSpec, RampJumpSpec, HeightSampler };

export const RAMP_BOUNDS = { x0: -60, x1: 60, z0: -30, z1: 30 } as const;

export const RAMP_PALETTE = {
  bg: '#0a0d14',
  fog: '#0e1320',
  ground: 0x1a2233,
  ramp: 0x915b3a,
  arc: 0xffd0a0,
  car: 0x55dcff,
  carPath: 0x7fe9ff,
  goal: 0xffcc33,
  gap: 0xff66aa,
} as const;

export interface RampGapSpec {
  /** Planner-only obstacle centre (no physical collider — the car flies
   *  over). */
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export interface RampCourse {
  bounds: typeof RAMP_BOUNDS;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  ramps: RampSpec[];
  gaps: RampGapSpec[];
  jumps: RampJumpSpec[];
  spawn: CarKinematicState;
  goal: CarKinematicState;
}

const GAP_INFLATE = 0.5;

export interface BuildRampOptions {
  /** Include the gap obstacle the planner has to deal with. Default true. */
  withGap?: boolean;
}

export function buildRampCourse(opts: BuildRampOptions = {}): RampCourse {
  const withGap = opts.withGap ?? true;
  const polygons: NavPolygon[] = [
    {
      id: 1,
      y: 0,
      ring: [
        [RAMP_BOUNDS.x0, RAMP_BOUNDS.z0],
        [RAMP_BOUNDS.x1, RAMP_BOUNDS.z0],
        [RAMP_BOUNDS.x1, RAMP_BOUNDS.z1],
        [RAMP_BOUNDS.x0, RAMP_BOUNDS.z1],
      ],
    },
  ];

  // Single ramp aligned with +X. Long gentle face (the car needs to climb it
  // at cruise speed) and a sharp lip at the crest (so it actually launches).
  // Up-slope spans x ∈ [-3, 11] (length 14), crest at x=11. Narrow width so
  // the ramp visually reads as a directional launch ramp, not a mound.
  const ramps: RampSpec[] = [
    {
      id: 'ramp-main',
      base: { x: 4, z: 0 },
      length: 14,
      width: 10,
      height: 2.5,
      heading: 0,
    },
  ];

  // Per-ramp ballistic launch -> land pose. `jumpSpecFromRamp` estimates the
  // landing distance for a car cruising up the ramp at `cruiseSpeed` and
  // launching off the lip at the slope angle; override `launchDist` here
  // (10 m) for a generous margin since the planner-only "gap" obstacle is
  // sized around it.
  const jumps: RampJumpSpec[] = ramps.map((r) =>
    jumpSpecFromRamp(r, { launchDist: 10 }),
  );

  // Planner-only "gap" sitting just past the ramp crest. Forces the planner
  // to either detour around (long) or take the jump affordance (short). Not
  // a physical collider — the car flies over it.
  //
  // Sizing: the gap MUST keep clear of launch + land by more than the agent
  // footprint forward radius (≈2.2m + inflate), otherwise the planner can't
  // reach the affordance entry. Margin = 3m on each side keeps the
  // footprint clear at both endpoints. Lateral half-width is wide (15m) so
  // the drive-around path is significantly longer than the affordance.
  const gaps: RampGapSpec[] = withGap
    ? jumps.map((j) => {
        const c = Math.cos(j.heading);
        const s = Math.sin(j.heading);
        const jumpDist = Math.hypot(j.land.x - j.launch.x, j.land.z - j.launch.z);
        const endpointMargin = 3;
        const halfAlong = Math.max(0.5, jumpDist / 2 - endpointMargin);
        const halfLateral = 15;
        // Centre is the midpoint of launch and land.
        const midX = (j.launch.x + j.land.x) / 2;
        const midZ = (j.launch.z + j.land.z) / 2;
        return {
          x: midX,
          z: midZ,
          hx: Math.abs(halfAlong * c) + Math.abs(halfLateral * s),
          hz: Math.abs(halfAlong * s) + Math.abs(halfLateral * c),
        };
      })
    : [];

  // Planner obstacles = the ramp's solid wedge walls (sides + back, so the car
  // can only drive up the front and must jump or reverse off the crest) PLUS
  // the planner-only "gap" boxes. Both funnel through the shared helper so the
  // ramp's side collision is never forgotten.
  const obstacles = buildCourseObstacles({
    boxes: gaps,
    ramps,
    inflate: GAP_INFLATE,
    rampOpts: { back: true },
  });

  return {
    bounds: RAMP_BOUNDS,
    polygons,
    obstacles,
    ramps,
    gaps,
    jumps,
    spawn: { x: -45, z: 0, heading: 0, speed: 0, t: 0 },
    goal: { x: 50, z: 0, heading: 0, speed: 0, t: 0 },
  };
}

// ---------------------------------------------------------------------------
// Agent + primitive library — reuse the obstacle-course profile (modest car,
// generous turn radius, reverse allowed).

export const RAMP_AGENT: VehicleAgent = defaultVehicleAgent({
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

export const RAMP_LIB = buildPrimitiveLibrary(RAMP_AGENT);

// ---------------------------------------------------------------------------
// Affordances. The jump is launch=ramp-crest, land=past-the-gap.

export function rampAffordances(course: RampCourse): AffordanceRegistry {
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
        cost: 0.2,
      }),
    );
  }
  return reg;
}

// ---------------------------------------------------------------------------
// Planning helper.

export const RAMP_REPLAN_BUDGET_MS = 100;
export const RAMP_MAX_EXPANSIONS = 20000;
export const RAMP_TEST_MAX_EXPANSIONS = 60000;

export interface RampPlanRequest {
  state: CarKinematicState;
  goal: CarKinematicState;
  course: RampCourse;
  world?: NavWorld;
  /** Disable affordances on this plan (without rebuilding the course). */
  withoutAffordances?: boolean;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planRampDemo(req: RampPlanRequest): PlanResult<CarKinematicState> {
  const world =
    req.world ?? new InMemoryNavWorld(req.course.polygons, req.course.obstacles);
  return planVehicleOnce({
    start: req.state,
    goal: req.goal,
    world,
    agent: RAMP_AGENT,
    lib: RAMP_LIB,
    affordances: req.withoutAffordances ? undefined : rampAffordances(req.course),
    deadlineMs: req.deadlineMs ?? RAMP_REPLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? RAMP_MAX_EXPANSIONS,
  });
}

// ---------------------------------------------------------------------------
// Headless snapshot for the test runner.

export interface RampSnapshot {
  course: RampCourse;
  start: CarKinematicState;
  goal: CarKinematicState;
  result: PlanResult<CarKinematicState>;
}

export function buildRampSnapshot(
  opts: { withAffordance?: boolean; withGap?: boolean } = {},
): RampSnapshot {
  const course = buildRampCourse({ withGap: opts.withGap });
  const world = new InMemoryNavWorld(course.polygons, course.obstacles);
  const goal = nudgeGoalClear(course.goal, course.spawn, world, RAMP_AGENT);
  const result = planRampDemo({
    state: course.spawn,
    goal,
    course,
    world,
    withoutAffordances: opts.withAffordance === false,
    deadlineMs: Infinity,
    maxExpansions: RAMP_TEST_MAX_EXPANSIONS,
  });
  return { course, start: course.spawn, goal, result };
}
