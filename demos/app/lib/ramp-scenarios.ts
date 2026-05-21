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
import { InMemoryNavWorld, nudgeGoalClear } from 'kinocat/environment';
import type { NavPolygon, NavWorld } from 'kinocat/environment';
import {
  AffordanceRegistry,
  createJumpAffordance,
} from 'kinocat/predict';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { VehicleAgent, VehicleState } from 'kinocat/agent';
import { characterizeVehicle, MotionPrimitiveLibrary } from 'kinocat/primitives';

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

export interface RampSpec {
  id: string;
  /** Ramp base centre on the XZ plane (the centre of the up-slope footprint). */
  base: { x: number; z: number };
  /** Up-slope length along `heading`. */
  length: number;
  /** Lateral width perpendicular to `heading`. */
  width: number;
  /** Crest height (world Y). */
  height: number;
  /** Forward direction of the slope (radians, 0 = +X). The slope rises from
   *  `base - (length/2) * heading_dir` to `base + (length/2) * heading_dir`,
   *  then drops off vertically beyond the crest. */
  heading: number;
}

export interface RampGapSpec {
  /** Planner-only obstacle centre (no physical collider — the car flies
   *  over). */
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export interface RampJumpSpec {
  id: string;
  /** Launch point — top of the ramp crest. */
  launch: { x: number; z: number };
  /** Landing pose on the far side of the gap. */
  land: { x: number; z: number; heading: number };
  /** Crest height in world Y, used by the arc helper. */
  height: number;
  /** Approach heading (radians, +X = 0). */
  heading: number;
}

export interface RampCourse {
  bounds: typeof RAMP_BOUNDS;
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  ramps: RampSpec[];
  gaps: RampGapSpec[];
  jumps: RampJumpSpec[];
  spawn: VehicleState;
  goal: VehicleState;
}

const GAP_INFLATE = 0.5;

function box(x: number, z: number, hx: number, hz: number): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

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

  const jumps: RampJumpSpec[] = ramps.map((r) => {
    const c = Math.cos(r.heading);
    const s = Math.sin(r.heading);
    const crestX = r.base.x + (r.length / 2) * c;
    const crestZ = r.base.z + (r.length / 2) * s;
    // With cruise ~12 m/s, ramp slope tan(2.5/14)=10°, the chassis launches
    // off the lip at ~12 m/s with ~2 m/s vertical (from the slope angle).
    // Ballistic range ≈ vx*(vy+sqrt(vy²+2g*h))/g ≈ 12*(2+sqrt(4+49))/9.81
    // ≈ 11.5 m — round to 10 m for a generous margin.
    const launchDist = 10;
    const landX = crestX + launchDist * c;
    const landZ = crestZ + launchDist * s;
    return {
      id: `${r.id}-jump`,
      launch: { x: crestX, z: crestZ },
      land: { x: landX, z: landZ, heading: r.heading },
      height: r.height,
      heading: r.heading,
    };
  });

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

  const obstacles: Array<[number, number][]> = gaps.map((g) =>
    box(g.x, g.z, g.hx + GAP_INFLATE, g.hz + GAP_INFLATE),
  );

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
// Height sampler — flat (y=0) everywhere except inside ramp footprints,
// where the surface rises linearly from base→crest along the ramp heading,
// with a small cosine ease laterally so wheels don't snag a corner. Beyond
// the crest the sampler returns 0 again — the heightfield sample-step gives
// a vertical "lip" the car launches off.

export type HeightSampler = (x: number, z: number) => number;

export function rampHeightSampler(ramps: ReadonlyArray<RampSpec>): HeightSampler {
  return (x, z) => {
    let y = 0;
    for (const r of ramps) {
      const c = Math.cos(r.heading);
      const s = Math.sin(r.heading);
      // Project (x-base, z-base) onto ramp-forward and ramp-lateral axes.
      const dx = x - r.base.x;
      const dz = z - r.base.z;
      const along = dx * c + dz * s;
      const lateral = -dx * s + dz * c;
      const halfL = r.length / 2;
      const halfW = r.width / 2;
      // Lateral skirt: wedge the height down over the last ~1.5m at the
      // sides so we never produce a vertical-side triangle in the
      // heightfield mesh. Vertical / near-vertical triangles intermittently
      // WASM-trap Rapier's wheel raycaster.
      const lateralSkirt = 1.5;
      const lateralInset = halfW - Math.abs(lateral);
      if (lateralInset <= 0) continue;
      const lateralScale = Math.min(1, lateralInset / lateralSkirt);
      // Hard "back" bound on the up-slope start: nothing behind along=-halfL.
      if (along < -halfL) continue;
      // Steep but continuous back-slope past the crest so the car still
      // ballistically launches off the lip (slope steeper than the car can
      // climb), without introducing a true cliff.
      const backSkirt = 2.5;
      if (along > halfL + backSkirt) continue;
      let alongH: number;
      if (along <= halfL) {
        // Linear up-slope from 0 at along=-halfL to height at along=+halfL.
        const u = (along + halfL) / r.length;
        alongH = r.height * u;
      } else {
        // Steep back-slope: full height at the crest, 0 at +backSkirt past it.
        const u = (along - halfL) / backSkirt;
        alongH = r.height * (1 - u);
      }
      const h = alongH * lateralScale;
      if (h > y) y = h;
    }
    return y;
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
  state: VehicleState;
  goal: VehicleState;
  course: RampCourse;
  world?: NavWorld;
  /** Disable affordances on this plan (without rebuilding the course). */
  withoutAffordances?: boolean;
  deadlineMs?: number;
  maxExpansions?: number;
}

export function planRampDemo(req: RampPlanRequest): PlanResult<VehicleState> {
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
  start: VehicleState;
  goal: VehicleState;
  result: PlanResult<VehicleState>;
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
