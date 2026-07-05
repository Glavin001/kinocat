// Pure, headless-testable car-chase scenario. The interactive CarChase demo
// imports the course geometry, NavWorld builder, AI tactical layer, and
// per-tick planning helper from here; the headless test (demos/test/
// scenarios.test.ts) imports the same to assert the cops + robber always
// produce a plan within budget for the spawn matchup. No React / three /
// Rapier imports — Rapier physics runs only inside CarChase.tsx; planning
// uses the kinematic forward model.
//
// Architecture mirror: `demos/app/lib/dogfight-scenarios.ts` is the airborne
// equivalent; many helpers here are direct ground-2D ports of its 3D ones
// (selectTacticalMode / tacticalGoal / planAI).
import { planVehicleOnce } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  nudgeGoalClear,
  jumpSpecFromRamp,
} from 'kinocat/environment';
import type {
  NavPolygon,
  RampSpec,
  RampJumpSpec,
} from 'kinocat/environment';
import {
  PlanRegistry,
  AffordanceRegistry,
  asObstacle,
  constantVelocity,
  createBoostAffordance,
  createJumpAffordance,
} from 'kinocat/predict';
import type { Predict, MovingObstacle } from 'kinocat/predict';
import { within, ahead, beside, near, LEFT, RIGHT } from 'kinocat/scenario';
import type { Region, RegionAgent, ScenarioState } from 'kinocat/scenario';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import type { ObstacleDescriptor } from 'kinocat/worker';

// ---------------------------------------------------------------------------
// Palette — kept on this module so headless asserts can reference it too.

export const CARCHASE_PALETTE = {
  bg: '#0a0d14',
  fog: '#0e1320',
  ground: '#1a2233',
  asphalt: '#2a2f3a',
  curb: '#3b4252',
  building: '#3a4458',
  buildingEdge: '#6c7a94',
  ramp: '#915b3a',
  gate: '#ffd070',
  cop: 0xff5566,
  robber: 0x55dcff,
  copPath: 0xff8aa0,
  robberPath: 0x7fe9ff,
  boostPad: '#ffe066',
  boostRing: '#ffa030',
  jumpArrow: '#ffd0a0',
  highway: '#3a4054',
  start: '#55ff88',
  goal: '#ffcc33',
} as const;

// ---------------------------------------------------------------------------
// Course bounds. The drivable area is a rectangle on the XZ plane around the
// origin; the +X half holds the downtown grid + alley, the −X half holds the
// ramp/jump + drift slalom + highway loop. Building footprints fit inside.

export const CARCHASE_BOUNDS = {
  x0: -120,
  x1: 120,
  z0: -90,
  z1: 90,
} as const;

export type BuildingSpec = {
  /** XZ-plane footprint half-extents. */
  x: number;
  z: number;
  hx: number;
  hz: number;
  /** Visual height (world Y units). Plan-time collision is XZ only. */
  height: number;
};

export type DriftGateSpec = { x: number; z: number; heading: number };

export type BoostPadSpec = {
  id: string;
  x: number;
  z: number;
  /** Heading of the boost exit (radians, 0 = +X). */
  exitHeading: number;
  /** World-units pushed along `exitHeading` for the boost exit pose. */
  exitDistance: number;
};

/** Re-exported under the historical car-chase name for back-compat with the
 *  Rapier wiring + UI code that used to assume a cuboid ramp. New code can
 *  use `RampJumpSpec` directly. */
export type JumpSpec = RampJumpSpec;

export interface CarChaseCourse {
  bounds: typeof CARCHASE_BOUNDS;
  polygons: NavPolygon[];
  /** Building footprint vertex rings (CCW). The ramp is NOT in this list —
   *  it's a drivable heightfield surface in `ramps`. */
  obstacles: Array<[number, number][]>;
  buildings: BuildingSpec[];
  boostPads: BoostPadSpec[];
  /** Drivable heightfield ramps the car physically climbs. */
  ramps: RampSpec[];
  /** `BallisticJump` affordance launch/land derived from each ramp. */
  jumps: JumpSpec[];
  driftGates: DriftGateSpec[];
  /** Closed loop of waypoints the robber drives in order (XZ + heading). */
  robberLoop: Array<{ x: number; z: number; heading: number }>;
}

function box(x: number, z: number, hx: number, hz: number): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

/** Build the static car-chase course. Pure: deterministic for the same call. */
export function buildCarChaseCourse(): CarChaseCourse {
  const b = CARCHASE_BOUNDS;
  // Single drivable polygon covering the rectangle; obstacles are subtractive.
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

  // Downtown grid on the +X half. A 3×3 block layout with avenues between
  // them so the planner has to thread the streets — Reeds-Shepp shines here.
  const buildings: BuildingSpec[] = [];
  const blockHX = 12;
  const blockHZ = 12;
  const avenue = 9; // half-spacing between blocks
  const downtownX0 = 30;
  const downtownZ0 = -50;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      // Leave a narrow alley right through the middle row (j=1, i=1) by
      // squashing that block to half-Z; the robber can slip through.
      const isAlley = i === 1 && j === 1;
      const cx = downtownX0 + i * (blockHX * 2 + avenue * 2);
      const cz = downtownZ0 + j * (blockHZ * 2 + avenue * 2);
      const hx = blockHX;
      const hz = isAlley ? 4 : blockHZ;
      buildings.push({ x: cx, z: cz, hx, hz, height: 18 + ((i * 7 + j * 3) % 12) });
    }
  }

  // Two wall-like buildings forming a tight pinch on the south boundary so
  // the cops can corner the robber there.
  buildings.push({ x: 0, z: -80, hx: 12, hz: 3, height: 12 });
  buildings.push({ x: -50, z: 70, hx: 16, hz: 4, height: 14 });

  // West "industrial yard": three pillars used by the drift-gate slalom.
  buildings.push({ x: -90, z: -10, hx: 4, hz: 4, height: 8 });
  buildings.push({ x: -90, z: 20, hx: 4, hz: 4, height: 8 });
  buildings.push({ x: -70, z: 5, hx: 4, hz: 4, height: 8 });

  // South ramp: a single drivable heightfield ramp the car climbs and (with
  // the jump affordance) can leap off. Approach is from +X going -X
  // (heading = π). The shared `rampHeightSampler` shapes the actual
  // surface; here we only commit to the high-level pose + dimensions.
  const ramps: RampSpec[] = [
    {
      id: 'jump-ramp-south',
      base: { x: -40, z: -50 },
      length: 10,
      width: 8,
      height: 3,
      heading: Math.PI,
    },
  ];

  // Off-mesh jump: from the ramp crest to a touchdown ~12 m past it
  // (ballistic range at cruise — `jumpSpecFromRamp` does the math). The
  // planner may take it as a shortcut but the ramp is also drivable; cops
  // pick whichever is shorter to wherever the robber currently is. Both
  // species can use this — the robber loves it for evasion.
  //
  // NOTE: unlike the /ramp demo we DO NOT add a planner-only "gap"
  // obstacle past the ramp. The ramp lives inside a busy chase course;
  // forcing every plan that touches the southwest quadrant to either
  // detour or take the jump made cops miss the robber whenever it left
  // that quadrant. Keeping the ramp open lets the planner choose its
  // own trade-off per replan.
  const jumps: JumpSpec[] = ramps.map((r) =>
    jumpSpecFromRamp(r, { launchDist: 12 }),
  );

  // Light building inflation (0.5 m past the visual face). The real
  // chassis-vs-wall margin comes from the inflated agent footprint in
  // `CARCHASE_AGENT`; pushing this much higher closes the south pinch
  // (z = -80 wall) and the downtown alley to the point where the
  // planner can't find any path at all under budget.
  const obstacles: Array<[number, number][]> = buildings.map((b1) =>
    box(b1.x, b1.z, b1.hx + 0.5, b1.hz + 0.5),
  );

  // Boost pads — one inside the downtown grid (rewards the robber for
  // committing to the alley shortcut), one on the west highway loop.
  const boostPads: BoostPadSpec[] = [
    {
      id: 'boost-downtown',
      x: 60,
      z: 0,
      exitHeading: Math.PI,
      exitDistance: 28,
    },
    {
      id: 'boost-highway-west',
      x: -100,
      z: 60,
      exitHeading: 0,
      exitDistance: 30,
    },
  ];

  // Drift-gate slalom — pure visual decoration; the pillars above are the
  // actual collision; the gates are markers between them.
  const driftGates: DriftGateSpec[] = [
    { x: -90, z: 5, heading: Math.PI / 2 },
    { x: -80, z: 20, heading: -Math.PI / 2 },
    { x: -70, z: -10, heading: Math.PI / 2 },
  ];

  // Robber waypoint loop. Anchors picked so the loop forces the robber
  // through every feature: alley → boost → highway → drift → jump → back.
  const robberLoop = [
    { x: 50, z: 70, heading: -Math.PI / 2 }, // top of downtown
    { x: 50, z: 0, heading: -Math.PI / 2 }, // through alley
    { x: 80, z: -70, heading: -Math.PI / 2 }, // south corner
    { x: 0, z: -75, heading: Math.PI }, // south corridor west
    { x: -30, z: -50, heading: Math.PI }, // approach jump (will trigger)
    { x: -80, z: -40, heading: Math.PI / 2 }, // post-jump, head north
    { x: -100, z: 30, heading: 0 }, // highway loop boost
    { x: -50, z: 70, heading: 0 }, // back across top
  ];

  return {
    bounds: b,
    polygons,
    obstacles,
    buildings,
    boostPads,
    ramps,
    jumps,
    driftGates,
    robberLoop,
  };
}

// ---------------------------------------------------------------------------
// Agent + motion-primitive library. Smaller turn radius than the dogfight
// aircraft (4 m vs 25 m); reverse allowed because cops sometimes have to back
// out of a corner. Footprint is ~5 m long, 2.4 m wide — a realistic sedan.

export const CARCHASE_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 4.5,
  maxSpeed: 14,
  maxReverseSpeed: 5,
  // Planner footprint is INFLATED past the actual chassis (2.4 × 1.0 m
  // half-extents) so plans keep a clearance buffer from walls. Without
  // this margin a tight Reeds-Shepp curve at full speed clips the inner
  // curb during execution — the chassis touches the wall a frame before
  // the planner thinks it should and the controller wedges there with
  // no obvious recovery. ~0.25 m margin all around fixes the worst
  // wall-grinding without closing the downtown alleys (avenues are
  // ~18 m wide and we don't want to overshrink them).
  footprint: [
    [2.65, 1.25],
    [-2.65, 1.25],
    [-2.65, -1.25],
    [2.65, -1.25],
  ],
  // Reverse is only modestly more expensive than forward so the planner is
  // willing to back out of a corner instead of grinding the chassis into a
  // wall when the only path to the robber is behind it.
  reverseCostMultiplier: 1.4,
  directionChangePenalty: 0.4,
});

function buildPrimitiveLibrary(agent: VehicleAgent): MotionPrimitiveLibrary {
  const k = 1 / agent.minTurnRadius; // tightest forward turn
  const kHalf = k / 2;               // wide forward turn
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    // Three forward speeds × five curvatures plus three reverse curvatures
    // gives the planner enough variety to (a) cruise straight, (b) negotiate
    // tight corners at moderate speed, and (c) back-and-fill out of a
    // wedged-against-a-wall failure.
    controlSets: [
      // Cruise / fast straight.
      [0, 14],
      [0, 10],
      // Forward gentle turns.
      [kHalf, 12],
      [-kHalf, 12],
      // Forward tight turns at lower speed (more controllable on cuboid wheels).
      [k, 8],
      [-k, 8],
      // Slow forward straight (for tight quarters).
      [0, 5],
      // Reverse straight.
      [0, -4],
      // Reverse gentle turns — the staple of "back out of a corner".
      [kHalf, -4],
      [-kHalf, -4],
      // Reverse tight turns (for sharp three-point turns).
      [k, -3],
      [-k, -3],
    ],
    duration: 0.55,
    substeps: 5,
    startSpeeds: [0],
  });
}

export const CARCHASE_LIB = buildPrimitiveLibrary(CARCHASE_AGENT);

// ---------------------------------------------------------------------------
// Static affordance registry for the course (jump links + boost pads).

export function carChaseAffordances(course: CarChaseCourse): AffordanceRegistry {
  const reg = new AffordanceRegistry();
  for (const j of course.jumps) {
    reg.add(
      createJumpAffordance({
        id: j.id,
        launch: j.launch,
        entryRadius: 4,
        land: {
          x: j.land.x,
          z: j.land.z,
          heading: j.land.heading,
          speed: 10,
          t: 0,
        },
        apexY: j.height + 2,
        duration: 1.2,
        cost: 2,
      }),
    );
  }
  for (const p of course.boostPads) {
    const c = Math.cos(p.exitHeading);
    const s = Math.sin(p.exitHeading);
    reg.add(
      createBoostAffordance({
        id: p.id,
        pad: { x: p.x, z: p.z },
        entryRadius: 4,
        exit: {
          x: p.x + p.exitDistance * c,
          z: p.z + p.exitDistance * s,
          heading: p.exitHeading,
          speed: 13,
          t: 0,
        },
        duration: 0.8,
        cost: 0.6,
      }),
    );
  }
  return reg;
}

// ---------------------------------------------------------------------------
// Tactical layer — ground-2D port of dogfight-scenarios.ts:405 / :432. The
// planner gets a goal pose; this picks WHICH pose from observable geometry.

// ---------------------------------------------------------------------------
// Cop tactical roles. Rather than swap roles by geometry every tick (which
// made the squad's behaviour hard to read), each cop now has a FIXED,
// distinct persona — four genuinely different pursuit heuristics that are
// interesting to watch play off each other. All four are authored as
// `kinocat/scenario` goal regions (see `copGoalRegion`).
//
//   INTERCEPT (Hunter)   — pure lead-pursuit: solve the collision quadratic
//                          and drive to the point where the cop meets the
//                          robber at cruise speed. Relentless direct chase.
//                          Region: `within(lead)`.
//   CUTOFF (Blocker)     — the "cut it off" role: don't chase the meeting
//                          point, race PAST it — well ahead along the
//                          robber's travel direction — to slam a roadblock
//                          across its escape lane. Because the cop then sits
//                          on the robber's forward path (and the robber
//                          treats cops as obstacles) it forces a detour.
//                          Region: `ahead(lead, big)`.
//   CONTAIN (Shepherd)   — pull up alongside on the robber's flank (the side
//                          the cop is already nearest) to herd / pin it,
//                          denying lateral escape. Region: `beside(lead)`.
//   AMBUSH (Predator)    — the clever one: run the robber's OWN escape
//                          heuristic to predict where it will flee next, then
//                          race to that escape point and spring the trap
//                          before the robber arrives. Region: `near(escape)`.
//   PURSUE               — degenerate fallback: robber (nearly) stationary,
//                          so leading is meaningless; converge on its pose.
export type CopTacticalMode =
  | 'INTERCEPT'
  | 'CUTOFF'
  | 'CONTAIN'
  | 'AMBUSH'
  | 'PURSUE';

// The four personas, in cop-index order. Cop i takes COP_PERSONAS[i % 4].
const COP_PERSONAS: CopTacticalMode[] = ['INTERCEPT', 'CUTOFF', 'CONTAIN', 'AMBUSH'];

// Below this robber speed the lead point of any interception collapses onto
// the robber itself, so every cop just drives straight at it (PURSUE).
const LEAD_MIN_SPEED = 2.0; // m/s

/** Solve for the time at which a cop travelling at `copSpeed` intercepts a
 *  target moving at constant velocity `(vx, vz)` from relative position
 *  `(rx, rz)` = target − cop. Returns the smallest strictly-positive root of
 *  the collision quadratic, or a straight-line fallback (`dist / copSpeed`)
 *  when the target is uncatchable under the constant-velocity model (e.g.
 *  moving directly away at ≥ cop speed). This is the standard
 *  lead-pursuit / missile-intercept solve. */
function interceptTime(
  rx: number,
  rz: number,
  vx: number,
  vz: number,
  copSpeed: number,
): number {
  const a = vx * vx + vz * vz - copSpeed * copSpeed;
  const b = 2 * (rx * vx + rz * vz);
  const c = rx * rx + rz * rz;
  const dist = Math.sqrt(c);
  const fallback = copSpeed > 1e-6 ? dist / copSpeed : 6;
  // Near-linear (robber at ~cop speed): a ≈ 0 → b t + c = 0.
  if (Math.abs(a) < 1e-6) {
    if (Math.abs(b) < 1e-9) return fallback;
    const t = -c / b;
    return t > 1e-3 ? t : fallback;
  }
  const disc = b * b - 4 * a * c;
  if (disc < 0) return fallback;
  const sq = Math.sqrt(disc);
  const t1 = (-b - sq) / (2 * a);
  const t2 = (-b + sq) / (2 * a);
  // Smallest strictly-positive root.
  const cands = [t1, t2].filter((t) => t > 1e-3).sort((p, q) => p - q);
  return cands.length > 0 ? cands[0]! : fallback;
}

/** Signed lateral position of `cop` relative to the robber's travel
 *  direction: negative = to the robber's left, positive = to its right.
 *  Used to hand each supporting cop the flank it is already nearest to. */
function copSide(robber: CarKinematicState, cop: CarKinematicState): number {
  // Perpendicular (to the left) of the robber heading.
  const px = -Math.sin(robber.heading);
  const pz = Math.cos(robber.heading);
  return (cop.x - robber.x) * px + (cop.z - robber.z) * pz;
}

/** Each cop's fixed persona by index (cycled if there are >4 cops). A robber
 *  that is nearly stationary collapses every cop to PURSUE, since none of the
 *  lead-based tactics mean anything against a parked target. Kept as an
 *  array-returning function so callers can index into it once per squad. */
export function assignCopModes(
  robber: CarKinematicState,
  cops: ReadonlyArray<CarKinematicState>,
): CopTacticalMode[] {
  if (Math.abs(robber.speed) < LEAD_MIN_SPEED) return cops.map(() => 'PURSUE');
  return cops.map((_, i) => COP_PERSONAS[i % COP_PERSONAS.length]!);
}

/** Back-compat single-cop wrapper: derive the whole squad's roles from the
 *  shared geometry and return this cop's. Callers that have all cop states
 *  should prefer `assignCopModes` and index into it once. */
export function selectTacticalMode(
  robber: CarKinematicState,
  cops: ReadonlyArray<CarKinematicState>,
  copIndex: number,
): CopTacticalMode {
  return assignCopModes(robber, cops)[copIndex] ?? 'PURSUE';
}

/** Translate a cop tactic into a goal `CarKinematicState`. The planner uses a
 *  generous goalRadius so these only need to be in the right neighbourhood.
 *
 *  Goal-nudging delegates to `kinocat/environment`'s generic helper, which
 *  uses the same `NavWorld.footprintClear` predicate the planner's
 *  `checkValidity` calls — so a goal accepted by `nudgeGoalToNavClear` is
 *  guaranteed to plan. The InMemoryNavWorld is built lazily and cached per
 *  course identity. */
let cachedCourseObstacles: Array<[number, number][]> | null = null;
let cachedNavWorld: InMemoryNavWorld | null = null;
function navWorldFor(course: CarChaseCourse): InMemoryNavWorld {
  if (cachedCourseObstacles !== course.obstacles) {
    cachedNavWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
    cachedCourseObstacles = course.obstacles;
  }
  return cachedNavWorld!;
}

function nudgeGoalToNavClear(
  goal: CarKinematicState,
  near: CarKinematicState,
  buildings: BuildingSpec[] | undefined,
  course?: CarChaseCourse,
): CarKinematicState {
  // Building list is no longer needed for the check itself (the NavWorld
  // owns inflated obstacle geometry); we only keep the parameter so callers
  // pass `course.buildings` if they want nudging enabled.
  if (!buildings || !course) return goal;
  const nudged = nudgeGoalClear(goal, near, navWorldFor(course), CARCHASE_AGENT);
  return clampGoalToBounds(nudged);
}

// Tactical goal geometry (metres). Because a cop's objective is now a
// `kinocat/scenario` Region, "tune a tactic" = change a region argument, and
// the exact same object can be compiled, validated, and drawn by the Goal-Lab
// tooling (`createRegionHelper`) — that's the introspection payoff.
const COP_GOAL_R = 3; // `within` ball radius (INTERCEPT / PURSUE contact)
const COP_CUTOFF_LEAD = 13; // `ahead` roadblock distance past the lead point (CUTOFF)
const COP_CUTOFF_TOL = 3.5; // CUTOFF ball tolerance
const COP_CONTAIN_GAP = 6; // `beside` lateral pinch gap (CONTAIN)
const COP_CONTAIN_TOL = 3; // beside ball tolerance
const COP_AMBUSH_R = 4; // `near` escape-trap radius (AMBUSH)
const COP_LEAD_MAX_S = 6; // clamp on the interception lead time

/** Extra context a couple of the personas need beyond the robber prediction:
 *  the full cop roster + course, so `AMBUSH` can run the robber's own escape
 *  heuristic to predict where it will flee. */
export interface CopContext {
  cops?: CarKinematicState[];
  buildings?: BuildingSpec[];
  course?: CarChaseCourse;
}

/** The canonical GOAL REGION a cop is trying to reach, authored in the
 *  `kinocat/scenario` DSL against a *lead-shifted* view of the robber — so the
 *  region is centred on where the robber WILL be at interception, not where it
 *  is now. Returning the `Region` (not a bare pose) is what makes the tactic
 *  readable, introspectable and reusable: the same object feeds the goal-pose
 *  extraction below AND the on-screen region overlay / `validate()`.
 *
 *  We deliberately plan to `region.representative()` with the fast pose
 *  planner rather than handing the dynamic region to `planVehicleScenario`:
 *  a moving `within` goal disables the Reeds-Shepp analytic shortcut and turns
 *  every replan into a full time-lattice search (~120 ms vs ~5 ms on this
 *  course), which the 4-agent real-time loop cannot afford. The lead-shifted
 *  agent means `representative()` already carries the interception lead. */
export function copGoalRegion(
  robber: CarKinematicState,
  robberPredict: Predict<CarKinematicState>,
  cop: CarKinematicState,
  mode: CopTacticalMode,
  ctx?: CopContext,
): Region {
  const speed = CARCHASE_AGENT.maxSpeed;
  const vx = Math.cos(robber.heading) * robber.speed;
  const vz = Math.sin(robber.heading) * robber.speed;
  // PURSUE (near-stationary robber) aims at its actual pose; the rest lead.
  const tti =
    mode === 'PURSUE'
      ? 0
      : Math.min(
          COP_LEAD_MAX_S,
          Math.max(0.3, interceptTime(robber.x - cop.x, robber.z - cop.z, vx, vz, speed)),
        );
  const lead: RegionAgent = {
    id: 'robber',
    predict: (t) => (robberPredict((cop.t ?? 0) + tti + t) ?? robber) as ScenarioState,
  };
  switch (mode) {
    case 'INTERCEPT':
    case 'PURSUE':
      return within(lead, COP_GOAL_R);
    case 'CUTOFF':
      // Race PAST the meeting point, down the robber's travel direction, to
      // block its escape lane.
      return ahead(lead, COP_CUTOFF_LEAD, COP_CUTOFF_TOL);
    case 'CONTAIN': {
      // Flank on whichever side the cop is already nearest, so it slides
      // alongside instead of cutting across the robber's nose.
      const side = copSide(robber, cop) >= 0 ? LEFT : RIGHT;
      return beside(lead, side, COP_CONTAIN_GAP, COP_CONTAIN_TOL);
    }
    case 'AMBUSH': {
      // Predict where the robber will FLEE (its own escape heuristic) and set
      // a trap there. Needs the roster + course; without them, fall back to a
      // straight intercept so the persona still does something sensible.
      if (ctx?.cops && ctx.course) {
        const escape = robberGoal(
          robber,
          ctx.course.robberLoop,
          0,
          ctx.cops,
          ctx.buildings,
          ctx.course,
        ).goal;
        return near({ x: escape.x, z: escape.z }, COP_AMBUSH_R);
      }
      return within(lead, COP_GOAL_R);
    }
  }
}

/** Concrete goal pose for the fast pose planner, extracted from the tactic's
 *  canonical goal region via its `representative()` aim point. */
export function tacticalGoal(
  robber: CarKinematicState,
  robberPredict: Predict<CarKinematicState>,
  cop: CarKinematicState,
  mode: CopTacticalMode,
  ctx?: CopContext,
): CarKinematicState {
  const rep = copGoalRegion(robber, robberPredict, cop, mode, ctx).representative();
  const goal = clampGoalToBounds({
    x: rep.x,
    z: rep.z,
    heading: rep.heading,
    speed: CARCHASE_AGENT.maxSpeed,
    t: 0,
  });
  return nudgeGoalToNavClear(goal, cop, ctx?.buildings, ctx?.course);
}

// ---------------------------------------------------------------------------
// Robber AI — an actual *evader*. Instead of chasing a fixed waypoint loop
// (the old behaviour, which happily drove past cops), the robber scores a fan
// of candidate escape headings and picks the one that best trades off three
// things a fleeing driver actually cares about:
//
//   1. OPENNESS   — how far it can run down that heading before a wall / the
//                   map edge stops it. This is what keeps it from fleeing
//                   into a dead-end or pinning itself in a corner.
//   2. COP CLEAR  — how much the heading points *away* from the cops, with
//                   nearer cops weighted much more heavily. A heading that
//                   splits two pursuers (drives between them) is penalised
//                   because it points toward both.
//   3. MOMENTUM   — a mild bonus for not reversing direction every tick, so
//                   the car commits to an escape lane instead of dithering.
//   4. EDGE       — a repulsion from the arena boundary once the robber gets
//                   near it. OPENNESS alone doesn't stop edge-hugging: a
//                   heading straight at a wall stays "open" until the very
//                   wall, and fleeing a cop that's *behind* means the outward
//                   (wall-ward) heading also scores well on COP CLEAR — so the
//                   evader used to run to the boundary and (past the padded
//                   slab) off the map. This term makes it peel away first.
//
// The chosen heading is projected out to a far goal; the kinocat planner then
// finds the feasible kinodynamic path there and still routes around the cops'
// predicted trajectories (they're passed as moving obstacles).

// How far a ray probes for free space, and how far out we plant the goal.
const ROBBER_LOOK = 55; // m — raycast probe length
const ROBBER_GOAL_DIST = 38; // m — nominal goal projection distance
const ROBBER_MIN_GOAL_DIST = 10; // m — never plant the goal on our bumper
// Cop influence falls off to ~0 past this range.
const ROBBER_THREAT_RANGE = 75; // m
// Extra clearance kept from inflated building faces when probing openness.
const ROBBER_WALL_MARGIN = 3.5; // m
// Number of candidate headings in the fan.
const ROBBER_RAYS = 24;
// Score weights. Cop-avoidance is *coupled* to openness (see scoring below):
// a heading you can't actually drive down is worthless as an escape no matter
// how far it points from the cops, so it can't lure the robber into a wall.
const ROBBER_W_OPEN = 1.2;
const ROBBER_W_COP = 1.3;
const ROBBER_W_MOM = 0.35;
// Start peeling away from a boundary once within this distance of it, and how
// hard. Unlike the cop term this is NOT gated by openness, so it fires even
// though there is technically open pavement up to the wall.
const ROBBER_EDGE_BUFFER = 34; // m
const ROBBER_W_EDGE = 2.2;

/** Distance from `(x,z)` along unit direction `(dx,dz)` until the ray leaves
 *  the course bounds or enters an inflated building footprint. Capped at
 *  `maxDist`. Pure ray-vs-AABB slab tests — cheap enough to run for every
 *  candidate heading every replan. */
function robberFreeDistance(
  x: number,
  z: number,
  dx: number,
  dz: number,
  course: CarChaseCourse,
  maxDist: number,
): number {
  let best = maxDist;
  const b = CARCHASE_BOUNDS;
  const m = 4; // matches clampGoalToBounds margin
  // Bounds: distance to each wall the ray is heading toward.
  if (dx > 1e-6) best = Math.min(best, (b.x1 - m - x) / dx);
  else if (dx < -1e-6) best = Math.min(best, (b.x0 + m - x) / dx);
  if (dz > 1e-6) best = Math.min(best, (b.z1 - m - z) / dz);
  else if (dz < -1e-6) best = Math.min(best, (b.z0 + m - z) / dz);
  best = Math.max(0, best);

  for (const bld of course.buildings) {
    const hx = bld.hx + ROBBER_WALL_MARGIN;
    const hz = bld.hz + ROBBER_WALL_MARGIN;
    // Slab test for the inflated box centred at (bld.x, bld.z).
    let tmin = 0;
    let tmax = best;
    // X slab.
    if (Math.abs(dx) < 1e-6) {
      if (x < bld.x - hx || x > bld.x + hx) continue; // parallel & outside
    } else {
      let t1 = (bld.x - hx - x) / dx;
      let t2 = (bld.x + hx - x) / dx;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) continue;
    }
    // Z slab.
    if (Math.abs(dz) < 1e-6) {
      if (z < bld.z - hz || z > bld.z + hz) continue;
    } else {
      let t1 = (bld.z - hz - z) / dz;
      let t2 = (bld.z + hz - z) / dz;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) continue;
    }
    // Ray enters the box at tmin ≥ 0 within [0,best] → openness stops there.
    if (tmax >= 0 && tmin >= 0 && tmin < best) best = tmin;
  }
  return Math.max(0, best);
}

/** Pick the robber's escape goal by scoring a fan of candidate headings.
 *  `loop` / `loopIndex` are retained for signature compatibility (the robber
 *  no longer follows the waypoint loop); `nextIndex` is echoed back unused. */
// Radius of the robber's `near` escape-goal region (also its on-screen ring).
const ROBBER_GOAL_R = 4;

export function robberGoal(
  robber: CarKinematicState,
  loop: CarChaseCourse['robberLoop'],
  loopIndex: number,
  cops: CarKinematicState[],
  buildings?: BuildingSpec[],
  course?: CarChaseCourse,
): { goal: CarKinematicState; nextIndex: number; region: Region } {
  // Without course geometry we can't probe openness — fall back to fleeing
  // directly away from the weighted cop centroid.
  if (!course) {
    let ax = 0;
    let az = 0;
    for (const c of cops) {
      const dx = robber.x - c.x;
      const dz = robber.z - c.z;
      const d = Math.max(1, Math.hypot(dx, dz));
      ax += dx / (d * d);
      az += dz / (d * d);
    }
    const mag = Math.hypot(ax, az) || 1;
    const goal = clampGoalToBounds({
      x: robber.x + (ax / mag) * ROBBER_GOAL_DIST,
      z: robber.z + (az / mag) * ROBBER_GOAL_DIST,
      heading: Math.atan2(az, ax),
      speed: CARCHASE_AGENT.maxSpeed,
      t: 0,
    });
    return { goal, nextIndex: loopIndex, region: near({ x: goal.x, z: goal.z }, ROBBER_GOAL_R) };
  }

  let bestScore = -Infinity;
  let bestAngle = robber.heading;
  let bestOpen = ROBBER_MIN_GOAL_DIST;

  for (let i = 0; i < ROBBER_RAYS; i++) {
    const ang = (i / ROBBER_RAYS) * 2 * Math.PI;
    const dx = Math.cos(ang);
    const dz = Math.sin(ang);
    const open = robberFreeDistance(robber.x, robber.z, dx, dz, course, ROBBER_LOOK);

    // Cop term: sum of how much this heading points AT each cop, weighted by
    // closeness. Positive dot = toward the cop = bad.
    let copTerm = 0;
    for (const c of cops) {
      const cx = c.x - robber.x;
      const cz = c.z - robber.z;
      const cd = Math.hypot(cx, cz);
      if (cd < 1e-3 || cd > ROBBER_THREAT_RANGE) continue;
      const toward = (dx * cx + dz * cz) / cd; // cos(angle to cop) ∈ [-1,1]
      const w = 1 - cd / ROBBER_THREAT_RANGE; // nearer cops dominate
      copTerm += w * toward;
    }

    // Edge term: how much this heading points toward a nearby arena wall,
    // weighted by how close that wall is (0 when > ROBBER_EDGE_BUFFER away).
    // Summed over the four walls so a corner repels on both axes.
    const b = CARCHASE_BOUNDS;
    let edgeTerm = 0;
    const west = Math.max(0, 1 - (robber.x - b.x0) / ROBBER_EDGE_BUFFER);
    const east = Math.max(0, 1 - (b.x1 - robber.x) / ROBBER_EDGE_BUFFER);
    const south = Math.max(0, 1 - (robber.z - b.z0) / ROBBER_EDGE_BUFFER);
    const north = Math.max(0, 1 - (b.z1 - robber.z) / ROBBER_EDGE_BUFFER);
    edgeTerm += west * Math.max(0, -dx); // heading has a −x (west-ward) part
    edgeTerm += east * Math.max(0, dx);
    edgeTerm += south * Math.max(0, -dz);
    edgeTerm += north * Math.max(0, dz);

    const momentum = Math.cos(ang - robber.heading);
    // Openness gates the cop-clearance judgement: the value of running away
    // from the cops down this heading is proportional to how far we can
    // actually run. A wall-facing heading (open ≈ 0) therefore scores ≈
    // momentum only. The edge term is applied on top (ungated) so the evader
    // turns away from the boundary *before* it gets pinned against it.
    const openFrac = open / ROBBER_LOOK;
    const score =
      openFrac * (ROBBER_W_OPEN - ROBBER_W_COP * copTerm) +
      ROBBER_W_MOM * momentum -
      ROBBER_W_EDGE * edgeTerm;

    if (score > bestScore) {
      bestScore = score;
      bestAngle = ang;
      bestOpen = open;
    }
  }

  // Plant the goal along the winning heading, but never past the free
  // distance we just measured (minus a margin) so it stays reachable.
  const dist = Math.max(
    ROBBER_MIN_GOAL_DIST,
    Math.min(ROBBER_GOAL_DIST, bestOpen - ROBBER_WALL_MARGIN),
  );
  const clamped = clampGoalToBounds({
    x: robber.x + Math.cos(bestAngle) * dist,
    z: robber.z + Math.sin(bestAngle) * dist,
    heading: bestAngle,
    speed: CARCHASE_AGENT.maxSpeed,
    t: 0,
  });
  const goal = nudgeGoalToNavClear(clamped, robber, buildings, course);
  return {
    goal,
    nextIndex: loopIndex,
    // Canonical goal region (introspection / overlay); the evader is a
    // "reach this open escape point" objective in the scenario DSL.
    region: near({ x: goal.x, z: goal.z }, ROBBER_GOAL_R),
  };
}

/** Pull the goal pose back inside the planning rectangle. Keeps a small
 *  margin so the agent's footprint still fits on the walkable mesh. */
function clampGoalToBounds(s: CarKinematicState): CarKinematicState {
  const m = 4;
  const b = CARCHASE_BOUNDS;
  return {
    ...s,
    x: Math.max(b.x0 + m, Math.min(b.x1 - m, s.x)),
    z: Math.max(b.z0 + m, Math.min(b.z1 - m, s.z)),
  };
}

// ---------------------------------------------------------------------------
// Per-tick planning helper. Wraps VehicleEnvironment + TimeAwareEnvironment +
// PlanRegistry + AffordanceRegistry. One call per AI per replan slot.

export interface CarChasePlanRequest {
  /** Stable id; matches the PlanRegistry key. */
  npcId: string;
  state: CarKinematicState;
  goal: CarKinematicState;
  /** Other agents whose published plans should be treated as moving
   *  obstacles (sibling cops + the robber/player for cop calls; the cops for
   *  robber calls). The robber state shows up as a Predict; sibling NPC
   *  states come from `registry`. */
  movingObstacles: MovingObstacle[];
  /** Shared kinocat plan registry; this AI's plan is NOT auto-published. */
  registry: PlanRegistry;
  /** Course (for the NavWorld + affordances). */
  course: CarChaseCourse;
  /** Anytime budget (wall clock ms). */
  deadlineMs?: number;
  maxExpansions?: number;
}

// Moving-obstacle inflation radii (metres) for the multi-agent plans.
//   • ROBBER_SEE_COP_R — a fleeing robber routes generously around the cops.
//   • COP_SEE_ROBBER_R — a chasing cop treats the robber as barely an
//     obstacle so its plan drives right up to the arrest instead of orbiting
//     a fat exclusion disc around the target (the old radius left the cop
//     unable to close the final few metres, so captures only ever happened
//     via the dumb-pursuit fallback).
//   • COP_COP_R — cops still keep clear of each other.
export const ROBBER_SEE_COP_R = 2.6;
export const COP_SEE_ROBBER_R = 0.6;
export const COP_COP_R = 2.6;

export const CARCHASE_REPLAN_BUDGET_MS = 120;
export const CARCHASE_MAX_EXPANSIONS = 25000;
/** Test budget — generous because the test runs one shot, not interactive. */
export const CARCHASE_TEST_MAX_EXPANSIONS = 25000;

/** Single-shot planning call for one car-chase AI. Thin wrapper around
 *  `kinocat/planner`'s `planVehicleOnce` that pins the course-specific
 *  affordances + 10 m affordance proximity. Everything else (env tuning,
 *  agent radius for the moving-obstacle test, broadphase) comes from the
 *  core defaults. */
export function planCarChaseAI(
  req: CarChasePlanRequest,
): PlanResult<CarKinematicState> {
  return planVehicleOnce({
    start: req.state,
    goal: req.goal,
    world: new InMemoryNavWorld(req.course.polygons, req.course.obstacles),
    agent: CARCHASE_AGENT,
    lib: CARCHASE_LIB,
    movingObstacles: req.movingObstacles,
    affordances: carChaseAffordances(req.course),
    timeOptions: { affordanceRadius: 10 },
    deadlineMs: req.deadlineMs ?? CARCHASE_REPLAN_BUDGET_MS,
    maxExpansions: req.maxExpansions ?? CARCHASE_MAX_EXPANSIONS,
  });
}

// ---------------------------------------------------------------------------
// Obstacle dehydration — extracts the data backing a MovingObstacle closure
// into a plain-data ObstacleDescriptor for structured-clone transfer to a
// Web Worker. The worker rehydrates via `rehydrateObstacle()`.

export function dehydrateObstacle(
  npcId: string,
  registry: PlanRegistry,
  fallbackState: CarKinematicState,
  radius: number,
  horizon = 4,
): ObstacleDescriptor {
  const pub = registry.get(npcId);
  if (pub && pub.states.length > 0) {
    return { kind: 'plan', path: pub.states as CarKinematicState[], radius };
  }
  return { kind: 'cv', state: fallbackState, horizon, radius };
}

// ---------------------------------------------------------------------------
// Predict the robber's trajectory for cops to plan against. The robber is a
// (mostly) cooperative AI that DOES publish to the registry — when it does,
// cops read its plan directly. When the robber has no plan yet, fall back to
// constant-velocity.

export function predictRobberFromState(
  robber: CarKinematicState,
  horizon = 4,
): Predict<CarKinematicState> {
  return constantVelocity(robber, horizon);
}

// ---------------------------------------------------------------------------
// Headless deterministic snapshot — what the scenario test asserts. Spawns
// the robber + N cops at canonical positions, picks a tactic + goal per cop,
// and runs one plan per AI. The test asserts every AI returns a plan within
// the (one-shot, generous) budget.

export interface CarChaseSnapshot {
  course: CarChaseCourse;
  robber: {
    state: CarKinematicState;
    goal: CarKinematicState;
    loopIndex: number;
    result: PlanResult<CarKinematicState>;
  };
  cops: Array<{
    id: string;
    state: CarKinematicState;
    mode: CopTacticalMode;
    goal: CarKinematicState;
    result: PlanResult<CarKinematicState>;
  }>;
}

const SPAWN_ROBBER: CarKinematicState = {
  x: 50,
  z: 70,
  heading: -Math.PI / 2,
  speed: 0,
  t: 0,
};

const SPAWN_COPS: CarKinematicState[] = [
  { x: 95, z: 70, heading: Math.PI, speed: 0, t: 0 }, // NE — Hunter
  { x: 50, z: -80, heading: Math.PI / 2, speed: 0, t: 0 }, // S — Blocker
  { x: -100, z: 0, heading: 0, speed: 0, t: 0 }, // W — Shepherd
  { x: -80, z: 75, heading: 0, speed: 0, t: 0 }, // NW — Ambusher
];

export function spawnPoses() {
  return { robber: SPAWN_ROBBER, cops: SPAWN_COPS.slice() };
}

export function buildCarChaseSnapshot(): CarChaseSnapshot {
  const course = buildCarChaseCourse();
  const registry = new PlanRegistry();

  // Step 1: plan the robber first. The robber has no sibling-cop plans yet
  // (registry empty), so its moving-obstacle list is just constant-velocity
  // predictions of the cops as opponents.
  const robberCopPreds: MovingObstacle[] = SPAWN_COPS.map((c) =>
    asObstacle(constantVelocity(c, 4), 2.6),
  );
  const robberPick = robberGoal(
    SPAWN_ROBBER,
    course.robberLoop,
    0,
    SPAWN_COPS,
    course.buildings,
    course,
  );
  const robberResult = planCarChaseAI({
    npcId: 'robber',
    state: SPAWN_ROBBER,
    goal: robberPick.goal,
    movingObstacles: robberCopPreds,
    registry,
    course,
    // 250 ms wall budget is plenty for the spawn matchup and keeps the
    // test file under the vitest worker RPC timeout when combined with the
    // other slow scenarios (dogfight + swarm) in this run.
    deadlineMs: 250,
    maxExpansions: CARCHASE_TEST_MAX_EXPANSIONS,
  });
  if (robberResult.found) registry.publish('robber', robberResult.path);

  // Step 2: plan each cop. They see the robber's published plan as a moving
  // obstacle (so they intercept its FUTURE position, not its current one) and
  // also see sibling cop plans (so two cops don't take the same route).
  const cops: CarChaseSnapshot['cops'] = [];
  for (let i = 0; i < SPAWN_COPS.length; i++) {
    const id = `cop${i}`;
    const cop = SPAWN_COPS[i]!;
    const mode = selectTacticalMode(SPAWN_ROBBER, SPAWN_COPS, i);
    // Predict the robber from its published plan when available; fall back
    // to constant-velocity. Either way, asObstacle wraps it.
    const robberPredict: Predict<CarKinematicState> = (t) => {
      const fromPlan = registry.predictNPC('robber')(t) as CarKinematicState | null;
      return fromPlan ?? predictRobberFromState(SPAWN_ROBBER, 4)(t);
    };
    const goal = tacticalGoal(SPAWN_ROBBER, robberPredict, cop, mode, {
      cops: SPAWN_COPS,
      buildings: course.buildings,
      course,
    });
    const siblingIds = SPAWN_COPS.map((_, j) => `cop${j}`).filter(
      (_, j) => j !== i,
    );
    const obstacles: MovingObstacle[] = [
      // Robber is barely an obstacle to a cop — we WANT the cop to drive all
      // the way in for the arrest, not swerve around it (see COP_SEE_ROBBER_R).
      asObstacle(robberPredict, COP_SEE_ROBBER_R),
      ...siblingIds.map((sid) =>
        asObstacle(
          registry.predictNPC(sid) as Predict<{ x: number; z: number }>,
          COP_COP_R,
        ),
      ),
    ];
    const result = planCarChaseAI({
      npcId: id,
      state: cop,
      goal,
      movingObstacles: obstacles,
      registry,
      course,
      deadlineMs: Infinity,
      maxExpansions: CARCHASE_TEST_MAX_EXPANSIONS,
    });
    if (result.found) registry.publish(id, result.path);
    cops.push({ id, state: cop, mode, goal, result });
  }

  return {
    course,
    robber: {
      state: SPAWN_ROBBER,
      goal: robberPick.goal,
      loopIndex: robberPick.nextIndex,
      result: robberResult,
    },
    cops,
  };
}
