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

export type CopTacticalMode =
  | 'PURSUE'
  | 'INTERCEPT'
  | 'FLANK_LEFT'
  | 'FLANK_RIGHT'
  | 'CUTOFF'
  | 'REGROUP';

function wrapPi(a: number): number {
  let r = ((a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/** Cop tactic from robber-relative geometry. Spread by cop index so the
 *  squad fans out instead of stacking.
 *
 *  Two important "collapse to PURSUE" cases:
 *  - **Slow robber.** INTERCEPT/CUTOFF/FLANK lead the robber along its
 *    *current* heading. If the robber is stationary the lead point is just
 *    empty space ahead of it — the cop drives there and parks, never
 *    doubling back. Below `LEAD_MIN_SPEED` ignore tactics and head
 *    straight at the robber.
 *  - **Cop already close.** Once a cop is within `CLOSE_RANGE`, leading
 *    by 10–20 m past the robber actively makes things worse (the cop
 *    overshoots its target and drives away). Collapse to PURSUE so the
 *    closing cop commits to the arrest. */
const LEAD_MIN_SPEED = 3.0; // m/s — robber slower than this → pursue
const CLOSE_RANGE = 18; // m — cop closer than this → pursue regardless

export function selectTacticalMode(
  robber: CarKinematicState,
  cop: CarKinematicState,
  copIndex: number,
): CopTacticalMode {
  const dx = robber.x - cop.x;
  const dz = robber.z - cop.z;
  const dist = Math.hypot(dx, dz);
  // Bearing from cop to robber (signed off cop's nose).
  const bearing = wrapPi(Math.atan2(dz, dx) - cop.heading);
  // Stationary / slow robber → tactics with a lead distance just park the
  // cop in empty space ahead of where the robber WAS heading. Pursue
  // directly so the cop always converges on the robber's actual pose.
  if (Math.abs(robber.speed) < LEAD_MIN_SPEED) return 'PURSUE';
  // Already close — commit to the arrest, don't try to lead past it.
  if (dist < CLOSE_RANGE) return 'PURSUE';
  // Far away — regroup at a wider arc.
  if (dist > 80) return 'REGROUP';
  // Robber inside cop's forward cone at close range → just chase.
  if (dist < 30 && Math.abs(bearing) < Math.PI / 6) return 'PURSUE';
  // Mid distance — pick INTERCEPT for cop 0, CUTOFF (ahead-of) for cop 1,
  // flanks for the rest. Gives every cop a distinct goal so the registry's
  // moving-obstacle test doesn't force them onto one route.
  if (dist > 45) {
    if (copIndex === 0) return 'INTERCEPT';
    if (copIndex === 1) return 'CUTOFF';
  }
  return copIndex % 2 === 0 ? 'FLANK_LEFT' : 'FLANK_RIGHT';
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

export function tacticalGoal(
  robber: CarKinematicState,
  robberPredict: Predict<CarKinematicState>,
  cop: CarKinematicState,
  mode: CopTacticalMode,
  buildings?: BuildingSpec[],
  course?: CarChaseCourse,
): CarKinematicState {
  const ahead = (t: number) => robberPredict(t) ?? robber;
  const dist = Math.hypot(robber.x - cop.x, robber.z - cop.z);
  const eta = Math.min(6, Math.max(0.8, dist / CARCHASE_AGENT.maxSpeed));
  const future = ahead(cop.t + eta);
  const speed = CARCHASE_AGENT.maxSpeed;
  const base = { speed, t: 0 };

  // Lead distance scales with how fast the robber is actually moving.
  // At cruise (≥8 m/s) we use the full nominal lead; below that we
  // smoothly collapse toward 0 so a coasting robber doesn't leave the
  // cop chasing empty pavement. (`selectTacticalMode` already promotes
  // very-slow robbers to PURSUE; this handles the in-between band.)
  const leadFrac = Math.min(1, Math.abs(robber.speed) / 8);

  let goal: CarKinematicState;
  switch (mode) {
    case 'PURSUE': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      goal = {
        ...base,
        x: future.x - 8 * c * leadFrac,
        z: future.z - 8 * s * leadFrac,
        heading: future.heading,
      };
      break;
    }
    case 'INTERCEPT': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      const lead = 10 * leadFrac;
      goal = {
        ...base,
        x: future.x + lead * c,
        z: future.z + lead * s,
        heading: future.heading + Math.PI,
      };
      break;
    }
    case 'CUTOFF': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      const lead = 20 * leadFrac;
      goal = {
        ...base,
        x: future.x + lead * c,
        z: future.z + lead * s,
        heading: future.heading,
      };
      break;
    }
    case 'FLANK_LEFT':
    case 'FLANK_RIGHT': {
      const sign = mode === 'FLANK_LEFT' ? -1 : 1;
      const phi = future.heading + (sign * Math.PI) / 2;
      const c = Math.cos(phi);
      const s = Math.sin(phi);
      // Flanks still keep a lateral offset (otherwise all cops stack on
      // the robber), but shrink with robber speed too.
      const lateral = 14 * Math.max(0.35, leadFrac);
      goal = {
        ...base,
        x: future.x + lateral * c,
        z: future.z + lateral * s,
        heading: future.heading,
      };
      break;
    }
    case 'REGROUP': {
      // Approach from a wide arc behind the robber. Distance is small enough
      // to stay inside the course bounds even when the robber is near a wall.
      const c = Math.cos(robber.heading);
      const s = Math.sin(robber.heading);
      goal = {
        ...base,
        x: robber.x - 25 * c,
        z: robber.z - 25 * s,
        heading: robber.heading,
      };
      break;
    }
  }
  const clamped = clampGoalToBounds(goal);
  return nudgeGoalToNavClear(clamped, cop, buildings, course);
}

// ---------------------------------------------------------------------------
// Robber AI — picks the next waypoint on the course loop. Adds a small bias
// AWAY from the nearest cop's predicted position so it isn't a fixed track.

// Don't aim at a waypoint that has a cop sitting on top of it — driving
// straight into the arrest is the robber's worst move. If the *next*
// scheduled waypoint is inside this radius of any cop, skip forward
// through the loop until we find one that isn't.
const ROBBER_AVOID_COP_RADIUS = 22; // m
// Cap waypoint look-ahead so an unlucky cop placement (cops surround the
// whole loop) doesn't infinite-loop. After this many skips we accept the
// blocked waypoint and rely on the planner's moving-obstacle avoidance.
const ROBBER_MAX_WP_SKIPS = 3;

function waypointBlockedByCop(
  wp: { x: number; z: number },
  cops: ReadonlyArray<CarKinematicState>,
  radius: number,
): boolean {
  for (const c of cops) {
    const dx = wp.x - c.x;
    const dz = wp.z - c.z;
    if (dx * dx + dz * dz < radius * radius) return true;
  }
  return false;
}

export function robberGoal(
  robber: CarKinematicState,
  loop: CarChaseCourse['robberLoop'],
  loopIndex: number,
  cops: CarKinematicState[],
  buildings?: BuildingSpec[],
  course?: CarChaseCourse,
): { goal: CarKinematicState; nextIndex: number } {
  // 1. Advance to the next waypoint well before reaching the current one.
  //    At maxSpeed 14 m/s, 20 m gives ~1.4 s of lead time — enough for a
  //    full replan cycle (~440 ms worst case) so the robber never pauses.
  const ROBBER_WP_ADVANCE_DIST = 20;
  const wp = loop[loopIndex]!;
  const reach = Math.hypot(robber.x - wp.x, robber.z - wp.z);
  let useIdx = reach < ROBBER_WP_ADVANCE_DIST ? (loopIndex + 1) % loop.length : loopIndex;

  // 2. Skip forward through the loop past any waypoint a cop is camping
  //    on. Bounded by ROBBER_MAX_WP_SKIPS so we don't spin forever when
  //    the squad surrounds the loop.
  for (let skip = 0; skip < ROBBER_MAX_WP_SKIPS; skip++) {
    const candidate = loop[useIdx]!;
    if (!waypointBlockedByCop(candidate, cops, ROBBER_AVOID_COP_RADIUS)) break;
    useIdx = (useIdx + 1) % loop.length;
  }
  const target = loop[useIdx]!;

  // 3. Compute a per-tick avoidance offset: project the (robber - cop)
  //    vector onto the perpendicular of the target heading, signed so we
  //    nudge AWAY from the cop. Clamped to ±5 m so the offset stays
  //    inside the downtown avenue (avenue half-width is ~8 m after
  //    building inflation) and never pushes the goal into a wall.
  let nearestDx = 0;
  let nearestDz = 0;
  let nearestD2 = Infinity;
  for (const c of cops) {
    const dx = robber.x - c.x;
    const dz = robber.z - c.z;
    const d2 = dx * dx + dz * dz;
    if (d2 < nearestD2) {
      nearestD2 = d2;
      nearestDx = dx;
      nearestDz = dz;
    }
  }
  const perpC = Math.cos(target.heading + Math.PI / 2);
  const perpS = Math.sin(target.heading + Math.PI / 2);
  let offset = 0;
  if (nearestD2 < 50 * 50) {
    const dot = nearestDx * perpC + nearestDz * perpS;
    offset = Math.max(-5, Math.min(5, dot));
  }
  const clamped = clampGoalToBounds({
    x: target.x + offset * perpC,
    z: target.z + offset * perpS,
    heading: target.heading,
    speed: CARCHASE_AGENT.maxSpeed,
    t: 0,
  });
  return {
    // `nudgeGoalToNavClear` pushes the goal out of any inflated obstacle
    // footprint — combined with the cop-skip above, the robber should
    // never plan straight at a wall OR at a cop.
    goal: nudgeGoalToNavClear(clamped, robber, buildings, course),
    nextIndex: useIdx,
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
  { x: 95, z: 70, heading: Math.PI, speed: 0, t: 0 },
  { x: 50, z: -80, heading: Math.PI / 2, speed: 0, t: 0 },
  { x: -100, z: 0, heading: 0, speed: 0, t: 0 },
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
    const mode = selectTacticalMode(SPAWN_ROBBER, cop, i);
    // Predict the robber from its published plan when available; fall back
    // to constant-velocity. Either way, asObstacle wraps it.
    const robberPredict: Predict<CarKinematicState> = (t) => {
      const fromPlan = registry.predictNPC('robber')(t) as CarKinematicState | null;
      return fromPlan ?? predictRobberFromState(SPAWN_ROBBER, 4)(t);
    };
    const goal = tacticalGoal(SPAWN_ROBBER, robberPredict, cop, mode, course.buildings, course);
    const siblingIds = SPAWN_COPS.map((_, j) => `cop${j}`).filter(
      (_, j) => j !== i,
    );
    const obstacles: MovingObstacle[] = [
      asObstacle(robberPredict, 2.6),
      ...siblingIds.map((sid) =>
        asObstacle(
          registry.predictNPC(sid) as Predict<{ x: number; z: number }>,
          2.6,
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
