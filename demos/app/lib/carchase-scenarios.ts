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
import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  VehicleEnvironment,
  TimeAwareEnvironment,
} from 'kinocat/environment';
import type { NavPolygon } from 'kinocat/environment';
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
import type { VehicleAgent, VehicleState } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';

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

export type JumpSpec = {
  id: string;
  /** Launch ramp midpoint (XZ). */
  launch: { x: number; z: number };
  /** Landing pose (heading kept from approach). */
  land: { x: number; z: number; heading: number };
  /** Ramp body half-extents (used to render + collide). */
  hx: number;
  hz: number;
  /** Ramp peak height. */
  height: number;
  /** Approach heading (radians, +X = 0). */
  heading: number;
};

export interface CarChaseCourse {
  bounds: typeof CARCHASE_BOUNDS;
  polygons: NavPolygon[];
  /** Building footprint vertex rings (CCW). */
  obstacles: Array<[number, number][]>;
  buildings: BuildingSpec[];
  boostPads: BoostPadSpec[];
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

  // The jump ramp is a building footprint too (cars cannot drive THROUGH the
  // ramp at plan-time — they use the jump affordance to leap over the gap).
  const jumpRampX = -40;
  const jumpRampZ = -50;
  const jumpHeading = Math.PI; // approach from +X going -X
  buildings.push({ x: jumpRampX, z: jumpRampZ, hx: 4, hz: 7, height: 5 });

  const obstacles: Array<[number, number][]> = buildings.map((b1) =>
    box(b1.x, b1.z, b1.hx + 0.5, b1.hz + 0.5),
  );

  // Off-mesh jump: from just east of the ramp to just west of the landing,
  // skipping over the ramp footprint. Costs a couple of seconds; usable by
  // both species (robber loves it for evasion).
  const jumps: JumpSpec[] = [
    {
      id: 'jump-ramp-south',
      launch: { x: jumpRampX + 12, z: jumpRampZ },
      land: { x: jumpRampX - 12, z: jumpRampZ, heading: jumpHeading },
      hx: 4,
      hz: 7,
      height: 5,
      heading: jumpHeading,
    },
  ];

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
  footprint: [
    [2.4, 1.0],
    [-2.4, 1.0],
    [-2.4, -1.0],
    [2.4, -1.0],
  ],
  reverseCostMultiplier: 2,
  directionChangePenalty: 0.6,
});

function buildPrimitiveLibrary(agent: VehicleAgent): MotionPrimitiveLibrary {
  const k = 1 / agent.minTurnRadius;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 10],
      [k, 10],
      [-k, 10],
      [k / 2, 12],
      [-k / 2, 12],
      [0, 14],
      [0, -4],
      [k, -4],
      [-k, -4],
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
 *  squad fans out instead of stacking. */
export function selectTacticalMode(
  robber: VehicleState,
  cop: VehicleState,
  copIndex: number,
): CopTacticalMode {
  const dx = robber.x - cop.x;
  const dz = robber.z - cop.z;
  const dist = Math.hypot(dx, dz);
  // Bearing from cop to robber (signed off cop's nose).
  const bearing = wrapPi(Math.atan2(dz, dx) - cop.heading);
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

/** Translate a cop tactic into a goal `VehicleState`. The planner uses a
 *  generous goalRadius so these only need to be in the right neighbourhood. */
/** True if (x,z) is inside any building footprint, inflated by `inflate`. */
function pointInAnyBuilding(
  x: number,
  z: number,
  buildings: BuildingSpec[],
  inflate: number,
): boolean {
  for (const b of buildings) {
    if (
      x >= b.x - b.hx - inflate &&
      x <= b.x + b.hx + inflate &&
      z >= b.z - b.hz - inflate &&
      z <= b.z + b.hz + inflate
    ) {
      return true;
    }
  }
  return false;
}

/** Walk the goal back along the (cop → goal) ray in `step` increments until
 *  it sits clear of every building. Bounded by `maxSteps` so a degenerate
 *  layout can never spin forever — falls back to the cop's own pose. */
function nudgeGoalClear(
  goal: VehicleState,
  cop: VehicleState,
  buildings: BuildingSpec[],
): VehicleState {
  const inflate = 2; // ~footprint half-length, keeps the goal pose clear too.
  if (!pointInAnyBuilding(goal.x, goal.z, buildings, inflate)) return goal;
  const dx = cop.x - goal.x;
  const dz = cop.z - goal.z;
  const total = Math.hypot(dx, dz);
  if (total < 1e-6) return cop;
  const ux = dx / total;
  const uz = dz / total;
  const step = 4;
  const maxSteps = Math.max(4, Math.ceil(total / step));
  for (let i = 1; i <= maxSteps; i++) {
    const nx = goal.x + ux * step * i;
    const nz = goal.z + uz * step * i;
    if (!pointInAnyBuilding(nx, nz, buildings, inflate)) {
      return clampGoalToBounds({ ...goal, x: nx, z: nz });
    }
  }
  // Last resort — at least the cop's own pose is collision-free.
  return { ...goal, x: cop.x, z: cop.z };
}

export function tacticalGoal(
  robber: VehicleState,
  robberPredict: Predict<VehicleState>,
  cop: VehicleState,
  mode: CopTacticalMode,
  buildings?: BuildingSpec[],
): VehicleState {
  const ahead = (t: number) => robberPredict(t) ?? robber;
  const dist = Math.hypot(robber.x - cop.x, robber.z - cop.z);
  const eta = Math.min(6, Math.max(0.8, dist / CARCHASE_AGENT.maxSpeed));
  const future = ahead(cop.t + eta);
  const speed = CARCHASE_AGENT.maxSpeed;
  const base = { speed, t: 0 };

  let goal: VehicleState;
  switch (mode) {
    case 'PURSUE': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      goal = {
        ...base,
        x: future.x - 8 * c,
        z: future.z - 8 * s,
        heading: future.heading,
      };
      break;
    }
    case 'INTERCEPT': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      goal = {
        ...base,
        x: future.x + 10 * c,
        z: future.z + 10 * s,
        heading: future.heading + Math.PI,
      };
      break;
    }
    case 'CUTOFF': {
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      goal = {
        ...base,
        x: future.x + 20 * c,
        z: future.z + 20 * s,
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
      goal = {
        ...base,
        x: future.x + 14 * c,
        z: future.z + 14 * s,
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
  return buildings ? nudgeGoalClear(clamped, cop, buildings) : clamped;
}

// ---------------------------------------------------------------------------
// Robber AI — picks the next waypoint on the course loop. Adds a small bias
// AWAY from the nearest cop's predicted position so it isn't a fixed track.

export function robberGoal(
  robber: VehicleState,
  loop: CarChaseCourse['robberLoop'],
  loopIndex: number,
  cops: VehicleState[],
  buildings?: BuildingSpec[],
): { goal: VehicleState; nextIndex: number } {
  // Advance to the next waypoint once close enough.
  const wp = loop[loopIndex]!;
  const reach = Math.hypot(robber.x - wp.x, robber.z - wp.z);
  const useIdx = reach < 8 ? (loopIndex + 1) % loop.length : loopIndex;
  const target = loop[useIdx]!;

  // Compute a per-tick avoidance offset: vector away from the closest cop,
  // clamped to ±12 m perpendicular to the target heading. Small enough not to
  // derail the loop, big enough to make the path interesting.
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
    // Project the away-vector onto the perpendicular of the target heading.
    // Clamp to ±5 m so the offset stays inside the downtown avenue (avenue
    // half-width is ~8 m after building inflation) and never pushes the goal
    // into a building wall.
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
    goal: buildings ? nudgeGoalClear(clamped, robber, buildings) : clamped,
    nextIndex: useIdx,
  };
}

/** Pull the goal pose back inside the planning rectangle. Keeps a small
 *  margin so the agent's footprint still fits on the walkable mesh. */
function clampGoalToBounds(s: VehicleState): VehicleState {
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
  state: VehicleState;
  goal: VehicleState;
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

// Re-use a single NavWorld per planning call would also work, but
// rebuilding is cheap (no obstacles change at runtime here) and avoids a
// hidden cross-NPC cache. For test determinism, the world IS rebuilt.
function buildPlanningWorld(course: CarChaseCourse): InMemoryNavWorld {
  return new InMemoryNavWorld(course.polygons, course.obstacles);
}

export const CARCHASE_REPLAN_BUDGET_MS = 120;
export const CARCHASE_MAX_EXPANSIONS = 25000;
/** Test budget — generous because the test runs one shot, not interactive. */
export const CARCHASE_TEST_MAX_EXPANSIONS = 80000;

export function planCarChaseAI(
  req: CarChasePlanRequest,
): PlanResult<VehicleState> {
  const world = buildPlanningWorld(req.course);
  const affordances = carChaseAffordances(req.course);
  const baseEnv = new VehicleEnvironment(world, CARCHASE_AGENT, CARCHASE_LIB, {
    posCell: 1.5,
    headingBuckets: 16,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 4,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: { everyN: 6, step: 0.6 },
  });
  // Agent circumscribed radius for the dynamic-obstacle test.
  let rCirc = 0;
  for (const [vx, vz] of CARCHASE_AGENT.footprint) {
    const r = Math.hypot(vx, vz);
    if (r > rCirc) rCirc = r;
  }
  const env = new TimeAwareEnvironment(baseEnv, {
    obstacles: req.movingObstacles,
    agentRadius: rCirc,
    affordances,
    affordanceRadius: 10,
    broadphase: { sampleStep: 0.5, maxSamples: 24 },
  });
  return plan(
    {
      start: req.state,
      goal: req.goal,
      environment: env,
      options: {
        maxExpansions: req.maxExpansions ?? CARCHASE_MAX_EXPANSIONS,
      },
    },
    req.deadlineMs ?? CARCHASE_REPLAN_BUDGET_MS,
  );
}

// ---------------------------------------------------------------------------
// Predict the robber's trajectory for cops to plan against. The robber is a
// (mostly) cooperative AI that DOES publish to the registry — when it does,
// cops read its plan directly. When the robber has no plan yet, fall back to
// constant-velocity.

export function predictRobberFromState(
  robber: VehicleState,
  horizon = 4,
): Predict<VehicleState> {
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
    state: VehicleState;
    goal: VehicleState;
    loopIndex: number;
    result: PlanResult<VehicleState>;
  };
  cops: Array<{
    id: string;
    state: VehicleState;
    mode: CopTacticalMode;
    goal: VehicleState;
    result: PlanResult<VehicleState>;
  }>;
}

const SPAWN_ROBBER: VehicleState = {
  x: 50,
  z: 70,
  heading: -Math.PI / 2,
  speed: 0,
  t: 0,
};

const SPAWN_COPS: VehicleState[] = [
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
  );
  const robberResult = planCarChaseAI({
    npcId: 'robber',
    state: SPAWN_ROBBER,
    goal: robberPick.goal,
    movingObstacles: robberCopPreds,
    registry,
    course,
    deadlineMs: Infinity,
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
    const robberPredict: Predict<VehicleState> = (t) => {
      const fromPlan = registry.predictNPC('robber')(t) as VehicleState | null;
      return fromPlan ?? predictRobberFromState(SPAWN_ROBBER, 4)(t);
    };
    const goal = tacticalGoal(SPAWN_ROBBER, robberPredict, cop, mode, course.buildings);
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
