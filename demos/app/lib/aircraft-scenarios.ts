// Pure, headless-testable aircraft planning. No React / three imports so the
// exact configuration each demo runs can be asserted by automated tests
// (demos/test/scenarios.test.ts). The aircraft planner searches a genuine 3D
// state (x, y, z, heading, pitch, speed, t) — altitude is planned, not
// derived — over the IGHA* core, unchanged.

import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  AircraftEnvironment,
  InMemoryAirspace,
} from 'kinocat/environment';
import type { AABB, MovingZone, AircraftEnvOptions } from 'kinocat/environment';
import { defaultAircraftAgent } from 'kinocat/agent';
import type { AircraftAgent, AircraftState } from 'kinocat/agent';

export const AIR_PALETTE = {
  bg: '#0a0e16',
  ground: '#16241c',
  wall: '#3a2330',
  path: '#44ddff',
  start: '#55ff88',
  goal: '#ffcc33',
  gate: '#7fd6ff',
  zone: '#ff5577',
  plane: '#e8edf4',
};

/** Interactive budget (single leg, replanned on input). */
export const AIRCRAFT_MAX_EXPANSIONS = 60000;
/** Computed-once scenarios (canyon, moving airspace) get a larger budget. */
export const AIRCRAFT_DYNAMIC_MAX_EXPANSIONS = 400000;

export const AIRCRAFT_AGENT: AircraftAgent = defaultAircraftAgent({
  minTurnRadius: 16,
  minSpeed: 8,
  maxSpeed: 18,
  maxClimbAngle: Math.PI / 6,
  radius: 1.6,
});

export const AIRCRAFT_BOUNDS = {
  x0: 0,
  x1: 160,
  z0: -60,
  z1: 60,
  floor: 0,
  ceiling: 80,
};

function envOpts(over: AircraftEnvOptions = {}): AircraftEnvOptions {
  return {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 9,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
    ...over,
  };
}

export function aircraftAirspace(
  boxes: AABB[] = [],
  zones: MovingZone[] = [],
): InMemoryAirspace {
  return new InMemoryAirspace({
    floor: AIRCRAFT_BOUNDS.floor,
    ceiling: AIRCRAFT_BOUNDS.ceiling,
    boxes,
    zones,
  });
}

/** Plan one leg between two waypoints over a given airspace. */
export function planAircraftLeg(
  airspace: InMemoryAirspace,
  start: AircraftState,
  goal: AircraftState,
  opts: { maxExpansions?: number; env?: AircraftEnvOptions } = {},
): PlanResult<AircraftState> {
  const env = new AircraftEnvironment(
    airspace,
    AIRCRAFT_AGENT,
    envOpts(opts.env),
  );
  return plan(
    {
      start,
      goal,
      environment: env,
      options: { maxExpansions: opts.maxExpansions ?? AIRCRAFT_MAX_EXPANSIONS },
    },
    Infinity,
  );
}

function gate(x: number, y: number, z: number): AircraftState {
  return { x, y, z, heading: 0, pitch: 0, speed: AIRCRAFT_AGENT.maxSpeed, t: 0 };
}

// ---------------------------------------------------------------------------
// Waypoint course — an ordered series of gates at varying altitude. Core has
// no multi-goal, so legs are planned sequentially (the end pose of one leg is
// the start of the next) and concatenated, exactly the one-plan()-per-query
// pattern the 2D demos use.

export interface AircraftScene {
  kind: 'waypoint' | 'canyon' | 'restricted' | 'gauntlet';
  path: AircraftState[];
  found: boolean;
  duration: number;
  start: AircraftState;
  goal: AircraftState;
  gates: AircraftState[];
  boxes: AABB[];
  zoneRadius?: number;
  zoneAt?: (t: number) => { x: number; y: number; z: number } | null;
  info: string;
}

export function buildWaypointCourse(): AircraftScene {
  const gates = [
    gate(55, 50, -30),
    gate(95, 22, 28),
    gate(148, 38, 0),
  ];
  const airspace = aircraftAirspace();
  const start = gate(10, 32, 0);
  const path: AircraftState[] = [start];
  let legStart = start;
  let found = true;
  for (const g of gates) {
    const r = planAircraftLeg(airspace, legStart, g, {
      maxExpansions: AIRCRAFT_MAX_EXPANSIONS,
    });
    if (!r.found) {
      found = false;
      break;
    }
    for (let i = 1; i < r.path.length; i++) path.push(r.path[i]!);
    legStart = path[path.length - 1]!;
  }
  return {
    kind: 'waypoint',
    path,
    found,
    duration: path[path.length - 1]!.t,
    start,
    goal: gates[gates.length - 1]!,
    gates,
    boxes: [],
    info: found
      ? `flew ${gates.length} gates across ${path.length} states`
      : 'no plan',
  };
}

// ---------------------------------------------------------------------------
// Canyon / terrain slalom — full-height walls leave only an alternating side
// gap, so the plane must weave *between* them (it cannot fly over); a final
// full-width ridge is low enough to climb. Both lateral routing and altitude
// must be searched to solve it.

export function buildCanyon(): AircraftScene {
  const f = AIRCRAFT_BOUNDS.floor;
  const c = AIRCRAFT_BOUNDS.ceiling;
  const boxes: AABB[] = [
    // full-height wall; the only gap is on the +z side → weave right
    { min: [44, f, -60], max: [52, c, 4] },
    // full-height wall; the only gap is on the -z side → weave left
    { min: [92, f, -4], max: [100, c, 60] },
    // full-width ridge, low enough to fly over → altitude is searched
    { min: [130, f, -60], max: [138, 34, 60] },
  ];
  const airspace = aircraftAirspace(boxes);
  const start = gate(8, 22, 0);
  const goal = gate(152, 22, 0);
  const r = planAircraftLeg(airspace, start, goal, {
    maxExpansions: AIRCRAFT_DYNAMIC_MAX_EXPANSIONS,
  });
  return {
    kind: 'canyon',
    path: r.found ? r.path : [start],
    found: r.found,
    duration: r.found ? r.path[r.path.length - 1]!.t : 0,
    start,
    goal,
    gates: [],
    boxes,
    info: r.found
      ? `weaved between the walls and climbed the ridge (${r.path.length} states)`
      : 'no plan',
  };
}

// ---------------------------------------------------------------------------
// Restricted airspace — a spherical no-fly zone (storm cell / traffic) drifts
// across the corridor. The plan is time-aware: it routes around where the
// zone *will be*, reusing the same Predict<T> seam the 2D /dynamic demo uses.

export function buildRestrictedAirspace(): AircraftScene {
  const radius = 22;
  const z0 = { x: 86, y: 34, z0: -54, vz: 7, horizon: 60 };
  const zoneAt = (t: number) => {
    if (t < 0 || t > z0.horizon) return null;
    return { x: z0.x, y: z0.y, z: z0.z0 + z0.vz * t };
  };
  const airspace = aircraftAirspace([], [{ radius, predict: zoneAt }]);
  const start = gate(8, 34, 0);
  const goal = gate(152, 34, 0);
  const r = planAircraftLeg(airspace, start, goal, {
    maxExpansions: AIRCRAFT_DYNAMIC_MAX_EXPANSIONS,
  });
  return {
    kind: 'restricted',
    path: r.found ? r.path : [start],
    found: r.found,
    duration: r.found ? r.path[r.path.length - 1]!.t : 0,
    start,
    goal,
    gates: [],
    boxes: [],
    zoneRadius: radius,
    zoneAt,
    info: r.found
      ? `routed around the moving no-fly zone (${r.path.length} states)`
      : 'no plan',
  };
}

// ---------------------------------------------------------------------------
// Grand tour — one combined showcase: weave a full-height wall, dodge a moving
// no-fly zone in the mid corridor (route around OR climb over it), weave the
// second wall, then climb the full-width ridge. Restricted airspace + canyon
// + altitude search + lateral routing, all in a single plan.

export function buildGauntlet(): AircraftScene {
  const f = AIRCRAFT_BOUNDS.floor;
  const c = AIRCRAFT_BOUNDS.ceiling;
  const boxes: AABB[] = [
    { min: [40, f, -60], max: [47, c, 4] }, // full height; weave right (+z)
    { min: [95, f, -4], max: [102, c, 60] }, // full height; weave left (-z)
    { min: [128, f, -60], max: [135, 34, 60] }, // ridge; climb over
  ];
  const radius = 15;
  const zone = { x: 70, y: 26, z0: -46, vz: 8, horizon: 45 };
  const zoneAt = (t: number) => {
    if (t < 0 || t > zone.horizon) return null;
    return { x: zone.x, y: zone.y, z: zone.z0 + zone.vz * t };
  };
  const airspace = aircraftAirspace(boxes, [{ radius, predict: zoneAt }]);
  const start = gate(8, 22, 0);
  const goal = gate(152, 22, 0);
  const r = planAircraftLeg(airspace, start, goal, {
    maxExpansions: AIRCRAFT_DYNAMIC_MAX_EXPANSIONS,
  });
  return {
    kind: 'gauntlet',
    path: r.found ? r.path : [start],
    found: r.found,
    duration: r.found ? r.path[r.path.length - 1]!.t : 0,
    start,
    goal,
    gates: [],
    boxes,
    zoneRadius: radius,
    zoneAt,
    info: r.found
      ? `weaved both walls, beat the moving zone, climbed the ridge (${r.path.length} states)`
      : 'no plan',
  };
}

// ---------------------------------------------------------------------------
// Interactive — default obstacle field for the tap-to-retarget mode. Two
// full-height walls (alternating side gaps) so the plane must weave *between*
// them on the way to the tapped destination; kept to two walls (no ridge) so
// every replan stays inside the interactive expansion budget.

const _IF = AIRCRAFT_BOUNDS.floor;
const _IC = AIRCRAFT_BOUNDS.ceiling;
export const INTERACTIVE_BOXES: AABB[] = [
  { min: [54, _IF, -60], max: [62, _IC, 6] },
  { min: [102, _IF, -6], max: [110, _IC, 60] },
];

export function planInteractive(
  boxes: AABB[],
  start: AircraftState,
  goal: AircraftState,
): PlanResult<AircraftState> {
  return planAircraftLeg(aircraftAirspace(boxes), start, goal, {
    maxExpansions: AIRCRAFT_MAX_EXPANSIONS,
  });
}
