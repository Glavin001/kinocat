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
import { defaultAircraftAgent, aircraftForwardSim } from 'kinocat/agent';
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
  maxBank: Math.PI / 2,
  halfLength: 2,
  halfSpan: 1.5,
  halfHeight: 0.3,
});

/** Convenience: body-frame half-extents matching AIRCRAFT_AGENT, in the
 *  order AirspaceWorld.clear expects (length, span, height). */
export const AIRCRAFT_HALF: [number, number, number] = [
  AIRCRAFT_AGENT.halfLength,
  AIRCRAFT_AGENT.halfSpan,
  AIRCRAFT_AGENT.halfHeight,
];

/** Pose helper for AirspaceWorld.clear from an AircraftState. */
export function aircraftPose(s: AircraftState) {
  return { x: s.x, y: s.y, z: s.z, yaw: s.heading, pitch: s.pitch, roll: s.roll };
}

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
    primDuration: AIRCRAFT_PRIM_DURATION,
    substeps: AIRCRAFT_PRIM_SUBSTEPS,
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
  return {
    x,
    y,
    z,
    heading: 0,
    pitch: 0,
    roll: 0,
    speed: AIRCRAFT_AGENT.maxSpeed,
    t: 0,
  };
}

// ---------------------------------------------------------------------------
// Waypoint course — an ordered series of gates at varying altitude. Core has
// no multi-goal, so legs are planned sequentially (the end pose of one leg is
// the start of the next) and concatenated, exactly the one-plan()-per-query
// pattern the 2D demos use.

export interface AircraftScene {
  kind: 'waypoint' | 'canyon' | 'restricted' | 'gauntlet' | 'knife-edge';
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
// Knife edge — a narrow vertical slot too tight for the wingspan. The only
// feasible plan banks to ±π/2 (wings vertical) so the OBB's thin axis lines
// up with the slot. Requires roll to be a searched dimension, opted in via
// rollFractions.

export function buildKnifeEdge(): AircraftScene {
  const f = AIRCRAFT_BOUNDS.floor;
  const c = AIRCRAFT_BOUNDS.ceiling;
  // Slot is 1.2 units wide in z, full height. Wingspan 3 (halfSpan 1.5) does
  // NOT fit level; thickness 0.6 (halfHeight 0.3) fits when banked ±90°.
  const boxes: AABB[] = [
    { min: [78, f, -60], max: [92, c, -0.6] },
    { min: [78, f, 0.6], max: [92, c, 60] },
  ];
  const airspace = aircraftAirspace(boxes);
  const start = gate(8, 24, 0);
  const goal = gate(152, 24, 0);
  const r = planAircraftLeg(airspace, start, goal, {
    maxExpansions: AIRCRAFT_DYNAMIC_MAX_EXPANSIONS,
    env: { rollFractions: [-1, 0, 1], goalRadius: 10 },
  });
  return {
    kind: 'knife-edge',
    path: r.found ? r.path : [start],
    found: r.found,
    duration: r.found ? r.path[r.path.length - 1]!.t : 0,
    start,
    goal,
    gates: [],
    boxes,
    info: r.found
      ? `knife-edged through a 1.2 m slot (${r.path.length} states)`
      : 'no plan — the slot is impassable level',
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

// ---------------------------------------------------------------------------
// Densify a planned path for rendering. The planner returns one state per
// motion primitive (~1 s), so straight-line interpolation between adjacent
// states cuts the chord of every arc — the rendered plane drifts sideways or
// even appears to fly backwards on sharp turns. We re-integrate
// aircraftForwardSim with the controls backed out from each segment so the
// rendered trajectory IS the arc the planner committed to.

/** Recover an attitude SETPOINT that reproduces the segment's ramp when
 *  re-integrated: a ramp that settled within the segment is reproduced by
 *  commanding the end value; a ramp still saturated at the end (|Δ| ≈
 *  rate·dt) is reproduced by commanding the envelope limit in that
 *  direction (any target at or beyond the end value yields the identical
 *  max-rate ramp). Infinity rates never saturate → end value (legacy). */
function rampedSetpoint(
  from: number,
  to: number,
  rate: number,
  dt: number,
  limit: number,
): number {
  if (Math.abs(to - from) >= rate * dt - 1e-9) {
    return to > from ? limit : -limit;
  }
  return to;
}

/** The substep resolution the demo planner collision-certifies (see
 *  planAircraftLeg's env options). densifyPath re-integrates at EXACTLY this
 *  resolution so its points are the planner's certified sample points —
 *  finer requested densities are linear subdivisions of that certified
 *  polygon, never a re-integration at a different Euler step (which drifts
 *  onto trajectories the planner never checked). */
export const AIRCRAFT_PRIM_SUBSTEPS = 4;

/** Duration of one demo motion primitive (see planAircraftLeg env options).
 *  densifyPath uses it to tell primitive segments (re-integrate the sim)
 *  from analytic-shot segments (linear, as certified). */
export const AIRCRAFT_PRIM_DURATION = 1;

export function densifyPath(
  path: AircraftState[],
  substepsPerSegment = AIRCRAFT_PRIM_SUBSTEPS,
): AircraftState[] {
  if (path.length < 2) return [...path];
  const sim = aircraftForwardSim(AIRCRAFT_AGENT);
  const lerpN = Math.max(
    1,
    Math.round(substepsPerSegment / AIRCRAFT_PRIM_SUBSTEPS),
  );
  const out: AircraftState[] = [path[0]!];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    const dt = b.t - a.t;
    if (dt <= 1e-9) continue;
    // Analytic-shot segments (recognizable by a non-primitive duration) are
    // certified by the planner as straight lines with synthetic poses — the
    // sim cannot re-fly them, so reproduce exactly what was certified.
    if (Math.abs(dt - AIRCRAFT_PRIM_DURATION) > 1e-6) {
      let dhh = b.heading - a.heading;
      if (dhh > Math.PI) dhh -= 2 * Math.PI;
      if (dhh < -Math.PI) dhh += 2 * Math.PI;
      for (let m = 1; m <= substepsPerSegment; m++) {
        const u = m / substepsPerSegment;
        out.push({
          x: a.x + (b.x - a.x) * u,
          y: a.y + (b.y - a.y) * u,
          z: a.z + (b.z - a.z) * u,
          heading: a.heading + dhh * u,
          pitch: a.pitch + (b.pitch - a.pitch) * u,
          roll: a.roll + (b.roll - a.roll) * u,
          speed: b.speed,
          t: a.t + dt * u,
        });
      }
      continue;
    }
    // The segment's commanded speed (speed snaps to its setpoint, so the
    // SUCCESSOR carries it; `a.speed` is the previous segment's command).
    const speed = b.speed > 1 ? b.speed : AIRCRAFT_AGENT.maxSpeed;
    // Recover the primitive's curvature from the heading delta; attitude is
    // rate-limited state, so back out setpoints that reproduce each ramp.
    let dh = ((b.heading - a.heading + Math.PI) % (2 * Math.PI)) - Math.PI;
    if (dh < -Math.PI) dh += 2 * Math.PI;
    const k = dh / (speed * dt);
    const climb = rampedSetpoint(
      a.pitch,
      b.pitch,
      AIRCRAFT_AGENT.maxPitchRate,
      dt,
      AIRCRAFT_AGENT.maxClimbAngle,
    );
    const roll = rampedSetpoint(
      a.roll,
      b.roll,
      AIRCRAFT_AGENT.maxRollRate,
      dt,
      AIRCRAFT_AGENT.maxBank,
    );
    const dtSub = dt / AIRCRAFT_PRIM_SUBSTEPS;
    let s = a;
    for (let j = 0; j < AIRCRAFT_PRIM_SUBSTEPS; j++) {
      const prev = s;
      s = sim(s, [k, climb, roll, speed], dtSub);
      // Linear subdivision of the certified step for rendering smoothness.
      for (let m = 1; m < lerpN; m++) {
        const u = m / lerpN;
        let dhh = s.heading - prev.heading;
        if (dhh > Math.PI) dhh -= 2 * Math.PI;
        if (dhh < -Math.PI) dhh += 2 * Math.PI;
        out.push({
          x: prev.x + (s.x - prev.x) * u,
          y: prev.y + (s.y - prev.y) * u,
          z: prev.z + (s.z - prev.z) * u,
          heading: prev.heading + dhh * u,
          pitch: prev.pitch + (s.pitch - prev.pitch) * u,
          roll: prev.roll + (s.roll - prev.roll) * u,
          speed,
          t: prev.t + (s.t - prev.t) * u,
        });
      }
      out.push(s);
    }
  }
  return out;
}
