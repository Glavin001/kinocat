// GoalLab presets — a catalog of canonical goals authored in the
// `kinocat/scenario` AST, each paired with everything needed to (a) plan it via
// the real ScenarioEnvironment bridge and (b) deterministically visualize the
// goal + progress. Pure module (no React/three) so the catalog is testable.

import { InMemoryNavWorld } from 'kinocat/environment';
import { planVehicleScenario } from 'kinocat/planner';
import type { ScenarioPlanResult } from 'kinocat/planner';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { characterizeVehicle } from 'kinocat/primitives';
import {
  defineScenario,
  reach,
  seq,
  any,
  near,
  at,
  inside,
  within,
  avoid,
  stayInside,
  minTime,
  smooth,
  deg,
} from 'kinocat/scenario';
import type { Scenario, RegionAgent } from 'kinocat/scenario';
import { authorParkingScenario, authorDraftingHold, planDrafting } from './scenario-goals';
import {
  buildParkingScenario,
  PARKING_AGENT,
  parkingLibrary,
  type ParkingScenarioId,
} from './parking-scenarios';

export interface GoalPreset {
  id: string;
  title: string;
  description: string;
  scenario: Scenario;
  bounds: { x0: number; x1: number; z0: number; z1: number };
  /** Obstacle polygons drawn + used for collision. */
  obstacles: Array<[number, number][]>;
  /** A scripted moving target (for dynamic/intercept presets), if any. */
  movingTarget?: RegionAgent;
  /** Plan the preset through the ScenarioEnvironment product search. */
  plan(): ScenarioPlanResult;
}

const FIELD = { x0: -30, x1: 30, z0: -20, z1: 20 };

function field(b: typeof FIELD): [number, number][] {
  return [
    [b.x0, b.z0],
    [b.x1, b.z0],
    [b.x1, b.z1],
    [b.x0, b.z1],
  ];
}
function fieldPolys(b: typeof FIELD) {
  return [{ id: 1, y: 0, ring: field(b) }];
}
function boxPoly(x: number, z: number, h: number): [number, number][] {
  return [
    [x - h, z - h],
    [x + h, z - h],
    [x + h, z + h],
    [x - h, z + h],
  ];
}

/** A finer cruising library for the open-field presets: a chassis that matches
 *  the rendered 4.8×2.0 m car, with several curvature options at a steady cruise
 *  speed (and a few reverse) at 0.4 s granularity. Denser + curvier than the
 *  demo's `startSpeeds:[0]` set, so the visualized path reads as smooth arcs
 *  rather than straight chords between sparse nodes. */
function goalLabVehicle(): { agent: VehicleAgent; lib: MotionPrimitiveLibrary } {
  const agent = defaultVehicleAgent({
    minTurnRadius: 4,
    maxSpeed: 8,
    maxReverseSpeed: 4,
    footprint: [
      [2.4, 1.0],
      [-2.4, 1.0],
      [-2.4, -1.0],
      [2.4, -1.0],
    ],
  });
  const k = 1 / agent.minTurnRadius;
  const cruise = 7;
  const lib = characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, cruise],
      [k, cruise],
      [-k, cruise],
      [k / 2, cruise],
      [-k / 2, cruise],
      [k / 3, cruise],
      [-k / 3, cruise],
      [0, -3],
      [k, -3],
      [-k, -3],
    ],
    duration: 0.4,
    substeps: 6,
    startSpeeds: [0, cruise],
  });
  return { agent, lib };
}

/** Env options for the open-field presets. The analytic Reeds-Shepp shot stays
 *  ON (it's what makes the goal reliably reachable); the GoalLab renderer
 *  smooths the resulting sparse path with heading-aware Hermite interpolation,
 *  so the long shot edge reads as a curve rather than a straight chord. */
const SMOOTH_ENV = {
  posCell: 1,
  headingBuckets: 24,
  goalRadius: 2,
  analyticExpansion: { everyN: 6, step: 0.5 },
};

function planWith(
  scenario: Scenario,
  world: InMemoryNavWorld,
  agent: VehicleAgent,
  lib: MotionPrimitiveLibrary,
  envOptions: Parameters<typeof planVehicleScenario>[0]['envOptions'],
  extra: Partial<Parameters<typeof planVehicleScenario>[0]> = {},
): ScenarioPlanResult {
  return planVehicleScenario({
    start: scenario.start,
    goal: scenario.goal,
    invariants: scenario.invariants,
    prefer: scenario.prefer,
    world,
    agent,
    lib,
    envOptions,
    // Expansion-bounded (not wall-clock) so the plan — and thus the
    // visualization — is DETERMINISTIC across machines / coverage runs.
    deadlineMs: Infinity,
    maxExpansions: 60_000,
    ...extra,
  });
}

// --- Point-to-point --------------------------------------------------------
function pointToPointPreset(): GoalPreset {
  const obstacles: Array<[number, number][]> = [boxPoly(0, 0, 3)];
  const start: CarKinematicState = { x: -22, z: -8, heading: 0, speed: 0, t: 0 };
  const scenario = defineScenario('Point-to-point', {
    start,
    goal: reach(near({ x: 22, z: 8 }, 2)),
    invariants: [stayInside(field(FIELD)), avoid(inside(obstacles[0]!))],
    prefer: [minTime(1)],
  });
  const { agent, lib } = goalLabVehicle();
  return {
    id: 'point-to-point',
    title: 'Point-to-point',
    description: 'reach(near(p)) + avoid(box) + stayInside(field)',
    scenario,
    bounds: FIELD,
    obstacles,
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), obstacles), agent, lib, SMOOTH_ENV),
  };
}

// --- A-or-B (disjunction) --------------------------------------------------
function aOrBPreset(): GoalPreset {
  const start: CarKinematicState = { x: -24, z: 0, heading: 0, speed: 0, t: 0 };
  const scenario = defineScenario('A or B', {
    start,
    goal: any(reach(near({ x: 20, z: 12 }, 2)), reach(near({ x: 20, z: -12 }, 2))),
    invariants: [stayInside(field(FIELD))],
    prefer: [minTime(1)],
  });
  const { agent, lib } = goalLabVehicle();
  return {
    id: 'a-or-b',
    title: 'A or B (any)',
    description: 'any(reach(bayA), reach(bayB)) — either open bay satisfies',
    scenario,
    bounds: FIELD,
    obstacles: [],
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), []), agent, lib, SMOOTH_ENV),
  };
}

// --- Sequenced gates (seq) -------------------------------------------------
function slalomPreset(): GoalPreset {
  const start: CarKinematicState = { x: -24, z: 0, heading: 0, speed: 0, t: 0 };
  const wps = [
    { x: -10, z: 8 },
    { x: 2, z: -8 },
    { x: 14, z: 8 },
    { x: 24, z: 0 },
  ];
  const scenario = defineScenario('Slalom (seq)', {
    start,
    goal: seq(...wps.map((w) => reach(near(w, 2.5)))),
    invariants: [stayInside(field(FIELD))],
    prefer: [minTime(1)],
  });
  const { agent, lib } = goalLabVehicle();
  return {
    id: 'slalom',
    title: 'Slalom (seq)',
    description: 'seq(reach g1, reach g2, …) — ordered gates',
    scenario,
    bounds: FIELD,
    obstacles: [],
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), []), agent, lib, {
        ...SMOOTH_ENV,
        goalRadius: 2.5,
      }),
  };
}

// --- Intercept (dynamic region -> clock) -----------------------------------
function interceptPreset(): GoalPreset {
  const start: CarKinematicState = { x: -24, z: -10, heading: 0, speed: 0, t: 0 };
  // Target crosses the field +x at 3 m/s starting from (-20, 10).
  const target: RegionAgent = {
    id: 'runner',
    predict: (t) => ({ x: -20 + 3 * t, z: 10, heading: 0, speed: 3, t }),
  };
  const scenario = defineScenario('Intercept', {
    start,
    goal: reach(within(target, 2.5)),
    invariants: [stayInside(field(FIELD))],
    prefer: [minTime(1)],
    agents: [target],
  });
  const { agent, lib } = goalLabVehicle();
  return {
    id: 'intercept',
    title: 'Intercept (dynamic)',
    description: 'reach(within(target)) — clock enters the search; aims where it WILL be',
    scenario,
    bounds: FIELD,
    obstacles: [],
    movingTarget: target,
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), []), agent, lib, {
        ...SMOOTH_ENV,
        goalRadius: 2.5,
      }),
  };
}

// --- Parking (at-pose, stop, aligned) — one preset per stall layout --------
const PARKING_VARIANTS: Array<{ id: ParkingScenarioId; title: string }> = [
  { id: 'forward-pullin', title: 'Parking — forward pull-in' },
  { id: 'reverse-perp', title: 'Parking — reverse into bay' },
  { id: 'parallel', title: 'Parking — parallel' },
];

function parkingPreset(id: ParkingScenarioId, title: string): GoalPreset {
  const s = buildParkingScenario(id);
  const scenario = authorParkingScenario(id);
  return {
    id: `parking-${id}`,
    title,
    description: 'reach(at(pose,margins),{speed:{max:0}}) + stayInside(lot) + avoid(cars)',
    scenario,
    bounds: s.bounds,
    obstacles: s.obstacles,
    plan: () =>
      planWith(
        scenario,
        new InMemoryNavWorld(s.polygons, s.obstacles),
        PARKING_AGENT,
        parkingLibrary(),
        {
          posCell: 0.3,
          headingBuckets: 36,
          goalRadius: 0.35,
          goalHeadingTol: 0.25,
          sweepSegmentCheck: true,
          analyticExpansion: { everyN: 3, step: 0.15 },
        },
        { invariants: [stayInside([
          [s.bounds.x0, s.bounds.z0],
          [s.bounds.x1, s.bounds.z0],
          [s.bounds.x1, s.bounds.z1],
          [s.bounds.x0, s.bounds.z1],
        ])], deadlineMs: Infinity, maxExpansions: 80_000 },
      ),
  };
}

// --- Drafting (CONTINUOUS close-follow behind a moving car) -----------------
// Uses the sustained `hold` variant so the visualization shows the ego FIRST
// reaching the slipstream slot and then CONTINUING to draft (a repeat/progress
// objective, never "done") — match-pace is `.while`-scoped to the slot.
function draftingPreset(): GoalPreset {
  const bounds = { x0: -40, x1: 70, z0: -25, z1: 25 };
  const lead: RegionAgent = {
    id: 'lead',
    predict: (t) => ({ x: 6 + 3 * t, z: 0, heading: 0, speed: 3, t }),
  };
  // Start near the lead's path so the (time-augmented) repeat search reaches
  // the slot quickly and then holds.
  const start: CarKinematicState = { x: -2, z: -5, heading: 0, speed: 0, t: 0 };
  const scenario = authorDraftingHold({ start, lead, gap: 6, tol: 2.5, safe: 2, bounds });
  return {
    id: 'drafting',
    title: 'Drafting (follow a moving car)',
    description: 'repeat(reach(behind(lead,6))) + hold station — enters the slipstream, then keeps following',
    scenario,
    bounds,
    obstacles: [],
    movingTarget: lead,
    plan: () =>
      planDrafting({ start, lead, gap: 6, tol: 2.5, safe: 2, bounds }, {
        hold: true,
        horizonSeconds: 6,
        maxExpansions: 55_000,
      }),
  };
}

export function goalLabPresets(): GoalPreset[] {
  return [
    pointToPointPreset(),
    slalomPreset(),
    aOrBPreset(),
    interceptPreset(),
    draftingPreset(),
    ...PARKING_VARIANTS.map((v) => parkingPreset(v.id, v.title)),
  ];
}
