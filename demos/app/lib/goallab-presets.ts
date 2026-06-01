// GoalLab presets — a catalog of canonical goals authored in the
// `kinocat/scenario` AST, each paired with everything needed to (a) plan it via
// the real ScenarioEnvironment bridge and (b) deterministically visualize the
// goal + progress. Pure module (no React/three) so the catalog is testable.

import { InMemoryNavWorld } from 'kinocat/environment';
import { planVehicleScenario } from 'kinocat/planner';
import type { ScenarioPlanResult } from 'kinocat/planner';
import type { VehicleAgent, CarKinematicState } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
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
import { demoVehicle } from './scenarios';
import { authorParkingScenario } from './scenario-goals';
import { buildParkingScenario, PARKING_AGENT, parkingLibrary } from './parking-scenarios';

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
    deadlineMs: 3500,
    maxExpansions: 120_000,
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
  const { agent, lib } = demoVehicle();
  return {
    id: 'point-to-point',
    title: 'Point-to-point',
    description: 'reach(near(p)) + avoid(box) + stayInside(field)',
    scenario,
    bounds: FIELD,
    obstacles,
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), obstacles), agent, lib, {
        posCell: 1,
        headingBuckets: 16,
        goalRadius: 2,
      }),
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
  const { agent, lib } = demoVehicle();
  return {
    id: 'a-or-b',
    title: 'A or B (any)',
    description: 'any(reach(bayA), reach(bayB)) — either open bay satisfies',
    scenario,
    bounds: FIELD,
    obstacles: [],
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), []), agent, lib, {
        posCell: 1,
        headingBuckets: 16,
        goalRadius: 2,
      }),
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
  const { agent, lib } = demoVehicle();
  return {
    id: 'slalom',
    title: 'Slalom (seq)',
    description: 'seq(reach g1, reach g2, …) — ordered gates',
    scenario,
    bounds: FIELD,
    obstacles: [],
    plan: () =>
      planWith(scenario, new InMemoryNavWorld(fieldPolys(FIELD), []), agent, lib, {
        posCell: 1,
        headingBuckets: 16,
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
  const { agent, lib } = demoVehicle({ maxSpeed: 10 });
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
        posCell: 1,
        headingBuckets: 16,
        goalRadius: 2.5,
      }),
  };
}

// --- Parking (at-pose, stop, aligned) --------------------------------------
function parkingPreset(): GoalPreset {
  const s = buildParkingScenario('forward-pullin');
  const scenario = authorParkingScenario('forward-pullin');
  return {
    id: 'parking',
    title: 'Parking (at + stop)',
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
        ])], deadlineMs: 4000, maxExpansions: 80_000 },
      ),
  };
}

export function goalLabPresets(): GoalPreset[] {
  return [
    pointToPointPreset(),
    slalomPreset(),
    aOrBPreset(),
    interceptPreset(),
    parkingPreset(),
  ];
}
