import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { InMemoryNavWorld } from '../src/environment/nav-world';
import { VehicleEnvironment } from '../src/environment/vehicle-environment';
import type { VehicleEnvOptions } from '../src/environment/vehicle-environment';
import { defaultVehicleAgent, kinematicForwardSim } from '../src/agent/vehicle';
import { characterizeVehicle } from '../src/primitives/characterize';
import type { CarKinematicState } from '../src/agent/types';

// Reeds-Shepp heuristic LUT (spec §12.3): the RS shortest-path heuristic is
// the dominant per-successor cost. This benchmarks the SAME query with the
// table disabled (exact RS per successor) vs enabled (O(1) cached lookup).

const agent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  maxReverseSpeed: 4,
  footprint: [
    [1.2, 0.6],
    [-1.2, 0.6],
    [-1.2, -0.6],
    [1.2, -0.6],
  ],
});
const k = 1 / agent.minTurnRadius;
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [
    [0, 6],
    [k, 6],
    [-k, 6],
    [k / 2, 6],
    [-k / 2, 6],
    [0, -4],
    [k, -4],
    [-k, -4],
  ],
  duration: 0.5,
  substeps: 4,
  startSpeeds: [0],
});

// An obstacle-rich world with the analytic shot-to-goal disabled, so the
// search is genuinely heuristic-bound (RS called for every successor).
function world(): InMemoryNavWorld {
  const b = { x0: 0, z0: -11, x1: 44, z1: 11 };
  const box = (x: number, z: number, h: number): [number, number][] => [
    [x - h, z - h],
    [x + h, z - h],
    [x + h, z + h],
    [x - h, z + h],
  ];
  return new InMemoryNavWorld(
    [
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
    ],
    [box(15, -3, 3), box(26, 4, 3), box(34, -2, 3)],
  );
}

const start: CarKinematicState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
const goal: CarKinematicState = { x: 41, z: 0, heading: 0, speed: 0, t: 0 };

const base: VehicleEnvOptions = {
  posCell: 0.8,
  headingBuckets: 16,
  speedQuant: 4,
  levelDivisors: [4, 2, 1],
  goalRadius: 2,
  goalHeadingTol: Infinity,
  sweepSegmentCheck: false,
  analyticExpansion: false,
};

function run(opts: VehicleEnvOptions): void {
  const env = new VehicleEnvironment(world(), agent, lib, opts);
  plan(
    { start, goal, environment: env, options: { maxExpansions: 30000 } },
    Infinity,
  );
}

describe('Reeds-Shepp heuristic table (before vs after)', () => {
  bench('before — heuristic table DISABLED (exact RS per successor)', () => {
    run(base);
  });
  bench('after — heuristic table ENABLED (cached LUT)', () => {
    run({ ...base, heuristicTable: {} });
  });
});
