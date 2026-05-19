import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { navWorldFromTriangleMesh } from '../src/adapters/navcat/index';
import { VehicleEnvironment } from '../src/environment/vehicle-environment';
import { defaultVehicleAgent, kinematicForwardSim } from '../src/agent/vehicle';
import { characterizeVehicle } from '../src/primitives/characterize';
import type { VehicleState } from '../src/agent/types';

// Opt 2 (spec §10.3): the Reeds-Shepp/Euclid heuristic is blind to
// obstacles, so on a serpentine it expands deep into dead-ends. The
// grid-Dijkstra goal lower bound is obstacle-aware → branch-and-bound
// prunes those. Same fixed query, gridHeuristic OFF vs ON.

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
  ],
  duration: 0.5,
  substeps: 4,
  startSpeeds: [0],
});

// A ground plane tessellated into `cell`-sized quads, skipping any cell
// whose centre falls inside a wall rect → a navmesh with real holes.
function serpentineMesh() {
  const W = 50;
  const D = 30;
  const cell = 2;
  const walls = [
    { x0: 14, z0: 0, x1: 16, z1: 22 }, // gap at the top
    { x0: 28, z0: 8, x1: 30, z1: 30 }, // gap at the bottom
    { x0: 40, z0: 0, x1: 42, z1: 22 }, // gap at the top
  ];
  const inWall = (x: number, z: number) =>
    walls.some((w) => x >= w.x0 && x <= w.x1 && z >= w.z0 && z <= w.z1);
  const positions: number[] = [];
  const indices: number[] = [];
  for (let z = 0; z < D; z += cell) {
    for (let x = 0; x < W; x += cell) {
      if (inWall(x + cell / 2, z + cell / 2)) continue;
      const b = positions.length / 3;
      positions.push(x, 0, z, x + cell, 0, z, x + cell, 0, z + cell, x, 0, z + cell);
      indices.push(b, b + 3, b + 2, b, b + 2, b + 1);
    }
  }
  return { positions, indices };
}

let built: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = serpentineMesh();
  built = navWorldFromTriangleMesh(
    m.positions,
    m.indices,
    { cellSize: 0.3, walkableSlopeAngleDegrees: 50 },
    { clearanceField: true, horizontalTolerance: 0.5 },
  );
} catch {
  built = null;
}

const start: VehicleState = { x: 4, z: 15, heading: 0, speed: 0, t: 0 };
const goal: VehicleState = { x: 47, z: 15, heading: 0, speed: 0, t: 0 };

function run(gridHeuristic: boolean): void {
  if (!built) return;
  const env = new VehicleEnvironment(built.world, agent, lib, {
    posCell: 0.8,
    headingBuckets: 16,
    speedQuant: 4,
    goalRadius: 2,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: false,
    gridHeuristic: gridHeuristic ? {} : false,
  });
  plan(
    { start, goal, environment: env, options: { maxExpansions: 200000 } },
    Infinity,
  );
}

describe('obstacle-aware grid-Dijkstra dual heuristic (before vs after)', () => {
  bench('before — gridHeuristic DISABLED (Reeds-Shepp/Euclid only)', () => {
    run(false);
  });
  bench('after — gridHeuristic ENABLED (obstacle-aware lower bound)', () => {
    run(true);
  });
});
