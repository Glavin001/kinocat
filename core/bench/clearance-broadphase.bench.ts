import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { navWorldFromTriangleMesh } from '../src/adapters/navcat/index';
import { VehicleEnvironment } from '../src/environment/vehicle-environment';
import { defaultVehicleAgent, kinematicForwardSim } from '../src/agent/vehicle';
import { characterizeVehicle } from '../src/primitives/characterize';
import type { VehicleState } from '../src/agent/types';

// Opt 1 (spec §10.2): on a real navcat world `footprintClear` is the
// bottleneck — several findNearestPoly + raycast calls per sweep sample.
// The CompactHeightfield clearance field lets the search skip the exact
// check wherever a disk of the circumscribed radius is provably clear. Same
// fixed query, navcat clearanceAt OFF vs ON.

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

// A 60×40 flat plane (single navcat navmesh) built once with the clearance
// field attached.
function planeMesh(w: number, d: number) {
  return {
    positions: [0, 0, 0, w, 0, 0, w, 0, d, 0, 0, d],
    indices: [0, 3, 2, 0, 2, 1],
  };
}
let built: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = planeMesh(60, 40);
  built = navWorldFromTriangleMesh(
    m.positions,
    m.indices,
    { cellSize: 0.5, walkableSlopeAngleDegrees: 50 },
    { clearanceField: true, horizontalTolerance: 0.6 },
  );
} catch {
  built = null;
}

const start: VehicleState = { x: 4, z: 20, heading: 0, speed: 0, t: 0 };
const goal: VehicleState = { x: 56, z: 20, heading: 0, speed: 0, t: 0 };

function run(clearanceBroadphase: boolean): void {
  if (!built) return;
  const env = new VehicleEnvironment(built.world, agent, lib, {
    posCell: 0.8,
    headingBuckets: 16,
    speedQuant: 4,
    goalRadius: 2,
    goalHeadingTol: Infinity,
    sweepSegmentCheck: false,
    analyticExpansion: false,
    clearanceBroadphase,
  });
  plan(
    { start, goal, environment: env, options: { maxExpansions: 30000 } },
    Infinity,
  );
}

describe('CompactHeightfield clearance broadphase (before vs after)', () => {
  bench('before — clearance broadphase DISABLED (exact navcat footprint)', () => {
    run(false);
  });
  bench('after — clearance broadphase ENABLED (O(1) distance-field)', () => {
    run(true);
  });
});
