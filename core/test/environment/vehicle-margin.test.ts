// Regression: the `footprintInflate` option on VehicleEnvironment actually
// keeps the planned path away from static obstacles by the requested margin.
// Without the knob a planner can route a chassis-sized footprint to graze a
// wall, which on a real Rapier raycast vehicle shows up as visible wall
// scraping during execution.

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { VehicleAgent, VehicleState } from '../../src/agent/types';
import { placeFootprint } from '../../src/internal/geom';

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const agent: VehicleAgent = defaultVehicleAgent({
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
  substeps: 6,
  startSpeeds: [0],
});

/** Distance from point (px,pz) to segment (ax,az)-(bx,bz). */
function distPointSeg(
  px: number,
  pz: number,
  ax: number,
  az: number,
  bx: number,
  bz: number,
): number {
  const dx = bx - ax;
  const dz = bz - az;
  const len2 = dx * dx + dz * dz;
  if (len2 === 0) return Math.hypot(px - ax, pz - az);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / len2));
  return Math.hypot(px - (ax + t * dx), pz - (az + t * dz));
}

/** True min distance between two convex polygons (assumes non-overlapping;
 *  callers verify overlap separately). For non-overlapping convex polygons,
 *  min distance is achieved at vertex-edge pairs, so iterate both ways. */
function polygonPolygonDistance(
  a: ReadonlyArray<readonly [number, number]>,
  b: ReadonlyArray<readonly [number, number]>,
): number {
  let min = Infinity;
  // Each vertex of A to each edge of B.
  for (let i = 0; i < a.length; i++) {
    const [px, pz] = a[i]!;
    for (let j = 0; j < b.length; j++) {
      const [bx0, bz0] = b[j]!;
      const [bx1, bz1] = b[(j + 1) % b.length]!;
      const d = distPointSeg(px, pz, bx0, bz0, bx1, bz1);
      if (d < min) min = d;
    }
  }
  // Each vertex of B to each edge of A.
  for (let i = 0; i < b.length; i++) {
    const [px, pz] = b[i]!;
    for (let j = 0; j < a.length; j++) {
      const [ax0, az0] = a[j]!;
      const [ax1, az1] = a[(j + 1) % a.length]!;
      const d = distPointSeg(px, pz, ax0, az0, ax1, az1);
      if (d < min) min = d;
    }
  }
  return min;
}

function pathMinDistanceToObstacle(
  path: VehicleState[],
  obstacle: ReadonlyArray<readonly [number, number]>,
): number {
  let min = Infinity;
  for (const s of path) {
    const fp = placeFootprint(agent.footprint, s.x, s.z, s.heading);
    const d = polygonPolygonDistance(fp, obstacle);
    if (d < min) min = d;
  }
  return min;
}

describe('VehicleEnvironment footprintInflate clearance', () => {
  // Corridor 0..40 in x, -10..10 in z. A wall obstacle is centred on the
  // y-axis: the planner must steer around it. With no inflation the planner
  // is free to graze the obstacle's edge; with inflation it must keep the
  // physical footprint away from the obstacle by at least `inflate`.
  const polygons = [rect(1, -5, -10, 45, 10)];
  const obstacle: [number, number][] = [
    [18, -2],
    [22, -2],
    [22, 4],
    [18, 4],
  ];
  const world = new InMemoryNavWorld(polygons, [obstacle]);
  const start: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: VehicleState = { x: 36, z: 0, heading: 0, speed: 0, t: 0 };

  function planWith(inflate: number) {
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
      footprintInflate: inflate,
      sweepSegmentCheck: true,
    });
    return plan(
      { start, goal, environment: env, options: { maxExpansions: 200000 } },
      Infinity,
    );
  }

  it('zero inflation: planner finds a path that may graze the wall', () => {
    const r = planWith(0);
    expect(r.found).toBe(true);
    expect(r.path.length).toBeGreaterThanOrEqual(2);
    // No assertion on clearance — baseline only.
  });

  it('positive inflation keeps the physical footprint clear by ~inflate', () => {
    const inflate = 0.5;
    const r = planWith(inflate);
    expect(r.found).toBe(true);
    const minDist = pathMinDistanceToObstacle(r.path, obstacle);
    // The planner only samples the path at primitive-substep cadence, so
    // intermediate poses on a fast segment may dip below the requested
    // margin by half a sub-step's worth of motion. A 1 cm slack covers
    // that without making the test useless.
    expect(minDist).toBeGreaterThan(inflate - 0.01);
  });

  it('larger inflation enforces strictly more clearance than smaller', () => {
    const small = planWith(0.2);
    const large = planWith(0.6);
    expect(small.found).toBe(true);
    expect(large.found).toBe(true);
    const dSmall = pathMinDistanceToObstacle(small.path, obstacle);
    const dLarge = pathMinDistanceToObstacle(large.path, obstacle);
    expect(dLarge).toBeGreaterThan(dSmall);
    expect(dLarge).toBeGreaterThan(0.6 - 0.01);
  });
});
