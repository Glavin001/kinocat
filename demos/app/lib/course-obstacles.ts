// Single source of truth for assembling a course's planner-collision polygons.
//
// Every composable piece a car must avoid funnels through here so the
// planner-obstacle sink can never silently forget one — the exact bug that let
// the planner drive straight through the broad side of a ramp (the ramp's
// footprint was never registered as an obstacle, only its heightfield +
// affordance were). Buildings/pillars become inflated boxes; ramps contribute
// their wedge walls via the core `rampNavObstacles` primitive, derived from the
// SAME `RampSpec` that drives the heightfield and visual mesh.
//
// This is the minimal, phased step toward a full `WorldPiece` union: it shares
// the *collision* derivation across demos today, while leaving the (already
// course-derived) physics/visual fan-out untouched. A later pass can dispatch
// per-piece via `navObstaclesOf(piece)` without changing these call sites.

import { rampNavObstacles } from 'kinocat/environment';
import type { RampSpec, RampNavObstacleOptions } from 'kinocat/environment';

export interface CourseBoxSpec {
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export interface CourseObstacleInput {
  /** Cuboid footprints (buildings + drift pillars) the car must go around. */
  boxes: ReadonlyArray<CourseBoxSpec>;
  /** Drivable ramps — each contributes solid side (+ back) walls. */
  ramps: ReadonlyArray<RampSpec>;
  /** Outward inflation past the visual face for every box (m). */
  inflate: number;
  /** Per-course ramp wall tuning (see `rampNavObstacles`). */
  rampOpts?: RampNavObstacleOptions;
}

/** Axis-aligned box footprint (CCW) centred at (x,z) with half-extents. */
export function box(
  x: number,
  z: number,
  hx: number,
  hz: number,
): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

/** Assemble the full planner-obstacle set for a course: inflated cuboids plus
 *  each ramp's wedge walls. */
export function buildCourseObstacles(
  input: CourseObstacleInput,
): Array<[number, number][]> {
  const obstacles: Array<[number, number][]> = input.boxes.map((b) =>
    box(b.x, b.z, b.hx + input.inflate, b.hz + input.inflate),
  );
  for (const r of input.ramps) {
    for (const wall of rampNavObstacles(r, input.rampOpts)) {
      obstacles.push(wall as [number, number][]);
    }
  }
  return obstacles;
}
