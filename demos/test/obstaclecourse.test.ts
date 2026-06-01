import { describe, it, expect } from 'vitest';
import {
  buildObstacleCourse,
  buildObstacleCourseSnapshot,
  OBS_AGENT,
  OBS_LIB,
  OBS_BLOCKS_ALL,
  OBS_TEST_MAX_EXPANSIONS,
} from '../app/lib/obstaclecourse-scenarios';
import { planVehicleOnce } from 'kinocat/planner';
import { InMemoryNavWorld, rampNavObstacles } from 'kinocat/environment';
import {
  placeFootprint,
  polygonsIntersect,
  type Pt,
} from '../../core/src/internal/geom';

describe('obstaclecourse demo: single-car building-block plan', () => {
  it('the AI plans from spawn to the first waypoint against the full course', () => {
    const s = buildObstacleCourseSnapshot();
    expect(s.result.found).toBe(true);
    expect(s.result.path.length).toBeGreaterThanOrEqual(2);
    expect(s.result.stats.expansions).toBeLessThan(OBS_TEST_MAX_EXPANSIONS);
    for (let i = 1; i < s.result.path.length; i++) {
      expect(s.result.path[i]!.t).toBeGreaterThan(s.result.path[i - 1]!.t - 1e-9);
    }
  });

  it('every block can be toggled independently — empty course still solves', () => {
    const empty = buildObstacleCourseSnapshot({
      heightfield: false,
      buildings: false,
      ramp: false,
      boost: false,
      driftGates: false,
    });
    expect(empty.result.found).toBe(true);
    expect(empty.course.buildings.length).toBe(0);
    expect(empty.course.jumps.length).toBe(0);
    expect(empty.course.boosts.length).toBe(0);
    expect(empty.course.driftGates.length).toBe(0);
  });

  it('full course exposes every advertised block', () => {
    const c = buildObstacleCourse(OBS_BLOCKS_ALL);
    expect(c.buildings.length).toBeGreaterThan(0);
    expect(c.jumps.length).toBeGreaterThanOrEqual(1);
    expect(c.boosts.length).toBeGreaterThanOrEqual(1);
    expect(c.driftGates.length).toBeGreaterThanOrEqual(2);
    expect(c.waypoints.length).toBeGreaterThanOrEqual(4);
  });

  // Spatial-awareness regression: the ramp is a solid wedge, so a plan that
  // needs to get from one broad side to the other must go AROUND it, never
  // straight through the side. Plan with affordances OFF so the car can't jump
  // over (the jump arc legitimately crosses the ramp in 2D); every footprint of
  // the resulting ground path must stay clear of the ramp walls.
  it('never plans a ground path through the broad side of the ramp', () => {
    const course = buildObstacleCourse(OBS_BLOCKS_ALL);
    const world = new InMemoryNavWorld(course.polygons, course.obstacles);
    const walls: Pt[][] = course.ramps.flatMap((r) =>
      rampNavObstacles(r, { back: true }),
    );
    expect(walls.length).toBeGreaterThan(0);

    // Ramp footprint is x∈[13,25.5], z∈[-3,3]; start south of it, goal north.
    const res = planVehicleOnce({
      start: { x: 18, z: -12, heading: Math.PI / 2, speed: 0, t: 0 },
      goal: { x: 18, z: 12, heading: Math.PI / 2, speed: 0, t: 0 },
      world,
      agent: OBS_AGENT,
      lib: OBS_LIB,
      deadlineMs: Number.POSITIVE_INFINITY,
      maxExpansions: OBS_TEST_MAX_EXPANSIONS,
    });
    expect(res.found).toBe(true);
    for (const p of res.path) {
      const fp = placeFootprint(OBS_AGENT.footprint, p.x, p.z, p.heading);
      for (const w of walls) {
        expect(polygonsIntersect(fp, w)).toBe(false);
      }
    }
  });
});
