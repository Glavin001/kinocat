import { describe, it, expect } from 'vitest';
import {
  buildObstacleCourse,
  buildObstacleCourseSnapshot,
  OBS_BLOCKS_ALL,
  OBS_TEST_MAX_EXPANSIONS,
} from '../app/lib/obstaclecourse-scenarios';

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
});
