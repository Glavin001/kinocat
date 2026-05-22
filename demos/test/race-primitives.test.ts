// Headless smoke for the side-by-side race scenario. Confirms BOTH library
// flavours (kinematic + learned-from-default-params) can plan a path from
// spawn → first waypoint on the race course. Pure planner test — no Rapier
// physics here (the interactive race demo exercises Rapier).

import { describe, it, expect } from 'vitest';
import { DEFAULT_LEARNED_PARAMS } from 'kinocat/agent';
import {
  buildRaceCourse,
  buildRaceSnapshot,
  RACE_TEST_MAX_EXPANSIONS,
} from '../app/lib/race-primitives-scenarios';

describe('raceprimitives demo', () => {
  it('the course has a non-trivial waypoint loop and clean bounds', () => {
    const c = buildRaceCourse();
    expect(c.waypoints.length).toBeGreaterThanOrEqual(6);
    expect(c.obstacles.length).toBe(0); // pure dynamics-stress course
    expect(c.bounds.x1).toBeGreaterThan(c.bounds.x0);
    expect(c.bounds.z1).toBeGreaterThan(c.bounds.z0);
  });

  it('both libraries find a plan from spawn → first waypoint', () => {
    const s = buildRaceSnapshot(DEFAULT_LEARNED_PARAMS);
    expect(s.kinematicResult.found).toBe(true);
    expect(s.kinematicResult.path.length).toBeGreaterThanOrEqual(2);
    expect(s.kinematicResult.stats.expansions).toBeLessThan(RACE_TEST_MAX_EXPANSIONS);
    expect(s.learnedResult.found).toBe(true);
    expect(s.learnedResult.path.length).toBeGreaterThanOrEqual(2);
    expect(s.learnedResult.stats.expansions).toBeLessThan(RACE_TEST_MAX_EXPANSIONS);
    // The two plans need not be identical (different libs) but both reach
    // close to the goal.
    const goal = s.goal;
    for (const path of [s.kinematicResult.path, s.learnedResult.path]) {
      const end = path[path.length - 1]!;
      expect(Math.hypot(end.x - goal.x, end.z - goal.z)).toBeLessThan(6);
    }
  });
});
