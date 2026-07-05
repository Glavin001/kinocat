// Planner-side benchmark: confirm a v2-derived race library still produces
// a feasible plan from spawn → first waypoint on the canonical race scenario,
// and (sanity-check) the v2 plan's total path length is comparable to the
// legacy learned plan. Headless — no Rapier execution; purely the planner +
// motion-primitive library.

import { describe, it, expect } from 'vitest';
import { DEFAULT_LEARNED_PARAMS } from 'kinocat/agent';
import { buildParametricOnlyModel, DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import {
  buildRaceCourse,
  buildLearnedRaceLibrary,
  buildLearnedRaceLibraryV2,
  planRace,
  RACE_TEST_MAX_EXPANSIONS,
} from '../app/lib/race-primitives-scenarios';

function pathLength(path: { x: number; z: number }[]): number {
  let s = 0;
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1]!;
    const b = path[i]!;
    s += Math.hypot(b.x - a.x, b.z - a.z);
  }
  return s;
}

describe('v2 learned race library plans through the canonical course', () => {
  it('finds a plan from spawn → first waypoint within budget', () => {
    const course = buildRaceCourse();
    const v2Model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const v2Lib = buildLearnedRaceLibraryV2(v2Model);

    const result = planRace({
      state: course.spawn,
      goal: course.waypoints[0]!,
      lib: v2Lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      deadlineMs: Infinity,
      maxExpansions: RACE_TEST_MAX_EXPANSIONS,
    });

    expect(result.found).toBe(true);
    expect(result.path.length).toBeGreaterThanOrEqual(2);
    // Path should end near the goal.
    const end = result.path[result.path.length - 1]!;
    expect(Math.hypot(end.x - course.waypoints[0]!.x, end.z - course.waypoints[0]!.z)).toBeLessThan(6);
  });

  it('v2 plan length is comparable to legacy plan length on the same goal', () => {
    const course = buildRaceCourse();
    const v2Model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const v2Lib = buildLearnedRaceLibraryV2(v2Model);
    const legacyLib = buildLearnedRaceLibrary(DEFAULT_LEARNED_PARAMS);

    const goal = course.waypoints[2]!; // a slalom apex — more discriminating
    const legacy = planRace({
      state: course.spawn,
      goal,
      lib: legacyLib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      deadlineMs: Infinity,
      maxExpansions: RACE_TEST_MAX_EXPANSIONS,
    });
    const v2 = planRace({
      state: course.spawn,
      goal,
      lib: v2Lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      deadlineMs: Infinity,
      maxExpansions: RACE_TEST_MAX_EXPANSIONS,
    });

    expect(legacy.found).toBe(true);
    expect(v2.found).toBe(true);

    const lLen = pathLength(legacy.path);
    const vLen = pathLength(v2.path);
    // Sanity: both plans should be similar order of magnitude (within 50%).
    // We don't enforce v2 < legacy here because primitives integrate
    // dynamics differently — the value comes from EXECUTION accuracy
    // (covered by race-primitives-model.test.ts) not raw planned distance.
    expect(vLen).toBeGreaterThan(0);
    expect(vLen).toBeLessThan(lLen * 1.5);
    expect(lLen).toBeLessThan(vLen * 1.5);
  });
});
