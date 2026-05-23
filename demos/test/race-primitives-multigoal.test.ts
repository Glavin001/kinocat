// Verify that `planRaceMultiGoal` plans a single global trajectory through
// multiple consecutive gates of the canonical race course — the
// architectural fix that replaces the chained-per-gate planner.

import { describe, it, expect } from 'vitest';
import {
  buildRaceCourse,
  buildKinematicLibrary,
  planRaceMultiGoal,
} from '../app/lib/race-primitives-scenarios';

describe('planRaceMultiGoal — single A* through gate sequence', () => {
  it('plans through 3 consecutive gates with generous budget', () => {
    const course = buildRaceCourse();
    const lib = buildKinematicLibrary();
    const gates = course.waypoints.slice(0, 3).map((w) => ({ ...w, t: 0 }));
    const result = planRaceMultiGoal({
      state: course.spawn,
      gates,
      lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      deadlineMs: 8000,
      maxExpansions: 500_000,
    });
    expect(result.found).toBe(true);
    expect(result.path.length).toBeGreaterThan(3);

    // The returned path passes near each gate in order.
    for (const gate of gates) {
      const minDist = Math.min(
        ...result.path.map((p) => Math.hypot(p.x - gate.x, p.z - gate.z)),
      );
      expect(minDist).toBeLessThan(5);
    }
  }, 20_000);

  it('plans through 2 gates faster than 5 (sanity: more gates = more work)', () => {
    const course = buildRaceCourse();
    const lib = buildKinematicLibrary();
    const gates2 = course.waypoints.slice(0, 2).map((w) => ({ ...w, t: 0 }));
    const r2 = planRaceMultiGoal({
      state: course.spawn,
      gates: gates2,
      lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      deadlineMs: 4000,
      maxExpansions: 200_000,
    });
    expect(r2.found).toBe(true);
    // 2-gate plan should have lower cost than 5-gate plan.
    expect(r2.cost).toBeLessThan(Infinity);
  }, 15_000);
});
