// Live goal-progress viewer channel (closed loop, real Rapier plant).
//
// The race course now carries its canonical scenario goal (`course.goal`,
// authored in buildRaceCourse via authorRaceCoursePlanes) and the scenario
// runner advances a compiled automaton over each car's EXECUTED trajectory —
// the read-only channel the /raceprimitives HUD renders with
// GoalProgressPanel. This test drives one kinematic car a few gates into the
// course and asserts the channel is (a) present, (b) consistent with the
// race's own waypoint bookkeeping, and (c) inert (planner behaviour on a
// multi-waypoint course is unchanged by the authored goal — pinned indirectly
// by every other race benchmark, and directly here by the car actually
// clearing gates).

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createRaceScenario, PHYSICS_DT } from '../app/lib/race-scenario';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { kinematicEntry } from '../app/lib/headless-race';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('live goal-automaton progress (executed trajectory)', () => {
  it('exposes the compiled automaton + a per-car snapshot that advances with the race', { timeout: 240_000 }, async () => {
    const course = buildRaceCourse();
    const scenario = await createRaceScenario({
      entries: [kinematicEntry('kin')],
      course,
      // Deterministic: expansion cap binds, not wall clock.
      tuning: { plannerBudgetMs: 10_000 },
    });
    try {
      const automaton = scenario.goalAutomaton();
      expect(automaton).not.toBeNull();
      // One phase per gate in the compiled circuit.
      const maxDepth = automaton!.states.reduce((m, s) => Math.max(m, s.depth), 0);
      expect(maxDepth).toBeGreaterThanOrEqual(course.waypoints.length - 1);

      // Drive until the car has cleared 2 gates (or 45 s sim, whichever
      // first). The kinematic car reaches gate 0 within ~8 s from spawn.
      let cleared = 0;
      let goalDepthAt2: number | null = null;
      const maxTicks = Math.round(45 / PHYSICS_DT);
      for (let i = 0; i < maxTicks; i++) {
        const r = scenario.tick(PHYSICS_DT);
        const car = r.cars[0]!;
        expect(car.goalProgress).not.toBeNull();
        cleared = car.metrics.waypointsCleared;
        if (cleared >= 2) {
          goalDepthAt2 = car.goalProgress!.depth;
          break;
        }
      }
      expect(cleared).toBeGreaterThanOrEqual(2);
      // The automaton advanced with the executed pass. It may lag the
      // position-only waypoint bookkeeping by one gate (its guards also
      // require the chord-aligned heading band), never lead it.
      expect(goalDepthAt2).not.toBeNull();
      expect(goalDepthAt2!).toBeGreaterThanOrEqual(1);
      expect(goalDepthAt2!).toBeLessThanOrEqual(2);

      // reset() rewinds the viewer channel with the rest of the scenario.
      scenario.reset();
      const s0 = scenario.status()[0]!;
      expect(s0.goalProgress!.depth).toBe(automaton!.states[automaton!.start]?.depth ?? 0);
      expect(s0.goalProgress!.laps).toBe(0);
    } finally {
      scenario.dispose();
    }
  });
});
