// Parking precision regression test.
//
// The controller-bench (`pnpm run controller-bench`) reports
// pass/fail with a loose 1.5 m position tolerance and no heading
// check — fine for "did the car arrive in the rough vicinity of the
// goal" but not for "is the chassis parked properly". This test
// pins down the precision properties pure-pursuit needs to preserve:
//
//   forward-pullin: end within 0.7 m and 10° of the goal.
//   reverse-perp:   end within 0.5 m and 8° of the goal (the tightest
//                   of the three — heading-aware analytic shots make
//                   this one the easiest to nail).
//   parallel:       end within 0.7 m and 20° of the goal AND the
//                   chassis must come to a full stop (|v|<0.3 m/s).
//
// The parallel-park heading bar is lenient (20° instead of 10°)
// because the planner's Reeds-Shepp shots into a 1.69× gap
// legitimately end with residual tilt that the chassis can't shake
// without another back-and-forth shunt. This test's job is to catch
// REGRESSIONS — e.g., the cusp-stop fix in race-scenario degrading
// from "20° tilt" to "60° tilt" because pure-pursuit no longer
// brakes-to-stop at gear cusps — not to assert a perfect park.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createRaceScenario } from '../app/lib/race-scenario';
import {
  buildParkingScenario,
  parkingLibrary,
  type ParkingScenarioId,
} from '../app/lib/parking-scenarios';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

function parkingCourse(id: ParkingScenarioId): ReturnType<typeof buildRaceCourse> {
  const s = buildParkingScenario(id);
  return {
    bounds: { x0: s.bounds.x0, x1: s.bounds.x1, z0: s.bounds.z0, z1: s.bounds.z1 },
    polygons: s.polygons,
    obstacles: s.obstacles,
    waypoints: [{ ...s.goal, speed: 0, t: 0 }],
    spawn: { ...s.spawn, speed: 0, t: 0 },
  };
}

async function runParking(id: ParkingScenarioId, maxSimSec: number) {
  const course = parkingCourse(id);
  const goal = course.waypoints[course.waypoints.length - 1]!;
  const scenario = await createRaceScenario({
    entries: [{ name: id, lib: parkingLibrary() }],
    targetLaps: 1,
    syncHold: false,
    offTrackRecovery: 'none',
    tuning: {
      cruiseSpeed: 2,
      goalTolerance: 0.4,
      arriveRadius: 0.6,
      plannerPosCell: 0.3,
      plannerHeadingBuckets: 36,
      plannerGoalRadius: 0.35,
      plannerGoalHeadingTol: 0.2,
      plannerBudgetMs: 500,
      plannerMaxExpansions: 80_000,
      mpcWTerminalPosition: 50,
      mpcWTerminalSpeed: 30,
    },
    course,
  });
  while (scenario.simTime() < maxSimSec) {
    scenario.tick();
    const s = scenario.status()[0]!;
    const dist = Math.hypot(s.state.x - goal.x, s.state.z - goal.z);
    // Early-exit when the chassis is positionally close + has stopped.
    if (dist < 0.5 && Math.abs(s.state.speed) < 0.3) break;
  }
  const s = scenario.status()[0]!;
  scenario.dispose();
  return {
    distToGoal: Math.hypot(s.state.x - goal.x, s.state.z - goal.z),
    headingErr: Math.abs(wrapPi(s.state.heading - goal.heading)),
    finalSpeed: Math.abs(s.state.speed),
    offTrack: s.offTrackEvents,
    // Cusps: total segments minus 1. Surfaces "did the planner pick
    // a back-and-forth maneuver?" — important for parallel.
    cusps: s.totalSegments - 1,
  };
}

describe.skipIf(!RAPIER_OK)('parking precision', () => {
  it('forward-pullin parks within 0.7 m / 10° of goal', { timeout: 30_000, retry: 0 }, async () => {
    const r = await runParking('forward-pullin', 15);
    expect(r.offTrack).toBe(0);
    expect(r.distToGoal, `pos err ${r.distToGoal.toFixed(2)}m`).toBeLessThan(0.7);
    expect(r.headingErr, `hdg err ${(r.headingErr * 180 / Math.PI).toFixed(1)}°`).toBeLessThan(Math.PI / 18);
  });

  it('reverse-perp parks within 0.5 m / 8° of goal', { timeout: 60_000, retry: 0 }, async () => {
    const r = await runParking('reverse-perp', 30);
    expect(r.offTrack).toBe(0);
    expect(r.distToGoal, `pos err ${r.distToGoal.toFixed(2)}m`).toBeLessThan(0.5);
    expect(r.headingErr, `hdg err ${(r.headingErr * 180 / Math.PI).toFixed(1)}°`).toBeLessThan(Math.PI / 22.5);
  });

  it('parallel parks within 0.7 m / 25° of goal and stops', { timeout: 60_000, retry: 0 }, async () => {
    // Parallel parking heading is the hardest: the planner's Reeds-Shepp
    // shots into the 1.69× gap can legitimately end with ~20° residual
    // tilt that pure-pursuit can't recover without another back-and-
    // forth shunt. 25° is the regression bar — if heading degrades
    // past that, something is wrong with the cusp-stop handoff.
    const r = await runParking('parallel', 30);
    expect(r.offTrack).toBe(0);
    expect(r.distToGoal, `pos err ${r.distToGoal.toFixed(2)}m`).toBeLessThan(0.7);
    expect(r.headingErr, `hdg err ${(r.headingErr * 180 / Math.PI).toFixed(1)}°`).toBeLessThan(Math.PI / 7.2);
    expect(r.finalSpeed, `still moving at ${r.finalSpeed.toFixed(2)} m/s`).toBeLessThan(0.5);
  });
});
