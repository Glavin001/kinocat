// Regression coverage for the "ramp stopped taking the affordance" bug.
//
// The /ramp demo's whole point: when the ballistic-jump affordance is enabled
// the planner should PREFER it over detouring around the planner-only gap,
// because the jump is strictly cheaper. A merge (204455a, the grid-Dijkstra
// float32 livelock fix) made the obstacle-aware goal-distance heuristic
// actually route around the gap — which is INADMISSIBLE once an affordance can
// jump over that obstacle. It overestimated cost-to-go near the launch, so
// branch-and-bound pruned the jump branch the moment a detour incumbent
// appeared, and the car drove around instead of jumping.
//
// The old static snapshot tests all planned from the exact spawn pose
// (speed 0, x=-45), which — by luck of the greedy straight-line approach —
// still found the jump, so they stayed green while the live demo (replanning
// every 120 ms from the MOVING car) always detoured. These tests plan from the
// moving / perturbed start states the live loop actually encounters, and
// assert the physical outcome, so the regression cannot hide behind the spawn
// pose again.

import { describe, expect, it } from 'vitest';
import {
  buildRampCourse,
  planRampDemo,
  planTakesJump,
  planLateralExcursion,
  RAMP_MAX_EXPANSIONS,
  type RampCourse,
} from '../app/lib/ramp-scenarios';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { CarKinematicState } from 'kinocat/agent';

function planFrom(
  course: RampCourse,
  start: Partial<CarKinematicState>,
  withAffordance = true,
) {
  // A fresh world per query mirrors the demo's reused InMemoryNavWorld while
  // keeping the tests independent. Live expansion budget (20k), not the
  // roomier test budget, so we exercise exactly what the page runs.
  const world = new InMemoryNavWorld(course.polygons, course.obstacles);
  return planRampDemo({
    state: { x: -45, z: 0, heading: 0, speed: 0, t: 0, ...start },
    goal: { ...course.goal, t: 0 },
    course,
    world,
    withoutAffordances: !withAffordance,
    deadlineMs: Infinity,
    maxExpansions: RAMP_MAX_EXPANSIONS,
  });
}

describe('ramp affordance selection (planner)', () => {
  const course = buildRampCourse();
  const gapHalfWidth = course.gaps[0]!.hz; // the detour must swing past this

  // The states the live replan loop actually hits: the car is moving, and it
  // is never at exactly x=-45 by the time the first post-spawn replan fires.
  const movingStarts: Array<Partial<CarKinematicState>> = [
    { x: -45, speed: 0 },
    { x: -44.75, speed: 2.7 }, // the exact pose the demo replanned at pre-fix
    { x: -44, speed: 2 },
    { x: -43, speed: 4 },
    { x: -40, speed: 8 },
    { x: -30, speed: 11 },
    { x: -20, speed: 11 },
  ];

  for (const s of movingStarts) {
    it(`takes the jump from x=${s.x} speed=${s.speed}`, () => {
      const res = planFrom(course, s, true);
      expect(res.found).toBe(true);
      const ctx = `lateral=${planLateralExcursion(res.path).toFixed(1)} exp=${res.stats.expansions}`;
      expect(planTakesJump(res.path, course), `expected JUMP; ${ctx}`).toBe(true);
      // A jump plan hugs the z=0 centreline; a detour swings out past the gap.
      expect(planLateralExcursion(res.path), ctx).toBeLessThan(gapHalfWidth / 2);
    });
  }

  it('detours (does NOT jump) when the affordance is disabled', () => {
    // The contrast case proves the gap genuinely forces a choice — the jump is
    // preferred only because it is available, not because the detour is blocked.
    const res = planFrom(course, { x: -44, speed: 2 }, false);
    expect(res.found).toBe(true);
    expect(planTakesJump(res.path, course)).toBe(false);
    expect(planLateralExcursion(res.path)).toBeGreaterThan(gapHalfWidth);
  });

  it('the jump plan arrives sooner than the detour from the same moving start', () => {
    const jump = planFrom(course, { x: -44, speed: 2 }, true);
    const detour = planFrom(course, { x: -44, speed: 2 }, false);
    expect(jump.found && detour.found).toBe(true);
    const tJump = jump.path[jump.path.length - 1]!.t;
    const tDetour = detour.path[detour.path.length - 1]!.t;
    expect(tJump, `jump ${tJump} should beat detour ${tDetour}`).toBeLessThan(tDetour);
  });
});
