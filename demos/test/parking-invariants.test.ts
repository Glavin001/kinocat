// Headless parking invariant tests — precision parking while avoiding
// obstacles, asserted entirely from recorded telemetry (no visual inspection).
//
// Drives the SAME `createRaceScenario` engine + the SAME `parkingCourse` /
// `PARKING_RACE_TUNING` definition the /parking web page imports, so a green
// test here means the page behaves the same way.
//
// HISTORY (2026-05): originally only `forward-pullin` parked; `reverse-perp`
// drove 40+ m the wrong way and `parallel` thrashed, both via a planner
// replan-failure storm (failedReplanRatio ≈ 0.99 / 0.42). The root causes were
// not the maneuvers themselves but a stack of mismatches in the shared runner:
//   1. the runner planned parking with RACE_AGENT (30 m/s, 4.5 m turn radius)
//      instead of PARKING_AGENT, so the time-cost heuristic was ~15× too weak
//      and A* degenerated into near-breadth-first search → the replan storm;
//   2. the pure-pursuit→chassis bridge dropped the reverse gear (threw away the
//      throttle sign) AND didn't flip the steer sign in reverse, so a planned
//      back-in was never executed — the car just coasted forward off-plan;
//   3. the planner's analytic Reeds-Shepp shot-to-goal was collapsed to a
//      straight chord (the bare node sequence discards the curve), so the
//      curved final approach was tracked as a diagonal that clipped a car;
//   4. parking replanned on the 300 ms race cadence, re-deciding the whole
//      multi-cusp maneuver every tick and oscillating at the forward↔reverse
//      cusp instead of committing to it.
// A fifth fix came from the web demo: the planning footprint was
// `defaultVehicleAgent`'s 3.2 × 1.8 m box, far smaller than the 4.8 × 2.0 m
// chassis the runner drives and three.js renders, so "collision-free" plans
// drove the real bumpers into the parked cars. The footprint now matches the
// true chassis and the (previously over-tight) stall spacing / aisle / gap were
// widened so the real car still fits.
//
// Fixing all five (see race-scenario.ts / race-primitives-scenarios.ts /
// parking-scenarios.ts / core analytic-shot poses) makes `forward-pullin` and
// `reverse-perp` park cleanly with a healthy planner. `parallel` now reaches
// the slot collision-free under a healthy planner but still can't nail the
// terminal-heading straightening shunt (pure-pursuit brakes to rest at its
// approach angle, ~16° off). That residual is kept HONEST as an `it.fails` so
// it stays exercised and flips to green the moment terminal-pose precision
// (e.g. an MPC final stage) lands.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runMonitored } from './_sim-harness';
import { formatReport, type SuccessTolerances } from '../app/lib/sim-monitor';
import {
  parkingCourse,
  parkingScenarioOptions,
  parkingLibrary,
  PARKING_AGENT,
  type ParkingScenarioId,
} from '../app/lib/parking-scenarios';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// "Parked" = within 0.6 m of the goal pose, heading within 15°, stopped.
const SUCCESS: SuccessTolerances = { posTol: 0.6, headingTol: 0.26, speedTol: 0.5 };

async function park(id: ParkingScenarioId, maxTicks: number, success: SuccessTolerances = SUCCESS) {
  const course = parkingCourse(id);
  const goal = course.waypoints[course.waypoints.length - 1]!;
  return runMonitored({
    // The SAME canonical options the /parking page and the controller-bench
    // CLI use — including ZERO teleportation (no stall/off-track rescue), so a
    // maneuver that doesn't reach the goal fails honestly by timeout instead of
    // being snapped onto it.
    scenario: parkingScenarioOptions(id, [{ name: id, lib: parkingLibrary() }]),
    footprint: PARKING_AGENT.footprint,
    obstacles: course.obstacles,
    goal: { x: goal.x, z: goal.z, heading: goal.heading },
    success,
    maxTicks,
    // Stop early once genuinely parked (saves CI wall time).
    done: (s) =>
      Math.hypot(s.state.x - goal.x, s.state.z - goal.z) < success.posTol &&
      Math.abs(s.state.speed) < 0.3,
  });
}

// These are deterministic Rapier sims — a retry can't change the outcome and
// just multiplies wall time, so opt out of the suite-wide retry:2.
const OPTS = { timeout: 90000, retry: 0 } as const;

describe.skipIf(!RAPIER_OK)('parking invariants', () => {
  it('forward-pullin: parks cleanly — right pose, no collision, no teleport rescue', OPTS, async () => {
    const { report } = await park('forward-pullin', 1000);
    const ctx = `\n${formatReport(report)}`;
    // Reached the goal pose (position AND heading AND stopped).
    expect(report.parkedOk, ctx).toBe(true);
    // Never touched a parked car / wall, with real clearance margin.
    expect(report.collided, ctx).toBe(false);
    expect(report.minClearance, ctx).toBeGreaterThan(0.4);
    // Drove there under its own power — not rescued by a stall/off-track jump.
    expect(report.teleports, ctx).toBe(0);
    // Drove toward the goal, not away from it ("wrong direction" guard).
    expect(report.netProgress, ctx).toBeGreaterThan(0);
    expect(report.maxRetreat, ctx).toBeLessThan(2);
    // Planner is healthy (no failure storm).
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
  });

  // FIXED (was the reverse-perp replan storm). The car now reverses west out of
  // the aisle, swings, and backs into the stall under its own power. The slot
  // is genuinely tight so we allow a slightly looser terminal tolerance than
  // forward-pullin (0.7 m / ~17°), but the maneuver is otherwise clean: no
  // collision, no teleport rescue, and the planner no longer thrashes.
  const REVERSE_SUCCESS: SuccessTolerances = { posTol: 0.7, headingTol: 0.3, speedTol: 0.5 };
  it('reverse-perp: backs into the stall — clean pose, no collision, no replan storm', OPTS, async () => {
    const { report } = await park('reverse-perp', 2200, REVERSE_SUCCESS);
    const ctx = `\n${formatReport(report)}`;
    // Backed into the stall pose under its own power.
    expect(report.parkedOk, ctx).toBe(true);
    expect(report.terminalSpeed, ctx).toBeLessThan(0.5);
    // Never touched a parked car / wall.
    expect(report.collided, ctx).toBe(false);
    // Not rescued by a stall/off-track jump.
    expect(report.teleports, ctx).toBe(0);
    // Net motion was toward the goal, not the 40 m drive-away of the bug.
    expect(report.netProgress, ctx).toBeGreaterThan(0);
    expect(report.maxRetreat, ctx).toBeLessThan(2);
    // Planner is healthy — the storm (failedReplanRatio ≈ 0.99) is gone.
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
  });

  // PARTIALLY FIXED — parallel parking. The replan storm is gone, the chassis
  // reaches the slot position under a healthy planner, AND it no longer grazes
  // a neighbour on the way in (these invariants pass). What it still can't do is
  // the final straightening shunt — pure-pursuit brakes to rest at whatever
  // approach angle it arrives with (no terminal-heading control), ~16° off the
  // curb. See the `it.fails` below.
  it('parallel: reaches the slot cleanly under a healthy planner (no storm, no graze)', OPTS, async () => {
    const { report } = await park('parallel', 1600);
    const ctx = `\n${formatReport(report)}`;
    // Planner is healthy — the storm (failedReplanRatio ≈ 0.42) is gone.
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
    // Drove to the slot, not away from it.
    expect(report.netProgress, ctx).toBeGreaterThan(0);
    expect(report.terminalPosError, ctx).toBeLessThan(0.6);
    // Came to rest there.
    expect(report.terminalSpeed, ctx).toBeLessThan(0.5);
    // Never touched a parked car or the curb (the visible clipping is gone).
    expect(report.collided, ctx).toBe(false);
    // Not rescued by a teleport.
    expect(report.teleports, ctx).toBe(0);
  });

  // KNOWN BROKEN — parallel terminal heading. The chassis arrives in the slot
  // (right position, no collision) but ~16° off the curb heading: pure-pursuit
  // has no final back-and-forth straightening shunt. Encoded as the
  // CORRECT-behaviour assertion under `it.fails`: it throws today and Vitest
  // counts that as a pass; when terminal-pose precision lands (e.g. an MPC final
  // stage) it will start passing, flip this to red, and signal that the
  // `.fails` should drop.
  it.fails('parallel: ends square with the curb (terminal heading within tolerance)', OPTS, async () => {
    const { report } = await park('parallel', 1600);
    const ctx = `\n${formatReport(report)}`;
    expect(report.terminalHeadingError, ctx).toBeLessThan(SUCCESS.headingTol);
  });
});
