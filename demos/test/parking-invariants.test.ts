// Headless parking invariant tests — precision parking while avoiding
// obstacles, asserted entirely from recorded telemetry (no visual inspection).
//
// Drives the SAME `createRaceScenario` engine + the SAME `parkingCourse` /
// `PARKING_RACE_TUNING` definition the /parking web page imports, so a green
// test here means the page behaves the same way.
//
// IMPORTANT FINDING (2026-05): only `forward-pullin` actually parks. The
// telemetry exposes that `reverse-perp` and `parallel` are broken today — the
// planner fails almost every replan and the chassis only reaches the goal via
// the stall-guard teleport (reverse-perp), or it collides and stops at the
// wrong heading (parallel). Those two are encoded as `it.fails` so CI stays
// honest: the assertions describe CORRECT parking and currently fail; when the
// underlying maneuvers are fixed, these flip and must drop the `.fails`.

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

async function park(id: ParkingScenarioId, maxTicks: number) {
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
    success: SUCCESS,
    maxTicks,
    // Stop early once genuinely parked (saves CI wall time).
    done: (s) =>
      Math.hypot(s.state.x - goal.x, s.state.z - goal.z) < SUCCESS.posTol &&
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

  // KNOWN BROKEN — reverse perpendicular. The root cause is a planner-failure
  // storm: measured failedReplanRatio≈0.99 (141 consecutive failures). With
  // teleportation now disabled, this is no longer masked — the chassis drives
  // away, never reaches the goal, and the run times out as an honest failure
  // (parkedOk=false). We assert the root-cause signal on a SHORT budget (it
  // manifests immediately; a long run of 500 ms failing replans is needlessly
  // slow).
  it.fails('reverse-perp: planner is healthy (no replan-failure storm)', OPTS, async () => {
    const { report } = await park('reverse-perp', 300);
    const ctx = `\n${formatReport(report)}`;
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
  });

  // KNOWN BROKEN — parallel parking. Same root-cause planner failure
  // (measured failedReplanRatio≈0.42); downstream the footprint overlaps a
  // parked car (collided=true, minClearance=0) and the chassis stops ~21° off
  // the goal heading. Assert the root-cause signal on a short budget.
  it.fails('parallel: planner is healthy (no replan-failure storm)', OPTS, async () => {
    const { report } = await park('parallel', 300);
    const ctx = `\n${formatReport(report)}`;
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
  });
});
