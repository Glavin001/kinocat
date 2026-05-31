// Headless racing invariant tests — "racing through waypoints" asserted from
// recorded telemetry. Drives the SAME `createRaceScenario` engine the
// /raceprimitives page uses.
//
// Like the existing headless-race smoke test, we do NOT assert lap completion
// (a full lap is ~40 s sim, too slow for a free CI runner). We tick a short
// budget and assert the behaviours you'd otherwise eyeball: the car makes
// waypoint progress, stays within speed/comfort limits, and the planner isn't
// thrashing.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runMonitored } from './_sim-harness';
import { formatReport } from '../app/lib/sim-monitor';
import {
  buildRaceCourse,
  buildKinematicLibrary,
  RACE_AGENT,
} from '../app/lib/race-primitives-scenarios';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

describe.skipIf(!RAPIER_OK)('race invariants', () => {
  it('races through waypoints within speed/comfort limits and without replan thrash', { timeout: 90000, retry: 0 }, async () => {
    const course = buildRaceCourse();
    const { status, report } = await runMonitored({
      scenario: {
        entries: [{ name: 'kin', lib: buildKinematicLibrary() }],
        syncHold: false,
        // No teleport rescue: if the car spins off-track it must fail honestly,
        // not get snapped back. In a clean 7 s run it never leaves the track.
        offTrackRecovery: 'none',
        stallTimeoutMs: Number.POSITIVE_INFINITY,
      },
      footprint: RACE_AGENT.footprint,
      obstacles: course.obstacles, // empty for the race course
      maxTicks: 420, // 7 s sim
    });
    const ctx = `\n${formatReport(report)}`;

    // Actually racing: at least one waypoint cleared in the first few seconds.
    expect(status.metrics.waypointsCleared, ctx).toBeGreaterThanOrEqual(1);

    // Finite, sane state stream.
    expect(Number.isFinite(status.state.x), ctx).toBe(true);
    expect(Number.isFinite(status.state.z), ctx).toBe(true);
    expect(Number.isFinite(status.state.speed), ctx).toBe(true);

    // Stays on track under its own power (no off-track teleport in 7 s).
    expect(report.teleports, ctx).toBe(0);

    // Speed within the physical ceiling.
    expect(report.peakSpeed, ctx).toBeLessThanOrEqual(RACE_AGENT.maxSpeed * 1.05);

    // Comfort / jitter bound (lateral-accel RMS — tracker-agnostic). Measured
    // ~6.5; assert generously and ratchet down later.
    expect(report.lateralAccelRms, ctx).toBeLessThan(15);

    // Planner is producing plans and not failing.
    expect(report.successfulReplans, ctx).toBeGreaterThan(0);
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
    // Replan cadence is bounded (300 ms base + adaptive triggers ⇒ a handful/s,
    // not a thrash). Measured ~5.3/s.
    expect(report.replansPerSec, ctx).toBeLessThan(12);
  });
});
