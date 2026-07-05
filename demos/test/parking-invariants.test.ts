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
// `reverse-perp` park cleanly with a healthy planner. `parallel` used to reach
// the slot but rest ~16° off the curb; a Stanley-style terminal heading term
// (`terminalHeadingGain`, gated by the runner on distance to the TRUE goal) now
// straightens it onto the curb, so all three park square + centred and the
// predicate (`evaluateParked`, incl. a centering bound) agrees.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { runMonitored } from './_sim-harness';
import { formatReport, type SuccessTolerances } from '../app/lib/sim-monitor';
import {
  parkingCourse,
  parkingScenarioOptions,
  parkingLibrary,
  buildParkingScenario,
  evaluateParked,
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
  const scenario = buildParkingScenario(id);
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
    // Stop early only once GENUINELY parked (footprint in the stall + squared up
    // + stopped). Gating on the shared predicate can't false-trigger mid-
    // maneuver (a forward↔reverse cusp pause isn't inside the stall) and lets a
    // scenario that never squares up (parallel) run the full budget and fail
    // honestly rather than being captured mid-settle.
    done: (s) => evaluateParked(s.state, scenario).parked,
  });
}

// These are deterministic Rapier sims — a retry can't change the outcome and
// just multiplies wall time, so opt out of the suite-wide retry:2.
const OPTS = { timeout: 90000, retry: 0 } as const;

describe.skipIf(!RAPIER_OK)('parking invariants', () => {
  it('forward-pullin: parks cleanly — fits the stall, no collision, no teleport rescue', OPTS, async () => {
    const { report, status } = await park('forward-pullin', 1000);
    // The SHARED "in-the-stall" predicate (same one the /parking HUD and the
    // controller-bench CLI use): the footprint must sit inside the stall
    // silhouette, squared up, and stopped — not merely be near the goal point.
    const ev = evaluateParked(status.state, buildParkingScenario('forward-pullin'));
    const ctx = `\n${formatReport(report)}\nparked=${JSON.stringify(ev)}`;
    expect(ev.parked, ctx).toBe(true);
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
  it('reverse-perp: backs into the stall — fits the silhouette, no collision, no replan storm', OPTS, async () => {
    const { report, status } = await park('reverse-perp', 2600, REVERSE_SUCCESS);
    const ev = evaluateParked(status.state, buildParkingScenario('reverse-perp'));
    const ctx = `\n${formatReport(report)}\nparked=${JSON.stringify(ev)}`;
    // Backed into the stall and fits the silhouette, squared up + stopped. This
    // PASSES since the terminal brake was tightened (goalTolerance 0.4 → 0.08):
    // the chassis now stops ON the goal instead of ~0.5 m short. (reverse-perp's
    // residual was position, not heading — see the parallel it.fails for the
    // heading gap that remains.)
    expect(ev.parked, ctx).toBe(true);
    // Never touched a parked car / wall.
    expect(report.collided, ctx).toBe(false);
    // Not rescued by a stall/off-track jump.
    expect(report.teleports, ctx).toBe(0);
    // Net motion was toward the goal, not the 40 m drive-away of the old bug.
    expect(report.netProgress, ctx).toBeGreaterThan(0);
    expect(report.maxRetreat, ctx).toBeLessThan(2);
    // Planner is healthy — the storm (failedReplanRatio ≈ 0.99) is gone.
    expect(report.failedReplanRatio, ctx).toBeLessThan(0.3);
  });

  // Parallel parking. The replan storm is gone, the chassis reaches the slot
  // under a healthy planner without grazing a neighbour, AND comes to rest
  // CENTRED (terminalPosError < 0.6 m) — the centering that the terminal heading
  // term + the `evaluateParked` centering bound now guarantee. Squareness is
  // asserted separately below.
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

  // FIXED — parallel terminal heading. The chassis reaches the slot position and
  // now squares up with the curb (heading → ~0°). The fix is a Stanley-style
  // heading-alignment term added to pure-pursuit (`terminalHeadingGain`),
  // confined to within 2 m of the goal so it corrects the short terminal
  // straightening curve — which pure-pursuit's lookahead otherwise cuts, leaving
  // the car ~16° off — without perturbing the tight clearance-critical dive past
  // the parked cars. (An earlier free-space smooth pose-regulator was rejected:
  // its own kinematic tests showed it over-rotates / limit-cycles. This works
  // because it follows the planner's collision-free path tangent instead.)
  it('parallel: ends square with the curb — fits squarely inside the stall', OPTS, async () => {
    const { report, status } = await park('parallel', 1600);
    const ev = evaluateParked(status.state, buildParkingScenario('parallel'));
    const ctx = `\n${formatReport(report)}\nparked=${JSON.stringify(ev)}`;
    expect(ev.parked, ctx).toBe(true);
    expect(report.collided, ctx).toBe(false);
    expect(report.teleports, ctx).toBe(0);
  });
});
