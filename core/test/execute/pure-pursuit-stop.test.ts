// Regression suite for pure-pursuit's stop behavior, exercised against a
// FINITE-deceleration plant (the perfect/instant-tracking sweeps hide stopping
// dynamics entirely). These cover two bugs fixed in pure-pursuit.ts:
//
//   BUG A — respectPathSpeed folded the goal sample's speed into the speed cap
//     unconditionally, so any plan ending in a stop pinned the target to ~0
//     from far away and the vehicle never moved. Fixed: the terminal speed is
//     only applied once the goal is within the lookahead window.
//
//   BUG B — standalone pure-pursuit had no passed-the-goal latch and floored
//     the brake-distance term at lookaheadMin, so above ~5 m/s approach it
//     couldn't stop, crossed the goal, re-accelerated and ran away. Fixed: a
//     stop-terminated plan lets the brake term decelerate to 0, and a stateless
//     passed-goal latch holds the stop once the vehicle is past the terminal.
//
// Both fixes are gated on the plan ending in a genuine stop (terminal speed
// ~0), so drive-through / racing-loop behavior is unchanged — the drive-through
// guard below pins that.

import { describe, it, expect } from 'vitest';
import { purePursuit } from '../../src/execute/pure-pursuit';
import { smoothSpeedProfile } from '../../src/execute/speed-profile';
import { learnedForwardSim, DEFAULT_LEARNED_PARAMS } from '../../src/agent/vehicle';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import { SWEEP_AGENT } from '../fixtures/vehicle-sweep';

const GOAL_X = 40;

const BASE_CFG: PurePursuitConfig = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: 10,
  maxAccel: 6.5,
  maxDecel: 8, // matches the plant's real decel (DEFAULT_LEARNED_PARAMS.maxDecel)
  cruiseSpeed: 6,
  goalTolerance: 0.5,
  minTurnRadius: 2,
};

/** Closed loop with a FINITE-deceleration plant (learnedForwardSim). Returns
 *  the end state and furthest x reached. */
function runStop(cfg: PurePursuitConfig, path: PlanPath, steps = 4000) {
  const sim = learnedForwardSim(DEFAULT_LEARNED_PARAMS, SWEEP_AGENT);
  let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
  const dt = 1 / 60;
  let maxX = 0;
  for (let k = 0; k < steps; k++) {
    const cmd = purePursuit(s, path, cfg);
    s = sim(s, [cmd.steering, cmd.targetSpeed], dt);
    maxX = Math.max(maxX, s.x);
    if (cmd.atGoal && Math.abs(s.speed) < 0.05) break;
  }
  return { end: s, maxX, overshoot: maxX - GOAL_X };
}

/** A straight plan that cruises, then STOPS at the goal (terminal speed 0). */
function stopPath(cruise: number): PlanPath {
  const p: PlanPath = [];
  for (let x = 0; x <= GOAL_X; x += 2) {
    p.push({ x, z: 0, heading: 0, speed: x >= GOAL_X ? 0 : cruise, t: 0 });
  }
  return p;
}

/** The production shape for respectPathSpeed: a friction-circle speed-profiled
 *  plan whose speeds ramp smoothly down to 0 at the terminal. */
function profiledStopPath(cruise: number): PlanPath {
  const raw = stopPath(cruise);
  return smoothSpeedProfile(raw, {
    aLatMax: 10,
    aLonMaxAccel: 6,
    aLonMaxDecel: 4,
    maxSpeed: cruise,
    minSpeed: 0,
    curvatureOverride: raw.map((_, i) => (i === raw.length - 1 ? 1e6 : 0)),
  });
}

describe('pure-pursuit stop — BUG A fixed: respectPathSpeed drives a stop-terminated plan', () => {
  it('drives the full plan and comes to rest (does not stall at the start)', () => {
    const { end } = runStop({ ...BASE_CFG, respectPathSpeed: true }, profiledStopPath(6));
    // Before the fix the terminal-speed clamp pinned the target to ~0 and the
    // vehicle never left x≈0 (stalled). Now it traverses the whole plan and
    // stops. NOTE: respectPathSpeed caps to the MIN planned speed over a
    // forward lookahead window, so it brakes for the terminal ~lookaheadMax
    // early and rests just short of the goal — a separate, pre-existing
    // conservatism of the window-min, not the stall bug fixed here. Terminal
    // precision is the job of the profile/MPC, not this safety cap.
    expect(end.x).toBeGreaterThan(GOAL_X - BASE_CFG.lookaheadMax - 2); // ≫ 0: no stall
    expect(Math.abs(end.speed)).toBeLessThan(0.3); // came to rest
  });

  it('respects an in-window slow zone (the feature still works)', () => {
    // A slow zone in the forward window must still cap the target speed.
    const slow: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 2, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 4, z: 0, heading: 0, speed: 2, t: 0 }, // slow zone, within lookahead
      { x: 6, z: 0, heading: 0, speed: 2, t: 0 },
      { x: 8, z: 0, heading: 0, speed: 6, t: 0 },
    ];
    const cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 6, t: 0 };
    const cmd = purePursuit(cur, slow, { ...BASE_CFG, respectPathSpeed: true });
    expect(Math.abs(cmd.targetSpeed)).toBeCloseTo(2, 5);
  });
});

describe('pure-pursuit stop — BUG B fixed: no runaway across the approach-speed sweep', () => {
  // No speed profile, no respectPathSpeed: the controller's own brake-distance
  // term must bring it to a stop near the goal at every approach speed up to
  // the agent's maxSpeed (8 m/s) — including the speeds that used to run away.
  for (const cruise of [3, 5, 6, 7, 8]) {
    it(`stops near the goal from a ${cruise} m/s approach (bounded overshoot)`, () => {
      const { end, overshoot } = runStop({ ...BASE_CFG, cruiseSpeed: cruise }, stopPath(cruise));
      const ctx = `cruise=${cruise} endX=${end.x.toFixed(2)} overshoot=${overshoot.toFixed(2)} endV=${end.speed.toFixed(3)}`;
      // Comes to rest...
      expect(Math.abs(end.speed), ctx).toBeLessThan(0.3);
      // ...near the goal, with overshoot bounded by physics (no runaway).
      expect(Math.hypot(end.x - GOAL_X, end.z), ctx).toBeLessThan(2.0);
      expect(overshoot, ctx).toBeLessThan(2.0);
    });
  }
});

describe('pure-pursuit stop — drive-through regression guard', () => {
  it('a drive-through plan (no stop terminal) is unaffected by the stop fixes', () => {
    // Terminal speed > 0 ⇒ NOT a stop terminal: the passed-goal latch must not
    // fire and the lookaheadMin brake-floor must still apply, so a tight-corner
    // plan reports the curvature-limited speed exactly as before.
    const driveThrough: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 0.1, z: 50, heading: Math.PI / 2, speed: 6, t: 0 },
    ];
    const cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cfg: PurePursuitConfig = { ...BASE_CFG, cruiseSpeed: 50, maxDecel: 50, minTurnRadius: undefined };
    const cmd = purePursuit(cur, driveThrough, cfg);
    const kappa = Math.abs(cmd.steering);
    expect(Math.abs(cmd.targetSpeed)).toBeCloseTo(Math.sqrt(cfg.maxLateralAccel / kappa), 6);
  });
});
