// KNOWN BUGS — characterization tests for pure-pursuit's stop behavior.
//
// These document two real defects found by closing the tracking loop with a
// FINITE-deceleration plant (the earlier sweep tests used perfect/instant speed
// tracking, which hid both). Per maintainer decision this PR does NOT change
// the controller; instead each bug is captured two ways:
//
//   • a CURRENT-BEHAVIOR test that asserts what actually happens today (green,
//     so it pins the symptom and will break if the behavior drifts), and
//   • an `it.fails` TRIPWIRE that asserts the DESIRED behavior — it passes only
//     because the desired property currently does NOT hold. The day the bug is
//     fixed, the tripwire flips red, signaling "remove .fails and keep the
//     positive assertion."
//
// ── BUG A ──────────────────────────────────────────────────────────────────
// pure-pursuit.ts (respectPathSpeed branch) folds the GOAL sample's speed into
// the speed cap UNCONDITIONALLY:
//     const last = Math.abs(path[path.length - 1]!.speed);
//     if (last < vPath) vPath = last;
// so for any plan that ends in a stop (terminal speed ~0) the target speed is
// pinned to ~0 from the very first tick — even when the goal is dozens of
// metres away — and the vehicle never moves. respectPathSpeed:true is the
// DEFAULT tuning (demos/app/lib/race-scenario.ts); racing only escapes it
// because loop waypoints are drive-through (never terminal speed 0). Correct
// behavior: only apply the terminal speed once the goal is within the lookahead
// window.
//
// ── BUG B ──────────────────────────────────────────────────────────────────
// Standalone (no speed-profile) pure-pursuit has no "passed-the-goal" latch and
// floors the brake-distance term `vGoal` at lookaheadMin, so above an approach
// speed of ~5 m/s the vehicle cannot stop inside goalTolerance, crosses to the
// far side (atGoal flips false), re-accelerates to cruise, and runs away by
// hundreds of metres. Correct behavior: latch a stop once the goal is reached /
// passed, so overshoot stays bounded at any approach speed.

import { describe, it, expect } from 'vitest';
import { purePursuit } from '../../src/execute/pure-pursuit';
import { learnedForwardSim, DEFAULT_LEARNED_PARAMS } from '../../src/agent/vehicle';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import { wrapAngle } from '../../src/internal/math';
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

/** Closed loop with a FINITE-deceleration plant (the key to exposing both
 *  bugs). Returns the end state and the furthest x reached. */
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

// A straight plan that cruises, then STOPS at the goal (terminal speed 0) — the
// ordinary "drive there and park" shape.
function stopPath(cruise: number): PlanPath {
  const p: PlanPath = [];
  for (let x = 0; x <= GOAL_X; x += 2) {
    p.push({ x, z: 0, heading: 0, speed: x >= GOAL_X ? 0 : cruise, t: 0 });
  }
  return p;
}

describe('pure-pursuit stop — BUG A: respectPathSpeed stalls on a stop-terminated path', () => {
  const path = stopPath(6);
  const cfg: PurePursuitConfig = { ...BASE_CFG, respectPathSpeed: true };

  it('CURRENT BEHAVIOR: the vehicle never leaves the start (target pinned to ~0)', () => {
    const { end } = runStop(cfg, path);
    // The terminal-speed clamp holds targetSpeed at ~0 from tick 0, so it sits
    // at the spawn and never approaches the goal 40 m away.
    expect(end.x).toBeLessThan(1);
    expect(Math.hypot(end.x - GOAL_X, end.z)).toBeGreaterThan(38);
  });

  it.fails('DESIRED (tripwire): should drive to and stop at the goal', () => {
    const { end } = runStop(cfg, path);
    expect(Math.hypot(end.x - GOAL_X, end.z)).toBeLessThan(1.0);
    expect(Math.abs(end.speed)).toBeLessThan(0.3);
  });

  it('control: with respectPathSpeed OFF the same plan drives off the start', () => {
    // Proves the stall is the respectPathSpeed clamp, not the plan/plant: the
    // identical path with the flag off makes real progress.
    const { end } = runStop({ ...BASE_CFG, respectPathSpeed: false }, path);
    expect(end.x).toBeGreaterThan(5);
  });
});

describe('pure-pursuit stop — BUG B: standalone runaway above ~5 m/s approach', () => {
  // No speed profile, no respectPathSpeed: rely on the controller's own vGoal
  // brake-distance term to stop. It works at low approach speed and fails
  // catastrophically above a sharp threshold.
  const cfg = (cruise: number): PurePursuitConfig => ({ ...BASE_CFG, cruiseSpeed: cruise });

  it('CURRENT BEHAVIOR: stops cleanly at low approach speed (≤4 m/s)', () => {
    for (const cruise of [2, 3, 4]) {
      const { overshoot, end } = runStop(cfg(cruise), stopPath(cruise));
      expect(overshoot, `cruise=${cruise} overshoot=${overshoot.toFixed(2)}`).toBeLessThan(0.6);
      expect(Math.abs(end.speed), `cruise=${cruise}`).toBeLessThan(0.1);
    }
  });

  it('CURRENT BEHAVIOR: runs away past the goal at high approach speed (≥6 m/s)', () => {
    for (const cruise of [6, 8]) {
      const { overshoot, end } = runStop(cfg(cruise), stopPath(cruise));
      // Overshoots by hundreds of metres and is still at cruise speed at the end.
      expect(overshoot, `cruise=${cruise} overshoot=${overshoot.toFixed(2)}`).toBeGreaterThan(50);
      expect(Math.abs(end.speed), `cruise=${cruise}`).toBeGreaterThan(1);
    }
  });

  it.fails('DESIRED (tripwire): an 8 m/s approach should still stop near the goal', () => {
    const { overshoot, end } = runStop(cfg(8), stopPath(8));
    expect(overshoot).toBeLessThan(2);
    expect(Math.abs(end.speed)).toBeLessThan(0.3);
  });
});
