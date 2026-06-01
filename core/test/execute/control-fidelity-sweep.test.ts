// Control fidelity + stability, as sweeps. A single "track this path" test
// either passes or fails; sweeping approach speed / path curvature / controller
// gain reveals WHERE behaviour degrades — overshoot at the stopping point,
// corner-cutting as the radius tightens, and limit-cycle hunting around a
// setpoint that looks fine for two seconds but never settles. All closed-loop
// sims here are deterministic and Rapier-free (perfect / first-order tracking).

import { describe, it, expect } from 'vitest';
import { purePursuit } from '../../src/execute/pure-pursuit';
import { smoothSpeedProfile } from '../../src/execute/speed-profile';
import { kinematicForwardSim } from '../../src/agent/vehicle';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import { wrapAngle } from '../../src/internal/math';
import { SWEEP_AGENT, linspace } from '../fixtures/vehicle-sweep';

const BASE_CFG: PurePursuitConfig = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: 10,
  maxAccel: 8,
  maxDecel: 8,
  cruiseSpeed: 6,
  goalTolerance: 0.5,
  minTurnRadius: 2,
};

/** Closed-loop tracker. Perfect speed tracking by default (matches the
 *  existing pure-pursuit test); records steering + states so stability metrics
 *  can be derived. Stops at goal. */
function track(
  start: CarKinematicState,
  path: PlanPath,
  cfg: PurePursuitConfig,
  steps = 800,
  dt = 0.05,
) {
  let s: CarKinematicState = { ...start };
  const states: CarKinematicState[] = [s];
  const steers: number[] = [];
  for (let k = 0; k < steps; k++) {
    const cmd = purePursuit(s, path, cfg);
    steers.push(cmd.steering);
    if (cmd.atGoal) break;
    const speed = cmd.targetSpeed;
    const heading = wrapAngle(s.heading + speed * cmd.steering * dt);
    s = {
      x: s.x + speed * Math.cos(s.heading) * dt,
      z: s.z + speed * Math.sin(s.heading) * dt,
      heading,
      speed,
      t: s.t + dt,
    };
    states.push(s);
  }
  return { final: s, states, steers };
}

describe('control fidelity — stop-at-point, approach-speed sweep', () => {
  // Build a straight braking profile that ramps to a full stop at the goal, then
  // track it with respectPathSpeed. The friction-circle smoother is what makes
  // the stop respect the deceleration limit instead of teleporting to zero.
  const GOAL_X = 40;
  const path: PlanPath = [];
  for (let x = 0; x <= GOAL_X; x += 2) {
    path.push({ x, z: 0, heading: 0, speed: 6, t: x / 6 });
  }

  // Final sample demands v=0 (curvatureOverride → tiny vCap there); minSpeed 0
  // so the backward pass may ramp all the way to a stop.
  const stopCurv = path.map((_, i) => (i === path.length - 1 ? 1e6 : 0));

  /** Approach at constant `cruise` (maxSpeed cap) then brake to a stop at the
   *  goal under `aDec`. */
  function brakingProfile(cruise: number, aDec: number) {
    return smoothSpeedProfile(path, {
      aLatMax: 10,
      aLonMaxAccel: 6,
      aLonMaxDecel: aDec,
      maxSpeed: cruise,
      minSpeed: 0,
      curvatureOverride: stopCurv,
    });
  }

  it('smoothed profile brakes to a stop without exceeding the decel limit', () => {
    const aDec = 4;
    for (const cruise of linspace(3, 12, 6)) {
      const prof = brakingProfile(cruise, aDec);
      // Terminal stop (the tiny floor from the override is ~0.003 m/s).
      expect(Math.abs(prof[prof.length - 1]!.speed), `cruise=${cruise}`).toBeLessThan(0.05);
      // No segment brakes harder than the limit (no "magic stopping power").
      for (let i = 1; i < prof.length; i++) {
        const v0 = Math.abs(prof[i - 1]!.speed);
        const v1 = Math.abs(prof[i]!.speed);
        const ds = Math.hypot(prof[i]!.x - prof[i - 1]!.x, prof[i]!.z - prof[i - 1]!.z);
        if (v1 < v0 && ds > 1e-9) {
          const implied = (v0 * v0 - v1 * v1) / (2 * ds);
          expect(implied, `cruise=${cruise} i=${i}`).toBeLessThanOrEqual(aDec + 1e-3);
        }
      }
    }
  });

  it('faster approach ⇒ braking begins no later (earlier or equal)', () => {
    const aDec = 4;
    const brakeStart = (cruise: number) => {
      const prof = brakingProfile(cruise, aDec);
      // First index whose speed has dropped meaningfully below the cruise peak.
      const i = prof.findIndex((p) => Math.abs(p.speed) < cruise - 0.25);
      return i < 0 ? path.length : i;
    };
    const cruises = linspace(4, 12, 5);
    const idxs = cruises.map(brakeStart);
    for (let i = 1; i < idxs.length; i++) {
      expect(idxs[i]!, `cruises=${cruises.join(',')} idxs=${idxs.join(',')}`).toBeLessThanOrEqual(idxs[i - 1]!);
    }
  });

  it('closed-loop tracking stops near the goal at every approach speed', () => {
    // pure-pursuit brakes once inside goalTolerance; sweeping cruise speed shows
    // the stop stays accurate (no overshoot growing without bound).
    for (const cruise of linspace(3, 9, 5)) {
      const cfg: PurePursuitConfig = { ...BASE_CFG, cruiseSpeed: cruise };
      const { final } = track({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, path, cfg);
      const posErr = Math.hypot(final.x - GOAL_X, final.z);
      expect(posErr, `cruise=${cruise} final.x=${final.x.toFixed(2)}`).toBeLessThan(BASE_CFG.goalTolerance + 0.6);
      // Overshoot past the goal stays bounded — does not blow up with speed.
      expect(final.x - GOAL_X, `cruise=${cruise}`).toBeLessThan(1.0);
    }
  });
});

describe('control fidelity — curved-path lateral-error sweep', () => {
  // Half-circle reference arcs of decreasing radius. Cross-track error is the
  // car's deviation from the circle. We measure the STEADY-STATE peak (after the
  // initial acquire transient) so the metric reflects sustained tracking, not
  // the one-off lurch onto the path. Tighter radius ⇒ harder to track.
  function arc(radius: number, n = 90): PlanPath {
    const out: PlanPath = [];
    for (let i = 0; i < n; i++) {
      const phi = -Math.PI / 2 + Math.PI * (i / (n - 1)); // -90°→+90° (half turn)
      out.push({
        x: radius * Math.cos(phi),
        z: radius + radius * Math.sin(phi),
        heading: wrapAngle(phi + Math.PI / 2),
        speed: 3,
        t: 0,
      });
    }
    return out;
  }

  const radii = [8, 6, 4, 3];
  // Tight lookahead so pursuit hugs the arc; minTurnRadius well below the radii.
  const cfg: PurePursuitConfig = {
    ...BASE_CFG,
    cruiseSpeed: 3,
    lookaheadMin: 1,
    lookaheadGain: 0.2,
    lookaheadMax: 4,
    minTurnRadius: 1.5,
  };

  const peakErrors = radii.map((R) => {
    const path = arc(R);
    const { states } = track({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, path, cfg, 2500, 0.04);
    let peak = 0;
    const skip = Math.floor(states.length * 0.3); // drop the acquire transient
    for (let i = skip; i < states.length; i++) {
      const s = states[i]!;
      const e = Math.abs(Math.hypot(s.x - 0, s.z - R) - R); // |dist to centre − R|
      if (e > peak) peak = e;
    }
    return peak;
  });

  it('tracks every arc (radius ≥ minTurnRadius) within a bounded steady-state error', () => {
    for (let i = 0; i < radii.length; i++) {
      expect(peakErrors[i]!, `R=${radii[i]} peak=${peakErrors[i]!.toFixed(3)}`).toBeLessThan(1.0);
    }
  });

  it('the tightest arc is the hardest to track (corner-cutting grows as R shrinks)', () => {
    const ctx = `\nradii=${radii.join(',')}\npeak =${peakErrors.map((e) => e.toFixed(3)).join(',')}`;
    const tightest = peakErrors[peakErrors.length - 1]!;
    const widest = peakErrors[0]!;
    expect(tightest, ctx).toBeGreaterThanOrEqual(widest - 1e-6);
  });
});

describe('control stability — oscillation / hunting around a straight setpoint', () => {
  const straight: PlanPath = [];
  for (let x = 0; x <= 60; x += 2) straight.push({ x, z: 0, heading: 0, speed: 5, t: x / 5 });

  it('converges from a lateral offset and does not hunt (low tail steering activity)', () => {
    for (const gain of linspace(0.1, 0.6, 6)) {
      const cfg: PurePursuitConfig = { ...BASE_CFG, cruiseSpeed: 5, lookaheadGain: gain };
      const { states, steers } = track({ x: 0, z: 1.0, heading: 0, speed: 0, t: 0 }, straight, cfg, 1500, 0.04);
      // Converged onto the line.
      const finalZ = Math.abs(states[states.length - 1]!.z);
      expect(finalZ, `gain=${gain} finalZ=${finalZ.toFixed(3)}`).toBeLessThan(0.2);
      // Tail steering must be quiet — a limit cycle would keep reversing sign.
      const tail = steers.slice(Math.floor(steers.length * 0.6));
      let reversals = 0;
      let prevSign = 0;
      for (const st of tail) {
        const sign = st > 0.02 ? 1 : st < -0.02 ? -1 : 0;
        if (sign !== 0) {
          if (prevSign !== 0 && sign !== prevSign) reversals++;
          prevSign = sign;
        }
      }
      expect(reversals, `gain=${gain} reversals=${reversals}`).toBeLessThanOrEqual(4);
    }
  });
});

describe('control fidelity — minimum-radius turn respects the kinematic limit', () => {
  it('commanded over-curvature is clamped to 1/minTurnRadius (never violated)', () => {
    const sim = kinematicForwardSim(SWEEP_AGENT); // minTurnRadius 3 ⇒ kMax = 1/3
    const kMax = 1 / SWEEP_AGENT.minTurnRadius;
    const dt = 0.1;
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 6, t: 0 };
    for (let i = 0; i < 80; i++) {
      const next = sim(s, [100, 6], dt); // absurd curvature command + full speed
      const dHeading = Math.abs(wrapAngle(next.heading - s.heading));
      const ds = Math.hypot(next.x - s.x, next.z - s.z);
      const effK = ds > 1e-9 ? dHeading / ds : 0; // achieved path curvature
      expect(effK).toBeLessThanOrEqual(kMax + 1e-6);
      expect(Number.isFinite(next.x) && Number.isFinite(next.heading)).toBe(true);
      s = next;
    }
  });
});
