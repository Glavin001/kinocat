import { describe, it, expect } from 'vitest';
import { purePursuit } from '../../src/execute/pure-pursuit';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import { wrapAngle } from '../../src/internal/math';

const cfg: PurePursuitConfig = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: 4,
  maxAccel: 8,
  maxDecel: 8,
  cruiseSpeed: 6,
  goalTolerance: 0.5,
  minTurnRadius: 2,
};

/** Closed-loop unicycle sim (perfect speed tracking) for tracking tests. */
function simulate(start: CarKinematicState, path: PlanPath, steps: number, dt = 0.05) {
  let s = { ...start };
  let maxCross = 0;
  for (let k = 0; k < steps; k++) {
    const cmd = purePursuit(s, path, cfg);
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
    maxCross = Math.max(maxCross, Math.abs(s.z)); // straight-path cross-track
  }
  return { final: s, maxCross };
}

describe('purePursuit', () => {
  it('converges onto and tracks a straight path from a lateral offset', () => {
    const path: PlanPath = [];
    for (let x = 0; x <= 24; x += 2) {
      path.push({ x, z: 0, heading: 0, speed: 6, t: x / 6 });
    }
    const start: CarKinematicState = { x: 0, z: 2, heading: 0, speed: 0, t: 0 };
    const { final } = simulate(start, path, 600);
    expect(Math.hypot(final.x - 24, final.z - 0)).toBeLessThan(0.6);
    expect(Math.abs(final.z)).toBeLessThan(0.2); // converged onto the line
  });

  it('reduces speed on tight curvature (v ≈ sqrt(aLat/|κ|))', () => {
    const longCfg = { ...cfg, cruiseSpeed: 50, maxDecel: 50, minTurnRadius: undefined };
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 0.1, z: 50, heading: Math.PI / 2, speed: 6, t: 10 },
    ];
    const cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cmd = purePursuit(cur, path, longCfg);
    const kappa = Math.abs(cmd.steering);
    expect(kappa).toBeGreaterThan(0.1);
    const expected = Math.sqrt(longCfg.maxLateralAccel / kappa);
    expect(Math.abs(cmd.targetSpeed)).toBeCloseTo(expected, 6);
    expect(Math.abs(cmd.targetSpeed)).toBeLessThan(longCfg.cruiseSpeed);
  });

  it('tracks a reverse path (negative planned speed)', () => {
    const path: PlanPath = [];
    for (let x = 0; x >= -12; x -= 2) {
      path.push({ x, z: 0, heading: 0, speed: -4, t: -x / 4 });
    }
    const start: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cmd0 = purePursuit(start, path, cfg);
    expect(cmd0.targetSpeed).toBeLessThan(0);
    const { final } = simulate(start, path, 600);
    expect(Math.hypot(final.x - -12, final.z - 0)).toBeLessThan(0.8);
  });

  it('is a pure function (identical inputs → identical output)', () => {
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 10, z: 1, heading: 0, speed: 6, t: 1.7 },
    ];
    const cur: CarKinematicState = { x: 1, z: 0.2, heading: 0.1, speed: 3, t: 0.2 };
    expect(purePursuit(cur, path, cfg)).toEqual(purePursuit(cur, path, cfg));
  });

  it('caps speed via the braking envelope of upcoming plan speeds', () => {
    // Straight path with a slow zone ahead. Plan speeds constrain
    // through the braking envelope sqrt(v² + 2·maxDecel·d): a slow zone
    // binds exactly when the chassis is inside braking distance of it,
    // not from arbitrarily far away (the old raw window-min meant any
    // coast-to-stop sample in the window pinned the target to ~0 —
    // measured closed-loop: race cars crawled to 0 laps in 100 s with
    // the toggle on).
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 2, z: 0, heading: 0, speed: 6, t: 0.3 },
      { x: 4, z: 0, heading: 0, speed: 2, t: 0.9 }, // slow zone
      { x: 6, z: 0, heading: 0, speed: 2, t: 1.5 },
      { x: 8, z: 0, heading: 0, speed: 6, t: 2.0 },
    ];
    // Just before the slow zone: inside braking distance, so the cap
    // binds at the envelope value sqrt(2² + 2·maxDecel·d).
    const near: CarKinematicState = { x: 3.5, z: 0, heading: 0, speed: 6, t: 0 };
    const without = purePursuit(near, path, { ...cfg, respectPathSpeed: false });
    const withCap = purePursuit(near, path, { ...cfg, respectPathSpeed: true });
    expect(Math.abs(without.targetSpeed)).toBeGreaterThan(Math.abs(withCap.targetSpeed));
    const envelope = Math.sqrt(2 * 2 + 2 * cfg.maxDecel * 0.5); // d = 0.5 m to the slow sample
    expect(Math.abs(withCap.targetSpeed)).toBeCloseTo(envelope, 5);

    // A stopped chassis on the plan's first sample (the rest echo) must
    // still launch: the d≈0 echo is excluded from the constraint.
    const resting: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const launch = purePursuit(resting, path, { ...cfg, respectPathSpeed: true });
    expect(Math.abs(launch.targetSpeed)).toBeGreaterThan(1);
  });

  it('brakes at the goal', () => {
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 4, t: 0 },
      { x: 5, z: 0, heading: 0, speed: 0, t: 1.25 },
    ];
    const atGoal: CarKinematicState = { x: 4.9, z: 0, heading: 0, speed: 1, t: 1.2 };
    const cmd = purePursuit(atGoal, path, cfg);
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBe(1);
    expect(cmd.targetSpeed).toBe(0);
  });

  describe('terminal heading term (headingGain)', () => {
    // A path whose tangent points along +x (heading 0) but the chassis sits on
    // it pointed 0.3 rad off. Pure-pursuit's lookahead point is straight ahead
    // (no cross-track), so WITHOUT a heading term it commands ~0 curvature and
    // never corrects the orientation; WITH the term it commands curvature toward
    // the path heading.
    const straight: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 1, t: 0 },
      { x: 1, z: 0, heading: 0, speed: 1, t: 1 },
      { x: 2, z: 0, heading: 0, speed: 1, t: 2 },
    ];
    const offHeading: CarKinematicState = { x: 0, z: 0, heading: 0.3, speed: 1, t: 0 };

    it('adds curvature toward the path tangent when the chassis is mis-aligned', () => {
      // Unclamped config so the exact delta is observable (the curvature limit
      // is exercised by its own test below).
      const noClamp: PurePursuitConfig = { ...cfg, minTurnRadius: undefined };
      const base = purePursuit(offHeading, straight, noClamp).steering;
      const withTerm = purePursuit(offHeading, straight, { ...noClamp, headingGain: 1.0 }).steering;
      // Chassis heading (+0.3) exceeds the path tangent (0) ⇒ correction is a
      // negative-curvature (turn back toward the path heading) delta of exactly
      // headingGain · (tangent − heading) = 1.0 · (0 − 0.3).
      expect(withTerm).toBeLessThan(base);
      expect(withTerm - base).toBeCloseTo(-0.3, 6);
    });

    it('is gated off beyond headingRadius', () => {
      // distToGoal from (0,0) to (2,0) is 2; a 1.5 m radius gates the term off.
      const gated = purePursuit(offHeading, straight, { ...cfg, headingGain: 1.0, headingRadius: 1.5 });
      const base = purePursuit(offHeading, straight, cfg);
      expect(gated.steering).toBeCloseTo(base.steering, 9);
    });

    it('respects the minimum turn radius after adding the term', () => {
      const kMax = 1 / cfg.minTurnRadius!;
      const cmd = purePursuit(
        { x: 0, z: 0, heading: 1.2, speed: 1, t: 0 },
        straight,
        { ...cfg, headingGain: 5.0 },
      );
      expect(Math.abs(cmd.steering)).toBeLessThanOrEqual(kMax + 1e-9);
    });

    it('applies in reverse gear with the same formula (travel-frame law)', () => {
      // The pursuit runs in the travel frame (body frame flipped by pi when
      // reversing), where d(heading)/dt = |v| * kappa holds for both gears and
      // the pose-heading error equals the travel-frame error. So the heading
      // term is gain * wrapAngle(tangent - heading) in reverse too. (It used
      // to be gated forward-only on the assumption that reverse maneuvers
      // arrive pre-aligned — false once the direction-change penalty fix made
      // plans legitimately terminate on a reverse leg.)
      const revPath: PlanPath = [
        { x: 0, z: 0, heading: 0, speed: -1, t: 0 },
        { x: -1, z: 0, heading: 0, speed: -1, t: 1 },
        { x: -2, z: 0, heading: 0, speed: -1, t: 2 },
      ];
      const noClamp: PurePursuitConfig = { ...cfg, minTurnRadius: undefined };
      const base = purePursuit(offHeading, revPath, noClamp).steering;
      const withTerm = purePursuit(offHeading, revPath, { ...noClamp, headingGain: 1.0 }).steering;
      // offHeading has heading +0.3 vs path tangent 0 -> term adds 1.0 * (0 - 0.3).
      expect(withTerm - base).toBeCloseTo(-0.3, 6);
    });
  });
});
