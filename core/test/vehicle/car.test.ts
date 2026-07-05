// Unit tests for the car-domain helpers. Exercises trimPlan, keyboard
// mapping, encoders (especially the steer-sign-flip rule), follow-plan,
// and the playback pattern driver.

import { describe, expect, it } from 'vitest';

import {
  KeyboardCarDriver,
  PlaybackPatternCarDriver,
  PlanFollowerCarDriver,
  encodeForKinematic,
  encodeForParametricV2,
  encodeWheeledRaw,
  followPlan,
  keyboardAckermann,
  keysFromSet,
  trimPlan,
  wheeledFromNormalized,
  ZERO_WHEELED,
  type CarKinematicState,
} from '../../src/vehicle/car';

describe('trimPlan', () => {
  it('drops plan samples already in the past', () => {
    const plan = [0, 1, 2, 3, 4].map((t) => ({ x: t, z: 0, heading: 0, speed: 0, t }));
    expect(trimPlan(plan, 2.5).map((s) => s.t)).toEqual([2, 3, 4]);
  });
  it('keeps at least one sample (the goal)', () => {
    const plan = [{ x: 0, z: 0, heading: 0, speed: 0, t: 0 }];
    expect(trimPlan(plan, 100).length).toBe(1);
  });
  it('returns empty for empty input', () => {
    expect(trimPlan([], 0)).toEqual([]);
  });
});

describe('keyboardAckermann', () => {
  it('left key steers negative', () => {
    const cmd = keyboardAckermann({ left: true, right: false, forward: false, backward: false, brake: false });
    expect(cmd.steer).toBeLessThan(0);
  });
  it('right key steers positive', () => {
    const cmd = keyboardAckermann({ left: false, right: true, forward: false, backward: false, brake: false });
    expect(cmd.steer).toBeGreaterThan(0);
  });
  it('forward / backward set throttle sign', () => {
    expect(keyboardAckermann({ left: false, right: false, forward: true, backward: false, brake: false }).throttle).toBe(1);
    expect(keyboardAckermann({ left: false, right: false, forward: false, backward: true, brake: false }).throttle).toBe(-1);
  });
  it('brake fires only on space', () => {
    expect(keyboardAckermann({ left: false, right: false, forward: false, backward: false, brake: true }).brake).toBe(1);
  });
});

describe('keysFromSet', () => {
  it('maps WASD and arrow keys', () => {
    const ks = keysFromSet(new Set(['w', 'arrowleft', ' ']));
    expect(ks.forward).toBe(true);
    expect(ks.left).toBe(true);
    expect(ks.brake).toBe(true);
    expect(ks.right).toBe(false);
  });
});

describe('encoders', () => {
  it('encodeForParametricV2 flips the steer sign', () => {
    const v = encodeForParametricV2({ steer: 0.5, driveForce: 1000, brakeForce: 0 });
    expect(v[0]).toBe(-0.5);
    expect(v[1]).toBe(1000);
    expect(v[2]).toBe(0);
  });
  it('encodeWheeledRaw does NOT flip', () => {
    const v = encodeWheeledRaw({ steer: 0.5, driveForce: 1000, brakeForce: 200 });
    expect(v).toEqual([0.5, 1000, 200]);
  });
  it('encodeForKinematic maps to (curvature, targetSpeed)', () => {
    const v = encodeForKinematic(
      { steer: 0.1, driveForce: 0, brakeForce: 0 },
      { wheelBase: 2, maxSpeed: 12, throttle: 0.5, brake: 0 },
    );
    // curvature = tan(-0.1) / 2 ≈ -0.0501
    expect(v[0]!).toBeCloseTo(Math.tan(-0.1) / 2, 6);
    // targetSpeed = (0.5 - 0) * 12 = 6
    expect(v[1]).toBeCloseTo(6, 6);
  });
});

describe('followPlan', () => {
  it('returns a stop command on empty trim', () => {
    const state: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cmd = followPlan(state, [], {
      config: {
        lookaheadMin: 1,
        lookaheadGain: 0.5,
        lookaheadMax: 10,
        maxLateralAccel: 3,
        maxAccel: 5,
        maxDecel: 5,
        cruiseSpeed: 10,
        goalTolerance: 1,
      },
      elapsed: 0,
    });
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBe(1);
  });
});

describe('KeyboardCarDriver', () => {
  it('maps active keys to (steer rad, driveForce N, brakeForce N) with Rapier sign', () => {
    const active = new Set<string>(['w', 'd']); // forward + right
    const d = new KeyboardCarDriver({
      keys: () => active,
      engineForceN: 4000,
      brakeForceN: 4000,
      maxSteerAngle: 0.5,
      steerGain: 1.0,
    });
    const c = d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1 / 60);
    // Rapier-frame steer = -planning steer; planning right = +0.55 -> Rapier -0.55 * 0.5 maxSteer = ...
    // With steerGain=1, planning steer = +1; clamped to ±1 (matches max). Rapier-side = -1 * 0.5 = -0.5
    expect(c.steer).toBeCloseTo(-0.5, 6);
    expect(c.driveForce).toBe(4000);
    expect(c.brakeForce).toBe(0);
  });
});

describe('PlaybackPatternCarDriver', () => {
  it('cycles through hold + slalom + ramp segments', () => {
    const d = new PlaybackPatternCarDriver([
      { kind: 'hold', duration: 1, controls: { steer: 0, driveForce: 1000, brakeForce: 0 } },
      { kind: 'slalom', duration: 1, periodSec: 1, steerAmp: 0.3, driveForce: 500 },
    ]);
    const c0 = d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1 / 60);
    expect(c0).toEqual({ steer: 0, driveForce: 1000, brakeForce: 0 });
    const c1 = d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 1.25, 1 / 60);
    // 0.25 s into slalom with period 1 -> phase = π/2 -> sin = 1
    expect(c1.steer).toBeCloseTo(0.3, 6);
    expect(c1.driveForce).toBe(500);
    expect(d.cycleSec()).toBe(2);
  });

  it('detects cycle boundaries', () => {
    const d = new PlaybackPatternCarDriver([
      { kind: 'hold', duration: 1, controls: { steer: 0, driveForce: 0, brakeForce: 0 } },
    ]);
    d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1 / 60);
    expect(d.isCycleBoundary(0, 0.01)).toBe(true);
    expect(d.isCycleBoundary(0.999, 0.05)).toBe(true);
    expect(d.isCycleBoundary(0.5, 0.05)).toBe(false);
  });
});

describe('PlanFollowerCarDriver', () => {
  it('returns zero when no plan is set', () => {
    const d = new PlanFollowerCarDriver({
      config: {
        lookaheadMin: 1,
        lookaheadGain: 0.5,
        lookaheadMax: 10,
        maxLateralAccel: 3,
        maxAccel: 5,
        maxDecel: 5,
        cruiseSpeed: 10,
        goalTolerance: 1,
      },
      wheelBase: 2,
      engineForceN: 4000,
      brakeForceN: 4000,
      maxSteerAngle: 0.5,
    });
    const c = d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1 / 60);
    expect(c).toEqual({ steer: 0, driveForce: 0, brakeForce: 0 });
  });

  const mkDriver = () =>
    new PlanFollowerCarDriver({
      config: {
        lookaheadMin: 1,
        lookaheadGain: 0.5,
        lookaheadMax: 10,
        maxLateralAccel: 6,
        maxAccel: 5,
        maxDecel: 5,
        cruiseSpeed: 5,
        goalTolerance: 0.5,
      },
      wheelBase: 3.2,
      engineForceN: 4000,
      brakeForceN: 4000,
      maxSteerAngle: 0.6,
    });

  it('steering sign: forward left-curving plan -> NEGATIVE Rapier steer', () => {
    // The car sits left of a straight +x path, so the tracker commands a
    // right-turning (negative planning-frame) curvature... use the clearer
    // case: car BELOW the path (z = -1) -> lookahead is up-left -> positive
    // lateral offset in body frame -> positive planning curvature kappa.
    // Net applied Rapier steer must be -atan(kappa * L) (frame flip), i.e.
    // NEGATIVE. A double negation here once inverted this and every forward
    // follower veered away from its plan on the first tick.
    const d = mkDriver();
    const path = Array.from({ length: 11 }, (_, i) => ({
      x: i * 2, z: 0, heading: 0, speed: 3, t: i * 2 / 3,
    }));
    d.setPlan(path, 0);
    const cmd = d.sample({ x: 0, z: -1, heading: 0, speed: 3, t: 0 }, 0, 1 / 60);
    expect(cmd.driveForce).toBeGreaterThan(0); // forward gear
    expect(cmd.steer).toBeLessThan(0); // frame-flipped left turn
  });

  it('steering sign: reverse plan flips both drive force and steer', () => {
    const d = mkDriver();
    // Straight reverse along -x; car sits at z = -1 (same lateral offset).
    const path = Array.from({ length: 11 }, (_, i) => ({
      x: -i * 2, z: 0, heading: 0, speed: -3, t: i * 2 / 3,
    }));
    d.setPlan(path, 0);
    const cmd = d.sample({ x: 0, z: -1, heading: 0, speed: -3, t: 0 }, 0, 1 / 60);
    expect(cmd.driveForce).toBeLessThan(0); // reverse gear now possible at all
    // Derivation (matches the demo runner's proven -gear*atan(kappa*L)):
    // reversed body frame he = pi; lookahead at (-Ld, 0) from (0,-1) gives
    // lateral offset yV = -1 -> kappa < 0 (travel frame). gear = -1, so
    // applied steer = -gear*atan(kappa*L) = atan(kappa*L) < 0.
    expect(cmd.steer).toBeLessThan(0);
  });
});

describe('wheeledFromNormalized', () => {
  const tuning = { engineForceN: 4000, brakeForceN: 2000 };
  it('pre-negates steer (planner-frame -> Rapier-frame)', () => {
    const out = wheeledFromNormalized({ steer: 0.3, throttle: 0, brake: 0 }, tuning);
    expect(out.steer).toBeCloseTo(-0.3, 12);
  });
  it('scales throttle by engineForceN, supports reverse', () => {
    expect(
      wheeledFromNormalized({ steer: 0, throttle: 0.5, brake: 0 }, tuning).driveForce,
    ).toBeCloseTo(2000, 12);
    expect(
      wheeledFromNormalized({ steer: 0, throttle: -1, brake: 0 }, tuning).driveForce,
    ).toBeCloseTo(-4000, 12);
  });
  it('scales brake by brakeForceN', () => {
    expect(
      wheeledFromNormalized({ steer: 0, throttle: 0, brake: 0.5 }, tuning).brakeForce,
    ).toBeCloseTo(1000, 12);
  });
  it('ZERO_WHEELED is the no-op input', () => {
    expect(ZERO_WHEELED).toEqual({ steer: 0, driveForce: 0, brakeForce: 0 });
  });
});

describe('PlanFollowerCarDriver (plan set)', () => {
  it('produces a finite command once a plan is set', () => {
    const d = new PlanFollowerCarDriver({
      config: {
        lookaheadMin: 1,
        lookaheadGain: 0.5,
        lookaheadMax: 10,
        maxLateralAccel: 3,
        maxAccel: 5,
        maxDecel: 5,
        cruiseSpeed: 10,
        goalTolerance: 0.5,
      },
      wheelBase: 2,
      engineForceN: 4000,
      brakeForceN: 4000,
      maxSteerAngle: 0.5,
    });
    const plan: CarKinematicState[] = [
      { x: 0, z: 0, heading: 0, speed: 5, t: 0 },
      { x: 5, z: 0, heading: 0, speed: 5, t: 1 },
      { x: 10, z: 0, heading: 0, speed: 5, t: 2 },
    ];
    d.setPlan(plan, 0);
    const c = d.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1 / 60);
    expect(Number.isFinite(c.steer)).toBe(true);
    expect(Number.isFinite(c.driveForce)).toBe(true);
  });
});
