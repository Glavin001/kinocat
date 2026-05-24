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
