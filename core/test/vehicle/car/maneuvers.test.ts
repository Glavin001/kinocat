import { describe, expect, it } from 'vitest';
import {
  ouControls,
  defaultManeuverBundle,
  seededRng,
  throttleRelease,
  steerReversal,
  slalom,
  trailBrake,
  panicTurn,
  type ManeuverLimits,
} from 'kinocat/vehicle/car';

const limits: ManeuverLimits = {
  maxSteerAngle: 0.6,
  maxDriveForce: 4000,
  maxBrakeForce: 2000,
};

describe('car maneuvers', () => {
  it('ouControls clips outputs to physical limits', () => {
    const drv = ouControls({
      params: { sigmaSteer: 100, sigmaDrive: 1e6, sigmaBrake: 1e6, tau: 0.3 },
      limits,
      rng: seededRng(7),
    });
    for (let i = 0; i < 200; i++) {
      const c = drv.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, i * 0.05, 0.05);
      expect(Math.abs(c.steer)).toBeLessThanOrEqual(limits.maxSteerAngle + 1e-9);
      expect(Math.abs(c.driveForce)).toBeLessThanOrEqual(limits.maxDriveForce + 1e-6);
      expect(c.brakeForce).toBeGreaterThanOrEqual(0);
      expect(c.brakeForce).toBeLessThanOrEqual(limits.maxBrakeForce + 1e-6);
    }
  });

  it('ouControls deterministic for same seed', () => {
    const make = () => ouControls({
      params: { sigmaSteer: 0.1, sigmaDrive: 500, sigmaBrake: 500, tau: 0.3 },
      limits,
      rng: seededRng(42),
    });
    const a = make();
    const b = make();
    for (let i = 0; i < 30; i++) {
      const sa = a.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, i * 0.05, 0.05);
      const sb = b.sample({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, i * 0.05, 0.05);
      expect(sa).toEqual(sb);
    }
  });

  it('throttleRelease emits two regimes around the transition', () => {
    const drv = throttleRelease({ throttle: 3000, steer: 0.1, transitionAt: 1.0 });
    const state = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    drv.sample(state, 0, 0.05); // prime t0=0
    const before = drv.sample(state, 0.5, 0.05);
    const after = drv.sample(state, 1.5, 0.05);
    expect(before.driveForce).toBe(3000);
    expect(after.driveForce).toBe(0);
    expect(after.steer).toBe(0.1);
  });

  it('steerReversal flips at the transition', () => {
    const drv = steerReversal({ steerLeft: -0.5, steerRight: 0.5, drive: 2000, transitionAt: 1.0 });
    const state = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    drv.sample(state, 0, 0.05); // prime t0=0
    const left = drv.sample(state, 0.5, 0.05);
    const right = drv.sample(state, 1.5, 0.05);
    expect(left.steer).toBe(-0.5);
    expect(right.steer).toBe(0.5);
  });

  it('slalom traces a sinusoid in steer', () => {
    const drv = slalom({ amplitude: 0.4, periodSec: 1.0, driveForce: 2000 });
    const state = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    drv.sample(state, 0, 0.05); // prime t0=0
    const s = drv.sample(state, 0.25, 0.05); // quarter-period later
    expect(s.steer).toBeCloseTo(0.4, 5);
  });

  it('trailBrake decays brake and ramps steer', () => {
    const drv = trailBrake({
      brakeForce: 1500, releaseTime: 1.0, steerRamp: 0.5, steerHold: 0.4, totalDuration: 2.0,
    });
    const state = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    drv.sample(state, 0, 0.05); // prime t0=0
    const early = drv.sample(state, 0.2, 0.05);
    const late = drv.sample(state, 1.5, 0.05);
    expect(early.brakeForce).toBeGreaterThan(0);
    expect(late.brakeForce).toBe(0);
    expect(late.steer).toBe(0.4); // ramp clipped to hold
  });

  it('panicTurn transitions into brake recovery', () => {
    const drv = panicTurn({
      limits, steer: limits.maxSteerAngle, turnDuration: 0.4, brakeRecovery: limits.maxBrakeForce,
    });
    const state = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    drv.sample(state, 0, 0.05); // prime t0=0
    const turn = drv.sample(state, 0.1, 0.05);
    const recover = drv.sample(state, 0.5, 0.05);
    expect(turn.brakeForce).toBe(0);
    expect(recover.brakeForce).toBeGreaterThan(0);
  });

  it('defaultManeuverBundle returns the requested count with mixed IDs', () => {
    const bundle = defaultManeuverBundle({ limits, count: 50, seed: 1 });
    expect(bundle.length).toBe(50);
    const ids = new Set(bundle.map((s) => s.id));
    expect(ids.size).toBeGreaterThan(4); // at least 5 distinct factories
    // OU should be the largest single class (60%).
    const ouCount = bundle.filter((s) => s.id === 'ou').length;
    expect(ouCount).toBeGreaterThanOrEqual(25);
  });

  it('defaultManeuverBundle is deterministic for same seed', () => {
    const a = defaultManeuverBundle({ limits, count: 20, seed: 9 });
    const b = defaultManeuverBundle({ limits, count: 20, seed: 9 });
    expect(a.map((s) => s.id)).toEqual(b.map((s) => s.id));
  });
});
