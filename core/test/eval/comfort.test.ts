import { describe, it, expect } from 'vitest';
import { comfortFlags, DEFAULT_COMFORT_BOUNDS } from '../../src/eval/comfort';
import type { CarKinematicState } from '../../src/agent/types';

/** Build a constant-speed straight-line executed trajectory. */
function cruise(speed: number, n: number, dt: number): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  let x = 0;
  for (let i = 0; i < n; i++) {
    out.push({ x, z: 0, heading: 0, speed, t: i * dt });
    x += speed * dt;
  }
  return out;
}

describe('comfortFlags', () => {
  it('marks smooth constant-speed cruising as comfortable', () => {
    const r = comfortFlags(cruise(6, 50, 0.05), 0.05);
    expect(r.comfortable).toBe(true);
    expect(r.violations).toEqual([]);
  });

  it('flags a hard lateral-accel corner as uncomfortable', () => {
    // Tight, fast circle ⇒ large v·yawRate.
    const dt = 0.05;
    const v = 12;
    const yawRate = 1.2; // rad/s ⇒ lat accel = 14.4 m/s² ≫ 4.89
    const out: CarKinematicState[] = [];
    let heading = 0;
    let x = 0;
    let z = 0;
    for (let i = 0; i < 40; i++) {
      out.push({ x, z, heading, speed: v, t: i * dt });
      heading += yawRate * dt;
      x += v * Math.cos(heading) * dt;
      z += v * Math.sin(heading) * dt;
    }
    const r = comfortFlags(out, dt);
    expect(r.comfortable).toBe(false);
    expect(r.violations).toContain('latAccel');
  });

  it('flags yaw-accel and jerk on a rapidly-varying maneuver', () => {
    // Oscillate yaw rate and speed every tick ⇒ large yaw accel + jerk.
    const dt = 0.05;
    const out: CarKinematicState[] = [];
    let heading = 0;
    let x = 0;
    let z = 0;
    for (let i = 0; i < 40; i++) {
      const yawRate = i % 2 === 0 ? 0.8 : -0.8;
      const speed = i % 2 === 0 ? 8 : 5;
      out.push({ x, z, heading, speed, t: i * dt });
      heading += yawRate * dt;
      x += speed * Math.cos(heading) * dt;
      z += speed * Math.sin(heading) * dt;
    }
    const r = comfortFlags(out, dt);
    expect(r.comfortable).toBe(false);
    expect(r.violations).toContain('yawAccel');
    expect(r.violations.some((v) => v === 'longJerk' || v === 'jerkVec')).toBe(true);
  });

  it('uses the populated state.yawRate field when present', () => {
    const dt = 0.05;
    const out: CarKinematicState[] = Array.from({ length: 20 }, (_, i) => ({
      x: i * 0.3,
      z: 0,
      heading: 0,
      speed: 6,
      t: i * dt,
      yawRate: 2.0, // explicit, exceeds the 0.95 rad/s bound
    }));
    const r = comfortFlags(out, dt);
    expect(r.peak.yawRate).toBeCloseTo(2.0, 5);
    expect(r.violations).toContain('yawRate');
  });

  it('respects custom (relaxed arcade) bounds', () => {
    const arcade = { ...DEFAULT_COMFORT_BOUNDS, latAccelAbs: 100 };
    const dt = 0.05;
    const v = 12;
    const yawRate = 1.2;
    const out: CarKinematicState[] = [];
    let heading = 0;
    let x = 0;
    let z = 0;
    for (let i = 0; i < 40; i++) {
      out.push({ x, z, heading, speed: v, t: i * dt });
      heading += yawRate * dt;
      x += v * Math.cos(heading) * dt;
      z += v * Math.sin(heading) * dt;
    }
    const r = comfortFlags(out, dt, arcade);
    expect(r.violations).not.toContain('latAccel');
  });
});
