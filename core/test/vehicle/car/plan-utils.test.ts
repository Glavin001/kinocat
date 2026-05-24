import { describe, it, expect } from 'vitest';
import { samplePlanAt, trimPlan } from '../../../src/vehicle/car/plan-utils';
import type { CarKinematicState } from '../../../src/vehicle/car/types';

function path(): CarKinematicState[] {
  return [
    { x: 0, z: 0, heading: 0, speed: 5, t: 0 },
    { x: 5, z: 0, heading: 0, speed: 5, t: 1 },
    { x: 10, z: 0, heading: Math.PI / 2, speed: 5, t: 2 },
  ];
}

describe('samplePlanAt', () => {
  it('returns first sample at t<=start', () => {
    const s = samplePlanAt(path(), -1)!;
    expect(s.x).toBe(0);
  });
  it('returns last sample at t>=end', () => {
    const s = samplePlanAt(path(), 99)!;
    expect(s.x).toBe(10);
  });
  it('linearly interpolates position within a bracket', () => {
    const s = samplePlanAt(path(), 0.5)!;
    expect(s.x).toBeCloseTo(2.5);
    expect(s.t).toBeCloseTo(0.5);
  });
  it('interpolates heading across the shorter arc', () => {
    // From 0 to π/2 across t=[1,2], at t=1.5 → π/4.
    const s = samplePlanAt(path(), 1.5)!;
    expect(s.heading).toBeCloseTo(Math.PI / 4, 4);
  });
  it('returns null on empty plan', () => {
    expect(samplePlanAt([], 0)).toBeNull();
  });
});

describe('trimPlan', () => {
  it('drops past samples but keeps at least one', () => {
    const trimmed = trimPlan(path(), 99);
    expect(trimmed.length).toBe(1);
    expect(trimmed[0]!.x).toBe(10);
  });
  it('keeps samples whose t > elapsed', () => {
    const trimmed = trimPlan(path(), 0.5);
    expect(trimmed.length).toBe(3);
  });
});
