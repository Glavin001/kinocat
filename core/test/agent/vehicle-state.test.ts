// Smoke tests for the VehicleState type extension: legacy producers
// continue to work; new optional fields default semantically to 0.

import { describe, expect, it } from 'vitest';
import type { VehicleState } from 'kinocat/agent';

describe('VehicleState — optional Markov-state fields', () => {
  it('legacy 4-D construction still type-checks and reads as undefined', () => {
    const legacy: VehicleState = { x: 1, z: 2, heading: 0.1, speed: 5, t: 0 };
    expect(legacy.yawRate).toBeUndefined();
    expect(legacy.lateralVelocity).toBeUndefined();
  });

  it('full 6-D construction preserves all fields', () => {
    const full: VehicleState = {
      x: 1, z: 2, heading: 0.1, speed: 5, t: 0,
      yawRate: 0.4, lateralVelocity: -0.7,
    };
    expect(full.yawRate).toBe(0.4);
    expect(full.lateralVelocity).toBe(-0.7);
  });

  it('consumers default ?? 0 reads sensibly', () => {
    const legacy: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const yr = legacy.yawRate ?? 0;
    const lv = legacy.lateralVelocity ?? 0;
    expect(yr).toBe(0);
    expect(lv).toBe(0);
  });
});
