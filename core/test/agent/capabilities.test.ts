// The capability envelope must be a DERIVATION from the vehicle config,
// and the config default must itself match the Rapier-derived config —
// this is the anti-drift layer. If any of these fail after a chassis
// option change, a hand-copied mirror somewhere was missed.

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNABLE_CONFIG,
  deriveVehicleCapabilities,
  plannerVehicleCapabilities,
} from 'kinocat/agent';
import { deriveLearnableConfig } from '../../src/adapters/rapier/raycast-vehicle';

const caps = deriveVehicleCapabilities(DEFAULT_LEARNABLE_CONFIG);

describe('deriveVehicleCapabilities — reference chassis', () => {
  it('derives the exact axle-to-axle wheelbase and mass', () => {
    expect(caps.wheelbaseLength).toBeCloseTo(3.2, 12);
    expect(caps.chassisMass).toBeCloseTo(576, 12);
  });

  it('derives the kinematic minimum turn radius L/tan(steerMax)', () => {
    expect(caps.minTurnRadius).toBeCloseTo(3.2 / Math.tan(0.6), 6); // ≈ 4.6805
  });

  it('acceleration is traction-limited for the RWD reference chassis', () => {
    // Drive limit: 2 wheels × 4000 N / 576 kg ≈ 13.9 m/s².
    // Traction limit: μ 1.8 × 0.5 axle share × g ≈ 8.83 m/s² — binds.
    expect(caps.maxAccel).toBeCloseTo(1.8 * 0.5 * 9.81, 6);
  });

  it('deceleration is brake-hardware-limited below the grip ceiling', () => {
    // Brakes: 4 wheels × 2000 N / 576 kg ≈ 13.9 < μ·g ≈ 17.66.
    expect(caps.maxDecel).toBeCloseTo((4 * 2000) / 576, 6);
    expect(caps.maxLateralAccel).toBeCloseTo(1.8 * 9.81, 6);
  });
});

describe('config/adapter drift', () => {
  it('DEFAULT_LEARNABLE_CONFIG equals the config derived from Rapier defaults', () => {
    const derived = deriveLearnableConfig({ id: 'drift-probe', position: { x: 0, z: 0 }, heading: 0 });
    expect(derived).toEqual(DEFAULT_LEARNABLE_CONFIG);
  });
});

describe('plannerVehicleCapabilities — containment', () => {
  it('planner envelope is strictly inside the plant envelope', () => {
    const planner = plannerVehicleCapabilities(DEFAULT_LEARNABLE_CONFIG);
    expect(planner.minTurnRadius).toBeGreaterThan(caps.minTurnRadius);
    expect(planner.maxAccel).toBeLessThan(caps.maxAccel);
    expect(planner.maxDecel).toBeLessThan(caps.maxDecel);
    expect(planner.maxLateralAccel).toBeLessThan(caps.maxLateralAccel);
  });

  it('zero margin reproduces the plant envelope', () => {
    expect(plannerVehicleCapabilities(DEFAULT_LEARNABLE_CONFIG, 0)).toEqual(caps);
  });
});
