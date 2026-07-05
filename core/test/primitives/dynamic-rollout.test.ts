// WS-2 — dynamic rollouts. The planner's baked motion library is
// characterized from zero-slip canonical states; a car mid-corner (yaw rate +
// sideslip) would expand as if it were rolling straight with no slip.
// `characterizeVehicleFromState` rolls the car's OWN model live from the true
// dynamic state, so the committed first primitive is model-consistent.

import { describe, it, expect } from 'vitest';
import {
  characterizeVehicle,
  characterizeVehicleFromState,
} from '../../src/primitives/characterize';
import {
  parametricForwardV2,
  DEFAULT_LEARNED_PARAMS_V2,
} from '../../src/agent/vehicle-model';
import { DEFAULT_LEARNABLE_CONFIG } from '../../src/agent/vehicle-config';
import type { CarKinematicState } from '../../src/agent/types';

const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
const drive = DEFAULT_LEARNABLE_CONFIG.maxDriveForce;
const brake = DEFAULT_LEARNABLE_CONFIG.maxBrakeForce;
const steer = DEFAULT_LEARNABLE_CONFIG.maxSteerAngle;
const controlSets = [
  [0, drive, 0],
  [steer * 0.5, 0, brake * 0.5],
  [-steer * 0.5, 0, brake * 0.5],
  [steer, 0, brake],
];
const duration = 0.55;
const substeps = 6;

describe('WS-2 dynamic rollouts (characterizeVehicleFromState)', () => {
  const midCorner: CarKinematicState = {
    x: 0, z: 0, heading: 0, speed: 18, yawRate: 0.8, lateralVelocity: 1.2, t: 0,
  };

  it('A2.1: root primitive endpoints equal a direct live model rollout', () => {
    const prims = characterizeVehicleFromState(sim, midCorner, controlSets, duration, substeps);
    const dt = duration / substeps;
    for (let ci = 0; ci < controlSets.length; ci++) {
      // Direct rollout in the local frame from the true dynamic state.
      let s: CarKinematicState = {
        x: 0, z: 0, heading: 0,
        speed: midCorner.speed, yawRate: midCorner.yawRate, lateralVelocity: midCorner.lateralVelocity, t: 0,
      };
      for (let k = 0; k < substeps; k++) s = sim(s, controlSets[ci]!, dt);
      const p = prims[ci]!;
      expect(Math.abs(p.end.dx - s.x)).toBeLessThan(1e-9);
      expect(Math.abs(p.end.dz - s.z)).toBeLessThan(1e-9);
      expect(Math.abs(p.end.speed - s.speed)).toBeLessThan(1e-9);
    }
  });

  it('A2.2: differs measurably from the baked zero-slip library', () => {
    // Baked library at the nearest start speed (no slip).
    const baked = characterizeVehicle({
      forwardSim: sim, controlSets, duration, substeps, startSpeeds: [18],
    });
    const dyn = characterizeVehicleFromState(sim, midCorner, controlSets, duration, substeps);
    // At least one control's endpoint should move by > 0.3 m due to the
    // carried yaw rate + sideslip that the baked (zero-slip) library ignores.
    let maxDelta = 0;
    for (let ci = 0; ci < controlSets.length; ci++) {
      const b = baked.primitives[ci]!;
      const d = dyn[ci]!;
      maxDelta = Math.max(maxDelta, Math.hypot(b.end.dx - d.end.dx, b.end.dz - d.end.dz));
    }
    expect(maxDelta).toBeGreaterThan(0.3);
  });

  it('produces primitives with the same shape as the baked library', () => {
    const dyn = characterizeVehicleFromState(sim, midCorner, controlSets, duration, substeps);
    expect(dyn).toHaveLength(controlSets.length);
    for (const p of dyn) {
      expect(p.sweep.length).toBe(substeps + 1);
      expect(p.sweep[0]).toEqual({ x: 0, z: 0, heading: 0 });
      expect(p.duration).toBe(duration);
    }
  });
});
