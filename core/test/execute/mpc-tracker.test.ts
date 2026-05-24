import { describe, it, expect } from 'vitest';
import { mpcTrack, createMPCTrackerState } from '../../src/execute/mpc-tracker';
import { parametricForwardV2 } from '../../src/agent/vehicle-model';
import { DEFAULT_LEARNED_PARAMS_V2 } from '../../src/agent/vehicle-model';
import { DEFAULT_LEARNABLE_CONFIG } from '../../src/agent/vehicle-config';
import type { CarKinematicState } from '../../src/agent/types';

function buildForwardSim() {
  return parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
}

function straightPath(length: number, speed: number, samples = 30): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  for (let i = 0; i < samples; i++) {
    const u = i / (samples - 1);
    out.push({ x: length * u, z: 0, heading: 0, speed, t: (length * u) / Math.max(speed, 1) });
  }
  return out;
}

describe('mpcTrack', () => {
  it('returns wheeled commands within actuator limits', () => {
    const path = straightPath(20, 6);
    const sim = buildForwardSim();
    const state = createMPCTrackerState(6, 42);
    const cur: CarKinematicState = { x: 0, z: 0.5, heading: 0, speed: 0, t: 0 };
    const cmd = mpcTrack(cur, path, sim, state, {
      maxSteer: 0.6,
      maxDriveForce: 4000,
      maxBrakeForce: 2000,
      samples: 16,
      horizonSteps: 6,
    });
    expect(Math.abs(cmd.steer)).toBeLessThanOrEqual(0.6 + 1e-9);
    expect(Math.abs(cmd.driveForce)).toBeLessThanOrEqual(4000 + 1e-6);
    expect(cmd.brakeForce).toBeGreaterThanOrEqual(0);
    expect(cmd.brakeForce).toBeLessThanOrEqual(2000 + 1e-6);
    expect(Number.isFinite(cmd.bestCost)).toBe(true);
  });

  it('accelerates from rest and makes forward progress along the path', () => {
    const path = straightPath(40, 6, 50);
    const sim = buildForwardSim();
    const state = createMPCTrackerState(6, 7);
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: 4000,
      maxBrakeForce: 2000,
      samples: 32,
      horizonSteps: 6,
      wLateral: 8,
      wHeading: 2,
      wSpeed: 1,
      wControlRate: 0.3,
    };
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let i = 0; i < 60; i++) {
      const cmd = mpcTrack(s, path, sim, state, cfg);
      s = sim(s, [cmd.steer, cmd.driveForce, cmd.brakeForce], 0.05);
    }
    // 3 s @ 20 Hz of closed-loop control: the chassis should have
    // accelerated from rest and travelled meaningfully along the path.
    expect(s.x).toBeGreaterThan(2);
    expect(s.speed).toBeGreaterThan(0.5);
  });

  it('updates warm-start state across calls in a non-stationary scenario', () => {
    const path = straightPath(20, 6);
    const sim = buildForwardSim();
    const state = createMPCTrackerState(6, 11);
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: 4000,
      maxBrakeForce: 2000,
      samples: 8,
      horizonSteps: 6,
    };
    // Stationary state → tracker can hold steady → warm-start may stay
    // approximately equal. Use a moving / disturbed state instead so
    // the optimal sequence actually shifts between calls.
    let s: CarKinematicState = { x: 0, z: 0.5, heading: 0.2, speed: 2, t: 0 };
    mpcTrack(s, path, sim, state, cfg);
    const before = Array.from(state.prev);
    s = sim(s, [state.prev[0]!, state.prev[1]!, state.prev[2]!], 0.05);
    mpcTrack(s, path, sim, state, cfg);
    const after = Array.from(state.prev);
    // Some control changed between calls. Allow a tiny floor for FP wobble.
    let maxAbsDelta = 0;
    for (let i = 0; i < before.length; i++) {
      maxAbsDelta = Math.max(maxAbsDelta, Math.abs(before[i]! - after[i]!));
    }
    expect(maxAbsDelta).toBeGreaterThan(1e-6);
  });

  it('deterministic with the same RNG seed', () => {
    const path = straightPath(20, 6);
    const sim = buildForwardSim();
    const cur: CarKinematicState = { x: 0, z: 0.5, heading: 0, speed: 0, t: 0 };
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: 4000,
      maxBrakeForce: 2000,
      samples: 16,
      horizonSteps: 6,
    };
    const a = mpcTrack(cur, path, sim, createMPCTrackerState(6, 5), cfg);
    const b = mpcTrack(cur, path, sim, createMPCTrackerState(6, 5), cfg);
    expect(a.steer).toBe(b.steer);
    expect(a.driveForce).toBe(b.driveForce);
    expect(a.brakeForce).toBe(b.brakeForce);
  });
});
