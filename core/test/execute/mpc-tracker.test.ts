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

  it('populates the warm-start buffer after a call', () => {
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
    // Pre-call: warm-start buffer is all zeros.
    expect(Array.from(state.prev).every((v) => v === 0)).toBe(true);
    const s: CarKinematicState = { x: 0, z: 0.5, heading: 0.2, speed: 0, t: 0 };
    mpcTrack(s, path, sim, state, cfg);
    // Post-call: at least one control is non-zero (deterministic anchors
    // like "full-throttle" or random samples guarantee this for any
    // sensible starting state).
    expect(Array.from(state.prev).some((v) => v !== 0)).toBe(true);
  });

  it('commits a reverse shunt after a gear flip instead of braking in place', () => {
    // Regression for the learned-model gate wedge: at a forward→reverse cusp
    // the warm-start prior holds the full brake latched while stopping in.
    // Carried into the reverse gear it saturates the pedal channel and no
    // sample discovers the shunt, so the car brakes in place forever. The
    // gear-flip prior reseed must let the tracker actually back up.
    const sim = buildForwardSim();
    // Reverse leg: travel tangent points +x (heading 0) but the plan speeds
    // are negative, so the chassis must move in −x (back up along it).
    const revPath: CarKinematicState[] = [];
    for (let i = 0; i < 20; i++) {
      revPath.push({ x: -0.5 * i, z: 0, heading: 0, speed: -6, t: 0.09 * i });
    }
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: 4000,
      maxBrakeForce: 2000,
      samples: 48,
      horizonSteps: 12,
      costMode: 'progress' as const,
      cruiseSpeed: 30,
      maxReverseSpeed: 6,
      wProgress: 6,
      corridorHalfWidth: 2.5,
    };
    const state = createMPCTrackerState(12, 99);
    // Simulate having just tracked a FORWARD segment that braked to the cusp:
    // lastGear forward, warm-start prior a full brake at rest.
    state.lastGear = 1;
    for (let i = 0; i < 12; i++) state.prev[i * 3 + 2] = 2000;
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let i = 0; i < 30; i++) {
      const cmd = mpcTrack(s, revPath, sim, state, cfg);
      s = sim(s, [cmd.steer, cmd.driveForce, cmd.brakeForce], 0.05);
    }
    // 1.5 s of closed loop: the chassis must have actually backed up along
    // the reverse leg (not sat braking at the origin).
    expect(s.x).toBeLessThan(-0.5);
    expect(s.speed).toBeLessThan(0);
    expect(state.lastGear).toBe(-1);
  });

  it('feedforward seeds the prior so the plan launch throttle fires from rest', () => {
    // WS-1½ control feedforward. From a dead stop one 0.05 s MPC step barely
    // moves, so the progress reward can't distinguish coast from throttle and a
    // zero-seeded prior emits ~no drive at the first tick (the classic
    // launch-from-rest weakness). When the plan carries its generating
    // primitive's full-throttle control as `ff`, the prior is seeded from it and
    // the tracker commits real throttle immediately. Same seed + state isolate
    // the feedforward as the cause.
    const sim = buildForwardSim();
    const MAXD = 4000;
    const straight = (ff?: readonly [number, number, number]): CarKinematicState[] => {
      const out: CarKinematicState[] = [];
      for (let i = 0; i < 40; i++) {
        out.push({ x: 0.75 * i, z: 0, heading: 0, speed: 12, t: 0.05 * i, ff });
      }
      return out;
    };
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: MAXD,
      maxBrakeForce: 2000,
      samples: 32,
      horizonSteps: 30,
      costMode: 'progress' as const,
      cruiseSpeed: 20,
      noStopAtEnd: true,
      referenceExtension: 40,
    };
    const rest: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const noFF = mpcTrack(rest, straight(), sim, createMPCTrackerState(30, 3), {
      ...cfg,
      useFeedforward: false,
    });
    const withFF = mpcTrack(rest, straight([0, MAXD, 0]), sim, createMPCTrackerState(30, 3), {
      ...cfg,
      useFeedforward: true,
    });
    // The plan's own full-throttle control fires immediately …
    expect(withFF.driveForce).toBeGreaterThan(0.5 * MAXD);
    // … and clearly harder than the geometry-only baseline from the same seed.
    expect(withFF.driveForce).toBeGreaterThan(noFF.driveForce + 0.25 * MAXD);
    expect(withFF.brakeForce).toBe(0);
  });

  it('feedforward respects a brake-hold plan control (does not invent throttle)', () => {
    // The feedforward baseline is the plan's control, not a throttle bias: a
    // plan sample carrying a hard-brake `ff` must not launch. Guards against a
    // feedforward implementation that only ever seeds drive.
    const sim = buildForwardSim();
    const MAXD = 4000;
    const MAXB = 2000;
    const braking: CarKinematicState[] = [];
    for (let i = 0; i < 40; i++) {
      braking.push({ x: 0.2 * i, z: 0, heading: 0, speed: 2, t: 0.1 * i, ff: [0, 0, MAXB] });
    }
    const cfg = {
      maxSteer: 0.6,
      maxDriveForce: MAXD,
      maxBrakeForce: MAXB,
      samples: 32,
      horizonSteps: 30,
      costMode: 'progress' as const,
      cruiseSpeed: 20,
      noStopAtEnd: true,
      referenceExtension: 40,
      useFeedforward: true,
    };
    const cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 6, t: 0 };
    const cmd = mpcTrack(cur, braking, sim, createMPCTrackerState(30, 3), cfg);
    // The prior is a brake-hold, so the emitted longitudinal is not a launch.
    expect(cmd.driveForce).toBeLessThan(0.3 * MAXD);
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
