// Lattice-fidelity invariants for the race primitive library.
//
// During search the planner rigid-composes primitive endpoints that were
// baked at the NEAREST start-speed bucket — so the bucket spacing bounds
// how much of the forward model's speed-dependence survives into the
// plan. These tests pin (a) the grid density and (b) the actual endpoint
// error introduced by nearest-bucket quantization, measured with the v2
// parametric model (the kinematic model is speed-independent, so only a
// dynamics-aware model exposes this error class).

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  parametricForwardV2,
} from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import { RACE_AGENT, RACE_START_SPEEDS } from '../app/lib/race-primitives-scenarios';

const cfg = DEFAULT_LEARNABLE_CONFIG;
const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);

/** Roll the v2 model for `duration` from rest-frame state at `startSpeed`,
 *  mirroring characterizeVehicle's substepping. */
function rollout(startSpeed: number, controls: number[], duration: number, substeps = 6): CarKinematicState {
  let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: startSpeed, yawRate: 0, lateralVelocity: 0, t: 0 };
  const dt = duration / substeps;
  for (let i = 0; i < substeps; i++) s = sim(s, controls, dt);
  return s;
}

function nearestBucket(speed: number, buckets: number[]): number {
  let best = buckets[0]!;
  for (const b of buckets) if (Math.abs(b - speed) < Math.abs(best - speed)) best = b;
  return best;
}

/** Worst-case endpoint position error of nearest-bucket quantization over
 *  a probe control set, sweeping true start speeds across [2, 28]. */
function worstQuantizationError(buckets: number[]): number {
  const st = cfg.maxSteerAngle;
  const drv = cfg.maxDriveForce;
  const brk = cfg.maxBrakeForce;
  const probeControls: number[][] = [
    [0, drv, 0],
    [0, 0, brk],
    [st * 0.5, 0.5 * drv, 0],
    [-st * 0.3, 0, 0.5 * brk],
  ];
  const duration = 0.8;
  let worst = 0;
  for (let v = 2; v <= 28; v += 1) {
    const b = nearestBucket(v, buckets);
    for (const c of probeControls) {
      const baked = rollout(b, c, duration);
      const truth = rollout(v, c, duration);
      // The planner composes the baked OFFSET from the true state, so the
      // positional error is the offset difference.
      const err = Math.hypot(baked.x - truth.x, baked.z - truth.z);
      if (err > worst) worst = err;
    }
  }
  return worst;
}

describe('race start-speed lattice', () => {
  it('bucket spacing keeps nearest-bucket speed error ≤ 2 m/s across the envelope', () => {
    const sorted = [...RACE_START_SPEEDS].sort((a, b) => a - b);
    expect(sorted[0]).toBe(0);
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]! - sorted[i - 1]!).toBeLessThanOrEqual(4);
    }
    // Top bucket close enough to the agent ceiling that top-speed nodes
    // aren't expanded with mid-speed dynamics.
    expect(RACE_AGENT.maxSpeed - sorted[sorted.length - 1]!).toBeLessThanOrEqual(4);
  });

  it('nearest-bucket quantization stays under budget (and beats the old grid)', () => {
    // Measured with the default v2 parametric model: the old [0,10,20,28]
    // grid produced up to ~3.9 m of endpoint error in a SINGLE 0.8 s
    // primitive (a 15 m/s node expanded with 10/20 m/s dynamics) — larger
    // than the model's entire open-loop 1 s error. The 4 m/s grid brings
    // the worst case to ~1.6 m. Budget has ~15% headroom so a re-fit of
    // the default params doesn't flap the test; a genuine grid regression
    // blows through it.
    const oldGrid = worstQuantizationError([0, 10, 20, 28]);
    const newGrid = worstQuantizationError([...RACE_START_SPEEDS]);
    expect(newGrid).toBeLessThan(oldGrid / 2);
    expect(newGrid).toBeLessThanOrEqual(1.85);
  });
});
