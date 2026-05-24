// Behavioural tests for the v2 parametric vehicle model.
//
// We don't test absolute physics fidelity here (that's measured against a
// Rapier trial in the integration tests); we test the *structure* of the
// model: friction-circle clamp engages, yaw-rate inertia decays, the
// closed-form is deterministic and dt-consistent.

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  parametricForwardV2,
  buildParametricOnlyModel,
  learnedForwardSimV2,
  predictWithUncertainty,
} from 'kinocat/agent';
import { DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const cfg = DEFAULT_LEARNABLE_CONFIG;

function rollFor(
  steer: number,
  driveForce: number,
  brakeForce: number,
  ticks: number,
  startSpeed = 8,
  initial?: Partial<CarKinematicState>,
): CarKinematicState {
  const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
  let s: CarKinematicState = {
    x: 0, z: 0, heading: 0, speed: startSpeed,
    yawRate: 0, lateralVelocity: 0, t: 0,
    ...initial,
  };
  const dt = 1 / 60;
  for (let i = 0; i < ticks; i++) {
    s = sim(s, [steer, driveForce, brakeForce], dt);
  }
  return s;
}

describe('parametricForwardV2 — structural invariants', () => {
  it('friction-circle clamp engages: steady-state max-brake + max-steer at speed caps total accel', () => {
    // Run for enough ticks that yaw rate reaches steady state at the
    // commanded curvature, then measure long + lat (centripetal) accel.
    // Cap: gripScale * frictionSlip * G * frictionCircleSlack.
    const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
    let s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 10, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const dt = 1 / 60;
    // Drive for ~0.3s so yaw rate has time to actualize despite yawRateTau.
    for (let i = 0; i < 18; i++) {
      const prev = s.speed;
      s = sim(s, [cfg.maxSteerAngle, 0, cfg.maxBrakeForce], dt);
      // Stop once stopped — measuring beyond that is meaningless.
      if (Math.abs(s.speed) < 0.1) break;
      const aLong = Math.abs((s.speed - prev) / dt);
      const aLat = Math.abs((s.yawRate ?? 0) * s.speed); // centripetal
      const mag = Math.sqrt(aLong * aLong + aLat * aLat);
      const cap = DEFAULT_LEARNED_PARAMS_V2.frictionCircleSlack
        * DEFAULT_LEARNED_PARAMS_V2.gripScale * cfg.frictionSlip * 9.81;
      // Allow 1.2× cap for transient overshoot from the first-order step
      // discretization.
      expect(mag).toBeLessThanOrEqual(cap * 1.2);
    }
  });

  it('yaw-rate inertia: with zero steer from yawRate=1, decays toward 0', () => {
    const after = rollFor(0, 0, 0, 60, 8, { yawRate: 1.0 }); // 1 second
    expect(Math.abs(after.yawRate ?? 0)).toBeLessThan(0.5);
  });

  it('deterministic — same inputs produce identical outputs', () => {
    const a = rollFor(0.2, 1500, 0, 30);
    const b = rollFor(0.2, 1500, 0, 30);
    expect(a).toEqual(b);
  });

  it('dt-consistent: two half-dt steps approximately equal one full-dt step', () => {
    const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
    const start: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 6, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const dt = 0.1;
    const full = sim(start, [0.1, 1000, 0], dt);
    const half = sim(sim(start, [0.1, 1000, 0], dt / 2), [0.1, 1000, 0], dt / 2);
    const dx = Math.abs(full.x - half.x);
    const dz = Math.abs(full.z - half.z);
    expect(Math.hypot(dx, dz)).toBeLessThan(0.05); // within 5 cm
  });

  it('lateralVelocity is excited by steer and damped to ~0 without steer', () => {
    const turning = rollFor(0.3, 1500, 0, 20);
    expect(Math.abs(turning.lateralVelocity ?? 0)).toBeGreaterThan(0.05);
    const straightened = rollFor(0, 0, 0, 120, 8, { lateralVelocity: 1.0 });
    expect(Math.abs(straightened.lateralVelocity ?? 0)).toBeLessThan(0.3);
  });
});

describe('learnedForwardSimV2 + predictWithUncertainty', () => {
  it('parametric-only model returns zero uncertainty', () => {
    const model = buildParametricOnlyModel();
    const r = predictWithUncertainty(model,
      { x: 0, z: 0, heading: 0, speed: 8, yawRate: 0, lateralVelocity: 0, t: 0 },
      [0.1, 1000, 0],
      1 / 60,
    );
    expect(r.std).toHaveLength(6);
    for (const v of r.std) expect(v).toBe(0);
  });

  it('drop-in via learnedForwardSimV2 produces identical states to direct parametric', () => {
    const model = buildParametricOnlyModel();
    const wrapped = learnedForwardSimV2(model);
    const direct = parametricForwardV2(model.params, model.config);
    const start: CarKinematicState = {
      x: 1, z: 2, heading: 0.3, speed: 5, yawRate: 0.1, lateralVelocity: -0.2, t: 0,
    };
    const ctrl = [0.05, 800, 100];
    expect(wrapped(start, ctrl, 1 / 60)).toEqual(direct(start, ctrl, 1 / 60));
  });
});
