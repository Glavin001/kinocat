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
  buildMLPInput,
  MLP_INPUT_DIM,
  createMLP,
  computeInputSupport,
  inputSupportDistance,
  type MLP,
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

describe('MLP input layout — translation + heading invariance', () => {
  it('buildMLPInput omits absolute position (translation-invariant)', () => {
    // Same dynamic state at two world positions → identical MLP input.
    const a = buildMLPInput(
      { x: 0, z: 0, heading: 0.3, speed: 5, yawRate: 0.1, lateralVelocity: -0.2, t: 0 },
      [0.05, 800, 100],
      DEFAULT_LEARNABLE_CONFIG,
    );
    const b = buildMLPInput(
      { x: -52, z: -1.5, heading: 0.3, speed: 5, yawRate: 0.1, lateralVelocity: -0.2, t: 0 },
      [0.05, 800, 100],
      DEFAULT_LEARNABLE_CONFIG,
    );
    expect(a).toEqual(b);
  });

  it('buildMLPInput represents heading as (sin, cos) — smooth across the ±π wrap', () => {
    const justBelowPi = buildMLPInput(
      { x: 0, z: 0, heading: Math.PI - 1e-6, speed: 5, yawRate: 0, lateralVelocity: 0, t: 0 },
      [0, 0, 0],
      DEFAULT_LEARNABLE_CONFIG,
    );
    const justAboveNegPi = buildMLPInput(
      { x: 0, z: 0, heading: -Math.PI + 1e-6, speed: 5, yawRate: 0, lateralVelocity: 0, t: 0 },
      [0, 0, 0],
      DEFAULT_LEARNABLE_CONFIG,
    );
    // Same physical heading wrapped on opposite sides of ±π → input
    // vectors are within FP noise of each other.
    for (let i = 0; i < justBelowPi.length; i++) {
      expect(Math.abs(justBelowPi[i]! - justAboveNegPi[i]!)).toBeLessThan(1e-3);
    }
  });

  it('MLP_INPUT_DIM is 21 (5 state + 3 controls + 13 config)', () => {
    expect(MLP_INPUT_DIM).toBe(21);
    const input = buildMLPInput(
      { x: 0, z: 0, heading: 0, speed: 0, t: 0 },
      [0, 0, 0],
      DEFAULT_LEARNABLE_CONFIG,
    );
    expect(input).toHaveLength(MLP_INPUT_DIM);
  });
});

describe('learnedForwardSimV2 — OOD fallback', () => {
  // Untrained MLPs initialised with different random seeds disagree
  // on most inputs (small but nonzero std) — perfect for testing
  // that the fallback FIRES under OOD-like conditions when we drop
  // the OOD threshold low enough to catch even untrained-ensemble
  // disagreement.
  function makeUntrainedEnsemble(size: number): MLP[] {
    return Array.from({ length: size }, (_, i) =>
      createMLP({ inputDim: MLP_INPUT_DIM, hiddenDims: [16], outputDim: 6 }, i + 1),
    );
  }

  it('falls back to parametric prediction when ensemble disagrees beyond threshold', () => {
    const para = buildParametricOnlyModel();
    // Tight threshold (0.001 on every channel) — even small disagreement
    // among untrained MLPs trips the fallback.
    const ood = {
      ...para,
      residualEnsemble: makeUntrainedEnsemble(3),
      residualReferenceDt: 1 / 60,
      oodStdThreshold: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    };
    const fwd = learnedForwardSimV2(ood);
    const directParam = parametricForwardV2(para.params, para.config);
    const s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 5, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const ctrl = [0, 0, 0];
    const got = fwd(s, ctrl, 1 / 60);
    const expected = directParam(s, ctrl, 1 / 60);
    // OOD fallback fires → output equals parametric exactly.
    expect(got.speed).toBeCloseTo(expected.speed, 10);
    expect(got.x).toBeCloseTo(expected.x, 10);
    expect(got.z).toBeCloseTo(expected.z, 10);
    expect(got.heading).toBeCloseTo(expected.heading, 10);
  });

  it('does NOT fall back when ensemble disagreement is within threshold', () => {
    const para = buildParametricOnlyModel();
    // Generous threshold (1.0 on every channel) — untrained MLPs disagree
    // by less than 1.0 on their output (final-layer std is 0.01), so the
    // fallback should NOT fire and the residual is applied.
    const model = {
      ...para,
      residualEnsemble: makeUntrainedEnsemble(3),
      residualReferenceDt: 1 / 60,
      oodStdThreshold: [100, 100, 100, 100, 100, 100],
    };
    const fwd = learnedForwardSimV2(model);
    const directParam = parametricForwardV2(para.params, para.config);
    const s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 5, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const ctrl = [0, 0, 0];
    const got = fwd(s, ctrl, 1 / 60);
    const baseParam = directParam(s, ctrl, 1 / 60);
    // The residual ensemble agrees-ish (untrained, small outputs) — its
    // mean is applied as a delta. Outputs differ from pure parametric
    // by a small amount.
    const diff = Math.abs(got.speed - baseParam.speed) +
      Math.abs(got.x - baseParam.x) +
      Math.abs(got.z - baseParam.z);
    // Untrained MLPs output ~0.01-scale values; with dt=1/60 the delta
    // is small but non-zero.
    expect(diff).toBeGreaterThan(0);
  });

  it('single-MLP "ensemble" skips OOD check (no variance to measure)', () => {
    const para = buildParametricOnlyModel();
    const single = {
      ...para,
      residualEnsemble: makeUntrainedEnsemble(1),
      residualReferenceDt: 1 / 60,
      oodStdThreshold: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    };
    const fwd = learnedForwardSimV2(single);
    const directParam = parametricForwardV2(para.params, para.config);
    const s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 5, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const ctrl = [0, 0, 0];
    const got = fwd(s, ctrl, 1 / 60);
    const baseParam = directParam(s, ctrl, 1 / 60);
    // 1-MLP ensemble always applies residual unconditionally (can't
    // measure variance over a single sample). Output differs from
    // pure parametric.
    const diff = Math.abs(got.speed - baseParam.speed) +
      Math.abs(got.x - baseParam.x) +
      Math.abs(got.z - baseParam.z);
    expect(diff).toBeGreaterThan(0);
  });
});

describe('coverage OOD gate (inputSupport)', () => {
  // Build a support cloud over a narrow forward-driving regime: speeds 6-10,
  // gentle steer, modest throttle, NO reverse and NO sliding. Reverse /
  // mid-slide queries are then genuinely out of support.
  function trainingInputs(): number[][] {
    const inputs: number[][] = [];
    for (let speed = 6; speed <= 10; speed += 0.5) {
      for (const steer of [-0.1, 0, 0.1]) {
        for (const drive of [800, 1600, 2400]) {
          const s: CarKinematicState = {
            x: 0, z: 0, heading: 0, speed, yawRate: 0, lateralVelocity: 0, t: 0,
          };
          inputs.push(buildMLPInput(s, [steer, drive, 0], cfg));
        }
      }
    }
    return inputs;
  }

  it('computeInputSupport: in-distribution distance < threshold < far-OOD distance', () => {
    const support = computeInputSupport(trainingInputs(), 0.99)!;
    expect(support).not.toBeNull();
    expect(support.mean).toHaveLength(MLP_INPUT_DIM);
    // A point near the centre of the cloud (speed 8, straight, mid throttle).
    const inDist = inputSupportDistance(
      support,
      buildMLPInput({ x: 0, z: 0, heading: 0, speed: 8, yawRate: 0, lateralVelocity: 0, t: 0 }, [0, 1600, 0], cfg),
    );
    // Reverse at speed -4 with full reverse force, mid-slide — far outside.
    const oodDist = inputSupportDistance(
      support,
      buildMLPInput({ x: 0, z: 0, heading: 0, speed: -4, yawRate: 1.2, lateralVelocity: 3, t: 0 }, [0, -2000, 0], cfg),
    );
    expect(inDist).toBeLessThan(support.threshold);
    expect(oodDist).toBeGreaterThan(support.threshold);
  });

  it('learnedForwardSimV2 falls back to parametric on an out-of-support query, even when the ensemble agrees', () => {
    const para = buildParametricOnlyModel();
    const model = {
      ...para,
      residualEnsemble: makeEnsemble(3),
      residualReferenceDt: 1 / 60,
      // Generous variance threshold: the variance gate would NOT fire here.
      oodStdThreshold: [100, 100, 100, 100, 100, 100],
      inputSupport: computeInputSupport(trainingInputs(), 0.99)!,
    };
    const fwd = learnedForwardSimV2(model);
    const directParam = parametricForwardV2(para.params, para.config);
    // Out-of-support: reverse + slide.
    const sOod: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: -4, yawRate: 1.2, lateralVelocity: 3, t: 0,
    };
    const ctrl = [0, -2000, 0];
    const got = fwd(sOod, ctrl, 1 / 60);
    const expected = directParam(sOod, ctrl, 1 / 60);
    expect(got.speed).toBeCloseTo(expected.speed, 10);
    expect(got.x).toBeCloseTo(expected.x, 10);
    expect(got.heading).toBeCloseTo(expected.heading, 10);

    // In-support: residual IS applied (output differs from pure parametric).
    const sIn: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 8, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    const inCtrl = [0, 1600, 0];
    const gotIn = fwd(sIn, inCtrl, 1 / 60);
    const baseIn = directParam(sIn, inCtrl, 1 / 60);
    const diff = Math.abs(gotIn.speed - baseIn.speed) + Math.abs(gotIn.x - baseIn.x);
    expect(diff).toBeGreaterThan(0);
  });

  it('predictWithUncertainty reports supportDistance and ood=true out of support', () => {
    const para = buildParametricOnlyModel();
    const model = {
      ...para,
      residualEnsemble: makeEnsemble(3),
      residualReferenceDt: 1 / 60,
      oodStdThreshold: [100, 100, 100, 100, 100, 100],
      inputSupport: computeInputSupport(trainingInputs(), 0.99)!,
    };
    const outOfSupport = predictWithUncertainty(
      model,
      { x: 0, z: 0, heading: 0, speed: -4, yawRate: 1.2, lateralVelocity: 3, t: 0 },
      [0, -2000, 0],
      1 / 60,
    );
    expect(outOfSupport.ood).toBe(true);
    expect(outOfSupport.supportDistance!).toBeGreaterThan(model.inputSupport.threshold);

    const inSupport = predictWithUncertainty(
      model,
      { x: 0, z: 0, heading: 0, speed: 8, yawRate: 0, lateralVelocity: 0, t: 0 },
      [0, 1600, 0],
      1 / 60,
    );
    expect(inSupport.ood).toBe(false);
    expect(inSupport.supportDistance!).toBeLessThan(model.inputSupport.threshold);
  });

  it('computeInputSupport returns null for an empty input set', () => {
    expect(computeInputSupport([])).toBeNull();
  });
});

// Deterministic small ensemble for the coverage tests (trained-ish: tiny
// random weights, all members near-identical so the variance gate stays quiet
// and we isolate the coverage gate's behaviour).
function makeEnsemble(size: number): MLP[] {
  return Array.from({ length: size }, (_, i) =>
    createMLP({ inputDim: MLP_INPUT_DIM, hiddenDims: [16], outputDim: 6 }, i + 1),
  );
}
