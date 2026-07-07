// Fit-parameter hygiene for the v2 model.
//
// Two params (`accelTau`, `loadTransferCoeff`) were historically declared,
// fit, and regularized while having zero effect on the integration body —
// burning optimizer dimensions on coefficients with no gradient. These
// tests pin the contract: deprecated params stay out of the fit vector,
// provably do not influence predictions, and out-of-bounds persisted
// params are detectable.

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  PARAMS_V2_ORDER,
  paramsV2ToVec,
  paramsV2FromVec,
  paramsV2OutOfBounds,
  parametricForwardV2,
  buildMLPInput,
  type LearnedVehicleParamsV2,
} from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const cfg = DEFAULT_LEARNABLE_CONFIG;

const PROBE_STATES: CarKinematicState[] = [
  { x: 0, z: 0, heading: 0, speed: 0, yawRate: 0, lateralVelocity: 0, t: 0 },
  { x: 3, z: -2, heading: 0.7, speed: 12, yawRate: 0.4, lateralVelocity: 0.8, t: 1 },
  { x: -5, z: 9, heading: -2.9, speed: -3, yawRate: -0.2, lateralVelocity: -0.3, t: 2 },
];
const PROBE_CONTROLS: number[][] = [
  [0, cfg.maxDriveForce, 0],
  [cfg.maxSteerAngle, 0.5 * cfg.maxDriveForce, 0],
  [-0.3, 0, cfg.maxBrakeForce],
  [0, -0.4 * cfg.maxDriveForce, 0],
];

function rollAll(params: LearnedVehicleParamsV2): CarKinematicState[] {
  const sim = parametricForwardV2(params, cfg);
  const out: CarKinematicState[] = [];
  for (const s of PROBE_STATES) {
    for (const c of PROBE_CONTROLS) {
      // Multi-step so any lag term would have visible effect.
      let cur = s;
      for (let i = 0; i < 12; i++) cur = sim(cur, c, 1 / 60);
      out.push(cur);
    }
  }
  return out;
}

describe('deprecated v2 params', () => {
  it('accelTau and loadTransferCoeff are excluded from the fit vector', () => {
    expect(PARAMS_V2_ORDER).not.toContain('accelTau');
    expect(PARAMS_V2_ORDER).not.toContain('loadTransferCoeff');
    expect(paramsV2ToVec(DEFAULT_LEARNED_PARAMS_V2)).toHaveLength(PARAMS_V2_ORDER.length);
  });

  it('changing them does not change predictions', () => {
    const a = rollAll({ ...DEFAULT_LEARNED_PARAMS_V2, accelTau: 0.08, loadTransferCoeff: 0 });
    const b = rollAll({ ...DEFAULT_LEARNED_PARAMS_V2, accelTau: 0.6, loadTransferCoeff: 0.1 });
    expect(a).toEqual(b);
  });

  it('fit vector round-trips through encode/decode', () => {
    const vec = paramsV2ToVec(DEFAULT_LEARNED_PARAMS_V2);
    const back = paramsV2FromVec(vec);
    for (const k of PARAMS_V2_ORDER) {
      expect(back[k]).toBeCloseTo(DEFAULT_LEARNED_PARAMS_V2[k], 12);
    }
  });
});

describe('paramsV2OutOfBounds', () => {
  it('default params are within bounds', () => {
    expect(paramsV2OutOfBounds(DEFAULT_LEARNED_PARAMS_V2)).toEqual([]);
  });

  it('flags params outside the current physical-plausibility bounds', () => {
    // engineScale ceiling is 2.2 (drivenWheelCount × F is the real
    // propulsion force — the WS-0 envelope corrected the old 1.05 bound).
    const stale = { ...DEFAULT_LEARNED_PARAMS_V2, brakeScale: 3.0, engineScale: 2.5 };
    const flagged = paramsV2OutOfBounds(stale);
    expect(flagged).toContain('brakeScale');
    expect(flagged).toContain('engineScale');
  });

  it('does not flag deprecated non-fit params', () => {
    const p = { ...DEFAULT_LEARNED_PARAMS_V2, accelTau: 99, loadTransferCoeff: 99 };
    expect(paramsV2OutOfBounds(p)).toEqual([]);
  });
});

describe('MLP input control normalisation', () => {
  it('scales drive/brake by the config own force limits', () => {
    const atLimit = buildMLPInput(
      PROBE_STATES[0]!,
      [0, cfg.maxDriveForce, cfg.maxBrakeForce],
      cfg,
    );
    // Control channels sit right after the 5 state dims.
    expect(atLimit[6]).toBeCloseTo(1, 12);
    expect(atLimit[7]).toBeCloseTo(1, 12);

    const doubled = { ...cfg, maxDriveForce: 2 * cfg.maxDriveForce, maxBrakeForce: 2 * cfg.maxBrakeForce };
    const atDoubledLimit = buildMLPInput(
      PROBE_STATES[0]!,
      [0, doubled.maxDriveForce, doubled.maxBrakeForce],
      doubled,
    );
    expect(atDoubledLimit[6]).toBeCloseTo(1, 12);
    expect(atDoubledLimit[7]).toBeCloseTo(1, 12);
  });
});
