// V3 learned vehicle model — PURELY data-driven neural dynamics.
//
// Motivation (learned the hard way from v2): the v2 model is a hand-written
// parametric backbone with fitted knobs, clamped to hand-coded
// "physical-plausibility" bounds, plus a residual MLP. One of those hand-set
// bounds was factually wrong about the plant (`engineScale ≤ 1.05` assumed
// `driveForce` is the TOTAL propulsion force; the Rapier adapter applies it
// to EACH driven wheel, so the real full-throttle acceleration is ~13.9 m/s²
// while the clamped backbone can only express ~7.3 m/s²). The residual MLP
// silently absorbed a 2× longitudinal error — and every OOD fallback to the
// "safe" backbone re-introduced it. The lesson: any hand-written internal
// structure is a place for hidden falsehoods to live.
//
// V3 therefore contains NO hand-written dynamics and NO hand-set parameter
// bounds. The transition function is a neural network trained directly on
// recorded plant transitions, and every constant it needs (input/output
// normalization) is a statistic computed from that same data.
//
// The only structure imposed is the exact symmetry of the plant itself:
// on a flat, uniform-friction plane the chassis dynamics are
// translation-invariant and rotation-equivariant. The model therefore works
// entirely in the chassis BODY frame — inputs carry no position and no
// absolute heading, and outputs are body-frame deltas rotated into the world
// frame at integration time. This is not a modeling assumption; it is a
// property of the plant we would otherwise waste capacity re-learning
// (v2's residual MLP received sin/cos(heading) and could therefore learn
// spurious direction-dependent corrections).
//
// State (6):    x, z, heading (frame — never model inputs),
//               speed, yawRate, lateralVelocity (body-frame dynamic state).
// Controls (3): [steer, driveForce, brakeForce] — the plant's native
//               actuator vocabulary, same encoding as v2.
// Prediction:   body-frame deltas over one reference step (the plant's own
//               physics tick): [dFwd, dRight, dHeading, dSpeed, dYawRate,
//               dLateralVelocity].

import type { CarKinematicState } from './types';
import type { LearnableVehicleConfig } from './vehicle-config';
import type { ForwardSim } from '../primitives/types';
import { wrapAngle } from '../internal/math';
import {
  type MLP,
  forward as mlpForward,
  serializeMLP,
  deserializeMLP,
} from '../internal/mlp';

/** Raw model inputs: [speed, yawRate, lateralVelocity, steer, driveForce,
 *  brakeForce]. Position and heading are deliberately absent — see the
 *  body-frame rationale in the file header. */
export const V3_INPUT_DIM = 6;

/** Body-frame deltas over one reference step:
 *  [dFwd, dRight, dHeading, dSpeed, dYawRate, dLateralVelocity]. */
export const V3_OUTPUT_DIM = 6;

/** Normalization statistics — computed from the training data, never
 *  hand-set. Inputs are z-scored; the network predicts z-scored targets. */
export interface V3Normalization {
  inputMean: number[];
  inputStd: number[];
  outputMean: number[];
  outputStd: number[];
}

export interface LearnedVehicleModelV3 {
  /** Vehicle config the training data was recorded with. Carried for
   *  bookkeeping/validation only — the dynamics themselves are entirely in
   *  the network weights. */
  config: LearnableVehicleConfig;
  /** Independently-seeded ensemble members; inference uses the mean.
   *  The per-dim spread is available as an epistemic-uncertainty signal
   *  (see `predictWithUncertaintyV3`) — there is deliberately NO fallback
   *  model to switch to. */
  ensemble: MLP[];
  norm: V3Normalization;
  /** The step the deltas were learned at — the plant's physics tick. */
  referenceDt: number;
}

/** Assemble the raw (un-normalized) input vector. Exported so training and
 *  inference share one encoding and cannot drift. */
export function buildV3Input(
  s: CarKinematicState,
  controls: ReadonlyArray<number>,
): number[] {
  return [
    s.speed,
    s.yawRate ?? 0,
    s.lateralVelocity ?? 0,
    controls[0] ?? 0,
    controls[1] ?? 0,
    controls[2] ?? 0,
  ];
}

/** Ensemble-mean prediction of the RAW (denormalized) body-frame delta for
 *  one full reference step. */
function predictDelta(model: LearnedVehicleModelV3, raw: number[]): Float64Array {
  const { norm, ensemble } = model;
  const input = new Array<number>(V3_INPUT_DIM);
  for (let i = 0; i < V3_INPUT_DIM; i++) {
    input[i] = (raw[i]! - norm.inputMean[i]!) / norm.inputStd[i]!;
  }
  const mean = new Float64Array(V3_OUTPUT_DIM);
  for (const mlp of ensemble) {
    const out = mlpForward(mlp, input).output;
    for (let i = 0; i < V3_OUTPUT_DIM; i++) mean[i]! += out[i]! / ensemble.length;
  }
  for (let i = 0; i < V3_OUTPUT_DIM; i++) {
    mean[i] = mean[i]! * norm.outputStd[i]! + norm.outputMean[i]!;
  }
  return mean;
}

/** Apply a (possibly fractionally-scaled) body-frame delta to a state. */
function applyDelta(
  s: CarKinematicState,
  d: Float64Array,
  frac: number,
  stepDt: number,
): CarKinematicState {
  const cosH = Math.cos(s.heading);
  const sinH = Math.sin(s.heading);
  const dFwd = d[0]! * frac;
  const dRight = d[1]! * frac;
  // Body → world under the kinocat convention (heading rotates +X toward
  // +Z): forward = (cos h, sin h), right = (sin h, -cos h).
  return {
    x: s.x + dFwd * cosH + dRight * sinH,
    z: s.z + dFwd * sinH - dRight * cosH,
    heading: wrapAngle(s.heading + d[2]! * frac),
    speed: s.speed + d[3]! * frac,
    yawRate: (s.yawRate ?? 0) + d[4]! * frac,
    lateralVelocity: (s.lateralVelocity ?? 0) + d[5]! * frac,
    t: s.t + stepDt,
  };
}

/** Drop-in `ForwardSim<CarKinematicState>` for `characterizeVehicle`, the
 *  planner, and MPPI. Arbitrary `dt` queries are decomposed into whole
 *  reference steps plus one fractional step (delta linearly scaled) — the
 *  model is only ever asked to predict at the step it was trained on. */
export function forwardSimV3(model: LearnedVehicleModelV3): ForwardSim<CarKinematicState> {
  const refDt = model.referenceDt;
  return (s: CarKinematicState, controls: number[], dt: number): CarKinematicState => {
    const raw = buildV3Input(s, controls);
    let cur = s;
    let remaining = dt;
    while (remaining > 1e-9) {
      const frac = Math.min(1, remaining / refDt);
      const stepDt = refDt * frac;
      // Controls are held constant across the query (planner contract), but
      // the dynamic state changes every internal step — re-encode it.
      raw[0] = cur.speed;
      raw[1] = cur.yawRate ?? 0;
      raw[2] = cur.lateralVelocity ?? 0;
      cur = applyDelta(cur, predictDelta(model, raw), frac, stepDt);
      remaining -= stepDt;
    }
    return cur;
  };
}

export interface V3PredictionWithUncertainty {
  next: CarKinematicState;
  /** Per-output-dim ensemble standard deviation (raw units) — epistemic
   *  uncertainty signal for monitoring / planner cost shaping. */
  std: number[];
}

/** Single reference-step prediction with ensemble spread. */
export function predictWithUncertaintyV3(
  model: LearnedVehicleModelV3,
  s: CarKinematicState,
  controls: ReadonlyArray<number>,
): V3PredictionWithUncertainty {
  const { norm, ensemble } = model;
  const raw = buildV3Input(s, controls);
  const input = new Array<number>(V3_INPUT_DIM);
  for (let i = 0; i < V3_INPUT_DIM; i++) {
    input[i] = (raw[i]! - norm.inputMean[i]!) / norm.inputStd[i]!;
  }
  const outputs = ensemble.map((mlp) => mlpForward(mlp, input).output);
  const mean = new Float64Array(V3_OUTPUT_DIM);
  for (const o of outputs) for (let i = 0; i < V3_OUTPUT_DIM; i++) mean[i]! += o[i]! / outputs.length;
  const std = new Array<number>(V3_OUTPUT_DIM).fill(0);
  for (const o of outputs) {
    for (let i = 0; i < V3_OUTPUT_DIM; i++) {
      const d = o[i]! - mean[i]!;
      std[i]! += (d * d) / outputs.length;
    }
  }
  for (let i = 0; i < V3_OUTPUT_DIM; i++) {
    std[i] = Math.sqrt(std[i]!) * Math.abs(norm.outputStd[i]!);
    mean[i] = mean[i]! * norm.outputStd[i]! + norm.outputMean[i]!;
  }
  return { next: applyDelta(s, mean, 1, model.referenceDt), std };
}

// ---------------------------------------------------------------------------
// Persistence

export interface PersistedV3Model {
  version: 1;
  kind: 'v3-dynamics';
  config: LearnableVehicleConfig;
  referenceDt: number;
  norm: V3Normalization;
  ensembleJson: string[];
  meta?: Record<string, unknown>;
}

export function v3ToJson(
  model: LearnedVehicleModelV3,
  meta?: Record<string, unknown>,
): PersistedV3Model {
  return {
    version: 1,
    kind: 'v3-dynamics',
    config: model.config,
    referenceDt: model.referenceDt,
    norm: model.norm,
    ensembleJson: model.ensemble.map((m) => serializeMLP(m)),
    ...(meta ? { meta } : {}),
  };
}

export function v3FromJson(payload: PersistedV3Model): LearnedVehicleModelV3 {
  if (payload.kind !== 'v3-dynamics' || payload.version !== 1) {
    throw new Error(`not a v3 dynamics model payload (kind=${(payload as { kind?: string }).kind})`);
  }
  return {
    config: payload.config,
    referenceDt: payload.referenceDt,
    norm: payload.norm,
    ensemble: payload.ensembleJson.map((j) => deserializeMLP(j)),
  };
}

/** Duck-type check for routing loaders (a v2 payload has `params` +
 *  `residualEnsembleJson`; a v3 payload has `kind: 'v3-dynamics'`). */
export function isV3Payload(payload: unknown): payload is PersistedV3Model {
  return (
    typeof payload === 'object' &&
    payload !== null &&
    (payload as { kind?: string }).kind === 'v3-dynamics'
  );
}
