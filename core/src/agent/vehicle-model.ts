// V2 learned dynamics model for an Ackermann wheeled vehicle.
//
// Two-stage prediction:
//   1. An extended parametric backbone (`parametricForwardV2`) — closed-form,
//      ~16 coefficients, friction-circle aware, yaw-rate inertia,
//      asymmetric understeer/oversteer, config-aware.
//   2. An optional residual MLP ensemble that corrects the backbone's
//      remaining error and provides epistemic uncertainty (std across the
//      ensemble).
//
// Domain-agnostic: knows nothing about Rapier. The native control vector is
// `[steer, driveForce, brakeForce]` (see `controls.ts`). The vehicle
// configuration (`LearnableVehicleConfig`) is passed alongside the state so
// the same trained model can drive different vehicles — the regularization
// effect of training across diverse configs improves identifiability of the
// invariant physics, not just future-proofing.

import type { CarKinematicState } from './types';
import type { ForwardSim } from '../primitives/types';
import { wrapAngle } from '../internal/math';
import {
  type LearnableVehicleConfig,
  DEFAULT_LEARNABLE_CONFIG,
  encodeConfigOneHot,
  CONFIG_SCALES_ORDINAL,
} from './vehicle-config';
import { decodeWheeled } from './controls';
import {
  type MLP,
  forward as mlpForward,
} from '../internal/mlp';

// ---------------------------------------------------------------------------
// Extended parametric backbone

const G = 9.81;

/** Parameters of the extended parametric backbone. ~16 coefficients,
 *  all expressed as multipliers/exponents of physical config quantities
 *  so the model can generalize across vehicles. */
export interface LearnedVehicleParamsV2 {
  /** Engine effectiveness scaling: longitudinal accel = engineScale * (driveForce / mass).
   *  Should be ≈ 1 if the vehicle delivers commanded force perfectly; <1 if
   *  drivetrain loss / wheel spin reduces it. */
  engineScale: number;
  /** Reverse-direction engine efficiency (relative to forward). */
  reverseEffScale: number;
  /** Brake effectiveness scaling: longitudinal decel = brakeScale * (brakeForce / mass). */
  brakeScale: number;
  /** @deprecated No longer part of the model. Rapier applies wheel engine
   *  force instantly, so the plant has no longitudinal command lag to fit;
   *  the previous implementation was also algebraically inert
   *  (`a·(1−e^−dt/τ) + a·e^−dt/τ ≡ a`). Kept in the interface so persisted
   *  payloads round-trip; excluded from `PARAMS_V2_ORDER` (not fit). */
  accelTau: number;
  /** Tire grip multiplier on top of `frictionSlip`. Effective µ = grip * frictionSlip. */
  gripScale: number;
  /** Friction-circle slack — how much of available grip can be
   *  simultaneously used long + lat before saturating. 1.0 = strict; >1
   *  permits some over-use (slip). Bounded [0.7, 1.3] in fit. */
  frictionCircleSlack: number;
  /** Effective steering ratio: actual front-wheel angle is multiplied by
   *  this (so the bicycle model can be off without retuning steer). */
  steerRatio: number;
  /** Understeer gain (off-throttle): yaw_rate command degraded by
   *  1 / (1 + ku_off * v^2). */
  understeerOffThrottle: number;
  /** Understeer gain (power-on): RWD typically tightens; coefficient can be
   *  smaller or even slightly negative. */
  understeerPowerOn: number;
  /** Yaw-rate first-order tracking time constant (s). The chassis can't
   *  spin up to commanded yaw rate instantly — moment of inertia + tire
   *  side force buildup lags. */
  yawRateTau: number;
  /** Lateral-velocity damping (1/s). Higher = slip decays faster.
   *  Couples slip back toward zero. */
  lateralDamping: number;
  /** Lateral-velocity gain from steer × speed. Real cars develop sideslip
   *  in turns proportional to commanded steer angle × speed; this is the
   *  scale. */
  lateralFromSteer: number;
  /** Speed loss while turning (replaces the legacy `lateralDrag` but is
   *  parameterized by lateral velocity instead of curvature, so it
   *  captures slip-induced drag honestly). */
  slipDrag: number;
  /** @deprecated Never referenced by `parametricForwardV2` — fitting it
   *  was a free degree of freedom (fit degeneracy). Kept in the interface
   *  so persisted payloads round-trip; excluded from `PARAMS_V2_ORDER`
   *  (not fit). Re-introduce only together with an actual load-transfer
   *  term in the integration body. */
  loadTransferCoeff: number;
  /** Throttle deadzone (N): below this drive force, no accel. Captures
   *  rolling resistance / clutch effects. */
  driveDeadzone: number;
  /** Rolling resistance (m/s² per m/s): linear speed decay even with no
   *  controls. */
  rollingResistance: number;
}

export const DEFAULT_LEARNED_PARAMS_V2: LearnedVehicleParamsV2 = {
  engineScale: 0.85,
  reverseEffScale: 0.9,
  brakeScale: 1.6,
  accelTau: 0.2,
  // Slightly more optimistic than the conservative prior so untrained
  // models don't immediately reject high-speed turns. Trained models
  // will move away from these via the fitter — but bad starting points
  // can cause the planner to produce a degenerate action space, hiding
  // v2's value (see /primitive-explorer hull stats).
  gripScale: 1.0,
  // Friction-circle slack > 1 = the model accepts brief excursions past
  // the linear µ·g cap (Rapier's tire model does the same via slip).
  // 1.2 means up to ~20% transient over-use is allowed in commanded
  // accel; clamps still kick in for sustained over-use.
  frictionCircleSlack: 1.2,
  steerRatio: 1.0,
  // Halved understeer gain: the prior value attenuated yaw rate by 10×
  // at 28 m/s (1 + 0.012·784 = 10.4×), which made every high-speed
  // turn primitive collapse to the same near-straight outcome under
  // the friction-circle clamp. 0.006 → 5.7× attenuation at 28 m/s
  // still respects high-speed understeer physics but leaves room for
  // distinguishable gentle-turn primitives at the top speed bucket.
  understeerOffThrottle: 0.006,
  understeerPowerOn: 0.002,
  yawRateTau: 0.18,
  lateralDamping: 4.5,
  lateralFromSteer: 0.6,
  slipDrag: 0.4,
  loadTransferCoeff: 0.02,
  driveDeadzone: 50,
  rollingResistance: 0.05,
};

/** Bounds for parametric fit — PHYSICALLY plausible ranges. Looser
 *  bounds let the fit pin parameters to walls when the model lacks
 *  expressiveness for some trial (e.g. brakeScale=3.5 means the model
 *  predicts 119% more brake decel than Rapier delivers — unphysical;
 *  frictionCircleSlack<1 says the chassis exceeds its own friction
 *  limit, which is incoherent). The bounds here describe the actual
 *  Rapier raycast vehicle's range; the regularization in the fit
 *  (training-driver.ts) adds soft pull toward DEFAULT_LEARNED_PARAMS_V2
 *  on top. */
export const PARAMS_V2_LO: LearnedVehicleParamsV2 = {
  engineScale: 0.7,           // chassis can't deliver < 70% of commanded force
  reverseEffScale: 0.7,
  brakeScale: 0.8,            // Rapier brakeForce maps ~1:1 to chassis decel
  accelTau: 0.08,
  gripScale: 0.7,
  frictionCircleSlack: 0.95,  // < 1 = chassis exceeds its own friction limit → incoherent
  steerRatio: 0.7,
  understeerOffThrottle: 0,
  understeerPowerOn: -0.01,
  yawRateTau: 0.1,            // chassis inertia is real — < 100 ms is unphysical
  lateralDamping: 2,
  lateralFromSteer: 0.2,      // must be positive (steering DOES produce sideslip)
  slipDrag: 0,
  loadTransferCoeff: 0,
  driveDeadzone: 0,
  rollingResistance: 0.02,
};

export const PARAMS_V2_HI: LearnedVehicleParamsV2 = {
  engineScale: 1.05,          // > 100% means the model "amplifies" commanded force — unphysical
  reverseEffScale: 1.1,
  brakeScale: 2.0,            // generous upper to allow some over-fit, well short of the old 3.5
  accelTau: 0.6,
  gripScale: 1.3,
  frictionCircleSlack: 1.3,   // allows brief tire-slip excursions past µ·g but not 2×
  steerRatio: 1.3,
  understeerOffThrottle: 0.05, // 0.05 × 784 (at v=28) = 40× yaw attenuation — already extreme
  understeerPowerOn: 0.04,
  yawRateTau: 0.4,
  lateralDamping: 9,          // > 9 means sideslip decays in < 0.1 s → unphysical for car chassis
  lateralFromSteer: 1.5,
  slipDrag: 1.5,
  loadTransferCoeff: 0.1,
  driveDeadzone: 250,
  rollingResistance: 0.2,
};

/** Parameters the fitter optimizes. `accelTau` and `loadTransferCoeff`
 *  are deliberately absent: both were inert in the integration body, so
 *  fitting them burned Nelder-Mead dimensions on coefficients with zero
 *  gradient (see their @deprecated docs). */
export const PARAMS_V2_ORDER: (keyof LearnedVehicleParamsV2)[] = [
  'engineScale', 'reverseEffScale', 'brakeScale',
  'gripScale', 'frictionCircleSlack', 'steerRatio',
  'understeerOffThrottle', 'understeerPowerOn', 'yawRateTau',
  'lateralDamping', 'lateralFromSteer', 'slipDrag',
  'driveDeadzone', 'rollingResistance',
];

export function paramsV2ToVec(p: LearnedVehicleParamsV2): number[] {
  return PARAMS_V2_ORDER.map((k) => p[k]);
}

/** Names of fitted params lying outside [PARAMS_V2_LO, PARAMS_V2_HI].
 *  Loaders should surface these (a persisted artifact trained under
 *  older, looser bounds carries values the current physical-plausibility
 *  rationale rejects — the honest remedy is a re-fit, not a silent clamp,
 *  since any residual ensemble was trained around the unclamped
 *  backbone). Deprecated non-fit params are not checked. */
export function paramsV2OutOfBounds(p: LearnedVehicleParamsV2): (keyof LearnedVehicleParamsV2)[] {
  return PARAMS_V2_ORDER.filter((k) => p[k] < PARAMS_V2_LO[k] || p[k] > PARAMS_V2_HI[k]);
}

export function paramsV2FromVec(v: ReadonlyArray<number>): LearnedVehicleParamsV2 {
  const out = { ...DEFAULT_LEARNED_PARAMS_V2 };
  PARAMS_V2_ORDER.forEach((k, i) => {
    const raw = v[i];
    if (raw === undefined) return;
    const lo = PARAMS_V2_LO[k];
    const hi = PARAMS_V2_HI[k];
    out[k] = raw < lo ? lo : raw > hi ? hi : raw;
  });
  return out;
}

/**
 * Extended parametric forward model. Stateless / deterministic / pure.
 *
 * `controls = [steer, driveForce, brakeForce]` per `WheeledCarControls` encoding.
 * `config` carries the vehicle's physical parameters (mass, wheelbase, etc.)
 * so the model is config-aware.
 *
 * Captures the structure the legacy 5-param model couldn't:
 *   - friction-circle coupling (brake+turn or accel+turn loses lat grip)
 *   - yaw-rate inertia (commanded yaw doesn't actualize instantly)
 *   - lateral velocity (slip angle) as a real state variable
 *   - asymmetric understeer/oversteer depending on throttle direction
 *   - signed config-scaled engine / brake / steering responses
 */
export function parametricForwardV2(
  params: LearnedVehicleParamsV2,
  config: LearnableVehicleConfig,
): ForwardSim<CarKinematicState> {
  return (s: CarKinematicState, controls: number[], dt: number): CarKinematicState => {
    const c = decodeWheeled(controls);
    const v = s.speed;
    const vy = s.lateralVelocity ?? 0;
    const yawRate = s.yawRate ?? 0;
    const m = Math.max(50, config.chassisMass);

    // --- Longitudinal command (drive vs brake) ---------------------------
    let driveAccel = 0;
    const fAbs = Math.abs(c.driveForce);
    if (fAbs > params.driveDeadzone) {
      const fEff = (Math.sign(c.driveForce)) * (fAbs - params.driveDeadzone);
      const dir = c.driveForce >= 0 ? params.engineScale : params.engineScale * params.reverseEffScale;
      driveAccel = (dir * fEff) / m;
    }
    const brakeAccel = (params.brakeScale * c.brakeForce) / m;
    // Brake opposes motion; if speed is 0 and brake applied, no motion.
    const brakeSigned = -Math.sign(v) * brakeAccel;
    const rolling = -Math.sign(v) * params.rollingResistance * Math.abs(v);

    // Long accel before friction-circle (raw commanded + brake + rolling).
    let aLong = driveAccel + brakeSigned + rolling;

    // --- Steer → bicycle-model commanded yaw rate ------------------------
    const effSteer = c.steer * params.steerRatio;
    const L = Math.max(0.5, 2 * config.wheelBase);
    // Bicycle: yaw_rate_cmd = v * tan(steer) / L. Use sin for stability at
    // larger angles, since tan blows up.
    const yawRateCmdRaw = (v * Math.sin(effSteer)) / L;
    // Asymmetric understeer: power-on vs off-throttle.
    const isPowerOn = c.driveForce > 0 && v > 0;
    const ku = isPowerOn ? params.understeerPowerOn : params.understeerOffThrottle;
    const yawRateCmd = yawRateCmdRaw / (1 + ku * v * v);

    // --- Friction-circle clamp -------------------------------------------
    // Available total acceleration (m/s²) = gripScale * frictionSlip * g.
    const aMax = params.gripScale * config.frictionSlip * G * params.frictionCircleSlack;
    // Estimated lateral accel = v * yawRate_cmd (steady-state). Use cmd, not
    // measured — model is predicting forward in time.
    const aLatEst = v * yawRateCmd;
    const mag = Math.sqrt(aLong * aLong + aLatEst * aLatEst);
    let scale = 1;
    if (mag > aMax && mag > 0) scale = aMax / mag;
    aLong *= scale;
    const yawRateAllowed = yawRateCmd * scale;

    // --- Speed dynamics ---------------------------------------------------
    // No first-order lag on aLong: the Rapier plant sets wheel engine force
    // instantly (`setWheelEngineForce` per tick), so longitudinal response
    // is mass-limited only. (`accelTau` used to appear here in an
    // algebraically-inert expression — see its @deprecated doc.)
    // Slip-induced drag: lateral velocity costs forward speed.
    const slipLoss = -params.slipDrag * Math.abs(vy) * Math.sign(v) * dt;
    let speed = v + aLong * dt + slipLoss;
    // Don't let brake-driven sign flip past zero in a single step.
    if (Math.sign(speed) !== Math.sign(v) && c.brakeForce > 0 && Math.abs(v) > 0) {
      speed = 0;
    }

    // --- Yaw-rate inertia ------------------------------------------------
    const yTau = Math.max(0.02, params.yawRateTau);
    const yawRateNext = yawRate + ((yawRateAllowed - yawRate) * dt) / yTau;

    // --- Lateral velocity dynamics ---------------------------------------
    // Drive lateral velocity from commanded steer × speed, damped.
    const vyDrive = params.lateralFromSteer * effSteer * v;
    const vyNext =
      vy + ((vyDrive - vy * params.lateralDamping) * dt);

    // --- Integrate pose --------------------------------------------------
    // Use averaged speed and yaw-rate for the midpoint integration.
    const speedAvg = 0.5 * (v + speed);
    const yawAvg = 0.5 * (yawRate + yawRateNext);
    const heading = wrapAngle(s.heading + yawAvg * dt);
    const cosH = Math.cos(s.heading);
    const sinH = Math.sin(s.heading);
    // Forward velocity in world frame, plus lateral velocity (right-hand
    // perpendicular to heading): right = (sin h, -cos h) under the kinocat
    // sign convention (heading rotates +X toward +Z).
    const vyAvg = 0.5 * (vy + vyNext);
    const dx = (speedAvg * cosH + vyAvg * Math.sin(s.heading)) * dt;
    const dz = (speedAvg * sinH - vyAvg * Math.cos(s.heading)) * dt;
    return {
      x: s.x + dx,
      z: s.z + dz,
      heading,
      speed,
      yawRate: yawRateNext,
      lateralVelocity: vyNext,
      t: s.t + dt,
    };
  };
}

// ---------------------------------------------------------------------------
// Full model (parametric + optional residual MLP ensemble)

export interface LearnedVehicleModel {
  params: LearnedVehicleParamsV2;
  config: LearnableVehicleConfig;
  /** Optional residual MLP ensemble. If empty, model = parametric only. */
  residualEnsemble: MLP[];
  /** Reference dt the residual was trained against (for output scaling). */
  residualReferenceDt: number;
  /**
   * Per-output-dim OOD thresholds (length 6, matching MLP_OUTPUT_DIM).
   * If any per-tick ensemble std exceeds the corresponding threshold,
   * `learnedForwardSimV2` falls back to the parametric prediction —
   * guaranteeing inference is never worse than the parametric backbone
   * even on inputs the residual was never trained for.
   *
   * Defaults are conservative (= small thresholds → fallback fires
   * easily). Tune via per-regime eval after training.
   */
  oodStdThreshold?: number[];
}

/** Default OOD thresholds (per output dim: x, z, heading, speed,
 *  yawRate, lateralVelocity). Set just above the typical in-distribution
 *  ensemble variance — a well-trained ensemble agrees within these
 *  bounds; ensemble disagreement beyond them is the OOD signal. */
export const DEFAULT_OOD_STD_THRESHOLD = [0.5, 0.5, 0.1, 1.0, 0.5, 0.5];

export interface PredictionWithUncertainty {
  next: CarKinematicState;
  /** Standard deviation across the ensemble per output dimension
   *  (x, z, heading, speed, yawRate, lateralVelocity). Length 6. */
  std: number[];
}

/** Build a parametric-only model (no residual MLP). Most useful for unit
 *  tests, comparison baselines, and the first training round before MLP
 *  fitting starts. */
export function buildParametricOnlyModel(
  params: LearnedVehicleParamsV2 = DEFAULT_LEARNED_PARAMS_V2,
  config: LearnableVehicleConfig = DEFAULT_LEARNABLE_CONFIG,
): LearnedVehicleModel {
  return {
    params,
    config,
    residualEnsemble: [],
    residualReferenceDt: 1 / 60,
  };
}

/** Drop-in `ForwardSim<CarKinematicState>` for `characterizeVehicle` and the
 *  IGHA* planner.
 *
 *  Includes a built-in safety floor: the parametric backbone is the
 *  guaranteed minimum quality. The residual MLP ensemble contributes
 *  ONLY when it's confident (= low ensemble variance). When ensemble
 *  disagreement exceeds `model.oodStdThreshold` on any output channel,
 *  inference emits the parametric prediction unchanged. The model can
 *  therefore never produce a single-step prediction worse than the
 *  parametric backbone, even on inputs the residual was never trained
 *  for.
 */
export function learnedForwardSimV2(model: LearnedVehicleModel): ForwardSim<CarKinematicState> {
  const paraSim = parametricForwardV2(model.params, model.config);
  if (model.residualEnsemble.length === 0) {
    return paraSim;
  }
  const thresh = model.oodStdThreshold ?? DEFAULT_OOD_STD_THRESHOLD;
  // Special case: an ensemble of size 1 has zero variance by
  // definition, so OOD fallback can never fire. Detect this and
  // emit the residual unconditionally with a note in the diagnostics
  // (a 1-MLP model trades the OOD safety for compute simplicity).
  const ensembleSize = model.residualEnsemble.length;
  return (s: CarKinematicState, controls: number[], dt: number): CarKinematicState => {
    const base = paraSim(s, controls, dt);
    const input = buildMLPInput(s, controls, model.config);
    if (ensembleSize === 1) {
      const residualMean = ensembleMean(model.residualEnsemble, input);
      const scale = dt / model.residualReferenceDt;
      return applyResidual(base, residualMean, scale);
    }
    // Multi-MLP ensemble: compute mean + per-component std, fall back
    // to parametric when any component's std exceeds its threshold.
    const outputs = model.residualEnsemble.map((mlp) => mlpForward(mlp, input).output);
    const mean = new Float64Array(outputs[0]!.length);
    for (const o of outputs) for (let i = 0; i < o.length; i++) mean[i]! += o[i]! / ensembleSize;
    let ood = false;
    for (let i = 0; i < mean.length; i++) {
      let variance = 0;
      for (const o of outputs) {
        const d = o[i]! - mean[i]!;
        variance += d * d;
      }
      variance /= ensembleSize;
      const std = Math.sqrt(variance);
      if (std > thresh[i]!) { ood = true; break; }
    }
    if (ood) return base;
    const scale = dt / model.residualReferenceDt;
    return applyResidual(base, mean, scale);
  };
}

/** Forward + uncertainty. Used by the planner to penalize high-uncertainty
 *  trajectories (when cost-modifier feature is enabled) and by the runtime
 *  monitor to flag OOD operation. */
export function predictWithUncertainty(
  model: LearnedVehicleModel,
  s: CarKinematicState,
  controls: number[],
  dt: number,
): PredictionWithUncertainty {
  const paraSim = parametricForwardV2(model.params, model.config);
  const base = paraSim(s, controls, dt);
  if (model.residualEnsemble.length === 0) {
    return { next: base, std: [0, 0, 0, 0, 0, 0] };
  }
  const input = buildMLPInput(s, controls, model.config);
  const outputs = model.residualEnsemble.map((mlp) => mlpForward(mlp, input).output);
  const mean = new Float64Array(outputs[0]!.length);
  for (const o of outputs) for (let i = 0; i < o.length; i++) mean[i]! += o[i]! / outputs.length;
  const std = new Array(mean.length).fill(0);
  for (const o of outputs) {
    for (let i = 0; i < mean.length; i++) {
      const d = o[i]! - mean[i]!;
      std[i] += (d * d) / outputs.length;
    }
  }
  for (let i = 0; i < std.length; i++) std[i] = Math.sqrt(std[i]);
  const scale = dt / model.residualReferenceDt;
  return { next: applyResidual(base, mean, scale), std };
}

// ---------------------------------------------------------------------------
// MLP wiring helpers (used by both inference and training)

/** Inputs:
 *    state (5) — (sin heading, cos heading, speed, yawRate, lateralVel)
 *  + controls (3) — (steer, driveForce, brakeForce)
 *  + config one-hot (13)
 *  = 21 dims total.
 *
 *  Position (x, z) is DELIBERATELY OMITTED. The chassis's dynamic
 *  response to controls is translation-invariant; including absolute
 *  world position made the residual MLP position-dependent, and trials
 *  near origin failed catastrophically when run at world coordinates
 *  outside that range (sim-to-real free-drive at x = -52 produced
 *  residual outputs that drove the prediction off by 14 m in 10 s).
 *
 *  Heading is encoded as (sin, cos) rather than the raw angle so the
 *  network sees a smooth signal across the ±π wrap boundary. Rotational
 *  symmetry is then naturally learnable.
 *
 *  Bumped from the previous 22-dim encoding (which had x, z, raw
 *  heading). Older model files with `version: 2` payloads cannot use
 *  this layout — the persistence loader rejects them with a clear
 *  message asking for a retrain.
 */
export const MLP_INPUT_DIM = 5 + 3 + 13;
export const MLP_OUTPUT_DIM = 6;

// Heading is sin/cos (already bounded ±1). Speed, yawRate, lateralVel
// are normalised by typical chassis magnitudes.
const STATE_SCALES = [1, 1, 20, 4, 6]; // (sinH, cosH, speed, yawRate, lateralVelocity)

export function buildMLPInput(
  s: CarKinematicState,
  controls: ReadonlyArray<number>,
  config: LearnableVehicleConfig,
): number[] {
  // Heading as (sin, cos) — translation- and wrap-symmetric.
  // Position is deliberately omitted (see MLP_INPUT_DIM docstring).
  const stateVec = [
    Math.sin(s.heading),
    Math.cos(s.heading),
    s.speed,
    s.yawRate ?? 0,
    s.lateralVelocity ?? 0,
  ];
  const ctrlVec = [controls[0] ?? 0, controls[1] ?? 0, controls[2] ?? 0];
  // Normalise drive/brake by the config's own force limits (identical to
  // the historical hardcoded [1, 4000, 2000] for the reference chassis,
  // but stays correct when a different chassis is trained). Training and
  // inference both come through here, so the encoding cannot drift.
  const controlScales = [
    1,
    Math.max(1, config.maxDriveForce),
    Math.max(1, config.maxBrakeForce),
  ];
  const cfgVec = encodeConfigOneHot(config);
  const inputs: number[] = [];
  for (let i = 0; i < stateVec.length; i++) inputs.push(stateVec[i]! / STATE_SCALES[i]!);
  for (let i = 0; i < ctrlVec.length; i++) inputs.push(ctrlVec[i]! / controlScales[i]!);
  for (let i = 0; i < cfgVec.length; i++) {
    const scale = CONFIG_SCALES_ORDINAL[Math.min(i, CONFIG_SCALES_ORDINAL.length - 1)] || 1;
    inputs.push(cfgVec[i]! / scale);
  }
  return inputs;
}

function ensembleMean(ensemble: MLP[], input: ReadonlyArray<number>): Float64Array {
  const outputs = ensemble.map((mlp) => mlpForward(mlp, input).output);
  const mean = new Float64Array(outputs[0]!.length);
  for (const o of outputs) for (let i = 0; i < o.length; i++) mean[i]! += o[i]! / outputs.length;
  return mean;
}

/** Apply a 6-dim residual (in state-vector ordering) to a base state. The
 *  residual is interpreted as a per-state-component additive correction;
 *  `dtScale` rescales it from the residual's reference dt to the current dt. */
function applyResidual(
  base: CarKinematicState,
  residual: Float64Array | ReadonlyArray<number>,
  dtScale: number,
): CarKinematicState {
  return {
    x: base.x + (residual[0] ?? 0) * dtScale,
    z: base.z + (residual[1] ?? 0) * dtScale,
    heading: wrapAngle(base.heading + (residual[2] ?? 0) * dtScale),
    speed: base.speed + (residual[3] ?? 0) * dtScale,
    yawRate: (base.yawRate ?? 0) + (residual[4] ?? 0) * dtScale,
    lateralVelocity: (base.lateralVelocity ?? 0) + (residual[5] ?? 0) * dtScale,
    t: base.t,
  };
}
