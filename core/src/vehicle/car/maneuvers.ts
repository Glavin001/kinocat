// Car maneuver library — Phase 1 of the training-dataset plan.
//
// Each factory returns a `Driver<CarKinematicState, WheeledCarControls>`
// that produces a time-varying controls trace, replacing the prior
// constant-hold seed grid. The training driver draws maneuvers from this
// library under a budget weighting (60 % OU random-walk, 15 % transition
// probes, 10 % saturation / panic, 10 % named identification, 5 % legacy
// constant-hold) so the trial distribution matches the planner's deployment
// query distribution and the lift-off / mid-corner regimes the prior
// dataset never visited become covered.

import type { Driver } from '../../scene/driver';
import type { CarKinematicState } from './types';
import type { WheeledCarControls } from './types';

const ZERO: WheeledCarControls = { steer: 0, driveForce: 0, brakeForce: 0 };

// ---------------------------------------------------------------------------
// Tiny seeded PRNG. Inline so we don't pull in a dep just for this; the
// only place it's used is the OU random-walk family. Mulberry32.

export function seededRng(seed: number): () => number {
  let s = (seed | 0) >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal via Box-Muller from a uniform RNG. */
function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

// ---------------------------------------------------------------------------
// Chassis limits — every maneuver factory accepts these so it can clip the
// emitted commands to physically realistic ranges.

export interface ManeuverLimits {
  maxSteerAngle: number;
  maxDriveForce: number;
  maxBrakeForce: number;
}

// ---------------------------------------------------------------------------
// Class 1 — Random-walk (Ornstein-Uhlenbeck) controls.
//
// Independent OU processes per channel, mean-reverting toward zero with
// timescale `tau`. Sigma is the per-channel diffusion. Clipped to chassis
// limits. Smooth, non-pathological, deeply transition-rich. The single class
// that does the most heavy lifting in this library.

export interface OuParams {
  /** Diffusion std-dev (rad / N / N per second) per channel. */
  sigmaSteer: number;
  sigmaDrive: number;
  sigmaBrake: number;
  /** Mean-reversion timescale (s). Larger = slower. */
  tau: number;
}

export function ouControls(opts: {
  params: OuParams;
  limits: ManeuverLimits;
  rng: () => number;
  /** Starting controls (default zero). */
  initial?: WheeledCarControls;
}): Driver<CarKinematicState, WheeledCarControls> {
  let last: WheeledCarControls = { ...(opts.initial ?? ZERO) };
  let lastSimTime: number | null = null;
  const { params, limits, rng } = opts;
  return {
    sample(_state, simTime, dt) {
      const stepDt = lastSimTime === null ? dt : Math.max(dt, simTime - lastSimTime);
      lastSimTime = simTime;
      const decay = Math.exp(-stepDt / Math.max(1e-3, params.tau));
      const diff = Math.sqrt(Math.max(0, 1 - decay * decay));
      const nextSteer = clamp(
        last.steer * decay + diff * params.sigmaSteer * gaussian(rng),
        -limits.maxSteerAngle, limits.maxSteerAngle,
      );
      const nextDrive = clamp(
        last.driveForce * decay + diff * params.sigmaDrive * gaussian(rng),
        -limits.maxDriveForce, limits.maxDriveForce,
      );
      // Brake is non-negative; reflect at 0.
      const brakeUpdate = last.brakeForce * decay + diff * params.sigmaBrake * gaussian(rng);
      const nextBrake = clamp(Math.abs(brakeUpdate), 0, limits.maxBrakeForce);
      last = { steer: nextSteer, driveForce: nextDrive, brakeForce: nextBrake };
      return last;
    },
    reset() {
      last = { ...(opts.initial ?? ZERO) };
      lastSimTime = null;
    },
  };
}

/** Mixture random-walk: switches OU parameters between named driving "modes"
 *  (cruise, attack, lift-off, brake-zone) at a fixed rate. One trial then
 *  contains multiple regime crossings — the kind of diversity the
 *  constant-hold grid entirely lacks. */
export interface MixtureMode extends OuParams {
  name: string;
  /** Bias added per tick — non-zero values pull the OU toward a regime.
   *  Defaults to all-zero (mean-reverting toward 0). */
  driveBias?: number;
  brakeBias?: number;
}

export function mixtureRandomWalk(opts: {
  modes: MixtureMode[];
  modeSwitchHz: number;
  limits: ManeuverLimits;
  rng: () => number;
}): Driver<CarKinematicState, WheeledCarControls> {
  if (opts.modes.length === 0) {
    return ouControls({
      params: { sigmaSteer: 0, sigmaDrive: 0, sigmaBrake: 0, tau: 1 },
      limits: opts.limits, rng: opts.rng,
    });
  }
  let last: WheeledCarControls = { ...ZERO };
  let lastSimTime: number | null = null;
  let modeIdx = Math.floor(opts.rng() * opts.modes.length);
  let nextSwitch = 1 / Math.max(0.01, opts.modeSwitchHz);
  return {
    sample(_state, simTime, dt) {
      const stepDt = lastSimTime === null ? dt : Math.max(dt, simTime - lastSimTime);
      lastSimTime = simTime;
      if (simTime >= nextSwitch) {
        modeIdx = Math.floor(opts.rng() * opts.modes.length) % opts.modes.length;
        nextSwitch = simTime + 1 / Math.max(0.01, opts.modeSwitchHz);
      }
      const m = opts.modes[modeIdx]!;
      const decay = Math.exp(-stepDt / Math.max(1e-3, m.tau));
      const diff = Math.sqrt(Math.max(0, 1 - decay * decay));
      const nextSteer = clamp(
        last.steer * decay + diff * m.sigmaSteer * gaussian(opts.rng),
        -opts.limits.maxSteerAngle, opts.limits.maxSteerAngle,
      );
      const nextDrive = clamp(
        last.driveForce * decay + diff * m.sigmaDrive * gaussian(opts.rng) + (m.driveBias ?? 0) * stepDt,
        -opts.limits.maxDriveForce, opts.limits.maxDriveForce,
      );
      const brakeUpdate = last.brakeForce * decay + diff * m.sigmaBrake * gaussian(opts.rng) + (m.brakeBias ?? 0) * stepDt;
      const nextBrake = clamp(Math.abs(brakeUpdate), 0, opts.limits.maxBrakeForce);
      last = { steer: nextSteer, driveForce: nextDrive, brakeForce: nextBrake };
      return last;
    },
    reset() {
      last = { ...ZERO };
      lastSimTime = null;
      modeIdx = 0;
      nextSwitch = 1 / Math.max(0.01, opts.modeSwitchHz);
    },
  };
}

// ---------------------------------------------------------------------------
// Class 2 — Transition probes. One transition per trial (typically at t=2s
// out of a 4s trial); the transition itself is the training signal the
// constant-hold dataset entirely lacks.

function constantSegment(c: WheeledCarControls): Driver<CarKinematicState, WheeledCarControls> {
  return { sample: () => c };
}

/** Generic two-phase driver: emits `before` until `transitionAt`, then `after`. */
function twoPhase(
  before: WheeledCarControls,
  after: WheeledCarControls,
  transitionAt: number,
): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      return local < transitionAt ? before : after;
    },
    reset() {
      t0 = null;
    },
  };
}

export function throttleRelease(opts: {
  throttle: number;
  steer: number;
  transitionAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steer, driveForce: opts.throttle, brakeForce: 0 },
    { steer: opts.steer, driveForce: 0, brakeForce: 0 },
    opts.transitionAt,
  );
}

export function throttleToBrake(opts: {
  throttle: number;
  brake: number;
  steer: number;
  transitionAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steer, driveForce: opts.throttle, brakeForce: 0 },
    { steer: opts.steer, driveForce: 0, brakeForce: opts.brake },
    opts.transitionAt,
  );
}

export function brakeToThrottle(opts: {
  brake: number;
  throttle: number;
  steer: number;
  transitionAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steer, driveForce: 0, brakeForce: opts.brake },
    { steer: opts.steer, driveForce: opts.throttle, brakeForce: 0 },
    opts.transitionAt,
  );
}

export function steerToZero(opts: {
  steer: number;
  drive: number;
  transitionAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steer, driveForce: opts.drive, brakeForce: 0 },
    { steer: 0, driveForce: opts.drive, brakeForce: 0 },
    opts.transitionAt,
  );
}

export function steerReversal(opts: {
  steerLeft: number;
  steerRight: number;
  drive: number;
  transitionAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steerLeft, driveForce: opts.drive, brakeForce: 0 },
    { steer: opts.steerRight, driveForce: opts.drive, brakeForce: 0 },
    opts.transitionAt,
  );
}

// ---------------------------------------------------------------------------
// Class 3 — Saturation / panic probes. Friction-circle-saturated regime
// under recovery, lift-off oversteer, reverse + steer. These exist in
// deployment (panic stops, evasive maneuvers) but never in the constant-
// hold grid.

export function panicTurn(opts: {
  limits: ManeuverLimits;
  steer: number;
  turnDuration: number;
  brakeRecovery: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: opts.steer, driveForce: 0, brakeForce: 0 },
    { steer: opts.steer * 0.3, driveForce: 0, brakeForce: opts.brakeRecovery },
    opts.turnDuration,
  );
}

export function liftOffOversteer(opts: {
  driveForce: number;
  steer: number;
  liftAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: 0, driveForce: opts.driveForce, brakeForce: 0 },
    { steer: opts.steer, driveForce: 0, brakeForce: 0 },
    opts.liftAt,
  );
}

export function reverseWithSteer(opts: {
  reverseDrive: number; // negative
  steer: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return constantSegment({ steer: opts.steer, driveForce: opts.reverseDrive, brakeForce: 0 });
}

// ---------------------------------------------------------------------------
// Class 4 — Named identification maneuvers (system-ID textbook set).

export function stepSteer(opts: {
  amplitude: number;
  driveForce: number;
  stepAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return twoPhase(
    { steer: 0, driveForce: opts.driveForce, brakeForce: 0 },
    { steer: opts.amplitude, driveForce: opts.driveForce, brakeForce: 0 },
    opts.stepAt,
  );
}

export function sinSweepSteer(opts: {
  amplitude: number;
  startFreqHz: number;
  endFreqHz: number;
  duration: number;
  driveForce: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = clamp(simTime - t0, 0, opts.duration);
      const u = local / opts.duration;
      const f = opts.startFreqHz + (opts.endFreqHz - opts.startFreqHz) * u;
      const phase = 2 * Math.PI * f * local;
      return {
        steer: opts.amplitude * Math.sin(phase),
        driveForce: opts.driveForce,
        brakeForce: 0,
      };
    },
    reset() { t0 = null; },
  };
}

export function slalom(opts: {
  amplitude: number;
  periodSec: number;
  driveForce: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      return {
        steer: opts.amplitude * Math.sin((2 * Math.PI * local) / Math.max(1e-3, opts.periodSec)),
        driveForce: opts.driveForce,
        brakeForce: 0,
      };
    },
    reset() { t0 = null; },
  };
}

export function trailBrake(opts: {
  brakeForce: number;
  releaseTime: number;
  steerRamp: number;
  steerHold: number;
  totalDuration: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = clamp(simTime - t0, 0, opts.totalDuration);
      const steer = clamp(opts.steerRamp * local, -opts.steerHold, opts.steerHold);
      const brake = local < opts.releaseTime ? opts.brakeForce * (1 - local / opts.releaseTime) : 0;
      return { steer, driveForce: 0, brakeForce: Math.max(0, brake) };
    },
    reset() { t0 = null; },
  };
}

export function throttleOnApex(opts: {
  initialBrake: number;
  brakeReleaseAt: number;
  throttleRamp: number;
  steer: number;
  maxDrive: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      if (local < opts.brakeReleaseAt) {
        return { steer: opts.steer, driveForce: 0, brakeForce: opts.initialBrake };
      }
      const drive = clamp(opts.throttleRamp * (local - opts.brakeReleaseAt), 0, opts.maxDrive);
      return { steer: opts.steer, driveForce: drive, brakeForce: 0 };
    },
    reset() { t0 = null; },
  };
}

export function jTurn(opts: {
  steerStep: number;
  driveForce: number;
  stepAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return stepSteer({
    amplitude: opts.steerStep,
    driveForce: opts.driveForce,
    stepAt: opts.stepAt,
  });
}

export function fishhook(opts: {
  steerAmp: number;
  driveForce: number;
  firstAt: number;
  reverseAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      if (local < opts.firstAt) return { steer: 0, driveForce: opts.driveForce, brakeForce: 0 };
      if (local < opts.reverseAt) return { steer: opts.steerAmp, driveForce: opts.driveForce, brakeForce: 0 };
      return { steer: -opts.steerAmp, driveForce: opts.driveForce, brakeForce: 0 };
    },
    reset() { t0 = null; },
  };
}

export function scandinavianFlick(opts: {
  counterSteer: number;
  steer: number;
  driveForce: number;
  counterAt: number;
  flickAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      if (local < opts.counterAt) return { steer: 0, driveForce: opts.driveForce, brakeForce: 0 };
      if (local < opts.flickAt) return { steer: opts.counterSteer, driveForce: opts.driveForce, brakeForce: 0 };
      return { steer: opts.steer, driveForce: opts.driveForce, brakeForce: 0 };
    },
    reset() { t0 = null; },
  };
}

export function doubleLaneChange(opts: {
  steerAmp: number;
  driveForce: number;
  firstAt: number;
  reverseAt: number;
  returnAt: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample(_s, simTime, _dt) {
      if (t0 === null) t0 = simTime;
      const local = simTime - t0;
      let steer = 0;
      if (local < opts.firstAt) steer = 0;
      else if (local < opts.reverseAt) steer = opts.steerAmp;
      else if (local < opts.returnAt) steer = -opts.steerAmp;
      else steer = 0;
      return { steer, driveForce: opts.driveForce, brakeForce: 0 };
    },
    reset() { t0 = null; },
  };
}

export function donut(opts: {
  steer: number;
  driveForce: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return constantSegment({ steer: opts.steer, driveForce: opts.driveForce, brakeForce: 0 });
}

// ---------------------------------------------------------------------------
// Trial-spec metadata factory. Pairs each Driver above with a stable
// `maneuverId` + `maneuverParams` record so the hash-based split policy +
// coverage meter index them consistently.

export interface ManeuverSpec {
  id: string;
  params: Record<string, number>;
  build: (limits: ManeuverLimits, seed: number) => Driver<CarKinematicState, WheeledCarControls>;
}

/** Build a budget-weighted bundle of maneuver specs. Each call returns
 *  deterministic outputs for the same seed. Useful as the default
 *  recipe the training driver consumes. */
export function defaultManeuverBundle(args: {
  limits: ManeuverLimits;
  count: number;
  seed?: number;
}): ManeuverSpec[] {
  const rng = seededRng(args.seed ?? 1);
  const { limits } = args;
  const specs: ManeuverSpec[] = [];

  // 60 % OU random walk — three sub-flavors so per-spec params have variety.
  const ouCount = Math.round(args.count * 0.60);
  for (let i = 0; i < ouCount; i++) {
    const sigSteer = 0.10 + 0.30 * rng();
    const sigDrive = 0.20 * limits.maxDriveForce + 0.50 * limits.maxDriveForce * rng();
    const sigBrake = 0.15 * limits.maxBrakeForce + 0.50 * limits.maxBrakeForce * rng();
    const tau = 0.18 + 0.45 * rng();
    const subSeed = (args.seed ?? 1) * 9301 + i;
    specs.push({
      id: 'ou',
      params: { sigSteer, sigDrive, sigBrake, tau, idx: i },
      build: (lim, _) => ouControls({
        params: { sigmaSteer: sigSteer, sigmaDrive: sigDrive, sigmaBrake: sigBrake, tau },
        limits: lim,
        rng: seededRng(subSeed),
      }),
    });
  }

  // 15 % transition probes.
  const transCount = Math.round(args.count * 0.15);
  for (let i = 0; i < transCount; i++) {
    const kind = i % 5;
    const drv = limits.maxDriveForce * (0.5 + 0.5 * rng());
    const brk = limits.maxBrakeForce * (0.5 + 0.5 * rng());
    const st = limits.maxSteerAngle * (rng() - 0.5);
    const stOther = limits.maxSteerAngle * (rng() - 0.5);
    if (kind === 0) specs.push({
      id: 'throttleRelease', params: { drv, st, idx: i },
      build: () => throttleRelease({ throttle: drv, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 1) specs.push({
      id: 'throttleToBrake', params: { drv, brk, st, idx: i },
      build: () => throttleToBrake({ throttle: drv, brake: brk, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 2) specs.push({
      id: 'brakeToThrottle', params: { brk, drv, st, idx: i },
      build: () => brakeToThrottle({ brake: brk, throttle: drv, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 3) specs.push({
      id: 'steerToZero', params: { st, drv, idx: i },
      build: () => steerToZero({ steer: st, drive: drv, transitionAt: 1.0 }),
    });
    else specs.push({
      id: 'steerReversal', params: { steerLeft: st, steerRight: stOther, drv, idx: i },
      build: () => steerReversal({ steerLeft: st, steerRight: stOther, drive: drv, transitionAt: 1.0 }),
    });
  }

  // 10 % saturation / panic probes.
  const satCount = Math.round(args.count * 0.10);
  for (let i = 0; i < satCount; i++) {
    const kind = i % 3;
    const st = limits.maxSteerAngle * (i % 2 === 0 ? 1 : -1) * (0.7 + 0.3 * rng());
    if (kind === 0) specs.push({
      id: 'panicTurn', params: { st, idx: i },
      build: (lim) => panicTurn({
        limits: lim, steer: st, turnDuration: 0.5, brakeRecovery: lim.maxBrakeForce,
      }),
    });
    else if (kind === 1) specs.push({
      id: 'liftOffOversteer', params: { st, drv: limits.maxDriveForce, idx: i },
      build: () => liftOffOversteer({
        driveForce: limits.maxDriveForce * 0.9, steer: st, liftAt: 1.2,
      }),
    });
    else specs.push({
      id: 'reverseWithSteer', params: { st, idx: i },
      build: () => reverseWithSteer({ reverseDrive: -0.5 * limits.maxDriveForce, steer: st }),
    });
  }

  // 10 % named identification maneuvers.
  const identCount = Math.round(args.count * 0.10);
  for (let i = 0; i < identCount; i++) {
    const kind = i % 7;
    const amp = limits.maxSteerAngle * (0.3 + 0.5 * rng());
    const drv = limits.maxDriveForce * (0.4 + 0.4 * rng());
    if (kind === 0) specs.push({
      id: 'stepSteer', params: { amp, drv, idx: i },
      build: () => stepSteer({ amplitude: amp, driveForce: drv, stepAt: 0.5 }),
    });
    else if (kind === 1) specs.push({
      id: 'sinSweepSteer', params: { amp, drv, idx: i },
      build: () => sinSweepSteer({
        amplitude: amp, startFreqHz: 0.2, endFreqHz: 1.5, duration: 3.0, driveForce: drv,
      }),
    });
    else if (kind === 2) specs.push({
      id: 'slalom', params: { amp, drv, idx: i },
      build: () => slalom({ amplitude: amp, periodSec: 1.0 + rng(), driveForce: drv }),
    });
    else if (kind === 3) specs.push({
      id: 'trailBrake', params: { amp, idx: i },
      build: (lim) => trailBrake({
        brakeForce: lim.maxBrakeForce * 0.8, releaseTime: 0.6, steerRamp: amp * 1.5,
        steerHold: amp, totalDuration: 2.0,
      }),
    });
    else if (kind === 4) specs.push({
      id: 'throttleOnApex', params: { amp, drv, idx: i },
      build: (lim) => throttleOnApex({
        initialBrake: lim.maxBrakeForce * 0.5, brakeReleaseAt: 0.7,
        throttleRamp: drv * 1.5, steer: amp, maxDrive: drv,
      }),
    });
    else if (kind === 5) specs.push({
      id: 'jTurn', params: { amp, drv, idx: i },
      build: () => jTurn({ steerStep: amp, driveForce: drv, stepAt: 0.5 }),
    });
    else specs.push({
      id: 'fishhook', params: { amp, drv, idx: i },
      build: () => fishhook({ steerAmp: amp, driveForce: drv, firstAt: 0.5, reverseAt: 1.2 }),
    });
  }

  // 5 % constant-hold baseline (legacy ablation).
  const holdCount = Math.max(0, args.count - specs.length);
  for (let i = 0; i < holdCount; i++) {
    const st = limits.maxSteerAngle * (rng() - 0.5);
    const drv = limits.maxDriveForce * rng();
    const brk = i % 3 === 0 ? limits.maxBrakeForce * rng() : 0;
    specs.push({
      id: 'constantHold', params: { st, drv, brk, idx: i },
      build: () => constantSegment({ steer: st, driveForce: drv, brakeForce: brk }),
    });
  }

  return specs;
}

// ---------------------------------------------------------------------------
// Section 2: passive / multi-cusp helpers + universalManeuverBundle.
//
// The default bundle above covers OU random-walks + named maneuvers but
// under-covers the regimes the user's sim-to-real free-drive trace
// exposed: passive coast (controls ≈ 0 at varying speeds), reverse from
// rest, multi-cusp parking sequences, and stuck states. The universal
// bundle adds these so a single trained model can drive the full
// spectrum of vehicle activities (race, park, reverse, recovery, edge
// cases) without out-of-distribution failures.

/** Passive coast: all controls = 0 for the full trial. The chassis
 *  decays under rolling resistance + friction. The most important
 *  "what does my model do when nothing is happening?" trial. */
export function passiveCoast(): Driver<CarKinematicState, WheeledCarControls> {
  return constantSegment({ steer: 0, driveForce: 0, brakeForce: 0 });
}

/** Full drive from rest with no steer — wheelspin / acceleration probe. */
export function wheelspin(opts: {
  driveForce: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return constantSegment({ steer: 0, driveForce: opts.driveForce, brakeForce: 0 });
}

/** Stuck state: full drive AND full brake. The chassis shouldn't move
 *  much. Important to teach the model "brake dominates drive at rest." */
export function stuckState(opts: {
  driveForce: number;
  brakeForce: number;
}): Driver<CarKinematicState, WheeledCarControls> {
  return constantSegment({
    steer: 0,
    driveForce: opts.driveForce,
    brakeForce: opts.brakeForce,
  });
}

/** Scripted multi-cusp parking sequence: forward → stop → reverse →
 *  stop → forward. Each phase has its own controls + duration. The
 *  driver emits the right phase for the elapsed sim time. */
export function multiCuspParkingScript(opts: {
  phases: Array<{ controls: WheeledCarControls; duration: number }>;
}): Driver<CarKinematicState, WheeledCarControls> {
  let t0: number | null = null;
  return {
    sample: (s) => {
      if (t0 === null) t0 = s.t;
      const elapsed = s.t - t0;
      let acc = 0;
      for (const phase of opts.phases) {
        acc += phase.duration;
        if (elapsed < acc) return phase.controls;
      }
      // Past the last phase — emit final phase's controls (typically a
      // hold at goal).
      return opts.phases[opts.phases.length - 1]?.controls ?? {
        steer: 0, driveForce: 0, brakeForce: 0,
      };
    },
  };
}

/** Three-point turn script: drive forward and right, stop, reverse and
 *  left, stop, drive forward. Single canonical sequence used for
 *  training coverage of cusp-heavy parking maneuvers. */
export function threePointTurnScript(args: {
  limits: ManeuverLimits;
}): Driver<CarKinematicState, WheeledCarControls> {
  const drv = args.limits.maxDriveForce;
  const brk = args.limits.maxBrakeForce;
  const st = args.limits.maxSteerAngle;
  return multiCuspParkingScript({
    phases: [
      { controls: { steer: +st * 0.7, driveForce: 0.5 * drv, brakeForce: 0 }, duration: 1.5 },
      { controls: { steer: 0, driveForce: 0, brakeForce: brk }, duration: 0.5 },
      { controls: { steer: -st * 0.7, driveForce: -0.5 * drv, brakeForce: 0 }, duration: 1.5 },
      { controls: { steer: 0, driveForce: 0, brakeForce: brk }, duration: 0.5 },
      { controls: { steer: 0, driveForce: 0.4 * drv, brakeForce: 0 }, duration: 1.0 },
    ],
  });
}

/**
 * Universal coverage bundle. Augments `defaultManeuverBundle` with
 * passive coast / reverse / multi-cusp / stuck-state trials at varied
 * intensities and durations.
 *
 * Use this bundle when training a model intended for ALL vehicle
 * activities (race + park + reverse + recovery), not just dynamics
 * identification. Recommended trial count: 1000-2000 per round for
 * adequate stratification across the 45-regime coverage matrix.
 *
 * Composition (over `count` total specs):
 *   - 35 %  OU random-walk (broad exploration)
 *   - 15 %  passive coast at varied starting speeds (FIXES the
 *           sim-to-real free-drive failure)
 *   - 10 %  named identification maneuvers (slalom, fishhook, jTurn,
 *           trailBrake, throttleOnApex, doubleLaneChange, etc.)
 *   - 10 %  reverse + parking-style (reverseWithSteer, multi-cusp,
 *           three-point turn)
 *   - 10 %  saturation / instability (panicTurn, liftOffOversteer,
 *           wheelspin, stuckState)
 *   - 10 %  transitions (throttleRelease, throttleToBrake, etc.)
 *   -  5 %  steady cruise at varied speeds (lets the residual learn
 *           "small" deltas in the dominant operating regime)
 *   -  5 %  constant-hold (legacy ablation baseline)
 */
export function universalManeuverBundle(args: {
  limits: ManeuverLimits;
  count: number;
  seed?: number;
}): ManeuverSpec[] {
  const rng = seededRng(args.seed ?? 1);
  const { limits } = args;
  const specs: ManeuverSpec[] = [];

  // 35 % OU random-walk (broad exploration, varied noise scales).
  const ouCount = Math.round(args.count * 0.35);
  for (let i = 0; i < ouCount; i++) {
    const sigSteer = 0.08 + 0.35 * rng();
    const sigDrive = 0.15 * limits.maxDriveForce + 0.55 * limits.maxDriveForce * rng();
    const sigBrake = 0.10 * limits.maxBrakeForce + 0.50 * limits.maxBrakeForce * rng();
    const tau = 0.15 + 0.50 * rng();
    const subSeed = (args.seed ?? 1) * 9301 + i;
    specs.push({
      id: 'ou',
      params: { sigSteer, sigDrive, sigBrake, tau, idx: i },
      build: (lim) => ouControls({
        params: { sigmaSteer: sigSteer, sigmaDrive: sigDrive, sigmaBrake: sigBrake, tau },
        limits: lim,
        rng: seededRng(subSeed),
      }),
    });
  }

  // 15 % passive coast — the key missing regime that caused sim-to-real
  // free-drive to fail. Different sub-trials start the chassis at
  // different speeds (the training-driver's `cells` grid determines
  // start state, not the maneuver itself).
  const coastCount = Math.round(args.count * 0.15);
  for (let i = 0; i < coastCount; i++) {
    specs.push({
      id: 'passiveCoast',
      params: { idx: i },
      build: () => passiveCoast(),
    });
  }

  // 10 % named identification maneuvers (rotated through 7 kinds).
  const identCount = Math.round(args.count * 0.10);
  for (let i = 0; i < identCount; i++) {
    const kind = i % 7;
    const amp = limits.maxSteerAngle * (0.3 + 0.5 * rng());
    const drv = limits.maxDriveForce * (0.4 + 0.5 * rng());
    if (kind === 0) specs.push({
      id: 'slalom', params: { amp, drv, idx: i },
      build: () => slalom({ amplitude: amp, periodSec: 0.8 + 1.0 * rng(), driveForce: drv }),
    });
    else if (kind === 1) specs.push({
      id: 'fishhook', params: { amp, drv, idx: i },
      build: () => fishhook({ steerAmp: amp, driveForce: drv, firstAt: 0.5, reverseAt: 1.2 }),
    });
    else if (kind === 2) specs.push({
      id: 'jTurn', params: { amp, drv, idx: i },
      build: () => jTurn({ steerStep: amp, driveForce: drv, stepAt: 0.5 }),
    });
    else if (kind === 3) specs.push({
      id: 'trailBrake', params: { amp, idx: i },
      build: (lim) => trailBrake({
        brakeForce: lim.maxBrakeForce * (0.5 + 0.4 * rng()),
        releaseTime: 0.4 + 0.5 * rng(),
        steerRamp: amp * 1.5,
        steerHold: amp,
        totalDuration: 1.5 + 1.0 * rng(),
      }),
    });
    else if (kind === 4) specs.push({
      id: 'throttleOnApex', params: { amp, drv, idx: i },
      build: (lim) => throttleOnApex({
        initialBrake: lim.maxBrakeForce * 0.5,
        brakeReleaseAt: 0.7,
        throttleRamp: drv * 1.5,
        steer: amp,
        maxDrive: drv,
      }),
    });
    else if (kind === 5) specs.push({
      id: 'doubleLaneChange', params: { amp, drv, idx: i },
      build: () => doubleLaneChange({
        steerAmp: amp,
        driveForce: drv,
        firstAt: 0.6,
        reverseAt: 1.1,
        returnAt: 1.8,
      }),
    });
    else specs.push({
      id: 'scandinavianFlick', params: { amp, drv, idx: i },
      build: () => scandinavianFlick({
        counterSteer: amp * 0.7,
        steer: -amp,
        driveForce: drv,
        counterAt: 0.3,
        flickAt: 0.7,
      }),
    });
  }

  // 10 % reverse + multi-cusp parking sequences.
  const revCount = Math.round(args.count * 0.10);
  for (let i = 0; i < revCount; i++) {
    const kind = i % 4;
    const st = limits.maxSteerAngle * (rng() - 0.5);
    const drv = limits.maxDriveForce * (0.4 + 0.4 * rng());
    if (kind === 0) specs.push({
      id: 'reverseWithSteer', params: { st, idx: i },
      build: (lim) => reverseWithSteer({ reverseDrive: -drv, steer: st }),
    });
    else if (kind === 1) specs.push({
      id: 'reverseStraight', params: { idx: i },
      build: (lim) => constantSegment({ steer: 0, driveForce: -drv, brakeForce: 0 }),
    });
    else if (kind === 2) specs.push({
      id: 'threePointTurn', params: { idx: i },
      build: (lim) => threePointTurnScript({ limits: lim }),
    });
    else specs.push({
      id: 'cuspSequence', params: { idx: i },
      build: (lim) => multiCuspParkingScript({
        phases: [
          { controls: { steer: +st, driveForce: 0.4 * drv, brakeForce: 0 }, duration: 1.0 },
          { controls: { steer: 0, driveForce: 0, brakeForce: lim.maxBrakeForce }, duration: 0.5 },
          { controls: { steer: -st, driveForce: -0.4 * drv, brakeForce: 0 }, duration: 1.0 },
          { controls: { steer: 0, driveForce: 0, brakeForce: lim.maxBrakeForce }, duration: 0.5 },
        ],
      }),
    });
  }

  // 10 % saturation / instability probes (panic, lift-off, wheelspin, stuck).
  const satCount = Math.round(args.count * 0.10);
  for (let i = 0; i < satCount; i++) {
    const kind = i % 4;
    const st = limits.maxSteerAngle * (i % 2 === 0 ? 1 : -1) * (0.7 + 0.3 * rng());
    if (kind === 0) specs.push({
      id: 'panicTurn', params: { st, idx: i },
      build: (lim) => panicTurn({
        limits: lim, steer: st, turnDuration: 0.5, brakeRecovery: lim.maxBrakeForce,
      }),
    });
    else if (kind === 1) specs.push({
      id: 'liftOffOversteer', params: { st, drv: limits.maxDriveForce, idx: i },
      build: () => liftOffOversteer({
        driveForce: limits.maxDriveForce * 0.9, steer: st, liftAt: 1.2,
      }),
    });
    else if (kind === 2) specs.push({
      id: 'wheelspin', params: { idx: i },
      build: (lim) => wheelspin({ driveForce: lim.maxDriveForce }),
    });
    else specs.push({
      id: 'stuckState', params: { idx: i },
      build: (lim) => stuckState({
        driveForce: lim.maxDriveForce * 0.8,
        brakeForce: lim.maxBrakeForce,
      }),
    });
  }

  // 10 % transition probes.
  const transCount = Math.round(args.count * 0.10);
  for (let i = 0; i < transCount; i++) {
    const kind = i % 5;
    const drv = limits.maxDriveForce * (0.4 + 0.5 * rng());
    const brk = limits.maxBrakeForce * (0.5 + 0.5 * rng());
    const st = limits.maxSteerAngle * (rng() - 0.5);
    const stOther = limits.maxSteerAngle * (rng() - 0.5);
    if (kind === 0) specs.push({
      id: 'throttleRelease', params: { drv, st, idx: i },
      build: () => throttleRelease({ throttle: drv, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 1) specs.push({
      id: 'throttleToBrake', params: { drv, brk, st, idx: i },
      build: () => throttleToBrake({ throttle: drv, brake: brk, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 2) specs.push({
      id: 'brakeToThrottle', params: { brk, drv, st, idx: i },
      build: () => brakeToThrottle({ brake: brk, throttle: drv, steer: st, transitionAt: 1.0 }),
    });
    else if (kind === 3) specs.push({
      id: 'steerToZero', params: { st, drv, idx: i },
      build: () => steerToZero({ steer: st, drive: drv, transitionAt: 1.0 }),
    });
    else specs.push({
      id: 'steerReversal', params: { steerLeft: st, steerRight: stOther, drv, idx: i },
      build: () => steerReversal({ steerLeft: st, steerRight: stOther, drive: drv, transitionAt: 1.0 }),
    });
  }

  // 5 % steady cruise — small constant drive at various speeds. Combined
  // with the training-driver's start-speed cells, this gives the
  // residual MLP "what does steady-state cruise look like" examples.
  const cruiseCount = Math.round(args.count * 0.05);
  for (let i = 0; i < cruiseCount; i++) {
    const drv = limits.maxDriveForce * (0.1 + 0.3 * rng());
    specs.push({
      id: 'cruiseHold', params: { drv, idx: i },
      build: () => constantSegment({ steer: 0, driveForce: drv, brakeForce: 0 }),
    });
  }

  // Remaining 5 % constant-hold (legacy ablation baseline).
  const holdCount = Math.max(0, args.count - specs.length);
  for (let i = 0; i < holdCount; i++) {
    const st = limits.maxSteerAngle * (rng() - 0.5);
    const drv = limits.maxDriveForce * rng();
    const brk = i % 3 === 0 ? limits.maxBrakeForce * rng() : 0;
    specs.push({
      id: 'constantHold', params: { st, drv, brk, idx: i },
      build: () => constantSegment({ steer: st, driveForce: drv, brakeForce: brk }),
    });
  }

  return specs;
}
