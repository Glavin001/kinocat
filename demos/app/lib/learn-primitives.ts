// Autonomous motion-primitive learner. Headless orchestration (no React,
// no three.js). The learner drives a Rapier vehicle like a human player —
// only steer/throttle/brake commands, NO teleports or forced velocities
// during a trial. For each `(startSpeed, controls)` pair the car first
// brakes itself to a stop, then accelerates physically to the target speed
// via open-loop throttle, then applies the test controls for the primitive
// duration while sample poses are recorded. The five-coefficient parametric
// dynamics model (`LearnedVehicleParams`) is then least-squares fit to the
// recorded trajectories with Nelder-Mead and fed into the standard
// `characterizeVehicle()` to produce a planner-ready `MotionPrimitiveLibrary`.
//
// The only teleports in this module are at world setup (chassis spawned at
// origin) — once trials start, the vehicle is purely physically driven.
//
// Used by `/learnprimitives` (interactive) and `demos/test/learn-primitives.
// test.ts` (headless). No browser-only APIs anywhere in this module.

import RAPIER from '@dimforge/rapier3d-compat';
import type { VehicleAgent, VehicleState, LearnedVehicleParams } from 'kinocat/agent';
import {
  DEFAULT_LEARNED_PARAMS,
  kinematicForwardSim,
  learnedForwardSim,
} from 'kinocat/agent';
import {
  characterizeVehicle,
  MotionPrimitiveLibrary,
} from 'kinocat/primitives';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  type CarHandle,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import { CARCHASE_AGENT } from './carchase-scenarios';

// ---------------------------------------------------------------------------
// Defaults — match the car-chase tuning so the learned library can be dropped
// in as a replacement for CARCHASE_LIB without retuning.

/** Wheel-base half-distance from chassis centre. The Ackermann conversion
 *  needs `2 * WHEEL_BASE`. */
export const LEARN_WHEEL_BASE = 1.6;

export const LEARN_VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: LEARN_WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.3,
  suspensionMaxTravel: 0.2,
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd',
};

/** Primitive duration in seconds. Matches `carchase-scenarios.ts`. */
export const PRIMITIVE_DURATION = 0.55;
/** Sweep substeps recorded per primitive (`characterizeVehicle`'s `substeps`). */
export const PRIMITIVE_SUBSTEPS = 6;
/** Physics tick. */
export const PHYSICS_DT = 1 / 60;
/** Number of physics ticks per primitive (covers PRIMITIVE_DURATION). */
export const TICKS_PER_TRIAL = Math.round(PRIMITIVE_DURATION / PHYSICS_DT); // 33

/** Start-speed buckets to characterise from. */
export const DEFAULT_START_SPEEDS: number[] = [0, 4, 8, 12];

/** Default control set: same shape as the car-chase library so the learned
 *  result is a true drop-in replacement. */
export function defaultControlSets(agent: VehicleAgent = CARCHASE_AGENT): number[][] {
  const k = 1 / agent.minTurnRadius;
  const kHalf = k / 2;
  return [
    // Cruise / fast straight.
    [0, 14],
    [0, 10],
    // Forward gentle turns.
    [kHalf, 12],
    [-kHalf, 12],
    // Forward tight turns at lower speed.
    [k, 8],
    [-k, 8],
    // Slow forward straight.
    [0, 5],
    // Reverse straight.
    [0, -4],
    // Reverse gentle turns.
    [kHalf, -4],
    [-kHalf, -4],
    // Reverse tight turns.
    [k, -3],
    [-k, -3],
  ];
}

// ---------------------------------------------------------------------------
// Data shapes.

export interface SampledPose {
  x: number;
  z: number;
  heading: number;
  speed: number;
  /** Time since trial start (seconds). */
  t: number;
}

export interface TrialResult {
  /** Index into the (startSpeed × controlSet) grid. */
  index: number;
  startSpeed: number;
  controls: [number, number];
  /** Recorded poses in the trial-local frame (origin at trial start, heading 0).
   *  Length = PRIMITIVE_SUBSTEPS + 1 (start + substeps). */
  samples: SampledPose[];
}

export interface SweepData {
  agent: VehicleAgent;
  startSpeeds: number[];
  controlSets: number[][];
  trials: TrialResult[];
}

export interface FitResult {
  params: LearnedVehicleParams;
  /** Final mean squared loss from the optimisation (lower = better). */
  loss: number;
  /** Mean planar position error (m) across all sampled substeps. */
  meanPosError: number;
  /** Max planar position error (m) across all sampled substeps. */
  maxPosError: number;
}

// ---------------------------------------------------------------------------
// World/vehicle setup.

export interface SweepWorld {
  rapier: typeof RAPIER;
  world: RAPIER.World;
  car: CarHandle;
  agent: VehicleAgent;
  dispose: () => void;
}

/** Build a fresh Rapier world + ground + raycast vehicle ready for trials.
 *  The ground is large (1km×1km) because trials accumulate drift over the
 *  full sweep — the car is driven continuously without teleports. */
export async function createSweepWorld(agent: VehicleAgent = CARCHASE_AGENT): Promise<SweepWorld> {
  const rapier = await ensureRapier();
  const world = new rapier.World({ x: 0, y: -9.81, z: 0 });
  createGroundCollider(world, {
    bounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
    pad: 20,
    friction: 1.5,
  });
  const car = createRaycastVehicle(world, {
    id: 'learn-trial',
    position: { x: 0, z: 0 },
    heading: 0,
    ...LEARN_VEHICLE_TUNING,
  });
  // Settle the chassis on its suspension once at the start so the wheels are
  // at rest length before the first trial begins driving.
  for (let i = 0; i < 30; i++) {
    car.applyControls({ steer: 0, throttle: 0, brake: 0 });
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    world.step();
  }
  return {
    rapier,
    world,
    car,
    agent,
    dispose() {
      car.dispose();
      world.free();
    },
  };
}

// ---------------------------------------------------------------------------
// Trial execution.

const ACCEL_SCALE = 6; // throttle/brake proportional-control scaling
const DECEL_SCALE = 8;

function controlsToWheelCommand(
  currentSpeed: number,
  curvature: number,
  targetSpeed: number,
  wheelBase: number,
): { steer: number; throttle: number; brake: number } {
  // Ackermann: steer = atan(curvature * L). Sign-flip matches the kinocat ↔
  // Rapier yaw convention applied inside `planToAckermannControls`.
  const steer = -Math.atan(curvature * wheelBase);
  const gear = targetSpeed >= 0 ? 1 : -1;
  const errMag = Math.abs(targetSpeed) - Math.abs(currentSpeed);
  let throttle = 0;
  let brake = 0;
  if (errMag > 0) {
    throttle = gear * Math.min(1, errMag / ACCEL_SCALE);
  } else if (errMag < 0) {
    brake = Math.min(1, -errMag / DECEL_SCALE);
  }
  return { steer, throttle, brake };
}

/** Max wall ticks (1/60s each) spent driving to the target start speed. The
 *  vehicle is fully physically driven — no setLinvel — so a generous cap
 *  matters for high target speeds. */
const MAX_RAMP_TICKS = 600; // up to 10s of simulated ramp-up
const SPEED_TOL = 0.25;     // m/s — "close enough" to target start speed

function physicsStep(sw: SweepWorld, cmd: { steer: number; throttle: number; brake: number }): void {
  sw.car.applyControls(cmd);
  sw.world.timestep = PHYSICS_DT;
  sw.car.vehicle.updateVehicle(PHYSICS_DT);
  sw.world.step();
}

/** Brake to a near-stop using ONLY the brake input. No teleport. */
function brakeToStop(sw: SweepWorld, maxTicks = 200): void {
  for (let i = 0; i < maxTicks; i++) {
    const s = sw.car.readState(0);
    if (Math.abs(s.speed) < 0.05) {
      // A few extra ticks of full brake to let suspension settle.
      for (let k = 0; k < 6; k++) physicsStep(sw, { steer: 0, throttle: 0, brake: 1 });
      return;
    }
    physicsStep(sw, { steer: 0, throttle: 0, brake: 1 });
  }
}

/** Drive the car to `target` m/s using only throttle/brake. Steers straight
 *  (curvature 0). Returns the number of ticks actually used. */
function driveToSpeed(sw: SweepWorld, target: number): number {
  const wheelBase = 2 * LEARN_WHEEL_BASE;
  for (let tick = 0; tick < MAX_RAMP_TICKS; tick++) {
    const s = sw.car.readState(0);
    if (Math.abs(s.speed - target) < SPEED_TOL) {
      return tick;
    }
    const cmd = controlsToWheelCommand(s.speed, 0, target, wheelBase);
    physicsStep(sw, cmd);
  }
  return MAX_RAMP_TICKS;
}

function rotate(dx: number, dz: number, c: number, s: number): [number, number] {
  return [dx * c + dz * s, -dx * s + dz * c];
}

/** Run one trial: brake to stop, drive to the target start speed under
 *  pure throttle/brake control (no teleport), then apply the test controls
 *  for PRIMITIVE_DURATION while sample poses are recorded in the trial-local
 *  frame (recording-start pose at origin, heading 0). */
export function runTrial(
  sw: SweepWorld,
  startSpeed: number,
  controls: [number, number],
  index: number,
): TrialResult {
  const { car, world } = sw;
  const wheelBase = 2 * LEARN_WHEEL_BASE;

  // Step 1: brake to a near-stop so each trial starts from a known dynamic
  // state. No teleport — purely the brake input.
  brakeToStop(sw);

  // Step 2: physically accelerate (or reverse) to the target start speed.
  driveToSpeed(sw, startSpeed);

  // Step 3: capture the trial-local frame, then apply the test controls for
  // PRIMITIVE_DURATION while recording.
  const origin = car.readState(0);
  const c0 = Math.cos(origin.heading);
  const s0 = Math.sin(origin.heading);
  const samples: SampledPose[] = [
    { x: 0, z: 0, heading: 0, speed: origin.speed, t: 0 },
  ];
  const sampleTicks = new Set<number>();
  for (let k = 1; k <= PRIMITIVE_SUBSTEPS; k++) {
    sampleTicks.add(Math.round((k * TICKS_PER_TRIAL) / PRIMITIVE_SUBSTEPS));
  }

  for (let tick = 1; tick <= TICKS_PER_TRIAL; tick++) {
    const state = car.readState(0);
    const cmd = controlsToWheelCommand(state.speed, controls[0], controls[1], wheelBase);
    car.applyControls(cmd);
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    world.step();
    if (sampleTicks.has(tick)) {
      const s = car.readState(0);
      const [lx, lz] = rotate(s.x - origin.x, s.z - origin.z, c0, s0);
      let lh = s.heading - origin.heading;
      while (lh > Math.PI) lh -= 2 * Math.PI;
      while (lh < -Math.PI) lh += 2 * Math.PI;
      samples.push({
        x: lx,
        z: lz,
        heading: lh,
        speed: s.speed,
        t: tick * PHYSICS_DT,
      });
    }
  }
  return { index, startSpeed, controls, samples };
}

export interface SweepOptions {
  agent?: VehicleAgent;
  startSpeeds?: number[];
  controlSets?: number[][];
  /** Fired after each completed trial (1..total). */
  onProgress?: (done: number, total: number) => void;
  /** Awaited between trials so the browser can paint a progress bar. */
  yieldEvery?: number;
  yieldFn?: () => Promise<void>;
}

/** Run the full sweep of trials. Async so the React UI can yield between
 *  batches via `yieldFn`; the test passes no `yieldFn` and the function
 *  resolves on the next microtask. */
export async function runSweep(
  sw: SweepWorld,
  opts: SweepOptions = {},
): Promise<SweepData> {
  const agent = opts.agent ?? sw.agent;
  const startSpeeds = opts.startSpeeds ?? DEFAULT_START_SPEEDS;
  const controlSets = opts.controlSets ?? defaultControlSets(agent);
  const yieldEvery = Math.max(1, opts.yieldEvery ?? 1);
  const total = startSpeeds.length * controlSets.length;
  const trials: TrialResult[] = [];
  let i = 0;
  for (const speed of startSpeeds) {
    for (const ctrl of controlSets) {
      const c: [number, number] = [ctrl[0]!, ctrl[1]!];
      trials.push(runTrial(sw, speed, c, i));
      i++;
      opts.onProgress?.(i, total);
      if (opts.yieldFn && i % yieldEvery === 0) await opts.yieldFn();
    }
  }
  return { agent, startSpeeds, controlSets, trials };
}

// ---------------------------------------------------------------------------
// Parameter fitting (Nelder-Mead simplex).

const POS_W = 1;
const HEADING_W = 5;
const SPEED_W = 1;

function angleDelta(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Substeps used by the fit's internal integration. Higher = finer integration
 *  but slower fit. 11 ticks per sampled substep keeps the fit error well below
 *  the noise floor of the Rapier trials. */
const FIT_SUBSTEPS_PER_SAMPLE = 6;

function rolloutAndLoss(
  params: LearnedVehicleParams,
  data: SweepData,
): { loss: number; errors: number[] } {
  const sim = learnedForwardSim(params, data.agent);
  let loss = 0;
  const errors: number[] = [];
  // Sub-divide each between-sample interval so the integration in the loss
  // matches the resolution of the open-loop trial (33 physics ticks across
  // 6 samples ≈ 5-6 ticks per sample window).
  for (const tr of data.trials) {
    let s: VehicleState = {
      x: 0,
      z: 0,
      heading: 0,
      speed: tr.startSpeed,
      t: 0,
    };
    for (let k = 1; k < tr.samples.length; k++) {
      const a = tr.samples[k - 1]!;
      const b = tr.samples[k]!;
      const dt = (b.t - a.t) / FIT_SUBSTEPS_PER_SAMPLE;
      for (let j = 0; j < FIT_SUBSTEPS_PER_SAMPLE; j++) {
        s = sim(s, tr.controls, dt);
      }
      const dx = s.x - b.x;
      const dz = s.z - b.z;
      const dh = angleDelta(s.heading, b.heading);
      const ds = s.speed - b.speed;
      loss += POS_W * (dx * dx + dz * dz) + HEADING_W * dh * dh + SPEED_W * ds * ds;
      errors.push(Math.sqrt(dx * dx + dz * dz));
    }
  }
  return { loss, errors };
}

const PARAM_ORDER = [
  'maxAccel',
  'maxDecel',
  'accelTau',
  'understeerGain',
  'lateralDrag',
] as const;
const PARAM_LO: Record<(typeof PARAM_ORDER)[number], number> = {
  maxAccel: 0.5,
  maxDecel: 0.5,
  accelTau: 0.02,
  understeerGain: 0,
  lateralDrag: 0,
};
const PARAM_HI: Record<(typeof PARAM_ORDER)[number], number> = {
  maxAccel: 30,
  maxDecel: 30,
  accelTau: 2,
  understeerGain: 1,
  lateralDrag: 5,
};

function toVec(p: LearnedVehicleParams): number[] {
  return PARAM_ORDER.map((k) => p[k]);
}

function fromVec(v: number[]): LearnedVehicleParams {
  const out = {} as LearnedVehicleParams;
  PARAM_ORDER.forEach((k, i) => {
    const lo = PARAM_LO[k];
    const hi = PARAM_HI[k];
    let x = v[i] ?? DEFAULT_LEARNED_PARAMS[k];
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    out[k] = x;
  });
  return out;
}

/** Nelder-Mead simplex search. Deterministic; max 400 iterations is enough for
 *  a 5-param, smooth-loss problem. */
function nelderMead(
  x0: number[],
  loss: (v: number[]) => number,
  opts: { maxIter?: number; tol?: number; step?: number } = {},
): number[] {
  const n = x0.length;
  const maxIter = opts.maxIter ?? 400;
  const tol = opts.tol ?? 1e-6;
  const step = opts.step ?? 0.1;
  // Initial simplex.
  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const v = x0.slice();
    v[i] = v[i]! * (1 + step) + (v[i] === 0 ? step : 0);
    simplex.push(v);
  }
  let scores = simplex.map(loss);
  for (let iter = 0; iter < maxIter; iter++) {
    // Sort by score ascending.
    const order = scores
      .map((s, i) => [s, i] as const)
      .sort((a, b) => a[0] - b[0])
      .map((p) => p[1]);
    const sortedSim = order.map((i) => simplex[i]!);
    const sortedScores = order.map((i) => scores[i]!);
    for (let i = 0; i < simplex.length; i++) {
      simplex[i] = sortedSim[i]!;
      scores[i] = sortedScores[i]!;
    }
    const best = scores[0]!;
    const worst = scores[n]!;
    if (worst - best < tol) break;
    // Centroid of all but worst.
    const centroid = new Array(n).fill(0) as number[];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]!;
    }
    for (let j = 0; j < n; j++) centroid[j]! /= n;
    // Reflection.
    const xr = centroid.map((c, j) => c + (c - simplex[n]![j]!));
    const fr = loss(xr);
    if (fr < scores[n - 1]! && fr >= scores[0]!) {
      simplex[n] = xr;
      scores[n] = fr;
      continue;
    }
    if (fr < scores[0]!) {
      // Expansion.
      const xe = centroid.map((c, j) => c + 2 * (c - simplex[n]![j]!));
      const fe = loss(xe);
      if (fe < fr) {
        simplex[n] = xe;
        scores[n] = fe;
      } else {
        simplex[n] = xr;
        scores[n] = fr;
      }
      continue;
    }
    // Contraction.
    const xc = centroid.map((c, j) => c + 0.5 * (simplex[n]![j]! - c));
    const fc = loss(xc);
    if (fc < scores[n]!) {
      simplex[n] = xc;
      scores[n] = fc;
      continue;
    }
    // Shrink.
    for (let i = 1; i <= n; i++) {
      simplex[i] = simplex[0]!.map((b, j) => b + 0.5 * (simplex[i]![j]! - b));
      scores[i] = loss(simplex[i]!);
    }
  }
  return simplex[0]!;
}

export interface FitOptions {
  /** Initial guess. Defaults to `DEFAULT_LEARNED_PARAMS`. */
  init?: LearnedVehicleParams;
  /** Optimiser iteration cap. */
  maxIter?: number;
}

/** Fit the five-parameter dynamics model to recorded sweep data via
 *  Nelder-Mead. Deterministic for a fixed `init`. */
export function fitParams(data: SweepData, opts: FitOptions = {}): FitResult {
  const init = opts.init ?? DEFAULT_LEARNED_PARAMS;
  const x0 = toVec(init);
  const lossFn = (v: number[]) => rolloutAndLoss(fromVec(v), data).loss;
  const xStar = nelderMead(x0, lossFn, { maxIter: opts.maxIter ?? 400 });
  const params = fromVec(xStar);
  const finalRoll = rolloutAndLoss(params, data);
  const errs = finalRoll.errors;
  const mean = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length ? Math.max(...errs) : 0;
  return { params, loss: finalRoll.loss, meanPosError: mean, maxPosError: max };
}

// ---------------------------------------------------------------------------
// Library generation.

export interface BuildLibraryOptions {
  agent?: VehicleAgent;
  startSpeeds?: number[];
  controlSets?: number[][];
}

/** Feed the fitted `learnedForwardSim` into the standard
 *  `characterizeVehicle()` to produce a `MotionPrimitiveLibrary` the planner
 *  uses unchanged. */
export function buildLearnedLibrary(
  params: LearnedVehicleParams,
  opts: BuildLibraryOptions = {},
): MotionPrimitiveLibrary {
  const agent = opts.agent ?? CARCHASE_AGENT;
  return characterizeVehicle({
    forwardSim: learnedForwardSim(params, agent),
    controlSets: opts.controlSets ?? defaultControlSets(agent),
    duration: PRIMITIVE_DURATION,
    substeps: PRIMITIVE_SUBSTEPS,
    startSpeeds: opts.startSpeeds ?? DEFAULT_START_SPEEDS,
  });
}

// ---------------------------------------------------------------------------
// Online fit: refit the 5 coefficients from arbitrary one-step (state,
// controls, dt, next-state) transitions. Unlike `fitParams` (which integrates
// each trial's controls across multiple substeps) the online loss compares
// `learnedSim(s, controls, dt)` directly against the observed next state.
// Used by `/raceprimitives` to refit between laps as the car races on the
// real course — no dedicated trial phase needed.

/** One physics tick of recorded driving data. `controls` is the
 *  (curvature, targetSpeed) pair the pure-pursuit tracker produced for this
 *  tick — the same form the parametric model accepts. */
export interface TransitionSample {
  state: VehicleState;
  controls: [number, number];
  dt: number;
  next: VehicleState;
}

function transitionLoss(
  samples: ReadonlyArray<TransitionSample>,
  params: LearnedVehicleParams,
  agent: VehicleAgent,
): { loss: number; errors: number[] } {
  const sim = learnedForwardSim(params, agent);
  let loss = 0;
  const errors: number[] = [];
  for (const t of samples) {
    const pred = sim(t.state, t.controls, t.dt);
    const dx = pred.x - t.next.x;
    const dz = pred.z - t.next.z;
    const dh = angleDelta(pred.heading, t.next.heading);
    const ds = pred.speed - t.next.speed;
    loss += POS_W * (dx * dx + dz * dz) + HEADING_W * dh * dh + SPEED_W * ds * ds;
    errors.push(Math.sqrt(dx * dx + dz * dz));
  }
  return { loss, errors };
}

export interface OnlineFitOptions {
  init?: LearnedVehicleParams;
  maxIter?: number;
}

/** Fit the 5-coefficient model directly to a buffer of one-step transitions
 *  recorded during real driving. Deterministic for a fixed `init`. */
export function fitParamsOnline(
  samples: ReadonlyArray<TransitionSample>,
  agent: VehicleAgent,
  opts: OnlineFitOptions = {},
): FitResult {
  const init = opts.init ?? DEFAULT_LEARNED_PARAMS;
  const x0 = toVec(init);
  const lossFn = (v: number[]) => transitionLoss(samples, fromVec(v), agent).loss;
  const xStar = nelderMead(x0, lossFn, { maxIter: opts.maxIter ?? 300 });
  const params = fromVec(xStar);
  const finalRoll = transitionLoss(samples, params, agent);
  const errs = finalRoll.errors;
  const mean = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length ? Math.max(...errs) : 0;
  return { params, loss: finalRoll.loss, meanPosError: mean, maxPosError: max };
}

/** Comparison summary: per-primitive max/mean error between the kinematic
 *  ghost and the recorded Rapier trial. Surfaced in the demo HUD. */
export interface DiscrepancySummary {
  meanPosError: number;
  maxPosError: number;
  meanSpeedError: number;
}

/** Compare the kinematic forward model (no fit needed) against recorded data
 *  so the HUD can show "this is the gap we just closed". */
export function summariseKinematicGap(data: SweepData): DiscrepancySummary {
  const sim = kinematicForwardSim(data.agent);
  const errors: number[] = [];
  const speedErrors: number[] = [];
  for (const tr of data.trials) {
    let s: VehicleState = {
      x: 0,
      z: 0,
      heading: 0,
      speed: tr.startSpeed,
      t: 0,
    };
    for (let k = 1; k < tr.samples.length; k++) {
      const a = tr.samples[k - 1]!;
      const b = tr.samples[k]!;
      const dt = (b.t - a.t) / FIT_SUBSTEPS_PER_SAMPLE;
      for (let j = 0; j < FIT_SUBSTEPS_PER_SAMPLE; j++) {
        s = sim(s, tr.controls, dt);
      }
      const dx = s.x - b.x;
      const dz = s.z - b.z;
      errors.push(Math.sqrt(dx * dx + dz * dz));
      speedErrors.push(Math.abs(s.speed - b.speed));
    }
  }
  const mean = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : 0;
  const max = errors.length ? Math.max(...errors) : 0;
  const sMean = speedErrors.length
    ? speedErrors.reduce((a, b) => a + b, 0) / speedErrors.length
    : 0;
  return { meanPosError: mean, maxPosError: max, meanSpeedError: sMean };
}
