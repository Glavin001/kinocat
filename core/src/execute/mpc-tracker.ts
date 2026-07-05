// Model Predictive Path Integral (MPPI) tracker for Ackermann vehicles.
//
// The single principled controller for the entire driving spectrum —
// from racing cruise to tight-space parking and every intermediate
// motion. Replaces the random-shooting MPC variant that preceded it
// (which was the "toy MPC" of pick-the-best-sample), and is the
// established production default in ROS Navigation 2 for ground
// vehicles. Reference: Williams et al., "Aggressive driving with
// model predictive path integral control" (ICRA 2016).
//
// The single-controller-for-everything property comes from how MPPI
// integrates samples: instead of selecting the best, every candidate
// is weighted by `exp(-(cost - min_cost) / λ)` (Boltzmann / Gibbs
// distribution) and the emitted control is the importance-weighted
// average. λ (the temperature) governs the regime:
//
//   λ → 0    Boltzmann collapses to pick-the-best → aggressive but
//            noisy, matches random-shooting MPC.
//   λ → ∞    Uniform average over samples → very smooth, very
//            cautious.
//   λ ~ cost-scale  Smooth weighted blend — the bias-variance sweet
//                   spot that handles racing AND parking with the
//                   same hyperparameter.
//
// The cost function does its own scenario detection: if the plan's
// terminal speed is near zero AND the goal is reachable within the
// MPC horizon, the terminal-pose weights kick in (precision mode).
// Otherwise the controller cruises with cost-weighted plan tracking
// (racing / general driving mode). No mode switching, no developer-
// visible knob — the cost shape derives from the plan content.

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import type { PlanPath } from './types';

/** Native wheeled-vehicle control set the tracker emits each tick. */
export interface MPCCommand {
  steer: number;       // rad (front-wheel angle)
  driveForce: number;  // N (signed; negative = reverse)
  brakeForce: number;  // N (>= 0)
  /** Reference speed at MPC step 1 — for telemetry / logging. */
  targetSpeed: number;
  /** Reference point at MPC step 1 — for visualisation. */
  lookahead: { x: number; z: number };
  /** Within the goal tolerance (terminal cost dominates). */
  atGoal: boolean;
  /** Min cost across sampled rollouts — useful for ablation. */
  bestCost: number;
}

export interface MPCTrackerConfig {
  /** Number of MPC steps in the rolling horizon. Default 10
   *  (~0.5 s at the default 0.05 s step). */
  horizonSteps?: number;
  /** Length of each MPC step (s). Default 0.05 (matches 60 Hz physics
   *  for cost-integration accuracy). */
  stepDt?: number;
  /** Number of control sequences sampled per tick. Default 64.
   *  Compute cost: K × H × (forward-sim cost) per tick. */
  samples?: number;
  /** Sampling stddev around the warm-started prior, in actuator units. */
  steerStd?: number;
  driveStd?: number;
  brakeStd?: number;
  /** Actuator limits — clamp every sample within these. */
  maxSteer: number;
  maxDriveForce: number;
  maxBrakeForce: number;
  /** Allow sampling negative `driveForce` (reverse). Default true. */
  allowReverse?: boolean;
  /**
   * MPPI temperature. Controls the importance-weighted average's
   * concentration: smaller λ → controller commits to the best-looking
   * sample (aggressive, noisy); larger λ → controller averages many
   * samples (smooth, cautious). Default 1.0 — works for both racing
   * cruise and parking with the same value because cost scaling is
   * normalised by `min(cost)`.
   */
  lambda?: number;
  /** Per-stage cost weights. */
  wLateral?: number;
  wHeading?: number;
  wSpeed?: number;
  wControlRate?: number;
  /** Separate steer-rate weight (much larger than wControlRate at high
   *  speed — see notes in the random-shooting precursor). Default 3. */
  wSteerRate?: number;
  /**
   * Terminal-pose costs. These are auto-activated when the plan's
   * terminal speed is near zero AND the goal is reachable within the
   * MPC horizon — i.e. when the plan is asking the chassis to STOP at
   * a pose (parking, multi-step back-and-forth target). For pure
   * cruise plans (terminal speed ≈ cruise speed) these don't fire and
   * the controller behaves like a standard tracking MPC. The config
   * values cap how strong the terminal cost can get; default to
   * sensible levels for parking.
   */
  wTerminalPosition?: number;
  wTerminalSpeed?: number;
  /** Distance below which `atGoal` is reported true. Default 0.5 m. */
  goalTolerance?: number;
  /**
   * Cruise speed (m/s) the reference advances at where the plan itself
   * doesn't dictate a speed (samples with |speed| ≤ 0.5, e.g. the rest state
   * at the start of a drive-through plan). Without this the tracker derives a
   * cruise from the plan's TERMINAL speed — which the planner leaves ≈ 0 even
   * for race gates — so the reference barely extends ahead and the chassis
   * crawls/stalls instead of accelerating to racing speed. Pass the scenario's
   * cruise (race: agent max speed; parking: the slow parking cap). Defaults to
   * `max(|goal.speed|, 1)` for backward compatibility. */
  cruiseSpeed?: number;
}

export interface MPCTrackerState {
  /** Best control sequence from the previous call, kept for warm-start.
   *  Flattened `[s0, d0, b0, s1, d1, b1, ...]`. Length = 3 × horizon. */
  prev: Float64Array;
  /** RNG state (linear congruential) so the tracker is deterministic
   *  given a seed — required for ablation reproducibility. */
  rngState: number;
}

/** Create a fresh MPPI tracker state. Deterministic on `seed`. */
export function createMPCTrackerState(horizonSteps: number, seed = 0x1337): MPCTrackerState {
  return {
    prev: new Float64Array(horizonSteps * 3),
    rngState: seed >>> 0 || 1,
  };
}

// 32-bit LCG (constants from Numerical Recipes). Returns [0, 1).
function lcg(state: MPCTrackerState): number {
  state.rngState = (Math.imul(state.rngState, 1664525) + 1013904223) >>> 0;
  return state.rngState / 0x100000000;
}

// Box–Muller — one standard normal sample per call.
function gauss(state: MPCTrackerState): number {
  const u1 = Math.max(lcg(state), 1e-9);
  const u2 = lcg(state);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/** Per-step reference state from the plan polyline by walking
 *  arc-length forward from the projection of `current`. */
function buildReference(
  current: CarKinematicState,
  plan: PlanPath,
  horizon: number,
  cruiseSpeed: number,
  stepDt: number,
): { ref: CarKinematicState[]; bestI: number } {
  if (plan.length === 0) {
    const ref: CarKinematicState[] = [];
    for (let i = 0; i < horizon; i++) ref.push({ ...current });
    return { ref, bestI: 0 };
  }
  let bestI = 0;
  let bestD = Infinity;
  for (let i = 0; i < plan.length; i++) {
    const dx = plan[i]!.x - current.x;
    const dz = plan[i]!.z - current.z;
    const d = dx * dx + dz * dz;
    if (d < bestD) { bestD = d; bestI = i; }
  }
  const cum: number[] = [0];
  for (let i = bestI + 1; i < plan.length; i++) {
    const a = plan[i - 1]!;
    const b = plan[i]!;
    cum.push(cum[cum.length - 1]! + Math.hypot(b.x - a.x, b.z - a.z));
  }
  // Speed the reference advances along the plan. Use the plan's local speed
  // where it dictates one; otherwise (rest state, or a plan that left speed
  // unset) fall back to the scenario cruise so the reference reaches a full
  // horizon ahead rather than bunching up at the chassis.
  const localSpeed =
    Math.abs(plan[bestI]!.speed) > 0.5 ? Math.abs(plan[bestI]!.speed) : cruiseSpeed;
  const ref: CarKinematicState[] = [];
  for (let k = 1; k <= horizon; k++) {
    const targetArc = k * stepDt * localSpeed;
    let j = 0;
    while (j < cum.length - 1 && cum[j + 1]! < targetArc) j++;
    const segLen = (j < cum.length - 1) ? (cum[j + 1]! - cum[j]!) : 0;
    const u = segLen > 1e-9 ? (targetArc - cum[j]!) / segLen : 0;
    const a = plan[bestI + j]!;
    const b = plan[Math.min(bestI + j + 1, plan.length - 1)]!;
    ref.push({
      x: a.x + (b.x - a.x) * u,
      z: a.z + (b.z - a.z) * u,
      heading: a.heading + (b.heading - a.heading) * u,
      // Reference speed = the advance speed, NOT the plan's stored per-sample
      // speed. The two must agree or the speed-tracking cost fights the
      // position-tracking cost: a plan that starts at rest (speed ≈ 0) but
      // whose positions are advanced at cruise would otherwise ask the chassis
      // to be both 15 m ahead AND stopped, so MPPI blends full throttle with
      // full brake and the car never leaves the line (the race DNF).
      speed: localSpeed,
      t: 0,
    });
  }
  return { ref, bestI };
}

function scoreRollout(
  rollout: CarKinematicState[],
  reference: CarKinematicState[],
  controls: Float64Array,
  prevSeed: Float64Array,
  goal: CarKinematicState | undefined,
  terminalActive: boolean,
  w: Required<Pick<MPCTrackerConfig,
    'wLateral' | 'wHeading' | 'wSpeed' | 'wControlRate' | 'wSteerRate' | 'wTerminalPosition' | 'wTerminalSpeed'
  >>,
): number {
  let cost = 0;
  const H = rollout.length;
  for (let i = 0; i < H; i++) {
    const r = reference[i]!;
    const s = rollout[i]!;
    const dx = s.x - r.x;
    const dz = s.z - r.z;
    cost += w.wLateral * (dx * dx + dz * dz);
    const dh = wrapPi(s.heading - r.heading);
    cost += w.wHeading * dh * dh;
    const dv = Math.abs(s.speed) - Math.abs(r.speed);
    cost += w.wSpeed * dv * dv;
    if (i > 0) {
      const ds = controls[i * 3]! - controls[(i - 1) * 3]!;
      const dd = controls[i * 3 + 1]! - controls[(i - 1) * 3 + 1]!;
      const db = controls[i * 3 + 2]! - controls[(i - 1) * 3 + 2]!;
      cost += w.wSteerRate * ds * ds;
      cost += w.wControlRate * (dd * dd * 1e-6 + db * db * 1e-6);
    } else {
      // Inter-tick steer rate (vs the previous tick's first command).
      const ds0 = controls[0]! - prevSeed[0]!;
      cost += w.wSteerRate * 0.5 * ds0 * ds0;
    }
  }
  // Terminal-pose cost. Auto-activated by `terminalActive` (set by
  // the caller when the plan asks the chassis to stop near a pose
  // and the goal is reachable within the horizon).
  if (terminalActive && goal !== undefined) {
    const last = rollout[rollout.length - 1]!;
    const dx = last.x - goal.x;
    const dz = last.z - goal.z;
    cost += w.wTerminalPosition * (dx * dx + dz * dz);
    cost += w.wTerminalSpeed * last.speed * last.speed;
  }
  return cost;
}

/**
 * One MPPI tracker step. Pure modulo the deterministic RNG state
 * mutation inside `state`. Returns the actuator command for THIS
 * tick; the importance-weighted optimal sequence is kept in `state`
 * for next-tick warm-start.
 */
export function mpcTrack(
  current: CarKinematicState,
  plan: PlanPath,
  forwardSim: ForwardSim<CarKinematicState>,
  state: MPCTrackerState,
  config: MPCTrackerConfig,
): MPCCommand {
  const H = config.horizonSteps ?? 10;
  const dt = config.stepDt ?? 0.05;
  const K = config.samples ?? 64;
  const sStd = config.steerStd ?? 0.10;
  const dStd = config.driveStd ?? 0.4 * config.maxDriveForce;
  const bStd = config.brakeStd ?? 0.10 * config.maxBrakeForce;
  const allowReverse = config.allowReverse ?? true;
  const lambda = Math.max(config.lambda ?? 1.0, 1e-6);
  const weights = {
    wLateral: config.wLateral ?? 5,
    wHeading: config.wHeading ?? 2,
    wSpeed: config.wSpeed ?? 3,
    wControlRate: config.wControlRate ?? 0.5,
    wSteerRate: config.wSteerRate ?? 8,
    wTerminalPosition: config.wTerminalPosition ?? 30,
    wTerminalSpeed: config.wTerminalSpeed ?? 20,
  };
  const tol = config.goalTolerance ?? 0.5;

  if (state.prev.length !== H * 3) state.prev = new Float64Array(H * 3);

  // Build the reference + decide whether to activate terminal cost.
  const goal = plan.length > 0 ? plan[plan.length - 1]! : undefined;
  const cruiseSpeed =
    config.cruiseSpeed ?? (goal ? Math.max(Math.abs(goal.speed), 1) : 5);
  const { ref } = buildReference(current, plan, H, cruiseSpeed, dt);

  // Terminal-cost activation. Requires:
  //   (a) caller opted in via non-zero `wTerminalPosition` or
  //       `wTerminalSpeed` weights (intent signal from the scenario:
  //       "this plan asks the chassis to come to rest at a pose"),
  //   (b) the plan's terminal speed is near zero (so the plan agrees
  //       this is a stop, not a drive-through gate), AND
  //   (c) the goal is reachable within the MPC horizon (so terminal
  //       cost meaningfully fires inside the rollout — telling MPPI
  //       to optimise for a goal 100 m away is just noise).
  // The race scenario sets `wTerminalPosition=0` so this never
  // triggers there even when individual gate poses happen to have
  // `speed=0` from the planner's pose() helper.
  let terminalActive = false;
  const wantsTerminal = weights.wTerminalPosition > 0 || weights.wTerminalSpeed > 0;
  if (wantsTerminal && goal && Math.abs(goal.speed) < 0.5) {
    const distToGoal = Math.hypot(current.x - goal.x, current.z - goal.z);
    const maxReachInHorizon = Math.max(Math.abs(current.speed), 1) * H * dt + 2.0;
    if (distToGoal <= maxReachInHorizon) terminalActive = true;
  }

  // Warm-start prior — previous solution shifted by one step.
  const prior = new Float64Array(H * 3);
  for (let i = 0; i < H - 1; i++) {
    prior[i * 3]! = state.prev[(i + 1) * 3]!;
    prior[i * 3 + 1]! = state.prev[(i + 1) * 3 + 1]!;
    prior[i * 3 + 2]! = state.prev[(i + 1) * 3 + 2]!;
  }
  prior[(H - 1) * 3]! = state.prev[(H - 1) * 3]!;
  prior[(H - 1) * 3 + 1]! = state.prev[(H - 1) * 3 + 1]!;
  prior[(H - 1) * 3 + 2]! = state.prev[(H - 1) * 3 + 2]!;

  // Allocate sample storage. We need every sample's controls AND its
  // cost — MPPI's emit step is the importance-weighted average of
  // them all (NOT pick-the-best).
  const samples = new Float64Array(K * H * 3);
  const costs = new Float64Array(K);
  let minCost = Infinity;
  const work = new Float64Array(H * 3);

  for (let k = 0; k < K; k++) {
    // First sample (k=0) is the unperturbed prior — guarantees the
    // warm-start is in the candidate set.
    for (let i = 0; i < H; i++) {
      const noiseFactor = k === 0 ? 0 : 1;
      let steer = prior[i * 3]! + sStd * gauss(state) * noiseFactor;
      let drive = prior[i * 3 + 1]! + dStd * gauss(state) * noiseFactor;
      let brake = prior[i * 3 + 2]! + bStd * gauss(state) * noiseFactor;
      steer = clamp(steer, -config.maxSteer, config.maxSteer);
      const minDrive = allowReverse ? -config.maxDriveForce : 0;
      drive = clamp(drive, minDrive, config.maxDriveForce);
      brake = clamp(brake, 0, config.maxBrakeForce);
      work[i * 3]! = steer;
      work[i * 3 + 1]! = drive;
      work[i * 3 + 2]! = brake;
    }
    let s: CarKinematicState = { ...current };
    const traj: CarKinematicState[] = [];
    for (let i = 0; i < H; i++) {
      const u = [work[i * 3]!, work[i * 3 + 1]!, work[i * 3 + 2]!];
      s = forwardSim(s, u, dt);
      traj.push(s);
    }
    const cost = scoreRollout(traj, ref, work, state.prev, goal, terminalActive, weights);
    costs[k] = cost;
    if (cost < minCost) minCost = cost;
    // Copy work → samples[k]
    for (let i = 0; i < H * 3; i++) samples[k * H * 3 + i] = work[i]!;
  }

  // MPPI importance-weighted average. Subtract minCost for numerical
  // stability before the softmax (a classic log-sum-exp trick).
  const weightsArr = new Float64Array(K);
  let weightSum = 0;
  for (let k = 0; k < K; k++) {
    const w = Math.exp(-(costs[k]! - minCost) / lambda);
    weightsArr[k] = w;
    weightSum += w;
  }
  // Compose the optimal sequence as the weighted average.
  const optimal = new Float64Array(H * 3);
  if (weightSum > 1e-9) {
    for (let k = 0; k < K; k++) {
      const w = weightsArr[k]! / weightSum;
      for (let i = 0; i < H * 3; i++) {
        optimal[i]! += w * samples[k * H * 3 + i]!;
      }
    }
  } else {
    // Degenerate case (all weights underflow to 0); fall back to
    // the lowest-cost sample.
    let bestK = 0;
    for (let k = 1; k < K; k++) if (costs[k]! < costs[bestK]!) bestK = k;
    for (let i = 0; i < H * 3; i++) optimal[i] = samples[bestK * H * 3 + i]!;
  }
  // Re-clamp the weighted-average controls (the average of valid
  // controls is always valid for box constraints, but float roundoff
  // can put us microscopically out).
  for (let i = 0; i < H; i++) {
    optimal[i * 3]! = clamp(optimal[i * 3]!, -config.maxSteer, config.maxSteer);
    const minDrive = allowReverse ? -config.maxDriveForce : 0;
    optimal[i * 3 + 1]! = clamp(optimal[i * 3 + 1]!, minDrive, config.maxDriveForce);
    optimal[i * 3 + 2]! = clamp(optimal[i * 3 + 2]!, 0, config.maxBrakeForce);
  }

  state.prev = optimal;

  const atGoal =
    goal !== undefined &&
    Math.hypot(current.x - goal.x, current.z - goal.z) <= tol;
  return {
    steer: optimal[0]!,
    driveForce: optimal[1]!,
    brakeForce: optimal[2]!,
    targetSpeed: ref[0]?.speed ?? 0,
    lookahead: { x: ref[0]?.x ?? current.x, z: ref[0]?.z ?? current.z },
    atGoal,
    bestCost: minCost,
  };
}
