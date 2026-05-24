// Sampling-based MPC tracker for Ackermann vehicles.
//
// Replaces (or augments) pure-pursuit when the application needs
// high-fidelity execution of a plan — e.g. parking in tight spaces,
// multi-step back-and-forth corrections, precise terminal poses, or
// any scenario where the controller's actuator → state mapping matters
// (which pure-pursuit ignores entirely).
//
// The flavour: random-shooting MPC with warm-start. Every tick we
//   1) sample K candidate control sequences (each N steps of
//      `[steer, driveForce, brakeForce]`), centred on the
//      previously-best sequence shifted by one step (the warm-start);
//   2) roll each sequence through `forwardSim` (typically the v2
//      learned model — the same dynamics model the planner trusts);
//   3) score every rolled trajectory against the plan polyline +
//      penalties for control-rate / jerk / terminal pose;
//   4) keep the lowest-cost sequence and emit its first control as
//      this tick's command.
//
// Sampling (not gradient) chosen because: it's robust to discontinuous
// constraints (friction circle, brake saturation), it parallelises
// trivially if we ever move to a worker, and it doesn't need
// differentiable rollouts. The classic CEM / MPPI literature would
// upgrade this to a weighted softmax over samples — a future refinement.

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import type { PlanPath } from './types';

/** The MPC tracker emits actuator commands directly (no curvature →
 *  steer conversion in between), so the natural return type is the
 *  native wheeled-vehicle control set. */
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
  /** Cost of the best rollout — useful for ablation diagnostics. */
  bestCost: number;
}

export interface MPCTrackerConfig {
  /** Number of MPC steps in the rolling horizon. Default 6 (~0.6 s
   *  at the default 0.1 s step). */
  horizonSteps?: number;
  /** Length of each MPC step (s). Default 0.1. */
  stepDt?: number;
  /** Number of control sequences sampled per tick. Default 32. */
  samples?: number;
  /** Sampling stddev around the warm-started prior, in actuator units. */
  steerStd?: number;        // rad
  driveStd?: number;        // N
  brakeStd?: number;        // N
  /** Actuator limits — clamp every sample within these. */
  maxSteer: number;
  maxDriveForce: number;
  maxBrakeForce: number;
  /** Allow sampling negative `driveForce` (reverse). Default true; turn
   *  off for forward-only scenarios like race cruising. */
  allowReverse?: boolean;
  /** Cost weights. Tune empirically; defaults work for race + parking. */
  wLateral?: number;
  wHeading?: number;
  wSpeed?: number;
  wControlRate?: number;
  wTerminalPosition?: number;
  wTerminalSpeed?: number;
  /** Goal pose at the end of the plan — used by the terminal-pose
   *  cost terms. Defaults to the last plan sample. */
  goalTolerance?: number;
}

export interface MPCTrackerState {
  /** Best control sequence from the previous call, kept for warm-start.
   *  Flattened `[s0, d0, b0, s1, d1, b1, ...]`. Length = 3 × horizon. */
  prev: Float64Array;
  /** RNG state (linear congruential) so the tracker is deterministic
   *  given a seed — required for ablation reproducibility. */
  rngState: number;
}

/** Create a fresh MPC tracker state. Deterministic on `seed`. */
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

/** Build a per-step reference state from the plan polyline by walking
 *  arc-length forward from the projection of `current`. Returns one
 *  reference point per MPC step (so we can score each rollout's
 *  matching step against the right plan point). */
function buildReference(
  current: CarKinematicState,
  plan: PlanPath,
  horizon: number,
  cruiseSpeed: number,
  stepDt: number,
): { ref: CarKinematicState[]; totalArc: number } {
  if (plan.length === 0) {
    const ref: CarKinematicState[] = [];
    for (let i = 0; i < horizon; i++) ref.push({ ...current });
    return { ref, totalArc: 0 };
  }
  // Project `current` onto the plan to find the starting arc-length.
  let bestI = 0;
  let bestD = Infinity;
  for (let i = 0; i < plan.length; i++) {
    const dx = plan[i]!.x - current.x;
    const dz = plan[i]!.z - current.z;
    const d = dx * dx + dz * dz;
    if (d < bestD) {
      bestD = d;
      bestI = i;
    }
  }
  // Cumulative arc from the projection point onward.
  const cum: number[] = [0];
  for (let i = bestI + 1; i < plan.length; i++) {
    const a = plan[i - 1]!;
    const b = plan[i]!;
    cum.push(cum[cum.length - 1]! + Math.hypot(b.x - a.x, b.z - a.z));
  }
  const ref: CarKinematicState[] = [];
  for (let k = 1; k <= horizon; k++) {
    // Step forward at the plan's local speed (use the planned per-sample
    // speed if it's a sensible magnitude, else fall back to cruise).
    const localSpeed =
      Math.abs(plan[bestI]!.speed) > 0.5 ? Math.abs(plan[bestI]!.speed) : cruiseSpeed;
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
      speed: a.speed + (b.speed - a.speed) * u,
      t: 0,
    });
  }
  return { ref, totalArc: cum[cum.length - 1] ?? 0 };
}

function scoreRollout(
  rollout: CarKinematicState[],
  reference: CarKinematicState[],
  controls: Float64Array,
  prevSeed: Float64Array,
  goal: CarKinematicState | undefined,
  w: Required<Pick<MPCTrackerConfig,
    'wLateral' | 'wHeading' | 'wSpeed' | 'wControlRate' | 'wTerminalPosition' | 'wTerminalSpeed'
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
    const dh = ((s.heading - r.heading + Math.PI) % (2 * Math.PI)) - Math.PI;
    cost += w.wHeading * dh * dh;
    const dv = Math.abs(s.speed) - Math.abs(r.speed);
    cost += w.wSpeed * dv * dv;
    if (w.wControlRate > 0 && i > 0) {
      const ds = controls[i * 3]! - controls[(i - 1) * 3]!;
      const dd = controls[i * 3 + 1]! - controls[(i - 1) * 3 + 1]!;
      const db = controls[i * 3 + 2]! - controls[(i - 1) * 3 + 2]!;
      cost += w.wControlRate * (ds * ds + dd * dd * 1e-6 + db * db * 1e-6);
    }
    if (i === 0) {
      // Rate vs the previous tick's first command (the actual previous
      // actuator setpoint) — penalises actuator chatter between ticks,
      // not just within one MPC horizon.
      const ds0 = controls[0]! - prevSeed[0]!;
      cost += w.wControlRate * 0.5 * ds0 * ds0;
    }
  }
  // Terminal pose cost — critical for parking precision.
  if (goal && (w.wTerminalPosition > 0 || w.wTerminalSpeed > 0)) {
    const last = rollout[rollout.length - 1]!;
    if (w.wTerminalPosition > 0) {
      const dx = last.x - goal.x;
      const dz = last.z - goal.z;
      cost += w.wTerminalPosition * (dx * dx + dz * dz);
    }
    if (w.wTerminalSpeed > 0) {
      // Reaching a parking goal means low |speed| at the end.
      cost += w.wTerminalSpeed * last.speed * last.speed;
    }
  }
  return cost;
}

/**
 * One tick of the sampling MPC tracker. Pure modulo the random-state
 * mutation inside `state` (which is bounded to `MPCTrackerState.rngState`
 * for determinism). Returns the actuator command for THIS tick; the
 * optimised future commands are kept in `state` for warm-start on the
 * next call.
 */
export function mpcTrack(
  current: CarKinematicState,
  plan: PlanPath,
  forwardSim: ForwardSim<CarKinematicState>,
  state: MPCTrackerState,
  config: MPCTrackerConfig,
): MPCCommand {
  const H = config.horizonSteps ?? 6;
  const dt = config.stepDt ?? 0.1;
  const K = config.samples ?? 32;
  const sStd = config.steerStd ?? 0.15;
  const dStd = config.driveStd ?? 0.3 * config.maxDriveForce;
  const bStd = config.brakeStd ?? 0.2 * config.maxBrakeForce;
  const allowReverse = config.allowReverse ?? true;
  const weights = {
    wLateral: config.wLateral ?? 5,
    wHeading: config.wHeading ?? 1,
    wSpeed: config.wSpeed ?? 1,
    wControlRate: config.wControlRate ?? 0.5,
    wTerminalPosition: config.wTerminalPosition ?? 0,
    wTerminalSpeed: config.wTerminalSpeed ?? 0,
  };

  // Resize warm-start buffer if horizon changed.
  if (state.prev.length !== H * 3) state.prev = new Float64Array(H * 3);

  // Build the reference trajectory (one ref point per MPC step).
  const goal = plan.length > 0 ? plan[plan.length - 1]! : undefined;
  const cruiseSpeed = goal ? Math.max(Math.abs(goal.speed), 1) : 5;
  const { ref } = buildReference(current, plan, H, cruiseSpeed, dt);

  // Warm-start prior — previous solution shifted by one step.
  const prior = new Float64Array(H * 3);
  for (let i = 0; i < H - 1; i++) {
    prior[i * 3]! = state.prev[(i + 1) * 3]!;
    prior[i * 3 + 1]! = state.prev[(i + 1) * 3 + 1]!;
    prior[i * 3 + 2]! = state.prev[(i + 1) * 3 + 2]!;
  }
  // The last step of the shifted prior repeats the previous final step
  // (a sensible coast continuation).
  prior[(H - 1) * 3]! = state.prev[(H - 1) * 3]!;
  prior[(H - 1) * 3 + 1]! = state.prev[(H - 1) * 3 + 1]!;
  prior[(H - 1) * 3 + 2]! = state.prev[(H - 1) * 3 + 2]!;

  let bestCost = Infinity;
  let bestSeq = new Float64Array(H * 3);
  // First "sample" is always the prior itself (no perturbation) so the
  // warm-start path is in the candidate set.
  const work = new Float64Array(H * 3);
  for (let k = 0; k < K; k++) {
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
    // Roll forward.
    let s: CarKinematicState = { ...current };
    const traj: CarKinematicState[] = [];
    for (let i = 0; i < H; i++) {
      const u = [work[i * 3]!, work[i * 3 + 1]!, work[i * 3 + 2]!];
      s = forwardSim(s, u, dt);
      traj.push(s);
    }
    const cost = scoreRollout(traj, ref, work, state.prev, goal, weights);
    if (cost < bestCost) {
      bestCost = cost;
      bestSeq = new Float64Array(work);
    }
  }

  // Persist for warm-start.
  state.prev = bestSeq;

  const tol = config.goalTolerance ?? 0.5;
  const atGoal =
    goal !== undefined &&
    Math.hypot(current.x - goal.x, current.z - goal.z) <= tol;
  return {
    steer: bestSeq[0]!,
    driveForce: bestSeq[1]!,
    brakeForce: bestSeq[2]!,
    targetSpeed: ref[0]?.speed ?? 0,
    lookahead: { x: ref[0]?.x ?? current.x, z: ref[0]?.z ?? current.z },
    atGoal,
    bestCost,
  };
}
