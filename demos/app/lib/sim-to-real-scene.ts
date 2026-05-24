// Pure helpers for the /sim-to-real scope. These are the building blocks
// the React scene uses to project each model forward open-loop and to
// accumulate the rolling "model-vs-Rapier" gap. Lives in `lib/` so the
// pure parts can be unit-tested without Three.js or Rapier.
//
// Three primitives:
//   1. `rolloutOpenLoop`: drive a `ForwardSim<CarKinematicState>` through a
//      controls trace at fixed `dt`. Returns one CarKinematicState per tick
//      (inclusive of t=0). Used by Playback mode: feed the trial's
//      `controlsTrace` and you get the model's predicted timeline.
//   2. `projectFuture`: take the current state + a HELD constant control
//      and project T seconds ahead. Used by Free Drive mode every
//      200 ms to draw the prediction polyline + ghost-at-T.
//   3. `GapAccumulator`: ring-buffer of recent (real - predicted)
//      pose+heading+speed gaps; reports rolling RMS over the last
//      `windowSec` seconds. Used by the HUD.

import type { CarKinematicState } from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';

// ---------------------------------------------------------------------------
// Open-loop rollout

/** Drive a forward-sim through a controls trace.
 *
 *  - `initial.t` is the rollout's starting time; each returned state's
 *    `t` advances by `dt`.
 *  - Output length = `controlsTrace.length + 1` (the initial state at
 *    index 0, then one state per applied control).
 *  - The forward sim is responsible for advancing kinematic state; this
 *    helper just chains it. Yaw-rate / lateral velocity propagation is
 *    whatever the sim returns. */
export function rolloutOpenLoop(
  initial: CarKinematicState,
  controlsTrace: ReadonlyArray<ReadonlyArray<number>>,
  dt: number,
  forwardSim: ForwardSim<CarKinematicState>,
): CarKinematicState[] {
  const out: CarKinematicState[] = [initial];
  let s = initial;
  for (let i = 0; i < controlsTrace.length; i++) {
    const c = controlsTrace[i]!;
    const next = forwardSim(s, c as number[], dt);
    out.push(next);
    s = next;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Future projection (constant-control short horizon)

export interface FutureProjectionOpts {
  /** Total horizon in seconds (typically 1.0). */
  horizonSec: number;
  /** Sub-step used internally; defaults to 1/60. Smaller = more samples
   *  in the polyline but the same final pose for any linear ODE. */
  stepDt?: number;
}

/** Roll the sim forward with a constant control vector for `horizonSec`
 *  seconds. Returns the full sampled path (initial state included). */
export function projectFuture(
  initial: CarKinematicState,
  controls: ReadonlyArray<number>,
  forwardSim: ForwardSim<CarKinematicState>,
  opts: FutureProjectionOpts,
): CarKinematicState[] {
  const dt = opts.stepDt ?? 1 / 60;
  const steps = Math.max(1, Math.round(opts.horizonSec / dt));
  const trace = Array.from({ length: steps }, () => controls);
  return rolloutOpenLoop(initial, trace, dt, forwardSim);
}

// ---------------------------------------------------------------------------
// Gap accumulator

export interface GapSample {
  /** Absolute time the sample was recorded (seconds). */
  t: number;
  /** Euclidean position error (m). */
  posErr: number;
  /** Wrapped heading error (rad, in [-π, π]). */
  headingErr: number;
  /** Signed speed error (m/s). */
  speedErr: number;
}

export interface GapRms {
  /** Number of samples in the rolling window. */
  count: number;
  posRms: number;
  headingRms: number;
  speedRms: number;
}

/** Wrap an angle into [-π, π]. */
export function wrapPi(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

/** Instantaneous pose-error between a Rapier-real state and a model
 *  state at the SAME time t. The two should be aligned: index-i in
 *  Playback, or the prediction-at-T compared to the actual-at-T in
 *  Free Drive. */
export function poseGap(real: CarKinematicState, pred: CarKinematicState): GapSample {
  const dx = pred.x - real.x;
  const dz = pred.z - real.z;
  return {
    t: real.t,
    posErr: Math.hypot(dx, dz),
    headingErr: wrapPi(pred.heading - real.heading),
    speedErr: pred.speed - real.speed,
  };
}

/** Ring-buffer rolling-RMS accumulator over the last `windowSec`
 *  seconds. Each `push()` evicts entries older than the window. */
export class GapAccumulator {
  private samples: GapSample[] = [];
  constructor(private readonly windowSec: number = 2.0) {}

  push(sample: GapSample): void {
    this.samples.push(sample);
    const cutoff = sample.t - this.windowSec;
    while (this.samples.length > 0 && this.samples[0]!.t < cutoff) {
      this.samples.shift();
    }
  }

  rms(): GapRms {
    const n = this.samples.length;
    if (n === 0) return { count: 0, posRms: 0, headingRms: 0, speedRms: 0 };
    let pos2 = 0, hdg2 = 0, spd2 = 0;
    for (const s of this.samples) {
      pos2 += s.posErr * s.posErr;
      hdg2 += s.headingErr * s.headingErr;
      spd2 += s.speedErr * s.speedErr;
    }
    return {
      count: n,
      posRms: Math.sqrt(pos2 / n),
      headingRms: Math.sqrt(hdg2 / n),
      speedRms: Math.sqrt(spd2 / n),
    };
  }

  reset(): void {
    this.samples = [];
  }
}

// ---------------------------------------------------------------------------
// Free Drive "prediction-at-T" tracker
//
// Records a predicted pose tagged with the time T at which it's
// supposed to be observed. When the real chassis reaches that time we
// compare and emit a sample. Used to measure how good each model's
// 1-second forward prediction is, online, as the user steers.

export interface ScheduledPrediction {
  /** World time the prediction is supposed to be matched at. */
  matchAt: number;
  /** Predicted state at matchAt. */
  pred: CarKinematicState;
}

export class FuturePredictionTracker {
  private pending: ScheduledPrediction[] = [];
  /** Add a prediction (current time + horizon = matchAt). */
  schedule(pred: CarKinematicState, currentTime: number, horizonSec: number): void {
    this.pending.push({ matchAt: currentTime + horizonSec, pred });
  }
  /** Match every pending prediction whose matchAt is <= realState.t.
   *  Emits a GapSample for each, in chronological order, and removes
   *  them from the queue. */
  drainMatured(realState: CarKinematicState): GapSample[] {
    const out: GapSample[] = [];
    while (this.pending.length > 0 && this.pending[0]!.matchAt <= realState.t) {
      const { pred } = this.pending.shift()!;
      out.push(poseGap(realState, pred));
    }
    return out;
  }
  reset(): void {
    this.pending = [];
  }
  size(): number {
    return this.pending.length;
  }
}

// ---------------------------------------------------------------------------
// Speed-to-color helper for trail ribbons.

/** Map a speed (m/s) onto a green-yellow-red ramp, returning hex
 *  0xRRGGBB. Used by `TrailRibbon` to color each segment by the
 *  chassis's speed when the sample was recorded. */
export function speedToColor(speed: number, vMax: number): number {
  const t = Math.max(0, Math.min(1, Math.abs(speed) / Math.max(1e-6, vMax)));
  // Green (0x00cc66) -> Yellow (0xffcc00) -> Red (0xff3030)
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const k = t / 0.5;
    r = Math.round(0x00 + k * (0xff - 0x00));
    g = Math.round(0xcc + k * (0xcc - 0xcc));
    b = Math.round(0x66 + k * (0x00 - 0x66));
  } else {
    const k = (t - 0.5) / 0.5;
    r = Math.round(0xff + k * (0xff - 0xff));
    g = Math.round(0xcc + k * (0x30 - 0xcc));
    b = Math.round(0x00 + k * (0x30 - 0x00));
  }
  return (r << 16) | (g << 8) | b;
}
