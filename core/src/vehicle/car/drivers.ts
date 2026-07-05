// Car-domain driver implementations.
//
// Every driver here implements the generic `Driver<CarKinematicState,
// WheeledCarControls>` from `kinocat/scene`. Same shape everywhere:
//   - keyboard / WASD live driving
//   - plan-following with pure-pursuit
//   - programmatic playback patterns (slalom, throttle pulse, brake cycle)
//
// Demos hold a `SwitchableDriver` and swap between these on mode change. The
// dataset-gen layer will add `RandomWalkDriver`, `StepInputDriver`, etc. as
// siblings.

import type { Driver } from '../../scene/driver';
import type { PurePursuitConfig } from '../../execute/types';
import { followPlan } from './follow-plan';
import { keyboardAckermann, keysFromSet, type KeyState } from './keyboard';
import type { CarKinematicState, WheeledCarControls } from './types';

const ZERO: WheeledCarControls = { steer: 0, driveForce: 0, brakeForce: 0 };

// ---------------------------------------------------------------------------
// Keyboard / WASD driver.

export interface KeyboardCarDriverOpts {
  /** Live key snapshot. Demos pass a `Set<string>` they update from keyup /
   *  keydown handlers; the driver reads it every tick (no React state). */
  keys: () => ReadonlySet<string>;
  /** Max engine force (N). Throttle scales linearly to this. */
  engineForceN: number;
  /** Max brake force (N). */
  brakeForceN: number;
  /** Max steer angle (rad). */
  maxSteerAngle: number;
  /** Keyboard mapping gain (default 0.55). */
  steerGain?: number;
  /** Optional override: if provided, the driver uses this `KeyState`
   *  directly (bypassing `keysFromSet`). Useful for tests. */
  keyState?: () => KeyState;
}

export class KeyboardCarDriver implements Driver<CarKinematicState, WheeledCarControls> {
  constructor(private readonly opts: KeyboardCarDriverOpts) {}
  sample(_state: CarKinematicState, _t: number, _dt: number): WheeledCarControls {
    const ks = this.opts.keyState ? this.opts.keyState() : keysFromSet(this.opts.keys());
    const cmd = keyboardAckermann(ks, { steerGain: this.opts.steerGain });
    // Rapier-frame steer = -planning-frame steer
    const steer = Math.max(-1, Math.min(1, -cmd.steer)) * this.opts.maxSteerAngle;
    const driveForce = Math.max(-1, Math.min(1, cmd.throttle)) * this.opts.engineForceN;
    const brakeForce = Math.max(0, Math.min(1, cmd.brake)) * this.opts.brakeForceN;
    return { steer, driveForce, brakeForce };
  }
}

// ---------------------------------------------------------------------------
// Plan-follower driver.

export interface PlanFollowerCarDriverOpts {
  /** Pure-pursuit config. */
  config: PurePursuitConfig;
  /** Wheelbase used to convert curvature -> steer angle (m). */
  wheelBase: number;
  /** Max engine force (N). */
  engineForceN: number;
  /** Max brake force (N). */
  brakeForceN: number;
  /** Max steer angle (rad). */
  maxSteerAngle: number;
}

export class PlanFollowerCarDriver implements Driver<CarKinematicState, WheeledCarControls> {
  private plan: ReadonlyArray<CarKinematicState> = [];
  private planStartTime = 0;
  private started = false;

  constructor(private readonly opts: PlanFollowerCarDriverOpts) {}

  /** Swap the plan in. `simStartTime` is the sim-time at which the first
   *  sample of `plan` should be tracked. */
  setPlan(plan: ReadonlyArray<CarKinematicState>, simStartTime: number): void {
    this.plan = plan;
    this.planStartTime = simStartTime;
    this.started = true;
  }

  clearPlan(): void {
    this.plan = [];
    this.started = false;
  }

  sample(state: CarKinematicState, simTime: number, _dt: number): WheeledCarControls {
    if (!this.started || this.plan.length === 0) return ZERO;
    const elapsed = simTime - this.planStartTime;
    const cmd = followPlan(state, this.plan, { config: this.opts.config, elapsed });
    // Gear from the tracker's signed target speed. Pure-pursuit reports
    // throttle as a non-negative magnitude and encodes direction in
    // targetSpeed's sign; it also computes curvature in the direction of
    // travel. Driving a chassis therefore needs BOTH corrections the demo
    // runner applies: signed drive force (reverse maneuvers were silently
    // impossible here — throttle was clamped to [0, 1]) and the reverse
    // steer sign flip (an Ackermann chassis produces opposite world-frame
    // curvature per travel direction).
    const gear = cmd.targetSpeed < 0 ? -1 : 1;
    // Convert curvature -> Ackermann steer angle. `wheelBase` here is the
    // FULL front-to-rear axle spacing (the Rapier adapter's `wheelBase`
    // option is the half-spacing — pass 2x that). Negate for Rapier frame.
    // Net applied steer must be -gear * atan(kappa * L): the leading minus is
    // the kinocat->Rapier frame flip (applied via the outer negation below),
    // the gear factor is the reverse-travel flip. (A double negation here
    // once inverted steering for FORWARD drivers — every consumer veered off
    // instantly; see the sign regression test.)
    const steerPlanning = gear * Math.atan(cmd.steering * this.opts.wheelBase);
    const steer = Math.max(-this.opts.maxSteerAngle, Math.min(this.opts.maxSteerAngle, -steerPlanning));
    const driveForce = gear * Math.max(0, Math.min(1, cmd.throttle)) * this.opts.engineForceN;
    const brakeForce = Math.max(0, Math.min(1, cmd.brake)) * this.opts.brakeForceN;
    return { steer, driveForce, brakeForce };
  }

  reset(): void {
    this.started = false;
    this.plan = [];
  }
}

// ---------------------------------------------------------------------------
// Programmatic playback patterns.
//
// Used by /sim-to-real Playback mode to cycle the car through a deterministic
// throttle / slalom / brake script. Replaces the inline switch statement
// that lived in the React scene.

export type PlaybackPatternSegment =
  | { kind: 'hold'; duration: number; controls: WheeledCarControls }
  | { kind: 'ramp'; duration: number; from: WheeledCarControls; to: WheeledCarControls }
  | { kind: 'slalom'; duration: number; periodSec: number; steerAmp: number; driveForce: number };

export class PlaybackPatternCarDriver implements Driver<CarKinematicState, WheeledCarControls> {
  private startTime: number | null = null;
  private readonly totalDuration: number;
  constructor(private readonly script: ReadonlyArray<PlaybackPatternSegment>) {
    this.totalDuration = script.reduce((acc, s) => acc + s.duration, 0);
  }

  sample(_state: CarKinematicState, simTime: number, _dt: number): WheeledCarControls {
    if (this.startTime === null) this.startTime = simTime;
    if (this.totalDuration <= 0) return ZERO;
    const elapsed = ((simTime - this.startTime) % this.totalDuration + this.totalDuration) % this.totalDuration;
    let acc = 0;
    for (const seg of this.script) {
      if (elapsed < acc + seg.duration) {
        const local = elapsed - acc;
        return evalSegment(seg, local);
      }
      acc += seg.duration;
    }
    return ZERO;
  }

  /** Total seconds per cycle. Demos use this to trigger ghost / trail
   *  resets on cycle boundaries. */
  cycleSec(): number {
    return this.totalDuration;
  }

  /** True iff the supplied simTime is within `eps` of a cycle boundary. */
  isCycleBoundary(simTime: number, eps: number): boolean {
    if (this.startTime === null || this.totalDuration <= 0) return false;
    const phase = ((simTime - this.startTime) % this.totalDuration + this.totalDuration) % this.totalDuration;
    return phase < eps || this.totalDuration - phase < eps;
  }

  reset(): void {
    this.startTime = null;
  }
}

function evalSegment(seg: PlaybackPatternSegment, local: number): WheeledCarControls {
  if (seg.kind === 'hold') return seg.controls;
  if (seg.kind === 'ramp') {
    const u = Math.max(0, Math.min(1, local / Math.max(1e-9, seg.duration)));
    return {
      steer: seg.from.steer + (seg.to.steer - seg.from.steer) * u,
      driveForce: seg.from.driveForce + (seg.to.driveForce - seg.from.driveForce) * u,
      brakeForce: seg.from.brakeForce + (seg.to.brakeForce - seg.from.brakeForce) * u,
    };
  }
  // slalom
  const phase = (2 * Math.PI * local) / Math.max(1e-9, seg.periodSec);
  return {
    steer: Math.sin(phase) * seg.steerAmp,
    driveForce: seg.driveForce,
    brakeForce: 0,
  };
}
