// Single source of truth for converting "legacy normalized car commands"
// (Ackermann steer in planner frame + normalized throttle/brake) into the
// canonical `WheeledCarControls` shape consumed by:
//   - `CarHandle.applyWheeledControls` (the Rapier raycast vehicle)
//   - the v2 training pipeline (`WheeledCarControls` is what trials record)
//   - every car `Driver<CarKinematicState, WheeledCarControls>` impl
//
// THE STEER-SIGN-FLIP RULE LIVES IN THIS FILE — paired with the encoders in
// `./encoders.ts` they form the complete account of the planner ⇄ Rapier
// frame negotiation. Demos, the headless harness, the sim-to-real recorder,
// and the offline training driver all consume this helper so a single edit
// here propagates everywhere — no per-call-site arithmetic, no copy-paste.
//
// Background: Rapier's raycast vehicle uses a right-handed yaw convention
// (+yaw rotates +X toward -Z) while kinocat's planning sign convention has
// +yaw rotating +X toward +Z. We standardize on the planner frame for all
// `steer` values flowing through the system and let this helper apply the
// one negation needed at the chassis boundary.

import type { WheeledCarControls } from './types';

/** Per-chassis force-scale constants. */
export interface CarForceTuning {
  /** Peak engine force in Newtons (multiplied by normalized throttle). */
  engineForceN: number;
  /** Peak brake force in Newtons (multiplied by normalized brake). */
  brakeForceN: number;
}

/** Legacy normalized command shape used by keyboard / pure-pursuit / scripted
 *  drivers. Steer is in radians in the kinocat planning sign convention. */
export interface NormalizedCarCommand {
  /** Steer angle in radians (planner-frame, +left). */
  steer: number;
  /** Throttle in [-1, 1] (negative = reverse drive force). */
  throttle: number;
  /** Brake in [0, 1]. */
  brake: number;
}

/**
 * Convert a normalized car command into canonical `WheeledCarControls`.
 *
 * Pre-negates `steer` so that a planner-frame steer of +0.2 rad ("turn left
 * visually") becomes a Rapier-frame steer of -0.2 rad on the chassis, which
 * is the same direction the user / model sees in the 3D scene.
 *
 * Scales `throttle`/`brake` by the supplied per-chassis force constants.
 */
export function wheeledFromNormalized(
  cmd: NormalizedCarCommand,
  tuning: CarForceTuning,
): WheeledCarControls {
  return {
    steer: -cmd.steer,
    driveForce: cmd.throttle * tuning.engineForceN,
    brakeForce: cmd.brake * tuning.brakeForceN,
  };
}

/** Convenience: a zero `WheeledCarControls` value for idle / settle loops. */
export const ZERO_WHEELED: Readonly<WheeledCarControls> = Object.freeze({
  steer: 0,
  driveForce: 0,
  brakeForce: 0,
});
