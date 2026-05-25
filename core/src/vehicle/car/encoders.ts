// Encoders mapping `WheeledCarControls` into the opaque `number[]` shape the
// car forward sims (v2 parametric / v2 learned / kinematic) expect.
//
// THE STEER-SIGN-FLIP RULE LIVES IN THIS FILE — and only this file. Rapier's
// raycast vehicle uses a right-handed yaw convention (+yaw rotates +X toward
// -Z) while kinocat's planning sign convention has +yaw rotating +X toward
// +Z. Every place that feeds a control vector into a kinocat-trained model
// from a Rapier-frame `steer` field must negate that steer.
//
// Putting the rule here, behind named encoders, means a future bug-hunt for
// "ghosts going the wrong way when WASD-driving" has a single grep target.

import type { WheeledCarControls } from './types';

/** Encode wheeled controls for the v2 parametric / v2 learned forward sims.
 *  Output: `[steer_planning_radians, driveForce_N, brakeForce_N]`.
 *
 *  The planning-frame steer is the NEGATIVE of the Rapier-frame steer so
 *  open-loop ghost predictions move the same direction as the real chassis. */
export function encodeForParametricV2(c: WheeledCarControls): number[] {
  return [-c.steer, c.driveForce, c.brakeForce];
}

/** Encode wheeled controls for the kinematic forward sim, which consumes
 *  `[curvature, targetSpeed]` rather than `[steer, driveForce, brakeForce]`.
 *
 *  Throttle/brake are mapped onto a +/- target-speed ramp via the supplied
 *  `maxSpeed`. The steer is interpreted as Ackermann steer angle in the
 *  Rapier frame; we convert to planning-frame curvature via `tan(-steer) /
 *  wheelBase` (the sign flip matches `encodeForParametricV2`). */
export function encodeForKinematic(
  c: WheeledCarControls,
  opts: { wheelBase: number; maxSpeed: number; throttle: number; brake: number },
): number[] {
  const curvature = Math.tan(-c.steer) / Math.max(opts.wheelBase, 1e-6);
  const target = (opts.throttle - opts.brake) * opts.maxSpeed;
  return [curvature, target];
}

/** Raw wheeled encoding: `[steer, driveForce, brakeForce]` with NO sign flip.
 *  Use when the consumer lives in the same sign convention as the Rapier
 *  raycast vehicle. */
export function encodeWheeledRaw(c: WheeledCarControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}
