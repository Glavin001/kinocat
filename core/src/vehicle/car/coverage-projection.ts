// Canonical 5-D coverage projection for the car domain — Phase 0 of the
// training-dataset plan.
//
// Axes chosen so the "holes" in the constant-hold dataset light up: speed,
// steer, lateralVel-normalized, yawRate, and a control-direction
// (drive-or-brake) discriminator. Pair with `createCoverageMeter` from
// `kinocat/training`.

import type { CarKinematicState, WheeledCarControls } from './types';
import type { CoverageAxis, CoverageProjection } from '../../training/coverage-meter';

export const CAR_COVERAGE_AXES: CoverageAxis[] = [
  // Forward speed. Includes the small reverse band so reverse trials light a
  // bin. Race envelope tops out near 30 m/s.
  { name: 'speed', lo: -8, hi: 32, bins: 8 },
  // Steering angle (radians). Defaults assume |max| ≈ 0.8 rad.
  { name: 'steer', lo: -0.8, hi: 0.8, bins: 6 },
  // Lateral velocity normalized by |speed|+ε. Sliding regime indicator.
  { name: 'lateralRel', lo: -0.8, hi: 0.8, bins: 5 },
  // Yaw rate (rad/s). |yawRate| > 0.5 = clearly mid-corner.
  { name: 'yawRate', lo: -2.5, hi: 2.5, bins: 5 },
  // Throttle vs. brake discriminator. Compact 4-bin code:
  // 0 = coast, 1 = throttle-only, 2 = brake-only, 3 = combined.
  { name: 'inputKind', lo: 0, hi: 4, bins: 4 },
];

export const carCoverageProjection: CoverageProjection<CarKinematicState, WheeledCarControls, unknown> = (
  state, controls, _cfg,
) => {
  const speed = state.speed;
  const steer = controls[0] ?? 0;
  const drive = controls[1] ?? 0;
  const brake = controls[2] ?? 0;
  const lateralRel = (state.lateralVelocity ?? 0) / Math.max(1, Math.abs(speed));
  const yawRate = state.yawRate ?? 0;
  const driving = Math.abs(drive) > 1; // > 1 N counts as engaged
  const braking = brake > 1;
  let inputKind = 0;
  if (driving && braking) inputKind = 3;
  else if (driving) inputKind = 1;
  else if (braking) inputKind = 2;
  return [speed, steer, lateralRel, yawRate, inputKind + 0.5];
};

/** Convenience helper: build the `controlsToVec` projection used by both
 *  the coverage meter and the evaluation harness for wheeled controls. */
export function wheeledControlsToVec(c: WheeledCarControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}
