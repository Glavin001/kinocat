// Physical capability envelope derived from a vehicle's configuration.
//
// The planner, tracker, and demo tuning have historically held hand-copied
// mirrors of these numbers (mass 580 vs the true 576, minTurnRadius 4.5 vs
// the true 4.68), which silently drift whenever the chassis options change.
// This module makes every capability a *derivation* from the single
// `LearnableVehicleConfig` source (itself derivable from Rapier options via
// `deriveLearnableConfig`), so "configurable vehicle parameters" stay safe
// to configure.
//
// Contract the derivation supports (assert it in tests, not prose):
//   planner envelope ⊂ tracker envelope ⊂ plant envelope.
// Use `plannerVehicleCapabilities` for search-side limits — it applies a
// conservative margin so the tracker always retains control authority to
// correct with.

import type { LearnableVehicleConfig } from './vehicle-config';

const G = 9.81;

export interface VehicleCapabilities {
  /** Chassis mass (kg) — echoed from config for convenience. */
  chassisMass: number;
  /** Front-to-rear axle distance (m) = 2 * config.wheelBase. */
  wheelbaseLength: number;
  /** Kinematic minimum turning radius (m) at full steering lock,
   *  measured at the rear axle: L / tan(maxSteerAngle). */
  minTurnRadius: number;
  /** Upper bound on longitudinal acceleration (m/s²): the smaller of the
   *  drivetrain limit (per-wheel engine force × driven wheels / mass —
   *  the Rapier adapter applies `engineForce` to EACH driven wheel) and
   *  the driven-axle traction limit (μ × static axle load share × g).
   *  The real plant sits below this (slip, suspension load transfer). */
  maxAccel: number;
  /** Upper bound on braking deceleration (m/s²): the smaller of the brake
   *  hardware limit (per-wheel brake force × 4 / mass — the adapter
   *  brakes all four wheels) and the tire grip ceiling (μ·g). */
  maxDecel: number;
  /** Tire grip ceiling on lateral acceleration (m/s²): μ·g. */
  maxLateralAccel: number;
}

function drivenWheelCount(config: LearnableVehicleConfig): number {
  return config.drivenWheels === 'awd' ? 4 : 2;
}

/** Static weight share carried by the driven axle(s). Cuboid chassis with
 *  symmetric wheel placement → 0.5 per axle; AWD uses all grip. */
function drivenAxleShare(config: LearnableVehicleConfig): number {
  return config.drivenWheels === 'awd' ? 1 : 0.5;
}

export function deriveVehicleCapabilities(
  config: LearnableVehicleConfig,
): VehicleCapabilities {
  const m = Math.max(1, config.chassisMass);
  const L = 2 * config.wheelBase;
  const mu = config.frictionSlip;
  const driveLimit = (drivenWheelCount(config) * config.maxDriveForce) / m;
  const tractionLimit = mu * drivenAxleShare(config) * G;
  const brakeLimit = (4 * config.maxBrakeForce) / m;
  const gripLimit = mu * G;
  return {
    chassisMass: m,
    wheelbaseLength: L,
    minTurnRadius: L / Math.tan(Math.min(1.5, Math.max(1e-3, config.maxSteerAngle))),
    maxAccel: Math.min(driveLimit, tractionLimit),
    maxDecel: Math.min(brakeLimit, gripLimit),
    maxLateralAccel: gripLimit,
  };
}

/** Default planner-side safety margin: the planner assumes 10% less
 *  capability than the plant has, so tracking error never requires the
 *  tracker to exceed physical limits to stay on plan. */
export const DEFAULT_PLANNER_CAPABILITY_MARGIN = 0.1;

/** Capability envelope for the SEARCH side: strictly inside the plant's.
 *  Turn radius grows by the margin; accelerations shrink by it. */
export function plannerVehicleCapabilities(
  config: LearnableVehicleConfig,
  margin: number = DEFAULT_PLANNER_CAPABILITY_MARGIN,
): VehicleCapabilities {
  const plant = deriveVehicleCapabilities(config);
  const shrink = Math.max(0, Math.min(0.9, margin));
  return {
    ...plant,
    minTurnRadius: plant.minTurnRadius * (1 + shrink),
    maxAccel: plant.maxAccel * (1 - shrink),
    maxDecel: plant.maxDecel * (1 - shrink),
    maxLateralAccel: plant.maxLateralAccel * (1 - shrink),
  };
}
