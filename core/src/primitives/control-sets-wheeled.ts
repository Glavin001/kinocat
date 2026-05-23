// Pre-built control sets for a wheeled-vehicle action shape
// `[steer, driveForce, brakeForce]`. Two resolutions ship for IGHA*'s
// coarse-vs-fine tier expansion:
//
//   - COARSE: 5 typed actions, 0.5s primitives (cruise, gentle turns, brake)
//   - FINE  : 15 typed actions, 0.15s primitives (precise apex / late-brake)
//
// All actions are parameterized on the vehicle's physical limits so the
// library scales with config (a stronger engine produces longer cruise
// segments; a tighter max steer angle clamps to its bound).
//
// Used by `characterizeVehicle` to build a `MotionPrimitiveLibrary` against
// any `ForwardSim<VehicleState>` that accepts this control vector.

import type { LearnableVehicleConfig } from '../agent/vehicle-config';
import { encodeWheeled, type WheeledControls } from '../agent/controls';

export interface WheeledControlTierOptions {
  config: LearnableVehicleConfig;
  /** Fraction of `maxDriveForce` used by the cruise/full-throttle level. */
  fullDriveFraction?: number;
  /** Fraction of `maxBrakeForce` used by the full-brake level. */
  fullBrakeFraction?: number;
}

/** Coarse tier — small action set for cheap exploration far from goal /
 *  away from tight gates. */
export function coarseWheeledControls(opts: WheeledControlTierOptions): number[][] {
  const { config } = opts;
  const driveF = (opts.fullDriveFraction ?? 0.9) * config.maxDriveForce;
  const halfSteer = 0.5 * config.maxSteerAngle;
  const brakeF = (opts.fullBrakeFraction ?? 0.8) * config.maxBrakeForce;
  const actions: WheeledControls[] = [
    // Cruise straight, full throttle.
    { steer: 0, driveForce: driveF, brakeForce: 0 },
    // Gentle left / right at moderate throttle.
    { steer: +halfSteer, driveForce: 0.7 * driveF, brakeForce: 0 },
    { steer: -halfSteer, driveForce: 0.7 * driveF, brakeForce: 0 },
    // Brake straight.
    { steer: 0, driveForce: 0, brakeForce: brakeF },
    // Reverse straight (only useful for recoveries; planner usually avoids).
    { steer: 0, driveForce: -0.5 * driveF, brakeForce: 0 },
  ];
  return actions.map(encodeWheeled);
}

/** Fine tier — denser action set for precision near apexes, braking points,
 *  and obstacle clearances. IGHA*'s hysteresis logic switches to this tier
 *  near bottlenecks. */
export function fineWheeledControls(opts: WheeledControlTierOptions): number[][] {
  const { config } = opts;
  const driveF = (opts.fullDriveFraction ?? 0.9) * config.maxDriveForce;
  const brakeF = (opts.fullBrakeFraction ?? 0.8) * config.maxBrakeForce;
  const s1 = 0.25 * config.maxSteerAngle;
  const s2 = 0.5 * config.maxSteerAngle;
  const s3 = config.maxSteerAngle;
  // 5 steer levels × 3 effort levels = 15 actions.
  const steerLevels = [-s3, -s2, -s1, 0, +s1, +s2, +s3];
  const efforts: Array<{ drive: number; brake: number }> = [
    { drive: driveF, brake: 0 },
    { drive: 0, brake: 0 }, // coast
    { drive: 0, brake: brakeF },
  ];
  const actions: WheeledControls[] = [];
  // Take a sparser steer set for full-throttle to avoid silly extreme-grip
  // combinations and keep the fine tier at ~15-18 actions.
  for (const eff of efforts) {
    const usedSteer = eff.drive > 0 ? [-s2, -s1, 0, +s1, +s2] : steerLevels;
    for (const st of usedSteer) {
      actions.push({ steer: st, driveForce: eff.drive, brakeForce: eff.brake });
    }
  }
  return actions.map(encodeWheeled);
}

/** Default start-speed buckets. Race scenarios typically span [0..14] m/s. */
export const DEFAULT_WHEELED_START_SPEEDS = [0, 4, 8, 12];

/** Finer start-speed bucket set used at the fine tier. */
export const FINE_WHEELED_START_SPEEDS = [0, 3, 6, 9, 12, 15];
