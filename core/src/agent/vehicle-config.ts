// Generic physical configuration of an Ackermann-steered wheeled vehicle.
//
// These are the parameters a learned dynamics model can identify (or be
// conditioned on, to generalize across vehicle setups). Domain-agnostic: any
// downstream consumer (kinocat Rapier adapter, alternate physics adapter, a
// closed-form ground truth) can express its vehicle setup in these terms.
//
// All quantities are in SI units. `drivenWheels` is a small set, not a free
// number — encoded as 0/1/2 internally when needed for ML inputs.

export type DrivenWheels = 'rwd' | 'fwd' | 'awd';

export interface LearnableVehicleConfig {
  /** Chassis mass (kg). */
  chassisMass: number;
  /** Distance from chassis centre to wheel hub along the forward axis (m).
   *  Effective wheelbase between front and rear axles is `2 * wheelBase`. */
  wheelBase: number;
  /** Distance from chassis centre to wheel hub along the lateral axis (m). */
  wheelTrack: number;
  /** Wheel radius (m). */
  wheelRadius: number;
  /** Suspension spring stiffness (N/m, or solver-units). */
  suspensionStiffness: number;
  /** Tire longitudinal traction coefficient (dimensionless). */
  frictionSlip: number;
  /** Tire lateral friction stiffness multiplier (dimensionless). */
  sideFrictionStiffness: number;
  /** Max engine force on the driven wheels (N). */
  maxDriveForce: number;
  /** Max brake force per wheel (N). */
  maxBrakeForce: number;
  /** Max |front-wheel steer angle| (radians). */
  maxSteerAngle: number;
  /** Which wheels receive engine torque. */
  drivenWheels: DrivenWheels;
}

/** Sensible defaults — match the kinocat sport-car reference chassis (the
 *  raycast-vehicle adapter's `DEFAULTS`, derived for the same vehicle).
 *  chassisMass is the exact cuboid derivation (8·2.4·0.5·1.0 halfExtents
 *  volume × density 60 = 576 kg), matching `deriveLearnableConfig` — a
 *  hand-rounded 580 lived here for a while and drifted from every
 *  Rapier-derived config, including the shipped model artifact. */
export const DEFAULT_LEARNABLE_CONFIG: LearnableVehicleConfig = {
  chassisMass: 576,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionStiffness: 80,
  frictionSlip: 1.8,
  sideFrictionStiffness: 1.0,
  maxDriveForce: 4000,
  maxBrakeForce: 2000,
  maxSteerAngle: 0.6,
  drivenWheels: 'rwd',
};

/** Numeric dimensionality of the config when encoded as a vector for ML
 *  inputs. (One-hot drivenWheels → 3, plus 10 continuous = 13.)
 *  The simpler 11-dim encoding (drivenWheels as a single 0/1/2 ordinal) is
 *  also exposed for use cases where one-hot is overkill. */
export const LEARNABLE_CONFIG_VEC_DIM_ORDINAL = 11;
export const LEARNABLE_CONFIG_VEC_DIM_ONEHOT = 13;

/** Encode the config as a flat numeric vector for ML inputs. `drivenWheels`
 *  is encoded as ordinal 0=rwd, 1=fwd, 2=awd. */
export function encodeConfigOrdinal(c: LearnableVehicleConfig): number[] {
  return [
    c.chassisMass,
    c.wheelBase,
    c.wheelTrack,
    c.wheelRadius,
    c.suspensionStiffness,
    c.frictionSlip,
    c.sideFrictionStiffness,
    c.maxDriveForce,
    c.maxBrakeForce,
    c.maxSteerAngle,
    c.drivenWheels === 'rwd' ? 0 : c.drivenWheels === 'fwd' ? 1 : 2,
  ];
}

/** Encode the config with one-hot `drivenWheels` (13 dims). */
export function encodeConfigOneHot(c: LearnableVehicleConfig): number[] {
  return [
    c.chassisMass,
    c.wheelBase,
    c.wheelTrack,
    c.wheelRadius,
    c.suspensionStiffness,
    c.frictionSlip,
    c.sideFrictionStiffness,
    c.maxDriveForce,
    c.maxBrakeForce,
    c.maxSteerAngle,
    c.drivenWheels === 'rwd' ? 1 : 0,
    c.drivenWheels === 'fwd' ? 1 : 0,
    c.drivenWheels === 'awd' ? 1 : 0,
  ];
}

/** Per-component normalisation scales for the config vector — used to make
 *  ML inputs roughly unit-magnitude, so a small MLP can fit without an
 *  upfront whitening pass. Tuned to the kinocat reference chassis range. */
export const CONFIG_SCALES_ORDINAL: number[] = [
  // chassisMass     wheelBase  wheelTrack  wheelRadius  susp.stiff.
  1000,              2,         1.2,        0.5,         150,
  // frictionSlip    sideFriction maxDrive   maxBrake    maxSteer  driveTrain
  3.0,               2.0,         8000,      4000,       1.2,      2,
];
