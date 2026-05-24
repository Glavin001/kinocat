// Generic native action shape for an Ackermann-steered wheeled vehicle.
//
// Convention for the opaque `controls: number[]` consumed by `ForwardSim`
// in this control family: `[steer, driveForce, brakeForce]`.
//
// `steer` is the actual front-wheel angle in radians (no pure-pursuit /
// curvature translation — the model learns physics directly from native
// inputs, not from the heuristic controller's quirks layered on top).
//
// `driveForce` is the signed engine force in Newtons; negative = reverse.
// `brakeForce` is the non-negative brake force in Newtons.
//
// Domain-agnostic: any consumer that drives an Ackermann vehicle (the kinocat
// Rapier adapter, an alternate physics adapter, a closed-form simulator) can
// accept these and translate them to its own wheel inputs.

export interface WheeledCarControls {
  /** Front-wheel steer angle in radians. */
  steer: number;
  /** Signed engine force in Newtons (negative = reverse). */
  driveForce: number;
  /** Brake force in Newtons (>= 0). */
  brakeForce: number;
}

/**
 * @deprecated Use `WheeledCarControls`. Retained as a structural alias so
 * existing imports keep compiling. The new name disambiguates from
 * non-wheeled vehicle controls (airplane throttle/elevator/aileron, etc.).
 */
export type WheeledControls = WheeledCarControls;

/** Dimensionality of the encoded controls vector. */
export const WHEELED_CONTROL_DIM = 3;

export function encodeWheeled(c: WheeledCarControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}

export function decodeWheeled(v: ReadonlyArray<number>): WheeledCarControls {
  return {
    steer: v[0] ?? 0,
    driveForce: v[1] ?? 0,
    brakeForce: v[2] ?? 0,
  };
}

/** Clamp a `WheeledCarControls` against an agent's physical limits. */
export function clampWheeled(
  c: WheeledCarControls,
  limits: {
    maxSteerAngle: number;
    maxDriveForce: number;
    maxBrakeForce: number;
  },
): WheeledCarControls {
  const s = Math.max(-limits.maxSteerAngle, Math.min(limits.maxSteerAngle, c.steer));
  const d = Math.max(-limits.maxDriveForce, Math.min(limits.maxDriveForce, c.driveForce));
  const b = Math.max(0, Math.min(limits.maxBrakeForce, c.brakeForce));
  return { steer: s, driveForce: d, brakeForce: b };
}
