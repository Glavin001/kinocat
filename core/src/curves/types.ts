/** A pose in the planning plane. The library is plane-agnostic; callers map
 *  (x,y) to whichever world axes they use (kinocat uses world XZ). */
export interface Pose {
  x: number;
  y: number;
  /** Heading in radians. */
  theta: number;
}

/** Steering primitive of a segment: left arc, straight, or right arc. */
export type Steer = 'L' | 'S' | 'R';

/** Direction of travel along a segment. +1 forward, -1 reverse. */
export type Gear = 1 | -1;

/** One segment of an analytical car curve. `length` is the actual
 *  (un-normalized) length in world units: arc length for L/R, distance for S. */
export interface CurveSegment {
  steer: Steer;
  gear: Gear;
  length: number;
}

export type CurveKind = 'dubins' | 'reeds-shepp';

export interface CurvePath {
  kind: CurveKind;
  /** Word label, e.g. "LSL" or "LpRmSmLm"-style letters (steer letters only). */
  word: string;
  segments: CurveSegment[];
  /** Total actual length (sum of |segment| lengths). */
  length: number;
}
