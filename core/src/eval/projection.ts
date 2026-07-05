// Geometric projection of a pose onto a reference path (evaluation guide §9:
// "compute cross-track / heading error by geometric projection onto the
// reference, handling the path being represented as discrete points;
// interpolate — don't just snap to the nearest stored point").

import type { ReferenceTrajectory } from './reference-trajectory';
import { lerp, lerpAngle } from '../internal/math';

export interface Projection {
  /** Arc-length of the foot point along the reference (m). */
  s: number;
  /** Signed perpendicular distance from the pose to the path (m); positive to
   *  the left of the path tangent direction. */
  crossTrack: number;
  /** Index of the path segment [i, i+1] the foot point lies on. */
  segIndex: number;
  /** Interpolation parameter within the segment, in [0, 1]. */
  u: number;
  /** Path heading at the foot point (rad). */
  psiAtFoot: number;
  /** Target speed at the foot point (m/s). */
  vAtFoot: number;
}

/** Project (x, z) onto the reference polyline, returning the closest foot point
 *  with a signed cross-track distance. Scans every segment for the true closest
 *  foot (interpolated within the segment, not snapped to a vertex). */
export function projectOntoPath(
  ref: ReferenceTrajectory,
  x: number,
  z: number,
): Projection {
  const n = ref.length;
  if (n === 0) {
    return { s: 0, crossTrack: 0, segIndex: 0, u: 0, psiAtFoot: 0, vAtFoot: 0 };
  }
  if (n === 1) {
    const p = ref[0]!;
    return {
      s: p.s,
      crossTrack: Math.hypot(x - p.x, z - p.z),
      segIndex: 0,
      u: 0,
      psiAtFoot: p.psi,
      vAtFoot: p.v,
    };
  }

  let bestDist = Infinity;
  let bestSeg = 0;
  let bestU = 0;
  let bestFx = ref[0]!.x;
  let bestFz = ref[0]!.z;

  for (let i = 0; i < n - 1; i++) {
    const a = ref[i]!;
    const b = ref[i + 1]!;
    const dx = b.x - a.x;
    const dz = b.z - a.z;
    const lenSq = dx * dx + dz * dz;
    let u = 0;
    if (lenSq > 1e-12) {
      u = ((x - a.x) * dx + (z - a.z) * dz) / lenSq;
      if (u < 0) u = 0;
      else if (u > 1) u = 1;
    }
    const fx = a.x + dx * u;
    const fz = a.z + dz * u;
    const d = Math.hypot(x - fx, z - fz);
    if (d < bestDist) {
      bestDist = d;
      bestSeg = i;
      bestU = u;
      bestFx = fx;
      bestFz = fz;
    }
  }

  const a = ref[bestSeg]!;
  const b = ref[bestSeg + 1]!;
  // Signed cross-track: sign from the 2D cross product of the segment tangent
  // and the vector from the segment start to the query point (positive ⇒ the
  // point lies to the left of the path direction).
  const tx = b.x - a.x;
  const tz = b.z - a.z;
  const cross = tx * (z - a.z) - tz * (x - a.x);
  const sign = cross >= 0 ? 1 : -1;

  return {
    s: lerp(a.s, b.s, bestU),
    crossTrack: sign * bestDist,
    segIndex: bestSeg,
    u: bestU,
    psiAtFoot: lerpAngle(a.psi, b.psi, bestU),
    vAtFoot: lerp(a.v, b.v, bestU),
  };
}
