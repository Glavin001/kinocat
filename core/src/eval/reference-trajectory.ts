// Reference-trajectory artifact — the explicit seam between planner and
// controller (evaluation guide §3). A plan in kinocat is a bare
// `CarKinematicState[]`; this module derives the richer per-point fields the
// feasibility check (§4.1) and the cross-track / heading error computation
// (§4.2) fall out of: arc-length `s`, curvature `kappa`, and longitudinal
// acceleration `a`. Geometry is reused, not recomputed: curvature comes from
// the existing `curvaturePerSample` (speed-profile.ts).

import type { CarKinematicState } from '../agent/types';
import { curvaturePerSample } from '../execute/speed-profile';
import { lerp, lerpAngle } from '../internal/math';

/** One sample of a reference trajectory (SI units, planning plane XZ). */
export interface ReferencePoint {
  /** Arc-length distance along the path from the start (m). */
  s: number;
  x: number;
  z: number;
  /** Heading / path tangent (rad). */
  psi: number;
  /** Curvature magnitude (1/m) = 1/radius. */
  kappa: number;
  /** Target speed at this point (m/s), signed (negative = reverse). */
  v: number;
  /** Target longitudinal acceleration (m/s²). */
  a: number;
}

export type ReferenceTrajectory = ReferencePoint[];

/** Build a reference trajectory from a plan path. `s` accumulates arc-length,
 *  `kappa` reuses the Menger-formula `curvaturePerSample`, `a` is a central
 *  difference of `v` over the per-sample time `t` (falling back to dv/ds·v when
 *  the time stamps are degenerate). */
export function toReferenceTrajectory(
  path: ReadonlyArray<CarKinematicState>,
): ReferenceTrajectory {
  const n = path.length;
  if (n === 0) return [];
  const kappa = curvaturePerSample(path);

  const s = new Array<number>(n).fill(0);
  for (let i = 1; i < n; i++) {
    s[i] = s[i - 1]! + Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z);
  }

  const out: ReferencePoint[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = path[i]!;
    out[i] = {
      s: s[i]!,
      x: p.x,
      z: p.z,
      psi: p.heading,
      kappa: Math.abs(kappa[i] ?? 0),
      v: p.speed,
      a: 0,
    };
  }

  // Longitudinal acceleration via central difference of speed over time.
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - 1);
    const hi = Math.min(n - 1, i + 1);
    const dtSpan = (path[hi]!.t ?? 0) - (path[lo]!.t ?? 0);
    if (dtSpan > 1e-6) {
      out[i]!.a = (path[hi]!.speed - path[lo]!.speed) / dtSpan;
    } else {
      // Fall back to v·dv/ds when the plan carries no usable timing.
      const ds = s[hi]! - s[lo]!;
      out[i]!.a = ds > 1e-6 ? path[i]!.speed * ((path[hi]!.speed - path[lo]!.speed) / ds) : 0;
    }
  }
  return out;
}

/** Total arc length of a reference trajectory (m). */
export function referenceLength(ref: ReferenceTrajectory): number {
  return ref.length === 0 ? 0 : ref[ref.length - 1]!.s;
}

/** Interpolate a reference point at arc-length `s` (clamped to the ends). */
export function referencePoseAt(ref: ReferenceTrajectory, s: number): ReferencePoint | null {
  const n = ref.length;
  if (n === 0) return null;
  const first = ref[0]!;
  const last = ref[n - 1]!;
  if (s <= first.s) return { ...first };
  if (s >= last.s) return { ...last };
  for (let i = 0; i < n - 1; i++) {
    const a = ref[i]!;
    const b = ref[i + 1]!;
    if (s >= a.s && s <= b.s) {
      const span = b.s - a.s;
      const u = span > 1e-12 ? (s - a.s) / span : 0;
      return {
        s,
        x: lerp(a.x, b.x, u),
        z: lerp(a.z, b.z, u),
        psi: lerpAngle(a.psi, b.psi, u),
        kappa: lerp(a.kappa, b.kappa, u),
        v: lerp(a.v, b.v, u),
        a: lerp(a.a, b.a, u),
      };
    }
  }
  return { ...last };
}
