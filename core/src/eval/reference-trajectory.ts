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

/** Per-sample curvature that is SPLIT at gear cusps (speed sign flips) before
 *  the Menger formula runs. At a Reeds-Shepp cusp the car reverses direction, so
 *  three consecutive samples straddle the reversal: the far side `c = |C − A|`
 *  collapses while the triangle area stays finite, and `κ = 4·area/(a·b·c)`
 *  explodes into a phantom spike (observed κ ≈ 13 m⁻¹ at a valid parking cusp),
 *  which makes `checkFeasibility` misdiagnose every real parking plan as
 *  planner-infeasible. Computing curvature within each maximal same-gear run
 *  keeps every Menger triple on one side of the cusp; the reversal sample itself
 *  (and the run endpoints) get κ = 0 — correct, since |v| → 0 there so the
 *  lateral-accel demand v²·κ vanishes regardless. */
function gearSegmentedCurvature(path: ReadonlyArray<CarKinematicState>): number[] {
  const n = path.length;
  const out = new Array<number>(n).fill(0);
  if (n < 3) return out;
  // Gear per sample: +1 forward, −1 reverse. Zero-speed samples inherit the
  // previous gear so a momentary stop mid-run doesn't fragment it, but a true
  // forward↔reverse sign change starts a new run.
  const gear = new Array<number>(n).fill(0);
  let cur = 0;
  for (let i = 0; i < n; i++) {
    const v = path[i]!.speed;
    if (v > 0) cur = 1;
    else if (v < 0) cur = -1;
    gear[i] = cur;
  }
  // Back-fill any leading zeros with the first resolved gear.
  const firstResolved = gear.find((g) => g !== 0) ?? 1;
  for (let i = 0; i < n && gear[i] === 0; i++) gear[i] = firstResolved;
  // Curvature within each maximal same-gear run only.
  let start = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || gear[i] !== gear[start]) {
      if (i - start >= 3) {
        const seg = curvaturePerSample(path.slice(start, i));
        for (let k = 0; k < seg.length; k++) out[start + k] = seg[k]!;
      }
      start = i;
    }
  }
  return out;
}


/** Build a reference trajectory from a plan path. `s` accumulates arc-length,
 *  `kappa` reuses the Menger-formula `curvaturePerSample`, `a` is a central
 *  difference of `v` over the per-sample time `t` (falling back to dv/ds·v when
 *  the time stamps are degenerate). */
export function toReferenceTrajectory(
  path: ReadonlyArray<CarKinematicState>,
): ReferenceTrajectory {
  const n = path.length;
  if (n === 0) return [];
  const kappa = gearSegmentedCurvature(path);

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
