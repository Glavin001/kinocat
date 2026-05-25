// Friction-circle aware speed-profile smoother.
//
// Apollo / Autoware style: after a primitive-based planner returns a path,
// re-derive the velocity profile so downstream curvature is respected,
// killing "hot-into-the-corner" overshoot. Three passes:
//
//   1. Curvature limit:     v_max(i) = sqrt(aLatMax / |κ_i|)
//   2. Forward (accel cap): v(i)     = min(v_max, sqrt(v(i-1)^2 + 2·aMax·ds))
//   3. Backward (brake):    v(i)     = min(v(i),  sqrt(v(i+1)^2 + 2·dMax·ds))
//
// Output is the same path with speeds replaced and times re-accumulated
// from the smoothed profile. Path geometry (x, z, heading) is preserved.
//
// Sign convention: input may include reverse segments (speed < 0). The
// smoother operates on speed magnitudes; the original sign is restored.

import type { CarKinematicState } from '../agent/types';

/** Discrete curvature at every sample of a polyline using the Menger
 *  formula on (prev, here, next). Endpoints are 0. Returns an array of
 *  the same length as `path`. Useful as `SpeedProfileOptions.curvatureOverride`
 *  when you want speeds derived from the original (sharp) primitive
 *  polyline applied to a separately-smoothed tracking polyline. */
export function curvaturePerSample(
  path: ReadonlyArray<CarKinematicState>,
): number[] {
  const n = path.length;
  const out = new Array<number>(n).fill(0);
  if (n < 3) return out;
  for (let i = 1; i < n - 1; i++) {
    const ax = path[i - 1]!.x;
    const az = path[i - 1]!.z;
    const bx = path[i]!.x;
    const bz = path[i]!.z;
    const cx = path[i + 1]!.x;
    const cz = path[i + 1]!.z;
    const a = Math.hypot(bx - ax, bz - az);
    const b = Math.hypot(cx - bx, cz - bz);
    const c = Math.hypot(cx - ax, cz - az);
    if (a < 1e-6 || b < 1e-6 || c < 1e-6) continue;
    const cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
    const area = 0.5 * Math.abs(cross);
    out[i] = (4 * area) / (a * b * c);
  }
  return out;
}

/** Resample a per-sample scalar field (e.g. curvature) from one polyline
 *  onto another by matched arc-length. Useful when post-processing has
 *  changed the sample count (e.g. trajectory smoother → dense) and we
 *  need a scalar attached to the original samples mapped onto the new
 *  ones. Both polylines are expected to cover the same physical span. */
export function resampleScalarByArcLength(
  fromPath: ReadonlyArray<CarKinematicState>,
  fromScalar: ReadonlyArray<number>,
  toPath: ReadonlyArray<CarKinematicState>,
): number[] {
  const m = fromPath.length;
  const n = toPath.length;
  if (m === 0 || n === 0) return [];
  if (m === 1) return new Array<number>(n).fill(fromScalar[0] ?? 0);
  // Cumulative arc lengths on both.
  const fromCum: number[] = [0];
  for (let i = 1; i < m; i++) {
    fromCum.push(fromCum[i - 1]! + Math.hypot(
      fromPath[i]!.x - fromPath[i - 1]!.x,
      fromPath[i]!.z - fromPath[i - 1]!.z,
    ));
  }
  const fromTotal = fromCum[m - 1]!;
  const toCum: number[] = [0];
  for (let i = 1; i < n; i++) {
    toCum.push(toCum[i - 1]! + Math.hypot(
      toPath[i]!.x - toPath[i - 1]!.x,
      toPath[i]!.z - toPath[i - 1]!.z,
    ));
  }
  const toTotal = toCum[n - 1]!;
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    // Map this `to` sample's normalized arc onto the `from` polyline.
    const u = toTotal > 1e-9 ? toCum[i]! / toTotal : 0;
    const targetS = u * fromTotal;
    let j = 0;
    while (j < m - 2 && fromCum[j + 1]! < targetS) j++;
    const seg = fromCum[j + 1]! - fromCum[j]!;
    const t = seg > 1e-9 ? (targetS - fromCum[j]!) / seg : 0;
    const a = fromScalar[j] ?? 0;
    const b = fromScalar[j + 1] ?? a;
    // For curvature take the MAX (worst-case) within the matched segment
    // rather than linear interp — keeps the safety bound tight.
    out[i] = Math.max(a, b) * (1 - 0) + 0 * t;
  }
  return out;
}

export interface SpeedProfileOptions {
  /** Max lateral accel for curvature speed cap (m/s²). */
  aLatMax: number;
  /** Max longitudinal accel for the forward pass (m/s²). */
  aLonMaxAccel: number;
  /** Max longitudinal decel for the backward pass (m/s², positive). */
  aLonMaxDecel: number;
  /**
   * Don't lower the entry speed below the current sample's speed — the
   * vehicle is *already* travelling at this speed, the planner cannot
   * teleport it slower. Default true.
   */
  honorEntrySpeed?: boolean;
  /** Optional cap on speed magnitude (cruise / posted limit). */
  maxSpeed?: number;
  /** Floor on speed magnitude when curvature would force v→0. Default 0.5 m/s. */
  minSpeed?: number;
  /**
   * Optional pre-computed per-sample curvature (1/m, unsigned). When
   * supplied, the friction-circle cap uses these values instead of
   * estimating curvature from the path samples. Essential when the
   * `path` argument has already been smoothed (which under-estimates
   * the real curvature the chassis must drive) but you still want
   * speeds derived from the ORIGINAL primitive geometry. Length must
   * equal `path.length`.
   */
  curvatureOverride?: ReadonlyArray<number>;
}

function curvatureAt(
  prev: { x: number; z: number },
  mid: { x: number; z: number },
  next: { x: number; z: number },
): number {
  // Discrete curvature via the Menger formula: 4·area / (a·b·c). Returns
  // 0 for colinear or degenerate triples.
  const ax = prev.x;
  const az = prev.z;
  const bx = mid.x;
  const bz = mid.z;
  const cx = next.x;
  const cz = next.z;
  const a = Math.hypot(bx - ax, bz - az);
  const b = Math.hypot(cx - bx, cz - bz);
  const c = Math.hypot(cx - ax, cz - az);
  if (a < 1e-6 || b < 1e-6 || c < 1e-6) return 0;
  // Twice the signed triangle area (cross product).
  const cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
  const area = 0.5 * Math.abs(cross);
  return (4 * area) / (a * b * c);
}

/** Returns a copy of `path` with speeds replaced by a friction-circle-aware
 *  forward/backward smoothed profile, and `t` re-accumulated from the new
 *  speeds. Path geometry (x, z, heading) is preserved sample-by-sample. */
export function smoothSpeedProfile(
  path: ReadonlyArray<CarKinematicState>,
  opts: SpeedProfileOptions,
): CarKinematicState[] {
  const n = path.length;
  if (n < 2) return path.map((p) => ({ ...p }));

  const aLat = Math.max(opts.aLatMax, 0.1);
  const aAcc = Math.max(opts.aLonMaxAccel, 0.1);
  const aDec = Math.max(opts.aLonMaxDecel, 0.1);
  const vMax = opts.maxSpeed ?? Infinity;
  const vMin = Math.max(opts.minSpeed ?? 0.5, 0);
  const honorEntry = opts.honorEntrySpeed ?? true;

  // Original signs (forward vs reverse). We smooth magnitudes only and
  // restore signs at the end.
  const sign = new Array<number>(n);
  for (let i = 0; i < n; i++) sign[i] = (path[i]!.speed ?? 0) < 0 ? -1 : 1;

  // Curvature speed cap per sample. Prefer caller-supplied curvature
  // (so we can compute the cap from the ORIGINAL primitive geometry
  // even when `path` here has been smoothed downstream).
  const vCap = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let k: number;
    if (opts.curvatureOverride && opts.curvatureOverride.length === n) {
      k = Math.abs(opts.curvatureOverride[i] ?? 0);
    } else if (i === 0 || i === n - 1) {
      k = 0;
    } else {
      k = curvatureAt(path[i - 1]!, path[i]!, path[i + 1]!);
    }
    const vK = k > 1e-6 ? Math.sqrt(aLat / k) : Infinity;
    vCap[i] = Math.max(Math.min(vK, vMax), vMin);
  }

  const v = new Array<number>(n);
  const entryMag = Math.abs(path[0]!.speed ?? 0);
  v[0] = honorEntry ? Math.max(entryMag, vMin) : Math.min(vCap[0]!, vMax);

  // Forward pass — limit by acceleration capability.
  for (let i = 1; i < n; i++) {
    const ds = Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z);
    const vAccel = Math.sqrt(v[i - 1]! * v[i - 1]! + 2 * aAcc * ds);
    v[i] = Math.min(vCap[i]!, vAccel, vMax);
  }

  // Backward pass — limit by braking capability.
  for (let i = n - 2; i >= 0; i--) {
    const ds = Math.hypot(path[i + 1]!.x - path[i]!.x, path[i + 1]!.z - path[i]!.z);
    const vBrake = Math.sqrt(v[i + 1]! * v[i + 1]! + 2 * aDec * ds);
    v[i] = Math.min(v[i]!, vBrake);
  }

  // Floor (after backward pass so true terminal stops at goals can be 0).
  if (vMin > 0) {
    for (let i = 0; i < n - 1; i++) {
      if (v[i]! < vMin) v[i] = vMin;
    }
  }

  // Re-accumulate time from the smoothed profile using the trapezoidal rule.
  const out: CarKinematicState[] = new Array(n);
  let t = path[0]!.t ?? 0;
  out[0] = { ...path[0]!, speed: sign[0]! * v[0]!, t };
  for (let i = 1; i < n; i++) {
    const ds = Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z);
    const vAvg = Math.max(0.5 * (v[i - 1]! + v[i]!), vMin);
    t += ds / Math.max(vAvg, 1e-3);
    out[i] = { ...path[i]!, speed: sign[i]! * v[i]!, t };
  }
  return out;
}
