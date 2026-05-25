// Geometric trajectory smoother — turns a sparse motion-primitive polyline
// (sharp seams at primitive boundaries) into a densely-sampled, C¹-
// continuous reference trajectory.
//
// This is the missing layer between "A* over discrete primitives" and the
// tracker / speed-profile smoother. Without it the controller follows a
// piecewise-linear interpolation of primitive endpoints — the polyline
// "sharp lines" visible on the race demo, and the underestimated
// curvature at every seam.
//
// The algorithm is the standard gradient-based path smoother (DARPA
// Urban Challenge era, used in many ROS / autonomous-vehicle stacks):
//
//   1. Resample input to uniform arc-length spacing.
//   2. For `iterations` rounds, every interior point is nudged by
//      `dataWeight · (orig - here)`  (anchor to the input geometry)
//      + `smoothWeight · (0.5·(prev + next) - here)` (Laplacian pull).
//      Endpoints are pinned so the plan's start/goal don't drift.
//   3. Heading recomputed from local tangent; speed/time linearly
//      interpolated from the input along matched arc-length.
//
// What you give up: the smoother can cut tight corners by up to a few
// times the inter-sample spacing. For obstacle-rich planning the caller
// should verify the smoothed path is still collision-free (we don't
// re-check here — the race demo's planner already enforces clearance on
// the underlying primitive sweep, and the smoothing displacement is
// bounded by dataWeight). Apollo / Autoware add an explicit clearance
// constraint inside the smoother itself; that's the natural follow-up.

import type { CarKinematicState } from '../agent/types';

export interface TrajectorySmoothOptions {
  /** Uniform arc-length spacing of the output polyline (m). Default 0.4. */
  sampleSpacing?: number;
  /** Number of Laplacian-smoothing iterations. Default 20. */
  iterations?: number;
  /** Pull-toward-original weight (∈[0,1]). Higher = stays closer to the
   *  A* polyline. Default 0.5. */
  dataWeight?: number;
  /** Pull-toward-midpoint-of-neighbours weight (∈[0,1]). Higher = smoother
   *  but more corner-cutting. Default 0.3. */
  smoothWeight?: number;
  /** Optional cap on point displacement (m) per iteration. Prevents the
   *  smoother from runaway when given pathological input. Default 0.5. */
  maxStep?: number;
  /** Anchor the start and end points (don't move them). Default true. */
  anchorEndpoints?: boolean;
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/** Linearly resample a polyline to uniform arc-length `spacing`. Preserves
 *  the start and end points; the last sample lands on the goal. */
function resampleByArcLength(
  path: ReadonlyArray<CarKinematicState>,
  spacing: number,
): CarKinematicState[] {
  const n = path.length;
  if (n < 2) return path.map((p) => ({ ...p }));
  // Cumulative arc length along the input polyline.
  const cum: number[] = [0];
  for (let i = 1; i < n; i++) {
    cum.push(cum[i - 1]! + Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z));
  }
  const total = cum[n - 1]!;
  if (total < spacing * 1.5) {
    // Path is shorter than ~one sample: return endpoints, nothing to do.
    return [{ ...path[0]! }, { ...path[n - 1]! }];
  }
  const numOut = Math.max(2, Math.ceil(total / spacing) + 1);
  const stepLen = total / (numOut - 1);
  const out: CarKinematicState[] = [];
  let j = 0;
  for (let i = 0; i < numOut; i++) {
    const targetS = i === numOut - 1 ? total : i * stepLen;
    while (j < n - 2 && cum[j + 1]! < targetS) j++;
    const segLen = cum[j + 1]! - cum[j]!;
    const u = segLen > 1e-9 ? (targetS - cum[j]!) / segLen : 0;
    const a = path[j]!;
    const b = path[j + 1]!;
    const dh = wrapPi(b.heading - a.heading);
    out.push({
      x: a.x + (b.x - a.x) * u,
      z: a.z + (b.z - a.z) * u,
      heading: wrapPi(a.heading + dh * u),
      speed: a.speed + (b.speed - a.speed) * u,
      t: a.t + (b.t - a.t) * u,
    });
  }
  return out;
}

/** Geometric trajectory smoother. Returns a dense, C¹-continuous polyline
 *  that stays close to `path` (Laplacian smoothing with original-position
 *  anchor). Heading is recomputed from tangents; speeds and times are
 *  arc-length-interpolated from the input so a downstream speed-profile
 *  pass has accurate curvature estimates to work with. */
export function smoothTrajectory(
  path: ReadonlyArray<CarKinematicState>,
  opts: TrajectorySmoothOptions = {},
): CarKinematicState[] {
  const spacing = opts.sampleSpacing ?? 0.4;
  const iters = opts.iterations ?? 20;
  const wData = opts.dataWeight ?? 0.5;
  const wSmooth = opts.smoothWeight ?? 0.3;
  const maxStep = opts.maxStep ?? 0.5;
  const anchor = opts.anchorEndpoints ?? true;

  if (path.length < 2) return path.map((p) => ({ ...p }));

  // 1. Resample to uniform arc-length spacing.
  const orig = resampleByArcLength(path, spacing);
  const n = orig.length;
  if (n < 3) return orig;

  // 2. Iterative Laplacian smoothing with original-position anchor.
  const sx = new Float64Array(n);
  const sz = new Float64Array(n);
  const ox = new Float64Array(n);
  const oz = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sx[i] = orig[i]!.x;
    sz[i] = orig[i]!.z;
    ox[i] = orig[i]!.x;
    oz[i] = orig[i]!.z;
  }
  for (let it = 0; it < iters; it++) {
    for (let i = 1; i < n - 1; i++) {
      const lapx = 0.5 * (sx[i - 1]! + sx[i + 1]!) - sx[i]!;
      const lapz = 0.5 * (sz[i - 1]! + sz[i + 1]!) - sz[i]!;
      const datx = ox[i]! - sx[i]!;
      const datz = oz[i]! - sz[i]!;
      let dx = wSmooth * lapx + wData * datx;
      let dz = wSmooth * lapz + wData * datz;
      const stepLen = Math.hypot(dx, dz);
      if (stepLen > maxStep) {
        const k = maxStep / stepLen;
        dx *= k;
        dz *= k;
      }
      sx[i]! += dx;
      sz[i]! += dz;
    }
    if (!anchor) {
      // Optionally let the endpoints drift toward the smoothed neighbours.
      // Off by default — keep start/goal pinned.
    }
  }

  // 3. Recompute heading from local tangent; re-interpolate speed/time
  //    from the input by matched arc-length.
  const out: CarKinematicState[] = new Array(n);
  // Build smoothed cumulative arc-length for time interpolation.
  const smCum: number[] = [0];
  for (let i = 1; i < n; i++) {
    smCum.push(smCum[i - 1]! + Math.hypot(sx[i]! - sx[i - 1]!, sz[i]! - sz[i - 1]!));
  }
  const smTotal = smCum[n - 1]!;
  // Map smoothed arc-length back to the *input* polyline to read speed/time.
  // Speeds are restored from the input by matched arc parameter; the
  // downstream speed-profile pass will replace them with a curvature-
  // respecting profile anyway, but having a reasonable starting point is
  // useful for callers that skip the speed pass.
  const inputCum: number[] = [0];
  for (let i = 1; i < path.length; i++) {
    inputCum.push(
      inputCum[i - 1]! + Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z),
    );
  }
  const inputTotal = inputCum[path.length - 1]!;
  for (let i = 0; i < n; i++) {
    // Heading from tangent.
    let hx: number;
    let hz: number;
    if (i === 0) {
      hx = sx[1]! - sx[0]!;
      hz = sz[1]! - sz[0]!;
    } else if (i === n - 1) {
      hx = sx[n - 1]! - sx[n - 2]!;
      hz = sz[n - 1]! - sz[n - 2]!;
    } else {
      hx = sx[i + 1]! - sx[i - 1]!;
      hz = sz[i + 1]! - sz[i - 1]!;
    }
    const heading =
      Math.abs(hx) + Math.abs(hz) > 1e-9 ? Math.atan2(hz, hx) : orig[i]!.heading;
    // Speed/time at matched arc parameter on the input polyline.
    const sFrac = smTotal > 1e-9 ? smCum[i]! / smTotal : 0;
    const sInput = sFrac * inputTotal;
    // Find bracket.
    let j = 0;
    while (j < path.length - 2 && inputCum[j + 1]! < sInput) j++;
    const segLen = inputCum[j + 1]! - inputCum[j]!;
    const u = segLen > 1e-9 ? (sInput - inputCum[j]!) / segLen : 0;
    const speed = path[j]!.speed + (path[j + 1]!.speed - path[j]!.speed) * u;
    const t = path[j]!.t + (path[j + 1]!.t - path[j]!.t) * u;
    out[i] = { x: sx[i]!, z: sz[i]!, heading, speed, t };
  }
  return out;
}
