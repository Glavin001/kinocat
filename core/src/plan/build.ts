// Thin builder: enrich an already-smoothed dense polyline into a rich Plan.
//
// This does NOT run the post-processing pipeline (Reeds-Shepp lift, geometric
// smoothing, friction-circle speed profile) — it takes their output (the
// dense `CarKinematicState[]` the controllers already consume) and annotates
// it with the per-point/segment richness the bare state array throws away.
// Reuses `curvaturePerSample` from the execute layer so curvature is computed
// exactly as the speed-profile pass does.

import type { CarKinematicState } from '../agent/types';
import { curvaturePerSample } from '../execute/speed-profile';
import { segmentByGear } from './segments';
import type { Plan, ReferencePoint } from './types';

export interface BuildPlanOptions {
  /**
   * EFFECTIVE wheelbase L (m) for the feedforward steer `steerFf = atan(L·κ)`.
   * Pass the effective wheelbase — for kinocat's car config that is
   * `2 * VehicleConfig.wheelBase`. When omitted, `steerFf` is left undefined
   * on every point (no kinematic model available to invert curvature).
   */
  wheelBase?: number;
}

/** Sign of the turn at sample `i` from the cross product of consecutive
 *  tangents in the XZ plane (positive = left turn). Endpoints and degenerate
 *  triples return +1 (their magnitude is 0 anyway). */
function turnSign(path: ReadonlyArray<CarKinematicState>, i: number): number {
  const n = path.length;
  if (i <= 0 || i >= n - 1) return 1;
  const ax = path[i - 1]!.x;
  const az = path[i - 1]!.z;
  const bx = path[i]!.x;
  const bz = path[i]!.z;
  const cx = path[i + 1]!.x;
  const cz = path[i + 1]!.z;
  const cross = (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
  return cross >= 0 ? 1 : -1;
}

/** Enrich a dense, post-processed polyline into a rich Plan. Reuses existing
 *  geometry utilities; populates everything computable today and reserves the
 *  control / dynamic / free-space slots for a future primitive upgrade. The
 *  input is never mutated. */
export function buildPlan(
  path: ReadonlyArray<CarKinematicState>,
  opts: BuildPlanOptions = {},
): Plan {
  const n = path.length;
  if (n === 0) return { points: [], segments: [] };

  const kMag = curvaturePerSample(path); // unsigned magnitude per sample.
  const L = opts.wheelBase;

  const points: ReferencePoint[] = new Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const st = path[i]!;
    if (i > 0) {
      s += Math.hypot(st.x - path[i - 1]!.x, st.z - path[i - 1]!.z);
    }
    const kappa = (kMag[i] ?? 0) * turnSign(path, i);

    // aRef = d(vRef)/dt, central difference (one-sided at the ends).
    let aRef = 0;
    if (n >= 2) {
      let dv: number;
      let dt: number;
      if (i === 0) {
        dv = path[1]!.speed - st.speed;
        dt = path[1]!.t - st.t;
      } else if (i === n - 1) {
        dv = st.speed - path[i - 1]!.speed;
        dt = st.t - path[i - 1]!.t;
      } else {
        dv = path[i + 1]!.speed - path[i - 1]!.speed;
        dt = path[i + 1]!.t - path[i - 1]!.t;
      }
      aRef = Math.abs(dt) > 1e-6 ? dv / dt : 0;
    }

    const p: ReferencePoint = {
      t: st.t,
      s,
      x: st.x,
      z: st.z,
      heading: st.heading,
      vRef: st.speed,
      kappa,
      aRef,
      // Tier 2: feedforward accel (approximated today; mirrors aRef).
      accelFf: aRef,
    };
    // Tier 3: dynamic state, only when the source state carried it.
    if (st.yawRate !== undefined) p.rRef = st.yawRate;
    if (st.lateralVelocity !== undefined) {
      p.betaRef = Math.atan2(st.lateralVelocity, Math.abs(st.speed) + 1e-6);
    }
    points[i] = p;
  }

  const segments = segmentByGear(points);

  // Feedforward steer needs the enclosing SEGMENT's gear, not sign(vRef):
  // `steerFf = atan(L·κ·dir)`. A car tracing a given geometric arc in reverse
  // must steer the opposite way from tracing it forward (the demo's chassis
  // conversion applies the same `-gear` flip), so ignoring direction reports
  // the negated steer on every reverse segment — exactly wrong for parking.
  // Deferred to a per-segment pass because a point's own vRef is ambiguous at
  // the zero-speed cusp; the segment carries the authoritative gear.
  if (L !== undefined) {
    for (const seg of segments) {
      for (let i = seg.startIdx; i <= seg.endIdx; i++) {
        points[i]!.steerFf = Math.atan(L * points[i]!.kappa * seg.direction);
      }
    }
  }

  return { points, segments };
}

/** Round-trip a Plan back to the plain `CarKinematicState[]` the controllers
 *  already accept, so producing a rich Plan never forces a consumer change.
 *  Re-attaches `yawRate` from `rRef` and reconstructs `lateralVelocity` from
 *  the sideslip angle when those were carried. */
export function toStatePath(plan: Plan): CarKinematicState[] {
  return plan.points.map((p) => {
    const st: CarKinematicState = {
      x: p.x,
      z: p.z,
      heading: p.heading,
      speed: p.vRef,
      t: p.t,
    };
    if (p.rRef !== undefined) st.yawRate = p.rRef;
    if (p.betaRef !== undefined) {
      st.lateralVelocity = Math.tan(p.betaRef) * Math.abs(p.vRef);
    }
    return st;
  });
}
