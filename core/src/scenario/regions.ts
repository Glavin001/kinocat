// Static (time-invariant) region constructors. Each returns a `Region` whose
// `kind` tag keeps it serializable + visualizable. Membership uses the
// zero-dependency planar geometry in `internal/geom.ts`; `costToGo` is the
// admissible, HEADING-AWARE Reeds-Shepp lower bound where orientation matters,
// and the cheaper Euclidean bound where it does not.

import type { Region, ScenarioState } from './types';
import {
  pointInPolygon,
  segmentsIntersect,
  pointSegmentDistance,
  type Pt,
} from '../internal/geom';
import { angleDiff } from '../internal/math';
import { reedsSheppShortestPath } from '../curves/reeds-shepp';

/** Default min turning radius used by heading-aware `costToGo` bounds. Real
 *  planning radius is the agent's; using a small radius here keeps the bound
 *  ADMISSIBLE (a tighter-turning car can only be cheaper). */
const RS_RADIUS = 0.5;

function rsCost(
  from: ScenarioState,
  gx: number,
  gz: number,
  gtheta: number,
): number {
  // curves plane maps (x, y=z, theta=heading).
  return reedsSheppShortestPath(
    { x: from.x, y: from.z, theta: from.heading },
    { x: gx, y: gz, theta: gtheta },
    RS_RADIUS,
  ).length;
}

function st(x: number, z: number, heading = 0): ScenarioState {
  return { x, z, heading, speed: 0, t: 0 };
}

// ---------------------------------------------------------------------------

export interface Pose {
  x: number;
  z: number;
  heading: number;
}
export interface PoseMargins {
  dx: number;
  dz: number;
  /** Heading half-tolerance, radians. Omit / Infinity = any heading. */
  dheading?: number;
}

/** A pose box in SE(2): |x-x0|<dx & |z-z0|<dz & |angleDiff(theta,theta0)|<dheading.
 *  The canonical "park here, aligned" / "dock at this pose" region. */
export function at(pose: Pose, m: PoseMargins): Region {
  const dh = m.dheading ?? Infinity;
  return {
    kind: 'at',
    key: `at:${pose.x},${pose.z},${pose.heading},${m.dx},${m.dz},${dh}`,
    dynamic: false,
    contains(s) {
      return (
        Math.abs(s.x - pose.x) <= m.dx &&
        Math.abs(s.z - pose.z) <= m.dz &&
        Math.abs(angleDiff(pose.heading, s.heading)) <= dh
      );
    },
    costToGo(s) {
      return rsCost(s, pose.x, pose.z, pose.heading);
    },
    representative() {
      return st(pose.x, pose.z, pose.heading);
    },
  };
}

/** A ball of radius `r` around a point, any heading. Point-to-point goals. */
export function near(point: { x: number; z: number }, r: number): Region {
  return {
    kind: 'near',
    key: `near:${point.x},${point.z},${r}`,
    dynamic: false,
    contains(s) {
      return Math.hypot(s.x - point.x, s.z - point.z) <= r;
    },
    costToGo(s) {
      // Heading-agnostic: Euclidean to the ball surface (never an over-estimate).
      return Math.max(0, Math.hypot(s.x - point.x, s.z - point.z) - r);
    },
    representative() {
      return st(point.x, point.z, 0);
    },
  };
}

/** An area (any heading): inside the given polygon ([x,z] vertices). */
export function inside(polygon: ReadonlyArray<Pt>): Region {
  // centroid as representative + a bounding circle for the costToGo LB.
  let cx = 0;
  let cz = 0;
  for (const [x, z] of polygon) {
    cx += x;
    cz += z;
  }
  cx /= polygon.length;
  cz /= polygon.length;
  let rad = 0;
  for (const [x, z] of polygon) {
    rad = Math.max(rad, Math.hypot(x - cx, z - cz));
  }
  return {
    kind: 'inside',
    key: `inside:${polygon.map((p) => `${p[0]},${p[1]}`).join(';')}`,
    dynamic: false,
    contains(s) {
      return pointInPolygon(s.x, s.z, polygon);
    },
    costToGo(s) {
      if (pointInPolygon(s.x, s.z, polygon)) return 0;
      // LB: distance to the polygon's bounding circle around its centroid.
      return Math.max(0, Math.hypot(s.x - cx, s.z - cz) - rad);
    },
    representative() {
      return st(cx, cz, 0);
    },
  };
}

/** Direction a gate must be crossed. +1 = a->b normal side, -1 = opposite. */
export type GateDir = 1 | -1;
export const FORWARD: GateDir = 1;
export const BACKWARD: GateDir = -1;

/** An oriented segment a->b. Membership is "near the segment" (within `half`
 *  of the line); the real semantics is `crossed()` with a direction sign — the
 *  edge must pass through the segment moving with the requested normal sign. */
export function gate(
  a: { x: number; z: number },
  b: { x: number; z: number },
  dir: GateDir = FORWARD,
  half = 1.5,
): Region {
  const mx = (a.x + b.x) / 2;
  const mz = (a.z + b.z) / 2;
  // Gate normal (rotate the a->b segment by +90deg). `dir` picks which way the
  // crossing must go.
  const ex = b.x - a.x;
  const ez = b.z - a.z;
  const nlen = Math.hypot(ex, ez) || 1;
  // Normal = segment a->b rotated -90deg (CW), so a "left-to-right" gate
  // (a below, b above) has its FORWARD normal pointing +x.
  const nx = ez / nlen;
  const nz = -ex / nlen;
  const repHeading = Math.atan2(dir * nz, dir * nx);
  return {
    kind: 'gate',
    key: `gate:${a.x},${a.z},${b.x},${b.z},${dir},${half}`,
    dynamic: false,
    contains(s) {
      return pointSegmentDistance(s.x, s.z, a.x, a.z, b.x, b.z) <= half;
    },
    crossed(from, to) {
      if (!segmentsIntersect(from.x, from.z, to.x, to.z, a.x, a.z, b.x, b.z)) {
        return false;
      }
      // Require the motion to advance along the gate's signed normal.
      const dot = (to.x - from.x) * nx + (to.z - from.z) * nz;
      return dir > 0 ? dot > 0 : dot < 0;
    },
    costToGo(s) {
      // Heading-aware: arrive at the gate midpoint pointing through it.
      return rsCost(s, mx, mz, repHeading);
    },
    representative() {
      return st(mx, mz, repHeading);
    },
  };
}

/** A track / lane tube of half-width `width/2` around a polyline centerline. */
export function corridor(
  centerline: ReadonlyArray<{ x: number; z: number }>,
  width: number,
): Region {
  const half = width / 2;
  const minDist = (px: number, pz: number): number => {
    let d = Infinity;
    for (let i = 0; i + 1 < centerline.length; i++) {
      const c0 = centerline[i]!;
      const c1 = centerline[i + 1]!;
      d = Math.min(d, pointSegmentDistance(px, pz, c0.x, c0.z, c1.x, c1.z));
    }
    return d;
  };
  const first = centerline[0] ?? { x: 0, z: 0 };
  return {
    kind: 'corridor',
    key: `corridor:${width}:${centerline.map((p) => `${p.x},${p.z}`).join(';')}`,
    dynamic: false,
    contains(s) {
      return minDist(s.x, s.z) <= half;
    },
    costToGo(s) {
      return Math.max(0, minDist(s.x, s.z) - half);
    },
    representative() {
      return st(first.x, first.z, 0);
    },
  };
}

/** One side of a line through `point` with outward `normal`: the half-plane
 *  where (s - point) . normal >= 0. */
export function halfPlane(
  point: { x: number; z: number },
  normal: { x: number; z: number },
): Region {
  const nlen = Math.hypot(normal.x, normal.z) || 1;
  const nx = normal.x / nlen;
  const nz = normal.z / nlen;
  return {
    kind: 'halfPlane',
    key: `halfPlane:${point.x},${point.z},${nx},${nz}`,
    dynamic: false,
    contains(s) {
      return (s.x - point.x) * nx + (s.z - point.z) * nz >= 0;
    },
    costToGo(s) {
      const signed = (s.x - point.x) * nx + (s.z - point.z) * nz;
      return signed >= 0 ? 0 : -signed;
    },
    representative() {
      return st(point.x, point.z, Math.atan2(nz, nx));
    },
  };
}
