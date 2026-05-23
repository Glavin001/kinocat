// Quantitative diagnostics on motion-primitive libraries — answers the
// "is the action space good enough resolution?" question with numbers
// you can read off, instead of pure eyeballing of the fan plot.
//
// All pure functions. No React, no DOM.

import type { MotionPrimitive } from 'kinocat/primitives';

/** Endpoints of forward primitives only — reverse primitives are usually
 *  recovery actions whose endpoint angles aren't part of the racing-line
 *  reachable space. Returns angle (rad) of the endpoint vector from origin. */
function forwardEndpointAngles(primitives: ReadonlyArray<MotionPrimitive>): number[] {
  const angles: number[] = [];
  for (const p of primitives) {
    if (p.reverse) continue;
    const dx = p.end.dx;
    const dz = p.end.dz;
    if (Math.hypot(dx, dz) < 1e-6) continue;
    angles.push(Math.atan2(dz, dx));
  }
  return angles;
}

/** Sorted angular gaps (degrees) between consecutive forward-primitive
 *  endpoints, swept around the origin. A large gap means the planner
 *  cannot aim its trajectory in that direction in a single primitive —
 *  a hole in the action space.
 *
 *  Returns one fewer entry than `forwardEndpointAngles(primitives)` since
 *  it's pairwise; the largest gap typically appears at the "back" of the
 *  fan where the forward-primitives don't reach. */
export function endpointAngularGaps(
  primitives: ReadonlyArray<MotionPrimitive>,
): number[] {
  const angles = forwardEndpointAngles(primitives).sort((a, b) => a - b);
  // A single endpoint is "all 360° gap" — the user can only steer in one
  // direction, so the action space is degenerate. Surface that explicitly.
  if (angles.length === 1) return [360];
  if (angles.length === 0) return [];
  const gaps: number[] = [];
  for (let i = 0; i < angles.length - 1; i++) {
    gaps.push(((angles[i + 1]! - angles[i]!) * 180) / Math.PI);
  }
  // Wrap gap (last → first, going through ±π).
  const wrap = 2 * Math.PI - (angles[angles.length - 1]! - angles[0]!);
  gaps.push((wrap * 180) / Math.PI);
  return gaps.sort((a, b) => a - b);
}

/** Max angular gap (degrees) — convenience for diagnostics summary. */
export function maxEndpointAngularGap(
  primitives: ReadonlyArray<MotionPrimitive>,
): number {
  const gaps = endpointAngularGaps(primitives);
  return gaps.length > 0 ? gaps[gaps.length - 1]! : 0;
}

/** Convex hull area (m²) of forward-primitive endpoints. A LARGER hull
 *  means the planner can place the chassis in more places in `duration`
 *  seconds — a wider, more capable action space. */
export function reachableHullArea(
  primitives: ReadonlyArray<MotionPrimitive>,
): number {
  const pts: [number, number][] = [];
  for (const p of primitives) {
    if (p.reverse) continue;
    pts.push([p.end.dx, p.end.dz]);
  }
  if (pts.length < 3) return 0;
  const hull = convexHull(pts);
  return polygonArea(hull);
}

/** Andrew's monotone chain convex hull, returns hull vertices in CCW order. */
function convexHull(points: [number, number][]): [number, number][] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i]!;
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function polygonArea(vertices: [number, number][]): number {
  if (vertices.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < vertices.length; i++) {
    const [xa, ya] = vertices[i]!;
    const [xb, yb] = vertices[(i + 1) % vertices.length]!;
    s += xa * yb - xb * ya;
  }
  return Math.abs(s) / 2;
}

export interface PrimitiveMismatch {
  /** Index into the (start-speed bucket) primitive list. */
  index: number;
  /** Controls vector (same in both libraries by construction). */
  controls: number[];
  /** Euclidean distance between the two libraries' endpoint predictions (m). */
  distance: number;
  /** End-state offset under library A. */
  endA: { dx: number; dz: number };
  /** End-state offset under library B. */
  endB: { dx: number; dz: number };
  /** Whether the primitive is reverse-gear. */
  reverse: boolean;
}

/** Pairwise endpoint mismatches between two libraries that were built
 *  from the IDENTICAL control set (i.e. one is kinematic, one is
 *  learned, both compiled from `raceControlSets` via `characterizeVehicle`).
 *
 *  Assumes both inputs are the primitives within the SAME start-speed
 *  bucket (caller should filter via `library.lookup(speed)`), and that
 *  control indexing matches. Used by the explorer page to surface
 *  "where do kinematic and v2 disagree most?" — directly answers the
 *  user's question about why v2 drives the course differently. */
export function pairwiseEndpointMismatch(
  primitivesA: ReadonlyArray<MotionPrimitive>,
  primitivesB: ReadonlyArray<MotionPrimitive>,
): PrimitiveMismatch[] {
  // Build a control-vector → primitive map for B so we can match A→B by
  // control identity (more robust than positional indexing if either
  // library re-orders).
  const keyOf = (c: number[]) => c.map((x) => Number(x.toFixed(4))).join('|');
  const byKey = new Map<string, MotionPrimitive>();
  for (const p of primitivesB) {
    byKey.set(keyOf(p.controls), p);
  }
  const out: PrimitiveMismatch[] = [];
  for (let i = 0; i < primitivesA.length; i++) {
    const a = primitivesA[i]!;
    const b = byKey.get(keyOf(a.controls));
    if (!b) continue;
    const dx = a.end.dx - b.end.dx;
    const dz = a.end.dz - b.end.dz;
    out.push({
      index: i,
      controls: a.controls,
      distance: Math.hypot(dx, dz),
      endA: { dx: a.end.dx, dz: a.end.dz },
      endB: { dx: b.end.dx, dz: b.end.dz },
      reverse: a.reverse,
    });
  }
  return out;
}

/** Summary statistics for one library + (optional) comparison library.
 *  Drives the diagnostics row in the explorer. */
export interface LibraryDiagnostics {
  count: number;
  forwardCount: number;
  reverseCount: number;
  maxAngularGapDeg: number;
  hullAreaM2: number;
  /** Bounding box of forward endpoints (helps the user see "in this many
   *  metres can I steer the chassis"). */
  forwardEndpointBBox: { xMin: number; xMax: number; zMin: number; zMax: number };
  /** Only when a comparison library was provided (paired by control). */
  pairedMismatches?: PrimitiveMismatch[];
  /** Mean / max mismatch (m) for quick HUD. */
  meanMismatch?: number;
  maxMismatch?: number;
  largestMismatch?: PrimitiveMismatch;
}

export function diagnoseLibrary(
  primitives: ReadonlyArray<MotionPrimitive>,
  compareAgainst?: ReadonlyArray<MotionPrimitive>,
): LibraryDiagnostics {
  let forwardCount = 0;
  let reverseCount = 0;
  let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (const p of primitives) {
    if (p.reverse) {
      reverseCount++;
    } else {
      forwardCount++;
      xMin = Math.min(xMin, p.end.dx);
      xMax = Math.max(xMax, p.end.dx);
      zMin = Math.min(zMin, p.end.dz);
      zMax = Math.max(zMax, p.end.dz);
    }
  }
  const out: LibraryDiagnostics = {
    count: primitives.length,
    forwardCount,
    reverseCount,
    maxAngularGapDeg: maxEndpointAngularGap(primitives),
    hullAreaM2: reachableHullArea(primitives),
    forwardEndpointBBox: forwardCount > 0
      ? { xMin, xMax, zMin, zMax }
      : { xMin: 0, xMax: 0, zMin: 0, zMax: 0 },
  };
  if (compareAgainst) {
    const mismatches = pairwiseEndpointMismatch(primitives, compareAgainst);
    out.pairedMismatches = mismatches;
    if (mismatches.length > 0) {
      let sum = 0;
      let mx = 0;
      let mxRef: PrimitiveMismatch | undefined;
      for (const m of mismatches) {
        sum += m.distance;
        if (m.distance > mx) { mx = m.distance; mxRef = m; }
      }
      out.meanMismatch = sum / mismatches.length;
      out.maxMismatch = mx;
      out.largestMismatch = mxRef;
    }
  }
  return out;
}
