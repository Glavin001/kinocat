import type { Direction, Segment } from './types';

/** Minimal structural view segmentByGear needs — just the signed speed. Lets
 *  callers pass a `ReferencePoint[]` OR any `{ vRef }[]` projection (e.g. a
 *  `CarKinematicState[]` mapped `speed → vRef`) without a cast. */
interface GearSample {
  vRef: number;
}

/** Speed deadband (m/s) below which a sample is treated as "stopped" for
 *  gear purposes. */
const GEAR_EPS = 1e-3;

/** Split a reference polyline into single-gear segments at every forward↔
 *  reverse cusp (sign flip of `vRef`). Each segment is `{startIdx, endIdx,
 *  direction}` with inclusive indices into `points`; adjacent segments share
 *  the cusp index.
 *
 *  The shared boundary is placed at the **rest sample** — the sample of
 *  minimum |vRef| in the transition band between the two opposite-gear runs
 *  (the pose where the chassis actually comes to rest before reversing).
 *  That is the stop target a segment-following controller brakes to, so it
 *  must be the segment endpoint, not the first sample already moving in the
 *  new gear. Near-zero samples (`|vRef| ≤ GEAR_EPS`) never open a segment on
 *  their own; they belong to the rest band.
 *
 *  A plan with no cusps returns a single segment spanning the whole polyline.
 *  Inputs shorter than two points return `[]` (no traversable segment). */
export function segmentByGear(points: ReadonlyArray<GearSample>): Segment[] {
  const n = points.length;
  if (n < 2) return [];

  const gearOf = (v: number): Direction | 0 =>
    v < -GEAR_EPS ? -1 : v > GEAR_EPS ? 1 : 0;

  const out: Segment[] = [];
  let segStart = 0;
  // Segment gear, set from the first decisive (non-rest) sample.
  let segDir: Direction | 0 = 0;
  // Last index that decisively carried `segDir` — the near end of any
  // transition band we later scan for the rest sample.
  let lastDecisive = 0;

  for (let i = 0; i < n; i++) {
    const g = gearOf(points[i]!.vRef);
    if (g === 0) continue; // rest sample: belongs to the band, never decisive.
    if (segDir === 0) {
      segDir = g;
      lastDecisive = i;
      continue;
    }
    if (g === segDir) {
      lastDecisive = i;
      continue;
    }
    // Cusp: gear flips between `lastDecisive` (segDir) and `i` (g). Place the
    // shared boundary at the rest sample — argmin |vRef| over [lastDecisive, i].
    let boundary = lastDecisive;
    let best = Math.abs(points[lastDecisive]!.vRef);
    for (let j = lastDecisive + 1; j <= i; j++) {
      const a = Math.abs(points[j]!.vRef);
      if (a < best) {
        best = a;
        boundary = j;
      }
    }
    out.push({ startIdx: segStart, endIdx: boundary, direction: segDir });
    segStart = boundary;
    segDir = g;
    lastDecisive = i;
  }

  // Trailing segment (or the whole plan when there were no cusps). An
  // all-rest plan has no decisive gear; default it to forward.
  out.push({ startIdx: segStart, endIdx: n - 1, direction: segDir === 0 ? 1 : segDir });
  return out;
}
