import type { Direction, ReferencePoint, Segment } from './types';

/** Speed deadband (m/s) below which a sample is treated as "stopped" for
 *  gear purposes. Matches the demo's `splitAtGearCusps` threshold so the
 *  segmentation produced here is identical to the established pipeline. */
const GEAR_EPS = 1e-3;

/** Split a reference polyline into single-gear segments at every forward↔
 *  reverse cusp (sign flip of `vRef`). Each segment is `{startIdx, endIdx,
 *  direction}` with inclusive indices into `points`; adjacent segments share
 *  the cusp index. Near-zero samples (`|vRef| ≤ GEAR_EPS`, e.g. the rest
 *  sample at a cusp or terminal) take the enclosing segment's gear rather
 *  than starting a spurious new segment.
 *
 *  A plan with no cusps returns a single segment spanning the whole polyline.
 *  Inputs shorter than two points return `[]` (no traversable segment). */
export function segmentByGear(points: ReadonlyArray<ReferencePoint>): Segment[] {
  const n = points.length;
  if (n < 2) return [];

  const out: Segment[] = [];
  let startIdx = 0;
  // Segment gear is taken from the first sample whose speed clears the
  // deadband; until then we don't know the gear, so seed forward and let
  // the first decisive sample set it.
  let dir: Direction = points[0]!.vRef < -GEAR_EPS ? -1 : 1;
  let dirKnown = Math.abs(points[0]!.vRef) > GEAR_EPS;

  for (let i = 1; i < n; i++) {
    const v = points[i]!.vRef;
    if (Math.abs(v) <= GEAR_EPS) continue; // stopped: inherit current gear.
    const sampleDir: Direction = v < 0 ? -1 : 1;
    if (!dirKnown) {
      dir = sampleDir;
      dirKnown = true;
      continue;
    }
    if (sampleDir !== dir) {
      // Cusp at i: close the current segment here (shared boundary), open
      // the next in the new gear.
      out.push({ startIdx, endIdx: i, direction: dir });
      startIdx = i;
      dir = sampleDir;
    }
  }
  out.push({ startIdx, endIdx: n - 1, direction: dir });
  return out;
}
