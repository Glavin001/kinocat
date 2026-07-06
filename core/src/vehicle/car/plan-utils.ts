// Plan polyline utilities for the car domain.
//
// `trimPlan` discards plan samples whose `t` is already in the past (relative
// to `elapsed`), keeping only the tail the pure-pursuit tracker still cares
// about. Used by every demo that follows a kinocat plan with a real chassis.

import type { CarKinematicState } from './types';
import type { MotionPrimitive } from '../../primitives/types';
import type { Node } from '../../environment/types';
import { wrapAngle } from '../../internal/math';

/** Drop plan samples whose `t` is <= `elapsed`. Keeps at least one sample
 *  (the goal pose) so downstream code can always read a non-empty path. */
export function trimPlan<S extends { t: number }>(
  plan: ReadonlyArray<S>,
  elapsed: number,
): S[] {
  if (plan.length === 0) return [];
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
}

/** Convenience alias when the elements are `CarKinematicState`. */
export function trimCarPlan(plan: ReadonlyArray<CarKinematicState>, elapsed: number): CarKinematicState[] {
  return trimPlan(plan, elapsed);
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/** Linearly interpolate a CarKinematicState plan at relative time `t`
 *  (seconds from the plan's start sample). Used by commit-window plan
 *  stitching to compute the predicted future state to replan from.
 *  Clamps to the first/last sample if `t` is out of range. */
export function samplePlanAt(
  plan: ReadonlyArray<CarKinematicState>,
  t: number,
): CarKinematicState | null {
  if (plan.length === 0) return null;
  if (plan.length === 1 || t <= plan[0]!.t) return { ...plan[0]! };
  const last = plan[plan.length - 1]!;
  if (t >= last.t) return { ...last };
  // Binary-search for the bracket [i, i+1] containing t.
  let lo = 0;
  let hi = plan.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (plan[mid]!.t <= t) lo = mid;
    else hi = mid;
  }
  const a = plan[lo]!;
  const b = plan[hi]!;
  const dt = b.t - a.t;
  const u = dt > 1e-9 ? (t - a.t) / dt : 0;
  // Heading is interpolated on the shorter arc.
  const dh = wrapPi(b.heading - a.heading);
  return {
    x: a.x + (b.x - a.x) * u,
    z: a.z + (b.z - a.z) * u,
    heading: wrapPi(a.heading + dh * u),
    speed: a.speed + (b.speed - a.speed) * u,
    t,
  };
}

/** Expand a sparse plan (primitive endpoints only) into a dense path by
 *  inserting intermediate sweep samples from each primitive. Each node's
 *  `edge.data.primId` identifies the primitive whose `sweep` array provides
 *  the local-frame intermediate poses. These are transformed into world
 *  coordinates using the parent node's pose as the reference frame.
 *
 *  Falls back to the sparse endpoint when the edge is missing or is a
 *  Reeds-Shepp analytic expansion (which already stores dense samples in
 *  `edge.data.samples`).
 *
 *  The result is a `CarKinematicState[]` with monotonically increasing `t`,
 *  drop-in compatible with the smoother / tracker pipeline. */
export function expandPlanSweeps(
  nodes: ReadonlyArray<Node<CarKinematicState>>,
  primitives: ReadonlyArray<MotionPrimitive>,
): CarKinematicState[] {
  if (nodes.length === 0) return [];
  // Build a lookup table for primitives by id.
  const primById = new Map<number, MotionPrimitive>();
  for (const p of primitives) primById.set(p.id, p);

  const out: CarKinematicState[] = [{ ...nodes[0]!.state }];

  for (let i = 1; i < nodes.length; i++) {
    const node = nodes[i]!;
    const parent = nodes[i - 1]!;
    const edge = node.edge;

    if (edge && edge.kind !== 'reeds-shepp') {
      const data = edge.data as { primId: number } | undefined;
      const prim = data ? primById.get(data.primId) : undefined;
      if (prim && prim.sweep.length > 1) {
        const st = parent.state;
        const c = Math.cos(st.heading);
        const s = Math.sin(st.heading);
        const sweepCount = prim.sweep.length;
        // Skip sweep[0] (the parent pose itself, already in `out`).
        for (let k = 1; k < sweepCount; k++) {
          const sp = prim.sweep[k]!;
          const wx = st.x + sp.x * c - sp.z * s;
          const wz = st.z + sp.x * s + sp.z * c;
          const wh = wrapAngle(st.heading + sp.heading);
          // Linearly interpolate speed and time across the sweep.
          const u = k / (sweepCount - 1);
          out.push({
            x: wx,
            z: wz,
            heading: wh,
            speed: st.speed + (prim.end.speed - st.speed) * u,
            t: st.t + prim.duration * u,
          });
        }
        continue;
      }
    }

    // Reeds-Shepp or missing primitive: insert the dense RS samples if
    // available, otherwise fall back to the endpoint.
    if (edge && edge.kind === 'reeds-shepp') {
      const data = edge.data as { samples?: [number, number][] } | undefined;
      if (data?.samples && data.samples.length > 0) {
        const st = parent.state;
        const end = node.state;
        const n = data.samples.length;
        for (let k = 0; k < n; k++) {
          const [sx, sz] = data.samples[k]!;
          const u = (k + 1) / n;
          out.push({
            x: sx,
            z: sz,
            heading: wrapPi(st.heading + wrapPi(end.heading - st.heading) * u),
            speed: st.speed + (end.speed - st.speed) * u,
            t: st.t + (end.t - st.t) * u,
          });
        }
        continue;
      }
    }

    // Fallback: just the endpoint.
    out.push({ ...node.state });
  }
  return out;
}
