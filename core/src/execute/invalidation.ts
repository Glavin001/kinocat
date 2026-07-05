// Region-scoped external invalidation. When the world changes under running
// agents (a navmesh tile rebuilt, ground destroyed, an off-mesh link added),
// the game describes the changed area as a `ChangedRegion` and this module
// decides WHICH committed trajectories actually cross it — only those agents
// get `ReplanState.markDirty`, everyone else keeps their plan. Pure planar
// geometry; world-agnostic (serves NavcatWorld, InMemoryNavWorld, and tests
// identically).

import type { PlanPath } from './types';
import type { ReplanState } from './replan';
import {
  pointInPolygon,
  pointSegmentDistance,
  segmentIntersectsAABB,
  segmentsIntersect,
  type Pt,
} from '../internal/geom';

/** A changed area of the world, in planning-plane (XZ) coordinates. The AABB
 *  is required — it is what a tile rebuild naturally produces, and testing
 *  against it alone is conservative (over-marking costs one extra replan,
 *  never a stale plan). `polygon` optionally refines the test for regions
 *  much smaller than their bounding box. */
export interface ChangedRegion {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  /** Optional tighter boundary; must lie within the AABB. */
  polygon?: ReadonlyArray<Pt>;
}

/** Circumradius of a local-frame footprint ring — the inflation that makes a
 *  polyline test cover the full swept body of the agent. */
export function footprintCircumradius(footprint: ReadonlyArray<Pt>): number {
  let r2 = 0;
  for (const [x, z] of footprint) {
    const d2 = x * x + z * z;
    if (d2 > r2) r2 = d2;
  }
  return Math.sqrt(r2);
}

function segmentNearPolygon(
  ax: number, az: number, bx: number, bz: number,
  poly: ReadonlyArray<Pt>,
  inflate: number,
): boolean {
  // Either endpoint (or the midpoint, for segments spanning the region)
  // inside the polygon.
  if (pointInPolygon(ax, az, poly)) return true;
  if (pointInPolygon(bx, bz, poly)) return true;
  if (pointInPolygon((ax + bx) / 2, (az + bz) / 2, poly)) return true;
  for (let j = 0; j < poly.length; j++) {
    const e0 = poly[j]!;
    const e1 = poly[(j + 1) % poly.length]!;
    if (segmentsIntersect(ax, az, bx, bz, e0[0], e0[1], e1[0], e1[1])) return true;
    if (inflate > 0) {
      // Inflated test: the polygon edge passes within `inflate` of the
      // segment. Vertex-vs-segment both ways is a conservative proxy that is
      // exact for the rectangular tiles this serves.
      if (pointSegmentDistance(e0[0], e0[1], ax, az, bx, bz) <= inflate) return true;
      if (pointSegmentDistance(ax, az, e0[0], e0[1], e1[0], e1[1]) <= inflate) return true;
      if (pointSegmentDistance(bx, bz, e0[0], e0[1], e1[0], e1[1]) <= inflate) return true;
    }
  }
  return false;
}

/** Does the committed trajectory pass through the changed region? `inflate`
 *  grows the region by the agent's footprint circumradius so the test covers
 *  the swept body, not just the reference point. Single poses (length-1
 *  paths) are tested as points. */
export function planCrossesRegion(
  path: PlanPath,
  region: ChangedRegion,
  inflate = 0,
): boolean {
  if (path.length === 0) return false;
  const minX = region.x0 - inflate;
  const minZ = region.z0 - inflate;
  const maxX = region.x1 + inflate;
  const maxZ = region.z1 + inflate;
  const n = Math.max(path.length - 1, 1);
  for (let i = 0; i < n; i++) {
    const a = path[i]!;
    const b = path[Math.min(i + 1, path.length - 1)]!;
    if (!segmentIntersectsAABB(a.x, a.z, b.x, b.z, minX, minZ, maxX, maxZ)) continue;
    // AABB hit; refine against the polygon when one is provided.
    if (!region.polygon) return true;
    if (segmentNearPolygon(a.x, a.z, b.x, b.z, region.polygon, inflate)) return true;
  }
  return false;
}

/** An agent whose committed plan should be checked against changed regions.
 *  `inflate` is typically `footprintCircumradius(agent.footprint)`. */
export interface AffectedAgent {
  replan: ReplanState;
  inflate?: number;
}

/** Mark dirty exactly the agents whose committed trajectory crosses `region`.
 *  Returns the marked states. `reason` feeds `ReplanState.markDirty` — use
 *  'tile-rebuild' for geometry changes, 'off-mesh' for new links (a new link
 *  never invalidates a committed plan, but agents may want to exploit it). */
export function markAffectedAgents(
  region: ChangedRegion,
  agents: ReadonlyArray<AffectedAgent>,
  reason = 'tile-rebuild',
): ReplanState[] {
  const marked: ReplanState[] = [];
  for (const a of agents) {
    const plan = a.replan.currentPlan;
    if (!plan) continue;
    if (planCrossesRegion(plan, region, a.inflate ?? 0)) {
      a.replan.markDirty(reason);
      marked.push(a.replan);
    }
  }
  return marked;
}
