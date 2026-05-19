// Mononen-style static off-mesh annotation. Given jump candidates (edge pairs
// separated by a real gap), register them as navcat off-mesh connections and
// mirror them into the NavWorld so the planner expands them as jump edges.
// Candidate *generation* from a full navmesh boundary scan is a future
// extension; the substantive part — validating and registering connections
// with rich metadata — is implemented and tested here.

import { addOffMeshConnection, OffMeshConnectionDirection } from 'navcat';
import type { NavMesh } from 'navcat';
import type { NavcatWorld } from './index';
import type { StaticAffordanceMetadata } from './types';

export interface JumpCandidate {
  from: readonly [number, number];
  to: readonly [number, number];
  cost?: number;
  kind?: 'jump' | 'drop' | 'climb';
}

export interface AnnotateOptions {
  /** Endpoint radius for the off-mesh connection. */
  radius?: number;
  flags?: number;
  area?: number;
}

/**
 * Validate and register jump candidates. A candidate is accepted only if both
 * endpoints are on the navmesh AND the straight segment between them is NOT
 * walkable (i.e., it spans a real gap — a genuine jump, not a shortcut).
 * Returns the metadata for every registered connection.
 */
export function annotateJumpLinks(
  world: NavcatWorld,
  navMesh: NavMesh,
  candidates: ReadonlyArray<JumpCandidate>,
  opts: AnnotateOptions = {},
): StaticAffordanceMetadata[] {
  const radius = opts.radius ?? 0.5;
  const flags = opts.flags ?? 1;
  const area = opts.area ?? 0;
  const out: StaticAffordanceMetadata[] = [];

  for (const c of candidates) {
    const fp = world.polygonAt(c.from[0], c.from[1]);
    const tp = world.polygonAt(c.to[0], c.to[1]);
    if (!fp || !tp) continue;
    if (world.segmentClear(c.from[0], c.from[1], c.to[0], c.to[1])) continue; // no gap

    const start: [number, number, number] = [c.from[0], fp.y, c.from[1]];
    const end: [number, number, number] = [c.to[0], tp.y, c.to[1]];
    const connectionId = addOffMeshConnection(navMesh, {
      start,
      end,
      radius,
      direction: OffMeshConnectionDirection.BIDIRECTIONAL,
      flags,
      area,
    });

    const link = {
      from: fp,
      to: tp,
      start,
      end,
      kind: c.kind ?? ('jump' as const),
      cost: c.cost ?? Math.hypot(end[0] - start[0], end[2] - start[2]),
    };
    world.addOffLink(link);
    out.push({ connectionId, link });
  }
  return out;
}
