// Tile-rebuild integration. When the game rebuilds a navmesh tile (e.g. a
// destructible wall collapses), bump the world revision so caches invalidate
// and trigger ReplanState.markDirty for NPCs whose plan crosses the tile.
// Layered on navcat's existing tile infrastructure — no navcat changes.

import type { NavcatWorld } from './index';
import type { ReplanState } from '../../execute/replan';
import {
  markAffectedAgents,
  type AffectedAgent,
  type ChangedRegion,
} from '../../execute/invalidation';

/** Legacy form: the caller decides who is affected — every passed state is
 *  marked dirty unconditionally. */
export function markTileRebuilt(
  world: NavcatWorld,
  affectedReplanStates?: ReplanState[],
): void;
/** Region-scoped form: pass the rebuilt tile's extent and ALL agents; only
 *  those whose committed trajectory crosses the region (inflated by their
 *  footprint circumradius) are marked. Returns the marked states. */
export function markTileRebuilt(
  world: NavcatWorld,
  change: ChangedRegion,
  agents: ReadonlyArray<AffectedAgent>,
): ReplanState[];
export function markTileRebuilt(
  world: NavcatWorld,
  changeOrStates: ChangedRegion | ReplanState[] = [],
  agents?: ReadonlyArray<AffectedAgent>,
): ReplanState[] | void {
  world.bumpRevision();
  if (Array.isArray(changeOrStates)) {
    for (const rs of changeOrStates) rs.markDirty('tile-rebuild');
    return;
  }
  return markAffectedAgents(changeOrStates, agents ?? [], 'tile-rebuild');
}
