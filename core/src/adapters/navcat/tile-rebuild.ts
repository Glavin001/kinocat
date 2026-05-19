// Tile-rebuild integration. When the game rebuilds a navmesh tile (e.g. a
// destructible wall collapses), bump the world revision so caches invalidate
// and trigger ReplanState.markDirty for NPCs whose plan crosses the tile.
// Layered on navcat's existing tile infrastructure — no navcat changes.

import type { NavcatWorld } from './index';
import type { ReplanState } from '../../execute/replan';

/** Bump the NavWorld revision and mark affected NPCs for replanning. */
export function markTileRebuilt(
  world: NavcatWorld,
  affectedReplanStates: ReplanState[] = [],
): void {
  world.bumpRevision();
  for (const rs of affectedReplanStates) rs.markDirty('tile-rebuild');
}
