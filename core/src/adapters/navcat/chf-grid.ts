// Shared world↔cell math over a navcat CompactHeightfield, used by the
// clearance field (Opt 1) and the goal-distance field (Opt 2). Lives under
// adapters/navcat so the core never imports navcat and this stays out of the
// coverage gate; correctness is covered by core/test/adapters/navcat.test.ts.

import type { CompactHeightfield } from 'navcat';

/** `mathcat` Box3 is the tuple [minX,minY,minZ,maxX,maxY,maxZ]. */
export interface ChfGrid {
  chf: CompactHeightfield;
  minX: number;
  minY: number;
  minZ: number;
  width: number;
  height: number;
  cellSize: number;
  cellHeight: number;
}

export function makeChfGrid(chf: CompactHeightfield): ChfGrid {
  const b = chf.bounds;
  return {
    chf,
    minX: b[0],
    minY: b[1],
    minZ: b[2],
    width: chf.width,
    height: chf.height,
    cellSize: chf.cellSize,
    cellHeight: chf.cellHeight,
  };
}

/** Linear column index for world (x,z), or null if outside the field. */
export function worldCell(g: ChfGrid, x: number, z: number): number | null {
  const cx = Math.floor((x - g.minX) / g.cellSize);
  const cz = Math.floor((z - g.minZ) / g.cellSize);
  if (cx < 0 || cx >= g.width || cz < 0 || cz >= g.height) return null;
  return cx + cz * g.width;
}

/** A span is walkable iff its area id is non-zero (navcat NULL_AREA = 0). */
export function isWalkable(g: ChfGrid, spanIndex: number): boolean {
  return (g.chf.areas[spanIndex] ?? 0) !== 0;
}
