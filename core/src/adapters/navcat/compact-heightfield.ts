// Opt 1 — O(1) clearance broadphase (spec §10.2). navcat builds a Recast
// chamfer distance field on the CompactHeightfield (`distances`, raw units:
// straight step +2, diagonal +3). Converting to a *strict lower bound* on the
// true world clearance lets VehicleEnvironment skip the expensive exact
// footprint check wherever a disk of the agent's circumscribed radius is
// provably clear (early-ACCEPT only — it never early-rejects, so it cannot
// introduce a false "clear").

import type { CompactHeightfield } from 'navcat';
import { type ChfGrid, makeChfGrid, worldCell } from './chf-grid';

export interface ClearanceFieldOptions {
  /**
   * Multiplier applied to the (raw/2)·cellSize chamfer estimate before a
   * further one-cell slack is subtracted. The (2,3) chamfer over-estimates
   * Euclidean distance by up to ~8% and navcat blurs the field, so a factor
   * &lt; 1/1.08 keeps the result a strict lower bound. Default 0.9; the
   * navcat.test.ts scale-pin asserts the result never exceeds true distance.
   */
  safety?: number;
}

export class ChfClearanceField {
  private readonly g: ChfGrid;
  private readonly safety: number;

  constructor(chf: CompactHeightfield, opts: ClearanceFieldOptions = {}) {
    this.g = makeChfGrid(chf);
    this.safety = opts.safety ?? 0.9;
  }

  /** Strict lower bound (world units) on the distance from (x,z) to the
   *  nearest obstacle / boundary, or null if off-field or no walkable span.
   *  With `queryY` undefined the most-conservative (minimum-clearance) span
   *  in the column is used, so the early-accept is sound on multi-level
   *  terrain regardless of which level the agent is on. */
  clearanceAt(x: number, z: number, queryY?: number): number | null {
    const g = this.g;
    const cell = worldCell(g, x, z);
    if (cell === null) return null;
    const col = g.chf.cells[cell];
    if (!col || col.count === 0) return null;
    const spans = g.chf.spans;
    const areas = g.chf.areas;
    const dist = g.chf.distances;
    let pick = -1;
    let key = Infinity;
    for (let s = col.index; s < col.index + col.count; s++) {
      if ((areas[s] ?? 0) === 0) continue; // NULL_AREA
      if (queryY === undefined) {
        const d = dist[s] ?? 0; // smallest clearance ⇒ safest early-accept
        if (d < key) {
          key = d;
          pick = s;
        }
      } else {
        const wy = g.minY + (spans[s]?.y ?? 0) * g.cellHeight;
        const k = Math.abs(wy - queryY);
        if (k < key) {
          key = k;
          pick = s;
        }
      }
    }
    if (pick < 0) return null;
    const raw = dist[pick] ?? 0;
    // raw/2 ≈ cells-to-boundary; ·cellSize → world; ·safety + one-cell slack
    // makes it a provable lower bound (verified by the scale-pin test).
    const world = (raw / 2) * g.cellSize * this.safety - g.cellSize;
    return world > 0 ? world : 0;
  }
}
