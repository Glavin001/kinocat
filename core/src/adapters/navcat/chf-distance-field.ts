// Opt 2 — obstacle-aware grid-Dijkstra dual heuristic (Dolgov et al.; spec
// §10.3). A single-source Dijkstra FROM THE GOAL over the CompactHeightfield's
// walkable spans gives the true 2D obstacle-respecting shortest distance — an
// admissible, consistent lower bound on remaining path length. Combined via
// max() with the Reeds-Shepp/Euclid term it stays admissible while pruning
// the search hard in obstacle-dense terrain. Lives under adapters/navcat so
// the core never imports navcat (coverage-excluded; navcat.test.ts covers it).
//
// Admissibility invariant: the CHF MUST be un-eroded (walkableRadiusWorld 0,
// the adapter default) so its walkable set is a superset of the true free
// space and the grid geodesic is a genuine lower bound. The per-query
// `−√2·cellSize` slack absorbs one cell of discretisation.

import type { CompactHeightfield } from 'navcat';
import { getCon } from 'navcat';
import { type ChfGrid, makeChfGrid, worldCell, isWalkable } from './chf-grid';
import { BinaryHeap } from '../../internal/heap';

const NOT_CONNECTED = 0x3f;
// [xOffset, zOffset] per direction; linear cell index = x + z*width.
const DIR: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];
// (cardinal dir, then this perpendicular dir from the neighbour) → the four
// diagonals, matching navcat's own distance-field corner handling so a
// diagonal is only taken when the corner is actually traversable.
const DIAG: ReadonlyArray<readonly [number, number]> = [
  [0, 3],
  [3, 2],
  [2, 1],
  [1, 0],
];

interface QItem {
  i: number;
  d: number;
}

export class ChfGoalDistanceField {
  private readonly g: ChfGrid;
  private readonly dist: Float64Array;
  private readonly slack: number;
  readonly available: boolean;

  /** Point-goal field. `regionSeeds` is internal — use `fromRegion`. */
  constructor(
    chf: CompactHeightfield,
    gx: number,
    gz: number,
    gy?: number,
    regionSeeds?: (g: ChfGrid) => number[],
  ) {
    const g = makeChfGrid(chf);
    this.g = g;
    this.slack = Math.SQRT2 * g.cellSize;
    this.dist = new Float64Array(chf.spanCount).fill(Infinity);
    const seeds = regionSeeds
      ? regionSeeds(g)
      : [this.spanInColumn(this.cellOf(gx, gz), gy, false)].filter((s) => s >= 0);
    if (seeds.length === 0) {
      this.available = false;
      return;
    }
    this.available = true;
    this.dijkstra(seeds);
  }

  /** Multi-source field seeded from EVERY walkable span whose column centre
   *  lies inside the region — `lookup` then gives the obstacle-avoiding
   *  distance to the NEAREST point of the region (a true region ETA bound).
   *  Null when no column passes `containsXZ` (region off-mesh / degenerate;
   *  callers fall back to their kinematic bound). */
  static fromRegion(
    chf: CompactHeightfield,
    containsXZ: (x: number, z: number, cellHalfDiag?: number) => boolean,
  ): ChfGoalDistanceField | null {
    const field = new ChfGoalDistanceField(chf, NaN, NaN, undefined, (g) => {
      const halfDiag = (g.cellSize * Math.SQRT2) / 2;
      const seeds: number[] = [];
      for (let cz = 0; cz < g.height; cz++) {
        for (let cx = 0; cx < g.width; cx++) {
          const wx = g.minX + (cx + 0.5) * g.cellSize;
          const wz = g.minZ + (cz + 0.5) * g.cellSize;
          if (!containsXZ(wx, wz, halfDiag)) continue;
          const col = g.chf.cells[cx + cz * g.width];
          if (!col) continue;
          for (let s = col.index; s < col.index + col.count; s++) {
            if (isWalkable(g, s)) seeds.push(s);
          }
        }
      }
      return seeds;
    });
    return field.available ? field : null;
  }

  private cellOf(x: number, z: number): number {
    return worldCell(this.g, x, z) ?? -1;
  }

  /** Walkable span in column `cell`: nearest world-Y `y` when given; else
   *  the smallest-goal-distance span when `preferMinDist` (post-Dijkstra
   *  query — the most conservative admissible choice) or the first walkable
   *  span otherwise (goal selection, pre-Dijkstra). -1 if none / off-grid. */
  private spanInColumn(
    cell: number,
    y: number | undefined,
    preferMinDist: boolean,
  ): number {
    if (cell < 0) return -1;
    const g = this.g;
    const col = g.chf.cells[cell];
    if (!col || col.count === 0) return -1;
    let pick = -1;
    let key = Infinity;
    for (let s = col.index; s < col.index + col.count; s++) {
      if ((g.chf.areas[s] ?? 0) === 0) continue;
      if (y !== undefined) {
        const wy = g.minY + (g.chf.spans[s]?.y ?? 0) * g.cellHeight;
        const k = Math.abs(wy - y);
        if (k < key) {
          key = k;
          pick = s;
        }
      } else if (preferMinDist) {
        const d = this.dist[s] ?? Infinity;
        if (pick < 0 || d < key) {
          key = d;
          pick = s;
        }
      } else {
        return s; // first walkable span (goal selection)
      }
    }
    return pick;
  }

  private dijkstra(seeds: ReadonlyArray<number>): void {
    const g = this.g;
    const { width, height } = g;
    const cells = g.chf.cells;
    const spans = g.chf.spans;
    const areas = g.chf.areas;
    const dist = this.dist;
    const step = g.cellSize;
    const diag = Math.SQRT2 * g.cellSize;
    // span index → linear cell, for O(1) (cx,cz) recovery.
    const spanCell = new Int32Array(g.chf.spanCount);
    for (let cz = 0; cz < height; cz++) {
      for (let cx = 0; cx < width; cx++) {
        const c = cells[cx + cz * width];
        if (!c) continue;
        for (let s = c.index; s < c.index + c.count; s++) spanCell[s] = cx + cz * width;
      }
    }
    const heap = new BinaryHeap<QItem>((a, b) => a.d - b.d);
    for (const s of seeds) {
      dist[s] = 0;
      heap.push({ i: s, d: 0 });
    }
    while (!heap.isEmpty()) {
      const top = heap.pop()!;
      const si = top.i;
      if (top.d > (dist[si] ?? Infinity)) continue;
      const span = spans[si];
      if (!span) continue;
      const col = spanCell[si]!;
      const cx = col % width;
      const cz = (col / width) | 0;
      const base = dist[si]!;
      for (let dir = 0; dir < 4; dir++) {
        const con = getCon(span, dir);
        if (con === NOT_CONNECTED) continue;
        const o = DIR[dir]!;
        const ax = cx + o[0];
        const az = cz + o[1];
        if (ax < 0 || ax >= width || az < 0 || az >= height) continue;
        const ac = cells[ax + az * width];
        if (!ac) continue;
        const ai = ac.index + con;
        if (ai < 0 || ai >= g.chf.spanCount || (areas[ai] ?? 0) === 0) continue;
        if (base + step < (dist[ai] ?? Infinity)) {
          dist[ai] = base + step;
          heap.push({ i: ai, d: dist[ai]! });
        }
      }
      for (const [d1, d2] of DIAG) {
        const c1 = getCon(span, d1);
        if (c1 === NOT_CONNECTED) continue;
        const o1 = DIR[d1]!;
        const ax = cx + o1[0];
        const az = cz + o1[1];
        if (ax < 0 || ax >= width || az < 0 || az >= height) continue;
        const ac = cells[ax + az * width];
        if (!ac) continue;
        const ai = ac.index + c1;
        const aspan = spans[ai];
        if (!aspan || (areas[ai] ?? 0) === 0) continue;
        const c2 = getCon(aspan, d2);
        if (c2 === NOT_CONNECTED) continue;
        const o2 = DIR[d2]!;
        const bx = ax + o2[0];
        const bz = az + o2[1];
        if (bx < 0 || bx >= width || bz < 0 || bz >= height) continue;
        const bc = cells[bx + bz * width];
        if (!bc) continue;
        const bi = bc.index + c2;
        if (bi < 0 || bi >= g.chf.spanCount || (areas[bi] ?? 0) === 0) continue;
        if (base + diag < (dist[bi] ?? Infinity)) {
          dist[bi] = base + diag;
          heap.push({ i: bi, d: dist[bi]! });
        }
      }
    }
  }

  /** Admissible lower bound (world units) on the remaining obstacle-avoiding
   *  path length from (x,z) to the goal, or null when off-grid / unreachable
   *  (the caller then uses the Reeds-Shepp term alone — still admissible). */
  lookup(x: number, z: number, y?: number): number | null {
    if (!this.available) return null;
    const si = this.spanInColumn(this.cellOf(x, z), y, true);
    if (si < 0) return null;
    const d = this.dist[si] ?? Infinity;
    if (!Number.isFinite(d)) return null;
    const lb = d - this.slack;
    return lb > 0 ? lb : 0;
  }
}
