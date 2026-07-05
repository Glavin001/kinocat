// The core's only model of "the world". kinocat never imports navcat;
// VehicleEnvironment / HumanoidEnvironment consume this seam. InMemoryNavWorld
// (polygon soup, zero deps) makes the whole algorithmic core unit-testable
// without any external runtime. adapters/navcat implements NavWorld over a
// real navcat NavMesh.

import { pointInPolygon, polygonsIntersect, segmentsIntersect, type Pt } from '../internal/geom';
import { BinaryHeap } from '../internal/heap';

/** Opaque planning-plane polygon reference. JSON-serializable; the planner
 *  does heuristic/dominance math from the cached centroid + derived Y without
 *  any geometry callback. */
export interface PolygonRef {
  readonly id: number;
  readonly cx: number;
  readonly cz: number;
  readonly y: number;
}

export interface OffMeshLink {
  readonly from: PolygonRef;
  readonly to: PolygonRef;
  readonly start: readonly [number, number, number];
  readonly end: readonly [number, number, number];
  readonly kind: 'jump' | 'drop' | 'climb';
  readonly cost: number;
}

export interface NavWorld {
  /** Walkable polygon containing (x,z), or null if off-mesh/blocked. */
  polygonAt(x: number, z: number): PolygonRef | null;
  /** Derived floor Y at (x,z) within/near `poly` (right-handed, +Y up). */
  heightAt(poly: PolygonRef, x: number, z: number): number;
  /** True if the world-space footprint polygon is collision-free. */
  footprintClear(footprint: ReadonlyArray<readonly [number, number]>): boolean;
  /** True if the straight XZ segment stays on walkable, obstacle-free mesh. */
  segmentClear(x0: number, z0: number, x1: number, z1: number): boolean;
  /** Static off-mesh connections leaving `poly` (jumps/drops/climbs). */
  offMeshFrom(poly: PolygonRef): ReadonlyArray<OffMeshLink>;
  /** Bumps when geometry changes (tile rebuild) — cache invalidation. */
  readonly revision: number;

  /** Optional fast clearance oracle (e.g. a CompactHeightfield distance
   *  field). Distance in world units from (x,z) to the nearest obstacle /
   *  boundary, or null if unavailable / off-field. A return ≥ r guarantees a
   *  disk of radius r centred at (x,z) is collision-free, so a caller may
   *  skip the exact footprint check (early-ACCEPT only — never reject). */
  clearanceAt?(x: number, z: number, queryY?: number): number | null;

  /** Optional admissible obstacle-aware distance-to-goal oracle factory.
   *  Returns a function giving a lower bound (world units) on the remaining
   *  path length from (x,z) to (gx,gz), or null when unavailable. The bound
   *  must never exceed the true shortest obstacle-avoiding path length so the
   *  planner's heuristic stays admissible. */
  buildGoalLowerBound?(
    gx: number,
    gz: number,
    gy?: number,
  ): ((x: number, z: number, y?: number) => number | null) | null;

  /** Optional REGION variant of `buildGoalLowerBound`: a multi-source
   *  distance field seeded from every walkable cell whose centre passes
   *  `containsXZ`. The predicate receives the grid's cell HALF-DIAGONAL as a
   *  third argument so callers can conservatively accept cells for regions
   *  smaller than a cell (which would otherwise fall between cell centres).
   *  The returned function gives an admissible lower bound on the
   *  obstacle-avoiding distance from (x,z) to the NEAREST point of the
   *  region, or null off-grid / when the region is unreachable. UNMEMOIZED —
   *  one field build per call; callers (the ETA oracle) own caching. */
  buildRegionLowerBound?(
    containsXZ: (x: number, z: number, cellHalfDiag?: number) => boolean,
  ): ((x: number, z: number, y?: number) => number | null) | null;
}

export interface NavPolygon {
  id: number;
  /** CCW or CW ring of [x,z] vertices. */
  ring: Pt[];
  /** Flat floor height for this polygon. */
  y: number;
}

/** Polygon-soup NavWorld for tests and dep-free planning. Walkable = union of
 *  `polygons`; `obstacles` are holes that block footprints/segments.
 *
 *  Indexed by a uniform-grid AABB broadphase: every obstacle (and every
 *  polygon) is bucketed by the cells its bounding box touches, and the
 *  collision queries narrow on candidates from the cells the query AABB
 *  touches. Behavior is identical to brute-force iteration — the index only
 *  changes which obstacles get the (unchanged) `polygonsIntersect` /
 *  `segmentsIntersect` test applied to them. */
export class InMemoryNavWorld implements NavWorld {
  private _revision = 0;
  private offMesh: OffMeshLink[] = [];

  // Pre-computed AABBs and a uniform-grid spatial index over them. The grid
  // is rebuilt lazily whenever obstacles change (`setObstacles`) — polygons
  // are immutable post-construction. `null` cell = obstacle-free bucket.
  private obstacleAABBs: Float64Array;
  private polygonAABBs: Float64Array;
  private obstacleGrid: Int32Array[] | null = null;
  private polygonGrid: Int32Array[] | null = null;
  private gridOriginX = 0;
  private gridOriginZ = 0;
  private gridCellsX = 0;
  private gridCellsZ = 0;
  private gridCell = 0;

  // Coarse occupancy + clearance grids for the dual heuristic (`gridHeuristic`)
  // and clearance broadphase (`clearanceBroadphase`) opt-ins on
  // `VehicleEnvironment`. Built once at construction, invalidated on
  // `setObstacles`. See `buildGoalLowerBound` / `clearanceAt` below.
  private h_cellSize = 0;
  private h_originX = 0;
  private h_originZ = 0;
  private h_width = 0;
  private h_height = 0;
  // 1 = cell is blocked (off-mesh or any obstacle AABB overlaps it).
  private h_blocked: Uint8Array | null = null;
  // World-units clearance at each cell centre — distance from the cell centre
  // to the nearest obstacle edge or off-mesh transition, clamped to 0 inside
  // an obstacle. `lookup` subtracts a √2·cell slack to stay a lower bound.
  private h_clearance: Float32Array | null = null;
  // Memoised goal-distance Dijkstra (keyed on rounded goal cell index). One
  // entry survives until the next `setObstacles` or a different goal cell.
  private h_goalKey = -1;
  private h_goalDist: Float32Array | null = null;
  // Set by `setObstacles`; the grid rebuild is deferred to the next consumer
  // (`clearanceAt` / `buildGoalLowerBound`) so a live tile update only pays
  // for the oracles it actually uses — the rebuild is the dominant cost of
  // an obstacle swap and would otherwise land on the replan latency path.
  private h_dirty = false;

  constructor(
    private polygons: NavPolygon[],
    private obstacles: Pt[][] = [],
  ) {
    this.obstacleAABBs = buildAABBs(this.obstacles);
    this.polygonAABBs = buildAABBs(this.polygons.map((p) => p.ring));
    this.rebuildIndex();
    this.buildHeuristicGrid();
  }

  get revision(): number {
    return this._revision;
  }

  private rebuildIndex(): void {
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    let diagSum = 0;
    let diagN = 0;
    const merge = (a: Float64Array): void => {
      for (let i = 0; i < a.length; i += 4) {
        const x0 = a[i]!, z0 = a[i + 1]!, x1 = a[i + 2]!, z1 = a[i + 3]!;
        if (x0 < minX) minX = x0;
        if (z0 < minZ) minZ = z0;
        if (x1 > maxX) maxX = x1;
        if (z1 > maxZ) maxZ = z1;
        diagSum += Math.max(x1 - x0, z1 - z0);
        diagN++;
      }
    };
    merge(this.obstacleAABBs);
    merge(this.polygonAABBs);
    if (diagN === 0 || !isFinite(minX)) {
      this.obstacleGrid = null;
      this.polygonGrid = null;
      return;
    }
    // Cell size ≈ median primitive AABB side. Keeps each obstacle in ~1-4
    // cells while bounding the per-query candidate set.
    const meanDiag = diagSum / diagN;
    this.gridCell = Math.max(meanDiag, 1);
    this.gridOriginX = minX;
    this.gridOriginZ = minZ;
    this.gridCellsX = Math.max(1, Math.ceil((maxX - minX) / this.gridCell) + 1);
    this.gridCellsZ = Math.max(1, Math.ceil((maxZ - minZ) / this.gridCell) + 1);
    this.obstacleGrid = this.bucket(this.obstacleAABBs);
    this.polygonGrid = this.bucket(this.polygonAABBs);
  }

  private bucket(aabbs: Float64Array): Int32Array[] {
    const cells = this.gridCellsX * this.gridCellsZ;
    // Two-pass: count per cell, allocate exact-size Int32Array per cell.
    const counts = new Int32Array(cells);
    const n = aabbs.length / 4;
    for (let i = 0; i < n; i++) {
      const x0 = aabbs[i * 4]!, z0 = aabbs[i * 4 + 1]!;
      const x1 = aabbs[i * 4 + 2]!, z1 = aabbs[i * 4 + 3]!;
      const cx0 = this.cellX(x0), cz0 = this.cellZ(z0);
      const cx1 = this.cellX(x1), cz1 = this.cellZ(z1);
      for (let cz = cz0; cz <= cz1; cz++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          counts[cz * this.gridCellsX + cx]!++;
        }
      }
    }
    const out: Int32Array[] = new Array(cells);
    for (let i = 0; i < cells; i++) out[i] = new Int32Array(counts[i]!);
    const cursors = new Int32Array(cells);
    for (let i = 0; i < n; i++) {
      const x0 = aabbs[i * 4]!, z0 = aabbs[i * 4 + 1]!;
      const x1 = aabbs[i * 4 + 2]!, z1 = aabbs[i * 4 + 3]!;
      const cx0 = this.cellX(x0), cz0 = this.cellZ(z0);
      const cx1 = this.cellX(x1), cz1 = this.cellZ(z1);
      for (let cz = cz0; cz <= cz1; cz++) {
        for (let cx = cx0; cx <= cx1; cx++) {
          const idx = cz * this.gridCellsX + cx;
          out[idx]![cursors[idx]!++] = i;
        }
      }
    }
    return out;
  }

  private cellX(x: number): number {
    const c = Math.floor((x - this.gridOriginX) / this.gridCell);
    if (c < 0) return 0;
    if (c >= this.gridCellsX) return this.gridCellsX - 1;
    return c;
  }
  private cellZ(z: number): number {
    const c = Math.floor((z - this.gridOriginZ) / this.gridCell);
    if (c < 0) return 0;
    if (c >= this.gridCellsZ) return this.gridCellsZ - 1;
    return c;
  }

  /** Visit (without dedup) every candidate index whose AABB cell-range
   *  overlaps the query AABB. Caller dedups via the `seen` Int8Array; the
   *  caller MUST clear `seen` between queries (or pass a fresh one). */
  private candidates(
    grid: Int32Array[] | null,
    qx0: number, qz0: number, qx1: number, qz1: number,
  ): Int32Array[] | null {
    if (!grid) return null;
    const cx0 = this.cellX(qx0), cz0 = this.cellZ(qz0);
    const cx1 = this.cellX(qx1), cz1 = this.cellZ(qz1);
    // Collect cell-buckets into a small array; callers stream over them.
    // For axis-aligned single-cell queries this is a one-element array.
    const out: Int32Array[] = [];
    for (let cz = cz0; cz <= cz1; cz++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        const bucket = grid[cz * this.gridCellsX + cx]!;
        if (bucket.length > 0) out.push(bucket);
      }
    }
    return out;
  }

  private refOf(p: NavPolygon): PolygonRef {
    let sx = 0;
    let sz = 0;
    for (const [x, z] of p.ring) {
      sx += x;
      sz += z;
    }
    const n = p.ring.length;
    return { id: p.id, cx: sx / n, cz: sz / n, y: p.y };
  }

  polygonAt(x: number, z: number): PolygonRef | null {
    const obBuckets = this.candidates(this.obstacleGrid, x, z, x, z);
    if (obBuckets) {
      const seen = new Set<number>();
      for (const bucket of obBuckets) {
        for (let k = 0; k < bucket.length; k++) {
          const i = bucket[k]!;
          if (seen.has(i)) continue;
          seen.add(i);
          if (aabbContainsPoint(this.obstacleAABBs, i, x, z) &&
              pointInPolygon(x, z, this.obstacles[i]!)) {
            return null;
          }
        }
      }
    } else {
      for (const ob of this.obstacles) {
        if (pointInPolygon(x, z, ob)) return null;
      }
    }
    const polyBuckets = this.candidates(this.polygonGrid, x, z, x, z);
    if (polyBuckets) {
      const seen = new Set<number>();
      for (const bucket of polyBuckets) {
        for (let k = 0; k < bucket.length; k++) {
          const i = bucket[k]!;
          if (seen.has(i)) continue;
          seen.add(i);
          if (aabbContainsPoint(this.polygonAABBs, i, x, z) &&
              pointInPolygon(x, z, this.polygons[i]!.ring)) {
            return this.refOf(this.polygons[i]!);
          }
        }
      }
      return null;
    }
    for (const p of this.polygons) {
      if (pointInPolygon(x, z, p.ring)) return this.refOf(p);
    }
    return null;
  }

  heightAt(poly: PolygonRef, _x: number, _z: number): number {
    return poly.y;
  }

  footprintClear(footprint: ReadonlyArray<readonly [number, number]>): boolean {
    const fp = footprint as ReadonlyArray<Pt>;
    // every vertex must be on walkable mesh...
    for (let i = 0; i < fp.length; i++) {
      if (this.polygonAt(fp[i]![0], fp[i]![1]) === null) return false;
    }
    // ...and the footprint must not overlap any obstacle.
    let qx0 = Infinity, qz0 = Infinity, qx1 = -Infinity, qz1 = -Infinity;
    for (let i = 0; i < fp.length; i++) {
      const x = fp[i]![0], z = fp[i]![1];
      if (x < qx0) qx0 = x;
      if (z < qz0) qz0 = z;
      if (x > qx1) qx1 = x;
      if (z > qz1) qz1 = z;
    }
    const buckets = this.candidates(this.obstacleGrid, qx0, qz0, qx1, qz1);
    if (buckets) {
      const seen = new Set<number>();
      for (const bucket of buckets) {
        for (let k = 0; k < bucket.length; k++) {
          const i = bucket[k]!;
          if (seen.has(i)) continue;
          seen.add(i);
          if (!aabbsOverlap(this.obstacleAABBs, i, qx0, qz0, qx1, qz1)) continue;
          if (polygonsIntersect(fp, this.obstacles[i]!)) return false;
        }
      }
      return true;
    }
    for (const ob of this.obstacles) {
      if (polygonsIntersect(fp, ob)) return false;
    }
    return true;
  }

  segmentClear(x0: number, z0: number, x1: number, z1: number): boolean {
    if (this.polygonAt(x0, z0) === null || this.polygonAt(x1, z1) === null) {
      return false;
    }
    const qx0 = Math.min(x0, x1), qz0 = Math.min(z0, z1);
    const qx1 = Math.max(x0, x1), qz1 = Math.max(z0, z1);
    const buckets = this.candidates(this.obstacleGrid, qx0, qz0, qx1, qz1);
    if (buckets) {
      const seen = new Set<number>();
      for (const bucket of buckets) {
        for (let k = 0; k < bucket.length; k++) {
          const i = bucket[k]!;
          if (seen.has(i)) continue;
          seen.add(i);
          if (!aabbsOverlap(this.obstacleAABBs, i, qx0, qz0, qx1, qz1)) continue;
          const ob = this.obstacles[i]!;
          for (let j = 0; j < ob.length; j++) {
            const a = ob[j]!;
            const b = ob[(j + 1) % ob.length]!;
            if (segmentsIntersect(x0, z0, x1, z1, a[0], a[1], b[0], b[1])) return false;
          }
        }
      }
      return true;
    }
    for (const ob of this.obstacles) {
      for (let i = 0; i < ob.length; i++) {
        const a = ob[i]!;
        const b = ob[(i + 1) % ob.length]!;
        if (segmentsIntersect(x0, z0, x1, z1, a[0], a[1], b[0], b[1])) return false;
      }
    }
    return true;
  }

  offMeshFrom(poly: PolygonRef): ReadonlyArray<OffMeshLink> {
    return this.offMesh.filter((l) => l.from.id === poly.id);
  }

  /** Lower bound on Euclidean clearance at (x, z): distance from (x, z) to
   *  the nearest obstacle edge or off-mesh boundary. Early-ACCEPT only — when
   *  this returns `r`, a disk of radius `r` centred at (x, z) is guaranteed
   *  collision-free. Returns null when off-grid.
   *
   *  Stored value is the grid-Dijkstra distance from the cell to the nearest
   *  blocked cell (centre-to-centre); the true blocked region may be up to
   *  cell_diagonal closer (half-cell at each end), so we subtract the full
   *  diagonal to keep the answer a true lower bound. */
  clearanceAt(x: number, z: number): number | null {
    this.ensureHeuristicGrid();
    if (!this.h_clearance) return null;
    const cs = this.h_cellSize;
    const cx = Math.floor((x - this.h_originX) / cs);
    const cz = Math.floor((z - this.h_originZ) / cs);
    if (cx < 0 || cx >= this.h_width || cz < 0 || cz >= this.h_height) return null;
    const v = this.h_clearance[cx + cz * this.h_width]!;
    if (!Number.isFinite(v)) return null;
    const lb = v - Math.SQRT2 * cs;
    return lb > 0 ? lb : 0;
  }

  /** Admissible obstacle-aware distance-to-goal oracle. Returns a function
   *  giving a lower bound on the remaining obstacle-avoiding path length from
   *  (x, z) to (gx, gz), or null when the goal is off-grid / unreachable.
   *  Memoised per goal cell across calls with the same goal. */
  buildGoalLowerBound(
    gx: number,
    gz: number,
  ): ((x: number, z: number, y?: number) => number | null) | null {
    this.ensureHeuristicGrid();
    if (!this.h_blocked) return null;
    const cs = this.h_cellSize;
    const w = this.h_width;
    const h = this.h_height;
    const gcx = Math.floor((gx - this.h_originX) / cs);
    const gcz = Math.floor((gz - this.h_originZ) / cs);
    if (gcx < 0 || gcx >= w || gcz < 0 || gcz >= h) return null;
    const goalIdx = gcx + gcz * w;
    if (this.h_goalKey !== goalIdx || !this.h_goalDist) {
      this.h_goalDist = this.dijkstraMulti([goalIdx]);
      this.h_goalKey = goalIdx;
    }
    const dist = this.h_goalDist;
    const slack = Math.SQRT2 * cs;
    return (x: number, z: number) => {
      const cx = Math.floor((x - this.h_originX) / cs);
      const cz = Math.floor((z - this.h_originZ) / cs);
      if (cx < 0 || cx >= w || cz < 0 || cz >= h) return null;
      const d = dist[cx + cz * w]!;
      if (!Number.isFinite(d)) return null;
      const lb = d - slack;
      return lb > 0 ? lb : 0;
    };
  }

  /** Region variant of `buildGoalLowerBound`: multi-source field seeded from
   *  every UNBLOCKED grid cell whose centre passes `containsXZ` — lookups
   *  give the obstacle-avoiding distance to the nearest point of the region.
   *  Unmemoized (callers own caching); null when the region covers no
   *  unblocked cell. */
  buildRegionLowerBound(
    containsXZ: (x: number, z: number, cellHalfDiag?: number) => boolean,
  ): ((x: number, z: number, y?: number) => number | null) | null {
    this.ensureHeuristicGrid();
    if (!this.h_blocked) return null;
    const cs = this.h_cellSize;
    const w = this.h_width;
    const h = this.h_height;
    const blocked = this.h_blocked;
    const halfDiag = (cs * Math.SQRT2) / 2;
    const seeds: number[] = [];
    for (let cz = 0; cz < h; cz++) {
      for (let cx = 0; cx < w; cx++) {
        const i = cx + cz * w;
        if (blocked[i]) continue;
        const x = this.h_originX + (cx + 0.5) * cs;
        const z = this.h_originZ + (cz + 0.5) * cs;
        if (containsXZ(x, z, halfDiag)) seeds.push(i);
      }
    }
    if (seeds.length === 0) return null;
    const dist = this.dijkstraMulti(seeds);
    const slack = Math.SQRT2 * cs;
    return (x: number, z: number) => {
      const cx = Math.floor((x - this.h_originX) / cs);
      const cz = Math.floor((z - this.h_originZ) / cs);
      if (cx < 0 || cx >= w || cz < 0 || cz >= h) return null;
      const d = dist[cx + cz * w]!;
      if (!Number.isFinite(d)) return null;
      const lb = d - slack;
      return lb > 0 ? lb : 0;
    };
  }

  /** 8-connected Dijkstra from the source cells over the heuristic grid,
   *  treating blocked cells as walls. Diagonal moves cost √2·cellSize.
   *  Diagonals through a blocked cardinal neighbour are disallowed so the
   *  bound never "cuts a corner" through an obstacle. */
  private dijkstraMulti(sources: ReadonlyArray<number>): Float32Array {
    const w = this.h_width;
    const h = this.h_height;
    const cs = this.h_cellSize;
    const blocked = this.h_blocked!;
    const dist = new Float32Array(w * h);
    dist.fill(Infinity);
    const heap = new BinaryHeap<{ i: number; d: number }>((a, b) => a.d - b.d);
    for (const s of sources) {
      dist[s] = 0;
      heap.push({ i: s, d: 0 });
    }
    const diag = Math.SQRT2 * cs;
    while (!heap.isEmpty()) {
      const top = heap.pop()!;
      const i = top.i;
      const d = top.d;
      if (d > dist[i]!) continue;
      const cx = i % w;
      const cz = (i / w) | 0;
      // Relax: store into the Float32Array first, then push the ROUNDED
      // read-back value. Pushing the raw float64 sum makes the stale-entry
      // guard above (`d > dist[i]`) reject the entry after float32 rounding
      // shrinks dist[i] — the cell then never expands and the wave dies at
      // diagonal-valued chokepoints.
      const relax = (ni: number, nd: number): void => {
        if (nd < dist[ni]!) {
          dist[ni] = nd;
          heap.push({ i: ni, d: dist[ni]! });
        }
      };
      // Cardinal neighbours.
      const cardOpen = [false, false, false, false]; // -x, +x, -z, +z
      if (cx > 0 && !blocked[i - 1]) {
        cardOpen[0] = true;
        relax(i - 1, d + cs);
      }
      if (cx + 1 < w && !blocked[i + 1]) {
        cardOpen[1] = true;
        relax(i + 1, d + cs);
      }
      if (cz > 0 && !blocked[i - w]) {
        cardOpen[2] = true;
        relax(i - w, d + cs);
      }
      if (cz + 1 < h && !blocked[i + w]) {
        cardOpen[3] = true;
        relax(i + w, d + cs);
      }
      // Diagonals — require both adjacent cardinals to be open so we never
      // squeeze through a corner. Keeps the bound admissible.
      if (cardOpen[0] && cardOpen[2] && !blocked[i - 1 - w]) relax(i - 1 - w, d + diag);
      if (cardOpen[1] && cardOpen[2] && !blocked[i + 1 - w]) relax(i + 1 - w, d + diag);
      if (cardOpen[0] && cardOpen[3] && !blocked[i - 1 + w]) relax(i - 1 + w, d + diag);
      if (cardOpen[1] && cardOpen[3] && !blocked[i + 1 + w]) relax(i + 1 + w, d + diag);
    }
    return dist;
  }

  /** Build the coarse occupancy + clearance grids used by
   *  `buildGoalLowerBound` and `clearanceAt`. Cell size is sized to the
   *  smallest obstacle so even narrow passages aren't fully fused into a
   *  blocked block. Called once at construction and on `setObstacles`. */
  private buildHeuristicGrid(): void {
    if (this.polygons.length === 0) {
      this.h_blocked = null;
      this.h_clearance = null;
      return;
    }
    // World bounds = polygon AABBs (walkable extent).
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.polygonAABBs.length; i += 4) {
      if (this.polygonAABBs[i]! < minX) minX = this.polygonAABBs[i]!;
      if (this.polygonAABBs[i + 1]! < minZ) minZ = this.polygonAABBs[i + 1]!;
      if (this.polygonAABBs[i + 2]! > maxX) maxX = this.polygonAABBs[i + 2]!;
      if (this.polygonAABBs[i + 3]! > maxZ) maxZ = this.polygonAABBs[i + 3]!;
    }
    if (!isFinite(minX)) {
      this.h_blocked = null;
      this.h_clearance = null;
      return;
    }
    // Cell size: half the smallest obstacle dimension, clamped to [1, 5] m.
    // Smaller cells → tighter bounds but slower Dijkstra. The clamp covers
    // both "no obstacles" and "tiny obstacles" cases.
    let minDim = Infinity;
    for (let i = 0; i < this.obstacleAABBs.length; i += 4) {
      const dx = this.obstacleAABBs[i + 2]! - this.obstacleAABBs[i]!;
      const dz = this.obstacleAABBs[i + 3]! - this.obstacleAABBs[i + 1]!;
      const d = dx < dz ? dx : dz;
      if (d < minDim) minDim = d;
    }
    let cs = isFinite(minDim) ? minDim * 0.5 : 5;
    if (cs < 1) cs = 1;
    if (cs > 5) cs = 5;
    this.h_cellSize = cs;
    this.h_originX = minX;
    this.h_originZ = minZ;
    this.h_width = Math.max(1, Math.ceil((maxX - minX) / cs) + 1);
    this.h_height = Math.max(1, Math.ceil((maxZ - minZ) / cs) + 1);
    const n = this.h_width * this.h_height;
    const blocked = new Uint8Array(n);
    // ---- Pass 1: blocked-cell mask. A cell is blocked when ANY of its four
    // corners is off-mesh OR any obstacle AABB overlaps the cell. Sampling
    // corners (not just the centre) avoids missing thin slivers where the
    // centre happens to land on-mesh but part of the cell hangs off.
    const halfCs = cs * 0.5;
    for (let cz = 0; cz < this.h_height; cz++) {
      for (let cx = 0; cx < this.h_width; cx++) {
        const wx = minX + cx * cs + halfCs;
        const wz = minZ + cz * cs + halfCs;
        const i = cx + cz * this.h_width;
        // Off-mesh test: any of the 4 corners fails → blocked.
        let allOnMesh = true;
        for (let dz = -1; dz <= 1; dz += 2) {
          for (let dx = -1; dx <= 1; dx += 2) {
            const qx = wx + dx * halfCs;
            const qz = wz + dz * halfCs;
            let onMesh = false;
            const polyBuckets = this.candidates(this.polygonGrid, qx, qz, qx, qz);
            if (polyBuckets) {
              outer: for (const bucket of polyBuckets) {
                for (let k = 0; k < bucket.length; k++) {
                  const pi = bucket[k]!;
                  if (aabbContainsPoint(this.polygonAABBs, pi, qx, qz) &&
                      pointInPolygon(qx, qz, this.polygons[pi]!.ring)) {
                    onMesh = true;
                    break outer;
                  }
                }
              }
            } else {
              for (const p of this.polygons) {
                if (pointInPolygon(qx, qz, p.ring)) { onMesh = true; break; }
              }
            }
            if (!onMesh) { allOnMesh = false; break; }
          }
          if (!allOnMesh) break;
        }
        if (!allOnMesh) { blocked[i] = 1; continue; }
        // Obstacle test: any obstacle AABB overlapping this cell blocks it.
        const cellX0 = wx - halfCs, cellZ0 = wz - halfCs;
        const cellX1 = wx + halfCs, cellZ1 = wz + halfCs;
        const obBuckets = this.candidates(
          this.obstacleGrid, cellX0, cellZ0, cellX1, cellZ1,
        );
        let blockedHere = false;
        if (obBuckets) {
          const seen = new Set<number>();
          for (const bucket of obBuckets) {
            for (let k = 0; k < bucket.length; k++) {
              const oi = bucket[k]!;
              if (seen.has(oi)) continue;
              seen.add(oi);
              if (aabbsOverlap(this.obstacleAABBs, oi, cellX0, cellZ0, cellX1, cellZ1)) {
                blockedHere = true;
                break;
              }
            }
            if (blockedHere) break;
          }
        } else {
          for (let oi = 0; oi < this.obstacles.length; oi++) {
            if (aabbsOverlap(this.obstacleAABBs, oi, cellX0, cellZ0, cellX1, cellZ1)) {
              blockedHere = true;
              break;
            }
          }
        }
        if (blockedHere) blocked[i] = 1;
      }
    }
    this.h_blocked = blocked;
    // ---- Pass 2: clearance via 8-connected multi-source Dijkstra from all
    // blocked cells. Result[i] = grid-shortest-path distance from cell i to
    // the nearest blocked cell. `clearanceAt` subtracts cell_diagonal to keep
    // the answer a true lower bound on Euclidean clearance anywhere in cell i.
    const clearance = new Float32Array(n);
    clearance.fill(Infinity);
    const heap = new BinaryHeap<{ i: number; d: number }>((a, b) => a.d - b.d);
    for (let i = 0; i < n; i++) {
      if (blocked[i]) {
        clearance[i] = 0;
        heap.push({ i, d: 0 });
      }
    }
    const w = this.h_width;
    const h = this.h_height;
    const diag = Math.SQRT2 * cs;
    while (!heap.isEmpty()) {
      const top = heap.pop()!;
      const i = top.i;
      const d = top.d;
      if (d > clearance[i]!) continue;
      const cx = i % w;
      const cz = (i / w) | 0;
      const cardOpen = [false, false, false, false];
      if (cx > 0) {
        cardOpen[0] = true;
        const nd = d + cs;
        if (nd < clearance[i - 1]!) { clearance[i - 1] = nd; heap.push({ i: i - 1, d: nd }); }
      }
      if (cx + 1 < w) {
        cardOpen[1] = true;
        const nd = d + cs;
        if (nd < clearance[i + 1]!) { clearance[i + 1] = nd; heap.push({ i: i + 1, d: nd }); }
      }
      if (cz > 0) {
        cardOpen[2] = true;
        const nd = d + cs;
        if (nd < clearance[i - w]!) { clearance[i - w] = nd; heap.push({ i: i - w, d: nd }); }
      }
      if (cz + 1 < h) {
        cardOpen[3] = true;
        const nd = d + cs;
        if (nd < clearance[i + w]!) { clearance[i + w] = nd; heap.push({ i: i + w, d: nd }); }
      }
      if (cardOpen[0] && cardOpen[2]) {
        const ni = i - 1 - w;
        const nd = d + diag;
        if (nd < clearance[ni]!) { clearance[ni] = nd; heap.push({ i: ni, d: nd }); }
      }
      if (cardOpen[1] && cardOpen[2]) {
        const ni = i + 1 - w;
        const nd = d + diag;
        if (nd < clearance[ni]!) { clearance[ni] = nd; heap.push({ i: ni, d: nd }); }
      }
      if (cardOpen[0] && cardOpen[3]) {
        const ni = i - 1 + w;
        const nd = d + diag;
        if (nd < clearance[ni]!) { clearance[ni] = nd; heap.push({ i: ni, d: nd }); }
      }
      if (cardOpen[1] && cardOpen[3]) {
        const ni = i + 1 + w;
        const nd = d + diag;
        if (nd < clearance[ni]!) { clearance[ni] = nd; heap.push({ i: ni, d: nd }); }
      }
    }
    this.h_clearance = clearance;
    this.h_goalKey = -1;
    this.h_goalDist = null;
  }

  // --- test/dynamic mutators (tile-rebuild analog, used by M4/M8) ---

  addOffMeshLink(link: OffMeshLink): void {
    this.offMesh.push(link);
    this._revision++;
  }

  setObstacles(obstacles: Pt[][]): void {
    this.obstacles = obstacles;
    this.obstacleAABBs = buildAABBs(obstacles);
    this.rebuildIndex();
    // Defer the heuristic-grid rebuild to its next consumer; collision
    // queries only need the spatial index rebuilt above. Drop the goal memo
    // now so nothing reads a stale field before the rebuild happens.
    this.h_dirty = true;
    this.h_goalKey = -1;
    this.h_goalDist = null;
    this._revision++;
  }

  private ensureHeuristicGrid(): void {
    if (!this.h_dirty) return;
    this.h_dirty = false;
    this.buildHeuristicGrid();
  }
}

// ---------------------------------------------------------------------------
// AABB helpers — `Float64Array` packed as [x0,z0,x1,z1, x0,z0,x1,z1, …] so
// the spatial-index hot path stays cache-friendly and allocation-free.

function buildAABBs(rings: ReadonlyArray<ReadonlyArray<Pt>>): Float64Array {
  const out = new Float64Array(rings.length * 4);
  for (let i = 0; i < rings.length; i++) {
    let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
    const r = rings[i]!;
    for (let j = 0; j < r.length; j++) {
      const x = r[j]![0], z = r[j]![1];
      if (x < x0) x0 = x;
      if (z < z0) z0 = z;
      if (x > x1) x1 = x;
      if (z > z1) z1 = z;
    }
    out[i * 4] = x0;
    out[i * 4 + 1] = z0;
    out[i * 4 + 2] = x1;
    out[i * 4 + 3] = z1;
  }
  return out;
}

function aabbsOverlap(
  a: Float64Array, i: number,
  qx0: number, qz0: number, qx1: number, qz1: number,
): boolean {
  return (
    a[i * 4]! <= qx1 &&
    a[i * 4 + 2]! >= qx0 &&
    a[i * 4 + 1]! <= qz1 &&
    a[i * 4 + 3]! >= qz0
  );
}

function aabbContainsPoint(a: Float64Array, i: number, x: number, z: number): boolean {
  return (
    a[i * 4]! <= x &&
    a[i * 4 + 2]! >= x &&
    a[i * 4 + 1]! <= z &&
    a[i * 4 + 3]! >= z
  );
}

/** Squared Euclidean distance from point (x, z) to the AABB at index `i` in
 *  the packed Float64Array. Returns 0 when the point is inside the AABB. */
function obstacleAABBDistance2(
  a: Float64Array, i: number, x: number, z: number,
): number {
  const x0 = a[i * 4]!, z0 = a[i * 4 + 1]!;
  const x1 = a[i * 4 + 2]!, z1 = a[i * 4 + 3]!;
  let dx = 0;
  if (x < x0) dx = x0 - x;
  else if (x > x1) dx = x - x1;
  let dz = 0;
  if (z < z0) dz = z0 - z;
  else if (z > z1) dz = z - z1;
  return dx * dx + dz * dz;
}
