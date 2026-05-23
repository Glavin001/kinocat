// The core's only model of "the world". kinocat never imports navcat;
// VehicleEnvironment / HumanoidEnvironment consume this seam. InMemoryNavWorld
// (polygon soup, zero deps) makes the whole algorithmic core unit-testable
// without any external runtime. adapters/navcat implements NavWorld over a
// real navcat NavMesh.

import { pointInPolygon, polygonsIntersect, segmentsIntersect, type Pt } from '../internal/geom';

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

  constructor(
    private polygons: NavPolygon[],
    private obstacles: Pt[][] = [],
  ) {
    this.obstacleAABBs = buildAABBs(this.obstacles);
    this.polygonAABBs = buildAABBs(this.polygons.map((p) => p.ring));
    this.rebuildIndex();
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

  // --- test/dynamic mutators (tile-rebuild analog, used by M4/M8) ---

  addOffMeshLink(link: OffMeshLink): void {
    this.offMesh.push(link);
    this._revision++;
  }

  setObstacles(obstacles: Pt[][]): void {
    this.obstacles = obstacles;
    this.obstacleAABBs = buildAABBs(obstacles);
    this.rebuildIndex();
    this._revision++;
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
