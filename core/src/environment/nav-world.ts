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
 *  `polygons`; `obstacles` are holes that block footprints/segments. */
export class InMemoryNavWorld implements NavWorld {
  private _revision = 0;
  private offMesh: OffMeshLink[] = [];

  constructor(
    private polygons: NavPolygon[],
    private obstacles: Pt[][] = [],
  ) {}

  get revision(): number {
    return this._revision;
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
    for (const ob of this.obstacles) {
      if (pointInPolygon(x, z, ob)) return null;
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
    for (const [x, z] of fp) {
      if (this.polygonAt(x, z) === null) return false;
    }
    // ...and the footprint must not overlap any obstacle.
    for (const ob of this.obstacles) {
      if (polygonsIntersect(fp, ob)) return false;
    }
    return true;
  }

  segmentClear(x0: number, z0: number, x1: number, z1: number): boolean {
    if (this.polygonAt(x0, z0) === null || this.polygonAt(x1, z1) === null) {
      return false;
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
    this._revision++;
  }
}
