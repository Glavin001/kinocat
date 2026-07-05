// kinocat/adapters/navcat — implements the core's NavWorld seam over a real
// navcat NavMesh. The core never imports navcat; only this subpath does.

import {
  createFindNearestPolyResult,
  findNearestPoly,
  createGetClosestPointOnPolyResult,
  getClosestPointOnPoly,
  raycast,
  DEFAULT_QUERY_FILTER,
} from 'navcat';
import { generateSoloNavMesh } from 'navcat/blocks';
import type {
  NavMesh,
  NodeRef,
  QueryFilter,
  FindNearestPolyResult,
  GetClosestPointOnPolyResult,
} from 'navcat';
import type {
  NavWorld,
  OffMeshLink,
  PolygonRef,
} from '../../environment/nav-world';
import {
  ChfClearanceField,
  type ClearanceFieldOptions,
} from './compact-heightfield';
import { ChfGoalDistanceField } from './chf-distance-field';
import type { CompactHeightfield } from 'navcat';

export interface NavcatWorldOptions {
  queryFilter?: QueryFilter;
  /** Horizontal snap tolerance for treating a point as "on this polygon". */
  horizontalTolerance?: number;
  /** Vertical search half-extent for findNearestPoly. */
  verticalExtent?: number;
  /** Y used for planar queries (Y is derived from polygon containment). */
  queryHeight?: number;
  /** Build the O(1) CompactHeightfield clearance field (Opt 1, spec §10.2)
   *  so `clearanceAt` is available for VehicleEnvironment's broadphase. */
  clearanceField?: boolean | ClearanceFieldOptions;
}

export class NavcatWorld implements NavWorld {
  private _revision = 0;
  private readonly offLinks: OffMeshLink[] = [];
  private readonly npResult: FindNearestPolyResult = createFindNearestPolyResult();
  private readonly cpResult: GetClosestPointOnPolyResult =
    createGetClosestPointOnPolyResult();
  private readonly filter: QueryFilter;
  private readonly hTol: number;
  private readonly vExt: number;
  private readonly qy: number;
  private chf?: ChfClearanceField;
  private rawChf?: CompactHeightfield;
  private chfOpts: ClearanceFieldOptions = {};
  private goalLB?: { key: string; field: ChfGoalDistanceField };

  constructor(
    private navMesh: NavMesh,
    opts: NavcatWorldOptions = {},
  ) {
    this.filter = opts.queryFilter ?? DEFAULT_QUERY_FILTER;
    this.hTol = opts.horizontalTolerance ?? 0.25;
    this.vExt = opts.verticalExtent ?? 1e4;
    this.qy = opts.queryHeight ?? 0;
  }

  get revision(): number {
    return this._revision;
  }

  /** Tile-rebuild hook: invalidate caches and bump the revision counter. The
   *  memoised goal field is dropped eagerly — a stale grid lower bound is
   *  INADMISSIBLE after additive changes (a new bridge shortens true paths
   *  below the old bound), not merely suboptimal. */
  bumpRevision(): void {
    this._revision++;
    this.goalLB = undefined;
  }

  addOffLink(link: OffMeshLink): void {
    this.offLinks.push(link);
  }

  /** Swap in a rebuilt navmesh (e.g. after destroying terrain) without
   *  reconstructing the world. Live queries (`polygonAt`/`segmentClear`) read
   *  the new mesh immediately; the revision bump invalidates derived caches.
   *
   *  Pass the regenerated CompactHeightfield to keep `clearanceAt` /
   *  `buildGoalLowerBound` available. When `chf` is omitted both oracles are
   *  CLEARED, not kept: the clearance field is an early-ACCEPT broadphase, so
   *  a stale one can approve footprints over destroyed ground. The planner
   *  falls back to the Reeds-Shepp heuristic (still admissible).
   *
   *  Off-mesh links are re-resolved against the new mesh (polygon node ids
   *  change per mesh); links whose endpoints are no longer on the mesh are
   *  dropped. Note navcat-side `addOffMeshConnection`s live inside the mesh
   *  object — a caller regenerating a mesh re-runs `annotateJumpLinks`; this
   *  heals only the adapter's mirror. */
  swapNavMesh(
    navMesh: NavMesh,
    chf?: CompactHeightfield,
    chfOpts?: ClearanceFieldOptions,
  ): void {
    this.navMesh = navMesh;
    if (chf) {
      this.attachClearanceField(chf, chfOpts ?? this.chfOpts);
    } else {
      this.chf = undefined;
      this.rawChf = undefined;
    }
    this.bumpRevision();
    // Re-resolve mirrored off-mesh links against the new mesh.
    const kept: OffMeshLink[] = [];
    for (const link of this.offLinks) {
      const from = this.polygonAt(link.start[0], link.start[2]);
      const to = this.polygonAt(link.end[0], link.end[2]);
      if (from && to) kept.push({ ...link, from, to });
    }
    this.offLinks.length = 0;
    this.offLinks.push(...kept);
  }

  /** Attach a CompactHeightfield clearance field (from the generator's
   *  intermediates) so `clearanceAt` becomes available. */
  attachClearanceField(
    chf: CompactHeightfield,
    opts: ClearanceFieldOptions = {},
  ): void {
    this.chf = new ChfClearanceField(chf, opts);
    this.rawChf = chf;
    this.chfOpts = opts;
  }

  /** O(1) lower-bound clearance, or null when no field is attached / the
   *  point is off-field — callers then fall back to the exact check. */
  clearanceAt(x: number, z: number, queryY?: number): number | null {
    return this.chf ? this.chf.clearanceAt(x, z, queryY) : null;
  }

  /** Admissible obstacle-aware distance-to-goal oracle (Opt 2, spec §10.3),
   *  memoised per goal. null when no CompactHeightfield is attached or the
   *  goal is off-mesh — the heuristic then uses the Reeds-Shepp term alone. */
  buildGoalLowerBound(
    gx: number,
    gz: number,
    gy?: number,
  ): ((x: number, z: number, y?: number) => number | null) | null {
    if (!this.rawChf) return null;
    const key = `${this._revision}|${gx},${gz},${gy ?? ''}`;
    if (!this.goalLB || this.goalLB.key !== key) {
      const field = new ChfGoalDistanceField(this.rawChf, gx, gz, gy);
      if (!field.available) return null;
      this.goalLB = { key, field };
    }
    const f = this.goalLB.field;
    return (x: number, z: number, y?: number) => f.lookup(x, z, y);
  }

  polygonAt(x: number, z: number): PolygonRef | null {
    const r = findNearestPoly(
      this.npResult,
      this.navMesh,
      [x, this.qy, z],
      [this.hTol, this.vExt, this.hTol],
      this.filter,
    );
    if (!r.success) return null;
    const dx = r.position[0] - x;
    const dz = r.position[2] - z;
    if (dx * dx + dz * dz > this.hTol * this.hTol) return null;
    return { id: r.nodeRef, cx: r.position[0], cz: r.position[2], y: r.position[1] };
  }

  heightAt(poly: PolygonRef, x: number, z: number): number {
    const r = getClosestPointOnPoly(
      this.cpResult,
      this.navMesh,
      poly.id as NodeRef,
      [x, this.qy, z],
    );
    return r.success ? r.position[1] : poly.y;
  }

  footprintClear(footprint: ReadonlyArray<readonly [number, number]>): boolean {
    for (const [x, z] of footprint) {
      if (this.polygonAt(x, z) === null) return false;
    }
    for (let i = 0; i < footprint.length; i++) {
      const a = footprint[i]!;
      const b = footprint[(i + 1) % footprint.length]!;
      if (!this.segmentClear(a[0], a[1], b[0], b[1])) return false;
    }
    return true;
  }

  segmentClear(x0: number, z0: number, x1: number, z1: number): boolean {
    const start = findNearestPoly(
      this.npResult,
      this.navMesh,
      [x0, this.qy, z0],
      [this.hTol, this.vExt, this.hTol],
      this.filter,
    );
    if (!start.success) return false;
    const dx = start.position[0] - x0;
    const dz = start.position[2] - z0;
    if (dx * dx + dz * dz > this.hTol * this.hTol) return false;
    const res = raycast(
      this.navMesh,
      start.nodeRef,
      [x0, this.qy, z0],
      [x1, this.qy, z1],
      this.filter,
    );
    // navcat: t === Number.MAX_VALUE ⇒ no wall hit ⇒ segment stays on-mesh.
    return res.t === Number.MAX_VALUE;
  }

  offMeshFrom(poly: PolygonRef): ReadonlyArray<OffMeshLink> {
    return this.offLinks.filter((l) => l.from.id === poly.id);
  }
}

export interface NavWorldFromMeshResult {
  world: NavcatWorld;
  navMesh: NavMesh;
  intermediates: ReturnType<typeof generateSoloNavMesh>['intermediates'];
  /** The generated CompactHeightfield (distance field) — handy for the
   *  `navcat/three` `createCompactHeightfieldDistancesHelper` overlay. */
  compactHeightfield: ReturnType<
    typeof generateSoloNavMesh
  >['intermediates']['compactHeightfield'];
}

/** Build a NavcatWorld from a triangle soup via navcat's solo generator. */
export function navWorldFromTriangleMesh(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  options: Partial<Parameters<typeof generateSoloNavMesh>[1]> = {},
  worldOptions: NavcatWorldOptions = {},
): NavWorldFromMeshResult {
  const cellSize = options.cellSize ?? 0.3;
  const radiusWorld = options.walkableRadiusWorld ?? 0;
  const climbWorld = options.walkableClimbWorld ?? 0.5;
  const heightWorld = options.walkableHeightWorld ?? 2;
  const full = {
    cellSize,
    cellHeight: options.cellHeight ?? 0.2,
    walkableRadiusWorld: radiusWorld,
    walkableRadiusVoxels: Math.ceil(radiusWorld / cellSize),
    walkableClimbWorld: climbWorld,
    walkableClimbVoxels: Math.ceil(climbWorld / (options.cellHeight ?? 0.2)),
    walkableHeightWorld: heightWorld,
    walkableHeightVoxels: Math.ceil(heightWorld / (options.cellHeight ?? 0.2)),
    walkableSlopeAngleDegrees: options.walkableSlopeAngleDegrees ?? 45,
    borderSize: options.borderSize ?? 0,
    minRegionArea: options.minRegionArea ?? 1,
    mergeRegionArea: options.mergeRegionArea ?? 4,
    maxSimplificationError: options.maxSimplificationError ?? 1.3,
    maxEdgeLength: options.maxEdgeLength ?? 12,
    maxVerticesPerPoly: options.maxVerticesPerPoly ?? 6,
    detailSampleDistance: options.detailSampleDistance ?? 6,
    detailSampleMaxError: options.detailSampleMaxError ?? 1,
  };
  const res = generateSoloNavMesh({ positions, indices }, full);
  const world = new NavcatWorld(res.navMesh, worldOptions);
  const cf = worldOptions.clearanceField;
  if (cf) {
    world.attachClearanceField(
      res.intermediates.compactHeightfield,
      typeof cf === 'object' ? cf : {},
    );
  }
  return {
    world,
    navMesh: res.navMesh,
    intermediates: res.intermediates,
    compactHeightfield: res.intermediates.compactHeightfield,
  };
}

export { annotateJumpLinks } from './offmesh';
export { markTileRebuilt } from './tile-rebuild';
export type { StaticAffordanceMetadata, JumpCandidate } from './types';
export { navWorldFromMeshes } from './from-meshes';
export type { NavWorldFromMeshesOptions } from './from-meshes';
