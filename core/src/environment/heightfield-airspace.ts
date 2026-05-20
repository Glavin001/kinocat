// HeightfieldAirspace — adds continuous ground-elevation collision to the
// AirspaceWorld contract. The existing InMemoryAirspace only knows about
// axis-aligned obstacle boxes and a flat altitude floor; this composes that
// with a user-supplied `height(x, z)` sampler so the aircraft planner can
// fly over rolling terrain (hills, ridges, canyons) without representing
// every terrain feature as a stack of AABBs.
//
// Soundness contract for the sampler: must be a deterministic function of
// (x, z) with no spatial frequencies finer than ~1 world unit relative to the
// agent's footprint. The implementation conservatively bounds the terrain
// inside the OBB's world-axis-aligned extent by sampling at its corners +
// centre; for smooth terrain this is exact within a small safety margin.
// Discontinuous / spiky terrain MUST be voxelized into AABBs instead.

import { InMemoryAirspace } from './airspace-world';
import type {
  AirspaceWorld,
  AirspaceOptions,
} from './airspace-world';
import {
  makeOBB,
  obbWorldExtentInto,
  poseToOBBInto,
  type OBB,
  type Pose,
  type Vec3,
} from '../internal/obb';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

/** Continuous ground-elevation lookup. Returns the terrain Y at world (x, z). */
export type HeightfieldSampler = (x: number, z: number) => number;

export interface HeightfieldAirspaceOptions extends AirspaceOptions {
  /** Required. Terrain elevation in world units, sampled at world (x, z). */
  sampler: HeightfieldSampler;
  /**
   * Extra clearance the agent's underside must keep above the sampled
   * terrain. World units. Default 0 — the agent can just kiss the terrain.
   * Bump it up to bias the planner toward conservative ground clearance.
   */
  sampleMargin?: number;
  /**
   * Grid resolution (world units) for `clearAABB` heightfield sampling. The
   * static broadphase sweeps a rectangle; we sample on a grid no coarser
   * than this. Default 4 — fine enough for analytic terrain, coarse enough
   * to keep the broadphase cheap. The OBB-aware `clear()` path uses a fixed
   * 5-point sampling pattern and ignores this.
   */
  clearAABBStep?: number;
}

/**
 * Composes InMemoryAirspace's obstacle/zone collision with a continuous
 * terrain heightfield. Implements the same AirspaceWorld interface, so
 * AircraftEnvironment uses it unchanged.
 */
export class HeightfieldAirspace implements AirspaceWorld {
  private readonly inner: InMemoryAirspace;
  private readonly sampler: HeightfieldSampler;
  private readonly margin: number;
  private readonly clearAABBStep: number;
  // Reused scratch to avoid per-call allocations on the hot path.
  private readonly _obb: OBB = makeOBB();
  private readonly _extMin: Vec3 = [0, 0, 0];
  private readonly _extMax: Vec3 = [0, 0, 0];
  private rec: PerfRecorder = NULL_RECORDER;

  constructor(opts: HeightfieldAirspaceOptions) {
    this.inner = new InMemoryAirspace(opts);
    this.sampler = opts.sampler;
    this.margin = opts.sampleMargin ?? 0;
    this.clearAABBStep = opts.clearAABBStep ?? 4;
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
    this.inner.attachRecorder(rec);
  }

  /** Maximum terrain Y over a rectangle, sampled at 4 corners + 4 edge
   *  midpoints + the centre (a 3×3 grid). Conservative for smooth terrain
   *  whose lateral feature size is ≳ half the rectangle's diagonal — the
   *  expected regime for analytic dogfight terrain. Spikier terrain MUST be
   *  represented as discrete AABB obstacles. */
  private maxTerrainInRect(
    minX: number,
    minZ: number,
    maxX: number,
    maxZ: number,
  ): number {
    const s = this.sampler;
    const mx = (minX + maxX) * 0.5;
    const mz = (minZ + maxZ) * 0.5;
    let m = s(minX, minZ);
    const v1 = s(mx, minZ);
    if (v1 > m) m = v1;
    const v2 = s(maxX, minZ);
    if (v2 > m) m = v2;
    const v3 = s(minX, mz);
    if (v3 > m) m = v3;
    const v4 = s(mx, mz);
    if (v4 > m) m = v4;
    const v5 = s(maxX, mz);
    if (v5 > m) m = v5;
    const v6 = s(minX, maxZ);
    if (v6 > m) m = v6;
    const v7 = s(mx, maxZ);
    if (v7 > m) m = v7;
    const v8 = s(maxX, maxZ);
    if (v8 > m) m = v8;
    return m;
  }

  /** Maximum terrain Y over a rectangle, sampled on a regular grid no coarser
   *  than `step`. Used by `clearAABB` where the swept envelope can be wide
   *  enough that corner-only sampling would alias over a hill in the middle. */
  private maxTerrainInRectGrid(
    minX: number,
    minZ: number,
    maxX: number,
    maxZ: number,
    step: number,
  ): number {
    const dx = maxX - minX;
    const dz = maxZ - minZ;
    const nx = Math.max(1, Math.ceil(dx / step));
    const nz = Math.max(1, Math.ceil(dz / step));
    const sx = dx / nx;
    const sz = dz / nz;
    const s = this.sampler;
    let m = -Infinity;
    for (let iz = 0; iz <= nz; iz++) {
      const z = minZ + iz * sz;
      for (let ix = 0; ix <= nx; ix++) {
        const x = minX + ix * sx;
        const h = s(x, z);
        if (h > m) m = h;
      }
    }
    return m;
  }

  clear(pose: Pose, half: [number, number, number], t: number): boolean {
    // Static obstacles + moving zones + floor/ceiling — delegate to the inner
    // airspace which already runs the OBB SAT broadphase.
    if (!this.inner.clear(pose, half, t)) return false;
    // Terrain — sample the heightfield over the OBB's world-axis-aligned
    // footprint. The OBB extent is conservatively larger than the body's
    // actual XZ projection, so a hill bumped only by the OBB's wingtip at
    // banking will still be detected.
    poseToOBBInto(this._obb, pose, half);
    obbWorldExtentInto(this._obb, this._extMin, this._extMax);
    const bottomY = this._extMin[1];
    const terrainMax = this.maxTerrainInRect(
      this._extMin[0],
      this._extMin[2],
      this._extMax[0],
      this._extMax[2],
    );
    if (terrainMax > bottomY - this.margin) {
      this.rec.counters.collisionRejects++;
      return false;
    }
    return true;
  }

  clearAABB(
    minX: number,
    minY: number,
    minZ: number,
    maxX: number,
    maxY: number,
    maxZ: number,
  ): boolean {
    // First defer to the inner static broadphase: cheap-rejects on boxes,
    // bails (returns false) if any moving zones exist — the aircraft env
    // then falls back to the per-substep narrowphase via `clear()`.
    if (!this.inner.clearAABB(minX, minY, minZ, maxX, maxY, maxZ)) return false;
    const terrainMax = this.maxTerrainInRectGrid(
      minX,
      minZ,
      maxX,
      maxZ,
      this.clearAABBStep,
    );
    return terrainMax <= minY - this.margin;
  }
}
