// The aircraft planner's only collision coupling. Volumetric and oriented:
// the agent is an OBB (length × span × height), oriented by yaw + pitch +
// roll from the searched state, so the planner can knife-edge through slots
// too narrow for a level-wing footprint. Static box volumes use a uniform-
// grid broadphase + SAT; moving spherical no-fly zones use closest-point-
// on-OBB.

import type { Predict } from '../predict/types';
import {
  computeOBBSepAxes,
  makeOBB,
  makeOBBSepAxes,
  obbHitsAABBCached,
  obbHitsSphereXYZ,
  obbWorldExtentInto,
  poseToOBBInto,
  type OBB,
  type OBBSepAxes,
  type Pose,
  type Vec3,
} from '../internal/obb';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

/** Axis-aligned box volume in world coordinates. */
export interface AABB {
  min: [number, number, number];
  max: [number, number, number];
}

/** A moving spherical no-fly zone: centre over time + radius. */
export interface MovingZone {
  predict: Predict<{ x: number; y: number; z: number }>;
  radius: number;
}

/** Is the agent's OBB at (pose) at absolute time `t` free of obstacles? The
 *  aircraft Environment depends on nothing else. */
export interface AirspaceWorld {
  clear(pose: Pose, half: [number, number, number], t: number): boolean;
  /** Optional: wire per-search counters (collisions, broadphase skips). */
  attachRecorder?(rec: PerfRecorder): void;
}

export interface AirspaceOptions {
  /** Inclusive flyable altitude band (defaults: unbounded). */
  floor?: number;
  ceiling?: number;
  boxes?: AABB[];
  zones?: MovingZone[];
  /**
   * Uniform-grid broadphase cell size for static `boxes` (world units). With
   * many boxes, `clear()` collapses from O(boxes) SAT tests to ~O(few) per
   * call. Default: median of box max-extent × 2 (or 8 if no boxes). Pass
   * `false` to disable (debug only).
   */
  broadphaseCell?: number | false;
}

interface BoxEntry {
  box: AABB;
  /** Cells (linear ids in the grid) this box overlaps. */
  cellIds: number[];
}

export class InMemoryAirspace implements AirspaceWorld {
  private readonly floor: number;
  private readonly ceiling: number;
  private readonly boxes: AABB[];
  private readonly zones: MovingZone[];
  // Broadphase grid (XZ plane; aircraft cluster around level Y).
  private readonly bpEnabled: boolean;
  private readonly bpCell: number;
  private readonly bpMinX: number;
  private readonly bpMinZ: number;
  private readonly bpCols: number;
  private readonly bpRows: number;
  private readonly bpCells: Uint32Array[];
  private readonly bpEntries: BoxEntry[];
  // Per-query scratch — avoids per-call Set allocation. The visited token
  // increments per query; cells store the token they were last visited at.
  private readonly bpVisited: Uint32Array;
  private bpToken: number = 0;
  private rec: PerfRecorder = NULL_RECORDER;
  // Scratch OBB + separating-axes table + extent vectors reused across
  // clear() calls. clear() is called >10⁶ times per long aircraft search;
  // these saves ~9 allocations per call.
  private readonly _obb: OBB = makeOBB();
  private readonly _sep: OBBSepAxes = makeOBBSepAxes();
  private readonly _extMin: Vec3 = [0, 0, 0];
  private readonly _extMax: Vec3 = [0, 0, 0];

  constructor(opts: AirspaceOptions = {}) {
    this.floor = opts.floor ?? -Infinity;
    this.ceiling = opts.ceiling ?? Infinity;
    this.boxes = opts.boxes ?? [];
    this.zones = opts.zones ?? [];

    const bpOpt = opts.broadphaseCell;
    this.bpEnabled = bpOpt !== false && this.boxes.length > 0;
    if (!this.bpEnabled) {
      this.bpCell = 0;
      this.bpMinX = 0;
      this.bpMinZ = 0;
      this.bpCols = 0;
      this.bpRows = 0;
      this.bpCells = [];
      this.bpEntries = [];
      this.bpVisited = new Uint32Array(0);
      return;
    }
    // Default cell size from median box max-extent.
    let cell: number;
    if (typeof bpOpt === 'number' && bpOpt > 0) {
      cell = bpOpt;
    } else {
      const extents: number[] = [];
      for (const b of this.boxes) {
        const dx = b.max[0] - b.min[0];
        const dz = b.max[2] - b.min[2];
        extents.push(Math.max(dx, dz));
      }
      extents.sort((a, b) => a - b);
      const med = extents.length > 0 ? extents[Math.floor(extents.length / 2)]! : 8;
      cell = Math.max(2, med * 2);
    }
    this.bpCell = cell;

    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const b of this.boxes) {
      if (b.min[0] < minX) minX = b.min[0];
      if (b.max[0] > maxX) maxX = b.max[0];
      if (b.min[2] < minZ) minZ = b.min[2];
      if (b.max[2] > maxZ) maxZ = b.max[2];
    }
    // Pad by a cell on each side so OBB AABBs that overshoot the box cloud
    // still hit a cell (their cell ids clamp into the grid via min/max).
    this.bpMinX = minX - cell;
    this.bpMinZ = minZ - cell;
    this.bpCols = Math.max(1, Math.ceil((maxX - minX) / cell) + 2);
    this.bpRows = Math.max(1, Math.ceil((maxZ - minZ) / cell) + 2);
    const total = this.bpCols * this.bpRows;
    const cellBuckets: number[][] = Array.from({ length: total }, () => []);
    const entries: BoxEntry[] = [];
    for (let i = 0; i < this.boxes.length; i++) {
      const b = this.boxes[i]!;
      const c0 = Math.max(0, Math.floor((b.min[0] - this.bpMinX) / cell));
      const c1 = Math.min(this.bpCols - 1, Math.floor((b.max[0] - this.bpMinX) / cell));
      const r0 = Math.max(0, Math.floor((b.min[2] - this.bpMinZ) / cell));
      const r1 = Math.min(this.bpRows - 1, Math.floor((b.max[2] - this.bpMinZ) / cell));
      const ids: number[] = [];
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const id = r * this.bpCols + c;
          cellBuckets[id]!.push(i);
          ids.push(id);
        }
      }
      entries.push({ box: b, cellIds: ids });
    }
    this.bpCells = cellBuckets.map((arr) => Uint32Array.from(arr));
    this.bpEntries = entries;
    this.bpVisited = new Uint32Array(this.boxes.length);
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
  }

  clear(pose: Pose, half: [number, number, number], t: number): boolean {
    this.rec.counters.collisionChecks++;
    const obb = this._obb;
    const extMin = this._extMin;
    const extMax = this._extMax;
    poseToOBBInto(obb, pose, half);
    obbWorldExtentInto(obb, extMin, extMax);
    if (extMin[1] < this.floor || extMax[1] > this.ceiling) {
      this.rec.counters.collisionRejects++;
      return false;
    }
    // Compute SAT cross-axes lazily — only when we will actually run SAT.
    // Many calls bail out on AABB pre-reject without needing them.
    let sepReady = false;
    const sep = this._sep;
    if (this.bpEnabled) {
      const cell = this.bpCell;
      const c0 = Math.max(0, Math.floor((extMin[0] - this.bpMinX) / cell));
      const c1 = Math.min(this.bpCols - 1, Math.floor((extMax[0] - this.bpMinX) / cell));
      const r0 = Math.max(0, Math.floor((extMin[2] - this.bpMinZ) / cell));
      const r1 = Math.min(this.bpRows - 1, Math.floor((extMax[2] - this.bpMinZ) / cell));
      const token = ++this.bpToken;
      // Wrap-around protection (unlikely in practice; reset on overflow).
      if (token === 0) {
        this.bpVisited.fill(0);
      }
      const visited = this.bpVisited;
      for (let r = r0; r <= r1; r++) {
        const rowBase = r * this.bpCols;
        for (let c = c0; c <= c1; c++) {
          const cellBoxes = this.bpCells[rowBase + c];
          if (!cellBoxes) continue;
          for (let k = 0; k < cellBoxes.length; k++) {
            const idx = cellBoxes[k]!;
            if (visited[idx] === token) continue;
            visited[idx] = token;
            const b = this.bpEntries[idx]!.box;
            // Cheap AABB-vs-AABB pre-reject on world-extent.
            if (
              extMax[0] < b.min[0] || extMin[0] > b.max[0] ||
              extMax[1] < b.min[1] || extMin[1] > b.max[1] ||
              extMax[2] < b.min[2] || extMin[2] > b.max[2]
            ) {
              this.rec.counters.broadphaseSkips++;
              continue;
            }
            if (!sepReady) {
              computeOBBSepAxes(obb, sep);
              sepReady = true;
            }
            if (obbHitsAABBCached(obb, sep, b.min, b.max)) {
              this.rec.counters.collisionRejects++;
              return false;
            }
          }
        }
      }
    } else {
      for (let i = 0; i < this.boxes.length; i++) {
        const b = this.boxes[i]!;
        if (!sepReady) {
          computeOBBSepAxes(obb, sep);
          sepReady = true;
        }
        if (obbHitsAABBCached(obb, sep, b.min, b.max)) {
          this.rec.counters.collisionRejects++;
          return false;
        }
      }
    }
    for (let i = 0; i < this.zones.length; i++) {
      const zone = this.zones[i]!;
      this.rec.counters.predictCalls++;
      const c = zone.predict(t);
      if (!c) continue;
      if (obbHitsSphereXYZ(obb, c.x, c.y, c.z, zone.radius)) {
        this.rec.counters.collisionRejects++;
        return false;
      }
    }
    return true;
  }
}
