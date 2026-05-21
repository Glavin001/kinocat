// THREE meshes → NavcatWorld convenience. The terrain plane, building boxes,
// and ramp meshes a demo already builds for rendering double as the
// triangle-soup input to navcat's solo generator: marker triangles steeper
// than `walkableSlopeAngleDegrees` are dropped, the rest become the navmesh
// the planner walks. One call.

import type { Mesh } from 'three';
import { getPositionsAndIndices } from 'navcat/three';
import {
  navWorldFromTriangleMesh,
  type NavcatWorldOptions,
  type NavWorldFromMeshResult,
} from './index';

export interface NavWorldFromMeshesOptions {
  /** Generator options (cellSize, walkableSlopeAngleDegrees, …). Defaults
   *  match `navWorldFromTriangleMesh`. */
  generator?: Parameters<typeof navWorldFromTriangleMesh>[2];
  world?: NavcatWorldOptions;
}

/** Build a `NavcatWorld` directly from a set of THREE meshes. Use this when
 *  your scene already has geometry on the GPU: terrain + obstacles → navmesh
 *  with no parallel triangle-soup bookkeeping. Returns the same shape as
 *  `navWorldFromTriangleMesh` plus the merged positions/indices for
 *  inspection / re-use (e.g. as a Rapier `TriMesh` collider). */
export function navWorldFromMeshes(
  meshes: Mesh[],
  opts: NavWorldFromMeshesOptions = {},
): NavWorldFromMeshResult & {
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
} {
  const [positions, indices] = getPositionsAndIndices(meshes);
  const res = navWorldFromTriangleMesh(positions, indices, opts.generator ?? {}, opts.world);
  return { ...res, positions, indices };
}
