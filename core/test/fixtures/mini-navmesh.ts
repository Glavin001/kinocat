// Programmatic navcat navmesh fixtures (no asset files, no browser). Two
// flat quad "islands" separated by a non-walkable gap, for adapter tests.

export interface TriMesh {
  positions: number[];
  indices: number[];
}

/** Append an upward-facing (CCW from +Y) flat quad at height y. */
function addQuad(
  mesh: TriMesh,
  x0: number,
  z0: number,
  x1: number,
  z1: number,
  y: number,
): void {
  const b = mesh.positions.length / 3;
  mesh.positions.push(
    x0, y, z0, // 0
    x1, y, z0, // 1
    x1, y, z1, // 2
    x0, y, z1, // 3
  );
  // upward normals: (0,3,2) and (0,2,1)
  mesh.indices.push(b + 0, b + 3, b + 2, b + 0, b + 2, b + 1);
}

/** Island A: x∈[0,8] z∈[0,10]; Island B: x∈[14,22] z∈[0,10]; gap x∈(8,14). */
export function twoIslandsMesh(): TriMesh {
  const mesh: TriMesh = { positions: [], indices: [] };
  addQuad(mesh, 0, 0, 8, 10, 0);
  addQuad(mesh, 14, 0, 22, 10, 0);
  return mesh;
}

export function singlePlaneMesh(): TriMesh {
  const mesh: TriMesh = { positions: [], indices: [] };
  addQuad(mesh, 0, 0, 30, 0 + 20, 0);
  return mesh;
}

/** The 30×20 plane with a rectangular hole punched out — "the ground was
 *  destroyed under you". Four quads frame the hole so everything else stays
 *  walkable. */
export function planeWithHoleMesh(hole: {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
}): TriMesh {
  const mesh: TriMesh = { positions: [], indices: [] };
  if (hole.x0 > 0) addQuad(mesh, 0, 0, hole.x0, 20, 0); // west
  if (hole.x1 < 30) addQuad(mesh, hole.x1, 0, 30, 20, 0); // east
  if (hole.z0 > 0) addQuad(mesh, hole.x0, 0, hole.x1, hole.z0, 0); // south strip
  if (hole.z1 < 20) addQuad(mesh, hole.x0, hole.z1, hole.x1, 20, 0); // north strip
  return mesh;
}
