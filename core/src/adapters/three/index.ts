// kinocat/adapters/three — debug visualization helpers (mirrors navcat's
// helper pattern). three is an OPTIONAL peer; only this subpath imports it.

import * as THREE from 'three';

export interface PlanarPoint {
  x: number;
  z: number;
}

export interface MotionPrimitiveLike {
  primitives: ReadonlyArray<{ sweep: ReadonlyArray<{ x: number; z: number }> }>;
}

function toVecs(pts: ReadonlyArray<PlanarPoint>, y: number): THREE.Vector3[] {
  return pts.map((p) => new THREE.Vector3(p.x, y, p.z));
}

/** A polyline for a planned path (states with x,z), drawn at height `y`. */
export function createPlanPathHelper(
  path: ReadonlyArray<PlanarPoint>,
  opts: { y?: number; color?: number } = {},
): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(toVecs(path, opts.y ?? 0.05));
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0x44ddff });
  return new THREE.Line(geo, mat);
}

/** A closed loop for a world-space footprint polygon. */
export function createFootprintHelper(
  footprint: ReadonlyArray<readonly [number, number]>,
  opts: { y?: number; color?: number } = {},
): THREE.LineLoop {
  const pts = footprint.map((p) => ({ x: p[0], z: p[1] }));
  const geo = new THREE.BufferGeometry().setFromPoints(toVecs(pts, opts.y ?? 0.05));
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0xffaa33 });
  return new THREE.LineLoop(geo, mat);
}

/** A group of polylines, one per motion primitive's local sweep. */
export function createMotionPrimitiveHelper(
  lib: MotionPrimitiveLike,
  opts: { y?: number; color?: number } = {},
): THREE.Group {
  const group = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0x88ff88 });
  for (const p of lib.primitives) {
    const geo = new THREE.BufferGeometry().setFromPoints(toVecs(p.sweep, opts.y ?? 0.05));
    group.add(new THREE.Line(geo, mat));
  }
  return group;
}
