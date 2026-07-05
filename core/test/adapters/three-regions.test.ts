import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { createRegionHelper, REGION_COLORS } from '../../src/adapters/three/regions';
import {
  at,
  near,
  inside,
  gate,
  corridor,
  cone,
  within,
  deg,
} from '../../src/scenario/index';
import type { RegionAgent } from '../../src/scenario/index';

function countVerts(obj: THREE.Object3D): number {
  let n = 0;
  obj.traverse((c) => {
    const g = (c as THREE.Line).geometry as THREE.BufferGeometry | undefined;
    const pos = g?.getAttribute?.('position');
    if (pos) n += pos.count;
  });
  return n;
}

describe('createRegionHelper', () => {
  it('near -> a ring (closed loop with many vertices)', () => {
    const g = createRegionHelper(near({ x: 1, z: 2 }, 3), { color: REGION_COLORS.objective });
    expect(g).toBeInstanceOf(THREE.Group);
    expect(countVerts(g)).toBeGreaterThan(8);
  });

  it('at -> a 4-corner box plus a heading tick', () => {
    const g = createRegionHelper(at({ x: 0, z: 0, heading: 0 }, { dx: 1, dz: 0.5, dheading: deg(5) }));
    // 4 box corners + 2 tick points.
    expect(countVerts(g)).toBe(6);
  });

  it('inside -> a polygon line loop with one vertex per ring vertex', () => {
    const poly: [number, number][] = [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4],
    ];
    const g = createRegionHelper(inside(poly));
    expect(countVerts(g)).toBe(4);
  });

  it('gate -> a segment plus a normal arrow', () => {
    const g = createRegionHelper(gate({ x: 0, z: -2 }, { x: 0, z: 2 }));
    expect(countVerts(g)).toBe(4); // 2 segment + 2 arrow
  });

  it('corridor -> centerline + two offset edges', () => {
    const g = createRegionHelper(
      corridor([
        { x: 0, z: 0 },
        { x: 10, z: 0 },
      ], 4),
    );
    expect(countVerts(g)).toBeGreaterThanOrEqual(6);
  });

  it('cone -> a wedge outline', () => {
    const moving: RegionAgent = { id: 'g', predict: (t) => ({ x: 0, z: 0, heading: 0, speed: 0, t }) };
    const g = createRegionHelper(cone(moving, deg(30), 10), { color: REGION_COLORS.avoid });
    expect(countVerts(g)).toBe(3);
  });

  it('within (dynamic) -> a ring at the predicted pose', () => {
    const moving: RegionAgent = { id: 'g', predict: (t) => ({ x: 5, z: 0, heading: 0, speed: 0, t }) };
    const g = createRegionHelper(within(moving, 2));
    expect(countVerts(g)).toBeGreaterThan(8);
  });
});
