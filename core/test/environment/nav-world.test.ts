import { describe, it, expect } from 'vitest';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import type { Pt } from '../../src/internal/geom';

function rect(id: number, x0: number, z0: number, x1: number, z1: number, y = 0): NavPolygon {
  return { id, y, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

describe('InMemoryNavWorld', () => {
  const floor = rect(1, 0, 0, 20, 20);
  const obstacle: Pt[] = [
    [8, 8],
    [12, 8],
    [12, 12],
    [8, 12],
  ];

  it('polygonAt: inside walkable, off-mesh, and inside obstacle', () => {
    const w = new InMemoryNavWorld([floor], [obstacle]);
    expect(w.polygonAt(5, 5)?.id).toBe(1);
    expect(w.polygonAt(-1, 5)).toBeNull();
    expect(w.polygonAt(10, 10)).toBeNull(); // inside obstacle
  });

  it('heightAt returns the polygon floor height', () => {
    const w = new InMemoryNavWorld([rect(1, 0, 0, 10, 10, 3.5)]);
    const p = w.polygonAt(5, 5)!;
    expect(w.heightAt(p, 5, 5)).toBe(3.5);
  });

  it('footprintClear: open ok, obstacle overlap blocked, off-mesh blocked', () => {
    const w = new InMemoryNavWorld([floor], [obstacle]);
    const open: Pt[] = [[2, 2], [3, 2], [3, 3], [2, 3]];
    expect(w.footprintClear(open)).toBe(true);
    const onObstacle: Pt[] = [[9, 9], [11, 9], [11, 11], [9, 11]];
    expect(w.footprintClear(onObstacle)).toBe(false);
    const offMesh: Pt[] = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    expect(w.footprintClear(offMesh)).toBe(false);
  });

  it('segmentClear: open ok, crossing obstacle blocked, off-mesh blocked', () => {
    const w = new InMemoryNavWorld([floor], [obstacle]);
    expect(w.segmentClear(1, 1, 5, 1)).toBe(true);
    expect(w.segmentClear(2, 10, 18, 10)).toBe(false); // passes through obstacle
    expect(w.segmentClear(1, 1, 25, 1)).toBe(false); // exits mesh
  });

  it('off-mesh links and revision bumps', () => {
    const w = new InMemoryNavWorld([rect(1, 0, 0, 10, 20), rect(2, 10, 0, 20, 20)]);
    const a = w.polygonAt(2, 2)!;
    const b = w.polygonAt(15, 15)!;
    expect(a.id).toBe(1);
    expect(b.id).toBe(2);
    expect(w.revision).toBe(0);
    expect(w.offMeshFrom(a)).toEqual([]);
    w.addOffMeshLink({
      from: a,
      to: b,
      start: [2, 0, 2],
      end: [15, 0, 15],
      kind: 'jump',
      cost: 5,
    });
    expect(w.revision).toBe(1);
    expect(w.offMeshFrom(a).length).toBe(1);
    expect(w.offMeshFrom(b)).toEqual([]);
    w.setObstacles([obstacle]);
    expect(w.revision).toBe(2);
  });
});
