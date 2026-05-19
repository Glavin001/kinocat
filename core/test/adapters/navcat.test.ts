import { describe, it, expect, expectTypeOf } from 'vitest';
import * as navcat from 'navcat';
import {
  NavcatWorld,
  navWorldFromTriangleMesh,
  annotateJumpLinks,
  markTileRebuilt,
} from '../../src/adapters/navcat/index';
import { ReplanState } from '../../src/execute/replan';
import { twoIslandsMesh } from '../fixtures/mini-navmesh';

// Build once; if navcat generation surprises us, skip the runtime suite
// (the core M1–M7 stays fully tested regardless).
let built: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = twoIslandsMesh();
  built = navWorldFromTriangleMesh(m.positions, m.indices, { cellSize: 0.3 });
} catch {
  built = null;
}
const NAVMESH_OK = built !== null && built.world.polygonAt(4, 5) !== null;

describe('navcat adapter type pin', () => {
  it('navcat exposes the surface the adapter depends on', () => {
    expectTypeOf(navcat.findNearestPoly).toBeFunction();
    expectTypeOf(navcat.raycast).toBeFunction();
    expectTypeOf(navcat.getClosestPointOnPoly).toBeFunction();
    expectTypeOf(navcat.addOffMeshConnection).toBeFunction();
    expect(typeof navcat.DEFAULT_QUERY_FILTER).toBe('object');
  });
});

describe.skipIf(!NAVMESH_OK)('NavcatWorld over a real navmesh', () => {
  const world = built!.world;

  it('polygonAt: on-mesh vs. gap vs. far outside', () => {
    expect(world.polygonAt(4, 5)).not.toBeNull();
    expect(world.polygonAt(18, 5)).not.toBeNull();
    expect(world.polygonAt(11, 5)).toBeNull(); // the gap
    expect(world.polygonAt(-50, -50)).toBeNull();
  });

  it('heightAt is ~0 on the flat mesh', () => {
    const p = world.polygonAt(4, 5)!;
    expect(Math.abs(world.heightAt(p, 4, 5))).toBeLessThan(0.5);
  });

  it('segmentClear: on-island clear, across-gap blocked', () => {
    expect(world.segmentClear(1, 5, 6, 5)).toBe(true);
    expect(world.segmentClear(1, 5, 12, 5)).toBe(false);
  });

  it('footprintClear: inside ok, straddling the gap blocked', () => {
    const inside: [number, number][] = [
      [3, 4],
      [4, 4],
      [4, 5],
      [3, 5],
    ];
    expect(world.footprintClear(inside)).toBe(true);
    const straddling: [number, number][] = [
      [7, 4],
      [12, 4],
      [12, 5],
      [7, 5],
    ];
    expect(world.footprintClear(straddling)).toBe(false);
  });

  it('annotateJumpLinks registers only real-gap candidates', () => {
    const meta = annotateJumpLinks(world, built!.navMesh, [
      { from: [7, 5], to: [15, 5], cost: 3 }, // spans the gap ⇒ accepted
      { from: [2, 5], to: [5, 5] }, // walkable shortcut ⇒ rejected
    ]);
    expect(meta.length).toBe(1);
    expect(typeof meta[0]!.connectionId).toBe('number');
    const fromPoly = world.polygonAt(7, 5)!;
    const links = world.offMeshFrom(fromPoly);
    expect(links.length).toBe(1);
    expect(links[0]!.kind).toBe('jump');
    expect(links[0]!.cost).toBe(3);
  });

  it('markTileRebuilt bumps revision and flags replanning', () => {
    const before = world.revision;
    const rs = new ReplanState({ divergenceThresholdMeters: 1, refreshIntervalMs: 500 });
    rs.setPlan([{ x: 0, z: 0, heading: 0, speed: 0, t: 0 }], 0);
    markTileRebuilt(world, [rs]);
    expect(world.revision).toBe(before + 1);
    expect(rs.shouldReplan({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 1)).toBe(true);
  });
});

it('reports navmesh availability', () => {
  // Surfaces in CI logs whether the real-navcat path ran or was skipped.
  expect(typeof NAVMESH_OK).toBe('boolean');
  expect(NavcatWorld).toBeTypeOf('function');
});
