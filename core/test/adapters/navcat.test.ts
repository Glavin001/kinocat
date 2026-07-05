import { describe, it, expect, expectTypeOf } from 'vitest';
import * as navcat from 'navcat';
import {
  NavcatWorld,
  navWorldFromTriangleMesh,
  annotateJumpLinks,
  markTileRebuilt,
} from '../../src/adapters/navcat/index';
import { ReplanState } from '../../src/execute/replan';
import { twoIslandsMesh, singlePlaneMesh } from '../fixtures/mini-navmesh';

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

// Opt 1: CompactHeightfield clearance field. Built on a 30×20 flat plane so
// true edge distances are known — the scale-pin asserts the reported value
// is a strict LOWER bound (early-accept safety) and never exceeds truth.
let clr: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = singlePlaneMesh(); // x∈[0,30] z∈[0,20], y=0
  clr = navWorldFromTriangleMesh(
    m.positions,
    m.indices,
    { cellSize: 0.3 },
    { clearanceField: true },
  );
} catch {
  clr = null;
}
const CLR_OK = clr !== null && clr.world.polygonAt(15, 10) !== null;

describe.skipIf(!CLR_OK)('CompactHeightfield clearance field (Opt 1)', () => {
  it('surfaces the compactHeightfield on the result', () => {
    expect(clr!.compactHeightfield).toBeTruthy();
    expect(clr!.compactHeightfield.distances.length).toBeGreaterThan(0);
  });

  it('is a strict lower bound on the true edge distance (scale-pin)', () => {
    const w = clr!.world;
    // (15,10): nearest boundary is z-edges at distance 10.
    const cMid = w.clearanceAt!(15, 10);
    expect(cMid).not.toBeNull();
    expect(cMid!).toBeGreaterThan(0);
    expect(cMid!).toBeLessThanOrEqual(10 + 1e-9); // never over-estimates
    // Nearer an edge ⇒ smaller; still a lower bound on true distance (~2).
    const cEdge = w.clearanceAt!(2, 10);
    expect(cEdge).not.toBeNull();
    expect(cEdge!).toBeLessThanOrEqual(2 + 1e-9);
    expect(cMid!).toBeGreaterThan(cEdge!);
  });

  it('returns null off-field', () => {
    expect(clr!.world.clearanceAt!(-50, -50)).toBeNull();
  });

  it('clearanceAt is null when no field was attached', () => {
    const w = built ? built.world : null;
    if (w) expect(w.clearanceAt!(4, 5)).toBeNull();
  });
});

describe.skipIf(!CLR_OK)('grid-Dijkstra goal lower bound (Opt 2)', () => {
  it('is a non-negative lower bound, ~0 at the goal, null off-grid', () => {
    const w = clr!.world;
    const lb = w.buildGoalLowerBound!(25, 10);
    expect(lb).not.toBeNull();
    const atGoal = lb!(25, 10);
    expect(atGoal).not.toBeNull();
    expect(atGoal!).toBeGreaterThanOrEqual(0);
    expect(atGoal!).toBeLessThan(2);
    const far = lb!(3, 10); // true straight distance ~22
    expect(far).not.toBeNull();
    expect(far!).toBeGreaterThan(0);
    expect(far!).toBeLessThanOrEqual(22 + 1e-6); // lower bound, never over
    expect(far!).toBeGreaterThan(atGoal!); // farther ⇒ larger
    expect(lb!(-50, -50)).toBeNull(); // off-grid
  });

  it('buildGoalLowerBound returns null for an off-mesh goal', () => {
    expect(clr!.world.buildGoalLowerBound!(-100, -100)).toBeNull();
  });
});

// Phase-1 integration: swapNavMesh + region-scoped markTileRebuilt. Fresh
// NavcatWorlds are constructed from the fixtures' already-generated meshes
// (no regeneration) so mutation stays isolated per test.
describe.skipIf(!NAVMESH_OK || !CLR_OK)('swapNavMesh + region-scoped invalidation', () => {
  const TRIGGER = { divergenceThresholdMeters: 1, refreshIntervalMs: 500 };

  it('live queries and the revision-keyed goal field see the new geometry', () => {
    const w = new NavcatWorld(built!.navMesh);
    w.attachClearanceField(built!.compactHeightfield);
    // Pre-swap (two islands): the gap is off-mesh and island B is
    // unreachable from island A.
    expect(w.polygonAt(11, 5)).toBeNull();
    const lb0 = w.buildGoalLowerBound!(18, 5);
    expect(lb0).not.toBeNull();
    expect(lb0!(4, 5)).toBeNull(); // cross-gap: unreachable
    // Swap in the single 30×20 plane. Same goal coords — with the old
    // revision-blind memo this would keep serving the stale field.
    w.swapNavMesh(clr!.navMesh, clr!.compactHeightfield);
    expect(w.polygonAt(11, 5)).not.toBeNull();
    const lb1 = w.buildGoalLowerBound!(18, 5);
    expect(lb1).not.toBeNull();
    const d = lb1!(4, 5);
    expect(d).not.toBeNull();
    expect(d!).toBeGreaterThan(0);
    expect(d!).toBeLessThanOrEqual(14 + 1e-6); // straight-line truth = 14
  });

  it('omitting the CHF clears both oracles instead of keeping stale ones', () => {
    const w = new NavcatWorld(built!.navMesh);
    w.attachClearanceField(built!.compactHeightfield);
    expect(w.clearanceAt(4, 5)).not.toBeNull();
    w.swapNavMesh(clr!.navMesh); // no chf
    expect(w.clearanceAt(15, 10)).toBeNull();
    expect(w.buildGoalLowerBound!(18, 5)).toBeNull();
  });

  it('re-resolves mirrored off-mesh links and drops off-mesh endpoints', () => {
    const w = new NavcatWorld(clr!.navMesh); // single plane
    const mk = (sx: number, sz: number, ex: number, ez: number) => ({
      from: w.polygonAt(sx, sz)!,
      to: w.polygonAt(ex, ez)!,
      start: [sx, 0, sz] as const,
      end: [ex, 0, ez] as const,
      kind: 'jump' as const,
      cost: 2,
    });
    w.addOffLink(mk(4, 5, 18, 5)); // both endpoints survive on the islands
    w.addOffLink(mk(11, 5, 18, 5)); // (11,5) lands in the islands' gap
    w.swapNavMesh(built!.navMesh);
    const from = w.polygonAt(4, 5)!;
    const links = w.offMeshFrom(from);
    expect(links.length).toBe(1);
    expect(links[0]!.from.id).toBe(from.id); // ref rewritten to the new mesh
    expect(links[0]!.to.id).toBe(w.polygonAt(18, 5)!.id);
  });

  it('markTileRebuilt(region) marks only crossing agents and returns them', () => {
    const w = new NavcatWorld(built!.navMesh);
    const crossing = new ReplanState(TRIGGER);
    crossing.setPlan(
      [
        { x: 0, z: 5, heading: 0, speed: 4, t: 0 },
        { x: 10, z: 5, heading: 0, speed: 4, t: 2.5 },
      ],
      0,
    );
    const far = new ReplanState(TRIGGER);
    far.setPlan(
      [
        { x: 0, z: 0.5, heading: 0, speed: 4, t: 0 },
        { x: 10, z: 0.5, heading: 0, speed: 4, t: 2.5 },
      ],
      0,
    );
    const before = w.revision;
    const marked = markTileRebuilt(w, { x0: 4, z0: 4, x1: 6, z1: 6 }, [
      { replan: crossing },
      { replan: far },
    ]);
    expect(w.revision).toBe(before + 1);
    expect(marked).toEqual([crossing]);
    const at = { x: 1, z: 5, heading: 0, speed: 4, t: 0.25 };
    expect(crossing.shouldReplan(at, 1)).toBe(true);
    expect(far.shouldReplan({ ...at, z: 0.5 }, 1)).toBe(false);
  });
});

it('reports navmesh availability', () => {
  // Surfaces in CI logs whether the real-navcat path ran or was skipped.
  expect(typeof NAVMESH_OK).toBe('boolean');
  expect(NavcatWorld).toBeTypeOf('function');
});
