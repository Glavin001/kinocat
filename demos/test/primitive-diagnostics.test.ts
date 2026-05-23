import { describe, expect, it } from 'vitest';
import {
  endpointAngularGaps,
  maxEndpointAngularGap,
  reachableHullArea,
  pairwiseEndpointMismatch,
  diagnoseLibrary,
} from '../app/lib/primitive-diagnostics';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  RACE_START_SPEEDS,
} from '../app/lib/race-primitives-scenarios';
import { buildParametricOnlyModel, DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import type { MotionPrimitive } from 'kinocat/primitives';

function makeStubPrimitive(opts: {
  id: number; dx: number; dz: number; reverse?: boolean; controls?: number[];
}): MotionPrimitive {
  return {
    id: opts.id,
    startSpeed: 0,
    controls: opts.controls ?? [0, 0],
    duration: 0.5,
    end: { dx: opts.dx, dz: opts.dz, dHeading: 0, speed: 0 },
    sweep: [{ x: 0, z: 0, heading: 0 }, { x: opts.dx, z: opts.dz, heading: 0 }],
    reverse: opts.reverse ?? false,
  };
}

describe('endpointAngularGaps', () => {
  it('returns sorted gaps; sum across the wrap-around equals 360°', () => {
    // 4 endpoints at the cardinal directions
    const prims = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0 }),     // 0°
      makeStubPrimitive({ id: 1, dx: 0, dz: 1 }),     // 90°
      makeStubPrimitive({ id: 2, dx: -1, dz: 0 }),    // 180°
      makeStubPrimitive({ id: 3, dx: 0, dz: -1 }),    // -90° / 270°
    ];
    const gaps = endpointAngularGaps(prims);
    expect(gaps).toHaveLength(4);
    for (const g of gaps) expect(g).toBeCloseTo(90, 5);
    const sum = gaps.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(360, 4);
  });

  it('ignores reverse primitives', () => {
    const prims = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0 }),
      makeStubPrimitive({ id: 1, dx: -1, dz: 0, reverse: true }),
    ];
    const gaps = endpointAngularGaps(prims);
    // Only one forward endpoint → no pairwise gap → only the wrap gap (360°)
    expect(gaps).toHaveLength(1);
    expect(gaps[0]).toBeCloseTo(360, 4);
  });

  it('max gap surfaces the "missing wedge" in a one-sided fan', () => {
    // 4 endpoints all in the right half-plane (curvatures from 0 to ~30°)
    const prims = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0 }),                         // 0°
      makeStubPrimitive({ id: 1, dx: 1, dz: 0.1 }),                       // ~6°
      makeStubPrimitive({ id: 2, dx: 1, dz: 0.3 }),                       // ~17°
      makeStubPrimitive({ id: 3, dx: 1, dz: 0.5 }),                       // ~27°
    ];
    expect(maxEndpointAngularGap(prims)).toBeGreaterThan(300);
  });
});

describe('reachableHullArea', () => {
  it('is positive for a non-degenerate forward fan', () => {
    const prims = [
      makeStubPrimitive({ id: 0, dx: 1, dz: -0.5 }),
      makeStubPrimitive({ id: 1, dx: 1, dz: 0.5 }),
      makeStubPrimitive({ id: 2, dx: 2, dz: 0 }),
    ];
    const area = reachableHullArea(prims);
    expect(area).toBeGreaterThan(0);
    // The triangle has base 1 (z range) and height 1 (x range) → area = 0.5
    expect(area).toBeCloseTo(0.5, 5);
  });

  it('is zero for fewer than 3 forward primitives', () => {
    expect(reachableHullArea([])).toBe(0);
    expect(reachableHullArea([makeStubPrimitive({ id: 0, dx: 1, dz: 0 })])).toBe(0);
    expect(reachableHullArea([
      makeStubPrimitive({ id: 0, dx: 1, dz: 0 }),
      makeStubPrimitive({ id: 1, dx: 2, dz: 0 }),
    ])).toBe(0);
  });
});

describe('pairwiseEndpointMismatch', () => {
  it('returns zero distance when both libraries match by controls (identical)', () => {
    const prims = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0, controls: [0, 10] }),
      makeStubPrimitive({ id: 1, dx: 2, dz: 0.5, controls: [0.1, 8] }),
    ];
    const mismatches = pairwiseEndpointMismatch(prims, prims);
    expect(mismatches).toHaveLength(2);
    for (const m of mismatches) expect(m.distance).toBe(0);
  });

  it('flags only the primitives whose endpoints differ', () => {
    const a = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0, controls: [0, 10] }),
      makeStubPrimitive({ id: 1, dx: 2, dz: 0, controls: [0.1, 8] }),
    ];
    const b = [
      makeStubPrimitive({ id: 0, dx: 1, dz: 0, controls: [0, 10] }),       // same
      makeStubPrimitive({ id: 1, dx: 1.5, dz: 0.5, controls: [0.1, 8] }),  // moved
    ];
    const mismatches = pairwiseEndpointMismatch(a, b);
    expect(mismatches).toHaveLength(2);
    expect(mismatches[0]!.distance).toBe(0);
    expect(mismatches[1]!.distance).toBeCloseTo(Math.hypot(0.5, 0.5), 5);
  });
});

describe('diagnoseLibrary — real race libraries', () => {
  it('kinematic race library has the expected shape at speed 0', () => {
    const lib = buildKinematicLibrary();
    const at0 = lib.lookup(0);
    expect(at0.length).toBeGreaterThan(0);
    const d = diagnoseLibrary(at0);
    expect(d.count).toBe(at0.length);
    expect(d.forwardCount + d.reverseCount).toBe(d.count);
    expect(d.hullAreaM2).toBeGreaterThanOrEqual(0);
  });

  it('comparing kinematic vs v2 at non-zero speed surfaces real mismatch', () => {
    const kin = buildKinematicLibrary();
    const v2 = buildLearnedRaceLibraryV2(
      buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG),
    );
    const speed = RACE_START_SPEEDS[2]!; // 20 m/s — high enough to show divergence
    const kinAtSpeed = kin.lookup(speed);
    const v2AtSpeed = v2.lookup(speed);
    const d = diagnoseLibrary(kinAtSpeed, v2AtSpeed);
    expect(d.pairedMismatches).toBeDefined();
    expect(d.pairedMismatches!.length).toBe(kinAtSpeed.length);
    // At high speed, the two models MUST disagree by more than millimetres,
    // otherwise the v2 model isn't doing anything useful.
    expect(d.meanMismatch!).toBeGreaterThan(0.05);
    expect(d.largestMismatch).toBeDefined();
  });

  it('comparing kinematic vs kinematic gives zero mismatch (sanity)', () => {
    const kin = buildKinematicLibrary();
    const kinAt = kin.lookup(20);
    const d = diagnoseLibrary(kinAt, kinAt);
    expect(d.meanMismatch!).toBeCloseTo(0, 6);
    expect(d.maxMismatch!).toBeCloseTo(0, 6);
  });
});
