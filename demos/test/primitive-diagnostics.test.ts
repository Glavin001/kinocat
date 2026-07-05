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
  buildLearnedRaceLibrary,
  buildLearnedRaceLibraryV2,
  RACE_START_SPEEDS,
} from '../app/lib/race-primitives-scenarios';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  DEFAULT_LEARNED_PARAMS,
} from 'kinocat/agent';
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

  it('comparing kinematic vs legacy-v1 (same control vocab) surfaces real mismatch', () => {
    // v2 library uses native wheeled controls [steer, drive, brake]; the
    // kinematic library uses [curvature, targetSpeed]. Different control
    // vocabularies → per-control mismatch is undefined (intentional). For
    // the pairing sanity check use the LEGACY 5-param learned library
    // which still uses the (curvature, targetSpeed) adapter and so shares
    // controls with kinematic.
    const kin = buildKinematicLibrary();
    const legacy = buildLearnedRaceLibrary(DEFAULT_LEARNED_PARAMS);
    const speed = RACE_START_SPEEDS[2]!; // 20 m/s — high enough to show divergence
    const kinAtSpeed = kin.lookup(speed);
    const legacyAtSpeed = legacy.lookup(speed);
    const d = diagnoseLibrary(kinAtSpeed, legacyAtSpeed);
    expect(d.pairedMismatches).toBeDefined();
    expect(d.pairedMismatches!.length).toBe(kinAtSpeed.length);
    expect(d.meanMismatch!).toBeGreaterThan(0.05);
    expect(d.largestMismatch).toBeDefined();
  });

  it('v2 race library has a non-degenerate fan at every speed bucket', () => {
    // The KEY visible signal that the fix landed: v2 should produce
    // distinguishable endpoints at all four buckets (0/10/20/28), not
    // collapse to ~0 hull at the speed extremes.
    const v2 = buildLearnedRaceLibraryV2(
      buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG),
    );
    for (const speed of RACE_START_SPEEDS) {
      const prims = v2.lookup(speed);
      expect(prims.length).toBeGreaterThanOrEqual(6);
      const d = diagnoseLibrary(prims);
      // Forward endpoint span across x — primitives MUST cover meaningful
      // forward distance (chassis accelerated / decelerated / coasted).
      const xSpan = d.forwardEndpointBBox.xMax - d.forwardEndpointBBox.xMin;
      expect(xSpan, `xSpan at speed ${speed}`).toBeGreaterThan(0.3);
      // Reachable area > 0.5 m² — confirms the action space isn't 1-D
      // (the old adapter-based v2 had hull = 0.2 m² at v=0, 0.4 m² at
      // v=28; both effectively unusable for planning).
      expect(d.hullAreaM2, `hull at speed ${speed}`).toBeGreaterThan(0.5);
    }
  });

  it('comparing kinematic vs kinematic gives zero mismatch (sanity)', () => {
    const kin = buildKinematicLibrary();
    const kinAt = kin.lookup(20);
    const d = diagnoseLibrary(kinAt, kinAt);
    expect(d.meanMismatch!).toBeCloseTo(0, 6);
    expect(d.maxMismatch!).toBeCloseTo(0, 6);
  });
});
