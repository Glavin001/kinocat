// Static avoidance, as parameterized boundary sweeps rather than single fixed
// setups. The headline is the gap-width sweep: a single "avoid this obstacle"
// test only tells you pass/fail, but sweeping the gap from clearly-impassable
// to clearly-passable tells you WHERE the decision threshold sits and — via
// countTransitions — that there is no ambiguous zone where the planner flips
// its mind. The rest are fail-safe cases (dead end, obstacle-on-goal, enclosed
// start) that a naive planner mishandles catastrophically, plus a plan-layer
// determinism check (the "same seed twice" invariant).

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import type { CarKinematicState } from '../../src/agent/types';
import {
  SWEEP_AGENT as agent,
  buildSweepLib,
  rect,
  footprintsClear,
  sweep,
  linspace,
  countTransitions,
  firstTrueIndex,
} from '../fixtures/vehicle-sweep';

const lib = buildSweepLib();

// Bounded corridor so an "impassable" gap genuinely has no go-around: the
// search exhausts its (small) budget and reports found:false, instead of
// detouring around the outside of the world forever.
const CORRIDOR = rect(1, 0, -8, 30, 8);
const WALL_X0 = 14;
const WALL_X1 = 17;
const START: CarKinematicState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
const GOAL: CarKinematicState = { x: 27, z: 0, heading: 0, speed: 0, t: 0 };

/** A wall fully spanning the corridor in x∈[14,17] with a centred gap of width
 *  `w` at z∈[-w/2, w/2] (w ≤ 0 ⇒ a solid wall, no gap). */
function gapWorld(w: number): InMemoryNavWorld {
  const half = w / 2;
  const obstacles: [number, number][][] = [];
  // top slab z∈[half, 8]
  if (8 - half > 1e-6) {
    obstacles.push([
      [WALL_X0, half],
      [WALL_X1, half],
      [WALL_X1, 8],
      [WALL_X0, 8],
    ]);
  }
  // bottom slab z∈[-8, -half]
  if (8 - half > 1e-6) {
    obstacles.push([
      [WALL_X0, -8],
      [WALL_X1, -8],
      [WALL_X1, -half],
      [WALL_X0, -half],
    ]);
  }
  return new InMemoryNavWorld([CORRIDOR], obstacles);
}

function mkEnv(world: InMemoryNavWorld) {
  return new VehicleEnvironment(world, agent, lib, {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
    analyticExpansion: {},
  });
}

function solve(world: InMemoryNavWorld, maxExpansions = 60000) {
  return plan({ start: START, goal: GOAL, environment: mkEnv(world), options: { maxExpansions } }, Infinity);
}

describe('static avoidance — gap-width sweep (where the threshold sits)', () => {
  // Vehicle full width is 1.2 m, so anything well below that cannot fit and
  // anything well above clearly can; the threshold must land in between.
  const widths = linspace(0.6, 3.2, 14);

  const runs = sweep(widths, (w) => solve(gapWorld(w)));
  const found = runs.map((r) => r.result.found);

  it('feasibility is monotone with a single clean threshold (no chattering)', () => {
    const ctx = `\nwidths=${widths.map((w) => w.toFixed(2)).join(',')}\nfound =${found.join(',')}`;
    // Exactly one false→true flip: a clear decision boundary, no ambiguous
    // zone where widening the gap toggles feasibility back and forth.
    expect(countTransitions(found), ctx).toBe(1);
    // Narrowest gap is impassable; widest is passable (the sweep straddles it).
    expect(found[0], ctx).toBe(false);
    expect(found[found.length - 1], ctx).toBe(true);
  });

  it('the threshold sits in a physically sane band around the vehicle width', () => {
    const idx = firstTrueIndex(found);
    expect(idx).toBeGreaterThan(0);
    const wStar = widths[idx]!;
    const ctx = `\nthreshold width = ${wStar.toFixed(2)} m (vehicle width 1.2 m)`;
    // Can't pass a gap narrower than the body; shouldn't need more than ~2.6 m.
    expect(wStar, ctx).toBeGreaterThanOrEqual(1.2);
    expect(wStar, ctx).toBeLessThanOrEqual(2.6);
  });

  it('every passable solution is collision-free and actually threads the gap', () => {
    const mid = (WALL_X0 + WALL_X1) / 2; // wall centre-line in x
    for (const { value: w, result } of runs) {
      if (!result.found) continue;
      const world = gapWorld(w);
      expect(footprintsClear(world, agent, result.path), `w=${w}`).toBe(true);
      // Interpolate the lateral offset where the path crosses the wall's
      // centre-line — samples are ~3 m apart so a state rarely lands exactly
      // inside the 3 m band, but the crossing z must lie within the gap.
      let crossedInsideGap = false;
      for (let i = 1; i < result.path.length; i++) {
        const a = result.path[i - 1]!;
        const b = result.path[i]!;
        if ((a.x - mid) * (b.x - mid) > 0) continue; // segment doesn't straddle mid
        const u = Math.abs(b.x - a.x) < 1e-9 ? 0 : (mid - a.x) / (b.x - a.x);
        const zc = a.z + (b.z - a.z) * u;
        if (Math.abs(zc) <= w / 2 + 1e-6) crossedInsideGap = true;
      }
      expect(crossedInsideGap, `w=${w}`).toBe(true);
    }
  });
});

describe('static avoidance — fail-safe degenerate cases', () => {
  it('dead end: a goal boxed in with no opening reports no plan (no hang/throw)', () => {
    // Box the goal region x∈[21,30] z∈[-3,3] shut (right side is the world edge).
    const world = new InMemoryNavWorld(
      [CORRIDOR],
      [
        [
          [20, 3],
          [30, 3],
          [30, 4],
          [20, 4],
        ],
        [
          [20, -4],
          [30, -4],
          [30, -3],
          [20, -3],
        ],
        [
          [20, -3],
          [21, -3],
          [21, 3],
          [20, 3],
        ],
      ],
    );
    let r!: ReturnType<typeof solve>;
    expect(() => {
      r = plan(
        { start: START, goal: GOAL, environment: mkEnv(world), options: { maxExpansions: 12000 } },
        Infinity,
      );
    }).not.toThrow();
    expect(r.found).toBe(false);
    // Fails safe within budget rather than running away to Infinity.
    expect(r.stats.expansions).toBeLessThanOrEqual(12000);
  });

  it('obstacle sitting on the goal: rejected up front (goal footprint blocked)', () => {
    const world = new InMemoryNavWorld(
      [CORRIDOR],
      [
        [
          [23, -2],
          [27, -2],
          [27, 2],
          [23, 2],
        ],
      ],
    );
    const r = solve(world, 30000);
    expect(r.found).toBe(false);
    // checkValidity rejects the blocked goal pose immediately — no search.
    expect(r.stats.expansions).toBe(0);
    expect(r.path).toEqual([]);
  });

  it('enclosed start: no escape reports no plan, fails safe', () => {
    const world = new InMemoryNavWorld(
      [CORRIDOR],
      [
        [
          [1, 3],
          [6, 3],
          [6, 4],
          [1, 4],
        ],
        [
          [1, -4],
          [6, -4],
          [6, -3],
          [1, -3],
        ],
        [
          [5, -3],
          [6, -3],
          [6, 3],
          [5, 3],
        ],
      ],
    );
    const r = plan(
      { start: START, goal: GOAL, environment: mkEnv(world), options: { maxExpansions: 30000 } },
      Infinity,
    );
    expect(r.found).toBe(false);
  });
});

describe('static avoidance — plan-layer determinism (same problem twice)', () => {
  it('re-planning an identical problem yields a byte-identical trajectory', () => {
    const world = gapWorld(2.0); // a passable but non-trivial detour
    const a = solve(world);
    const b = solve(world);
    expect(a.found).toBe(true);
    expect(b.found).toBe(true);
    expect(a.cost).toBe(b.cost);
    expect(a.stats.expansions).toBe(b.stats.expansions);
    expect(a.path.length).toBe(b.path.length);
    for (let i = 0; i < a.path.length; i++) {
      expect(b.path[i]!.x).toBe(a.path[i]!.x);
      expect(b.path[i]!.z).toBe(a.path[i]!.z);
      expect(b.path[i]!.heading).toBe(a.path[i]!.heading);
    }
  });
});
