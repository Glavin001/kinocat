// Shared scaffolding for the parameterized "boundary-sweep" tests. The best
// scenario tests aren't single fixed setups — they're families swept across a
// parameter so the test pins down WHERE behaviour changes (the decision
// threshold) and forbids a dangerous ambiguous zone where the planner flips
// back and forth. This module holds the standard agent + primitive library
// (previously duplicated across vehicle.test.ts / time-aware.test.ts), the
// world/footprint helpers, and the boundary-finding primitives.
//
// Not a `.test.ts` file, so Vitest ignores it (cf. `_util.ts`, `_sim-harness.ts`).

import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { placeFootprint } from '../../src/internal/geom';
import type { VehicleAgent, CarKinematicState } from '../../src/agent/types';

/** Axis-aligned rectangle as a NavPolygon (matches vehicle.test.ts). */
export function rect(
  id: number,
  x0: number,
  z0: number,
  x1: number,
  z1: number,
): NavPolygon {
  return {
    id,
    y: 0,
    ring: [
      [x0, z0],
      [x1, z0],
      [x1, z1],
      [x0, z1],
    ],
  };
}

/** Standard sweep agent: a ~2.4×1.2 m car, 3 m min turn radius. The footprint
 *  half-extents are 1.2 (length) × 0.6 (width), so its circumscribed radius is
 *  hypot(1.2, 0.6) ≈ 1.34 m — use SWEEP_AGENT_RADIUS for moving-obstacle pads. */
export const SWEEP_AGENT: VehicleAgent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  maxReverseSpeed: 4,
  footprint: [
    [1.2, 0.6],
    [-1.2, 0.6],
    [-1.2, -0.6],
    [1.2, -0.6],
  ],
});

/** Circumscribed-radius pad for the standard agent (matches time-aware.test.ts). */
export const SWEEP_AGENT_RADIUS = 1.4;

/** Forward + reverse + half-curvature control set, as used by the existing
 *  environment tests. `startSpeeds: [0]` keeps the library small/fast. */
export function buildSweepLib(agent: VehicleAgent = SWEEP_AGENT) {
  const k = 1 / agent.minTurnRadius;
  return characterizeVehicle({
    forwardSim: kinematicForwardSim(agent),
    controlSets: [
      [0, 6],
      [k, 6],
      [-k, 6],
      [k / 2, 6],
      [-k / 2, 6],
      [0, -4],
      [k, -4],
      [-k, -4],
    ],
    duration: 0.5,
    substeps: 6,
    startSpeeds: [0],
  });
}

/** True iff every state's placed footprint is collision-free in `world`. */
export function footprintsClear(
  world: InMemoryNavWorld,
  agent: VehicleAgent,
  path: ReadonlyArray<CarKinematicState>,
): boolean {
  return path.every((s) =>
    world.footprintClear(placeFootprint(agent.footprint, s.x, s.z, s.heading)),
  );
}

/** Run `fn` across a parameter range, pairing each input with its result. */
export function sweep<P, R>(
  values: ReadonlyArray<P>,
  fn: (value: P) => R,
): Array<{ value: P; result: R }> {
  return values.map((value) => ({ value, result: fn(value) }));
}

/** Inclusive linear ramp of `n` samples from `lo` to `hi` (n ≥ 2). */
export function linspace(lo: number, hi: number, n: number): number[] {
  if (n < 2) return [lo];
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = lo + ((hi - lo) * i) / (n - 1);
  return out;
}

/** Number of adjacent flips in a boolean sweep. 0 ⇒ never changed; 1 ⇒ a clean
 *  single threshold; ≥2 ⇒ the dangerous ambiguous/chattering zone where the
 *  planner's decision oscillates across the boundary. */
export function countTransitions(flags: ReadonlyArray<boolean>): number {
  let n = 0;
  for (let i = 1; i < flags.length; i++) if (flags[i] !== flags[i - 1]) n++;
  return n;
}

/** Index of the first `true` (the decision threshold for a monotone-rising
 *  feasibility sweep), or -1 if none. */
export function firstTrueIndex(flags: ReadonlyArray<boolean>): number {
  return flags.findIndex((f) => f);
}
