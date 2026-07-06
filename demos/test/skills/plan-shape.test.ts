// Skill tests — planner tier (plan-only, no execution, no Rapier).
//
// These plan a maneuver on a hand-built minimal course and assert the PLAN's
// shape: is it a dynamically-feasible spline, or does it stop / kink sharply?
// Any failure here is a PLANNER problem, independent of the executor.
//
// See docs/racing-skills-test-plan.md — skills K5 (slalom spline), K6
// (reverse-cost honesty), K8 (feasibility invariant).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  buildLearnedRaceLibraryV3,
  planRaceMultiGoal,
  RACE_AGENT,
  RACE_PLANNER_GATE_RADIUS,
} from '../../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';

const repoRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models', f), 'utf-8'));

const libs: Record<string, () => MotionPrimitiveLibrary> = {
  kin: () => buildKinematicLibrary(),
  v2: () => buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json'))),
  v3: () => buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json'))),
};

const BOUNDS = { x0: -60, z0: -40, x1: 60, z1: 40 };
const POLY = [{ id: 0, y: 0, ring: [[BOUNDS.x0, BOUNDS.z0], [BOUNDS.x1, BOUNDS.z0], [BOUNDS.x1, BOUNDS.z1], [BOUNDS.x0, BOUNDS.z1]] as [number, number][] }];

function planGates(
  lib: MotionPrimitiveLibrary,
  spawn: CarKinematicState,
  gates: CarKinematicState[],
  analyticDriveThrough = false,
) {
  return planRaceMultiGoal({
    state: spawn,
    gates,
    lib,
    polygons: POLY,
    obstacles: [],
    gateRadius: RACE_PLANNER_GATE_RADIUS,
    deadlineMs: 6000,
    maxExpansions: 400_000,
    analyticDriveThrough,
  });
}

/** Feasibility check: between consecutive plan samples, can the speed change
 *  have physically happened? |dv| <= aMax * dt (aMax generous = 15 m/s^2, above
 *  the derived 13.9 to avoid false positives from primitive granularity). */
function maxInfeasibleDv(path: CarKinematicState[], aMax = 15): number {
  let worst = 0;
  for (let i = 1; i < path.length; i++) {
    const dt = Math.max(1e-3, path[i]!.t - path[i - 1]!.t);
    const dv = Math.abs(path[i]!.speed - path[i - 1]!.speed);
    const excess = dv - aMax * dt;
    if (excess > worst) worst = excess;
  }
  return worst; // >0 means an impossible speed jump exists
}

/** Sharpest heading change per metre of travel along the plan (proxy for
 *  curvature / kink). */
function maxTurnPerMetre(path: CarKinematicState[]): number {
  let worst = 0;
  for (let i = 1; i < path.length; i++) {
    const ds = Math.max(0.1, Math.hypot(path[i]!.x - path[i - 1]!.x, path[i]!.z - path[i - 1]!.z));
    let dh = Math.abs(path[i]!.heading - path[i - 1]!.heading);
    if (dh > Math.PI) dh = 2 * Math.PI - dh;
    if (dh / ds > worst) worst = dh / ds;
  }
  return worst; // rad/m; 1/minRadius(4.5) = 0.22 is the geometric ceiling
}

const minInteriorSpeed = (path: CarKinematicState[]) =>
  Math.min(...path.slice(1, -1).map((p) => Math.abs(p.speed)));

describe('skill K8 — honest-model plans are dynamically feasible', () => {
  const spawn: CarKinematicState = { x: -40, z: 0, heading: 0, speed: 12, t: 0 };
  const gates: CarKinematicState[] = [
    { x: -20, z: 6, heading: 0, speed: 5, t: 0 },
    { x: 0, z: -6, heading: 0, speed: 5, t: 0 },
    { x: 20, z: 6, heading: 0, speed: 5, t: 0 },
  ];

  it('v2 slalom plan has no impossible speed jump', () => {
    const res = planGates(libs.v2!(), spawn, gates);
    expect(res.found).toBe(true);
    expect(maxInfeasibleDv(res.path)).toBeLessThan(2);
  });

  // TDD marker: v3's library currently bakes a primitive with a physically
  // impossible speed transition (measured dv ~9.7 m/s beyond the ~14 m/s^2
  // envelope over one step). When the v3 library is fixed this `it.fails`
  // starts failing → flip to `it`.
  it.fails('TARGET: v3 slalom plan has no impossible speed jump (v3 library bug)', () => {
    const res = planGates(libs.v3!(), spawn, gates);
    expect(res.found).toBe(true);
    expect(maxInfeasibleDv(res.path)).toBeLessThan(2);
  });
});

describe('skill K5 — slalom is planned as a spline, not stop-pivots', () => {
  // 3 alternating gates, gentle offset (feasible as an S at speed). A good
  // planner carries speed through; a stop-pivot plan drops interior speed to ~0
  // and kinks at the geometric limit.
  const spawn: CarKinematicState = { x: -40, z: 0, heading: 0, speed: 14, t: 0 };
  const gates: CarKinematicState[] = [
    { x: -18, z: 5, heading: 0, speed: 5, t: 0 },
    { x: 0, z: -5, heading: 0, speed: 5, t: 0 },
    { x: 18, z: 5, heading: 0, speed: 5, t: 0 },
  ];

  // Diagnostic (always green): record what each model currently plans.
  for (const model of ['kin', 'v2', 'v3'] as const) {
    it(`${model}: diagnostic — interior speed + max kink`, () => {
      const res = planGates(libs[model]!(), spawn, gates);
      expect(res.found).toBe(true);
      const minV = minInteriorSpeed(res.path);
      const kink = maxTurnPerMetre(res.path);
      // eslint-disable-next-line no-console
      console.log(`  K5 ${model}: minInteriorV=${minV.toFixed(1)} maxKink=${kink.toFixed(3)} rad/m len=${res.path.length}`);
    });
  }

  // TDD marker: a gentle slalom must be takeable WITHOUT stopping. Today the
  // planner prefers a zero-speed Reeds-Shepp analytic shot to each gate
  // (mispriced as length/maxSpeed while it actually stops), so interior speed
  // collapses to 0. When the analytic-shot pricing / medium-radius primitives
  // are fixed this `it.fails` starts failing → flip to `it`.
  it.fails('TARGET: v3 carries speed through the slalom (currently stops)', () => {
    const res = planGates(libs.v3!(), spawn, gates);
    expect(minInteriorSpeed(res.path)).toBeGreaterThan(3);
  });

  // The FIX (correctness branch): dispersion-designed control set + the
  // analytic drive-through repricing. The planner now splines the slalom
  // carrying speed instead of stopping at each gate. Proves the fix works.
  for (const model of ['v2', 'v3'] as const) {
    it(`${model}: FIXED (generated + drive-through) carries speed through the slalom`, () => {
      const lib =
        model === 'v3'
          ? buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')), { generatedControls: true })
          : buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json')), { generatedControls: true });
      const res = planGates(lib, spawn, gates, /* analyticDriveThrough */ true);
      expect(res.found).toBe(true);
      const minV = minInteriorSpeed(res.path);
      // eslint-disable-next-line no-console
      console.log(`  K5 ${model} FIXED: minInteriorV=${minV.toFixed(1)} len=${res.path.length}`);
      expect(minV).toBeGreaterThan(3);
    });
  }
});
