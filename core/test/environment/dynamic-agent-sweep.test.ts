// Dynamic agents, as parameterized sweeps — the highest-value group, because
// with moving traffic timing is everything and the boundary between "dodge it"
// and "can't" is where safety failures hide. Everything here is deterministic:
// obstacles are TimeAwareEnvironment + linearObstacle predictors (no Rapier, no
// RNG), so a failure is a real planner regression, not sampling noise.
//
// Two assertion shapes recur:
//   • a hard invariant — when a plan is found, NO path state ever overlaps a
//     predicted obstacle at that state's arrival time, and timestamps rise;
//   • a boundary sweep — vary one parameter (obstacle radius / corridor width)
//     and assert feasibility flips exactly once (countTransitions === 1).

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { linearObstacle } from '../../src/predict/factories';
import type { MovingObstacle } from '../../src/predict/types';
import type { CarKinematicState } from '../../src/agent/types';
import {
  SWEEP_AGENT as agent,
  SWEEP_AGENT_RADIUS as AGENT_R,
  buildSweepLib,
  rect,
  sweep,
  linspace,
  countTransitions,
} from '../fixtures/vehicle-sweep';

const lib = buildSweepLib();

function mkEnv(
  world: InMemoryNavWorld,
  obstacles: MovingObstacle[],
) {
  const base = new VehicleEnvironment(world, agent, lib, {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
  });
  return new TimeAwareEnvironment(base, { obstacles, agentRadius: AGENT_R });
}

/** Hard safety invariant: no path state overlaps any predicted obstacle at its
 *  arrival time, and timestamps strictly increase. Returns the min observed
 *  clearance margin (gap beyond the required radius) for diagnostics. */
function assertCollisionFree(
  path: ReadonlyArray<CarKinematicState>,
  obstacles: ReadonlyArray<MovingObstacle>,
): number {
  let minMargin = Infinity;
  for (const s of path) {
    for (const obs of obstacles) {
      const p = obs.predict(s.t);
      if (!p) continue;
      const d = Math.hypot(s.x - p.x, s.z - p.z);
      const margin = d - (obs.radius + AGENT_R);
      expect(margin).toBeGreaterThan(-1e-6);
      if (margin < minMargin) minMargin = margin;
    }
  }
  for (let i = 1; i < path.length; i++) {
    expect(path[i]!.t).toBeGreaterThan(path[i - 1]!.t - 1e-9);
  }
  return minMargin;
}

// Wide corridor so lateral evasion / passing is geometrically possible; the
// only thing that can make a scenario infeasible is the dynamic timing.
const CORRIDOR = rect(1, 0, -14, 32, 14);
const START: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
const GOAL: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };
const BUDGET = { maxExpansions: 500000 };

describe('dynamic agents — perpendicular crossing, closing-speed sweep', () => {
  // An obstacle starts below the corridor at the midpoint and crosses upward.
  // Sweeping its speed sweeps WHEN it occupies the centre — i.e. the timing
  // conflict with the ego vehicle. The plan must never collide, whatever the
  // timing.
  const speeds = linspace(2, 10, 9);
  const runs = sweep(speeds, (vz) => {
    const obs = linearObstacle(15, -12, 0, vz, 2.5, 0, 60);
    const r = plan({ start: START, goal: GOAL, environment: mkEnv(new InMemoryNavWorld([CORRIDOR]), [obs]), options: BUDGET }, Infinity);
    return { r, obs };
  });

  it('never collides at any crossing speed (yields or goes, but stays clear)', () => {
    for (const { value: vz, result } of runs) {
      if (!result.r.found) continue;
      const minMargin = assertCollisionFree(result.r.path, [result.obs]);
      expect(minMargin, `vz=${vz}`).toBeGreaterThan(-1e-6);
    }
  });

  it('feasibility transitions at most once across the speed sweep', () => {
    const found = runs.map((x) => x.result.r.found);
    // Most/all of these are dodgeable in a wide corridor, but if some closing
    // speed is genuinely un-passable the change must be a single clean flip,
    // never a chattering in/out as speed rises.
    expect(countTransitions(found), `found=${found.join(',')}`).toBeLessThanOrEqual(1);
  });
});

describe('dynamic agents — crossing-obstacle radius sweep (the boundary)', () => {
  // Fix the timing (the obstacle sits on the centre-line for the whole run) and
  // grow its radius from a small dodgeable disc to one that plugs the corridor.
  // This is the cleanest single-threshold boundary in the dynamic set.
  const radii = linspace(1.5, 13, 12);
  const corridorHalf = 14;

  const found = radii.map((radius) => {
    // Stationary blocker on the centre-line (vx=vz=0) — purely geometric, so
    // the threshold is analytic: passable while a gap ≥ vehicle width remains
    // on one side (radius + AGENT_R < corridorHalf − halfWidth).
    const obs = linearObstacle(15, 0, 0, 0, radius, 0, 1e6);
    const r = plan({ start: START, goal: GOAL, environment: mkEnv(new InMemoryNavWorld([CORRIDOR]), [obs]), options: BUDGET }, Infinity);
    if (r.found) assertCollisionFree(r.path, [obs]);
    return r.found;
  });

  it('a single clean threshold from passable to blocked (no chattering)', () => {
    const ctx = `\nradii =${radii.map((r) => r.toFixed(1)).join(',')}\nfound =${found.join(',')}`;
    expect(countTransitions(found), ctx).toBe(1);
    expect(found[0], ctx).toBe(true); // small disc: dodgeable
    expect(found[found.length - 1], ctx).toBe(false); // fat disc: corridor plugged
  });

  it('the threshold matches the geometric "a vehicle-width gap remains" bound', () => {
    const idx = found.indexOf(false);
    const rStar = radii[idx]!;
    // The obstacle (centre z=0) leaves a gap of (corridorHalf − radius − AGENT_R)
    // on each side; the half-width of the body is 0.6 m. So it should stay
    // passable until roughly radius ≈ corridorHalf − AGENT_R − 0.6 ≈ 12.0.
    const geomBound = corridorHalf - AGENT_R - 0.6;
    expect(Math.abs(rStar - geomBound), `rStar=${rStar.toFixed(1)} bound≈${geomBound.toFixed(1)}`).toBeLessThanOrEqual(2.5);
  });
});

describe('dynamic agents — lead / cut-in / oncoming invariants', () => {
  it('lead vehicle ahead, same direction: follows/passes without collision (speed sweep)', () => {
    for (const v of linspace(1, 5, 5)) {
      const obs = linearObstacle(10, 0, v, 0, 2.0, 0, 60); // slower, dead ahead
      const r = plan({ start: START, goal: GOAL, environment: mkEnv(new InMemoryNavWorld([CORRIDOR]), [obs]), options: BUDGET }, Infinity);
      expect(r.found, `lead v=${v}`).toBe(true);
      assertCollisionFree(r.path, [obs]);
    }
  });

  it('late cut-in from the side stays clear', () => {
    // Starts off to the +z side, slides into the lane around t≈2.5 s.
    const obs = linearObstacle(16, 9, 0, -3.2, 2.2, 0, 60);
    const r = plan({ start: START, goal: GOAL, environment: mkEnv(new InMemoryNavWorld([CORRIDOR]), [obs]), options: BUDGET }, Infinity);
    expect(r.found).toBe(true);
    assertCollisionFree(r.path, [obs]);
  });

  it('oncoming obstacle in a shared corridor: width sweep finds the pass/blocked threshold', () => {
    // Obstacle drives toward the ego down the centre-line. Narrow the corridor
    // until there is no room to slip past — a clean single threshold.
    const halfWidths = linspace(2.5, 7, 10);
    const found = halfWidths.map((hw) => {
      const world = new InMemoryNavWorld([rect(1, 0, -hw, 32, hw)]);
      const obs = linearObstacle(26, 0, -4, 0, 1.8, 0, 60); // heading −x toward ego
      const r = plan({ start: START, goal: GOAL, environment: mkEnv(world, [obs]), options: BUDGET }, Infinity);
      if (r.found) assertCollisionFree(r.path, [obs]);
      return r.found;
    });
    const ctx = `\nhalfWidths=${halfWidths.map((h) => h.toFixed(1)).join(',')}\nfound =${found.join(',')}`;
    expect(countTransitions(found), ctx).toBeLessThanOrEqual(1);
    expect(found[found.length - 1], ctx).toBe(true); // widest: always passable
  });
});
