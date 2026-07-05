// Regression: planning from a start state that ALREADY satisfies the goal
// must return a trivial "already satisfied" plan — not a maneuver.
//
// Scenario guards fire on edges, so before `scenarioStart` learned to advance
// the automaton on the zero-length start->start edge, a car parked exactly on
// the goal pose got a 3-point plan: reverse 0.75 m at -1.5 m/s and come back
// (cost 1.05). With an unconditional replan cadence this was the engine of
// the post-arrival shuffle: every replan commanded a settled car to un-park.

import { describe, expect, it } from 'vitest';
import { planVehicleScenarioCar } from '../../src/planner/plan-vehicle-scenario';
import { InMemoryNavWorld } from '../../src/environment';
import { characterizeVehicle } from '../../src/primitives';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent';
import type { CarKinematicState } from '../../src/agent/types';
import { reach, at, stopped, stayInside } from '../../src/scenario';

const agent = defaultVehicleAgent({ maxSpeed: 2, maxReverseSpeed: 1.5 });
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [
    [0, 1.5],
    [0, -1.5],
    [1 / agent.minTurnRadius, 1.0],
    [-1 / agent.minTurnRadius, 1.0],
  ],
  duration: 0.5,
  substeps: 4,
  startSpeeds: [-1.5, 0, 1.5],
});
const world = new InMemoryNavWorld([
  { id: 1, y: 0, ring: [[-40, -40], [40, -40], [40, 40], [-40, 40]] },
]);
const lot: [number, number][] = [[-40, -40], [40, -40], [40, 40], [-40, 40]];
const goalPose: CarKinematicState = { x: 5, z: 6, heading: Math.PI / 2, speed: 0, t: 0 };

function planFrom(start: CarKinematicState) {
  return planVehicleScenarioCar({
    start,
    goal: reach(
      at({ x: goalPose.x, z: goalPose.z, heading: goalPose.heading }, { radius: 0.35, dheading: 0.1 }),
      stopped(),
    ),
    invariants: [stayInside(lot)],
    world,
    agent,
    lib,
    envOptions: {
      posCell: 0.3,
      headingBuckets: 36,
      goalRadius: 0.35,
      goalHeadingTol: 0.1,
      analyticExpansion: { everyN: 3, step: 0.15 },
    },
    deadlineMs: 2000,
    maxExpansions: 20000,
  });
}

describe('start-state acceptance (already-satisfied goals)', () => {
  it('a start exactly on the goal pose, stopped, yields a trivial plan — no shuffle', () => {
    const res = planFrom({ ...goalPose });
    expect(res.found).toBe(true);
    // Trivial: the plan is the start state alone; zero cost; no motion.
    expect(res.path.length).toBe(1);
    expect(res.cost).toBe(0);
  });

  it('a start inside the tolerance region, stopped, is also already satisfied', () => {
    const res = planFrom({ x: 5.1, z: 6.1, heading: Math.PI / 2 - 0.05, speed: 0, t: 0 });
    expect(res.found).toBe(true);
    expect(res.path.length).toBe(1);
    expect(res.cost).toBe(0);
  });

  it('a start on the pose but still MOVING is NOT satisfied (stop condition)', () => {
    const res = planFrom({ ...goalPose, speed: -1.5 });
    expect(res.found).toBe(true);
    // Must actually plan something (come to rest / re-enter at rest).
    expect(res.path.length).toBeGreaterThan(1);
  });
});
