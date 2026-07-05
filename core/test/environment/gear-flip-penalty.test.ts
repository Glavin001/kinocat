// Regression: the direction-change penalty must price ACTUAL gear flips —
// and only gear flips — including through the ScenarioEnvironment bridge.
//
// The historic bug had two halves:
//   1. `parentReverse = node.edge && (...)` evaluated to `null` (not
//      `undefined`) for edge-less nodes, so the `!== undefined` guard never
//      disabled the penalty and every successor of a root paid it;
//   2. the scenario/multi-goal bridges rebuilt the inner node with a NULL
//      edge, so in bridge mode EVERY node looked like a root.
// Net effect: a constant per-edge tax — flipping gear was relatively free and
// the planner emitted multi-cusp shuffle parking plans.

import { describe, expect, it } from 'vitest';
import { VehicleEnvironment, InMemoryNavWorld } from '../../src/environment';
import { ScenarioEnvironment, scenarioStart } from '../../src/environment/scenario-environment';
import { characterizeVehicle } from '../../src/primitives';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent';
import type { CarKinematicState } from '../../src/agent/types';
import { compile, reach, near } from '../../src/scenario';
import type { Node } from '../../src/environment/types';

const PENALTY = 5; // large and unmistakable in edge costs
const REV_MULT = 1.05;
const DURATION = 0.5;

const agent = defaultVehicleAgent({
  maxSpeed: 2,
  maxReverseSpeed: 1.5,
  reverseCostMultiplier: REV_MULT,
  directionChangePenalty: PENALTY,
});
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [
    [0, 1.5], // forward
    [0, -1.5], // reverse
  ],
  duration: DURATION,
  substeps: 4,
  startSpeeds: [-1.5, 0, 1.5],
});
const world = new InMemoryNavWorld([
  { id: 1, y: 0, ring: [[-40, -40], [40, -40], [40, 40], [-40, 40]] },
]);
const envOptions = {
  posCell: 0.5,
  headingBuckets: 16,
  goalRadius: 1,
  goalHeadingTol: Infinity,
  analyticExpansion: false as const,
};

const start: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
const goalState: CarKinematicState = { x: 30, z: 0, heading: 0, speed: 0, t: 0 };

function edgeCost(n: Node<unknown>): number {
  return n.edge!.cost;
}
function isReverseEdge(n: Node<unknown>): boolean {
  return (n.edge!.data as { reverse?: boolean } | undefined)?.reverse === true;
}

describe('direction-change penalty pricing', () => {
  it('bare env: no penalty from a root; penalty only on actual flips', () => {
    const env = new VehicleEnvironment(world, agent, lib, envOptions);
    const root = env.createNode(start, null, null);
    root.g = 0;
    const goal = env.createNode(goalState, null, null);

    const first = env.succ(root, goal);
    expect(first.length).toBeGreaterThan(0);
    for (const n of first) {
      const expected = DURATION * (isReverseEdge(n) ? REV_MULT : 1);
      expect(edgeCost(n)).toBeCloseTo(expected, 6);
    }

    // Expand a REVERSE child: reverse-again is unpenalized, forward pays.
    const rev = first.find(isReverseEdge)!;
    const second = env.succ(rev, goal);
    const revAgain = second.filter(isReverseEdge);
    const fwdAfterRev = second.filter((n) => !isReverseEdge(n));
    expect(revAgain.length).toBeGreaterThan(0);
    expect(fwdAfterRev.length).toBeGreaterThan(0);
    for (const n of revAgain) expect(edgeCost(n)).toBeCloseTo(DURATION * REV_MULT, 6);
    for (const n of fwdAfterRev) expect(edgeCost(n)).toBeCloseTo(DURATION + PENALTY, 6);
  });

  it('scenario bridge: gear history survives the wrapper (the shuffle-plan bug)', () => {
    const base = new VehicleEnvironment(world, agent, lib, envOptions);
    const automaton = compile(reach(near({ x: 30, z: 0 }, 1)));
    const senv = new ScenarioEnvironment<CarKinematicState>(base, {
      automaton,
      invariants: [],
      costTerms: [],
    });
    const s0 = scenarioStart(start, automaton);
    const root = senv.createNode(s0, null, null);
    root.g = 0;
    const goal = senv.createNode(s0, null, null);

    const first = senv.succ(root, goal);
    expect(first.length).toBeGreaterThan(0);
    // Root: NO successor pays the penalty (this failed before the fix —
    // every bridge edge cost included it).
    for (const n of first) {
      const expected = DURATION * (isReverseEdge(n) ? REV_MULT : 1);
      expect(edgeCost(n)).toBeCloseTo(expected, 6);
    }

    // Reverse-after-reverse through the bridge: unpenalized (was 0.675
    // instead of 0.525 before the fix). Forward-after-reverse: penalized.
    const rev = first.find(isReverseEdge)!;
    const second = senv.succ(rev, goal);
    const revAgain = second.filter(isReverseEdge);
    const fwdAfterRev = second.filter((n) => !isReverseEdge(n));
    expect(revAgain.length).toBeGreaterThan(0);
    expect(fwdAfterRev.length).toBeGreaterThan(0);
    for (const n of revAgain) expect(edgeCost(n)).toBeCloseTo(DURATION * REV_MULT, 6);
    for (const n of fwdAfterRev) expect(edgeCost(n)).toBeCloseTo(DURATION + PENALTY, 6);
  });
});
