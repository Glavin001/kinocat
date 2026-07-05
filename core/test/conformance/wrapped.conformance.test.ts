// Composing wrappers must conform too: whatever the base guarantees, the
// wrapped environment must still guarantee. TimeAware(Vehicle) with a real
// moving obstacle is the load-bearing case — it exercises the time-extended
// dominance keys, obstacle pruning, and (post-WS2a) level forwarding.

import { describe, it, expect } from 'vitest';
import { runConformance, type DomainHarness } from '../../src/testing';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { linearObstacle } from '../../src/predict/factories';
import type { CarKinematicState } from '../../src/agent/types';
import {
  SWEEP_AGENT,
  SWEEP_AGENT_RADIUS,
  buildSweepLib,
  rect,
} from '../fixtures/vehicle-sweep';

const lib = buildSweepLib();

const harness: DomainHarness<CarKinematicState> = {
  makeEnv: () =>
    new TimeAwareEnvironment(
      new VehicleEnvironment(new InMemoryNavWorld([rect(1, 0, 0, 40, 30)]), SWEEP_AGENT, lib),
      {
        // Crosses the field perpendicular to the start→goal line.
        obstacles: [linearObstacle(20, 0, 0, 2, 1.5)],
        agentRadius: SWEEP_AGENT_RADIUS,
      },
    ),
  sampleState: (rand) => ({
    x: 2 + rand() * 36,
    z: 2 + rand() * 26,
    heading: (rand() - 0.5) * 2 * Math.PI,
    speed: 0,
    t: rand() * 50,
  }),
  scenarios: [
    {
      name: 'cross-with-moving-obstacle',
      start: { x: 5, z: 15, heading: 0, speed: 0, t: 0 },
      goal: { x: 35, z: 15, heading: 0, speed: 0, t: 0 },
      maxExpansions: 300_000,
    },
  ],
};

describe('TimeAware(Vehicle) conformance', () => {
  it('the wrapped environment passes the full battery', () => {
    const report = runConformance(harness);
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
  });
});
