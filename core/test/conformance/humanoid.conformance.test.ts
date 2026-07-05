// The humanoid domain through the conformance battery. Note the fixture
// worlds carry no off-mesh links: a link whose cost undercuts straight-line
// travel time would (correctly) fail heuristic consistency against the
// Euclidean heuristic — see docs/adding-a-domain.md.

import { describe, it, expect } from 'vitest';
import { runConformance, type DomainHarness } from '../../src/testing';
import { HumanoidEnvironment } from '../../src/environment/humanoid-environment';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { defaultHumanoidAgent } from '../../src/agent';
import type { HumanoidState } from '../../src/agent/types';
import { rect } from '../fixtures/vehicle-sweep';

const human = defaultHumanoidAgent({ radius: 0.3, maxSpeed: 4 });

const harness: DomainHarness<HumanoidState> = {
  makeEnv: () =>
    new HumanoidEnvironment(
      new InMemoryNavWorld([
        rect(1, 0, 0, 20, 1.5),
        rect(2, 18.5, 0, 20, 20),
      ]),
      human,
      { goalRadius: 0.7 },
    ),
  sampleState: (rand) =>
    rand() < 0.5
      ? { x: 0.5 + rand() * 19, z: 0.75, heading: (rand() - 0.5) * 2 * Math.PI, t: rand() * 50 }
      : { x: 19.25, z: 0.5 + rand() * 19, heading: (rand() - 0.5) * 2 * Math.PI, t: rand() * 50 },
  scenarios: [
    {
      name: 'l-corridor',
      start: { x: 2, z: 0.75, heading: 0, t: 0 },
      goal: { x: 19.25, z: 18, heading: Math.PI / 2, t: 0 },
      maxExpansions: 300_000,
    },
    {
      name: 'short-hop',
      start: { x: 2, z: 0.75, heading: 0, t: 0 },
      goal: { x: 10, z: 0.75, heading: 0, t: 0 },
      maxExpansions: 100_000,
    },
  ],
};

describe('HumanoidEnvironment conformance', () => {
  it('passes the full battery', () => {
    const report = runConformance(harness);
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
  });
});
