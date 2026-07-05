// The car domain run through the kinocat/testing conformance battery — the
// packaged answer to "how do we know this Environment works?". Also serves
// as the kit's reference usage example for docs/adding-a-domain.md.

import { describe, it, expect } from 'vitest';
import { runConformance, type DomainHarness, type FidelityHooks } from '../../src/testing';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { kinematicForwardSim } from '../../src/agent/vehicle';
import type { CarKinematicState } from '../../src/agent/types';
import { SWEEP_AGENT, buildSweepLib, rect } from '../fixtures/vehicle-sweep';

const lib = buildSweepLib();

// The sampler holds speed at 0 — exactly the library's only start-speed
// bucket — so re-simulating a drive edge from the actual state must match
// the rigid-transformed cached primitive to float noise. This pins the
// transform machinery itself (a frame/rotation bug shows up immediately).
const sim = kinematicForwardSim(SWEEP_AGENT);
const PRIM_DURATION = 0.5; // buildSweepLib's characterization parameters
const SUBSTEPS = 6;
const fidelity: FidelityHooks<CarKinematicState> = {
  tolerance: 1e-9,
  angularFields: ['heading'],
  resimulate: (parent, edge) => {
    if (edge.kind !== 'drive' && edge.kind !== 'drive-reverse') return null;
    const d = edge.data as { primId: number };
    const controls = lib.primitives[d.primId]!.controls;
    const dt = PRIM_DURATION / SUBSTEPS;
    let s = parent;
    for (let i = 0; i < SUBSTEPS; i++) s = sim(s, controls, dt);
    return s;
  },
};

function openWorld() {
  return new InMemoryNavWorld([rect(1, 0, 0, 40, 30)]);
}

/** Two rooms joined by a 6 m-wide bridge — wide enough for the sweep car's
 *  3 m turn radius, but obstacle-constrained (the direct line is blocked). */
function doorwayWorld() {
  return new InMemoryNavWorld([
    rect(1, 0, 0, 19, 30),
    rect(2, 21, 0, 40, 30),
    rect(3, 17, 12, 23, 18),
  ]);
}

function harness(world: () => InMemoryNavWorld): DomainHarness<CarKinematicState> {
  return {
    makeEnv: () => new VehicleEnvironment(world(), SWEEP_AGENT, lib),
    sampleState: (rand) => ({
      x: 2 + rand() * 36,
      z: 2 + rand() * 26,
      heading: (rand() - 0.5) * 2 * Math.PI,
      speed: 0,
      t: rand() * 50,
    }),
    fidelity,
    scenarios: [
      {
        name: 'open-field',
        start: { x: 5, z: 5, heading: 0, speed: 0, t: 0 },
        goal: { x: 35, z: 25, heading: 0, speed: 0, t: 0 },
        maxExpansions: 200_000,
      },
      {
        name: 'across-the-doorway',
        start: { x: 5, z: 5, heading: 0, speed: 0, t: 0 },
        goal: { x: 35, z: 25, heading: 0, speed: 0, t: 0 },
        maxExpansions: 300_000,
      },
    ],
  };
}

describe('VehicleEnvironment conformance', () => {
  it('open world passes the full battery', () => {
    const report = runConformance(harness(openWorld));
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
  });

  it('doorway world passes the full battery', () => {
    const report = runConformance(harness(doorwayWorld));
    expect(report.failures).toEqual([]);
  });
});
