// The aircraft domain — the genuinely-3D Environment — through the
// conformance battery, in both the default config and the per-level
// primitive-set (levelControls) config used for knife-edge scenarios.

import { describe, it, expect } from 'vitest';
import { runConformance, type DomainHarness } from '../../src/testing';
import {
  AircraftEnvironment,
  type AircraftEnvOptions,
} from '../../src/environment/aircraft-environment';
import { InMemoryAirspace } from '../../src/environment/airspace-world';
import {
  aircraftForwardSim,
  defaultAircraftAgent,
} from '../../src/agent/aircraft';
import type { AircraftState } from '../../src/agent/types';

const agent = defaultAircraftAgent({
  minTurnRadius: 12,
  minSpeed: 6,
  maxSpeed: 18,
  maxClimbAngle: Math.PI / 6,
  maxBank: Math.PI / 2,
  halfLength: 2,
  halfSpan: 1.5,
  halfHeight: 0.3,
});

/** A wall across the middle with a gap to route through. */
function airspace() {
  return new InMemoryAirspace({
    floor: 2,
    ceiling: 60,
    boxes: [
      { min: [45, 0, -100], max: [55, 60, -12] },
      { min: [45, 0, 12], max: [55, 60, 100] },
    ],
  });
}

// Env defaults (primDuration 1, substeps 6) — the fidelity hook must
// re-integrate with the exact same partition succ() uses.
const PRIM_DURATION = 1;
const SUBSTEPS = 6;
const sim = aircraftForwardSim(agent);

function harness(opts: AircraftEnvOptions = {}): DomainHarness<AircraftState> {
  return {
    makeEnv: () => new AircraftEnvironment(airspace(), agent, opts),
    sampleState: (rand) => ({
      x: -20 + rand() * 60, // keep samples out of the wall band
      y: 10 + rand() * 40,
      z: -40 + rand() * 80,
      heading: (rand() - 0.5) * 2 * Math.PI,
      // Attitude is rate-limited state — sample the whole envelope so the
      // fidelity check exercises mid-ramp starts, not just level flight.
      pitch: (rand() - 0.5) * 1.8 * agent.maxClimbAngle,
      roll: (rand() - 0.5) * 1.8 * agent.maxBank,
      speed: agent.maxSpeed,
      t: rand() * 50,
    }),
    // succ() rolls the sim live from the actual state, so re-simulation is
    // EXACT — machine-eps tolerance, no bucket slack.
    fidelity: {
      tolerance: 1e-9,
      angularFields: ['heading', 'pitch', 'roll'],
      resimulate: (parent, edge) => {
        if (edge.kind !== 'fly') return null;
        const d = edge.data as { k: number; climb: number; roll: number; v: number };
        const dt = PRIM_DURATION / SUBSTEPS;
        let s = parent;
        for (let i = 0; i < SUBSTEPS; i++) s = sim(s, [d.k, d.climb, d.roll, d.v], dt);
        return s;
      },
    },
    scenarios: [
      {
        name: 'through-the-gap',
        start: { x: 0, y: 20, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 },
        goal: { x: 100, y: 25, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 },
        maxExpansions: 200_000,
      },
      {
        name: 'climb-and-turn',
        start: { x: 0, y: 10, z: -30, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 },
        goal: { x: 20, y: 40, z: 30, heading: Math.PI / 2, pitch: 0, roll: 0, speed: 18, t: 0 },
        maxExpansions: 200_000,
      },
    ],
  };
}

describe('AircraftEnvironment conformance', () => {
  it('default config passes the full battery', () => {
    const report = runConformance(harness());
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
  });

  it('per-level primitive sets (levelControls) pass the full battery', () => {
    const report = runConformance(
      harness({
        levelControls: [
          { rollFractions: [0] },
          { rollFractions: [0] },
          { rollFractions: [-1, 0, 1] },
        ],
      }),
    );
    expect(report.failures).toEqual([]);
  });
});
