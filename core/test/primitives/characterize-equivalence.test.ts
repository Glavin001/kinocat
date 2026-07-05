// Behavior-pinning for the characterize<S> refactor: vehicle libraries and
// aircraft successor sets must be numerically IDENTICAL to the fixture
// captured before the shared rollout harness was extracted. Exact equality,
// no epsilon — the refactor moves code, it must not move floats. (The
// fixture also guards future accidental changes to primitive rollout.)

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import {
  AircraftEnvironment,
  type AircraftEnvOptions,
} from '../../src/environment/aircraft-environment';
import { InMemoryAirspace } from '../../src/environment/airspace-world';
import { defaultAircraftAgent } from '../../src/agent/aircraft';
import type { AircraftState } from '../../src/agent/types';

const fixture = JSON.parse(
  readFileSync(join(__dirname, '../fixtures/characterize-equivalence.json'), 'utf-8'),
);

describe('characterizeVehicle equivalence', () => {
  it('produces the exact pre-refactor library', () => {
    const agent = defaultVehicleAgent({ minTurnRadius: 3, maxSpeed: 8, maxReverseSpeed: 4 });
    const k = 1 / agent.minTurnRadius;
    const lib = characterizeVehicle({
      forwardSim: kinematicForwardSim(agent),
      controlSets: [
        [0, 6], [k, 6], [-k, 6], [k / 2, 6], [-k / 2, 6],
        [0, -4], [k, -4], [-k, -4],
      ],
      duration: 0.7,
      substeps: 5,
      startSpeeds: [0, 4, 8],
    });
    expect(JSON.parse(lib.toJSON())).toEqual(fixture.vehicle);
  });
});

describe('aircraft primitive-cache equivalence (via succ)', () => {
  // Infinity attitude rates reproduce the legacy quasi-static snap model
  // bit-exactly, so this fixture keeps pinning the rigid-transform
  // machinery itself; the rate-limited behavior has its own tests.
  const air = defaultAircraftAgent({
    minTurnRadius: 12, minSpeed: 6, maxSpeed: 18,
    maxClimbAngle: Math.PI / 6, maxBank: Math.PI / 2,
    maxRollRate: Infinity, maxPitchRate: Infinity,
    halfLength: 2, halfSpan: 1.5, halfHeight: 0.3,
  });
  function aircraftSucc(opts: AircraftEnvOptions, from: AircraftState, level?: number) {
    const env = new AircraftEnvironment(
      new InMemoryAirspace({ floor: 0, ceiling: 100 }),
      air,
      opts,
    );
    const node = env.createNode(from, null, null);
    const goal = env.createNode({ ...from, x: from.x + 500 }, null, null);
    return env.succ(node, goal, level).map((n) => ({ state: n.state, edge: n.edge, g: n.g }));
  }
  const s0: AircraftState = { x: 0, y: 20, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 };
  const s1: AircraftState = { x: 13.7, y: 31.2, z: -8.4, heading: 2.3, pitch: 0.1, roll: -0.4, speed: 18, t: 4.5 };
  const LC: AircraftEnvOptions = {
    levelControls: [
      { rollFractions: [0] },
      { rollFractions: [0] },
      { rollFractions: [-1, 0, 1] },
    ],
  };

  it('default config, finest level, canonical start', () => {
    expect(aircraftSucc({}, s0, 2)).toEqual(fixture.aircraft.defaultL2_s0);
  });
  it('default config, coarse level, translated/rotated start', () => {
    expect(aircraftSucc({}, s1, 0)).toEqual(fixture.aircraft.defaultL0_s1);
  });
  it('levelControls, coarse level (sparse set)', () => {
    expect(aircraftSucc(LC, s0, 0)).toEqual(fixture.aircraft.lcL0_s0);
  });
  it('levelControls, finest level (dense roll set)', () => {
    expect(aircraftSucc(LC, s1, 2)).toEqual(fixture.aircraft.lcL2_s1);
  });
});
