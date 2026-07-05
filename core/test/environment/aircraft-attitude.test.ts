// Attitude is state, not a snap-to output. The sim integrates pitch/roll
// toward their setpoints at the airframe's rates, and the emergent planning
// consequences fall out of search with no special-case code: the plane must
// BEGIN rolling before a knife-edge slot, must HOLD the bank through a
// double slot whose gap is too short to unroll and re-roll, and relaxes to
// wings level when the gap affords it (commanded bank costs rollCost/s;
// commanding level is free).

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import {
  AircraftEnvironment,
  type AircraftEnvOptions,
} from '../../src/environment/aircraft-environment';
import { InMemoryAirspace, type AABB } from '../../src/environment/airspace-world';
import {
  defaultAircraftAgent,
  aircraftForwardSim,
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

function level(over: Partial<AircraftState> = {}): AircraftState {
  return {
    x: 0, y: 24, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0,
    ...over,
  };
}

describe('aircraftForwardSim: attitude integrates', () => {
  const sim = aircraftForwardSim(agent);

  it('one short step gets maxRollRate·dt of bank, not the full command', () => {
    const s = sim(level(), [0, 0, agent.maxBank, 18], 0.1);
    expect(s.roll).toBeCloseTo(agent.maxRollRate * 0.1, 9);
    expect(s.roll).toBeLessThan(agent.maxBank);
  });

  it('holding the command converges to the target and stops there', () => {
    let s = level();
    for (let i = 0; i < 10; i++) s = sim(s, [0, 0, agent.maxBank, 18], 0.1);
    expect(s.roll).toBeCloseTo(agent.maxBank, 9);
  });

  it('pitch ramps the same way, and altitude follows the RAMP, not the command', () => {
    const snap = aircraftForwardSim({
      ...agent,
      maxPitchRate: Infinity,
    });
    const ramped = sim(level(), [0, agent.maxClimbAngle, 0, 18], 0.2);
    const snapped = snap(level(), [0, agent.maxClimbAngle, 0, 18], 0.2);
    expect(ramped.pitch).toBeCloseTo(agent.maxPitchRate * 0.2, 9);
    // Less altitude gained while the flight path is still bending up.
    expect(ramped.y).toBeLessThan(snapped.y);
  });

  it('Infinity rates reproduce the legacy quasi-static snap exactly', () => {
    const legacy = aircraftForwardSim({
      ...agent,
      maxRollRate: Infinity,
      maxPitchRate: Infinity,
    });
    const s = legacy(level(), [0.05, 0.3, agent.maxBank, 15], 1);
    expect(s.roll).toBeCloseTo(agent.maxBank, 12);
    expect(s.pitch).toBeCloseTo(0.3, 12);
  });
});

// A 1.2 m-wide vertical slot: 1.5cos θ + 0.3sin θ ≤ 0.6 needs |roll| ≥ ~74°.
function slotWalls(x0: number, x1: number): AABB[] {
  return [
    { min: [x0, 0, -60], max: [x1, 80, -0.6] },
    { min: [x0, 0, 0.6], max: [x1, 80, 60] },
  ];
}

const ENV_OPTS: AircraftEnvOptions = {
  posCell: 4,
  altCell: 4,
  goalRadius: 10,
  rollFractions: [-1, 0, 1],
  primDuration: 0.5,
  substeps: 4,
  levelControls: [
    { rollFractions: [0] },
    { rollFractions: [0] },
    { rollFractions: [-1, 0, 1] },
  ],
};

function planThrough(boxes: AABB[], goalX: number) {
  const air = new InMemoryAirspace({ floor: 0, ceiling: 80, boxes });
  const env = new AircraftEnvironment(air, agent, ENV_OPTS);
  const r = plan(
    {
      start: level(),
      goal: level({ x: goalX }),
      environment: env,
      options: { maxExpansions: 400_000 },
    },
    Infinity,
  );
  expect(r.found).toBe(true);
  return r;
}

describe('emergent knife-edge timing (no special-case code)', () => {
  it('begins the roll BEFORE the slot — setup is a matter of timing', () => {
    const r = planThrough(slotWalls(40, 50), 90);
    // Fully banked inside the slot…
    const inside = r.path.filter((s) => s.x > 41 && s.x < 49);
    expect(inside.length).toBeGreaterThan(0);
    for (const s of inside) expect(Math.abs(s.roll)).toBeGreaterThan(1.25);
    // …and already substantially rolled while still approaching it. With
    // maxRollRate π the 90° ramp takes 0.5 s ≈ 9 m of approach.
    const approaching = r.path.filter((s) => s.x < 40 && Math.abs(s.roll) > 0.6);
    expect(approaching.length).toBeGreaterThan(0);
  });

  it('holds the bank through a double slot whose gap is too short to unroll', () => {
    // Gap 50→54 = 4 m ≈ 0.22 s of flight; a full unroll+re-roll needs 1 s.
    const r = planThrough([...slotWalls(40, 50), ...slotWalls(54, 64)], 104);
    const through = r.path.filter((s) => s.x > 41 && s.x < 63);
    expect(through.length).toBeGreaterThanOrEqual(2);
    for (const s of through) expect(Math.abs(s.roll)).toBeGreaterThan(0.5);
    expect(Math.max(...through.map((s) => Math.abs(s.roll)))).toBeGreaterThan(1.25);
  });

  it('relaxes toward level when the gap affords it (commanded bank costs, level is free)', () => {
    // Gap 50→104 = 54 m ≈ 3 s — room to unroll, cruise level, re-roll.
    const r = planThrough([...slotWalls(40, 50), ...slotWalls(104, 114)], 154);
    const between = r.path.filter((s) => s.x > 57 && s.x < 100);
    expect(between.length).toBeGreaterThan(0);
    expect(Math.min(...between.map((s) => Math.abs(s.roll)))).toBeLessThan(0.35);
    // And it is banked again inside the second slot.
    const slot2 = r.path.filter((s) => s.x > 105 && s.x < 113);
    expect(slot2.length).toBeGreaterThan(0);
    for (const s of slot2) expect(Math.abs(s.roll)).toBeGreaterThan(1.25);
  });
});
