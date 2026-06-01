// Unit tests for the shared "correctly parked" predicate — the single source of
// truth (`evaluateParked`) that the /parking web HUD, the controller-bench CLI,
// and the Vitest invariant tests all consume. Pure geometry: no Rapier, no
// physics sim, so the predicate's pass/fail boundary is pinned independently of
// the controller that has to reach it.

import { describe, it, expect } from 'vitest';
import {
  buildParkingScenario,
  evaluateParked,
  PARKING_AGENT,
  PARKING_SUCCESS,
} from '../app/lib/parking-scenarios';

const stopped = { speed: 0 };

describe('evaluateParked', () => {
  it('forward-pullin: exactly on the goal pose is parked (full coverage)', () => {
    const s = buildParkingScenario('forward-pullin');
    const ev = evaluateParked({ x: s.goal.x, z: s.goal.z, heading: s.goal.heading, ...stopped }, s);
    expect(ev.parked).toBe(true);
    expect(ev.footprintInStall).toBe(true);
    expect(ev.stopped).toBe(true);
    expect(ev.coverage).toBeGreaterThan(0.99);
    expect(ev.headingError).toBeCloseTo(0, 6);
  });

  it('reverse-perp: exactly on the goal pose is parked', () => {
    const s = buildParkingScenario('reverse-perp');
    const ev = evaluateParked({ x: s.goal.x, z: s.goal.z, heading: s.goal.heading, ...stopped }, s);
    expect(ev.parked).toBe(true);
    expect(ev.coverage).toBeGreaterThan(0.99);
  });

  it('parallel: exactly on the goal pose is parked', () => {
    const s = buildParkingScenario('parallel');
    const ev = evaluateParked({ x: s.goal.x, z: s.goal.z, heading: s.goal.heading, ...stopped }, s);
    expect(ev.parked).toBe(true);
    expect(ev.coverage).toBeGreaterThan(0.99);
  });

  it('is NOT parked when still moving (right pose, nonzero speed)', () => {
    const s = buildParkingScenario('forward-pullin');
    const ev = evaluateParked(
      { x: s.goal.x, z: s.goal.z, heading: s.goal.heading, speed: 1.0 },
      s,
    );
    expect(ev.stopped).toBe(false);
    expect(ev.parked).toBe(false);
  });

  it('parallel: stopped in the slot but angled (~16° off) is NOT parked', () => {
    // The slot is long, so an angled car still mostly overlaps the box and
    // coverage stays high — the heading tolerance is what rejects it. This is
    // the exact failure mode the demo showed (parallel car offset/angled).
    const s = buildParkingScenario('parallel');
    const ev = evaluateParked(
      { x: s.goal.x, z: s.goal.z, heading: s.goal.heading + 0.28, ...stopped },
      s,
    );
    expect(ev.coverage).toBeGreaterThan(0.6); // still overlaps the long box
    expect(ev.headingError).toBeGreaterThan(PARKING_SUCCESS.headingTol);
    expect(ev.footprintInStall).toBe(false);
    expect(ev.parked).toBe(false);
  });

  it('forward-pullin: stopped but laterally offset out of the stall is NOT parked', () => {
    // Same-size stall: a 1.2 m lateral offset pushes the footprint mostly out
    // of the box, so coverage collapses even though it is squared up.
    const s = buildParkingScenario('forward-pullin');
    const ev = evaluateParked(
      { x: s.goal.x + 1.2, z: s.goal.z, heading: s.goal.heading, ...stopped },
      s,
    );
    expect(ev.coverage).toBeLessThan(PARKING_SUCCESS.coverageMin);
    expect(ev.parked).toBe(false);
  });

  it('uses the agent footprint by default and accepts an override', () => {
    const s = buildParkingScenario('forward-pullin');
    const onGoal = { x: s.goal.x, z: s.goal.z, heading: s.goal.heading, ...stopped };
    expect(evaluateParked(onGoal, s).coverage).toBeCloseTo(
      evaluateParked(onGoal, s, PARKING_AGENT.footprint).coverage,
      9,
    );
  });
});
