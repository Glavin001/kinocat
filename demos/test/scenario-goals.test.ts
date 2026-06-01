// Canonical goal-expression tests for the core ground-vehicle scenarios. Each
// authored Scenario must (1) compile, (2) validate clean, and (3) be solvable
// by the real ScenarioEnvironment product search (`planVehicleScenario`).

import { describe, it, expect } from 'vitest';
import { compile, validate } from 'kinocat/scenario';
import {
  authorParkingScenario,
  planParkingScenario,
  authorRaceLap,
  authorRaceCircuit,
  planRaceLap,
  authorPointToPoint,
} from '../app/lib/scenario-goals';
import { planVehicleScenario } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { demoVehicle } from '../app/lib/scenarios';
import type { ParkingScenarioId } from '../app/lib/parking-scenarios';

const PARKING_IDS: ParkingScenarioId[] = ['forward-pullin', 'reverse-perp', 'parallel'];

describe('authored scenarios compile + validate clean', () => {
  it('parking variants compile and validate', () => {
    for (const id of PARKING_IDS) {
      const sc = authorParkingScenario(id);
      expect(() => compile(sc.goal)).not.toThrow();
      const errors = validate(sc, { posCell: 0.3 }).filter((d) => d.severity === 'error');
      expect(errors).toEqual([]);
    }
  });

  it('race lap + circuit compile and validate', () => {
    const lap = authorRaceLap();
    const circuit = authorRaceCircuit();
    expect(compile(lap.goal).accepting.length).toBeGreaterThan(0);
    expect(compile(circuit.goal).progress).toBe(true);
    expect(validate(lap, { posCell: 1.5 }).filter((d) => d.severity === 'error')).toEqual([]);
    expect(validate(circuit, { posCell: 1.5 }).filter((d) => d.severity === 'error')).toEqual([]);
  });

  it('point-to-point compiles and validates', () => {
    const sc = authorPointToPoint({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 40, z: 0 },
      obstacles: [{ x: 22, z: 0 }],
    });
    expect(validate(sc, { posCell: 1 }).filter((d) => d.severity === 'error')).toEqual([]);
  });
});

describe('planVehicleScenario solves the authored goals', () => {
  it('point-to-point reaches the goal around an obstacle', () => {
    const sc = authorPointToPoint({
      start: { x: 4, z: 0, heading: 0, speed: 0, t: 0 },
      goal: { x: 40, z: 0 },
      obstacles: [{ x: 22, z: 0 }],
    });
    const { agent, lib } = demoVehicle();
    const world = new InMemoryNavWorld(
      [{ id: 1, y: 0, ring: [[0, -11], [44, -11], [44, 11], [0, 11]] }],
      [],
    );
    const r = planVehicleScenario({
      start: sc.start,
      goal: sc.goal,
      invariants: sc.invariants,
      prefer: sc.prefer,
      world,
      agent,
      lib,
      envOptions: { posCell: 1, headingBuckets: 12, goalRadius: 2 },
      deadlineMs: Infinity, // deterministic, expansion-bounded
      maxExpansions: 40000,
    });
    expect(r.raw.found).toBe(true);
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - 40, last.z)).toBeLessThanOrEqual(2.5);
  });

  it('a parking goal yields a plan that advances toward the stall', () => {
    const r = planParkingScenario('forward-pullin', { maxExpansions: 80000 });
    // The product search should find a plan (full or best-progress partial).
    expect(r.raw.found).toBe(true);
    expect(r.path.length).toBeGreaterThanOrEqual(2);
  });

  it('a single race lap reaches the final waypoint', () => {
    const r = planRaceLap(undefined, { maxExpansions: 200000 });
    expect(r.raw.found).toBe(true);
  });
});
