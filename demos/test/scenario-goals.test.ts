// Canonical goal-expression tests for the core ground-vehicle scenarios. Each
// authored Scenario must (1) compile, (2) validate clean, and (3) be solvable
// by the real ScenarioEnvironment product search (`planVehicleScenario`).

import { describe, it, expect } from 'vitest';
import { compile, validate, evaluateProgress } from 'kinocat/scenario';
import type { Scenario } from 'kinocat/scenario';
import {
  authorParkingScenario,
  planParkingScenario,
  authorRaceLap,
  authorRaceCircuit,
  planRaceLap,
  authorPointToPoint,
} from '../app/lib/scenario-goals';
import { buildRaceCourse, RACE_ARRIVE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { planVehicleScenario } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { demoVehicle } from '../app/lib/scenarios';
import type { CarKinematicState } from 'kinocat/agent';
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

describe('race course carries its canonical scenario goal (course.goal)', () => {
  it('both variants author goal/invariants/prefer that compile + validate clean', () => {
    for (const variant of ['open', 'technical'] as const) {
      const course = buildRaceCourse(variant);
      expect(course.goal).toBeDefined();
      expect(course.invariants?.length).toBeGreaterThan(0);
      expect(course.prefer?.length).toBeGreaterThan(0);
      const automaton = compile(course.goal!);
      // The circuit is a repeat(...) — a progress objective, one phase per gate.
      expect(automaton.progress).toBe(true);
      const maxDepth = automaton.states.reduce((m, s) => Math.max(m, s.depth), 0);
      expect(maxDepth).toBeGreaterThanOrEqual(course.waypoints.length - 1);
      const sc: Scenario = {
        name: `race-${variant}`,
        start: course.spawn,
        goal: course.goal!,
        invariants: course.invariants,
        prefer: course.prefer,
      };
      expect(validate(sc, { posCell: 1.5 }).filter((d) => d.severity === 'error')).toEqual([]);
    }
  });

  it('replaying a synthetic lap advances the automaton and completes one objective lap', () => {
    const course = buildRaceCourse();
    const automaton = compile(course.goal!);
    // Dense piecewise-linear lap: spawn → w0 → … → w10 → w0, sampled every
    // 0.5 m with heading = the CURRENT segment's direction. On the samples
    // that leave gate i's capture disk the heading sits exactly on the
    // outgoing chord — inside the authored chord-aligned heading band — so
    // every gate guard fires while its disk still contains the car. (Samples
    // ENTERING a slalom gate carry the incoming chord heading, which can sit
    // outside the band; the guard correctly waits for the post-corner
    // samples. That asymmetry is the heading prior working as designed.)
    const pts = [course.spawn, ...course.waypoints, course.waypoints[0]!];
    const traj: CarKinematicState[] = [];
    let t = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i]!;
      const b = pts[i + 1]!;
      const len = Math.hypot(b.x - a.x, b.z - a.z);
      const heading = Math.atan2(b.z - a.z, b.x - a.x);
      const n = Math.max(2, Math.ceil(len / 0.5));
      for (let k = 0; k < n; k++) {
        const u = k / n;
        traj.push({
          x: a.x + (b.x - a.x) * u,
          z: a.z + (b.z - a.z) * u,
          heading,
          speed: 10,
          t: (t += 0.05),
        });
      }
    }
    const p = evaluateProgress(automaton, traj);
    // The final leg re-enters gate 0 → the repeat back-edge fires: one full
    // objective lap, with the automaton reset near the start of the cycle.
    expect(p.laps).toBe(1);
    // Sanity: the gate disks the trajectory was built from match the runner's
    // arrive radius (the authored goal describes what counts as CLEARING).
    expect(RACE_ARRIVE_RADIUS).toBe(2.5);
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
