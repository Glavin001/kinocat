// Technical race-course variant: walls, guard blocks, and a thread-the-gate
// corridor. These tests confirm the geometry is well-formed, the walls are
// reflected as planner obstacles, the racing line stays feasible (a plan
// exists through the walled slalom), and the open variant is unchanged
// (backward compatibility for every existing race consumer).

import { describe, it, expect } from 'vitest';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
} from 'kinocat/agent';
import { InMemoryNavWorld } from 'kinocat/environment';
import { hashGoal } from 'kinocat/scenario';
import {
  buildRaceCourse,
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  planRaceMultiGoal,
  RACE_AGENT,
  RACE_WALL_INFLATE,
} from '../app/lib/race-primitives-scenarios';

describe('technical race course geometry', () => {
  it('open variant is unchanged: no walls, no obstacles', () => {
    const open = buildRaceCourse();
    expect(open.variant).toBe('open');
    expect(open.walls).toEqual([]);
    expect(open.obstacles).toEqual([]);
    // Default arg matches explicit 'open'. The authored scenario planes
    // (goal / invariants / prefer) carry Region CLOSURES, which are
    // reference-unequal across two builds by nature — compare those
    // STRUCTURALLY (hashGoal / region keys / cost-term identity) and
    // everything else (geometry, waypoints, spawn) by deep equality.
    const again = buildRaceCourse('open');
    const { goal: gA, invariants: iA, prefer: pA, ...restA } = again;
    const { goal: gO, invariants: iO, prefer: pO, ...restO } = open;
    expect(restA).toEqual(restO);
    expect(gA).toBeDefined();
    expect(hashGoal(gA!)).toBe(hashGoal(gO!));
    expect(iA!.map((inv) => `${inv.kind}:${inv.region.key}`)).toEqual(
      iO!.map((inv) => `${inv.kind}:${inv.region.key}`),
    );
    expect(pA!.map((c) => `${c.name}:${c.weight}`)).toEqual(
      pO!.map((c) => `${c.name}:${c.weight}`),
    );
  });

  it('technical variant adds walls, mirrored as inflated planner obstacles', () => {
    const tech = buildRaceCourse('technical');
    const walls = tech.walls ?? [];
    expect(tech.variant).toBe('technical');
    expect(walls.length).toBeGreaterThan(0);
    // Every wall has a matching obstacle polygon inflated by RACE_WALL_INFLATE.
    expect(tech.obstacles.length).toBe(walls.length);
    for (let i = 0; i < walls.length; i++) {
      const w = walls[i]!;
      const ring = tech.obstacles[i]!;
      const xs = ring.map((p) => p[0]);
      const zs = ring.map((p) => p[1]);
      expect(Math.min(...xs)).toBeCloseTo(w.x - w.hx - RACE_WALL_INFLATE, 6);
      expect(Math.max(...xs)).toBeCloseTo(w.x + w.hx + RACE_WALL_INFLATE, 6);
      expect(Math.min(...zs)).toBeCloseTo(w.z - w.hz - RACE_WALL_INFLATE, 6);
      expect(Math.max(...zs)).toBeCloseTo(w.z + w.hz + RACE_WALL_INFLATE, 6);
    }
    // Same waypoints / bounds / spawn as open (same racing line).
    const open = buildRaceCourse('open');
    expect(tech.waypoints).toEqual(open.waypoints);
    expect(tech.bounds).toEqual(open.bounds);
    expect(tech.spawn).toEqual(open.spawn);
  });

  it('no wall obstacle covers a waypoint or the spawn (racing line stays clear)', () => {
    const tech = buildRaceCourse('technical');
    const world = new InMemoryNavWorld(tech.polygons, tech.obstacles);
    const pts = [tech.spawn, ...tech.waypoints];
    for (const p of pts) {
      // A waypoint must lie on the walkable surface (not inside an obstacle
      // hole) — otherwise the planner could never reach it.
      expect(world.polygonAt(p.x, p.z), `waypoint (${p.x},${p.z})`).not.toBeNull();
    }
  });

  it('the walled slalom is still solvable at the demo lookahead (2-3 gates)', () => {
    // The runtime plans PLAN_LOOKAHEAD_COUNT = 2 gates per replan; 4+ gate
    // joint searches exceed the multi-goal budget on BOTH open and technical
    // courses (a pre-existing planner scaling limit, not a wall issue). We
    // validate the horizon the demo actually uses: threading gates 0→1→2,
    // which passes the guard wall north of gate 1.
    const tech = buildRaceCourse('technical');
    const model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const lib = buildLearnedRaceLibraryV2(model);
    const gates = tech.waypoints.slice(0, 3).map((w) => ({ ...w, t: 0 }));
    const result = planRaceMultiGoal({
      state: tech.spawn,
      gates,
      lib,
      polygons: tech.polygons,
      obstacles: tech.obstacles,
      deadlineMs: 8_000,
      maxExpansions: 500_000,
      gateRadius: 1.8,
      agent: RACE_AGENT,
    });
    expect(result.found).toBe(true);
    // Every gate is reached, and every planned pose clears the walls (the
    // planner's footprint-collision check guarantees the latter, but assert
    // gate coverage explicitly).
    for (const gate of gates) {
      const minDist = Math.min(
        ...result.path.map((p) => Math.hypot(p.x - gate.x, p.z - gate.z)),
      );
      expect(minDist, `gate (${gate.x},${gate.z})`).toBeLessThan(2.5);
    }
  }, 20_000);

  it('the kinematic library also finds a feasible plan through the corridor', () => {
    const tech = buildRaceCourse('technical');
    const lib = buildKinematicLibrary();
    // Gates 9-10 straddle the return-leg thread-the-gate corridor.
    const gates = [tech.waypoints[8]!, tech.waypoints[9]!, tech.waypoints[10]!].map((w) => ({ ...w, t: 0 }));
    const result = planRaceMultiGoal({
      state: { ...tech.waypoints[7]!, t: 0 },
      gates,
      lib,
      polygons: tech.polygons,
      obstacles: tech.obstacles,
      deadlineMs: 15_000,
      maxExpansions: 800_000,
      gateRadius: 1.8,
      agent: RACE_AGENT,
    });
    expect(result.found).toBe(true);
  }, 30_000);
});
