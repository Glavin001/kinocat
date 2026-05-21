import { describe, it, expect } from 'vitest';
import { InMemoryNavWorld, type NavPolygon } from '../../src/environment/nav-world';
import { nudgeGoalClear } from '../../src/environment/nudge-goal';
import { defaultVehicleAgent } from '../../src/agent/vehicle';
import type { Pt } from '../../src/internal/geom';

function rect(x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id: 1, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

const agent = defaultVehicleAgent({
  minTurnRadius: 4,
  maxSpeed: 10,
  maxReverseSpeed: 3,
  footprint: [
    [2, 1],
    [-2, 1],
    [-2, -1],
    [2, -1],
  ] as Pt[],
});

describe('nudgeGoalClear', () => {
  const floor = rect(-50, -50, 50, 50);
  const obstacle: Pt[] = [
    [-5, -5],
    [5, -5],
    [5, 5],
    [-5, 5],
  ];
  const world = new InMemoryNavWorld([floor], [obstacle]);

  it('returns the goal unchanged when already clear', () => {
    const goal = { x: 20, z: 0, heading: 0, speed: 0, t: 0 };
    expect(nudgeGoalClear(goal, { x: -20, z: 0 }, world, agent)).toBe(goal);
  });

  it('walks the goal toward `near` until it is clear', () => {
    const goal = { x: 0, z: 0, heading: 0, speed: 5, t: 0 };
    const near = { x: -20, z: 0 };
    const out = nudgeGoalClear(goal, near, world, agent);
    // Should have moved toward `near` (negative-X direction) until clear of
    // the obstacle (which extends to x = -5) plus the 2-unit half-footprint.
    expect(out.x).toBeLessThan(-5);
    expect(out.heading).toBe(0);
    expect(out.speed).toBe(5);
    expect(out.t).toBe(0);
    // And the result actually passes the planner's footprint check.
    const c = Math.cos(out.heading);
    const s = Math.sin(out.heading);
    const fp = agent.footprint.map(
      ([lx, lz]) => [out.x + lx * c - lz * s, out.z + lx * s + lz * c] as const,
    );
    expect(world.footprintClear(fp)).toBe(true);
  });

  it('falls back to `near` when no clear pose is reachable in maxSteps', () => {
    // Tiny step + tiny maxSteps so the walk never escapes the obstacle.
    const goal = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const near = { x: -20, z: 0 };
    const out = nudgeGoalClear(goal, near, world, agent, { step: 0.01, maxSteps: 2 });
    expect(out.x).toBe(near.x);
    expect(out.z).toBe(near.z);
  });

  it('returns the goal unchanged when goal and near are the same point', () => {
    const goal = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const out = nudgeGoalClear(goal, { x: 0, z: 0 }, world, agent);
    expect(out.x).toBe(0);
    expect(out.z).toBe(0);
  });
});
