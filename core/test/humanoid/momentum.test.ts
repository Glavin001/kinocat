// Behavior tests: the momentum humanoid must actually behave like a body
// with mass — braking distance is bounded by maxDecel, a sprinter needs
// more room to corner than a walker, and plans through a doorway slow down
// on approach instead of teleport-turning.

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { MomentumHumanoidEnvironment } from '../../src/environment/momentum-humanoid-environment';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import {
  defaultMomentumHumanoidAgent,
  momentumHumanoidForwardSim,
  turnRateAt,
} from '../../src/agent/momentum-humanoid';
import type { MomentumHumanoidState } from '../../src/agent/types';
import { rect } from '../fixtures/vehicle-sweep';

const agent = defaultMomentumHumanoidAgent();

function speed(s: MomentumHumanoidState): number {
  return Math.hypot(s.vx, s.vz);
}

describe('momentumHumanoidForwardSim', () => {
  const sim = momentumHumanoidForwardSim(agent);
  const rest: MomentumHumanoidState = { x: 0, z: 0, heading: 0, vx: 0, vz: 0, t: 0 };

  it('accelerates forward under full throttle, capped at maxSpeed', () => {
    let s = rest;
    for (let i = 0; i < 50; i++) s = sim(s, [1, 0, 0], 0.1);
    expect(speed(s)).toBeCloseTo(agent.maxSpeed, 6);
    expect(s.x).toBeGreaterThan(0);
    expect(Math.abs(s.z)).toBeLessThan(1e-9);
  });

  it('launch is bounded by maxAccel, braking by maxDecel through rest', () => {
    const launched = sim(rest, [1, 0, 0], 0.5);
    expect(speed(launched)).toBeLessThanOrEqual(agent.maxAccel * 0.5 + 1e-9);

    let s = rest;
    for (let i = 0; i < 50; i++) s = sim(s, [1, 0, 0], 0.1); // sprint
    let minSpeed = speed(s);
    const brakeStep = sim(s, [1, Math.PI, 0], 0.1);
    // One braking step sheds at most maxDecel·dt.
    expect(speed(s) - speed(brakeStep)).toBeLessThanOrEqual(
      agent.maxDecel * 0.1 + 1e-9,
    );
    // Held backward push: brakes THROUGH rest (the zero-crossing clamp),
    // then becomes a backpedal capped at the strafe speed — never a
    // backward sprint.
    for (let i = 0; i < 40; i++) {
      s = sim(s, [1, Math.PI, 0], 0.1);
      minSpeed = Math.min(minSpeed, speed(s));
    }
    expect(minSpeed).toBeLessThan(1e-9);
    expect(speed(s)).toBeLessThanOrEqual(agent.strafeSpeed + 1e-9);
    expect(s.vx).toBeLessThan(0); // backpedaling, facing unchanged
    expect(s.heading).toBeCloseTo(0, 9);
  });

  it('cannot sustain sprint sideways — the strafe cap binds', () => {
    // Accelerate hard 90° off the facing, never turning.
    let s = rest;
    for (let i = 0; i < 100; i++) s = sim(s, [1, Math.PI / 2, 0], 0.1);
    expect(speed(s)).toBeLessThanOrEqual(agent.strafeSpeed + 1e-9);
  });

  it('turn rate degrades with speed', () => {
    expect(turnRateAt(agent, agent.maxSpeed)).toBeLessThan(
      turnRateAt(agent, 0) * 0.35,
    );
  });
});

describe('momentum shows up in plans', () => {
  it('a sprint start needs more room to reverse direction than a standing start', () => {
    const world = new InMemoryNavWorld([rect(1, 0, 0, 40, 12)]);
    const env = () => new MomentumHumanoidEnvironment(world, agent);
    const goal: MomentumHumanoidState = { x: 5, z: 6, heading: 0, vx: 0, vz: 0, t: 0 };
    const standing: MomentumHumanoidState = { x: 15, z: 6, heading: 0, vx: 0, vz: 0, t: 0 };
    const sprinting: MomentumHumanoidState = {
      x: 15,
      z: 6,
      heading: 0,
      vx: agent.maxSpeed, // running AWAY from the goal
      vz: 0,
      t: 0,
    };
    const a = plan(
      { start: standing, goal, environment: env(), options: { maxExpansions: 300_000 } },
      Infinity,
    );
    const b = plan(
      { start: sprinting, goal, environment: env(), options: { maxExpansions: 300_000 } },
      Infinity,
    );
    expect(a.found).toBe(true);
    expect(b.found).toBe(true);
    // Momentum carries the sprinter past the start before they can turn back.
    const overshootA = Math.max(...a.path.map((s) => s.x)) - 15;
    const overshootB = Math.max(...b.path.map((s) => s.x)) - 15;
    expect(overshootB).toBeGreaterThan(overshootA + 1);
    expect(b.cost).toBeGreaterThan(a.cost);
  });

  it('threads a tight doorway on a straight run without clipping it', () => {
    // A 1.4 m doorway dead ahead: the straight line to the goal goes through
    // it, so the search is well-guided; the interesting part is that the
    // committed trajectory fits the gap with an inertial body.
    const world = new InMemoryNavWorld([
      rect(1, 0, 4, 14, 15),
      rect(2, 16, 4, 28, 15),
      rect(3, 13, 9, 17, 10.4), // 1.4 m-wide doorway bridge
    ]);
    const env = new MomentumHumanoidEnvironment(world, agent);
    const start: MomentumHumanoidState = { x: 4, z: 9.7, heading: 0, vx: 0, vz: 0, t: 0 };
    const goal: MomentumHumanoidState = { x: 24, z: 9.7, heading: 0, vx: 0, vz: 0, t: 0 };
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 250_000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    // Builds real speed in the open stretch...
    expect(Math.max(...r.path.map(speed))).toBeGreaterThan(agent.strafeSpeed);
    // ...and the crossing itself threads the 1.4 m slot: interpolate the
    // committed states across the bridge-only span (path states are 0.5 s
    // apart, so a single step can hop the whole span; the chord is a close
    // stand-in for the primitive curve at door speeds — allow small slack).
    const crossings: Array<{ x: number; z: number }> = [];
    for (let i = 1; i < r.path.length; i++) {
      const a = r.path[i - 1]!;
      const b = r.path[i]!;
      for (const u of [0, 0.25, 0.5, 0.75]) {
        const x = a.x + (b.x - a.x) * u;
        if (x > 14.3 && x < 15.7) {
          crossings.push({ x, z: a.z + (b.z - a.z) * u });
        }
      }
    }
    expect(crossings.length).toBeGreaterThan(0);
    for (const cr of crossings) {
      expect(cr.z).toBeGreaterThan(9 + agent.radius - 0.15);
      expect(cr.z).toBeLessThan(10.4 - agent.radius + 0.15);
    }
    // Time is monotone along the plan (kinodynamic contract).
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t);
    }
  });
});
