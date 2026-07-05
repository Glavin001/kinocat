// The generalized dynamic-world layer: TimeAwareEnvironment accepts any
// {x, z, t(, y)} state. When both the state and an obstacle's prediction
// carry `y`, the collision proxy is a 3D sphere; a y-less pairing degrades
// to the planning-plane circle (an infinite vertical cylinder). The padded-
// AABB broadphase must stay a pure accelerator in 3D exactly as it is in 2D.

import { describe, it, expect } from 'vitest';
import type { Environment, EdgeRef, Node } from '../../src/environment/types';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { AircraftEnvironment } from '../../src/environment/aircraft-environment';
import { InMemoryAirspace } from '../../src/environment/airspace-world';
import { defaultAircraftAgent } from '../../src/agent/aircraft';
import { plan } from '../../src/planner/ighastar';
import { makeNode } from '../../src/planner/node';
import type { MovingObstacle } from '../../src/predict/types';
import type { AircraftState } from '../../src/agent/types';
import { rng } from '../../src/testing';

interface S3 {
  x: number;
  y: number;
  z: number;
  t: number;
}

/** Minimal 3D base env: one straight successor, no static collision. */
class Free3DEnv implements Environment<S3> {
  readonly levels = 2;
  createNode(state: S3, parent: Node<S3> | null, edge: EdgeRef | null): Node<S3> {
    const k = `${Math.round(state.x)},${Math.round(state.y)},${Math.round(state.z)}`;
    return makeNode(state, parent, edge, [k, k], `${k},${Math.round(state.t / 0.25)}`);
  }
  succ(node: Node<S3>, goal: Node<S3>): Node<S3>[] {
    const st = node.state;
    const next: S3 = { x: st.x + 2, y: st.y, z: st.z, t: st.t + 1 };
    const edge: EdgeRef = { cost: 1, kind: 'step' };
    const n = this.createNode(next, node, edge);
    n.g = node.g + 1;
    n.h = this.heuristic(next, goal.state);
    n.f = n.g + n.h;
    return [n];
  }
  heuristic(from: S3, to: S3): number {
    return Math.hypot(from.x - to.x, from.y - to.y, from.z - to.z) / 2;
  }
  checkValidity(): [boolean, boolean] {
    return [true, true];
  }
  reachedGoalRegion(node: Node<S3>, goal: Node<S3>): boolean {
    return Math.hypot(node.state.x - goal.state.x, node.state.z - goal.state.z) <= 1;
  }
}

/** A stationary obstacle hovering at (x, y?, z). */
function hover(x: number, z: number, radius: number, y?: number): MovingObstacle {
  return {
    radius,
    predict: () => (y === undefined ? { x, z } : { x, z, y }),
  };
}

function succStates(env: Environment<S3>, from: S3): S3[] {
  const start = env.createNode(from, null, null);
  const goal = env.createNode({ x: 100, y: from.y, z: from.z, t: 0 }, null, null);
  return env.succ(start, goal).map((n) => n.state);
}

describe('TimeAware 3D obstacle semantics', () => {
  // The single successor of {x:0} lands at (2, y, 0) at t=1.
  it('a 3D obstacle at a different altitude does NOT prune (sphere, not circle)', () => {
    const env = new TimeAwareEnvironment<S3>(new Free3DEnv(), {
      obstacles: [hover(2, 0, 3, /* y */ 40)],
    });
    expect(succStates(env, { x: 0, y: 10, z: 0, t: 0 })).toHaveLength(1);
  });

  it('a 3D obstacle at the same altitude prunes', () => {
    const env = new TimeAwareEnvironment<S3>(new Free3DEnv(), {
      obstacles: [hover(2, 0, 3, /* y */ 10)],
    });
    expect(succStates(env, { x: 0, y: 10, z: 0, t: 0 })).toHaveLength(0);
  });

  it('a y-less obstacle acts as a vertical cylinder against a 3D state', () => {
    const env = new TimeAwareEnvironment<S3>(new Free3DEnv(), {
      obstacles: [hover(2, 0, 3)],
    });
    // Altitude 10 vs unspecified: still pruned — conservative fallback.
    expect(succStates(env, { x: 0, y: 10, z: 0, t: 0 })).toHaveLength(0);
  });
});

describe('TimeAware broadphase is a pure accelerator in 3D', () => {
  it('with and without broadphase produce identical successor sets', () => {
    const rand = rng(0xbeef);
    const obstacles: MovingObstacle[] = [];
    for (let i = 0; i < 12; i++) {
      const x0 = rand() * 40 - 20;
      const z0 = rand() * 40 - 20;
      const y0 = rand() * 40;
      const vx = rand() * 4 - 2;
      const vz = rand() * 4 - 2;
      const vy = rand() * 2 - 1;
      const r = 1 + rand() * 3;
      const planar = rand() < 0.3;
      const horizon = rand() < 0.2 ? 5 : 60; // some short validity windows
      obstacles.push({
        radius: r,
        predict: (t) =>
          t < 0 || t > horizon
            ? null
            : planar
              ? { x: x0 + vx * t, z: z0 + vz * t }
              : { x: x0 + vx * t, z: z0 + vz * t, y: y0 + vy * t },
      });
    }
    const exact = new TimeAwareEnvironment<S3>(new Free3DEnv(), {
      obstacles,
      agentRadius: 0.5,
      broadphase: false,
    });
    const fast = new TimeAwareEnvironment<S3>(new Free3DEnv(), {
      obstacles,
      agentRadius: 0.5,
      broadphase: {},
    });
    const probes: S3[] = [];
    for (let i = 0; i < 400; i++) {
      probes.push({
        x: rand() * 60 - 30,
        y: rand() * 50,
        z: rand() * 60 - 30,
        t: rand() * 40,
      });
    }
    for (const p of probes) {
      const a = succStates(exact, p);
      const b = succStates(fast, p);
      expect(b).toEqual(a);
    }
  });
});

describe('TimeAware(AircraftEnvironment) composition', () => {
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
  const RR = 8 + 2; // obstacle radius + agentRadius

  function makeEnvs() {
    const world = new InMemoryAirspace({ floor: 2, ceiling: 60 });
    // The inner time bucket is redundant under composition — exercise the
    // opt-out added for exactly this case.
    const base = new AircraftEnvironment(world, agent, { timeInHash: false });
    const obstacle = hover(60, 0, 8, /* y */ 20);
    const wrapped = new TimeAwareEnvironment<AircraftState>(base, {
      obstacles: [obstacle],
      agentRadius: 2,
    });
    return { base, wrapped, obstacle };
  }

  const start: AircraftState = { x: 0, y: 20, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 };
  const goal: AircraftState = { x: 120, y: 20, z: 0, heading: 0, pitch: 0, roll: 0, speed: 18, t: 0 };

  it('the plane routes around a hovering 3D no-fly sphere on its straight line', () => {
    const { base, wrapped, obstacle } = makeEnvs();

    // Contrast: the unwrapped env flies straight through the sphere.
    const straight = plan(
      { start, goal, environment: base, options: { maxExpansions: 100_000 } },
      Infinity,
    );
    expect(straight.found).toBe(true);
    const straightMin = Math.min(
      ...straight.path.map((s) => {
        const p = obstacle.predict(s.t)!;
        return Math.hypot(s.x - p.x, s.y - (p.y ?? s.y), s.z - p.z);
      }),
    );
    expect(straightMin).toBeLessThan(RR);

    // Wrapped: every committed state keeps 3D clearance at its own time.
    const r = plan(
      { start, goal, environment: wrapped, options: { maxExpansions: 200_000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    for (const s of r.path) {
      const p = obstacle.predict(s.t)!;
      const d = Math.hypot(s.x - p.x, s.y - (p.y ?? s.y), s.z - p.z);
      expect(d).toBeGreaterThan(RR - 1e-9);
    }
    for (let i = 1; i < r.path.length; i++) {
      expect(r.path[i]!.t).toBeGreaterThan(r.path[i - 1]!.t);
    }
  });
});
