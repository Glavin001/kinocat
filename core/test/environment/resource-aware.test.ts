import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { R2Environment } from '../../src/environment/r2-environment';
import {
  ResourceAwareEnvironment,
  type ResourceAwareOptions,
} from '../../src/environment/resource-aware';
import type { R2State } from '../../src/environment/r2-environment';

// Smoke tests for the generic resource-aware wrapper. We exercise it on R2
// (the simplest domain) to keep the assertions about wrapper behavior pure
// — independent of any aircraft/vehicle/affordance specifics.

type Fuel = { fuel: number };

function makeR2(): R2Environment {
  return new R2Environment({
    step: 1,
    blocked: () => false,
    bounds: { minCx: -50, maxCx: 50, minCy: -50, maxCy: 50 },
  });
}

function fuelOpts(
  overrides: Partial<ResourceAwareOptions<R2State, Fuel>> = {},
): ResourceAwareOptions<R2State, Fuel> {
  return {
    initial: { fuel: 100 },
    bucket: ({ fuel }) => String(Math.round(fuel / 25)),
    allow: () => true,
    step: ({ fuel }) => ({ fuel }),
    ...overrides,
  };
}

describe('ResourceAwareEnvironment', () => {
  it('augments node hash/index with the resource bucket', () => {
    const base = makeR2();
    const env = new ResourceAwareEnvironment<R2State, Fuel>(base, fuelOpts());

    const a = env.createNode({ x: 1, y: 1 }, null, null);
    expect(a.hash).toMatch(/\|r/); // augmented
    expect(a.index.every((s) => s.includes('|r'))).toBe(true);
    expect(env.resourceOf(a)?.fuel).toBe(100);
  });

  it('two envs with different initial resources produce distinct hashes for the same state', () => {
    const base = makeR2();
    const env = new ResourceAwareEnvironment<R2State, Fuel>(base, fuelOpts());
    const env2 = new ResourceAwareEnvironment<R2State, Fuel>(
      base,
      fuelOpts({ initial: { fuel: 0 } }),
    );
    const a = env.createNode({ x: 1, y: 1 }, null, null);
    const b = env2.createNode({ x: 1, y: 1 }, null, null);
    expect(a.hash).not.toBe(b.hash);
  });

  it('allow() drops successors entirely', () => {
    const base = makeR2();
    const startBase = base.createNode({ x: 0, y: 0 }, null, null);
    const goalBase = base.createNode({ x: 5, y: 0 }, null, null);
    expect(base.succ(startBase, goalBase).length).toBeGreaterThan(0);

    const env = new ResourceAwareEnvironment<R2State, Fuel>(
      base,
      fuelOpts({ allow: () => false }),
    );
    const start = env.createNode({ x: 0, y: 0 }, null, null);
    const goal = env.createNode({ x: 5, y: 0 }, null, null);
    expect(env.succ(start, goal).length).toBe(0);
  });

  it('step() evolves the resource along an edge; resourceOf reflects it', () => {
    const base = makeR2();
    const env = new ResourceAwareEnvironment<R2State, Fuel>(
      base,
      fuelOpts({
        initial: { fuel: 50 },
        // every edge drains 10 fuel (dt is 0 for R2State so we ignore it)
        step: ({ fuel }) => ({ fuel: Math.max(0, fuel - 10) }),
      }),
    );
    const start = env.createNode({ x: 0, y: 0 }, null, null);
    const goal = env.createNode({ x: 5, y: 0 }, null, null);
    const succs = env.succ(start, goal);
    expect(succs.length).toBeGreaterThan(0);
    for (const s of succs) expect(env.resourceOf(s)?.fuel).toBe(40);
  });

  it('affordance() can top up the resource past step()', () => {
    const base = makeR2();
    const env = new ResourceAwareEnvironment<R2State, Fuel>(
      base,
      fuelOpts({
        initial: { fuel: 50 },
        step: ({ fuel }) => ({ fuel: Math.max(0, fuel - 10) }),
        affordance: ({ fuel }, from, to) => {
          // Any edge that crosses x = 1 → +50.
          if (Math.max(from.x, to.x) >= 1 && Math.min(from.x, to.x) < 1) {
            return { fuel: Math.min(100, fuel + 50) };
          }
          return null;
        },
      }),
    );
    const start = env.createNode({ x: 0, y: 0 }, null, null);
    const goal = env.createNode({ x: 5, y: 0 }, null, null);
    const succs = env.succ(start, goal);
    const fuels = succs.map((s) => env.resourceOf(s)!.fuel);
    expect(Math.max(...fuels)).toBeGreaterThan(40);
  });

  it('integrates with the planner end-to-end', () => {
    const base = makeR2();
    const env = new ResourceAwareEnvironment<R2State, Fuel>(base, fuelOpts());
    const res = plan(
      {
        start: { x: 0, y: 0 },
        goal: { x: 8, y: 0 },
        environment: env,
      },
      Infinity,
    );
    expect(res.found).toBe(true);
    expect(res.path.length).toBeGreaterThan(1);
  });

  it('gating BOOST-like edges by resource prunes a fast path when fuel is low', () => {
    // Tiny demo: edges with "cost === SQRT2" are diagonals — call them
    // "BOOST" and gate them by fuel. Force fuel = 0 and verify only axis-
    // aligned successors survive.
    const base = makeR2();
    const SQRT2 = Math.SQRT2;
    const env = new ResourceAwareEnvironment<R2State, Fuel>(base, {
      initial: { fuel: 0 },
      bucket: ({ fuel }) => String(Math.round(fuel / 25)),
      allow: ({ fuel }, _from, edge) =>
        edge.cost < SQRT2 - 1e-6 || fuel > 0,
      step: ({ fuel }, _from, edge) =>
        edge.cost >= SQRT2 - 1e-6
          ? { fuel: Math.max(0, fuel - 10) }
          : { fuel },
    });
    const start = env.createNode({ x: 0, y: 0 }, null, null);
    const goal = env.createNode({ x: 5, y: 0 }, null, null);
    const succs = env.succ(start, goal);
    // Diagonals require fuel > 0; we set fuel = 0, so only axis edges survive.
    for (const s of succs) {
      expect(s.edge!.cost).toBeLessThan(SQRT2 - 1e-6);
    }
  });
});
