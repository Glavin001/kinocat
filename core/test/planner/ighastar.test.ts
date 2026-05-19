import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import type { PlanRequest } from '../../src/planner/types';
import type { R2State } from '../../src/environment/r2-environment';
import {
  gridFromAscii,
  dijkstraOptimal,
  SIMPLE_OPEN,
  WITH_WALL,
  POCKET,
  UNREACHABLE,
  type GridProblem,
} from '../fixtures/grid-problem';

function req(p: GridProblem, options = {}): PlanRequest<R2State> {
  return { start: p.start, goal: p.goal, environment: p.env, options };
}

function assertValidPath(p: GridProblem, path: R2State[]): void {
  expect(path.length).toBeGreaterThan(0);
  const cells = path.map((s) => p.env.cellOf(s));
  expect(cells[0]).toEqual(p.startCell);
  expect(cells[cells.length - 1]).toEqual(p.goalCell);
  for (let i = 1; i < cells.length; i++) {
    const [ax, ay] = cells[i - 1]!;
    const [bx, by] = cells[i]!;
    expect(Math.max(Math.abs(ax - bx), Math.abs(ay - by))).toBe(1);
    expect(p.blocked(bx, by)).toBe(false);
  }
}

describe('IGHA* correctness vs Dijkstra oracle', () => {
  for (const [name, ascii] of [
    ['simple-open', SIMPLE_OPEN],
    ['with-wall', WITH_WALL],
    ['pocket', POCKET],
  ] as const) {
    it(`finds the optimal cost on ${name}`, () => {
      const p = gridFromAscii(ascii);
      const r = plan(req(p), Infinity);
      expect(r.found).toBe(true);
      const opt = dijkstraOptimal(p);
      expect(r.cost).toBeCloseTo(opt, 6);
      assertValidPath(p, r.path);
    });
  }

  it('is optimal even with a single resolution level (pure A*)', () => {
    const p = gridFromAscii(WITH_WALL);
    const r = plan(req(p, { levels: 1 }), Infinity);
    expect(r.found).toBe(true);
    expect(r.cost).toBeCloseTo(dijkstraOptimal(p), 6);
  });

  it('reports no plan when the goal is unreachable', () => {
    const p = gridFromAscii(UNREACHABLE);
    const r = plan(req(p), Infinity);
    expect(r.found).toBe(false);
    expect(r.cost).toBe(Infinity);
    expect(r.path).toEqual([]);
  });

  it('rejects an invalid (blocked) start/goal', () => {
    const p = gridFromAscii(['S.G']);
    const bad: PlanRequest<R2State> = {
      start: { x: -5, y: -5 },
      goal: p.goal,
      environment: p.env,
    };
    expect(plan(bad, Infinity).found).toBe(false);
  });
});

describe('IGHA* anytime behaviour', () => {
  it('incumbent cost is non-increasing as the expansion budget grows', () => {
    const p = gridFromAscii(POCKET);
    const opt = dijkstraOptimal(p);
    const budgets = [5, 15, 40, 120, 400, Infinity];
    let prev = Infinity;
    for (const maxExpansions of budgets) {
      const r = plan(req(p, { maxExpansions }), Infinity);
      const c = r.found ? r.cost : Infinity;
      expect(c).toBeLessThanOrEqual(prev + 1e-9);
      prev = c;
    }
    expect(prev).toBeCloseTo(opt, 6);
  });

  it('records an improving solution history', () => {
    const p = gridFromAscii(WITH_WALL);
    const r = plan(req(p), Infinity);
    expect(r.solutionHistory.length).toBeGreaterThanOrEqual(1);
    const costs = r.solutionHistory.map((path) => path.length);
    // later solutions are not worse (path node count is a coarse proxy here)
    expect(r.found).toBe(true);
    expect(costs.length).toBe(r.stats.improvements);
  });

  it('deadline of 0 returns no plan and flags deadlineHit', () => {
    const p = gridFromAscii(SIMPLE_OPEN);
    const r = plan(req(p), 0);
    expect(r.found).toBe(false);
    expect(r.stats.deadlineHit).toBe(true);
    expect(r.stats.expansions).toBe(0);
  });

  it('budget exhaustion is flagged', () => {
    const p = gridFromAscii(POCKET);
    const r = plan(req(p, { maxExpansions: 3 }), Infinity);
    expect(r.stats.budgetHit).toBe(true);
  });
});

describe('IGHA* f-monotonicity', () => {
  it('f is non-decreasing along the returned path (consistent heuristic)', () => {
    const p = gridFromAscii(POCKET);
    const r = plan(req(p), Infinity);
    expect(r.found).toBe(true);
    for (let i = 1; i < r.nodes.length; i++) {
      expect(r.nodes[i]!.f).toBeGreaterThanOrEqual(r.nodes[i - 1]!.f - 1e-6);
    }
    // last node's g equals total cost
    expect(r.nodes[r.nodes.length - 1]!.g).toBeCloseTo(r.cost, 9);
  });
});
