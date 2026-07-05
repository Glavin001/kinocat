// Property-based optimality: the existing oracle test pins IGHA* to three
// hand-built grids. Those can't exercise the configurations that stress the
// multi-resolution dominance — a coarse-level merge that prunes a node the
// optimal path needs, a tie broken the wrong way, an anytime pass that fails to
// tighten. So fuzz it: hundreds of random grids, each checked against an
// INDEPENDENT 8-connected Dijkstra. The oracle uses the identical connectivity
// and cost model as R2Environment.succ (diagonal allowed when the destination
// cell is free — neither forbids corner-cutting; √2 / 1 step costs), so any
// divergence is a genuine planner bug, not an oracle mismatch.
//
// This is the test most likely to FAIL if the planner has a subtle correctness
// hole — which is exactly why it's worth having.

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import type { R2State } from '../../src/environment/r2-environment';
import {
  gridFromAscii,
  dijkstraOptimal,
  type GridProblem,
} from '../fixtures/grid-problem';
import { rng } from '../curves/_util';

/** Build a random ASCII grid: random size, random wall density, S/G on two
 *  distinct free cells. Returns null if it couldn't place S and G (retry). */
function randomGrid(rand: () => number): string[] | null {
  const cols = 6 + Math.floor(rand() * 9); // 6..14
  const rows = 6 + Math.floor(rand() * 9);
  const density = 0.12 + rand() * 0.28; // 12%..40% walls
  const cells: string[][] = [];
  const free: Array<[number, number]> = [];
  for (let y = 0; y < rows; y++) {
    const row: string[] = [];
    for (let x = 0; x < cols; x++) {
      if (rand() < density) row.push('#');
      else {
        row.push('.');
        free.push([x, y]);
      }
    }
    cells.push(row);
  }
  if (free.length < 2) return null;
  const a = free[Math.floor(rand() * free.length)]!;
  let b = free[Math.floor(rand() * free.length)]!;
  for (let tries = 0; tries < 8 && b[0] === a[0] && b[1] === a[1]; tries++) {
    b = free[Math.floor(rand() * free.length)]!;
  }
  if (b[0] === a[0] && b[1] === a[1]) return null;
  cells[a[1]]![a[0]] = 'S';
  cells[b[1]]![b[0]] = 'G';
  return cells.map((r) => r.join(''));
}

/** Re-measure a path's cost under the same 8-connected model — a second,
 *  independent check that the returned path is real and as cheap as claimed. */
function measure(p: GridProblem, path: R2State[]): number {
  let c = 0;
  for (let i = 1; i < path.length; i++) {
    const [ax, ay] = p.env.cellOf(path[i - 1]!);
    const [bx, by] = p.env.cellOf(path[i]!);
    const dx = Math.abs(ax - bx);
    const dy = Math.abs(ay - by);
    expect(Math.max(dx, dy), `non-adjacent step ${i}`).toBe(1);
    expect(p.blocked(bx, by), `step ${i} into a wall`).toBe(false);
    c += dx !== 0 && dy !== 0 ? Math.SQRT2 : 1;
  }
  return c;
}

describe('IGHA* optimality — randomized fuzz vs Dijkstra oracle', () => {
  const N = 300;
  const rand = rng(0xc0ffee);

  it(`matches Dijkstra cost on ${N} random grids (solvable and unsolvable)`, () => {
    let solvable = 0;
    let unsolvable = 0;
    let attempts = 0;
    for (let i = 0; i < N; ) {
      attempts++;
      if (attempts > N * 20) break; // safety, should never hit
      const ascii = randomGrid(rand);
      if (!ascii) continue;
      const p = gridFromAscii(ascii);
      const opt = dijkstraOptimal(p);
      const r = plan({ start: p.start, goal: p.goal, environment: p.env, options: {} }, Infinity);
      const ctx = `\nseed-case ${i} (${p.cols}x${p.rows}) S=${p.startCell} G=${p.goalCell}\n${ascii.join('\n')}\nopt=${opt} cost=${r.cost} found=${r.found}`;

      if (!Number.isFinite(opt)) {
        // No route exists ⇒ the planner must fail safe, not invent a path.
        expect(r.found, ctx).toBe(false);
        expect(r.cost, ctx).toBe(Infinity);
        unsolvable++;
      } else {
        expect(r.found, ctx).toBe(true);
        // Optimal cost (admissible heuristic + exact finest level ⇒ optimal).
        expect(r.cost, ctx).toBeCloseTo(opt, 6);
        // The returned path is real and its measured cost equals the reported one.
        const measured = measure(p, r.path);
        expect(measured, ctx).toBeCloseTo(r.cost, 6);
        solvable++;
      }
      i++;
    }
    // Sanity: the corpus actually exercised both branches (not all trivial).
    expect(solvable, `solvable=${solvable} unsolvable=${unsolvable}`).toBeGreaterThan(N * 0.5);
    expect(unsolvable, `solvable=${solvable} unsolvable=${unsolvable}`).toBeGreaterThan(0);
  });
});
