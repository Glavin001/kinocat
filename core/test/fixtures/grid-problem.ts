import { R2Environment, type R2State } from '../../src/environment/r2-environment';

export interface GridProblem {
  env: R2Environment;
  start: R2State;
  goal: R2State;
  startCell: [number, number];
  goalCell: [number, number];
  cols: number;
  rows: number;
  blocked: (cx: number, cy: number) => boolean;
}

const DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

/** Build a problem from an ASCII map: '#'=wall, '.'=free, 'S'=start, 'G'=goal.
 *  Cell (cx,cy) = (column, row). step = 1. */
export function gridFromAscii(rowsAscii: string[], levelDivisors?: number[]): GridProblem {
  const rows = rowsAscii.length;
  const cols = Math.max(...rowsAscii.map((r) => r.length));
  const walls = new Set<string>();
  let startCell: [number, number] = [0, 0];
  let goalCell: [number, number] = [0, 0];
  for (let cy = 0; cy < rows; cy++) {
    const line = rowsAscii[cy]!;
    for (let cx = 0; cx < cols; cx++) {
      const ch = line[cx] ?? '#';
      if (ch === '#') walls.add(`${cx},${cy}`);
      else if (ch === 'S') startCell = [cx, cy];
      else if (ch === 'G') goalCell = [cx, cy];
    }
  }
  const blocked = (cx: number, cy: number): boolean => walls.has(`${cx},${cy}`);
  const env = new R2Environment({
    step: 1,
    blocked,
    bounds: { minCx: 0, maxCx: cols - 1, minCy: 0, maxCy: rows - 1 },
    levelDivisors,
  });
  return {
    env,
    start: { x: startCell[0], y: startCell[1] },
    goal: { x: goalCell[0], y: goalCell[1] },
    startCell,
    goalCell,
    cols,
    rows,
    blocked,
  };
}

/** Independent ground truth: 8-connected Dijkstra on the identical model. */
export function dijkstraOptimal(p: GridProblem): number {
  const key = (x: number, y: number): string => `${x},${y}`;
  const dist = new Map<string, number>();
  const [sx, sy] = p.startCell;
  const [gx, gy] = p.goalCell;
  dist.set(key(sx, sy), 0);
  // simple array-based PQ; grids here are small
  const pq: Array<{ x: number; y: number; d: number }> = [{ x: sx, y: sy, d: 0 }];
  while (pq.length > 0) {
    let bi = 0;
    for (let i = 1; i < pq.length; i++) if (pq[i]!.d < pq[bi]!.d) bi = i;
    const cur = pq.splice(bi, 1)[0]!;
    if (cur.x === gx && cur.y === gy) return cur.d;
    if (cur.d > (dist.get(key(cur.x, cur.y)) ?? Infinity)) continue;
    for (const [dx, dy] of DIRS) {
      const nx = cur.x + dx;
      const ny = cur.y + dy;
      if (nx < 0 || ny < 0 || nx >= p.cols || ny >= p.rows) continue;
      if (p.blocked(nx, ny)) continue;
      const nd = cur.d + (dx !== 0 && dy !== 0 ? Math.SQRT2 : 1);
      if (nd < (dist.get(key(nx, ny)) ?? Infinity) - 1e-12) {
        dist.set(key(nx, ny), nd);
        pq.push({ x: nx, y: ny, d: nd });
      }
    }
  }
  return Infinity;
}

export const SIMPLE_OPEN = ['S......', '.......', '.......', '......G'];

export const WITH_WALL = [
  'S.........',
  '.....#....',
  '.....#....',
  '.....#....',
  '.....#....',
  '.....#....',
  '.........G',
];

export const POCKET = [
  '..........',
  '.S........',
  '.####.....',
  '.#..#.....',
  '.#..#.....',
  '.####.....',
  '........G.',
  '..........',
];

export const UNREACHABLE = [
  'S...#...',
  '....#...',
  '....#..G',
  '....#...',
];
