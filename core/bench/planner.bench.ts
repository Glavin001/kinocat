import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { R2Environment } from '../src/environment/r2-environment';

// Not a CI gate — informational throughput (spec §15.5/§15.7).
const blocked = (cx: number, cy: number): boolean =>
  cx === 20 && cy >= 0 && cy < 30; // a wall with a gap at the bottom

const env = new R2Environment({
  step: 1,
  blocked,
  bounds: { minCx: 0, maxCx: 50, minCy: 0, maxCy: 40 },
});
const start = { x: 1, y: 1 };
const goal = { x: 48, y: 38 };

describe('IGHA* throughput (R2 50x40 with wall)', () => {
  bench('plan to optimality', () => {
    plan({ start, goal, environment: env }, Infinity);
  });

  bench('plan with 2000-expansion anytime budget', () => {
    plan({ start, goal, environment: env, options: { maxExpansions: 2000 } }, Infinity);
  });
});
