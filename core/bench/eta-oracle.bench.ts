// ETA/feasibility oracle (charter gate: < 2 ms per steady-state query; the
// assertion lives in core/test/predict/eta-oracle.test.ts — benches can't
// assert). Rows separate the two cost regimes so a tactical layer can budget
// them independently:
//
//   • field build   — one multi-source Dijkstra per (region, world revision);
//                     amortised via the oracle's LRU / `prebuild`
//   • warm query    — LRU hit + O(1) field lookup + Region.costToGo floor
//
// Fixtures: a real navcat CHF world (fine grid, ~10k spans) and an
// InMemoryNavWorld with obstacles (coarse adaptive grid).

import { bench, describe } from 'vitest';
import { navWorldFromTriangleMesh } from '../src/adapters/navcat/index';
import { InMemoryNavWorld, type NavPolygon } from '../src/environment/nav-world';
import { createEtaOracle } from '../src/predict/eta-oracle';
import { near } from '../src/scenario/regions';
import type { ScenarioState } from '../src/scenario/types';

const AGENT = { maxSpeed: 10 };

function st(x: number, z: number): ScenarioState {
  return { x, z, heading: 0, speed: 0, t: 0 };
}

// --- navcat CHF fixture (60×40 plane, cellSize 0.5 → ~9.6k spans) ---
function planeMesh(w: number, d: number) {
  return {
    positions: [0, 0, 0, w, 0, 0, w, 0, d, 0, 0, d],
    indices: [0, 3, 2, 0, 2, 1],
  };
}
let chfWorld: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = planeMesh(60, 40);
  chfWorld = navWorldFromTriangleMesh(
    m.positions,
    m.indices,
    { cellSize: 0.5 },
    { clearanceField: true },
  );
} catch {
  chfWorld = null;
}

// --- InMemoryNavWorld fixture (walled floor, adaptive coarse grid) ---
const FLOOR: NavPolygon = {
  id: 1,
  y: 0,
  ring: [
    [0, 0],
    [60, 0],
    [60, 40],
    [0, 40],
  ],
};
const memWorld = new InMemoryNavWorld(
  [FLOOR],
  [
    [
      [28, 0],
      [32, 0],
      [32, 28],
      [28, 28],
    ],
  ],
);

const REGION = near({ x: 50, z: 20 }, 3);

describe.skipIf(!chfWorld)('eta-oracle: navcat CHF world', () => {
  const world = chfWorld!.world;
  const oracle = createEtaOracle(world, AGENT);
  oracle.prebuild(REGION);

  bench('field build (per region × revision)', () => {
    // Bypass the LRU: build the field directly.
    world.buildRegionLowerBound!((x, z) => Math.hypot(x - 50, z - 20) <= 3);
  });

  let i = 0;
  bench('warm eta query', () => {
    oracle.eta(st(2 + (i++ % 50), 10), REGION);
  });
});

describe('eta-oracle: InMemoryNavWorld', () => {
  const oracle = createEtaOracle(memWorld, AGENT);
  oracle.prebuild(REGION);

  bench('field build (per region × revision)', () => {
    memWorld.buildRegionLowerBound((x, z) => Math.hypot(x - 50, z - 20) <= 3);
  });

  let i = 0;
  bench('warm eta query', () => {
    oracle.eta(st(2 + (i++ % 50), 10), REGION);
  });
});
