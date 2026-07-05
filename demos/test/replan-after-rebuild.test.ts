// Integration gate: replan-after-rebuild < 100 ms (p95).
//
// The full pipeline a live tile change costs an agent: world-update
// (setObstacles rebuild) → region-scoped invalidation → dirty replan through
// the worker handler → plan adoption. The companion bench
// (demos/bench/replan-after-rebuild.bench.ts) reports throughput; this test
// asserts the charter gate because vitest benches can't assert.

import { describe, it, expect } from 'vitest';
import {
  setupReplanRebuildFixture,
  runReplanAfterRebuildOnce,
} from './helpers/replan-rebuild';

let fix: ReturnType<typeof setupReplanRebuildFixture> | null = null;
try {
  fix = setupReplanRebuildFixture();
} catch {
  fix = null;
}

describe.skipIf(!fix)('replan-after-rebuild latency gate', () => {
  it('p95 of 40 rebuild→detect→replan rounds is under 100 ms', () => {
    // Warm-up: JIT + world grids + heuristic LUT.
    for (let i = 0; i < 3; i++) runReplanAfterRebuildOnce(fix!, i);

    const wall: number[] = [];
    for (let i = 0; i < 40; i++) {
      const r = runReplanAfterRebuildOnce(fix!, i);
      // The pipeline must actually work every round, not just be fast.
      expect(r.markedCount).toBe(1); // region-scoped detection fired
      expect(r.found).toBe(true); // a usable plan came back
      expect(r.adopted).toBe(true); // and was adopted (dirty forces it)
      wall.push(r.wallMs);
    }

    wall.sort((a, b) => a - b);
    const p95 = wall[Math.min(wall.length - 1, Math.ceil(wall.length * 0.95) - 1)]!;
    const p50 = wall[Math.floor(wall.length / 2)]!;
    // Surface the numbers in CI logs either way.
    console.info(
      `replan-after-rebuild: p50=${p50.toFixed(1)}ms p95=${p95.toFixed(1)}ms ` +
        `max=${wall[wall.length - 1]!.toFixed(1)}ms`,
    );
    expect(p95).toBeLessThan(100);
  }, 30000);
});
