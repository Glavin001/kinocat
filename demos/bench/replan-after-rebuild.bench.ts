// Bench for the replan-after-rebuild pipeline (charter gate: < 100 ms p95;
// the assertion lives in demos/test/replan-after-rebuild.test.ts — benches
// can't assert). Rows:
//
//   • replan only            — the planner cost alone (no world mutation)
//   • rebuild → … → replan   — the full pipeline: setObstacles (AABB +
//                              spatial-index + heuristic-grid rebuild),
//                              region-scoped invalidation, dirty replan,
//                              adoption
//
// The delta between the rows is what a live tile change costs on top of an
// ordinary replan. Blockers alternate every iteration so setObstacles always
// sees a real data change.

import { bench, describe } from 'vitest';
import {
  setupReplanRebuildFixture,
  runReplanAfterRebuildOnce,
  REPLAN_DEADLINE_MS,
} from '../test/helpers/replan-rebuild';
import { handlePlanMessage } from 'kinocat/worker';

let fix: ReturnType<typeof setupReplanRebuildFixture> | null = null;
try {
  fix = setupReplanRebuildFixture();
} catch {
  fix = null;
}

let iter = 0;

describe.skipIf(!fix)('replan-after-rebuild', () => {
  bench('replan only (no rebuild)', () => {
    handlePlanMessage(
      {
        type: 'plan',
        reqId: iter++,
        npcId: 'robber',
        start: fix!.robber,
        goal: fix!.goal,
        obstacles: fix!.copDescriptors,
        deadlineMs: REPLAN_DEADLINE_MS,
        maxExpansions: 25000,
      },
      () => {},
    );
  });

  bench('rebuild → world-update → dirty → replan (full pipeline)', () => {
    runReplanAfterRebuildOnce(fix!, iter++);
  });
});
