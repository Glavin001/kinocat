// One-off diagnostic probe: does tightening `plannerBudgetMs` reproduce
// the web demo's "parking-reverse-perp drives the wrong way" failure?
//
// Hypothesis (user-facing message in PR claude/dreamy-faraday-Ohjjn):
// the browser's slower JS execution makes the planner hit its 500 ms
// budget more often, so it returns partial / forward-only plans that
// drive the chassis away from the goal. The CLI's faster JS doesn't
// hit the deadline as often, hence the disconnect between bench PASS
// and browser failure.
//
// This probe runs the SAME parking-reverse-perp scenario at a range of
// budgets and reports, per budget:
//   - did the chassis park within the bench/HUD acceptance criterion?
//   - terminal pose error vs goal
//   - fraction of replans where the planner hit its deadline
//   - mean plan-search ms (sanity check)
// If success degrades as budget tightens, the hypothesis is confirmed.

import { createRaceScenario } from '../app/lib/race-scenario';
import {
  buildParkingScenario,
  parkingLibrary,
  checkParkingGoal,
} from '../app/lib/parking-scenarios';

async function probeBudget(budgetMs: number): Promise<void> {
  const s = buildParkingScenario('reverse-perp');
  const course = {
    bounds: { x0: s.bounds.x0, x1: s.bounds.x1, z0: s.bounds.z0, z1: s.bounds.z1 },
    polygons: s.polygons,
    obstacles: s.obstacles,
    waypoints: [{ ...s.goal, speed: 0, t: 0 }],
    spawn: { ...s.spawn, speed: 0, t: 0 },
  };
  const scenario = await createRaceScenario({
    entries: [{ name: 'kinematic', lib: parkingLibrary() }],
    targetLaps: 1,
    syncHold: false,
    offTrackRecovery: 'none',
    tuning: {
      cruiseSpeed: 2,
      goalTolerance: 0.4,
      arriveRadius: 0.6,
      plannerPosCell: 0.3,
      plannerHeadingBuckets: 36,
      plannerGoalRadius: 0.35,
      plannerGoalHeadingTol: 0.2,
      plannerBudgetMs: budgetMs,
      plannerMaxExpansions: 80_000,
      mpcWTerminalPosition: 50,
      mpcWTerminalSpeed: 30,
    },
    course,
  });

  const MAX_SIM = 40;
  while (scenario.simTime() < MAX_SIM) {
    scenario.tick();
    const st = scenario.status()[0]!;
    if (checkParkingGoal(st.state, s.goal).passed) break;
  }
  const st = scenario.status()[0]!;
  const sim = scenario.simTime();
  const check = checkParkingGoal(st.state, s.goal);
  scenario.dispose();
  const total = st.diagnostics.totalReplans;
  const deadlineHits = st.diagnostics.plannerDeadlineHitsTotal;
  const meanMs =
    total > 0 ? st.diagnostics.plannerMsTotal / total : 0;
  const failPart = !check.posOk
    ? `pos ${check.posM.toFixed(2)}m`
    : !check.hdgOk
      ? `hdg ${(check.hdgRad * 180 / Math.PI).toFixed(1)}°`
      : !check.spdOk
        ? `spd ${check.spdMS.toFixed(2)}m/s`
        : 'sim timeout';
  const verdict = check.passed && sim < MAX_SIM ? 'PARKED' : `FAIL (${failPart})`;
  // eslint-disable-next-line no-console
  console.log(
    `budget=${String(budgetMs).padStart(5)}ms  sim=${sim.toFixed(2).padStart(6)}s  ` +
      `replans=${String(total).padStart(3)}  deadlineHits=${String(deadlineHits).padStart(3)}  ` +
      `meanPlanMs=${meanMs.toFixed(0).padStart(4)}  ` +
      `terminal: pos=${check.posM.toFixed(2)}m hdg=${(check.hdgRad * 180 / Math.PI).toFixed(1)}° ` +
      `|v|=${check.spdMS.toFixed(2)}  → ${verdict}`,
  );
}

async function main(): Promise<void> {
  // Sweep from generous to tight. Browser's observed plan-ms was ~500-510;
  // CLI's typical is ~50-200. Probing both regimes (plus extremes) shows
  // the threshold where success breaks.
  const budgets = [2000, 500, 200, 100, 50, 20, 10];
  console.log('parking-reverse-perp · planner-budget sweep\n');
  for (const b of budgets) await probeBudget(b);
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error(e);
  process.exit(1);
});
