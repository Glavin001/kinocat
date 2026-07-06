// Per-model timing distribution: where does the wall-clock go?
// Runs a short closed-loop segment and reports the distribution of planner
// replan times and the MPPI solve cost, so "why is v3 slow" is a measurement.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/perf-profile.mts [secs]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const models = (process.argv[3] ? [process.argv[3]] : ['kin', 'v2', 'v3']) as ('kin' | 'v2' | 'v3')[];
const secs = Number(process.argv[2] ?? 25);
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const sp = process.env.KINOCAT_SPEED_PROFILE === '1';

const pct = (arr: number[], p: number): number => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
};

for (const which of models) {
  const entry = which === 'v3' ? v3Entry('v3', v3FromJson(readModel('v3-default.json')))
    : which === 'v2' ? v2Entry('v2', modelFromJson(readModel('v2-default.json')))
      : kinematicEntry('kin');
  const scenario = await createRaceScenario({
    entries: [entry], targetLaps: 3, syncHold: false, course: buildRaceCourse('open'),
    tuning: { plannerBudgetMs: 12000, tracker: 'mpc', analyticDriveThrough: dt, enableSpeedProfile: sp },
  });
  const replanMs: number[] = [];
  let lastReplans = 0;
  let lastSolveTotal = 0, lastSolveCount = 0;
  const solveMs: number[] = [];
  const wall0 = performance.now();
  while (scenario.simTime() < secs) {
    const r = scenario.tick();
    const c = r.cars[0]!;
    const d = c.diagnostics;
    if (d.totalReplans > lastReplans) { lastReplans = d.totalReplans; replanMs.push(d.lastReplanMs); }
    const m = c.metrics;
    if (m.mpcSolveCount > lastSolveCount) {
      const dCount = m.mpcSolveCount - lastSolveCount;
      const dTotal = m.mpcSolveMsAvg * m.mpcSolveCount - lastSolveTotal;
      if (dCount > 0) solveMs.push(dTotal / dCount);
      lastSolveCount = m.mpcSolveCount; lastSolveTotal = m.mpcSolveMsAvg * m.mpcSolveCount;
    }
    if (r.allFinished) break;
  }
  const wall = performance.now() - wall0;
  const s = scenario.status()[0]!;
  const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
  console.log(`\n=== ${which} (${secs}s sim, ${gen ? 'gen ' : ''}${dt ? 'reprice ' : ''}${sp ? 'speedprofile' : ''}) ===`);
  console.log(`  wall=${(wall / 1000).toFixed(1)}s  laps=${s.laps.length}  meanSpd=${s.quality.meanSpeed.toFixed(1)}`);
  console.log(`  REPLAN ms: n=${replanMs.length} min=${Math.min(...replanMs).toFixed(0)} med=${pct(replanMs, 0.5).toFixed(0)} p90=${pct(replanMs, 0.9).toFixed(0)} p99=${pct(replanMs, 0.99).toFixed(0)} max=${Math.max(...replanMs).toFixed(0)} | totalPlanTime=${(sum(replanMs) / 1000).toFixed(1)}s`);
  console.log(`  MPPI solve ms/tick: n=${solveMs.length} min=${Math.min(...solveMs).toFixed(1)} med=${pct(solveMs, 0.5).toFixed(1)} p90=${pct(solveMs, 0.9).toFixed(1)} max=${Math.max(...solveMs).toFixed(1)} | totalSolveTime=${(sum(solveMs) / 1000).toFixed(1)}s`);
  scenario.dispose();
}
