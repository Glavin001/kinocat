// Real-time planner health probe. The pause-clock benchmarks answer "is the
// plan correct given unlimited time"; this answers "does the planner keep a
// FRESH plan on the car's clock within a real-time budget" — the actual
// browser failure (screenshot: 35% success, 1.6 s plan age, wedged).
//
// Runs one config headless with a FIXED planner budget (no pause-clock) and
// reports the health signals: planner success rate, plan staleness (age),
// wedge time, and lap outcome. Use it to A/B the Tier-1 perf changes (budget /
// cadence / commit window / anytime weight) — see docs/v3-realtime-performance-plan.md.
//
// usage:
//   KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 KINOCAT_SPEED_PROFILE=1 \
//     npx tsx scripts/realtime-probe.mts <kin|v2|v3> [open|technical] \
//       [budgetMs=120] [cadenceMs=300] [commitMs=0] [weight=1] [ff=1] [secs=60]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, type RaceTuning } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import { openRunLog } from './lib/run-log';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const { path: logPath, log } = openRunLog('realtime-probe');

const which = (process.argv[2] ?? 'v3') as 'kin' | 'v2' | 'v3';
const variant = (process.argv[3] === 'open' ? 'open' : 'technical') as 'open' | 'technical';
const budgetMs = Number(process.argv[4] ?? 120);
const cadenceMs = Number(process.argv[5] ?? 300);
const commitMs = Number(process.argv[6] ?? 0);
const weight = Number(process.argv[7] ?? 1);
const ff = (process.argv[8] ?? '1') === '1';
const secs = Number(process.argv[9] ?? 60);
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const sp = process.env.KINOCAT_SPEED_PROFILE === '1';

const entry = which === 'v3' ? v3Entry('v3', v3FromJson(readModel('v3-default.json')))
  : which === 'v2' ? v2Entry('v2', modelFromJson(readModel('v2-default.json')))
    : kinematicEntry('kin');

const tuning: Partial<RaceTuning> = {
  plannerBudgetMs: budgetMs,
  replanIntervalMs: cadenceMs,
  commitWindowMs: commitMs,
  plannerWeight: weight,
  tracker: 'mpc',
  controlFeedforward: ff,
  analyticDriveThrough: dt,
  enableSpeedProfile: sp,
};

const pct = (arr: number[], p: number): number => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
};

const scenario = await createRaceScenario({
  entries: [entry], targetLaps: 3, syncHold: false, course: buildRaceCourse(variant), tuning,
});

log(`\n[${which} · ${variant} · FF ${ff ? 'on' : 'off'}] budget=${budgetMs}ms cadence=${cadenceMs}ms commit=${commitMs}ms weight=${weight}`);
const planAges: number[] = [];
const replanMs: number[] = [];
let lastReplans = 0;
const wall0 = performance.now();
let nextBeat = 15;
while (scenario.simTime() < secs) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  planAges.push(c.diagnostics.planAgeMs);
  if (c.diagnostics.totalReplans > lastReplans) {
    lastReplans = c.diagnostics.totalReplans;
    replanMs.push(c.diagnostics.lastReplanMs);
  }
  if (scenario.simTime() >= nextBeat) {
    log(`  … sim ${scenario.simTime().toFixed(0)}s  laps ${c.laps.length}  planAge ${c.diagnostics.planAgeMs.toFixed(0)}ms  wall ${((performance.now() - wall0) / 1000).toFixed(0)}s`);
    nextBeat += 15;
  }
  if (r.allFinished) break;
}
const s = scenario.status()[0]!;
const d = s.diagnostics;
const q = s.quality;
const successRate = d.totalReplans > 0 ? d.successfulReplans / d.totalReplans : 0;
const lapDurs = s.laps.map((l) => l.duration);
scenario.dispose();

log(`\n=== ${which} · ${variant} · FF ${ff ? 'on' : 'off'} — budget ${budgetMs}ms / cadence ${cadenceMs}ms / commit ${commitMs}ms / weight ${weight} ===`);
log(`  PLANNER success     ${(successRate * 100).toFixed(0)}%  (${d.successfulReplans}/${d.totalReplans})   ← the headline`);
log(`  replan ms           med ${pct(replanMs, 0.5).toFixed(0)}  p90 ${pct(replanMs, 0.9).toFixed(0)}  max ${Math.max(0, ...replanMs).toFixed(0)}   (budget ${budgetMs})`);
log(`  plan age            mean ${(planAges.reduce((a, b) => a + b, 0) / planAges.length).toFixed(0)}ms  p90 ${pct(planAges, 0.9).toFixed(0)}ms  max ${Math.max(...planAges).toFixed(0)}ms`);
log(`  laps                ${s.laps.length}   best ${lapDurs.length ? Math.min(...lapDurs).toFixed(1) + 's' : '—'}`);
log(`  mean speed          ${q.meanSpeed.toFixed(2)} m/s`);
log(`  time stopped        ${q.timeStopped.toFixed(2)} s   (wedge proxy)`);
log(`  recoveries          ${q.recoveryCount}`);
log(`  predErrRms          ${d.predErrorRms.toFixed(2)} m`);
log(`\nDone. Full log: ${logPath}`);
