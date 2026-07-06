// Best one-lap stats per model at its optimal control-feedforward setting.
// Feedforward helps only when the plan is model-faithful, so the winning config
// is: kinematic OFF, v2 OFF, v3 ON (see plan-vs-plant.mts / feedforward-compare.mts).
// Runs each single-car on the open course until it completes laps (or a time
// cap) and prints the full one-lap driving-quality stats.
//
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/best-config-bench.mts [secsCap] [budgetMs]
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
// Unique, live-tailable progress log (stdout is buffered until exit).
const { path: logPath, log } = openRunLog('best-config-bench');
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const secsCap = Number(process.argv[2] ?? 90);
const budgetMs = Number(process.argv[3] ?? 12000);
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const sp = process.env.KINOCAT_SPEED_PROFILE === '1';

// (label, entry factory, feedforward) — each model at its BEST feedforward.
const CONFIGS = [
  { label: 'kinematic (FF off)', mk: () => kinematicEntry('kin'), ff: false },
  { label: 'v2 learned (FF off)', mk: () => v2Entry('v2', modelFromJson(readModel('v2-default.json'))), ff: false },
  { label: 'v3 learned (FF on)', mk: () => v3Entry('v3', v3FromJson(readModel('v3-default.json'))), ff: true },
] as const;

for (const cfg of CONFIGS) {
  const tuning: Partial<RaceTuning> = {
    plannerBudgetMs: budgetMs, tracker: 'mpc',
    analyticDriveThrough: dt, enableSpeedProfile: sp, controlFeedforward: cfg.ff,
  };
  const scenario = await createRaceScenario({
    entries: [cfg.mk()], targetLaps: 3, syncHold: false, course: buildRaceCourse('open'), tuning,
  });
  const wall0 = performance.now();
  log(`\n[${cfg.label}] running (cap ${secsCap}s sim, budget ${budgetMs}ms)…`);
  let nextBeat = 15;
  // Stop once the target laps are in, or the time cap hits. Target 1 lap keeps
  // wall time tractable at the generous budget; the launch is part of it (same
  // basis as the feedforward-compare best-lap figure).
  const targetLaps = Number(process.env.KINOCAT_BENCH_LAPS ?? 1);
  while (scenario.simTime() < secsCap) {
    const r = scenario.tick();
    if (r.cars[0]!.laps.length >= targetLaps || r.allFinished) break;
    if (scenario.simTime() >= nextBeat) {
      const c0 = scenario.status()[0]!;
      log(`  … sim ${scenario.simTime().toFixed(0)}s  laps ${c0.laps.length}  mean ${c0.quality.meanSpeed.toFixed(1)}  wall ${((performance.now() - wall0) / 1000).toFixed(0)}s`);
      nextBeat += 15;
    }
  }
  const wall = (performance.now() - wall0) / 1000;
  const s = scenario.status()[0]!;
  const q = s.quality;
  const lapDurs = s.laps.map((l) => l.duration);
  const best = lapDurs.length ? Math.min(...lapDurs) : 0;
  scenario.dispose();
  log(`\n=== ${cfg.label} — open course, ${budgetMs}ms budget${dt ? ', reprice' : ''}${sp ? ', speedprofile' : ''} ===`);
  log(`  laps completed      ${s.laps.length}   (sim ${secsCap}s cap, wall ${wall.toFixed(0)}s)`);
  log(`  best lap            ${best ? best.toFixed(2) + ' s' : '— (no full lap)'}`);
  log(`  lap durations       ${lapDurs.map((d) => d.toFixed(1)).join(', ') || '—'}`);
  log(`  mean speed          ${q.meanSpeed.toFixed(2)} m/s`);
  log(`  predErrRms          ${s.diagnostics.predErrorRms.toFixed(2)} m   (plan vs actual, lower = truer tracking)`);
  log(`  plan churn (mean)   ${q.planChurnMean.toFixed(2)} m`);
  log(`  time stopped        ${q.timeStopped.toFixed(2)} s`);
  log(`  time reversing      ${q.timeReversing.toFixed(2)} s`);
  log(`  recoveries          ${q.recoveryCount}`);
  log(`  distance travelled  ${q.distanceTravelled.toFixed(0)} m`);
  log(`  g-g utilization     mean ${(q.ggMeanUtil * 100).toFixed(0)}%  peak ${(q.ggPeakUtil * 100).toFixed(0)}%`);
  log(`  MPPI solve ms/tick  ${s.diagnostics.mpcSolveMsAvg.toFixed(1)}`);
}

log(`\nDone. Full log: ${logPath}`);
