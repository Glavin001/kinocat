// Best one-lap stats per model at ITS OWN best CONFIGURATION (not a fixed
// tracker). The three models want different control stacks:
//   - kinematic: the model is delusional, so a model-in-the-loop tracker (MPPI)
//     over-drives every corner and wedges. Its home is pure-pursuit — geometric,
//     no model — where it's fast (the main-branch config).
//   - v2: decent but its MPPI rollout under-predicts accel + is slow; try both
//     pure-pursuit and MPPI-without-feedforward and keep whichever laps best.
//   - v3: the accurate model — the whole point is MPPI + control feedforward,
//     which turns its fidelity into the driven line.
// Runs each config single-car on the open course to a lap (or a time cap) and
// prints the full one-lap driving-quality stats + a winner-per-model summary.
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
const variant = (process.argv[4] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const sp = process.env.KINOCAT_SPEED_PROFILE === '1';

type Tracker = 'pure-pursuit' | 'mpc';
// (model, label, entry factory, tracker, feedforward). Multiple candidate
// configs per model where the best isn't obvious (v2); the summary picks the
// fastest per model.
const CONFIGS: { model: string; label: string; mk: () => ReturnType<typeof kinematicEntry>; tracker: Tracker; ff: boolean }[] = [
  { model: 'kinematic', label: 'kinematic · pure-pursuit', mk: () => kinematicEntry('kin'), tracker: 'pure-pursuit', ff: false },
  { model: 'v2', label: 'v2 · pure-pursuit', mk: () => v2Entry('v2', modelFromJson(readModel('v2-default.json'))), tracker: 'pure-pursuit', ff: false },
  { model: 'v2', label: 'v2 · MPPI (no FF)', mk: () => v2Entry('v2', modelFromJson(readModel('v2-default.json'))), tracker: 'mpc', ff: false },
  { model: 'v3', label: 'v3 · MPPI + feedforward', mk: () => v3Entry('v3', v3FromJson(readModel('v3-default.json'))), tracker: 'mpc', ff: true },
];

interface Result { model: string; label: string; laps: number; best: number; meanSpeed: number; predErr: number }
const results: Result[] = [];

for (const cfg of CONFIGS) {
  const tuning: Partial<RaceTuning> = {
    plannerBudgetMs: budgetMs, tracker: cfg.tracker,
    analyticDriveThrough: dt, enableSpeedProfile: sp, controlFeedforward: cfg.ff,
  };
  const scenario = await createRaceScenario({
    entries: [cfg.mk()], targetLaps: 3, syncHold: false, course: buildRaceCourse(variant), tuning,
  });
  const wall0 = performance.now();
  log(`\n[${cfg.label}] running (${cfg.tracker}${cfg.ff ? ' +FF' : ''}, cap ${secsCap}s sim, budget ${budgetMs}ms)…`);
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
  results.push({ model: cfg.model, label: cfg.label, laps: s.laps.length, best, meanSpeed: q.meanSpeed, predErr: s.diagnostics.predErrorRms });
  log(`\n=== ${cfg.label} — ${variant} course, ${budgetMs}ms budget${dt ? ', reprice' : ''}${sp ? ', speedprofile' : ''} ===`);
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

// Winner per model (fastest completed lap; a config with no full lap loses to
// any that lapped).
log(`\n======== BEST CONFIG PER MODEL (${variant} course) ========`);
const rank = (r: Result) => (r.laps > 0 && r.best > 0 ? r.best : Infinity);
for (const model of ['kinematic', 'v2', 'v3']) {
  const cands = results.filter((r) => r.model === model).sort((a, b) => rank(a) - rank(b));
  if (!cands.length) continue;
  const win = cands[0]!;
  const bestStr = rank(win) === Infinity ? 'no clean lap' : `${win.best.toFixed(2)}s best lap`;
  log(`  ${model.padEnd(10)} → ${win.label.padEnd(26)} ${bestStr}  (mean ${win.meanSpeed.toFixed(1)} m/s, predErr ${win.predErr.toFixed(2)} m)`);
  for (const alt of cands.slice(1)) {
    const altStr = rank(alt) === Infinity ? 'no clean lap' : `${alt.best.toFixed(2)}s`;
    log(`             alt: ${alt.label.padEnd(26)} ${altStr}`);
  }
}
log(`\nDone. Full log: ${logPath}`);
