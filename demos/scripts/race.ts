// Headless race benchmark CLI — `pnpm run race`.
//
// Loads the preloaded `models/v2-default.json` (or any caller-supplied
// model files), builds matching race entries, runs the headless race
// scenario, and prints a comparison table. Exits 0 iff every entry
// completed all laps and v2-default's average lap is ≤ the kinematic
// baseline's. Used as the Phase 3 acceptance gate.

import { readFileSync, existsSync, mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { resolve, basename, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  runHeadlessRace,
  kinematicEntry,
  parametricOnlyEntry,
  v2Entry,
  type RaceEntry,
  type RaceResult,
  type CarTrace,
} from '../app/lib/headless-race';
import { modelFromJson } from '../app/lib/v2-model-file';
import type { PersistedV2Model } from '../app/lib/v2-model-persistence';

function loadModelEntry(path: string, displayName?: string): RaceEntry {
  const text = readFileSync(path, 'utf-8');
  const payload = JSON.parse(text) as PersistedV2Model;
  const model = modelFromJson(payload);
  return v2Entry(displayName ?? basename(path).replace(/\.json$/, ''), model);
}

function fmt(s: number): string {
  return Number.isFinite(s) ? `${s.toFixed(2)}s` : '---';
}

function tableLine(
  cols: (string | number)[],
  widths: number[],
): string {
  return cols.map((c, i) => String(c).padEnd(widths[i] ?? 0)).join('  ');
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    allowPositionals: true,
    options: {
      models: { type: 'string', default: 'demos/public/models/v2-default.json' },
      laps: { type: 'string', default: '3' },
      seed: { type: 'string', default: '42' },
      'max-sim': { type: 'string', default: '180' },
      json: { type: 'string' },
      ledger: { type: 'string' },
      'dump-replans': { type: 'string' },
      'debug-dir': { type: 'string' },
      quick: { type: 'boolean', default: false },
      'no-kinematic': { type: 'boolean', default: false },
      'no-parametric': { type: 'boolean', default: false },
      tracker: { type: 'string', default: 'pure-pursuit' },
      deterministic: { type: 'boolean', default: false },
      'steer-rate': { type: 'string' },
      'lat-debounce': { type: 'string' },
      'consistency': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run race -- [--models=a.json,b.json,...]
                      [--laps=N] [--seed=N] [--max-sim=N]
                      [--json=path] [--ledger=dir] [--quick]
                      [--no-kinematic] [--no-parametric]
                      [--tracker=pure-pursuit|mpc]
                      [--dump-replans=path.json]
                      [--debug-dir=path] (auto-named timestamped debug bundle)
`);
    return;
  }

  const laps = Number(values.laps);
  const maxSimTime = Number(values['max-sim']);
  const quick = values.quick === true;

  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const entries: RaceEntry[] = [];
  const modelPaths = String(values.models).split(',').map((s) => s.trim()).filter(Boolean);
  for (const p of modelPaths) {
    const abs = isAbsolute(p) ? p : resolve(repoRoot, p);
    if (!existsSync(abs)) {
      process.stderr.write(`warning: model file not found: ${abs} — skipping\n`);
      continue;
    }
    entries.push(loadModelEntry(abs));
  }
  if (!values['no-kinematic']) entries.push(kinematicEntry('kinematic'));
  if (!values['no-parametric']) entries.push(parametricOnlyEntry('parametric-only'));
  if (entries.length === 0) {
    process.stderr.write('No race entries to run.\n');
    process.exit(2);
  }

  const targetLaps = quick ? 1 : laps;
  const tracker = (values.tracker === 'mpc' ? 'mpc' : 'pure-pursuit') as 'pure-pursuit' | 'mpc';
  process.stdout.write(`race-primitives benchmark — seed=${values.seed} laps=${targetLaps} tracker=${tracker} dt=1/60\n`);

  // --debug-dir captures per-tick traces (0.1 s stride) plus the
  // structured replan history; output goes to a timestamped
  // sub-directory so multiple runs don't overwrite each other.
  const debugDir = values['debug-dir'] ? String(values['debug-dir']) : null;
  const debugRoot = debugDir
    ? (isAbsolute(debugDir) ? debugDir : resolve(repoRoot, debugDir))
    : null;
  const runStamp = new Date().toISOString().replace(/[:.]/g, '-');
  const runDir = debugRoot ? `${debugRoot}/${runStamp}` : null;
  if (runDir) mkdirSync(runDir, { recursive: true });
  let capturedTraces: CarTrace[] = [];

  const results = await runHeadlessRace({
    entries,
    targetLaps,
    maxSimTime,
    onProgress: (msg) => process.stdout.write(`  · ${msg}\n`),
    progressEverySec: 10,
    tuning: {
      tracker,
      deterministicPlanner: Boolean(values.deterministic),
      // Only set when the flag is provided, otherwise the spread leaves
      // an `undefined` value that OVERRIDES the default in
      // `DEFAULT_TUNING` instead of falling back to it.
      ...(values['steer-rate'] !== undefined ? { maxSteerRateRadPerSec: Number(values['steer-rate']) } : {}),
      ...(values['lat-debounce'] !== undefined ? { lateralErrorReplanMinTicks: Number(values['lat-debounce']) } : {}),
      ...(values['consistency'] !== undefined ? { consistencyWeight: Number(values['consistency']) } : {}),
    },
    traceEverySec: runDir ? 0.1 : undefined,
    onTrace: runDir ? (t) => { capturedTraces = t; } : undefined,
  });

  // Print table.
  const widths = [22, 7, 7, 7, 7, 7, 7, 9];
  const header = ['model', 'status', 'laps', 'lap1', 'lap2', 'lap3', 'best', 'avg'];
  process.stdout.write('\n' + tableLine(header, widths) + '\n');
  for (const r of results) {
    const status = r.finished ? 'OK' : 'DNF';
    const lap1 = r.laps[0] ? fmt(r.laps[0].duration) : '---';
    const lap2 = r.laps[1] ? fmt(r.laps[1].duration) : '---';
    const lap3 = r.laps[2] ? fmt(r.laps[2].duration) : '---';
    process.stdout.write(
      tableLine(
        [r.name, status, `${r.laps.length}/${targetLaps}`, lap1, lap2, lap3, fmt(r.best), fmt(r.avg)],
        widths,
      ) + '\n',
    );
  }
  process.stdout.write('\n');

  // Per-car diagnostics — surface the planner / replan / off-track signals
  // the harness collected, so debugging "why is car X slow?" doesn't need
  // a second run. Replan reasons isolate which trigger dominates (mostly
  // cadence is healthy; mostly lateral-error or failure-retry means the
  // controller and planner are fighting).
  const dwidths = [22, 9, 12, 18, 11, 9, 11, 24];
  const dheader = ['model', 'off-track', 'replan(ok/total)', 'planner ms (mean/max)', 'deadlineHit', 'predErrRMS', 'sharpSteer', 'reasons (cad/lat/wp/fail)'];
  process.stdout.write(tableLine(dheader, dwidths) + '\n');
  for (const r of results) {
    const reasons = `${r.replanReasonCounts.cadence}/${r.replanReasonCounts['lateral-error']}/${r.replanReasonCounts['waypoint-advance']}/${r.replanReasonCounts['failure-retry']}`;
    process.stdout.write(
      tableLine(
        [
          r.name,
          String(r.offTrackEvents),
          `${r.successfulReplans}/${r.totalReplans}`,
          `${r.plannerMsMean.toFixed(1)}/${r.plannerMsMax.toFixed(1)}`,
          String(r.plannerDeadlineHits),
          r.predErrorRms.toFixed(2),
          String(r.sharpSteerTicks),
          reasons,
        ],
        dwidths,
      ) + '\n',
    );
  }
  process.stdout.write('\n');

  // Plan & execution health table. Domain-agnostic metrics that answer
  // "is the planner emitting clean plans?" and "is the tracker
  // executing them faithfully?" — works for racing, parking, anything.
  // cuspsKept = real gear changes the tracker acted on (forward↔reverse
  //   transitions in the plan).
  // cuspsRaw  = pre-filter sign-flip count (smoother artifacts +
  //   real cusps). cuspsRaw ≫ cuspsKept means the smoother is
  //   manufacturing fake cusps that the detector then absorbs.
  // infCurv% = fraction of plan samples whose |κ|·v² exceeds the
  //   tracker's lateral-accel cap. Planner asking for impossible
  //   cornering grip.
  // infAcc%  = fraction of plan samples where |Δv|/Δt exceeds accel
  //   caps. Planner asking for an instantaneous speed jump.
  // spdErrP95 = p95 of |targetSpeed - actualSpeed| (m/s). Tracker
  //   following the plan's speed channel.
  // latErrP95 = p95 of lateral error to plan (m). Tracker following
  //   plan geometry.
  // infNowTk  = tick count where the controller was asked to do
  //   the impossible (|κ|·v² > cap) right at execution time.
  // lapCv     = std/mean of completed lap times — direct measure of
  //   race-to-race lap consistency.
  const hwidths = [22, 9, 9, 8, 8, 10, 10, 8, 6];
  const hheader = ['model', 'cuspsRaw', 'cuspsKept', 'infCurv%', 'infAcc%', 'spdErrP95', 'latErrP95', 'infNowTk', 'lapCv'];
  process.stdout.write(tableLine(hheader, hwidths) + '\n');
  for (const r of results) {
    const total = Math.max(1, r.planSamplesTotal);
    const infCurvPct = (100 * r.infeasibleCurvatureSamples / total).toFixed(1);
    const infAccPct = (100 * r.infeasibleAccelSamples / total).toFixed(1);
    process.stdout.write(
      tableLine(
        [
          r.name,
          String(r.cuspsRawTotal),
          String(r.cuspsKeptTotal),
          infCurvPct,
          infAccPct,
          r.speedErrP95.toFixed(2),
          r.lateralErrP95.toFixed(2),
          String(r.infeasibleNowTicks),
          r.lapTimeCv.toFixed(3),
        ],
        hwidths,
      ) + '\n',
    );
  }
  process.stdout.write('\n');

  // Comparison commentary.
  const v2 = results.find((r) => r.name.includes('v2') || modelPaths.some((p) => r.name === basename(p).replace(/\.json$/, '')));
  const kin = results.find((r) => r.name === 'kinematic');
  if (v2 && kin && Number.isFinite(v2.avg) && Number.isFinite(kin.avg)) {
    const diff = (kin.avg - v2.avg) / kin.avg * 100;
    if (diff > 0) {
      process.stdout.write(`→ ${v2.name} beats ${kin.name} by ${diff.toFixed(1)}% (avg)\n`);
    } else {
      process.stdout.write(`→ ${v2.name} loses to ${kin.name} by ${(-diff).toFixed(1)}% (avg)\n`);
    }
  }

  // Optional JSON output.
  if (values.json) {
    const arg = String(values.json);
    const jsonPath = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    writeFileSync(jsonPath, JSON.stringify({ seed: Number(values.seed), targetLaps, results }, null, 2), 'utf-8');
    process.stdout.write(`wrote ${jsonPath}\n`);
  }

  // Auto-named debug bundle.
  if (runDir) {
    const summary = {
      timestamp: runStamp,
      seed: Number(values.seed),
      targetLaps,
      tracker,
      maxSimTime,
      results: results.map((r) => ({
        name: r.name,
        finished: r.finished,
        laps: r.laps,
        best: r.best,
        avg: r.avg,
        stddev: r.stddev,
        offTrackEvents: r.offTrackEvents,
        predErrorRms: r.predErrorRms,
        totalReplans: r.totalReplans,
        successfulReplans: r.successfulReplans,
        replanReasonCounts: r.replanReasonCounts,
        plannerMsMean: r.plannerMsMean,
        plannerMsMax: r.plannerMsMax,
        plannerDeadlineHits: r.plannerDeadlineHits,
        sharpSteerTicks: r.sharpSteerTicks,
        cuspsRawTotal: r.cuspsRawTotal,
        cuspsKeptTotal: r.cuspsKeptTotal,
        infeasibleCurvatureSamples: r.infeasibleCurvatureSamples,
        infeasibleAccelSamples: r.infeasibleAccelSamples,
        planSamplesTotal: r.planSamplesTotal,
        speedErrP95: r.speedErrP95,
        lateralErrP95: r.lateralErrP95,
        infeasibleNowTicks: r.infeasibleNowTicks,
        lapTimeCv: r.lapTimeCv,
        perLapOffTrackTicks: r.perLapOffTrackTicks,
        perLapReplanCounts: r.perLapReplanCounts,
      })),
    };
    writeFileSync(`${runDir}/summary.json`, JSON.stringify(summary, null, 2), 'utf-8');
    writeFileSync(
      `${runDir}/replan-history.json`,
      JSON.stringify(
        results.map((r) => ({
          name: r.name,
          replanHistory: r.replanHistory,
        })),
        null,
        2,
      ),
      'utf-8',
    );
    if (capturedTraces.length > 0) {
      writeFileSync(`${runDir}/traces.json`, JSON.stringify(capturedTraces, null, 2), 'utf-8');
    }
    process.stdout.write(`wrote debug bundle to ${runDir}\n`);
  }

  // Optional replan-history dump — full per-replan structured records
  // (ring buffer, max 30 most recent per car) for offline analysis of
  // "what was the planner thinking at this point in the race?".
  if (values['dump-replans']) {
    const arg = String(values['dump-replans']);
    const dumpPath = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    const dump = results.map((r) => ({
      name: r.name,
      replanReasonCounts: r.replanReasonCounts,
      plannerMsMean: r.plannerMsMean,
      plannerMsMax: r.plannerMsMax,
      plannerDeadlineHits: r.plannerDeadlineHits,
      sharpSteerTicks: r.sharpSteerTicks,
      replanHistory: r.replanHistory,
    }));
    writeFileSync(dumpPath, JSON.stringify(dump, null, 2), 'utf-8');
    process.stdout.write(`wrote ${dumpPath} (replan history)\n`);
  }

  // Optional ledger append.
  if (values.ledger) {
    const arg = String(values.ledger);
    const dir = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    mkdirSync(dir, { recursive: true });
    const line = JSON.stringify({
      at: Date.now(),
      seed: Number(values.seed),
      targetLaps,
      results: results.map((r) => ({
        name: r.name,
        finished: r.finished,
        best: r.best,
        avg: r.avg,
        offTrackEvents: r.offTrackEvents,
      })),
    });
    appendFileSync(`${dir}/results.jsonl`, line + '\n');
    process.stdout.write(`appended to ${dir}/results.jsonl\n`);
  }

  // Exit code: pass iff AT LEAST ONE entry finished (so CI doesn't trip
  // on a slow runner where one baseline DNFs). For v2 < kinematic — that's
  // a quality signal, not a contract — never fails the run. Local users
  // chasing parity can read the printed table.
  const anyFinished = results.some((r) => r.finished);
  process.exit(anyFinished ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`race failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
  process.exit(1);
});
