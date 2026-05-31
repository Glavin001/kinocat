// `pnpm exec tsx scripts/analyze-race-debug.ts <run-dir>` — offline
// analyser for the `--debug-dir` bundles produced by `pnpm run race`.
//
// Reads the timestamped run directory (containing summary.json,
// traces.json, replan-history.json) and prints a human-friendly
// "what was each car doing?" report. Designed to answer the questions
// you would otherwise have to dig out by hand:
//
//   - WHERE on the lap is the chassis braking? (controller fighting
//     the plan vs. honest corner-entry braking)
//   - WHERE is the lateral error spiking? (planner-executor mismatch)
//   - WHERE is the steering wrenched to the limit? (sharp-turn red
//     dots on the web demo)
//   - WHICH replans were triggered by what? (cadence is healthy;
//     lateral-error storms mean the controller can't track the plan)
//
// No assertions, no formatting boilerplate — pure introspection on the
// captured JSON. Each report cell is a "this lap segment / this 2-
// second sim window" summary.

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

interface TickSample {
  simTime: number;
  x: number;
  z: number;
  heading: number;
  speed: number;
  steer: number;
  throttle: number;
  brake: number;
  targetSpeed: number;
  loopIndex: number;
  lateralErr: number;
  planNearestIdx: number;
  planLength: number;
}

interface CarTrace {
  name: string;
  samples: TickSample[];
}

interface ReplanSnapshot {
  simTime: number;
  reason: string;
  searchMs: number;
  found: boolean;
  expansions: number;
  generated: number;
  deadlineHit: boolean;
  cost: number;
  improvements: number;
  startState: { x: number; z: number; heading: number; speed: number };
  chassisState: { x: number; z: number; heading: number; speed: number };
  gates: Array<{ x: number; z: number }>;
  loopIndex: number;
  planLength: number;
  vsLastPlan: { meanDist: number; maxDist: number };
}

interface Summary {
  timestamp: string;
  seed: number;
  targetLaps: number;
  tracker: string;
  maxSimTime: number;
  results: Array<{
    name: string;
    finished: boolean;
    laps: Array<{ lap: number; simTime: number; duration: number; sectors: number[] }>;
    best: number;
    avg: number;
    stddev: number;
    offTrackEvents: number;
    predErrorRms: number;
    totalReplans: number;
    successfulReplans: number;
    replanReasonCounts: Record<string, number>;
    plannerMsMean: number;
    plannerMsMax: number;
    plannerDeadlineHits: number;
    sharpSteerTicks: number;
  }>;
}

function median(xs: number[]): number {
  if (xs.length === 0) return NaN;
  const s = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function pct(xs: number[], p: number): number {
  if (xs.length === 0) return NaN;
  const s = xs.slice().sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.floor(p * s.length));
  return s[i]!;
}

function analyseTrace(name: string, samples: TickSample[]): string {
  if (samples.length === 0) return `${name}: no samples\n`;
  const lateralErrs = samples.map((s) => s.lateralErr).filter((x) => Number.isFinite(x));
  const speeds = samples.map((s) => Math.abs(s.speed));
  const steers = samples.map((s) => Math.abs(s.steer));
  const brakeOn = samples.filter((s) => s.brake > 0.3).length;
  const throttleOn = samples.filter((s) => s.throttle > 0.3).length;
  const total = samples.length;
  const sharp = steers.filter((s) => s > 0.15).length;  // ~|κ| > 0.15
  const lines = [
    `=== ${name} ===`,
    `  samples=${total}  duration=${(samples[samples.length - 1]!.simTime - samples[0]!.simTime).toFixed(1)}s`,
    `  speed       mean=${(speeds.reduce((a, b) => a + b, 0) / total).toFixed(1)}  p50=${median(speeds).toFixed(1)}  p95=${pct(speeds, 0.95).toFixed(1)}  max=${Math.max(...speeds).toFixed(1)}`,
    `  lateral err mean=${(lateralErrs.reduce((a, b) => a + b, 0) / lateralErrs.length).toFixed(2)}m  p50=${median(lateralErrs).toFixed(2)}  p95=${pct(lateralErrs, 0.95).toFixed(2)}  max=${Math.max(...lateralErrs).toFixed(2)}`,
    `  |steer|     mean=${(steers.reduce((a, b) => a + b, 0) / total).toFixed(3)}  p50=${median(steers).toFixed(3)}  p95=${pct(steers, 0.95).toFixed(3)}  max=${Math.max(...steers).toFixed(3)}`,
    `  brake on=${(brakeOn / total * 100).toFixed(1)}%   throttle on=${(throttleOn / total * 100).toFixed(1)}%   sharp(|κ|>0.15)=${(sharp / total * 100).toFixed(1)}%`,
  ];
  // Lateral-error hotspots: 5 worst windows of 2 s each.
  const windowSec = 2;
  const buckets: Array<{ start: number; samples: TickSample[] }> = [];
  let bucketStart = samples[0]!.simTime;
  let bucketSamples: TickSample[] = [];
  for (const s of samples) {
    if (s.simTime >= bucketStart + windowSec) {
      buckets.push({ start: bucketStart, samples: bucketSamples });
      bucketStart += windowSec;
      bucketSamples = [];
    }
    bucketSamples.push(s);
  }
  if (bucketSamples.length > 0) buckets.push({ start: bucketStart, samples: bucketSamples });
  const bucketStats = buckets.map((b) => ({
    start: b.start,
    maxLat: Math.max(...b.samples.map((s) => s.lateralErr)),
    meanSpeed: b.samples.reduce((a, s) => a + Math.abs(s.speed), 0) / b.samples.length,
    sharpFrac: b.samples.filter((s) => Math.abs(s.steer) > 0.15).length / b.samples.length,
    avgX: b.samples.reduce((a, s) => a + s.x, 0) / b.samples.length,
    avgZ: b.samples.reduce((a, s) => a + s.z, 0) / b.samples.length,
  }));
  const worstLat = bucketStats.slice().sort((a, b) => b.maxLat - a.maxLat).slice(0, 5);
  lines.push('  lateral-error hotspots (worst 5 2s windows):');
  for (const w of worstLat) {
    lines.push(`    t=${w.start.toFixed(1)}s  pos=(${w.avgX.toFixed(0)}, ${w.avgZ.toFixed(0)})  maxLat=${w.maxLat.toFixed(2)}m  spd=${w.meanSpeed.toFixed(1)}  sharp%=${(w.sharpFrac * 100).toFixed(0)}`);
  }
  const sharpHotspots = bucketStats.slice().sort((a, b) => b.sharpFrac - a.sharpFrac).slice(0, 5);
  lines.push('  sharp-steer hotspots (worst 5 2s windows):');
  for (const w of sharpHotspots) {
    lines.push(`    t=${w.start.toFixed(1)}s  pos=(${w.avgX.toFixed(0)}, ${w.avgZ.toFixed(0)})  sharp%=${(w.sharpFrac * 100).toFixed(0)}  spd=${w.meanSpeed.toFixed(1)}  maxLat=${w.maxLat.toFixed(2)}`);
  }
  return lines.join('\n') + '\n';
}

function analyseReplans(name: string, history: ReplanSnapshot[]): string {
  if (history.length === 0) return `${name}: no replans captured\n`;
  const lines = [
    `=== ${name} replans (ring buffer; up to 30 most recent) ===`,
  ];
  // Group by reason.
  const byReason: Record<string, ReplanSnapshot[]> = {};
  for (const r of history) (byReason[r.reason] ??= []).push(r);
  for (const [reason, rs] of Object.entries(byReason)) {
    const searches = rs.map((r) => r.searchMs);
    const expansions = rs.map((r) => r.expansions);
    const deadlineHits = rs.filter((r) => r.deadlineHit).length;
    const meanVsLast = rs
      .map((r) => r.vsLastPlan.maxDist)
      .filter((d) => d >= 0);
    lines.push(`  ${reason}  count=${rs.length}  ms p50=${median(searches).toFixed(1)} p95=${pct(searches, 0.95).toFixed(1)}  exp p50=${median(expansions).toFixed(0)}  deadlineHits=${deadlineHits}  plan-drift p95=${pct(meanVsLast, 0.95).toFixed(2)}m`);
  }
  // 3 most-deadline-hit replans.
  const slow = history.slice().sort((a, b) => b.searchMs - a.searchMs).slice(0, 3);
  lines.push('  slowest 3 replans:');
  for (const r of slow) {
    lines.push(`    t=${r.simTime.toFixed(1)}s  ${r.searchMs.toFixed(1)}ms (deadline=${r.deadlineHit ? 'YES' : 'no'})  reason=${r.reason}  cost=${Number.isFinite(r.cost) ? r.cost.toFixed(1) : 'INF'}  exp=${r.expansions}`);
  }
  return lines.join('\n') + '\n';
}

function main(): void {
  const arg = process.argv[2];
  if (!arg) {
    process.stderr.write('Usage: tsx scripts/analyze-race-debug.ts <run-dir>\n');
    process.stderr.write('   or: tsx scripts/analyze-race-debug.ts <root-with-timestamped-subdirs>\n');
    process.exit(2);
  }
  let runDir = resolve(arg);
  // If the arg points to the parent of timestamped subdirs, pick the
  // newest. Saves typing.
  if (!existsSync(`${runDir}/summary.json`)) {
    const subs = readdirSync(runDir).filter((s: string) => statSync(`${runDir}/${s}`).isDirectory());
    if (subs.length === 0) {
      process.stderr.write(`no summary.json in ${runDir} and no subdirs\n`);
      process.exit(1);
    }
    subs.sort();
    runDir = `${runDir}/${subs[subs.length - 1]}`;
    process.stdout.write(`using latest run: ${runDir}\n\n`);
  }
  const summary: Summary = JSON.parse(readFileSync(`${runDir}/summary.json`, 'utf-8'));
  process.stdout.write(`run: seed=${summary.seed} laps=${summary.targetLaps} tracker=${summary.tracker} timestamp=${summary.timestamp}\n\n`);
  for (const r of summary.results) {
    process.stdout.write(`${r.name}: ${r.finished ? 'OK' : 'DNF'}  laps=${r.laps.length}/${summary.targetLaps}  best=${r.best.toFixed(2)}s  avg=${r.avg.toFixed(2)}s  off-track=${r.offTrackEvents}  predErrRMS=${r.predErrorRms.toFixed(2)}\n`);
    process.stdout.write(`  replans total=${r.totalReplans} ok=${r.successfulReplans} deadlineHits=${r.plannerDeadlineHits}  planner ms mean=${r.plannerMsMean.toFixed(1)} max=${r.plannerMsMax.toFixed(1)}\n`);
    process.stdout.write(`  reasons: ${Object.entries(r.replanReasonCounts).map(([k, v]) => `${k}=${v}`).join('  ')}\n`);
    process.stdout.write(`  sharp-steer ticks=${r.sharpSteerTicks}\n`);
  }
  process.stdout.write('\n');
  const tracesPath = `${runDir}/traces.json`;
  if (existsSync(tracesPath)) {
    const traces: CarTrace[] = JSON.parse(readFileSync(tracesPath, 'utf-8'));
    for (const t of traces) process.stdout.write(analyseTrace(t.name, t.samples) + '\n');
  } else {
    process.stdout.write('(no traces.json — run with --debug-dir for per-tick trace capture)\n\n');
  }
  const historyPath = `${runDir}/replan-history.json`;
  if (existsSync(historyPath)) {
    const histories: Array<{ name: string; replanHistory: ReplanSnapshot[] }> = JSON.parse(readFileSync(historyPath, 'utf-8'));
    for (const h of histories) process.stdout.write(analyseReplans(h.name, h.replanHistory) + '\n');
  }
}

main();
