// Feature-ablation harness — `pnpm run ablation`.
//
// Runs the headless race scenario across a matrix of `RaceTuning`
// presets so we can answer "what does each improvement actually buy
// us?" with hard numbers, and catch regressions when a new feature
// lands. Every preset uses the SAME entry (default: kinematic baseline)
// and the same physics seed, so the only varying input is the tuning.
//
// Output: a table comparing best/avg/stddev lap times, prediction RMS,
// and replan counts vs. the LEGACY baseline (everything OFF). A positive
// delta means the feature helps; a negative delta means it hurts on
// this scenario / entry combination.

import { readFileSync, existsSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, basename, isAbsolute, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  runHeadlessRace,
  kinematicEntry,
  v2Entry,
  DEFAULT_TUNING,
  LEGACY_TUNING,
  type RaceEntry,
  type RaceResult,
  type RaceTuning,
} from '../app/lib/headless-race';
import { modelFromJson } from '../app/lib/v2-model-file';
import type { PersistedV2Model } from '../app/lib/v2-model-persistence';

// ---------------------------------------------------------------------------
// Preset definitions: which feature does this row isolate?
//
// "all-on"  = DEFAULT_TUNING (current best-in-class state)
// "legacy"  = LEGACY_TUNING (every improvement disabled)
// "no-X"    = DEFAULT_TUNING with feature X disabled (so the gap vs.
//             all-on isolates the contribution of X)
// "only-X"  = LEGACY_TUNING with feature X enabled (so the gap vs.
//             legacy isolates the standalone contribution of X)

interface Preset {
  name: string;
  tuning: RaceTuning;
  notes: string;
}

function preset(name: string, overrides: Partial<RaceTuning>, notes: string, base = DEFAULT_TUNING): Preset {
  return { name, tuning: { ...base, ...overrides }, notes };
}

const DEFAULT_PRESETS: Preset[] = [
  { name: 'all-on (default)', tuning: { ...DEFAULT_TUNING }, notes: 'everything enabled' },
  { name: 'legacy (baseline)', tuning: { ...LEGACY_TUNING }, notes: 'pre-improvement baseline' },
  preset('no-commit-window', { commitWindowMs: 0 }, 'plan stitching off'),
  preset('no-consistency', { consistencyWeight: 0 }, 'trajectory-consistency hysteresis off'),
  preset('no-speed-profile', { enableSpeedProfile: false }, 'friction-circle pass off'),
  preset('no-trajectory-smoother', { enableTrajectorySmoother: false }, 'geometric smoother off'),
  preset('no-respect-path-speed', { respectPathSpeed: false }, 'controller ignores smoothed speeds'),
  preset('no-adaptive-replan', { enableAdaptiveReplan: false, enableWaypointAdvanceReplan: false }, 'cadence-only replans'),
  preset('no-heuristic-table', { enableHeuristicTable: false }, 'no RS LUT in env'),
];

function parsePreset(arg: string): Preset {
  // Format: "name:flag=val,flag=val,..."
  // Example: "stitch-only:commitWindowMs=200,enableSpeedProfile=false"
  const [name, rest] = arg.includes(':') ? arg.split(/:(.+)/) : [arg, ''];
  const flagPairs = rest ? rest.split(',') : [];
  const overrides: Partial<RaceTuning> = {};
  for (const pair of flagPairs) {
    const [k, v] = pair.split('=');
    if (!k || v === undefined) continue;
    const value =
      v === 'true' ? true :
      v === 'false' ? false :
      Number.isFinite(Number(v)) ? Number(v) :
      (v as unknown as never);
    (overrides as Record<string, unknown>)[k] = value;
  }
  return { name: name!, tuning: { ...DEFAULT_TUNING, ...overrides }, notes: 'custom' };
}

function loadEntry(modelPath: string | undefined): RaceEntry {
  if (!modelPath) return kinematicEntry('kinematic');
  if (!existsSync(modelPath)) {
    throw new Error(`Model file not found: ${modelPath}`);
  }
  const payload = JSON.parse(readFileSync(modelPath, 'utf-8')) as PersistedV2Model;
  const model = modelFromJson(payload);
  return v2Entry(basename(modelPath).replace(/\.json$/, ''), model);
}

function fmt(n: number, digits = 2): string {
  return Number.isFinite(n) ? n.toFixed(digits) : '---';
}

function fmtDelta(value: number, base: number, lowerIsBetter: boolean): string {
  if (!Number.isFinite(value) || !Number.isFinite(base) || base === 0) return '  -  ';
  const pct = ((value - base) / base) * 100;
  const sign = pct > 0 ? '+' : '';
  const arrow =
    Math.abs(pct) < 0.5 ? '·' : lowerIsBetter ? (pct < 0 ? '▼' : '▲') : (pct > 0 ? '▲' : '▼');
  return `${sign}${pct.toFixed(1)}%${arrow}`;
}

interface AblationRow {
  preset: Preset;
  result: RaceResult;
}

function printTable(rows: AblationRow[], baselineName: string): void {
  const baseline = rows.find((r) => r.preset.name === baselineName)?.result;
  const headers = ['preset', 'best', 'avg', 'stddev', 'pred-rms', 'replans', 'off-track', 'notes'];
  const widths = [26, 8, 8, 8, 10, 8, 10, 40];
  const sep = '─'.repeat(widths.reduce((a, b) => a + b + 2, 0));
  process.stdout.write(
    headers.map((h, i) => h.padEnd(widths[i] ?? 0)).join('  ') + '\n' + sep + '\n',
  );
  for (const { preset: p, result: r } of rows) {
    const cols = [
      p.name,
      fmt(r.best),
      fmt(r.avg),
      fmt(r.stddev),
      fmt(r.predErrorRms),
      String(r.totalReplans),
      String(r.offTrackEvents),
      p.notes,
    ];
    process.stdout.write(cols.map((c, i) => String(c).padEnd(widths[i] ?? 0)).join('  ') + '\n');
    if (baseline && p.name !== baselineName) {
      const deltaCols = [
        `  Δ vs ${baselineName}`,
        fmtDelta(r.best, baseline.best, true),
        fmtDelta(r.avg, baseline.avg, true),
        fmtDelta(r.stddev, baseline.stddev, true),
        fmtDelta(r.predErrorRms, baseline.predErrorRms, true),
        '',
        '',
        '',
      ];
      process.stdout.write(deltaCols.map((c, i) => String(c).padEnd(widths[i] ?? 0)).join('  ') + '\n');
    }
  }
  process.stdout.write(sep + '\n');
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    allowPositionals: true,
    options: {
      model: { type: 'string' },
      laps: { type: 'string', default: '4' },
      'max-sim': { type: 'string', default: '300' },
      preset: { type: 'string', multiple: true },
      baseline: { type: 'string', default: 'legacy (baseline)' },
      json: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run ablation -- [--model=path.json] [--laps=N] [--max-sim=N]
                          [--preset=name:flag=val,... ...] [--baseline=NAME]
                          [--json=out.json]

Default presets cover commit-window, consistency cost, speed profile,
trajectory smoother, path-speed clamp, adaptive replan, and RS heuristic
table. Override with --preset to run any custom combination, e.g.:

  pnpm run ablation -- --preset=stitch-only:commitWindowMs=200
\n`);
    return;
  }
  const presets: Preset[] = values.preset && values.preset.length > 0
    ? values.preset.map(parsePreset)
    : DEFAULT_PRESETS;
  const targetLaps = parseInt(values.laps!, 10);
  const maxSimTime = parseInt(values['max-sim']!, 10);
  const entry = loadEntry(values.model);
  process.stdout.write(`entry: ${entry.name} · laps: ${targetLaps} · max-sim: ${maxSimTime}s\n\n`);

  const rows: AblationRow[] = [];
  for (const p of presets) {
    process.stdout.write(`▶ running ${p.name} ...`);
    const start = performance.now();
    const result = (
      await runHeadlessRace({
        entries: [{ name: entry.name, lib: entry.lib }],
        targetLaps,
        maxSimTime,
        tuning: p.tuning,
      })
    )[0]!;
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    process.stdout.write(` ${elapsed}s · best=${fmt(result.best)}s avg=${fmt(result.avg)}s\n`);
    rows.push({ preset: p, result });
  }
  process.stdout.write('\n');
  printTable(rows, values.baseline ?? 'legacy (baseline)');

  if (values.json) {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const out = isAbsolute(values.json) ? values.json : resolve(__dirname, '..', values.json);
    mkdirSync(dirname(out), { recursive: true });
    const payload = rows.map((r) => ({
      preset: r.preset.name,
      tuning: r.preset.tuning,
      notes: r.preset.notes,
      result: r.result,
    }));
    writeFileSync(out, JSON.stringify(payload, null, 2));
    process.stdout.write(`\nWrote ${out}\n`);
  }
}

main().catch((e) => {
  process.stderr.write(`ablation failed: ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
  process.exit(1);
});
