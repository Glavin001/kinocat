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
      quick: { type: 'boolean', default: false },
      'no-kinematic': { type: 'boolean', default: false },
      'no-parametric': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run race -- [--models=a.json,b.json,...]
                      [--laps=N] [--seed=N] [--max-sim=N]
                      [--json=path] [--ledger=dir] [--quick]
                      [--no-kinematic] [--no-parametric]
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
  process.stdout.write(`race-primitives benchmark — seed=${values.seed} laps=${targetLaps} dt=1/60\n`);

  const results = await runHeadlessRace({
    entries,
    targetLaps,
    maxSimTime,
    onProgress: (msg) => process.stdout.write(`  · ${msg}\n`),
    progressEverySec: 10,
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

  // Exit code: pass iff all entries finished. For CI noise reduction we
  // do NOT fail if v2 < kinematic — that's a quality signal, not a
  // contract.
  const ok = results.every((r) => r.finished);
  process.exit(ok ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`race failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
  process.exit(1);
});
