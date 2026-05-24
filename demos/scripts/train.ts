// Headless training CLI — `pnpm run train`.
//
// Drives the same maneuver-based training pipeline the in-browser Model
// Lab uses (`runManeuverTraining`), streams progress to stdout, and on
// completion writes `demos/public/models/v2-default.json` (the
// `PersistedV2Model` payload the demos load on first visit) +
// `demos/public/models/v2-default.manifest.json` (provenance sidecar).

import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { execSync } from 'node:child_process';
import {
  runManeuverTraining,
  type TrainingEvent,
  DEFAULT_VEHICLE_OPTS,
} from '../app/lib/training-driver';
import { modelToJson } from '../app/lib/v2-model-file';

interface Profile {
  rounds: number;
  trialsPerRound: number;
  trialTicks: number;
  sampleEveryNTicks: number;
}

const PROFILES: Record<string, Profile> = {
  // Smoke profile for CI / refactor sanity checks — ~30 s.
  quick: { rounds: 1, trialsPerRound: 30, trialTicks: 60, sampleEveryNTicks: 6 },
  // Default: ~5-10 min on a laptop.
  default: { rounds: 3, trialsPerRound: 120, trialTicks: 120, sampleEveryNTicks: 6 },
  // Sweep profile: Phase 4 multi-config (reserved for the future).
  sweep: { rounds: 5, trialsPerRound: 200, trialTicks: 150, sampleEveryNTicks: 6 },
  // Overnight: max rounds till val plateau (single config for now).
  overnight: { rounds: 12, trialsPerRound: 300, trialTicks: 180, sampleEveryNTicks: 6 },
};

function gitSha(): string {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
  } catch {
    return 'unknown';
  }
}

function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}min`;
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    allowPositionals: true,
    options: {
      profile: { type: 'string', default: 'default' },
      seed: { type: 'string', default: '42' },
      rounds: { type: 'string' },
      trials: { type: 'string' },
      ticks: { type: 'string' },
      out: { type: 'string' },
      dagger: { type: 'string' },
      'import-mined': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run train -- [--profile=quick|default|sweep|overnight]
                       [--seed=N] [--rounds=N] [--trials=N] [--ticks=N]
                       [--out=path/to/v2-default.json]

Profiles:
${Object.entries(PROFILES).map(([k, v]) => `  ${k.padEnd(10)} rounds=${v.rounds} trials/round=${v.trialsPerRound} ticks=${v.trialTicks}`).join('\n')}
`);
    return;
  }

  const profileName = String(values.profile);
  const profile = PROFILES[profileName];
  if (!profile) {
    throw new Error(`Unknown profile "${profileName}". Available: ${Object.keys(PROFILES).join(', ')}`);
  }
  const rounds = values.rounds ? Number(values.rounds) : profile.rounds;
  const trialsPerRound = values.trials ? Number(values.trials) : profile.trialsPerRound;
  const trialTicks = values.ticks ? Number(values.ticks) : profile.trialTicks;
  const sampleEveryNTicks = profile.sampleEveryNTicks;
  const seed = Number(values.seed);
  // Repo root = parent of `demos/` directory; this file lives at
  // demos/scripts/train.ts so root = ../.. from here.
  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const outArg = values.out ? String(values.out) : 'demos/public/models/v2-default.json';
  const outPath = isAbsolute(outArg) ? outArg : resolve(repoRoot, outArg);

  process.stdout.write(`kinocat train — profile=${profileName} seed=${seed} rounds=${rounds} trials/round=${trialsPerRound} ticks=${trialTicks}\n`);

  const t0 = Date.now();
  let lastRoundT = t0;

  const daggerStartRound = values.dagger !== undefined ? Number(values.dagger) : undefined;
  if (daggerStartRound !== undefined) {
    process.stdout.write(`  · DAgger mode: race-collect starting round ${daggerStartRound + 1}\n`);
  }

  // Phase 3.5: import sim-to-real-mined trials before round 0 so the
  // residual MLP sees the hard regimes from the first fit.
  let minedTrials = undefined;
  if (values['import-mined']) {
    const arg = String(values['import-mined']);
    const minedPath = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    const raw = readFileSync(minedPath, 'utf-8');
    const parsed = JSON.parse(raw);
    minedTrials = parsed.trials ?? [];
    process.stdout.write(`  · Imported ${minedTrials.length} mined trial${minedTrials.length === 1 ? '' : 's'} from ${minedPath}\n`);
  }

  const result = await runManeuverTraining({
    rounds,
    trialsPerRound,
    trialTicks,
    sampleEveryNTicks,
    seed,
    daggerStartRound,
    minedTrials,
    onEvent: (e: TrainingEvent) => {
      switch (e.type) {
        case 'round-start':
          process.stdout.write(`  round ${e.round + 1}/${rounds} — collecting ${trialsPerRound} trials...\n`);
          break;
        case 'trial-batch':
          process.stdout.write(`    + ${e.collected} trials (${e.discarded} discarded)\n`);
          break;
        case 'evaluation': {
          const rms = e.diagnostics.openLoopDivergence.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          const blKin = e.diagnostics.baselines['kinematic']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          const blParam = e.diagnostics.baselines['parametricOnly']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          process.stdout.write(`    eval: posRms@1s v2=${rms.toFixed(3)}m parametric=${blParam.toFixed(3)}m kinematic=${blKin.toFixed(3)}m\n`);
          break;
        }
        case 'round-end': {
          const now = Date.now();
          process.stdout.write(`    round ${e.round + 1} done — store=${e.trialsAfter} trials, ${fmtMs(now - lastRoundT)}\n`);
          lastRoundT = now;
          break;
        }
        case 'done':
          process.stdout.write(`done — ${e.totalTrials} trials total, ${fmtMs(Date.now() - t0)} wall time\n`);
          break;
      }
    },
  });

  // Headline diagnostics for the manifest sidecar.
  const headline = (h: number): number =>
    result.finalDiagnostics.openLoopDivergence.find((r) => Math.abs(r.tSec - h) < 0.05)?.posRms ?? NaN;

  const meta = {
    trialsUsed: result.trials.size(),
    openLoopRmsAt1s: headline(1.0),
    legacyRmsAt1s: result.finalDiagnostics.baselines['legacyV1']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms,
    kinematicRmsAt1s: result.finalDiagnostics.baselines['kinematic']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms,
    createdAt: Date.now(),
  };

  const payload = modelToJson(result.model, meta);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf-8');
  process.stdout.write(`wrote ${outPath}\n`);

  // Manifest sidecar.
  const manifestPath = outPath.replace(/\.json$/, '.manifest.json');
  const manifest = {
    version: 1,
    profile: profileName,
    seed,
    rounds,
    trialsPerRound,
    trialTicks,
    sampleEveryNTicks,
    totalTrials: result.trials.size(),
    sampleDt: result.sampleDt,
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    git: gitSha(),
    diagnostics: {
      openLoopDivergence: result.finalDiagnostics.openLoopDivergence,
      perStateRms: result.finalDiagnostics.perStateRms,
      baselines: result.finalDiagnostics.baselines,
    },
    runtimeMs: Date.now() - t0,
    createdAt: Date.now(),
  };
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf-8');
  process.stdout.write(`wrote ${manifestPath}\n`);
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(`train failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
    process.exit(1);
  },
);
