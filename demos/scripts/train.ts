// Headless training CLI — `pnpm run train`.
//
// Drives the same maneuver-based training pipeline the in-browser Model
// Lab uses (`runManeuverTraining`), streams progress to stdout, and on
// completion writes `demos/public/models/v2-default.json` (the
// `PersistedV2Model` payload the demos load on first visit) +
// `demos/public/models/v2-default.manifest.json` (provenance sidecar).

import { writeFileSync, mkdirSync, readFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve, isAbsolute, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { execSync, spawnSync } from 'node:child_process';
import {
  runManeuverTraining,
  type TrainingEvent,
  DEFAULT_VEHICLE_OPTS,
  type PythonFitRequest,
  type PythonFitResult,
} from '../app/lib/training-driver';
import { writeTrialsNpz } from './lib/trial-npz';
import { readResidualEnsembleNpz } from './lib/residual-npz';
import { modelToJson } from '../app/lib/v2-model-file';
import {
  computeCacheKey,
  tryReadRoundCache,
  writeRoundCache,
  getRapierVersionTag,
  CACHE_BUSTER,
  type TrialCacheKeyInputs,
} from '../app/lib/trial-cache';

interface TrialCacheOpts {
  cacheDir: string;
  seed: number;
  trialsPerRound: number;
  trialTicks: number;
  sampleEveryNTicks: number;
  bundle: 'default' | 'universal';
}

function buildTrialCache(opts: TrialCacheOpts) {
  const rapierVersion = getRapierVersionTag();
  const makeKey = (round: number): TrialCacheKeyInputs => ({
    seed: opts.seed,
    round,
    trialsPerRound: opts.trialsPerRound,
    trialTicks: opts.trialTicks,
    sampleEveryNTicks: opts.sampleEveryNTicks,
    bundle: opts.bundle,
    startSpeedSchedule: [0, 4, 8, 12, 16, 20, 24, 28],
    vehicleOptions: DEFAULT_VEHICLE_OPTS as Record<string, unknown>,
    rapierVersion,
    cacheBuster: CACHE_BUSTER,
  });
  return {
    tryRead(round: number) {
      const key = computeCacheKey(makeKey(round));
      const result = tryReadRoundCache(opts.cacheDir, key);
      if (result) {
        process.stdout.write(`    (cache hit: round ${round + 1}, key=${key})\n`);
      }
      return result;
    },
    write(round: number, trials: unknown[]) {
      const inputs = makeKey(round);
      const key = computeCacheKey(inputs);
      writeRoundCache(opts.cacheDir, key, trials as never, inputs);
      process.stdout.write(`    (cache write: round ${round + 1}, key=${key})\n`);
    },
  };
}

interface Profile {
  rounds: number;
  trialsPerRound: number;
  trialTicks: number;
  sampleEveryNTicks: number;
}

const PROFILES: Record<string, Profile & { bundle?: 'default' | 'universal' }> = {
  // Smoke profile for CI / refactor sanity checks — ~30 s.
  quick: { rounds: 1, trialsPerRound: 30, trialTicks: 60, sampleEveryNTicks: 6 },
  // Default: ~5-10 min on a laptop.
  default: { rounds: 4, trialsPerRound: 200, trialTicks: 150, sampleEveryNTicks: 6 },
  // Sweep profile: Phase 4 multi-config (reserved for the future).
  sweep: { rounds: 5, trialsPerRound: 200, trialTicks: 150, sampleEveryNTicks: 6 },
  // Overnight: max rounds till val plateau (single config for now).
  overnight: { rounds: 12, trialsPerRound: 300, trialTicks: 180, sampleEveryNTicks: 6 },
  // Universal: covers all 45 regimes from the universal coverage matrix
  // (passive coast, reverse, multi-cusp parking, etc.) — the bundle the
  // shipped v2-default.json should be trained on for "one model, all
  // driving" capability. Larger trial count + the universal bundle.
  universal: { rounds: 4, trialsPerRound: 500, trialTicks: 180, sampleEveryNTicks: 6, bundle: 'universal' },
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
      bundle: { type: 'string' },
      'mine-from-trace': { type: 'string' },
      'cache-dir': { type: 'string' },
      'no-cache': { type: 'boolean', default: false },
      trainer: { type: 'string', default: 'js' },          // 'js' | 'python'
      python: { type: 'string', default: 'python3' },      // python interpreter for --trainer=python
      'max-iter': { type: 'string', default: '50' },       // LM iters for python trainer
      'mlp-shape': { type: 'string', default: '64,64' },
      'ensemble-size': { type: 'string', default: '3' },
      epochs: { type: 'string', default: '200' },
      'no-mlp': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run train -- [--profile=quick|default|sweep|overnight]
                       [--seed=N] [--rounds=N] [--trials=N] [--ticks=N]
                       [--out=path/to/v2-default.json]
                       [--cache-dir=path] [--no-cache]
                       [--trainer=js|python]  [--python=python3]
                       [--max-iter=N] [--mlp-shape=64,64] [--ensemble-size=3] [--epochs=200]

Trainer modes:
  js (default)  legacy Nelder-Mead + SGD path
  python        JAX-based LM + Adam (see demos/scripts/python/README.md)

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
  const bundle = (values.bundle ? String(values.bundle) : (profile.bundle ?? 'default')) as 'default' | 'universal';
  const seed = Number(values.seed);
  // Repo root = parent of `demos/` directory; this file lives at
  // demos/scripts/train.ts so root = ../.. from here.
  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const outArg = values.out ? String(values.out) : 'demos/public/models/v2-default.json';
  const outPath = isAbsolute(outArg) ? outArg : resolve(repoRoot, outArg);

  // Trial cache: enabled by default, disabled with --no-cache.
  const cacheDir = values['no-cache']
    ? undefined
    : (() => {
        const arg = values['cache-dir'] ? String(values['cache-dir']) : 'demos/.cache/training';
        return isAbsolute(arg) ? arg : resolve(repoRoot, arg);
      })();

  process.stdout.write(`kinocat train — profile=${profileName} seed=${seed} rounds=${rounds} trials/round=${trialsPerRound} ticks=${trialTicks}\n`);
  if (cacheDir) {
    process.stdout.write(`  · trial cache: ${cacheDir}\n`);
  } else {
    process.stdout.write(`  · trial cache: disabled\n`);
  }

  const t0 = Date.now();
  let lastRoundT = t0;
  let phaseT = t0;

  const daggerStartRound = values.dagger !== undefined ? Number(values.dagger) : undefined;
  if (daggerStartRound !== undefined) {
    process.stdout.write(`  · DAgger mode: race-collect starting round ${daggerStartRound + 1}\n`);
  }

  // Phase 3.5: import sim-to-real-mined trials before round 0 so the
  // residual MLP sees the hard regimes from the first fit.
  let minedTrials: unknown[] = [];
  if (values['import-mined']) {
    const arg = String(values['import-mined']);
    const minedPath = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    const raw = readFileSync(minedPath, 'utf-8');
    const parsed = JSON.parse(raw);
    minedTrials = parsed.trials ?? [];
    process.stdout.write(`  · Imported ${minedTrials.length} mined trial${minedTrials.length === 1 ? '' : 's'} from ${minedPath}\n`);
  }
  // New: convert a sim-to-real debug JSON's frames into Trial format
  // (each consecutive frame pair → one Trial). Lets us train on the
  // user's reported failure trace by direct import.
  if (values['mine-from-trace']) {
    const { DEFAULT_LEARNABLE_CONFIG } = await import('kinocat/agent');
    const arg = String(values['mine-from-trace']);
    const tracePath = isAbsolute(arg) ? arg : resolve(repoRoot, arg);
    const trace = JSON.parse(readFileSync(tracePath, 'utf-8'));
    const frames = trace.frames ?? [];
    const sampleDt = trace.meta?.physicsDt ?? 1 / 60;
    const cfg = trace.meta?.vehicleConfig ?? DEFAULT_LEARNABLE_CONFIG;
    // The codebase uses string config keys like 'rwd-default' (see
    // training-driver / sim-to-real). Use the same convention so
    // mined trials group with the synthetic ones during training.
    const cfgKey = 'rwd-default';
    let added = 0;
    for (let i = 0; i + 1 < frames.length; i++) {
      const cur = frames[i];
      const nxt = frames[i + 1];
      if (!cur?.real || !cur?.controls || !nxt?.real) continue;
      const wheeled = {
        steer: cur.controls.steer,
        driveForce: cur.controls.driveForce,
        brakeForce: cur.controls.brakeForce,
      };
      minedTrials.push({
        id: `mined-trace-${i}`,
        initialState: cur.real,
        controlsTrace: [wheeled],
        dt: sampleDt,
        samples: [
          { t: 0, state: cur.real },
          { t: sampleDt, state: nxt.real },
        ],
        config: cfg,
        configKey: cfgKey,
      });
      added++;
    }
    process.stdout.write(`  · Mined ${added} trial${added === 1 ? '' : 's'} from trace ${tracePath}\n`);
  }

  // --trainer=python: replace the JS Nelder-Mead + SGD fits with a
  // JAX-based pipeline via `demos/scripts/python/train_fit.py`. The hook
  // runs once per round; on the final round it also fits the residual
  // MLP ensemble and ships it back as an npz.
  const trainerMode = String(values.trainer);
  if (trainerMode !== 'js' && trainerMode !== 'python') {
    throw new Error(`Unknown --trainer=${trainerMode}. Use 'js' or 'python'.`);
  }
  let pythonFitHook: ((req: PythonFitRequest) => Promise<PythonFitResult>) | undefined;
  if (trainerMode === 'python') {
    const pythonInterp = String(values.python);
    const pyDir = resolve(repoRoot, 'demos/scripts/python');
    const maxIter = Number(values['max-iter']);
    const mlpShape = String(values['mlp-shape']);
    const ensembleSize = Number(values['ensemble-size']);
    const epochs = Number(values.epochs);
    const tmpRoot = mkdtempSync(join(tmpdir(), 'kinocat-jax-'));
    process.stdout.write(`  · trainer: python (JAX) interp=${pythonInterp} tmp=${tmpRoot}\n`);

    pythonFitHook = async (req) => {
      const roundDir = join(tmpRoot, `round-${req.round}`);
      mkdirSync(roundDir, { recursive: true });
      const trialsPath = join(roundDir, 'trials.npz');
      const inParamsPath = join(roundDir, 'params-in.json');
      const outParamsPath = join(roundDir, 'params-out.json');
      const skipMlp = Boolean(values['no-mlp']);
      const outResidualPath = (!skipMlp && req.isFinalRound) ? join(roundDir, 'mlp.npz') : null;

      writeTrialsNpz(trialsPath, req.trials);
      writeFileSync(inParamsPath, JSON.stringify(req.currentParams, null, 2));

      const args = [
        '-m', 'train_fit',
        '--trials', trialsPath,
        '--init-params', inParamsPath,
        '--out-params', outParamsPath,
        '--max-iter', String(maxIter),
        '--mlp-shape', mlpShape,
        '--ensemble-size', String(ensembleSize),
        '--epochs', String(epochs),
        '--verbose',
      ];
      if (outResidualPath) {
        args.push('--out-residual', outResidualPath);
      } else {
        args.push('--no-residual');
      }

      process.stdout.write(`    [python] round ${req.round + 1} (${req.trials.length} trials)\n`);
      const t0 = Date.now();
      const res = spawnSync(pythonInterp, args, {
        cwd: pyDir,
        stdio: ['ignore', 'inherit', 'inherit'],
        env: { ...process.env, PYTHONPATH: pyDir },
      });
      if (res.status !== 0) {
        throw new Error(`python trainer failed with status ${res.status}`);
      }
      process.stdout.write(`    [python] round ${req.round + 1} done in ${fmtMs(Date.now() - t0)}\n`);

      const fittedParams = JSON.parse(readFileSync(outParamsPath, 'utf-8'));
      let residualEnsemble;
      if (outResidualPath) {
        residualEnsemble = readResidualEnsembleNpz(outResidualPath);
      }
      return {
        params: fittedParams,
        residualEnsemble,
        residualReferenceDt: req.sampleDt,
      };
    };
  }

  const result = await runManeuverTraining({
    rounds,
    trialsPerRound,
    trialTicks,
    sampleEveryNTicks,
    seed,
    daggerStartRound,
    minedTrials: minedTrials.length > 0 ? (minedTrials as never) : undefined,
    bundle,
    trialCache: cacheDir ? buildTrialCache({ cacheDir, seed, trialsPerRound, trialTicks, sampleEveryNTicks, bundle }) : undefined,
    pythonFit: pythonFitHook,
    // Node runs without a UI to keep responsive — drop the per-iter setTimeout(0).
    cooperativeYield: () => {},
    onEvent: (e: TrainingEvent) => {
      switch (e.type) {
        case 'round-start':
          process.stdout.write(`  round ${e.round + 1}/${rounds} — collecting ${trialsPerRound} trials...\n`);
          break;
        case 'trial-batch':
          process.stdout.write(`    + ${e.collected} trials (${e.discarded} discarded)\n`);
          break;
        case 'phase': {
          // 'collecting' is redundant with round-start; skip it
          if (e.phase === 'collecting') break;
          const elapsed = Date.now() - phaseT;
          // Only show elapsed if a meaningful phase just finished (not first event)
          const suffix = phaseT !== t0 && elapsed > 500 ? ` (prev phase ${fmtMs(elapsed)})` : '';
          phaseT = Date.now();
          process.stdout.write(`    [${e.phase}]${suffix}\n`);
          break;
        }
        case 'fit-progress': {
          const i = e.iterIndex ?? e.event.iter;
          const n = e.iterTotal ?? 0;
          // Log every 20th iteration (or first/last) to avoid flooding stdout
          if (n === 0) break;
          const isFirst = i === 0;
          const isLast = i === n - 1;
          const isPeriodic = (i + 1) % 20 === 0;
          if (isFirst || isLast || isPeriodic) {
            const normLoss = e.event.lossNormalized !== undefined
              ? ` loss/sample=${e.event.lossNormalized.toFixed(6)}`
              : ` loss=${e.event.loss.toFixed(6)}`;
            const pct = Math.round(((i + 1) / n) * 100);
            process.stdout.write(`    [${e.phase}] ${i + 1}/${n} (${pct}%)${normLoss}\n`);
          }
          break;
        }
        case 'evaluation': {
          const rms = e.diagnostics.openLoopDivergence.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          const blKin = e.diagnostics.baselines['kinematic']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          const blParam = e.diagnostics.baselines['parametricOnly']?.find((r) => Math.abs(r.tSec - 1.0) < 0.05)?.posRms ?? NaN;
          process.stdout.write(`    eval: posRms@1s v2=${rms.toFixed(3)}m parametric=${blParam.toFixed(3)}m kinematic=${blKin.toFixed(3)}m\n`);
          break;
        }
        case 'round-end': {
          const now = Date.now();
          const roundMs = now - lastRoundT;
          const roundsDone = e.round + 1;
          const avgMs = (now - t0) / roundsDone;
          const remaining = rounds - roundsDone;
          const etaMs = avgMs * remaining;
          process.stdout.write(
            `    round ${roundsDone}/${rounds} done — store=${e.trialsAfter} trials, ${fmtMs(roundMs)}` +
            ` (avg ${fmtMs(avgMs)}/round, eta ${fmtMs(etaMs)})\n`,
          );
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
  const payloadStr = JSON.stringify(payload, null, 2);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, payloadStr, 'utf-8');
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
  const manifestStr = JSON.stringify(manifest, null, 2);
  writeFileSync(manifestPath, manifestStr, 'utf-8');
  process.stdout.write(`wrote ${manifestPath}\n`);

  // Timestamped snapshot for rollback — e.g. v2-20260525T143012-abc1234.json
  const now = new Date();
  const ts = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, '').replace('T', 'T');
  const sha = gitSha();
  const tag = `${ts}-${sha}`;
  const historyDir = resolve(dirname(outPath), 'history');
  mkdirSync(historyDir, { recursive: true });
  const baseName = outPath.replace(/.*\//, '').replace(/\.json$/, '');
  const snapPath = resolve(historyDir, `${baseName}-${tag}.json`);
  const snapManifest = resolve(historyDir, `${baseName}-${tag}.manifest.json`);
  writeFileSync(snapPath, payloadStr, 'utf-8');
  writeFileSync(snapManifest, manifestStr, 'utf-8');
  process.stdout.write(`wrote ${snapPath}\n`);
  process.stdout.write(`wrote ${snapManifest}\n`);
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(`train failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
    process.exit(1);
  },
);
