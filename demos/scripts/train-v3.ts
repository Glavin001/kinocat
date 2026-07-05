// Headless v3 training CLI — `pnpm exec tsx demos/scripts/train-v3.ts`.
//
// Trains the PURELY data-driven v3 neural dynamics model
// (`core/src/agent/vehicle-model-v3.ts`): collects diverse maneuver trials
// on the real Rapier plant at per-tick resolution, fits the transition
// network directly on those transitions (no parametric backbone, no
// hand-set bounds), evaluates endpoint fidelity against the plant
// side-by-side with the shipped v2 model, and writes
// `demos/public/models/v3-default.json` + a provenance manifest.

import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { execSync } from 'node:child_process';
import {
  createHeadlessTrialHarness,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import {
  forwardSimV3,
  v3ToJson,
  learnedForwardSimV2,
  deserializeMLP,
  type LearnedVehicleModel,
  type CarKinematicState,
  type WheeledCarControls,
  type MLP,
} from 'kinocat/agent';
import { runDynamicsV3Fit, type CarTrial } from 'kinocat/learning';
import type { ForwardSim } from 'kinocat/primitives';
import {
  DEFAULT_VEHICLE_OPTS,
  collectManeuverBatch,
  buildDefaultManeuverBundle,
} from '../app/lib/training-driver';

const PHYSICS_DT = 1 / 60;

interface Profile {
  trials: number;
  ticks: number;
  epochs: number;
  hidden: number[];
  ensemble: number;
}

const PROFILES: Record<string, Profile> = {
  // Smoke profile for CI / refactor sanity checks.
  quick: { trials: 40, ticks: 120, epochs: 8, hidden: [32, 32], ensemble: 1 },
  // Default: full artifact quality.
  default: { trials: 400, ticks: 150, epochs: 60, hidden: [64, 64], ensemble: 3 },
};

// Start speeds cycled across maneuvers. Extends past the 28 m/s v2 grid so
// the model has seen the above-race-cap regime plans can briefly enter
// (race policy cap is 30 m/s; full-throttle endpoints overshoot it).
const SPEED_SCHEDULE = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36];

function gitSha(): string {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
  } catch {
    return 'unknown';
  }
}

// ---------------------------------------------------------------------------
// Fidelity evaluation — same probe grid as model-vs-plant-fidelity.test.ts

const PROBES: { name: string; c: WheeledCarControls }[] = [
  { name: 'coast half-steer', c: { steer: 0.3, driveForce: 0, brakeForce: 0 } },
  { name: 'coast full-lock', c: { steer: 0.6, driveForce: 0, brakeForce: 0 } },
  { name: 'full throttle straight', c: { steer: 0, driveForce: 4000, brakeForce: 0 } },
  { name: 'brake-in-turn', c: { steer: 0.3, driveForce: 0, brakeForce: 1000 } },
];
const EVAL_SPEEDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
const EVAL_TICKS = 48; // 0.8 s

function rollModel(
  sim: ForwardSim<CarKinematicState>,
  start: CarKinematicState,
  controls: number[],
): CarKinematicState {
  let s = start;
  for (let i = 0; i < EVAL_TICKS; i++) s = sim(s, controls, PHYSICS_DT);
  return s;
}

function evaluateFidelity(
  harness: HeadlessTrialHarness,
  sims: Record<string, ForwardSim<CarKinematicState>>,
): Record<string, { mean: number; worst: number }> {
  const stats: Record<string, { sum: number; worst: number; n: number }> = {};
  for (const k of Object.keys(sims)) stats[k] = { sum: 0, worst: 0, n: 0 };
  for (const speed of EVAL_SPEEDS) {
    for (const probe of PROBES) {
      const outcome = harness.runTrial({
        pose: { x: 0, z: 0, heading: 0 },
        kin: { forwardSpeed: speed },
        controlsTrace: new Array(EVAL_TICKS).fill(probe.c),
        sampleEveryNTicks: EVAL_TICKS,
        id: `eval-${speed}-${probe.name}`,
      });
      if (!outcome.ok) throw new Error(`eval trial failed: ${outcome.reason}`);
      const start = outcome.trial.samples[0]!;
      const truth = outcome.trial.samples[outcome.trial.samples.length - 1]!;
      const vec = [probe.c.steer, probe.c.driveForce, probe.c.brakeForce];
      for (const [k, sim] of Object.entries(sims)) {
        const end = rollModel(sim, start, vec);
        const err = Math.hypot(end.x - truth.x, end.z - truth.z);
        const st = stats[k]!;
        st.sum += err;
        st.n += 1;
        if (err > st.worst) st.worst = err;
      }
    }
  }
  const out: Record<string, { mean: number; worst: number }> = {};
  for (const [k, st] of Object.entries(stats)) {
    out[k] = { mean: st.sum / Math.max(1, st.n), worst: st.worst };
  }
  return out;
}

function loadV2(repoRoot: string): LearnedVehicleModel | undefined {
  const p = resolve(repoRoot, 'demos/public/models/v2-default.json');
  if (!existsSync(p)) return undefined;
  const payload = JSON.parse(readFileSync(p, 'utf8'));
  const ensemble: MLP[] = (payload.residualEnsembleJson ?? []).map((j: string) => deserializeMLP(j));
  return {
    params: payload.params,
    config: payload.config,
    residualEnsemble: ensemble,
    residualReferenceDt: payload.residualReferenceDt ?? 0.1,
    oodStdThreshold: payload.oodStdThreshold,
  };
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    allowPositionals: true,
    options: {
      profile: { type: 'string', default: 'default' },
      seed: { type: 'string', default: '42' },
      trials: { type: 'string' },
      ticks: { type: 'string' },
      epochs: { type: 'string' },
      out: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm exec tsx demos/scripts/train-v3.ts
       [--profile=quick|default] [--seed=N] [--trials=N] [--ticks=N]
       [--epochs=N] [--out=path/to/v3-default.json]\n`);
    return;
  }
  const profileName = String(values.profile);
  const profile = PROFILES[profileName];
  if (!profile) throw new Error(`Unknown profile "${profileName}"`);
  const seed = Number(values.seed);
  const trialsCount = values.trials ? Number(values.trials) : profile.trials;
  const ticks = values.ticks ? Number(values.ticks) : profile.ticks;
  const epochs = values.epochs ? Number(values.epochs) : profile.epochs;

  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const outArg = values.out ? String(values.out) : 'demos/public/models/v3-default.json';
  const outPath = isAbsolute(outArg) ? outArg : resolve(repoRoot, outArg);

  process.stdout.write(
    `kinocat train-v3 — profile=${profileName} seed=${seed} trials=${trialsCount} ` +
    `ticks=${ticks} epochs=${epochs} hidden=[${profile.hidden}] ensemble=${profile.ensemble}\n`,
  );
  const t0 = Date.now();

  // --- Collect: per-tick plant transitions across the maneuver bundle ----
  const harness = await createHeadlessTrialHarness({
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    groundBounds: { x0: -1000, x1: 1000, z0: -1000, z1: 1000 },
    offArenaThreshold: 2000,
  });
  const bundle = buildDefaultManeuverBundle({ count: trialsCount, seed });
  const { collected, discarded } = await collectManeuverBatch(
    harness,
    bundle,
    { ticks, sampleEveryNTicks: 1 }, // per-tick: every pair = one exact plant transition
    0,
    SPEED_SCHEDULE,
  );
  process.stdout.write(
    `collected ${collected.length} trials (${discarded} discarded) — ` +
    `${collected.reduce((a, t) => a + t.samples.length - 1, 0)} transitions\n`,
  );

  // --- Fit --------------------------------------------------------------
  const fit = runDynamicsV3Fit({
    trials: collected as CarTrial[],
    hiddenDims: profile.hidden,
    ensembleSize: profile.ensemble,
    seed,
    epochs,
    onProgress: (e) =>
      process.stdout.write(
        `  epoch ${String(e.epoch).padStart(3)}  train=${e.trainLoss.toExponential(3)}  val=${e.valLoss.toExponential(3)}\n`,
      ),
  });
  process.stdout.write(
    `fit done — ${fit.trainPairs} train / ${fit.valPairs} val pairs\n` +
    `  val RMS per step: dFwd=${fit.valRmsRaw[0]!.toExponential(2)}m ` +
    `dRight=${fit.valRmsRaw[1]!.toExponential(2)}m dHead=${fit.valRmsRaw[2]!.toExponential(2)}rad ` +
    `dSpd=${fit.valRmsRaw[3]!.toExponential(2)}m/s dYaw=${fit.valRmsRaw[4]!.toExponential(2)}rad/s ` +
    `dLat=${fit.valRmsRaw[5]!.toExponential(2)}m/s\n`,
  );

  // --- Evaluate: endpoint fidelity vs plant, side-by-side with v2 --------
  const sims: Record<string, ForwardSim<CarKinematicState>> = {
    v3: forwardSimV3(fit.model),
  };
  const v2 = loadV2(repoRoot);
  if (v2) sims['v2-trained'] = learnedForwardSimV2(v2);
  const fidelity = evaluateFidelity(harness, sims);
  for (const [k, f] of Object.entries(fidelity)) {
    process.stdout.write(
      `  fidelity(0.8s endpoint) ${k.padEnd(11)} mean=${f.mean.toFixed(3)}m worst=${f.worst.toFixed(3)}m\n`,
    );
  }

  // Full-throttle launch comparison — the channel the v2 backbone got 2× wrong.
  const launch = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: 0 },
    controlsTrace: new Array(Math.round(3 / PHYSICS_DT)).fill({ steer: 0, driveForce: DEFAULT_VEHICLE_OPTS.engineForce, brakeForce: 0 }),
    sampleEveryNTicks: 60,
    id: 'launch-eval',
  });
  if (launch.ok) {
    const plantAt = (s: number) => launch.trial.samples[s]!.speed;
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, yawRate: 0, lateralVelocity: 0, t: 0 };
    const v3sim = sims['v3']!;
    const v3At: number[] = [0];
    for (let sec = 0; sec < 3; sec++) {
      for (let i = 0; i < 60; i++) s = v3sim(s, [0, DEFAULT_VEHICLE_OPTS.engineForce, 0], PHYSICS_DT);
      v3At.push(s.speed);
    }
    process.stdout.write(
      `  full-throttle from rest: plant 1s=${plantAt(1).toFixed(1)} 2s=${plantAt(2).toFixed(1)} 3s=${plantAt(3).toFixed(1)} | ` +
      `v3 1s=${v3At[1]!.toFixed(1)} 2s=${v3At[2]!.toFixed(1)} 3s=${v3At[3]!.toFixed(1)}\n`,
    );
  }
  harness.dispose();

  // --- Persist ------------------------------------------------------------
  const meta = {
    profile: profileName,
    seed,
    trials: collected.length,
    ticks,
    epochs,
    fidelity,
    valRmsRaw: fit.valRmsRaw,
    createdAt: Date.now(),
  };
  const payload = v3ToJson(fit.model, meta);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(payload, null, 2), 'utf-8');
  process.stdout.write(`wrote ${outPath}\n`);

  const manifest = {
    version: 1,
    kind: 'v3-dynamics',
    profile: profileName,
    seed,
    trials: collected.length,
    ticks,
    epochs,
    hidden: profile.hidden,
    ensemble: profile.ensemble,
    speedSchedule: SPEED_SCHEDULE,
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    fidelity,
    valRmsRaw: fit.valRmsRaw,
    finalTrainLoss: fit.finalTrainLoss,
    finalValLoss: fit.finalValLoss,
    git: gitSha(),
    runtimeMs: Date.now() - t0,
    createdAt: Date.now(),
  };
  const manifestPath = outPath.replace(/\.json$/, '.manifest.json');
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf-8');
  process.stdout.write(`wrote ${manifestPath} (${((Date.now() - t0) / 60000).toFixed(1)} min)\n`);
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(`train-v3 failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`);
    process.exit(1);
  },
);
