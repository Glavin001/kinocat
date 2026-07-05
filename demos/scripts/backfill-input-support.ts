// One-shot: backfill the `inputSupport` coverage stats into an existing
// trained model JSON that predates the coverage OOD gate.
//
// New models get `inputSupport` populated at train time (see
// training-driver). This script reconstructs a representative cloud of the
// MLP inputs the shipped model was trained on — by re-running the SAME
// default maneuver bundle + start-speed schedule the overnight profile uses —
// computes the support stats, and patches them into the JSON in place. The
// rest of the payload (params, config, residual ensemble, meta) is untouched.
//
// Run: cd demos && npx tsx scripts/backfill-input-support.ts
//   [--model=public/models/v2-default.json] [--quantile=0.995]

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { buildMLPInput, computeInputSupport } from 'kinocat/agent';
import type { CarKinematicState, LearnableVehicleConfig, WheeledCarControls } from 'kinocat/agent';
import { createHeadlessTrialHarness } from 'kinocat/adapters/rapier';
import {
  DEFAULT_VEHICLE_OPTS,
  buildDefaultManeuverBundle,
  collectManeuverBatch,
} from '../app/lib/training-driver';
import { deriveLearnableConfig } from 'kinocat/adapters/rapier';

// Mirror the overnight profile's collection knobs so the reconstructed input
// distribution matches what the shipped model was actually fit on.
const TICKS = 180;
const SAMPLE_EVERY = 6;
const SPEED_SCHEDULE = [0, 4, 8, 12, 16, 20, 24, 28];
const SEED = 42;
const ROUNDS = 3; // a representative sample of the 12-round distribution
const PER_ROUND = 300;

function controlsToVec(c: WheeledCarControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      model: { type: 'string', default: 'public/models/v2-default.json' },
      quantile: { type: 'string', default: '0.995' },
    },
  });
  const demosRoot = resolve(fileURLToPath(new URL('..', import.meta.url)));
  const modelPath = isAbsolute(String(values.model)) ? String(values.model) : resolve(demosRoot, String(values.model));
  const quantile = Number(values.quantile);

  const payload = JSON.parse(readFileSync(modelPath, 'utf-8'));
  if (!payload.residualEnsembleJson?.length) {
    process.stdout.write('Model has no residual ensemble — coverage gate is a no-op; nothing to backfill.\n');
    return;
  }

  const config = deriveLearnableConfig({ id: 'backfill', position: { x: 0, z: 0 }, heading: 0, ...DEFAULT_VEHICLE_OPTS }) as LearnableVehicleConfig;
  const harness = await createHeadlessTrialHarness({ vehicleOptions: DEFAULT_VEHICLE_OPTS });

  const inputs: number[][] = [];
  let collectedCount = 0;
  for (let round = 0; round < ROUNDS; round++) {
    const bundle = buildDefaultManeuverBundle({ count: PER_ROUND, seed: SEED + round * 17 });
    const { collected } = await collectManeuverBatch(
      harness, bundle,
      { ticks: TICKS, sampleEveryNTicks: SAMPLE_EVERY },
      collectedCount, SPEED_SCHEDULE,
    );
    collectedCount += collected.length;
    for (const t of collected) {
      const cfg = t.config ?? config;
      for (const sample of t.samples) {
        const tick = Math.min(t.controlsTrace.length - 1, Math.max(0, Math.round(sample.t / t.dt)));
        const c = t.controlsTrace[tick];
        if (!c) continue;
        inputs.push(buildMLPInput(sample.state, controlsToVec(c), cfg));
      }
    }
    process.stdout.write(`  round ${round + 1}/${ROUNDS}: ${collected.length} trials → ${inputs.length} inputs so far\n`);
  }
  harness.dispose();

  const support = computeInputSupport(inputs, quantile);
  if (!support) {
    process.stderr.write('No inputs collected — aborting.\n');
    process.exit(1);
  }
  payload.inputSupport = support;
  writeFileSync(modelPath, JSON.stringify(payload, null, 2), 'utf-8');
  process.stdout.write(
    `\nBackfilled inputSupport into ${modelPath}\n` +
      `  inputs=${inputs.length}  dims=${support.mean.length}  threshold(p${(quantile * 100).toFixed(1)})=${support.threshold.toFixed(3)}\n`,
  );
}

main().then(
  () => process.exit(0),
  (err) => { process.stderr.write(`backfill failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`); process.exit(1); },
);
