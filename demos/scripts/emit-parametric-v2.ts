// One-off: emit a parametric-only LearnedVehicleModel as `v2-default.json`.
//
// Why this exists: the residual-MLP-trained model overfits and produces
// catastrophic out-of-distribution predictions (sim-to-real free-drive
// trace: v2-full RMS = 14.87 m vs parametric backbone RMS = 0.82 m).
// Until the training data covers a wider state distribution AND the
// MLP is regularised to fall back to parametric outside its support,
// the parametric backbone IS the most honest "v2" model — it's
// chassis-physics-aware (16-coefficient friction-circle model) without
// the brittle residual.
//
// `learnedForwardSimV2` already short-circuits to `parametricForwardV2`
// when `residualEnsemble.length === 0`, so consumers (planner library
// builder, MPPI rollout, sim-to-real ghost) automatically use the
// parametric backbone with no other code changes.

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
} from 'kinocat/agent';
import { modelToJson } from '../app/lib/v2-model-file';

const model = buildParametricOnlyModel(
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
);

const payload = modelToJson(model, {
  trialsUsed: 0,
  openLoopRmsAt1s: 0,
  createdAt: Date.now(),
});

const __dirname = dirname(fileURLToPath(import.meta.url));
const out = resolve(__dirname, '..', 'public', 'models', 'v2-default.json');
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(payload, null, 2));
process.stdout.write(`wrote ${out}\n`);

const manifest = {
  version: 1,
  source: 'parametric-only-backbone',
  rounds: 0,
  trialsPerRound: 0,
  totalTrials: 0,
  notes: 'See header of scripts/emit-parametric-v2.ts for rationale.',
  createdAt: Date.now(),
};
const manifestPath = out.replace(/\.json$/, '.manifest.json');
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
process.stdout.write(`wrote ${manifestPath}\n`);
