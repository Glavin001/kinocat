// Node-side loader for the shipped trained v2 artifact
// (`demos/public/models/v2-default.json`). The browser pages load the
// same payload via `loadV2ModelFromUrl`; tests and CLI benchmarks read
// it straight from disk so headless runs use the exact model the demo
// pages run with.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { deserializeMLP, type LearnedVehicleModel, type MLP } from 'kinocat/agent';

export const TRAINED_V2_ARTIFACT_PATH = join(
  __dirname, '..', '..', 'public', 'models', 'v2-default.json',
);

export function loadTrainedV2FromDisk(
  path: string = TRAINED_V2_ARTIFACT_PATH,
): LearnedVehicleModel {
  const payload = JSON.parse(readFileSync(path, 'utf8'));
  const ensemble: MLP[] = (payload.residualEnsembleJson ?? []).map(
    (j: string) => deserializeMLP(j),
  );
  return {
    params: payload.params,
    config: payload.config,
    residualEnsemble: ensemble,
    residualReferenceDt: payload.residualReferenceDt ?? 0.1,
    oodStdThreshold: payload.oodStdThreshold,
  };
}
