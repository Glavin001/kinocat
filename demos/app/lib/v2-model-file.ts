// Node-friendly serialization for the v2 learned model. Shares the
// `PersistedV2Model` payload shape with `v2-model-persistence.ts` so the
// browser-side localStorage loader can consume CLI-trained artifacts and
// vice versa — but does NOT touch `window` / `require()` so it works
// inside CLI scripts (`pnpm run train`).

import {
  serializeMLP,
  deserializeMLP,
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  type LearnedVehicleModel,
} from 'kinocat/agent';
import type { PersistedV2Model } from './v2-model-persistence';

export function modelToJson(
  model: LearnedVehicleModel,
  meta: PersistedV2Model['meta'],
): PersistedV2Model {
  const residualEnsembleJson: string[] = [];
  for (const mlp of model.residualEnsemble ?? []) {
    residualEnsembleJson.push(serializeMLP(mlp));
  }
  return {
    version: 3,
    params: model.params,
    config: model.config,
    residualEnsembleJson,
    residualReferenceDt: model.residualReferenceDt ?? 0.1,
    oodStdThreshold: model.oodStdThreshold,
    meta,
  };
}

export function modelFromJson(payload: PersistedV2Model | { version?: number; [k: string]: unknown }): LearnedVehicleModel {
  // Reject obsolete v2 payloads' residual ensembles: their MLP first
  // layer expects the old 22-dim input (with absolute position) and
  // would produce destructive predictions in v3's 21-dim layout. Drop
  // the ensemble during load — model becomes parametric-only, with a
  // clear retrain prompt surfaced to the caller via the returned
  // model's empty `residualEnsemble`.
  const version = (payload as { version?: number }).version;
  if (version !== 3 && version !== undefined) {
    const p = payload as Partial<PersistedV2Model>;
    return buildParametricOnlyModel(
      p.params ?? DEFAULT_LEARNED_PARAMS_V2,
      p.config ?? DEFAULT_LEARNABLE_CONFIG,
    );
  }
  const p = payload as PersistedV2Model;
  const base = buildParametricOnlyModel(
    p.params ?? DEFAULT_LEARNED_PARAMS_V2,
    p.config ?? DEFAULT_LEARNABLE_CONFIG,
  );
  if (p.residualEnsembleJson?.length) {
    return {
      ...base,
      residualEnsemble: p.residualEnsembleJson.map((j) => deserializeMLP(j)),
      residualReferenceDt: p.residualReferenceDt ?? 0.1,
      oodStdThreshold: p.oodStdThreshold,
    };
  }
  return base;
}
