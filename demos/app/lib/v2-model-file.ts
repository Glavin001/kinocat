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
    version: 2,
    params: model.params,
    config: model.config,
    residualEnsembleJson,
    residualReferenceDt: model.residualReferenceDt ?? 0.1,
    meta,
  };
}

export function modelFromJson(payload: PersistedV2Model): LearnedVehicleModel {
  const base = buildParametricOnlyModel(
    payload.params ?? DEFAULT_LEARNED_PARAMS_V2,
    payload.config ?? DEFAULT_LEARNABLE_CONFIG,
  );
  if (payload.residualEnsembleJson?.length) {
    return {
      ...base,
      residualEnsemble: payload.residualEnsembleJson.map((j) => deserializeMLP(j)),
      residualReferenceDt: payload.residualReferenceDt ?? 0.1,
    };
  }
  return base;
}
