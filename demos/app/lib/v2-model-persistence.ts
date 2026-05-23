// localStorage persistence + JSON export/import for the v2 trained model.
// Lightweight: stores the parametric params + LearnableVehicleConfig. The
// residual MLP ensemble is OUT of scope here (large weight matrices; load
// from a separate slot if/when residual training is wired in).

import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  buildParametricOnlyModel,
  type LearnedVehicleParamsV2,
  type LearnableVehicleConfig,
  type LearnedVehicleModel,
} from 'kinocat/agent';

const STORAGE_KEY = 'kinocat:v2-learned-model:v1';

export interface PersistedV2Model {
  version: 1;
  params: LearnedVehicleParamsV2;
  config: LearnableVehicleConfig;
  /** Headline diagnostic at save time, for the UI to show "trained on …". */
  meta: {
    trialsUsed: number;
    openLoopRmsAt1s: number;
    legacyRmsAt1s?: number;
    kinematicRmsAt1s?: number;
    createdAt: number;
  };
}

export function loadV2Model(): { model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw) as PersistedV2Model;
    if (obj.version !== 1) return null;
    return {
      model: buildParametricOnlyModel(obj.params ?? DEFAULT_LEARNED_PARAMS_V2, obj.config ?? DEFAULT_LEARNABLE_CONFIG),
      meta: obj.meta,
    };
  } catch {
    return null;
  }
}

export function saveV2Model(model: LearnedVehicleModel, meta: PersistedV2Model['meta']): void {
  if (typeof window === 'undefined') return;
  try {
    const payload: PersistedV2Model = {
      version: 1,
      params: model.params,
      config: model.config,
      meta,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Quota or serialization failure — silently ignore; the live model still works.
  }
}

export function clearV2Model(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

/** Build a downloadable Blob URL for the persisted v2 model JSON. Caller is
 *  responsible for revoking the URL after the download click finishes. */
export function buildV2ModelDownloadUrl(model: LearnedVehicleModel, meta: PersistedV2Model['meta']): string {
  const payload: PersistedV2Model = {
    version: 1,
    params: model.params,
    config: model.config,
    meta,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  return URL.createObjectURL(blob);
}

/** Parse an imported JSON file string back into a model. Returns null on
 *  any validation failure (unknown version, missing fields). */
export function importV2ModelFromText(text: string): { model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null {
  try {
    const obj = JSON.parse(text) as PersistedV2Model;
    if (obj.version !== 1) return null;
    if (!obj.params || !obj.config) return null;
    return {
      model: buildParametricOnlyModel(obj.params, obj.config),
      meta: obj.meta ?? { trialsUsed: 0, openLoopRmsAt1s: 0, createdAt: Date.now() },
    };
  } catch {
    return null;
  }
}
