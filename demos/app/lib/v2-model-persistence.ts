// localStorage persistence + JSON export/import for the v2 trained model.
// Stores the parametric params + LearnableVehicleConfig + residual MLP
// ensemble (each MLP is serialized via core's serializeMLP). Bumped to
// version 2; v1 cached models load with empty ensemble (parametric only).

import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  buildParametricOnlyModel,
  type LearnedVehicleParamsV2,
  type LearnableVehicleConfig,
  type LearnedVehicleModel,
} from 'kinocat/agent';

const STORAGE_KEY = 'kinocat:v2-learned-model:v2';
const STORAGE_KEY_LEGACY = 'kinocat:v2-learned-model:v1';

export interface PersistedV2Model {
  version: 2;
  params: LearnedVehicleParamsV2;
  config: LearnableVehicleConfig;
  /** Residual MLP ensemble — each entry is a `serializeMLP(mlp)` JSON
   *  string. Empty array when the model has no residual ensemble. */
  residualEnsembleJson: string[];
  /** Reference dt the residual was trained against (seconds). */
  residualReferenceDt: number;
  /** Headline diagnostic at save time, for the UI to show "trained on …". */
  meta: {
    trialsUsed: number;
    openLoopRmsAt1s: number;
    legacyRmsAt1s?: number;
    kinematicRmsAt1s?: number;
    createdAt: number;
  };
}

// Backward-compat: v1 payload had no residualEnsemble.
interface PersistedV2ModelV1 {
  version: 1;
  params: LearnedVehicleParamsV2;
  config: LearnableVehicleConfig;
  meta: PersistedV2Model['meta'];
}

function migrate(parsed: unknown): PersistedV2Model | null {
  if (!parsed || typeof parsed !== 'object') return null;
  const obj = parsed as { version?: number };
  if (obj.version === 2) return obj as PersistedV2Model;
  if (obj.version === 1) {
    const v1 = obj as PersistedV2ModelV1;
    return {
      version: 2,
      params: v1.params,
      config: v1.config,
      residualEnsembleJson: [],
      residualReferenceDt: 0.1,
      meta: v1.meta,
    };
  }
  return null;
}

function rebuildModel(payload: PersistedV2Model): LearnedVehicleModel {
  // Lazy import to avoid pulling core MLP deserialization on cold paths.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { deserializeMLP } = require('kinocat/agent') as {
    deserializeMLP?: (s: string) => unknown;
  };
  const base = buildParametricOnlyModel(
    payload.params ?? DEFAULT_LEARNED_PARAMS_V2,
    payload.config ?? DEFAULT_LEARNABLE_CONFIG,
  );
  if (deserializeMLP && payload.residualEnsembleJson?.length) {
    return {
      ...base,
      residualEnsemble: payload.residualEnsembleJson.map((j) => deserializeMLP(j) as never),
      residualReferenceDt: payload.residualReferenceDt ?? 0.1,
    };
  }
  return base;
}

export function loadV2Model(): { model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null {
  if (typeof window === 'undefined') return null;
  try {
    // Prefer the v2 slot; fall back to v1 for legacy users.
    let raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) raw = window.localStorage.getItem(STORAGE_KEY_LEGACY);
    if (!raw) return null;
    const migrated = migrate(JSON.parse(raw));
    if (!migrated) return null;
    return { model: rebuildModel(migrated), meta: migrated.meta };
  } catch {
    return null;
  }
}

function buildPayload(model: LearnedVehicleModel, meta: PersistedV2Model['meta']): PersistedV2Model {
  // Lazy import to avoid SSR / cold-path issues.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { serializeMLP } = require('kinocat/agent') as {
    serializeMLP?: (mlp: unknown) => string;
  };
  const ensembleJson: string[] = [];
  if (serializeMLP && model.residualEnsemble?.length) {
    for (const mlp of model.residualEnsemble) {
      ensembleJson.push(serializeMLP(mlp));
    }
  }
  return {
    version: 2,
    params: model.params,
    config: model.config,
    residualEnsembleJson: ensembleJson,
    residualReferenceDt: model.residualReferenceDt ?? 0.1,
    meta,
  };
}

export function saveV2Model(model: LearnedVehicleModel, meta: PersistedV2Model['meta']): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(buildPayload(model, meta)));
    // Best-effort cleanup of the legacy slot so cached v1 doesn't shadow
    // a freshly-trained v2.
    window.localStorage.removeItem(STORAGE_KEY_LEGACY);
  } catch {
    // Quota or serialization failure — silently ignore; the live model still works.
  }
}

export function clearV2Model(): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
    window.localStorage.removeItem(STORAGE_KEY_LEGACY);
  } catch {
    // ignore
  }
}

/** Build a downloadable Blob URL for the persisted v2 model JSON. Caller is
 *  responsible for revoking the URL after the download click finishes. */
export function buildV2ModelDownloadUrl(model: LearnedVehicleModel, meta: PersistedV2Model['meta']): string {
  const blob = new Blob([JSON.stringify(buildPayload(model, meta), null, 2)], { type: 'application/json' });
  return URL.createObjectURL(blob);
}

/** Parse an imported JSON file string back into a model. Returns null on
 *  any validation failure (unknown version, missing fields). */
export function importV2ModelFromText(text: string): { model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null {
  try {
    const migrated = migrate(JSON.parse(text));
    if (!migrated) return null;
    return {
      model: rebuildModel(migrated),
      meta: migrated.meta ?? { trialsUsed: 0, openLoopRmsAt1s: 0, createdAt: Date.now() },
    };
  } catch {
    return null;
  }
}
