// localStorage persistence + JSON export/import for the v2 trained model.
// Stores the parametric params + LearnableVehicleConfig + residual MLP
// ensemble (each MLP is serialized via core's serializeMLP).
//
// Version history:
//   v1  → no residual ensemble (parametric-only legacy payload).
//   v2  → adds residualEnsembleJson + residualReferenceDt. MLP input was
//         22 dims: [x, z, heading, speed, yawRate, lateralVel, steer,
//         drive, brake, config13]. Position-dependent (BROKEN).
//   v3  → MLP input is 21 dims: [sin h, cos h, speed, yawRate, lateralVel,
//         steer, drive, brake, config13]. Position-invariant +
//         heading-wrap-symmetric. v2 payloads CANNOT be re-used because
//         the MLP layer dimensions differ — the loader rejects them
//         with a clear retrain prompt rather than silently producing
//         garbage (the same kind of failure the sim-to-real free-drive
//         trace exposed on v2 payloads).

import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  buildParametricOnlyModel,
  type LearnedVehicleParamsV2,
  type LearnableVehicleConfig,
  type LearnedVehicleModel,
} from 'kinocat/agent';

const STORAGE_KEY = 'kinocat:v2-learned-model:v3';
const STORAGE_KEY_LEGACY = 'kinocat:v2-learned-model:v1';
const STORAGE_KEY_V2 = 'kinocat:v2-learned-model:v2';

export interface PersistedV2Model {
  version: 3;
  params: LearnedVehicleParamsV2;
  config: LearnableVehicleConfig;
  /** Residual MLP ensemble — each entry is a `serializeMLP(mlp)` JSON
   *  string. Empty array when the model has no residual ensemble. */
  residualEnsembleJson: string[];
  /** Reference dt the residual was trained against (seconds). */
  residualReferenceDt: number;
  /** Per-output OOD threshold (length 6). Inference falls back to
   *  parametric prediction when ensemble std on any output exceeds
   *  the corresponding threshold. Optional — sensible defaults applied
   *  by the loader. */
  oodStdThreshold?: number[];
  /** Coverage gate: training-input distribution stats + threshold. The
   *  primary OOD trigger (distance-to-training-support); inference falls back
   *  to parametric for queries outside it. Optional — absent on legacy
   *  models, which keep variance-only gating. */
  inputSupport?: import('kinocat/agent').InputSupport;
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
  if (obj.version === 3) return obj as PersistedV2Model;
  if (obj.version === 2) {
    // v2 payloads contain residual MLPs whose first layer expects the
    // OBSOLETE 22-dim input (with absolute position). They cannot be
    // loaded into the 21-dim v3 architecture without retraining; doing
    // so produces destructive predictions (the failure mode the
    // sim-to-real debug exposed). Discard the residual ensemble and
    // fall back to parametric — equivalent to "this model needs to be
    // retrained against the new input layout."
    const v2 = obj as {
      params: LearnedVehicleParamsV2;
      config: LearnableVehicleConfig;
      meta: PersistedV2Model['meta'];
      residualReferenceDt?: number;
    };
    return {
      version: 3,
      params: v2.params,
      config: v2.config,
      residualEnsembleJson: [],
      residualReferenceDt: v2.residualReferenceDt ?? 0.1,
      meta: v2.meta,
    };
  }
  if (obj.version === 1) {
    const v1 = obj as PersistedV2ModelV1;
    return {
      version: 3,
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
      oodStdThreshold: payload.oodStdThreshold,
      inputSupport: payload.inputSupport,
    };
  }
  return base;
}

export function loadV2Model(): { model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null {
  if (typeof window === 'undefined') return null;
  try {
    // Prefer the current (v3) slot; fall back to v2 then v1 for legacy
    // users. Both fallbacks discard the residual ensemble during
    // migration — see comments on `migrate`.
    let raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) raw = window.localStorage.getItem(STORAGE_KEY_V2);
    if (!raw) raw = window.localStorage.getItem(STORAGE_KEY_LEGACY);
    if (!raw) return null;
    const migrated = migrate(JSON.parse(raw));
    if (!migrated) return null;
    return { model: rebuildModel(migrated), meta: migrated.meta };
  } catch {
    return null;
  }
}

/** Default location the headless training CLI (`pnpm run train`) writes
 *  the preloaded model artifact to. Served by Next.js from `public/`. */
export const DEFAULT_PRELOADED_V2_URL = '/models/v2-default.json';

/** Fetch + rebuild a v2 model from a URL (typically the preloaded
 *  `/models/v2-default.json` artifact). Returns null on any error
 *  (404, parse failure, unknown version) so demo pages can fall back
 *  to default-parametric without ceremony. */
export async function loadV2ModelFromUrl(
  url: string = DEFAULT_PRELOADED_V2_URL,
): Promise<{ model: LearnedVehicleModel; meta: PersistedV2Model['meta'] } | null> {
  if (typeof window === 'undefined') return null;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const migrated = migrate(await res.json());
    if (!migrated) return null;
    return { model: rebuildModel(migrated), meta: migrated.meta };
  } catch {
    return null;
  }
}

/** Load the cached v2 model from localStorage; if absent, fall back to
 *  the preloaded artifact at `url`. Cache hit is synchronous; the
 *  preloaded fetch is async. Callers typically wire this into a mount
 *  effect so the first paint shows the parametric baseline and the
 *  preloaded model lands a tick later. */
export async function loadV2ModelWithFallback(
  url: string = DEFAULT_PRELOADED_V2_URL,
): Promise<{ model: LearnedVehicleModel; meta: PersistedV2Model['meta']; source: 'localStorage' | 'preloaded' } | null> {
  const cached = loadV2Model();
  if (cached) return { ...cached, source: 'localStorage' };
  const preloaded = await loadV2ModelFromUrl(url);
  if (preloaded) return { ...preloaded, source: 'preloaded' };
  return null;
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
    version: 3,
    params: model.params,
    config: model.config,
    residualEnsembleJson: ensembleJson,
    residualReferenceDt: model.residualReferenceDt ?? 0.1,
    oodStdThreshold: model.oodStdThreshold,
    inputSupport: model.inputSupport,
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
