// The shipped preloaded model artifact must be consistent with the
// CURRENT model contract: params inside the fit bounds and a config that
// matches the reference chassis derivation. An artifact trained under
// older, looser bounds is honest about being stale via the failing test
// below (repo convention: known-broken behavior stays red with
// `it.fails` until actually fixed — here, by re-fitting and re-emitting
// `public/models/v2-default.json`).

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  paramsV2OutOfBounds,
  type LearnedVehicleParamsV2,
  type LearnableVehicleConfig,
} from 'kinocat/agent';

const ARTIFACT_PATH = join(__dirname, '..', 'public', 'models', 'v2-default.json');

function loadArtifact(): { params: LearnedVehicleParamsV2; config: LearnableVehicleConfig } {
  return JSON.parse(readFileSync(ARTIFACT_PATH, 'utf8'));
}

describe('shipped v2-default.json artifact', () => {
  // KNOWN STALE: the shipped artifact was fit when PARAMS_V2_HI allowed
  // brakeScale 3.5 / engineScale 1.2 / steerRatio 1.35; the bounds were
  // later tightened on physical-plausibility grounds but the artifact
  // was never re-fit. Loading is still permitted (the residual ensemble
  // was trained around these exact backbone values, so clamping would
  // degrade it) — but the artifact should be regenerated. Flip this to
  // `it` when a re-fit artifact lands.
  it.fails('params are within the current fit bounds (requires artifact re-fit)', () => {
    const { params } = loadArtifact();
    expect(paramsV2OutOfBounds(params)).toEqual([]);
  });

  it('embeds the derived reference-chassis mass (not the hand-copied one)', () => {
    const { config } = loadArtifact();
    // 8 * 2.4 * 0.5 * 1.0 halfExtents volume * density 60 = 576 kg.
    expect(config.chassisMass).toBeCloseTo(576, 6);
  });
});
