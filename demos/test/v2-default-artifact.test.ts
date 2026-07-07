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
  // The artifact is now regenerated reproducibly (fixed seed, current
  // bounds, grip-saturating brake model) by `pnpm run train` — see the
  // manifest sidecar for the git SHA / seed / trial count. Its params are
  // within the current fit bounds, so this is a passing `it` (flipped from
  // the historical `it.fails` that recorded the stale overnight artifact).
  it('params are within the current fit bounds', () => {
    const { params } = loadArtifact();
    expect(paramsV2OutOfBounds(params)).toEqual([]);
  });

  it('embeds the derived reference-chassis mass (not the hand-copied one)', () => {
    const { config } = loadArtifact();
    // 8 * 2.4 * 0.5 * 1.0 halfExtents volume * density 60 = 576 kg.
    expect(config.chassisMass).toBeCloseTo(576, 6);
  });
});
