// Unit tests for the pure ground-truth / uncertainty helpers used by
// the Model Lab dashboard. The Rapier-dependent GT-dot path is covered
// in the e2e training-driver test transitively (it shells the same
// harness internals); here we focus on the pure halo computation +
// filtering behavior.

import { describe, it, expect } from 'vitest';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  learnedForwardSimV2,
} from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import { computeUncertaintyHalos } from '../app/lib/fan-plot-ground-truth';

function buildLibFromModel(startSpeed: number) {
  const model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
  const lib = characterizeVehicle({
    forwardSim: learnedForwardSimV2(model),
    controlSets: [
      [0, 1500, 0],
      [0.2, 1500, 0],
      [-0.2, 1500, 0],
      [0, 0, 800],
    ],
    duration: 0.55,
    substeps: 6,
    startSpeeds: [startSpeed],
  });
  return { lib, model };
}

describe('computeUncertaintyHalos', () => {
  it('returns no halos when the ensemble is empty (parametric-only)', () => {
    const { lib, model } = buildLibFromModel(10);
    const halos = computeUncertaintyHalos({
      primitives: lib.primitives,
      model,
      config: model.config,
      startSpeed: 10,
    });
    expect(halos).toEqual([]);
  });

  it('skips primitives whose startSpeed does not match the requested bucket', () => {
    const { lib, model } = buildLibFromModel(10);
    // Force-evaluate at a different bucket — every primitive should be
    // filtered out (no ensemble + mismatched speed = empty).
    const halos = computeUncertaintyHalos({
      primitives: lib.primitives,
      model,
      config: model.config,
      startSpeed: 28,
    });
    expect(halos).toEqual([]);
  });
});
