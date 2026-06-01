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
import {
  computeUncertaintyHalos,
  computePrimitiveComparisonStats,
} from '../app/lib/fan-plot-ground-truth';
import type { GroundTruthDot } from '../app/lib/fan-plot-ground-truth';

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

describe('computePrimitiveComparisonStats', () => {
  it('returns null when there is no ground truth yet', () => {
    const { lib, model } = buildLibFromModel(10);
    const stats = computePrimitiveComparisonStats({
      full: lib.primitives, parametric: lib.primitives, groundTruth: [],
      model, startSpeed: 10,
    });
    expect(stats).toBeNull();
  });

  it('measures full/parametric endpoint RMS against ground truth and ranks worst', () => {
    const { lib, model } = buildLibFromModel(10);
    // Synthesize ground truth exactly on the parametric endpoints EXCEPT one
    // primitive (index 1) offset by a known amount — para should be perfect,
    // and the offset primitive should sort to the top.
    const gt: GroundTruthDot[] = lib.primitives.map((p, i) => ({
      index: i,
      dx: p.end.dx + (i === 1 ? 1.0 : 0),
      dz: p.end.dz,
    }));
    const stats = computePrimitiveComparisonStats({
      full: lib.primitives, parametric: lib.primitives, groundTruth: gt,
      model, startSpeed: 10, topN: 4,
    });
    expect(stats).not.toBeNull();
    // full === parametric here (same lib), so both RMS match and residual is
    // neither helping nor hurting.
    expect(stats!.count).toBe(lib.primitives.length);
    expect(stats!.fullRmsM).toBeCloseTo(stats!.paraRmsM, 9);
    // Only one primitive is off by 1m → RMS = 1/sqrt(n).
    expect(stats!.fullRmsM).toBeCloseTo(1 / Math.sqrt(lib.primitives.length), 6);
    // Parametric-only model → no ensemble → gate never fires.
    expect(stats!.gateFires).toBe(0);
    // Worst row is the 1m-offset primitive.
    expect(stats!.worst[0]!.fullErrM).toBeCloseTo(1.0, 6);
  });

  it('flags the residual as harmful when it moves endpoints away from truth', () => {
    const { lib: paraLib } = buildLibFromModel(10);
    // Ground truth sits 0.1m off the parametric endpoints; "full" is 0.5m off
    // (residual pushed it further away). The residual is strictly worse →
    // negative help %.
    const gt: GroundTruthDot[] = paraLib.primitives.map((p, i) => ({
      index: i, dx: p.end.dx + 0.1, dz: p.end.dz,
    }));
    const full = paraLib.primitives.map((p) => ({
      ...p, end: { ...p.end, dx: p.end.dx + 0.6 }, // 0.5m beyond the GT offset
    }));
    const model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const stats = computePrimitiveComparisonStats({
      full, parametric: paraLib.primitives, groundTruth: gt, model, startSpeed: 10,
    });
    expect(stats!.paraRmsM).toBeCloseTo(0.1, 6);
    expect(stats!.fullRmsM).toBeCloseTo(0.5, 6);
    expect(stats!.residualHelpPct).toBeLessThan(0); // residual hurts
  });
});
