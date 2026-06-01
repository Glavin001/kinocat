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
  computeActionComparison,
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

describe('computeActionComparison', () => {
  it('returns null when there is no ground truth yet', () => {
    const { lib, model } = buildLibFromModel(10);
    const summary = computeActionComparison({
      full: lib.primitives, parametric: lib.primitives, groundTruth: [],
      model, startSpeed: 10,
    });
    expect(summary).toBeNull();
  });

  it('measures full/parametric endpoint RMS and sorts actions worst-first', () => {
    const { lib, model } = buildLibFromModel(10);
    // Ground truth on the model endpoints EXCEPT index 1, offset 1m — that
    // action should sort to the top.
    const gt: GroundTruthDot[] = lib.primitives.map((p, i) => ({
      index: i, dx: p.end.dx + (i === 1 ? 1.0 : 0), dz: p.end.dz,
    }));
    const summary = computeActionComparison({
      full: lib.primitives, parametric: lib.primitives, groundTruth: gt,
      model, startSpeed: 10,
    });
    expect(summary).not.toBeNull();
    expect(summary!.count).toBe(lib.primitives.length);
    // full === parametric here → equal RMS, no net help.
    expect(summary!.fullRmsM).toBeCloseTo(summary!.paraRmsM, 9);
    expect(summary!.fullRmsM).toBeCloseTo(1 / Math.sqrt(lib.primitives.length), 6);
    // Parametric-only model → no ensemble → gate never fires.
    expect(summary!.actions.every((a) => a.gate === false)).toBe(true);
    // Worst action is the 1m-offset one, classified confident-bias (gate off).
    expect(summary!.actions[0]!.fullErrM).toBeCloseTo(1.0, 6);
    expect(summary!.actions[0]!.verdict).toBe('confident-bias');
    // The on-truth actions are accurate.
    expect(summary!.accurate).toBe(lib.primitives.length - 1);
    expect(summary!.confidentBias).toBe(1);
  });

  it('flags the residual as harmful when it moves endpoints away from truth', () => {
    const { lib: paraLib } = buildLibFromModel(10);
    // GT 0.1m off parametric; "full" 0.5m off (residual pushed it further).
    const gt: GroundTruthDot[] = paraLib.primitives.map((p, i) => ({
      index: i, dx: p.end.dx + 0.1, dz: p.end.dz,
    }));
    const full = paraLib.primitives.map((p) => ({
      ...p, end: { ...p.end, dx: p.end.dx + 0.6 },
    }));
    const model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const summary = computeActionComparison({
      full, parametric: paraLib.primitives, groundTruth: gt, model, startSpeed: 10,
    });
    expect(summary!.paraRmsM).toBeCloseTo(0.1, 6);
    expect(summary!.fullRmsM).toBeCloseTo(0.5, 6);
    expect(summary!.residualHelpPct).toBeLessThan(0); // net harmful
    // Every action's residual delta is negative (worse than parametric).
    expect(summary!.actions.every((a) => a.residualDeltaPct < 0)).toBe(true);
  });
});
