// Verify the generic Nelder-Mead fitter recovers parameters from a synthetic
// ForwardSim whose ground-truth is known. Uses a tiny 2-param linear-damping
// model: v_{t+1} = v_t + (a - k*v_t) * dt, position integrated.

import { describe, expect, it } from 'vitest';
import { runParametricFit } from 'kinocat/learning';
import {
  createTrialStore,
  type Trial,
} from 'kinocat/learning';
import type { ForwardSim } from 'kinocat/primitives';

interface MiniState { x: number; v: number; t: number; }
interface MiniParams { accel: number; damping: number; }
interface MiniControls { effort: number; }

function makeSim(p: MiniParams): ForwardSim<MiniState> {
  return (s, controls, dt) => {
    const e = controls[0] ?? 0;
    const v = s.v + (p.accel * e - p.damping * s.v) * dt;
    return { x: s.x + 0.5 * (s.v + v) * dt, v, t: s.t + dt };
  };
}

describe('runParametricFit — recovers known params from synthetic trial data', () => {
  it('recovers (accel, damping) within 5%', () => {
    const truth: MiniParams = { accel: 4.0, damping: 1.2 };
    const trueSim = makeSim(truth);

    // Build 4 trials with constant-effort traces at different effort levels.
    const store = createTrialStore<MiniState, MiniControls, null>();
    const efforts = [0.3, 0.7, 1.0, -0.5];
    for (let i = 0; i < efforts.length; i++) {
      const eff = efforts[i]!;
      let s: MiniState = { x: 0, v: 0, t: 0 };
      const dt = 1 / 30;
      const ticks = 60; // 2 seconds
      const controlsTrace: MiniControls[] = [];
      const samples: Trial<MiniState, MiniControls, null>['samples'] = [{ t: 0, state: { ...s } }];
      for (let k = 0; k < ticks; k++) {
        controlsTrace.push({ effort: eff });
        s = trueSim(s, [eff], dt);
        if ((k + 1) % 10 === 0) samples.push({ t: (k + 1) * dt, state: { ...s } });
      }
      store.add({
        id: `t${i}`, initialState: { x: 0, v: 0, t: 0 }, controlsTrace, dt,
        samples, config: null, configKey: 'cfg',
      });
    }

    const init: MiniParams = { accel: 1, damping: 0.1 };
    const result = runParametricFit<MiniParams, MiniState, MiniControls, null>({
      init,
      encode: (p) => [p.accel, p.damping],
      decode: (v) => ({
        accel: Math.max(0.1, Math.min(10, v[0] ?? 1)),
        damping: Math.max(0, Math.min(5, v[1] ?? 0)),
      }),
      makeSim,
      stateDelta: (pred, act) => {
        const dx = pred.x - act.x;
        const dv = pred.v - act.v;
        return dx * dx + dv * dv;
      },
      trials: store.all(),
      controlsToVec: (c) => [c.effort],
      maxIter: 800,
    });

    expect(Math.abs(result.params.accel - truth.accel) / truth.accel).toBeLessThan(0.05);
    expect(Math.abs(result.params.damping - truth.damping) / truth.damping).toBeLessThan(0.05);
    expect(result.history.length).toBeGreaterThan(5);
    // Monotonically decreasing best loss.
    for (let i = 1; i < result.history.length; i++) {
      expect(result.history[i]!.loss).toBeLessThanOrEqual(result.history[i - 1]!.loss + 1e-9);
    }
  });
});
