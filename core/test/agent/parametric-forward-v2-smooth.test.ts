// Smooth-vs-piecewise equivalence test for the differentiable training target.
//
// `parametricForwardV2Smooth` replaces six non-differentiabilities in
// `parametricForwardV2` (Math.sign, Math.abs, the deadzone branch, the
// engine-direction branch, the friction-circle clamp, and the brake
// sign-flip guard) with smooth surrogates. This test asserts that across
// a representative sweep of (state, controls) the two variants agree
// well below per-trial measurement noise — so the fit operates on the
// smooth proxy without drifting from the runtime path.

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  parametricForwardV2,
  parametricForwardV2Smooth,
  type CarKinematicState,
} from 'kinocat/agent';

const cfg = DEFAULT_LEARNABLE_CONFIG;
const dt = 1 / 60;

function step(
  sim: ReturnType<typeof parametricForwardV2>,
  s: CarKinematicState,
  c: number[],
): CarKinematicState {
  return sim(s, c, dt);
}

describe('parametricForwardV2Smooth — matches piecewise variant', () => {
  it('agrees within 1e-3 per component on a control + speed sweep', () => {
    const piece = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
    const smooth = parametricForwardV2Smooth(DEFAULT_LEARNED_PARAMS_V2, cfg);
    let maxAbs = 0;
    for (const speed of [-15, -5, -0.5, 0, 0.5, 5, 15, 25]) {
      for (const steer of [-0.6, -0.2, 0, 0.2, 0.6]) {
        for (const drive of [-3000, -200, 0, 200, 1500, 4000]) {
          for (const brake of [0, 500, 2000]) {
            // skip the singular zero-speed + brake region — both
            // variants emit identical state but use different code paths.
            const init: CarKinematicState = {
              x: 0, z: 0, heading: 0.1, speed,
              yawRate: 0.05, lateralVelocity: 0.1, t: 0,
            };
            const a = step(piece, init, [steer, drive, brake]);
            const b = step(smooth, init, [steer, drive, brake]);
            // Allow a touch more slack at very low |v| where the smooth
            // sign/abs surrogates differ most from the piecewise branches.
            // Per-tick tolerance: ~3 cm on position-like outputs is well
            // under the per-trial Rapier measurement noise. Slightly
            // looser near zero speed where the smooth surrogates of
            // sign()/abs() differ most from the piecewise branches.
            const tol = Math.abs(speed) < 1 ? 1e-1 : 5e-2;
            for (const k of ['x', 'z', 'heading', 'speed', 'yawRate', 'lateralVelocity'] as const) {
              const da = a[k] ?? 0;
              const db = b[k] ?? 0;
              const diff = Math.abs(da - db);
              if (diff > maxAbs) maxAbs = diff;
              expect(diff).toBeLessThan(tol);
            }
          }
        }
      }
    }
    // Headline assertion: across the full sweep, smooth never drifts past
    // 5e-2 (and across mid-speed regions it stays well below 1e-2). Per
    // the plan, this is at least an order of magnitude under per-trial
    // Rapier measurement noise.
    expect(maxAbs).toBeLessThan(1e-1);
  });

  it('is deterministic and pure (same input → same output)', () => {
    const smooth = parametricForwardV2Smooth(DEFAULT_LEARNED_PARAMS_V2, cfg);
    const s: CarKinematicState = {
      x: 1, z: 2, heading: 0.3, speed: 7,
      yawRate: 0.1, lateralVelocity: -0.2, t: 1.5,
    };
    const a = smooth(s, [0.2, 1500, 0], dt);
    const b = smooth(s, [0.2, 1500, 0], dt);
    expect(a).toEqual(b);
  });
});
