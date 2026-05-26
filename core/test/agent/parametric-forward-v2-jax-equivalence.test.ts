// TS half of the equivalence guard described in
// `demos/scripts/python/test_equivalence.py`. Pins the smooth TS
// implementation against the same golden file the JAX trainer is pinned
// to. If either side moves, both tests fail until the golden is
// regenerated deliberately (`python -m test_equivalence --write-golden`).
//
// Tolerance: 1e-3 per component. Looser than the Python side's 1e-6
// because Math.tanh / Math.log1p have platform-dependent rounding at
// extreme inputs, while NumPy / JAX uses libm directly. The two values
// are well below per-trial Rapier measurement noise.

import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  parametricForwardV2Smooth,
  paramsV2FromVec,
  DEFAULT_LEARNABLE_CONFIG,
  type CarKinematicState,
} from 'kinocat/agent';

interface GoldenCase {
  state: number[];
  controls: number[];
  dt: number;
  expected: number[];
}

interface GoldenFile {
  schema: number;
  params: number[];
  param_names: string[];
  config: number[];
  config_names: string[];
  cases: GoldenCase[];
}

const here = resolve(fileURLToPath(import.meta.url), '..');
const goldenPath = resolve(here, 'parametric-forward-v2-golden.json');

describe('parametricForwardV2Smooth — JAX equivalence (golden)', () => {
  if (!existsSync(goldenPath)) {
    it.skip('golden file not yet generated — run `python -m test_equivalence --write-golden`', () => {});
    return;
  }
  const golden = JSON.parse(readFileSync(goldenPath, 'utf-8')) as GoldenFile;
  const params = paramsV2FromVec(golden.params);
  const config = { ...DEFAULT_LEARNABLE_CONFIG };
  // Golden config encodes [chassisMass, wheelBase, frictionSlip].
  config.chassisMass = golden.config[0]!;
  config.wheelBase = golden.config[1]!;
  config.frictionSlip = golden.config[2]!;
  const sim = parametricForwardV2Smooth(params, config);

  it('matches the JAX-generated golden for every pinned case', () => {
    for (let i = 0; i < golden.cases.length; i++) {
      const c = golden.cases[i]!;
      const init: CarKinematicState = {
        x: c.state[0]!, z: c.state[1]!, heading: c.state[2]!, speed: c.state[3]!,
        yawRate: c.state[4]!, lateralVelocity: c.state[5]!, t: c.state[6]!,
      };
      const next = sim(init, c.controls, c.dt);
      const got = [
        next.x, next.z, next.heading, next.speed,
        next.yawRate ?? 0, next.lateralVelocity ?? 0, next.t,
      ];
      for (let k = 0; k < 7; k++) {
        const diff = Math.abs(got[k]! - c.expected[k]!);
        if (diff > 1e-3) {
          throw new Error(
            `Case ${i}, component ${k}: got ${got[k]} expected ${c.expected[k]} (diff ${diff})`,
          );
        }
        expect(diff).toBeLessThan(1e-3);
      }
    }
  });
});
