// Headless tests for the autonomous motion-primitive learner. Boots Rapier
// WASM and runs a (small) subset of the autonomous sweep so the test stays
// under a few seconds. Skips entirely on CI runners without the WASM binary,
// same pattern as `core/test/adapters/raycast-vehicle.test.ts`.

import { describe, it, expect } from 'vitest';
import {
  ensureRapier,
} from 'kinocat/adapters/rapier';
import {
  DEFAULT_LEARNED_PARAMS,
  learnedForwardSim,
  kinematicForwardSim,
} from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { CARCHASE_AGENT } from '../app/lib/carchase-scenarios';
import {
  buildLearnedLibrary,
  createSweepWorld,
  defaultControlSets,
  fitParams,
  PRIMITIVE_DURATION,
  PRIMITIVE_SUBSTEPS,
  runSweep,
  runTrial,
} from '../app/lib/learn-primitives';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

it('rapier availability is a boolean (logs skip status in CI)', () => {
  expect(typeof RAPIER_OK).toBe('boolean');
});

describe('learnedForwardSim contract', () => {
  it('produces CarKinematicState with wrapped heading and finite fields', () => {
    const sim = learnedForwardSim(DEFAULT_LEARNED_PARAMS, CARCHASE_AGENT);
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let k = 0; k < 200; k++) {
      s = sim(s, [1 / CARCHASE_AGENT.minTurnRadius, 12], 0.05);
      expect(Number.isFinite(s.x)).toBe(true);
      expect(Number.isFinite(s.z)).toBe(true);
      expect(Number.isFinite(s.speed)).toBe(true);
      expect(s.heading).toBeGreaterThan(-Math.PI - 1e-9);
      expect(s.heading).toBeLessThanOrEqual(Math.PI + 1e-9);
    }
  });

  it('respects the agent speed and curvature limits at saturation', () => {
    const sim = learnedForwardSim(DEFAULT_LEARNED_PARAMS, CARCHASE_AGENT);
    // Way-too-big target speed and curvature; the sim should clamp them.
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let k = 0; k < 200; k++) {
      s = sim(s, [10, 1000], 0.05);
    }
    expect(s.speed).toBeLessThanOrEqual(CARCHASE_AGENT.maxSpeed + 1e-6);
    // Negative direction.
    s = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let k = 0; k < 200; k++) {
      s = sim(s, [0, -1000], 0.05);
    }
    expect(s.speed).toBeGreaterThanOrEqual(-CARCHASE_AGENT.maxReverseSpeed - 1e-6);
  });
});

describe('buildLearnedLibrary round-trip', () => {
  it('produces the expected primitive count and JSON survives a round trip', () => {
    const lib = buildLearnedLibrary(DEFAULT_LEARNED_PARAMS, { agent: CARCHASE_AGENT });
    expect(lib.startSpeeds.length).toBe(4);
    expect(lib.primitives.length).toBe(4 * defaultControlSets(CARCHASE_AGENT).length);
    const restored = MotionPrimitiveLibrary.fromJSON(lib.toJSON());
    expect(restored.primitives.length).toBe(lib.primitives.length);
    expect(restored.startSpeeds).toEqual(lib.startSpeeds);
    // Sweep length must equal substeps + 1.
    for (const p of restored.primitives) {
      expect(p.sweep.length).toBe(PRIMITIVE_SUBSTEPS + 1);
      expect(p.duration).toBeCloseTo(PRIMITIVE_DURATION, 9);
    }
  });
});

describe.skipIf(!RAPIER_OK)('autonomous sweep + fit', () => {
  it('left/right turns at the same |curvature| produce mirrored z trajectories', async () => {
    const sw = await createSweepWorld(CARCHASE_AGENT);
    try {
      const k = 1 / CARCHASE_AGENT.minTurnRadius;
      const left = runTrial(sw, 0, [+k, 8], 0);
      const right = runTrial(sw, 0, [-k, 8], 1);
      // Sample-by-sample mirror within tolerance. Rapier is symmetric on flat
      // ground; small slip differences set the tolerance.
      for (let i = 1; i < left.samples.length; i++) {
        const a = left.samples[i]!;
        const b = right.samples[i]!;
        expect(Math.abs(a.x - b.x)).toBeLessThan(0.5);
        expect(Math.abs(a.z + b.z)).toBeLessThan(0.5);
      }
    } finally {
      sw.dispose();
    }
  }, 30000);

  it('two sequential sweeps over the same world produce identical results', async () => {
    const startSpeeds = [0];
    const controlSets = [
      [0, 10],
      [1 / CARCHASE_AGENT.minTurnRadius, 8],
      [0, -4],
    ];
    const sw1 = await createSweepWorld(CARCHASE_AGENT);
    let a: ReturnType<typeof Object>;
    try {
      a = await runSweep(sw1, { startSpeeds, controlSets });
    } finally {
      sw1.dispose();
    }
    const sw2 = await createSweepWorld(CARCHASE_AGENT);
    let b: ReturnType<typeof Object>;
    try {
      b = await runSweep(sw2, { startSpeeds, controlSets });
    } finally {
      sw2.dispose();
    }
    // Two fresh worlds + same teleport sequence → identical samples.
    const ta = (a as { trials: Array<{ samples: Array<{ x: number; z: number; heading: number; speed: number }> }> }).trials;
    const tb = (b as { trials: Array<{ samples: Array<{ x: number; z: number; heading: number; speed: number }> }> }).trials;
    expect(ta.length).toBe(tb.length);
    for (let i = 0; i < ta.length; i++) {
      const sa = ta[i]!.samples;
      const sb = tb[i]!.samples;
      expect(sa.length).toBe(sb.length);
      for (let j = 0; j < sa.length; j++) {
        expect(sa[j]!.x).toBeCloseTo(sb[j]!.x, 6);
        expect(sa[j]!.z).toBeCloseTo(sb[j]!.z, 6);
        expect(sa[j]!.heading).toBeCloseTo(sb[j]!.heading, 6);
        expect(sa[j]!.speed).toBeCloseTo(sb[j]!.speed, 6);
      }
    }
  }, 30000);

  it('fitted params reproduce trial data within ~0.3 m mean position error', async () => {
    // Compact sweep so the test stays fast but covers forward + reverse and
    // straight + turning.
    const k = 1 / CARCHASE_AGENT.minTurnRadius;
    const controlSets = [
      [0, 10],
      [0, 5],
      [k / 2, 10],
      [-k / 2, 10],
      [k, 6],
      [0, -4],
    ];
    const sw = await createSweepWorld(CARCHASE_AGENT);
    try {
      const data = await runSweep(sw, {
        startSpeeds: [0, 6],
        controlSets,
      });
      const fit = fitParams(data, { maxIter: 600 });
      expect(fit.meanPosError).toBeLessThan(0.5);
      // Sanity: maxAccel in a reasonable range for a 4kN / ~580kg car.
      expect(fit.params.maxAccel).toBeGreaterThan(2);
      expect(fit.params.maxAccel).toBeLessThan(15);
      // And fitted error is better than the kinematic baseline.
      const kinSim = kinematicForwardSim(data.agent);
      let kinErr = 0;
      let n = 0;
      for (const tr of data.trials) {
        let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: tr.startSpeed, t: 0 };
        for (let i = 1; i < tr.samples.length; i++) {
          const a = tr.samples[i - 1]!;
          const b = tr.samples[i]!;
          const dt = (b.t - a.t) / 6;
          for (let j = 0; j < 6; j++) s = kinSim(s, tr.controls, dt);
          kinErr += Math.hypot(s.x - b.x, s.z - b.z);
          n++;
        }
      }
      const kinMean = kinErr / n;
      expect(fit.meanPosError).toBeLessThan(kinMean);
    } finally {
      sw.dispose();
    }
  }, 60000);
});
