// Skill test — primitive COVERAGE regression (control-set design safeguard).
//
// Asserts the dispersion-designed race control sets actually span the chassis's
// reachable set per speed bucket, so a future edit that opens a coverage hole
// (like the K5 medium-radius gap) fails CI instead of surfacing three layers
// downstream as a wedged car. See docs/racing-skills-test-plan.md.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { v3FromJson, forwardSimV3, learnedForwardSimV2 } from 'kinocat/agent';
import { modelFromJson } from '../../app/lib/v2-model-file';
import { designControlSet, coverageReport } from 'kinocat/primitives';
import type { ForwardSim } from 'kinocat/primitives';
import type { CarKinematicState } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));

const v2Model = modelFromJson(readModel('v2-default.json'));
const v3Model = v3FromJson(readModel('v3-default.json'));
const sims: Record<string, ForwardSim<CarKinematicState>> = {
  v2: learnedForwardSimV2(v2Model),
  v3: forwardSimV3(v3Model),
};
// Actuator limits from the model config — the same values the library builder
// rolls with (cfg.maxSteerAngle / maxDriveForce / maxBrakeForce).
const cfg = v3Model.config;
const common = { maxSteer: cfg.maxSteerAngle, maxDrive: cfg.maxDriveForce, maxBrake: cfg.maxBrakeForce };
// (speed, duration, reverseSlots, budget) buckets mirroring the library builder.
const BUCKETS = [
  { v: 0, dur: 1.5, rev: 5, budget: 18 },
  { v: 8, dur: 0.55, rev: 0, budget: 14 },
  { v: 20, dur: 0.55, rev: 0, budget: 14 },
  { v: 28, dur: 0.55, rev: 0, budget: 14 },
] as const;

describe('skill: primitive coverage of the dispersion-designed race sets', () => {
  for (const model of ['v2', 'v3'] as const) {
    for (const b of BUCKETS) {
      it(`${model} @ ${b.v} m/s spans the reachable set (no coverage hole)`, () => {
        const sim = sims[model]!;
        const controls = designControlSet({
          forwardSim: sim, startSpeed: b.v, duration: b.dur, substeps: 6,
          budget: b.budget, reverseSlots: b.rev, ...common,
        });
        const r = coverageReport(controls, { forwardSim: sim, startSpeed: b.v, duration: b.dur, substeps: 6, ...common });
        // eslint-disable-next-line no-console
        console.log(`  cov ${model}@${b.v}: disp=${r.dispersion.toFixed(2)} minPair=${r.minPairwise.toFixed(2)} maxHead ${r.maxHeadingSet.toFixed(2)}/${r.maxHeadingReachable.toFixed(2)} slots=${r.slots}`);

        // EXTREME-REACH: the set must expose most of the chassis's real
        // cornering envelope (the K5 fix — old sets reached only ~half).
        expect(r.maxHeadingSet).toBeGreaterThan(0.85 * r.maxHeadingReachable);
        // EXTREME-REACH straight: near the fastest reachable straight.
        expect(r.maxDxSet).toBeGreaterThan(0.9 * r.maxDxReachable);
        // NO REDUNDANCY: no two forward primitives nearly coincide.
        expect(r.minPairwise).toBeGreaterThan(0.15);
        // BOUNDED DISPERSION: no large uncovered reachable region. The launch
        // bucket (1.5 s) reaches farther so allow a larger absolute gap.
        expect(r.dispersion).toBeLessThan(b.v < 2 ? 4.0 : 2.2);
        // SYMMETRY (by construction): every steered control has its mirror.
        const key = (u: number[]) => `${u[0]!.toFixed(3)}|${u[1]}|${u[2]}`;
        const set = new Set(controls.map(key));
        for (const u of controls) {
          if (Math.abs(u[0]!) > 1e-6) expect(set.has(key([-u[0]!, u[1]!, u[2]!]))).toBe(true);
        }
        // Reverse allocation where required.
        if (b.rev > 0) expect(controls.filter((u) => u[1]! < 0).length).toBeGreaterThan(0);
      });
    }
  }
});
