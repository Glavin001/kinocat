// Acceptance gate for the shipped learned forward model
// (`public/models/v2-default.json`).
//
// This is the CI ratchet that turns "is the surrogate faithful?" from a hope
// into a release requirement. It grades the SHIPPED model against fresh Rapier
// ground truth on the metrics the MPC tracker actually depends on — NOT the
// long-horizon library endpoints (those now come from ground truth), but:
//
//   1. single-step accuracy (dt = 0.1 s) — the per-tick prediction MPPI rolls,
//   2. short-horizon accuracy (0.5 s = the MPPI horizon),
//   3. gradient/directional fidelity — +steer must move the endpoint the same
//      way Rapier does (a model can have low RMS but a wrong local gradient,
//      which makes MPPI pick BAD controls while looking accurate),
//   4. the coverage gate ships populated (inputSupport present),
//
// across the historically-hard regimes (reverse, high-speed brake, brake into
// a corner) plus the easy ones. Budgets are ratcheted just above current
// measured error: they block REGRESSIONS and document the known-loose regimes.
// Tighten them as DAgger improves those regimes — a tightening that fails here
// is the signal the model got better and the ratchet should follow.

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { learnedForwardSimV2, decodeWheeled } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import {
  createHeadlessTrialHarness,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import { modelFromJson } from '../app/lib/v2-model-file';
import { DEFAULT_VEHICLE_OPTS } from '../app/lib/training-driver';

const PHYSICS_DT = 1 / 60;
const MODEL_PATH = resolve(__dirname, '../public/models/v2-default.json');

type Regime = 'precise' | 'hard';
interface Probe {
  name: string;
  regime: Regime;
  startSpeed: number;
  controls: number[]; // [steer, driveForce, brakeForce]
}

// Per-regime budgets (metres), ratcheted ~just above current measured error.
const BUDGET: Record<Regime, { singleStep: number; horizon: number }> = {
  precise: { singleStep: 0.15, horizon: 0.8 },
  hard: { singleStep: 1.0, horizon: 2.5 },
};

const PROBES: Probe[] = [
  { name: 'cruise straight @10', regime: 'precise', startSpeed: 10, controls: [0, 2000, 0] },
  { name: 'gentle turn @10', regime: 'precise', startSpeed: 10, controls: [0.4, 1600, 0] },
  { name: 'coast @14', regime: 'precise', startSpeed: 14, controls: [0, 0, 0] },
  { name: 'reverse @0', regime: 'hard', startSpeed: 0, controls: [0, -1600, 0] },
  { name: 'brake @20', regime: 'hard', startSpeed: 20, controls: [0, 0, 2000] },
  { name: 'brake into corner @28', regime: 'hard', startSpeed: 28, controls: [0.4, 0, 2000] },
];

function toLocal(
  start: { x: number; z: number; heading: number },
  end: { x: number; z: number },
): { dx: number; dz: number } {
  const dx = end.x - start.x;
  const dz = end.z - start.z;
  const h = start.heading;
  return { dx: dx * Math.cos(h) + dz * Math.sin(h), dz: -dx * Math.sin(h) + dz * Math.cos(h) };
}

describe('shipped model acceptance gate (vs Rapier ground truth)', () => {
  let harness: HeadlessTrialHarness;
  let sim: ForwardSim<CarKinematicState>;
  let hasInputSupport = false;

  beforeAll(async () => {
    const model = modelFromJson(JSON.parse(readFileSync(MODEL_PATH, 'utf-8')));
    hasInputSupport = !!model.inputSupport;
    sim = learnedForwardSimV2(model);
    harness = await createHeadlessTrialHarness({ vehicleOptions: DEFAULT_VEHICLE_OPTS });
  }, 30_000);

  afterAll(() => harness?.dispose());

  // Ground truth + matched-IC model rollout for a control held `dur` seconds.
  function errorFor(startSpeed: number, controls: number[], dur: number, steps: number): number {
    const ticks = Math.round(dur / PHYSICS_DT);
    const wc = decodeWheeled(controls);
    const out = harness.runTrial({
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: startSpeed },
      controlsTrace: Array.from({ length: ticks }, () => ({ ...wc })),
      sampleEveryNTicks: ticks,
    });
    expect(out.ok).toBe(true);
    if (!out.ok) return NaN;
    const samples = out.trial.samples;
    const start = samples[0]!;
    const gt = toLocal(start, samples[samples.length - 1]!);
    // Model from matched ICs.
    let s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: start.speed,
      yawRate: start.yawRate ?? 0, lateralVelocity: start.lateralVelocity ?? 0, t: 0,
    };
    const dt = dur / steps;
    for (let i = 0; i < steps; i++) s = sim(s, controls, dt);
    return Math.hypot(s.x - gt.dx, s.z - gt.dz);
  }

  it('ships with the coverage gate populated (inputSupport present)', () => {
    expect(hasInputSupport).toBe(true);
  });

  for (const p of PROBES) {
    it(`${p.name}: single-step error within ${BUDGET[p.regime].singleStep} m`, () => {
      const e = errorFor(p.startSpeed, p.controls, 0.1, 1);
      expect(e).toBeLessThan(BUDGET[p.regime].singleStep);
    });
    it(`${p.name}: 0.5 s horizon error within ${BUDGET[p.regime].horizon} m`, () => {
      const e = errorFor(p.startSpeed, p.controls, 0.5, 5);
      expect(e).toBeLessThan(BUDGET[p.regime].horizon);
    });
  }

  it('gradient/directional fidelity: +steer turns the same way as Rapier', () => {
    // Lateral displacement (local dz) must share Rapier's sign and be
    // monotonic across the steer sweep — otherwise MPPI would optimise the
    // wrong direction even with low RMS.
    const lateralOf = (steer: number): { model: number; truth: number } => {
      const controls = [steer, 1600, 0];
      const ticks = Math.round(0.5 / PHYSICS_DT);
      const wc = decodeWheeled(controls);
      const out = harness.runTrial({
        pose: { x: 0, z: 0, heading: 0 }, kin: { forwardSpeed: 10 },
        controlsTrace: Array.from({ length: ticks }, () => ({ ...wc })), sampleEveryNTicks: ticks,
      });
      if (!out.ok) throw new Error('trial discarded');
      const samples = out.trial.samples;
      const gt = toLocal(samples[0]!, samples[samples.length - 1]!);
      let s: CarKinematicState = {
        x: 0, z: 0, heading: 0, speed: samples[0]!.speed,
        yawRate: samples[0]!.yawRate ?? 0, lateralVelocity: samples[0]!.lateralVelocity ?? 0, t: 0,
      };
      for (let i = 0; i < 5; i++) s = sim(s, controls, 0.1);
      return { model: s.z, truth: gt.dz };
    };
    const left = lateralOf(-0.4);
    const right = lateralOf(0.4);
    // Sign agreement with Rapier on each side.
    expect(Math.sign(left.model)).toBe(Math.sign(left.truth));
    expect(Math.sign(right.model)).toBe(Math.sign(right.truth));
    // Monotonic: right turn lands further +dz than left, in BOTH model and truth.
    expect(right.model).toBeGreaterThan(left.model);
    expect(right.truth).toBeGreaterThan(left.truth);
  });
});
