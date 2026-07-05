// Ground-truth model-fidelity stats: how well does each prediction layer
// match the REAL Rapier raycast vehicle, for the same chassis
// configuration, across the speed envelope?
//
// This is the measurement the model-fidelity roadmap hangs on:
//   1. The v2 parametric model (config derived straight from the Rapier
//      options via the harness) must beat the kinematic model against the
//      actual plant — it encodes speed-dependent turning (understeer,
//      friction circle, yaw inertia) the kinematic model cannot express.
//   2. The PLANNER'S view of the plant — pre-baked primitive endpoints
//      looked up by nearest start-speed bucket — must not throw that
//      fidelity away: the dense 4 m/s grid must track the plant strictly
//      better than the old [0, 10, 20, 28] grid.
//
// Every probe rolls the real chassis 0.8 s (one race-primitive duration)
// from a settled state at a swept start speed, then measures each
// predictor's endpoint position error against the recorded truth.

import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import {
  createHeadlessTrialHarness,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  parametricForwardV2,
  kinematicForwardSim,
  defaultVehicleAgent,
  deriveVehicleCapabilities,
  type CarKinematicState,
  type WheeledCarControls,
} from 'kinocat/agent';
import { learnedForwardSimV2, deserializeMLP, type LearnedVehicleModel, type MLP } from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { RACE_START_SPEEDS } from '../app/lib/race-primitives-scenarios';

/** The shipped artifact, trained on Rapier trial data (overnight profile,
 *  3600 trials). Loaded raw — same values the demo pages run with. */
function loadTrainedModel(): LearnedVehicleModel {
  const payload = JSON.parse(
    readFileSync(join(__dirname, '..', 'public', 'models', 'v2-default.json'), 'utf8'),
  );
  const ensemble: MLP[] = (payload.residualEnsembleJson ?? []).map((j: string) => deserializeMLP(j));
  return {
    params: payload.params,
    config: payload.config,
    residualEnsemble: ensemble,
    residualReferenceDt: payload.residualReferenceDt ?? 0.1,
    oodStdThreshold: payload.oodStdThreshold,
  };
}

const PHYSICS_DT = 1 / 60;
const DURATION_TICKS = 48; // 0.8 s — one race-primitive duration
const OLD_GRID = [0, 10, 20, 28];

// Planning-frame probe controls: pure-coast turns isolate speed-dependent
// yaw response; throttle/brake probes exercise the longitudinal channel.
const PROBES: { name: string; c: WheeledCarControls }[] = [
  { name: 'coast half-steer', c: { steer: 0.3, driveForce: 0, brakeForce: 0 } },
  { name: 'coast full-lock', c: { steer: 0.6, driveForce: 0, brakeForce: 0 } },
  { name: 'full throttle straight', c: { steer: 0, driveForce: 4000, brakeForce: 0 } },
  { name: 'brake-in-turn', c: { steer: 0.3, driveForce: 0, brakeForce: 1000 } },
];

const SPEEDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];

let harness: HeadlessTrialHarness;

beforeAll(async () => {
  harness = await createHeadlessTrialHarness({ vehicleOptions: {} });
});
afterAll(() => harness?.dispose());

function rollModel(
  sim: ForwardSim<CarKinematicState>,
  start: CarKinematicState,
  controls: number[],
): CarKinematicState {
  let s = start;
  for (let i = 0; i < DURATION_TICKS; i++) s = sim(s, controls, PHYSICS_DT);
  return s;
}

function nearestBucket(speed: number, buckets: number[]): number {
  let best = buckets[0]!;
  for (const b of buckets) if (Math.abs(b - speed) < Math.abs(best - speed)) best = b;
  return best;
}

/** Bake a primitive offset at the bucket speed (rest frame, heading 0) and
 *  rigid-compose it onto the true start — exactly what the planner's succ
 *  does with a library endpoint. */
function bucketEndpoint(
  sim: ForwardSim<CarKinematicState>,
  start: CarKinematicState,
  bucket: number,
  controls: number[],
): { x: number; z: number } {
  const baked = rollModel(
    sim,
    { x: 0, z: 0, heading: 0, speed: bucket, yawRate: 0, lateralVelocity: 0, t: 0 },
    controls,
  );
  const cos = Math.cos(start.heading);
  const sin = Math.sin(start.heading);
  return {
    x: start.x + baked.x * cos - baked.z * sin,
    z: start.z + baked.x * sin + baked.z * cos,
  };
}

interface ErrStats { sum: number; worst: number; n: number }
const mk = (): ErrStats => ({ sum: 0, worst: 0, n: 0 });
const add = (s: ErrStats, e: number) => { s.sum += e; s.n += 1; if (e > s.worst) s.worst = e; };
const mean = (s: ErrStats) => s.sum / Math.max(1, s.n);

describe('model vs Rapier plant — endpoint fidelity across the speed envelope', () => {
  it('config-derived v2 model beats kinematic; dense lattice beats the old grid', () => {
    // The harness derives its config straight from the Rapier options —
    // this equality IS the "model built from the same car parameters"
    // guarantee (drift here would invalidate everything below).
    expect(harness.config).toEqual(DEFAULT_LEARNABLE_CONFIG);

    const caps = deriveVehicleCapabilities(harness.config);
    const v2 = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, harness.config);
    const kinAgent = defaultVehicleAgent({
      minTurnRadius: caps.minTurnRadius,
      maxSpeed: 30,
      maxReverseSpeed: 6,
    });
    const kin = kinematicForwardSim(kinAgent);
    const L = caps.wheelbaseLength;

    const trained = learnedForwardSimV2(loadTrainedModel());

    const kinStats = mk();
    const v2Stats = mk();
    const trainedStats = mk();
    const oldGridStats = mk();
    const newGridStats = mk();
    const rows: string[] = [];

    for (const speed of SPEEDS) {
      for (const probe of PROBES) {
        const outcome = harness.runTrial({
          pose: { x: 0, z: 0, heading: 0 },
          kin: { forwardSpeed: speed },
          controlsTrace: new Array(DURATION_TICKS).fill(probe.c),
          sampleEveryNTicks: DURATION_TICKS,
          id: `probe-${speed}-${probe.name}`,
        });
        expect(outcome.ok, `trial ${speed} m/s ${probe.name}`).toBe(true);
        if (!outcome.ok) continue;
        const start = outcome.trial.samples[0]!;
        const truth = outcome.trial.samples[outcome.trial.samples.length - 1]!;

        const ctrlVec = [probe.c.steer, probe.c.driveForce, probe.c.brakeForce];
        // Kinematic worldview: steer → constant curvature; throttle → top
        // speed, brake → 0, coast → hold current speed.
        const kinTarget = probe.c.driveForce > 0 ? kinAgent.maxSpeed : probe.c.brakeForce > 0 ? 0 : start.speed;
        const kinVec = [Math.tan(probe.c.steer) / L, kinTarget];

        const errOf = (p: { x: number; z: number }) => Math.hypot(p.x - truth.x, p.z - truth.z);
        const kinErr = errOf(rollModel(kin, start, kinVec));
        const v2Err = errOf(rollModel(v2, start, ctrlVec));
        const trainedErr = errOf(rollModel(trained, start, ctrlVec));
        const oldErr = errOf(bucketEndpoint(trained, start, nearestBucket(start.speed, OLD_GRID), ctrlVec));
        const newErr = errOf(bucketEndpoint(trained, start, nearestBucket(start.speed, [...RACE_START_SPEEDS]), ctrlVec));
        add(kinStats, kinErr); add(v2Stats, v2Err); add(trainedStats, trainedErr);
        add(oldGridStats, oldErr); add(newGridStats, newErr);
        rows.push(
          `${String(speed).padStart(2)} m/s  ${probe.name.padEnd(24)} kin=${kinErr.toFixed(2)}  v2-prior=${v2Err.toFixed(2)}  v2-trained=${trainedErr.toFixed(2)}  old-grid=${oldErr.toFixed(2)}  new-grid=${newErr.toFixed(2)}`,
        );
      }
    }

    console.log('\nEndpoint position error vs real Rapier chassis after one 0.8 s primitive (m):');
    for (const r of rows) console.log('  ' + r);
    console.log(
      `\n  MEAN   kinematic=${mean(kinStats).toFixed(3)}  v2-prior=${mean(v2Stats).toFixed(3)}  ` +
      `v2-trained=${mean(trainedStats).toFixed(3)}  ` +
      `old-grid=${mean(oldGridStats).toFixed(3)}  new-grid=${mean(newGridStats).toFixed(3)}\n` +
      `  WORST  kinematic=${kinStats.worst.toFixed(3)}  v2-prior=${v2Stats.worst.toFixed(3)}  ` +
      `v2-trained=${trainedStats.worst.toFixed(3)}  ` +
      `old-grid=${oldGridStats.worst.toFixed(3)}  new-grid=${newGridStats.worst.toFixed(3)}`,
    );

    // 1. Speed-dependent dynamics matter: the config-derived v2 model must
    //    beat the kinematic worldview against the real plant — even with
    //    UNTRAINED prior params.
    expect(mean(v2Stats)).toBeLessThan(mean(kinStats));
    expect(v2Stats.worst).toBeLessThan(kinStats.worst);

    // 2. Training on Rapier trial data must pay: the shipped trained model
    //    beats both the kinematic model and the untrained prior.
    expect(mean(trainedStats)).toBeLessThan(mean(kinStats));
    expect(mean(trainedStats)).toBeLessThan(mean(v2Stats));

    // 3. The planner's bucketed view must preserve that fidelity: dense
    //    grid at least as good as the old one on mean AND worst case
    //    (strictly better on the isolated quantization metric — see
    //    primitive-lattice.test.ts; against the plant, model bias adds a
    //    shared floor to both).
    expect(mean(newGridStats)).toBeLessThan(mean(oldGridStats));
    expect(newGridStats.worst).toBeLessThanOrEqual(oldGridStats.worst);

    // 4. The dense grid costs almost nothing relative to the un-bucketed
    //    trained model: quantization overhead stays a small additive term,
    //    so planner fidelity ≈ model fidelity.
    expect(mean(newGridStats)).toBeLessThan(mean(trainedStats) + 1.0);

    // 5. Absolute regression budgets (measured 2026-07: trained mean
    //    1.08 m, planner-view mean 1.23 m over one 0.8 s primitive; ~60%
    //    headroom so re-fits don't flap, real regressions still trip).
    expect(mean(trainedStats)).toBeLessThan(1.8);
    expect(mean(newGridStats)).toBeLessThan(2.0);
  }, 120_000);
});
