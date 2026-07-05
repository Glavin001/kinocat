// V3 purely-learned dynamics model — machinery tests.
//
// These cover the model-independent guarantees (frame handling, dt
// decomposition, persistence round-trip, fit determinism and mirror
// symmetry) on tiny synthetic data. Fidelity against the real Rapier plant
// is covered by demos/test (which can spin up the physics harness).

import { describe, expect, it } from 'vitest';
import {
  forwardSimV3,
  v3ToJson,
  v3FromJson,
  isV3Payload,
  buildV3Input,
  V3_INPUT_DIM,
  V3_OUTPUT_DIM,
  DEFAULT_LEARNABLE_CONFIG,
  type CarKinematicState,
  type LearnedVehicleModelV3,
} from '../../src/agent';
import { runDynamicsV3Fit, type CarTrial } from '../../src/learning';
import type { WheeledCarControls } from '../../src/agent/controls';

const DT = 1 / 60;

/** Synthetic "plant": trivially-learnable linear dynamics used to produce
 *  consistent trials (constant accel from drive, yaw from steer × speed). */
function syntheticStep(s: CarKinematicState, c: WheeledCarControls): CarKinematicState {
  const accel = (c.driveForce / 4000) * 10 - (c.brakeForce / 2000) * 12 * Math.sign(s.speed);
  const speed = s.speed + accel * DT;
  const yawRate = 0.5 * c.steer * s.speed;
  const heading = s.heading + yawRate * DT;
  return {
    x: s.x + speed * Math.cos(s.heading) * DT,
    z: s.z + speed * Math.sin(s.heading) * DT,
    heading,
    speed,
    yawRate,
    lateralVelocity: 0.1 * c.steer * s.speed,
    t: s.t + DT,
  };
}

function syntheticTrial(
  id: string,
  startSpeed: number,
  controls: WheeledCarControls,
  ticks: number,
  startHeading = 0,
): CarTrial {
  let s: CarKinematicState = {
    x: 0, z: 0, heading: startHeading, speed: startSpeed,
    yawRate: 0, lateralVelocity: 0, t: 0,
  };
  const samples = [{ t: 0, state: s }];
  const trace: WheeledCarControls[] = [];
  for (let i = 0; i < ticks; i++) {
    trace.push(controls);
    s = syntheticStep(s, controls);
    samples.push({ t: (i + 1) * DT, state: s });
  }
  return {
    id,
    initialState: samples[0]!.state,
    controlsTrace: trace,
    dt: DT,
    samples,
    config: DEFAULT_LEARNABLE_CONFIG,
    configKey: 'rwd-default',
  };
}

function trainTinyModel(seed = 7): LearnedVehicleModelV3 {
  const trials: CarTrial[] = [];
  let n = 0;
  for (const v of [0, 5, 10, 15, 20]) {
    for (const c of [
      { steer: 0, driveForce: 4000, brakeForce: 0 },
      { steer: 0, driveForce: 0, brakeForce: 1000 },
      { steer: 0.3, driveForce: 2000, brakeForce: 0 },
      { steer: -0.3, driveForce: 2000, brakeForce: 0 },
      { steer: 0, driveForce: 0, brakeForce: 0 },
    ]) {
      trials.push(syntheticTrial(`t${n++}`, v, c, 30, (n % 5) * 1.1));
    }
  }
  return runDynamicsV3Fit({
    trials,
    hiddenDims: [24, 24],
    ensembleSize: 1,
    epochs: 150,
    seed,
    noiseScale: 0,
  }).model;
}

describe('v3 dynamics model', () => {
  const model = trainTinyModel();

  it('input encoding carries only body-frame dynamic state + controls', () => {
    const s: CarKinematicState = {
      x: 123, z: -77, heading: 2.4, speed: 8, yawRate: 0.3, lateralVelocity: -0.5, t: 9,
    };
    const input = buildV3Input(s, [0.2, 1000, 50]);
    expect(input).toHaveLength(V3_INPUT_DIM);
    expect(input).toEqual([8, 0.3, -0.5, 0.2, 1000, 50]);
  });

  it('learns the synthetic dynamics (sanity: closes on a held-out rollout)', () => {
    const sim = forwardSimV3(model);
    // Trained control combo from a start speed BETWEEN grid points —
    // machinery sanity on a tiny net, not a fidelity budget (that's
    // measured against the real plant in demos/test).
    let truth: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 7.5, yawRate: 0, lateralVelocity: 0, t: 0 };
    let pred = truth;
    const c: WheeledCarControls = { steer: 0.3, driveForce: 2000, brakeForce: 0 };
    for (let i = 0; i < 30; i++) {
      truth = syntheticStep(truth, c);
      pred = sim(pred, [c.steer, c.driveForce, c.brakeForce], DT);
    }
    expect(Math.hypot(pred.x - truth.x, pred.z - truth.z)).toBeLessThan(1.0);
    expect(Math.abs(pred.speed - truth.speed)).toBeLessThan(1.5);
  });

  it('is exactly translation- and rotation-equivariant', () => {
    const sim = forwardSimV3(model);
    const controls = [0.2, 2500, 0];
    const base: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 10, yawRate: 0.1, lateralVelocity: 0.2, t: 0 };
    const endA = sim(base, controls, 0.4);
    const yaw = 1.234;
    const moved: CarKinematicState = { ...base, x: 50, z: -20, heading: yaw };
    const endB = sim(moved, controls, 0.4);
    // Rigid-transform endA by (50, -20, yaw) — must match endB exactly.
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    expect(endB.x).toBeCloseTo(50 + endA.x * cos - endA.z * sin, 10);
    expect(endB.z).toBeCloseTo(-20 + endA.x * sin + endA.z * cos, 10);
    expect(endB.speed).toBeCloseTo(endA.speed, 10);
    expect(endB.yawRate ?? 0).toBeCloseTo(endA.yawRate ?? 0, 10);
  });

  it('decomposes arbitrary dt into reference steps (self-consistent)', () => {
    const sim = forwardSimV3(model);
    const s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 12, yawRate: 0, lateralVelocity: 0, t: 0 };
    const controls = [0, 4000, 0];
    // One query of 4 ticks == 4 queries of 1 tick.
    const oneShot = sim(s, controls, 4 * DT);
    let stepped = s;
    for (let i = 0; i < 4; i++) stepped = sim(stepped, controls, DT);
    expect(oneShot.x).toBeCloseTo(stepped.x, 9);
    expect(oneShot.speed).toBeCloseTo(stepped.speed, 9);
    expect(oneShot.t).toBeCloseTo(stepped.t, 9);
    // Fractional remainder is honored in elapsed time.
    const frac = sim(s, controls, 5.5 * DT);
    expect(frac.t).toBeCloseTo(5.5 * DT, 9);
  });

  it('mirror augmentation yields a left/right-symmetric model', () => {
    // The symmetry is enforced by data (every transition trains alongside
    // its mirror image), not by architecture — so it holds statistically,
    // not bit-exactly. Over a 24-step rollout the residual asymmetry must
    // stay a small fraction of the trajectory scale.
    const sim = forwardSimV3(model);
    const s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 10, yawRate: 0, lateralVelocity: 0, t: 0 };
    const left = sim(s, [0.3, 2000, 0], 0.4);
    const right = sim(s, [-0.3, 2000, 0], 0.4);
    const scale = Math.hypot(left.x, left.z);
    expect(Math.abs(right.x - left.x)).toBeLessThan(0.05 * scale);
    expect(Math.abs(right.z + left.z)).toBeLessThan(0.05 * scale);
    expect(Math.abs(right.heading + left.heading)).toBeLessThan(0.05);
    expect(Math.abs(right.speed - left.speed)).toBeLessThan(0.5);
  });

  it('persists through JSON round-trip bit-exactly', () => {
    const payload = v3ToJson(model, { note: 'test' });
    expect(isV3Payload(payload)).toBe(true);
    expect(isV3Payload({ params: {}, residualEnsembleJson: [] })).toBe(false);
    const back = v3FromJson(JSON.parse(JSON.stringify(payload)));
    const sim = forwardSimV3(model);
    const sim2 = forwardSimV3(back);
    const s: CarKinematicState = { x: 1, z: 2, heading: 0.3, speed: 14, yawRate: -0.2, lateralVelocity: 0.1, t: 0 };
    const a = sim(s, [0.1, 1500, 200], 0.55 / 6);
    const b = sim2(s, [0.1, 1500, 200], 0.55 / 6);
    expect(b).toEqual(a);
  });

  it('fit is deterministic given the seed', () => {
    const m1 = trainTinyModel(11);
    const m2 = trainTinyModel(11);
    const s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 6, yawRate: 0, lateralVelocity: 0, t: 0 };
    const a = forwardSimV3(m1)(s, [0.2, 3000, 0], 0.2);
    const b = forwardSimV3(m2)(s, [0.2, 3000, 0], 0.2);
    expect(b).toEqual(a);
    expect(m1.norm).toEqual(m2.norm);
  });

  it('normalization statistics come from the data, with sane shapes', () => {
    expect(model.norm.inputMean).toHaveLength(V3_INPUT_DIM);
    expect(model.norm.inputStd).toHaveLength(V3_INPUT_DIM);
    expect(model.norm.outputMean).toHaveLength(V3_OUTPUT_DIM);
    expect(model.norm.outputStd).toHaveLength(V3_OUTPUT_DIM);
    for (const v of model.norm.inputStd) expect(v).toBeGreaterThan(0);
    for (const v of model.norm.outputStd) expect(v).toBeGreaterThan(0);
    expect(model.referenceDt).toBeCloseTo(DT, 12);
  });
});
