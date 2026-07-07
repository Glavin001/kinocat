// TEMP — model-vs-plant fidelity check in the wedge-escape regime. Not committed.
// From rest (the wedge state), apply candidate escape maneuvers for 1.5 s and
// compare the Rapier plant truth vs each MPPI forward model's rollout,
// integrated exactly the way mpcTrack does (30 × 0.05 s steps, 3 substeps).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHeadlessTrialHarness } from 'kinocat/adapters/rapier';
import { DEFAULT_VEHICLE_OPTS } from '../app/lib/training-driver';
import { modelFromJson } from '../app/lib/v2-model-file';
import {
  v3FromJson, forwardSimV3Rollout, learnedForwardSimV2, parametricForwardV2,
  KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG,
  type CarKinematicState,
} from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const v3 = v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8')));
const v2 = modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8')));

const models: Array<[string, ForwardSim<CarKinematicState>]> = [
  ['kin', parametricForwardV2(KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG)],
  ['v2', learnedForwardSimV2(v2)],
  ['v3', forwardSimV3Rollout(v3)],
];

const harness = await createHeadlessTrialHarness({
  vehicleOptions: DEFAULT_VEHICLE_OPTS,
  groundFriction: 1.5,
  groundBounds: { x0: -2000, x1: 2000, z0: -2000, z1: 2000 },
  offArenaThreshold: 5000,
});

const H = 30;
const stepDt = 0.05;
const substeps = 3;
const ticksPerStep = 3; // 0.05 s at 1/60

interface Maneuver { name: string; steer: number; drive: number; brake: number; v0: number }
const maneuvers: Maneuver[] = [
  { name: 'rest full-steer full-throttle', steer: 0.6, drive: 4000, brake: 0, v0: 0 },
  { name: 'rest full-steer half-throttle', steer: 0.6, drive: 2000, brake: 0, v0: 0 },
  { name: 'rest full-steer quart-throttle', steer: 0.6, drive: 1000, brake: 0, v0: 0 },
  { name: 'rest straight full-throttle', steer: 0.0, drive: 4000, brake: 0, v0: 0 },
  { name: 'rest full-steer reverse-40%', steer: 0.6, drive: -1600, brake: 0, v0: 0 },
  { name: 'v8 full-steer coast', steer: 0.6, drive: 0, brake: 0, v0: 8 },
  { name: 'v8 full-steer half-throttle', steer: 0.6, drive: 2000, brake: 0, v0: 8 },
  { name: 'v8 straight full-brake', steer: 0.0, drive: 0, brake: 2000, v0: 8 },
];

console.log('maneuver'.padEnd(34) + 'who'.padEnd(7) + 'end(x,z)'.padEnd(18) + 'dh(deg)'.padEnd(9) + 'v_end'.padEnd(7) + 'posErr'.padEnd(8) + 'hErr(deg)');
for (const m of maneuvers) {
  const trace = Array.from({ length: H * ticksPerStep }, () => ({
    steer: m.steer, driveForce: m.drive, brakeForce: m.brake,
  }));
  const r = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: m.v0 },
    controlsTrace: trace,
    sampleEveryNTicks: 1,
    id: m.name,
  });
  if (!r.ok) { console.log(`${m.name}: plant trial failed (${r.reason})`); continue; }
  const truth = r.trial.samples[r.trial.samples.length - 1]!;
  const wrap = (a: number): number => {
    let d = a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d;
  };
  console.log(
    m.name.padEnd(34) + 'PLANT'.padEnd(7) +
    `(${truth.x.toFixed(1)},${truth.z.toFixed(1)})`.padEnd(18) +
    `${(wrap(truth.heading) * 180 / Math.PI).toFixed(0)}`.padEnd(9) +
    `${truth.speed.toFixed(1)}`.padEnd(7),
  );
  for (const [name, sim] of models) {
    let s: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: m.v0, yawRate: 0, lateralVelocity: 0, t: 0,
    };
    for (let i = 0; i < H; i++) {
      for (let sub = 0; sub < substeps; sub++) {
        s = sim(s, [m.steer, m.drive, m.brake], stepDt / substeps);
      }
    }
    const posErr = Math.hypot(s.x - truth.x, s.z - truth.z);
    const hErr = Math.abs(wrap(s.heading - truth.heading)) * 180 / Math.PI;
    console.log(
      ''.padEnd(34) + name.padEnd(7) +
      `(${s.x.toFixed(1)},${s.z.toFixed(1)})`.padEnd(18) +
      `${(wrap(s.heading) * 180 / Math.PI).toFixed(0)}`.padEnd(9) +
      `${s.speed.toFixed(1)}`.padEnd(7) +
      `${posErr.toFixed(2)}`.padEnd(8) +
      `${hErr.toFixed(0)}`,
    );
  }
}
harness.dispose();
