// Is the plan PLAUSIBLE, or is the model too optimistic? Takes the planner's
// exact primitive controls, rolls them OPEN-LOOP through (a) the model that
// planned them and (b) the real Rapier plant, and reports where the plant
// diverges from what the model predicted. Large divergence at a fast turn =
// the model "rounded up" its capability (optimistic, implausible plan). Small
// divergence = the plan is feasible and any wedge is a CONTROLLER problem.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/plan-vs-plant.mts <v2|v3>
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  RACE_AGENT, RACE_PLANNER_GATE_RADIUS, RACE_ARRIVE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson, forwardSimV3, learnedForwardSimV2 } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';
import { createHeadlessTrialHarness } from 'kinocat/adapters/rapier';
import { plotTrajectory } from './lib/trajectory-plot';
import type { CarKinematicState } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const v3m = which === 'v3' ? v3FromJson(readModel('v3-default.json')) : null;
const v2m = which === 'v2' ? modelFromJson(readModel('v2-default.json')) : null;
const forwardSim = which === 'v3' ? forwardSimV3(v3m!) : learnedForwardSimV2(v2m!);
const lib = which === 'v3' ? buildLearnedRaceLibraryV3(v3m!, { generatedControls: gen }) : buildLearnedRaceLibraryV2(v2m!, { generatedControls: gen });

const bounds = { x0: -50, z0: -25, x1: 30, z1: 25 };
const POLY = [{ id: 0, y: 0, ring: [[bounds.x0, bounds.z0], [bounds.x1, bounds.z0], [bounds.x1, bounds.z1], [bounds.x0, bounds.z1]] as [number, number][] }];
const world = new InMemoryNavWorld(POLY, []);
const spawn = { x: -40, z: 0, heading: 0, speed: 14, t: 0 };
const gates = [
  { x: -18, z: 5, heading: 0, speed: 5, t: 0 },
  { x: 0, z: -5, heading: 0, speed: 5, t: 0 },
  { x: 18, z: 5, heading: 0, speed: 5, t: 0 },
];
const res = planVehicleMultiGoal({
  start: spawn, gates, world, agent: RACE_AGENT, lib,
  deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
  envOptions: dt ? { analyticDriveThrough: true } : undefined,
});

// Extract the planned DRIVE controls into a per-physics-tick trace (skip the
// terminal Reeds-Shepp analytic segment — it isn't a model-driven maneuver).
const DT = 1 / 60;
const primById = new Map<number, { controls: number[]; duration: number }>();
for (const p of lib.primitives) primById.set(p.id, p);
const trace: { steer: number; driveForce: number; brakeForce: number }[] = [];
const nodes = res.nodes ?? [];
for (let i = 1; i < res.path.length; i++) {
  const edge = nodes[i]?.edge;
  if (edge?.kind !== 'drive') continue;
  const primId = (edge.data as { primId?: number }).primId;
  const prim = primId !== undefined ? primById.get(primId) : undefined;
  if (!prim) continue;
  const ticks = Math.max(1, Math.round(prim.duration / DT));
  for (let k = 0; k < ticks; k++) trace.push({ steer: prim.controls[0]!, driveForce: prim.controls[1]!, brakeForce: prim.controls[2]! });
}

// (a) Roll the MODEL open-loop (what the planner predicts these controls do).
const modelTraj: CarKinematicState[] = [{ ...spawn }];
{
  let s: CarKinematicState = { ...spawn };
  for (const c of trace) { s = forwardSim(s, [c.steer, c.driveForce, c.brakeForce], DT); modelTraj.push({ ...s }); }
}
// (b) Roll the real Rapier PLANT with the same controls.
const harness = await createHeadlessTrialHarness({ vehicleOptions: {} });
const outcome = harness.runTrial({ pose: { x: spawn.x, z: spawn.z, heading: spawn.heading }, kin: { forwardSpeed: spawn.speed }, controlsTrace: trace, sampleEveryNTicks: 1 });
harness.dispose();
if (!outcome.ok) { console.log('plant trial failed:', outcome.reason); process.exit(1); }
const plantTraj = outcome.trial.samples;

// Compare: position divergence + speed over-prediction along the maneuver.
const n = Math.min(modelTraj.length, plantTraj.length);
let maxDiv = 0, maxAt = 0;
const speedErr: number[] = [];
for (let i = 0; i < n; i++) {
  const m = modelTraj[i]!, p = plantTraj[i]!;
  const d = Math.hypot(m.x - p.x, m.z - p.z);
  if (d > maxDiv) { maxDiv = d; maxAt = i; }
  speedErr.push(Math.abs(m.speed) - Math.abs(p.speed)); // >0 = model predicted FASTER than plant
}
const endDiv = Math.hypot(modelTraj[n - 1]!.x - plantTraj[n - 1]!.x, modelTraj[n - 1]!.z - plantTraj[n - 1]!.z);
const mPeak = Math.max(...modelTraj.map((s) => Math.abs(s.speed)));
const pPeak = Math.max(...plantTraj.map((s) => Math.abs(s.speed)));
console.log(`${which}: planned controls rolled open-loop through model vs real plant`);
console.log(`  position divergence: max=${maxDiv.toFixed(2)}m @t=${(maxAt * DT).toFixed(1)}s  endpoint=${endDiv.toFixed(2)}m  (maneuver ~${(n * DT).toFixed(1)}s, ${(modelTraj.reduce((a, s, i) => i ? a + Math.hypot(s.x - modelTraj[i-1]!.x, s.z - modelTraj[i-1]!.z) : 0, 0)).toFixed(0)}m)`);
console.log(`  peak speed: model predicts ${mPeak.toFixed(1)} m/s, plant reaches ${pPeak.toFixed(1)} m/s  (model over-predict by ${(mPeak - pPeak).toFixed(1)})`);
console.log(`  mean speed over-prediction (model - plant): ${(speedErr.reduce((a, b) => a + b, 0) / speedErr.length).toFixed(2)} m/s`);
// Plot both trajectories overlaid.
const toSamp = (tr: CarKinematicState[], every = 3) => tr.filter((_, i) => i % every === 0).map((s, i) => ({ t: i, x: s.x, z: s.z, speed: Math.abs(s.speed) }));
plotTrajectory(`/tmp/plant-${which}-model.png`, { bounds, waypoints: [spawn, ...gates], spawn, arriveRadius: RACE_ARRIVE_RADIUS }, toSamp(modelTraj), { title: `${which} MODEL prediction (open-loop planned controls) peak=${mPeak.toFixed(1)}`, vMax: 30 });
plotTrajectory(`/tmp/plant-${which}-plant.png`, { bounds, waypoints: [spawn, ...gates], spawn, arriveRadius: RACE_ARRIVE_RADIUS }, toSamp(plantTraj), { title: `${which} REAL PLANT (same controls) peak=${pPeak.toFixed(1)} — divergence ${maxDiv.toFixed(1)}m`, vMax: 30 });
console.log(`  plots: /tmp/plant-${which}-model.png  /tmp/plant-${which}-plant.png`);
