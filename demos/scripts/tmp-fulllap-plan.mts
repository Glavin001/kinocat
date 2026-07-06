// Full-lap PLAN (no execution) via receding-horizon stitch: what each model's
// planner thinks the best racing line + speed profile is, carrying its OWN
// predicted state forward (its model of reality). No physics.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/tmp-fulllap-plan.mts <kin|v2|v3> [out]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary, buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  RACE_AGENT, RACE_PLANNER_GATE_RADIUS, RACE_ARRIVE_RADIUS, buildRaceCourse,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';
import { plotTrajectory } from './lib/trajectory-plot';
import type { CarKinematicState } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const out = process.argv[3] ?? `/tmp/fulllap-${which}.png`;
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const lib = which === 'v3'
  ? buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')), { generatedControls: gen })
  : which === 'v2'
    ? buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json')), { generatedControls: gen })
    : buildKinematicLibrary();
const course = buildRaceCourse('open');
const world = new InMemoryNavWorld(course.polygons, course.obstacles);
const wps = course.waypoints;
const N = wps.length;

let state: CarKinematicState = { ...course.spawn, speed: 0 };
const full: CarKinematicState[] = [state];
const nearestIdx = (path: CarKinematicState[], g: { x: number; z: number }): number => {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < path.length; i++) {
    const d = Math.hypot(path[i]!.x - g.x, path[i]!.z - g.z);
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
};
let ok = true;
const t0 = performance.now();
for (let i = 0; i < N; i++) {
  const window = [wps[i % N]!, wps[(i + 1) % N]!, wps[(i + 2) % N]!].map((w) => ({ ...w }));
  const res = planVehicleMultiGoal({
    start: state, gates: window, world, agent: RACE_AGENT, lib,
    deadlineMs: 15000, maxExpansions: 2_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
    envOptions: dt ? { analyticDriveThrough: true } : undefined,
  });
  if (!res.found || res.path.length < 2) {
    // Skip a failed window (keep the viz continuous): jump to the next gate.
    ok = false;
    state = { ...wps[(i + 1) % N]!, heading: state.heading, speed: 8, t: state.t };
    full.push(state);
    continue;
  }
  // Commit a full gate of progress: up to the SECOND window gate.
  const commitTo = Math.max(1, nearestIdx(res.path, wps[(i + 1) % N]!));
  for (const p of res.path.slice(1, commitTo + 1)) full.push(p);
  state = full[full.length - 1]!;
}
const ms = performance.now() - t0;
const dist = full.reduce((a, p, i) => (i === 0 ? 0 : a + Math.hypot(p.x - full[i - 1]!.x, p.z - full[i - 1]!.z)), 0);
const peak = Math.max(0, ...full.map((p) => Math.abs(p.speed)));
const planT = full[full.length - 1]!.t;
console.log(`${which} gen=${gen} dt=${dt}: completed=${ok} len=${full.length} dist=${dist.toFixed(0)}m peakV=${peak.toFixed(1)} planT=${planT.toFixed(1)}s ms=${ms.toFixed(0)}`);
const samples = full.map((p, i) => ({ t: i, x: p.x, z: p.z, speed: Math.abs(p.speed) }));
const png = plotTrajectory(out,
  { bounds: course.bounds, waypoints: course.waypoints, walls: course.walls, spawn: course.spawn, arriveRadius: RACE_ARRIVE_RADIUS },
  samples,
  { title: `${which} FULL-LAP PLAN (no exec, ${gen ? 'gen+' : ''}${dt ? 'reprice' : 'baseline'}) — dist=${dist.toFixed(0)}m peakV=${peak.toFixed(1)} planT=${planT.toFixed(1)}s`, vMax: 30 });
console.log(`plot: ${png}`);
