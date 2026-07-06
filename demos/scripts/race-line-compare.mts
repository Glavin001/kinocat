// Clean single-plan slalom comparison: the isolated 3-gate slalom (plans in one
// call, no receding stitch), rendered per model with the fix on. Shows each
// model's intended line + speed profile through the same corner sequence.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/race-line-compare.mts <kin|v2|v3> [out]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary, buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  RACE_AGENT, RACE_PLANNER_GATE_RADIUS, RACE_ARRIVE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';
import { plotTrajectory } from './lib/trajectory-plot';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const out = process.argv[3] ?? `/tmp/slalom-${which}.png`;
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const lib = which === 'v3'
  ? buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')), { generatedControls: gen })
  : which === 'v2'
    ? buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json')), { generatedControls: gen })
    : buildKinematicLibrary();

const bounds = { x0: -50, z0: -25, x1: 30, z1: 25 };
const POLY = [{ id: 0, y: 0, ring: [[bounds.x0, bounds.z0], [bounds.x1, bounds.z0], [bounds.x1, bounds.z1], [bounds.x0, bounds.z1]] as [number, number][] }];
const world = new InMemoryNavWorld(POLY, []);
const spawn = { x: -40, z: 0, heading: 0, speed: 14, t: 0 };
const gates = [
  { x: -18, z: 5, heading: 0, speed: 5, t: 0 },
  { x: 0, z: -5, heading: 0, speed: 5, t: 0 },
  { x: 18, z: 5, heading: 0, speed: 5, t: 0 },
];
const t0 = performance.now();
const res = planVehicleMultiGoal({
  start: spawn, gates, world, agent: RACE_AGENT, lib,
  deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
  envOptions: dt ? { analyticDriveThrough: true } : undefined,
});
const ms = performance.now() - t0;
const path = res.path;
const speeds = path.map((p) => Math.abs(p.speed));
const peak = Math.max(0, ...speeds);
const minInt = path.length > 2 ? Math.min(...speeds.slice(1, -1)) : 0;
const dist = path.reduce((a, p, i) => (i === 0 ? 0 : a + Math.hypot(p.x - path[i - 1]!.x, p.z - path[i - 1]!.z)), 0);
console.log(`${which} gen=${gen} dt=${dt}: found=${res.found} pts=${path.length} dist=${dist.toFixed(0)}m peakV=${peak.toFixed(1)} minInteriorV=${minInt.toFixed(1)} cost=${res.cost.toFixed(1)} exp=${res.stats.expansions} ms=${ms.toFixed(0)}`);
// Densify: interpolate between sparse plan endpoints (~7-15 m apart) so the
// plotter's 10 m teleport guard doesn't drop segments.
const samples: { t: number; x: number; z: number; speed: number }[] = [];
for (let i = 0; i < path.length; i++) {
  if (i > 0) {
    const a = path[i - 1]!, b = path[i]!;
    const d = Math.hypot(b.x - a.x, b.z - a.z);
    const steps = Math.max(1, Math.ceil(d / 1.0));
    for (let s = 1; s < steps; s++) {
      const u = s / steps;
      samples.push({ t: samples.length, x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u, speed: Math.abs(a.speed + (b.speed - a.speed) * u) });
    }
  }
  samples.push({ t: samples.length, x: path[i]!.x, z: path[i]!.z, speed: Math.abs(path[i]!.speed) });
}
const png = plotTrajectory(out,
  { bounds, waypoints: [spawn, ...gates], spawn, arriveRadius: RACE_ARRIVE_RADIUS },
  samples,
  { title: `${which} SLALOM PLAN (${gen ? 'gen+' : 'hand+'}${dt ? 'reprice' : 'baseline'}) — peakV=${peak.toFixed(1)} minInteriorV=${minInt.toFixed(1)} (0=stops)`, vMax: 30 });
console.log(`plot: ${png}`);
