// SEGMENT plan under PERFECT TRACKING (no physics): receding-horizon replan
// over a slice of the track, advancing the car exactly along each committed
// plan. The resulting line IS what the planner intends, realized perfectly —
// so comparing kin/v2/v3 shows each model's worldview (line + speed profile).
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/tmp-segment-plan.mts <kin|v2|v3> [startGate] [endGate] [out]
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
const gStart = Number(process.argv[3] ?? 0);
const gEnd = Number(process.argv[4] ?? 6);
const out = process.argv[5] ?? `/tmp/seg-${which}.png`;
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

const nearestIdx = (path: CarKinematicState[], g: { x: number; z: number }): number => {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < path.length; i++) {
    const d = Math.hypot(path[i]!.x - g.x, path[i]!.z - g.z);
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
};

// Start at the segment's first gate, at a modest entry speed.
let state: CarKinematicState = { ...wps[gStart]!, speed: 8, t: 0 };
const line: CarKinematicState[] = [state];
const t0 = performance.now();
let failed = 0;
for (let i = gStart; i < gEnd; i++) {
  const window = [wps[i]!, wps[i + 1]!, wps[Math.min(i + 2, wps.length - 1)]!].map((w) => ({ ...w }));
  const res = planVehicleMultiGoal({
    start: state, gates: window, world, agent: RACE_AGENT, lib,
    deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
    envOptions: dt ? { analyticDriveThrough: true } : undefined,
  });
  if (!res.found || res.path.length < 2) { failed++; state = { ...wps[i + 1]!, speed: 6, t: state.t }; line.push(state); continue; }
  // PERFECT TRACKING: commit up to the NEXT gate; the car IS the plan.
  const commitTo = Math.max(1, nearestIdx(res.path, wps[i + 1]!));
  for (const p of res.path.slice(1, commitTo + 1)) line.push({ ...p, t: p.t });
  state = line[line.length - 1]!;
}
const ms = performance.now() - t0;
const dist = line.reduce((a, p, i) => (i === 0 ? 0 : a + Math.hypot(p.x - line[i - 1]!.x, p.z - line[i - 1]!.z)), 0);
const speeds = line.map((p) => Math.abs(p.speed));
const peak = Math.max(0, ...speeds);
const mean = speeds.reduce((a, b) => a + b, 0) / speeds.length;
const minInterior = Math.min(...speeds.slice(1, -1));
console.log(`${which} gen=${gen} dt=${dt} gates ${gStart}-${gEnd}: pts=${line.length} failedWindows=${failed} dist=${dist.toFixed(0)}m peakV=${peak.toFixed(1)} meanV=${mean.toFixed(1)} minInteriorV=${minInterior.toFixed(1)} ms=${ms.toFixed(0)}`);
const samples = line.map((p, i) => ({ t: i, x: p.x, z: p.z, speed: Math.abs(p.speed) }));
const png = plotTrajectory(out,
  { bounds: course.bounds, waypoints: course.waypoints, walls: course.walls, spawn: course.spawn, arriveRadius: RACE_ARRIVE_RADIUS },
  samples,
  { title: `${which} SEGMENT PLAN gates ${gStart}-${gEnd} (perfect tracking) — peakV=${peak.toFixed(1)} meanV=${mean.toFixed(1)} minV=${minInterior.toFixed(1)}`, vMax: 30 });
console.log(`plot: ${png}`);
