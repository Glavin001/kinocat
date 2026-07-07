// Direct core-search cost A/B: how many NODES does a real technical-course
// replan expand, with the obstacle-aware grid heuristic OFF vs ON? Expansions
// (not wall-clock, which is noise under load) is the honest measure of core
// search efficiency. Plans the same gate window from several representative
// start poses and reports expansions + generated + wall ms per config.
//
// usage: KINOCAT_SPEED_PROFILE=1 npx tsx scripts/plan-expansions.mts [v3|v2|kin]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary, buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  buildRaceCourse, RACE_AGENT, RACE_PLANNER_GATE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';
import type { CarKinematicState } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const lib: MotionPrimitiveLibrary =
  which === 'v3' ? buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')))
    : which === 'v2' ? buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json')))
      : buildKinematicLibrary();

const course = buildRaceCourse('technical');
const world = new InMemoryNavWorld(course.polygons, course.obstacles);
const wps = course.waypoints;

// Representative start poses: the spawn, plus a few gate-relative poses that
// stress the walled sections (offset from a waypoint, angled).
const starts: CarKinematicState[] = [
  { x: course.spawn.x, z: course.spawn.z, heading: course.spawn.heading, speed: 8, t: 0 },
  { x: wps[0]!.x - 3, z: wps[0]!.z + 2, heading: 0.4, speed: 10, t: 0 },
  { x: wps[1 % wps.length]!.x + 2, z: wps[1 % wps.length]!.z - 3, heading: -0.6, speed: 6, t: 0 },
  { x: wps[2 % wps.length]!.x, z: wps[2 % wps.length]!.z + 3, heading: 1.2, speed: 12, t: 0 },
];

const plan = (start: CarKinematicState, gi: number, grid: boolean) => {
  const gates = [wps[gi % wps.length]!, wps[(gi + 1) % wps.length]!].map(
    (w) => ({ x: w.x, z: w.z, heading: 0, speed: 0, t: 0 }),
  );
  const t0 = performance.now();
  const res = planVehicleMultiGoal({
    start, gates, world, agent: RACE_AGENT, lib,
    deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
    // grid heuristic is ON by default; pass false to isolate its effect.
    envOptions: grid ? undefined : { gridHeuristic: false },
  });
  return { ms: performance.now() - t0, exp: res.stats.expansions, gen: res.stats.generated, found: res.found, cost: res.cost };
};

console.log(`\n${which} technical — grid heuristic OFF vs ON (unbounded budget, so expansions reflect true search size)\n`);
console.log(`  start  gates | ${'OFF exp'.padStart(9)} ${'ON exp'.padStart(9)} ${'Δexp'.padStart(7)} | ${'OFF ms'.padStart(7)} ${'ON ms'.padStart(7)} | cost match`);
let totOff = 0, totOn = 0;
for (let s = 0; s < starts.length; s++) {
  for (let gi = 0; gi < 2; gi++) {
    const off = plan(starts[s]!, gi, false);
    const on = plan(starts[s]!, gi, true);
    totOff += off.exp; totOn += on.exp;
    const dpct = off.exp > 0 ? `${(((on.exp - off.exp) / off.exp) * 100).toFixed(0)}%` : '—';
    const costMatch = Math.abs(off.cost - on.cost) < 0.01 ? 'ok' : `OFF ${off.cost.toFixed(1)} ON ${on.cost.toFixed(1)}`;
    console.log(`   s${s}    g${gi}   | ${String(off.exp).padStart(9)} ${String(on.exp).padStart(9)} ${dpct.padStart(7)} | ${off.ms.toFixed(0).padStart(7)} ${on.ms.toFixed(0).padStart(7)} | ${off.found && on.found ? costMatch : 'FOUND MISMATCH'}`);
  }
}
console.log(`\n  TOTAL expansions: OFF ${totOff}  ON ${totOn}  → ${((1 - totOn / totOff) * 100).toFixed(0)}% fewer with the grid heuristic\n`);
