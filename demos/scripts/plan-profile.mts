// Per-expansion cost profile at the real operating point (weighted-A*). Tells
// us which core cost to attack next: the Reeds-Shepp heuristic or the footprint
// collision sweeps. Reports counter ratios + wall so the target is measured,
// not guessed. Run with NO other jobs (wall time is meaningful only uncontended).
//
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 KINOCAT_SPEED_PROFILE=1 \
//          npx tsx scripts/plan-profile.mts [weight=2]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildLearnedRaceLibraryV3, buildRaceCourse, RACE_AGENT, RACE_PLANNER_GATE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const weight = Number(process.argv[2] ?? 2);
const lib = buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')), { generatedControls: gen });
const course = buildRaceCourse('technical');
const world = new InMemoryNavWorld(course.polygons, course.obstacles);
const wps = course.waypoints;
const starts: CarKinematicState[] = [
  { x: course.spawn.x, z: course.spawn.z, heading: course.spawn.heading, speed: 8, t: 0 },
  { x: wps[0]!.x - 3, z: wps[0]!.z + 2, heading: 0.4, speed: 10, t: 0 },
  { x: wps[1 % wps.length]!.x + 2, z: wps[1 % wps.length]!.z - 3, heading: -0.6, speed: 6, t: 0 },
  { x: wps[2 % wps.length]!.x, z: wps[2 % wps.length]!.z + 3, heading: 1.2, speed: 12, t: 0 },
];

// Warm up (JIT) then profile.
for (let w = 0; w < 2; w++) {
  for (let s = 0; s < starts.length; s++) {
    const gates = [wps[s % wps.length]!, wps[(s + 1) % wps.length]!].map((g) => ({ x: g.x, z: g.z, heading: 0, speed: 0, t: 0 }));
    planVehicleMultiGoal({ start: starts[s]!, gates, world, agent: RACE_AGENT, lib, deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS, plannerOptions: { weight, profile: 'counts' } });
  }
}

console.log(`\nv3 technical per-expansion profile — weight ${weight} gen=${gen} reprice=${dt}\n`);
console.log(`  pose | ${'exp'.padStart(6)} ${'ms'.padStart(6)} ${'µs/exp'.padStart(7)} | ${'heur/exp'.padStart(8)} ${'coll/exp'.padStart(8)} ${'succ/exp'.padStart(8)} ${'analytic'.padStart(8)}`);
for (let s = 0; s < starts.length; s++) {
  const gates = [wps[s % wps.length]!, wps[(s + 1) % wps.length]!].map((g) => ({ x: g.x, z: g.z, heading: 0, speed: 0, t: 0 }));
  const t0 = performance.now();
  const res = planVehicleMultiGoal({ start: starts[s]!, gates, world, agent: RACE_AGENT, lib, deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS, plannerOptions: { weight, profile: 'counts' } });
  const ms = performance.now() - t0;
  const c = res.stats.counters;
  const exp = res.stats.expansions || 1;
  console.log(`   s${s}  | ${String(res.stats.expansions).padStart(6)} ${ms.toFixed(0).padStart(6)} ${(ms * 1000 / exp).toFixed(1).padStart(7)} | ${(c.heuristicCalls / exp).toFixed(1).padStart(8)} ${(c.collisionChecks / exp).toFixed(1).padStart(8)} ${(c.successorsTotal / exp).toFixed(1).padStart(8)} ${String(c.analyticShots).padStart(8)}`);
}
console.log(`\n  heur/exp = Reeds-Shepp heuristic calls per expansion; coll/exp = footprint collision tests per expansion.`);
console.log(`  Whichever is larger × its unit cost is the dominant per-expansion term to attack next.\n`);
