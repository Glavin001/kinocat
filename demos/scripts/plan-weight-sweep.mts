// Weighted-A* expansion/quality tradeoff on the technical course. The core
// search blows the real-time budget because some poses expand tens of
// thousands of nodes (measured up to 40k). Weighted-A* (f = g + weight·h) is
// the classic lever: a small ε over-estimate collapses expansions at a bounded
// cost increase. Reports expansions + plan cost per weight per (start, gate) so
// we can pick the weight that fits the budget without wrecking the line.
//
// usage: KINOCAT_SPEED_PROFILE=1 npx tsx scripts/plan-weight-sweep.mts [v3|v2|kin]
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
const starts: CarKinematicState[] = [
  { x: course.spawn.x, z: course.spawn.z, heading: course.spawn.heading, speed: 8, t: 0 },
  { x: wps[0]!.x - 3, z: wps[0]!.z + 2, heading: 0.4, speed: 10, t: 0 },
  { x: wps[1 % wps.length]!.x + 2, z: wps[1 % wps.length]!.z - 3, heading: -0.6, speed: 6, t: 0 },
  { x: wps[2 % wps.length]!.x, z: wps[2 % wps.length]!.z + 3, heading: 1.2, speed: 12, t: 0 },
];
const WEIGHTS = [1, 1.2, 1.5, 2, 3];

const plan = (start: CarKinematicState, gi: number, weight: number) => {
  const gates = [wps[gi % wps.length]!, wps[(gi + 1) % wps.length]!].map(
    (w) => ({ x: w.x, z: w.z, heading: 0, speed: 0, t: 0 }),
  );
  const t0 = performance.now();
  const res = planVehicleMultiGoal({
    start, gates, world, agent: RACE_AGENT, lib,
    deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
    plannerOptions: weight !== 1 ? { weight } : undefined,
  });
  return { ms: performance.now() - t0, exp: res.stats.expansions, found: res.found, cost: res.cost };
};

console.log(`\n${which} technical — weighted-A* expansion/cost tradeoff\n`);
const header = ['start/gate', ...WEIGHTS.map((w) => `w=${w}`)];
console.log('  ' + header.map((s) => s.padStart(12)).join(''));
const totals: Record<number, number> = {};
for (const w of WEIGHTS) totals[w] = 0;
const baseCostTot: Record<number, number> = {};
for (let s = 0; s < starts.length; s++) {
  for (let gi = 0; gi < 2; gi++) {
    const cells: string[] = [`s${s}g${gi}`];
    let baseCost = 0;
    for (const w of WEIGHTS) {
      const r = plan(starts[s]!, gi, w);
      totals[w]! += r.exp;
      if (w === 1) baseCost = r.cost;
      const costTag = w === 1 ? '' : baseCost > 0 ? `+${(((r.cost - baseCost) / baseCost) * 100).toFixed(0)}%` : '';
      cells.push(`${r.exp}${costTag ? '(' + costTag + ')' : ''}`);
    }
    console.log('  ' + cells.map((s) => s.padStart(12)).join(''));
  }
}
console.log('\n  TOTAL expansions per weight:');
const base = totals[1]!;
for (const w of WEIGHTS) {
  console.log(`    w=${w}: ${totals[w]}  (${w === 1 ? 'baseline' : ((1 - totals[w]! / base) * 100).toFixed(0) + '% fewer'})`);
}
console.log('');
