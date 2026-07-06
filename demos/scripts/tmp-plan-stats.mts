// Planning-cost breakdown: same slalom, 4 configs, full PlanStats.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLearnedRaceLibraryV3, RACE_AGENT, RACE_PLANNER_GATE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const model = v3FromJson(JSON.parse(readFileSync(resolve(root,'demos/public/models/v3-default.json'),'utf-8')));
const POLY=[{id:0,y:0,ring:[[-60,-40],[60,-40],[60,40],[-60,40]] as [number,number][]}];
const world=new InMemoryNavWorld(POLY,[]);
const spawn={x:-40,z:0,heading:0,speed:14,t:0};
const gates=[{x:-18,z:5,heading:0,speed:5,t:0},{x:0,z:-5,heading:0,speed:5,t:0},{x:18,z:5,heading:0,speed:5,t:0}];

const configs = [
  { name: 'hand,  no-reprice', gen: false, dt: false },
  { name: 'hand,  reprice   ', gen: false, dt: true },
  { name: 'gen+dense, no-rep ', gen: true,  dt: false },
  { name: 'gen+dense, reprice', gen: true,  dt: true },
];
console.log('config             | found cost  exp   gen    dhit  h.calls  col.chk col.rej  ms    prims/bucket@14');
for (const cfg of configs) {
  const lib = buildLearnedRaceLibraryV3(model, { generatedControls: cfg.gen });
  const branch = lib.lookup(14).length;
  const t0 = performance.now();
  const res = planVehicleMultiGoal({ start: spawn, gates, world, agent: RACE_AGENT, lib,
    deadlineMs: 8000, maxExpansions: 600000, gateRadius: RACE_PLANNER_GATE_RADIUS,
    envOptions: cfg.dt ? { analyticDriveThrough: true } : undefined });
  const ms = performance.now() - t0;
  const c = res.stats.counters;
  console.log(`${cfg.name} | ${res.found?'Y':'N'}   ${res.cost.toFixed(1).padStart(4)} ${String(res.stats.expansions).padStart(5)} ${String(res.stats.generated).padStart(6)}  ${res.stats.deadlineHit?'HIT':' - '}  ${String(c.heuristicCalls).padStart(6)}  ${String(c.collisionChecks).padStart(6)} ${String(c.collisionRejects).padStart(6)}  ${ms.toFixed(0).padStart(4)}   ${branch}`);
}
