import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLearnedRaceLibraryV3, RACE_AGENT, RACE_PLANNER_GATE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const lib = buildLearnedRaceLibraryV3(v3FromJson(JSON.parse(readFileSync(resolve(root,'demos/public/models/v3-default.json'),'utf-8'))));
const POLY=[{id:0,y:0,ring:[[-60,-40],[60,-40],[60,40],[-60,40]] as [number,number][]}];
const world=new InMemoryNavWorld(POLY,[]);
const spawn={x:-40,z:0,heading:0,speed:14,t:0};
const gates=[{x:-18,z:5,heading:0,speed:5,t:0},{x:0,z:-5,heading:0,speed:5,t:0},{x:18,z:5,heading:0,speed:5,t:0}];
for (const [label, ana, exp] of [['analytic OFF 400k', {analyticExpansion: undefined as any}, 400000],['analytic OFF 3M', {analyticExpansion: undefined as any}, 3000000]] as const){
  const t0=performance.now();
  const res=planVehicleMultiGoal({start:spawn,gates,world,agent:RACE_AGENT,lib,deadlineMs:20000,maxExpansions:exp,gateRadius:RACE_PLANNER_GATE_RADIUS,envOptions:ana as any});
  const minInt=res.path.length>2?Math.min(...res.path.slice(1,-1).map(p=>Math.abs(p.speed))):-1;
  console.log(`${label}: found=${res.found} cost=${res.cost.toFixed(2)} exp=${res.stats.expansions} len=${res.path.length} minIntV=${minInt.toFixed(1)} ms=${(performance.now()-t0).toFixed(0)}`);
}
