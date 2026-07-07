import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { RACE_AGENT } from '../app/lib/race-primitives-scenarios';
import { v3FromJson, forwardSimV3 } from 'kinocat/agent';
import { designControlSet, rollEndpoint } from 'kinocat/primitives';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const m = v3FromJson(JSON.parse(readFileSync(resolve(root,'demos/public/models/v3-default.json'),'utf-8')));
const sim = forwardSimV3(m);
const common = { forwardSim: sim, maxSteer: RACE_AGENT.maxSteerAngle ?? 0.6, maxDrive: 4000, maxBrake: 2000 };
console.log('Per-bucket generated sets (rolled through v3 dynamics AT that speed):');
for (const [v, dur, rev] of [[0,1.5,5],[8,0.55,0],[20,0.55,0],[28,0.55,0]] as const){
  const ctrls = designControlSet({ ...common, startSpeed: v, duration: dur, substeps: 6, budget: v<2?18:14, reverseSlots: rev });
  const ends = ctrls.map(u => rollEndpoint(sim, u, v, dur, 6));
  const ft = ends.find(e => Math.abs(e.dHeading)<0.05 && e.controls[1]===4000);
  const maxTurn = ends.reduce((a,b)=>Math.abs(b.dHeading)>Math.abs(a.dHeading)?b:a);
  const accel = ft ? ft.dv/dur : NaN;
  console.log(`  v=${String(v).padStart(2)}: full-throttle accel=${accel.toFixed(1)} m/s² | max turn dh=${maxTurn.dHeading.toFixed(2)} R≈${(Math.hypot(maxTurn.dx,maxTurn.dz)/Math.max(1e-3,Math.abs(maxTurn.dHeading))).toFixed(1)}m | slots=${ctrls.length} rev=${ctrls.filter(u=>u[1]<0).length}`);
}
