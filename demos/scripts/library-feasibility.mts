import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLearnedRaceLibraryV3 } from '../app/lib/race-primitives-scenarios';
import { v3FromJson } from 'kinocat/agent';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const m = v3FromJson(JSON.parse(readFileSync(resolve(root,'demos/public/models/v3-default.json'),'utf-8')));
for (const gen of [false, true]) {
  const lib = buildLearnedRaceLibraryV3(m, { generatedControls: gen });
  let worst = { dvps: 0, sp: 0, es: 0, dur: 0, u: [] as number[] };
  for (const p of lib.primitives) {
    const dvps = Math.abs(p.end.speed - p.startSpeed) / p.duration; // m/s per s
    if (dvps > worst.dvps) worst = { dvps, sp: p.startSpeed, es: p.end.speed, dur: p.duration, u: p.controls };
  }
  console.log(`generated=${gen}: worst |dv/dt| = ${worst.dvps.toFixed(1)} m/s² (start ${worst.sp} -> end ${worst.es.toFixed(1)} over ${worst.dur}s, controls [${worst.u.map(x=>x.toFixed(0)).join(',')}])  [plant accel<=13.9 brake<=~26]`);
}
