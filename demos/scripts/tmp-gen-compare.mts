import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLearnedRaceLibraryV3, RACE_AGENT } from '../app/lib/race-primitives-scenarios';
import { v3FromJson, forwardSimV3 } from 'kinocat/agent';
import { designControlSet, coverageReport } from 'kinocat/primitives';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const m = v3FromJson(JSON.parse(readFileSync(resolve(root,'demos/public/models/v3-default.json'),'utf-8')));
const sim = forwardSimV3(m);
const lib = buildLearnedRaceLibraryV3(m);
const common = { forwardSim: sim, duration: 0.55, substeps: 6, maxSteer: RACE_AGENT.maxSteerAngle ?? 0.6, maxDrive: 4000, maxBrake: 2000 };
for (const v of [14, 20]) {
  const handControls = lib.lookup(v).map(p => p.controls);
  const budget = handControls.length;
  const gen = designControlSet({ ...common, startSpeed: v, budget, reverseSlots: 0 });
  const rh = coverageReport(handControls, { ...common, startSpeed: v });
  const rg = coverageReport(gen, { ...common, startSpeed: v });
  console.log(`\n@ ${v} m/s (budget ${budget})`);
  console.log(`  HAND: disp=${rh.dispersion.toFixed(2)} minPair=${rh.minPairwise.toFixed(2)} maxHead ${rh.maxHeadingSet.toFixed(2)}/${rh.maxHeadingReachable.toFixed(2)} asym=${rh.asymmetry.toFixed(2)}`);
  console.log(`  GEN : disp=${rg.dispersion.toFixed(2)} minPair=${rg.minPairwise.toFixed(2)} maxHead ${rg.maxHeadingSet.toFixed(2)}/${rg.maxHeadingReachable.toFixed(2)} asym=${rg.asymmetry.toFixed(2)} slots=${gen.length}`);
}
