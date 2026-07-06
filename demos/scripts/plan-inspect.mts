// Inspect a plan's true per-primitive profile: endpoint speeds, segment length,
// heading change (curvature), and edge kind (primitive arc vs Reeds-Shepp
// analytic). Reveals what the coarse straight-chord render hides.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/plan-inspect.mts <kin|v2|v3>
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary, buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  RACE_AGENT, RACE_PLANNER_GATE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const lib = which === 'v3' ? buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')), { generatedControls: gen })
  : which === 'v2' ? buildLearnedRaceLibraryV2(modelFromJson(readModel('v2-default.json')), { generatedControls: gen })
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
const res = planVehicleMultiGoal({
  start: spawn, gates, world, agent: RACE_AGENT, lib,
  deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
  envOptions: dt ? { analyticDriveThrough: true } : undefined,
});
console.log(`${which}: found=${res.found} pts=${res.path.length} cost=${res.cost.toFixed(2)}`);
const gateStr = (p: { x: number; z: number }) => {
  for (let g = 0; g < gates.length; g++) if (Math.hypot(p.x - gates[g]!.x, p.z - gates[g]!.z) < RACE_PLANNER_GATE_RADIUS + 0.5) return ` <- GATE${g + 1}`;
  return '';
};
console.log(' idx |    x     z   | head  | speed |  dt  | segLen | dHead |  R(m)  | a(m/s²) | edge');
const nodes = res.nodes ?? [];
for (let i = 0; i < res.path.length; i++) {
  const p = res.path[i]!;
  const prev = i > 0 ? res.path[i - 1]! : null;
  const segLen = prev ? Math.hypot(p.x - prev.x, p.z - prev.z) : 0;
  let dHead = prev ? p.heading - prev.heading : 0;
  while (dHead > Math.PI) dHead -= 2 * Math.PI; while (dHead < -Math.PI) dHead += 2 * Math.PI;
  const R = Math.abs(dHead) > 1e-3 ? segLen / Math.abs(dHead) : Infinity;
  const dtSeg = prev ? p.t - prev.t : 0;
  const accel = dtSeg > 1e-6 && prev ? (p.speed - prev.speed) / dtSeg : 0;
  const edge = nodes[i]?.edge;
  const kind = edge?.kind ?? '-';
  console.log(` ${String(i).padStart(3)} | ${p.x.toFixed(1).padStart(5)} ${p.z.toFixed(1).padStart(5)} | ${p.heading.toFixed(2).padStart(5)} | ${p.speed.toFixed(1).padStart(5)} | ${dtSeg.toFixed(2)} | ${segLen.toFixed(1).padStart(5)} | ${dHead.toFixed(2).padStart(5)} | ${(R === Infinity ? '  inf' : R.toFixed(1)).padStart(6)} | ${accel.toFixed(1).padStart(6)} | ${kind}${gateStr(p)}`);
}
