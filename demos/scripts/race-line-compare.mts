// Clean single-plan slalom comparison: the isolated 3-gate slalom (plans in one
// call, no receding stitch), rendered per model with the fix on. Shows each
// model's intended line + speed profile through the same corner sequence.
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/race-line-compare.mts <kin|v2|v3> [out]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildKinematicLibrary, buildLearnedRaceLibraryV2, buildLearnedRaceLibraryV3,
  RACE_AGENT, RACE_PLANNER_GATE_RADIUS, RACE_ARRIVE_RADIUS,
} from '../app/lib/race-primitives-scenarios';
import { planVehicleMultiGoal } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import { v3FromJson, forwardSimV3, learnedForwardSimV2, kinematicForwardSim } from 'kinocat/agent';
import { modelFromJson } from '../app/lib/v2-model-file';
import { plotTrajectory } from './lib/trajectory-plot';
import type { CarKinematicState } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const out = process.argv[3] ?? `/tmp/slalom-${which}.png`;
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
// Keep BOTH the library (for planning) and the raw forward model (to re-roll
// primitives at render time for the true per-substep speed along the spline).
const v3m = which === 'v3' ? v3FromJson(readModel('v3-default.json')) : null;
const v2m = which === 'v2' ? modelFromJson(readModel('v2-default.json')) : null;
const forwardSim = which === 'v3' ? forwardSimV3(v3m!)
  : which === 'v2' ? learnedForwardSimV2(v2m!)
    : kinematicForwardSim(RACE_AGENT);
const lib = which === 'v3' ? buildLearnedRaceLibraryV3(v3m!, { generatedControls: gen })
  : which === 'v2' ? buildLearnedRaceLibraryV2(v2m!, { generatedControls: gen })
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
const t0 = performance.now();
const res = planVehicleMultiGoal({
  start: spawn, gates, world, agent: RACE_AGENT, lib,
  deadlineMs: 20000, maxExpansions: 3_000_000, gateRadius: RACE_PLANNER_GATE_RADIUS,
  envOptions: dt ? { analyticDriveThrough: true } : undefined,
});
const ms = performance.now() - t0;
const path = res.path;
const speeds = path.map((p) => Math.abs(p.speed));
const peak = Math.max(0, ...speeds);
const minInt = path.length > 2 ? Math.min(...speeds.slice(1, -1)) : 0;
const dist = path.reduce((a, p, i) => (i === 0 ? 0 : a + Math.hypot(p.x - path[i - 1]!.x, p.z - path[i - 1]!.z)), 0);
console.log(`${which} gen=${gen} dt=${dt}: found=${res.found} pts=${path.length} dist=${dist.toFixed(0)}m peakV=${peak.toFixed(1)} minInteriorV=${minInt.toFixed(1)} cost=${res.cost.toFixed(1)} exp=${res.stats.expansions} ms=${ms.toFixed(0)}`);
// Reconstruct the TRUE swept curve (arcs), not straight chords: each drive
// edge replays its primitive's local sweep transformed to world by the parent
// pose; each Reeds-Shepp edge uses its stored world-frame poses. Speed is
// interpolated across each segment (substeps carry no speed).
const primById = new Map<number, { controls: number[]; startSpeed: number; duration: number }>();
for (const p of lib.primitives) primById.set(p.id, p);
const SUB = 16; // render substeps per primitive (finer than the 6 baked in)
const samples: { t: number; x: number; z: number; speed: number }[] = [];
samples.push({ t: 0, x: path[0]!.x, z: path[0]!.z, speed: Math.abs(path[0]!.speed) });
const nodes = res.nodes ?? [];
for (let i = 1; i < path.length; i++) {
  const a = path[i - 1]!, b = path[i]!;
  const edge = nodes[i]?.edge;
  if (edge?.kind === 'drive') {
    const primId = (edge.data as { primId?: number }).primId;
    const prim = primId !== undefined ? primById.get(primId) : undefined;
    if (prim) {
      // Re-roll the primitive's controls through the model — reproduces the
      // EXACT trajectory + per-substep SPEED the planner projected. Local frame
      // (start at rest pose, the bucket start speed), then place into the world
      // at the parent pose.
      const cos = Math.cos(a.heading), sin = Math.sin(a.heading);
      let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: prim.startSpeed, t: 0 };
      const subDt = prim.duration / SUB;
      for (let k = 0; k < SUB; k++) {
        s = forwardSim(s, prim.controls, subDt);
        samples.push({
          t: samples.length,
          x: a.x + s.x * cos - s.z * sin,
          z: a.z + s.x * sin + s.z * cos,
          speed: Math.abs(s.speed),
        });
      }
      continue;
    }
  }
  // Reeds-Shepp analytic edge (geometric, not a model rollout): use its world
  // poses; speed decays from entry to the terminal linearly.
  const v0 = Math.abs(a.speed), v1 = Math.abs(b.speed);
  const poses = (edge?.data as { poses?: { x: number; z: number }[] } | undefined)?.poses
    ?? [{ x: a.x, z: a.z }, { x: b.x, z: b.z }];
  for (let s2 = 1; s2 < poses.length; s2++) {
    const u = s2 / (poses.length - 1);
    samples.push({ t: samples.length, x: poses[s2]!.x, z: poses[s2]!.z, speed: v0 + (v1 - v0) * u });
  }
}
const png = plotTrajectory(out,
  { bounds, waypoints: [spawn, ...gates], spawn, arriveRadius: RACE_ARRIVE_RADIUS },
  samples,
  { title: `${which} SLALOM PLAN (${gen ? 'gen+' : 'hand+'}${dt ? 'reprice' : 'baseline'}) — peakV=${peak.toFixed(1)} minInteriorV=${minInt.toFixed(1)} (0=stops)`, vMax: 30 });
console.log(`plot: ${png}`);
