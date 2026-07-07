// Isolated high-speed CORNER-ENTRY harness (reusable). A straight run-up into a
// single hard corner, driven closed-loop under MPPI — reproduces the v3
// high-speed wedge in ~15 s sim instead of a full lap, for fast iteration on
// the executor. Reports whether the car makes the corner or overshoots/wedges,
// plus peak speed / stopped time / recoveries, and plots the trajectory.
//
// usage: KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 npx tsx scripts/corner-entry.mts <kin|v2|v3> [maxSec] [overridesJson] [out]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, PHYSICS_DT } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { RACE_ARRIVE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import { TrajectoryRecorder } from './lib/trajectory-plot';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = process.argv[2] ?? 'v3';
const maxSec = Number(process.argv[3] ?? 18);
const overrides = process.argv[4] ? JSON.parse(process.argv[4]) : {};
const out = process.argv[5];
const entry = which === 'v3' ? v3Entry('v3', v3FromJson(readModel('v3-default.json')))
  : which === 'v2' ? v2Entry('v2', modelFromJson(readModel('v2-default.json')))
    : kinematicEntry('kin');

// Course: a straight to build speed, then a TIGHT slalom (rapid alternating
// gates) entered hot — the sequence that stresses the executor at speed. Set
// KINOCAT_CORNER=single for the simpler one-corner variant.
const g = (x: number, z: number): CarKinematicState => ({ x, z, heading: 0, speed: 5, t: 0 });
const waypoints = process.env.KINOCAT_CORNER === 'single'
  ? [g(0, 0), g(55, 0), g(72, -14), g(66, -34), g(45, -42)]
  : [g(0, 0), g(45, 0), g(63, 6), g(85, -6), g(107, 6), g(129, -6), g(150, 2)];
const bounds = { x0: -20, z0: -60, x1: 165, z1: 25 };
const course = {
  bounds,
  polygons: [{ id: 0, y: 0, ring: [[bounds.x0, bounds.z0], [bounds.x1, bounds.z0], [bounds.x1, bounds.z1], [bounds.x0, bounds.z1]] as [number, number][] }],
  obstacles: [] as [number, number][][],
  walls: [] as { x: number; z: number; hx: number; hz: number; height: number }[],
  waypoints,
  spawn: { x: -10, z: 0, heading: 0, speed: 0, t: 0 } as CarKinematicState,
};

const scenario = await createRaceScenario({
  entries: [entry], targetLaps: 1, syncHold: false, course: course as never,
  tuning: {
    plannerBudgetMs: Number(process.env.KINOCAT_PLANNER_BUDGET ?? 3000),
    tracker: 'mpc',
    analyticDriveThrough: process.env.KINOCAT_ANALYTIC_DT === '1',
    enableSpeedProfile: process.env.KINOCAT_SPEED_PROFILE === '1',
    mpcOverrides: overrides,
  },
});
const rec = out ? new TrajectoryRecorder() : null;
let maxWp = 0;
while (scenario.simTime() < maxSec) {
  const r = scenario.tick();
  rec?.record(r.simTime, r.cars[0]!);
  maxWp = Math.max(maxWp, r.cars[0]!.loopIndex);
  if (r.allFinished) break;
}
const s = scenario.status()[0]!;
console.log(
  `${which} corner-entry: reachedWp=${maxWp}/${waypoints.length - 1} laps=${s.laps.length} ` +
  `peakV=${s.metrics.peakSpeed.toFixed(1)} meanV=${s.quality.meanSpeed.toFixed(1)} ` +
  `stopped=${s.quality.timeStopped.toFixed(1)}s recov=${s.quality.recoveryCount} ` +
  `churn=${s.quality.planChurnMean.toFixed(2)}m offTrack=${s.offTrackEvents} t=${scenario.simTime().toFixed(1)}`,
);
if (rec && out) {
  const png = rec.save(resolve(out),
    { bounds: course.bounds, waypoints: course.waypoints, spawn: course.spawn, arriveRadius: RACE_ARRIVE_RADIUS },
    `${which} corner-entry — reachedWp=${maxWp} peakV=${s.metrics.peakSpeed.toFixed(1)} recov=${s.quality.recoveryCount}`, 30);
  console.log(`plot: ${png}`);
}
scenario.dispose();
void PHYSICS_DT;
