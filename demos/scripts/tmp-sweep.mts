// TEMP — MPPI knob sweep. Not committed.
// usage: npx tsx scripts/tmp-sweep.mts <kin|v2|v3> '<json overrides>' [maxSec] [course] [plotPath]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse, RACE_ARRIVE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import { TrajectoryRecorder } from './lib/trajectory-plot';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'kin';
const overrides = process.argv[3] ? JSON.parse(process.argv[3]) : {};
const maxSec = Number(process.argv[4] ?? 120);
const variant = (process.argv[5] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const plotPath = process.argv[6];
const entry =
  which === 'v3'
    ? v3Entry('v3', v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8'))))
    : which === 'v2'
      ? v2Entry('v2', modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8'))))
      : kinematicEntry('kin');
const course = buildRaceCourse(variant);
const scenario = await createRaceScenario({
  entries: [entry],
  targetLaps: 2,
  syncHold: false,
  course,
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc', mpcOverrides: overrides, analyticDriveThrough: process.env.KINOCAT_ANALYTIC_DT === '1' },
});
const rec = plotPath ? new TrajectoryRecorder() : null;
while (scenario.simTime() < maxSec) {
  const r = scenario.tick();
  rec?.record(r.simTime, r.cars[0]!);
  if (r.allFinished) break;
}
const s = scenario.status()[0]!;
const lapStr = s.laps.map((l) => l.duration.toFixed(1)).join(',');
console.log(
  `${which} ${JSON.stringify(overrides)} → laps=${s.laps.length} [${lapStr}] t=${scenario.simTime().toFixed(1)} ` +
  `recov=${s.quality.recoveryCount} stopped=${s.quality.timeStopped.toFixed(1)}s meanSpd=${s.quality.meanSpeed.toFixed(1)} offTrack=${s.offTrackEvents}`,
);
if (rec && plotPath) {
  const saved = rec.save(
    resolve(plotPath),
    { bounds: course.bounds, waypoints: course.waypoints, walls: course.walls, spawn: course.spawn, arriveRadius: RACE_ARRIVE_RADIUS },
    `${which} MPPI ${variant} — laps=${s.laps.length} [${lapStr}] recov=${s.quality.recoveryCount}`,
    30,
  );
  console.log(`plot: ${saved}`);
}
scenario.dispose();
