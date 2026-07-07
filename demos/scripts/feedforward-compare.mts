// WS-1½ control-feedforward A/B. Does seeding the MPPI prior from the plan's
// OWN primitive controls (instead of re-deriving them from geometry every
// tick) make a model-faithful plan execute better? Runs the SAME closed-loop
// segment twice for one model — feedforward OFF vs ON — and reports the
// tracking-fidelity + pace + stability deltas that matter, then plots both
// executed lines. This is the closed-loop counterpart to plan-vs-plant.mts
// (which measured that v3's plan is open-loop faithful in the first place).
//
// usage:
//   KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1 \
//     npx tsx scripts/feedforward-compare.mts <kin|v2|v3> [secs] [open|technical] [budgetMs]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, type RaceTuning } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse, RACE_ARRIVE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import { plotTrajectory, type TrajectorySample } from './lib/trajectory-plot';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));
const which = (process.argv[2] ?? 'v3') as 'kin' | 'v2' | 'v3';
const secs = Number(process.argv[3] ?? 60);
const variant = (process.argv[4] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const budgetMs = Number(process.argv[5] ?? 12000);
const gen = process.env.KINOCAT_GEN_CONTROLS === '1';
const dt = process.env.KINOCAT_ANALYTIC_DT === '1';
const sp = process.env.KINOCAT_SPEED_PROFILE === '1';

const makeEntry = () =>
  which === 'v3' ? v3Entry('v3', v3FromJson(readModel('v3-default.json')))
    : which === 'v2' ? v2Entry('v2', modelFromJson(readModel('v2-default.json')))
      : kinematicEntry('kin');

const course = buildRaceCourse(variant);
const geom = {
  bounds: course.bounds,
  waypoints: course.waypoints,
  walls: course.walls,
  spawn: course.spawn,
  arriveRadius: RACE_ARRIVE_RADIUS,
};

interface Run {
  samples: TrajectorySample[];
  meanSpeed: number;
  predErrRms: number;
  laps: number;
  bestLap: number;
  churnMean: number;
  recoveries: number;
  timeStopped: number;
  dist: number;
  wallStrikes: number;
  solveMsAvg: number;
}

async function run(feedforward: boolean): Promise<Run> {
  const tuning: Partial<RaceTuning> = {
    plannerBudgetMs: budgetMs,
    tracker: 'mpc',
    analyticDriveThrough: dt,
    enableSpeedProfile: sp,
    controlFeedforward: feedforward,
  };
  const scenario = await createRaceScenario({
    entries: [makeEntry()], targetLaps: 3, syncHold: false, course, tuning,
  });
  const samples: TrajectorySample[] = [];
  let solveTotal = 0, solveN = 0;
  while (scenario.simTime() < secs) {
    const r = scenario.tick();
    const c = r.cars[0]!;
    if (samples.length === 0 || scenario.simTime() - samples[samples.length - 1]!.t > 0.1) {
      samples.push({ t: scenario.simTime(), x: c.state.x, z: c.state.z, speed: Math.abs(c.state.speed) });
    }
    const m = c.metrics;
    if (m.mpcSolveCount > solveN) { solveTotal = m.mpcSolveMsAvg * m.mpcSolveCount; solveN = m.mpcSolveCount; }
    if (r.allFinished) break;
  }
  const s = scenario.status()[0]!;
  const lapDurs = s.laps.map((l) => l.duration);
  scenario.dispose();
  return {
    samples,
    meanSpeed: s.quality.meanSpeed,
    predErrRms: s.diagnostics.predErrorRms,
    laps: s.laps.length,
    bestLap: lapDurs.length ? Math.min(...lapDurs) : 0,
    churnMean: s.quality.planChurnMean,
    recoveries: s.quality.recoveryCount,
    timeStopped: s.quality.timeStopped,
    dist: s.quality.distanceTravelled,
    wallStrikes: s.wallStrikes,
    solveMsAvg: solveN ? solveTotal / solveN : 0,
  };
}

const off = await run(false);
const on = await run(true);

const pct = (a: number, b: number) => (a === 0 ? (b === 0 ? '  0.0%' : ' +inf ') : `${(((b - a) / Math.abs(a)) * 100 >= 0 ? '+' : '')}${(((b - a) / Math.abs(a)) * 100).toFixed(1)}%`);
const row = (label: string, a: number, b: number, unit = '', better: 'lower' | 'higher' = 'lower') => {
  const delta = better === 'lower' ? a - b : b - a; // positive = ON is better
  const mark = Math.abs(b - a) < 1e-9 ? ' ' : delta > 0 ? '✓' : '✗';
  console.log(`  ${label.padEnd(22)} ${a.toFixed(2).padStart(9)}${unit} ${b.toFixed(2).padStart(9)}${unit}   ${pct(a, b).padStart(7)} ${mark}`);
};

console.log(`\n=== control feedforward A/B — ${which} on ${variant} (${secs}s sim, ${gen ? 'gen ' : ''}${dt ? 'reprice ' : ''}${sp ? 'speedprofile ' : ''}budget=${budgetMs}ms) ===`);
console.log(`  ${'metric'.padEnd(22)} ${'FF off'.padStart(10)} ${'FF on'.padStart(10)}   ${'Δ'.padStart(7)}   (✓ = feedforward better)`);
row('predErrRms (m)', off.predErrRms, on.predErrRms, '', 'lower');
row('meanSpeed (m/s)', off.meanSpeed, on.meanSpeed, '', 'higher');
row('laps', off.laps, on.laps, '', 'higher');
row('bestLap (s)', off.bestLap, on.bestLap, '', 'lower');
row('planChurnMean (m)', off.churnMean, on.churnMean, '', 'lower');
row('recoveries', off.recoveries, on.recoveries, '', 'lower');
row('timeStopped (s)', off.timeStopped, on.timeStopped, '', 'lower');
row('distanceTravelled (m)', off.dist, on.dist, '', 'lower');
row('wallStrikes', off.wallStrikes, on.wallStrikes, '', 'lower');
row('mpcSolveMsAvg', off.solveMsAvg, on.solveMsAvg, '', 'lower');

plotTrajectory(`/tmp/ff-${which}-${variant}-off.png`, geom, off.samples, { title: `${which} ${variant} — feedforward OFF (predErr ${off.predErrRms.toFixed(2)}m, mean ${off.meanSpeed.toFixed(1)})`, vMax: 30 });
plotTrajectory(`/tmp/ff-${which}-${variant}-on.png`, geom, on.samples, { title: `${which} ${variant} — feedforward ON (predErr ${on.predErrRms.toFixed(2)}m, mean ${on.meanSpeed.toFixed(1)})`, vMax: 30 });
console.log(`\n  plots: /tmp/ff-${which}-${variant}-off.png  /tmp/ff-${which}-${variant}-on.png\n`);
