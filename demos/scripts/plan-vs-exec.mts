// plan STABILITY / churn diagnostic. Not committed.
//
// Separates PLANNING error from EXECUTION error, and measures replan churn.
//   <out>-plans.png  every committed plan overlaid, coloured by replan order
//                    (blue = early → red = late). Plans stacked on one line =
//                    the planner commits/agrees; a fan of colours = thrash.
//                    Waypoint-advance replans (the lookahead goal shifting one
//                    gate forward) are drawn THICK so goal-shift jumps show.
//   <out>-exec.png   executed trajectory (speed-coloured) + committed plans in
//                    grey. Gap between grey plan and coloured drive = execution
//                    error.
//
// Also prints, per replan: churn vs the previous plan (mean/max lateral gap
// over their spatial overlap) and whether it was a waypoint-advance — so the
// "hits a gate, goal jumps 1 forward, plan overcorrects" hypothesis is
// measurable, not eyeballed.
//
// usage: npx tsx scripts/plan-vs-exec.mts <kin|v2|v3> [maxSec] [open|technical] <outPrefix>
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse, RACE_ARRIVE_RADIUS } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import { plotTrajectory, type PlanOverlay, type TrajectorySample } from './lib/trajectory-plot';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'v3';
const maxSec = Number(process.argv[3] ?? 100);
const variant = (process.argv[4] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const outPrefix = process.argv[5] ?? `/tmp/${which}-${variant}`;
const entry =
  which === 'v3'
    ? v3Entry('v3', v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8'))))
    : which === 'v2'
      ? v2Entry('v2', modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8'))))
      : kinematicEntry('kin');
const course = buildRaceCourse(variant);
const geom = {
  bounds: course.bounds,
  waypoints: course.waypoints,
  walls: course.walls,
  spawn: course.spawn,
  arriveRadius: RACE_ARRIVE_RADIUS,
};

type Pt = { x: number; z: number };
/** Mean & max lateral gap of `a`'s points to polyline `b`, over the first
 *  `arcCap` metres of `a` (the near-car region both plans share). */
function planGap(a: Pt[], b: Pt[], arcCap = 18): { mean: number; max: number } {
  if (a.length < 2 || b.length < 2) return { mean: 0, max: 0 };
  let sum = 0, n = 0, mx = 0, arc = 0;
  for (let i = 0; i < a.length; i++) {
    if (i > 0) arc += Math.hypot(a[i]!.x - a[i - 1]!.x, a[i]!.z - a[i - 1]!.z);
    if (arc > arcCap) break;
    let best = Infinity;
    for (let j = 1; j < b.length; j++) best = Math.min(best, segDist(a[i]!, b[j - 1]!, b[j]!));
    sum += best; n++; if (best > mx) mx = best;
  }
  return { mean: n ? sum / n : 0, max: mx };
}
function segDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x, dz = b.z - a.z;
  const L2 = dx * dx + dz * dz;
  const t = L2 > 1e-9 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.z - a.z) * dz) / L2)) : 0;
  return Math.hypot(p.x - (a.x + t * dx), p.z - (a.z + t * dz));
}

const scenario = await createRaceScenario({
  entries: [entry],
  targetLaps: 2,
  syncHold: false,
  course,
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc' },
});

const samples: TrajectorySample[] = [];
interface Commit { t: number; wp: number; cleared: number; advance: boolean; pts: Pt[]; gap: { mean: number; max: number } }
const commits: Commit[] = [];
let lastPlanRef: unknown = null;
let lastLoop = -1;
let prevPts: Pt[] | null = null;

while (scenario.simTime() < maxSec) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  samples.push({ t: r.simTime, x: c.state.x, z: c.state.z, speed: c.state.speed });
  if (c.plan && c.plan !== lastPlanRef) {
    lastPlanRef = c.plan;
    const pts = c.plan.map((p) => ({ x: p.x, z: p.z }));
    const advance = lastLoop >= 0 && c.loopIndex !== lastLoop;
    const gap = prevPts ? planGap(pts, prevPts) : { mean: 0, max: 0 };
    commits.push({ t: r.simTime, wp: c.loopIndex, cleared: c.loopIndex, advance, pts, gap });
    lastLoop = c.loopIndex;
    prevPts = pts;
  }
  if (r.allFinished) break;
}
const s = scenario.status()[0]!;
const lapStr = s.laps.map((l) => l.duration.toFixed(1)).join(',');

// ---- churn report ----
const sameCh = commits.filter((c) => !c.advance && c.gap.mean > 0);
const advCh = commits.filter((c) => c.advance && c.gap.mean > 0);
const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
console.log(
  `EXEC(${which}) laps=${s.laps.length} [${lapStr}] t=${scenario.simTime().toFixed(1)} ` +
  `recov=${s.quality.recoveryCount} stopped=${s.quality.timeStopped.toFixed(1)}s meanSpd=${s.quality.meanSpeed.toFixed(1)}`,
);
console.log(
  `REPLANS: total=${commits.length} advances=${advCh.length + commits.filter((c) => c.advance && c.gap.mean === 0).length} | ` +
  `churn(mean gap m): same-wp avg=${avg(sameCh.map((c) => c.gap.mean)).toFixed(2)} max=${Math.max(0, ...sameCh.map((c) => c.gap.max)).toFixed(2)} | ` +
  `advance avg=${avg(advCh.map((c) => c.gap.mean)).toFixed(2)} max=${Math.max(0, ...advCh.map((c) => c.gap.max)).toFixed(2)}`,
);
// The 8 largest-churn replans, with advance flag — the overcorrection suspects.
[...commits].sort((a, b) => b.gap.mean - a.gap.mean).slice(0, 8).forEach((c) => {
  console.log(`  churn t=${c.t.toFixed(1).padStart(5)} wp=${c.wp} ${c.advance ? 'ADVANCE' : 'same   '} meanGap=${c.gap.mean.toFixed(2)}m maxGap=${c.gap.max.toFixed(2)}m`);
});

// ---- plots ----
// (a) all plans coloured by replan order; advances thick.
const N = Math.max(1, commits.length - 1);
const planOverlays: PlanOverlay[] = commits.map((c, i) => {
  const frac = i / N; // 0 → 1
  const hue = (240 * (1 - frac)).toFixed(0); // blue(early) → red(late)
  return { t: c.t, pts: c.pts, stroke: `hsla(${hue}, 85%, 55%, ${c.advance ? 0.95 : 0.5})` };
});
const plansPng = plotTrajectory(`${outPrefix}-plans.png`, geom, samples, {
  title: `${which} REPLAN STABILITY (blue→red = early→late; bright = waypoint-advance) — ${commits.length} plans`,
  vMax: 30,
  plans: planOverlays,
});
console.log(`plans plot: ${plansPng}`);

// (b) exec + committed plans in grey.
const execPng = plotTrajectory(`${outPrefix}-exec.png`, geom, samples, {
  title: `${which} EXEC vs PLAN (grey = committed plans) — laps=${s.laps.length} [${lapStr}] recov=${s.quality.recoveryCount}`,
  vMax: 30,
  plans: commits.map((c) => ({ t: c.t, pts: c.pts })),
});
console.log(`exec plot: ${execPng}`);
scenario.dispose();
