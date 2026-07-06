// TEMP — dump the 5 s leading up to each stuck-recovery trigger. Not committed.
// usage: npx tsx scripts/tmp-wedge3.mts <kin|v2|v3> [maxSec] [maxDumps]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, splitAtGearCusps } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'v3';
const maxSec = Number(process.argv[3] ?? 60);
const maxDumps = Number(process.argv[4] ?? 3);
const entry =
  which === 'v3'
    ? v3Entry('v3', v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8'))))
    : which === 'v2'
      ? v2Entry('v2', modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8'))))
      : kinematicEntry('kin');
const scenario = await createRaceScenario({
  entries: [entry],
  targetLaps: 2,
  syncHold: false,
  course: buildRaceCourse('open'),
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc' },
});

interface Row { t: number; x: number; z: number; h: number; v: number; steer: number; thr: number; brk: number; segs: string; planAge: number; wp: number }
const ring: Row[] = [];
let lastRecov = 0;
let dumps = 0;
let lastRowT = -1;
while (scenario.simTime() < maxSec && dumps < maxDumps) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  if (r.simTime - lastRowT >= 0.25) {
    lastRowT = r.simTime;
    const lc = c.metrics.liveControls;
    const segs = c.plan
      ? splitAtGearCusps(c.plan).map((s0) => `${s0.length}${s0.reduce((a, p) => a + p.speed, 0) < 0 ? 'R' : 'F'}`).join('/')
      : '-';
    ring.push({
      t: r.simTime, x: c.state.x, z: c.state.z, h: c.state.heading, v: c.state.speed,
      steer: lc?.steer ?? 0, thr: lc?.throttle ?? 0, brk: lc?.brake ?? 0,
      segs, planAge: c.diagnostics.planAgeMs, wp: c.loopIndex,
    });
    while (ring.length > 24) ring.shift();
  }
  if (c.quality.recoveryCount > lastRecov) {
    lastRecov = c.quality.recoveryCount;
    dumps++;
    console.log(`\n=== RECOVERY #${lastRecov} at t=${r.simTime.toFixed(1)} (dump of preceding ${(ring.length * 0.25).toFixed(0)}s) ===`);
    for (const row of ring) {
      console.log(
        `t=${row.t.toFixed(2).padStart(6)} wp=${row.wp} pos=(${row.x.toFixed(1)},${row.z.toFixed(1)}) h=${row.h.toFixed(2)} v=${row.v.toFixed(1).padStart(5)} ` +
        `steer=${row.steer.toFixed(2).padStart(5)} thr=${row.thr.toFixed(2).padStart(5)} brk=${row.brk.toFixed(2)} segs=${row.segs} planAge=${row.planAge.toFixed(0)}`,
      );
    }
    ring.length = 0;
  }
}
scenario.dispose();
