// TEMP — single-car MPPI diagnosis. Not committed.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, splitAtGearCusps } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import { extendPlanForTracking, buildProgressGeometry } from 'kinocat/execute';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const variant = (process.argv[2] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const which = process.argv[3] ?? 'kin';
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
  course: buildRaceCourse(variant),
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc' },
});
let lastPrint = 0;
while (scenario.simTime() < 90) {
  const r = scenario.tick();
  if (r.allFinished) break;
  if (r.simTime - lastPrint >= 1) {
    lastPrint = r.simTime;
    const c = r.cars[0]!;
    const lc = c.metrics.liveControls;
    // Instrument the allowed-speed profile the progress cost would build.
    let vMin = Infinity;
    let vAt = 0;
    if (c.plan && c.plan.length >= 2) {
      const ext = extendPlanForTracking(c.plan, 50, 30);
      const geom = buildProgressGeometry(ext, {
        envelopeDecel: 8, envelopeLateralAccel: 12, usePlanSpeeds: false, ignoreTerminalSpeed: true,
      });
      // nearest sample to the car
      let ni = 0; let nd = Infinity;
      for (let i = 0; i < ext.length; i++) {
        const d = Math.hypot(ext[i]!.x - c.state.x, ext[i]!.z - c.state.z);
        if (d < nd) { nd = d; ni = i; }
      }
      vAt = geom.vAllow[ni]!;
      const sHere = geom.cum[ni]!;
      for (let i = ni; i < ext.length && geom.cum[i]! - sHere < 30; i++) {
        if (geom.vAllow[i]! < vMin) vMin = geom.vAllow[i]!;
      }
    }
    const segs = c.plan ? splitAtGearCusps(c.plan) : [];
    const segStr = segs.map((s0) => `${s0.length}${(s0.reduce((a, p) => a + p.speed, 0) < 0 ? 'R' : 'F')}`).join('/');
    console.log(
      `t=${r.simTime.toFixed(1)} pos=(${c.state.x.toFixed(1)},${c.state.z.toFixed(1)}) v=${c.state.speed.toFixed(1)} ` +
      `wp=${c.loopIndex} laps=${c.laps.length} steer=${lc?.steer.toFixed(2)} thr=${lc?.throttle.toFixed(2)} brk=${lc?.brake.toFixed(2)} ` +
      `planLen=${c.plan?.length ?? 0} segs=${segStr} vAllowHere=${vAt.toFixed(1)} vAllowMin30=${vMin.toFixed(1)} mpcMs=${c.diagnostics.mpcSolveMsAvg.toFixed(2)}`,
    );
  }
}
const s = scenario.status()[0]!;
console.log(`laps=${s.laps.length} offTrack=${s.offTrackEvents} recov=${s.quality.recoveryCount} mpcMsAvg=${s.diagnostics.mpcSolveMsAvg.toFixed(2)} solves=${s.diagnostics.mpcSolveCount}`);
scenario.dispose();
