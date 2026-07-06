// TEMP — capture a wedged MPPI state for a LEARNED entry and dissect. Not committed.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario, splitAtGearCusps } from '../app/lib/race-scenario';
import { v2Entry, v3Entry, kinematicEntry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import {
  extendPlanForTracking, buildProgressGeometry, scoreRolloutProgress,
} from 'kinocat/execute';
import { v3FromJson, forwardSimV3Rollout, learnedForwardSimV2 } from 'kinocat/agent';
import type { CarKinematicState, ForwardSim } from 'kinocat/agent';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'v2';
let entry;
let sim: ForwardSim<CarKinematicState>;
if (which === 'v3') {
  const m = v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8')));
  entry = v3Entry('v3', m);
  sim = forwardSimV3Rollout(m);
} else {
  const m = modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8')));
  entry = v2Entry('v2', m);
  sim = learnedForwardSimV2(m);
}

const scenario = await createRaceScenario({
  entries: [entry],
  targetLaps: 2,
  syncHold: false,
  course: buildRaceCourse('open'),
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc' },
});

let stopSince = -1;
let captured = 0;
while (scenario.simTime() < 120 && captured < 3) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  if (Math.abs(c.state.speed) < 0.2 && r.simTime > 3) {
    if (stopSince < 0) stopSince = r.simTime;
    if (r.simTime - stopSince > 1.0 && c.plan && c.plan.length > 2) {
      captured++;
      stopSince = r.simTime + 5; // don't recapture immediately
      const cur = c.state;
      console.log(`\nWEDGE#${captured} t=${r.simTime.toFixed(1)} pos=(${cur.x.toFixed(2)},${cur.z.toFixed(2)}) heading=${cur.heading.toFixed(2)} v=${cur.speed.toFixed(2)} yawRate=${(cur.yawRate ?? 0).toFixed(2)}`);
      console.log(`diag: planAgeMs=${c.diagnostics.planAgeMs.toFixed(0)} lastReplanFound=${c.diagnostics.lastReplanFound} consecFailed=${c.diagnostics.consecutiveFailedReplans} lastReplanMs=${c.diagnostics.lastReplanMs.toFixed(0)}`);
      {
        const segs = splitAtGearCusps(c.plan);
        console.log(`segments: ${segs.map((s0) => `${s0.length}pts gear=${Math.sign(s0.reduce((a, p) => a + p.speed, 0)) || 0}`).join(' | ')}`);
        const p0 = c.plan[0]!;
        console.log(`plan[0] vs car: dx=${(p0.x - cur.x).toFixed(2)} dz=${(p0.z - cur.z).toFixed(2)} dh=${(p0.heading - cur.heading).toFixed(2)}`);
        const raw = (globalThis as Record<string, unknown>)[`__rawPlan_v3`] as CarKinematicState[] | undefined
          ?? (globalThis as Record<string, unknown>)[`__rawPlan_v2`] as CarKinematicState[] | undefined;
        if (raw) {
          console.log(`RAW planner path (${raw.length} nodes):`);
          for (let i = 0; i < Math.min(8, raw.length); i++) {
            const p = raw[i]!;
            console.log(`  raw[${i}] x=${p.x.toFixed(2)} z=${p.z.toFixed(2)} h=${p.heading.toFixed(2)} v=${p.speed.toFixed(2)}`);
          }
        }
      }
      const ext = extendPlanForTracking(c.plan, 50, 30);
      const geom = buildProgressGeometry(ext, {
        envelopeDecel: 8, envelopeLateralAccel: 12, usePlanSpeeds: false, ignoreTerminalSpeed: true,
      });
      let ni = 0; let nd = Infinity;
      for (let i = 0; i < Math.min(ext.length - 1, 20); i++) {
        const d = Math.hypot(ext[i]!.x - cur.x, ext[i]!.z - cur.z);
        if (d < nd) { nd = d; ni = i; }
      }
      console.log(`anchor idx=${ni} lat=${nd.toFixed(2)} planHeading=${ext[ni]!.heading.toFixed(2)} dh=${(((ext[ni]!.heading - cur.heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI).toFixed(2)} vAllow=${geom.vAllow[ni]!.toFixed(1)}`);
      for (let i = 0; i < Math.min(14, c.plan.length); i += 2) {
        const p = c.plan[i]!;
        console.log(`  plan[${i}] x=${p.x.toFixed(2)} z=${p.z.toFixed(2)} h=${p.heading.toFixed(2)} v=${p.speed.toFixed(2)}`);
      }
      const H = 30;
      const w = { wProgress: 6, wCorridor: 20, corridorHalfWidth: 2.5, wCenterline: 0.08, wOverspeed: 4, envelopeDecel: 8, wControlRate: 0.15, wSteerRate: 10, wHeadingAlign: 1.5 };
      const zeros = new Float64Array(H * 3);
      const anchorProj = { s: geom.cum[ni]!, idx: ni };
      const target = ext[Math.min(ni + 10, ext.length - 1)]!;
      function tryManeuver(name: string, steerFn: (i: number, s: CarKinematicState) => number, drive: number, brake = 0): void {
        let s: CarKinematicState = { ...cur };
        const traj: CarKinematicState[] = [];
        const ctrl = new Float64Array(H * 3);
        for (let i = 0; i < H; i++) {
          const st = steerFn(i, s);
          ctrl[i * 3] = st; ctrl[i * 3 + 1] = drive; ctrl[i * 3 + 2] = brake;
          for (let sub = 0; sub < 3; sub++) s = sim(s, [st, drive, brake], 0.05 / 3);
          traj.push(s);
        }
        const cost = scoreRolloutProgress(traj, geom, anchorProj, ctrl, zeros, w);
        const end = traj[H - 1]!;
        console.log(`  ${name.padEnd(22)} cost=${cost.toFixed(1)} end=(${end.x.toFixed(1)},${end.z.toFixed(1)}) v=${end.speed.toFixed(1)}`);
      }
      const steerToPlan = (_i: number, s: CarKinematicState): number => {
        const dx = target.x - s.x; const dz = target.z - s.z;
        const desired = Math.atan2(dz, dx);
        let e = desired - s.heading;
        while (e > Math.PI) e -= 2 * Math.PI;
        while (e < -Math.PI) e += 2 * Math.PI;
        return Math.max(-0.6, Math.min(0.6, 1.5 * e));
      };
      tryManeuver('hold-still', () => 0, 0, 400);
      tryManeuver('floor-straight', () => 0, 4000);
      tryManeuver('floor-steer-to-plan', steerToPlan, 4000);
      tryManeuver('half-steer-to-plan', steerToPlan, 2000);
      tryManeuver('quarter-steer-to-plan', steerToPlan, 1000);
    }
  } else {
    stopSince = -1;
  }
}
scenario.dispose();
