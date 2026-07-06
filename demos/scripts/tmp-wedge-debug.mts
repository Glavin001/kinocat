// TEMP — capture a wedged MPPI state and dissect the cost landscape. Not committed.
import { writeFileSync } from 'node:fs';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import {
  extendPlanForTracking, buildProgressGeometry, scoreRolloutProgress,
} from 'kinocat/execute';
import { parametricForwardV2, KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const scenario = await createRaceScenario({
  entries: [kinematicEntry('kin')],
  targetLaps: 2,
  syncHold: false,
  course: buildRaceCourse('open'),
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc' },
});

let stopSince = -1;
let captured = false;
while (scenario.simTime() < 120 && !captured) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  if (Math.abs(c.state.speed) < 0.2 && r.simTime > 3) {
    if (stopSince < 0) stopSince = r.simTime;
    if (r.simTime - stopSince > 0.8 && c.plan && c.plan.length > 2) {
      captured = true;
      const cur = c.state;
      console.log(`WEDGE at t=${r.simTime.toFixed(1)} pos=(${cur.x.toFixed(2)},${cur.z.toFixed(2)}) heading=${cur.heading.toFixed(2)} v=${cur.speed.toFixed(2)}`);
      writeFileSync('/tmp/wedge.json', JSON.stringify({ cur, plan: c.plan }, null, 1));
      // Dissect: cost of candidate maneuvers under the same geometry.
      const ext = extendPlanForTracking(c.plan, 50, 30);
      const geom = buildProgressGeometry(ext, {
        envelopeDecel: 8, envelopeLateralAccel: 12, usePlanSpeeds: false, ignoreTerminalSpeed: true,
      });
      // anchor
      let ni = 0; let nd = Infinity;
      for (let i = 0; i < ext.length - 1; i++) {
        const d = Math.hypot(ext[i]!.x - cur.x, ext[i]!.z - cur.z);
        if (d < nd) { nd = d; ni = i; }
      }
      console.log(`anchor idx=${ni} of ${ext.length} lat=${nd.toFixed(2)} vAllow[ni..ni+20]=${geom.vAllow.slice(ni, ni + 20).map((v) => v.toFixed(1)).join(',')}`);
      // plan heading vs car heading at anchor
      console.log(`planHeading=${ext[ni]!.heading.toFixed(2)} carHeading=${cur.heading.toFixed(2)}`);
      const sim = parametricForwardV2(KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG);
      const H = 30;
      const w = { wProgress: 6, wCorridor: 20, corridorHalfWidth: 2.5, wCenterline: 0.08, wOverspeed: 4, envelopeDecel: 8, wControlRate: 0.15, wSteerRate: 25 };
      const zeros = new Float64Array(H * 3);
      const anchorProj = { s: geom.cum[ni]!, idx: ni };
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
        console.log(`${name.padEnd(24)} cost=${cost.toFixed(1)} end=(${end.x.toFixed(1)},${end.z.toFixed(1)}) v=${end.speed.toFixed(1)}`);
      }
      // steer toward plan tangent a few samples ahead
      const target = ext[Math.min(ni + 8, ext.length - 1)]!;
      tryManeuver('hold-still', () => 0, 0, 400);
      tryManeuver('floor-straight', () => 0, 4000);
      tryManeuver('floor-half', () => 0, 2000);
      tryManeuver('floor-steer-to-plan', (_i, s) => {
        const dx = target.x - s.x; const dz = target.z - s.z;
        const desired = Math.atan2(dz, dx);
        let e = desired - s.heading;
        while (e > Math.PI) e -= 2 * Math.PI;
        while (e < -Math.PI) e += 2 * Math.PI;
        return Math.max(-0.6, Math.min(0.6, 1.5 * e));
      }, 4000);
      tryManeuver('half-steer-to-plan', (_i, s) => {
        const dx = target.x - s.x; const dz = target.z - s.z;
        const desired = Math.atan2(dz, dx);
        let e = desired - s.heading;
        while (e > Math.PI) e -= 2 * Math.PI;
        while (e < -Math.PI) e += 2 * Math.PI;
        return Math.max(-0.6, Math.min(0.6, 1.5 * e));
      }, 2000);
    }
  } else {
    stopSince = -1;
  }
}
scenario.dispose();
