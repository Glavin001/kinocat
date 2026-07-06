// TEMP — debug progress cost. Not committed.
import { buildProgressGeometry, scoreRolloutProgress, extendPlanForTracking } from 'kinocat/execute';
import { parametricForwardV2, DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
const plan: CarKinematicState[] = [];
for (let i = 0; i <= 150; i++) plan.push({ x: i * 0.4, z: 0, heading: 0, speed: 0, t: i * 0.1 });
const ext = extendPlanForTracking(plan, 50, 30);
const geom = buildProgressGeometry(ext, {
  envelopeDecel: 8, envelopeLateralAccel: 12, usePlanSpeeds: false, ignoreTerminalSpeed: true,
});
console.log('plan len', ext.length, 'cum end', geom.cum[geom.cum.length-1], 'vAllow[0..5]', geom.vAllow.slice(0,5));

const H = 30;
const w = { wProgress: 6, wCorridor: 20, corridorHalfWidth: 2.5, wCenterline: 0.08, wOverspeed: 4, envelopeDecel: 8, wControlRate: 0.15, wSteerRate: 25 };
function rollout(drive: number): CarKinematicState[] {
  let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
  const out: CarKinematicState[] = [];
  for (let i = 0; i < H; i++) {
    for (let sub = 0; sub < 3; sub++) s = sim(s, [0, drive, 0], 0.05 / 3);
    out.push(s);
  }
  return out;
}
const zeros = new Float64Array(H * 3);
const full = rollout(4000);
console.log('full-throttle endpoint', full[H-1]);
const ctrlFull = new Float64Array(H * 3);
for (let i = 0; i < H; i++) ctrlFull[i*3+1] = 4000;
console.log('cost(full)', scoreRolloutProgress(full, geom, { s: 0, idx: 0 }, ctrlFull, zeros, w));
console.log('cost(zero)', scoreRolloutProgress(rollout(0), geom, { s: 0, idx: 0 }, zeros, zeros, w));
