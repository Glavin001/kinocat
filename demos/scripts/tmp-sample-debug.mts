// TEMP — replicate mpcTrack sampling loop to debug. Not committed.
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
const H = 30, K = 16;
const w = { wProgress: 6, wCorridor: 20, corridorHalfWidth: 2.5, wCenterline: 0.08, wOverspeed: 4, envelopeDecel: 8, wControlRate: 0.15, wSteerRate: 25 };

let rng = 0x1337 >>> 0;
function lcg(): number { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng / 0x100000000; }
function gauss(): number { const u1 = Math.max(lcg(), 1e-9); const u2 = lcg(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

const prev = new Float64Array(H * 3);
const current: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
for (let k = 0; k < K; k++) {
  const work = new Float64Array(H * 3);
  for (let i = 0; i < H; i++) {
    const nf = k === 0 ? 0 : 1;
    work[i*3] = Math.max(-0.6, Math.min(0.6, 0 + 0.10 * gauss() * nf));
    work[i*3+1] = Math.max(0, Math.min(4000, 0 + 2000 * gauss() * nf));
    work[i*3+2] = Math.max(0, Math.min(2000, 0 + 200 * gauss() * nf));
  }
  let s = { ...current };
  const traj: CarKinematicState[] = [];
  for (let i = 0; i < H; i++) {
    const u = [work[i*3]!, work[i*3+1]!, work[i*3+2]!];
    for (let sub = 0; sub < 3; sub++) s = sim(s, u, 0.05/3);
    traj.push(s);
  }
  const cost = scoreRolloutProgress(traj, geom, { s: 0, idx: 0 }, work, prev, w);
  console.log(`k=${k} cost=${cost.toFixed(2)} end=(${s.x.toFixed(2)},${s.z.toFixed(2)}) v=${s.speed.toFixed(2)} meanDrive=${(Array.from({length:H},(_,i)=>work[i*3+1]!).reduce((a,b)=>a+b,0)/H).toFixed(0)} meanBrake=${(Array.from({length:H},(_,i)=>work[i*3+2]!).reduce((a,b)=>a+b,0)/H).toFixed(0)}`);
}
