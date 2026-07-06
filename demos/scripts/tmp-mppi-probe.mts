// TEMP — probe progress-mode MPPI in isolation. Not committed.
import { mpcTrack, createMPCTrackerState } from 'kinocat/execute';
import {
  parametricForwardV2,
  DEFAULT_LEARNED_PARAMS_V2,
} from 'kinocat/agent';
import { DEFAULT_LEARNABLE_CONFIG } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';

const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);

// Straight 60 m plan, gate-style speeds (0 at terminal), 0.4 m spacing.
const plan: CarKinematicState[] = [];
for (let i = 0; i <= 150; i++) {
  plan.push({ x: i * 0.4, z: 0, heading: 0, speed: 0, t: i * 0.1 });
}

const cfg = {
  horizonSteps: 30,
  stepDt: 0.05,
  samples: 64,
  maxSteer: 0.6,
  maxDriveForce: 4000,
  maxBrakeForce: 2000,
  allowReverse: false,
  lambda: 0.5,
  steerStd: 0.10,
  driveStd: 2000,
  brakeStd: 200,
  wControlRate: 0.15,
  wSteerRate: 25,
  wTerminalPosition: 0,
  wTerminalSpeed: 0,
  cruiseSpeed: 30,
  costMode: 'progress' as const,
  noStopAtEnd: true,
  referenceExtension: 50,
  substeps: 3,
  wProgress: 6,
  wCorridor: 20,
  corridorHalfWidth: 2.5,
  wCenterline: 0.08,
  wOverspeed: 4,
  envelopeDecel: 8,
  envelopeLateralAccel: 12,
  usePlanSpeeds: false,
};

const st = createMPCTrackerState(30, 0x1337);
let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
for (let i = 0; i < 60; i++) {
  const t0 = performance.now();
  const cmd = mpcTrack(s, plan, sim, st, cfg);
  const ms = performance.now() - t0;
  if (i % 5 === 0) {
    console.log(
      `t=${(i * 0.05).toFixed(2)} x=${s.x.toFixed(2)} v=${s.speed.toFixed(2)} ` +
      `steer=${cmd.steer.toFixed(3)} drive=${cmd.driveForce.toFixed(0)} brake=${cmd.brakeForce.toFixed(0)} ` +
      `bestCost=${cmd.bestCost.toFixed(2)} solveMs=${ms.toFixed(2)}`,
    );
  }
  for (let sub = 0; sub < 3; sub++) s = sim(s, [cmd.steer, cmd.driveForce, cmd.brakeForce], 0.05 / 3);
}
