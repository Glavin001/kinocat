// Skill tests — executor tier (mpcTrack + pure-JS model, sub-second, no Rapier).
//
// These drive HAND-BUILT clean plans (a straight, a constant-radius arc, a
// straight-into-corner) through the real MPPI tracker + the parametric v2
// forward model, and assert the resulting closed-loop behaviour. Because the
// plan is hand-built and feasible, any failure here is an EXECUTOR problem, not
// a planner one — this is the planning-vs-execution separation.
//
// See docs/racing-skills-test-plan.md — skills K1 (floor a straight), K2 (hold
// a sweeper), K10 (brake late). The K10 contrast (timid decel=8 vs a realistic
// envelope-derived decel) is the quantified late-braking bug.

import { describe, it, expect } from 'vitest';
import { mpcTrack, createMPCTrackerState } from '../../src/execute/mpc-tracker';
import { parametricForwardV2, DEFAULT_LEARNED_PARAMS_V2 } from '../../src/agent/vehicle-model';
import { DEFAULT_LEARNABLE_CONFIG } from '../../src/agent/vehicle-config';
import type { CarKinematicState } from '../../src/agent/types';
import type { PlanPath } from '../../src/execute/types';

const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);

/** Base MPPI config mirroring the race MPC_CONFIG (progress mode). `envelopeDecel`
 *  is the knob the K10 fix changes; default 8 is the current (timid) scenario
 *  value. */
function raceCfg(over: Record<string, unknown> = {}) {
  return {
    maxSteer: 0.6,
    maxDriveForce: 4000,
    maxBrakeForce: 2000,
    samples: 64,
    horizonSteps: 30,
    stepDt: 0.05,
    substeps: 3,
    costMode: 'progress' as const,
    cruiseSpeed: 30,
    maxReverseSpeed: 6,
    wProgress: 6,
    wCorridor: 20,
    corridorHalfWidth: 2.5,
    wCenterline: 0.08,
    wOverspeed: 4,
    envelopeDecel: 8,
    envelopeLateralAccel: 12,
    wControlRate: 0.15,
    wSteerRate: 10,
    wHeadingAlign: 1.5,
    lambda: 3,
    noStopAtEnd: true,
    referenceExtension: 50,
    ...over,
  };
}

/** Drive a plan closed-loop from `start` for `secs`, sampling at 60 Hz but
 *  re-solving MPPI every 3 ticks (like the scenario's MPC_TICKS_PER_SOLVE).
 *  Returns the trajectory. */
function drive(plan: PlanPath, start: CarKinematicState, secs: number, cfg = raceCfg()): CarKinematicState[] {
  const state = createMPCTrackerState(30, 1234);
  const traj: CarKinematicState[] = [{ ...start }];
  let s = { ...start };
  const dt = 1 / 60;
  const ticks = Math.round(secs * 60);
  let hold: { steer: number; driveForce: number; brakeForce: number } | null = null;
  for (let i = 0; i < ticks; i++) {
    if (hold === null || i % 3 === 0) {
      const cmd = mpcTrack(s, plan, sim, state, cfg);
      hold = { steer: cmd.steer, driveForce: cmd.driveForce, brakeForce: cmd.brakeForce };
    }
    s = sim(s, [hold.steer, hold.driveForce, hold.brakeForce], dt);
    traj.push({ ...s });
  }
  return traj;
}

function straightPlan(length: number, speed: number, ds = 0.5): PlanPath {
  const n = Math.max(2, Math.round(length / ds));
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    const u = (i / (n - 1)) * length;
    out.push({ x: u, z: 0, heading: 0, speed, t: u / Math.max(speed, 1) });
  }
  return out;
}

function arcPlan(R: number, turn: number, speed: number, ds = 0.5): PlanPath {
  const arcLen = R * turn;
  const n = Math.max(2, Math.round(arcLen / ds));
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    const th = (i / (n - 1)) * turn;
    out.push({ x: R * Math.sin(th), z: R * (1 - Math.cos(th)), heading: th, speed, t: 0 });
  }
  let acc = 0;
  for (let i = 1; i < out.length; i++) {
    acc += Math.hypot(out[i]!.x - out[i - 1]!.x, out[i]!.z - out[i - 1]!.z);
    out[i]!.t = acc / Math.max(speed, 1);
  }
  return out;
}

function straightIntoCorner(runup: number, R: number, turn: number, speed: number, ds = 0.5): PlanPath {
  const out = straightPlan(runup, speed, ds);
  const arc = arcPlan(R, turn, speed, ds);
  const last = out[out.length - 1]!;
  for (const p of arc.slice(1)) out.push({ ...p, x: p.x + last.x, z: p.z + last.z });
  let acc = 0;
  for (let i = 1; i < out.length; i++) {
    acc += Math.hypot(out[i]!.x - out[i - 1]!.x, out[i]!.z - out[i - 1]!.z);
    out[i]!.t = acc / Math.max(speed, 1);
  }
  return out;
}

const peak = (t: CarKinematicState[]) => Math.max(...t.map((s) => Math.abs(s.speed)));

describe('skill K1 — the executor floors a straight drive-through', () => {
  it('accelerates from rest to near cruise on an open straight', () => {
    // parametric model accelerates ~5.8 m/s^2 => needs ~5 s to reach ~28.
    const traj = drive(straightPlan(400, 30), { x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 9);
    expect(peak(traj)).toBeGreaterThan(25);
  });
});

describe('skill K2 — the executor holds a constant-radius sweeper', () => {
  it('sustains a large fraction of sqrt(aLat*R) instead of crawling', () => {
    const R = 25;
    // Feasible corner speed at aLat=12 is sqrt(12*25)=17.3; at the measured
    // 13.7 it is 18.5. The car should SUSTAIN well above a crawl.
    const traj = drive(arcPlan(R, Math.PI, 16), { x: 0, z: 0, heading: 0, speed: 16, t: 0 }, 6);
    // Mean speed over the second half (steady state), should hold the corner.
    const half = traj.slice(Math.floor(traj.length / 2));
    const mean = half.reduce((a, s) => a + Math.abs(s.speed), 0) / half.length;
    expect(mean).toBeGreaterThan(12);
  });
});

describe('skill K10 — the executor brakes LATE, and a realistic decel brakes later', () => {
  // Enter a straight-into-corner at 28 m/s; measure how far before the corner
  // the car has already dropped below 24 m/s (started braking). A realistic
  // deceleration budget must brake substantially LATER (closer to the corner)
  // than the timid default 8.
  function brakeLead(decel: number): number {
    const runup = 140;
    const plan = straightIntoCorner(runup, 8, Math.PI / 2, 28);
    const traj = drive(plan, { x: 0, z: 0, heading: 0, speed: 28, t: 0 }, 7, raceCfg({ envelopeDecel: decel }));
    // First trajectory x where speed < 24 (braking has begun).
    const braked = traj.find((s) => Math.abs(s.speed) < 24 && s.x > 5);
    const brakeX = braked ? braked.x : runup;
    return runup - brakeX; // metres of straight still ahead when braking began
  }

  it('a realistic decel budget brakes at least 12 m closer to the corner than decel=8', () => {
    const leadTimid = brakeLead(8);
    const leadReal = brakeLead(22);
    // Larger lead = braked earlier (further from corner). Timid must brake
    // earlier (bigger lead); realistic later (smaller lead).
    expect(leadTimid - leadReal).toBeGreaterThan(12);
  });
});
