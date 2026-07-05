// Unit tests for the telemetry + diagnostics monitor. Pure, no Rapier — feeds
// hand-built sample streams and checks each observable-behaviour metric maps to
// the phenomenon it is meant to catch. Always runs (no skipIf).

import { describe, it, expect } from 'vitest';
import {
  createSimMonitor,
  type MonitorSample,
  type Pt,
} from '../app/lib/sim-monitor';
import type { MonitorConfig } from '../app/lib/sim-monitor';

const DT = 1 / 60;

const FOOTPRINT: Pt[] = [
  [1.6, 0.9],
  [-1.6, 0.9],
  [-1.6, -0.9],
  [1.6, -0.9],
];

function mk(partial: {
  x?: number;
  z?: number;
  heading?: number;
  speed?: number;
  steer?: number;
  totalReplans?: number;
  successfulReplans?: number;
  consecutiveFailedReplans?: number;
  plan?: ReadonlyArray<{ x: number; z: number; heading: number }> | null;
  loopIndex?: number;
}): MonitorSample {
  return {
    state: {
      x: partial.x ?? 0,
      z: partial.z ?? 0,
      heading: partial.heading ?? 0,
      speed: partial.speed ?? 0,
    },
    metrics: {
      liveControls: {
        steer: partial.steer ?? 0,
        throttle: 0,
        brake: 0,
        targetSpeed: 0,
      },
    },
    diagnostics: {
      totalReplans: partial.totalReplans ?? 0,
      successfulReplans: partial.successfulReplans ?? 0,
      consecutiveFailedReplans: partial.consecutiveFailedReplans ?? 0,
    },
    plan: partial.plan ?? null,
    loopIndex: partial.loopIndex ?? 0,
  };
}

const baseCfg: MonitorConfig = { footprint: FOOTPRINT, obstacles: [], dt: DT };

describe('sim-monitor: safety invariants', () => {
  it('reports ~0 accel/jerk and 0 reversals for constant-speed straight motion', () => {
    const m = createSimMonitor(baseCfg);
    for (let i = 0; i < 60; i++) m.sample(mk({ x: i * 10 * DT, speed: 10 }));
    const r = m.summary();
    expect(r.maxAccel).toBeCloseTo(0, 6);
    expect(r.maxJerk).toBeCloseTo(0, 6);
    expect(r.steerReversals).toBe(0);
    expect(r.peakSpeed).toBeCloseTo(10, 9);
  });

  it('recovers a constant accel from a linear speed ramp, with ~0 jerk', () => {
    const a = 3; // m/s^2
    const m = createSimMonitor(baseCfg);
    let x = 0;
    let v = 0;
    for (let i = 0; i < 60; i++) {
      m.sample(mk({ x, speed: v }));
      x += v * DT;
      v += a * DT;
    }
    const r = m.summary();
    expect(r.maxAccel).toBeCloseTo(a, 6);
    expect(r.maxJerk).toBeCloseTo(0, 6);
  });

  it('flags a collision and zero clearance when the footprint overlaps an obstacle', () => {
    const obstacle: Array<[number, number]> = [
      [9, -1],
      [11, -1],
      [11, 1],
      [9, 1],
    ];
    const m = createSimMonitor({ ...baseCfg, obstacles: [obstacle] });
    // Drive straight through x=10 where the obstacle sits.
    for (let i = 0; i < 30; i++) m.sample(mk({ x: i * 0.5, speed: 5 }));
    const r = m.summary();
    expect(r.collided).toBe(true);
    expect(r.minClearance).toBe(0);
  });

  it('measures a positive clearance when the footprint passes at a known gap', () => {
    // Footprint half-width 0.9; obstacle lower edge at z=3 ⇒ gap ≈ 3-0.9 = 2.1.
    const obstacle: Array<[number, number]> = [
      [-2, 3],
      [2, 3],
      [2, 5],
      [-2, 5],
    ];
    const m = createSimMonitor({ ...baseCfg, obstacles: [obstacle] });
    m.sample(mk({ x: 0, z: 0, heading: 0, speed: 1 }));
    const r = m.summary();
    expect(r.minClearance).toBeCloseTo(2.1, 6);
    expect(r.collided).toBe(false);
  });
});

describe('sim-monitor: teleport detection', () => {
  it('counts a teleport and excludes it from accel and away-progress', () => {
    const m = createSimMonitor({ ...baseCfg, goal: { x: 0, z: 0, heading: 0 } });
    // Approach the goal smoothly, then a single-tick 30 m jump (stall rescue
    // back onto the goal), then sit still.
    m.sample(mk({ x: 5, speed: 2 }));
    m.sample(mk({ x: 4.9, speed: 2 }));
    m.sample(mk({ x: 35, speed: 2 })); // teleport: +30 m in one tick
    m.sample(mk({ x: 0, z: 0, speed: 0 })); // teleport back to goal
    const r = m.summary();
    expect(r.teleports).toBe(2);
    // The 30 m/tick jump would be ~1800 m/s^2 if counted — it must not be.
    expect(r.maxAccel).toBeLessThan(50);
  });
});

describe('sim-monitor: jitter', () => {
  it('counts steering reversals and nonzero steer-rate RMS for a zig-zag trace', () => {
    const m = createSimMonitor(baseCfg);
    // Alternate steer sign every tick ⇒ many reversals.
    for (let i = 0; i < 40; i++) {
      m.sample(mk({ x: i * 5 * DT, speed: 5, steer: i % 2 === 0 ? 0.3 : -0.3 }));
    }
    const r = m.summary();
    expect(r.steerReversals).toBeGreaterThan(10);
    expect(r.steerRateRms).toBeGreaterThan(0);
  });
});

describe('sim-monitor: direction / wrong-way', () => {
  it('reports negative net progress and away-ticks when driving away from goal', () => {
    const m = createSimMonitor({ ...baseCfg, goal: { x: 0, z: 0, heading: 0 } });
    // Start near the goal and drive away along +x.
    for (let i = 0; i < 30; i++) m.sample(mk({ x: 1 + i * 0.2, speed: 1, heading: 0 }));
    const r = m.summary();
    expect(r.netProgress).toBeLessThan(0);
    expect(r.movedAwayFromGoalTicks).toBeGreaterThan(20);
    expect(r.awayHeadingTicks).toBeGreaterThan(20);
  });

  it('reports positive net progress and parkedOk when arriving at the goal pose', () => {
    const m = createSimMonitor({ ...baseCfg, goal: { x: 10, z: 0, heading: 0 } });
    // Drive from x=0 to x=10, decelerating to a stop at the goal.
    for (let i = 0; i <= 20; i++) {
      const x = (i / 20) * 10;
      const speed = i === 20 ? 0 : 1;
      m.sample(mk({ x, z: 0, heading: 0, speed }));
    }
    const r = m.summary();
    expect(r.netProgress).toBeCloseTo(10, 1);
    expect(r.parkedOk).toBe(true);
  });

  it('parkedOk is false when stopped at the right spot but wrong heading', () => {
    const m = createSimMonitor({ ...baseCfg, goal: { x: 0, z: 0, heading: 0 } });
    m.sample(mk({ x: 0, z: 0, heading: 0, speed: 1 }));
    m.sample(mk({ x: 0, z: 0, heading: Math.PI / 2, speed: 0 })); // 90° off
    const r = m.summary();
    expect(r.terminalPosError).toBeCloseTo(0, 6);
    expect(r.parkedOk).toBe(false);
  });
});

describe('sim-monitor: replan health', () => {
  it('counts plan updates and direction flips for an oscillating planner', () => {
    const m = createSimMonitor(baseCfg);
    // New plan object each step, alternating aim left/right of +x heading.
    for (let i = 0; i < 10; i++) {
      const dz = i % 2 === 0 ? 3 : -3; // aim +z then -z
      const plan = [
        { x: 0, z: 0, heading: 0 },
        { x: 4, z: dz, heading: 0 },
      ];
      m.sample(mk({ x: 0, z: 0, heading: 0, speed: 2, plan, totalReplans: i + 1, successfulReplans: i + 1 }));
    }
    const r = m.summary();
    expect(r.planUpdates).toBe(10);
    expect(r.planDirectionFlips).toBeGreaterThan(5);
    expect(r.replansPerSec).toBeGreaterThan(0);
  });

  it('tracks max consecutive failed replans and failed ratio', () => {
    const m = createSimMonitor(baseCfg);
    m.sample(mk({ totalReplans: 1, successfulReplans: 1, consecutiveFailedReplans: 0 }));
    m.sample(mk({ totalReplans: 2, successfulReplans: 1, consecutiveFailedReplans: 1 }));
    m.sample(mk({ totalReplans: 4, successfulReplans: 1, consecutiveFailedReplans: 3 }));
    const r = m.summary();
    expect(r.consecutiveFailedReplansMax).toBe(3);
    expect(r.failedReplanRatio).toBeCloseTo(0.75, 6);
  });
});
