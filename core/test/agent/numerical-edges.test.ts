// Numerical / adversarial edge cases. These are the inputs that crash naive
// vehicle math — zero speed, absurd speed, start coincident with goal, an agent
// exactly on your heading axis, two obstacles at the identical position. The
// invariant under test is uniform: stay FINITE, do not throw, and fail safe.
// (Empty paths are intentionally out of scope — purePursuit's contract requires
// at least one path point; a zero-length displacement path is the realistic
// "already at the goal" degenerate case and IS covered.)

import { describe, it, expect } from 'vitest';
import { plan } from '../../src/planner/ighastar';
import { VehicleEnvironment } from '../../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { linearObstacle } from '../../src/predict/factories';
import { purePursuit } from '../../src/execute/pure-pursuit';
import {
  kinematicForwardSim,
  learnedForwardSim,
  DEFAULT_LEARNED_PARAMS,
} from '../../src/agent/vehicle';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import {
  SWEEP_AGENT as agent,
  SWEEP_AGENT_RADIUS as AGENT_R,
  buildSweepLib,
  rect,
} from '../fixtures/vehicle-sweep';

const lib = buildSweepLib();

function isFiniteState(s: CarKinematicState): boolean {
  return (
    Number.isFinite(s.x) &&
    Number.isFinite(s.z) &&
    Number.isFinite(s.heading) &&
    Number.isFinite(s.speed) &&
    Number.isFinite(s.t)
  );
}

describe('numerical edges — forward dynamics stay finite', () => {
  const kin = kinematicForwardSim(agent);
  const learned = learnedForwardSim(DEFAULT_LEARNED_PARAMS, agent);
  const dt = 1 / 60;

  it('zero speed with any control produces a finite, motionless-ish step', () => {
    const s0: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (const controls of [[0, 0], [10, 0], [-10, 0], [0, 5], [1e9, 0]]) {
      expect(isFiniteState(kin(s0, controls, dt)), `kin ${controls}`).toBe(true);
      expect(isFiniteState(learned(s0, controls, dt)), `learned ${controls}`).toBe(true);
    }
  });

  it('absurd target speed is clamped to maxSpeed (no overflow / NaN)', () => {
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let i = 0; i < 300; i++) {
      s = kin(s, [0, 1e12], dt);
      expect(isFiniteState(s)).toBe(true);
      expect(Math.abs(s.speed)).toBeLessThanOrEqual(agent.maxSpeed + 1e-6);
    }
  });

  it('very high speed + max curvature over a long roll never NaNs', () => {
    const kMax = 1 / agent.minTurnRadius;
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: agent.maxSpeed, t: 0 };
    for (let i = 0; i < 1000; i++) {
      s = learned(s, [i % 2 === 0 ? kMax : -kMax, 1e9], dt);
      expect(isFiniteState(s), `step ${i}`).toBe(true);
    }
  });
});

describe('numerical edges — degenerate planning queries', () => {
  const world = new InMemoryNavWorld([rect(1, 0, -10, 30, 10)]);
  const env = new VehicleEnvironment(world, agent, lib, {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
  });

  it('start === goal returns immediately without throwing', () => {
    const at: CarKinematicState = { x: 10, z: 0, heading: 0, speed: 0, t: 0 };
    let r!: ReturnType<typeof plan<CarKinematicState>>;
    expect(() => {
      r = plan({ start: { ...at }, goal: { ...at }, environment: env, options: { maxExpansions: 5000 } }, Infinity);
    }).not.toThrow();
    expect(r.found).toBe(true);
    expect(Number.isFinite(r.cost)).toBe(true);
    for (const s of r.path) expect(isFiniteState(s)).toBe(true);
  });
});

describe('numerical edges — degenerate tracker inputs', () => {
  const cfg: PurePursuitConfig = {
    lookaheadMin: 2,
    lookaheadGain: 0.3,
    lookaheadMax: 6,
    maxLateralAccel: 4,
    maxAccel: 8,
    maxDecel: 8,
    cruiseSpeed: 6,
    goalTolerance: 0.5,
    minTurnRadius: 2,
  };

  it('single-point path: finite command, reports atGoal when on it', () => {
    const path: PlanPath = [{ x: 5, z: 0, heading: 0, speed: 0, t: 0 }];
    const cmd = purePursuit({ x: 5, z: 0, heading: 0, speed: 0, t: 0 }, path, cfg);
    expect(Number.isFinite(cmd.steering) && Number.isFinite(cmd.targetSpeed)).toBe(true);
    expect(cmd.atGoal).toBe(true);
  });

  it('zero-length path (all points coincident) yields a finite, safe command', () => {
    const path: PlanPath = [
      { x: 3, z: 3, heading: 0, speed: 0, t: 0 },
      { x: 3, z: 3, heading: 0, speed: 0, t: 0.1 },
      { x: 3, z: 3, heading: 0, speed: 0, t: 0.2 },
    ];
    const cmd = purePursuit({ x: 3, z: 3, heading: 1.0, speed: 0, t: 0 }, path, cfg);
    expect(Number.isFinite(cmd.steering)).toBe(true);
    expect(Number.isFinite(cmd.targetSpeed)).toBe(true);
    expect(Number.isFinite(cmd.lookahead.x) && Number.isFinite(cmd.lookahead.z)).toBe(true);
  });
});

describe('numerical edges — adversarial obstacle geometry', () => {
  const world = new InMemoryNavWorld([rect(1, 0, -14, 32, 14)]);
  const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
  const goal: CarKinematicState = { x: 28, z: 0, heading: 0, speed: 0, t: 0 };

  function solveWith(obstacles: ReturnType<typeof linearObstacle>[]) {
    const base = new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity });
    const env = new TimeAwareEnvironment(base, { obstacles, agentRadius: AGENT_R });
    return plan({ start, goal, environment: env, options: { maxExpansions: 300000 } }, Infinity);
  }

  it('two obstacles at the identical position do not crash the planner', () => {
    const dup = [linearObstacle(15, 0, 0, 0, 2.5, 0, 1e6), linearObstacle(15, 0, 0, 0, 2.5, 0, 1e6)];
    let r!: ReturnType<typeof solveWith>;
    expect(() => {
      r = solveWith(dup);
    }).not.toThrow();
    expect(typeof r.found).toBe('boolean');
    if (r.found) for (const s of r.path) expect(isFiniteState(s)).toBe(true);
  });

  it('an obstacle exactly on the heading axis stays finite and never collides', () => {
    // Obstacle approaches head-on along z=0 (the ego's heading axis) — the
    // singular case for any bearing/atan2 math.
    const obs = linearObstacle(26, 0, -4, 0, 1.8, 0, 60);
    const r = solveWith([obs]);
    expect(typeof r.found).toBe('boolean');
    if (r.found) {
      for (const s of r.path) {
        expect(isFiniteState(s)).toBe(true);
        const p = obs.predict(s.t);
        if (p) {
          expect(Math.hypot(s.x - p.x, s.z - p.z)).toBeGreaterThan(obs.radius + AGENT_R - 1e-6);
        }
      }
    }
  });
});
