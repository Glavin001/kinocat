import { describe, it, expect } from 'vitest';
import { purePursuit } from '../../src/execute/pure-pursuit';
import type { PurePursuitConfig, PlanPath } from '../../src/execute/types';
import type { CarKinematicState } from '../../src/agent/types';
import { wrapAngle } from '../../src/internal/math';

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

/** Closed-loop unicycle sim (perfect speed tracking) for tracking tests. */
function simulate(start: CarKinematicState, path: PlanPath, steps: number, dt = 0.05) {
  let s = { ...start };
  let maxCross = 0;
  for (let k = 0; k < steps; k++) {
    const cmd = purePursuit(s, path, cfg);
    if (cmd.atGoal) break;
    const speed = cmd.targetSpeed;
    const heading = wrapAngle(s.heading + speed * cmd.steering * dt);
    s = {
      x: s.x + speed * Math.cos(s.heading) * dt,
      z: s.z + speed * Math.sin(s.heading) * dt,
      heading,
      speed,
      t: s.t + dt,
    };
    maxCross = Math.max(maxCross, Math.abs(s.z)); // straight-path cross-track
  }
  return { final: s, maxCross };
}

describe('purePursuit', () => {
  it('converges onto and tracks a straight path from a lateral offset', () => {
    const path: PlanPath = [];
    for (let x = 0; x <= 24; x += 2) {
      path.push({ x, z: 0, heading: 0, speed: 6, t: x / 6 });
    }
    const start: CarKinematicState = { x: 0, z: 2, heading: 0, speed: 0, t: 0 };
    const { final } = simulate(start, path, 600);
    expect(Math.hypot(final.x - 24, final.z - 0)).toBeLessThan(0.6);
    expect(Math.abs(final.z)).toBeLessThan(0.2); // converged onto the line
  });

  it('reduces speed on tight curvature (v ≈ sqrt(aLat/|κ|))', () => {
    const longCfg = { ...cfg, cruiseSpeed: 50, maxDecel: 50, minTurnRadius: undefined };
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 0.1, z: 50, heading: Math.PI / 2, speed: 6, t: 10 },
    ];
    const cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cmd = purePursuit(cur, path, longCfg);
    const kappa = Math.abs(cmd.steering);
    expect(kappa).toBeGreaterThan(0.1);
    const expected = Math.sqrt(longCfg.maxLateralAccel / kappa);
    expect(Math.abs(cmd.targetSpeed)).toBeCloseTo(expected, 6);
    expect(Math.abs(cmd.targetSpeed)).toBeLessThan(longCfg.cruiseSpeed);
  });

  it('tracks a reverse path (negative planned speed)', () => {
    const path: PlanPath = [];
    for (let x = 0; x >= -12; x -= 2) {
      path.push({ x, z: 0, heading: 0, speed: -4, t: -x / 4 });
    }
    const start: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const cmd0 = purePursuit(start, path, cfg);
    expect(cmd0.targetSpeed).toBeLessThan(0);
    const { final } = simulate(start, path, 600);
    expect(Math.hypot(final.x - -12, final.z - 0)).toBeLessThan(0.8);
  });

  it('is a pure function (identical inputs → identical output)', () => {
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 10, z: 1, heading: 0, speed: 6, t: 1.7 },
    ];
    const cur: CarKinematicState = { x: 1, z: 0.2, heading: 0.1, speed: 3, t: 0.2 };
    expect(purePursuit(cur, path, cfg)).toEqual(purePursuit(cur, path, cfg));
  });

  it('brakes at the goal', () => {
    const path: PlanPath = [
      { x: 0, z: 0, heading: 0, speed: 4, t: 0 },
      { x: 5, z: 0, heading: 0, speed: 0, t: 1.25 },
    ];
    const atGoal: CarKinematicState = { x: 4.9, z: 0, heading: 0, speed: 1, t: 1.2 };
    const cmd = purePursuit(atGoal, path, cfg);
    expect(cmd.atGoal).toBe(true);
    expect(cmd.brake).toBe(1);
    expect(cmd.targetSpeed).toBe(0);
  });
});
