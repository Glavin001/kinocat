import { describe, it, expect } from 'vitest';
import {
  buildDogLegCorridor,
  assessPassability,
  sweptClearance,
  runGauntlet,
  type RefController,
} from '../../src/eval';
import type { DynamicLimits } from '../../src/eval';
import { purePursuit } from '../../src/execute/pure-pursuit';
import type { PurePursuitConfig } from '../../src/execute/types';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { CarKinematicState, Pt } from '../../src/agent/types';

const agent = defaultVehicleAgent();
const footprint = agent.footprint as Pt[];
const limits: DynamicLimits = {
  frictionLimit: 4,
  minTurnRadius: agent.minTurnRadius,
  maxAccel: 6.5,
  maxDecel: 8,
};

const ppConfig: PurePursuitConfig = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 5,
  maxLateralAccel: 4,
  maxAccel: 6.5,
  maxDecel: 8,
  cruiseSpeed: 8,
  goalTolerance: 1.0,
  minTurnRadius: agent.minTurnRadius,
};
const controller: RefController = (state, path) => {
  const cmd = purePursuit(state, path as CarKinematicState[], ppConfig);
  return { controls: [cmd.steering, cmd.targetSpeed], steer: cmd.steering, atGoal: cmd.atGoal };
};

describe('buildDogLegCorridor', () => {
  it('builds an offset corridor (dog-leg) with walls sized to footprint + margin', () => {
    const w = buildDogLegCorridor({ footprint, margin: 1.0, offset: 4, speed: 6 });
    // Footprint half-width 0.9 ⇒ width = 1.8 + 1.0 = 2.8.
    expect(w.width).toBeCloseTo(2.8, 6);
    // Start in one lot at z≈0, goal in the offset lot at z≈4.
    expect(Math.abs(w.start.z)).toBeLessThan(0.01);
    expect(w.goal.z).toBeCloseTo(4, 1);
    expect(w.leftWall.length).toBeGreaterThan(2);
    expect(w.rightWall.length).toBeGreaterThan(2);
  });
});

describe('passability oracle', () => {
  it('a generous corridor is provably passable', () => {
    const w = buildDogLegCorridor({ footprint, margin: 1.5, offset: 4, speed: 6 });
    const p = assessPassability(w, footprint, limits);
    expect(p.passable).toBe(true);
    expect(p.oracleMinClearance).toBeGreaterThan(0);
    expect(p.feasible).toBe(true);
  });

  it('a negative-margin corridor is NOT passable (oracle clips)', () => {
    const w = buildDogLegCorridor({ footprint, margin: -0.2, offset: 4, speed: 6 });
    const p = assessPassability(w, footprint, limits);
    expect(p.passable).toBe(false);
    // The ideal swept footprint already touches/overlaps the walls.
    expect(p.oracleMinClearance).toBe(0);
  });

  it('flags dynamic infeasibility when the dog-leg is too sharp for the radius', () => {
    // Big offset over a short corridor ⇒ tight S-curve below min turn radius.
    const w = buildDogLegCorridor({ footprint, margin: 2, offset: 8, corridorLength: 6, speed: 6 });
    const p = assessPassability(w, footprint, limits);
    expect(p.feasible).toBe(false);
  });
});

describe('runGauntlet', () => {
  it('threads a generous corridor: reaches goal, no collision, high gated score', () => {
    const w = buildDogLegCorridor({ footprint, margin: 1.5, offset: 4, speed: 5 });
    const r = runGauntlet(w, footprint, controller, kinematicForwardSim(agent), { dt: 1 / 60, limits });
    expect(r.passability.passable).toBe(true);
    expect(r.reachedGoal).toBe(true);
    expect(r.collided).toBe(false);
    expect(r.gatedScore).toBeGreaterThan(0);
    expect(r.corridorCrossTrackMax).toBeLessThan(w.margin / 2 + 0.5);
  });

  it('gated score is 0 when the controller clips the wall', () => {
    // The oracle is passable here, but a tight margin + the controller s curve
    // deviation should clip → gate fires.
    const w = buildDogLegCorridor({ footprint, margin: 0.05, offset: 4, speed: 8 });
    const r = runGauntlet(w, footprint, controller, kinematicForwardSim(agent), { dt: 1 / 60, limits });
    if (r.collided) {
      expect(r.gatedScore).toBe(0);
    }
  });

  it('is deterministic across reruns', () => {
    const make = () => buildDogLegCorridor({ footprint, margin: 1.0, offset: 4, speed: 6 });
    const a = runGauntlet(make(), footprint, controller, kinematicForwardSim(agent), { dt: 1 / 60, limits });
    const b = runGauntlet(make(), footprint, controller, kinematicForwardSim(agent), { dt: 1 / 60, limits });
    expect(a.executedMinClearance).toBe(b.executedMinClearance);
    expect(a.corridorCrossTrackRmse).toBe(b.corridorCrossTrackRmse);
  });
});

describe('sweptClearance', () => {
  it('detects a clean pass vs a wall hit', () => {
    const w = buildDogLegCorridor({ footprint, margin: 1.5, offset: 4, speed: 6 });
    const clean = sweptClearance(w.centerline, footprint, w);
    expect(clean.collided).toBe(false);
    expect(clean.minClearance).toBeGreaterThan(0);
    // Shove the trajectory laterally so the footprint straddles a wall (the
    // realistic clip regime — a small deviation, not teleporting fully past it).
    const shoved = w.centerline.map((s) => ({ ...s, z: s.z + w.width / 2 }));
    const hit = sweptClearance(shoved, footprint, w);
    expect(hit.collided).toBe(true);
  });
});
