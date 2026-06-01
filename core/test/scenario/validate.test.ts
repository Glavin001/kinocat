import { describe, it, expect } from 'vitest';
import {
  defineScenario,
  reach,
  seq,
  any,
  at,
  near,
  within,
  avoid,
  maintain,
  stayInside,
  speed,
  distanceFrom,
  lte,
  gte,
  validate,
  deg,
  minTime,
  smooth,
  keepClear,
  racingLine,
  maxProgress,
} from '../../src/scenario/index';
import type { ScenarioState, RegionAgent } from '../../src/scenario/index';

const start: ScenarioState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
const lot: [number, number][] = [
  [-20, -20],
  [20, -20],
  [20, 20],
  [-20, 20],
];

describe('validate', () => {
  it('clean scenario -> no errors', () => {
    const sc = defineScenario('ok', {
      start,
      goal: reach(at({ x: 10, z: 5, heading: 0 }, { dx: 0.5, dz: 0.5, dheading: deg(5) }), {
        speed: { max: 0 },
      }),
      invariants: [stayInside(lot)],
      prefer: [minTime(1)],
    });
    const diags = validate(sc, { posCell: 0.3 });
    expect(diags.filter((d) => d.severity === 'error')).toHaveLength(0);
  });

  it('flags margins below planner discretization', () => {
    const sc = defineScenario('tight', {
      start,
      goal: reach(at({ x: 10, z: 5, heading: 0 }, { dx: 0.1, dz: 0.1 })),
    });
    const diags = validate(sc, { posCell: 0.3 });
    expect(diags.some((d) => d.check === 'margins-vs-resolution')).toBe(true);
  });

  it('flags unsatisfiable structure (any of nothing)', () => {
    const sc = defineScenario('empty', { start, goal: any() });
    const diags = validate(sc);
    expect(diags.some((d) => d.check === 'unsatisfiable-structure')).toBe(true);
  });

  it('flags contradictory maintain(speed) bounds', () => {
    const sc = defineScenario('contradiction', {
      start,
      goal: reach(near({ x: 5, z: 0 }, 1)),
      invariants: [maintain(speed(gte(5))), maintain(speed(lte(3)))],
    });
    const diags = validate(sc);
    expect(diags.some((d) => d.check === 'contradictory-invariants')).toBe(true);
  });

  it('flags a dynamic region whose agent is missing', () => {
    const cop: RegionAgent = { id: 'cop', predict: () => ({ ...start }) };
    const sc = defineScenario('missing-agent', {
      start,
      goal: reach(within(cop, 2)),
      // agents omitted on purpose
    });
    const diags = validate(sc);
    expect(diags.some((d) => d.check === 'dynamic-region-agent')).toBe(true);
  });

  it('accepts a dynamic region when the agent IS registered', () => {
    const cop: RegionAgent = { id: 'cop', predict: () => ({ ...start }) };
    const sc = defineScenario('with-agent', {
      start,
      goal: reach(within(cop, 2)),
      invariants: [maintain(distanceFrom(cop, gte(1)))],
      agents: [cop],
    });
    const diags = validate(sc);
    expect(diags.filter((d) => d.check === 'dynamic-region-agent')).toHaveLength(0);
  });

  it('flags an unreachable deadline (coarse time feasibility)', () => {
    const sc = defineScenario('late', {
      start,
      goal: reach(near({ x: 1000, z: 0 }, 1), { by: 1 }), // 1000 m in 1 s -> impossible
    });
    const diags = validate(sc, { refSpeed: 10 });
    expect(diags.some((d) => d.check === 'time-feasibility')).toBe(true);
  });
});

describe('cost terms', () => {
  const a: ScenarioState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
  const b: ScenarioState = { x: 3, z: 4, heading: deg(90), speed: 5, t: 1 };

  it('minTime penalizes elapsed dt', () => {
    expect(minTime(2).edgeCost(a, b, 1)).toBe(2);
  });
  it('smooth penalizes heading + speed change', () => {
    expect(smooth(1).edgeCost(a, b, 1)).toBeGreaterThan(0);
  });
  it('keepClear ramps up as clearance drops', () => {
    const term = keepClear(10, 2, [{ x: 3, z: 4 }]);
    expect(term.edgeCost(a, b, 1)).toBeGreaterThan(0); // at the obstacle
    const far = keepClear(10, 2, [{ x: 100, z: 100 }]);
    expect(far.edgeCost(a, b, 1)).toBe(0);
  });
  it('racingLine penalizes lateral deviation', () => {
    const term = racingLine(1, [
      { x: 0, z: 0 },
      { x: 10, z: 0 },
    ]);
    expect(term.edgeCost(a, b, 1)).toBeCloseTo(4, 5); // b is 4 off the z=0 line
  });
  it('maxProgress penalizes shortfall from ideal advance', () => {
    const term = maxProgress(1, 10);
    // advanced 5 m, ideal 10 m -> penalty 5.
    expect(term.edgeCost(a, b, 1)).toBeCloseTo(5, 5);
  });
});
