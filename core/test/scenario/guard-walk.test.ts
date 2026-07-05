import { describe, it, expect } from 'vitest';
import {
  checkAcceptance,
  guardSatisfied,
  goalRegions,
  avoidRegions,
  maintainRegions,
  collectScenarioRegions,
  defineScenario,
  reach,
  seq,
  any,
  near,
  gate,
  inside,
  avoid,
  maintain,
  stayInside,
  stopped,
  speed,
  distanceFrom,
  closingSpeed,
  lte,
  gte,
  inRange,
  deg,
} from '../../src/scenario/index';
import type { GuardPredicate } from '../../src/scenario/index';
import type { ScenarioState, RegionAgent } from '../../src/scenario/index';

const S = (x: number, z: number, heading = 0, speed = 0, t = 0): ScenarioState => ({
  x, z, heading, speed, t,
});

describe('checkAcceptance', () => {
  it('undefined acceptance always holds', () => {
    expect(checkAcceptance(undefined, S(0, 0))).toBe(true);
  });
  it('speed band', () => {
    expect(checkAcceptance({ speed: { max: 0 } }, S(0, 0, 0, 0))).toBe(true);
    expect(checkAcceptance({ speed: { max: 0 } }, S(0, 0, 0, 1))).toBe(false);
    expect(checkAcceptance({ speed: { min: 8 } }, S(0, 0, 0, 9))).toBe(true);
    expect(checkAcceptance({ speed: { min: 8 } }, S(0, 0, 0, 5))).toBe(false);
  });
  it('stopped() is a symmetric band: reverse motion is NOT stopped', () => {
    // The footgun this replaces: {max: 0} bounds SIGNED speed, so a car
    // reversing through the goal at -1.5 m/s "satisfied" the stop condition
    // (the /parking DONE-while-reversing bug). stopped() rejects it.
    expect(checkAcceptance({ speed: { max: 0 } }, S(0, 0, 0, -1.5))).toBe(true); // the footgun, documented
    expect(checkAcceptance(stopped(), S(0, 0, 0, -1.5))).toBe(false);
    expect(checkAcceptance(stopped(), S(0, 0, 0, 1.5))).toBe(false);
    expect(checkAcceptance(stopped(), S(0, 0, 0, 0))).toBe(true);
    expect(checkAcceptance(stopped(), S(0, 0, 0, -0.04))).toBe(true);
    expect(checkAcceptance(stopped(0.2), S(0, 0, 0, -0.15))).toBe(true);
  });
  it('heading band (wrap-aware arc + one-sided)', () => {
    expect(checkAcceptance({ heading: { min: -deg(10), max: deg(10) } }, S(0, 0, deg(5)))).toBe(true);
    expect(checkAcceptance({ heading: { min: -deg(10), max: deg(10) } }, S(0, 0, deg(40)))).toBe(false);
    expect(checkAcceptance({ heading: { min: 0 } }, S(0, 0, 1))).toBe(true);
    expect(checkAcceptance({ heading: { min: 0 } }, S(0, 0, -1))).toBe(false);
    expect(checkAcceptance({ heading: { max: 1 } }, S(0, 0, 0.5))).toBe(true);
    expect(checkAcceptance({ heading: { max: 1 } }, S(0, 0, 2))).toBe(false);
  });
  it('window + deadline read the clock', () => {
    expect(checkAcceptance({ window: [2, 5] }, S(0, 0, 0, 0, 3))).toBe(true);
    expect(checkAcceptance({ window: [2, 5] }, S(0, 0, 0, 0, 6))).toBe(false);
    expect(checkAcceptance({ by: 10 }, S(0, 0, 0, 0, 9))).toBe(true);
    expect(checkAcceptance({ by: 10 }, S(0, 0, 0, 0, 11))).toBe(false);
  });
});

describe('guardSatisfied', () => {
  it('uses crossed for gates (catches tunneling) + falls back to contains', () => {
    const guard: GuardPredicate = { region: gate({ x: 0, z: -2 }, { x: 0, z: 2 }) };
    // edge tunnels through the thin gate.
    expect(guardSatisfied(guard, S(-1, 0), S(1, 0))).toBe(true);
    // endpoint inside (membership fallback).
    expect(guardSatisfied(guard, S(-1, 0), S(0, 0))).toBe(true);
  });
  it('combines spatial membership AND acceptance', () => {
    const guard: GuardPredicate = { region: near({ x: 10, z: 0 }, 1), accept: { speed: { max: 0 } } };
    expect(guardSatisfied(guard, S(9, 0), S(10, 0, 0, 0))).toBe(true);
    expect(guardSatisfied(guard, S(9, 0), S(10, 0, 0, 5))).toBe(false); // moving -> accept fails
    expect(guardSatisfied(guard, S(9, 0), S(5, 0, 0, 0))).toBe(false); // not in region
  });
});

describe('AST walkers', () => {
  const cop: RegionAgent = { id: 'cop', predict: () => S(0, 0) };
  const sc = defineScenario('walk', {
    start: S(0, 0),
    goal: seq(reach(near({ x: 1, z: 0 }, 1)), any(reach(near({ x: 2, z: 0 }, 1)), reach(near({ x: 3, z: 0 }, 1)))),
    invariants: [stayInside(inside([[0, 0], [10, 0], [10, 10], [0, 10]])), avoid(near({ x: 5, z: 5 }, 1)), maintain(speed(lte(8)))],
    agents: [cop],
  });
  it('goalRegions yields every reach region in order', () => {
    expect(goalRegions(sc.goal)).toHaveLength(3);
  });
  it('avoidRegions / maintainRegions split by plane', () => {
    expect(avoidRegions(sc.invariants)).toHaveLength(1);
    expect(maintainRegions(sc.invariants)).toHaveLength(2); // stayInside + maintain(speed)
  });
  it('collectScenarioRegions buckets all regions', () => {
    const r = collectScenarioRegions(sc);
    expect(r.objective).toHaveLength(3);
    expect(r.avoid).toHaveLength(1);
    expect(r.maintain).toHaveLength(2);
  });
});

describe('distanceFrom condition-region', () => {
  const lead: RegionAgent = { id: 'lead', predict: (t) => S(2 * t, 0, 0, 2, t) };
  it('respects the distance bound against the predicted pose', () => {
    const r = distanceFrom(lead, gte(1));
    // at t=0 lead at (0,0); chaser 5 m away -> >= 1 holds.
    expect(r.contains(S(5, 0, 0, 0, 0), 0)).toBe(true);
    // chaser right on the lead -> < 1 -> violated.
    expect(r.contains(S(0, 0, 0, 0, 0), 0)).toBe(false);
    expect(r.dynamic).toBe(true);
    expect(r.costToGo(S(0, 0))).toBe(0);
  });
  it('treats an unknown predictor horizon as not-violated', () => {
    const gone: RegionAgent = { id: 'gone', predict: () => null };
    expect(distanceFrom(gone, gte(1)).contains(S(0, 0, 0, 0, 99), 99)).toBe(true);
  });
});

describe('closingSpeed condition-region', () => {
  // lead drives +x at 5 m/s; ego sits behind it at the origin, heading +x.
  const lead: RegionAgent = { id: 'lead', predict: (t) => S(20 + 5 * t, 0, 0, 5, t) };

  it('positive when ego is catching up (closing the gap)', () => {
    const r = closingSpeed(lead, gte(2));
    expect(r.contains(S(0, 0, 0, /*speed*/ 8, 0), 0)).toBe(true); // 8 vs 5 -> closing 3
    expect(closingSpeed(lead, lte(0)).contains(S(0, 0, 0, 8, 0), 0)).toBe(false);
  });

  it('~zero when matched (holding station / steady draft)', () => {
    expect(closingSpeed(lead, inRange(-1, 1)).contains(S(0, 0, 0, 5, 0), 0)).toBe(true);
  });

  it('negative when falling back', () => {
    expect(closingSpeed(lead, lte(0)).contains(S(0, 0, 0, 2, 0), 0)).toBe(true); // 2 vs 5
    expect(closingSpeed(lead, gte(0)).contains(S(0, 0, 0, 2, 0), 0)).toBe(false);
  });

  it('dynamic + non-spatial (costToGo 0); unknown horizon not violated', () => {
    expect(closingSpeed(lead, gte(0)).dynamic).toBe(true);
    expect(closingSpeed(lead, gte(0)).costToGo(S(0, 0))).toBe(0);
    const gone: RegionAgent = { id: 'gone', predict: () => null };
    expect(closingSpeed(gone, gte(99)).contains(S(0, 0, 0, 0, 5), 5)).toBe(true);
  });
});
