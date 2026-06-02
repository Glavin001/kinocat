import { describe, it, expect } from 'vitest';
import {
  at,
  near,
  inside,
  gate,
  corridor,
  halfPlane,
  FORWARD,
  BACKWARD,
  within,
  ahead,
  behind,
  beside,
  cone,
  LEFT,
  RIGHT,
  deg,
} from '../../src/scenario/index';
import type { ScenarioState, RegionAgent } from '../../src/scenario/index';

function s(x: number, z: number, heading = 0, speed = 0, t = 0): ScenarioState {
  return { x, z, heading, speed, t };
}

describe('static regions', () => {
  it('at: pose box with heading tolerance', () => {
    const r = at({ x: 10, z: 5, heading: 0 }, { dx: 0.3, dz: 0.3, dheading: deg(5) });
    expect(r.kind).toBe('at');
    expect(r.dynamic).toBe(false);
    expect(r.contains(s(10, 5, 0))).toBe(true);
    expect(r.contains(s(10.2, 5.1, deg(3)))).toBe(true);
    expect(r.contains(s(11, 5, 0))).toBe(false); // outside dx
    expect(r.contains(s(10, 5, deg(20)))).toBe(false); // outside heading tol
    expect(r.costToGo(s(10, 5, 0))).toBeLessThan(0.01);
    expect(r.costToGo(s(0, 0, 0))).toBeGreaterThan(5);
  });

  it('at: infinite heading tolerance by default', () => {
    const r = at({ x: 0, z: 0, heading: 0 }, { dx: 1, dz: 1 });
    expect(r.contains(s(0, 0, Math.PI))).toBe(true);
  });

  it('near: ball, any heading', () => {
    const r = near({ x: 0, z: 0 }, 2);
    expect(r.contains(s(1, 1))).toBe(true);
    expect(r.contains(s(3, 0))).toBe(false);
    expect(r.costToGo(s(5, 0))).toBeCloseTo(3, 5); // 5 - r(2)
    expect(r.costToGo(s(1, 0))).toBe(0);
    expect(r.representative()).toMatchObject({ x: 0, z: 0 });
  });

  it('inside: polygon area', () => {
    const sq: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ];
    const r = inside(sq);
    expect(r.contains(s(5, 5))).toBe(true);
    expect(r.contains(s(-1, 5))).toBe(false);
    expect(r.costToGo(s(5, 5))).toBe(0);
    expect(r.costToGo(s(50, 5))).toBeGreaterThan(0);
    expect(r.representative()).toMatchObject({ x: 5, z: 5 });
  });

  it('gate: oriented crossing with direction', () => {
    // gate segment from (0,-2) to (0,2): a vertical line at x=0.
    const g = gate({ x: 0, z: -2 }, { x: 0, z: 2 }, FORWARD);
    // moving +x crosses the gate forward.
    expect(g.crossed!(s(-1, 0), s(1, 0))).toBe(true);
    // moving -x crosses but in the wrong direction.
    expect(g.crossed!(s(1, 0), s(-1, 0))).toBe(false);
    // not crossing at all.
    expect(g.crossed!(s(-1, 0), s(-0.5, 0))).toBe(false);
    const back = gate({ x: 0, z: -2 }, { x: 0, z: 2 }, BACKWARD);
    expect(back.crossed!(s(1, 0), s(-1, 0))).toBe(true);
    expect(g.contains(s(0, 0))).toBe(true);
  });

  it('corridor: tube around a centerline', () => {
    const r = corridor(
      [
        { x: 0, z: 0 },
        { x: 10, z: 0 },
      ],
      4,
    );
    expect(r.contains(s(5, 1))).toBe(true); // within half-width 2
    expect(r.contains(s(5, 3))).toBe(false);
    expect(r.costToGo(s(5, 5))).toBeCloseTo(3, 5); // 5 - 2
  });

  it('halfPlane: one side of a line', () => {
    const r = halfPlane({ x: 0, z: 0 }, { x: 1, z: 0 }); // x >= 0
    expect(r.contains(s(5, 0))).toBe(true);
    expect(r.contains(s(-5, 0))).toBe(false);
    expect(r.costToGo(s(-3, 0))).toBeCloseTo(3, 5);
    expect(r.costToGo(s(3, 0))).toBe(0);
  });

  it('region keys are stable + distinct', () => {
    expect(near({ x: 0, z: 0 }, 2).key).toBe(near({ x: 0, z: 0 }, 2).key);
    expect(near({ x: 0, z: 0 }, 2).key).not.toBe(near({ x: 1, z: 0 }, 2).key);
  });
});

describe('dynamic regions', () => {
  const moving: RegionAgent = {
    id: 'lead',
    // moves +x at 2 m/s starting from (0,0) heading 0.
    predict: (t) => ({ x: 2 * t, z: 0, heading: 0, speed: 2, t }),
  };

  it('within: ball around predicted pose (intercept)', () => {
    const r = within(moving, 1);
    expect(r.dynamic).toBe(true);
    // at t=5 the lead is at x=10; a chaser at x=10 is within.
    expect(r.contains(s(10, 0, 0, 0, 5), 5)).toBe(true);
    // at t=0 the lead is at x=0; the same chaser pose is NOT within.
    expect(r.contains(s(10, 0, 0, 0, 0), 0)).toBe(false);
  });

  it('ahead / behind relative to heading', () => {
    const a = ahead(moving, 4); // 4 m in front of the lead
    // at t=0 lead at (0,0) heading 0 => ahead point at (4,0).
    expect(a.contains(s(4, 0, 0, 0, 0), 0)).toBe(true);
    const b = behind(moving, 4);
    expect(b.kind).toBe('behind');
    expect(b.contains(s(-4, 0, 0, 0, 0), 0)).toBe(true);
  });

  it('beside: lateral offset on a side', () => {
    const l = beside(moving, LEFT, 2);
    // heading 0, left is +z => beside point at (0, 2) at t=0.
    expect(l.contains(s(0, 2, 0, 0, 0), 0)).toBe(true);
    const rt = beside(moving, RIGHT, 2);
    expect(rt.contains(s(0, -2, 0, 0, 0), 0)).toBe(true);
  });

  it('cone: vision wedge in front', () => {
    const c = cone(moving, deg(30), 10);
    expect(c.dynamic).toBe(true);
    // directly ahead within range -> seen.
    expect(c.contains(s(5, 0, 0, 0, 0), 0)).toBe(true);
    // behind -> not seen.
    expect(c.contains(s(-5, 0, 0, 0, 0), 0)).toBe(false);
    // ahead but beyond range -> not seen.
    expect(c.contains(s(20, 0, 0, 0, 0), 0)).toBe(false);
    // ahead but outside fov.
    expect(c.contains(s(5, 5, 0, 0, 0), 0)).toBe(false);
  });
});
