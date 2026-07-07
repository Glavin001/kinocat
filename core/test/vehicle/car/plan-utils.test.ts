import { describe, it, expect } from 'vitest';
import { samplePlanAt, trimPlan, expandPlanSweeps } from '../../../src/vehicle/car/plan-utils';
import type { CarKinematicState } from '../../../src/vehicle/car/types';
import type { MotionPrimitive } from '../../../src/primitives/types';
import type { Node } from '../../../src/environment/types';

function path(): CarKinematicState[] {
  return [
    { x: 0, z: 0, heading: 0, speed: 5, t: 0 },
    { x: 5, z: 0, heading: 0, speed: 5, t: 1 },
    { x: 10, z: 0, heading: Math.PI / 2, speed: 5, t: 2 },
  ];
}

describe('samplePlanAt', () => {
  it('returns first sample at t<=start', () => {
    const s = samplePlanAt(path(), -1)!;
    expect(s.x).toBe(0);
  });
  it('returns last sample at t>=end', () => {
    const s = samplePlanAt(path(), 99)!;
    expect(s.x).toBe(10);
  });
  it('linearly interpolates position within a bracket', () => {
    const s = samplePlanAt(path(), 0.5)!;
    expect(s.x).toBeCloseTo(2.5);
    expect(s.t).toBeCloseTo(0.5);
  });
  it('interpolates heading across the shorter arc', () => {
    // From 0 to π/2 across t=[1,2], at t=1.5 → π/4.
    const s = samplePlanAt(path(), 1.5)!;
    expect(s.heading).toBeCloseTo(Math.PI / 4, 4);
  });
  it('returns null on empty plan', () => {
    expect(samplePlanAt([], 0)).toBeNull();
  });
});

describe('trimPlan', () => {
  it('drops past samples but keeps at least one', () => {
    const trimmed = trimPlan(path(), 99);
    expect(trimmed.length).toBe(1);
    expect(trimmed[0]!.x).toBe(10);
  });
  it('keeps samples whose t > elapsed', () => {
    const trimmed = trimPlan(path(), 0.5);
    expect(trimmed.length).toBe(3);
  });
});

function node(state: CarKinematicState, edge: Node<CarKinematicState>['edge']): Node<CarKinematicState> {
  return {
    state,
    g: 0, h: 0, f: 0, parent: null, edge,
    index: [], hash: '', level: 0, active: true, seq: 0,
  };
}

describe('expandPlanSweeps', () => {
  it('returns empty for no nodes', () => {
    expect(expandPlanSweeps([], [])).toEqual([]);
  });

  it('inserts a primitive\'s swept poses in world coords with monotonic t', () => {
    // A single primitive: a quarter arc turning left, sampled at 3 local poses.
    const prim: MotionPrimitive = {
      id: 7,
      startSpeed: 4,
      controls: [],
      duration: 1,
      end: { dx: 1, dz: 1, dHeading: Math.PI / 2, speed: 6 },
      sweep: [
        { x: 0, z: 0, heading: 0 },
        { x: 0.7, z: 0.3, heading: Math.PI / 4 },
        { x: 1, z: 1, heading: Math.PI / 2 },
      ],
      reverse: false,
    };
    const start: CarKinematicState = { x: 10, z: 20, heading: 0, speed: 4, t: 0 };
    const end: CarKinematicState = { x: 11, z: 21, heading: Math.PI / 2, speed: 6, t: 1 };
    const nodes = [
      node(start, null),
      node(end, { cost: 1, kind: 'primitive', data: { primId: 7 } }),
    ];
    const out = expandPlanSweeps(nodes, [prim]);
    // start + 2 intermediate sweep poses (sweep[0] skipped as parent).
    expect(out.length).toBe(3);
    expect(out[0]).toMatchObject({ x: 10, z: 20 });
    // heading==0 → world == start + local offset directly.
    expect(out[2]!.x).toBeCloseTo(11);
    expect(out[2]!.z).toBeCloseTo(21);
    expect(out[2]!.heading).toBeCloseTo(Math.PI / 2);
    // speed interpolated start.speed(4) → prim.end.speed(6).
    expect(out[1]!.speed).toBeCloseTo(5);
    expect(out[2]!.speed).toBeCloseTo(6);
    // t strictly increasing.
    for (let i = 1; i < out.length; i++) expect(out[i]!.t).toBeGreaterThan(out[i - 1]!.t);
  });

  it('rotates sweep poses into the parent frame', () => {
    // Parent heading = 90°: local +x maps to world +z.
    const prim: MotionPrimitive = {
      id: 1, startSpeed: 0, controls: [], duration: 1,
      end: { dx: 2, dz: 0, dHeading: 0, speed: 0 },
      sweep: [{ x: 0, z: 0, heading: 0 }, { x: 2, z: 0, heading: 0 }],
      reverse: false,
    };
    const start: CarKinematicState = { x: 0, z: 0, heading: Math.PI / 2, speed: 0, t: 0 };
    const nodes = [
      node(start, null),
      node({ x: 0, z: 2, heading: Math.PI / 2, speed: 0, t: 1 }, { cost: 1, kind: 'primitive', data: { primId: 1 } }),
    ];
    const out = expandPlanSweeps(nodes, [prim]);
    expect(out[1]!.x).toBeCloseTo(0);
    expect(out[1]!.z).toBeCloseTo(2);
  });

  it('falls back to the endpoint when the primitive is missing', () => {
    const start: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 1, t: 0 };
    const end: CarKinematicState = { x: 5, z: 0, heading: 0, speed: 1, t: 1 };
    const nodes = [
      node(start, null),
      node(end, { cost: 1, kind: 'primitive', data: { primId: 999 } }),
    ];
    const out = expandPlanSweeps(nodes, []); // no primitives registered
    expect(out.length).toBe(2);
    expect(out[1]).toMatchObject({ x: 5, z: 0 });
  });
});
