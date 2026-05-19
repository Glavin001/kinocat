import { describe, it, expect } from 'vitest';
import { ReplanState, planPoseAt } from '../../src/execute/replan';
import type { PlanPath } from '../../src/execute/types';
import type { VehicleState } from '../../src/agent/types';

const path: PlanPath = [
  { x: 0, z: 0, heading: 0, speed: 4, t: 0 },
  { x: 8, z: 0, heading: 0, speed: 4, t: 2 },
  { x: 8, z: 8, heading: Math.PI / 2, speed: 4, t: 4 },
];

describe('planPoseAt', () => {
  it('interpolates and clamps to the ends', () => {
    expect(planPoseAt([], 1)).toBeNull();
    expect(planPoseAt(path, -1)).toEqual(path[0]);
    expect(planPoseAt(path, 99)).toEqual(path[2]);
    const mid = planPoseAt(path, 1)!;
    expect(mid.x).toBeCloseTo(4, 9);
    expect(mid.z).toBeCloseTo(0, 9);
    const mid2 = planPoseAt(path, 3)!;
    expect(mid2.x).toBeCloseTo(8, 9);
    expect(mid2.z).toBeCloseTo(4, 9);
  });
});

describe('ReplanState', () => {
  const trigger = { divergenceThresholdMeters: 1.5, refreshIntervalMs: 500 };

  it('requires a plan before anything else', () => {
    const rs = new ReplanState(trigger);
    expect(rs.currentPlan).toBeNull();
    expect(rs.shouldReplan({ x: 0, z: 0, heading: 0, speed: 0, t: 0 }, 0)).toBe(true);
  });

  it('does not replan immediately after setPlan when on-track', () => {
    const rs = new ReplanState(trigger);
    rs.setPlan(path, 1000);
    const onTrack: VehicleState = { x: 4, z: 0, heading: 0, speed: 4, t: 1 };
    expect(rs.shouldReplan(onTrack, 1100)).toBe(false);
  });

  it('replans on periodic refresh', () => {
    const rs = new ReplanState(trigger);
    rs.setPlan(path, 1000);
    const onTrack: VehicleState = { x: 4, z: 0, heading: 0, speed: 4, t: 1 };
    expect(rs.shouldReplan(onTrack, 1600)).toBe(true); // 600ms > 500ms
  });

  it('replans on divergence beyond threshold', () => {
    const rs = new ReplanState(trigger);
    rs.setPlan(path, 1000);
    const off: VehicleState = { x: 4, z: 5, heading: 0, speed: 4, t: 1 };
    expect(rs.divergence(off)).toBeCloseTo(5, 9);
    expect(rs.shouldReplan(off, 1100)).toBe(true);
  });

  it('replans when marked dirty (event-driven)', () => {
    const rs = new ReplanState(trigger);
    rs.setPlan(path, 1000);
    rs.markDirty('tile-rebuild');
    const onTrack: VehicleState = { x: 4, z: 0, heading: 0, speed: 4, t: 1 };
    expect(rs.shouldReplan(onTrack, 1100)).toBe(true);
    rs.setPlan(path, 1100); // setPlan clears the dirty flag
    expect(rs.shouldReplan(onTrack, 1150)).toBe(false);
  });
});
