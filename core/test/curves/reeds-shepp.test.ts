import { describe, it, expect } from 'vitest';
import { reedsSheppShortestPath } from '../../src/curves/reeds-shepp';
import { dubinsShortestPath } from '../../src/curves/dubins';
import { curveEndpoint } from '../../src/curves/sample';
import type { Pose } from '../../src/curves/types';
import { rng, poseClose } from './_util';

describe('reeds-shepp', () => {
  it('identical pose has zero length and no segments', () => {
    const p: Pose = { x: 1, y: 2, theta: 0.7 };
    const path = reedsSheppShortestPath(p, p, 1.2);
    expect(path.length).toBeCloseTo(0, 9);
    expect(path.segments.length).toBe(0);
  });

  it('straight-ahead goal is a single forward straight', () => {
    const start: Pose = { x: 0, y: 0, theta: 0 };
    const goal: Pose = { x: 4, y: 0, theta: 0 };
    const path = reedsSheppShortestPath(start, goal, 1);
    expect(path.length).toBeCloseTo(4, 6);
    expect(path.segments).toEqual([{ steer: 'S', gear: 1, length: 4 }]);
  });

  it('goal directly behind is a single reverse straight', () => {
    const start: Pose = { x: 0, y: 0, theta: 0 };
    const goal: Pose = { x: -3, y: 0, theta: 0 };
    const path = reedsSheppShortestPath(start, goal, 1);
    expect(path.length).toBeCloseTo(3, 6);
    expect(path.segments.length).toBe(1);
    expect(path.segments[0]!.steer).toBe('S');
    expect(path.segments[0]!.gear).toBe(-1);
  });

  it('endpoint reconstruction matches the goal (fuzz, 500 cases)', () => {
    const r = rng(12345);
    let worst = 0;
    for (let i = 0; i < 500; i++) {
      const radius = 0.5 + r() * 3;
      const start: Pose = {
        x: (r() - 0.5) * 10,
        y: (r() - 0.5) * 10,
        theta: (r() - 0.5) * 2 * Math.PI,
      };
      const goal: Pose = {
        x: (r() - 0.5) * 10,
        y: (r() - 0.5) * 10,
        theta: (r() - 0.5) * 2 * Math.PI,
      };
      const path = reedsSheppShortestPath(start, goal, radius);
      const end = curveEndpoint(start, path, radius);
      worst = Math.max(worst, Math.abs(end.x - goal.x), Math.abs(end.y - goal.y));
      expect(poseClose(end, goal, 1e-6, 1e-6)).toBe(true);
      expect(path.length).toBeGreaterThan(0);
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it('is never longer than the Dubins path for the same query', () => {
    const r = rng(999);
    for (let i = 0; i < 300; i++) {
      const radius = 0.5 + r() * 2;
      const start: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const goal: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const rs = reedsSheppShortestPath(start, goal, radius);
      const db = dubinsShortestPath(start, goal, radius);
      expect(rs.length).toBeLessThanOrEqual(db.length + 1e-6);
    }
  });
});
