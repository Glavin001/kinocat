import { describe, it, expect } from 'vitest';
import { dubinsShortestPath } from '../../src/curves/dubins';
import { curveEndpoint } from '../../src/curves/sample';
import type { Pose } from '../../src/curves/types';
import { rng, poseClose } from './_util';

describe('dubins', () => {
  it('straight-ahead goal is a single forward straight', () => {
    const start: Pose = { x: 0, y: 0, theta: 0 };
    const goal: Pose = { x: 5, y: 0, theta: 0 };
    const path = dubinsShortestPath(start, goal, 1);
    expect(path.length).toBeCloseTo(5, 6);
    expect(path.segments).toEqual([{ steer: 'S', gear: 1, length: 5 }]);
  });

  it('all segments are forward (gear = 1)', () => {
    const r = rng(7);
    for (let i = 0; i < 200; i++) {
      const radius = 0.5 + r() * 2;
      const start: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const goal: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const path = dubinsShortestPath(start, goal, radius);
      for (const s of path.segments) expect(s.gear).toBe(1);
    }
  });

  it('endpoint reconstruction matches the goal (fuzz, 500 cases)', () => {
    const r = rng(424242);
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
      const path = dubinsShortestPath(start, goal, radius);
      const end = curveEndpoint(start, path, radius);
      expect(poseClose(end, goal, 1e-6, 1e-6)).toBe(true);
    }
  });

  it('is at least the straight-line distance long', () => {
    const r = rng(3);
    for (let i = 0; i < 200; i++) {
      const radius = 0.5 + r() * 2;
      const start: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const goal: Pose = { x: (r() - 0.5) * 8, y: (r() - 0.5) * 8, theta: (r() - 0.5) * 6 };
      const path = dubinsShortestPath(start, goal, radius);
      const straight = Math.hypot(goal.x - start.x, goal.y - start.y);
      expect(path.length).toBeGreaterThanOrEqual(straight - 1e-6);
    }
  });
});
