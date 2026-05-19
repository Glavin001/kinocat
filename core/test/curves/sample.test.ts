import { describe, it, expect } from 'vitest';
import { reedsSheppShortestPath } from '../../src/curves/reeds-shepp';
import { dubinsShortestPath } from '../../src/curves/dubins';
import { sampleCurve, curveEndpoint } from '../../src/curves/sample';
import type { Pose } from '../../src/curves/types';
import { rng, poseClose } from './_util';

describe('sampleCurve', () => {
  it('starts at start and ends at the exact endpoint', () => {
    const start: Pose = { x: 1, y: -2, theta: 0.4 };
    const goal: Pose = { x: 5, y: 3, theta: -1.1 };
    const radius = 1.5;
    const path = reedsSheppShortestPath(start, goal, radius);
    const poses = sampleCurve(start, path, radius, 0.25);
    expect(poseClose(poses[0]!, start, 1e-12, 1e-12)).toBe(true);
    const end = curveEndpoint(start, path, radius);
    expect(poseClose(poses[poses.length - 1]!, end, 1e-9, 1e-9)).toBe(true);
  });

  it('consecutive samples are no farther apart than the step', () => {
    const r = rng(55);
    for (let i = 0; i < 50; i++) {
      const radius = 0.6 + r() * 2;
      const step = 0.2;
      const start: Pose = { x: (r() - 0.5) * 6, y: (r() - 0.5) * 6, theta: (r() - 0.5) * 6 };
      const goal: Pose = { x: (r() - 0.5) * 6, y: (r() - 0.5) * 6, theta: (r() - 0.5) * 6 };
      const path = dubinsShortestPath(start, goal, radius);
      const poses = sampleCurve(start, path, radius, step);
      for (let k = 1; k < poses.length; k++) {
        const d = Math.hypot(poses[k]!.x - poses[k - 1]!.x, poses[k]!.y - poses[k - 1]!.y);
        expect(d).toBeLessThanOrEqual(step + 1e-9);
      }
      if (path.length > 0) {
        expect(poses.length).toBeGreaterThanOrEqual(path.length / step);
      }
    }
  });

  it('samples a reverse Reeds-Shepp path correctly', () => {
    const start: Pose = { x: 0, y: 0, theta: 0 };
    const goal: Pose = { x: -2, y: 0, theta: 0 };
    const radius = 1;
    const path = reedsSheppShortestPath(start, goal, radius);
    const poses = sampleCurve(start, path, radius, 0.1);
    // every sample should be on the negative-x axis heading ~0
    for (const p of poses) {
      expect(Math.abs(p.y)).toBeLessThan(1e-9);
      expect(p.x).toBeLessThanOrEqual(1e-9);
    }
    expect(poseClose(poses[poses.length - 1]!, goal, 1e-6, 1e-6)).toBe(true);
  });
});
