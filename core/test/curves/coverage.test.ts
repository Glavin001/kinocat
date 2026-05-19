import { describe, it, expect } from 'vitest';
import { dubinsShortestPath } from '../../src/curves/dubins';
import { reedsSheppShortestPath } from '../../src/curves/reeds-shepp';
import { curveEndpoint } from '../../src/curves/sample';
import { buildPath, type SegType } from '../../src/curves/internal';
import type { Pose } from '../../src/curves/types';
import { poseClose } from './_util';

/** Realize a Pose pair for a desired (d, alpha, beta) with radius 1. */
function poses(d: number, alpha: number, beta: number): [Pose, Pose] {
  const th = -alpha;
  return [
    { x: 0, y: 0, theta: 0 },
    { x: d * Math.cos(th), y: d * Math.sin(th), theta: beta + th },
  ];
}

describe('curves: buildPath edge cases', () => {
  it("treats a missing value for a non-N slot as 0 (?? branch)", () => {
    const tmpl: SegType[] = ['L', 'S', 'L'];
    const p = buildPath('dubins', tmpl, [1], 1); // only one value supplied
    expect(p.segments).toEqual([{ steer: 'L', gear: 1, length: 1 }]);
    expect(p.length).toBe(1);
  });

  it('skips near-zero segments and encodes reverse gear', () => {
    const p = buildPath('reeds-shepp', ['L', 'S', 'R'], [0, -2, 1e-12], 2);
    expect(p.segments).toEqual([{ steer: 'S', gear: -1, length: 4 }]);
    expect(p.word).toBe('S');
  });
});

describe('curves: Dubins degenerate poses', () => {
  it('identical pose returns an empty zero-length path (early return)', () => {
    const p: Pose = { x: 3, y: -1, theta: 0.9 };
    const path = dubinsShortestPath(p, p, 1.5);
    expect(path.length).toBe(0);
    expect(path.segments).toEqual([]);
    expect(path.word).toBe('');
  });

  it('same position, different heading does NOT early-return', () => {
    const a: Pose = { x: 0, y: 0, theta: 0 };
    const b: Pose = { x: 0, y: 0, theta: Math.PI };
    const path = dubinsShortestPath(a, b, 1);
    expect(path.length).toBeGreaterThan(0);
    expect(poseClose(curveEndpoint(a, path, 1), b, 1e-6, 1e-6)).toBe(true);
  });
});

describe('curves: structured (d, alpha, beta) sweep', () => {
  it('covers all Dubins/RS word families and reconstructs every endpoint', () => {
    const ds = [0, 1e-8, 0.05, 0.3, 1, 2.5, 8, 25];
    const angles: number[] = [];
    for (let i = 0; i < 24; i++) angles.push(-Math.PI + (i * (2 * Math.PI)) / 24);
    let dubinsCount = 0;
    let rsCount = 0;
    for (const d of ds) {
      for (const a of angles) {
        for (const b of angles) {
          const [s, g] = poses(d, a, b);
          for (const radius of [1, 3.5]) {
            const db = dubinsShortestPath(s, g, radius);
            if (db.segments.length > 0) {
              expect(
                poseClose(curveEndpoint(s, db, radius), g, 1e-5, 1e-5),
              ).toBe(true);
              dubinsCount++;
            }
            const rs = reedsSheppShortestPath(s, g, radius);
            if (rs.segments.length > 0) {
              expect(
                poseClose(curveEndpoint(s, rs, radius), g, 1e-5, 1e-5),
              ).toBe(true);
              // RS ≤ Dubins is asserted strictly in reeds-shepp.test.ts;
              // here a loose bound suffices on near-degenerate sweep inputs.
              expect(rs.length).toBeLessThanOrEqual(db.length + 1e-2);
              rsCount++;
            }
          }
        }
      }
    }
    expect(dubinsCount).toBeGreaterThan(1000);
    expect(rsCount).toBeGreaterThan(1000);
  });
});
