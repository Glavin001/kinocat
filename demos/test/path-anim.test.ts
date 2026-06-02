// Pins the GoalLab path-animation invariant: the car must always face the
// direction it travels along the drawn path (the source of the earlier "car
// rotated perpendicular" bug).
import { describe, it, expect } from 'vitest';
import {
  hermitePose,
  hermiteHeading,
  densifyPath,
  type PathSample,
} from '../app/lib/path-anim';

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

describe('path-anim heading == travel tangent', () => {
  it('forward: car heading matches the direction of motion along the path', () => {
    // A smooth quarter-arc (radius 10), node headings = the arc tangent.
    const path: PathSample[] = [];
    for (let i = 0; i <= 4; i++) {
      const a = (i / 4) * (Math.PI / 2);
      path.push({ x: 10 * Math.sin(a), z: 10 * (1 - Math.cos(a)), heading: a, speed: 5, t: i });
    }
    for (let t = 0.3; t < 3.7; t += 0.2) {
      const a = hermitePose(path, t);
      const b = hermitePose(path, t + 0.02);
      const travel = Math.atan2(b.z - a.z, b.x - a.x);
      expect(Math.abs(wrap(a.heading - travel))).toBeLessThan(0.25);
    }
  });

  it('reverse: the nose points OPPOSITE travel (gear-aware)', () => {
    const path: PathSample[] = [
      { x: 0, z: 0, heading: Math.PI, speed: -3, t: 0 },
      { x: 5, z: 0, heading: Math.PI, speed: -3, t: 2 },
    ];
    const a = hermitePose(path, 0.5);
    const b = hermitePose(path, 0.52);
    const travel = Math.atan2(b.z - a.z, b.x - a.x); // ≈ 0 (moving +x)
    expect(Math.abs(wrap(travel))).toBeLessThan(0.25);
    expect(Math.abs(wrap(a.heading - (travel + Math.PI)))).toBeLessThan(0.25);
  });

  it('long-edge case (the bug): heading tracks the curve, not the stale node heading', () => {
    const a: PathSample = { x: 0, z: 0, heading: (86 * Math.PI) / 180, speed: 6, t: 0 };
    const c: PathSample = { x: 20, z: 4, heading: 0, speed: 6, t: 4 };
    const h = hermiteHeading(a, c, 0.95);
    expect(Math.abs(wrap(h - (86 * Math.PI) / 180))).toBeGreaterThan(0.5);
  });

  it('densifyPath returns a dense polyline through the endpoints', () => {
    const path: PathSample[] = [
      { x: 0, z: 0, heading: 0, speed: 5, t: 0 },
      { x: 10, z: 5, heading: 0, speed: 5, t: 2 },
    ];
    const dense = densifyPath(path, 10);
    expect(dense.length).toBe(11);
    expect(dense[0]).toMatchObject({ x: 0, z: 0 });
    expect(dense[dense.length - 1]).toMatchObject({ x: 10, z: 5 });
  });
});
