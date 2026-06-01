import { describe, it, expect } from 'vitest';
import { rampNavObstacles } from '../../src/environment/ramp';
import type { RampSpec } from '../../src/environment/ramp';
import {
  placeFootprint,
  polygonsIntersect,
  segmentsIntersect,
  type Pt,
} from '../../src/internal/geom';

// A modest car footprint (heading 0 = +x), ~4.4 m × 1.9 m.
const FOOTPRINT: Pt[] = [
  [2.2, 0.95],
  [-2.2, 0.95],
  [-2.2, -0.95],
  [2.2, -0.95],
];

function footprintHitsAny(
  walls: Pt[][],
  x: number,
  z: number,
  heading: number,
): boolean {
  const fp = placeFootprint(FOOTPRINT, x, z, heading);
  return walls.some((w) => polygonsIntersect(fp, w));
}

function segmentHitsAny(
  walls: Pt[][],
  x0: number,
  z0: number,
  x1: number,
  z1: number,
): boolean {
  return walls.some((w) => {
    for (let i = 0; i < w.length; i++) {
      const a = w[i]!;
      const b = w[(i + 1) % w.length]!;
      if (segmentsIntersect(x0, z0, x1, z1, a[0], a[1], b[0], b[1])) return true;
    }
    return false;
  });
}

describe('rampNavObstacles — solid-wedge planner collision', () => {
  const ramp: RampSpec = {
    id: 'r',
    base: { x: 0, z: 0 },
    length: 10,
    width: 6,
    height: 2,
    heading: 0,
  };
  // Geometry for heading 0: along = world x, lateral = world z.
  // halfL=5, halfW=3 → crest at x=5, foot at x=-5, sides at |z|∈[2.4,3].

  it('returns 3 walls with back (default), 2 without', () => {
    expect(rampNavObstacles(ramp).length).toBe(3);
    expect(rampNavObstacles(ramp, { back: false }).length).toBe(2);
  });

  it('blocks the broad sides', () => {
    const walls = rampNavObstacles(ramp);
    // Footprint sitting on the −Z side overlaps a wall.
    expect(footprintHitsAny(walls, 0, -3, 0)).toBe(true);
    // ...and on the +Z side.
    expect(footprintHitsAny(walls, 0, 3, 0)).toBe(true);
    // A path that crosses the body laterally hits a wall.
    expect(segmentHitsAny(walls, 0, -6, 0, 6)).toBe(true);
  });

  it('blocks the back face', () => {
    const walls = rampNavObstacles(ramp);
    // Approaching the crest from behind (past the back-skirt) hits the back.
    expect(footprintHitsAny(walls, 7, 0, 0)).toBe(true);
    // ...but with back:false the rear is open.
    const open = rampNavObstacles(ramp, { back: false });
    expect(footprintHitsAny(open, 7, 0, 0)).toBe(false);
  });

  it('leaves the front foot open', () => {
    const walls = rampNavObstacles(ramp);
    // Footprint at the foot, and the centreline drive-up approach, are clear.
    expect(footprintHitsAny(walls, -5, 0, 0)).toBe(false);
    expect(segmentHitsAny(walls, -8, 0, 0, 0)).toBe(false);
  });

  it('leaves the crest jump-entry zone reachable up the centreline', () => {
    const walls = rampNavObstacles(ramp);
    // A centreline pose within the jump entryRadius (~3.5 m) of the crest (5,0)
    // has a collision-free footprint, so the affordance can still trigger.
    const x = 2.5; // dist to crest = 2.5 < entryRadius
    expect(Math.hypot(5 - x, 0)).toBeLessThan(3.5);
    expect(footprintHitsAny(walls, x, 0, 0)).toBe(false);
  });

  it('orients with heading (π/2 case)', () => {
    const turned: RampSpec = { ...ramp, heading: Math.PI / 2 };
    const walls = rampNavObstacles(turned);
    // For heading π/2 the ramp points +Z; the broad sides face ±X.
    expect(footprintHitsAny(walls, 3, 0, Math.PI / 2)).toBe(true);
    expect(footprintHitsAny(walls, -3, 0, Math.PI / 2)).toBe(true);
    // The foot (now at z=-5) on the centreline is open.
    expect(footprintHitsAny(walls, 0, -5, Math.PI / 2)).toBe(false);
  });
});
