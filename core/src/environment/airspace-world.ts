// The aircraft planner's only collision coupling. Unlike NavWorld (a 2D
// polygon nav graph with derived height), airspace is genuinely volumetric:
// a flyable altitude band, static box volumes (terrain / canyon walls /
// buildings), and moving spherical no-fly zones (storms, traffic) queried at
// the successor's absolute time — the same Predict<T> seam the rest of kinocat
// uses for everything dynamic.

import type { Predict } from '../predict/types';

/** Axis-aligned box volume in world coordinates. */
export interface AABB {
  min: [number, number, number];
  max: [number, number, number];
}

/** A moving spherical no-fly zone: centre over time + radius. */
export interface MovingZone {
  predict: Predict<{ x: number; y: number; z: number }>;
  radius: number;
}

/** Is a sphere of `radius` centred at (x,y,z) at absolute time `t` free of
 *  obstacles? The aircraft Environment depends on nothing else. */
export interface AirspaceWorld {
  clear(x: number, y: number, z: number, t: number, radius: number): boolean;
}

export interface AirspaceOptions {
  /** Inclusive flyable altitude band (defaults: unbounded). */
  floor?: number;
  ceiling?: number;
  boxes?: AABB[];
  zones?: MovingZone[];
}

function sphereHitsAABB(
  x: number,
  y: number,
  z: number,
  r: number,
  b: AABB,
): boolean {
  const cx = Math.max(b.min[0], Math.min(x, b.max[0]));
  const cy = Math.max(b.min[1], Math.min(y, b.max[1]));
  const cz = Math.max(b.min[2], Math.min(z, b.max[2]));
  const dx = x - cx;
  const dy = y - cy;
  const dz = z - cz;
  return dx * dx + dy * dy + dz * dz < r * r;
}

export class InMemoryAirspace implements AirspaceWorld {
  private readonly floor: number;
  private readonly ceiling: number;
  private readonly boxes: AABB[];
  private readonly zones: MovingZone[];

  constructor(opts: AirspaceOptions = {}) {
    this.floor = opts.floor ?? -Infinity;
    this.ceiling = opts.ceiling ?? Infinity;
    this.boxes = opts.boxes ?? [];
    this.zones = opts.zones ?? [];
  }

  clear(x: number, y: number, z: number, t: number, radius: number): boolean {
    if (y - radius < this.floor || y + radius > this.ceiling) return false;
    for (const b of this.boxes) {
      if (sphereHitsAABB(x, y, z, radius, b)) return false;
    }
    for (const zone of this.zones) {
      const c = zone.predict(t);
      if (!c) continue;
      const dx = x - c.x;
      const dy = y - c.y;
      const dz = z - c.z;
      const rr = radius + zone.radius;
      if (dx * dx + dy * dy + dz * dz < rr * rr) return false;
    }
    return true;
  }
}
