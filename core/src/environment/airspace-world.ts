// The aircraft planner's only collision coupling. Volumetric and oriented:
// the agent is an OBB (length × span × height), oriented by yaw + pitch +
// roll from the searched state, so the planner can knife-edge through slots
// too narrow for a level-wing footprint. Static box volumes use SAT; moving
// spherical no-fly zones use closest-point-on-OBB.

import type { Predict } from '../predict/types';
import {
  poseToOBB,
  obbHitsAABB,
  obbHitsSphere,
  obbWorldExtent,
  type Pose,
} from '../internal/obb';

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

/** Is the agent's OBB at (pose) at absolute time `t` free of obstacles? The
 *  aircraft Environment depends on nothing else. */
export interface AirspaceWorld {
  clear(pose: Pose, half: [number, number, number], t: number): boolean;
}

export interface AirspaceOptions {
  /** Inclusive flyable altitude band (defaults: unbounded). */
  floor?: number;
  ceiling?: number;
  boxes?: AABB[];
  zones?: MovingZone[];
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

  clear(pose: Pose, half: [number, number, number], t: number): boolean {
    const obb = poseToOBB(pose, half);
    const ext = obbWorldExtent(obb);
    if (ext.min[1] < this.floor || ext.max[1] > this.ceiling) return false;
    for (const b of this.boxes) {
      if (obbHitsAABB(obb, b.min, b.max)) return false;
    }
    for (const zone of this.zones) {
      const c = zone.predict(t);
      if (!c) continue;
      if (obbHitsSphere(obb, [c.x, c.y, c.z], zone.radius)) return false;
    }
    return true;
  }
}
