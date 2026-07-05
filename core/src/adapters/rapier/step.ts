// Sub-stepped Rapier raycast-vehicle step helper.
//
// The canonical tick pattern for ANY demo running a raycast vehicle is:
//   - split `dt` into `substeps` mini-ticks
//   - per mini-tick: `vehicle.updateVehicle(subDt, wheelFilter)` then `world.step()`
//
// This helper bakes the pattern into one call so the four demos that
// re-implement it inline can stop. Multi-vehicle scenes (e.g. /carchase
// with robber + cops) pass all `CarHandle`s in a single array — they share
// one world and one `world.step()` per sub-tick.
//
// `wheelFilter` defaults to `EXCLUDE_DYNAMIC` (raycast hits static colliders
// only — matches what every demo currently uses).

import RAPIER from '@dimforge/rapier3d-compat';
import type { CarHandle } from './raycast-vehicle';

export interface StepRaycastVehicleOptions {
  /** Total tick (s) — typically `1 / 60`. */
  dt: number;
  /** Mini-ticks per call (default 4). Higher = more stable, slower. */
  substeps?: number;
  /** Query filter for the wheel raycasts. Defaults to `EXCLUDE_DYNAMIC`. */
  wheelFilter?: number;
}

/** Step one or more raycast vehicles in `world` for `dt`, in `substeps`
 *  mini-ticks. After this call returns the new physics state is available
 *  via each car's `readState`. */
export function stepRaycastVehicle(
  world: RAPIER.World,
  cars: ReadonlyArray<CarHandle>,
  opts: StepRaycastVehicleOptions,
): void {
  const substeps = Math.max(1, Math.floor(opts.substeps ?? 4));
  const subDt = opts.dt / substeps;
  const filter = opts.wheelFilter ?? RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
  for (let s = 0; s < substeps; s++) {
    world.timestep = subDt;
    for (const c of cars) c.vehicle.updateVehicle(subDt, filter);
    world.step();
  }
}
