// `Body<CarKinematicState, WheeledCarControls>` implementation backed by the
// kinocat Rapier raycast vehicle (`CarHandle`).
//
// This is the bridge between the generic `kinocat/scene` runtime and the
// existing Rapier adapter. A `RapierCarBody` owns one `CarHandle` plus the
// `world` it lives in and serves three roles:
//   - `readState`        → `car.readState(now)` (planner-frame state)
//   - `applyControls(c)` → `car.applyWheeledControls(c)` (planning-sign safe)
//   - `step(dt)`         → sub-stepped `stepRaycastVehicle(world, [car], ...)`
//
// Multi-car scenes (e.g. /carchase) instantiate one `RapierCarBody` per car
// but share the underlying world. To keep the world step from running
// `cars.length` times per tick, pass `sharedStepPolicy: 'external'` and
// drive `stepRaycastVehicle(world, allCars, ...)` from the demo (one step
// per simulated tick, not per car).

import type RAPIER from '@dimforge/rapier3d-compat';
import type { Body } from '../../scene/body';
import type { CarKinematicState } from '../../agent/types';
import type { WheeledCarControls } from '../../agent/controls';
import type { CarHandle, WheelTelemetry } from './raycast-vehicle';
import { stepRaycastVehicle } from './step';

export interface RapierCarBodyOptions {
  /** Rapier world the chassis lives in. */
  world: RAPIER.World;
  /** Live car handle (returned by `createRaycastVehicle`). */
  car: CarHandle;
  /** Sub-stepping per `step(dt)` call. Default 4. Ignored when
   *  `stepPolicy === 'external'`. */
  substeps?: number;
  /** Initial sim-time (s) used by `readState`. Advanced internally each
   *  step. Defaults to 0. */
  startTime?: number;
  /** Whether this body owns the world step.
   *  - `'self'` (default): `step(dt)` calls `stepRaycastVehicle(world,
   *    [car], ...)`. Use for single-car scenes.
   *  - `'external'`: `step(dt)` only advances internal time + writes any
   *    pending controls; the world is stepped elsewhere (multi-car). */
  stepPolicy?: 'self' | 'external';
}

export class RapierCarBody implements Body<CarKinematicState, WheeledCarControls> {
  private readonly world: RAPIER.World;
  private readonly car: CarHandle;
  private readonly substeps: number;
  private readonly stepPolicy: 'self' | 'external';
  private time: number;
  private pending: WheeledCarControls | null = null;

  constructor(opts: RapierCarBodyOptions) {
    this.world = opts.world;
    this.car = opts.car;
    this.substeps = opts.substeps ?? 4;
    this.stepPolicy = opts.stepPolicy ?? 'self';
    this.time = opts.startTime ?? 0;
  }

  readState(): CarKinematicState {
    return this.car.readState(this.time);
  }

  applyControls(c: WheeledCarControls): void {
    this.pending = c;
    this.car.applyWheeledControls(c);
  }

  step(dt: number): void {
    if (this.stepPolicy === 'self') {
      stepRaycastVehicle(this.world, [this.car], { dt, substeps: this.substeps });
    }
    this.time += dt;
  }

  teleport(state: CarKinematicState): void {
    this.car.teleportFull(
      { x: state.x, z: state.z, heading: state.heading },
      {
        forwardSpeed: state.speed,
        lateralVelocity: state.lateralVelocity ?? 0,
        yawRate: state.yawRate ?? 0,
      },
    );
    this.time = state.t;
    this.pending = null;
  }

  dispose(): void {
    this.car.dispose();
  }

  /** Escape hatch: the underlying handle for wheel-telemetry / Rapier-side
   *  introspection that doesn't fit the `Body` interface. Use sparingly. */
  getHandle(): CarHandle {
    return this.car;
  }

  /** Per-wheel telemetry from the last sub-step. Useful for diagnostics
   *  overlays and the DebugRecorder's `extras` blob. */
  readWheelTelemetry(): WheelTelemetry[] {
    return this.car.readWheelTelemetry();
  }

  /** Last applied controls (or `null` if none have been applied since the
   *  most recent teleport). Read by the recorder so it can capture the
   *  controls actually sent to the chassis. */
  lastControls(): WheeledCarControls | null {
    return this.pending;
  }
}
