// Headless Rapier trial harness.
//
// Spawns a fresh Rapier world + raycast vehicle, teleports the chassis to a
// requested initial kinematic state (forward speed, lateral velocity, yaw
// rate), settles the suspension briefly, then drives the vehicle through a
// caller-supplied native-controls trace at the fixed physics tick (1/60s).
// Samples the resulting `CarKinematicState` at a requested rate and returns them.
//
// Used by the offline-training pipeline to collect ground-truth dynamics
// trials at arbitrary initial conditions (including ones the running planner
// would never naturally visit, like mid-drift starts).
//
// Single-threaded reference implementation; a Web Worker shim wrapping this
// can run multiple in parallel (see `headless-trial-worker.ts`).

import RAPIER from '@dimforge/rapier3d-compat';
import type { CarKinematicState } from '../../agent/types';
import type { WheeledCarControls } from '../../agent/controls';
import { ZERO_WHEELED } from '../../vehicle/car/wheeled';
import {
  type LearnableVehicleConfig,
  DEFAULT_LEARNABLE_CONFIG,
} from '../../agent/vehicle-config';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  type CarHandle,
  type RaycastVehicleOptions,
} from './raycast-vehicle';

export interface TrialSpec {
  /** Initial pose (planning frame). */
  pose: { x: number; z: number; heading: number };
  /** Initial kinematic state. */
  kin: { forwardSpeed: number; lateralVelocity?: number; yawRate?: number };
  /** Native-controls trace, one entry per physics tick (PHYSICS_DT seconds). */
  controlsTrace: WheeledCarControls[];
  /** How many ticks per recorded sample (e.g., 6 = record every 6 ticks). */
  sampleEveryNTicks: number;
  /** Optional id to attach to the result for downstream bookkeeping. */
  id?: string;
}

export type TrialOutcome =
  | { ok: true; trial: HeadlessTrialResult }
  | { ok: false; reason: string };

export interface HeadlessTrialResult {
  id: string;
  spec: TrialSpec;
  /** Recorded samples in world frame (NOT trial-local). Length =
   *  floor(controlsTrace.length / sampleEveryNTicks) + 1 (start + samples). */
  samples: CarKinematicState[];
  /** The vehicle config the chassis was created with. */
  config: LearnableVehicleConfig;
  /** Physics dt the trial was run at (always 1/60 in this implementation). */
  dt: number;
}

export interface HeadlessTrialOptions {
  vehicleOptions: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'>;
  /** World gravity. Default { y: -9.81 }. */
  gravityY?: number;
  /** Ground bounds. Default ±500 each side. */
  groundBounds?: { x0: number; x1: number; z0: number; z1: number };
  /** Friction of the ground plane. */
  groundFriction?: number;
  /** Settle ticks after teleport before recording starts. Default 9 (~0.15s). */
  settleTicks?: number;
  /** Off-arena detection: if |x| or |z| exceeds this, discard the trial. */
  offArenaThreshold?: number;
  /** Perpetual-spin detection: if |yawRate| exceeds this for the full last
   *  half of the trial, discard. Default 10 rad/s. */
  spinThreshold?: number;
}

const PHYSICS_DT = 1 / 60;

export interface HeadlessTrialHarness {
  runTrial(spec: TrialSpec): TrialOutcome;
  config: LearnableVehicleConfig;
  dispose(): void;
}

/** Build a single Rapier world + vehicle and return a harness that can run
 *  many trials, teleporting between them. Significantly faster than tearing
 *  down the world per trial. */
export async function createHeadlessTrialHarness(
  opts: HeadlessTrialOptions,
): Promise<HeadlessTrialHarness> {
  const rapier = await ensureRapier();
  const world = new rapier.World({ x: 0, y: opts.gravityY ?? -9.81, z: 0 });
  const bounds = opts.groundBounds ?? { x0: -500, x1: 500, z0: -500, z1: 500 };
  createGroundCollider(world, {
    bounds,
    pad: 20,
    friction: opts.groundFriction ?? 1.5,
  });
  const car: CarHandle = createRaycastVehicle(world, {
    id: 'headless-trial',
    position: { x: 0, z: 0 },
    heading: 0,
    ...opts.vehicleOptions,
  });
  // Initial settle so the suspension is at rest length.
  for (let i = 0; i < 30; i++) {
    car.applyWheeledControls(ZERO_WHEELED);
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    world.step();
  }
  // Derive config once from the supplied vehicle options.
  const config = deriveConfigFromOptions(opts.vehicleOptions);
  const settleTicks = opts.settleTicks ?? 9;
  const offArena = opts.offArenaThreshold ?? Math.max(
    Math.abs(bounds.x0), Math.abs(bounds.x1), Math.abs(bounds.z0), Math.abs(bounds.z1),
  );
  const spinLim = opts.spinThreshold ?? 10;

  function step(c: WheeledCarControls): void {
    car.applyWheeledControls(c);
    world.timestep = PHYSICS_DT;
    car.vehicle.updateVehicle(PHYSICS_DT);
    world.step();
  }

  function runTrial(spec: TrialSpec): TrialOutcome {
    if (spec.sampleEveryNTicks < 1) {
      return { ok: false, reason: 'sampleEveryNTicks must be >= 1' };
    }
    if (spec.controlsTrace.length === 0) {
      return { ok: false, reason: 'controlsTrace must be non-empty' };
    }
    car.teleportFull(spec.pose, spec.kin);
    // Settle.
    for (let i = 0; i < settleTicks; i++) {
      step({ steer: 0, driveForce: 0, brakeForce: 0 });
    }
    // Restore the requested initial kinematic state after settle, because the
    // settle phase may have dissipated some of it (no controls = drag).
    // For lateral velocity / yaw rate specifically this is important.
    if (
      Math.abs(spec.kin.lateralVelocity ?? 0) > 0.05 ||
      Math.abs(spec.kin.yawRate ?? 0) > 0.05
    ) {
      car.teleportFull(spec.pose, spec.kin);
    }
    const samples: CarKinematicState[] = [];
    const startTime = 0;
    samples.push({ ...car.readState(startTime) });
    let lateSpinTicks = 0;
    const half = Math.floor(spec.controlsTrace.length / 2);
    for (let tick = 0; tick < spec.controlsTrace.length; tick++) {
      const c = spec.controlsTrace[tick]!;
      step(c);
      const st = car.readState((tick + 1) * PHYSICS_DT);
      // Pathological detection.
      if (!Number.isFinite(st.x) || !Number.isFinite(st.z) || !Number.isFinite(st.heading) || !Number.isFinite(st.speed)) {
        return { ok: false, reason: 'NaN in state' };
      }
      if (Math.abs(st.x) > offArena || Math.abs(st.z) > offArena) {
        return { ok: false, reason: 'off-arena' };
      }
      if (tick >= half && Math.abs(st.yawRate ?? 0) > spinLim) {
        lateSpinTicks++;
      }
      if ((tick + 1) % spec.sampleEveryNTicks === 0) {
        samples.push({ ...st });
      }
    }
    if (lateSpinTicks > Math.floor((spec.controlsTrace.length - half) * 0.8)) {
      return { ok: false, reason: 'perpetual spin' };
    }
    return {
      ok: true,
      trial: {
        id: spec.id ?? `trial-${Math.random().toString(36).slice(2, 10)}`,
        spec,
        samples,
        config,
        dt: PHYSICS_DT,
      },
    };
  }

  function dispose(): void {
    car.dispose();
    world.free();
  }

  return { runTrial, config, dispose };
}

function deriveConfigFromOptions(
  opts: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'>,
): LearnableVehicleConfig {
  // We don't have the `withDefaults` helper exported, but the relevant
  // defaults match `DEFAULT_LEARNABLE_CONFIG`. Lift any explicitly-set
  // values from `opts`; fall back otherwise.
  const half = opts.chassisHalf ?? { x: 2.4, y: 0.5, z: 1.0 };
  const density = opts.chassisDensity ?? 60;
  const volume = 8 * half.x * half.y * half.z;
  return {
    chassisMass: volume * density,
    wheelBase: opts.wheelBase ?? DEFAULT_LEARNABLE_CONFIG.wheelBase,
    wheelTrack: opts.wheelTrack ?? DEFAULT_LEARNABLE_CONFIG.wheelTrack,
    wheelRadius: opts.wheelRadius ?? DEFAULT_LEARNABLE_CONFIG.wheelRadius,
    suspensionStiffness: opts.suspensionStiffness ?? DEFAULT_LEARNABLE_CONFIG.suspensionStiffness,
    frictionSlip: opts.frictionSlip ?? DEFAULT_LEARNABLE_CONFIG.frictionSlip,
    sideFrictionStiffness: opts.sideFrictionStiffness ?? DEFAULT_LEARNABLE_CONFIG.sideFrictionStiffness,
    maxDriveForce: opts.engineForce ?? DEFAULT_LEARNABLE_CONFIG.maxDriveForce,
    maxBrakeForce: opts.brakeForce ?? DEFAULT_LEARNABLE_CONFIG.maxBrakeForce,
    maxSteerAngle: opts.maxSteerAngle ?? DEFAULT_LEARNABLE_CONFIG.maxSteerAngle,
    drivenWheels: (opts.driveTrain ?? DEFAULT_LEARNABLE_CONFIG.drivenWheels) as LearnableVehicleConfig['drivenWheels'],
  };
}
