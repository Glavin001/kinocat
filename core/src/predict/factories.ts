import type { Predict, MovingObstacle } from './types';
import type { VehicleState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import { wrapAngle } from '../internal/math';

const DEFAULT_HORIZON = 10; // seconds

/** Constant-velocity extrapolation of a vehicle along its current heading. */
export function constantVelocity(
  state: VehicleState,
  horizon = DEFAULT_HORIZON,
): Predict<VehicleState> {
  const vx = state.speed * Math.cos(state.heading);
  const vz = state.speed * Math.sin(state.heading);
  return (t) => {
    const dt = t - state.t;
    if (dt < 0 || dt > horizon) return null;
    return {
      x: state.x + vx * dt,
      z: state.z + vz * dt,
      heading: state.heading,
      speed: state.speed,
      t,
    };
  };
}

/** Constant-acceleration extrapolation (world-frame accel a = [ax, az]). */
export function constantAcceleration(
  state: VehicleState,
  accel: { ax: number; az: number },
  horizon = DEFAULT_HORIZON,
): Predict<VehicleState> {
  const vx0 = state.speed * Math.cos(state.heading);
  const vz0 = state.speed * Math.sin(state.heading);
  return (t) => {
    const dt = t - state.t;
    if (dt < 0 || dt > horizon) return null;
    const vx = vx0 + accel.ax * dt;
    const vz = vz0 + accel.az * dt;
    return {
      x: state.x + vx0 * dt + 0.5 * accel.ax * dt * dt,
      z: state.z + vz0 * dt + 0.5 * accel.az * dt * dt,
      heading: wrapAngle(Math.atan2(vz, vx)),
      speed: Math.hypot(vx, vz),
      t,
    };
  };
}

/** Roll a ForwardSim forward once, sample at `dtStep`, and lerp on query. */
export function fromPhysicsRollout(
  initial: VehicleState,
  controls: number[],
  forwardSim: ForwardSim<VehicleState>,
  dtStep: number,
  horizon = DEFAULT_HORIZON,
): Predict<VehicleState> {
  const samples: VehicleState[] = [initial];
  let s = initial;
  const steps = Math.ceil(horizon / dtStep);
  for (let i = 0; i < steps; i++) {
    s = forwardSim(s, controls, dtStep);
    samples.push(s);
  }
  const t0 = initial.t;
  return (t) => {
    const dt = t - t0;
    if (dt < 0 || dt > horizon) return null;
    const fi = dt / dtStep;
    const i = Math.min(Math.floor(fi), samples.length - 2);
    const a = samples[i]!;
    const b = samples[i + 1]!;
    const u = fi - i;
    return {
      x: a.x + (b.x - a.x) * u,
      z: a.z + (b.z - a.z) * u,
      heading: a.heading,
      speed: a.speed + (b.speed - a.speed) * u,
      t,
    };
  };
}

/** Observe an agent's current state on each query and constant-velocity
 *  extrapolate (for players / adversarial NPCs that publish no plan).
 *  `smoothing` ∈ [0,1] EMA-blends the observed speed. */
export function fromObservations(
  getCurrentState: () => VehicleState,
  opts: { horizon?: number; smoothing?: number } = {},
): Predict<VehicleState> {
  const horizon = opts.horizon ?? DEFAULT_HORIZON;
  const alpha = opts.smoothing ?? 1;
  let smoothedSpeed: number | null = null;
  return (t) => {
    const cur = getCurrentState();
    smoothedSpeed =
      smoothedSpeed === null ? cur.speed : alpha * cur.speed + (1 - alpha) * smoothedSpeed;
    const dt = t - cur.t;
    if (dt < 0 || dt > horizon) return null;
    const v = smoothedSpeed;
    return {
      x: cur.x + v * Math.cos(cur.heading) * dt,
      z: cur.z + v * Math.sin(cur.heading) * dt,
      heading: cur.heading,
      speed: v,
      t,
    };
  };
}

/** A constant-velocity point predictor for circular moving obstacles. */
export function linearObstacle(
  x: number,
  z: number,
  vx: number,
  vz: number,
  radius: number,
  t0 = 0,
  horizon = DEFAULT_HORIZON,
): MovingObstacle {
  return {
    radius,
    predict: (t) => {
      const dt = t - t0;
      if (dt < 0 || dt > horizon) return null;
      return { x: x + vx * dt, z: z + vz * dt };
    },
  };
}

/** Adapt any {x,z}-bearing predictor into a circular moving obstacle. */
export function asObstacle<T extends { x: number; z: number }>(
  predict: Predict<T>,
  radius: number,
): MovingObstacle {
  return {
    radius,
    predict: (t) => {
      const p = predict(t);
      return p ? { x: p.x, z: p.z } : null;
    },
  };
}
