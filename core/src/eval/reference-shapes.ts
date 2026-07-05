// Hand-authored ideal lines for the controller-isolation test (evaluation guide
// §4.1) and for unit tests. Each generator returns a `CarKinematicState[]` with
// a constant speed profile and headings set to the local path tangent, so it
// can be fed straight into `toReferenceTrajectory` and `runControllerIsolation`.

import type { CarKinematicState } from '../agent/types';

function build(
  points: ReadonlyArray<{ x: number; z: number }>,
  speed: number,
): CarKinematicState[] {
  const n = points.length;
  const out: CarKinematicState[] = new Array(n);
  let t = 0;
  for (let i = 0; i < n; i++) {
    const p = points[i]!;
    // Tangent heading from the forward (or backward at the end) difference.
    const a = points[Math.max(0, i - 1)]!;
    const b = points[Math.min(n - 1, i + 1)]!;
    const heading = Math.atan2(b.z - a.z, b.x - a.x);
    if (i > 0) {
      const prev = points[i - 1]!;
      const ds = Math.hypot(p.x - prev.x, p.z - prev.z);
      t += ds / Math.max(Math.abs(speed), 1e-3);
    }
    out[i] = { x: p.x, z: p.z, heading, speed, t };
  }
  return out;
}

/** A straight line of `length` m along +x, sampled every `ds` m. */
export function straightLine(opts: {
  length: number;
  speed: number;
  ds?: number;
}): CarKinematicState[] {
  const ds = opts.ds ?? 0.5;
  const n = Math.max(2, Math.round(opts.length / ds) + 1);
  const pts = Array.from({ length: n }, (_, i) => ({ x: (i / (n - 1)) * opts.length, z: 0 }));
  return build(pts, opts.speed);
}

/** A circular arc of the given `radius` sweeping `sweep` radians, starting at
 *  the origin heading +x (turning left for positive sweep). */
export function arcPath(opts: {
  radius: number;
  sweep: number;
  speed: number;
  ds?: number;
}): CarKinematicState[] {
  const ds = opts.ds ?? 0.5;
  const arcLen = Math.abs(opts.radius * opts.sweep);
  const n = Math.max(3, Math.round(arcLen / ds) + 1);
  const dir = Math.sign(opts.sweep) || 1;
  // Centre to the left of the start tangent (+x) at (0, +radius·dir).
  const cx = 0;
  const cz = opts.radius * dir;
  const pts = Array.from({ length: n }, (_, i) => {
    const theta = (i / (n - 1)) * opts.sweep;
    // Rotate the start point (0,0) about the centre by theta.
    const angle0 = Math.atan2(0 - cz, 0 - cx);
    const angle = angle0 + theta;
    return { x: cx + Math.abs(opts.radius) * Math.cos(angle), z: cz + Math.abs(opts.radius) * Math.sin(angle) };
  });
  return build(pts, opts.speed);
}

/** A single lane-change maneuver: a smooth sigmoid offset of `width` m over a
 *  longitudinal `length` m (the control literature's standard test). */
export function laneChange(opts: {
  width: number;
  length: number;
  speed: number;
  ds?: number;
}): CarKinematicState[] {
  const ds = opts.ds ?? 0.5;
  const n = Math.max(3, Math.round(opts.length / ds) + 1);
  const pts = Array.from({ length: n }, (_, i) => {
    const x = (i / (n - 1)) * opts.length;
    // Smoothstep from 0 to width over [0, length].
    const u = x / opts.length;
    const s = u * u * (3 - 2 * u);
    return { x, z: s * opts.width };
  });
  return build(pts, opts.speed);
}

/** A reverse-park maneuver: drive forward on a left arc, stop at a CUSP, then
 *  reverse on the opposite steer into the bay and stop. Built by integrating a
 *  kinematic bicycle so `heading` is the car's true orientation — on the reverse
 *  leg the car faces OPPOSITE its direction of travel, and `speed` is signed
 *  (negative in reverse). This exercises the whole reverse pipeline the forward
 *  shapes cannot: the gear-cusp curvature split, the reverse accel/decel
 *  classification, and (as an isolation reference) the terminal pose/heading/
 *  speed accuracy that catches stop-short and wrong-terminal-heading bugs.
 *  Parameters are chosen so the maneuver is comfortably feasible (lat accel
 *  V²/R, ramped longitudinal accel) for a typical parking budget. */
export function reversePark(opts?: {
  /** Cruise |speed| on each leg (m/s). */
  speed?: number;
  /** Turn radius on each leg (m). */
  radius?: number;
  /** Integration step (s). */
  dt?: number;
  /** Duration of each leg including ramps (s). */
  legSeconds?: number;
}): CarKinematicState[] {
  const V = opts?.speed ?? 2;
  // Default radius 6 m keeps curvature (≈1/6) comfortably inside a 4 m minimum
  // turn radius even after the Menger discretization noise the low-speed ramp
  // regions add — so the shape is a genuinely FEASIBLE known-good reference.
  const R = opts?.radius ?? 6;
  const dt = opts?.dt ?? 0.1;
  const legT = opts?.legSeconds ?? 3;
  const k = 1 / R;
  const ramp = Math.min(0.6, legT / 2); // s to ramp |speed| up / down

  const states: CarKinematicState[] = [];
  let x = 0;
  let z = 0;
  let heading = 0;
  let t = 0;
  states.push({ x, z, heading, speed: 0, t });

  // Trapezoidal signed-speed profile: ramp 0→dir·V, cruise, ramp back to 0.
  const speedAt = (tau: number, dir: number): number => {
    const up = Math.min(1, tau / ramp);
    const down = Math.min(1, (legT - tau) / ramp);
    return dir * V * Math.max(0, Math.min(up, down));
  };
  const integrateLeg = (dir: number, kk: number): void => {
    let tau = 0;
    while (tau < legT - 1e-9) {
      const v = speedAt(Math.min(legT, tau + dt), dir);
      heading += v * kk * dt;
      x += v * Math.cos(heading) * dt;
      z += v * Math.sin(heading) * dt;
      t += dt;
      tau += dt;
      states.push({ x, z, heading, speed: v, t });
    }
    // Pin the leg's last sample to rest — the cusp (and final stop) are v = 0.
    states[states.length - 1] = { ...states[states.length - 1]!, speed: 0 };
  };

  integrateLeg(+1, +k); // forward, steer left
  integrateLeg(-1, -k); // reverse, opposite steer → curls into the bay, stops
  return states;
}

/** A cone slalom: a sine weave of `amplitude` m with `cones` peaks spaced
 *  `spacing` m apart (the canonical fluidity / capability test). */
export function slalom(opts: {
  spacing: number;
  amplitude: number;
  cones: number;
  speed: number;
  ds?: number;
}): CarKinematicState[] {
  const ds = opts.ds ?? 0.5;
  const length = opts.spacing * opts.cones;
  const n = Math.max(3, Math.round(length / ds) + 1);
  const pts = Array.from({ length: n }, (_, i) => {
    const x = (i / (n - 1)) * length;
    const z = opts.amplitude * Math.sin((Math.PI * x) / opts.spacing);
    return { x, z };
  });
  return build(pts, opts.speed);
}
