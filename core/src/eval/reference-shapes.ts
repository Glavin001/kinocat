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
