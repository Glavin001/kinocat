// Shared internals for the analytical-curve solvers. Math is ported from the
// canonical OMPL Reeds-Shepp / Dubins state spaces (BSD), kept in normalized
// unit-turning-radius space; callers scale by the real radius.

import type { CurvePath, CurveKind, CurveSegment, Pose, Steer } from './types';

export const TWO_PI = 2 * Math.PI;
export const PI = Math.PI;
export const HALF_PI = Math.PI / 2;
export const ZERO = 1e-9;

/** Wrap to (-pi, pi], matching OMPL's mod2pi. */
export function mod2pi(x: number): number {
  let v = x % TWO_PI;
  if (v < -PI) v += TWO_PI;
  else if (v > PI) v -= TWO_PI;
  return v;
}

export function polar(x: number, y: number): { r: number; theta: number } {
  return { r: Math.hypot(x, y), theta: Math.atan2(y, x) };
}

/** Internal segment letter; 'N' = no-op (skipped). */
export type SegType = Steer | 'N';

/** A normalized segment: `value` is signed (negative ⇒ reverse gear), in
 *  unit-radius space (radians for arcs, distance for straight). */
export interface NormSeg {
  steer: Steer;
  value: number;
}

/** Convert a path-type template + signed normalized values into a CurvePath. */
export function buildPath(
  kind: CurveKind,
  template: readonly SegType[],
  values: readonly number[],
  radius: number,
): CurvePath {
  const segments: CurveSegment[] = [];
  let word = '';
  let length = 0;
  for (let i = 0; i < template.length; i++) {
    const t = template[i]!;
    if (t === 'N') continue;
    const v = values[i] ?? 0;
    if (Math.abs(v) < ZERO) continue;
    const gear: 1 | -1 = v < 0 ? -1 : 1;
    const segLen = Math.abs(v) * radius;
    segments.push({ steer: t, gear, length: segLen });
    word += t;
    length += segLen;
  }
  return { kind, word, segments, length };
}

/** Apply one normalized step (signed `seg`, unit radius) to a unit-space pose
 *  whose heading is absolute. Mirrors OMPL's interpolate integration. */
export function stepUnit(
  ux: number,
  uy: number,
  uyaw: number,
  steer: Steer,
  seg: number,
): { ux: number; uy: number; uyaw: number } {
  if (steer === 'S') {
    return { ux: ux + seg * Math.cos(uyaw), uy: uy + seg * Math.sin(uyaw), uyaw };
  }
  if (steer === 'L') {
    return {
      ux: ux + Math.sin(uyaw + seg) - Math.sin(uyaw),
      uy: uy - Math.cos(uyaw + seg) + Math.cos(uyaw),
      uyaw: uyaw + seg,
    };
  }
  // 'R'
  return {
    ux: ux - Math.sin(uyaw - seg) + Math.sin(uyaw),
    uy: uy + Math.cos(uyaw - seg) - Math.cos(uyaw),
    uyaw: uyaw - seg,
  };
}

/** Signed normalized value of a public segment (inverse of buildPath). */
export function segValue(s: CurveSegment, radius: number): number {
  return (s.gear * s.length) / radius;
}

/** Transform `goal` into `start`'s frame, normalized by turning radius. */
export function toLocal(
  start: Pose,
  goal: Pose,
  radius: number,
): { x: number; y: number; phi: number } {
  const dx = goal.x - start.x;
  const dy = goal.y - start.y;
  const c = Math.cos(start.theta);
  const s = Math.sin(start.theta);
  return {
    x: (c * dx + s * dy) / radius,
    y: (-s * dx + c * dy) / radius,
    phi: goal.theta - start.theta,
  };
}
