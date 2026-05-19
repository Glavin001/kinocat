import type { CurvePath, Pose } from './types';
import { segValue, stepUnit } from './internal';
import { wrapAngle } from '../internal/math';

/** Exact endpoint of following `path` from `start` with the given turning
 *  radius. Integrates per-segment in closed form (no subdivision). */
export function curveEndpoint(start: Pose, path: CurvePath, radius: number): Pose {
  let ux = 0;
  let uy = 0;
  let uyaw = start.theta;
  for (const s of path.segments) {
    ({ ux, uy, uyaw } = stepUnit(ux, uy, uyaw, s.steer, segValue(s, radius)));
  }
  return { x: ux * radius + start.x, y: uy * radius + start.y, theta: wrapAngle(uyaw) };
}

/** Sample poses along `path` at ~`step` world-distance spacing. The first
 *  pose is `start`; the last is the exact endpoint. */
export function sampleCurve(
  start: Pose,
  path: CurvePath,
  radius: number,
  step: number,
): Pose[] {
  const out: Pose[] = [{ x: start.x, y: start.y, theta: wrapAngle(start.theta) }];
  const dUnit = Math.max(step / radius, 1e-6);
  let ux = 0;
  let uy = 0;
  let uyaw = start.theta;
  for (const s of path.segments) {
    const value = segValue(s, radius);
    const sgn = value < 0 ? -1 : 1;
    let remaining = Math.abs(value);
    while (remaining > 1e-12) {
      const d = Math.min(dUnit, remaining);
      ({ ux, uy, uyaw } = stepUnit(ux, uy, uyaw, s.steer, sgn * d));
      remaining -= d;
      out.push({ x: ux * radius + start.x, y: uy * radius + start.y, theta: wrapAngle(uyaw) });
    }
  }
  return out;
}
