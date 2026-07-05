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

/** A sampled pose plus the gear (forward/reverse) of the curve segment it
 *  lies on. The start pose adopts the gear of the first segment. */
export interface GearedPose extends Pose {
  /** True when this pose is on a reverse-gear (backing-up) segment. */
  reverse: boolean;
}

/** Like {@link sampleCurve} but tags every pose with the gear of the segment
 *  it belongs to. Used to reconstruct an executable trajectory through a
 *  Reeds-Shepp analytic shot: the controller needs the per-sample heading AND
 *  the forward/reverse sign so a back-in maneuver is driven in the right gear
 *  (a straight chord from the last grid node to the goal — which is all the
 *  bare node sequence carries — throws that information away). */
export function sampleCurveWithGear(
  start: Pose,
  path: CurvePath,
  radius: number,
  step: number,
): GearedPose[] {
  const firstReverse = path.segments[0] ? path.segments[0].gear < 0 : false;
  const out: GearedPose[] = [
    { x: start.x, y: start.y, theta: wrapAngle(start.theta), reverse: firstReverse },
  ];
  const dUnit = Math.max(step / radius, 1e-6);
  let ux = 0;
  let uy = 0;
  let uyaw = start.theta;
  for (const s of path.segments) {
    const value = segValue(s, radius);
    const sgn = value < 0 ? -1 : 1;
    const reverse = s.gear < 0;
    let remaining = Math.abs(value);
    while (remaining > 1e-12) {
      const d = Math.min(dUnit, remaining);
      ({ ux, uy, uyaw } = stepUnit(ux, uy, uyaw, s.steer, sgn * d));
      remaining -= d;
      out.push({ x: ux * radius + start.x, y: uy * radius + start.y, theta: wrapAngle(uyaw), reverse });
    }
  }
  return out;
}
