// Pure-pursuit path tracker with curvature-aware speed. A single pure
// function evaluated every physics tick — no execution state machine, no
// driving/airborne/landing modes. Whatever physics does is just the start
// state of the next plan; replanning is the universal correction.

import type { CarKinematicState } from '../agent/types';
import type { PlanPath, PurePursuitConfig, TrackingCommand } from './types';
import { clamp, dist } from '../internal/math';

function nearestIndex(path: PlanPath, x: number, z: number): number {
  let bi = 0;
  let bd = Infinity;
  for (let i = 0; i < path.length; i++) {
    const d = dist(x, z, path[i]!.x, path[i]!.z);
    if (d < bd) {
      bd = d;
      bi = i;
    }
  }
  return bi;
}

/** Point on the polyline `lookahead` units ahead of `path[fromIdx]`. */
function lookaheadPoint(
  path: PlanPath,
  fromIdx: number,
  lookahead: number,
): { x: number; z: number; idx: number } {
  let acc = 0;
  for (let i = fromIdx; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    const seg = dist(a.x, a.z, b.x, b.z);
    if (acc + seg >= lookahead) {
      const u = seg > 1e-9 ? (lookahead - acc) / seg : 0;
      return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u, idx: i };
    }
    acc += seg;
  }
  const last = path[path.length - 1]!;
  return { x: last.x, z: last.z, idx: path.length - 1 };
}

/** Compute the tracking command for `current` following `path`. Pure. */
export function purePursuit(
  current: CarKinematicState,
  path: PlanPath,
  config: PurePursuitConfig,
): TrackingCommand {
  const goal = path[path.length - 1]!;
  const distToGoal = dist(current.x, current.z, goal.x, goal.z);
  const atGoal = distToGoal <= config.goalTolerance;

  const ni = nearestIndex(path, current.x, current.z);
  // gear from the planned speed sign just ahead on the path
  const aheadSpeed = path[Math.min(ni + 1, path.length - 1)]!.speed;
  const gear = aheadSpeed < 0 ? -1 : 1;

  const Ld = clamp(
    config.lookaheadMin + config.lookaheadGain * Math.abs(current.speed),
    config.lookaheadMin,
    config.lookaheadMax,
  );
  const lp = lookaheadPoint(path, ni, Ld);

  // lookahead in the (possibly reversed) body frame
  const he = gear < 0 ? current.heading + Math.PI : current.heading;
  const c = Math.cos(he);
  const s = Math.sin(he);
  const dx = lp.x - current.x;
  const dz = lp.z - current.z;
  const yV = -dx * s + dz * c; // lateral offset
  let kappa = (2 * yV) / (Ld * Ld);
  if (config.minTurnRadius) {
    const kMax = 1 / config.minTurnRadius;
    kappa = clamp(kappa, -kMax, kMax);
  }

  const vCurve = Math.sqrt(config.maxLateralAccel / Math.max(Math.abs(kappa), 1e-3));
  const vGoal = Math.sqrt(
    2 * config.maxDecel * Math.max(distToGoal - config.goalTolerance, 0),
  );
  const speedMag = atGoal
    ? 0
    : Math.min(config.cruiseSpeed, vCurve, Math.max(vGoal, config.lookaheadMin));
  const targetSpeed = gear * speedMag;

  let throttle = 0;
  let brake = 0;
  if (atGoal) {
    brake = 1;
  } else if (Math.abs(targetSpeed) > Math.abs(current.speed) + 1e-6) {
    throttle = clamp(
      (Math.abs(targetSpeed) - Math.abs(current.speed)) / config.maxAccel,
      0,
      1,
    );
  } else if (Math.abs(targetSpeed) < Math.abs(current.speed) - 1e-6) {
    brake = clamp(
      (Math.abs(current.speed) - Math.abs(targetSpeed)) / config.maxDecel,
      0,
      1,
    );
  }

  return {
    steering: kappa,
    throttle,
    brake,
    targetSpeed,
    lookahead: { x: lp.x, z: lp.z },
    atGoal,
  };
}
