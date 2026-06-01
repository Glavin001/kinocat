// Pure-pursuit path tracker with curvature-aware speed. A single pure
// function evaluated every physics tick — no execution state machine, no
// driving/airborne/landing modes. Whatever physics does is just the start
// state of the next plan; replanning is the universal correction.

import type { CarKinematicState } from '../agent/types';
import type { PlanPath, PurePursuitConfig, TrackingCommand } from './types';
import { clamp, dist, wrapAngle } from '../internal/math';

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

  // Optional Stanley-style heading-alignment term. Pure-pursuit chases a
  // lookahead POINT and ignores the path's HEADING, so on a tight terminal
  // straightening curve (e.g. a parking final approach) — where the lookahead
  // overshoots the short curve — it cuts the corner and comes to rest at the
  // approach angle instead of the planned terminal heading. Blending in a term
  // proportional to the error against the local path tangent makes the chassis
  // actively rotate onto the planned heading, the component pure-pursuit
  // structurally lacks. Self-gating: on a straight run the tangent equals the
  // chassis heading so the term is ~0; it only acts where the plan's heading
  // diverges from the chassis (the straighten). Forward gear only — the parking
  // terminal approach is forward, reverse maneuvers already arrive aligned, and
  // the curvature sign convention differs in reverse. Off (0) for racing.
  if (
    config.headingGain &&
    gear > 0 &&
    path.length >= 2 &&
    distToGoal <= (config.headingRadius ?? Infinity)
  ) {
    const tangent = path[Math.min(ni + 1, path.length - 1)]!.heading;
    kappa += config.headingGain * wrapAngle(tangent - current.heading);
    if (config.minTurnRadius) {
      const kMax = 1 / config.minTurnRadius;
      kappa = clamp(kappa, -kMax, kMax);
    }
  }

  const vCurve = Math.sqrt(config.maxLateralAccel / Math.max(Math.abs(kappa), 1e-3));
  const vGoal = Math.sqrt(
    2 * config.maxDecel * Math.max(distToGoal - config.goalTolerance, 0),
  );

  // Optional path-speed cap: when the upstream planner attached a
  // speed-profile-smoothed plan, the per-sample `speed` already encodes
  // curvature- and brake-distance-aware targets. Take the min of the
  // planned speeds in a forward window so the controller actually
  // consumes that information (without this, the smoother has no effect).
  let vPath = Infinity;
  if (config.respectPathSpeed) {
    let acc = 0;
    const window = config.lookaheadMax;
    for (let i = ni; i < path.length - 1 && acc <= window; i++) {
      const s2 = Math.abs(path[i]!.speed);
      if (s2 < vPath) vPath = s2;
      acc += dist(path[i]!.x, path[i]!.z, path[i + 1]!.x, path[i + 1]!.z);
    }
    const last = Math.abs(path[path.length - 1]!.speed);
    if (last < vPath) vPath = last;
    if (!Number.isFinite(vPath)) vPath = Infinity;
  }

  const speedMag = atGoal
    ? 0
    : Math.min(
        config.cruiseSpeed,
        vCurve,
        vPath,
        Math.max(vGoal, config.lookaheadMin),
      );
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
