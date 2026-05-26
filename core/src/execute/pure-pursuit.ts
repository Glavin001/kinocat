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
  // The plan's terminal speed encodes intent: near-zero means the plan
  // asks the chassis to STOP at this pose (parking, terminal maneuver);
  // non-zero means DRIVE THROUGH (racing — the plan endpoint is just
  // where the planner's horizon stops, and a fresh replan will extend it
  // before the chassis arrives). The brake-to-goal cap and full-brake-on-
  // arrival only fire for stop-intent plans — applying them to drive-
  // through plans makes the chassis brake at every gate.
  const terminalIsStop = Math.abs(goal.speed) < 0.5;

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

  // vCurve: max speed allowed by lateral acceleration limit.
  // Default mode is reactive — uses the single curvature command from
  // the lookahead point above. When `lookaheadCurvature` is set, the
  // controller instead sweeps the plan polyline within `lookaheadMax`,
  // computing the brake-distance-aware speed required to decelerate
  // to each future corner's cornering limit. The lookahead mode is
  // strictly more cautious and matters most for fast plans into
  // sharp turns (racing); parking should leave it off so the
  // chassis doesn't crawl through every tight in-stall arc.
  const kMaxClamp = config.minTurnRadius ? 1 / config.minTurnRadius : Infinity;
  let vCurve = Math.sqrt(
    config.maxLateralAccel / Math.max(Math.min(Math.abs(kappa), kMaxClamp), 1e-3),
  );
  if (config.lookaheadCurvature) {
    let acc = 0;
    const window = config.lookaheadMax;
    for (let i = ni; i < path.length - 2 && acc < window; i++) {
      const a = path[i]!;
      const b = path[i + 1]!;
      const c2 = path[i + 2]!;
      const h1 = Math.atan2(b.z - a.z, b.x - a.x);
      const h2 = Math.atan2(c2.z - b.z, c2.x - b.x);
      let dh = h2 - h1;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const segLen = dist(a.x, a.z, b.x, b.z);
      const localKappa = Math.min(Math.abs(dh) / Math.max(segLen, 1e-6), kMaxClamp);
      const cornerV = Math.sqrt(config.maxLateralAccel / Math.max(localKappa, 1e-3));
      const vReach = Math.sqrt(cornerV * cornerV + 2 * config.maxDecel * acc);
      if (vReach < vCurve) vCurve = vReach;
      acc += segLen;
    }
  }
  // Brake-to-goal targets the plan's TERMINAL speed (planner intent),
  // not zero. Solves v² = v_term² + 2·a·d for the entry speed needed to
  // reach the plan endpoint at the planned terminal speed. Cases:
  //   parking (v_term=0): collapses to sqrt(2·a·d) — classic brake-to-
  //     stop.
  //   racing (v_term≈cruise): vGoal stays at or above cruise, so the
  //     cap doesn't fire — the chassis flows through the gate at race
  //     pace.
  //   slow gate (v_term=6): the cap kicks in within ~5–10 m of the
  //     gate, slowing the chassis to the planner's intended entry speed
  //     — the planner's per-primitive speed choice becomes actionable
  //     execution guidance instead of a number that gets ignored.
  const terminalSpeedMag = Math.abs(goal.speed);
  const brakeDist = Math.max(distToGoal - config.goalTolerance, 0);
  const vGoal = Math.sqrt(
    terminalSpeedMag * terminalSpeedMag + 2 * config.maxDecel * brakeDist,
  );

  // Path-speed cap with brake-distance awareness. For each forward
  // sample within the lookahead window, compute the speed I should be
  // at NOW so I can decelerate to the planned speed at that sample's
  // arc-distance: v_now² = v_plan² + 2·a·d. Take the minimum. This is
  // the standard friction-circle backward sweep, applied online.
  //
  // The old behaviour (raw min of plan speeds in window) made the
  // chassis brake to a HALT instantly when the plan contained any
  // [0,0,brake]-ending primitive, even 14 m away — chassis would
  // stall, the stall guard would teleport it to the next gate, and
  // the "race" turned into a series of 2 s teleport hops. The
  // brake-distance pass turns that into "slow gracefully to match
  // the planned speed BY the time you arrive", which is what the
  // smoothed speed profile means.
  let vPath = Infinity;
  if (config.respectPathSpeed) {
    // Skip the current sample (i=ni): plan[ni] is the chassis's
    // initial state — its planSpd reflects where the chassis IS, not
    // a future target. If the chassis is at rest (planSpd[ni]=0)
    // honouring it would brake to 0 and stall. Start from ni+1 so
    // we're asking "what speed should I be at NOW to match the
    // planner's FUTURE intent?". Samples below `minPathSpeed` are
    // ignored — used by racing scenarios to prevent the controller
    // from crawling along the planner's slow-start primitives when
    // it should be accelerating to cruise.
    const minPath = config.minPathSpeed ?? 0;
    let acc = (ni + 1 < path.length)
      ? dist(path[ni]!.x, path[ni]!.z, path[ni + 1]!.x, path[ni + 1]!.z)
      : 0;
    const window = config.lookaheadMax;
    for (let i = ni + 1; i < path.length - 1 && acc <= window; i++) {
      const planSpd = Math.abs(path[i]!.speed);
      if (planSpd >= minPath) {
        const vReach = Math.sqrt(planSpd * planSpd + 2 * config.maxDecel * acc);
        if (vReach < vPath) vPath = vReach;
      }
      acc += dist(path[i]!.x, path[i]!.z, path[i + 1]!.x, path[i + 1]!.z);
    }
    if (path.length - 1 > ni) {
      const lastPlanSpd = Math.abs(path[path.length - 1]!.speed);
      if (lastPlanSpd >= minPath) {
        const lastReach = Math.sqrt(lastPlanSpd * lastPlanSpd + 2 * config.maxDecel * acc);
        if (lastReach < vPath) vPath = lastReach;
      }
    }
    if (!Number.isFinite(vPath)) vPath = Infinity;
  }

  const stopAtGoal = atGoal && terminalIsStop;
  const speedMag = stopAtGoal
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
  if (stopAtGoal) {
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
