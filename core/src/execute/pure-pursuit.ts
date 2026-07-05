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
  const ni = nearestIndex(path, current.x, current.z);

  // A plan whose final sample is (near) stopped encodes a real "stop here"
  // terminal, as opposed to a drive-through waypoint (e.g. a racing loop) whose
  // last point is just the current planning horizon.
  const stopsAtEnd = Math.abs(goal.speed) < 0.05;

  // Stop latch. A stop-terminated segment is done once the nearest sample is
  // its terminal — the vehicle has consumed the whole path it can see. Without
  // this, a purely distance-based `atGoal` lets a fast approach sail past the
  // goal, where the symmetric brake-distance term (`vGoal` grows again with
  // distance) re-commands cruise and the vehicle runs away. Keying on the
  // nearest-sample index — rather than a final-tangent half-plane test — keeps
  // it robust on curved parking approaches (a Reeds-Shepp hook bends, so a
  // half-plane test misfires metres early). Drive-through plans (terminal speed
  // > 0) are exempt so racing is unaffected.
  const reachedEnd = stopsAtEnd && ni >= path.length - 1;
  const atGoal = distToGoal <= config.goalTolerance || reachedEnd;
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

  // Optional path-speed cap: consume the plan's per-sample speed intent
  // through the BRAKING ENVELOPE, not a raw window-min. A future planned
  // speed v_i at arc distance d_i constrains the speed HERE only to
  // sqrt(v_i² + 2·maxDecel·d_i) — the fastest we can go now and still
  // brake down to v_i by the time we get there. The previous raw
  // window-min meant any near-zero sample (a coast-to-stop primitive, a
  // cusp, a plan terminal) inside the 14 m window pinned the target to
  // ~0 from far away — measured closed-loop: BOTH race cars crawled to
  // 0 laps in 100 s with the toggle on, which is why it was disabled.
  // With the envelope, distant slow-downs bind exactly when physics says
  // they should, so the planner's honest entry speeds become executable.
  // Anticipatory curvature braking: cap speed by UPCOMING path curvature
  // through the braking envelope. `vCurve` above only sees the chord to
  // the lookahead point, so a plan whose geometry runs straight into a
  // tight corner is entered at full speed and overshot (measured on the
  // race track: the v2 car's honest brake-then-turn plan blew through
  // the 90° corner ~10 m wide and triggered a 3 s failed-replan U-turn).
  // Pure geometry — works identically for any library.
  let vPreview = Infinity;
  if (config.previewCurvature && path.length >= 3) {
    // Look as far ahead as braking from the CURRENT speed requires —
    // previewing from cruise speed capped mild curves 50+ m out and
    // slowed clean laps by ~50% (measured).
    const v0 = Math.abs(current.speed);
    const previewHorizon = Math.max(
      config.lookaheadMax,
      (v0 * v0) / (2 * config.maxDecel),
    );
    let acc = 0;
    for (let i = Math.max(ni, 1); i < path.length - 1 && acc <= previewHorizon; i++) {
      acc += dist(path[i - 1]!.x, path[i - 1]!.z, path[i]!.x, path[i]!.z);
      // Menger curvature through samples i-1, i, i+1.
      const ax = path[i - 1]!.x, az = path[i - 1]!.z;
      const bx = path[i]!.x, bz = path[i]!.z;
      const cx = path[i + 1]!.x, cz = path[i + 1]!.z;
      const area2 = Math.abs((bx - ax) * (cz - az) - (bz - az) * (cx - ax));
      const ab = Math.hypot(bx - ax, bz - az);
      const bc = Math.hypot(cx - bx, cz - bz);
      const ca = Math.hypot(cx - ax, cz - az);
      const denom = ab * bc * ca;
      if (denom < 1e-9) continue;
      const kappaAt = (2 * area2) / denom;
      // Only GENUINE corners bind (R ≤ 12.5 m). Replanned chord paths
      // carry curvature noise on straights; previewing every wiggle
      // capped both cars well below their clean pace (measured).
      if (kappaAt < 0.08) continue;
      const aLatPreview = config.previewLateralAccel ?? config.maxLateralAccel;
      const vAtCorner2 = aLatPreview / kappaAt;
      const allowed = Math.sqrt(vAtCorner2 + 2 * config.maxDecel * acc);
      if (allowed < vPreview) vPreview = allowed;
    }
  }

  let vPath = Infinity;
  if (config.respectPathSpeed) {
    const window = config.lookaheadMax;
    // The nearest sample constrains at its true distance — but ONLY when
    // it lies AHEAD along the direction of travel. At d≈0 it is the
    // plan's echo of the current state (folding it in pins a stopped car
    // to 0 — never launches); when BEHIND, it caps the car to a speed it
    // already passed (race plans have metres between samples, so the
    // rest-speed spawn node kept the whole field crawling). Samples past
    // the nearest constrain through their braking envelope regardless.
    const dxs = path[ni]!.x - current.x;
    const dzs = path[ni]!.z - current.z;
    const aheadSigned =
      (dxs * Math.cos(current.heading) + dzs * Math.sin(current.heading)) * gear;
    const d0 = dist(current.x, current.z, path[ni]!.x, path[ni]!.z);
    if (aheadSigned > 0.05) {
      const s0 = Math.abs(path[ni]!.speed);
      const allowed0 = Math.sqrt(s0 * s0 + 2 * config.maxDecel * d0);
      if (allowed0 < vPath) vPath = allowed0;
    }
    let acc = aheadSigned > 0 ? d0 : 0;
    for (let i = ni; i < path.length - 1 && acc <= window; i++) {
      acc += dist(path[i]!.x, path[i]!.z, path[i + 1]!.x, path[i + 1]!.z);
      const s2 = Math.abs(path[i + 1]!.speed);
      const allowed = Math.sqrt(s2 * s2 + 2 * config.maxDecel * acc);
      if (allowed < vPath) vPath = allowed;
    }
    if (!Number.isFinite(vPath)) vPath = Infinity;
  }

  // Reverse cruise is capped by BOTH limits: the chassis's reverse
  // envelope AND the scenario cruise (a parking scenario cruising at
  // 2 m/s must not back in at the 6 m/s chassis reverse limit —
  // measured: 5.4 m/s reverse approaches parked 12° crooked).
  const cruise =
    gear < 0
      ? Math.min(config.cruiseSpeed, config.reverseCruiseSpeed ?? config.cruiseSpeed)
      : config.cruiseSpeed;
  const speedMag = atGoal
    ? 0
    : Math.min(
        cruise,
        vCurve,
        vPreview,
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
