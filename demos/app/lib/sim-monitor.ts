// Telemetry + diagnostics monitor — the data analog of watching the screen.
//
// The North Star for headless testing: every phenomenon you would notice
// visually (the car driving the wrong way, stopping at the wrong heading,
// jittering, replanning in an oscillating loop) must show up in recorded
// numbers so a test can assert on it and a failure reads as obviously as the
// render would. This monitor samples a scenario's per-tick status (it never
// feeds anything back, so it cannot affect determinism) and produces:
//
//   • a full telemetry stream (pose, speed, controls, plan length) — dumpable
//     to JSON/CSV for offline plotting;
//   • a RunReport of scalar diagnostics grouped by the visible behaviour they
//     capture (safety invariants, direction/progress, parking success, jitter,
//     replan health).
//
// It is framework-free and Rapier-free: it reads only the structural fields
// below, so unit tests can feed it hand-built sample streams, and the web HUD
// can attach the same monitor to display the exact numbers the tests assert on.

import {
  placeFootprint,
  polygonsIntersect,
  polygonDistance,
  type Pt,
} from '../../../core/src/internal/geom';

export type { Pt } from '../../../core/src/internal/geom';

// ---------------------------------------------------------------------------
// Input shape. Structurally a subset of `RaceCarStatus` (race-scenario.ts), so
// `monitor.sample(scenario.status()[i])` type-checks directly — but declared
// locally so the monitor stays decoupled and trivially stubbable in unit tests.

export interface MonitorSample {
  state: { x: number; z: number; heading: number; speed: number };
  metrics: {
    liveControls: { steer: number; throttle: number; brake: number; targetSpeed: number };
  };
  diagnostics: {
    totalReplans: number;
    successfulReplans: number;
    consecutiveFailedReplans: number;
  };
  plan: ReadonlyArray<{ x: number; z: number; heading: number }> | null;
  loopIndex: number;
}

export interface MonitorGoal {
  x: number;
  z: number;
  heading: number;
}

export interface SuccessTolerances {
  /** Max terminal position error to count as parked (m). */
  posTol: number;
  /** Max terminal heading error to count as parked (rad). */
  headingTol: number;
  /** Max terminal |speed| to count as stopped (m/s). */
  speedTol: number;
}

export interface MonitorConfig {
  /** Body-local footprint polygon (heading 0 = +x), e.g. agent.footprint. */
  footprint: ReadonlyArray<Pt>;
  /** Static obstacle polygons in world XZ. Empty ⇒ clearance is Infinity. */
  obstacles: ReadonlyArray<ReadonlyArray<readonly [number, number]>>;
  /** Fixed tick dt (s) — used for finite-difference accel/jerk/steer-rate. */
  dt: number;
  /** Optional goal pose for progress + terminal-success diagnostics. */
  goal?: MonitorGoal;
  /** Tolerances for the `parkedOk` success flag. */
  success?: SuccessTolerances;
  /** Deadband (rad/s) below which a steering-rate change is treated as noise
   *  and does not count as a reversal. Defaults to 0.05. */
  steerRateDeadband?: number;
}

/** One recorded telemetry row — the full run is reconstructable from these. */
export interface TelemetryRow {
  t: number;
  x: number;
  z: number;
  heading: number;
  speed: number;
  steer: number;
  throttle: number;
  brake: number;
  targetSpeed: number;
  loopIndex: number;
  planLen: number;
  replanCount: number;
  /** Distance to goal this tick (NaN when no goal configured). */
  distToGoal: number;
}

/** Scalar summary — the golden record + the assertion surface for tests. */
export interface RunReport {
  ticks: number;
  durationSec: number;
  // --- safety invariants ---
  /** Min footprint-to-obstacle gap over the run (m); Infinity with no obstacles. */
  minClearance: number;
  /** True if the footprint ever overlapped an obstacle. */
  collided: boolean;
  /** Peak |longitudinal accel| (m/s^2). */
  maxAccel: number;
  /** Peak |jerk| (m/s^3). */
  maxJerk: number;
  /** Peak |speed| (m/s). */
  peakSpeed: number;
  // --- direction / wrong-way ---
  /** initialDist − finalDist to goal (m); negative ⇒ ended farther away. NaN w/o goal. */
  netProgress: number;
  /** Ticks where distance-to-goal increased beyond noise. */
  movedAwayFromGoalTicks: number;
  /** Largest backslide past the closest approach so far (m). */
  maxRetreat: number;
  /** Ticks where the velocity pointed >90° away from the goal bearing while moving. */
  awayHeadingTicks: number;
  // --- terminal / parking success ---
  terminalPosError: number;
  terminalHeadingError: number;
  terminalSpeed: number;
  /** Position AND heading AND stopped all within tolerance. NaN-safe false w/o goal. */
  parkedOk: boolean;
  // --- jitter ---
  /** Sign-changes of steering rate (deadbanded) — "jitters a lot". */
  steerReversals: number;
  /** RMS of steering rate (rad/s). */
  steerRateRms: number;
  /** RMS of lateral acceleration ≈ speed·yawRate (m/s^2). */
  lateralAccelRms: number;
  // --- replan health ---
  totalReplans: number;
  successfulReplans: number;
  replansPerSec: number;
  failedReplanRatio: number;
  consecutiveFailedReplansMax: number;
  /** Number of distinct committed plans observed (new plan references). */
  planUpdates: number;
  /** Times a new plan flipped the aim from left↔right vs the previous plan —
   *  the signature of an oscillating replanner. */
  planDirectionFlips: number;
}

export interface SimMonitor {
  /** Call once per tick AFTER scenario.tick(), with that car's status. */
  sample(s: MonitorSample): void;
  /** Finalize and read the scalar report (uses the last sample for terminals). */
  summary(): RunReport;
  /** The full recorded telemetry stream. */
  trajectory(): readonly TelemetryRow[];
}

const DEFAULT_SUCCESS: SuccessTolerances = { posTol: 0.5, headingTol: 0.26, speedTol: 0.5 };

/** Wrap an angle delta to [-pi, pi]. */
function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Signed aim of a plan relative to a heading: angle between the car heading
 *  and the bearing from the plan start to its first point ≳1.5 m ahead (or the
 *  last point for short plans). Returns 0 for unusable plans. */
function planAimSign(
  plan: ReadonlyArray<{ x: number; z: number }>,
  heading: number,
): number {
  if (plan.length < 2) return 0;
  const p0 = plan[0]!;
  let aim = plan[plan.length - 1]!;
  for (let i = 1; i < plan.length; i++) {
    const p = plan[i]!;
    if (Math.hypot(p.x - p0.x, p.z - p0.z) > 1.5) {
      aim = p;
      break;
    }
  }
  const bearing = Math.atan2(aim.z - p0.z, aim.x - p0.x);
  const rel = angleDiff(bearing, heading);
  if (rel > 1e-3) return 1;
  if (rel < -1e-3) return -1;
  return 0;
}

export function createSimMonitor(cfg: MonitorConfig): SimMonitor {
  const dt = cfg.dt;
  const success = cfg.success ?? DEFAULT_SUCCESS;
  const deadband = cfg.steerRateDeadband ?? 0.05;
  const hasObstacles = cfg.obstacles.length > 0;

  const rows: TelemetryRow[] = [];

  let peakSpeed = 0;
  let minClearance = Infinity;
  let collided = false;
  let maxAccel = 0;
  let maxJerk = 0;

  // direction / progress
  let initialDist = NaN;
  let minDistSoFar = Infinity;
  let maxRetreat = 0;
  let movedAwayTicks = 0;
  let awayHeadingTicks = 0;

  // jitter
  let steerReversals = 0;
  let steerRateSumSq = 0;
  let lateralAccelSumSq = 0;
  let diffCount = 0; // ticks with a valid finite-difference (i ≥ 1)

  // replan health
  let consecMax = 0;
  let planUpdates = 0;
  let planDirectionFlips = 0;

  // previous-tick state for finite differences
  let prev: MonitorSample | null = null;
  let prevAccel = NaN;
  let prevSteerRateSign = 0;
  let prevPlanRef: MonitorSample['plan'] = null;
  let prevPlanSign = 0;

  let last: MonitorSample | null = null;
  let ticks = 0;

  function sample(s: MonitorSample): void {
    const t = ticks * dt;
    const sp = Math.abs(s.state.speed);
    peakSpeed = Math.max(peakSpeed, sp);

    // --- clearance / collision ---
    let clearanceThisTick = Infinity;
    if (hasObstacles) {
      const fp = placeFootprint(cfg.footprint, s.state.x, s.state.z, s.state.heading);
      clearanceThisTick = Infinity;
      for (const obs of cfg.obstacles) {
        if (polygonsIntersect(fp, obs as ReadonlyArray<Pt>)) {
          collided = true;
          clearanceThisTick = 0;
          break;
        }
        const d = polygonDistance(fp, obs as ReadonlyArray<Pt>);
        if (d < clearanceThisTick) clearanceThisTick = d;
      }
      if (clearanceThisTick < minClearance) minClearance = clearanceThisTick;
    }

    // --- goal progress ---
    let distToGoal = NaN;
    if (cfg.goal) {
      distToGoal = Math.hypot(s.state.x - cfg.goal.x, s.state.z - cfg.goal.z);
      if (Number.isNaN(initialDist)) initialDist = distToGoal;
      if (prev) {
        const prevDist = Math.hypot(prev.state.x - cfg.goal.x, prev.state.z - cfg.goal.z);
        // Receding from the goal beyond a small noise band.
        if (distToGoal - prevDist > 1e-3) movedAwayTicks++;
        // Velocity direction vs bearing to goal, only when actually moving.
        const moved = Math.hypot(s.state.x - prev.state.x, s.state.z - prev.state.z);
        if (moved > 1e-4) {
          const travel = Math.atan2(s.state.z - prev.state.z, s.state.x - prev.state.x);
          const bearing = Math.atan2(cfg.goal.z - s.state.z, cfg.goal.x - s.state.x);
          if (Math.abs(angleDiff(travel, bearing)) > Math.PI / 2) awayHeadingTicks++;
        }
      }
      if (distToGoal < minDistSoFar) minDistSoFar = distToGoal;
      const retreat = distToGoal - minDistSoFar;
      if (retreat > maxRetreat) maxRetreat = retreat;
    }

    // --- finite differences: accel, jerk, steer rate, lateral accel ---
    if (prev) {
      const accel = (s.state.speed - prev.state.speed) / dt;
      // Skip the first finite-difference (settle/teleport from spawn).
      if (diffCount >= 1) {
        if (Math.abs(accel) > maxAccel) maxAccel = Math.abs(accel);
        if (Number.isFinite(prevAccel)) {
          const jerk = (accel - prevAccel) / dt;
          if (Math.abs(jerk) > maxJerk) maxJerk = Math.abs(jerk);
        }
      }
      prevAccel = accel;

      const steerRate = (s.metrics.liveControls.steer - prev.metrics.liveControls.steer) / dt;
      steerRateSumSq += steerRate * steerRate;
      const sign = steerRate > deadband ? 1 : steerRate < -deadband ? -1 : 0;
      if (sign !== 0) {
        if (prevSteerRateSign !== 0 && sign !== prevSteerRateSign) steerReversals++;
        prevSteerRateSign = sign;
      }

      const yawRate = angleDiff(s.state.heading, prev.state.heading) / dt;
      const latAccel = s.state.speed * yawRate;
      lateralAccelSumSq += latAccel * latAccel;

      diffCount++;
    }

    // --- replan health ---
    consecMax = Math.max(consecMax, s.diagnostics.consecutiveFailedReplans);
    if (s.plan && s.plan !== prevPlanRef) {
      planUpdates++;
      const sign = planAimSign(s.plan, s.state.heading);
      if (sign !== 0 && prevPlanSign !== 0 && sign !== prevPlanSign) planDirectionFlips++;
      if (sign !== 0) prevPlanSign = sign;
    }
    prevPlanRef = s.plan;

    rows.push({
      t,
      x: s.state.x,
      z: s.state.z,
      heading: s.state.heading,
      speed: s.state.speed,
      steer: s.metrics.liveControls.steer,
      throttle: s.metrics.liveControls.throttle,
      brake: s.metrics.liveControls.brake,
      targetSpeed: s.metrics.liveControls.targetSpeed,
      loopIndex: s.loopIndex,
      planLen: s.plan?.length ?? 0,
      replanCount: s.diagnostics.totalReplans,
      distToGoal,
    });

    prev = s;
    last = s;
    ticks++;
  }

  function summary(): RunReport {
    const durationSec = ticks * dt;
    const total = last?.diagnostics.totalReplans ?? 0;
    const ok = last?.diagnostics.successfulReplans ?? 0;

    let terminalPosError = NaN;
    let terminalHeadingError = NaN;
    let parkedOk = false;
    const terminalSpeed = last ? Math.abs(last.state.speed) : NaN;
    if (cfg.goal && last) {
      terminalPosError = Math.hypot(last.state.x - cfg.goal.x, last.state.z - cfg.goal.z);
      terminalHeadingError = Math.abs(angleDiff(last.state.heading, cfg.goal.heading));
      parkedOk =
        terminalPosError <= success.posTol &&
        terminalHeadingError <= success.headingTol &&
        terminalSpeed <= success.speedTol;
    }

    const netProgress = Number.isNaN(initialDist)
      ? NaN
      : initialDist - (Number.isFinite(terminalPosError) ? terminalPosError : initialDist);

    return {
      ticks,
      durationSec,
      minClearance,
      collided,
      maxAccel,
      maxJerk,
      peakSpeed,
      netProgress,
      movedAwayFromGoalTicks: movedAwayTicks,
      maxRetreat,
      awayHeadingTicks,
      terminalPosError,
      terminalHeadingError,
      terminalSpeed,
      parkedOk,
      steerReversals,
      steerRateRms: diffCount > 0 ? Math.sqrt(steerRateSumSq / diffCount) : 0,
      lateralAccelRms: diffCount > 0 ? Math.sqrt(lateralAccelSumSq / diffCount) : 0,
      totalReplans: total,
      successfulReplans: ok,
      replansPerSec: durationSec > 0 ? total / durationSec : 0,
      failedReplanRatio: total > 0 ? (total - ok) / total : 0,
      consecutiveFailedReplansMax: consecMax,
      planUpdates,
      planDirectionFlips,
    };
  }

  return {
    sample,
    summary,
    trajectory: () => rows,
  };
}

/** Human-readable one-block report — so a failing test prints what you'd see
 *  on screen. */
export function formatReport(r: RunReport): string {
  const f = (n: number, d = 2) => (Number.isFinite(n) ? n.toFixed(d) : '---');
  return [
    `ticks=${r.ticks} (${f(r.durationSec, 1)}s)`,
    `parkedOk=${r.parkedOk} posErr=${f(r.terminalPosError)}m hdgErr=${f(r.terminalHeadingError)}rad |v|=${f(r.terminalSpeed)}`,
    `progress: net=${f(r.netProgress)}m awayTicks=${r.movedAwayFromGoalTicks} maxRetreat=${f(r.maxRetreat)}m awayHeading=${r.awayHeadingTicks}`,
    `safety: minClear=${f(r.minClearance)}m collided=${r.collided} maxAccel=${f(r.maxAccel)} maxJerk=${f(r.maxJerk)} peakSpd=${f(r.peakSpeed)}`,
    `jitter: steerReversals=${r.steerReversals} steerRateRms=${f(r.steerRateRms)} latAccRms=${f(r.lateralAccelRms)}`,
    `replan: total=${r.totalReplans} ok=${r.successfulReplans} /s=${f(r.replansPerSec)} failRatio=${f(r.failedReplanRatio)} consecMax=${r.consecutiveFailedReplansMax} updates=${r.planUpdates} dirFlips=${r.planDirectionFlips}`,
  ].join('\n');
}
