// EvalProbe — the backbone that makes any demo scenario observable. It wraps
// the existing framework-free `SimMonitor` AND the pure `kinocat/eval` metrics,
// fed one per-tick status sample, and emits a compact `EvalSnapshot` the HUD
// renders. It computes NOTHING of its own: every number comes from the same
// functions the CLI harness (`pnpm run eval`) and the Vitest tests use, so the
// numbers on screen are exactly the numbers the tests assert on.
//
// It is strictly read-only / no-feedback (like SimMonitor), so attaching it can
// never perturb determinism.

import type { CarKinematicState } from 'kinocat/agent';
import {
  toReferenceTrajectory,
  projectOntoPath,
  ggUtilization,
  comfortFlags,
  checkFeasibility,
  diagnose,
  DEFAULT_DIAGNOSIS_THRESHOLDS,
  type DynamicLimits,
  type ReferenceTrajectory,
  type Verdict,
} from 'kinocat/eval';
import { smoothTrajectory } from 'kinocat/execute';
import { createSimMonitor, type SimMonitor, type MonitorSample, type RunReport } from './sim-monitor';

/** Per-tick input — structurally `MonitorSample` but with the richer
 *  `CarKinematicState[]` plan (carrying speed) the eval metrics need. Any
 *  scenario's `status()` row satisfies this. */
export interface EvalProbeSample extends Omit<MonitorSample, 'plan'> {
  plan: CarKinematicState[] | null;
}

export interface EvalProbeConfig {
  /** Body-local footprint polygon (for SimMonitor clearance, if obstacles set). */
  footprint: ReadonlyArray<readonly [number, number]>;
  /** Fixed tick dt (s). */
  dt: number;
  /** The car's true dynamic limits — plan feasibility & g-g are judged here. */
  limits: DynamicLimits;
  /** Static obstacle polygons (XZ) for clearance; empty ⇒ clearance Infinity. */
  obstacles?: ReadonlyArray<ReadonlyArray<readonly [number, number]>>;
  /** Optional goal pose for SimMonitor terminal/progress diagnostics. */
  goal?: { x: number; z: number; heading: number };
  /** Rolling window (ticks) for the live comfort flag. Default 120 (~2 s @60Hz). */
  comfortWindow?: number;
}

/** The compact, render-ready metric bundle. */
export interface EvalSnapshot {
  // --- controller fidelity: the gap to the committed plan ---
  crossTrackNow: number;
  crossTrackRmse: number;
  crossTrackMax: number;
  headingErrNow: number;
  // --- capability utilization (g-g / friction circle) ---
  ggNow: { aLong: number; aLat: number; util: number };
  planMeanUtil: number;
  planPeakUtil: number;
  frictionLimit: number;
  // --- plan feasibility (the committed plan, smoothed as tracked) ---
  planFeasible: boolean;
  planWorstRatio: number;
  // --- executed comfort (rolling window) ---
  comfortable: boolean;
  comfortViolations: string[];
  // --- the diagnosis 2×2 ---
  verdict: Verdict;
  // --- safety / jitter / replan (from SimMonitor) ---
  report: RunReport;
}

export interface EvalProbe {
  sample(s: EvalProbeSample): void;
  snapshot(): EvalSnapshot;
  monitor: SimMonitor;
}

export function createEvalProbe(cfg: EvalProbeConfig): EvalProbe {
  const monitor = createSimMonitor({
    footprint: cfg.footprint,
    obstacles: cfg.obstacles ?? [],
    dt: cfg.dt,
    goal: cfg.goal,
  });
  const comfortWindow = cfg.comfortWindow ?? 120;

  // Rolling controller-fidelity accumulators (vs the live committed plan).
  let ctSumSq = 0;
  let ctCount = 0;
  let ctMax = 0;
  let crossTrackNow = 0;
  let headingErrNow = 0;

  // Live friction-circle point (needs accel = Δspeed/dt, latAccel = speed·yawRate).
  let prev: { speed: number; heading: number } | null = null;
  let ggNow = { aLong: 0, aLat: 0, util: 0 };

  // Plan-quality cache — recomputed only when the committed plan changes (the
  // raw plan is smoothed first so the friction-circle reading isn't poisoned by
  // Menger curvature spikes on the unevenly-sampled lifted polyline).
  let lastPlanRef: CarKinematicState[] | null = null;
  let projRef: ReferenceTrajectory = [];
  let planMeanUtil = 0;
  let planPeakUtil = 0;
  let planFeasible = true;
  let planWorstRatio = 0;

  const recent: CarKinematicState[] = [];

  function refreshPlanQuality(plan: CarKinematicState[]): void {
    projRef = toReferenceTrajectory(plan);
    const tracked = plan.length >= 6 ? smoothTrajectory(plan, { sampleSpacing: 0.5, iterations: 8 }) : plan;
    const ref = toReferenceTrajectory(tracked);
    const gg = ggUtilization(ref, cfg.limits.frictionLimit);
    const feas = checkFeasibility(ref, cfg.limits);
    planMeanUtil = gg.meanUtil;
    planPeakUtil = gg.peakUtil;
    planFeasible = feas.feasible;
    planWorstRatio = feas.worstRatio;
  }

  function sample(s: EvalProbeSample): void {
    monitor.sample(s as MonitorSample);

    const st = s.state;
    // Live g-g point.
    if (prev) {
      const aLong = (st.speed - prev.speed) / cfg.dt;
      let dh = st.heading - prev.heading;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const aLat = st.speed * (dh / cfg.dt);
      ggNow = { aLong, aLat, util: Math.hypot(aLong, aLat) / Math.max(cfg.limits.frictionLimit, 1e-6) };
    }
    prev = { speed: st.speed, heading: st.heading };

    // Rolling comfort window (needs t; synthesize from tick index if absent).
    recent.push({ x: st.x, z: st.z, heading: st.heading, speed: st.speed, t: recent.length * cfg.dt });
    if (recent.length > comfortWindow) recent.shift();

    // Controller fidelity vs the committed plan.
    if (s.plan && s.plan.length >= 2) {
      if (s.plan !== lastPlanRef) {
        lastPlanRef = s.plan;
        refreshPlanQuality(s.plan);
      }
      const proj = projectOntoPath(projRef, st.x, st.z);
      crossTrackNow = Math.abs(proj.crossTrack);
      let he = st.heading - proj.psiAtFoot;
      while (he > Math.PI) he -= 2 * Math.PI;
      while (he < -Math.PI) he += 2 * Math.PI;
      headingErrNow = Math.abs(he);
      ctSumSq += crossTrackNow * crossTrackNow;
      ctCount++;
      if (crossTrackNow > ctMax) ctMax = crossTrackNow;
    }
  }

  function snapshot(): EvalSnapshot {
    const report = monitor.summary();
    const comfort = comfortFlags(recent, cfg.dt);
    const crossTrackRmse = ctCount > 0 ? Math.sqrt(ctSumSq / ctCount) : 0;

    // Diagnosis: reuse the 2×2 with the live plan-quality + rolling tracking.
    const verdict = diagnose(
      {
        feasibility: {
          feasible: planFeasible,
          violations: [],
          counts: { 'lateral-accel': 0, 'turn-radius': 0, 'longitudinal-accel': 0, 'curvature-rate': 0 },
          worstRatio: planWorstRatio,
        },
        gg: { meanUtil: planMeanUtil, peakUtil: planPeakUtil, cloud: [] },
        terminal: null,
        pathLength: 0,
        timeToGoal: 0,
      },
      {
        crossTrack: { rmse: crossTrackRmse, max: ctMax, p95: ctMax },
        heading: { rmse: 0, max: 0 },
        velocity: { rmse: 0, max: 0 },
        steerRateRms: report.steerRateRms,
        steerReversals: report.steerReversals,
        samples: ctCount,
      },
      DEFAULT_DIAGNOSIS_THRESHOLDS,
    ).verdict;

    return {
      crossTrackNow,
      crossTrackRmse,
      crossTrackMax: ctMax,
      headingErrNow,
      ggNow,
      planMeanUtil,
      planPeakUtil,
      frictionLimit: cfg.limits.frictionLimit,
      planFeasible,
      planWorstRatio,
      comfortable: comfort.comfortable,
      comfortViolations: comfort.violations,
      verdict,
      report,
    };
  }

  return { sample, snapshot, monitor };
}
