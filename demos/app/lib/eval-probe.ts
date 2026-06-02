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
  comfortFlags,
  scorePlan,
  diagnose,
  DEFAULT_DIAGNOSIS_THRESHOLDS,
  type DynamicLimits,
  type ReferenceTrajectory,
  type PlanQualityReport,
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
  /** Nominal tick dt (s) — fallback when a per-sample dt isn't passed. */
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
  /** Live executed friction-circle point (a_long, a_lat) this tick. */
  ggNow: { aLong: number; aLat: number; util: number };
  /** Mean g-g utilization averaged across all scored plans this run (matches
   *  the harness's `meanUtil`; a stable read, not a single plan's peak). */
  planMeanUtil: number;
  /** Fraction of scored plans whose mean util exceeded the friction circle. */
  planOverEnvelopeFrac: number;
  frictionLimit: number;
  // --- plan feasibility (over all scored, stub-guarded plans) ---
  /** Fraction of scored plans that were dynamically feasible. */
  planFeasibleFrac: number;
  /** How many plans contributed to the rolling plan-quality stats. */
  plansScored: number;
  // --- executed comfort (rolling window) ---
  comfortable: boolean;
  comfortViolations: string[];
  // --- the diagnosis 2×2 ---
  verdict: Verdict;
  // --- safety / jitter / replan (from SimMonitor) ---
  report: RunReport;
}

export interface EvalProbe {
  /** `dtSample` is the real sim-time advanced by this tick (variable under a
   *  browser rAF loop); omit for fixed-step runs. */
  sample(s: EvalProbeSample, dtSample?: number): void;
  snapshot(): EvalSnapshot;
  monitor: SimMonitor;
}

// Plan-scoring guards — identical to demos/scripts/eval-harness.ts. Short
// off-track / near-degenerate plan stubs have meaningless curvature (v²·κ
// spikes) and must not pollute the utilization / feasibility stats.
const MIN_PLAN_SAMPLES = 10;
const MIN_SMOOTHED_SAMPLES = 6;
const MIN_PATH_LENGTH_M = 5;

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

  // Plan-quality accumulators — averaged across plans, like the harness, so a
  // single noisy plan can't dominate. Recomputed only when the committed plan
  // changes (dedupe by reference); stubs are skipped, never overwriting stats.
  let lastPlanRef: CarKinematicState[] | null = null;
  let projRef: ReferenceTrajectory = []; // raw geometry of the last GOOD plan, for cross-track
  let plansScored = 0;
  let feasiblePlans = 0;
  let overEnvPlans = 0;
  let utilSum = 0;
  let lastQuality: PlanQualityReport | null = null;

  const recent: CarKinematicState[] = [];

  /** Score a newly-committed plan, mirroring the harness exactly. Returns true
   *  if the plan was representative enough to count (so the caller knows the
   *  projection reference is fresh). */
  function scoreNewPlan(plan: CarKinematicState[]): boolean {
    if (plan.length < MIN_PLAN_SAMPLES) return false;
    const tracked = smoothTrajectory(plan, { sampleSpacing: 0.5, iterations: 8 });
    if (tracked.length < MIN_SMOOTHED_SAMPLES) return false;
    const q = scorePlan(tracked, cfg.limits);
    if (q.pathLength < MIN_PATH_LENGTH_M) return false;
    plansScored++;
    utilSum += q.gg.meanUtil;
    if (q.gg.meanUtil > 1) overEnvPlans++;
    if (q.feasibility.feasible) feasiblePlans++;
    lastQuality = q;
    projRef = toReferenceTrajectory(plan); // raw geometry is fine for projection
    return true;
  }

  function sample(s: EvalProbeSample, dtSample?: number): void {
    const sd = dtSample !== undefined && dtSample > 0 ? dtSample : cfg.dt;
    monitor.sample(s as MonitorSample, sd);

    const st = s.state;
    // Live g-g point (executed).
    if (prev) {
      const aLong = (st.speed - prev.speed) / sd;
      let dh = st.heading - prev.heading;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const aLat = st.speed * (dh / sd);
      ggNow = { aLong, aLat, util: Math.hypot(aLong, aLat) / Math.max(cfg.limits.frictionLimit, 1e-6) };
    }
    prev = { speed: st.speed, heading: st.heading };

    // Rolling comfort window (needs t; synthesize from accumulated dt).
    const tNow = (recent[recent.length - 1]?.t ?? 0) + sd;
    recent.push({ x: st.x, z: st.z, heading: st.heading, speed: st.speed, t: tNow });
    if (recent.length > comfortWindow) recent.shift();

    // On a new committed plan, score it (stub-guarded). Only refresh the
    // projection reference for representative plans.
    if (s.plan && s.plan !== lastPlanRef) {
      lastPlanRef = s.plan;
      scoreNewPlan(s.plan);
    }

    // Controller fidelity vs the last representative plan's geometry.
    if (projRef.length >= 2) {
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
    const planMeanUtil = plansScored > 0 ? utilSum / plansScored : 0;
    const planFeasibleFrac = plansScored > 0 ? feasiblePlans / plansScored : 1;

    // Diagnosis 2×2: feed the latest scored plan's structure but with the
    // run-aggregate scalars (majority feasibility + mean util) so the verdict
    // is stable rather than flickering on a single noisy plan.
    let verdict: Verdict = 'ok';
    if (lastQuality) {
      const aggPlan: PlanQualityReport = {
        ...lastQuality,
        feasibility: { ...lastQuality.feasibility, feasible: planFeasibleFrac >= 0.5 },
        gg: { ...lastQuality.gg, meanUtil: planMeanUtil },
      };
      verdict = diagnose(
        aggPlan,
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
    }

    return {
      crossTrackNow,
      crossTrackRmse,
      crossTrackMax: ctMax,
      headingErrNow,
      ggNow,
      planMeanUtil,
      planOverEnvelopeFrac: plansScored > 0 ? overEnvPlans / plansScored : 0,
      frictionLimit: cfg.limits.frictionLimit,
      planFeasibleFrac,
      plansScored,
      comfortable: comfort.comfortable,
      comfortViolations: comfort.violations,
      verdict,
      report,
    };
  }

  return { sample, snapshot, monitor };
}
