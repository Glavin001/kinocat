// Always-on planner instrumentation. Counters are plain integer fields on a
// mutable record and update on every relevant event (collisions, dominance
// hits, predict() calls, …) at negligible cost (~1% in default mode). Wall-
// clock timing is opt-in via `PlannerOptions.profile = 'timings'` — only then
// does `performance.now()` get sampled at the top-level brackets that the
// planner cares about (env.succ, env.heuristic from outside succ, path
// reconstruct). Sub-component breakdowns (collision time, dominance time,
// …) are inferable from `counters` plus the bracketed totals — we
// deliberately do NOT bracket per-collision because that would dominate the
// thing we're measuring.
//
// Environments hold a `rec: PerfRecorder` field (default `NULL_RECORDER`);
// `plan()` attaches a fresh recorder via `env.attachRecorder?.()` before the
// search and the recorder's counters become `result.stats.counters`. A null
// recorder is a non-null shared singleton whose `mode === 'off'` flag lets
// hot paths branch-predict the increment as a no-op write to a throwaway
// struct. The only invariant: never expose `NULL_RECORDER.counters` to user
// code (it accumulates garbage); always pass the per-search recorder's own
// counters through `result.stats`.

export type PerfMode = 'off' | 'counts' | 'timings';

/** Per-search integer event counters. Always populated (see top-of-file). */
export interface PlanCounters {
  /** env.succ() invocations. */
  succCalls: number;
  /** Total successor nodes returned across all succ calls. */
  successorsTotal: number;
  /** Successors pruned by branch-and-bound (f > Ω). */
  rejectedByOmega: number;
  /** Successors pruned by exact-hash dedup (already known better g). */
  rejectedByExact: number;
  /** Successors pruned by coarse-level dominance. */
  rejectedByDominance: number;
  /** DominanceTable.best() lookups (coarse passes only). */
  domLookups: number;
  /** DominanceTable.relax() calls that improved the front. */
  domRelaxes: number;
  /** env.heuristic() calls (inside or outside succ). */
  heuristicCalls: number;
  /** env.reachedGoalRegion() calls. */
  goalChecks: number;
  /** Static-collision queries (world.clear / footprintClear / segmentClear). */
  collisionChecks: number;
  /** Collision queries that returned "occupied". */
  collisionRejects: number;
  /** Moving-obstacle predict() calls (time-aware envs). */
  predictCalls: number;
  /** Moving-obstacle broadphase cheap-rejects. */
  broadphaseSkips: number;
  /** Per-primitive swept-AABB pre-check skipped the substep narrowphase
   *  (entire primitive provably clear). Aircraft-env only. */
  primitiveSweptSkips: number;
  /** Analytic shot-to-goal attempted (aircraft-env / vehicle-env). */
  analyticShots: number;
  /** Analytic shot succeeded (emitted a goal-reaching successor). */
  analyticShotsClear: number;
  /** Wall-clock ms since plan() start at each improving incumbent. */
  improvementWallMs: number[];
}

/** Wall-clock ms per top-level bracket. Populated only when
 *  `PlannerOptions.profile === 'timings'`; undefined otherwise. */
export interface PlanTimings {
  total: number;
  /** All time inside env.succ() (includes its internal collision/heuristic). */
  succ: number;
  /** env.heuristic() calls from planner top-level (start-node h). */
  heuristic: number;
  /** Path reconstruction at each anytime improvement. */
  pathReconstruct: number;
}

/** Per-pass breakdown. One entry per pass (resolution level) actually run. */
export interface PassStats {
  level: number;
  expansions: number;
  generated: number;
  improvements: number;
  /** Wall-clock ms; only set when `profile === 'timings'`. */
  ms?: number;
}

/** The recorder threaded through planner + environments. Mutable by design. */
export interface PerfRecorder {
  mode: PerfMode;
  /** `mode === 'timings'`. Hot loops branch on this for `performance.now()`. */
  timingsOn: boolean;
  counters: PlanCounters;
  /** Only updated when `timingsOn`. Undefined otherwise (allocated lazily). */
  timings: PlanTimings;
}

export function makeCounters(): PlanCounters {
  return {
    succCalls: 0,
    successorsTotal: 0,
    rejectedByOmega: 0,
    rejectedByExact: 0,
    rejectedByDominance: 0,
    domLookups: 0,
    domRelaxes: 0,
    heuristicCalls: 0,
    goalChecks: 0,
    collisionChecks: 0,
    collisionRejects: 0,
    predictCalls: 0,
    broadphaseSkips: 0,
    primitiveSweptSkips: 0,
    analyticShots: 0,
    analyticShotsClear: 0,
    improvementWallMs: [],
  };
}

export function makeTimings(): PlanTimings {
  return { total: 0, succ: 0, heuristic: 0, pathReconstruct: 0 };
}

export function makeRecorder(mode: PerfMode): PerfRecorder {
  return {
    mode,
    timingsOn: mode === 'timings',
    counters: makeCounters(),
    timings: makeTimings(),
  };
}

/** Shared no-op sink. Environments hold this when no plan() is in flight.
 *  Its counters accumulate garbage and must never be read by user code. */
export const NULL_RECORDER: PerfRecorder = makeRecorder('off');

/** Human-readable one-line breakdown for the bench / console. */
export function formatPerf(
  stats: {
    expansions: number;
    generated: number;
    passesRun: number;
    improvements: number;
    counters: PlanCounters;
    timings?: PlanTimings;
    perPass: PassStats[];
  },
): string {
  const c = stats.counters;
  const t = stats.timings;
  const lines: string[] = [];
  lines.push(
    `expansions=${stats.expansions}  generated=${stats.generated}  ` +
      `passes=${stats.passesRun}  improvements=${stats.improvements}`,
  );
  lines.push(
    `succ.calls=${c.succCalls}  succ.total=${c.successorsTotal}  ` +
      `rej.omega=${c.rejectedByOmega}  rej.exact=${c.rejectedByExact}  ` +
      `rej.dom=${c.rejectedByDominance}`,
  );
  lines.push(
    `dom.lookups=${c.domLookups}  dom.relaxes=${c.domRelaxes}  ` +
      `h.calls=${c.heuristicCalls}  goal.checks=${c.goalChecks}`,
  );
  lines.push(
    `collision.checks=${c.collisionChecks}  collision.rejects=${c.collisionRejects}  ` +
      `predict.calls=${c.predictCalls}  broadphase.skips=${c.broadphaseSkips}  ` +
      `prim.swept.skips=${c.primitiveSweptSkips}`,
  );
  if (c.analyticShots > 0) {
    lines.push(
      `analytic.shots=${c.analyticShots}  analytic.shots.clear=${c.analyticShotsClear}  ` +
        `hit.rate=${((100 * c.analyticShotsClear) / c.analyticShots).toFixed(1)}%`,
    );
  }
  if (stats.expansions > 0) {
    const perExp = (x: number) => (x / stats.expansions).toFixed(2);
    lines.push(
      `per-expansion:  collisions=${perExp(c.collisionChecks)}  ` +
        `successors=${perExp(c.successorsTotal)}  ` +
        `predicts=${perExp(c.predictCalls)}`,
    );
  }
  if (t) {
    lines.push(
      `timings(ms): total=${t.total.toFixed(2)}  succ=${t.succ.toFixed(2)} ` +
        `(${((100 * t.succ) / Math.max(t.total, 1e-9)).toFixed(1)}%)  ` +
        `h=${t.heuristic.toFixed(2)}  reconstruct=${t.pathReconstruct.toFixed(2)}`,
    );
    if (stats.expansions > 0) {
      lines.push(
        `        per-expansion: ${(t.total / stats.expansions).toFixed(3)} ms total, ` +
          `${(t.succ / stats.expansions).toFixed(3)} ms succ`,
      );
    }
  }
  if (stats.perPass.length > 0) {
    lines.push('per-pass:');
    for (const p of stats.perPass) {
      lines.push(
        `  L${p.level}: expansions=${p.expansions}  generated=${p.generated}  ` +
          `improvements=${p.improvements}` +
          (p.ms !== undefined ? `  ms=${p.ms.toFixed(2)}` : ''),
      );
    }
  }
  if (c.improvementWallMs.length > 0) {
    lines.push(
      `improvement.timeline.ms = [${c.improvementWallMs
        .map((x) => x.toFixed(1))
        .join(', ')}]`,
    );
  }
  return lines.join('\n');
}
