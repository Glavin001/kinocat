// Pure helpers for computing fan-plot ground-truth dots + ensemble
// uncertainty halos. Lives in `lib/` so it's unit-testable separately
// from the React component that consumes it.
//
// Strategy:
//   - For ground truth, we run the same control trace through the
//     Rapier headless harness and record the chassis position at the
//     primitive's duration. Cached per (speedBucket, controls) tuple.
//   - For uncertainty, we step the v2 model once with the primitive's
//     controls and take the per-output std reported by
//     `predictWithUncertainty`; halo radius = max(stdX, stdZ) at the
//     primitive's duration. We approximate the multi-step propagation
//     by scaling the per-tick std by sqrt(N) (random-walk lower bound)
//     — coarse but visually meaningful.

import type {
  LearnedVehicleModel,
  CarKinematicState,
  LearnableVehicleConfig,
} from 'kinocat/agent';
import { predictWithUncertainty } from 'kinocat/agent';
import type { MotionPrimitive } from 'kinocat/primitives';
import type { HeadlessTrialHarness } from 'kinocat/adapters/rapier';

export interface GroundTruthDot {
  index: number;
  dx: number;
  dz: number;
}

export interface UncertaintyHalo {
  index: number;
  radiusM: number;
}

/** Local-frame end position of running a single-control trial through
 *  Rapier. Returns null on a discarded trial (off-arena / spin etc).
 *
 *  IMPORTANT: the harness coasts for ~9 settle ticks (zero controls) after
 *  the teleport before it starts recording, so by the first recorded sample
 *  the chassis has already drifted forward by ~startSpeed × 0.15 s (up to
 *  several metres at race speeds). We therefore express the end pose in the
 *  frame of the POST-SETTLE start sample — not the world origin — so the
 *  reported displacement is the true motion under the control, free of the
 *  settle-coast offset. (Earlier this returned raw world coords, which added
 *  the coast distance to every error arrow and made the model look far more
 *  wrong than it is at speed.) */
function runHeadlessForControl(
  harness: HeadlessTrialHarness,
  startSpeed: number,
  controls: number[],
  durationSec: number,
): { dx: number; dz: number } | null {
  const PHYSICS_DT = 1 / 60;
  const ticks = Math.max(1, Math.round(durationSec / PHYSICS_DT));
  const trace = Array.from({ length: ticks }, () => ({
    steer: controls[0] ?? 0,
    driveForce: controls[1] ?? 0,
    brakeForce: controls[2] ?? 0,
  }));
  const result = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: startSpeed },
    controlsTrace: trace,
    sampleEveryNTicks: ticks, // post-settle start sample + end sample
    id: `gt-${startSpeed}-${controls.join('_')}`,
  });
  if (!result.ok) return null;
  const samples = result.trial.samples;
  const start = samples[0];
  const last = samples[samples.length - 1];
  if (!start || !last) return null;
  // Re-express the end in the post-settle start's local frame (origin at the
  // start sample, +x along its heading) so the settle coast cancels out.
  const ddx = last.x - start.x;
  const ddz = last.z - start.z;
  const h = start.heading;
  return {
    dx: ddx * Math.cos(h) + ddz * Math.sin(h),
    dz: -ddx * Math.sin(h) + ddz * Math.cos(h),
  };
}

export interface ComputeGroundTruthOpts {
  primitives: ReadonlyArray<MotionPrimitive>;
  /** Speed used to teleport-spawn the chassis for each GT trial.
   *  Required because the primitives' start-speed bucket isn't stored
   *  per-primitive — caller passes the active bucket. */
  startSpeed: number;
  /** Default 0.55 s — must match what the primitive library was built
   *  with. Pass the actual duration when libraries use per-bucket
   *  durations (e.g. v2 race library: 1.5/0.8/0.8/0.8). */
  duration: number;
  harness: HeadlessTrialHarness;
  /** Optional cache of already-computed GT per (startSpeed, controlsKey). */
  cache?: Map<string, { dx: number; dz: number }>;
}

export function computeGroundTruthDots(opts: ComputeGroundTruthOpts): GroundTruthDot[] {
  const out: GroundTruthDot[] = [];
  const cache = opts.cache;
  for (let i = 0; i < opts.primitives.length; i++) {
    const p = opts.primitives[i]!;
    // Skip primitives whose start-speed bucket doesn't match the
    // selected speed — the GT trial only makes sense at the right
    // initial speed. (Primitives in different buckets are mixed into
    // one library by `buildLearnedRaceLibraryV2`.)
    if (Math.abs(p.startSpeed - opts.startSpeed) > 0.5) continue;
    const key = `${opts.startSpeed}|${p.controls.join(',')}|${opts.duration.toFixed(3)}`;
    let gt = cache?.get(key);
    if (!gt) {
      const result = runHeadlessForControl(
        opts.harness,
        opts.startSpeed,
        p.controls,
        p.duration ?? opts.duration,
      );
      if (!result) continue;
      gt = result;
      cache?.set(key, gt);
    }
    out.push({ index: i, dx: gt.dx, dz: gt.dz });
  }
  return out;
}

// Default per-output-dim OOD thresholds (x, z, heading, speed, yawRate,
// lateralVelocity) — mirrors `DEFAULT_OOD_STD_THRESHOLD` in the core model
// (not re-exported from the agent barrel). Used only to report whether the
// runtime gate WOULD fall back to parametric for a given primitive.
const OOD_STD_THRESHOLD = [0.5, 0.5, 0.1, 1.0, 0.5, 0.5];

// Classification thresholds (kept here, surfaced verbatim in the UI legend so
// the viewer knows exactly what "accurate" means).
/** An action is "accurate" if the model endpoint lands within this absolute
 *  distance of Rapier, OR within ACCURATE_PCT of how far the chassis travelled
 *  (whichever is more forgiving). The percentage term keeps the bar fair as
 *  primitives get longer/faster. */
export const ACCURATE_ABS_M = 0.6;
export const ACCURATE_PCT = 0.08;

export type Verdict = 'accurate' | 'flagged' | 'confident-bias';

export interface XZ { dx: number; dz: number }

/** Everything the UI needs to render one action's row + map glyph. */
export interface ActionComparison {
  /** Index into the (full) primitive array for the bucket. */
  index: number;
  label: string;
  /** Endpoints in the start-local frame (chassis at origin facing +x). */
  full: XZ; // full learned model
  para: XZ; // parametric-only floor
  truth: XZ; // Rapier ground truth
  /** Full model swept path (start-local) for the map's selected overlay. */
  sweep: ReadonlyArray<{ x: number; z: number }>;
  /** Reverse gear (drawn differently on the map). */
  reverse: boolean;
  fullErrM: number; // |full − truth|
  paraErrM: number; // |para − truth|
  /** Straight-line distance origin→truth — "how far the chassis went". */
  travelM: number;
  /** fullErrM / max(travelM, 1) — error as a fraction of travel. */
  errFrac: number;
  /** Ensemble 1σ position spread at the first step (the OOD signal). */
  ensSigmaPos: number;
  /** Would the runtime OOD gate fall back to parametric here? */
  gate: boolean;
  /** Did the residual actually move the endpoint off the parametric path? */
  residualActive: boolean;
  /** (paraErr − fullErr) / paraErr × 100. >0 ⇒ residual pulled toward truth
   *  (helped); <0 ⇒ pushed away (hurt). */
  residualDeltaPct: number;
  verdict: Verdict;
}

export interface ActionComparisonSummary {
  startSpeed: number;
  count: number;
  /** Full / parametric endpoint RMS vs Rapier (m). */
  fullRmsM: number;
  paraRmsM: number;
  /** Net residual help across the bucket (%). Negative ⇒ net harmful. */
  residualHelpPct: number;
  accurate: number;
  flagged: number;
  confidentBias: number;
  /** Largest single full-model error — used to scale the scorecard bars. */
  maxErrM: number;
  /** All actions, sorted worst-first by full error. */
  actions: ActionComparison[];
}

function controlLabel(c: ReadonlyArray<number>): string {
  const steer = c[0] ?? 0;
  const drive = c[1] ?? 0;
  const brake = c[2] ?? 0;
  const dir = Math.abs(steer) < 1e-6 ? 'straight' : steer > 0 ? 'left' : 'right';
  const eff = brake > 0 ? 'brake' : drive < 0 ? 'reverse' : drive > 0 ? 'drive' : 'coast';
  return `${eff} · ${dir}`;
}

function classify(fullErrM: number, errFrac: number, gate: boolean): Verdict {
  const accurate = fullErrM < ACCURATE_ABS_M || errFrac < ACCURATE_PCT;
  if (accurate) return 'accurate';
  // Wrong, but the system KNOWS it's unsure → it falls back to the safe
  // parametric backbone. Honest, not dangerous.
  if (gate) return 'flagged';
  // Wrong AND the ensemble was confident → bias flows into the plan.
  return 'confident-bias';
}

/** Build the full per-action comparison (verdicts + endpoints + summary) for a
 *  speed bucket. `groundTruth[i].index` indexes into `full`; `parametric` must
 *  be the same control set in the same order. */
export function computeActionComparison(opts: {
  full: ReadonlyArray<MotionPrimitive>;
  parametric: ReadonlyArray<MotionPrimitive>;
  groundTruth: ReadonlyArray<GroundTruthDot>;
  model: LearnedVehicleModel;
  startSpeed: number;
  perStepDt?: number;
}): ActionComparisonSummary | null {
  if (opts.groundTruth.length === 0) return null;
  const dt = opts.perStepDt ?? opts.model.residualReferenceDt;
  const hasEnsemble = opts.model.residualEnsemble.length > 0;
  const actions: ActionComparison[] = [];
  let fullSq = 0;
  let paraSq = 0;
  for (const g of opts.groundTruth) {
    const full = opts.full[g.index];
    const para = opts.parametric[g.index];
    if (!full || !para) continue;
    const truth: XZ = { dx: g.dx, dz: g.dz };
    const fullErrM = Math.hypot(full.end.dx - truth.dx, full.end.dz - truth.dz);
    const paraErrM = Math.hypot(para.end.dx - truth.dx, para.end.dz - truth.dz);
    fullSq += fullErrM * fullErrM;
    paraSq += paraErrM * paraErrM;
    const travelM = Math.hypot(truth.dx, truth.dz);
    const errFrac = fullErrM / Math.max(travelM, 1);
    const state: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: opts.startSpeed, t: 0, yawRate: 0, lateralVelocity: 0,
    };
    const pred = hasEnsemble
      ? predictWithUncertainty(opts.model, state, full.controls, dt)
      : { std: [0, 0, 0, 0, 0, 0] };
    const gate = pred.std.some((s, i) => s > (OOD_STD_THRESHOLD[i] ?? Infinity));
    const residualActive = Math.hypot(full.end.dx - para.end.dx, full.end.dz - para.end.dz) > 1e-4;
    actions.push({
      index: g.index,
      label: controlLabel(full.controls),
      full: { dx: full.end.dx, dz: full.end.dz },
      para: { dx: para.end.dx, dz: para.end.dz },
      truth,
      sweep: full.sweep.map((s) => ({ x: s.x, z: s.z })),
      reverse: full.reverse,
      fullErrM,
      paraErrM,
      travelM,
      errFrac,
      ensSigmaPos: Math.hypot(pred.std[0] ?? 0, pred.std[1] ?? 0),
      gate,
      residualActive,
      residualDeltaPct: paraErrM > 1e-6 ? ((paraErrM - fullErrM) / paraErrM) * 100 : 0,
      verdict: classify(fullErrM, errFrac, gate),
    });
  }
  const n = actions.length;
  if (n === 0) return null;
  actions.sort((a, b) => b.fullErrM - a.fullErrM);
  const fullRmsM = Math.sqrt(fullSq / n);
  const paraRmsM = Math.sqrt(paraSq / n);
  return {
    startSpeed: opts.startSpeed,
    count: n,
    fullRmsM,
    paraRmsM,
    residualHelpPct: paraRmsM > 0 ? (1 - fullRmsM / paraRmsM) * 100 : 0,
    accurate: actions.filter((a) => a.verdict === 'accurate').length,
    flagged: actions.filter((a) => a.verdict === 'flagged').length,
    confidentBias: actions.filter((a) => a.verdict === 'confident-bias').length,
    maxErrM: Math.max(...actions.map((a) => a.fullErrM)),
    actions,
  };
}

export interface ComputeUncertaintyOpts {
  primitives: ReadonlyArray<MotionPrimitive>;
  model: LearnedVehicleModel;
  config: LearnableVehicleConfig;
  /** Same as fan plot: only primitives at this start speed bucket. */
  startSpeed: number;
  /** Substep used internally by `predictWithUncertainty` — typically
   *  the trained-residual reference dt. */
  perStepDt?: number;
}

/** Coarse ensemble-uncertainty estimate: per-tick std from
 *  `predictWithUncertainty` propagated to the primitive duration by a
 *  random-walk sqrt(N) heuristic. Acceptable as a "is this region OOD?"
 *  signal even if not a calibrated CI. */
export function computeUncertaintyHalos(opts: ComputeUncertaintyOpts): UncertaintyHalo[] {
  const dt = opts.perStepDt ?? opts.model.residualReferenceDt;
  // Empty ensemble → no useful uncertainty, skip.
  if (opts.model.residualEnsemble.length === 0) return [];
  const out: UncertaintyHalo[] = [];
  for (let i = 0; i < opts.primitives.length; i++) {
    const p = opts.primitives[i]!;
    if (Math.abs(p.startSpeed - opts.startSpeed) > 0.5) continue;
    const state: CarKinematicState = {
      x: 0, z: 0, heading: 0,
      speed: p.startSpeed, t: 0,
      yawRate: 0, lateralVelocity: 0,
    };
    const pred = predictWithUncertainty(opts.model, state, p.controls, dt);
    // pred.std = [x, z, heading, speed, yawRate, lateralVelocity].
    const perStepXZ = Math.hypot(pred.std[0] ?? 0, pred.std[1] ?? 0);
    const steps = Math.max(1, (p.duration ?? 0.55) / dt);
    const radius = perStepXZ * Math.sqrt(steps);
    if (radius > 0.05) out.push({ index: i, radiusM: radius });
  }
  return out;
}
