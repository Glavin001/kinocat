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
 *  Rapier. Returns null on a discarded trial (off-arena / spin etc). */
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
    sampleEveryNTicks: ticks, // single end-sample is enough
    id: `gt-${startSpeed}-${controls.join('_')}`,
  });
  if (!result.ok) return null;
  const last = result.trial.samples[result.trial.samples.length - 1];
  if (!last) return null;
  // Trial frame: spawn at (0,0) heading 0. Samples are in world frame
  // but the harness teleports to origin/heading-0 → world = local.
  return { dx: last.x, dz: last.z };
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

export interface PrimitiveComparisonRow {
  label: string;
  fullErrM: number;
  paraErrM: number;
  ensSigmaPos: number;
  gate: boolean;
}

export interface PrimitiveComparisonStats {
  startSpeed: number;
  count: number;
  /** Full learned-model endpoint RMS vs Rapier ground truth (m). */
  fullRmsM: number;
  /** Parametric-only (residual-stripped) endpoint RMS vs Rapier (m). */
  paraRmsM: number;
  /** Percentage error reduction the residual buys over parametric.
   *  Negative ⇒ the residual is net-HARMFUL in this bucket. */
  residualHelpPct: number;
  /** How many primitives the OOD gate would fall back to parametric on. */
  gateFires: number;
  /** How many primitives the residual actually moves off the parametric path. */
  residualActive: number;
  /** Worst offenders by full-model endpoint error, descending. */
  worst: PrimitiveComparisonRow[];
}

function controlLabel(c: ReadonlyArray<number>): string {
  const steer = c[0] ?? 0;
  const drive = c[1] ?? 0;
  const brake = c[2] ?? 0;
  const dir = Math.abs(steer) < 1e-6 ? 'straight' : steer > 0 ? 'left' : 'right';
  const eff = brake > 0 ? 'brake' : drive < 0 ? 'reverse' : drive > 0 ? 'drive' : 'coast';
  return `${eff}-${dir}`;
}

/** Quantify, for the selected start-speed bucket, how the parametric floor and
 *  the full learned model each diverge from the Rapier ground-truth endpoints,
 *  plus where the OOD gate fires. `groundTruth[i].index` indexes into `full`;
 *  `parametric` must be the same control set in the same order. */
export function computePrimitiveComparisonStats(opts: {
  full: ReadonlyArray<MotionPrimitive>;
  parametric: ReadonlyArray<MotionPrimitive>;
  groundTruth: ReadonlyArray<GroundTruthDot>;
  model: LearnedVehicleModel;
  startSpeed: number;
  perStepDt?: number;
  topN?: number;
}): PrimitiveComparisonStats | null {
  if (opts.groundTruth.length === 0) return null;
  const dt = opts.perStepDt ?? opts.model.residualReferenceDt;
  const hasEnsemble = opts.model.residualEnsemble.length > 0;
  const rows: PrimitiveComparisonRow[] = [];
  let fullSq = 0;
  let paraSq = 0;
  let gateFires = 0;
  let residualActive = 0;
  for (const g of opts.groundTruth) {
    const full = opts.full[g.index];
    const para = opts.parametric[g.index];
    if (!full || !para) continue;
    const fullErr = Math.hypot(full.end.dx - g.dx, full.end.dz - g.dz);
    const paraErr = Math.hypot(para.end.dx - g.dx, para.end.dz - g.dz);
    fullSq += fullErr * fullErr;
    paraSq += paraErr * paraErr;
    const state: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: opts.startSpeed, t: 0, yawRate: 0, lateralVelocity: 0,
    };
    const pred = hasEnsemble
      ? predictWithUncertainty(opts.model, state, full.controls, dt)
      : { std: [0, 0, 0, 0, 0, 0] };
    const gate = pred.std.some((s, i) => s > (OOD_STD_THRESHOLD[i] ?? Infinity));
    if (gate) gateFires++;
    const moved = Math.hypot(full.end.dx - para.end.dx, full.end.dz - para.end.dz) > 1e-4;
    if (moved) residualActive++;
    rows.push({
      label: controlLabel(full.controls),
      fullErrM: fullErr,
      paraErrM: paraErr,
      ensSigmaPos: Math.hypot(pred.std[0] ?? 0, pred.std[1] ?? 0),
      gate,
    });
  }
  const n = rows.length;
  if (n === 0) return null;
  const fullRmsM = Math.sqrt(fullSq / n);
  const paraRmsM = Math.sqrt(paraSq / n);
  return {
    startSpeed: opts.startSpeed,
    count: n,
    fullRmsM,
    paraRmsM,
    residualHelpPct: paraRmsM > 0 ? (1 - fullRmsM / paraRmsM) * 100 : 0,
    gateFires,
    residualActive,
    worst: [...rows].sort((a, b) => b.fullErrM - a.fullErrM).slice(0, opts.topN ?? 5),
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
