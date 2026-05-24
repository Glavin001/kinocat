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
  VehicleState,
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
    const state: VehicleState = {
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
