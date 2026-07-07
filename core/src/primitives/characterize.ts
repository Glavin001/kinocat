import type { ForwardSim, LocalPose, MotionPrimitive } from './types';
import { MotionPrimitiveLibrary } from './library';
import type { CarKinematicState } from '../agent/types';
import { wrapAngle } from '../internal/math';

/** One rollout: a canonical start state paired with the control vector held
 *  for the primitive's whole duration. */
export interface CharacterizeRun<S> {
  startState: S;
  controls: number[];
}

export interface CharacterizeOptions<S, Sample> {
  forwardSim: ForwardSim<S>;
  /** Rollouts to perform, in order (order defines primitive identity — keep
   *  it stable across builds). Use `crossRuns` for the common
   *  startStates × controlSets grid. */
  runs: CharacterizeRun<S>[];
  /** Wall-clock duration of each primitive (seconds). */
  duration: number;
  /** Integration / sweep-sampling substeps per primitive. */
  substeps: number;
  /** Project a simulated state into the domain's local-frame sample record
   *  (collision-sweep pose, attitude, altitude delta, …). Pure; called once
   *  per substep. */
  record: (s: S) => Sample;
}

export interface CharacterizedPrimitive<S, Sample> {
  startState: S;
  controls: number[];
  duration: number;
  /** One record per substep (post-integration), in rollout order. Does NOT
   *  include the start pose — prepend it if the domain's sweep needs it. */
  samples: Sample[];
  /** Raw final state after the last substep (source of samples at the end);
   *  carries fields the Sample projection may drop (speed, gear, …). */
  endState: S;
}

/**
 * The shared rollout harness behind every primitive set: roll the supplied
 * ForwardSim through each run, recording a local-frame sample per substep.
 * Deterministic and physics-engine agnostic.
 *
 * CONTRACT — local-frame soundness: environments apply these primitives by
 * rigid-transforming the recorded samples by a node's world pose, instead of
 * re-simulating. That is only valid when the forward sim is translation- and
 * yaw-equivariant: its output must not depend on absolute position or
 * absolute heading (no global wind, no position-dependent surface grip). If
 * a learned or physics-backed sim gains such dependence, characterize at
 * plan time from the actual state instead of caching.
 */
export function characterize<S, Sample>(
  opts: CharacterizeOptions<S, Sample>,
): CharacterizedPrimitive<S, Sample>[] {
  const { forwardSim, runs, duration, substeps, record } = opts;
  const dt = duration / substeps;
  const out: CharacterizedPrimitive<S, Sample>[] = [];
  for (const run of runs) {
    let s = run.startState;
    const samples: Sample[] = [];
    for (let k = 0; k < substeps; k++) {
      s = forwardSim(s, run.controls, dt);
      samples.push(record(s));
    }
    out.push({
      startState: run.startState,
      controls: [...run.controls],
      duration,
      samples,
      endState: s,
    });
  }
  return out;
}

/** The common grid: every start state × every control set, start-major order
 *  (matches the historical characterizeVehicle iteration order). */
export function crossRuns<S>(
  startStates: S[],
  controlSets: number[][],
): CharacterizeRun<S>[] {
  const out: CharacterizeRun<S>[] = [];
  for (const startState of startStates) {
    for (const controls of controlSets) {
      out.push({ startState, controls });
    }
  }
  return out;
}

export interface CharacterizeVehicleOptions {
  forwardSim: ForwardSim<CarKinematicState>;
  /** Opaque control vectors to sweep (one primitive per control × speed). */
  controlSets: number[][];
  /** Wall-clock duration of each primitive (seconds). */
  duration: number;
  /** Integration / sweep-sampling substeps per primitive. */
  substeps: number;
  /** Start-speed buckets to characterize from. */
  startSpeeds: number[];
}

/**
 * Roll the supplied ForwardSim across the control × start-speed grid and
 * record each resulting short trajectory as a motion primitive (in the
 * start-local frame: start at origin, heading 0). A thin domain wrapper
 * over `characterize` — the ground-vehicle Sample is a LocalPose and the
 * gear is recovered from the raw end state.
 */
export function characterizeVehicle(
  opts: CharacterizeVehicleOptions,
): MotionPrimitiveLibrary {
  const { forwardSim, controlSets, duration, substeps, startSpeeds } = opts;
  const rolled = characterize<CarKinematicState, LocalPose>({
    forwardSim,
    runs: crossRuns(
      startSpeeds.map((speed) => ({ x: 0, z: 0, heading: 0, speed, t: 0 })),
      controlSets,
    ),
    duration,
    substeps,
    record: (s) => ({ x: s.x, z: s.z, heading: wrapAngle(s.heading) }),
  });
  const primitives: MotionPrimitive[] = rolled.map((r, id) => ({
    id,
    startSpeed: r.startState.speed,
    controls: r.controls,
    duration,
    end: {
      dx: r.endState.x,
      dz: r.endState.z,
      dHeading: wrapAngle(r.endState.heading),
      speed: r.endState.speed,
    },
    sweep: [{ x: 0, z: 0, heading: 0 }, ...r.samples],
    reverse: r.endState.speed < 0 || (r.controls[1] ?? 0) < 0,
  }));
  return new MotionPrimitiveLibrary(primitives, startSpeeds);
}

/**
 * WS-2 — DYNAMIC ROLLOUTS. Characterize primitives on the fly from the
 * chassis's ACTUAL dynamic state (including yaw rate and lateral velocity),
 * not the zero-slip canonical start states `characterizeVehicle` bakes. The
 * rollout starts at the local-frame origin (x=0, z=0, heading=0) but carries
 * the real `speed`/`yawRate`/`lateralVelocity`, so the produced primitives are
 * rigid-transform-valid in the node's world frame (the forward sim is
 * translation- and yaw-equivariant — see the CONTRACT in `characterize`).
 *
 * The point: the planner's motion library is baked from zero-slip states, so a
 * car sweeping through a corner (large yaw rate + sideslip) would otherwise
 * expand as if it were rolling straight with no slip — discarding exactly the
 * dynamic state the learned v2 model is stateful in. Rolling live from the
 * true state makes the committed first primitive model-consistent with reality.
 */
export function characterizeVehicleFromState(
  forwardSim: ForwardSim<CarKinematicState>,
  state: CarKinematicState,
  controlSets: number[][],
  duration: number,
  substeps: number,
): MotionPrimitive[] {
  const start: CarKinematicState = {
    x: 0, z: 0, heading: 0,
    speed: state.speed,
    yawRate: state.yawRate ?? 0,
    lateralVelocity: state.lateralVelocity ?? 0,
    t: 0,
  };
  const rolled = characterize<CarKinematicState, LocalPose>({
    forwardSim,
    runs: controlSets.map((controls) => ({ startState: start, controls })),
    duration,
    substeps,
    record: (s) => ({ x: s.x, z: s.z, heading: wrapAngle(s.heading) }),
  });
  return rolled.map((r, id) => ({
    id,
    startSpeed: r.startState.speed,
    controls: r.controls,
    duration,
    end: {
      dx: r.endState.x,
      dz: r.endState.z,
      dHeading: wrapAngle(r.endState.heading),
      speed: r.endState.speed,
    },
    sweep: [{ x: 0, z: 0, heading: 0 }, ...r.samples],
    reverse: r.endState.speed < 0 || (r.controls[1] ?? 0) < 0,
  }));
}
