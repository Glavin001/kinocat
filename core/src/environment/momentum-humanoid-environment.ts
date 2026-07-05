// MomentumHumanoidEnvironment — the inertial person, planned by the same
// IGHA* core. Built deliberately from the public seams only (the fourth
// domain is the proof the seams are general): state + agent metadata in
// kinocat/agent, dynamics as a ForwardSim, primitives via the shared
// characterize() harness, collision on the NavWorld seam, time/moving
// obstacles/affordances via the TimeAwareEnvironment wrapper.
//
// Primitives are cached per (start-speed, velocity-direction-relative-to-
// facing) bucket: a person at walking speed can strafe or backpedal, a
// sprinter's primitives all start with the velocity out in front. succ()
// rigid-transforms the cached sweeps by the node's pose (valid because the
// forward sim is translation- and yaw-equivariant — see characterize()).

import type { Environment, EdgeRef, Node } from './types';
import type { NavWorld } from './nav-world';
import type {
  MomentumHumanoidAgent,
  MomentumHumanoidState,
} from '../agent/types';
import { momentumHumanoidForwardSim } from '../agent/momentum-humanoid';
import { characterize, crossRuns } from '../primitives/characterize';
import { makeNode } from '../planner/node';
import { pack3 } from '../planner/resolution';
import { dist, wrapAngle } from '../internal/math';
import { placeFootprintInto, type Pt } from '../internal/geom';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

export interface MomentumHumanoidEnvOptions {
  posCell?: number;
  headingBuckets?: number;
  /** Velocity-direction buckets in the exact hash (momentum is Markov
   *  state — two poses with different velocities are different vertices). */
  velocityDirBuckets?: number;
  /** Speed quantum (m/s) for the exact hash. */
  speedQuant?: number;
  /** Start-speed buckets to characterize primitives from. Defaults to
   *  [0, strafeSpeed, maxSpeed]. */
  speedBuckets?: number[];
  /** Wall-clock duration of one primitive (s). */
  primDuration?: number;
  /** Integration / collision-sweep substeps per primitive. */
  substeps?: number;
  levelDivisors?: number[];
  goalRadius?: number;
  /** Octagon segments approximating the round footprint. */
  footprintSegments?: number;
}

/** Per-substep local-frame record: pose for the collision sweep + velocity
 *  so the end state can be rebuilt in world frame. */
interface LocalSample {
  x: number;
  z: number;
  heading: number;
  vx: number;
  vz: number;
}

interface CachedPrimitive {
  samples: LocalSample[];
  end: LocalSample;
  cost: number;
  edgeData: { ci: number };
}

/** Body-local octagon approximating the round footprint (heading-invariant,
 *  so placement is translation-only via placeFootprintInto). */
function circleLocal(r: number, segs: number): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * 2 * Math.PI;
    out.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return out;
}

export class MomentumHumanoidEnvironment
  implements Environment<MomentumHumanoidState>
{
  readonly levels: number;
  private readonly posCell: number;
  private readonly headingBuckets: number;
  private readonly velocityDirBuckets: number;
  private readonly speedQuant: number;
  private readonly speedBuckets: number[];
  private readonly primDuration: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly footprintSegments: number;
  /** cache[speedBucketIndex] = map from relDir bucket value → primitives. */
  private readonly cache: Map<number, CachedPrimitive[]>[];
  private readonly relDirsPerSpeed: number[][];
  // Reused by every collision check to avoid allocating a polygon per call
  // (millions per plan) — same pattern as VehicleEnvironment.fpScratch.
  private readonly fpLocal: Pt[];
  private readonly fpScratch: Array<[number, number]>;
  private rec: PerfRecorder = NULL_RECORDER;

  constructor(
    private readonly world: NavWorld,
    private readonly agent: MomentumHumanoidAgent,
    opts: MomentumHumanoidEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.4;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.velocityDirBuckets = opts.velocityDirBuckets ?? 8;
    this.speedQuant = opts.speedQuant ?? 1;
    this.speedBuckets =
      opts.speedBuckets ?? [0, agent.strafeSpeed, agent.maxSpeed];
    this.primDuration = opts.primDuration ?? 0.5;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 0.6;
    this.footprintSegments = opts.footprintSegments ?? 8;
    this.levels = this.divisors.length;
    this.fpLocal = circleLocal(agent.radius, this.footprintSegments);
    this.fpScratch = this.fpLocal.map(() => [0, 0]);

    // Control grid: coast / full accel toward {forward, strafes, brake},
    // each with {left, none, right} turn. aFrac 0 makes aDir moot — one
    // coast entry per turn command only.
    const controlSets: number[][] = [];
    for (const turn of [-1, 0, 1]) controlSets.push([0, 0, turn]);
    for (const aDir of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
      for (const turn of [-1, 0, 1]) controlSets.push([1, aDir, turn]);
    }

    // Start buckets: at rest the velocity direction is meaningless; at
    // walking speed people move omnidirectionally; at sprint the velocity
    // is out in front (the envelope forbids fast non-facing motion).
    const substeps = opts.substeps ?? 4;
    const sim = momentumHumanoidForwardSim(agent);
    this.relDirsPerSpeed = this.speedBuckets.map((s) =>
      s === 0
        ? [0]
        : s <= agent.strafeSpeed
          ? [0, Math.PI / 2, Math.PI, -Math.PI / 2]
          : [0],
    );
    this.cache = this.speedBuckets.map((speed, si) => {
      const byRelDir = new Map<number, CachedPrimitive[]>();
      for (const relDir of this.relDirsPerSpeed[si]!) {
        const start: MomentumHumanoidState = {
          x: 0,
          z: 0,
          heading: 0,
          vx: speed * Math.cos(relDir),
          vz: speed * Math.sin(relDir),
          t: 0,
        };
        const rolled = characterize<MomentumHumanoidState, LocalSample>({
          forwardSim: sim,
          runs: crossRuns([start], controlSets),
          duration: this.primDuration,
          substeps,
          record: (s) => ({
            x: s.x,
            z: s.z,
            heading: wrapAngle(s.heading),
            vx: s.vx,
            vz: s.vz,
          }),
        });
        // Envelope clamps make several control combos land on near-identical
        // end states (e.g. strafe-accel at sprint ≈ coast). Deduplicate at
        // build time — duplicates would burn a full collision sweep each and
        // then die in the planner's exact-hash dedup anyway.
        const seen = new Set<string>();
        const prims: CachedPrimitive[] = [];
        rolled.forEach((r, ci) => {
          const e = r.samples[r.samples.length - 1]!;
          const key = [e.x, e.z, e.heading, e.vx, e.vz]
            .map((v) => Math.round(v * 1e3))
            .join(',');
          if (seen.has(key)) return;
          seen.add(key);
          prims.push({
            samples: r.samples,
            end: e,
            cost: this.primDuration,
            edgeData: { ci },
          });
        });
        byRelDir.set(relDir, prims);
      }
      return byRelDir;
    });
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
  }

  private headingBucket(h: number): number {
    const step = (2 * Math.PI) / this.headingBuckets;
    return Math.round(wrapAngle(h) / step) % this.headingBuckets;
  }

  private clear(x: number, z: number): boolean {
    this.rec.counters.collisionChecks++;
    const ok = this.world.footprintClear(
      placeFootprintInto(this.fpLocal, x, z, 0, this.fpScratch),
    );
    if (!ok) this.rec.counters.collisionRejects++;
    return ok;
  }

  /** Nearest cached primitive set for the node's (speed, rel. direction). */
  private primitivesFor(st: MomentumHumanoidState): CachedPrimitive[] {
    const speed = Math.hypot(st.vx, st.vz);
    let si = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.speedBuckets.length; i++) {
      const d = Math.abs(this.speedBuckets[i]! - speed);
      if (d < bestD) {
        bestD = d;
        si = i;
      }
    }
    const dirs = this.relDirsPerSpeed[si]!;
    let relDir = dirs[0]!;
    if (this.speedBuckets[si]! > 0 && speed > 1e-9 && dirs.length > 1) {
      const actual = wrapAngle(Math.atan2(st.vz, st.vx) - st.heading);
      let bestA = Infinity;
      for (const d of dirs) {
        const err = Math.abs(wrapAngle(actual - d));
        if (err < bestA) {
          bestA = err;
          relDir = d;
        }
      }
    }
    return this.cache[si]!.get(relDir)!;
  }

  createNode(
    state: MomentumHumanoidState,
    parent: Node<MomentumHumanoidState> | null,
    edge: EdgeRef | null,
  ): Node<MomentumHumanoidState> {
    const ix = Math.round(state.x / this.posCell);
    const iz = Math.round(state.z / this.posCell);
    const ih = this.headingBucket(state.heading);
    const speed = Math.hypot(state.vx, state.vz);
    const isp = Math.round(speed / this.speedQuant);
    // Velocity direction participates in the exact hash only when moving —
    // at rest every direction is the same state class.
    const dirStep = (2 * Math.PI) / this.velocityDirBuckets;
    const ivd =
      speed < 0.5
        ? 0
        : ((Math.round(wrapAngle(Math.atan2(state.vz, state.vx)) / dirStep) %
            this.velocityDirBuckets) +
            this.velocityDirBuckets) %
          this.velocityDirBuckets;
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack3(Math.floor(ix / d), Math.floor(iz / d), ih));
    }
    // NO time bucket in the exact hash: this is a static environment with
    // time-invariant dynamics and cost = time, so among states that differ
    // only in `t` the earliest arrival dominates — dedup on the physical
    // dims is both correct and essential (a time bucket would let every
    // cell be re-expanded once per arrival time, an unbounded ladder).
    // Under TimeAwareEnvironment composition the wrapper appends its own
    // time buckets to hash AND index, restoring time-distinctness exactly
    // where moving obstacles make it meaningful.
    return makeNode(state, parent, edge, index, `${ix},${iz},${ih},${isp},${ivd}`);
  }

  succ(
    node: Node<MomentumHumanoidState>,
    goal: Node<MomentumHumanoidState>,
  ): Node<MomentumHumanoidState>[] {
    const st = node.state;
    const c = Math.cos(st.heading);
    const s = Math.sin(st.heading);
    const out: Node<MomentumHumanoidState>[] = [];

    for (const prim of this.primitivesFor(st)) {
      // Sweep: centre segment per substep + footprint at the END pose only
      // (the HumanoidEnvironment approximation — a disc this small skimming
      // off-mesh mid-primitive with its centre segment on-mesh is bounded
      // by the radius, and the next expansion's checks catch it).
      let px = st.x;
      let pz = st.z;
      let clearPath = true;
      for (const sp of prim.samples) {
        const wx = st.x + sp.x * c - sp.z * s;
        const wz = st.z + sp.x * s + sp.z * c;
        this.rec.counters.collisionChecks++;
        if (!this.world.segmentClear(px, pz, wx, wz)) {
          this.rec.counters.collisionRejects++;
          clearPath = false;
          break;
        }
        px = wx;
        pz = wz;
      }
      if (!clearPath || !this.clear(px, pz)) continue;

      const end = prim.end;
      const next: MomentumHumanoidState = {
        x: st.x + end.x * c - end.z * s,
        z: st.z + end.x * s + end.z * c,
        heading: wrapAngle(st.heading + end.heading),
        vx: end.vx * c - end.vz * s,
        vz: end.vx * s + end.vz * c,
        t: st.t + this.primDuration,
      };
      const edge: EdgeRef = { cost: prim.cost, kind: 'move', data: prim.edgeData };
      const n = this.createNode(next, node, edge);
      n.g = node.g + prim.cost;
      n.h = this.heuristic(next, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  /**
   * Euclidean time bound: distance / maxSpeed. Deliberately IGNORES the
   * current speed, and this is a bucketing lesson the conformance kit
   * taught: primitives are characterized from canonical start-speed
   * buckets, so applying one "teleports" the state's speed to the bucket —
   * a successor's speed can jump by more than maxAccel·dt. Any heuristic
   * that rewards stored speed (an accel-aware double-integrator bound)
   * then violates consistency along bucket-up edges. The speed-independent
   * bound is immune: the forward sim clamps |v| ≤ maxSpeed during every
   * rollout, so an edge of duration T closes at most maxSpeed·T of
   * distance, and h(a) ≤ T + h(b) holds regardless of bucket jumps.
   * (The car's Reeds-Shepp heuristic is speed-independent for the same
   * structural reason.)
   */
  heuristic(from: MomentumHumanoidState, to: MomentumHumanoidState): number {
    this.rec.counters.heuristicCalls++;
    return (
      dist(from.x, from.z, to.x, to.z) / this.agent.maxSpeed
    );
  }

  checkValidity(
    start: MomentumHumanoidState,
    goal: MomentumHumanoidState,
  ): [boolean, boolean] {
    return [this.clear(start.x, start.z), this.clear(goal.x, goal.z)];
  }

  reachedGoalRegion(
    node: Node<MomentumHumanoidState>,
    goal: Node<MomentumHumanoidState>,
  ): boolean {
    return (
      dist(node.state.x, node.state.z, goal.state.x, goal.state.z) <=
      this.goalRadius
    );
  }
}
