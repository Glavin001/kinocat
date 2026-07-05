// Composable time-extension. Wraps any static-world Environment to (a) treat
// time as an extra dimension in the per-level dominance key and the exact
// hash, and (b) prune any successor that would collide with a predicted
// moving obstacle at its arrival time. This — time participating in the
// multi-resolution dominance — is kinocat's novel contribution over the
// IGHA* paper. The static env stays independently unit-testable; the
// time-aware behaviour composes on top.

import type { Environment, EdgeRef, Node } from './types';
import type { MovingObstacle } from '../predict/types';
import type { AffordanceRegistry } from '../predict/affordance-registry';
import type { CarKinematicState } from '../agent/types';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

export interface TimeAwareOptions<State = CarKinematicState> {
  /** Obstacles are XZ circles for planar states; when both an obstacle's
   *  prediction and the wrapped state carry `y`, the test is a 3D sphere. */
  obstacles?: MovingObstacle[];
  /** Circumscribed agent radius added to each obstacle radius. */
  agentRadius?: number;
  /** Fine time bucket (seconds) for the exact hash. */
  timeQuantum?: number;
  /** Per-level time-bucket divisors (coarse → fine); length = base.levels.
   *  Defaults to coupled halving (2^(levels-1-L)). */
  levelTimeDivisors?: number[];
  /** Lazily-generated affordance edges (ramps/jumps/boosts/…), typed to the
   *  wrapped state so any domain can publish them. */
  affordances?: AffordanceRegistry<State>;
  /** Proximity radius for affordance queries (world units). */
  affordanceRadius?: number;
  /**
   * Moving-obstacle broadphase. `collides()` is O(obstacles) per successor
   * (O(agents²) in multi-agent scenes). When enabled, each obstacle's
   * predicted motion is pre-sampled once into a padded AABB + active time
   * window; `collides()` then cheap-rejects by time/AABB before the exact
   * predict+circle test. A pure accelerator — it only skips the exact test
   * where that test would also report no collision (the AABB strictly
   * contains every sampled position plus the max between-sample motion;
   * beyond the sampled window the exact test always runs). Assumes each
   * predictor resolves at `sampleStep` (true for kinocat's continuous
   * factories). Disabled by default — `{}` enables, `false` disables.
   */
  broadphase?: false | { sampleStep?: number; maxSamples?: number };
}

interface ObstacleBound {
  active: boolean;
  tMaxSampled: number;
  tLo: number;
  tHi: number;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  /** Altitude bounds — used for rejection only when EVERY sampled
   *  prediction carried `y` (see `hasY`); otherwise some exact tests fall
   *  back to the XZ circle and a Y-based reject would be unsound. */
  hasY: boolean;
  minY: number;
  maxY: number;
  pad: number;
  rr: number;
}

/** Planar states satisfy this as-is; 3D states (e.g. AircraftState) also
 *  expose `y` and get spherical obstacle tests + altitude broadphase. */
type HasXZT = { x: number; z: number; t: number; y?: number };

export class TimeAwareEnvironment<State extends HasXZT>
  implements Environment<State>
{
  readonly levels: number;
  private readonly obstacles: MovingObstacle[];
  private readonly agentRadius: number;
  private readonly timeQuantum: number;
  private readonly divisors: number[];
  private readonly affordances?: AffordanceRegistry<State>;
  private readonly affordanceRadius: number;
  private readonly bp: ObstacleBound[] | null;
  private rec: PerfRecorder = NULL_RECORDER;
  /** Present exactly when the base env has a `progress` hook — the planner
   *  only pays for the best-progress fallback when the method exists, so a
   *  composing wrapper must forward it without introducing one the base
   *  doesn't have (contract on Environment.progress). */
  progress?: (node: Node<State>) => number;

  constructor(
    private readonly base: Environment<State>,
    opts: TimeAwareOptions<State> = {},
  ) {
    this.levels = base.levels;
    if (base.progress) {
      this.progress = (node) => base.progress!(node);
    }
    this.obstacles = opts.obstacles ?? [];
    this.agentRadius = opts.agentRadius ?? 0;
    this.timeQuantum = opts.timeQuantum ?? 0.2;
    this.divisors =
      opts.levelTimeDivisors ??
      Array.from({ length: this.levels }, (_, L) => 2 ** (this.levels - 1 - L));
    this.affordances = opts.affordances;
    this.affordanceRadius = opts.affordanceRadius ?? 15;
    const bpo = opts.broadphase;
    this.bp =
      bpo !== undefined && bpo !== false
        ? this.buildBroadphase(bpo.sampleStep ?? 0.4, bpo.maxSamples ?? 64)
        : null;
  }

  /** Pre-sample each obstacle's predicted motion into a conservative padded
   *  AABB + active time window (built once; obstacles are fixed for a
   *  search). The AABB is padded by the largest between-sample displacement
   *  so a point outside it provably never overlaps the obstacle within the
   *  sampled window. */
  private buildBroadphase(step: number, maxSamples: number): ObstacleBound[] {
    const out: ObstacleBound[] = [];
    const tMaxSampled = step * (maxSamples - 1);
    for (const obs of this.obstacles) {
      let firstT = Infinity;
      let lastT = -Infinity;
      let minX = Infinity;
      let maxX = -Infinity;
      let minZ = Infinity;
      let maxZ = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      let hasY = true;
      let maxStep = 0;
      let prev: { x: number; z: number; y?: number } | null = null;
      for (let k = 0; k < maxSamples; k++) {
        const t = k * step;
        const p = obs.predict(t);
        if (!p) {
          prev = null;
          continue;
        }
        if (t < firstT) firstT = t;
        if (t > lastT) lastT = t;
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.z < minZ) minZ = p.z;
        if (p.z > maxZ) maxZ = p.z;
        if (p.y === undefined) {
          hasY = false;
        } else {
          if (p.y < minY) minY = p.y;
          if (p.y > maxY) maxY = p.y;
        }
        if (prev) {
          const dy =
            p.y !== undefined && prev.y !== undefined ? p.y - prev.y : 0;
          const d = Math.hypot(p.x - prev.x, dy, p.z - prev.z);
          if (d > maxStep) maxStep = d;
        }
        prev = p;
      }
      const rr = obs.radius + this.agentRadius;
      out.push(
        firstT === Infinity
          ? {
              active: false,
              tMaxSampled,
              tLo: 0,
              tHi: 0,
              minX: 0,
              maxX: 0,
              minZ: 0,
              maxZ: 0,
              hasY: false,
              minY: 0,
              maxY: 0,
              pad: 0,
              rr,
            }
          : {
              active: true,
              tMaxSampled,
              tLo: firstT - step,
              tHi: lastT + step,
              minX,
              maxX,
              minZ,
              maxZ,
              hasY,
              minY,
              maxY,
              pad: rr + maxStep,
              rr,
            },
      );
    }
    return out;
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
    this.base.attachRecorder?.(rec);
  }

  private augment(node: Node<State>): Node<State> {
    const tb = Math.round(node.state.t / this.timeQuantum);
    // In-place mutation: createNode just allocated `node.index`, we own it.
    const idx = node.index;
    for (let L = 0; L < idx.length; L++) {
      const d = this.divisors[L] ?? 1;
      idx[L] = `${idx[L]}@${Math.floor(tb / d)}`;
    }
    node.hash = `${node.hash}@t${tb}`;
    return node;
  }

  /** True if `state` overlaps any predicted obstacle at its own time.
   *  XZ circle for planar states; 3D sphere when both the state and the
   *  prediction carry `y` (a y-less pairing degrades to the circle, i.e.
   *  an infinite vertical cylinder — conservative). */
  private collides(state: State): boolean {
    const bp = this.bp;
    const counters = this.rec.counters;
    const sy = state.y;
    if (!bp) {
      for (const obs of this.obstacles) {
        counters.predictCalls++;
        const p = obs.predict(state.t);
        if (!p) continue;
        const rr = obs.radius + this.agentRadius;
        const dx = state.x - p.x;
        const dz = state.z - p.z;
        const dy = sy !== undefined && p.y !== undefined ? sy - p.y : 0;
        if (dx * dx + dy * dy + dz * dz <= rr * rr) return true;
      }
      return false;
    }
    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i]!;
      const b = bp[i]!;
      if (state.t <= b.tMaxSampled) {
        // Full knowledge inside the sampled window.
        if (!b.active) {
          counters.broadphaseSkips++;
          continue; // predictor null across the window
        }
        if (state.t < b.tLo || state.t > b.tHi) {
          counters.broadphaseSkips++;
          continue; // outside active span
        }
        if (
          state.x < b.minX - b.pad ||
          state.x > b.maxX + b.pad ||
          state.z < b.minZ - b.pad ||
          state.z > b.maxZ + b.pad
        ) {
          counters.broadphaseSkips++;
          continue; // provably farther than rr from the obstacle
        }
        // Altitude reject — only when every sampled prediction carried `y`
        // (so every exact test in this window uses the sphere) AND the
        // state does too. An XZ-only reject above is always sound because
        // adding dy² can only increase the distance.
        if (
          b.hasY &&
          sy !== undefined &&
          (sy < b.minY - b.pad || sy > b.maxY + b.pad)
        ) {
          counters.broadphaseSkips++;
          continue; // provably farther than rr in altitude alone
        }
      }
      // Inside the uncertain band, or beyond the sampled window: exact test.
      counters.predictCalls++;
      const p = obs.predict(state.t);
      if (!p) continue;
      const dx = state.x - p.x;
      const dz = state.z - p.z;
      const dy = sy !== undefined && p.y !== undefined ? sy - p.y : 0;
      if (dx * dx + dy * dy + dz * dz <= b.rr * b.rr) return true;
    }
    return false;
  }

  createNode(
    state: State,
    parent: Node<State> | null,
    edge: EdgeRef | null,
  ): Node<State> {
    return this.augment(this.base.createNode(state, parent, edge));
  }

  succ(node: Node<State>, goal: Node<State>, level?: number): Node<State>[] {
    const out: Node<State>[] = [];
    // Forward `level` so base envs with per-level primitive sets (e.g.
    // AircraftEnvironment levelControls) keep working under composition —
    // same pattern as MultiGoalEnvironment / ScenarioEnvironment.
    const succs =
      level !== undefined
        ? this.base.succ(node, goal, level)
        : this.base.succ(node, goal);
    for (const c of succs) {
      if (this.collides(c.state)) continue;
      out.push(this.augment(c));
    }
    this.addAffordanceEdges(node, goal, out);
    return out;
  }

  /** Lazily generate affordance successors usable from `node` at its time. */
  private addAffordanceEdges(
    node: Node<State>,
    goal: Node<State>,
    out: Node<State>[],
  ): void {
    const reg = this.affordances;
    if (!reg) return;
    const st = node.state;
    for (const aff of reg.queryNearby(st.x, st.z, st.t, this.affordanceRadius)) {
      const r = aff.tryUse(st, st.t);
      if (!r) continue;
      const resState = r.resultState;
      if (this.collides(resState)) continue;
      const edge: EdgeRef = {
        cost: r.cost,
        kind: 'affordance',
        data: { affordanceId: aff.id, type: aff.type },
      };
      const n = this.createNode(resState, node, edge);
      n.g = node.g + r.cost;
      n.h = this.base.heuristic(resState, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
  }

  heuristic(from: State, to: State): number {
    return this.base.heuristic(from, to);
  }

  checkValidity(start: State, goal: State): [boolean, boolean] {
    // Intentionally only delegate to the static (positional) base check for
    // the start. Rejecting the start because some moving obstacle overlaps
    // it at t=0 makes the planner allergic to its own initial conditions —
    // when several agents (cops, the robber) cluster, every cop's predicted
    // plan overlaps every other cop's start position, and they all bail
    // simultaneously. The successor-expansion path still uses `collides()`
    // to reject states that *remain* in conflict with a moving obstacle, so
    // real time-windowed conflicts are still avoided; we just don't refuse
    // to plan from a current pose we can't change.
    return this.base.checkValidity(start, goal);
  }

  reachedGoalRegion(node: Node<State>, goal: Node<State>): boolean {
    return this.base.reachedGoalRegion(node, goal);
  }
}
