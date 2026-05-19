import type { Environment, EdgeRef, Node } from './types';
import type { NavWorld } from './nav-world';
import type { HumanoidAgent, HumanoidState } from '../agent/types';
import { makeNode } from '../planner/node';
import { pack3 } from '../planner/resolution';
import { dist, wrapAngle } from '../internal/math';
import type { Pt } from '../internal/geom';

export interface HumanoidEnvOptions {
  posCell?: number;
  headingBuckets?: number;
  /** Number of discretized travel directions per expansion. */
  directions?: number;
  /** Wall-clock duration of one step primitive. */
  stepDuration?: number;
  levelDivisors?: number[];
  goalRadius?: number;
  /** Octagon segments approximating the round footprint. */
  footprintSegments?: number;
}

function circlePoly(x: number, z: number, r: number, segs: number): Pt[] {
  const out: Pt[] = [];
  for (let i = 0; i < segs; i++) {
    const a = (i / segs) * 2 * Math.PI;
    out.push([x + r * Math.cos(a), z + r * Math.sin(a)]);
  }
  return out;
}

/**
 * Humanoid environment. State is (x, z, heading, t) — no inertial speed
 * dimension (humans have no turning-radius constraint at game speeds; speed is
 * baked into the step primitive). Omnidirectional steps + navmesh off-mesh
 * jump links. Reuses the entire IGHA* / time-extension / affordance stack.
 */
export class HumanoidEnvironment implements Environment<HumanoidState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly headingBuckets: number;
  private readonly directions: number;
  private readonly stepDuration: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly footprintSegments: number;
  private readonly stepDist: number;

  constructor(
    private readonly world: NavWorld,
    private readonly agent: HumanoidAgent,
    opts: HumanoidEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.4;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.directions = opts.directions ?? 16;
    this.stepDuration = opts.stepDuration ?? 0.3;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 0.6;
    this.footprintSegments = opts.footprintSegments ?? 8;
    this.stepDist = this.agent.maxSpeed * this.stepDuration;
    this.levels = this.divisors.length;
  }

  private headingBucket(h: number): number {
    const step = (2 * Math.PI) / this.headingBuckets;
    return Math.round(wrapAngle(h) / step) % this.headingBuckets;
  }

  private clear(x: number, z: number): boolean {
    return this.world.footprintClear(
      circlePoly(x, z, this.agent.radius, this.footprintSegments),
    );
  }

  createNode(
    state: HumanoidState,
    parent: Node<HumanoidState> | null,
    edge: EdgeRef | null,
  ): Node<HumanoidState> {
    const ix = Math.round(state.x / this.posCell);
    const iz = Math.round(state.z / this.posCell);
    const ih = this.headingBucket(state.heading);
    const it = Math.round(state.t / 0.25);
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack3(Math.floor(ix / d), Math.floor(iz / d), ih));
    }
    return makeNode(state, parent, edge, index, `${ix},${iz},${ih},${it}`);
  }

  succ(node: Node<HumanoidState>, goal: Node<HumanoidState>): Node<HumanoidState>[] {
    const st = node.state;
    const out: Node<HumanoidState>[] = [];

    for (let i = 0; i < this.directions; i++) {
      const dir = (i / this.directions) * 2 * Math.PI;
      const nx = st.x + this.stepDist * Math.cos(dir);
      const nz = st.z + this.stepDist * Math.sin(dir);
      if (!this.clear(nx, nz)) continue;
      if (!this.world.segmentClear(st.x, st.z, nx, nz)) continue;
      const next: HumanoidState = {
        x: nx,
        z: nz,
        heading: wrapAngle(dir),
        t: st.t + this.stepDuration,
      };
      const edge: EdgeRef = { cost: this.stepDuration, kind: 'walk' };
      out.push(this.finish(node, next, edge, goal));
    }

    // navmesh off-mesh jump/drop/climb links
    const poly = this.world.polygonAt(st.x, st.z);
    if (poly) {
      for (const link of this.world.offMeshFrom(poly)) {
        const next: HumanoidState = {
          x: link.end[0],
          z: link.end[2],
          heading: wrapAngle(Math.atan2(link.end[2] - st.z, link.end[0] - st.x)),
          t: st.t + link.cost,
        };
        if (!this.clear(next.x, next.z)) continue;
        const edge: EdgeRef = { cost: link.cost, kind: link.kind, data: { offMesh: true } };
        out.push(this.finish(node, next, edge, goal));
      }
    }
    return out;
  }

  private finish(
    parent: Node<HumanoidState>,
    next: HumanoidState,
    edge: EdgeRef,
    goal: Node<HumanoidState>,
  ): Node<HumanoidState> {
    const n = this.createNode(next, parent, edge);
    n.g = parent.g + edge.cost;
    n.h = this.heuristic(next, goal.state);
    n.f = n.g + n.h;
    return n;
  }

  heuristic(from: HumanoidState, to: HumanoidState): number {
    return dist(from.x, from.z, to.x, to.z) / this.agent.maxSpeed;
  }

  checkValidity(start: HumanoidState, goal: HumanoidState): [boolean, boolean] {
    return [this.clear(start.x, start.z), this.clear(goal.x, goal.z)];
  }

  reachedGoalRegion(node: Node<HumanoidState>, goal: Node<HumanoidState>): boolean {
    return dist(node.state.x, node.state.z, goal.state.x, goal.state.z) <= this.goalRadius;
  }
}
