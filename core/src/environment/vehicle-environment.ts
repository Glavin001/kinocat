import type { Environment, EdgeRef, Node } from './types';
import type { NavWorld } from './nav-world';
import type { VehicleAgent, VehicleState } from '../agent/types';
import type { MotionPrimitiveLibrary } from '../primitives/library';
import { makeNode } from '../planner/node';
import { pack3 } from '../planner/resolution';
import { placeFootprint } from '../internal/geom';
import { angleDiff, dist, wrapAngle } from '../internal/math';
import { reedsSheppShortestPath } from '../curves/reeds-shepp';

export interface VehicleEnvOptions {
  posCell?: number;
  headingBuckets?: number;
  speedQuant?: number;
  /** Position-index divisors, coarse → fine; last MUST be 1. */
  levelDivisors?: number[];
  goalRadius?: number;
  /** Max |heading error| to accept the goal; default ∞ (ignore heading). */
  goalHeadingTol?: number;
  /** Also require straight segments between sweep samples to be clear. */
  sweepSegmentCheck?: boolean;
}

interface DriveEdgeData {
  primId: number;
  reverse: boolean;
}

export class VehicleEnvironment implements Environment<VehicleState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly headingBuckets: number;
  private readonly speedQuant: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly goalHeadingTol: number;
  private readonly sweepSegmentCheck: boolean;

  constructor(
    private readonly world: NavWorld,
    private readonly agent: VehicleAgent,
    private readonly lib: MotionPrimitiveLibrary,
    opts: VehicleEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.5;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.speedQuant = opts.speedQuant ?? 2;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 1.5;
    this.goalHeadingTol = opts.goalHeadingTol ?? Infinity;
    this.sweepSegmentCheck = opts.sweepSegmentCheck ?? true;
    this.levels = this.divisors.length;
  }

  private headingBucket(h: number): number {
    const step = (2 * Math.PI) / this.headingBuckets;
    return Math.round(wrapAngle(h) / step) % this.headingBuckets;
  }

  createNode(
    state: VehicleState,
    parent: Node<VehicleState> | null,
    edge: EdgeRef | null,
  ): Node<VehicleState> {
    const ix = Math.round(state.x / this.posCell);
    const iz = Math.round(state.z / this.posCell);
    const ih = this.headingBucket(state.heading);
    const isp = Math.round(state.speed / this.speedQuant);
    const it = Math.round(state.t / 0.25);
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack3(Math.floor(ix / d), Math.floor(iz / d), ih));
    }
    return makeNode(state, parent, edge, index, `${ix},${iz},${ih},${isp},${it}`);
  }

  private sweepClear(node: VehicleState, primSweep: ReadonlyArray<{ x: number; z: number; heading: number }>): boolean {
    const c = Math.cos(node.heading);
    const s = Math.sin(node.heading);
    let px = node.x;
    let pz = node.z;
    for (let i = 0; i < primSweep.length; i++) {
      const sp = primSweep[i]!;
      const wx = node.x + sp.x * c - sp.z * s;
      const wz = node.z + sp.x * s + sp.z * c;
      const wh = wrapAngle(node.heading + sp.heading);
      const fp = placeFootprint(this.agent.footprint, wx, wz, wh);
      if (!this.world.footprintClear(fp)) return false;
      if (this.sweepSegmentCheck && i > 0) {
        if (!this.world.segmentClear(px, pz, wx, wz)) return false;
      }
      px = wx;
      pz = wz;
    }
    return true;
  }

  succ(node: Node<VehicleState>, goal: Node<VehicleState>): Node<VehicleState>[] {
    const st = node.state;
    const c = Math.cos(st.heading);
    const s = Math.sin(st.heading);
    const parentReverse =
      node.edge && (node.edge.data as DriveEdgeData | undefined)?.reverse === true;
    const out: Node<VehicleState>[] = [];

    for (const prim of this.lib.lookup(st.speed)) {
      if (!this.sweepClear(st, prim.sweep)) continue;

      const ex = st.x + prim.end.dx * c - prim.end.dz * s;
      const ez = st.z + prim.end.dx * s + prim.end.dz * c;
      const next: VehicleState = {
        x: ex,
        z: ez,
        heading: wrapAngle(st.heading + prim.end.dHeading),
        speed: prim.end.speed,
        t: st.t + prim.duration,
      };

      const gearFlip = parentReverse !== undefined && parentReverse !== prim.reverse;
      const cost =
        prim.duration * (prim.reverse ? this.agent.reverseCostMultiplier : 1) +
        (gearFlip ? this.agent.directionChangePenalty : 0);

      const edge: EdgeRef = {
        cost,
        kind: prim.reverse ? 'drive-reverse' : 'drive',
        data: { primId: prim.id, reverse: prim.reverse } satisfies DriveEdgeData,
      };
      const n = this.createNode(next, node, edge);
      n.g = node.g + cost;
      n.h = this.heuristic(next, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  heuristic(from: VehicleState, to: VehicleState): number {
    const rs = reedsSheppShortestPath(
      { x: from.x, y: from.z, theta: from.heading },
      { x: to.x, y: to.z, theta: to.heading },
      this.agent.minTurnRadius,
    ).length;
    const euclid = dist(from.x, from.z, to.x, to.z);
    return Math.max(rs, euclid) / this.agent.maxSpeed;
  }

  private poseClear(s: VehicleState): boolean {
    return this.world.footprintClear(
      placeFootprint(this.agent.footprint, s.x, s.z, s.heading),
    );
  }

  checkValidity(start: VehicleState, goal: VehicleState): [boolean, boolean] {
    return [this.poseClear(start), this.poseClear(goal)];
  }

  reachedGoalRegion(node: Node<VehicleState>, goal: Node<VehicleState>): boolean {
    const a = node.state;
    const b = goal.state;
    if (dist(a.x, a.z, b.x, b.z) > this.goalRadius) return false;
    return Math.abs(angleDiff(a.heading, b.heading)) <= this.goalHeadingTol;
  }
}
