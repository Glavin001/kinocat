// AircraftEnvironment — a true 3D Environment<AircraftState> for the IGHA*
// core. Altitude is a searched dimension: it participates in the exact `hash`
// (finest-pass dedup) AND the per-level dominance key, so coarse passes
// genuinely separate over/under routes. The planner itself is unchanged —
// this is just a new Environment implementation, exactly the extension point
// the 6-method interface is designed for.

import type { Environment, EdgeRef, Node } from './types';
import type { AirspaceWorld } from './airspace-world';
import type { AircraftAgent, AircraftState } from '../agent/types';
import { aircraftForwardSim } from '../agent/aircraft';
import type { ForwardSim } from '../primitives/types';
import { makeNode } from '../planner/node';
import { angleDiff, wrapAngle } from '../internal/math';

export interface AircraftEnvOptions {
  /** Horizontal position cell (x, z) for quantization. */
  posCell?: number;
  /** Altitude (y) cell — the third searched dimension. */
  altCell?: number;
  headingBuckets?: number;
  pitchBuckets?: number;
  speedQuant?: number;
  /** Position-index divisors, coarse → fine; last MUST be 1. */
  levelDivisors?: number[];
  goalRadius?: number;
  /** Max |heading error| to accept the goal; default ∞ (ignore heading). */
  goalHeadingTol?: number;
  /** Duration of one motion primitive (s). */
  primDuration?: number;
  /** Forward-sim substeps per primitive (collision-sweep resolution). */
  substeps?: number;
  /** Turn-curvature fractions of 1/minTurnRadius. */
  turnFractions?: number[];
  /** Climb-angle fractions of maxClimbAngle. */
  climbFractions?: number[];
  /** Target speeds; default `[maxSpeed]`. */
  speeds?: number[];
}

interface ControlTriple {
  k: number;
  climb: number;
  v: number;
}

interface FlyEdgeData {
  k: number;
  climb: number;
}

export class AircraftEnvironment implements Environment<AircraftState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly altCell: number;
  private readonly headingBuckets: number;
  private readonly pitchBuckets: number;
  private readonly speedQuant: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly goalHeadingTol: number;
  private readonly primDuration: number;
  private readonly substeps: number;
  private readonly controls: ControlTriple[];
  private readonly sim: ForwardSim<AircraftState>;
  private readonly invMaxSpeed: number;

  constructor(
    private readonly world: AirspaceWorld,
    private readonly agent: AircraftAgent,
    opts: AircraftEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 3;
    this.altCell = opts.altCell ?? 3;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.pitchBuckets = opts.pitchBuckets ?? 4;
    this.speedQuant = opts.speedQuant ?? 4;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 6;
    this.goalHeadingTol = opts.goalHeadingTol ?? Infinity;
    this.primDuration = opts.primDuration ?? 1;
    this.substeps = opts.substeps ?? 4;
    this.sim = aircraftForwardSim(agent);
    this.invMaxSpeed = 1 / agent.maxSpeed;

    const kMax = 1 / agent.minTurnRadius;
    const turns = opts.turnFractions ?? [-1, -0.5, 0, 0.5, 1];
    const climbs = opts.climbFractions ?? [-1, 0, 1];
    const speeds = opts.speeds ?? [agent.maxSpeed];
    const triples: ControlTriple[] = [];
    for (const tf of turns) {
      for (const cf of climbs) {
        for (const v of speeds) {
          triples.push({
            k: tf * kMax,
            climb: cf * agent.maxClimbAngle,
            v,
          });
        }
      }
    }
    this.controls = triples;
    this.levels = this.divisors.length;
  }

  private headingBucket(h: number): number {
    const step = (2 * Math.PI) / this.headingBuckets;
    return (
      ((Math.round(wrapAngle(h) / step) % this.headingBuckets) +
        this.headingBuckets) %
      this.headingBuckets
    );
  }

  createNode(
    state: AircraftState,
    parent: Node<AircraftState> | null,
    edge: EdgeRef | null,
  ): Node<AircraftState> {
    const ix = Math.round(state.x / this.posCell);
    const iy = Math.round(state.y / this.altCell);
    const iz = Math.round(state.z / this.posCell);
    const ih = this.headingBucket(state.heading);
    const ip = Math.round(
      (state.pitch / Math.max(this.agent.maxClimbAngle, 1e-6)) *
        this.pitchBuckets,
    );
    const isp = Math.round(state.speed / this.speedQuant);
    const it = Math.round(state.t / 0.25);
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(
        `${Math.floor(ix / d)}:${Math.floor(iy / d)}:${Math.floor(iz / d)}:${ih}`,
      );
    }
    return makeNode(
      state,
      parent,
      edge,
      index,
      `${ix},${iy},${iz},${ih},${ip},${isp},${it}`,
    );
  }

  succ(
    node: Node<AircraftState>,
    goal: Node<AircraftState>,
  ): Node<AircraftState>[] {
    const dtSub = this.primDuration / this.substeps;
    const out: Node<AircraftState>[] = [];
    for (const c of this.controls) {
      const ctl = [c.k, c.climb, c.v];
      let s = node.state;
      let clear = true;
      for (let i = 0; i < this.substeps; i++) {
        s = this.sim(s, ctl, dtSub);
        if (!this.world.clear(s.x, s.y, s.z, s.t, this.agent.radius)) {
          clear = false;
          break;
        }
      }
      if (!clear) continue;
      const cost = this.primDuration;
      const edge: EdgeRef = {
        cost,
        kind: 'fly',
        data: { k: c.k, climb: c.climb } satisfies FlyEdgeData,
      };
      const n = this.createNode(s, node, edge);
      n.g = node.g + cost;
      n.h = this.heuristic(s, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  /** 3D straight-line time. Admissible & consistent: airspeed is constant
   *  along the path so the 3D arc length per edge is `speed·duration` with
   *  `speed ≤ maxSpeed`, hence the per-edge h-drop never exceeds the edge
   *  cost. With enough budget the finest pass returns the optimal plan. */
  heuristic(from: AircraftState, to: AircraftState): number {
    const dx = from.x - to.x;
    const dy = from.y - to.y;
    const dz = from.z - to.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) * this.invMaxSpeed;
  }

  private clearAt(s: AircraftState): boolean {
    return this.world.clear(s.x, s.y, s.z, s.t, this.agent.radius);
  }

  checkValidity(
    start: AircraftState,
    goal: AircraftState,
  ): [boolean, boolean] {
    return [this.clearAt(start), this.clearAt(goal)];
  }

  reachedGoalRegion(
    node: Node<AircraftState>,
    goal: Node<AircraftState>,
  ): boolean {
    const a = node.state;
    const b = goal.state;
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    if (Math.sqrt(dx * dx + dy * dy + dz * dz) > this.goalRadius) return false;
    return Math.abs(angleDiff(a.heading, b.heading)) <= this.goalHeadingTol;
  }
}
