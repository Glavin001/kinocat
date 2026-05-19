// AircraftEnvironment — a true 3D Environment<AircraftState> for the IGHA*
// core. Altitude is a searched dimension (in the hash and the per-level
// dominance key); roll is a tactical searched dimension (in the hash only,
// so coarse passes don't fragment over equivalent-altitude routes at
// different banks). Collision uses an OBB oriented by yaw + pitch + roll so
// the planner can knife-edge through slots too narrow for level wings.

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
  rollBuckets?: number;
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
  /** Bank-angle fractions of maxBank. */
  rollFractions?: number[];
  /** Target speeds; default `[maxSpeed]`. */
  speeds?: number[];
  /** Per-edge penalty added to cost as `rollCost · |roll| · primDuration` (in
   *  cost units per radian per second). Biases the planner toward wings-level
   *  flight; banking is still chosen when geometry demands it (the alternative
   *  is collision rejection). Default 0.5: a full ±π/2 bank for 1 s costs
   *  ~0.79, roughly doubling the edge cost of wings-level cruise. Kept
   *  admissible w.r.t. the 3D-Euclidean / maxSpeed heuristic because edge
   *  cost stays ≥ primDuration ≥ h-decrease. */
  rollCost?: number;
}

interface ControlQuad {
  k: number;
  climb: number;
  roll: number;
  v: number;
}

interface FlyEdgeData {
  k: number;
  climb: number;
  roll: number;
}

export class AircraftEnvironment implements Environment<AircraftState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly altCell: number;
  private readonly headingBuckets: number;
  private readonly pitchBuckets: number;
  private readonly rollBuckets: number;
  private readonly speedQuant: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly goalHeadingTol: number;
  private readonly primDuration: number;
  private readonly substeps: number;
  private readonly rollCost: number;
  private readonly controls: ControlQuad[];
  private readonly sim: ForwardSim<AircraftState>;
  private readonly invMaxSpeed: number;
  private readonly half: [number, number, number];

  constructor(
    private readonly world: AirspaceWorld,
    private readonly agent: AircraftAgent,
    opts: AircraftEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 3;
    this.altCell = opts.altCell ?? 3;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.pitchBuckets = opts.pitchBuckets ?? 4;
    this.rollBuckets = opts.rollBuckets ?? 4;
    this.speedQuant = opts.speedQuant ?? 4;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 6;
    this.goalHeadingTol = opts.goalHeadingTol ?? Infinity;
    this.primDuration = opts.primDuration ?? 1;
    this.substeps = opts.substeps ?? 6;
    this.rollCost = opts.rollCost ?? 0.5;
    this.sim = aircraftForwardSim(agent);
    this.invMaxSpeed = 1 / agent.maxSpeed;
    this.half = [agent.halfLength, agent.halfSpan, agent.halfHeight];

    const kMax = 1 / agent.minTurnRadius;
    const turns = opts.turnFractions ?? [-1, -0.5, 0, 0.5, 1];
    const climbs = opts.climbFractions ?? [-1, 0, 1];
    // Roll search is opt-in: it lets the planner knife-edge through tight
    // slots but triples the branching factor. Scenarios that need it (e.g.,
    // narrow vertical slots) pass `rollFractions: [-1, 0, 1]` explicitly.
    const rolls = opts.rollFractions ?? [0];
    const speeds = opts.speeds ?? [agent.maxSpeed];
    const quads: ControlQuad[] = [];
    for (const tf of turns) {
      for (const cf of climbs) {
        for (const rf of rolls) {
          for (const v of speeds) {
            quads.push({
              k: tf * kMax,
              climb: cf * agent.maxClimbAngle,
              roll: rf * agent.maxBank,
              v,
            });
          }
        }
      }
    }
    this.controls = quads;
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
    const ir = Math.round(
      (state.roll / Math.max(this.agent.maxBank, 1e-6)) * this.rollBuckets,
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
      `${ix},${iy},${iz},${ih},${ip},${ir},${isp},${it}`,
    );
  }

  private poseOf(s: AircraftState) {
    return {
      x: s.x,
      y: s.y,
      z: s.z,
      yaw: s.heading,
      pitch: s.pitch,
      roll: s.roll,
    };
  }

  succ(
    node: Node<AircraftState>,
    goal: Node<AircraftState>,
  ): Node<AircraftState>[] {
    const dtSub = this.primDuration / this.substeps;
    const out: Node<AircraftState>[] = [];
    for (const c of this.controls) {
      const ctl = [c.k, c.climb, c.roll, c.v];
      let s = node.state;
      let clear = true;
      for (let i = 0; i < this.substeps; i++) {
        s = this.sim(s, ctl, dtSub);
        if (!this.world.clear(this.poseOf(s), this.half, s.t)) {
          clear = false;
          break;
        }
      }
      if (!clear) continue;
      const cost =
        this.primDuration + this.rollCost * Math.abs(c.roll) * this.primDuration;
      const edge: EdgeRef = {
        cost,
        kind: 'fly',
        data: { k: c.k, climb: c.climb, roll: c.roll } satisfies FlyEdgeData,
      };
      const n = this.createNode(s, node, edge);
      n.g = node.g + cost;
      n.h = this.heuristic(s, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  /** 3D straight-line time. Admissible & consistent (airspeed is constant
   *  along the path, so the per-edge h-drop never exceeds the edge cost). */
  heuristic(from: AircraftState, to: AircraftState): number {
    const dx = from.x - to.x;
    const dy = from.y - to.y;
    const dz = from.z - to.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) * this.invMaxSpeed;
  }

  checkValidity(
    start: AircraftState,
    goal: AircraftState,
  ): [boolean, boolean] {
    return [
      this.world.clear(this.poseOf(start), this.half, start.t),
      this.world.clear(this.poseOf(goal), this.half, goal.t),
    ];
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
