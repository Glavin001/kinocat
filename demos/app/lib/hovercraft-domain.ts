// The hovercraft motion body, defined DOWNSTREAM of kinocat using only
// public seams — this file is what a game would write for its own body,
// following docs/adding-a-domain.md. The minimal teaching version lives in
// core/test/examples/hovercraft.test.ts; this one is tuned for the 3D demo
// (bigger arena speeds, drop-obstacle worlds) but is the same five-part
// API: State, Agent envelope, ForwardSim, Environment, conformance harness
// (see demos/test/scenarios.test.ts for the headless proof).
//
// The body: inertial, thrust-vectored, drifting — the hull's facing is
// fully decoupled from the velocity vector, so it slides through corners
// sideways and momentum carries it past overshoots.

import type { Environment, EdgeRef, Node } from 'kinocat/environment';
import type { NavWorld } from 'kinocat/environment';
import type { ForwardSim } from 'kinocat/primitives';
import { makeNode, pack3 } from 'kinocat/planner';

// ── 1. State ────────────────────────────────────────────────────────────

export interface HovercraftState {
  x: number;
  z: number;
  /** Hull facing (rad) — thrust is body-mounted; independent of motion. */
  heading: number;
  /** World-frame velocity (m/s). */
  vx: number;
  vz: number;
  t: number;
}

// ── 2. Agent (the envelope) ─────────────────────────────────────────────

export interface HovercraftAgent {
  kind: 'hovercraft';
  radius: number;
  maxSpeed: number;
  /** Peak thrust acceleration (m/s²). */
  maxThrust: number;
  /** Yaw rate (rad/s) — fully decoupled from translation. */
  maxYawRate: number;
  /** Velocity decay per second (the skirt scrubs a little energy). */
  drag: number;
}

export const HOVER_AGENT: HovercraftAgent = {
  kind: 'hovercraft',
  radius: 1.1,
  maxSpeed: 12,
  maxThrust: 6,
  maxYawRate: Math.PI,
  drag: 0.18,
};

// ── 3. ForwardSim — controls are setpoints, state evolves ──────────────
// controls = [thrustFrac 0..1, thrustAngle rel. facing, yawFrac -1..1]

export function hovercraftForwardSim(
  a: HovercraftAgent,
): ForwardSim<HovercraftState> {
  return (s, controls, dt) => {
    const thrust = Math.max(0, Math.min(1, controls[0] ?? 0)) * a.maxThrust;
    const thrustAngle = controls[1] ?? 0;
    const yaw = Math.max(-1, Math.min(1, controls[2] ?? 0)) * a.maxYawRate;
    let heading = s.heading + yaw * dt;
    if (heading > Math.PI) heading -= 2 * Math.PI;
    if (heading < -Math.PI) heading += 2 * Math.PI;
    const dir = heading + thrustAngle;
    const decay = Math.max(0, 1 - a.drag * dt);
    let vx = s.vx * decay + Math.cos(dir) * thrust * dt;
    let vz = s.vz * decay + Math.sin(dir) * thrust * dt;
    const speed = Math.hypot(vx, vz);
    if (speed > a.maxSpeed) {
      vx *= a.maxSpeed / speed;
      vz *= a.maxSpeed / speed;
    }
    return {
      x: s.x + vx * dt,
      z: s.z + vz * dt,
      heading,
      vx,
      vz,
      t: s.t + dt,
    };
  };
}

// ── 4. Environment — live rollout (see docs/architecture.md Seam 2) ────

export interface HovercraftEnvOptions {
  posCell?: number;
  headingBuckets?: number;
  speedQuant?: number;
  velocityDirBuckets?: number;
  primDuration?: number;
  substeps?: number;
  levelDivisors?: number[];
  goalRadius?: number;
}

export const HOVER_PRIM_DURATION = 0.5;
export const HOVER_SUBSTEPS = 4;

type Pt = [number, number];

export class HovercraftEnvironment implements Environment<HovercraftState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly headingBuckets: number;
  private readonly speedQuant: number;
  private readonly velocityDirBuckets: number;
  private readonly primDuration: number;
  private readonly substeps: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly sim: ForwardSim<HovercraftState>;
  /** Every primitive is "hold these setpoints for primDuration". */
  private readonly actions: number[][];
  private readonly fpLocal: Pt[];
  private readonly fpScratch: Pt[];

  constructor(
    private readonly world: NavWorld,
    private readonly agent: HovercraftAgent,
    opts: HovercraftEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.8;
    this.headingBuckets = opts.headingBuckets ?? 12;
    this.speedQuant = opts.speedQuant ?? 3;
    this.velocityDirBuckets = opts.velocityDirBuckets ?? 8;
    this.primDuration = opts.primDuration ?? HOVER_PRIM_DURATION;
    this.substeps = opts.substeps ?? HOVER_SUBSTEPS;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 1.6;
    this.levels = this.divisors.length;
    this.sim = hovercraftForwardSim(agent);
    this.actions = [];
    for (const yaw of [-1, 0, 1]) this.actions.push([0, 0, yaw]);
    for (const angle of [0, Math.PI]) {
      for (const yaw of [-1, 0, 1]) this.actions.push([1, angle, yaw]);
    }
    const segs = 8;
    this.fpLocal = Array.from({ length: segs }, (_, i) => {
      const a = (i / segs) * 2 * Math.PI;
      return [agent.radius * Math.cos(a), agent.radius * Math.sin(a)] as Pt;
    });
    this.fpScratch = this.fpLocal.map(() => [0, 0] as Pt);
  }

  private clear(x: number, z: number): boolean {
    const local = this.fpLocal;
    const out = this.fpScratch;
    for (let i = 0; i < local.length; i++) {
      out[i]![0] = x + local[i]![0];
      out[i]![1] = z + local[i]![1];
    }
    return this.world.footprintClear(out);
  }

  createNode(
    state: HovercraftState,
    parent: Node<HovercraftState> | null,
    edge: EdgeRef | null,
  ): Node<HovercraftState> {
    const ix = Math.round(state.x / this.posCell);
    const iz = Math.round(state.z / this.posCell);
    const step = (2 * Math.PI) / this.headingBuckets;
    const ih =
      ((Math.round(state.heading / step) % this.headingBuckets) +
        this.headingBuckets) %
      this.headingBuckets;
    const speed = Math.hypot(state.vx, state.vz);
    const isp = Math.round(speed / this.speedQuant);
    const dirStep = (2 * Math.PI) / this.velocityDirBuckets;
    const ivd =
      speed < 0.6
        ? 0
        : ((Math.round(Math.atan2(state.vz, state.vx) / dirStep) %
            this.velocityDirBuckets) +
            this.velocityDirBuckets) %
          this.velocityDirBuckets;
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack3(Math.floor(ix / d), Math.floor(iz / d), ih));
    }
    // Every Markov dim in the hash; no time (static env — TimeAware adds it).
    return makeNode(state, parent, edge, index, `${ix},${iz},${ih},${isp},${ivd}`);
  }

  succ(
    node: Node<HovercraftState>,
    goal: Node<HovercraftState>,
  ): Node<HovercraftState>[] {
    const st = node.state;
    const dt = this.primDuration / this.substeps;
    const out: Node<HovercraftState>[] = [];
    for (let ci = 0; ci < this.actions.length; ci++) {
      const controls = this.actions[ci]!;
      let s = st;
      let px = st.x;
      let pz = st.z;
      let ok = true;
      for (let i = 0; i < this.substeps; i++) {
        s = this.sim(s, controls, dt);
        if (!this.world.segmentClear(px, pz, s.x, s.z)) {
          ok = false;
          break;
        }
        px = s.x;
        pz = s.z;
      }
      if (!ok || !this.clear(s.x, s.z)) continue;
      const edge: EdgeRef = {
        cost: this.primDuration,
        kind: 'hover',
        data: { controls },
      };
      const n = this.createNode(s, node, edge);
      n.g = node.g + this.primDuration;
      n.h = this.heuristic(s, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  heuristic(from: HovercraftState, to: HovercraftState): number {
    return Math.hypot(to.x - from.x, to.z - from.z) / this.agent.maxSpeed;
  }

  checkValidity(
    start: HovercraftState,
    goal: HovercraftState,
  ): [boolean, boolean] {
    return [this.clear(start.x, start.z), this.clear(goal.x, goal.z)];
  }

  reachedGoalRegion(
    node: Node<HovercraftState>,
    goal: Node<HovercraftState>,
  ): boolean {
    return (
      Math.hypot(node.state.x - goal.state.x, node.state.z - goal.state.z) <=
      this.goalRadius
    );
  }
}
