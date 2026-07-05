// THE WORKED EXAMPLE for docs/adding-a-domain.md — a complete new
// controllable motion body ("agent domain"), end to end, in one file. The
// doc walks through this code section by section; if you change one, change
// both. It is deliberately a body unlike any that ships: a HOVERCRAFT —
// inertial, thrust-vectored, drifting, with yaw fully decoupled from the
// velocity vector. Everything below uses only public kinocat seams.
//
// The five things you define:
//   1. a State        — every dim the dynamics need to be Markov, plus `t`
//   2. an Agent       — the body's envelope (limits, size, cost knobs)
//   3. a ForwardSim   — controls are setpoints; state evolves under the envelope
//   4. an Environment — quantization, succ (roll the sim live), heuristic, goal
//   5. a DomainHarness — run the conformance battery; it defines "works"
// Affordances, moving obstacles, and goal automata then compose for free.

import { describe, it, expect } from 'vitest';
import type { Environment, EdgeRef, Node } from '../../src/environment/types';
import type { NavWorld } from '../../src/environment/nav-world';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import type { ForwardSim } from '../../src/primitives/types';
import { makeNode } from '../../src/planner/node';
import { pack3 } from '../../src/planner/resolution';
import { plan } from '../../src/planner/ighastar';
import { wrapAngle } from '../../src/internal/math';
import { placeFootprintInto, type Pt } from '../../src/internal/geom';
import {
  AffordanceRegistry,
  AffordanceType,
  type Affordance,
} from '../../src/predict/affordance-registry';
import { runConformance, type DomainHarness } from '../../src/testing';
import { rect } from '../fixtures/vehicle-sweep';

// ── 1. State ────────────────────────────────────────────────────────────
// Ask: "do two hovercraft at the same pose with different X behave
// differently going forward?" If yes, X is state. Velocity is world-frame
// and independent of heading (a hovercraft drifts); `t` is absolute time.

interface HovercraftState {
  x: number;
  z: number;
  /** Hull facing (rad) — thrust is body-mounted, so facing matters even
   *  though it does not constrain the velocity direction. */
  heading: number;
  vx: number;
  vz: number;
  t: number;
}

// ── 2. Agent (the envelope) ─────────────────────────────────────────────

interface HovercraftAgent {
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

const AGENT: HovercraftAgent = {
  kind: 'hovercraft',
  radius: 0.8,
  maxSpeed: 8,
  maxThrust: 4,
  maxYawRate: Math.PI,
  drag: 0.15,
};

// ── 3. ForwardSim — the single definition of what a primitive can do ────
// Controls are SETPOINTS: [thrustFrac 0..1, thrustAngle rel. facing, yawFrac
// -1..1]. Clamp to the envelope INSIDE the sim so the planner can never
// command the impossible. Keep it translation- and yaw-equivariant (no
// absolute-position effects) if you ever want to cache primitives.

function hovercraftForwardSim(a: HovercraftAgent): ForwardSim<HovercraftState> {
  return (s, controls, dt) => {
    const thrust = Math.max(0, Math.min(1, controls[0] ?? 0)) * a.maxThrust;
    const thrustAngle = controls[1] ?? 0;
    const yaw = Math.max(-1, Math.min(1, controls[2] ?? 0)) * a.maxYawRate;
    const heading = wrapAngle(s.heading + yaw * dt);
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

// ── 4. Environment — the planner's only view of the domain ─────────────
// succ() rolls the sim LIVE from each node's actual state (the recommended
// default: exact by construction, and the sim is trivially cheap next to
// collision checks). See docs/architecture.md Seam 2 for when to cache
// characterized primitives instead (car/momentum-humanoid pattern).

interface HovercraftEnvOptions {
  posCell?: number;
  headingBuckets?: number;
  speedQuant?: number;
  velocityDirBuckets?: number;
  primDuration?: number;
  substeps?: number;
  levelDivisors?: number[];
  goalRadius?: number;
}

class HovercraftEnvironment implements Environment<HovercraftState> {
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
  /** The action set: every primitive is "hold these setpoints for
   *  primDuration". [thrustFrac, thrustAngle, yawFrac] */
  private readonly actions: number[][];
  // Reusable scratch (collision checks run millions of times per plan).
  private readonly fpLocal: Pt[];
  private readonly fpScratch: Array<[number, number]>;

  constructor(
    private readonly world: NavWorld,
    private readonly agent: HovercraftAgent,
    opts: HovercraftEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.6;
    this.headingBuckets = opts.headingBuckets ?? 12;
    this.speedQuant = opts.speedQuant ?? 2;
    this.velocityDirBuckets = opts.velocityDirBuckets ?? 8;
    this.primDuration = opts.primDuration ?? 0.5;
    this.substeps = opts.substeps ?? 4;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 1.2;
    this.levels = this.divisors.length;
    this.sim = hovercraftForwardSim(agent);
    // Coast / thrust forward / reverse-thrust brake, each with yaw options.
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
    this.fpScratch = this.fpLocal.map(() => [0, 0]);
  }

  private clear(x: number, z: number): boolean {
    return this.world.footprintClear(
      placeFootprintInto(this.fpLocal, x, z, 0, this.fpScratch),
    );
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
      ((Math.round(wrapAngle(state.heading) / step) % this.headingBuckets) +
        this.headingBuckets) %
      this.headingBuckets;
    const speed = Math.hypot(state.vx, state.vz);
    const isp = Math.round(speed / this.speedQuant);
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
    // Hash every Markov dim; do NOT hash time in a static env (see
    // docs/architecture.md Seam 1 — TimeAwareEnvironment adds time when
    // moving obstacles make it meaningful).
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
        data: { controls }, // enough to re-simulate — the fidelity hook needs it
      };
      const n = this.createNode(s, node, edge);
      n.g = node.g + this.primDuration;
      n.h = this.heuristic(s, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  /** Speed-independent time bound — admissible and consistent because the
   *  sim caps |v| at maxSpeed (see the heuristic traps in
   *  docs/architecture.md before designing anything cleverer). */
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

// ── 5. Prove it: the conformance battery + fidelity hook ───────────────

const sim = hovercraftForwardSim(AGENT);
const PRIM_DURATION = 0.5;
const SUBSTEPS = 4;

const harness: DomainHarness<HovercraftState> = {
  makeEnv: () =>
    new HovercraftEnvironment(new InMemoryNavWorld([rect(1, 0, 0, 24, 16)]), AGENT),
  sampleState: (rand) => {
    const speed = rand() * AGENT.maxSpeed;
    const dir = (rand() - 0.5) * 2 * Math.PI;
    return {
      x: 2 + rand() * 20,
      z: 2 + rand() * 12,
      heading: (rand() - 0.5) * 2 * Math.PI,
      vx: speed * Math.cos(dir),
      vz: speed * Math.sin(dir),
      t: rand() * 30,
    };
  },
  scenarios: [
    {
      name: 'glide-across',
      start: { x: 2, z: 2, heading: 0, vx: 0, vz: 0, t: 0 },
      goal: { x: 21, z: 13, heading: 0, vx: 0, vz: 0, t: 0 },
      maxExpansions: 150_000,
    },
  ],
  // Live rollout ⇒ exact fidelity, no bucket slack.
  fidelity: {
    tolerance: 1e-9,
    angularFields: ['heading'],
    resimulate: (parent, edge) => {
      if (edge.kind !== 'hover') return null;
      const { controls } = edge.data as { controls: number[] };
      const dt = PRIM_DURATION / SUBSTEPS;
      let s = parent;
      for (let i = 0; i < SUBSTEPS; i++) s = sim(s, controls, dt);
      return s;
    },
  },
};

describe('worked example: a new controllable motion body (hovercraft)', () => {
  it('passes the full conformance battery', () => {
    const report = runConformance(harness);
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
    expect(report.checks).toContain('successor-fidelity');
  });

  it('drifts: momentum shows up in the plan', () => {
    const env = harness.makeEnv();
    const start: HovercraftState = {
      x: 4, z: 8, heading: 0, vx: AGENT.maxSpeed, vz: 0, t: 0,
    };
    const goal: HovercraftState = { x: 4, z: 8, heading: 0, vx: 0, vz: 0, t: 0 };
    // Full speed away from a goal AT the start point: momentum carries it
    // past before it can thrust back.
    const r = plan(
      { start: { ...start, x: 8 }, goal, environment: env, options: { maxExpansions: 150_000 } },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(Math.max(...r.path.map((s) => s.x))).toBeGreaterThan(10);
  });

  it('composes with the shared seams: a hovercraft-typed jump affordance', () => {
    // Two islands, a void between, an Affordance<HovercraftState> bridging
    // them — the SAME registry/wrapper machinery every other body uses.
    const islands = new InMemoryNavWorld([
      rect(1, 0, 0, 8, 8),
      rect(2, 20, 0, 28, 8),
    ]);
    const boost: Affordance<HovercraftState> = {
      id: 'gap-boost',
      type: AffordanceType.BoostPad,
      validFrom: -Infinity,
      validTo: Infinity,
      spatialBound: { x: 6, z: 4, radius: 2.5 },
      predict: () => ({ position: { x: 6, y: 0, z: 4 } }),
      tryUse: (s, useTime) => {
        const dx = s.x - 6;
        const dz = s.z - 4;
        if (dx * dx + dz * dz > 2.5 * 2.5) return null;
        return {
          resultState: { x: 22, z: 4, heading: 0, vx: 4, vz: 0, t: useTime + 1.5 },
          duration: 1.5,
          cost: 1.5,
          trajectory: [
            { x: s.x, y: 0, z: s.z, t: useTime },
            { x: 22, y: 0, z: 4, t: useTime + 1.5 },
          ],
        };
      },
    };
    const reg = new AffordanceRegistry<HovercraftState>();
    reg.add(boost);
    const env = new TimeAwareEnvironment(
      new HovercraftEnvironment(islands, AGENT),
      { affordances: reg, affordanceRadius: 4 },
    );
    const r = plan(
      {
        start: { x: 2, z: 4, heading: 0, vx: 0, vz: 0, t: 0 },
        goal: { x: 26, z: 4, heading: 0, vx: 0, vz: 0, t: 0 },
        environment: env,
        options: { maxExpansions: 120_000 },
      },
      Infinity,
    );
    expect(r.found).toBe(true);
    expect(r.nodes.some((n) => n.edge?.kind === 'affordance')).toBe(true);
  });
});
