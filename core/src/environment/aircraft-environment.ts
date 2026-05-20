// AircraftEnvironment — a true 3D Environment<AircraftState> for the IGHA*
// core. Altitude is a searched dimension. Pitch and roll participate in the
// coarse-level dominance index at a coarser bucket (so equivalent-position
// routes that differ only in attitude collapse on coarse passes) but the
// finest-level exact hash keeps all 8 dimensions — preserving optimality.
// Collision uses an OBB oriented by yaw + pitch + roll so the planner can
// knife-edge through slots too narrow for level wings. Motion primitives are
// pre-characterized as local-frame sweeps (mirror of the vehicle pattern),
// so `succ()` is rigid-transform + collision-check per substep, not
// forward-sim per substep.

import type { Environment, EdgeRef, Node } from './types';
import type { AirspaceWorld } from './airspace-world';
import type { AircraftAgent, AircraftState } from '../agent/types';
import { aircraftForwardSim } from '../agent/aircraft';
import type { ForwardSim } from '../primitives/types';
import { makeNode } from '../planner/node';
import { wrapAngle, angleDiff } from '../internal/math';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

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
  /** Per-edge penalty added to cost as `rollCost · |roll| · primDuration`. */
  rollCost?: number;
  /**
   * Analytic straight-line "shot to goal" (Hybrid A*'s analytic-expansion
   * trick adapted for the aircraft). Periodically — once every `everyN`
   * expansions of this env — try a straight 3D segment from the current
   * node to the goal: sample at `step` intervals, swept-OBB collision
   * check at each, and if all clear emit a single goal-reaching successor.
   *
   * Massive speedup in sparse-complexity scenes where most of the path is
   * straight (canyon, restricted-airspace gauntlets): the lattice search
   * still drives around obstacles, but the moment a straight line to the
   * goal is feasible the planner terminates immediately instead of
   * stepping one primitive at a time.
   *
   * Opt-in. `false` (default) disables. `{}` enables with sensible defaults
   * (everyN=8, step=2). Heading and pitch on the synthetic poses are
   * derived from the segment direction; roll = 0 (wings level for the
   * straight). STATIC collision only — if moving zones exist they're still
   * sampled via `world.clear()` per substep, but the shot may pass between
   * a zone's future positions.
   */
  analyticExpansion?: false | { everyN?: number; step?: number };
  /**
   * Per-resolution-level motion-primitive sets (Item 4). When supplied,
   * coarse passes use a sparse primitive set (e.g. wings-level only) for
   * low branching; the finest pass uses a dense set for refinement. One
   * entry per level (length must equal `levelDivisors.length`). Any
   * unspecified field on a per-level entry inherits the global default
   * (`turnFractions`, `climbFractions`, `rollFractions`, `speeds`). When
   * absent, every level uses the global default — current behavior.
   *
   * Example for knife-edge: coarse plans a level skeleton, finest banks:
   *   levelControls: [
   *     { rollFractions: [0] },          // L0 coarse: no roll search
   *     { rollFractions: [0] },          // L1 medium: still no roll
   *     { rollFractions: [-1, 0, 1] },   // L2 finest: full roll set
   *   ]
   */
  levelControls?: Array<{
    turnFractions?: number[];
    climbFractions?: number[];
    rollFractions?: number[];
    speeds?: number[];
  }>;
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

/** A local-frame swept pose along one primitive (start-relative). */
interface LocalSweep {
  dx: number; // forward (body +x at heading 0)
  dz: number; // lateral (body +z at heading 0; world +z when heading 0)
  dy: number; // altitude delta
  dHeading: number;
  pitch: number; // absolute (controls clamp; constant within primitive)
  roll: number; // absolute
  dt: number;
}

interface CachedPrimitive {
  control: ControlQuad;
  /** Per-substep local-frame poses; length = substeps. */
  samples: LocalSweep[];
  /** End-of-primitive deltas in local frame (== samples[last] for convenience). */
  end: LocalSweep;
  /** Pre-built `controls` array for legacy/sim use; reused, not reallocated. */
  ctlArray: readonly [number, number, number, number];
  /** Edge data preset (the `data` is mutated per use? No — it's read-only). */
  cost: number;
  edgeData: FlyEdgeData;
  /**
   * Local-frame swept envelope AABB (parent at origin, heading 0).
   * Conservative for any world heading: each substep's OBB is bounded by
   * a sphere of radius `R = sqrt(halfL² + halfS² + halfH²)` (orientation-
   * independent), so the swept envelope is `union(center_i ± R)`.
   * Includes the start pose `(0,0,0) ± R` for completeness.
   * At runtime, rotated by parent heading + translated to parent position
   * → world-frame AABB, queried via `world.clearAABB`.
   */
  sweptLocal: {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
    zmin: number;
    zmax: number;
  };
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
  /** One primitive cache per resolution level (length == levels). Coarse
   *  passes may use a sparse subset; the finest pass uses the full set.
   *  If `levelControls` was not supplied, every level points at the same
   *  CachedPrimitive[] (the global default). */
  private readonly levelPrimitives: CachedPrimitive[][];
  private readonly sim: ForwardSim<AircraftState>;
  private readonly invMaxSpeed: number;
  private readonly half: [number, number, number];
  private readonly analyticEnabled: boolean;
  private readonly analyticEveryN: number;
  private readonly analyticStep: number;
  private succCount = 0;
  // Scratch pose object reused by collision checks (poseOf).
  private readonly _scratchPose = {
    x: 0,
    y: 0,
    z: 0,
    yaw: 0,
    pitch: 0,
    roll: 0,
  };
  private rec: PerfRecorder = NULL_RECORDER;

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
    const ae = opts.analyticExpansion;
    this.analyticEnabled = ae !== undefined && ae !== false;
    this.analyticEveryN = this.analyticEnabled
      ? ((ae as { everyN?: number }).everyN ?? 8)
      : 0;
    this.analyticStep = this.analyticEnabled
      ? ((ae as { step?: number }).step ?? 2)
      : 0;

    const kMax = 1 / agent.minTurnRadius;
    const defaultTurns = opts.turnFractions ?? [-1, -0.5, 0, 0.5, 1];
    const defaultClimbs = opts.climbFractions ?? [-1, 0, 1];
    // Roll search is opt-in: it lets the planner knife-edge through tight
    // slots but multiplies the branching factor.
    const defaultRolls = opts.rollFractions ?? [0];
    const defaultSpeeds = opts.speeds ?? [agent.maxSpeed];
    this.controls = this.makeControlQuads(
      kMax,
      defaultTurns,
      defaultClimbs,
      defaultRolls,
      defaultSpeeds,
    );
    this.levels = this.divisors.length;

    // Build per-level primitive caches. If levelControls is absent, every
    // level shares one cache (== current behavior). Otherwise, each entry
    // selects its own primitive subset; missing fields inherit the global
    // defaults. Length is clamped to `this.levels`.
    if (opts.levelControls && opts.levelControls.length > 0) {
      const lc = opts.levelControls;
      this.levelPrimitives = new Array(this.levels);
      for (let L = 0; L < this.levels; L++) {
        const entry = lc[Math.min(L, lc.length - 1)]!;
        const t = entry.turnFractions ?? defaultTurns;
        const c = entry.climbFractions ?? defaultClimbs;
        const r = entry.rollFractions ?? defaultRolls;
        const v = entry.speeds ?? defaultSpeeds;
        const quads = this.makeControlQuads(kMax, t, c, r, v);
        this.levelPrimitives[L] = this.buildPrimitiveCacheFor(quads);
      }
    } else {
      const shared = this.buildPrimitiveCacheFor(this.controls);
      this.levelPrimitives = new Array(this.levels).fill(shared);
    }
  }

  private makeControlQuads(
    kMax: number,
    turns: number[],
    climbs: number[],
    rolls: number[],
    speeds: number[],
  ): ControlQuad[] {
    const quads: ControlQuad[] = [];
    // Each speed band may have its own min-turn-radius (via the agent's
    // optional `turnRadiusAt` hook). Without the hook, every band shares
    // `kMax` and behavior is identical to before.
    for (const v of speeds) {
      const kMaxV = this.agent.turnRadiusAt
        ? 1 / this.agent.turnRadiusAt(v)
        : kMax;
      for (const tf of turns) {
        for (const cf of climbs) {
          for (const rf of rolls) {
            quads.push({
              k: tf * kMaxV,
              climb: cf * this.agent.maxClimbAngle,
              roll: rf * this.agent.maxBank,
              v,
            });
          }
        }
      }
    }
    return quads;
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
    this.world.attachRecorder?.(rec);
  }

  /**
   * Pre-characterize each control quad against the kinematic forward sim,
   * recording substep-by-substep local-frame deltas. At runtime `succ()`
   * rigid-transforms these by the parent node's heading + position rather
   * than re-simulating, mirroring `characterizeVehicle()` for vehicles.
   *
   * Soundness: `aircraftForwardSim` (in `core/src/agent/aircraft.ts`) is a
   * pure function of input state, control, and dt. The XZ-plane translation
   * depends only on heading + speed + pitch; rotating the world-frame
   * outputs by the parent's heading reproduces the simulated trajectory
   * exactly because the body-axes' yaw appears linearly in (cos h, sin h)
   * factors. Altitude is heading-independent. If the agent's forward sim is
   * later swapped for one that depends on global wind or absolute position,
   * gate this cache on that property.
   */
  private buildPrimitiveCacheFor(quads: ControlQuad[]): CachedPrimitive[] {
    const out: CachedPrimitive[] = [];
    const dt = this.primDuration / this.substeps;
    // Orientation-independent bound on the agent OBB: a sphere of radius R
    // contains the OBB at every yaw/pitch/roll. Used to build a swept
    // envelope that's valid for ANY world heading the primitive is applied
    // at (the runtime rotation only changes the substep centers).
    const R = Math.sqrt(
      this.agent.halfLength * this.agent.halfLength +
        this.agent.halfSpan * this.agent.halfSpan +
        this.agent.halfHeight * this.agent.halfHeight,
    );
    for (const c of quads) {
      const ctl: readonly [number, number, number, number] = [
        c.k,
        c.climb,
        c.roll,
        c.v,
      ];
      // Simulate from a canonical start (heading 0, origin, level wings),
      // then store world-frame deltas — they're the local-frame deltas.
      let s: AircraftState = {
        x: 0,
        y: 0,
        z: 0,
        heading: 0,
        pitch: 0,
        roll: 0,
        speed: c.v,
        t: 0,
      };
      const samples: LocalSweep[] = [];
      // Seed the envelope with the start pose (parent origin, heading 0).
      let xmin = -R;
      let xmax = R;
      let ymin = -R;
      let ymax = R;
      let zmin = -R;
      let zmax = R;
      for (let i = 0; i < this.substeps; i++) {
        s = this.sim(s, ctl as unknown as number[], dt);
        samples.push({
          dx: s.x,
          dz: s.z,
          dy: s.y,
          dHeading: s.heading,
          pitch: s.pitch,
          roll: s.roll,
          dt: s.t,
        });
        if (s.x - R < xmin) xmin = s.x - R;
        if (s.x + R > xmax) xmax = s.x + R;
        if (s.y - R < ymin) ymin = s.y - R;
        if (s.y + R > ymax) ymax = s.y + R;
        if (s.z - R < zmin) zmin = s.z - R;
        if (s.z + R > zmax) zmax = s.z + R;
      }
      const end = samples[samples.length - 1]!;
      const cost =
        this.primDuration +
        this.rollCost * Math.abs(c.roll) * this.primDuration;
      out.push({
        control: c,
        samples,
        end,
        ctlArray: ctl,
        cost,
        edgeData: { k: c.k, climb: c.climb, roll: c.roll },
        sweptLocal: { xmin, xmax, ymin, ymax, zmin, zmax },
      });
    }
    return out;
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
    const index: string[] = new Array(this.divisors.length);
    for (let L = 0; L < this.divisors.length; L++) {
      const d = this.divisors[L]!;
      // Coarse passes (d > 1) include pitch/roll buckets at coarser bins so
      // equivalent-(x,y,z,heading) routes that differ only in attitude
      // collapse to one dominance cell. The finest level (d === 1) keeps
      // (ix,iy,iz,ih) only — its exact-hash dedup carries the (pitch,roll)
      // distinction, so finest-pass optimality is unaffected.
      if (d > 1) {
        index[L] =
          `${Math.floor(ix / d)}:${Math.floor(iy / d)}:${Math.floor(iz / d)}:${ih}:${Math.floor(ip / d)}:${Math.floor(ir / d)}`;
      } else {
        index[L] = `${ix}:${iy}:${iz}:${ih}`;
      }
    }
    return makeNode(
      state,
      parent,
      edge,
      index,
      `${ix},${iy},${iz},${ih},${ip},${ir},${isp},${it}`,
    );
  }

  succ(
    node: Node<AircraftState>,
    goal: Node<AircraftState>,
    level?: number,
  ): Node<AircraftState>[] {
    const out: Node<AircraftState>[] = [];
    const st = node.state;
    const ch = Math.cos(st.heading);
    const sh = Math.sin(st.heading);
    const absCh = Math.abs(ch);
    const absSh = Math.abs(sh);
    const pose = this._scratchPose;
    const half = this.half;
    const clearAABB = this.world.clearAABB;
    // Select per-level primitive set (Item 4). Coarse passes may use a
    // sparse subset for low branching; finest pass uses the full set.
    const L = level === undefined ? this.levels - 1 : level;
    const primitives = this.levelPrimitives[Math.min(L, this.levels - 1)]!;

    for (let pi = 0; pi < primitives.length; pi++) {
      const prim = primitives[pi]!;

      // Per-primitive swept-AABB pre-check (Item 1): rotate the local-frame
      // swept envelope by the parent heading and ask the world for a fast
      // static-only clearance. If clear, skip the per-substep narrowphase
      // entirely. Sound because the swept envelope contains every OBB at
      // every substep — AABB-clear ⇒ OBB-clear.
      let fastClear = false;
      if (clearAABB) {
        const sw = prim.sweptLocal;
        const lxMid = (sw.xmin + sw.xmax) * 0.5;
        const lzMid = (sw.zmin + sw.zmax) * 0.5;
        const lxHalf = (sw.xmax - sw.xmin) * 0.5;
        const lzHalf = (sw.zmax - sw.zmin) * 0.5;
        const wxCenter = st.x + lxMid * ch - lzMid * sh;
        const wzCenter = st.z + lxMid * sh + lzMid * ch;
        const wxHalf = lxHalf * absCh + lzHalf * absSh;
        const wzHalf = lxHalf * absSh + lzHalf * absCh;
        fastClear = clearAABB.call(
          this.world,
          wxCenter - wxHalf,
          st.y + sw.ymin,
          wzCenter - wzHalf,
          wxCenter + wxHalf,
          st.y + sw.ymax,
          wzCenter + wzHalf,
        );
      }

      let clear = true;
      if (fastClear) {
        this.rec.counters.primitiveSweptSkips++;
      } else {
        // Rigid-transform each local-frame substep pose into world space
        // and collision-check. The primitive cache stored (dx, dz, dy,
        // dHeading, pitch, roll) at heading 0; rotate (dx, dz) by parent
        // heading.
        for (let i = 0; i < prim.samples.length; i++) {
          const sp = prim.samples[i]!;
          pose.x = st.x + sp.dx * ch - sp.dz * sh;
          pose.z = st.z + sp.dx * sh + sp.dz * ch;
          pose.y = st.y + sp.dy;
          pose.yaw = wrapAngle(st.heading + sp.dHeading);
          pose.pitch = sp.pitch;
          pose.roll = sp.roll;
          const tNow = st.t + sp.dt;
          if (!this.world.clear(pose, half, tNow)) {
            clear = false;
            break;
          }
        }
      }
      if (!clear) continue;

      const end = prim.end;
      const nextState: AircraftState = {
        x: st.x + end.dx * ch - end.dz * sh,
        y: st.y + end.dy,
        z: st.z + end.dx * sh + end.dz * ch,
        heading: wrapAngle(st.heading + end.dHeading),
        pitch: end.pitch,
        roll: end.roll,
        speed: prim.control.v,
        t: st.t + end.dt,
      };
      const edge: EdgeRef = {
        cost: prim.cost,
        kind: 'fly',
        data: prim.edgeData,
      };
      const n = this.createNode(nextState, node, edge);
      n.g = node.g + prim.cost;
      this.rec.counters.heuristicCalls++;
      n.h = this.heuristicState(nextState, goal.state);
      n.f = n.g + n.h;
      out.push(n);
    }

    // Item 2 — analytic straight-line shot to goal. Mirrors
    // VehicleEnvironment.tryAnalyticShot: every analyticEveryN expansions,
    // try a single straight 3D segment to the goal. If swept-clear, push
    // a goal-reaching successor (one extra branch from the planner's view).
    if (this.analyticEnabled) {
      this.succCount++;
      if (this.succCount === 1 || this.succCount % this.analyticEveryN === 0) {
        const shot = this.tryAnalyticShot(node, goal);
        if (shot) out.push(shot);
      }
    }
    return out;
  }

  /**
   * Straight-line 3D shot from `node.state` to `goal.state`. Samples the
   * segment at `analyticStep` intervals, builds a synthetic OBB pose at
   * each (heading = horizontal bearing, pitch = climb angle, roll = 0),
   * and tests collision. Returns a single goal-reaching successor on
   * success, null on first collision. Cost = path length / maxSpeed,
   * independent of the pitch (constant airspeed model). Heading and
   * pitch at the goal are derived from the segment direction, not the
   * goal's stored heading/pitch — the planner's `reachedGoalRegion` is
   * heading-tolerant by default.
   */
  private tryAnalyticShot(
    node: Node<AircraftState>,
    goal: Node<AircraftState>,
  ): Node<AircraftState> | null {
    this.rec.counters.analyticShots++;
    const a = node.state;
    const b = goal.state;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dz = b.z - a.z;
    const dxz = Math.sqrt(dx * dx + dz * dz);
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-6) return null;
    const heading = Math.atan2(dz, dx);
    const pitch = Math.atan2(dy, dxz);
    // Pitch must be within the airframe's climb envelope; if not, the
    // straight is kinematically infeasible. Steep descents to a goal below
    // an obstacle (e.g. canyon) won't shoot — the lattice handles them.
    if (Math.abs(pitch) > this.agent.maxClimbAngle + 1e-6) return null;

    // Cheap swept-AABB pre-reject: build the segment's bounding box
    // (extended by the agent's circumscribed radius) and ask the world's
    // static broadphase. Avoids per-sample work on shots obviously blocked
    // by static geometry — the common case in obstacle-rich scenes.
    const clearAABB = this.world.clearAABB;
    if (clearAABB) {
      const R = Math.sqrt(
        this.agent.halfLength * this.agent.halfLength +
          this.agent.halfSpan * this.agent.halfSpan +
          this.agent.halfHeight * this.agent.halfHeight,
      );
      const minX = Math.min(a.x, b.x) - R;
      const maxX = Math.max(a.x, b.x) + R;
      const minY = Math.min(a.y, b.y) - R;
      const maxY = Math.max(a.y, b.y) + R;
      const minZ = Math.min(a.z, b.z) - R;
      const maxZ = Math.max(a.z, b.z) + R;
      if (!clearAABB.call(this.world, minX, minY, minZ, maxX, maxY, maxZ)) {
        return null;
      }
      // Static-broadphase clear ⇒ no per-sample static collision possible.
      // We still need per-sample if moving zones exist (clearAABB returns
      // false in that case anyway, so we wouldn't be here). Skip the
      // expensive per-sample loop entirely.
      const speed = this.agent.maxSpeed;
      const flightTime = len / speed;
      const nextState: AircraftState = {
        x: b.x,
        y: b.y,
        z: b.z,
        heading,
        pitch,
        roll: 0,
        speed,
        t: a.t + flightTime,
      };
      const edge: EdgeRef = {
        cost: flightTime,
        kind: 'fly-shot',
        data: { straight: true, length: len },
      };
      const n = this.createNode(nextState, node, edge);
      n.g = node.g + flightTime;
      this.rec.counters.heuristicCalls++;
      n.h = this.heuristicState(nextState, goal.state);
      n.f = n.g + n.h;
      this.rec.counters.analyticShotsClear++;
      return n;
    }

    // No fast broadphase — fall through to per-sample world.clear loop.
    const speed = this.agent.maxSpeed;
    const flightTime = len / speed;
    const step = this.analyticStep;
    const nSteps = Math.max(1, Math.ceil(len / step));
    const pose = this._scratchPose;
    pose.yaw = heading;
    pose.pitch = pitch;
    pose.roll = 0;
    const half = this.half;
    // Sample along the segment. Skip i=0 (parent pose, already verified)
    // and check up to and including i=nSteps (the goal pose).
    for (let i = 1; i <= nSteps; i++) {
      const u = i / nSteps;
      pose.x = a.x + dx * u;
      pose.y = a.y + dy * u;
      pose.z = a.z + dz * u;
      const tSample = a.t + flightTime * u;
      if (!this.world.clear(pose, half, tSample)) return null;
    }

    const nextState: AircraftState = {
      x: b.x,
      y: b.y,
      z: b.z,
      heading,
      pitch,
      roll: 0,
      speed,
      t: a.t + flightTime,
    };
    const edge: EdgeRef = {
      cost: flightTime,
      kind: 'fly-shot',
      data: { straight: true, length: len },
    };
    const n = this.createNode(nextState, node, edge);
    n.g = node.g + flightTime;
    this.rec.counters.heuristicCalls++;
    n.h = this.heuristicState(nextState, goal.state);
    n.f = n.g + n.h;
    this.rec.counters.analyticShotsClear++;
    return n;
  }

  /** Internal heuristic that bypasses the public counter (caller already
   *  incremented for the per-successor case). */
  private heuristicState(from: AircraftState, to: AircraftState): number {
    const dx = from.x - to.x;
    const dy = from.y - to.y;
    const dz = from.z - to.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) * this.invMaxSpeed;
  }

  /** 3D straight-line time. Admissible & consistent. */
  heuristic(from: AircraftState, to: AircraftState): number {
    this.rec.counters.heuristicCalls++;
    return this.heuristicState(from, to);
  }

  private poseOf(s: AircraftState) {
    const p = this._scratchPose;
    p.x = s.x;
    p.y = s.y;
    p.z = s.z;
    p.yaw = s.heading;
    p.pitch = s.pitch;
    p.roll = s.roll;
    return p;
  }

  checkValidity(
    start: AircraftState,
    goal: AircraftState,
  ): [boolean, boolean] {
    const a = this.world.clear(this.poseOf(start), this.half, start.t);
    const b = this.world.clear(this.poseOf(goal), this.half, goal.t);
    return [a, b];
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
