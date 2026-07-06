import type { Environment, EdgeRef, Node } from './types';
import type { NavWorld } from './nav-world';
import type { VehicleAgent, CarKinematicState } from '../agent/types';
import type { MotionPrimitiveLibrary } from '../primitives/library';
import type { MotionPrimitive } from '../primitives/types';
import { makeNode } from '../planner/node';
import { pack3 } from '../planner/resolution';
import { placeFootprintInto } from '../internal/geom';
import { angleDiff, dist, wrapAngle } from '../internal/math';
import { reedsSheppShortestPath } from '../curves/reeds-shepp';
import { sampleCurveWithGear } from '../curves/sample';
import { NULL_RECORDER, type PerfRecorder } from '../planner/perf';

/** Minimum traversal speed (m/s) used when pricing a drive-through analytic
 *  shot by decel-to-stop time — guards the division when the entry speed is
 *  near rest so the cost stays finite and the maneuver stays selectable. */
const ANALYTIC_DRIVETHROUGH_VFLOOR = 4;
/** Fraction of entry speed used as the average traversal speed when pricing a
 *  drive-through analytic shot (it decelerates to rest, so avg < entry). Lower
 *  = more punitive (fewer stop-shortcuts, but heavier search); 0.5 = the honest
 *  cruise-then-brake average. Raising it toward ~0.75 discourages the mispriced
 *  stop while keeping the analytic shot cheap enough that the search stays
 *  tractable (fewer expansions). */
const ANALYTIC_DRIVETHROUGH_AVG_FRAC = 0.7;

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
  /**
   * Reeds-Shepp analytic expansion ("shot to goal"): periodically try the
   * exact RS curve from the current node to the goal and, if its swept
   * footprint is collision-free, finish in one step. Makes trivial / far
   * queries terminate immediately and slashes expansions on the common case
   * (Dolgov et al. Hybrid A*). STATIC collision only — wrap with disabled
   * analytic expansion, or accept best-effort + replanning, when there are
   * predicted dynamic obstacles. Disabled by default — pass `{}` (or a tuned
   * `{ everyN, step }`) to enable; `false` is the explicit disable.
   */
  analyticExpansion?: false | { everyN?: number; step?: number };
  /**
   * Price the analytic (Reeds-Shepp) shot as a DRIVE-THROUGH rather than a
   * stop. The analytic shot always ends at `speed: 0` and is normally
   * time-costed `length / maxSpeed` — free-flowing pricing for a maneuver that
   * actually comes to rest. For a STOP goal (parking) that is correct. For a
   * drive-through goal (racing gates) it mis-sells the stop as fast, so the
   * planner takes an analytic stop-shortcut to every gate instead of a
   * speed-carrying spline. When true, the analytic edge is instead costed by
   * the honest decel-to-stop time (~`length / (vEntry/2)`), so a stop only wins
   * where the maneuver genuinely must slow. Default false (parking / single-goal
   * unchanged). Set by the multi-goal race planner.
   */
  analyticDriveThrough?: boolean;
  /**
   * Reeds-Shepp heuristic lookup table (Dolgov et al. Hybrid A*; spec §12.3).
   * The RS shortest-path heuristic is the dominant per-successor cost. Since
   * the goal is fixed for a search, caching RS by quantized *source* pose
   * turns it into an O(1) lookup after the first touch of each cell. A
   * conservative slack (RS is 1-Lipschitz in translation, R-Lipschitz in
   * heading) keeps the estimate admissible, so optimality is preserved.
   * Disabled by default — pass `{}` (or a tuned `{ posCell, headingBuckets }`)
   * to enable; `false` is the explicit disable.
   */
  heuristicTable?: false | { posCell?: number; headingBuckets?: number };
  /**
   * Whether the time bucket participates in the exact dedup hash. REQUIRED
   * for time-varying worlds (moving obstacles, affordances) where the same
   * pose at different times genuinely differs. In STATIC worlds it only
   * prevents deduplication — measured 3.8x expansion inflation (83,968 vs
   * 22,220 for an identical parking query) — so wrappers disable it when the
   * request carries no dynamics. Default: true (safe for any world).
   */
  timeInHash?: boolean;
  /**
   * O(1) clearance broadphase (Opt 1, spec §10.2). When the `NavWorld`
   * provides a `clearanceAt` oracle (e.g. a `NavcatWorld` built with
   * `clearanceField`), skip the expensive exact footprint check at any sweep
   * sample where a disk of the agent's circumscribed radius is provably
   * clear. Early-ACCEPT only — never early-rejects — so it cannot introduce
   * a false "clear"; the exact check still runs in the uncertain band.
   * Disabled by default. No-op on worlds without `clearanceAt`.
   */
  clearanceBroadphase?: boolean;
  /**
   * Obstacle-aware grid-Dijkstra dual heuristic (Opt 2, Dolgov et al.; spec
   * §10.3). When the `NavWorld` provides `buildGoalLowerBound` (a
   * `NavcatWorld` with `clearanceField`), max() an admissible obstacle-
   * respecting distance-to-goal into the Reeds-Shepp heuristic so the search
   * stops expanding into walls/dead-ends — large win on obstacle-dense
   * terrain. Stays admissible (max of two lower bounds; the CHF must be
   * un-eroded — the adapter default). `weight` (default 1, keep ≤ 1 for
   * admissibility) scales the grid term. Disabled by default; `{}` enables.
   */
  gridHeuristic?: false | { weight?: number };
  /**
   * Trajectory-consistency (a.k.a. "stay close to the previously-committed
   * plan") cost. When provided, every successor pays an extra
   * `referenceWeight * perpDist(successor.xz, referencePath)` on its
   * primitive cost. Cheap hysteresis: with a small weight the planner only
   * abandons the previous geometry when an alternative is meaningfully
   * faster. Solves the flip-flopping between near-equal-cost paths that
   * gives the demo car its visibly jittery plan stream. Added only to g
   * (not h), so the heuristic remains admissible. Pass `undefined` /
   * empty to disable.
   */
  referencePath?: ReadonlyArray<{ x: number; z: number }>;
  /** Cost per metre of perpendicular deviation from `referencePath`.
   *  Default 0.1 (s/m if you read the time-cost as seconds). */
  referenceWeight?: number;
  /**
   * WS-2 — DYNAMIC ROLLOUTS. When provided, any node whose state carries
   * meaningful slip (|yawRate| or |lateralVelocity| above a small threshold)
   * is expanded by rolling THIS function's live-characterized primitives
   * instead of the baked, zero-slip `lib.lookup(speed)`. Because successors
   * drop the slip dims, this fires only at the search root (a mid-corner
   * replan start state), which is exactly where the baked zero-slip library
   * mis-predicts — the committed first primitive becomes model-consistent
   * with the chassis's true dynamic state. The callback receives the node
   * state (with slip) and must return primitives in the node's LOCAL frame
   * (start at origin, heading 0), as `characterizeVehicleFromState` does.
   * Pure + deterministic. Omitted ⇒ classic baked-library expansion.
   */
  rootRollout?: (state: CarKinematicState) => MotionPrimitive[];
}

interface DriveEdgeData {
  primId: number;
  reverse: boolean;
}

export interface AnalyticEdgeData {
  reedsShepp: true;
  reverse: boolean;
  /** Sampled world-space (x,z) of the curve, for tracking / drawing. */
  samples: [number, number][];
  /** Per-sample pose (heading) + gear of the curve. Lets a consumer rebuild
   *  an executable trajectory through the analytic shot instead of the bare
   *  node sequence's straight chord (which discards the curve geometry and
   *  the forward/reverse gear). Same length + order as `samples`. */
  poses: Array<{ x: number; z: number; heading: number; reverse: boolean }>;
}

export class VehicleEnvironment implements Environment<CarKinematicState> {
  readonly levels: number;
  private readonly posCell: number;
  private readonly timeInHash: boolean;
  private readonly headingBuckets: number;
  private readonly speedQuant: number;
  private readonly divisors: number[];
  private readonly goalRadius: number;
  private readonly goalHeadingTol: number;
  private readonly sweepSegmentCheck: boolean;
  private readonly analyticEnabled: boolean;
  private readonly analyticEveryN: number;
  private readonly analyticStep: number;
  private readonly analyticDriveThrough: boolean;
  private succCount = 0;
  private readonly htEnabled: boolean;
  private readonly htPos: number;
  private readonly htHead: number;
  private readonly htSlack: number;
  private readonly hCache = new Map<string, number>();
  private readonly cbEnabled: boolean;
  private readonly rCirc: number;
  // Scratch buffer reused by every collision-check footprint placement to
  // avoid allocating a fresh Pt[] on each call (thousands per plan). The
  // length matches `agent.footprint`; values are overwritten each use.
  private readonly fpScratch: Array<[number, number]>;
  private readonly ghEnabled: boolean;
  private readonly ghWeight: number;
  private ghGoalX = NaN;
  private ghGoalZ = NaN;
  private ghRev = -1;
  private ghLB: ((x: number, z: number, y?: number) => number | null) | null = null;
  private rec: PerfRecorder = NULL_RECORDER;
  private readonly refPath: ReadonlyArray<{ x: number; z: number }>;
  private readonly refWeight: number;
  private readonly rootRollout?: (state: CarKinematicState) => MotionPrimitive[];

  constructor(
    private readonly world: NavWorld,
    private readonly agent: VehicleAgent,
    private readonly lib: MotionPrimitiveLibrary,
    opts: VehicleEnvOptions = {},
  ) {
    this.posCell = opts.posCell ?? 0.5;
    this.timeInHash = opts.timeInHash ?? true;
    this.headingBuckets = opts.headingBuckets ?? 16;
    this.speedQuant = opts.speedQuant ?? 2;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalRadius = opts.goalRadius ?? 1.5;
    this.goalHeadingTol = opts.goalHeadingTol ?? Infinity;
    this.sweepSegmentCheck = opts.sweepSegmentCheck ?? true;
    const ae = opts.analyticExpansion; // opt-in: disabled unless provided
    this.analyticEnabled = ae !== undefined && ae !== false;
    this.analyticEveryN = this.analyticEnabled ? ((ae as { everyN?: number }).everyN ?? 6) : 0;
    this.analyticStep = this.analyticEnabled ? ((ae as { step?: number }).step ?? 0.4) : 0;
    this.analyticDriveThrough = opts.analyticDriveThrough ?? false;
    const ht = opts.heuristicTable; // opt-in: disabled unless provided
    this.htEnabled = ht !== undefined && ht !== false;
    this.htPos = this.htEnabled ? ((ht as { posCell?: number }).posCell ?? this.posCell) : 1;
    this.htHead = this.htEnabled
      ? ((ht as { headingBuckets?: number }).headingBuckets ?? this.headingBuckets)
      : 1;
    this.htSlack = this.htEnabled
      ? 0.5 * this.htPos * Math.SQRT2 + this.agent.minTurnRadius * (Math.PI / this.htHead)
      : 0;
    let rc = 0;
    for (const [vx, vz] of this.agent.footprint) {
      const r = Math.hypot(vx, vz);
      if (r > rc) rc = r;
    }
    this.rCirc = rc;
    this.fpScratch = this.agent.footprint.map(() => [0, 0]);
    this.cbEnabled =
      opts.clearanceBroadphase === true &&
      typeof this.world.clearanceAt === 'function';
    const gh = opts.gridHeuristic; // opt-in: disabled unless provided
    this.ghEnabled =
      gh !== undefined &&
      gh !== false &&
      typeof this.world.buildGoalLowerBound === 'function';
    this.ghWeight = this.ghEnabled ? ((gh as { weight?: number }).weight ?? 1) : 1;
    this.refPath = opts.referencePath ?? [];
    this.refWeight = this.refPath.length >= 2 ? (opts.referenceWeight ?? 0.1) : 0;
    this.rootRollout = opts.rootRollout;
    this.levels = this.divisors.length;
  }

  attachRecorder(rec: PerfRecorder): void {
    this.rec = rec;
  }

  /** Nearest perpendicular distance from (x, z) to the reference polyline.
   *  Returns 0 when there is no reference path. Linear scan — fine for the
   *  ~10–60 sample polylines that primitive planners produce. */
  private refDist(x: number, z: number): number {
    const rp = this.refPath;
    const n = rp.length;
    if (n < 2 || this.refWeight === 0) return 0;
    let best = Infinity;
    for (let i = 0; i < n - 1; i++) {
      const ax = rp[i]!.x;
      const az = rp[i]!.z;
      const bx = rp[i + 1]!.x;
      const bz = rp[i + 1]!.z;
      const dx = bx - ax;
      const dz = bz - az;
      const lenSq = dx * dx + dz * dz;
      let u = 0;
      if (lenSq > 1e-9) {
        u = ((x - ax) * dx + (z - az) * dz) / lenSq;
        if (u < 0) u = 0;
        else if (u > 1) u = 1;
      }
      const px = ax + dx * u;
      const pz = az + dz * u;
      const ddx = x - px;
      const ddz = z - pz;
      const d2 = ddx * ddx + ddz * ddz;
      if (d2 < best) best = d2;
    }
    return Math.sqrt(best);
  }

  private headingBucket(h: number): number {
    const step = (2 * Math.PI) / this.headingBuckets;
    return Math.round(wrapAngle(h) / step) % this.headingBuckets;
  }

  createNode(
    state: CarKinematicState,
    parent: Node<CarKinematicState> | null,
    edge: EdgeRef | null,
  ): Node<CarKinematicState> {
    const ix = Math.round(state.x / this.posCell);
    const iz = Math.round(state.z / this.posCell);
    const ih = this.headingBucket(state.heading);
    const isp = Math.round(state.speed / this.speedQuant);
    const it = this.timeInHash ? Math.round(state.t / 0.25) : 0;
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack3(Math.floor(ix / d), Math.floor(iz / d), ih));
    }
    return makeNode(state, parent, edge, index, `${ix},${iz},${ih},${isp},${it}`);
  }

  private sweepClear(node: CarKinematicState, primSweep: ReadonlyArray<{ x: number; z: number; heading: number }>): boolean {
    const c = Math.cos(node.heading);
    const s = Math.sin(node.heading);
    let px = node.x;
    let pz = node.z;
    // Sample 0 of every primitive sweep is the parent pose itself. Re-
    // checking its footprint is redundant — the parent was either
    // (a) accepted as the search start, or (b) produced as a successor
    // whose own sweep was already cleared. Skipping it here also means a
    // start state whose kinematic footprint slightly clips an obstacle (a
    // chassis pinned against a wall in physics) can still expand any
    // primitive that *moves out* of the clip. The segment check from
    // sample 0 to sample 1 still rejects forward primitives that drive
    // deeper into the wall, so we don't admit unsafe trajectories.
    for (let i = 1; i < primSweep.length; i++) {
      const sp = primSweep[i]!;
      const wx = node.x + sp.x * c - sp.z * s;
      const wz = node.z + sp.x * s + sp.z * c;
      const wh = wrapAngle(node.heading + sp.heading);
      // Clearance broadphase: if a disk of the circumscribed footprint
      // radius at (wx,wz) is provably clear, the footprint is too — skip the
      // exact check (early-accept only; never rejects).
      let cleared = false;
      if (this.cbEnabled) {
        const cl = this.world.clearanceAt!(wx, wz);
        if (cl !== null && cl >= this.rCirc) cleared = true;
      }
      if (!cleared) {
        const fp = placeFootprintInto(this.agent.footprint, wx, wz, wh, this.fpScratch);
        this.rec.counters.collisionChecks++;
        if (!this.world.footprintClear(fp)) {
          this.rec.counters.collisionRejects++;
          return false;
        }
      }
      if (this.sweepSegmentCheck) {
        this.rec.counters.collisionChecks++;
        if (!this.world.segmentClear(px, pz, wx, wz)) {
          this.rec.counters.collisionRejects++;
          return false;
        }
      }
      px = wx;
      pz = wz;
    }
    return true;
  }

  succ(node: Node<CarKinematicState>, goal: Node<CarKinematicState>): Node<CarKinematicState>[] {
    const st = node.state;
    const c = Math.cos(st.heading);
    const s = Math.sin(st.heading);
    // Gear of the edge that PRODUCED this node; `undefined` = no history (a
    // true root), which charges no direction-change penalty on the first move.
    // NOTE the null-guard: `node.edge && ...` evaluates to `null` for roots,
    // and `null !== undefined` made EVERY successor of an edge-less node pay
    // the flip penalty — in scenario/multi-goal bridge mode (which used to
    // rebuild inner nodes without their edge) that taxed every single edge,
    // making actual gear flips relatively FREE and shuffle plans cheap.
    const parentReverse =
      node.edge != null
        ? (node.edge.data as DriveEdgeData | undefined)?.reverse === true
        : undefined;
    const out: Node<CarKinematicState>[] = [];

    // WS-2: expand from the chassis's true dynamic state when it carries
    // meaningful slip (the mid-corner replan root), rolling the caller's live
    // model instead of the baked zero-slip library. Successors drop the slip
    // dims, so this naturally scopes to the root node.
    const hasSlip =
      this.rootRollout !== undefined &&
      (Math.abs(st.yawRate ?? 0) > 1e-3 || Math.abs(st.lateralVelocity ?? 0) > 1e-3);
    const prims: readonly MotionPrimitive[] = hasSlip
      ? this.rootRollout!(st)
      : this.lib.lookup(st.speed);

    for (const prim of prims) {
      if (!this.sweepClear(st, prim.sweep)) continue;

      const ex = st.x + prim.end.dx * c - prim.end.dz * s;
      const ez = st.z + prim.end.dx * s + prim.end.dz * c;
      const next: CarKinematicState = {
        x: ex,
        z: ez,
        heading: wrapAngle(st.heading + prim.end.dHeading),
        speed: prim.end.speed,
        t: st.t + prim.duration,
      };

      const gearFlip = parentReverse !== undefined && parentReverse !== prim.reverse;
      let cost =
        prim.duration * (prim.reverse ? this.agent.reverseCostMultiplier : 1) +
        (gearFlip ? this.agent.directionChangePenalty : 0);
      if (this.refWeight > 0) {
        cost += this.refWeight * this.refDist(ex, ez);
      }

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

    if (this.analyticEnabled) {
      this.succCount++;
      if (this.succCount === 1 || this.succCount % this.analyticEveryN === 0) {
        const shot = this.tryAnalyticShot(node, goal);
        if (shot) out.push(shot);
      }
    }
    return out;
  }

  /** Reeds-Shepp shot from `node` to the goal; a single goal-reaching
   *  successor if the swept footprint is collision-free, else null. */
  private tryAnalyticShot(
    node: Node<CarKinematicState>,
    goal: Node<CarKinematicState>,
  ): Node<CarKinematicState> | null {
    const a = node.state;
    const b = goal.state;
    const path = reedsSheppShortestPath(
      { x: a.x, y: a.z, theta: a.heading },
      { x: b.x, y: b.z, theta: b.heading },
      this.agent.minTurnRadius,
    );
    if (path.segments.length === 0) return null;

    const poses = sampleCurveWithGear(
      { x: a.x, y: a.z, theta: a.heading },
      path,
      this.agent.minTurnRadius,
      this.analyticStep,
    );
    const samples: [number, number][] = [];
    const posePath: AnalyticEdgeData['poses'] = [];
    let px = a.x;
    let pz = a.z;
    for (const p of poses) {
      const fp = placeFootprintInto(this.agent.footprint, p.x, p.y, p.theta, this.fpScratch);
      this.rec.counters.collisionChecks++;
      if (!this.world.footprintClear(fp)) {
        this.rec.counters.collisionRejects++;
        return null;
      }
      if (this.sweepSegmentCheck && (p.x !== px || p.y !== pz)) {
        this.rec.counters.collisionChecks++;
        if (!this.world.segmentClear(px, pz, p.x, p.y)) {
          this.rec.counters.collisionRejects++;
          return null;
        }
      }
      samples.push([p.x, p.y]);
      posePath.push({ x: p.x, z: p.y, heading: p.theta, reverse: p.reverse });
      px = p.x;
      pz = p.y;
    }

    const parentReverse =
      (node.edge?.data as DriveEdgeData | undefined)?.reverse === true;
    let cost = 0;
    let hasReverse = false;
    let prevReverse = parentReverse;
    // Speed used to convert analytic path length → time cost. A stop goal
    // (default) is priced optimistically at maxSpeed — cheap-is-right, you were
    // stopping anyway. A DRIVE-THROUGH goal must pay the honest decel-to-stop
    // time: the shot ends at rest (see `next` below), so from entry speed the
    // average traversal speed is ~vEntry/2. Without this the planner takes a
    // mispriced analytic stop-shortcut to every drive-through gate instead of
    // carrying speed through a spline (skill K5).
    const vTraverse = this.analyticDriveThrough
      ? Math.max(a.speed * ANALYTIC_DRIVETHROUGH_AVG_FRAC, ANALYTIC_DRIVETHROUGH_VFLOOR)
      : this.agent.maxSpeed;
    for (const seg of path.segments) {
      const rev = seg.gear < 0;
      hasReverse ||= rev;
      cost += (seg.length * (rev ? this.agent.reverseCostMultiplier : 1)) / vTraverse;
      if (rev !== prevReverse) cost += this.agent.directionChangePenalty;
      prevReverse = rev;
    }

    const next: CarKinematicState = {
      x: b.x,
      z: b.z,
      heading: b.heading,
      speed: 0,
      // Traverse TIME must match the cost's traversal speed, else the plan's
      // timestamp is too short and the executor sees a fake huge decel across
      // this segment. Drive-through: honest decel-to-stop average; else maxSpeed.
      t: a.t + path.length / vTraverse,
    };
    const edge: EdgeRef = {
      cost,
      kind: 'reeds-shepp',
      data: { reedsShepp: true, reverse: hasReverse, samples, poses: posePath } satisfies AnalyticEdgeData,
    };
    const n = this.createNode(next, node, edge);
    n.g = node.g + cost;
    n.h = this.heuristic(next, b);
    n.f = n.g + n.h;
    return n;
  }

  heuristic(from: CarKinematicState, to: CarKinematicState): number {
    this.rec.counters.heuristicCalls++;
    const euclid = dist(from.x, from.z, to.x, to.z);
    let tRS: number;
    if (this.htEnabled) {
      const cx = Math.round(from.x / this.htPos);
      const cz = Math.round(from.z / this.htPos);
      const stepH = (2 * Math.PI) / this.htHead;
      const ch = Math.round(wrapAngle(from.heading) / stepH);
      // Key on BOTH the source cell AND the goal pose. A multi-goal search
      // flips the goal between gates as interleaved gateIndex nodes expand;
      // keying only on the source and clearing the cache on every goal change
      // (the old behaviour) wiped it on each flip — defeating the table in
      // exactly the multi-goal case where it's enabled by default. Goals are
      // few (the gates), so a goal-prefixed key stays small and the RS solves
      // are reused across flips instead of recomputed.
      const key = `${to.x}:${to.z}:${to.heading}|${cx}:${cz}:${ch}`;
      let rs = this.hCache.get(key);
      if (rs === undefined) {
        rs = reedsSheppShortestPath(
          { x: cx * this.htPos, y: cz * this.htPos, theta: ch * stepH },
          { x: to.x, y: to.z, theta: to.heading },
          this.agent.minTurnRadius,
        ).length;
        this.hCache.set(key, rs);
      }
      // rs is computed at the cell centre; subtract the worst-case in-cell
      // deviation so the estimate is a true lower bound (admissible).
      tRS = Math.max(rs - this.htSlack, euclid) / this.agent.maxSpeed;
    } else {
      const rs = reedsSheppShortestPath(
        { x: from.x, y: from.z, theta: from.heading },
        { x: to.x, y: to.z, theta: to.heading },
        this.agent.minTurnRadius,
      ).length;
      tRS = Math.max(rs, euclid) / this.agent.maxSpeed;
    }
    if (this.ghEnabled) {
      // Revision guard: a tile rebuild under a long-lived env must not keep
      // serving a lower bound computed against the old geometry.
      if (
        to.x !== this.ghGoalX ||
        to.z !== this.ghGoalZ ||
        this.world.revision !== this.ghRev
      ) {
        this.ghGoalX = to.x;
        this.ghGoalZ = to.z;
        this.ghRev = this.world.revision;
        this.ghLB = this.world.buildGoalLowerBound!(to.x, to.z);
      }
      if (this.ghLB) {
        const d = this.ghLB(from.x, from.z);
        if (d !== null) {
          const tGrid = (d * this.ghWeight) / this.agent.maxSpeed;
          if (tGrid > tRS) return tGrid;
        }
      }
    }
    return tRS;
  }

  private poseClear(s: CarKinematicState): boolean {
    this.rec.counters.collisionChecks++;
    const ok = this.world.footprintClear(
      placeFootprintInto(this.agent.footprint, s.x, s.z, s.heading, this.fpScratch),
    );
    if (!ok) this.rec.counters.collisionRejects++;
    return ok;
  }

  checkValidity(_start: CarKinematicState, goal: CarKinematicState): [boolean, boolean] {
    // Always accept the start. In a chase / contact-rich physics simulation
    // the chassis can spend frames slightly clipping a wall or another car,
    // even though it is on the verge of breaking free. Refusing to plan
    // from those states leaves the controller without any path at all and
    // the chassis sits there until something else nudges it. The successor
    // expansion in `sweepClear` skips the parent pose's footprint check, so
    // any primitive whose post-start substeps are clear (notably reverse
    // out of the wall) is still accepted as a valid escape — the planner
    // figures out the way out instead of refusing the question.
    return [true, this.poseClear(goal)];
  }

  reachedGoalRegion(node: Node<CarKinematicState>, goal: Node<CarKinematicState>): boolean {
    const a = node.state;
    const b = goal.state;
    if (dist(a.x, a.z, b.x, b.z) > this.goalRadius) return false;
    return Math.abs(angleDiff(a.heading, b.heading)) <= this.goalHeadingTol;
  }
}
