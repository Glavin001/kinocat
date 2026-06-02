// Scenario & Goal Specification Layer — shared vocabulary.
//
// A scenario decomposes into THREE orthogonal planes (never mixed):
//   - Objective  (reach / repeat, composed by seq / all / any) -> an AUTOMATON
//   - Invariant  (avoid / maintain[.while])                     -> successor PRUNING
//   - Cost       (prefer)                                       -> edge cost g
// over a shared spatial primitive, the `Region`.
//
// STATE CONVENTION. The design spec writes vehicle state as (x, y, theta, v).
// kinocat plans on the world XZ plane with `CarKinematicState =
// {x, z, heading, speed, t}`, so this layer adopts THAT convention
// (`ScenarioState` below) — it composes directly with `VehicleEnvironment` /
// `TimeAwareEnvironment` and the existing three.js helpers without a coordinate
// remap. `CarKinematicState` is structurally assignable to `ScenarioState`.

/** Minimal vehicle pose the scenario layer reads by name. `heading` is the XZ
 *  bearing (rad), `speed` is signed (negative = reverse), `t` is absolute sim
 *  time (seconds). Acceptance / regions reference these fields declaratively. */
export interface ScenarioState {
  x: number;
  z: number;
  heading: number;
  speed: number;
  t: number;
}

/** Something whose pose can be predicted at a future time — the abstraction
 *  behind every dynamic, agent-relative region (intercept / overtake / cone).
 *  Mirrors `kinocat/predict`'s `Predict<T>`: `predict(t)` returns null outside
 *  the predictor's validity horizon. */
export interface RegionAgent {
  readonly id: string;
  predict(t: number): ScenarioState | null;
}

/** The spatial primitive. One interface unifies three semantics:
 *  state membership (`contains`), edge crossing (`crossed`, for thin gates a
 *  fast car straddles between nodes), and time-varying / agent-relative
 *  (`dynamic`). `kind` keeps regions serializable + visualizable. */
export interface Region {
  /** Serializable discriminator (e.g. 'at', 'near', 'gate', 'cone'). Drives
   *  the deterministic visualizer + validation diagnostics. */
  readonly kind: string;
  /** Stable, parameter-derived signature (`kind` + its construction params /
   *  agent id). Two regions are STRUCTURALLY EQUAL iff their keys match — this
   *  is what makes goals hashable, diffable, and dedupable even though regions
   *  carry closures. */
  readonly key: string;
  /** Membership. May depend on `t` for dynamic regions. */
  contains(s: ScenarioState, t?: number): boolean;
  /** Admissible LOWER BOUND on cost-to-go to ENTER this region from `s`.
   *  Heading-aware (Reeds-Shepp), not Euclidean, for oriented regions. */
  costToGo(s: ScenarioState, t?: number): number;
  /** Did the motion from->to cross this region? Required for gates and any
   *  region thinner than the motion-primitive step length. Optional: regions
   *  thicker than a step can rely on endpoint `contains`. */
  crossed?(from: ScenarioState, to: ScenarioState, t0?: number): boolean;
  /** A representative interior pose — the heuristic chain LB and the planner
   *  bridge ("aim the base env at the next guard") both need a concrete pose.
   *  For dynamic regions this is the pose at the predictor's reference time. */
  representative(): ScenarioState;
  /** True when `contains` / `costToGo` depend on `t` — pulls a clock dimension
   *  into the search (see the spec's section 8). */
  readonly dynamic: boolean;
}

/** A one-sided numeric bound. `{ max: 0 }` = "must stop"; `{ min: 8 }` = ">=8". */
export interface Bound {
  min?: number;
  max?: number;
}

/** Extra conjuncts a `reach` guard must satisfy beyond spatial membership.
 *  `speed`/`heading` read off the candidate state; `window`/`by` read its
 *  clock `t` (absolute sim time). */
export interface Acceptance {
  /** Speed band, m/s (signed). */
  speed?: Bound;
  /** Heading band, radians, as an absolute-angle arc [min, max] (wrap-aware). */
  heading?: Bound;
  /** Must satisfy within absolute-time window [t0, t1] (inclusive). */
  window?: [number, number];
  /** Deadline: must satisfy by absolute time t (<= by). */
  by?: number;
}

// ---------------------------------------------------------------------------
// Objective plane — compiles to the automaton.

/** Reach a region (the first time membership + acceptance hold). */
export interface ReachGoal {
  kind: 'reach';
  region: Region;
  accept?: Acceptance;
}
/** Ordered conjunction — each child satisfied in order (the racing backbone). */
export interface SeqGoal {
  kind: 'seq';
  goals: Goal[];
}
/** Unordered conjunction — all children satisfied, any order. */
export interface AllGoal {
  kind: 'all';
  goals: Goal[];
}
/** Disjunction — any one child satisfied (e.g. either open bay). */
export interface AnyGoal {
  kind: 'any';
  goals: Goal[];
}
/** Liveness / laps — `goal` repeatedly re-satisfied; no terminal accept, so
 *  the objective becomes bounded progress maximization (see the env horizon). */
export interface RepeatGoal {
  kind: 'repeat';
  goal: Goal;
}

export type Goal = ReachGoal | SeqGoal | AllGoal | AnyGoal | RepeatGoal;

// ---------------------------------------------------------------------------
// Constraint plane — compiles to invariants (successor pruning).

/** A region that must NEVER be entered. */
export interface AvoidInvariant {
  kind: 'avoid';
  region: Region;
}
/** A region the vehicle must ALWAYS be inside (e.g. the lot / corridor).
 *  `scope` (authored via the builder's `.while(region)`) restricts the
 *  invariant to states inside that region only. */
export interface MaintainInvariant {
  kind: 'maintain';
  region: Region;
  scope?: Region;
}

export type Invariant = AvoidInvariant | MaintainInvariant;

// ---------------------------------------------------------------------------
// Cost plane — weighted edge-cost terms summed into g.

export interface CostTerm {
  readonly name: string;
  weight: number;
  /** Additive g contribution for the motion from->to over dt seconds. Must be
   *  >= 0 (a penalty), so the summed g stays a valid A* cost. */
  edgeCost(from: ScenarioState, to: ScenarioState, dt: number): number;
}

// ---------------------------------------------------------------------------
// A complete scenario.

export interface Scenario {
  name: string;
  start: ScenarioState;
  /** Objective plane. */
  goal: Goal;
  /** Constraint plane. */
  invariants?: Invariant[];
  /** Cost plane. */
  prefer?: CostTerm[];
  /** Dynamic-region agents, by id — referenced by within/ahead/cone/... */
  agents?: RegionAgent[];
}
