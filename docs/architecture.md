# Architecture: the five seams

kinocat is one generic search core behind five named seams. Everything
domain-specific — cars, humanoids, momentum humanoids, aircraft, and whatever
you add next — lives in implementations of these seams; the planner itself
(`core/src/planner/ighastar.ts`) never changes.

```
                 ┌────────────────────────────────────────────┐
                 │  IGHA* core     plan<State>(req, deadline)  │
                 │  anytime · multi-resolution · time-extended │
                 └──────────────────┬─────────────────────────┘
                                    │ five methods, nothing else
                 ┌──────────────────▼─────────────────────────┐
   Seam 1        │           Environment<State>               │
                 │  createNode · succ · heuristic ·            │
                 │  checkValidity · reachedGoalRegion          │
                 └───┬──────────────┬─────────────────┬───────┘
                     │              │                 │
   Seam 2       ForwardSim<S>   Seam 3            Seam 5 (wrappers)
   dynamics     + characterize  world seam        TimeAware · MultiGoal ·
                (primitives)    NavWorld (2.5D)   Scenario — compose over
                                AirspaceWorld(3D) any base Environment
                                    │
   Seam 4                      Predict<T> — everything time-varying:
                               moving obstacles, published plans,
                               affordances, zones
```

The proof the seams are real: four agent domains ship on them today —
`VehicleEnvironment` (car), `HumanoidEnvironment` (step-based walker),
`MomentumHumanoidEnvironment` (inertial person), `AircraftEnvironment`
(genuinely 3D: searched altitude, pitch, roll, OBB collision) — and the
fourth was added with zero planner-core edits.

## Seam 1 — `Environment<State>` (`core/src/environment/types.ts`)

The planner's ONLY coupling to a domain. Contract fine print (each point is
enforced by the conformance kit, `kinocat/testing`):

- **`succ` must set `g`, `h`, `f`** on every returned node, with
  `g = parent.g + edge.cost`, `f = g + h`, and `edge.cost > 0` (termination).
- **`index.length === levels`** on every node — one packed dominance cell per
  resolution level, coarse → fine.
- **`hash` keys the exact state class.** Identical states must hash
  identically (optimal dedup is unsound otherwise). Include every Markov
  dimension (the momentum humanoid hashes speed and velocity direction; the
  car hashes speed and gear-relevant time bucket). Do NOT include time in a
  static environment's hash — with cost = time, earliest arrival dominates,
  and a time bucket lets every cell be re-expanded once per arrival time
  (an unbounded ladder; the momentum humanoid learned this the hard way).
  Time-distinctness is `TimeAwareEnvironment`'s job.
- **`heuristic` must be admissible** (never overestimate cost-to-go) and
  should be consistent. Consistency traps discovered by the kit, worth
  knowing before you design one:
  - A velocity-*projection* bound (speed toward the goal) is inconsistent:
    motion rotates the goal bearing, so the projection can improve faster
    than any acceleration cap allows.
  - Any speed-*rewarding* bound is inconsistent with bucketed primitives:
    applying a primitive characterized from a canonical start-speed bucket
    "teleports" the state's speed to the bucket, jumping more than
    `maxAccel·dt`. Speed-independent bounds (Euclidean/maxSpeed,
    Reeds-Shepp length/maxSpeed) are immune because the forward sim clamps
    speed during rollout, so an edge of duration T closes at most
    `maxSpeed·T` of distance.
  - Off-mesh links / affordances whose cost undercuts the straight-line
    travel time break admissibility of a Euclidean heuristic — price them
    honestly.
- **`checkValidity` is deliberately lenient about the start** under the
  time-aware wrapper: rejecting a start because a moving obstacle overlaps
  it at t=0 makes multi-agent scenes collapse (every clustered agent bails
  simultaneously). See the comment in `time-aware.ts` — validate statics,
  let successor pruning handle time conflicts.
- **Optional hooks (`attachRecorder`, `progress`) must be forwarded by
  composing wrappers**, and `progress` only when the base has it (the
  planner pays for the hook only when the method exists). `succ`'s optional
  `level` argument must be forwarded too — dropping it silently disables
  per-level primitive sets (this was a real bug in `TimeAwareEnvironment`).

## Seam 2 — `ForwardSim<S>` (`core/src/primitives/`, `core/src/agent/`)

Dynamics are a pure function `(state, controls, dt) => state`, and **the sim
is the single definition of what a primitive can express**. Controls are
SETPOINTS; every state variable evolves continuously from its actual current
value under the agent's envelope. There is no "magical" instant maneuver: a
primitive commanding a bank gets `maxRollRate·dt` of it per step, exactly as
a primitive commanding thrust gets displacement (see `aircraftForwardSim` —
`Infinity` rates recover a quasi-static snap model). Don't hand-add per-DOF
features to environments; put the physics in the sim and let primitives
inherit it.

**Applying primitives — two strategies, one contract.** An environment's
`succ()` must produce successors that are what the sim actually yields from
the parent's ACTUAL state (`checkSuccessorFidelity` in `kinocat/testing`
verifies this):

- **Live rollout** (aircraft): integrate the sim per substep inside `succ()`.
  Exact by construction — right whenever the sim is cheap relative to the
  collision narrowphase it feeds (a few dozen flops vs an OBB SAT test), or
  when the sim's output depends on continuous state dims (rate-limited
  attitude) that a cache would quantize. The aircraft tried an
  attitude-bucketed cache first; its teleport error clipped near-margin
  obstacles, and the cache was buying nothing.
- **Cached characterization** (car, momentum humanoid): `characterize()`
  rolls the sim from canonical start-bucket states across a control grid,
  recording local-frame samples that `succ()` rigid-transforms by the node's
  pose. Right when rollouts are expensive or the start-dependence is
  low-dimensional (speed buckets). Two contracts apply:
  - **Equivariance**: rigid transform is only valid when the sim is
    translation- and yaw-equivariant — no absolute-position or
    absolute-heading dependence (no global wind, no position-dependent
    grip).
  - **Bucket coverage**: cache per start bucket over every state dim the
    sim's output depends on; the bucket teleport is a declared, measured
    tolerance in the domain's fidelity hook, not an invisible lie. Its
    heuristic consequence is documented under Seam 1.

## Seam 3 — the world seam (two fidelities, pick one per domain)

- **`NavWorld`** (`environment/nav-world.ts`): 2.5D polygon surface. Planning
  is in XZ; floor Y is derived from polygon containment. Footprint/segment
  clearance queries; off-mesh links. Ground agents live here.
  `InMemoryNavWorld` is the zero-dep implementation; `adapters/navcat` maps
  it onto a real navmesh.
- **`AirspaceWorld`** (`environment/airspace-world.ts`): 3D free space. The
  agent is an oriented box (yaw+pitch+roll); static AABBs via SAT + uniform
  grid; `clearAABB` broadphase; moving spherical zones with exact
  sphere-vs-OBB narrowphase.

## Seam 4 — `Predict<T>` (`core/src/predict/`)

`(t) => T | null` is the single abstraction for everything time-varying.
`MovingObstacle` predictions may carry an optional `y`: when both the
obstacle and the state have altitude, the collision proxy is a 3D sphere;
otherwise the planning-plane circle (a y-less obstacle against a 3D agent is
an infinite vertical cylinder — conservative). `MovingZone` is structurally a
`MovingObstacle` whose predictions always carry `y` — the same object can be
given to `TimeAwareEnvironment` (fast proxy, dominance participation,
broadphase) or to `AirspaceWorld.zones` (exact OBB narrowphase).

Affordances (`Affordance<S>`, `AffordanceRegistry<S>`) are extra edges, typed
to the agent state — any domain can publish them.

## Seam 5 — composing wrappers (`TimeAware`, `MultiGoal`, `Scenario`)

Each wraps any base `Environment` and returns another `Environment`:

- **`TimeAwareEnvironment`** adds time to the dominance keys and exact hash,
  prunes successors against predicted moving obstacles at their arrival
  times, and generates affordance edges. Works for any `{x, z, t(, y)}`
  state — car, humanoid, momentum humanoid, aircraft.
- **`MultiGoalEnvironment`** turns one search into an ordered gate sequence.
  It deliberately does not surface a `progress` score (its natural score,
  gateIndex, would change partial-result semantics; revisit as an opt-in).
- **`ScenarioEnvironment`** drives the search from a compiled goal automaton
  (reach/seq/all/any/repeat) with invariants and cost terms, and surfaces
  lap progress via the `progress` hook.

Composition notes: `AircraftEnvironment` bakes a time bucket into its own
hash for standalone use; pass `timeInHash: false` when wrapping it with
`TimeAwareEnvironment` (leaving it on is sound but redundant).

## Proving a domain: `kinocat/testing`

`runConformance(harness)` is the definition of "this environment works":
heuristic consistency and admissibility, successor invariants (positive
costs, monotone time, g/h/f bookkeeping, index arity), hash stability,
cross-instance determinism, anytime monotonicity, budgeted scenario
solvability, and — when the harness supplies re-simulation hooks —
successor fidelity (succ()'s output vs the forward sim run from the actual
parent state; exact for live-rollout environments, a declared tolerance for
bucket-cached ones). All framework-agnostic (no test-runner dependency) so
game projects can run it against their own domains. See
`docs/adding-a-domain.md` for the recipe and
`core/test/conformance/*.conformance.test.ts` for the four in-repo
harnesses.
