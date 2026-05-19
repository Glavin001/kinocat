# kinocat — Kinodynamic Planning for the Web

A TypeScript library for NPC vehicle navigation in browser-based games, built on top of [navcat](https://github.com/isaac-mason/navcat). Handles physics-aware planning over (x, y, θ, v) state with motion primitives learned from the host physics engine, opportunistic dynamic affordances, and ballistic jump edges — without leaving the browser.

---

## 1. Project Overview

navcat solves walkable-surface navigation for humanoid-style agents: 2D pathfinding over a polygon navmesh with string-pulled paths. That model breaks for vehicles, which have non-holonomic motion constraints (minimum turning radius, forward/reverse semantics, velocity-dependent feasibility). It also breaks for opportunistic AI behaviors — using moving ramps, drafting, predicted-destruction routing — that need to reason about the world in space and time.

kinocat fills that gap. It provides kinodynamic planning for vehicles via Hybrid A*, motion primitive libraries characterized per-vehicle from physics rollouts, ballistic jump edges precomputed at build time, and a local trajectory sampler that handles dynamic obstacles and opportunistic affordances through a uniform space-time intersection mechanism.

It is designed as a runtime dependency on navcat, not a fork. navcat provides the static topology (navmesh + off-mesh connections) and collision representation (CompactHeightfield). kinocat consumes those artifacts and adds the kinodynamic layer on top.

---

## 2. Goals

**Primary goal.** Enable NPC vehicles to plan and execute physically-feasible trajectories through complex 3D game environments, including reverse maneuvers, terrain-aware routing, dynamic obstacle avoidance, and ballistic jumps.

**Secondary goals.**

- Be web-native: pure TypeScript, tree-shakeable, no Emscripten, no Boost-derived bundle weight, no Node-only dependencies in the runtime path.
- Be physics-engine-agnostic at the abstraction boundary. Adapters for Rapier ship in-tree; the core never imports a physics library directly.
- Be navcat-aligned in style: JSON-serializable data structures, blocks-style high-level wrappers users can eject from, generic algorithms with pluggable filters and cost functions.
- Be small. Target 3500–5000 lines of TypeScript including tests, not 30,000.
- Be tunable without recompilation. Cost weights, primitive counts, replanning frequency are all runtime parameters.

**Use case anchor.** NPC vehicles in physics-based destruction/sandbox games where vehicles have fixed kinematic models known at design time, environments have static topology plus dynamic objects, and AI must produce believable driving including using ramps and reacting to moving hazards.

---

## 3. Non-Goals (Out of Scope)

These are deliberately excluded. Each could be added later as a separate package, but is not part of v1.

- **Robotic arm planning.** PRM in joint space is a different algorithm, different use case. Out of scope; revisit as a sibling package (`armcat`?) when there's a concrete game need.
- **Player-built/dynamically-assembled vehicles.** Kinodynamic planning requires a known forward model; user-assembled contraptions don't provide one. Such vehicles fall back to navcat + steering controller, which is outside kinocat.
- **Trajectory optimization (QP/SQP-based).** Sampling-based local planning covers the use case at game-appropriate quality. No quadratic programming solvers.
- **Model Predictive Control.** Pure pursuit with curvature-aware speed is sufficient. No MPC.
- **Learned dynamics models.** Physics-characterized motion primitives are deterministic and good enough. No neural network forward models.
- **Multi-modal probabilistic prediction of dynamic objects.** Linear and constant-acceleration extrapolation plus plan-sharing among friendly NPCs covers realistic cases. No mixture-density nets, no Gaussian processes.
- **RRT, RRT*, BIT*, PRM*, FMT* for vehicles.** Hybrid A* covers the vehicle case; alternative planners are not needed.
- **Mid-air controllability modeling.** Aerial phase is modeled as ballistic; the planner does not search over mid-air control inputs.
- **Multi-agent joint optimization.** Cooperative behavior emerges from plan-sharing through a registry and frequent replanning, not from explicit joint solving.
- **State machine for execution modes.** The executor runs one controller continuously; physics decides effectiveness. Replanning handles "stuck", "aerial", "landing" implicitly.
- **HD map / road network primitives.** Lane modeling, traffic rules, intersection semantics are game-specific concerns built on top of kinocat, not in it.

---

## 4. Design Principles

These are the load-bearing decisions that shape the entire library. Each one was settled deliberately.

**The planner reasons in plans, the executor reasons in physics, replanning bridges them.** No execution state machine. The executor runs a single tracking controller every tick; physics applies its outputs with whatever authority it has. The planner re-runs periodically (and on significant state divergence) from current actual physics state. "Stuck", "airborne", and "recovering" are not modes — they are states from which the planner produces appropriate plans (reverse maneuvers, ballistic-continuation plans, etc.) without special-case logic.

**Static topology and dynamic prediction live at different layers.** The navmesh and its off-mesh connections handle geometry that doesn't change at human-perception timescales. Moving vehicles, debris, transient affordances, other NPCs' published plans live in a separate tracked-object registry queried only by the local trajectory sampler. The global Hybrid A* plan does not enumerate dynamic objects.

**`predict(t) → state` is the abstraction boundary for everything dynamic.** Linear extrapolation, constant-acceleration, plan-registry lookup, physics rollout — all implement the same interface. The planner never knows what kind of predictor produced a function; it just queries it.

**Avoidance and exploitation are the same mechanism.** Space-time intersection between a candidate trajectory and a tracked object's predicted trajectory produces a cost contribution. Negative for hazards, positive for affordances, plus a state transform when applicable. New behaviors are new cost contributors, not new algorithms.

**Discretize at build time, search at runtime.** Where literature uses continuous constraints, we use enumerated edges. Ramps become tens of off-mesh connections at speed × heading buckets. Vehicles get motion primitive libraries characterized once per asset. Standard graph search then finds the right plan; no constraint-aware planner needed.

**navcat is a dependency, not a fork.** We extend through metadata in `area`/`flags`, custom `QueryFilter` implementations, and the existing off-mesh connection mechanism. We never modify navcat internals.

**One algorithm per planning problem.** Hybrid A* for vehicle global planning. Trajectory sampling for local planning. Pure pursuit for tracking. No alternates "in case." Resist additions until a specific use case provably requires them.

---

## 5. How It Leverages navcat

kinocat treats navcat as the topology and collision foundation. Specific integration points:

**Navmesh as collision input.** The Hybrid A* collision checker reads the `NavMesh` to determine which states and motion-primitive arcs are inside walkable polygons. We use navcat's polygon adjacency and `findNearestPoly`-style queries through public APIs; we do not assume internal layout.

**CompactHeightfield as fast clearance lookup.** During navmesh generation navcat builds a `CompactHeightfield` with a distance field (via `buildDistanceField`). The harness retrieves this intermediate via the `generateSoloNavMesh`/`generateTiledNavMesh` result's `intermediates` field and uses it directly for O(1) clearance queries during Hybrid A* expansion — much faster than polygon containment tests when sweeping arcs.

**navmesh node-path distance as Hybrid A* heuristic.** Dolgov's Hybrid A* paper recommends taking the max of two heuristics: "non-holonomic without obstacles" (Reeds-Shepp distance) and "holonomic with obstacles" (2D shortest path respecting walls). We compute the second via navcat's `findNodePath` + path-cost accumulation. This dramatically improves Hybrid A* search quality near obstacles without requiring our own A*-on-grid implementation.

**Off-mesh connections as jump-edge carriers.** navcat's `OffMeshConnectionParams` schema — `{ start, end, direction, radius, area, flags }` — is used as-is. We encode jump-edge type in the `area` field (e.g. `AREA_BALLISTIC_JUMP = 0x10`) and store extended metadata (entry velocity range, entry heading tolerance, exit velocity vector, precomputed trajectory polyline) in a parallel `JumpEdgeMetadata` map keyed by connection ID. The example pattern from navcat's `example-off-mesh-connections.ts` — `getNodeByRef(navMesh, curRef).area` to dispatch in `QueryFilter.getCost` — extends directly.

**Tile-rebuild infrastructure for destruction events.** navcat's tiled-navmesh mode supports `removeTile` + `addTile` rebuilds on geometry changes. When the game world is destructed (e.g. a building collapses), we trust navcat's tile rebuild and only invalidate kinocat's per-tile precomputed data (jump edges sourced from ramps in the affected tile, motion primitive heuristic caches if any). The `example-fps-dynamic-navmesh.ts` pattern of tile-throttled rebuild with per-tile object tracking is the canonical reference.

**`QueryFilter` for global plan customization.** kinocat ships a `vehicleQueryFilter(vehicleModel)` factory that produces a `QueryFilter` whose `passFilter` rejects edges incompatible with the vehicle (e.g. ballistic jumps whose entry-speed range exceeds vehicle capability) and whose `getCost` produces vehicle-appropriate edge costs (reverse penalty, terrain cost, jump-edge time cost). The factory uses the standard navcat `QueryFilter` interface — no extensions to navcat are required.

**Three.js helpers via `navcat/three`.** Debug visualization of plans, primitives, and jump trajectories follows navcat's helper pattern (`createNavMeshHelper`, `createNavMeshOffMeshConnectionsHelper`). kinocat's debug helpers ship under `kinocat/three`.

---

## 6. Architecture

Eight components, layered top to bottom by who depends on whom.

**1. Math primitives (`kinocat/curves`)** — Reeds-Shepp and Dubins curves. Pure analytical math, no dependencies beyond the core math module. Standalone, useful independently of the planner. Used as Hybrid A* heuristic and terminal "shot to goal" expansion.

**2. Motion primitive system (`kinocat/primitives`)** — primitive representation, library format, lookup index, and the characterization harness. A primitive is a feasible short trajectory the vehicle is known to execute. The harness consumes a `ForwardSim` function (the game's physics, wrapped) and produces a library by sweeping controls. Library is JSON-serializable; built once per vehicle asset.

**3. Vehicle model (`kinocat/vehicle`)** — small per-vehicle config: kinematic class (bicycle / skid-steer / etc.), max forward/reverse speed, reverse-cost multiplier, primitive library reference, optional damage-state alternates. Read by the planner and the executor.

**4. Hybrid A* (`kinocat/hybrid-astar`)** — global vehicle planner. Generic graph search over (x, y, θ, v) state. Pluggable collision check, pluggable heuristic, motion-primitive-based expansion. Outputs a sequence of typed edges (`DriveEdge` for primitive sequences, `JumpEdge` for off-mesh ballistic transitions).

**5. Jump-edge precomputation (`kinocat/jumps`)** — build-time tool that identifies ramps in a level, sweeps takeoff speed × heading buckets per ramp, simulates ballistic trajectories via the game's physics, validates landing zones against the navmesh, and emits `OffMeshConnectionParams` + parallel `JumpEdgeMetadata` records. Runs offline; output is serialized alongside the navmesh.

**6. Local trajectory sampler (`kinocat/sampler`)** — runs at executor tick rate (default 10 Hz). Generates N candidate short-horizon trajectories by branching motion-primitive sequences from the current state. Scores each against the cost function. Picks the best; first primitive of the best becomes the current execution target.

**7. Cost function and tracked-object registry (`kinocat/cost`)** — the trajectory sampler's scoring infrastructure. Cost function is a registry of contributor functions; built-ins are `pathTracking`, `spaceTimeCollision`, `affordanceUsage`, `smoothness`, `speedPreference`. The tracked-object registry holds dynamic obstacles, affordances, and other NPCs' published plans, all behind the `predict(t)` interface. This is the open extension surface — most game-specific behavior is a new cost contributor.

**8. Execution layer (`kinocat/execute`)** — pure pursuit with curvature-aware speed control, plus a divergence detector for triggering replans. No state machine; no mode logic. Reads the current target primitive, outputs throttle and steering each physics tick. Reports divergence events to the planner.

**Adapters (`kinocat/adapters/*`)** — navcat collision checker, navcat heuristic, Rapier `ForwardSim` wrapper, Three.js debug visualizers. Each adapter is small (~50–150 lines) and isolates the dependency.

---

## 7. Data Structures and API Contracts

These are the load-bearing types. Where a type is settled, it's specified concretely; where it's open, that's noted.

### Core state

```ts
// Vehicle state in planning space. The y coordinate is in the planar frame
// (it's the heightmap value, not the search dimension).
type VehicleState = {
  x: number;       // world X (planar)
  z: number;       // world Z (planar)  
  heading: number; // radians, 0 = +X
  speed: number;   // signed; negative = reversing
  // y is queried from the navmesh/heightmap on demand; not part of search state.
};

// 4D quantized lattice cell for duplicate detection.
type LatticeIndex = {
  ix: number;
  iz: number;
  iheading: number;  // discretized to e.g. 16 buckets
  ispeed: number;    // discretized to e.g. 4 buckets (heavy-reverse, light-reverse, slow-fwd, fast-fwd)
};
```

### Motion primitives

```ts
type MotionPrimitive = {
  id: number;
  
  // Start state class — primitive is only valid when starting in a state
  // matching this speed bucket and (optionally) heading offset.
  startSpeedBucket: number;
  
  // The control inputs. Opaque to kinocat; meaningful only to ForwardSim.
  // For a bicycle model this might be [steerAngle, throttle, durationSec].
  controls: number[];
  
  // Sampled trajectory in vehicle-local coordinates. Used for collision
  // checking and execution. Each sample is (dx, dz, dheading, speed, t).
  trajectory: Float32Array;  // packed; length = samples * 5
  
  // End-state offset in vehicle-local frame.
  endOffset: {
    dx: number;
    dz: number;
    dheading: number;
    endSpeed: number;
  };
  
  // Wall-clock duration of executing this primitive.
  duration: number;
};

type MotionPrimitiveLibrary = {
  vehicleId: string;
  
  // Indexed by speed bucket: primitives applicable from each bucket.
  byStartBucket: MotionPrimitive[][];
  
  // Speed bucket boundaries. byStartBucket[i] applies when
  // current speed falls in [speedBuckets[i], speedBuckets[i+1]).
  speedBuckets: number[];
  
  // Lattice resolution this library was built against.
  lattice: {
    cellSize: number;        // world meters
    headingBuckets: number;
    speedBuckets: number;
  };
};
```

### Characterization harness

```ts
// The single abstraction that decouples kinocat from any specific physics engine.
type ForwardSim = (
  state: VehicleState,
  controls: number[],
  dt: number,
) => VehicleState;

type CharacterizeOptions = {
  forwardSim: ForwardSim;
  controlSchema: ControlSchema;  // describes what each control input is, ranges, defaults
  speedBuckets: number[];
  headingBuckets: number;
  primitivesPerBucket: number;   // typical: 10-30
  primitiveDuration: number;     // typical: 0.3-0.6 seconds
};

function characterize(opts: CharacterizeOptions): MotionPrimitiveLibrary;
```

### Hybrid A*

```ts
type CollisionCheck = (
  primitive: MotionPrimitive,
  fromState: VehicleState,
) => boolean;  // true if collision-free

type Heuristic = (
  state: VehicleState,
  goal: VehicleState,
) => number;  // admissible lower-bound cost-to-go

type PlanRequest = {
  start: VehicleState;
  goal: VehicleState;
  goalTolerance: {
    position: number;       // meters
    heading: number;        // radians, optional (e.g. parking vs arrival)
    speed?: number;
  };
  vehicleModel: VehicleModel;
  collisionCheck: CollisionCheck;
  heuristic: Heuristic;
  maxExpansions?: number;   // safety bound; default 100000
  reverseCostMultiplier?: number;  // default 2.0
  directionChangePenalty?: number;  // default 0.5 (seconds-equivalent)
};

type PlannedEdge =
  | { type: 'drive'; primitive: MotionPrimitive; startState: VehicleState; endState: VehicleState }
  | { type: 'jump'; connectionId: number; entryState: VehicleState; exitState: VehicleState; trajectory: Float32Array };

type Plan = {
  edges: PlannedEdge[];
  totalCost: number;
  totalDuration: number;
};

function plan(req: PlanRequest): Plan | null;
```

### Jump edges

```ts
type JumpEdgeMetadata = {
  connectionId: number;  // navcat off-mesh connection ID
  
  // Entry constraints.
  entrySpeedRange: [number, number];
  entryHeading: number;          // radians; vehicle frame
  entryHeadingTolerance: number; // radians
  
  // Exit state after ballistic phase.
  exitState: VehicleState;
  
  // Sampled ballistic trajectory in world coordinates. (x, y, z, t) packed.
  trajectory: Float32Array;
  
  // Wall-clock duration of the ballistic phase.
  duration: number;
};

// Per-level: identify ramps, sweep, simulate, emit connections + metadata.
type PrecomputeJumpsInput = {
  navMesh: NavMesh;                 // from navcat
  ramps: RampDescriptor[];          // game-provided
  vehicleModel: VehicleModel;       // sweep is per-vehicle
  forwardSim: ForwardSim;           // for ballistic simulation
  speedBuckets: number[];           // default 5-10 evenly spaced
  headingBuckets: number;           // default 4-8
  navMeshHalfExtents: Vec3;         // for landing-zone validation
};

type PrecomputeJumpsResult = {
  connections: OffMeshConnectionParams[];  // ready to addOffMeshConnection
  metadata: Map<number, JumpEdgeMetadata>; // keyed by connection ID after add
};
```

### Tracked-object registry and predictors

```ts
type Predict = (t: number) => ObjectStateAtTime | null;

type ObjectStateAtTime = {
  position: Vec3;
  heading?: number;
  velocity?: Vec3;
  footprint: BoundingShape;  // for collision checks
};

type TrackedObject = {
  id: number;
  predict: Predict;
  // null entryConstraints/exitTransform = pure obstacle.
  // present = affordance.
  entryConstraints?: EntryConstraint;
  exitTransform?: (vehicleState: VehicleState) => VehicleState;
  // Validity window. predict() may also return null outside this.
  validFrom: number;
  validTo: number;
};

type EntryConstraint = {
  speedRange?: [number, number];
  headingTolerance?: number;
  // approach must be within this distance of the affordance position at intersection time
  positionTolerance: number;
};

type TrackedObjectRegistry = {
  add(obj: TrackedObject): void;
  remove(id: number): void;
  // Query: objects whose validity overlaps [tStart, tEnd] and whose
  // predicted positions are within radius of any point along a candidate trajectory.
  queryAlong(trajectory: SampledTrajectory, radius: number): TrackedObject[];
};

// Predictor factories.
function constantVelocity(p: Vec3, v: Vec3): Predict;
function constantAcceleration(p: Vec3, v: Vec3, a: Vec3): Predict;
function fromPublishedPlan(npcId: string, registry: PlanRegistry): Predict;
function fromPhysicsRollout(forwardSim: ForwardSim, initialState: VehicleState, controls: number[], dtStep: number): Predict;
```

### Local trajectory sampler

```ts
type SampleRequest = {
  currentState: VehicleState;
  globalPlan: Plan;                  // for path-tracking cost
  trackedObjects: TrackedObjectRegistry;
  primitiveLibrary: MotionPrimitiveLibrary;
  horizon: number;                   // seconds; default 1.5
  candidateCount: number;            // default 30
  costFunction: CostFunction;
};

type CostFunction = {
  contributors: CostContributor[];
  weights: Record<string, number>;
};

type CostContributor = {
  name: string;
  evaluate: (candidate: SampledTrajectory, req: SampleRequest) => number;
};

type SampleResult = {
  bestCandidate: SampledTrajectory;
  bestPrimitive: MotionPrimitive;    // the immediate-execution target
  scores: { name: string; total: number }[];  // for debugging
};

function sampleLocal(req: SampleRequest): SampleResult;
```

### Built-in cost contributors

```ts
// Negative cost (penalty) for predicted intersection with hazards;
// positive cost (reward) for predicted intersection with affordances.
const spaceTimeCollision: CostContributor;

// Penalty for deviation from the global plan path.
const pathTracking: CostContributor;

// Penalty for direction changes within the candidate (reverse-forward etc.).
const smoothness: CostContributor;

// Soft preference toward a target speed (NPC personality dial).
const speedPreference: CostContributor;

// Reward for using affordances; integrates with affordanceUsage state transform.
const affordanceUsage: CostContributor;
```

### Execution

```ts
type ExecutorInput = {
  currentState: VehicleState;
  targetPrimitive: MotionPrimitive;
  vehicleModel: VehicleModel;
  controlSchema: ControlSchema;
};

type ExecutorOutput = {
  controls: number[];     // same shape as primitive controls
  divergence: number;     // meters from expected position
};

function pureSpursuit(input: ExecutorInput): ExecutorOutput;

// Triggers replan if divergence exceeds threshold OR periodic refresh elapses.
type ReplanTrigger = {
  divergenceThreshold: number;  // meters
  refreshIntervalMs: number;
};

function shouldReplan(divergence: number, lastReplanMs: number, trigger: ReplanTrigger): boolean;
```

---

## 8. Build-Time Pipeline

Per game level, run once during asset processing or level load:

1. **navcat generates the navmesh** from level geometry using `generateTiledNavMesh`. Standard navcat flow; no kinocat involvement.
2. **Ramp detection** identifies game-tagged ramp surfaces (game-provided; level designers tag with metadata).
3. **For each (vehicle asset, level) pair:**
   - For each ramp, sweep takeoff speeds (e.g. 5 buckets between 5–20 m/s) and approach headings (e.g. 4 buckets within ±30° of ramp normal).
   - For each combination, simulate ballistic trajectory using the game's physics with the vehicle treated as a point mass on takeoff.
   - Validate the landing point: project onto navmesh, check the polygon is walkable, check the landing velocity is in a reasonable range.
   - Emit a `OffMeshConnectionParams` with `area = AREA_BALLISTIC_JUMP` and a `JumpEdgeMetadata` record.
4. **Serialize**: navmesh tiles + off-mesh connections + jump-edge metadata + per-vehicle motion primitive libraries → game asset bundle.

Per vehicle asset, run once when the vehicle is designed (or per game-build):

1. **Spawn the vehicle in a test scene** (flat empty plane, no obstacles).
2. **Run the characterization harness**: sweep controls across speed and steering ranges, record resulting trajectories.
3. **Extract effective parameters**: minimum turning radius, max forward/reverse speed, acceleration profile.
4. **Construct motion primitive library**: cluster control inputs into a canonical set (typically 20–60 primitives total across speed buckets), quantize endpoints to lattice cells.
5. **Serialize** the library alongside the vehicle asset.

Optionally, per (vehicle, level) pair: precompute a Reeds-Shepp distance table specific to the vehicle's turning radius for faster runtime heuristic evaluation. Optional; the closed-form formula works at runtime.

---

## 9. Runtime Loop

Three loops at different rates, communicating through shared state.

**Physics tick (game-determined, typically 60 Hz).** Executor reads the current target primitive, computes pure-pursuit controls, sends them to the physics engine. Physics applies them. Updates divergence-from-plan metric.

**Sampler tick (default 10 Hz).** Local trajectory sampler runs:
- Generates 20–50 candidate short-horizon trajectories from current state via motion primitive branching
- Queries tracked-object registry for predictions over the horizon
- Scores each candidate against the cost function
- Sets the chosen candidate's first primitive as the executor's current target

**Planner tick (event-driven + periodic).** Global Hybrid A* runs when:
- Divergence-from-plan exceeds threshold (default 1.5 m)
- Refresh interval elapses (default 750 ms)
- Plan invalidation event fires (target moved, navmesh tile rebuilt in a tile the plan crosses)

When a planner tick fires, Hybrid A* searches from current actual state to goal and replaces the global plan. The sampler picks up the new plan on its next tick.

**Asynchronous: plan-sharing among NPCs.** Each NPC publishes its current plan to a shared registry. Other NPCs' samplers read from this registry via `fromPublishedPlan` predictors. No explicit coordination protocol — cooperation emerges from cost function alignment and the frequency of replanning.

---

## 10. Acceptance Criteria

The library is "done v1" when the following are demonstrably working in a reference integration:

**Correctness criteria.**
- A vehicle can plan a path from one side of a level to the other, including a section that requires reversing (e.g. a dead-end pocket the planner enters before backing out, OR a parking maneuver at the goal).
- Hybrid A* with the navcat heuristic produces visibly better plans (shorter, more direct) than Hybrid A* with Euclidean heuristic, on the same query.
- Reeds-Shepp terminal shot reliably completes plans where the goal is reachable analytically (no obstacles between current state and goal).
- Jump-edge precomputation produces a non-zero number of valid jumps for a level with explicit ramps, and Hybrid A* finds plans that use them when they're on the optimal route.

**Dynamic behavior criteria.**
- Local sampler avoids a single linearly-extrapolated moving obstacle by either slowing, accelerating, or swerving — chosen by cost function, not hard-coded.
- Local sampler uses a moving ramp affordance when its goal is on the far side of an otherwise-unreachable gap.
- Two cooperating NPCs (one with a ramp on its back, one needing the jump) coordinate without explicit negotiation, using only plan-sharing + replanning.

**Recovery criteria.**
- Vehicle pinned against a wall by external collision triggers replan; new plan begins with a reverse maneuver.
- Vehicle launched unexpectedly off a slope (knocked airborne) does not produce control errors; controller keeps running, physics handles the aerial phase, planner replans on landing.

**Performance criteria.**
- Hybrid A* plan from one side of a 50×50 m level to the other (with moderate obstacles) completes in < 50 ms on a recent laptop browser.
- Sampler tick (30 candidates, 1.5 s horizon, with one dozen tracked objects) completes in < 10 ms.
- Characterization harness for a single vehicle asset completes in < 30 seconds.
- Jump-edge precomputation for a level with 10 ramps and one vehicle completes in < 60 seconds.

**Size criteria.**
- Core library (excluding `kinocat/three`, `kinocat/adapters/rapier`, and tests) under 4000 lines of TypeScript.
- Total minified bundle of `kinocat` core + `kinocat/adapters/navcat` under 80 KB.

**API criteria.**
- All planner inputs and outputs are JSON-serializable (no class instances at the boundary).
- The library has zero non-peerDependency dependencies on physics engines or rendering libraries in the core path.
- Replacing the collision check is a one-function adapter swap.

---

## 11. Algorithm and Solution References

The algorithmic content of kinocat. Each is a single chosen approach; no alternates in v1.

**Hybrid A*.** Dolgov, Thrun, Montemerlo, Diebel — "Path Planning for Autonomous Vehicles in Unknown Semi-structured Environments" (Stanford AI Lab Technical Report, 2010). The canonical reference. Autoware's `hybrid_astar` implementation is a clean OSS port for cross-checking.

**Reeds-Shepp curves.** Reeds, Shepp — "Optimal paths for a car that goes both forwards and backwards" (Pacific Journal of Mathematics, 1990). Closed-form solution; 48 path types. Modern implementations (multiple OSS ports available) typically handle the case enumeration as a switch over the analytical solutions.

**Dubins curves.** Dubins — "On Curves of Minimal Length with a Constraint on Average Curvature, and with Prescribed Initial and Terminal Positions and Tangents" (American Journal of Mathematics, 1957). Six path types; forward-only car. Used as a simpler heuristic where reverse is irrelevant.

**State lattice planning** (for the motion primitive structure). Pivtoraiko, Kelly — "Generating Near Minimal Spanning Control Sets for Constrained Motion Planning in Discrete State Spaces" (IROS 2005), and subsequent papers. Used as conceptual grounding for the lattice + primitive library approach; we implement a simpler version.

**Trajectory sampling for local planning.** No single canonical paper; this is a pattern from autonomous driving practice (Frenet-frame sampling à la Werling 2010 is one variant, but we use a vehicle-frame primitive-branching variant). Gran Turismo Sophy and similar racing-AI work are practical references.

**Pure pursuit controller.** Coulter — "Implementation of the Pure Pursuit Path Tracking Algorithm" (Robotics Institute, CMU, 1992). The simplest tracking controller that works. With curvature-aware speed it handles cornering adequately.

**Off-mesh connections as discrete graph edges.** The pattern is used directly from navcat (see `example-off-mesh-connections.ts`). We extend with metadata for physics constraints but the routing mechanism is unchanged.

---

## 12. Tuning Parameters (Knobs, Not Architecture)

These have sensible defaults but are exposed for game-specific tuning. None of them are architectural decisions.

| Parameter | Default | Range |
|---|---|---|
| Hybrid A* state quantization (position) | 0.5 m | 0.2 – 1.0 m |
| Hybrid A* heading quantization | 22.5° (16 buckets) | 8 – 32 buckets |
| Hybrid A* speed quantization | 4 buckets | 2 – 8 |
| Hybrid A* max expansions before fail | 100000 | 10k – 500k |
| Reverse cost multiplier | 2.0 | 1.5 – 5.0 |
| Direction change penalty | 0.5 sec-equivalent | 0.2 – 2.0 |
| Primitive count per speed bucket | 15 | 6 – 30 |
| Primitive duration | 0.5 sec | 0.2 – 1.0 sec |
| Jump speed buckets per ramp | 5 | 3 – 12 |
| Jump heading buckets per ramp | 4 | 2 – 10 |
| Sampler candidate count | 30 | 10 – 80 |
| Sampler horizon | 1.5 sec | 0.5 – 3.0 sec |
| Sampler tick rate | 10 Hz | 5 – 30 Hz |
| Replan refresh interval | 750 ms | 250 – 2000 ms |
| Replan divergence threshold | 1.5 m | 0.5 – 5.0 m |
| Cost weights (per-contributor) | varies | 0.0 – 10.0 each |

---

## 13. Implementation Order

Strict dependency order. Each phase produces a working, testable subset of the library; the project has value at every phase boundary.

**Phase 1: Curves.** Implement Reeds-Shepp and Dubins as standalone modules with unit tests against published reference paths. Self-contained, no dependencies. *Useful immediately; ships as `kinocat/curves`.*

**Phase 2: Hybrid A* with analytical bicycle.** Implement Hybrid A* against a hardcoded analytical bicycle model and a stub collision checker (empty world or hand-rolled occupancy grid). Verify reverse maneuvers, three-point turns. *Internal milestone; not yet usable in a real game.*

**Phase 3: navcat collision adapter.** Wire Hybrid A* against `NavMesh` and `CompactHeightfield` from a real generated navmesh. Verify it plans through real level geometry. *First externally-useful milestone.*

**Phase 4: Characterization harness + primitive library.** Build the `characterize()` function. Run it on a Rapier-driven test vehicle. Swap the analytical-bicycle primitives for characterized ones. *kinocat now plans for any vehicle that has physics.*

**Phase 5: Jump-edge precomputation.** Build-time tool. Verify Hybrid A* finds plans that use jumps when they're optimal. *Vertical traversal works.*

**Phase 6: Local trajectory sampler.** Start with a single cost contributor (`pathTracking`). Verify the sampler tracks the global plan. *Continuous execution works.*

**Phase 7: Tracked-object registry + predictors.** Implement `constantVelocity` and `constantAcceleration`. Add `spaceTimeCollision` cost contributor. Verify dynamic obstacle avoidance. *Reactive behavior works.*

**Phase 8: Affordances.** Extend the registry with entry constraints and exit transforms. Add `affordanceUsage` cost contributor. Verify ramp affordance is used opportunistically. *Mario Kart-style behavior works.*

**Phase 9: Plan sharing.** Add the plan registry and `fromPublishedPlan` predictor. Verify cooperative jumps between two NPCs. *Multi-agent behavior works.*

**Phase 10: Polish.** Three.js debug helpers, documentation, examples mirroring navcat's example structure, tuning guide. *Ready for v1 release.*

The first three phases produce a usable kinodynamic planner for static-world NPC driving. Phases 4–5 cover the per-vehicle and vertical-traversal stories. Phases 6–9 are the rich-behavior layer. Phase 10 is what makes it adoptable beyond the original author.

A phase is "done" when its acceptance subset passes and it has a working example in `kinocat/examples` modeled on navcat's example structure.

---

## 14. Open Questions

Things we deliberately haven't decided yet, with notes on when they'll be settled.

**Whether to expose Hybrid A* in (x, y, θ) only vs (x, y, θ, v).** Including velocity in the search state is more accurate but multiplies the state space by the speed bucket count. Default: include velocity. Revisit during Phase 2 if search times are problematic.

**Coordinate system convention.** navcat follows OpenGL (right-handed, +Y up, Z-forward). kinocat will follow the same convention end-to-end, with the planning plane being XZ and Y queried from the navmesh as terrain height. Confirmed during Phase 1.

**Whether to support tile-local planning for very large worlds.** Hybrid A* on a 5km × 5km world is impractical; you'd want hierarchical decomposition (region-level coarse plan, tile-local Hybrid A*). Out of v1; revisit if game requirements demand.

**Cost-contributor registration mechanism.** Static list at construction time vs. dynamic add/remove. Default: static list constructed per-NPC, allowing different NPCs to have different cost functions. Confirm during Phase 6.

**Plan-registry write semantics.** Last-writer-wins vs. versioned vs. explicit ownership. For v1: last-writer-wins, single-threaded write. Multi-threading (Web Workers) is out of v1.

**Damage-state primitive switching.** Whether and how to swap motion primitive libraries when a vehicle loses parts. Default for v1: not supported; damaged vehicles fall back to navmesh-only navigation via game-side mode swap. Add later if needed.

---

## 15. Naming

Working name `kinocat` parallels `navcat`. Plays nicely with future companions if needed (`armcat` for manipulation, etc.). Open to alternatives at first release. The library identity isn't load-bearing in v1; the technical content is.
