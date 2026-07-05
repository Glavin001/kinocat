# Adding a motion body (agent domain)

kinocat's planner is generic; what you add is a **controllable motion body**
— a car, a person, an aircraft, a hovercraft, anything with state that
evolves under bounded controls. "Vehicle" is just the historical name of one
of them; in code the neutral term is an *agent domain*: the package of state
+ envelope + dynamics + environment that plugs a body into the IGHA* core.

**The canonical worked example is executable**:
[`core/test/examples/hovercraft.test.ts`](../core/test/examples/hovercraft.test.ts)
defines a complete new body — inertial, thrust-vectored, drifting, yaw
decoupled from velocity, unlike anything that ships — in one file, using
only public seams, and proves it with the conformance battery. This guide
walks that file; if the code and the doc disagree, the code wins (CI runs
it). Read `docs/architecture.md` first for the seam map and contract fine
print.

## The API at a glance — five things you define

| # | You define | Contract | Hovercraft example |
|---|---|---|---|
| 1 | **State** — a plain JSON object | Every dim the dynamics need to be Markov, plus absolute time `t` | `{x, z, heading, vx, vz, t}` |
| 2 | **Agent** — the body's envelope | Limits and size as data (`maxSpeed`, rates, radius/footprint, cost knobs) | `{radius, maxSpeed, maxThrust, maxYawRate, drag}` |
| 3 | **`ForwardSim<S>`** — the dynamics | `(state, controls, dt) => state`. Controls are SETPOINTS; state evolves from its current value; clamp to the envelope inside | `[thrustFrac, thrustAngle, yawFrac]` |
| 4 | **`Environment<State>`** — the planner's view | `createNode / succ / heuristic / checkValidity / reachedGoalRegion` + `levels` | `HovercraftEnvironment`, ~150 lines |
| 5 | **`DomainHarness`** — the proof | `runConformance(harness)` green defines "this body works" | battery + exact fidelity hook |

Affordances (`Affordance<YourState>`), moving obstacles, goal automata, and
multi-goal sequences then compose for free through the shared wrappers — no
per-body code.

## 1. State

Ask: *"do two bodies at the same pose with different X behave differently
going forward?"* If yes, X is state. The hovercraft carries a world-frame
velocity vector separate from `heading` because it drifts; the momentum
humanoid does the same because people strafe; the aircraft carries pitch
and roll because attitude is rate-limited. Always carry absolute `t`.

## 2. Agent (the envelope)

Plain data describing what the body *can do* — the analog of
`VehicleAgent` / `AircraftAgent` in `core/src/agent/types.ts`. Rates belong
here too (`maxRollRate`, `maxYawRate`): they are physical facts like
`minTurnRadius`, not features.

## 3. ForwardSim — the single definition of what a primitive can express

A **motion primitive is just "hold these control setpoints for T seconds
through the sim."** Whatever the sim integrates — displacement, attitude,
drift — is the primitive's effect. Do not script maneuvers or add per-DOF
special cases in the environment; put the physics in the sim and timing
falls out of the search (the aircraft begins its knife-edge roll *before*
the slot purely because `maxRollRate` lives in its sim).

Rules of thumb:
- Clamp to the envelope **inside** the sim — the planner can then never
  command the impossible.
- Keep it pure and deterministic (no randomness, no wall clock).
- Keep it translation- and yaw-equivariant (no absolute-position effects)
  if you ever want to cache characterized primitives.
- A learned model (`kinocat/learning`) plugs in as the same function type.

## 4. Environment

The five methods, generic quantization decisions, and one structural choice:

**How does succ() apply primitives?** (details: architecture doc, Seam 2)
- **Live rollout** — integrate the sim per substep inside `succ()`, from the
  node's ACTUAL state. Exact by construction. **Recommended default**: right
  whenever the sim is cheap next to your collision checks, and mandatory in
  spirit when the sim depends on continuous dims a cache would quantize
  (rate-limited attitude). The hovercraft and aircraft do this.
- **Cached characterization** — `characterize<S>()` + `crossRuns()`
  (`core/src/primitives/characterize.ts`) roll the sim once from canonical
  start buckets; `succ()` rigid-transforms the cached sweeps. Right when
  rollouts are expensive or start-dependence is coarse (car speed buckets,
  momentum-humanoid speed × direction buckets). The bucket teleport is a
  declared tolerance in your fidelity hook, not a hidden lie.

Non-obvious contract points (all enforced by the kit):
- `succ` sets `g/h/f` on every successor; `edge.cost > 0`; put **enough in
  `edge.data` to re-simulate the edge** (the hovercraft stores its controls
  array — zero-allocation, it's the shared action reference).
- `index.length === levels`; hash every Markov dim; **never hash time in a
  static environment** (earliest arrival dominates; `TimeAwareEnvironment`
  adds time when moving obstacles make it meaningful).
- Heuristic: admissible, ideally consistent. Read the traps in the
  architecture doc before designing a clever one — speed-rewarding bounds
  break with bucketed primitives; `distance / maxSpeed` is safe because the
  sim caps speed.
- Reuse scratch buffers for footprint placement (`placeFootprintInto`) —
  collision checks run millions of times per plan.
- Pick the world seam: `NavWorld` (2.5D surfaces) or `AirspaceWorld`
  (3D volume); define your own if neither fits — the planner never sees it.

## 5. Prove it: the conformance battery

```ts
import { runConformance, type DomainHarness } from 'kinocat/testing';

const harness: DomainHarness<HovercraftState> = {
  makeEnv: () => new HovercraftEnvironment(world(), AGENT), // deterministic
  sampleState: (rand) => ({ /* seeded; cover the whole envelope */ }),
  scenarios: [{ name: 'glide-across', start, goal, maxExpansions: 150_000 }],
  fidelity: {
    tolerance: 1e-9,                 // live rollout ⇒ exact
    angularFields: ['heading'],      // compared on the circle
    resimulate: (parent, edge) => {  // reconstruct controls from edge.data
      if (edge.kind !== 'hover') return null;
      /* roll the sim for primDuration/substeps from `parent` */
    },
  },
};
expect(runConformance(harness).failures).toEqual([]);
```

The battery checks heuristic consistency + admissibility, successor
invariants, hash stability, determinism, anytime monotonicity, budgeted
solvability, and — via your `fidelity` hook — that `succ()` faithfully
applies the sim from the actual state (exact for live rollout; the
bucket-teleport bound for cached primitives). Run it standalone AND wrapped
(`TimeAware(YourEnvironment)`); keep fixture worlds small — the battery
replans each scenario ~8 times.

## 6. Compose the shared capabilities

No new code per body:
- **Moving obstacles / time**: wrap with `TimeAwareEnvironment` (any
  `{x, z, t(, y)}` state).
- **Affordances**: implement `Affordance<YourState>` (jump pads, boosts,
  teleporters — extra edges with honest costs), register in an
  `AffordanceRegistry<YourState>`, pass to the wrapper. The hovercraft
  example's third test crosses a void on a boost affordance.
- **Gate sequences / goal automata**: `MultiGoalEnvironment` /
  `ScenarioEnvironment`.

## 7. Ship a demo + headless scenario test

Scenario logic in a pure builder (`demos/app/lib/<slug>-scenario.ts`), the
route in `demos/app/<slug>/page.tsx`, a card in `demos/app/page.tsx`, and a
headless test asserting the shipped config plans within budget. The
coverage manifest in `demos/test/scenarios.test.ts` (`TESTED_DEMOS`) fails
CI until you do. The crowd demo (`demos/app/crowd/`) is the reference.

## The four shipped bodies as references

| Body | State highlights | Primitive strategy | Files |
|---|---|---|---|
| Car | signed speed, gears | cached, speed buckets | `environment/vehicle-environment.ts` |
| Humanoid (step) | no inertia | generated in succ | `environment/humanoid-environment.ts` |
| Momentum humanoid | world-frame velocity | cached, speed × direction buckets | `environment/momentum-humanoid-environment.ts` |
| Aircraft | altitude + rate-limited attitude | live rollout | `environment/aircraft-environment.ts` |
| **Hovercraft (example)** | drift, decoupled yaw | live rollout | `core/test/examples/hovercraft.test.ts` |
