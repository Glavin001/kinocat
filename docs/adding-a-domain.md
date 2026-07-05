# Adding an agent domain

kinocat's planner is generic; an "agent domain" is everything else: a state
type, dynamics, primitives, a collision world, and an `Environment`
implementation. This is the recipe, with the aircraft
(`core/src/environment/aircraft-environment.ts` — 3D, attitude, OBB) as the
richest worked example and the momentum humanoid
(`core/src/environment/momentum-humanoid-environment.ts`) as the "added with
zero core edits" proof it works. Read `docs/architecture.md` first for the
seam map and the contract fine print.

## 1. Define the state and agent metadata (`core/src/agent/types.ts`)

The state is whatever your dynamics need to be Markov — position, attitude,
velocities, and absolute time `t`. Ask "do two agents at the same pose with
different X behave differently going forward?" — if yes, X belongs in the
state (the momentum humanoid carries `vx/vz` because a sprinting and a
standing person at the same pose have different futures). The agent metadata
is the envelope: limits, footprint/half-extents, cost knobs.

```ts
export interface AircraftState {
  x: number; y: number; z: number;      // y IS searched — genuinely 3D
  heading: number; pitch: number; roll: number;
  speed: number; t: number;
}
```

## 2. Write (or learn) a `ForwardSim<S>` (`core/src/agent/`)

A pure function `(state, controls, dt) => state`. Clamp to the envelope
inside the sim — the planner then cannot ever command the impossible. Treat
controls as SETPOINTS and every state variable as evolving from its current
value at bounded rates (the aircraft's `maxRollRate`/`maxPitchRate` pattern):
the sim is the single definition of what a primitive can express, so
maneuver timing (begin the roll before the slot; hold the bank between
nearby slots) falls out of the search instead of being scripted.

Honor the **equivariance contract** (documented on `characterize()`): output
must not depend on absolute position or heading, or cached primitives cannot
be rigidly transformed. Learned models (see `agent/vehicle-model.ts` and the
`kinocat/learning` pipeline) plug in the same way.

## 3. Turn the sim into primitives — live rollout or `characterize<S>`

**Decide how `succ()` applies primitives** (see `docs/architecture.md`
Seam 2): roll the sim LIVE from each node's actual state when the sim is
cheap next to your collision narrowphase or depends on continuous state
dims (the aircraft), or pre-characterize per start bucket when rollouts are
costly and the start-dependence is coarse (car speed buckets, momentum
humanoid speed × direction buckets). Either way, wire the
`checkSuccessorFidelity` hook in step 7 so the choice is verified, not
assumed.

For the cached strategy (`core/src/primitives/characterize.ts`):

```ts
const rolled = characterize<MyState, MyLocalSample>({
  forwardSim, runs: crossRuns(canonicalStarts, controlSets),
  duration, substeps,
  record: (s) => ({ /* local-frame pose + whatever succ needs */ }),
});
```

Choose canonical starts to cover the state dims your dynamics care about
(car: speed buckets; momentum humanoid: speed × velocity-direction buckets;
aircraft: one canonical start but per-resolution-level control sets via
`levelControls`). Dedupe primitives whose end states coincide after envelope
clamping — duplicates burn a collision sweep each and then die in dedup
anyway. Domains with trivial dynamics can skip caching and generate steps in
`succ()` directly (the base `HumanoidEnvironment` does).

## 4. Pick the world seam

Ground agent on surfaces → `NavWorld`. Free 3D volume → `AirspaceWorld`.
See `docs/architecture.md` Seam 3. If neither fits, define your own — the
planner never sees the world type, only your `Environment`.

## 5. Implement `Environment<State>` (`core/src/environment/`)

The five methods plus `levels`. The non-obvious parts:

- **`createNode`**: quantize each state dim; build the per-level `index`
  (coarse → fine) and the exact `hash`. Hash every Markov dim; do NOT hash
  time in a static env (see architecture doc — the momentum-ladder lesson).
  Extra state dims can join the COARSE dominance cells at coarser buckets so
  attitude-equivalent routes collapse early while the finest level keeps
  them distinct — the aircraft's pitch/roll handling
  (`aircraft-environment.ts` `createNode`) is the pattern.
- **`succ`**: look up the primitive bucket, rigid-transform cached sweeps by
  the node pose, collision-check the sweep, build successors with g/h/f set.
  Reuse scratch buffers for footprint placement (`placeFootprintInto` +
  a preallocated array) — collision checks run millions of times per plan.
  Accept the optional `level` argument if you have per-level primitive sets.
- **`heuristic`**: admissible, ideally consistent. Read the consistency
  traps in `docs/architecture.md` BEFORE inventing a clever one — two of the
  three clever ideas we tried were inconsistent, and the conformance kit is
  what caught them.
- **`checkValidity` / `reachedGoalRegion`**: static footprint checks; a
  radius (+ optional heading tolerance) goal disk.

## 6. Compose the wrappers you need

Moving obstacles / affordances / time: wrap with `TimeAwareEnvironment`
(any `{x, z, t(, y)}` state). Gate sequences: `MultiGoalEnvironment`. Goal
automata: `ScenarioEnvironment`. No domain code changes required.

## 7. Prove it with the conformance kit (`kinocat/testing`)

```ts
import { runConformance, type DomainHarness } from 'kinocat/testing';

const harness: DomainHarness<MyState> = {
  makeEnv: () => new MyEnvironment(world(), agent),   // fresh + deterministic
  sampleState: (rand) => ({ /* cover the envelope, seeded rng */ }),
  scenarios: [{ name: 'open', start, goal, maxExpansions: 120_000 }],
};
const report = runConformance(harness);
expect(report.failures).toEqual([]);   // failures print with samples
```

Run it standalone AND wrapped (`TimeAware(MyEnvironment)`) — wrappers must
conform too. Keep fixture worlds small: the battery replans each scenario
~8 times, and kinodynamic domains with more exact-hash dims are
expansion-hungrier than geometric ones.

Supply the **fidelity hooks** so the battery also proves succ() applies the
forward sim faithfully: put enough in `edge.data` to reconstruct the edge's
controls, re-simulate from the actual parent state, and declare the
tolerance (machine-eps for live rollout; the bucket-teleport bound for
cached primitives — see the three in-repo harnesses in
`core/test/conformance/` for both flavors, and `angularFields` for
heading-like dims that compare on the circle).

## 8. Ship a demo + headless scenario test

Add `demos/app/<slug>/page.tsx` with the scenario logic in a
`demos/app/lib/<slug>-scenario.ts` builder (pure, headless-testable — no
React), register the route card in `demos/app/page.tsx`, and add a headless
test asserting the shipped config plans within budget. The coverage manifest
in `demos/test/scenarios.test.ts` (`TESTED_DEMOS`) fails CI until you do.
The crowd demo (`demos/app/crowd/`, `demos/app/lib/crowd-scenario.ts`) is
the reference for this step.

## Worked deltas

| Step | Aircraft | Momentum humanoid |
|---|---|---|
| State | 8-dim, altitude+attitude searched | 6-dim, world-frame velocity vector |
| Sim | `aircraftForwardSim` (turn/climb/roll/speed clamps) | `momentumHumanoidForwardSim` (launch/brake/strafe/turn-degrade) |
| Primitives | inline via shared harness, per-level `levelControls` | `characterize<S>` over speed × rel-direction buckets |
| World | `AirspaceWorld` (OBB vs AABB/spheres) | `NavWorld` (octagon footprint) |
| Heuristic | 3D Euclidean / maxSpeed | 2D Euclidean / maxSpeed (see consistency traps) |
| Conformance | `core/test/conformance/aircraft.conformance.test.ts` | `.../momentum-humanoid.conformance.test.ts` |
| Demo | `/plane`, `/dogfight` | `/crowd` |
