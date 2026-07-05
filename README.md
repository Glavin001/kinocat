# kinocat

**Time-extended kinodynamic motion planning for the web.**

kinocat is a pure-TypeScript, tree-shakeable, web-native motion planner that
gives browser-based 3D games F.E.A.R.-grade emergent NPC navigation: reverse
maneuvers, ballistic jumps, anticipatory routing over *predicted* future world
states, opportunistic affordance use, and emergent multi-agent cooperation —
for NPC vehicles, humanoids, **and aircraft**, with the same planner. Four
agent domains ship today (car, step humanoid, inertial momentum humanoid,
fixed-wing aircraft in true 3D); adding a fifth is a documented recipe
([`docs/adding-a-domain.md`](docs/adding-a-domain.md)) proven by a packaged
conformance kit (`kinocat/testing`).

The planner core is a TypeScript port of **IGHA\*** (Incremental Generalized
Hybrid A\*, Talia / Salzman / Srinivasa, RA-L 2025), extended with **time as a
state dimension** participating in multi-resolution dominance — the novel
kinocat contribution. It builds on
[navcat](https://github.com/isaac-mason/navcat) as its topology / collision
substrate, but the core never imports a physics or rendering library.

## Why

The capability bar is not "competent game AI." It is
autonomous-racing-research-meets-multi-robot-coordination, adapted to
web-realtime budgets. NPCs should be genuinely *anticipatory* — pre-committing
to plans that depend on predicted future world states: the future positions of
moving ramps, the future trajectories of other agents, the future geometry of
destructible environments.

The spiritual reference is F.E.A.R. (Monolith, 2005), whose emergent tactical
AI came from Jeff Orkin's GOAP planner finding action combinations the
designers never scripted. kinocat applies the same *planner-as-substrate*
principle to navigation: motion primitives are the actions, moving affordances
and ramps are the smart objects, time-varying spatial targets are the goals,
and time-extended Hybrid A\* over kinodynamic state is the planner. Intelligence
emerges because the planner finds combinations — *arrive at ramp X at time T
with velocity V to launch over barrier B, knowing NPC Y will be at P at T+0.8s
to land on, knowing the player will be at Q at T+2.1s to cut off.*

A high-level tactical layer (GOAP-style, **outside** kinocat's scope) selects
the goal. kinocat finds the physically-feasible trajectory that achieves it.

## What it does

- **Kinodynamic planning** for any agent with a characterizable forward model —
  reverse maneuvers and parking pockets fall out of the search, no special-case
  logic.
- **3D, not 2.5D** — ground agents plan on navcat's 3D polygon graph
  (multi-floor, overhangs, ramps; Y derived from polygon containment), and
  aircraft plan through free 3D volume: altitude, pitch, and roll are
  *searched* dimensions with oriented-box collision (knife-edge through a
  slot that level wings can't fit).
- **Time-aware** — `predict(t) → State | null` is the single abstraction for
  everything dynamic. Collisions are hard constraints (infeasible edges pruned
  from the search); affordances are extra edges.
- **Affordances** — ramps, boost pads, moving platforms, elevators. Static ones
  are Mononen-style pre-baked as navcat off-mesh connections; dynamic ones
  (carrier-mounted ramps, Mario-Kart-style) are generated lazily at expansion
  time.
- **Multi-agent** — NPCs publish plans to a shared registry; others predict via
  `fromPublishedPlan`. Cooperation (convoys, coordinated jumps, interception)
  is emergent — no negotiation protocol.
- **Anytime contract** — `plan(req, deadlineMs)` runs to a deadline and returns
  its best plan so far. Generous deadline → near-optimal; tight deadline →
  rough but valid. The NPC always has a plan; replanning is the universal
  correction mechanism (no execution state machine).

**Anchor use cases:** NPC vehicles in physics-based destruction/sandbox games;
NPC humanoids in 3D action games (walk / sidestep / jump / climb as
primitives); mixed scenarios where humanoids and vehicles plan around each
other and around human players.

## Architecture

Eight components, layered top-to-bottom by dependency direction. All are
implemented and tested in `core/src` with zero external runtime.

| Layer | Subpath | Responsibility |
|---|---|---|
| Curves | `kinocat/curves` | Reeds-Shepp & Dubins analytical curves (heuristics + shot-to-goal) |
| Primitives | `kinocat/primitives` | Generic `characterize<S>` rollout harness + motion-primitive library |
| Agent | `kinocat/agent` | Vehicle / humanoid / momentum-humanoid / aircraft metadata + forward sims |
| Planner | `kinocat/planner` | IGHA\* anytime, time-extended core (agent-agnostic) |
| Environment | `kinocat/environment` | `Environment` interface, `NavWorld` + `AirspaceWorld` seams, vehicle / humanoid / momentum-humanoid / aircraft / time-aware / R² impls |
| Predict | `kinocat/predict` | `Predict<T>` factories, generic `Affordance<S>` registry, plan registry |
| Execute | `kinocat/execute` | Curvature-aware pure-pursuit tracker + divergence / periodic replan |
| Testing | `kinocat/testing` | Domain conformance kit — prove any `Environment<State>` satisfies the planner's contract |
| Adapters | `kinocat/adapters/{navcat,rapier,three}` | Optional-peer integrations |

The IGHA\* search loop never changes; everything kinocat adds —
time-extension, lazy affordances, navcat collision, `predict(t)`-based dynamic
obstacle avoidance — lives in concrete `Environment` implementations behind a
five-method interface (`succ`, `heuristic`, `checkValidity`,
`reachedGoalRegion`, `createNode`). `TimeAwareEnvironment` is a composable
wrapper, so the static-world env and the time-aware behavior unit-test
independently.

### Load-bearing design decisions

- **The planner reasons in plans; the executor reasons in physics; replanning
  bridges them.** No mode logic. "Stuck", "airborne", "recovering" are just
  states from which IGHA\* produces appropriate plans.
- **Discretize at build time, search at runtime.** Motion primitives are
  pre-characterized per agent; static jumps are pre-baked. The runtime planner
  is graph search over a rich precomputed graph plus on-demand dynamic edges.
- **One algorithm per problem.** IGHA\* for global planning, pure pursuit for
  tracking, `predict(t)` for prediction. No alternates "in case."
- **navcat is a dependency, not a fork.** Extend through metadata, custom
  `QueryFilter`s, and the existing off-mesh / tile-rebuild mechanisms — never
  modify navcat internals.

See the in-repo implementation plan (this document's source spec) for the full
rationale, non-goals, acceptance criteria, and tuning knobs.

## Status

Algorithmic core is built and tested with **zero external dependencies**;
navcat / Rapier / three.js integrations live behind optional adapters.

| Phase | State |
|---|---|
| 1 — Curves (Reeds-Shepp / Dubins) | done, 100% coverage |
| 2 — IGHA\* port + R² stub environment | done |
| 3 — Vehicle environment + characterization | done |
| 4 — Time extension (novel contribution) | done |
| 5 — Static affordances (Mononen off-mesh) | done (registration + metadata; full boundary auto-scan is a future extension) |
| 6 — Dynamic affordances (carrier-mounted ramp) | done |
| 7 — Multi-NPC plan-sharing | done |
| 8 — Executor + replanning | done |
| 9 — Humanoid environment | done |
| 10 — Polish / docs / examples | in progress (demos shipping) |
| 11 — Domain convergence: conformance kit (`kinocat/testing`), dynamic-world layer generalized beyond ground vehicles (3D moving obstacles, `Affordance<S>` for any state), shared `characterize<S>` harness, momentum-humanoid as the fourth domain | done |

Four agent domains share the one planner core today — car, humanoid,
momentum humanoid (inertial person), and a genuinely-3D aircraft (searched
altitude / pitch / roll, OBB collision). The seams that make that possible,
and the recipe for adding a fifth, are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/adding-a-domain.md`](docs/adding-a-domain.md); a new domain proves
itself by passing `runConformance` from `kinocat/testing`.

The core never imports navcat: environments consume a kinocat-owned
`NavWorld` / `PolygonRef` seam, and `InMemoryNavWorld` (polygon soup, zero
deps) makes the entire algorithmic core unit-testable with no external runtime.

## Packages

| Package | Path | What |
|---|---|---|
| `kinocat` | `core/` | The library (multi-entry, subpath exports) |
| `@kinocat/three` | `three/` | three.js helper workspace (foundation stub; the implemented debug helpers ship in `kinocat/adapters/three`) |
| `@kinocat/demos` | `demos/` | Next.js demo app (deployable on Vercel) |

`sideEffects: false` + per-subpath entry points: importing `kinocat/curves`
pulls zero planner/environment code. Peers (`navcat`, `@dimforge/rapier3d-compat`,
`three`) are optional and externalized from the bundle.

## Demos

`pnpm dev` builds the core and runs the Next.js app:

- **3D navmesh world** — a vehicle plans through a 3D world and drives the path
  with pure-pursuit; orbit / zoom, click to move the goal.
- **Time-aware + multi-agent** — a moving obstacle with a time scrubber, a
  second NPC coordinated via the plan registry, and a jump affordance across a
  gap.
- **Interactive 2D playground** — drag start / goal, add and move obstacles,
  tune the anytime deadline and reverse cost; replans instantly.
- **Navmesh view** — 3D navmesh debug rendering.

A headless test (`demos/test/scenarios.test.ts`) asserts the exact shipped demo
configs always find a plan within budget, so a "no plan" regression fails CI.

### Preview the rich plan (visual debugging)

The planner→controller reference is captured as a rich `Plan`
(`kinocat/plan`): per-point arc length, signed curvature, target speed /
accel, feedforward steer, reserved dynamic-state / free-space slots, and the
single-gear segment / cusp structure. The **Parking demo** renders it as a
visual-debug overlay:

```sh
pnpm dev            # builds core, starts Next.js on http://localhost:3000
```

Open **http://localhost:3000/parking**, then press **`d`** (or the
`[d] plan-debug` button). The reverse-perp and parallel scenarios (`[2]` /
`[3]`) show it best, since they have forward↔reverse cusps. It has two halves,
each in the medium that fits the data:

**3-D overlay — the spatial story (where/how it drives):**

- **Speed-colored path** — the reference line shaded by `|vRef|` (slow → fast
  = red → green), so you can *see* where the profile slows for the stall.
- **Reverse spans in blue** — segments the plan drives in reverse gear.
- **Cusp/stop markers** (yellow) — where the chassis stops and flips gear.
- **Feedforward-steer wheel glyphs** (orange) — sparse, fixed-length marks
  rotated to the wheel direction (`heading + steerFf`); they show *which way*
  the wheel points, not a magnitude.

**2-D profile strip — the quantitative story (the reference signals):**
`vRef`, `steerFf`, and `aRef` plotted against **arc length** (a 1-D signal is
legible as a curve, unreadable as 3-D glyphs), with cusps as vertical dashed
lines. This is where you read the reference precisely — and the natural place
to later overlay planned-vs-executed for MPC/LQR tuning.

Both are built from the live committed plan (`buildPlan(smoothed, …)` in
`demos/app/lib/race-scenario.ts`), so they update on every replan. It is
*produce-but-don't-consume*: the controller still tracks the plain path today;
a follow-up feeds `Plan.kappa` / `steerFf` into the tracker.

**Test it.** The builder is unit-tested (curvature/arc-length/accel,
feedforward sign in reverse, cusp-boundary placement, round-trip):

```sh
pnpm test plan/build                            # core/test/plan/build.test.ts
pnpm test parking-invariants parking-success    # flagship parking still tracks
```

## Develop

```sh
pnpm install
pnpm test            # vitest (core + demo scenarios)
pnpm test:coverage   # 80% gate on core algorithm + environments
pnpm typecheck       # tsc, all workspaces
pnpm bench           # planner / curves microbenchmarks
pnpm --filter kinocat build   # tsup -> ESM + .d.ts per subpath
pnpm size            # asserts core + navcat-adapter < 100 KB minified
pnpm verify          # typecheck + test + build + size (CI gate)
pnpm dev             # build core, then run the Next.js demos
```

Requires Node ≥ 22 and pnpm. No bundler-specific globals; the core is
browser / Node agnostic. **No ESLint / Prettier / Biome by design** — strict
`tsc --noEmit` plus vitest are the correctness gates (see `core/README.md`).

## Quick start

```ts
import { plan } from 'kinocat/planner';
import { InMemoryNavWorld, VehicleEnvironment } from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';

const agent = defaultVehicleAgent();
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [[0, 6], [1 / agent.minTurnRadius, 6], [-1 / agent.minTurnRadius, 6], [0, -4]],
  duration: 0.5, substeps: 6, startSpeeds: [0],
});
const world = new InMemoryNavWorld([{ id: 1, y: 0, ring: [[0,-10],[40,-10],[40,10],[0,10]] }]);
const env = new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity });

const result = plan(
  {
    start: { x: 2, z: 0, heading: 0, speed: 0, t: 0 },
    goal: { x: 30, z: 4, heading: 0, speed: 0, t: 0 },
    environment: env,
  },
  50, // anytime deadline (ms)
);
// result.found, result.path, result.cost, result.stats
```

Wrap `env` in `TimeAwareEnvironment` to add predicted-obstacle avoidance and
lazy affordance edges; track the plan with `purePursuit` from
`kinocat/execute`.

Building your own agent domain? Follow
[`docs/adding-a-domain.md`](docs/adding-a-domain.md), then prove it:

```ts
import { runConformance } from 'kinocat/testing';

const report = runConformance({
  makeEnv: () => new MyEnvironment(world(), agent),
  sampleState: (rand) => ({ /* seeded valid-state sampler */ }),
  scenarios: [{ name: 'open', start, goal, maxExpansions: 120_000 }],
});
// report.ok — heuristic consistency/admissibility, successor invariants,
// hash stability, determinism, anytime monotonicity, budgeted solvability
```

## References

- Talia, Salzman, Srinivasa — *Incremental Generalized Hybrid A\** (RA-L 2025) — planner core
- Dolgov et al. (2010) — Hybrid A\* (conceptual ancestor)
- Reeds & Shepp (1990); Dubins (1957) — optimal car curves
- Folkers, Rick, Büskens (2019) — Time-Dependent Hybrid-State A\*
- Mononen — Recast Navigation auto-annotation (Paris Game AI 2011)
- Orkin — *Three States and a Plan: The A.I. of F.E.A.R.* (GDC 2006)
- Coulter (1992) — Pure Pursuit path tracking
- Sharon et al. (2015) — Conflict-Based Search (MAPF; simplified plan-sharing variant)
- navcat — Isaac Mason, https://github.com/isaac-mason/navcat

## License

BSD-3-Clause (matches the IGHA\* reference implementation).
