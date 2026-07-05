# Why the learned model isn't making the car drive well — a first-principles review

*Question under review: "I have a Rapier raycast vehicle with configurable
parameters. I want to sample its physical reality into motion primitives /
models so the kinodynamic planner plans accurately and the executor controls
it precisely. It's still buggy (/raceprimitives, /sim-to-real with learned
model v2) and I don't know what I'm doing wrong."*

*Date: 2026-07-05. All file:line references verified on the current branch.*

---

## 1. The first-principles frame

Your stack is a chain of five boxes, and the fidelity of the whole is the
fidelity of the **weakest seam**, not the best box:

```
Rapier plant params
      │  (1) calibration
      ▼
Forward model  (kinematic / parametric v2 / +residual MLP)
      │  (2) sampling into primitives
      ▼
Search lattice (IGHA* over pre-baked primitive endpoints + RS analytic shots)
      │  (3) plan representation
      ▼
Plan (poses + speeds + implicit controls)
      │  (4) tracking
      ▼
Tracker (pure-pursuit / MPPI) → controls → Rapier plant
```

You have been investing almost entirely in box 2's *input* — making the
forward model more faithful (parametric v2 + residual ensemble). And that
investment worked, on its own terms: the shipped v2 model's open-loop
position RMS at 1 s is **0.97 m vs 7.4 m for kinematic** (manifest in
`demos/public/models/v2-default.json`). The model is not the problem.

The problem is that **almost none of that fidelity survives the seams**:

1. **Seam (2) quantizes it away.** Primitives are pre-rolled at only 4
   start speeds, 10 m/s apart, and looked up by *nearest bucket*.
2. **Seam (3) discards the controls.** The plan hands the tracker only
   poses and speeds; the primitive's actual `[steer, driveForce,
   brakeForce]` inputs — the thing the model predicted the response to —
   are thrown away.
3. **Seam (4) ignores even the speeds.** The default tracker is
   pure-pursuit with `respectPathSpeed: false` and the speed profile
   disabled — it re-derives its own speed law and its own steering from
   geometry. The learned model's plan quality *literally cannot appear on
   /raceprimitives* because the executor doesn't consume it.
4. **Seam (1) has drift and inversions.** Three hand-synced copies of the
   vehicle parameters disagree (mass 576 vs 580; min turn radius 4.5 used
   vs 4.68 physical vs 3.5 planned in parking), and the planner's assumed
   capability envelope is in places *larger* than the plant's — which no
   tracker can compensate for.

So the honest answer to "what am I doing wrong" is: **you're improving the
model while the architecture around it is still shaped for the kinematic
model** — pre-baked chord endpoints, geometry-only tracking, hand-copied
constants. The rest of this doc walks each seam with evidence, then gives
the ordered fix list.

---

## 2. Seam-by-seam findings

### 2.1 The model is never in the planning loop, and barely in the execution loop

Structural fact that colors everything: the forward model is invoked
**exactly once per (startSpeed × control) pair, offline**, in
`characterizeVehicle` (`core/src/primitives/characterize.ts:37`). During
search, `VehicleEnvironment.succ`
(`core/src/environment/vehicle-environment.ts:290-339`) only rigid-transforms
those pre-baked endpoint offsets. At execution time, the default tracker
(pure-pursuit) is model-free; only the MPPI tracker rolls the model, and
MPPI is not the default (`DEFAULT_TUNING.tracker = 'pure-pursuit'`,
`demos/app/lib/race-scenario.ts:417`).

That is a legitimate lattice-planner design — but it means model fidelity
only reaches the road through the primitives' *sampling density* and the
tracker's *feedforward*, and both are currently too coarse to carry it.

### 2.2 Primitive quantization destroys the model's accuracy (the biggest gap)

- Race start-speed buckets: `RACE_START_SPEEDS = [0, 10, 20, 28]`
  (`demos/app/lib/race-primitives-scenarios.ts:206`). Lookup is
  **nearest-bucket** (`core/src/primitives/library.ts:21-37`).
- In `succ`, the successor's speed is the *baked endpoint speed* of the
  bucket's primitive, not anything derived from the node's actual speed
  (`vehicle-environment.ts:301-309`).

Concretely: a node at 15 m/s expands with primitives characterized at
10 or 20 m/s. With a speed-dependent model (understeer ∝ v², friction
circle), the arc a 15 m/s car actually drives differs from the 10 m/s
bake by *meters* over a 0.55–0.8 s primitive. You paid for a model with
sub-meter 1 s accuracy and then sample it on a 10 m/s grid — the
quantization error dwarfs the model error. **This is the main reason "the
learned car" doesn't visibly plan better than the kinematic car.**

Second-order versions of the same problem:

- `yawRate` / `lateralVelocity` are part of the v2 model's state and are
  carried on `CarKinematicState`, but characterization always starts
  primitives from zero yaw rate / zero slip. Mid-corner expansions
  therefore use straight-line-entry primitives — exactly where the
  dynamics matter most.
- Speed is in the exact dedup hash but **not** in the coarse dominance
  buckets (`vehicle-environment.ts:235-238`), so on non-finest passes a
  slow arrival can dominate-prune a fast arrival at the same cell even
  though they have completely different reachable futures.

### 2.3 The search's other model is pure geometry, with an infeasible radius

The Reeds-Shepp analytic shot and the heuristic
(`vehicle-environment.ts:343-469`) use only `agent.minTurnRadius` —
speed-independent, dynamics-free. Fine in principle (heuristics may be
optimistic; the shot is a shortcut generator), but two things make it
actively harmful today:

- **The radius is infeasible.** Physical plant minimum radius is
  `L/tan(δmax) = 3.2/tan(0.6) ≈ 4.68 m` (raycast-vehicle defaults).
  `RACE_AGENT.minTurnRadius = 4.5` (hand-mirrored, 4 % too tight);
  parking plans at **3.5 m** (`parking-scenarios.ts:143`) — 25–34 %
  beyond what the chassis can execute. Every tight arc the planner emits
  is one the car *cannot* drive; the tracker then reads its own failure
  as divergence and replans, forever.
- **The shot's cost is priced at top speed** with no accel/cusp dwell
  (`vehicle-environment.ts:394`), so incumbent comparisons prefer
  timelines the chassis can't keep, feeding time-divergence replans.

First principle violated: **the planner's capability envelope must be a
strict subset of the plant's** (plan conservatively; let feedback absorb
the residual). Several places currently invert this — turn radius in
parking, 30 m/s corner speeds in the kinematic library, optimistic shot
pricing.

### 2.4 The executor throws away everything the model knew

The plan reaches the tracker as `CarKinematicState[]` — poses + speeds.
Then, in the default configuration:

- `respectPathSpeed: false` and `enableSpeedProfile: false`
  (`race-scenario.ts:398-413`) — the plan's speeds are **ignored**;
  pure-pursuit re-derives speed from its own curvature cap.
- Steering is re-derived from geometry: lookahead chord → curvature →
  `steer = -gear·atan(κ · 2·WHEEL_BASE)` (`race-scenario.ts:1240`). This
  is a steady-state kinematic inversion; it contains none of the v2
  model's understeer, yaw lag, or friction-circle knowledge. The
  primitive's *actual* control inputs — which the model rollout proved
  produce the desired arc — were discarded at plan-build time.
- The race lookahead (`lookaheadMin: 3 … lookaheadMax: 14`) applied to
  parking-scale segments degenerates to "aim at the endpoint" (documented
  in the production-readiness review as E1).

So even a perfect plan is executed by a controller that re-solves the
problem with a cruder model than the one that planned it. In classic
terms: **you have feedback with no feedforward.** The standard
architecture for exactly this stack is:

> feedforward = the plan's own controls (or model-inverted controls),
> feedback = a small corrective term (pure-pursuit / Stanley / MPC) on
> the *residual* error.

PR #34's rich `Plan` structure (curvature, `steerFf`, segments) is
precisely the missing carrier for this — it's the highest-leverage merge
in flight for this goal, together with the follow-up that actually feeds
`kappa`/`steerFf` into the tracker.

### 2.5 Calibration drift: three hand-synced copies of the truth

`deriveLearnableConfig(opts)` (`raycast-vehicle.ts:409-428`) computes the
model config *from* the Rapier options — but only sim-to-real, training,
and tests use it. The race path uses hand-copied mirrors:

- `DEFAULT_LEARNABLE_CONFIG.chassisMass: 580` vs true derived **576**
  (`core/src/vehicle/car/vehicle-config.ts:41-53`).
- `RACE_AGENT.minTurnRadius: 4.5` vs physical 4.68; `maxSpeed: 30`
  empirical, hand-entered.
- `ENGINE_FORCE_N/BRAKE_FORCE_N/WHEEL_BASE` re-declared in at least three
  files; MLP `CONTROL_SCALES = [1, 4000, 2000]`
  (`vehicle-model.ts:480`) hardcodes the force scales — change
  `engineForce` in the Rapier options and the learned model's input
  normalization silently breaks.

The moment you tune a Rapier parameter (your stated use case: "configurable
parameters for how it drives"), every one of these copies is stale and
nothing tells you.

### 2.6 Model-layer bugs (real, but second-order to the seams)

- **`accelTau` is algebraically inert** in v2:
  `aEff = aLong·(1−e^{−dt/τ}) + aLong·e^{−dt/τ} ≡ aLong`
  (`vehicle-model.ts:269`). The fitter is spending a degree of freedom on
  a parameter with zero effect; the model has no longitudinal lag even
  though it claims to.
- **`loadTransferCoeff` is declared, fit, regularized — and never used**
  in the integration body (dead parameter, fit degeneracy).
- **The shipped `v2-default.json` was trained under older, looser bounds
  and is loaded un-clamped**: `engineScale 1.15` (current HI 1.05),
  `brakeScale 3.0` (HI 2.0), `steerRatio 1.31` (HI 1.30),
  `reverseEffScale 1.3` (HI 1.1) — `rebuildModel`
  (`v2-model-persistence.ts:104-122`) bypasses `paramsV2FromVec`. The
  current physical-plausibility rationale and the shipped artifact
  disagree; the artifact should be re-emitted from a fresh fit.
- On /sim-to-real, if neither a localStorage model nor the fetch of
  `/models/v2-default.json` succeeds, the page silently runs
  **parametric-only hand-tuned defaults** (banner at
  `SimToRealScope.tsx:881-885`) — worth checking you weren't judging v2
  by its fallback.
- Timestep asymmetry: the plant integrates at 1/240 s (4 substeps of
  1/60), MPPI rolls the model at 0.05 s single steps — 12× coarser; the
  model's dt-consistency test allows 5 cm per halving, which compounds
  at MPPI horizon.
- The plant applies steering **instantly** (`setWheelSteering` every
  tick, no slew), while the model has `yawRateTau = 0.18 s`. The yaw lag
  is real chassis physics (inertia + tire relaxation) so a fitted lag is
  correct — but be aware the *steer channel itself* is lag-free in the
  plant, so any additional command smoothing in the executor is a
  model/plant mismatch you introduce, not remove.

---

## 3. What "doing it right" looks like (the education part)

For a lattice/kinodynamic planner over a physics plant, there are three
contracts, and every mature stack (Urmson/Boss, LaValle's texts, the
state-lattice literature) enforces them explicitly:

**Contract 1 — Calibration: one source of truth, derived not copied.**
Everything downstream of the Rapier options must be *computed* from them:
`mass = f(density, halfExtents)`, `minTurnRadius = L/tan(δmax)·(1+margin)`,
`maxAccel = F_engine/m`, `maxDecel = f(F_brake, μ)`, `aLatMax = μg·slack`,
control scales for the MLP. You already have `deriveLearnableConfig`;
extend it to a `deriveAgentCapabilities(opts)` and make `RACE_AGENT`,
`PURE_PURSUIT_CONFIG`, `MPC_CONFIG`, the parking agent, and
`CONTROL_SCALES` consume it. Add a unit test asserting the derived values
equal the used values, so drift becomes a red test.

**Contract 2 — Containment: planner envelope ⊂ tracker envelope ⊂ plant
envelope.** The planner should assume slightly *less* grip, *larger*
radius, *lower* accel than the plant has (5–10 % margin), so the tracker
always has control authority left to correct with. Assert it in code
(one-line invariant tests). Today parking inverts it (3.5 < 4.68) and
racing rides the edge (4.5 < 4.68 — also inverted, just less).

**Contract 3 — Consistency: the model that plans is the model that
executes.** Two valid patterns:
- *Feedforward + feedback*: store each primitive's control trace in the
  plan; the tracker replays it as feedforward and adds a small
  pure-pursuit/Stanley correction. Cheap, deterministic, and the model's
  knowledge survives to the wheels.
- *Model-in-the-loop tracking*: MPPI/MPC with the **same** v2 model as
  the rollout model, substepped to match the plant dt. You already have
  both pieces (`mpc-tracker.ts` + `parametricForwardV2`); they're just
  not the default and roll at 0.05 s.

Either satisfies the contract; today's default (geometry-only
pure-pursuit that ignores plan speeds) satisfies neither.

And one lattice-specific principle: **sampling density must match model
curvature.** A model whose behavior varies strongly with v (understeer
∝ v²) sampled at 10 m/s intervals is aliased. Either densify the buckets
until endpoint interpolation error < your tracking tolerance, or stop
pre-baking and roll the model on demand in `succ` (the v2 parametric
backbone is ~a few hundred ns/step of straight-line math; 15 controls ×
6 substeps per expansion is well within your 120 ms replan budget —
measure, but it's likely fine, and it gives you exact-speed,
exact-yawRate primitives with the residual MLP optionally off during
search for speed).

---

## 4. Ordered fix list (each step observable before the next)

Sequenced so each step has a measurable acceptance criterion, using the
harness you already have (controller-bench, sim-to-real, eval module from
PR #37 once merged).

1. **Single source of truth for capabilities** (Contract 1). Add
   `deriveAgentCapabilities` next to `deriveLearnableConfig`; consume it
   everywhere; delete the hand copies; add drift tests.
   *Acceptance: change `engineForce` in one place, everything re-derives;
   drift test red if anyone re-hardcodes.*
2. **Fix the containment inversions** (Contract 2). Planner
   `minTurnRadius ≥ 4.68·1.05` everywhere (parking especially); shot cost
   priced with accel/cusp dwell; invariant tests
   (`plannerRadius ≥ plantRadius`, `plannerGoalRadius < arriveRadius`,
   etc.).
   *Acceptance: parking plans contain no arc the chassis can't drive;
   the frozen 0.29 m lateral residual on reverse-perp shrinks.*
3. **Make the executor consume the plan** (Contract 3, cheap half).
   Land PR #34 (with its reverse `steerFf` sign fix), store per-primitive
   control traces, and add feedforward to pure-pursuit:
   `steer = steerFf(s) + feedbackGain·(pure-pursuit correction)`; turn
   `respectPathSpeed` back on with the speed profile fed by *plan*
   curvature. Re-tune lookahead per scenario class (race vs parking).
   *Acceptance: controller-isolation eval (PR #37) shows cross-track RMS
   on a min-radius arc drops by ~the understeer term; parking heading and
   centering both inside tolerance in one run.*
4. **Fix the v2 model bugs and re-emit the artifact.** Make `accelTau`
   real (`aEff = ...` actually first-order), implement or delete
   `loadTransferCoeff`, clamp params on load in `rebuildModel`, re-fit
   with current bounds, re-ship `v2-default.json`.
   *Acceptance: model acceptance test ratchet holds; no shipped param
   outside `PARAMS_V2_LO/HI`.*
5. **Densify the lattice where it's aliased.** Short term: race start
   speeds `[0,4,8,12,16,20,24,28]` (you already use exactly this grid for
   training), add speed to coarse dominance keys. Medium term: roll
   primitives on demand in `succ` from the node's true
   (speed, yawRate, lateralVelocity) with memoization on a fine
   quantization — this is the change that makes "sample the physical
   reality" actually reach the search.
   *Acceptance: per-bucket primitive-endpoint-vs-Rapier error (new test:
   characterize vs headless-trial rollout at off-bucket speeds) below a
   budget; learned car visibly out-plans kinematic car on /raceprimitives.*
6. **Model-in-the-loop tracking as the default for racing.** Switch
   `DEFAULT_TUNING.tracker` to MPPI with the v2 model, substepped
   (e.g. 0.05 s horizon step integrated as 3× 1/60 model steps), warm-
   started, with the plan as reference. Keep pure-pursuit+FF for parking
   (low speed; determinism) until MPPI is tuned there.
   *Acceptance: closed-loop bench budgets (lap time, off-track events,
   replan count) beat pure-pursuit on the same scenarios.*
7. **Close the loop in CI.** The production-readiness review's §4 plan:
   settle-latched success oracle, closed-loop budgets (time-to-settled,
   replan count, final pose error), post-success hold window, and the
   per-layer eval harness (PR #37) with a reverse/parking-shaped
   reference. This is what keeps every seam above honest as you tune.

Steps 1–3 are days of work and require no retraining; they will move the
visible behavior more than any model improvement, because they let the
model's existing accuracy reach the wheels. Steps 5–6 are the
architectural payoff of your learned-model investment. Step 4 can proceed
in parallel.

---

## 5. Direct answers to "what am I doing wrong"

- You are **not** wrong that kinematic/parametric-only is too simplistic,
  and your instinct to sample the real plant is the right one — the v2
  pipeline (native controls, settle handling, ensemble OOD gate,
  maneuver excitation) is genuinely well built.
- The gap is that the model's fidelity is consumed at only two points —
  4 offline start speeds, and (optionally) the non-default MPPI tracker —
  while the layers users actually watch (search geometry, pure-pursuit,
  hand-copied constants) still embody the kinematic worldview.
- "Precisely control the dynamic raycast vehicle" is a *feedforward*
  problem before it is a model problem: keep the primitive's control
  trace, replay it, and let feedback correct the residual — or track with
  the model itself (MPPI). Either way, stop re-deriving controls from
  geometry alone.
- And make every number the planner believes about the car a *derivation*
  from the Rapier options, with a margin, enforced by a test. That is
  what makes "configurable parameters" safe to configure.
