# Training Dataset Plan — Exhaustive Realistic Coverage

> **Self-contained** — this document assumes no prior chat context.
> A new contributor should be able to pick this up cold, read it
> top-to-bottom, and start on Phase 0 without asking questions.
> Companion doc: `docs/v2-model-handoff.md` (deeper architectural
> overview of the v2 model itself; this plan is the "next step"
> after that one).

Goal: replace the current synthetic constant-control trial grid with a
training dataset that **matches the distribution of states + controls
the model is actually queried on at planning time**, so the v2 learned
model finally beats the kinematic baseline on lap time and ghost
predictions match the real chassis across every demo.

This plan is **domain-agnostic where possible**. Generic primitives live
under `kinocat/training/*` and `kinocat/scene/*`; the car-specific
maneuver library + closed-loop scenarios live under
`kinocat/vehicle/car/*`. An airplane domain would plug into the same
infrastructure.

---

## 0. Project context (skip if you already know the codebase)

**What kinocat is**: a TypeScript monorepo with two packages —
`core/` (`kinocat` published name, framework-agnostic) and `demos/`
(Next.js app with interactive 3D scenes). The core implements a
generic planner + vehicle-simulator + learning pipeline; the demos
wire it up to Rapier physics + Three.js rendering and present a set
of scenarios.

**The v2 learned vehicle model** (the thing this plan is about):
a hybrid dynamics model with a **parametric backbone** (16 physical
parameters like `engineScale`, `gripScale`, `understeerOffThrottle`)
+ a **residual MLP ensemble** that corrects what the parametric
backbone can't capture. It predicts the next `CarKinematicState`
given the current state, the applied `WheeledCarControls`, the
chassis `LearnableVehicleConfig`, and dt. It is trained **offline**
from simulated trial data, then used by the planner at race time
(via `learnedForwardSimV2`) to evaluate candidate paths.

**The user-visible problem this plan fixes**: the v2 model loses
to the simpler kinematic baseline on lap time, and ghost predictions
visibly diverge from the real chassis in `/sim-to-real` debug mode.
Both symptoms trace to the same cause — the training dataset doesn't
match the deployment query distribution.

**Demo routes** (in `demos/app/<route>/`):

| Route | What it does |
|---|---|
| `/sim-to-real` | Side-by-side: real Rapier chassis vs. N model "ghosts" predicting open-loop. The debugging surface for sim-to-real divergence. |
| `/raceprimitives` | Two cars race each other using the v2 model + planner. Hosts the in-browser training UI. |
| `/model-lab` | Live training diagnostics (loss curves, coverage, per-component RMS). |
| `/primitive-explorer` | Fan-plot diagnostic: what the model thinks each (state, control) primitive produces. |
| `/ramp`, `/obstaclecourse`, `/carchase` | Other interactive scenarios sharing the same chassis + planner. |

**Key types and where they live**:

| Type / module | Path | Purpose |
|---|---|---|
| `CarKinematicState` | `core/src/agent/types.ts` | `{x, z, heading, speed, t, yawRate?, lateralVelocity?}` — the car's Markov state. |
| `WheeledCarControls` | `core/src/agent/controls.ts` | `{steer (rad), driveForce (N), brakeForce (N)}` — the canonical action shape. Every driving source (user, planner, model rollout, scripted trial) emits this. |
| `LearnableVehicleConfig` | `core/src/agent/vehicle-config.ts` | Physical chassis params a learned model can identify (engineForce, brakeForce, maxSteerAngle, wheelBase, …). |
| `LearnedVehicleParamsV2` | `core/src/agent/vehicle-model.ts` | The 16 parametric coefficients. |
| `learnedForwardSimV2` | `core/src/agent/vehicle-model.ts` | The model's forward simulator (parametric + residual MLP). What the planner calls. |
| `Body<S, C>` | `core/src/scene/body.ts` | Generic interface for "a simulator I can `step()` and `readState()`". `RapierCarBody` is the car impl. |
| `Driver<S, C>` | `core/src/scene/driver.ts` | Generic interface for control policies. Existing impls: `IdleDriver`, `ScriptedDriver`, `SwitchableDriver`, `RecordingDriver`. |
| `SceneController<S, C>` | `core/src/scene/scene-controller.ts` | Orchestrates a `Body` + `Driver` + ghost trackers + recorder at 60 Hz. |
| `runTrial` | `core/src/scene/run-trial.ts` | Headless simulation kernel. |
| `DebugRecorder<S, C>` | `core/src/diagnostics/debug-recorder.ts` | Rolling-buffer recorder used by `/sim-to-real`. Exports JSON / markdown. |
| `OpenLoopGhostTracker<S, C>` | `core/src/scene/open-loop-ghost.ts` | Maintains open-loop predictions of N models against the real body; re-anchors on drift. |
| `Trial<S, C, Cfg>` + `TrialStore` | `core/src/learning/trial-store.ts` | The training-data container. |
| `runOfflineTrainingCore` + `TrainingPipeline<S,C,P,Cfg>` | `core/src/training/index.ts` | Generic orchestrator + pipeline contract. |
| `CarV2TrainingPipeline` | `demos/app/lib/training-driver.ts` | Car-v2-specific impl of `TrainingPipeline`. The "what the in-browser train button calls" entry point. |
| `loadV2Model` / `saveV2Model` | `demos/app/lib/v2-model-persistence.ts` | Serialize/deserialize a trained model. |
| `createHeadlessTrialHarness` | `core/src/adapters/rapier/headless-trial.ts` | Spins up a Rapier world + chassis + scripted control trace in Node (no DOM). |
| `wheeledFromNormalized` + `ZERO_WHEELED` | `core/src/vehicle/car/wheeled.ts` | Convert legacy normalized `{steer, throttle, brake}` to canonical `WheeledCarControls`. **Single source of truth** for the planner→Rapier sign convention. |

**Architecture rule** (enforced by `core/test/agnostic-core.test.ts`):
nothing under `core/src/` except `core/src/adapters/` may mention
Rapier or any specific scenario. Generic stuff in core; Rapier-specific
stuff in `core/src/adapters/rapier/`; use-case glue in `demos/`. New
code from this plan must respect that rule.

**How to run things locally**:

```bash
pnpm install                       # once
pnpm -F ./core typecheck           # core type-check
pnpm -F ./demos typecheck          # demos type-check
NODE_OPTIONS="--experimental-require-module" pnpm test   # full vitest suite
pnpm -F ./demos dev                # Next.js dev server (then visit /raceprimitives etc.)
```

Pre-existing flake to ignore: `demos/test/carchase-scenarios.test.ts`
sometimes returns `found=false`; not caused by anything in this plan.

---

## Diagnosis — why the current dataset is insufficient

The current pipeline (`demos/app/lib/training-driver.ts`) produces
~172 trials over 3 rounds. Every single one is a **constant-control
hold of ~2 s starting from `(x=0, z=0, heading=0, lateralVelocity=0,
yawRate=0)`** at one of 8 forward speeds.

```
Round 0 — seed grid: 76 trials
  for each (startSpeed ∈ {0,4,8,12,16,20,24,28},
            steer    ∈ {0, ±maxSt·{0.2,0.3,0.4,0.5,1.0}},
            drive    ∈ {0, mid, strong},
            brake    ∈ {0, low, mid}):
      hold (steer, drive, brake) CONSTANT for ~2 s, record

Rounds 1+ — active exploration: 48 trials/round
  same 8-speed × 3-steer grid with jitter, drive ≈ 2800-4000 N, brake = 0
```

**What's covered vs. what's missing**:

| Regime | Covered? |
|---|---|
| Straight throttle 0-28 m/s | ✅ |
| Straight brake 0-28 m/s | ✅ |
| Coast (no input) at all speeds | ✅ |
| Gentle turn (±0.12 rad) at high speed | ✅ |
| Hard turn (>0.3 rad) at >16 m/s | ❌ |
| **Combined throttle + steer + brake** | ❌ (mostly separated cells) |
| **Sustained turn > 2 s** | ❌ (every trial is 2 s flat) |
| **Step responses / input transitions** | ❌ (all constant) |
| **Steer reversal (slalom-like)** | ❌ |
| **Spawn mid-drift (lateralVel ≠ 0)** | ❌ — every spawn is 0 |
| **Spawn mid-turn (yawRate ≠ 0)** | ❌ — every spawn is 0 |
| **Reverse + steer** | ❌ |
| **Throttle release / lift-off transition** | ❌ |

**The concrete bug that motivates this plan** ("the Free-Drive
lift-off bug" referenced throughout): in `/sim-to-real`'s manual
driving mode (toggle "Free Drive", drive the chassis with WASD), the
user holds `W+A` (throttle + left turn), then releases. The chassis enters a
"coast-from-speed-with-residual-yaw-and-lateral-velocity" state. The
v2 residual MLP has **never been shown a single trial** that enters
this state — every training trial started at `(yawRate=0,
lateralVel=0)` and held one constant input. The MLP extrapolates from
the nearest training point ("0 m/s constant coast" or "12 m/s constant
coast at steer=0") and emits garbage residuals. The ghost diverges
immediately. This is the smoking-gun OOD failure the dataset has to fix.

**Net effect**: the model is essentially default-extrapolated on the
exact regimes the race exercises *and* the regimes a human driver
generates in Free Drive. The `/primitive-explorer` page shows this
directly — the v2 fan collapses to near-default behavior at race
speeds. The simulator can generate any of the missing regimes for
free; we just aren't asking it to.

| The model sees at training | The planner / user actually queries |
|---|---|
| Constant `(steer, drive, brake)` for ~2 s | Continuously-varying controls every 16.7 ms |
| Initial state: `(speed, yawRate=0, lateralVel=0)` | Mid-corner, already sliding |
| Default RWD chassis only | Every chassis used by CarChase / Ramp / RacePrim |
| Flat ground | Heightfield ramps, slopes, jumps |
| Discrete speed buckets | Continuous speed |
| No coupled brake+steer transitions | Trail-braking is the planner's bread and butter |
| Symmetric, well-behaved inputs | Saturation, sign flips, lift-off, panic |

---

## Strategy — 7 phases, each independently shippable

The phases are ordered so each lands a measurable improvement on its
own. Stop after any phase if the metric gate is met; don't ship the
later phases speculatively.

### Phase 0 — Coverage observatory + train/val/test split (must land first)

We cannot fix what we cannot measure. Build the visualization that
makes the gap obvious — *and* fix the eval-set instability — before
writing more dataset code.

**Train / validation / test split policy** (replaces the
last-25%-of-trials policy currently in `evaluate()` of
`training-driver.ts`):

- **train (~70 %)** — fit set; what the parametric fit + residual MLP
  see. Re-shuffled each round, but only trials with `split: 'train'`
  ever flow into the optimizer.
- **validation (~15 %)** — used during training for early-stopping +
  hyperparameter choices + the live training/val loss curve. The
  optimizer never directly minimizes on this.
- **test (~15 %)** — **fixed, named, frozen for the life of the
  project**. Never touched by any fit, never modified by active
  exploration. The number quoted in PRs / docs / acceptance gates
  is always the test-set number. This is the only way to honestly
  compare phase N vs phase N+1.
- Split is assigned **at trial creation time** by hashing
  `(maneuverId, maneuverParams, configKey, scenarioId)` so the same
  logical maneuver always lands in the same split — no leakage when
  the active explorer asks for "more like this one".
- For chassis-config diversification (Phase 4) the split is *also*
  by-config: ~2 of the 16 configs in the Latin-hypercube sample are
  held out entirely → test set proves cross-config generalization.

**Deliverables**
- `kinocat/training/coverage-meter.ts` (generic, `<S, C>`-parameterized):
  - N-dimensional histogram over arbitrary state+control projections
    supplied by the caller (e.g. for cars: speed × steer × lateralVel ×
    yawRate × throttleSign × brakeSign).
  - `record(trial)`, `query(bin)`, `summary()` returning per-bin
    trial-count + held-out RMS.
- `kinocat/vehicle/car/coverage-projection.ts`: canonical car
  projection (5-D: speed, steer, lateralVel/speed, yawRate, drive-or-brake).
- `kinocat/learning/trial-store.ts` extended: each `Trial<S,C,Cfg>`
  carries `split: 'train' | 'val' | 'test'`. `TrialStore` exposes
  `all(split?)`. Hash-based assignment described above is the default.
- `evaluateModel` extended to report **train, val, and test** RMS
  side-by-side at each horizon.
- Model Lab — three new cards:
  1. **Coverage Heatmap** — 2D slice picker, color = log(trial count),
     overlay = held-out (test) RMS per cell. Hot red = wrong AND
     under-sampled.
  2. **Train / Val Loss Curves** — both curves on one chart per round.
     Train↓ while Val↑ = overfitting; both flat = under-capacity;
     both ↓ in parallel = healthy. Mirrors the loss diagnostic every
     ML practitioner expects.
  3. **Per-Horizon Train / Val / Test RMS table** — the honest
     comparison surface for cross-phase progress reports.
- Same coverage heatmap rendered on `/sim-to-real`: highlights cells
  the current ghost is exercising that have low training count.
  Smoking gun for "ghost diverges right here" debug sessions.

**Acceptance gate**
- Heatmap renders for the current model and visibly shows the
  expected holes (high lateralVel, mid-corner yawRate≠0, transitions, etc.).
- Train / val loss curves both populated each round; an intentionally
  over-trained run (high MLP capacity, low data) visibly shows the
  train↓ val↑ overfitting signature.
- Click a hot cell → see the controls trace the planner pushed
  through it and the actual training trials (if any) near it.

### Phase 1 — Maneuver library (replace constant-hold trials)

The fundamental fix: trials should be **time-varying control scripts**,
not flat constants. A `ScriptedDriver<S, C>` already exists in
`kinocat/scene` — we use it.

**Trial-budget recommendation** (per round, replacing the current
constant-hold seed grid):

| Class | % of budget | Why |
|---|---|---|
| 1. **Random-walk OU controls** | **60%** | Single class covers transitions, reversals, varied magnitudes — i.e. the "any human driving" regime. Cheapest realism per trial. |
| 2. **Transition probes** | 15% | Targeted: throttle-release, throttle→brake, brake→throttle, hard-steer-then-zero, steer-reversal. One transition per trial; the *transitions themselves* are the training signal the constant-hold dataset entirely lacks. |
| 3. **Saturation / panic probes** | 10% | High-speed full-lock with brake recovery; lift-off-during-turn; reverse + steer. These exist in deployment (panic stops, evasive maneuvers) but never in the current grid. |
| 4. **Named identification maneuvers** | 10% | Step-steer, sin-sweep, slalom, fishhook, double-lane-change, J-turn, Scandinavian flick, donut. Cover the system-identification "textbook" set so the parametric backbone is well-conditioned. |
| 5. **Constant-hold grid (legacy)** | 5% | Kept as a small ablation-A/B baseline so we can measure the realism gain quantitatively. |

**Deliverables**
- `kinocat/vehicle/car/maneuvers.ts` — parameterized factories returning
  `Driver<CarKinematicState, WheeledCarControls>`:
  - **Random-walk family** (class 1):
    - `ouControls({ sigmaSteer, sigmaDrive, sigmaBrake, tau, clamp,
       durationSeconds, seed })` — independent Ornstein-Uhlenbeck
       processes per channel with mean-reversion timescale `tau ≈ 0.3 s`,
       clipped to chassis limits. Smooth, non-pathological, deeply
       transition-rich.
    - `mixtureRandomWalk({ ouParams, modeSwitchHz, modes: ['cruise',
       'attack', 'lift-off', 'brake-zone'] })` — switches OU
       parameters between driving modes mid-trial, so a single trial
       contains multiple regime crossings.
  - **Transition probes** (class 2) — explicit 4-second trials with a
    single transition at `t = 2 s`. These are what the lift-off bug
    needs:
    - `throttleRelease({ startSpeed, throttleHold, steerHold })` —
      `(steer, drive, 0)` then `(steer, 0, 0)`. The exact regime the
      Free-Drive lift-off bug enters.
    - `throttleToBrake({ startSpeed, throttle, brake, steer })`.
    - `brakeToThrottle({ startSpeed, brake, throttle, steer })`.
    - `steerToZero({ startSpeed, steer, drive })` — hard-steer-then-release.
    - `steerReversal({ startSpeed, steerLeft, steerRight, drive })` —
      slalom-flick: full-left → instant full-right.
  - **Saturation / panic probes** (class 3):
    - `panicTurn({ startSpeed: 25, steer: ±maxSt, durationS: 0.4 })`
      followed by `brakeRecovery({ brake: maxBrake })`. The
      friction-circle-saturated regime under recovery.
    - `liftOffOversteer({ startSpeed, driveForce, liftTime, steerStep })`
      — load-shift oversteer probe.
    - `reverseWithSteer({ throttle: -driveMid, steer: ±maxSt })` —
      currently 0% of dataset.
  - **Identification maneuvers** (class 4):
    - `stepSteer(amplitude, holdSeconds)`, `sinSweepSteer(amplRange,
       freqRange, durationS)`, `slalom(period, amplitude)`, `trailBrake
       (brakeForce, releaseTime, steerRamp)`, `throttleOnApex
       (initialBrake, holdSeconds, throttleRamp)`, `jTurn(speed,
       steerStep)`, `fishhook(speedRange, steerSequence)`,
       `scandinavianFlick(speed, counterSteer, throttle)`,
       `doubleLaneChange(width, length)`, `donut(steer, drive)`.
- Each factory is parameterized so we can run a sweep over the
  parameter range and emit a `TrialSpec[]`.
- `kinocat/training/maneuver-runner.ts` (generic): runs a `Driver` for
  N seconds against a `Body`, returns a `Trial<S, C, Cfg>`. Reuses
  `runTrial` from `kinocat/scene`.
- Replace `buildSeedGrid()` / `extremeProbes()` in
  `demos/app/lib/training-driver.ts` with a budget-weighted draw from
  the maneuver library. The constant-hold grid stays available as the
  5% ablation baseline.

**Acceptance gate**
- Coverage heatmap (Phase 0) shows fuller cells in the mid-corner /
  trail-brake / lift-off / fishhook / transition regimes — the
  lateralVel and yawRate axes are populated, not zero-pinned.
- Held-out RMS at `horizon=1.6 s` drops by ≥ 20% on a **fixed**
  evaluation set (eval-set lock-in policy, see Phase 0, eliminates
  the moving-target problem the current last-25%-of-trials policy
  has).
- The Free-Drive lift-off regression test (recorded sim-to-real JSON
  from the original bug) — model prediction now within tolerance of
  the real chassis through the lift-off transition.

### Phase 2 — Initial-state diversification

Every current trial enters the recording window from `(speed,
yawRate=0, lateralVel=0)` (the brake-to-stop-then-accel-to-speed
pattern). The model never sees a mid-corner initial condition, so it
has no idea what to do at the very moment the planner queries it
hardest.

**Two implementation options — use both, they have different tradeoffs**:

| Approach | Pro | Con |
|---|---|---|
| **A. Pre-roll script** (preferred) — stitch a Phase-1 maneuver before recording starts | Realistic suspension state, wheel loads, slip angles | Slower (each trial has a warm-up cost) |
| **B. `teleportFull(state)`** — directly write the chassis state we want | Instant, deterministic, covers states pre-roll can't reach (heavy oversteer, sliding) | Suspension/wheel-load state is at rest; first few ticks of the trial are slightly artificial |

Use B to *bootstrap* coverage cheaply, then re-record the same target
states with A once Phase 1 is producing the entry maneuvers naturally.

**Deliverables**
- `kinocat/training/state-conditioner.ts` (generic): two implementations
  of the same interface — `PreRollConditioner` (drives a `Body` through
  a script until `readState()` matches target within tolerance) and
  `TeleportConditioner` (calls `Body.teleport(state)` directly).
- Car-specific `carStateConditioner` targets — populated densely on the
  axes the constant-hold dataset leaves at zero:
  - `(speed=12, yawRate=0.6, lateralVel=-1.5)` — mid-left-corner sliding.
  - `(speed=20, yawRate=0.1, lateralVel=0)` — gentle high-speed turn.
  - `(speed=4, yawRate=-1.2, lateralVel=2)` — heavy oversteer at low speed.
  - `(speed=-3, yawRate=0, lateralVel=0)` — reverse start.
  - Latin-hypercube sample over the 5-D coverage projection.
- Maneuver library extended so every factory accepts an
  `initialCondition` target.

**Acceptance gate**
- Coverage heatmap's `lateralVel` and `yawRate` axes are populated
  uniformly, not zero-pinned.
- Replay the recorded sim-to-real debug JSON from a known
  divergence: the model now predicts within tolerance there.

### Phase 3 — Closed-loop scenario trials

The richest source of "realistic" data is **the model driving itself
around the actual race track**. The planner queries become the
training inputs by construction. This is the standard "iterative
imitation learning / DAgger" pattern — DAgger (Dataset Aggregation,
Ross et al. 2011) means: deploy the current model, log every
`(state, control, real-next-state)` it actually visits, mix those
into the training set, retrain. The training distribution is
guaranteed to track the deployment distribution.

**Deliverables**
- `kinocat/training/scenario-collector.ts` (generic): wraps a
  `SceneController<S, C>` and a `RecordingDriver<S, C>`; runs N
  closed-loop scenarios; emits trials from each window of the run
  using `OpenLoopGhostTracker`'s re-anchoring policy as the natural
  trial boundary.
- Car-specific scenario suite:
  - `/raceprimitives` track laps (planner-driven; the planner already
    uses the model under training — DAgger-style).
  - `/carchase` robber-vs-cop episodes (varied chassis configs!).
  - `/ramp` ramp-launch-and-recover.
  - `/obstaclecourse` weaving.
  - User WASD recordings (optional — gated by a "save my drive"
    toggle in each demo).
- Each closed-loop episode generates many short trials (e.g.
  every 0.5 s window of the run is a fresh trial with its own
  initial state). The dataset *automatically* matches the planner's
  query distribution.
- Cross-validation guard: held-out scenarios stay held out; the
  evaluation set never includes a scenario whose trials are in train.

**Acceptance gate**
- `pnpm race` (the headless race benchmark CLI — see Cross-cutting
  Infrastructure) reports the v2 model **beats** the kinematic
  baseline on average lap time over ≥ 3 laps with a fixed seed. This
  is the headline gate that was originally missed; everything before
  this phase is in service of crossing it.
- Sim-to-real ghost RMS at horizon=1.6 s drops to within "imperceptible"
  threshold on every demo.

### Phase 3.5 — Sim-to-real hard-example miner ⭐

Every divergence the user sees in the `/sim-to-real` scope is a
training trial we are missing. Today the user spots the divergence
visually, exports debug JSON, and the loop ends. We close it: when
the recorder detects a high-gap state, it automatically writes the
`(state, controls, dt, real_next_state)` tuple to a "hard-cases"
pool, and a background trigger schedules a residual-MLP refit when
the pool crosses a size threshold.

This is **DAgger turned into a daemon**: every demo session
contributes training data for the regimes the model is most wrong
about, with no user action required.

**Deliverables**
- `kinocat/training/hard-example-miner.ts` (generic, `<S, C>`-shaped):
  - Subscribes to a `DebugRecorder`'s frame stream.
  - Gap predicate is caller-supplied (default for cars: instantaneous
    `Δheading > 10°` between any ghost and the real chassis at any
    horizon, OR `Δposition > 1.5 m` at horizon=1.0 s).
  - When the predicate fires, emits a `TrialSpec` covering the
    surrounding window (e.g. ±1 s around the divergence) into a
    persisted hard-cases pool (IndexedDB in-browser, JSONL on disk).
- Wire the miner into `/sim-to-real`, `/ramp`, `/carchase`,
  `/raceprimitives`. Every demo session adds to the pool.
- Refit trigger: when the pool grows by > N new trials *or* on
  explicit user click, the Model Lab schedules a residual-MLP
  refit that includes the pool as oversampled examples.
- A "hard-cases" tab on Model Lab that visualizes each captured
  trial (the same recorder JSON the user already exports manually).

**Acceptance gate**
- After 10 minutes of mixed Free-Drive across the demos, the pool
  contains ≥ 50 hard examples and a one-click refit drops `/sim-to-real`
  gap-RMS on the *most-recent-session* slice by ≥ 30 %.
- The Free-Drive lift-off regression continues to pass after refit
  (no catastrophic forgetting on the original eval set).

### Phase 4 — Chassis-config diversification

`LearnableVehicleConfig` is supposed to be an *input* to the v2 model
(so one model generalizes across vehicles). In practice the dataset
has been collected with a single default RWD config. The model can
parrot the default config but cannot interpolate to a CarChase robber
chassis or a Ramp chassis — because it has no evidence those exist.

**Deliverables**
- Sweep dataset across realistic config variations:
  - `engineForce ∈ [2500, 6000]`, `brakeForce ∈ [1500, 3500]`,
    `maxSteerAngle ∈ [0.4, 0.8]`, `wheelBase ∈ [1.2, 2.0]`,
    `wheelTrack ∈ [0.7, 1.0]`, `driveTrain ∈ {rwd, fwd, awd}`.
- Phase 1 and Phase 3 trials repeated across the config Latin-hypercube
  sample (~16 configs is enough for a small parametric backbone).
- `configKey` in the trial store finally has more than one value;
  fitter learns the config dimensions of the parametric backbone.
- Split policy (per Phase 0): of the ~16 sampled configs, **2 are
  held out entirely as the test split** (no train, no val, no active
  exploration). The cross-config generalization number is the
  test-RMS on those held-out configs — the honest measure of "did
  the model learn the config dimensions, or memorize the train set".

**Acceptance gate**
- Held-out config evaluation: train on configs `[A..N]`, evaluate on
  config `O` (unseen). RMS at horizon=1.6 s within 1.5× the
  in-distribution RMS.
- Visible in the Model Lab: a "config interpolation" slider that
  shows the model's prediction smoothly varying as engineForce
  sweeps from 2500 → 6000.

### Phase 5 — Surface / terrain variation

Demos drive on heightfields (Ramp), inclines (ObstacleCourse landing),
and flat ground (RacePrimitives). The training collector currently
runs on flat ground only. The model has no notion of pitched chassis
attitude affecting yaw response.

**Deliverables**
- `kinocat/vehicle/car/terrain-conditions.ts`: parameterized terrain
  factories (flat, single-slope, washboard, heightfield-fragment).
- Headless trial harness accepts a `terrain` option; reuses
  `createHeightfieldCollider` from the rapier adapter.
- Phase 1 and Phase 3 trials repeated with terrain variation.
- Optional state addition: chassis pitch + roll (already in the
  Rapier state, just not exposed via `CarKinematicState`). Phase 5
  is the natural moment to widen the state vector if needed.

**Acceptance gate**
- Sim-to-real ghost on `/ramp` matches the real chassis through the
  ramp launch + landing.

### Phase 6 — Active uncertainty-driven exploration v2

The current active explorer (`proposeNextBatch`) is uncertainty-weighted
but constrained to the same 24-cell grid. With Phases 1-5 in place,
the proposer should operate on the **continuous maneuver-parameter
space** so it can ask for "more `trailBrake(brakeForce=2800,
releaseTime=0.18, steerRamp=0.45)` near `(speed=18, yawRate=0.7)`".

**Deliverables**
- Maneuver parameter space treated as the action space for the
  proposer. The residual MLP ensemble's disagreement at the predicted
  trajectory of a candidate maneuver = the acquisition function.
- Bayesian-optimization-flavored picker (any black-box optimizer is
  fine — random-shooting with the ensemble-disagreement score is the
  cheapest acceptable baseline).
- Stops when ensemble disagreement plateaus across the coverage grid.

**Acceptance gate**
- Convergence: adding 100 more trials chosen by the explorer drops
  held-out RMS by < 2%. (Diminishing returns = we've sufficiently
  covered the regime.)

---

## Cross-cutting infrastructure changes

Land alongside the phases above; not phase-gated.

### Headless training CLI + preloaded model artifact ⭐

The browser-side `/raceprimitives` "train" button is fine for
exploration, but a serious training run wants:
- to be reproducible (deterministic seed, no flaky tab focus),
- to run unattended (overnight, in CI, after a refactor),
- to ship its output as the **default model every demo loads on
  first visit** — so a fresh user sees a competent v2 from tick zero
  instead of staring at the default-parametric baseline.

**Deliverables**
- `core/scripts/train-v2.ts` (or `demos/scripts/train-v2.ts` — wherever
  the CarV2 specifics live; the generic loop lives in
  `kinocat/training`). Pure Node entrypoint:
  - Boots Rapier headlessly (already supported by the headless trial
    harness — no DOM).
  - Reuses `CarV2TrainingPipeline` + `runOfflineTrainingCore` —
    **identical code path** to the in-browser training, so the CLI
    cannot diverge. (Same lesson as the action-space unification that
    landed previously — `wheeledFromNormalized` in
    `core/src/vehicle/car/wheeled.ts` is the single source of truth
    for normalized-→-wheeled controls used by every driving source.
    Apply the same single-path discipline here.)
  - Streams the same `TrainingEvent` stream the browser consumes,
    pretty-printed to stdout (progress bars per round, per-horizon
    train / val / test RMS, coverage summary).
  - On completion, writes:
    - `demos/public/models/v2-default.json` — the serialized
      `LearnedVehicleModel` (params + residual MLP ensemble + config),
      using the existing `v2-model-persistence.ts` format that
      `loadV2Model` already understands.
    - `demos/public/models/v2-default.manifest.json` — provenance
      sidecar: git sha, dataset hash, train/val/test RMS at each
      horizon, coverage summary, runtime, seed. Reviewable artifact.
    - `docs/datasets/<timestamp>.json` — the dataset manifest from
      the cross-cutting "Dataset persistence" section below.
- `package.json` scripts:
  - `pnpm train` — default profile: ~10 rounds, ~600 trials/round,
    full maneuver library, single default config. ~5-10 min local.
  - `pnpm train:quick` — smoke profile: 2 rounds, 60 trials/round.
    For "did I break the pipeline" checks during refactors. ~30 s.
  - `pnpm train:sweep` — Phase 4 config sweep. ~30-60 min local.
  - `pnpm train:overnight` — Phase 1+2+4 combined, max rounds until
    val-RMS plateau. For users who want to throw a laptop at it.
- Each script takes optional CLI flags (`--seed`, `--rounds`,
  `--out`, `--profile`) parsed via `node:util parseArgs` — no
  dependency added.
- **Preloaded model wiring**: `/raceprimitives`, `/sim-to-real`,
  and `/model-lab` check for `models/v2-default.json` on mount and
  load it as the starting model. The "train" button now reads
  "retrain from preloaded" — first-tick behavior is competent,
  iteration is incremental. Strict offline-first: if the file is
  absent (fresh checkout, never ran CLI) demos fall back to the
  default-parametric model as today.
- CI smoke check: `pnpm train:quick` runs in CI on PRs that touch
  `core/src/{agent,training,learning,vehicle}/` so a refactor that
  silently breaks training is caught before it lands.

**Why this matters for the divergence story**
One code path for browser training + CLI training + CI smoke means a
single edit propagates everywhere — no risk of the CLI silently
diverging from the in-browser pipeline. The preloaded artifact also
closes the UX loop: fresh visitors shouldn't have to wait for a
5-minute training run to see the model actually behave.

### Headless race benchmark CLI ⭐

Training produces a model; we need an honest, repeatable way to
**actually race it** without opening a browser. The race scenario in
`/raceprimitives` is the natural benchmark — it already exercises the
planner, the model, the chassis, and the lap-time metric we care
about — but it's currently locked behind Three.js + a manual page
load. Strip the renderer, keep Rapier, run it headless from CLI.

Use cases:
- **Track progress over time**: `pnpm race` after each training run,
  log lap time to a JSON ledger. Plot lap-time-vs-commit to see
  whether dataset / model changes actually help, free of placebo.
- **A/B different models or algos**: race v2-default.json against
  v2-experimental.json against the kinematic baseline against the
  parametric-only baseline, all in one command, all with identical
  seeds + tracks + opponent.
- **Catch regressions before merge**: a CI job runs the benchmark
  with the preloaded model and fails if median lap time regresses
  by > X% vs the main branch baseline.
- **Phase 3 acceptance gate is measurable**: "v2 beats kinematic on
  lap time" becomes `pnpm race --models v2,kinematic --laps 5 --seed 42`
  with a pass/fail exit code, not a subjective "looked faster".

**Deliverables**
- `core/src/scene/race-scenario.ts` — extract the race logic that
  currently lives inside `demos/app/raceprimitives/RacePrimitives.tsx`
  into a renderer-free, generic `<S, C>` scenario runner:
  - `RaceScenario<S, C>` interface: `setup(world)`, `tick(dt) →
    {laps, status}`, `done(): RaceResult`. Generic so airplane /
    other vehicles can reuse it.
  - `runHeadlessRace<S, C>(opts: { scenario, body, drivers,
    maxSimTime, seed })` — uses `SceneController` (already
    renderer-free) to drive the sim. No Three.js dependency
    anywhere in the call graph. Verified by the existing
    `agnostic-core.test.ts` rule (nothing under `core/src/`
    outside `core/src/adapters/` may import Three.js or Rapier
    directly).
- `demos/app/lib/race-primitives-scenario.ts` — car-specific
  `RaceScenario<CarKinematicState, WheeledCarControls>` impl that
  builds the same track + opponents the demo page uses. Both the
  CLI and the React page consume this. Same single-path discipline
  as the training pipeline: no duplicate scenario code.
- `demos/app/raceprimitives/RacePrimitives.tsx` refactor: the React
  component becomes a thin shell — instantiates the scenario,
  mounts the Three.js renderer, and renders one frame per tick.
  All race logic moves out. Mirrors the broader "thin React layer"
  goal driving the rest of this codebase.
- `demos/scripts/race.ts` — Node entrypoint that boots Rapier
  headlessly, loads N models from disk, runs the scenario with
  each, prints a comparison table, optionally writes JSON results.
- `package.json` scripts:
  - `pnpm race` — default: race the preloaded `models/v2-default.json`
    against the kinematic baseline + parametric-only baseline. 3
    laps, fixed seed, ~30 s wall time.
  - `pnpm race -- --models a.json,b.json,c.json` — custom matchup.
  - `pnpm race -- --laps 5 --seed 17 --json out.json` — long run
    with explicit seed, persist results.
  - `pnpm race -- --ledger docs/race-results/` — append to a
    versioned-on-disk results ledger (git-tracked JSONL) so we
    can plot lap-time-vs-commit over the project's life.

**Output format**

```
$ pnpm race
race-primitives benchmark — seed=42 laps=3 dt=1/60

model                  status  laps  lap1   lap2   lap3   best   avg     crashes
v2-default.json        ✓        3   18.4s  17.9s  17.7s  17.7s  18.0s   0
kinematic-baseline     ✓        3   19.1s  18.8s  18.6s  18.6s  18.8s   0
parametric-only        ✗        2   22.1s  DNF    -      22.1s  -       1 (off-track t=44s)

→ v2-default beats kinematic by 4.1% (avg) — PASS
```

A row's `lap_n` value is the wall-clock lap time the chassis took
on lap n; `DNF` = did-not-finish within `maxSimTime`; `crashes` =
times the chassis left the track or rolled.

**Acceptance gate**
- `pnpm race` runs in ≤ 60 s on a laptop, deterministic for a fixed
  seed, prints the comparison table, exits 0 if every model
  completed all laps and v2 ≥ baseline.
- The same `RaceScenario` instance, when given to the React page,
  produces the same lap times bit-for-bit (modulo render-frame
  scheduling) — proves no logic divergence between CLI and demo.
- CI job runs `pnpm race -- --quick` on every PR; fails if median
  lap regresses > 3 % vs main.

**How this slots into the rest of the plan**
- Phase 3's acceptance gate ("v2 beats kinematic on lap time") is now
  measured by this CLI, not by eyeballing the React page.
- Phase 4's acceptance gate ("cross-config generalization") gets a
  second column: race the model against held-out chassis configs
  via `--chassis <config.json>` flag.
- The `docs/race-results/` ledger pairs with the `docs/datasets/`
  manifests — together they form a complete provenance trail:
  *which dataset trained which model and how it raced*. No more
  "I think v2 was faster last week" guesses.

### Trial store schema additions
- Add `maneuverId: string` and `maneuverParams: Record<string, number>`
  fields to `Trial<S, C, Cfg>` so the coverage meter can group by
  maneuver type.
- Add `scenarioId?: string` for Phase 3 closed-loop trials.
- Add `terrainKind?: string` for Phase 5.

### Dataset persistence
- `/raceprimitives` already serializes the trained model. Extend to
  serialize the trial store too (or at least its hash + summary) so
  the model artifact is reproducible and the dataset itself is a
  reviewable artifact.
- Per-phase dataset snapshots committed under `docs/datasets/` as
  small JSON manifests (counts, coverage summary, sha) — not the
  trial payloads themselves.

### Coverage acceptance gates as CI checks
- `core/test/training/coverage-acceptance.test.ts`: loads the manifest
  for the current dataset, asserts minimum trial counts per
  coverage-bin, and fails the build if any bin falls below the gate.
  This prevents the silent-coverage-regression mode the current
  pipeline is in.

### Manual replay shortcut
- The sim-to-real debug recorder already emits JSON. Even before the
  Phase 3.5 automated miner ships, wire a one-click "convert this
  recording to a TrialSpec" button so the user can promote any
  notable divergence into the training set manually. (Subsumed by
  Phase 3.5 once that lands; this is the cheap stopgap.)

---

## Suggested execution order

1. **Phase 0** — week 1. Pure observability; no model change. Lock the
   eval set here.
2. **Phase 1** — week 1-2. Lands the bulk of the realism gain. The OU
   random-walk class alone (60 % of budget) should fix the Free-Drive
   lift-off bug.
3. **Phase 3** — week 2-3. Closed-loop trials are where lap-time
   actually crosses the kinematic baseline.
4. **Phase 3.5** — week 3 (can run in parallel with Phase 3 since it
   only depends on Phase 0's recorder integration). Turns every demo
   session into a continuous data source.
5. **Phase 2** — week 3-4. Only if Phase 1 + 3 + 3.5 don't cover
   mid-corner initial conditions adequately.
6. **Phase 4** — week 4. After lap-time is won on the default config,
   widen to generalize.
7. **Phase 5** — week 5. Only required if a demo with non-flat terrain
   still shows sim-to-real divergence.
8. **Phase 6** — week 5-6. Final polish; only worthwhile once Phases
   1-5 have populated the coverage map.

---

## Open questions to resolve before Phase 1

1. **Eval-set lock-in policy**: fix a *named* held-out scenario set
   (e.g. 5 specific track laps + 5 specific maneuvers) and never
   train on it, OR keep the last-25%-of-trials policy? The named set
   is mandatory for honest cross-phase comparison.
2. **Maneuver-parameter ranges**: should they be set by domain
   knowledge (current draft above) or learned from the planner's
   own historical query distribution? Probably both — bootstrap from
   domain knowledge, refine from planner logs.
3. **Closed-loop bootstrap**: the planner in Phase 3 needs *some*
   model to drive. Use the parametric-only v2 (from Phase 1 fit) as
   the bootstrap; then iterate.
4. **Demo coverage of `/sim-to-real`**: should every Phase 3 scenario
   record into the sim-to-real recorder format so we get the visual
   debug for free? (Yes, recommended; trivial wiring.)
