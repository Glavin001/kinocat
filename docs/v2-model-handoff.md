# v2 Learned-Vehicle Model & Race Demo — Handoff

PR: [Glavin001/kinocat#19](https://github.com/Glavin001/kinocat/pull/19) (branch `claude/kind-gates-R5Lyf`)

This document captures the state of the v2 learned-vehicle-model work on
the `/raceprimitives` page (and the `/primitive-explorer` diagnostic
page) as of commit `6a3f0e3`. Self-contained — assumes no prior context
from the PR conversation.

---

## TL;DR

The original `/raceprimitives` demo had a learned dynamics model with a
low ceiling (5 parameters, online refit only on race data). This PR
turned it into a model-based-planning stack: richer state, native action
space, config-aware extended-parametric model + residual MLP ensemble,
offline + active-exploration training pipeline, multi-goal A* planner,
and observability tooling (action-space explorer page, live planner
diagnostics, one-click debug-report export).

**End state**: all infrastructure shipped + 296 tests pass. **The v2
car is not yet beating the kinematic car on lap time.** The most-recent
fix (residual MLP wired into the training driver, commit `6a3f0e3`)
hasn't been validated on the track yet — user has not retrained + raced
since that commit landed.

---

## Architecture overview

### Conceptual stack

```
                              agnostic core (kinocat/*)
                                       │
                                       ▼
state (Markov, generic)        ┌──── apply_motion (learned) ────┐
controls (generic vehicle) ────│  parametric backbone (16 params)│── next_state
config (vehicle params) ───────│  + residual MLP ensemble (3 nets)│
                               └─────────────────────────────────┘
                                          ▲                │
                              trained via │                │ evaluated via
                              core/learning ──┐            ▼
                                              │     core/planner (IGHA* + multi-goal)
                                              │           │
                                              │           │
                                              ▼           ▼
                                ┌─────────────────────────────────────┐
                                │     Rapier adapter (optional dep)    │
                                │   readState / applyWheeledControls   │
                                │      teleportFull / headless-trial   │
                                └─────────────────────────────────────┘
                                              ▲           │
                                              │           ▼
                                         demos/app (use case + UI)
                                         /raceprimitives + /primitive-explorer
                                         + Model Lab, debug-report, etc.
```

**Hard architecture rule** (codified by a test — see `core/test/agnostic-core.test.ts`):
- No file under `core/src/` (excluding `core/src/adapters/`) may mention
  Rapier, race, or any specific scenario.
- Rapier-specific code lives in `core/src/adapters/rapier/`.
- Use-case-specific code lives in `demos/`.

### Key generic abstractions added in core

| File | Purpose |
|---|---|
| `core/src/agent/types.ts` | `VehicleState` now has optional `yawRate` + `lateralVelocity` (Markov state for v2) |
| `core/src/agent/controls.ts` | `WheeledControls = (steer, driveForce, brakeForce)` — generic native action shape |
| `core/src/agent/vehicle-config.ts` | `LearnableVehicleConfig` — physical params a learned model can identify |
| `core/src/agent/vehicle-model.ts` | `LearnedVehicleParamsV2` (16 params); `parametricForwardV2`; `learnedForwardSimV2` (parametric + optional MLP ensemble); `predictWithUncertainty` |
| `core/src/internal/mlp.ts` | Hand-coded dense net + analytic backward + Adam; `serializeMLP` / `deserializeMLP` |
| `core/src/learning/trial-store.ts` | Generic typed trial DB |
| `core/src/learning/parametric-fit.ts` | Generic Nelder-Mead fitter (lifted from demos) with regularization support |
| `core/src/learning/residual-mlp-fit.ts` | Generic SGD/Adam loop for MLP ensemble training |
| `core/src/learning/evaluate.ts` | Open-loop divergence metric at horizons |
| `core/src/learning/active-explorer.ts` | Generic uncertainty-weighted trial proposer |
| `core/src/primitives/control-sets-wheeled.ts` | Coarse + fine native-action primitive generators (built but the demo uses speed-aware sets instead — see below) |
| `core/src/environment/multi-goal.ts` | `MultiGoalEnvironment<S>` — wraps any Environment to do single A* through a sequence of gates |
| `core/src/planner/plan-vehicle-multi.ts` | `planVehicleMultiGoal` convenience wrapper |

### Rapier adapter extensions

| File | Purpose |
|---|---|
| `core/src/adapters/rapier/raycast-vehicle.ts` | `readState` populates yawRate + lateralVelocity; `teleportFull` seeds full kinematic state; `applyWheeledControls` writes native action to wheels |
| `core/src/adapters/rapier/headless-trial.ts` | Headless trial harness — teleport, settle, run controls trace, sample, detect NaN / off-arena / spin |

### Demo-side wiring (use-case specific)

| File | Purpose |
|---|---|
| `demos/app/lib/race-primitives-scenarios.ts` | Race course geometry; `buildKinematicLibrary` (legacy); `buildLearnedRaceLibrary` (legacy 5-param); **`buildLearnedRaceLibraryV2`** (the working v2 path — per-bucket durations + speed-aware controls); `planRaceMultiGoal` |
| `demos/app/lib/training-driver.ts` | `runOfflineTraining` orchestrating headless trials → parametric fit → residual MLP fit → evaluate → repeat |
| `demos/app/lib/v2-model-persistence.ts` | localStorage + JSON round-trip for the trained v2 model (params + config + residual ensemble) |
| `demos/app/lib/primitive-diagnostics.ts` | Pure functions for primitive-library resolution metrics (hull area, angular gaps, pairwise mismatch) |
| `demos/app/lib/debug-report.ts` | Markdown debug-report generator (the 🐛 export button) |
| `demos/app/components/ModelLab.tsx` | Top-right panel in `/raceprimitives`: training controls + loss charts + headline divergence stats |
| `demos/app/components/PrimitiveFanPlot.tsx` | Canvas fan plot used by `/primitive-explorer` |
| `demos/app/components/PrimitiveOverlayPlot.tsx` | Endpoint overlay with kinematic↔v2 connector lines |
| `demos/app/primitive-explorer/PrimitiveExplorer.tsx` | The standalone diagnostic page |
| `demos/app/raceprimitives/RacePrimitives.tsx` | Main race page (≈ 2400 lines now) — wires everything together; mobile-responsive |

---

## What's working ✅

### Infrastructure (all verified by tests)

- **Generic agnostic core** — `core/` has zero domain knowledge of
  Rapier or racing. The architecture-invariant test
  (`core/test/agnostic-core.test.ts`) grep-fails any leak.
- **Extended v2 parametric model** — friction-circle clamp, yaw-rate
  inertia, asymmetric understeer/oversteer, lateral-velocity dynamics,
  config-aware (`core/test/agent/vehicle-model-v2.test.ts` passes).
- **MLP utility** — analytic gradients match finite-differences to
  1e-4; Adam reduces synthetic regression loss
  (`core/test/internal/mlp.test.ts`).
- **Generic Nelder-Mead fitter** — recovers known parameters from
  synthetic data within 1% (`core/test/learning/parametric-fit.test.ts`).
- **Multi-goal A* wrapper** — works against R2 and the race scenarios;
  proper dedup with gate-index in keys; admissible heuristic
  (`core/test/environment/multi-goal.test.ts`,
  `demos/test/race-primitives-multigoal.test.ts`).
- **Headless trial harness** — deterministic under fixed seed;
  pathological outcomes discarded with reason
  (`core/test/adapters/headless-trial.test.ts`).
- **Active explorer** — cells with high error × low count score
  highest (`core/test/learning/active-explorer.test.ts`).
- **Trial-store serialization** — byte-stable round trip
  (`core/test/learning/trial-store.test.ts`).
- **Debug-report generator** — all sections present, handles
  no-v2-loaded case (`demos/test/debug-report.test.ts`).

### Demo UI

- **`/raceprimitives`** loads, runs, races. Both cars finish laps.
- **Model Lab panel** (top-right) opens, trains a v2 model offline
  (~30s with parametric, +~30s with residual MLP), displays loss curves
  and headline divergence vs legacy + kinematic baselines.
- **`/primitive-explorer`** standalone page shows side-by-side fan
  plots for kinematic vs v2 at each speed bucket; overlay-and-diff
  view; resolution diagnostics.
- **🐛 export debug** button (TopBar, always visible) generates a
  Markdown report covering everything, copies to clipboard, downloads
  as `.md`.
- **Mobile-responsive** — uses `useIsMobile(820)` to swap absolute-
  positioned overlays for stacked / bottom-sheet layouts on phones.
- **Persistence** — trained v2 model + `useV2` toggle survive page
  reload (localStorage). Backward-compat for v1 persistence format
  (parametric-only) auto-migrates.
- **Live planner diagnostics** — per-car HUD shows last replan ms,
  plan age, success rate, failed streak with green/yellow/red color
  thresholds.

### Test status (last full run, commit `6a3f0e3`)

```
Test Files  1 failed | 51 passed (52)
Tests       2 failed | 296 passed (298)
```

The 2 failures are **pre-existing** in `demos/test/carchase-scenarios.test.ts`
(line 33 assertion + line 47 timeout). They were failing on the merge
commit that landed before this PR opened (`e7a76ec`) and have never been
touched by anything in this PR. Verified via `git stash && pnpm test`
early in the PR's life.

---

## What's NOT working ❌ (the open problem)

### The headline issue

**The v2 car is still slower than the kinematic car on lap time.**

User's debug exports across the PR's lifetime show v2 lap times in the
50–60s range vs kinematic at 32–40s. v2's open-loop pred err has
oscillated:

| State | Open-loop RMS @ 1s |
|---|---|
| Original (5-param online refit) | ~2.4 m |
| v2 parametric, unphysical bounds | 0.92 m (overfit to noise via bound-pinning) |
| v2 parametric, physical bounds | 2.50 m |
| v2 parametric, tight bounds + regularization | 4.49 m (the regularization made it worse — over-pulled) |
| v2 parametric + residual MLP (commit `6a3f0e3`) | **UNVERIFIED — user hasn't retrained since this landed** |

### Root cause (theory, partially confirmed)

The 16-parameter `parametricForwardV2` model is **structurally
insufficient** to represent Rapier's full nonlinear behaviors:

- Engine torque falloff at high RPM (not in the model)
- Tire slip nonlinearity (the friction-circle is a linear clamp; real
  tires Pacejka-curve into peak grip and out)
- Suspension load transfer under brake/accel (not modeled at all)
- Combined-input interactions (brake-while-turning has nonlinear
  yaw response)

When the fit was given loose bounds, it found locally-optimal but
unphysical parameter values (e.g. `brakeScale = 3.5` claiming the
chassis brakes at 2× the commanded force). With tight bounds, it gets
stuck near defaults and pred err climbs because the bounds prevent
the over-fitting that was — at least — fitting the training data.

**The residual MLP (commit `6a3f0e3`) is the architectural fix:**
3-member ensemble × 200 epochs trained on per-tick residuals between
the parametric prediction and the recorded Rapier sample. The
parametric stays physical; the MLP captures everything the parametric
can't.

### What needs to happen to validate

1. Hard-refresh `/raceprimitives` (Vercel deploy from commit `6a3f0e3`)
2. Click **🐛 export debug** — verify the loaded v2 model has the
   new persistence v2 format (the v2-export should include
   `residualEnsembleJson` field)
3. Click **Clear cached** in Model Lab (the previously-saved model
   has no ensemble; need a fresh train)
4. Train v2 (4 rounds × 96 trials × 90 ticks recommended). Training
   will take **~60s now** (30s parametric + ~30s residual MLP on the
   final round)
5. Click **🐛 export debug** again — check:
   - Open-loop RMS @ 1s should drop substantially (target: <1.5m)
   - A new `parametricOnly` baseline appears in the divergence chart
     showing how much the MLP contributed
6. Race with v2 toggled on; check lap times

If pred err is still > 1.5m after step 5, the next levers are:
- Bump MLP hidden-layer size (32 → 64)
- Bump ensemble size (3 → 5)
- More training epochs (200 → 400)
- Larger / more diverse trial sweep

### Other open issues

- **Pre-existing carchase test failures** — not caused by this PR but
  trip CI on every push. The 2 failures are in
  `demos/test/carchase-scenarios.test.ts` (line 33: `co.result.found`
  returning false; line 47: timeout). Probably a planner-budget or
  course-tuning issue on the carchase scenario. Worth fixing
  separately so CI is actually informative.

---

## How to use the tools

### Workflow for diagnosing v2 driving issues

1. **Race on `/raceprimitives`** — observe behavior (overshoots,
   circles, slowness)
2. **Click 🐛 export debug** during or just after the race
3. **Paste the markdown** into a chat / issue — it includes:
   - v2 model parameters (look for values pinned to bounds — that
     indicates the fit is over-extending)
   - Open-loop pred err per car (the v2 vs kinematic comparison is
     the headline accuracy number)
   - Planner diagnostics per car (success rate, plan age, failed
     streak)
   - Primitive-library resolution table (hull area at each speed
     bucket — sub-1 m² is degenerate)

### Workflow for inspecting the action space

1. Go to `/primitive-explorer`
2. Cycle through speed buckets (0 / 10 / 20 / 28 m/s)
3. Compare kinematic (left, pink) vs v2 (right, cyan) fan plots
4. v2 hull should be > 5 m² at every speed bucket; if it collapses,
   primitives can't differentiate
5. The overlay-and-diff view below the fans shows per-control
   disagreement when both libraries share controls (only works for
   legacy v1 vs kinematic; v2 uses native wheeled controls so the
   pairing is intentionally N/A)

### Workflow for retraining

1. Open Model Lab (top-right ▲ button on `/raceprimitives`)
2. Settings (recommended): **4 rounds × 96 trials × 90 ticks × seed 42**
3. Click **Train v2 model** — ~60s
4. Watch loss curves — total loss should decrease across rounds
5. After training: open-loop divergence chart shows v2 vs legacy vs
   kinematic at multiple horizons
6. **Click "Use v2 library for learned car"** to apply

---

## Critical files map (where to look)

### To change how v2 dynamics work
- `core/src/agent/vehicle-model.ts` — parametric backbone + MLP-ensemble
  application. `PARAMS_V2_LO / PARAMS_V2_HI` are the fit bounds;
  `DEFAULT_LEARNED_PARAMS_V2` is the prior the regularization pulls
  toward.

### To change how the v2 library is built (primitive shapes / durations)
- `demos/app/lib/race-primitives-scenarios.ts` — `buildLearnedRaceLibraryV2`
  (per-bucket durations + speed-aware controls). The four helpers
  `lowSpeedV2Controls`, `midSpeedV2Controls`, `highSpeedV2Controls`,
  `topSpeedV2Controls` define what actions exist at each speed bucket.

### To change training behavior
- `demos/app/lib/training-driver.ts` — `runOfflineTraining`. Look for
  `buildSeedGrid()` (the round-0 trial menu — includes speeds 0–28
  m/s as of commit `4571abe`), `runParametricFit` call (regularization
  config is just above it), `runResidualMLPFit` call (MLP shape +
  epochs + ensemble size).

### To change the planner
- `core/src/planner/plan-vehicle-multi.ts` — multi-goal wrapper
- `core/src/environment/multi-goal.ts` — generic multi-goal Environment
- `demos/app/raceprimitives/RacePrimitives.tsx` line ~83 —
  `PLAN_LOOKAHEAD_COUNT = 2` (number of gates the multi-goal A* sees
  per replan; bump to 3 if you increase replan budget)

### To change race behavior
- `demos/app/raceprimitives/RacePrimitives.tsx` — the giant component.
  `replan()` function does the multi-goal call. The course geometry
  (waypoints, RACE_BOUNDS, RACE_PALETTE) is in
  `demos/app/lib/race-primitives-scenarios.ts`.

### To regenerate / change diagnostics
- `demos/app/lib/primitive-diagnostics.ts` — pure functions
- `demos/app/lib/debug-report.ts` — markdown report builder

---

## Configuration notes

### Planner & race-cycle config

Located at top of `demos/app/lib/race-primitives-scenarios.ts`:

```ts
export const RACE_REPLAN_BUDGET_MS = 120;   // per-car per-replan deadline
export const RACE_MAX_EXPANSIONS = 30000;
export const RACE_ARRIVE_RADIUS = 2.5;       // demo advances loopIndex inside this
export const RACE_PLANNER_GATE_RADIUS = 1.8; // planner radius — STRICTLY < arrive radius
                                             // (so every valid plan reaches the advance circle)
```

And in `RacePrimitives.tsx`:
```ts
const REPLAN_INTERVAL_MS = 300;          // ~3.3 Hz replanning
const PLAN_LOOKAHEAD_COUNT = 2;          // multi-goal lookahead
const TRACKER_MAX_LATERAL_ACCEL = 12;    // pure-pursuit clip
```

### Bounds & regularization for v2 fit

Bounds are in `core/src/agent/vehicle-model.ts`
(`PARAMS_V2_LO` / `PARAMS_V2_HI`). Tightened in commit `602ec27`
to physically reasonable values.

Regularization is in `demos/app/lib/training-driver.ts` (line ~365):
- `strength: 0.05` — moderate pull toward defaults
- Per-parameter `REG_SCALES` define how tight each param's pull is

If the fit still pins to walls, either widen bounds (re-checking
they're physical) or stiffen regularization (bump strength).

### Trial seed grid

`buildSeedGrid()` in `training-driver.ts`. Currently includes speeds
0–28 m/s with focused high-speed brake-into-corner and gentle-turn
trials. `extremeProbes()` adds always-included saturation probes.

---

## Build / test / deploy

```bash
# from repo root
pnpm install              # one-time
pnpm --filter kinocat build  # builds core (required before demos can find it)
pnpm test                 # full test suite — ~60s
pnpm --filter @kinocat/demos typecheck
cd demos && pnpm dev      # Next.js dev server (port 3000)
cd demos && pnpm build    # production build
```

Vercel deploys automatically on push to the PR branch. Wait ~1
minute after pushing, then hard-refresh (Cmd+Shift+R).

---

## Commit history of major changes

| Commit | Summary |
|---|---|
| `3d4056f` | Foundation: VehicleState extensions, WheeledControls, vehicle-config, MLP utility, learning helpers, Rapier adapter extensions |
| `(early)` | Plan file: `/root/.claude/plans/in-http-localhost-3000-raceprimitives-pa-enchanted-micali.md` |
| `(several)` | Tests for all the new core helpers |
| `(several)` | ModelLab UI, race scenario wiring, primitive-explorer page |
| `4571abe` | The "all-four fixes": per-bucket primitive durations, native wheeled controls, speed-aware control sets, broader training sweep + relaxed defaults |
| `5f961be` | Aligned planner gate radius (1.8m) below demo advance radius (2.5m) — fixed the "U-turn back to missed gate" loop |
| `8b59fba` | Multi-goal A* (global trade-off across N gates) |
| `ad37d3b` | Per-replan planner diagnostics in HUD |
| `5506a84` | Mobile-responsive UI overhaul |
| `a6bcb44` | `/primitive-explorer` diagnostic page |
| `576ce06` | 🐛 export debug button |
| `602ec27` | Tightened parameter bounds + wired regularization (responded to user's debug showing 6 pinned params) |
| `6a3f0e3` | **LATEST**: residual MLP ensemble wired into training-driver (responded to user's debug showing structural fit limit) |

---

## Things to remember / gotchas

1. **The cached v2 model in localStorage is stale after major model
   changes.** Always click "Clear cached" in Model Lab after pulling
   new code that changes the v2 model architecture, then retrain.

2. **Vercel can take 1–2 minutes to redeploy.** A debug report showing
   old behavior may be because the deploy hasn't propagated. Hard
   refresh.

3. **`buildLearnedRaceLibraryV2` does NOT take training-style args**
   like `params`. It takes a `LearnedVehicleModel` (params + config +
   ensemble bundle). The model comes from `runOfflineTraining` or
   `loadV2Model()`.

4. **Per-control mismatch in `/primitive-explorer` requires same
   control vocabulary.** Kinematic library uses `[curvature, targetSpeed]`;
   v2 library uses `[steer, driveForce, brakeForce]`. The connector-line
   overlay only works between libraries that share controls (e.g.
   kinematic vs legacy 5-param). For kinematic vs v2 the fans render
   side-by-side but no per-control lines.

5. **Pure-pursuit execution is opaque to plan provenance.** The
   planner can use any primitive vocabulary; the execution layer
   just follows the resulting state path with pure-pursuit. This is
   why v2 with native wheeled controls doesn't require a new
   executor.

6. **The plan file** at `/root/.claude/plans/in-http-localhost-3000-raceprimitives-pa-enchanted-micali.md`
   has the original-approved architecture doc and follow-up plans.
   May not be accessible to future sessions but worth a try.

7. **The architecture-invariant test will block any future PR** that
   imports `@dimforge/rapier3d-compat` or mentions "Rapier" / "race" /
   "carchase" tokens in `core/src/` outside `core/src/adapters/`. If
   you intentionally want to leak something into core, you'll need to
   update `core/test/agnostic-core.test.ts` along with it.

8. **MLP serialization** for the residual ensemble uses the existing
   `serializeMLP` / `deserializeMLP` in `core/src/internal/mlp.ts`,
   re-exported through `kinocat/agent`. Storage size: ~25 KB per
   trained model (well within localStorage quota).

9. **The training-driver test** (`demos/test/training-driver.test.ts`)
   takes ~19 seconds because the residual MLP fit now runs as part
   of the orchestration. Don't be alarmed.

10. **CI carchase failures are pre-existing.** Don't waste time
    investigating them as PR regressions. Worth fixing separately
    so CI noise stops.

---

## Suggested next steps

### Immediate (validate the residual MLP)

1. Retrain v2 after the `6a3f0e3` deploy lands
2. Compare debug-report numbers: `parametricOnly` baseline vs the
   full v2 model — that's the MLP's contribution
3. If pred err < 1.5 m, race and see if v2 wins lap time
4. If pred err still > 1.5 m, the next knobs to try:
   - Bump MLP hidden width (32 → 64 in `training-driver.ts`)
   - Bump ensemble size (3 → 5)
   - More epochs (200 → 400)
   - More diverse trial sweep (add speed-yaw-rate cross product)

### Medium-term improvements (if v2 still doesn't win)

1. **Uncertainty-aware planner cost** — the v2 model already returns
   an ensemble std via `predictWithUncertainty`. The planner could
   penalize high-uncertainty trajectories. Add a `costModifier` hook
   to `PlanRequest.environment`.
2. **Online learning** — currently disabled when v2 is active. Could
   refit a small residual head from race data (similar to the
   original 5-param online refit).
3. **iLQR / DDP refinement** on the committed plan prefix. Multi-goal
   A* picks the strategy; iLQR perfects execution. Already discussed
   in the original plan as a non-goal but the right next step if the
   raw model accuracy is good but lap times still lag.
4. **Replace pure-pursuit with MPC** — pure-pursuit's lateral-accel
   clip (12 m/s²) is currently the execution-side bottleneck. An MPC
   tracker that uses the same v2 model would let the chassis push
   harder when v2 predicts grip is available.

### Architectural cleanups (lower priority)

1. **Migrate the legacy 5-param fitter** (`fitParams` /
   `fitParamsOnline` in `demos/app/lib/learn-primitives.ts`) onto the
   generic `runParametricFit` in core. Currently legacy still has its
   own Nelder-Mead implementation.
2. **Headless trial harness in Web Workers** for parallel collection.
   Currently single-threaded; would dramatically speed up training.
3. **Generalize the speed-aware control-set helpers** in
   `demos/app/lib/race-primitives-scenarios.ts` (`lowSpeedV2Controls`
   etc.) into a generic `speedAwareWheeledControls(config, schedule)`
   in core.
4. **Fix the 2 pre-existing carchase tests.** Probably a planner-
   budget issue but worth investigating.

### UX polish

1. **Toast confirmation for debug-export** — currently the button
   text changes briefly ("✓ copied · saved"). Could be a proper
   toast with the file path or a "View in chat" hint.
2. **Per-car "show plan polyline" toggle** in the race UI for
   visual debugging without opening the explorer.
3. **Save/load training profiles** in Model Lab — preset combinations
   of rounds / trials / ticks / seed.

---

## Glossary

- **kinematic library**: the original `[curvature, targetSpeed]`
  primitive library generated from `kinematicForwardSim`, which
  assumes instant speed tracking and no understeer. Used by the
  "control" car (pink).
- **v2 library**: the new `[steer, driveForce, brakeForce]` primitive
  library generated from `learnedForwardSimV2`. Per-bucket durations
  + speed-aware controls. Used by the "learned" car (cyan) when
  "Use v2 library" is toggled.
- **legacy library**: `buildLearnedLibrary` (5-param model with
  online refit). Still wired in for backward compat and as a
  comparison baseline; not used by either race car when v2 is active.
- **Multi-goal A***: single A* over `(chassis state, next-gate-index)`
  joint state space. Lets the planner globally optimize through N
  gates rather than chaining N independent single-goal plans.
- **Open-loop divergence**: starting from a known state, roll the
  model forward through a recorded control trace and measure RMS
  position error from the recorded Rapier ground truth at horizons
  (0.5s / 1.0s / 1.6s). The primary accuracy metric.
- **Hull area**: convex-hull area of the forward-primitive endpoints
  in start-local frame. A measure of the action space's reachable
  region. <1 m² = degenerate.
- **Pred err RMS**: per-primitive prediction error — the chassis's
  actual position 0.55s after a primitive boundary vs where the
  planner predicted it would be. Displayed in the per-car HUD as
  "0.55s pred err (rms)".
