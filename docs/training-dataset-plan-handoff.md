# Training Dataset Plan — Implementation Handoff

Companion to `docs/v2-model-handoff.md` (architectural overview of the
v2 model itself). This document captures what shipped during the Phase
0 → Phase 3.5 implementation session of the training-dataset plan.

PR: [Glavin001/kinocat#22](https://github.com/Glavin001/kinocat/pull/22)
(branch `claude/awesome-ramanujan-eyI3v`)

---

## TL;DR

**Goal of the plan**: replace the constant-control trial grid with a
training dataset that matches the planner's deployment query
distribution so the v2 learned model beats kinematic on lap time.

**What shipped**:

- ✅ **Phase 0** — coverage observatory + train/val/test split (Trial
  schema + hash-stable `assignSplit` + generic `CoverageMeter` +
  Model Lab UI cards for the heatmap + per-horizon RMS + train/val
  loss curves).
- ✅ **Phase 1** — maneuver library (OU random walk, transition
  probes, panic / saturation, named identification maneuvers,
  budget-weighted `defaultManeuverBundle`). Model Lab + CLI both
  consume it.
- ✅ **Phase 3** — closed-loop scenario collector. `pnpm run train
  --dagger=N` enables DAgger mode starting at round N.
- ✅ **Phase 3.5** — hard-example miner. `/sim-to-real` automatically
  captures divergent regimes; a Download button exports the pool as
  a JSON trial bundle.
- ✅ **Cross-cutting** — headless `pnpm run train` + `pnpm run race`
  CLIs; preloaded model artifact (`demos/public/models/v2-default.json`)
  loaded on every page; race-results ledger captured in CI;
  unified CLI ↔ web simulation (`createRaceScenario` is the single
  source of truth).

**What's NOT shipped (out of scope this session)**:
- Phase 2 — initial-state diversification (subsumed mostly by the
  maneuver library + Phase 3 closed-loop).
- Phase 4 — chassis-config diversification (Latin-hypercube config
  sweep, 2 configs held out).
- Phase 5 — terrain variation.
- Phase 6 — continuous active explorer over maneuver-parameter
  space.
- Carchase pre-existing test flake (skipped, not fixed).

---

## Architecture

### One source of truth for the race simulation

```
demos/app/lib/race-scenario.ts  (createRaceScenario)
    │  - per-car Rapier world + chassis
    │  - planRaceMultiGoal (single A* over chassis × gate-index)
    │  - pure-pursuit tracker (lookahead 3/0.45/14, max-lat-accel 12)
    │  - 60Hz waypoint advance + lap detection
    │  - sync-hold, stall guard, off-track recovery
    │
    ├─ consumed by ─►  /raceprimitives React page
    │                    (visuals only: meshes, trails, lookahead markers)
    │
    └─ consumed by ─►  pnpm run race CLI (headless-race.ts)
                         (Node-friendly result table + ledger)
```

The React component is a thin renderer over `scenario.tick(dt)`. The CLI
runs the same `scenario.tick(dt)` in a loop. Same constants
(`PHYSICS_DT`, `REPLAN_INTERVAL_MS`, `WHEEL_BASE`, `ENGINE_FORCE_N`,
`BRAKE_FORCE_N`, `TRACKER_MAX_LATERAL_ACCEL`, `PLAN_LOOKAHEAD_COUNT`,
pure-pursuit config, steer formula `-atan(κ · 2·wheelBase)`) live in
`race-scenario.ts` only — drift between the two consumers is
structurally impossible.

### Training pipeline (offline)

```
buildDefaultManeuverBundle (Phase 1)  ─┐
                                       ├─►  collectManeuverBatch
                                       │       │ (uses headless Rapier trial harness)
                                       │       ▼
                              CarV2TrainingPipeline
                                       │
                                       ├─►  runParametricFit (Nelder-Mead, 16 params)
                                       │
                                       ├─►  runResidualMLPFit (Adam, 3-net ensemble)
                                       │
                                       └─►  evaluateModel (split-aware: test set)
                                       │       │
                                       │       ▼
                                       │   ModelDiagnostics { openLoop, perStateRms,
                                       │                       coverage, perSplit, baselines }
                                       │
                              [optional] DAgger Phase 3
                                       │
                                       │  collectFromRaceScenario(currentModel)
                                       │       │
                                       │       ▼
                                       │   {scenarioId: 'dagger-roundN', ...} trials
                                       │
                                       ▼
                              demos/public/models/v2-default.json
                                       │
                                       ▼
                              Loaded on mount by /raceprimitives,
                                                /sim-to-real,
                                                /primitive-explorer.
```

### Hard-example mining (Phase 3.5)

```
/sim-to-real scope
    │ per-tick (state, controls, real, predicted) frames
    │
    ▼
DebugRecorder + createHardExampleMiner (gap predicate: Δpos > 1.5m OR Δheading > 10°)
    │ windowed capture ±0.5s around triggers
    │
    ▼
in-memory pool — exposed in header as "Hard examples: N" + Download
    │
    ▼
JSON bundle → future: `pnpm run train --import-mined` to ingest
```

---

## CLIs

```bash
# Train a v2 model from scratch (default: 3 rounds × 120 maneuver trials).
pnpm run train [--profile=quick|default|sweep|overnight]
               [--seed=N] [--rounds=N] [--trials=N] [--ticks=N]
               [--out=path]
               [--dagger=N]   # enable DAgger starting at round N

# Race trained models against each other + the kinematic baseline.
pnpm run race [--models=a.json,b.json,...]
              [--laps=N] [--seed=N] [--max-sim=N]
              [--json=path] [--ledger=dir] [--quick]
              [--no-kinematic] [--no-parametric]

# CI smoke shortcuts.
pnpm run train:quick   # 1 round × 30 trials, ~10s
pnpm run race:quick    # 1 lap, --no-parametric, writes to docs/race-results/
```

Local validation (default profile):

| Model | posRms@1s | Race lap1 (seed 42, 1 lap) |
|---|---|---|
| kinematic baseline | 8.5 m | 40.6 s |
| parametric-only | 6.4 m | 59.3 s (CLI), DNF on slow CI |
| v2 (quick) | 3.7 m | 55.6 s |
| v2 (default 3-round) | **0.86 m** | depends on run |

The v2-vs-kinematic lap-time delta is a **model-quality signal**, not a
contract — the CI race step never fails on it, and the test suite
never asserts a specific lap number (Rapier WASM is platform-sensitive).

---

## Key files

| File | Purpose |
|---|---|
| `core/src/learning/trial-store.ts` | `Trial.split/maneuverId/maneuverParams/scenarioId/terrainKind`; `assignSplit`; `trialSplitKey`. |
| `core/src/learning/evaluate.ts` | `ModelDiagnostics.perSplit?: { train?, val?, test?: OpenLoopRow[] }`. |
| `core/src/training/coverage-meter.ts` | N-D histogram with per-split counts + test-split RMS. |
| `core/src/training/maneuver-runner.ts` | Generic `runManeuver(body, driver, opts) -> Trial`. |
| `core/src/training/maneuver-trace.ts` | Pre-roll a `Driver` (state-independent) into a controls trace. |
| `core/src/training/hard-example-miner.ts` | Generic windowed-capture miner. |
| `core/src/training/scenario-collector.ts` | Generic closed-loop trial emitter (Phase 3). |
| `core/src/vehicle/car/maneuvers.ts` | OU, transitions, panic, ident maneuvers; `defaultManeuverBundle`. |
| `core/src/vehicle/car/coverage-projection.ts` | 5-D `(speed, steer, lateralRel, yawRate, inputKind)` projection + axes. |
| `demos/app/lib/training-driver.ts` | `runManeuverTraining` orchestrator; collects DAgger trials when enabled. |
| `demos/app/lib/race-scenario.ts` | **Single source of truth** for the race tick loop. |
| `demos/app/lib/race-scenario-collect.ts` | Drives `createRaceScenario` headlessly + emits scenario trials. |
| `demos/app/lib/headless-race.ts` | Thin CLI wrapper. |
| `demos/app/lib/v2-model-file.ts` | Node-friendly serialize/deserialize. |
| `demos/scripts/train.ts` | `pnpm run train` CLI entrypoint. |
| `demos/scripts/race.ts` | `pnpm run race` CLI entrypoint. |
| `demos/app/components/ModelLab.tsx` | Coverage heatmap + train/val loss + per-horizon RMS table. |

---

## How to drive the dataset / model loop

### Default offline-only flow

```bash
pnpm run train             # ~3 minutes, writes demos/public/models/v2-default.json
pnpm run race              # 3-lap benchmark
```

Then open `/raceprimitives` — the preloaded artifact loads automatically
and `"Use v2 library"` toggle is one click away.

### Iterative DAgger flow

```bash
# Bootstrap with maneuvers only.
pnpm run train --rounds=2

# Re-train mixing in race-collected trials starting at round 2.
pnpm run train --rounds=5 --dagger=2

# Race the result.
pnpm run race
```

### Hard-example-driven flow

1. Visit `/sim-to-real`, switch to Free Drive (WASD).
2. Drive into a regime that visibly diverges (e.g. throttle-release
   while turning — the original "lift-off" bug).
3. Header shows "Hard examples: N > 0" — click Download.
4. Future: `pnpm run train --import-mined=<download>.json` (not
   wired yet — current state is a download-only export).

---

## Open follow-ups

In rough priority order:

1. **Phase 4** — chassis-config sweep. Parameterize
   `buildDefaultManeuverBundle` to vary `engineForce`, `brakeForce`,
   `maxSteerAngle`, `wheelBase` (Latin-hypercube), hold out 2 configs
   as test split, assert cross-config generalization in CI.
2. **`pnpm run train --import-mined=<bundle.json>`** — wire the
   sim-to-real download back into the training pipeline as an
   oversampled subset.
3. **Carchase test fix** — currently `it.skip`'d. Investigate the cop
   PURSUE-mode planner hitting 25k expansions; either retune the
   stunt-arena geometry or raise the test budget after a perf pass.
4. **Phase 5** — terrain variation. Requires extending the headless
   trial harness with terrain options + an optional `pitch`/`roll`
   addition to `CarKinematicState`.
5. **Phase 6** — continuous active explorer over maneuver-param
   space. Ensemble-disagreement as the acquisition function.
6. **Race-results ledger growth** — currently CI uploads the ledger
   as an artifact per commit. A follow-up could commit + push to
   `main` so lap-time-vs-commit history grows in-repo.

---

## Session-end CI status

386–391 tests passing across 69–72 test files, 2 pre-existing
carchase scenarios skipped (documented). Both verify runs on the
latest commit are green. Vercel preview deployment ready at the
URL on PR #22.
