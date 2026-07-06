# Racing skills: a decomposed test-and-fix plan

*Status: PLAN for review. Splits the racing problem into isolated "skills,"
each with a fast, precise, reproducible test, so every failure is attributable
to one layer (planner / executor / replanner) and fixable in isolation —
before spending 6–8 minutes on a full-lap run. Every file:line reference below
was verified against the branch `claude/mppi-racing-ws3-794hwp`.*

## Context — why this exists

We have been debugging on the **full lap**, which is slow (a 2-lap MPPI run is
~6–8 min wall-clock) and **conflates three layers**: the planner (does it make a
good spline?), the executor/MPPI (does it drive the spline at pace without
wedging?), and the replanner (is the plan stable across replans, especially
when the lookahead goal shifts forward at a gate?). A single aggregate lap time
can't tell us which layer failed.

This plan breaks the lap into **skills** — a hairpin, a slalom, a straight, a
sweeper, a reverse-cost decision — each reproduced by a **tiny, deterministic,
sub-second-to-few-second test** that asserts the *right* behavior for that skill
and pins which layer is responsible. We fix each skill against its fast test,
then re-run the full lap as the integration check.

## What we already measured (three findings that shape the skills)

1. **The planner IS time-optimal — but only to the fidelity of the model that
   baked its primitive library.** Edge cost is genuine seconds:
   `prim.duration · reverseCostMultiplier + directionChangePenalty`
   (`core/src/environment/vehicle-environment.ts:360-366`; race values 3.0× /
   1.0s at `demos/app/lib/race-primitives-scenarios.ts:273-274`). **But the
   kinematic library's forward sim teleports speed** (`speed = target`
   instantly — `core/src/agent/vehicle.ts:38`), so a stop-and-reverse costs
   almost nothing in its worldview (no decel time, no forward distance lost
   braking). That is *why* the kinematic delusion plans and drives sharp
   pivots freely. The learned v2/v3 sims model finite decel
   (`core/src/agent/vehicle-model.ts:369-373`) but only across *chained*
   braking primitives; the gear-flip is a flat penalty that doesn't scale with
   the speed being scrubbed. The A* **heuristic is distance-based**
   (`/ maxSpeed`, speed-agnostic — `vehicle-environment.ts:494,501,519`), so it
   never prefers "longer-but-faster." Plans carry correct monotonic `.t`/`.speed`
   but **one sample per primitive** (~0.55s / up to ~15m apart —
   `vehicle-environment.ts:352-358`), with no intra-primitive speed profile.

2. **MPPI does NOT cap a straight below `cruiseSpeed = 30`.** On a straight
   drive-through the allowed-speed profile is `Infinity` and the only
   longitudinal term is the progress *reward*, so the static optimum is full
   throttle to 30 (`core/src/execute/mpc-tracker.ts` `buildProgressGeometry`
   418-467, `scoreRolloutProgress` 640-650; `wProgress=6`, `wOverspeed=4`,
   overspeed dead-banded off below 30.5). **The ~5 m/s crawl on the real course
   comes from the curvature `vAllow` cap + the backward braking-envelope pass
   firing at real *and phantom* gate corners**: a gate at library radius
   R≈3.4–4.5m gives `vAllow≈8–9 m/s`, and the backward pass
   (`envelopeDecel=8`) holds the approach below 30 for ~51m before the gate.
   `envelopeLateralAccel=12` is conservative for a 21 m/s² friction-circle
   chassis, and replanned chord paths inject phantom curvature. Config:
   `MPC_CONFIG` at `demos/app/lib/race-scenario.ts:1100-1180` (H=30 → 1.5s,
   stepDt 0.05, 64 samples, λ=3, `usePlanSpeeds:false`).

3. **Plan churn is present in BOTH the clean kinematic car and the wedging v2/v3
   cars** (~1m mean, ~1.5× higher on waypoint-advance replans — measured via the
   new `DrivingQuality.planChurnMean/Max` and `tmp-plan-plot.mts`), so churn is
   **not** the differentiator. The wedge is the honest model correctly braking
   at an infeasible slalom kink while the kinematic delusion powers through — on
   the **open course there is no wall to punish that**.

## Layer taxonomy (what each test attributes a failure to)

- **L0 — plant/model ground truth.** What the real Rapier plant can physically
  do (max speed, sustained corner speed, brake decel). Measured by
  `measurePlantEnvelope` (`demos/scripts/plant-envelope.ts:59`) and pinned in
  `demos/test/plant-envelope.test.ts`.
- **L1 — planner.** Produce a *dynamically feasible*, *time-optimal* spline
  (`planRace` / `planRaceMultiGoal` / `planThroughWaypoints` in
  `demos/app/lib/race-primitives-scenarios.ts`).
- **L2 — executor / MPPI.** Realize the plan's pace: floor straights, hold
  corners at the real limit, commit reverse shunts, never wedge
  (`mpcTrack`, `core/src/execute/mpc-tracker.ts`).
- **L3 — replanner.** Commit a stable plan across replans and waypoint
  advances; no overcorrection when the lookahead goal shifts
  (`replanCar` in `demos/app/lib/race-scenario.ts`; `planChurnMean` metric).

## The skills

Each skill = a minimal scenario + the behavior a *correct* system must show +
the layer(s) it exercises + the cheapest test that reproduces it + our current
read on pass/fail + the fix hypothesis.

| # | Skill / scenario | Correct behavior | Layer | Test tier | Current | Fix hypothesis |
|---|---|---|---|---|---|---|
| **K1** | **Straight drive-through** — one gate 60m ahead on a straight, no stop | Floor it to ≥28 m/s the whole way | L2 | cost-unit + executor-unit | **passes** (guard) | none — this test *proves* the crawl is not on straights; lock it as a regression |
| **K2** | **Sustained sweeper** — constant-radius arc (R≈20–35m) | Hold ≈√(aLat·R) at the plant's real corner limit, not a crawl | L2, L0 | executor-closed-loop + L0 envelope | **fails** (crawls) | raise/uncap `envelopeLateralAccel` toward measured µg; kill phantom-curvature throttling |
| **K3** | **Corner-approach speed profile** — one gate turn in a plan | `vAllow` at the approach = the *feasible* corner speed, not phantom-throttled | L2 | `buildProgressGeometry` unit | **fails** (over-conservative) | tune `envelopeLateralAccel`, `corridorSlack`, `PROGRESS_CURVATURE_MIN`/`CURVATURE_ARC_BASELINE` guards |
| **K4** | **Single hairpin** — tight 180° | Feasible arc (no kink tighter than min-radius at plan speed); brake in, power out; **no wedge/reverse** | L1, L2 | plan-feasibility unit + capped closed-loop | partial | ensure plan feasibility (K8); executor pace (K2/K3) |
| **K5** | **Slalom / chicane** — 3 alternating gates | A smooth **S-spline** carrying speed; **not** stop-pivot-at-each-gate; `timeStopped≈0`, `recov=0` | L1, L2 | plan-shape unit + capped closed-loop | **fails** (wedge zone) | feasible planner spline + executor pace; measure V-shape vs S-shape |
| **K6** | **Reverse-cost honesty** — a pose where stop-and-reverse *looks* short but a wider forward arc is faster | Honest-model plan prefers the **forward arc**; a reverse leg is preceded by real braking-to-rest | L1, L0 | plan-only unit | **exposes gap** (kinematic teleport) | assert honest plan feasibility; document kinematic teleport as the intended "delusion" baseline |
| **K7** | **Waypoint-advance stability** — cross a gate so the 2-gate lookahead shifts one forward | `planChurnMean` bounded; **no** spurious stop/overcorrection at the shift | L3 | capped closed-loop, assert churn + no-stop | churn ~1.5× at advance (tolerated) | reference-path hysteresis / commit-window at the advance if it ever induces a stop |
| **K8** | **Plan feasibility invariant** (cross-cutting) — any committed plan | Consecutive-sample `|Δv| ≤ maxDecel·Δt` and curvature ≤ 1/minRadius at the sample's speed | L1 | plan-only unit | kinematic **violates**, honest models ~ok | assert on honest-model plans; this is the "is the spline physically real" gate |

## The fast test harness (what to build, reusing existing blocks)

A new `demos/test/skills/` suite (fast tier) plus one shared helper file
`demos/test/skills/_skill-harness.ts`. Reuse, don't reinvent:

- **Minimal course builder** — `RaceCourse` is a plain object
  (`race-primitives-scenarios.ts:104-125`); a skill course is ~6 lines: one
  rectangle `polygon`, empty `obstacles`/`walls`, hand-placed `waypoints`
  (drive-through pose helper pattern at `:68-74`, `speed:5`), a `spawn`. No need
  to go through `buildRaceCourse`.
- **Plan-only assertions (L1, sub-second)** — `planRace` / `planRaceMultiGoal` /
  `planThroughWaypoints` (`race-primitives-scenarios.ts:611/728/779`) return
  `PlanResult` (`core/src/planner/types.ts:59`). Assert on `result.path`:
  `peakSpeed = max(path.speed)`, `stops = last.speed≈0`, `pathLength` (helper in
  `demos/test/race-primitives-planner-v2.test.ts:18-26`), per-sample curvature,
  and the **K8 feasibility** deltas. Templates:
  `race-primitives-multigoal.test.ts` (slalom subset), `-planner-v2.test.ts`.
- **Executor-only closed loop (L2, sub-second, no Rapier)** — `mpcTrack`
  (`core/src/execute/mpc-tracker.ts:781`) + `parametricForwardV2` on a synthetic
  plan; exact template `core/test/execute/mpc-tracker.test.ts` (`straightPath`
  builder `:12`, closed-loop assert `:41-65`, the gear-flip test I just added
  `:88-128`). K1/K3 go here as **cost-ordering unit tests**: call
  `scoreRolloutProgress` / `buildProgressGeometry` directly and assert
  full-throttle < coast on a straight, and correct `vAllow` on a corner.
- **Capped closed-loop (L1+L2+L3 together, few seconds)** — `createRaceScenario`
  with a `course:` override and one entry (`kinematicEntry`/`v2Entry`/`v3Entry`
  from `demos/app/lib/headless-race.ts`), `tuning.tracker:'mpc'`, driven by
  `runMonitored` (`demos/test/_sim-harness.ts:64`) with a hard `maxTicks` cap
  (3–5 s sim, like `headless-race.test.ts`, **not** the 150s benchmarks). Read
  `status().quality` (`meanSpeed`, `timeStopped`, `recoveryCount`,
  `planChurnMean`, `ggMeanUtil`) and `metrics.peakSpeed`.
- **L0 ground truth** — `measurePlantEnvelope` blocks (`plant-envelope.ts`
  max-speed `:70`, cornering `:124`, braking `:96`) give the *real* numbers K2/K3
  assert against, so a "hold the corner at the limit" test compares to physics,
  not a guessed constant.

## Systematic fix order

1. **Land the fast skills suite first (all skills, tests only).** Let K2/K3/K5/K6
   fail — the failing tests *are* the precise, fast reproductions we've been
   missing. Each failing assertion names its layer.
2. **L2 pace (K1→K3→K2).** Fix the corner-speed cap so MPPI drives at the plant's
   real limit (this is the single biggest lever — pace is what makes fidelity
   matter; at 5 m/s nothing else does). Cost-ordering + `buildProgressGeometry`
   unit tests gate it; the sweeper closed-loop confirms.
3. **L1 feasibility & reverse-cost (K8→K4→K5→K6).** Assert honest-model plans are
   dynamically feasible and reverse legs are properly braked; confirm the slalom
   plan is an S-spline, not V-pivots. (Decide explicitly whether the kinematic
   teleport stays as the intended "delusion" baseline — recommended — or gets a
   feasibility guard.)
4. **L3 stability (K7).** Only if the advance-replan churn ever induces a real
   stop; the metric already exists to catch it.
5. **Integration: re-run the full lap** (`tmp-sweep.mts`, then the benchmark
   suite) and compare `meanSpeed`, `timeStopped`, `recoveryCount`, lap time
   against the pre-fix baseline. **Only after v2/v3 actually beat kin under MPPI**
   do we pin a winning benchmark (per the handoff's own rule — no WIP-goal
   assertions in CI).

## Verification

- Fast tier: `npx vitest run core/test/execute demos/test/skills` — target
  sub-30s for the whole skills suite (plan-only + executor-unit dominate).
- L0/closed-loop tier: the few capped-closed-loop skills + `plant-envelope`.
- Integration: `cd demos && npx tsx scripts/tmp-sweep.mts <kin|v2|v3> '{}' 100
  <open|technical> /tmp/x.png` and the diagnostic `scripts/tmp-plan-plot.mts`
  for plan-vs-exec + churn; then `npx vitest run core/test demos/test` (full
  suite, currently 154 files / 869 pass / 2 skip) before any PR update.
