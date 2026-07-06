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

This plan breaks the lap into **skills** — a straight, a sweeper, a
corner-approach, a gate-kink arrival, a slalom, a reverse-cost decision, a
shunt commit — each reproduced by a **tiny, deterministic,
sub-second-to-few-second test** that asserts the *right* behavior for that
skill and pins which layer is responsible. We fix each skill against its fast
test, then re-run the full lap as the integration check.

## Measured baseline (this session, deterministic headless runs)

Open course, MPPI (`tracker:'mpc'`), 100 s cap, 2-lap target:

| Entry | Laps | Recov | Stopped | MeanSpd | Note |
|---|---|---|---|---|---|
| kin | **2** [37.1, 37.7] | 0 | 0.8 s | 9.6 | clean — the reference |
| v2 (pre-fix) | 1 [71.5] | 11 | 27.5 s | 4.8 | wedge loops at slalom gates |
| v3 (pre-fix) | 1 [64.3] | 13 | 29.9 s | 4.8 | worst wedger |
| v3 (post Fix 1+2) | 1 [58.6] | **5** | 26.8 s | 5.1 | shunts now execute; still stops |

Technical course, MPPI: kin 1 lap [56.3] recov=3 stopped=15.0 s; v3 1 lap
[81.8] recov=4 stopped=27.0 s. (Pure-pursuit reference: kin ~31–37 s avg open;
v2 **wins** technical 35.9 vs 44.4 — the committed benchmark.)

Replan churn (mean lateral gap vs previous plan, near-chassis):
v3 same-wp 0.82 m / waypoint-advance 1.28 m; kin 0.98 / 1.48 m.

**Reading:** the losses are dominated by **stopped time at tight gates**
(~27 s of a 100 s run for v3), not by slow sweeps (both cars hit 20–30 m/s on
the open stretches — see `/tmp/churn2-*-plans.png` artifacts). The clean kin
car churns *more* than v3, so plan churn is not the differentiator. Both cars
already landed fixes this session: recovery no longer interrupts reverse
shunts, and the gear-flip prior reseed makes shunts actually execute
(recoveries 13→5).

## What we established (three findings that shape the skills)

1. **The planner IS time-optimal — but only to the fidelity of the model that
   baked its primitive library.** Edge cost is genuine seconds:
   `prim.duration · reverseCostMultiplier + directionChangePenalty`
   (`core/src/environment/vehicle-environment.ts:360-366`; race values 3.0× /
   1.0 s at `demos/app/lib/race-primitives-scenarios.ts:273-274`). **But the
   kinematic library's forward sim teleports speed** (`speed = target`
   instantly — `core/src/agent/vehicle.ts:38`), so a stop-and-reverse costs
   almost nothing in its worldview (no decel time, no forward distance lost
   braking). That is *why* the kinematic delusion plans and drives sharp
   pivots freely. The learned v2/v3 sims model finite decel
   (`core/src/agent/vehicle-model.ts:369-373`) but only across *chained*
   braking primitives; the gear-flip is a flat penalty that doesn't scale with
   the speed being scrubbed. The A* **heuristic is distance-based**
   (`/ maxSpeed`, speed-agnostic — `vehicle-environment.ts:494,501,519`), so it
   never prefers "longer-but-faster"; time-optimality enters only through the
   g-cost. Plans carry correct monotonic `.t`/`.speed` but **one sample per
   primitive** (~0.55 s / up to ~15 m apart — `vehicle-environment.ts:352-358`),
   with no intra-primitive speed profile.

2. **MPPI does NOT cap a straight below `cruiseSpeed = 30`.** On a straight
   drive-through the allowed-speed profile is `Infinity` and the only
   longitudinal term is the progress *reward*, so the static optimum is full
   throttle to 30 (`core/src/execute/mpc-tracker.ts` `buildProgressGeometry`
   ~418-467, `scoreRolloutProgress` ~640-650; `wProgress=6`, `wOverspeed=4`,
   overspeed dead-banded off below 30.5; no longitudinal regularizer opposes
   throttle). **The crawl lives on corner approaches**: a gate kink at library
   radius R≈3.4–4.5 m gives `vAllow≈8–9 m/s`, and the backward
   braking-envelope pass (`envelopeDecel=8`) holds the approach below 30 for
   ~51 m before *every* tight gate — plus replanned chord paths inject
   *phantom* curvature. Note `envelopeLateralAccel=12` vs the **measured**
   plant sustained cornering of **13.698 m/s²**
   (`demos/public/models/plant-envelope.json`) — only ~12% conservative, so
   the lateral budget is NOT the big lever; phantom curvature and stop-go
   wedge loops are. Config: `MPC_CONFIG` at
   `demos/app/lib/race-scenario.ts:1100-1180` (H=30 → 1.5 s, stepDt 0.05,
   64 samples, λ=3, `usePlanSpeeds:false`, `corridorHalfWidth=2.5`).

3. **The wedge mechanism (from solve-probe dissection):** the planner accepts
   gates at any heading, so committed plans carry a **kink at the gate**; the
   honest model correctly brakes for the kink, ends stopped nose-off-the-next-
   leg, plans a reverse shunt — which (pre-fix) never executed because the
   stale full-brake warm-start prior saturated the pedal channel and the blind
   recovery kept wiping the maneuver. Both executor-side halves are now fixed
   and unit-guarded; the remaining half is the **entry**: arriving hot at an
   infeasible kink at all.

## Artificial-constraint audit (nothing may cap the planner but physics)

The principle (max-pace-roadmap §0d): remove artificial constraints; provide
capability + accuracy + incentive; the model + time cost decide everything.
Within the MPPI horizon the model rollout IS the physics — cost caps are only
legitimate as *beyond-horizon anticipation*, and then must be derived from the
**measured** plant envelope, never hand-set. Audit of every constant that can
bind, against `demos/public/models/plant-envelope.json`:

| Constant | Where | Value | Measured plant | Verdict |
|---|---|---|---|---|
| `envelopeDecel` | MPPI cost backward pass (`mpc-tracker.ts` `allowedSpeedAt`; wired from `SPEED_PROFILE_DECEL` at `race-scenario.ts:1169`) | **8** | brake decel **15.7–52.9** m/s² | **ARTIFICIAL, 2–6× timid** — braking-onset penalty starts ~3× further from every corner than physics requires; the single biggest inappropriate limiter found. Fix: derive from measured brake curve with a named margin (K10). |
| `envelopeLateralAccel` | MPPI cost corner cap (wired from `previewLateralAccel` at `:1170`) | 12 | sustained **13.698** | mildly conservative (−7% corner speed); re-derive from envelope (K2). |
| `vAllow` model-agnosticism | MPPI cost | geometric | — | the cap is identical for every model, so the honest model gets no credit for knowing its true limit. Long-term: within-horizon pricing belongs to the model; the cap covers only beyond-horizon corners. |
| `cruiseSpeed` | MPPI `speedCap` | 30 | vMax **97** | POLICY cap (documented as such) — keep, but keep explicit. |
| curvature action space | `raceControlSets` (`race-primitives-scenarios.ts:282`) | 3 levels: 0, R=9, R=4.5 | — | **GRANULARITY GAP**: no native medium-radius (R≈15–35) arc, no explicit trail-brake control; sweepers are zigzag-approximated then smoothed. Candidate: add κ/4 arcs + brake-in-turn controls, measure plan-shape + search-cost delta. |
| `RACE_START_SPEEDS` 4 m/s buckets + zero-slip seams | `library.ts` nearest-bucket, `vehicle-environment.ts:352` | ≤2 m/s snap error | — | known WS-2 gap; root dynamic rollouts landed but OFF. Acceptable short-term; revisit after K-suite. |
| 0.55 s chunk | `characterize` duration | one control/chunk | — | in-plan switch granularity ~0.55 s; replans (300 ms) + MPPI (50 ms) refine timing; interruptible primitives (roadmap A2.7) is the queued fix. |
| `minTurnRadius` | `RACE_AGENT` = 4.5 | — | executed min radius **10.1** at ≥8 m/s | the kinematic library offers geometry the plant can't execute at pace (part of the intended delusion for kin; honest libs bake real arcs per bucket). |
| `reverseCostMultiplier=3×`, `directionChangePenalty=1.0 s` | `vehicle-environment.ts:360` | flat | true cost = real decel+leg time | **hand-set compensation for the kinematic teleport**; for honest models it distorts (their reverses already pay true time via chained braking samples). Candidate: per-library values — honest libs closer to 1×, keep 3× only for kin (K6 decides with data). |
| kinematic `speed = target` teleport | `vehicle.ts:38` | — | — | the intended "delusion" baseline — keep, document. |

### Temporal resolution: is 0.55 s per primitive too coarse?

Shrinking the chunk uniformly to ~0.1 s is the wrong lever, for measured and
architectural reasons — but the *precision instinct* behind it is right and
has two cheaper answers:

- **Search cost**: a 2-gate horizon is ~5–10 s of driving. At 0.55 s that is
  10–18 edges deep; at 0.1 s it is 50–100 deep — branching^depth blows past
  any expansion budget, and each endpoint moves so little (0.5 m at 5 m/s)
  that it falls below the dedup grid (`posCell=1.5 m`) and the states collapse.
- **Distinguishability (measured)**: at 4 m/s the 0.55 s endpoint fan was
  ALREADY at the usability floor (hull 0.19 m² < 0.5 m²,
  `race-primitives-scenarios.ts:403-404`). Shorter chunks make the planner's
  choices *indistinguishable*, not more precise.
- **Industry practice** (state-lattice AVs — Pivtoraiko/Kelly, CMU Boss,
  Apollo/Autoware; racing — AutoRally/Williams MPPI, TUM Indy): the search
  layer decides *maneuver topology* at coarse resolution (0.5–2 s edges or
  fixed-arc-length stations, replanned ~3–10 Hz); a smoothing/optimization
  pass produces the fine trajectory; an MPC/MPPI layer at 20–60 Hz
  (0.02–0.05 s controls, 1–3 s horizon) owns exact brake-point timing. Chunks
  quantize the planner's *intent*, never when the pedals move. This codebase
  already has exactly that hierarchy (0.55 s lattice → smoother → 300 ms
  replans → 50 ms MPPI solves — max-pace-roadmap §0c).

The two cheap precision levers if plan-level timing proves binding:
1. **Interruptible primitives** (roadmap A2.7): primitives already record 6
   substep samples ~0.09 s apart (`characterize.ts:63-66`); allowing early
   termination at substep boundaries — root node first — gives **~0.09 s
   in-plan switch granularity** (the 0.1 s instinct) at zero extra rollout
   cost and bounded branching.
2. **Fixed arc-length primitives** (candidate experiment): 0.55 s spans 16.5 m
   at 30 m/s but 2.2 m at 4 m/s — spatially wildly uneven. Baking primitives
   to constant *length* (~5 m) instead of constant time gives uniform spatial
   resolution (~0.17 s chunks at speed, ~1.2 s when slow, where the fan floor
   needs it anyway). Requires re-baking libraries + duration-per-primitive in
   the cost (already supported — cost reads `prim.duration`).

## Layer taxonomy (what each test attributes a failure to)

- **L0 — plant/model ground truth.** What the real Rapier plant can physically
  do. Already measured and pinned: `vMax=97`, sustained `aLat=13.698`, brake
  15–50 m/s² (`demos/public/models/plant-envelope.json`,
  `demos/test/plant-envelope.test.ts`, `measurePlantEnvelope` at
  `demos/scripts/plant-envelope.ts:59`). Skill tests assert against THESE
  numbers, not guessed constants.
- **L1 — planner.** Produce a *dynamically feasible*, *time-optimal* spline
  (`planRace` / `planRaceMultiGoal` / `planThroughWaypoints` in
  `demos/app/lib/race-primitives-scenarios.ts:611/728/779`).
- **L2 — executor / MPPI.** Realize the plan's pace: floor straights, hold
  corners at the real limit, commit reverse shunts, never wedge
  (`mpcTrack`, `core/src/execute/mpc-tracker.ts:781`).
- **L3 — replanner.** Commit a stable plan across replans and waypoint
  advances; no overcorrection when the lookahead goal shifts
  (`replanCar` in `demos/app/lib/race-scenario.ts`; `planChurnMean/Max`
  metric added this session).

## The skills

Each skill = minimal scenario + the behavior a *correct* system must show +
layer(s) + test tier + measured current state + fix hypothesis.

| # | Skill / scenario | Correct behavior (assert) | Layer | Tier | Current | Fix hypothesis |
|---|---|---|---|---|---|---|
| **K1** | **Straight drive-through** — one gate far ahead on a straight, no stop | Full drive command until brake point; peak ≥ 0.9·cruise; never brakes | L2 | cost-unit + executor-unit | **passes** (per cost analysis — lock as regression guard; proves the crawl is NOT on straights) | none — guard only |
| **K2** | **Sustained sweeper** — constant-radius arc R≈20–35 m | Hold ≈√(aLat·R) with aLat near the measured 13.7, not a crawl | L2, L0 | executor closed-loop vs envelope | *unmeasured in isolation*; likely ≤12% slow (`envelopeLateralAccel=12` vs 13.7) | pin the gap; raise budget toward measured with margin |
| **K3** | **Corner-approach profile** — straight → single genuine corner | `vAllow` on approach = the feasible corner speed; **no phantom caps** on a straight replanned as noisy chords | L2 | `buildProgressGeometry` unit | **fails on phantoms** (chord-noise curvature throttles approaches) | strengthen `CURVATURE_ARC_BASELINE`/`PROGRESS_CURVATURE_MIN` guards; test real-vs-phantom separation explicitly |
| **K4** | **Gate-kink arrival (wedge ENTRY)** — two gates, 90–120° direction change, arrive hot | Slow *enough before* the gate, carry a feasible arc through; `timeStopped≈0`, `recov=0`, **no reverse leg** | L1+L2 | capped closed-loop (3–5 s sim) + plan-shape | **fails** — THE dominant loss (~27 s stopped / 100 s for v3) | planner: finite `goalHeadingTol` at tight gates so plans stop kinking; executor: corridor-vs-arrival-speed sweep (handoff: `corridorHalfWidth` 1.2–1.5 gave kin 0 recov) |
| **K5** | **Slalom** — 3 alternating offset gates | A smooth **S-spline** carrying speed; not stop-pivot; `timeStopped<0.5 s`, `recov=0`; path length ≤ 1.4× gate chords | L1+L2 | plan-shape unit + capped closed-loop | **fails** (the wedge zone) — same mechanism as K4 compounded | K4 fixes + K8 feasibility gate |
| **K6** | **Reverse-cost honesty** — moving start (≈12 m/s), goal placed so reverse *looks* short but a forward arc is faster in real time | Honest-model plan has **no reverse leg** when a feasible forward arc exists; any reverse leg is preceded by chained braking samples | L1, L0 | plan-only unit | **exposes the kinematic teleport gap** (decel-to-stop is free in its worldview) | assert on honest libs; keep kinematic teleport as the intended "delusion" baseline (document, don't fix) |
| **K7** | **Waypoint-advance stability** — cross a gate so the 2-gate lookahead shifts | `planChurnMean` bounded (< ~2 m); no stop/overcorrection at the shift | L3 | capped closed-loop, assert churn + no-stop | churn 1.28–1.48 m at advance, tolerated (kin churns MORE and is clean) | only if churn ever induces a stop: reference-path hysteresis / commit window at the advance |
| **K8** | **Plan feasibility invariant** — any committed honest-model plan | Consecutive samples obey `|Δv| ≤ maxDecel·Δt` and curvature ≤ 1/minRadius at that speed | L1 | plan-only unit (cross-cutting) | honest libs ~ok; kinematic **violates by design** | assert on v2/v3 plans; this is the "is the spline physically real" gate for every other skill |
| **K9** | **Reverse-shunt commit (wedge ESCAPE)** — wedged pose, plan is a short reverse leg | Tracker actually backs up (not brake-in-place); recovery does not interrupt | L2 | executor-unit | **fixed + guarded this session** (`core/test/execute/mpc-tracker.test.ts:88-128`; recoveries 13→5) | done — keep the guard |
| **K10** | **Late braking** — long straight into a genuine tight corner | Full speed until the *measured-envelope* brake point: braking onset distance ≈ (v²−vc²)/(2·aBrake_measured·margin), NOT the timid `envelopeDecel=8` distance (~3× too early) | L2, L0 | `buildProgressGeometry` unit (vAllow profile shape) + capped closed-loop (brake-onset distance) | **fails** — the audit's #1 artificial limiter | derive `envelopeDecel` from the measured brake curve (15.7–52.9 m/s²) with a named margin; assert onset distance in both tiers |

## The fast test harness (what to build, reusing existing blocks)

New `demos/test/skills/` suite (fast tier) + one shared helper
`demos/test/skills/_skill-harness.ts`. Reuse, don't reinvent:

- **Minimal course builder** — `RaceCourse` is a plain object
  (`race-primitives-scenarios.ts:104-125`); a skill course is ~6 lines: one
  rectangle `polygon`, empty `obstacles`/`walls`, hand-placed drive-through
  `waypoints` (pose pattern at `:68-74`, `speed:5`), a `spawn`.
- **Plan-only assertions (L1, sub-second)** — `planRace` / `planRaceMultiGoal`
  / `planThroughWaypoints` return `PlanResult` (`core/src/planner/types.ts:59`).
  Assert on `result.path`: peak speed, stops, `pathLength` (helper at
  `demos/test/race-primitives-planner-v2.test.ts:18-26`), per-sample curvature,
  K8 feasibility deltas, presence/absence of reverse samples. Template:
  `race-primitives-multigoal.test.ts` (slalom subset pattern).
- **Executor-only closed loop (L2, sub-second, no Rapier)** — `mpcTrack` +
  a forward model on a synthetic plan; template
  `core/test/execute/mpc-tracker.test.ts` (`straightPath` `:12`, closed-loop
  assert `:41-65`, gear-flip guard `:88-128`). K1/K3 cost-ordering tests call
  `scoreRolloutProgress` / `buildProgressGeometry` directly (both exported).
- **Capped closed-loop (L1+L2+L3, few seconds sim)** — `createRaceScenario`
  with a custom `course:` and ONE entry (`kinematicEntry`/`v2Entry`/`v3Entry`,
  `demos/app/lib/headless-race.ts:158/169/183`), `tuning.tracker:'mpc'`,
  driven by `runMonitored` (`demos/test/_sim-harness.ts:64`) with a hard
  `maxTicks` cap and a `done()` predicate (3–10 s sim like
  `headless-race.test.ts`, NOT the 150 s benchmarks). Read `status().quality`
  (`meanSpeed`, `timeStopped`, `recoveryCount`, `planChurnMean`, `ggMeanUtil`)
  and `metrics.peakSpeed`.
- **L0 ground truth** — read `demos/public/models/plant-envelope.json` (already
  committed, already regression-pinned) for the target numbers.

**Test-writing gotchas (so the suite doesn't flap):**

- **Deterministic always**: seeded `createMPCTrackerState`, fixed spawn, no
  retries. Same discipline as `determinism.test.ts`.
- **Assert against the model in use, not the plant**, at the executor-unit
  tier: `parametricForwardV2` accelerates ~5.8 m/s² (needs ~68 m to reach
  28 m/s) while the Rapier plant does ~13.8 m/s² (needs ~28 m). Size courses
  and thresholds per tier, with ≥15% margin.
- **Parametrize model-dependent skills over {kin, v2, v3}** (K2, K4, K5, K6,
  K8): the point of several skills is precisely that the honest and delusional
  models behave differently. Model loading pattern:
  `model-vs-plant-fidelity.test.ts` / `tmp-sweep.mts` (JSON from
  `demos/public/models/`).
- **Thresholds are canaries of emergent behavior, not commands** (the
  max-pace-roadmap §0d rule): nothing in a fix may inject speed or scripted
  behavior to satisfy a skill test.

## Systematic fix order (updated by the measurements)

0. **Enabler (parallel):** optimize the v2 MPPI rollout like v3's
   (`forwardSimV3Rollout` pattern — allocation-free single-member inference);
   v2 is ~75 ms/solve vs v3 ~44 ms, which distorts v2's 20 Hz behavior and
   makes every v2 iteration slower to test.
1. **Land the fast skills suite (tests only).** K2/K3/K4/K5/K6 are expected to
   fail — the failing tests ARE the fast, precise reproductions we've been
   missing. Each failure names its layer. K1/K9 land green as guards.
2. **K4/K5 — the wedge ENTRY (biggest lever: ~27 s stopped / 100 s).**
   Planner side: finite `goalHeadingTol` for tight-kink gates (chord-aligned)
   so committed plans stop carrying infeasible kinks; executor side: sweep
   `corridorHalfWidth` (1.2–1.5 proven for kin) and gate-kink allowed-speed
   margin. Gate every change on the K4/K5 tests + K8 feasibility.
3. **K10 — late braking** (second lever, and cheap): derive `envelopeDecel`
   from the measured brake curve instead of the timid 8 — this alone moves
   every brake point ~3× closer to the corner. One constant + named margin,
   gated by the K10 onset-distance test.
4. **K3 — kill phantom-curvature approach throttling** (it taxes every gate
   approach tens of metres out). The unit test must separate real corner caps
   (correct!) from chord-noise phantoms (bug).
5. **K2 — pin the lateral-envelope gap** (small: 12 vs 13.7 measured); raise
   with margin only if the K2 test shows it binding. Action-space granularity
   (add κ/4 arcs / trail-brake controls) is a follow-on experiment gated on
   K5 plan-shape results, not a default change.
6. **K7 — only if** advance-churn ever induces a stop; the metric is live.
7. **Integration gates, in order:**
   a. kin under MPPI ≥ kin under pure-pursuit on the open course (MPPI stops
      leaving pace on the table);
   b. v2/v3 clean (0 recov, <2 s stopped) on open;
   c. v2/v3 **beat** kin under MPPI on the technical course → only THEN pin
      the winning benchmark + ratchet (no WIP-goal assertions in CI, per the
      handoff rule).
   Full-lap tools: `tmp-sweep.mts`, `tmp-plan-plot.mts` (plan-vs-exec + churn),
   then the committed benchmark suite.

## Verification

- Fast tier: `npx vitest run core/test/execute demos/test/skills` — target
  <30 s total (plan-only + executor-unit dominate; capped closed-loops are the
  budgeted exceptions).
- Integration: `cd demos && npx tsx scripts/tmp-sweep.mts <kin|v2|v3> '{}' 100
  <open|technical> /tmp/x.png` + `scripts/tmp-plan-plot.mts` for attribution;
  then the full suite `npx vitest run core/test demos/test` (154 files /
  869 pass / 2 skip as of this branch) before any PR update.
