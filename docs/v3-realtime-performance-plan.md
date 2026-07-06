# v3 + MPPI + Control-Feedforward — Real-Time Performance Plan

Goal: make the **v3 learned model, MPPI tracker, control-feedforward ON** stack
drive the technical course **in real time in the browser**, not just under the
pause-clock. Today it is correct but not real-time: the planner cannot keep a
fresh plan on the car's clock, so the car wedges.

This document (1) states the problem with the measured evidence, (2) maps where
the wall-clock actually goes, grounded in the search code, and (3) ranks the
opportunities to close the gap, with a phased, each-step-measurable sequence.

---

## 1. The problem, in one screenshot

Live `/raceprimitives?course=technical&tracker=mpc&ff=1`, v3 selected:

```
PLANNER   last replan  79ms · ok
          plan age     1590 ms          ← red: the plan is 1.6 s stale
          success rate 35% (29/84)      ← 65% of replans FAIL
LIVE      throttle 0%  brake 63%  target spd 0.9 m/s   ← wedged, braking
MPPI      solve 50.7 ms · 316
time 28 s   laps 0   waypoints 1        ← 1 gate in 28 s
```

The planner succeeds on barely a third of replans. Every failure leaves the
executor tracking an ever-older plan; by the time the plan is ~1.6 s stale the
chassis has drifted off it, wedges against a wall, and now the planner must
solve a *harder* (tight-clearance, from-rest) problem — which fails more often.
It is a **compounding failure loop**, and its root is planning throughput.

---

## 2. Measured baseline (the evidence)

| Where | Budget / clock | Result |
|---|---|---|
| Browser, real-time | **120 ms**/replan, 300 ms cadence, RAF loop | v3 planner **35% success**, plan age **1.6 s**, wedged, 0 laps |
| Headless, pause-clock | **12,000 ms**/replan | v3+MPPI+FF completes a **controlled 49.5 s lap** on technical (mean 7.2, predErr 0.28 m, g-g-mean 49% — the tightest, most-within-envelope line of any model) |

The v3 stack is proven **correct** only when the planner is given ~**100×** the
real-time budget. Closing that ~100× effective-throughput gap is the whole task.

**Fixed costs that frame the budget** (all confirmed in code):

- Real-time planner budget `RACE_REPLAN_BUDGET_MS = 120` ms; the browser does
  not override it (`race-scenario.ts:1542`, `race-primitives-scenarios.ts:586`).
- Replan cadence `REPLAN_INTERVAL_MS = 300` ms; commit window **off** by default.
- `maxExpansions` = `RACE_MAX_EXPANSIONS*2 = 60,000` for the joint multi-goal search.
- MPPI solve **50.7 ms/tick**, run every 3 physics ticks (~50 ms sim), so it
  **alone nearly saturates the real-time frame** and competes with planning for
  the same wall-clock. Cost = 64 samples × 30 horizon × 3 substeps × 1 ensemble
  member of a 6-64-64-6 MLP ≈ **5,760 MLP forwards/solve**.
- Planner library (browser, non-generated v3): **8 speed buckets, 95 primitives,
  branching factor 11–14** per expansion (`race-primitives-scenarios.ts:397-476`).

Note the planner library is **pre-baked** — the v3 MLP is *not* evaluated during
search — so planning cost is pure A* (heuristic + collision + bookkeeping), and
MPPI is where the v3 MLP cost lands. They are two separate budgets that share one
real-time frame.

---

## 3. Where the wall-clock goes (bottleneck anatomy)

Per replan the demo runs a **single joint IGHA\*** search over
`chassis-pose × gate-index` for a **2-gate** lookahead
(`plan-vehicle-multi.ts`, `multi-goal.ts`). Multi-resolution anytime, 3 levels,
binary min-heap open list, `Map<string,number>` dedup. Per expansion it generates
11–14 successors. The dominant per-successor costs, in order:

### 3.1 Reeds-Shepp heuristic — paid, effectively, **twice** per successor
- Base heuristic = Reeds-Shepp shortest-path length / maxSpeed
  (`vehicle-environment.ts:507-565`), the in-code "dominant per-successor cost".
- Each RS miss runs **all ~48 candidate words** (CSC/CCC/CCCC/CCSC/CCSCC) with
  heavy `sin/cos/atan2/asin/acos/sqrt` and **no early-out** once a short CSC
  solution exists (`reeds-shepp.ts:330-338`).
- **The multi-goal wrapper recomputes it.** `MultiGoalEnvironment.succ`/
  `createNode` call `base.createNode` again and recompute `this.heuristic` for a
  successor that **already computed `h`** inside `vehicle-environment.succ:402`
  (`multi-goal.ts:86,94,120,123,139`). So the single most expensive operation in
  the search runs ~2× per successor.
- **Heuristic table thrashes in exactly the multi-goal case.** `hCache` is keyed
  only on the *source* pose and is `clear()`ed whenever the *goal* pose changes
  (`vehicle-environment.ts:512-517`). The joint search interleaves nodes of
  different `gateIndex`, so the goal flips constantly and the RS cache is wiped
  repeatedly — the O(1) cache it's meant to be degrades toward all-misses.

### 3.2 Footprint collision sweeps — no broadphase early-accept in multi-goal
- Each primitive runs `sweepClear` = **6 exact footprint-vs-navmesh tests**
  (`vehicle-environment.ts:298-344`), first-hit short-circuit.
- `clearanceBroadphase` (the cheap "far from any obstacle → accept without the
  exact test" early-out) is **single-goal-only** and **not set** in the
  multi-goal defaults, so on the technical course every one of those 6 samples
  pays the full polygon test even in wide-open track sections.
- The analytic shot (below) collision-sweeps an **entire curve to the gate**.

### 3.3 Analytic (Reeds-Shepp) shot — uncached, whole-curve, ~1-in-6 expansions
- `tryAnalyticShot` fires on expansion 1 then every `analyticEveryN = 6`
  (`vehicle-environment.ts:407-413`). Each fire = a **fresh uncached RS solve**
  to the gate + `sampleCurveWithGear` at 0.6 m + a **full collision sweep of the
  whole curve** (`:419-505`) — far heavier than an ordinary successor.
- `analyticDriveThrough` (the correctness-branch reprice) only changes the shot's
  *price*, but by removing the mispriced free stop it **balloons total expansions
  ~6×** — which is exactly why the correctness config needs a 12 s budget.

### 3.4 String-keyed everything + per-expansion allocation (GC pressure)
- Dedup (`gExact`), the coarse `DominanceTable`, and `hCache` are all
  `Map<string,number>`. Every `createNode` builds **3 `pack3` index strings + 1
  template-literal hash string** (`vehicle-environment.ts:281-296`) — allocated
  and hashed per node, and the multi-goal wrapper does it **twice**.
- Each successor also allocates a state object, an `EdgeRef` + `data` object, and
  a `Node`. **No node/edge pooling.** Each resolution pass rebuilds the heap +
  maps from scratch (3 passes).

### 3.5 Every replan is a **full fresh search** — nothing is reused
- There is no warm-start, no plan reuse, no incremental repair. Every 300 ms the
  planner throws away the previous solution and re-searches the 2-gate window
  from the live chassis state. On the technical course from a tight pose that
  full search does not fit in 120 ms → the 35% success rate.

### 3.6 The MPPI solve shares the same 16 ms frame
- 50.7 ms/solve at ~20 Hz means MPPI *by itself* exceeds a real-time frame; add a
  70–120 ms planner replan on the same thread and the loop cannot stay real-time.
- Feedforward currently only *consumes* the plan; it does not yet *reduce* the
  MPPI sample count, even though its warm-start prior should let it.

---

## 4. Opportunities, ranked by leverage

Ordered by expected impact-per-effort. Impact figures are **hypotheses to
verify**, not measurements, except where they cite the map above.

### Tier 1 — Structural: stop re-searching from scratch every 300 ms
This is where the ~100× lives. Two complementary moves:

1. **Plan reuse + longer commit window + bigger per-replan budget (feedforward
   synergy).** Feedforward makes the executor ride the plan's *own proven
   controls*, so it tolerates an older plan far better than a geometry-tracker
   does. Exploit that: turn on the existing `commitWindowMs`, drop the replan
   cadence (300 ms → ~800–1200 ms), and hand each replan a correspondingly
   larger budget (120 ms → 300–600 ms). Same wall-clock spent on planning, but
   spent as *fewer, deeper* searches that actually finish — success rate should
   jump and plan-age should stop climbing. **Cheapest high-leverage change; do it
   first.** (Machinery already exists: `commitWindowMs`, `pendingPlan`,
   `tuning.plannerBudgetMs`, `tuning.replanIntervalMs`.)

2. **Incremental repair instead of full re-plan.** When the new start pose is
   near the committed plan (the common case), repair the divergent prefix and
   keep the validated tail rather than re-searching both gates. Options, cheapest
   first: (a) seed the open list / consistency reference with the previous plan
   (`referencePath` already biases toward it — make it a *warm start*, not just a
   soft cost); (b) true LPA\*/D\*-Lite-style incremental A\* that reuses `g`-values
   across replans. (a) is a few days; (b) is the real fix but larger.

3. **Anytime weighted-A\* for a guaranteed first plan.** `plan()` already supports
   a heuristic `weight` (ε-suboptimal). Run the first pass at weight ≈ 1.5–3 to
   get a *usable* plan in far fewer expansions, then tighten with remaining
   budget. Converts "no plan in 120 ms" into "a good-enough plan in 30 ms, refined
   if time remains." Nearly free to try.

### Tier 2 — Search efficiency: make each expansion cheaper (the map's targets)
4. **Kill the double heuristic / double `createNode` in the multi-goal wrapper**
   (§3.1). Reuse the inner successor's already-computed `h` and node instead of
   recomputing (`multi-goal.ts:86,94,139`). This directly halves the dominant
   per-successor cost. **Highest impact-per-line-changed in the search core.**
5. **Fix the heuristic-table thrash** (§3.1): key `hCache` on `(source, goal)` (or
   keep one cache per gate) so goal flips in the joint search stop wiping it.
6. **RS early-out** (§3.1): return as soon as a CSC/short word beats the current
   best by more than the remaining families can improve; most heuristic calls
   don't need all 48 words.
7. **Enable `clearanceBroadphase` in the multi-goal env** (§3.2) so open-track
   samples skip the exact footprint test — a large fraction of the technical lap
   is not near a wall.
8. **Packed-integer keys + node/edge pooling** (§3.4): replace the `pack3`/hash
   *strings* with numeric keys (the state already quantizes to a small grid) and
   pool node/edge objects across passes. Removes the pervasive third cost and the
   GC pauses that hurt the p99 replan tail.
9. **Cheaper / rarer analytic shot** (§3.3): cache its RS solve (share `hCache`),
   only fire it when the gate is within a distance where a shot is plausible, and
   reuse the heuristic's RS result instead of recomputing.

### Tier 3 — Trim the MPPI solve that shares the frame
10. **Fewer samples via the feedforward prior.** With a good feedforward
    baseline the sampler no longer needs 64 samples to *discover* the maneuver —
    it needs a handful to *correct* it. Try 64 → 24–32 and measure predErr/lap;
    feedforward should hold quality while ~halving the 50.7 ms.
11. **Shorter horizon / fewer substeps** where the plan is already feedforward-
    accurate (30 → 20 steps; 3 → 2 substeps), measured against predErr.
12. **Batch the MLP** (one matmul over all samples' step-`i` inputs instead of
    5,760 scalar forwards) and/or move it to `Float32Array`/WASM. This is the
    single biggest lever on the MPPI half.

### Tier 4 — Offload & precompute
13. **Run the planner in a Web Worker** so a 300–600 ms search never blocks the
    render/physics frame (pairs naturally with the longer commit window of #1).
14. **Model-aware heuristic:** precompute a v3-specific cost-to-go table (the
    current RS heuristic is the *agent's kinematic* lower bound and mismatches
    v3's true reachability, over-expanding). A tighter, model-matched heuristic
    is the principled long-term fix to expansion count.

---

## 5. Recommended sequence (each step measurable, guard-railed)

Every step stays behind the existing flags and must not move the default
(flag-off) pinned benchmarks.

- **Phase 0 — Instrument.** Add a real-time (non-pause-clock) success-rate probe
  and expose per-replan `expansions / heuristicCalls / collisionChecks` (the
  counters already exist, `perf.ts`) so every later change is a measured delta.
  Extend `perf-profile.mts` to report planner success-rate at a *fixed real-time*
  budget, not just percentiles at a generous one.
- **Phase 1 — Tier 1 (cadence + commit + budget + anytime weight).** Cheapest,
  and it directly attacks the 35%. Target: >90% success at a real-time-affordable
  effective budget, plan age bounded. Feedforward is the enabler.
- **Phase 2 — Tier 2 items 4, 5, 7 (double-heuristic, cache thrash,
  broadphase).** Pure speedups, no behavior change; re-pin laps.
- **Phase 3 — Tier 2 items 8, 9 (pooling/packed keys, analytic shot).** Attacks
  the p99 tail and GC.
- **Phase 4 — Tier 3 (MPPI trim) + Tier 4 (worker / model-aware heuristic).** The
  larger, higher-ceiling work once the search is lean.

Stop and re-measure between phases; the target is a *reliable* real-time plan
(success ≳ 95%, plan age < ~1 replan interval), not a single fast search.

---

## 6. How to measure (reusable)

- **HUD fields already present** (screenshot): `success rate`, `plan age`,
  `last replan ms`, `MPPI solve ms` — the live regression surface.
- **`demos/scripts/perf-profile.mts`** — replan + MPPI-solve percentiles per
  model. Extend with a fixed real-time budget + success-rate column (Phase 0).
- **`demos/scripts/best-config-bench.mts`** — one-lap driving-quality per config;
  add wall-strike count and run it at real-time budgets to see the wedge.
- **`core/src/planner/perf.ts` counters** — `expansions`, `heuristicCalls`,
  `collisionChecks`, `rejectedByExact/Dominance/Omega`, formatted per-expansion.
  These attribute a change to the exact term it moved.
- **`scripts/lib/run-log.ts`** — stream long runs to a unique tailable log.

---

## 7. Risks / guardrails

- **Longer commit window + staler plan** could reintroduce divergence — but this
  is precisely what feedforward buys down; gate it on feedforward being ON and
  watch predErr.
- **Weighted-A\* / repaired plans are ε-suboptimal** — watch lap time and churn,
  not just success rate; a fast ugly plan that the executor can't hold is not a win.
- **Every change stays flag-gated and default-off**; re-pin the correctness-branch
  laps (open 41.3 s, technical 49.5 s for v3+FF) after each phase so a speedup
  never silently costs correctness.
- Do not chase MPPI micro-opts before Phase 1 — planning is the dominant real-time
  failure; MPPI is a secondary (though real) frame-budget competitor.
