# kinocat production-readiness review

*Review of open PRs, bugs on `main` (6ac1268), test strategy, and remaining gaps.*
*Date: 2026-07-05. All numbers below were measured on this commit unless noted.*

---

## 0. Executive summary

The paradox that motivated this review — **all 620 tests pass, the controller
bench reports 5/6, and yet the /parking page visibly drives badly** — is not a
coverage gap in the usual sense. The web page, the CLI bench, and the Vitest
suites genuinely share one runner (`createRaceScenario` +
`parkingScenarioOptions` + `evaluateParked`); a headless re-run of the web
page's exact configuration reproduces the live site's numbers **to the tick**
(124.20 s, 171 replans, 0.191 m final error, 89.9 % in stall). The problem is
that the *observers* disagree:

- the **bench** breaks out of its loop at the *first transient instant* the
  loose `parked` predicate holds (coverage ≥ 0.85, heading ≤ 8°, **|v| ≤ 0.3
  m/s — a moving car counts as parked**) → "PASS 22.3 s";
- the **web page** keeps ticking the same sim and shows the closed-loop truth:
  the runner destroys its own success ~300 ms later (unconditional 800 ms
  replan cadence) and grinds for **another ~102 seconds** hunting a
  termination condition it can barely satisfy;
- the **goal automaton** latched "DONE ✓" at t = 22.08 s *while the car was
  reversing at −0.84 m/s*, because the "stop" condition `speed: { max: 0 }`
  is evaluated on **signed** speed — any reverse motion satisfies it.

A small set of cross-layer contract bugs — plus two genuine planner-bridge
bugs — cause nearly everything the user sees. The IGHA\* search core itself
is sound (heap, dominance, dedup, anytime loop all verified), but the
*scenario bridge around it* and the *seams between layers* are not:

| # | Bug | Effect |
|---|-----|--------|
| 1 | "stop" encoded as `speed ≤ 0` (signed) — scenario layer *and* planner goal test | planner emits terminal stubs ending at −0.8…−1.5 m/s; tracker's stop latch (`|v| < 0.05`) can never fire on them; HUD latches DONE mid-reverse |
| 2 | tolerance inversion: planner may stop anywhere in a **0.35 m** goal disk, but the runner's finish requires crossing a **0.25 m** disk | the shuffle attractor sits 2–3 cm *outside* the finish disk → termination is literally luck (took 124 s) |
| 3 | pure-pursuit speed floor `max(vGoal, lookaheadMin)` treats a **3 m distance as a 3 m/s speed** | the brake-to-goal ramp is dead for parking (floor > 2 m/s cruise); terminal approach is bang-bang → 0.1–0.5 m stopping scatter |
| 4 | parking runs with **race lookahead** (`lookaheadMin = 3 m` on 0.5–3 m parking segments) | the lookahead point is *always the path endpoint* — pure pursuit degenerates to "aim at the goal point" and structurally cannot remove lateral error (probe: x-offset frozen at 0.29 m across ~180 shunts) |
| 5 | curvature infeasibility chain: planner plans **R = 3.5 m** arcs, tracker clamps at 4.5 m, the physical plant can only drive **≈ 4.68 m** | every tight parking arc is 25–34 % beyond what the chassis can execute → systematic wide tracking → the 0.29 m residual that the finish disk (bug 2) then can't absorb |

Two additional planner-bridge bugs compound this (§3.5): in scenario-bridge
mode the **gear-flip penalty is charged on every edge** (a `null !==
undefined` guard bug), making direction changes effectively *free* — the
planner happily emits multi-cusp shuffle plans; and the **time bucket sits in
the dedup hash even in static scenes**, inflating search 3.8× (measured:
83,968 vs 22,220 expansions for the identical query) so that any replan whose
Reeds-Shepp shot is blocked blows the 500 ms budget and returns junk.

The strategic conclusion: **the search algorithm is sound; the product breaks
in the seams** — success semantics, tolerances, termination, and the scenario
bridge each redefine the contract slightly differently. The highest-leverage work is (a) one shared,
*settle-latched* success oracle used by planner, tracker, runner, bench, tests
and HUD alike, and (b) closed-loop quality budgets (time-to-settled, replan
count, final pose error) asserted in CI.

---

## 1. Measured baseline

### 1.1 Test suite (main @ 6ac1268)

`pnpm test`: **620 passed, 2 skipped, 0 failed** (104 files, 208 s).
Parallel parking is a *documented known failure* kept honest via `it.fails`
in `demos/test/parking-invariants.test.ts`.

### 1.2 Controller bench (`pnpm run controller-bench`, pure-pursuit, kinematic)

| scenario | pass | sim(s) | goalErr(m) | hdgErr | \|v\| at "success" | note |
|---|---|---|---|---|---|---|
| race | PASS | 36.73 | 0.00 | 0.00 | 9.81 | 36.7 s lap |
| parking-forward-pullin | PASS | 7.77 | 0.14 | 0.00 | 0.26 | 97 % in stall |
| parking-reverse-perp | PASS | 22.33 | **0.27** | 0.05 | **0.15** | 86 % in stall |
| parking-parallel | **FAIL** | 8.58 | 0.15 | **0.28 (16°)** | 0.00 | pure-pursuit has no heading control |
| obstaclecourse | PASS | 8.00 | 0.00 | 0.00 | 9.42 | |
| ramp | PASS | 9.40 | 2.96 | 0.00 | 3.87 | reached (jumped) |

Note the reverse-perp row: the bench declares success while the car is still
moving (0.15 m/s) and 0.27 m off-center — that terminal error is *outside*
the runner's own 0.25 m finish radius, which is exactly why the web page ran
another 100 s after this instant.

Also: the usage example in `controller-bench.ts`'s own header
(`--filter=parking`) matches zero scenarios — the filter is exact-match on
names like `parking-reverse-perp`.

### 1.3 Headless reproduction of the web page

Running the web page's exact options headlessly (same tick, same tuning)
reproduced the live-site numbers deterministically: FINISHED at t = 124.20 s,
171/171 successful replans, final posError 0.191 m, coverage 89.9 %,
goal-automaton DONE latched at t = 22.08 s while reversing at −0.84 m/s;
transiently `parked` at t = 22.33 s. **~10 s of legitimate driving, ~12 s of
cusp maneuvering, ~102 s of pure termination-layer limit cycle.**

---

## 2. Open PR reviews

> **Status update (2026-07-05, later the same day):** PR **#37** was merged
> into main (`4edc567`) after a review-fixes commit. PR **#36** was reworked
> on-branch exactly along the lines below (`7e43b9a`: center-aware
> `evaluateParked` + true-goal gating of the heading term, main merged in)
> and re-verified independently: full suite 685 passed / 0 failed on the
> branch head, controller bench **6/6** including parking-parallel
> (17.4 s, 86 % in stall, 5° heading error — previously FAIL at 16°).
> The #36 verdict below is therefore superseded: **ready to merge**.
> Known follow-up from the #37 integration: the bench's new
> "plan N % feasible" annotation still reports 0 % for multi-cusp/racing
> plans (the cusp phantom-curvature issue, §2.2 fix 1) — advisory only,
> does not gate pass/fail.

Verdict summary as of the original review (details per PR below):

| PR | Title (short) | Importance | Verdict |
|---|---|---|---|
| #36 | Stanley heading term for pure-pursuit | **HIGH** — targets the only failing bench scenario | REWORK (proven regression vs current main) |
| #37 | Component-level eval harness (`core/src/eval`) | MEDIUM-HIGH (diagnosis infrastructure) | MERGE-WITH-FIXES — land **first**, use it to re-tune #36 |
| #34 | Rich `Plan` structure (curvature/feedforward/segments) | MEDIUM now, foundation for HIGH fix | MERGE-WITH-FIXES |
| #33 | Learned-vs-parametric-vs-Rapier comparison + coverage OOD gate | MEDIUM (research track) | MERGE |
| #30 | JAX parametric trainer + smooth model variant | MEDIUM (dev velocity) | REBASE-THEN-MERGE (decouple from #24) |
| #25 | "chore: gitignore" (actually trial-cache + analysis CLIs) | LOW | MERGE-WITH-FIXES |
| #24 | Dense sweep-sampled plan path + tuning experiments | MEDIUM (superseded in parts) | REWORK — salvage as 4–5 focused PRs |
| #41 | Integration-plan doc | LOW for driving quality | MERGE-AFTER-UPDATE (or close if charter stale) |

### 2.1 PR #36 — Stanley-style heading term (branch `claude/parallel-terminal-heading`)

**Problem**: pure-pursuit chases a lookahead *point* and structurally cannot
null terminal heading error; parallel parking ends ~16° misaligned. This is
the **only currently failing bench scenario** and the repo tracks it as
`it.fails`. The problem is real and important.

**Status vs main**: the PR predates two merged changes to the same file
(PR #40's nearest-index stop latch; PR #38's follow controller), so it must
be reconciled — the review below includes whether the control law itself is
correct.

**Review result — the control law is correct, the merge is not. VERDICT:
REWORK.**

- The math checks out against the codebase's actual conventions: sign
  (`kappa += headingGain · wrapAngle(tangent − heading)` drives heading →
  tangent given `heading += speed·κ·dt`), forward-only gating (correct and
  necessary — the term would destabilize in reverse), min-turn-radius clamp
  re-applied after the addition, and stability at gain 4.0 / 2 m/s / 60 Hz is
  comfortable (k·v·dt ≈ 0.16 ≪ 2; observed zero steer reversals).
- One semantic subtlety: the executor feeds pure-pursuit *per-segment* paths,
  so the `headingRadius` gate engages within 2 m of **every forward cusp**,
  not just the final goal, contrary to the PR's rationale.
- **The blocker is empirical**: `git merge-tree` is textually clean, but the
  actually-merged tree deterministically fails
  `parking-invariants > parallel: reaches the slot cleanly` with
  `terminalPosError = 1.42 m` (limit 0.6). Bisected to the interaction with
  PR #40's stop latch: with the heading term the car ends square in the
  stall (4° error) but **1.42 m off-center along it**, and `evaluateParked`
  has *no centering requirement* so the shared predicate calls it parked.
  Net effect of merging as-is: heading 16° → 4°, centering 0.15 m → 1.42 m —
  it trades one production bug for a worse one.
- Rework: rebase, gate the term on the *final* segment / true goal distance,
  re-tune against the stop latch until both parallel invariants pass in the
  same run, fix the stale comment in `parking-scenarios.ts`, and add a
  centering term to `evaluateParked` so "parked" cannot drift 1.4 m along
  the stall.

### 2.2 PR #37 — Component-level evaluation harness

**Problem**: today a driving failure cannot be attributed to planner vs
controller. The eval module (reference trajectories, projection-based
cross-track error, feasibility vs friction circle, comfort bounds, gauntlet
corridor with a passability oracle, 2×2 diagnosis matrix) is exactly the
missing measurement layer.

**Review result — right framework, not yet pointed at the failing behaviors.
VERDICT: MERGE-WITH-FIXES (and land it before re-working #36).**

- Merges clean; full typecheck + all 50 eval tests pass on the merged tree;
  exposed only via the `kinocat/eval` subpath → zero runtime risk.
- The math was verified in detail: projection (true per-segment foot points,
  signed cross-track matching the left-positive convention), friction-circle
  utilization, arc-length/curvature/accel derivations — all correct.
- Required fixes: (1) **cusp bug** — Menger curvature at a Reeds-Shepp gear
  cusp yields phantom κ (probe: κ = 13.27 m⁻¹ at a valid parking plan's
  cusp), so `checkFeasibility`/`scorePlan` misdiagnose *every real parking
  plan* as planner-infeasible — split at speed-sign flips before computing
  curvature; (2) reverse-gear accel/decel classification is swapped in the
  longitudinal feasibility check; (3) drop the committed machine-generated
  `demos/docs/eval-results/latest.json`; (4) add terminal pose/speed accuracy
  to the controller-isolation result and at least one reverse/parking-shaped
  reference so the harness can actually catch the stop-short and 16°-heading
  bug classes; (5) recalibrate the comfort/steer-reversal deadbands (they
  saturate on this stack's stepwise commands). Wire its reports into the
  existing sim-monitor/controller-bench rather than growing a parallel
  measurement stack.

### 2.3 PR #34 — Rich `Plan` structure — **MERGE-WITH-FIXES**

Solves a real representational gap: controllers consume bare
`CarKinematicState[]`, discarding curvature, feedforward steer, and cusp/gear
topology. The structure is well-designed (tiered optional fields, no forced
migration, `toStatePath()` round-trip) and sits directly under the
highest-value tracking fix this codebase needs (κ feedforward into
pure-pursuit — the repo currently caps parking speed at 2 m/s because the
tracker cuts tight Reeds-Shepp arcs by 20–30 cm).

Required fixes before merge (all verified, not speculative):

1. **`steerFf` sign bug in reverse** (`core/src/plan/build.ts:94`):
   `steerFf = atan(L·κ)` ignores travel direction; on every reverse segment
   the reported feedforward steer is the *negation* of the steer that
   produced the arc — wrong precisely for parking, the flagship use case.
   Main's own conversion (`race-scenario.ts:1206`,
   `steer = -gear * atan(...)`) proves the flip is required.
   Fix: `steerFf = atan(L · κ · dir)` with `dir` from the enclosing segment;
   add a reverse-arc test.
2. **Cusp boundary off-by-one** (`core/src/plan/segments.ts:41`): segment
   closes at the first decisively-new-gear sample, not the documented rest
   sample — would misplace stop targets for a segment-following controller.
3. Rebase over main (conflict with #39's rewritten `Parking.tsx`).
4. Unify with `splitAtGearCusps` — the runner now has two gear-splitting
   implementations with different deadband semantics (`segmentByGear` is the
   better one; `splitAtGearCusps` fails to split across an exact rest sample).

The follow-up PR that actually feeds `Plan.kappa` into pure-pursuit is what
converts this from speculative structure into a tracking-accuracy fix.

### 2.4 PR #33 — Model-comparison tooling + coverage OOD gate — **MERGE**

Zero conflicts with main; all 36 affected tests pass on the merged tree.
Adds honest ground-truth accountability for the learned dynamics model:
matched post-settle initial conditions, a *coverage* OOD gate (fixes the
dangerous "confident bias" failure mode of variance-only gating — ensembles
agreeing on a wrong answer), a settle-coast frame-fix that removes up to
metres of phantom model error, a shipped default model, and a CI ratchet
(`model-acceptance.test.ts`). Also fixes a latent bug (`rebuildModel`
dropped `oodStdThreshold` on load).

Important scoping fact verified during review: **the learned model is not in
the production driving loop** (parking primitives are kinematic; race MPC
uses `parametricForwardV2` with default params; pure-pursuit is model-free).
So this PR is research-track hardening — the right kind, and near-zero risk.
Minor follow-ups: two hardcoded mirrors of `DEFAULT_OOD_STD_THRESHOLD` should
import the now-exported constant; one log label says "mean" but computes RMS.

### 2.5 PR #30 — JAX trainer — **REBASE-THEN-MERGE, decoupled from #24**

All of its merge conflicts are inherited from its stacked base (#24), not its
own 27 files; its own changes plug into `training-driver.ts`, unchanged on
main. Well-guarded design (NM fits on runtime-identical piecewise physics by
default; per-stage no-regress gates; residual shipped only if >5 % better on
val; golden-file lockstep between the TS and JAX models). Required fixes:

1. `train_fit.py` hard-codes `PARAM_LO/HI` / `DEFAULT_PARAMS_VEC` /
   `REG_SCALES` that mirror **#24's widened bounds**, not main's — a Python
   fit would be silently clamped on load. Emit bounds from JS into the NPZ or
   a JSON sidecar so they cannot drift.
2. Dead code: `demos/scripts/workers/pool.ts` + `rapier-trial-worker.ts`
   (~190 lines) have no call sites — the `--workers` flag was never wired.
3. `vehicle-model.ts:422`: the smooth variant's brake gate
   `0.5·(1+tanh(brakeForce·0.01))` is 0.5 at zero brake input — damps speed
   ~25–30 %/tick near standstill even when coasting. Fix the gate to be 0 at
   0, regenerate the golden. Also correct the "<1e-3 agreement" comments
   (the test actually asserts 5e-2).

### 2.6 PR #25 — "chore: ignore artifacts" — **MERGE-WITH-FIXES**

Mistitled: 4 commits, +602/−31 — a content-addressed trial cache wired into
the training CLI, two offline analysis CLIs, and the gitignore chore. The
gitignore patterns are safe (match zero tracked files). One trivial union
conflict in `demos/package.json` (keep both script sets). Note
`analyze-race-debug.ts` reads bundles produced by a `--debug-dir` flag that
exists on no merged branch — inert until that producer lands. Retitle on
merge.

### 2.7 PR #24 — dense sweep path + a month of tuning experiments — **REWORK**

The title covers 1 of 39 commits (+4,873/−442 over 50 files). The headline
idea (expand primitive sweep samples so smoother/tracker/viz see the true
curve, not chords) is still valid — main fixed the *analytic-segment* chords
via `liftAnalyticPath` but primitive edges remain ~10–15 m chords at race
speed. However: 7 conflicted files vs main including a semantic conflict in
pure-pursuit (this branch and merged PR #40 solve the same two stop bugs
differently), and the branch's Reeds-Shepp expansion path is buggy
(duplicate parent pose with later timestamp; lerped heading across cusps;
**lerped speed discards gear sign** — a back-in shot reads as forward).
Salvage as focused PRs:

1. `expandPlanSweeps` primitive-branch only (delete its RS branch — main's
   `liftAnalyticPath` already does that correctly) + speed-colored viz + tests.
2. Pure-pursuit brake-to-terminal-speed sweep (`sqrt(v_t² + 2ad)` — a better
   formulation than main's raw min) reconciled with main's stop latch.
3. MPC drive/brake mutual exclusion + smoother sign-preservation at cusps +
   time-honest reverse pricing of RS segments (small, uncontroversial, main
   lacks all three).
4. Trial cache / train logging (overlaps PR #25 — pick one).
5. Separately debated: widened v2 parameter bounds & relaxed OOD thresholds
   (`vehicle-model.ts:155-172,342` — the original comments call these values
   unphysical; needs bench evidence).

### 2.8 PR #41 — integration-plan doc — **MERGE-AFTER-UPDATE** (or CLOSE)

The doc's Phase 0 was already executed by the repo owner *before the PR was
opened* (PRs #18/#40/#39 merged 90 minutes prior), and it classifies PR #39
as "not needed" when the owner merged it to HEAD. Its remaining gap analysis
is accurate and source-verified (see §3.4 — the dead `bumpRevision` /stale
`goalLB` memo finding is a real latent bug), but the plan contains **nothing
about driving quality** — no phase touches tracking, replan churn, or parking
time. Merging it un-edited would misdirect the next 7–9 days into integration
plumbing for a hypothetical host game while the user-visible pain is the
controller/termination stack. Update Part A to reflect merged state, add a
scope disclaimer, or close it.

---

## 3. Bugs on `main`

Ranked by user-visible impact. (F-numbers from the driving-stack audit;
all **confirmed** = reproduced by probe or traced concretely in code.)

### 3.1 Critical — the parking limit cycle

- **B1 (F1) — "stop" is `speed ≤ 0` (signed).** `parking-scenarios.ts:679`
  and `scenario-goals.ts:71` encode terminal stop as
  `{ speed: { max: 0 } }`; `core/src/scenario/guard.ts:14-16` compares the
  **signed** speed, so any reverse motion satisfies "stopped". Consequences:
  (a) the goal automaton latches DONE mid-reverse (HUD shows DONE ✓ with
  v ≠ 0 — monotone `stepAutomaton`, `core/src/scenario/progress.ts:30-52`);
  (b) the *same* guard is the planner's goal test
  (`scenario-environment.ts:220`), so near the goal A\* prefers cheap 0.5 s
  reverse stubs that terminate mid-reverse (probe: every plan from replan
  #26 on ends at −0.8…−1.5 m/s); (c) those plans defeat pure-pursuit's stop
  latch (`stopsAtEnd = |goal.speed| < 0.05`), so the tracker sails past the
  end and the shuffle repeats. **Fix:** stop = `|speed| ≤ ε` (e.g.
  `{min: −ε, max: ε}`), and validate direction-aware conditions at
  authoring time.
- **B2 (F2) — tolerance inversion.** `plannerGoalRadius: 0.35` >
  `arriveRadius: 0.25` (`parking-scenarios.ts:628,632`). The planner may
  declare success anywhere in the 0.35 m disk; probes show terminal poses at
  0.24–0.33 m; the only run-ending condition is a position-only 0.25 m
  crossing (`pickNextWaypoint`, `race-primitives-scenarios.ts:686-701`). The
  race configuration documents the required invariant (gate radius strictly
  < arrive radius, `race-primitives-scenarios.ts:406-411`) — parking inverts
  it. The shuffle attractor sits 2–3 cm outside the finish disk; the 124 s
  finish was luck. **Fix:** enforce `plannerGoalRadius + terminal scatter <
  arriveRadius` (or better: make the finish condition the shared success
  oracle, not a raw radius), and assert the invariant in code.
- **B3 — replanning from a parked pose commands a shuffle.** Planner probe:
  planning with start == goal pose (stopped, inside the goal region) returns
  a 3-point path `(0,6,v=0) → (0,5.25,v=−1.5) → (0,6,v=0)`, cost 1.05 —
  i.e. "reverse 0.75 m and come back" — instead of a trivial/empty path.
  Combined with the unconditional 800 ms cadence (B5) this is the engine of
  the post-arrival churn. Mechanism confirmed in §3.5b/P4: scenario guards
  are evaluated only on edges, so the search must *move* to fire one.
  **Fix:** test the start state against acceptance before searching.

### 3.2 High — terminal precision

- **B4 (F3) — dead deceleration ramp / unit bug.**
  `core/src/execute/pure-pursuit.ts:131`:
  `Math.max(vGoal, config.lookaheadMin)` uses `lookaheadMin` (**3 m**,
  a distance; `execute/types.ts:4-5`) as a **3 m/s** speed floor. Parking
  cruise is 2 m/s, so the brake-to-goal ramp `vGoal` *never* engages;
  terminal approach is bang-bang (full speed until 0.08 m or the end latch,
  then brake=1), stopping from 1–2 m/s takes 0.1–0.5 m → the observed
  0.19–0.53 m scatter. The repo has been tuning around this
  (`goalTolerance` 0.4 → 0.08 "to re-center the skid",
  `parking-scenarios.ts:615-626`). **Fix:** a dedicated `minApproachSpeed`
  (m/s) parameter; let `vGoal` engage below it near stop-terminated ends.
  (PR #24's brake-to-terminal-speed sweep is the principled version.)
- **B5 (F6) — replan cadence has no "still good / at rest" gate**
  (`race-scenario.ts:1067`), and the plan-expiry path conflates time with
  divergence: once sim time passes the plan's horizon, `lateralFromPlan`
  returns **Infinity** (`race-scenario.ts:1004-1030`) which reads as
  "lateral error > 2 m" → early replan every 0.5 s stub. Separately,
  `core/src/execute/replan.ts:61-66` measures divergence **time-indexed**
  (`planPoseAt(plan, current.t)`) — a tracker merely slower than the plan's
  time parameterization reads as diverged while perfectly on path, and
  `consider()` force-adopts on every 'divergence'. **Fix:** geometric
  cross-track + heading divergence; gate cadence replans on "plan invalid,
  goal not yet settled".
- **B6 — no terminal heading control in pure-pursuit** — parallel parking
  ends 16° off (bench FAIL; `it.fails` test). PR #36 is the candidate fix.

### 3.3 Medium — measurement and semantics

- **B7 (F5) — bench/tests stop at a transient snapshot.**
  `controller-bench.ts:196-202` breaks the first tick `parked` is true;
  `PARKING_SUCCESS.speedTol = 0.3 m/s` counts a creeping car as stopped;
  `coverageMin = 0.85` tolerates ~0.3 m lateral offset. The web page runs
  the closed loop and exposes what the bench masked. Same code, different
  stop condition, opposite verdicts (PASS 22.3 s vs 124 s of thrash).
- **B8 (F4) — three success verdicts displayed simultaneously** (automaton
  DONE at 22 s; `evaluateParked` flickering true/false; runner not finished
  until 124 s). One screen, three answers.
- **B9 — degenerate-plan fallback creeps forward.** When a car has no plan
  or its live segment has < 2 points, the runner commands constant
  `throttle 0.2` forward (`race-scenario.ts:1218,1230` area) — reasonable
  mid-course, dangerous at a goal or near obstacles.
- **B10 — navcat adapter: `bumpRevision()` is dead** — nothing reads
  `.revision`, and the `goalLB` distance-field memo
  (`core/src/adapters/navcat/index.ts:74-76,107-111`) is keyed by goal
  coords only → a rebuilt world serves a stale goal-distance field. Latent
  (demos construct fresh worlds), but a real trap for live-world use.
- **B11 — planner pass hysteresis carry-over** — see §3.5b/P5 (confirmed:
  `perPass = [32, 1, 2967]`). Degrades multi-resolution search exactly when
  the goal is hard.

### 3.4 Low

- **B12 (F8)** — web tick pacing is display-refresh dependent
  (`Parking.tsx:376-377`: no accumulator; 60 Hz → 4× realtime, 120 Hz → 8×).
  Sim results unaffected (fixed 1/60 ticks); playback speed varies.
- **B13 (F9)** — planner deadline is wall-clock inside the search; a loaded
  browser can time out where Node doesn't (rarely binds; plans ~1 ms).
- **B14** — `controller-bench --filter=parking` (the header's own example)
  matches nothing (exact-name filter).
- **B15** — steer conversion applies `atan(κ · 2·WHEEL_BASE)`
  (`race-scenario.ts:1206`) — the 2× factor is an uncalibrated fudge; fine
  as tuning, but it means "curvature" commands are not dimensionally honest
  through the Rapier path.

### 3.5 Executor-layer audit (additional confirmed findings)

An independent headless probe of the reverse-perp scenario over 150 sim-s
confirmed the run is a **permanent shunt livelock**, not "slow parking":
`finished` never latches, 208 replans by 150 s (≈171 at t≈123 s — the exact
HUD numbers), lateral offset frozen at x ≈ 0.29 m for 120+ s, **144 gear
flips**, last-30 s mean |v| = 0.58 m/s.

- **E1 (critical) — parking runs with race lookahead.**
  `PURE_PURSUIT_CONFIG` (`race-scenario.ts:135-150`) sets
  `lookaheadMin: 3, lookaheadMax: 14`; parking overrides only
  cruiseSpeed/goalTolerance/respectPathSpeed (`:637-642`). Parking segments
  are 0.5–3 m, so the lookahead point **is always the path endpoint** —
  pure pursuit degenerates to "aim at the goal point", which cannot remove
  lateral error (the frozen 0.29 m offset) and corner-cuts every entry hook.
  Fix: speed-scaled lookahead floor ≈ 0.5–1 m for parking-class speeds.
- **E2 (high) — curvature infeasibility chain.** Planner plans at
  `minTurnRadius: 3.5` (`parking-scenarios.ts:143`); the tracker clamps at
  4.5 (`RACE_AGENT.minTurnRadius`); the physical plant can only do
  R = L/tan(δmax) = 3.2/tan(0.6) ≈ **4.68 m**
  (`raycast-vehicle.ts:181,253-258`). Every tight parking arc is 25–34 %
  beyond the chassis's capability → systematic wide tracking → the lateral
  residual that the finish disk can't absorb. Fix: one agent-capability
  source of truth; planner radius ≥ plant radius (with margin).
- **E3 (high) — intermediate cusps carry no stop semantics.**
  `splitAtGearCusps` (`race-scenario.ts:166-185`) ends non-final segments at
  ±cruise speed, not 0; pure-pursuit's stop latch needs
  `|goal.speed| < 0.05`, so cusps are braked only inside the 0.08 m disk and
  the segment-advance gate (`dist ≤ 0.25 && |v| < 0.5`) is rarely met
  cleanly — most gear changes happen as side effects of replans (144 flips).
  Fix: zero the terminal sample's speed when splitting segments.
- **E4 (high) — `lateralFromPlan` conflates along-track lag with lateral
  error** (`race-scenario.ts:1004-1030`): the plan is trimmed by *elapsed
  time*, cusps have no dwell, so an on-path car lagging plan-time reads as
  > 2 m "lateral" → spurious adaptive replans, and each adopted replan
  resets `activeSegIdx = 0` mid-maneuver. Same design flaw in
  `ReplanState.divergence` (time-indexed, `replan.ts:61-66`).
- **E5 (medium, latent) — `smoothSpeedProfile` erases terminal and cusp
  stops** (`speed-profile.ts:174-211`): endpoint curvature forced to 0, the
  backward pass starts *from* an unconstrained `v[n-1]`, magnitudes smoothed
  across sign flips, `vMin` floor applied after the brake-feasibility pass.
  Currently disabled for race and parking — a landmine if enabled.
- **E6 (medium, latent) — `smoothTrajectory` breaks reverse legs**
  (`trajectory-smoother.ts:91,176-202`): recomputed headings flip by π on
  reverse segments; speed lerped across sign flips mints bogus near-zero
  samples that shift cusp boundaries. Parking correctly disables it.
- **E7 (medium) — the *public* `PlanFollowerCarDriver` still has the bugs
  the demo runner fixed privately** (`core/src/vehicle/car/drivers.ts:95-97`):
  throttle clamped to [0,1] (reverse impossible), no reverse steer-sign
  flip, wheelbase misuse vs the Rapier adapter's half-spacing convention.
  Library users get the broken version of what the demo already fixed.
- **E8 (medium) — MPPI reference interpolates heading without angle wrap**
  (`mpc-tracker.ts:201`): plans crossing ±π poison the heading cost. Latent
  for /parking (pure-pursuit), live for any MPC scenario near ±π.
- **E9 (low) — gear flips forward one sample early at the end of reverse
  approaches** (`pure-pursuit.ts:71-72`: terminal sample speed ≈ 0 reads as
  forward) — small window, lands exactly where parking precision matters.

### 3.5b Planner-bridge audit (additional confirmed findings)

All verified by running probes against the real reverse-perp geometry.

- **P1 (critical) — gear flips are effectively free in scenario/multi-goal
  bridge mode.** `VehicleEnvironment.succ` guards the direction-change
  penalty with `parentReverse !== undefined`, but when `node.edge` is null
  `parentReverse` is **`null`**, and `ScenarioEnvironment.succ` /
  `MultiGoalEnvironment.succ` rebuild the inner node with
  `createNode(state, null, null)` — edge dropped — so **every** node looks
  edge-less and **every successor pays the penalty**
  (`vehicle-environment.ts:294-311`, `scenario-environment.ts:194`,
  `multi-goal.ts:118`). Measured: reverse-after-reverse edges cost 0.675
  (with penalty) in bridge mode vs the correct 0.525 in the legacy path.
  A constant per-edge tax means *changing gear has zero marginal cost* —
  the planner freely emits multi-cusp shuffle plans, and g-inflation weakens
  heuristic guidance. Fix: thread the parent edge through the bridge node
  rebuild (or use `?.` semantics like `tryAnalyticShot` already does).
- **P2 (critical) — time bucket in the dedup hash even in static scenes.**
  `Math.round(state.t / 0.25)` is baked into the exact hash
  (`vehicle-environment.ts:234-239`) and TimeAware appends a second,
  differently-quantized tag (`time-aware.ts:164-173`), so the same pose at
  different times never dedups. Measured on the identical static query:
  **83,968 expansions / 14.8 s with t-hash vs 22,220 / 4.0 s without**,
  same cost and terminal. Inside the 500 ms replan budget this is the
  difference between solving and failing whenever the analytic shot is
  blocked. Fix: exclude t from hash/dominance when the world has no moving
  obstacles or affordances.
- **P3 (high) — the best-progress fallback returns the START node as
  `found: true`.** `ScenarioEnvironment.progress` = depth − g·1e-6; for a
  single-phase `reach` every non-accepting node has depth 0, so the best
  "progress" is the start (g = 0) and a deadline-hit returns
  `{found: true, partial: true, cost: 0, path: [start]}`
  (`ighastar.ts:235-244`, `scenario-environment.ts:272-276`). The demo
  survives only because it also checks `path.length > 1`, converting these
  into failed replans that feed the 150 ms retry storm. Fix: progress
  tie-break on −h (distance-to-goal), and/or `found: false` for
  degenerate partials.
- **P4 (high) — the bridge cannot recognize an already-satisfied start.**
  Scenario guards fire only on *edges* (`scenario-environment.ts:216-234`),
  so a car starting exactly at the goal, stopped, gets a 3-point plan:
  reverse 0.75 m at −1.5 m/s and come back (cost 1.05) instead of trivial
  success. With the 800 ms cadence this is the engine of the post-arrival
  shuffle. Fix: evaluate acceptance on the start state before searching.
- **P5 (medium) — hysteresis stagnation counter never resets across
  passes** (`ighastar.ts:81,157,211-217`): after a coarse pass breaks at
  threshold+band, every subsequent non-finest pass breaks after **one**
  expansion (measured `perPass = [32, 1, 2967]`). The multi-resolution
  ladder degenerates to "coarsest then finest".
- **P6 (medium) — the RS heuristic LUT is cleared whenever the goal
  changes** (`vehicle-environment.ts:422-428`) — in multi-goal search nodes
  with different gates interleave on the open list, wiping the shared cache
  nearly every alternation, precisely where the table is advertised.
- **P7 (medium) — moving-obstacle collision is only checked at primitive
  endpoints** (`time-aware.ts:234-242`) — a 0.5 s primitive can tunnel
  through a predicted obstacle mid-edge (multi-agent scenes).
- **P8 (low-medium) — heading-bucket seam split at ±π**
  (`vehicle-environment.ts:220-223`): the same physical direction gets two
  hash/dominance keys near the seam → duplicated subtrees for west-facing
  maneuvers (perf only, no correctness risk).
- **P9 (medium) — analytic-shot cost model is optimistic** (whole RS curve
  priced at top speed, no accel/cusp dwell, `vehicle-environment.ts:394`),
  biasing incumbent comparisons toward shots and producing plan timelines
  the chassis cannot keep — feeding time-divergence replans.
- Notes: `TimeAwareEnvironment.succ` drops the `level` argument;
  `kinematicForwardSim` implements instant speed change (plans assume
  instantaneous gear reversal at full speed); ~10-12 string allocations per
  generated successor across the wrapper stack (GC hotspot).

### 3.6 Things done well (do not break these)

- **Single-source-of-truth runner architecture is real** — web, CLI bench and
  Vitest share `createRaceScenario`/`parkingScenarioOptions`/`evaluateParked`;
  the fixed-dt deterministic design allowed tick-exact headless reproduction
  of the live site. This is rare and is the foundation everything in §4
  builds on.
- **Honest failure encoding** — known-broken behavior is kept red via
  `it.fails` rather than deleted; teleport/rescue hacks were deliberately
  removed with rationale.
- **The reverse-gear execution bridge is correct** — signed throttle, steer
  sign flip in reverse, per-sample gear in `liftAnalyticPath`, cusp-splitting
  segment executor. The car really drives a reverse-S; failures are confined
  to the terminal half-meter and the layers that judge it.

---

## 4. Q3 — Test strategy: making tests see exactly what users see

### 4.1 What already holds

Code-path parity is genuinely in place (see §3.6). The missing half is
**measurement parity**: the bench/tests must observe the same closed loop the
user watches, for as long as the user watches it, and judge it with the same
oracle the HUD displays.

### 4.2 The plan

1. **One success oracle, settle-latched.** A single exported predicate —
   conceptually `settled(state, scenario) = parked(state) held continuously
   for ≥ 2 s with |v| ≤ 0.05 m/s` — used by: the runner itself (entering a
   terminal HOLD state that stops cadence replanning — this is what makes
   the web page *able* to look right), the bench loop, the Vitest suites,
   and the HUD banner. One definition, one module, no per-layer thresholds.
   The scenario `stop` condition becomes `|speed| ≤ ε` (B1) and the runner's
   finish becomes this oracle instead of a raw radius crossing (B2).
2. **Closed-loop quality budgets as CI assertions.** Per scenario:
   max time-to-settled, max replans, min coverage, max final pos/heading
   error, min clearance, zero collisions/teleports. E.g. reverse-perp:
   settled ≤ 30 s, ≤ 45 replans, ≥ 95 % coverage, ≤ 3° heading error.
   Today's suite asserts none of the first two — which is precisely how
   124 s / 171 replans passes CI. Keep budgets in one data table shared by
   bench + tests so the bar cannot drift per-harness.
3. **Post-success hold window.** After the oracle first fires, keep ticking
   N more seconds and assert it *stays* true (no creep-out, no replan
   shuffle). This single change converts the bench's transient snapshot
   (B7) into the user's experience.
4. **Controller unit-envelope property tests.** Steady-state tracking error
   of pure-pursuit on canonical geometries (straight approach, min-radius
   arc, cusp approach at various entry speeds) must be < arriveRadius/2.
   This one property test would have caught B2/B4 (0.27 m steady-state vs
   0.25 m finish disk) years before a human noticed. PR #37's eval module
   (projection-based cross-track, reference shapes, gauntlet corridor with
   passability oracle) is exactly this layer — merge it and wire it into CI.
5. **Cross-layer tolerance invariants as tests.** Assert in code:
   `plannerGoalRadius + expected terminal scatter < arriveRadius`,
   `trackerGoalTolerance < plannerGoalRadius`, `at() box ≥ planner goal
   region`, scenario stop ε == tracker stop ε. These are one-line tests that
   make the five-layer contract explicit and unbreakable.
6. **Replay artifacts for failures.** The bench already emits JSON; extend
   to a JSONL trajectory dump on failure + a /replay page that loads it in
   the browser. "What CI saw" becomes pixel-identical to "what I see",
   closing the last parity gap (visual inspection).
7. **Keep the web thin.** `Parking.tsx` should contain: rendering, input,
   and *calls into* the shared runner + shared oracle. It already mostly
   does; the remaining page-local logic (lap-time HUD semantics, status
   banner selection) should read the oracle's state machine instead of
   recomputing verdicts (B8). Add a fixed-timestep accumulator (B12) so
   playback speed is display-independent.

### 4.3 Sequencing with the bug fixes

Write the new bench assertions **first** (they fail today — that's the
point): settle-latch oracle + budgets turn the current pathology into red
tests. Then fix B1 → B2 → B4 → B5 (each should flip specific assertions
green). This is the TDD loop the user asked for, and it runs identically
headless and in the browser.

---

## 5. Q4 — Remaining gaps to "production-ready"

### 5.1 Driving quality (the current fire) — suggested order

Small, surgical, high-leverage first (each should flip specific new bench
assertions from §4 green):

1. **B1** stop = `|speed| ≤ ε` in scenario guard semantics (one comparison).
2. **P1** thread the parent edge through the bridge node rebuild so the
   gear-flip penalty is charged only on actual flips.
3. **P4/B3** start-state acceptance check ("already satisfied" result).
4. **B2/P5-chain** enforce `plannerGoalRadius + scatter < arriveRadius <
   success region` as a code-level invariant; make the runner's finish the
   shared settle-latched oracle, then enter a HOLD state (stop cadence
   replans).
5. **E1** parking-scale lookahead (speed-scaled floor ~0.5–1 m).
6. **E2** one source of truth for turn radius; plan at ≥ plant capability.
7. **B4** real `minApproachSpeed` (m/s) so the decel ramp engages
   (PR #24's brake-to-terminal-speed sweep is the principled formulation).
8. **E3** zero cusp-sample speeds when splitting segments.
9. **P2** drop t from hash/dominance in static worlds (3.8× search speedup —
   makes the 500 ms budget reliable when the analytic shot is blocked).
10. **E4/B5** geometric divergence (cross-track + heading), replans gated on
    "plan invalid & not settled"; stop resetting `activeSegIdx` on adopt
    when the maneuver is unchanged.
11. **P3** fix the best-progress fallback (tie-break on −h; degenerate
    partials are not `found`).

Expected outcome per the audits: reverse-perp settles in ~20 s with < 40
replans and stays settled; the 102 s limit cycle disappears; parallel
becomes fixable by a re-tuned #36.

Then terminal-pose quality:

- Re-work + land PR #36's heading term (after #37's harness is in CI).
- Curvature feedforward (PR #34 + follow-up consumer) to lift the 2 m/s
  parking cap and stop cutting Reeds-Shepp arcs.
- Adopt PR #24's salvageable executor pieces: MPC drive/brake exclusivity,
  smoother cusp sign preservation, time-honest reverse pricing.
- Fix the public `PlanFollowerCarDriver` (E7) — library users currently get
  the broken version of what the demo runner fixed privately.

### 5.2 API/contract hardening (for robotics sims & game NPCs)

- **Success semantics as first-class API**: scenario goals need `settled`
  /hold semantics, |speed| stop conditions, and a queryable terminal state
  ("achieved and holding", not a latched DONE while moving).
- **Start-state acceptance** (B3): `plan()` from a satisfied state returns
  "already satisfied", not a shuffle.
- **Divergence API**: geometric (cross-track + heading + gear-aware), not
  time-indexed (B5); expose "plan still valid" so hosts can gate replans.
- **Live-world updates**: `bumpRevision` is dead (B10) — revision-keyed
  cache invalidation, region-scoped invalidation with
  trajectory-intersection tests, and worker-protocol support for tile/link
  updates (PR #41's Parts B/C are an accurate audit of this gap).
- **ETA/feasibility oracle** as a public API (games need "can I make it in
  T?" at ~ms cost).
- **Multi-agent in core**: plan-registry cooperation exists; a planner pool
  + frame-budget governor (N NPCs at 60 fps) is still demo-level.

### 5.3 Performance & robustness

- Planning is already fast (~1 ms parking replans; 500 ms budget unused).
  The risk is the *loop*, not the search: allocation pressure in per-tick
  code paths, and the B11 hysteresis carry-over degrading hard searches.
- Determinism: already good headlessly; remove wall-clock deadline effects
  (B13) with an expansion-budget-first policy for reproducible plans across
  environments.

### 5.4 Productization

- Versioning/release discipline: semver, changelog, API-stability statement,
  size budget already enforced (< 100 KB — good).
- Docs: the README oversells relative to measured behavior (e.g.
  "battle-tested" driving is not, yet); the honest bench table (§1.2) should
  be the public status page, auto-generated in CI.
- The learned-dynamics track (PRs #30/#33) is promising but **keep it off
  the driving-quality critical path** — the kinematic/parametric stack must
  pass the bench first; the learned model then improves fidelity behind the
  same gates.

---

*Full per-PR review notes and the probe methodology live in the PR thread /
session log. F-numbers reference the driving-stack audit; B-numbers are the
canonical bug list above.*
