# Max-pace roadmap: plant envelope, dynamic rollouts, racing MPPI, and a course that demands them

*Successor to `docs/racing-at-the-limit-plan.md`. That document diagnosed WHY
both cars drive at ~30% of capability and sketched the executor fixes
(Track 1 = MPPI cost redesign, Track 2 = pure-pursuit speed-policy surgery).
This document turns it into the full next-phase roadmap, adding the three
things the executor fixes alone don't cover:*

1. *drive the Rapier `DynamicRayCastVehicleController` at its **measured**
   limit, not a hand-derived fraction of it — floor it on straights like a
   human would;*
2. ***dynamic rollouts in planning** — expand the search from the car's true
   dynamic state (speed, yaw rate, sideslip) projected through its own
   forward model, instead of zero-slip table lookups, so the v2 learned
   model's statefulness is finally used where it matters;*
3. *a **course that requires an advanced driving AI** — where late braking,
   corner-entry speed, and line choice are decided by model knowledge and
   mistakes have physical consequences.*

*Status: PLAN. Every file:line reference verified against this branch
(post-PR #50: grip-saturating brake model, technical course, per-car MPPI
model wiring, driving-quality metrics).*

---

## 0. Where we are (measured)

| Signal | Kinematic | v2-trained | Plant capability |
|---|---|---|---|
| Mean executed speed (open, 2 laps) | 9.3 m/s | 8.8 m/s | 30 m/s ceiling |
| g-g mean utilization | 0.445 | 0.435 | 1.0 |
| Open-loop endpoint error @0.8 s | 6.1 m mean, 22.8 m worst @28 m/s | 0.68 m mean, 1.5 m worst | — |
| Closed-loop avg lap (open) | 33.0 s | 39.2 s (ratio 1.188) | — |
| Technical course avg lap | 38.4 s | 43.1 s (ratio 1.12) | — |

The model war is won (9× open-loop advantage, ratcheted); the *pipeline*
war is not: an executor that never exceeds 0.44 of the friction circle and a
planner that quantizes away dynamic state price model fidelity at zero. The
kinematic car wins on line length alone because sharp intent + plant
saturation is free at 9 m/s.

**Definition of done for this phase** (each measurable, each ratcheted):

- **D1 — Floor it.** On the 105 m straight (gate 7→8), the executed
  trajectory holds ≥ 95% throttle (or ≥ 95% `maxDriveForce` under MPPI)
  until the braking point and peaks ≥ 28 m/s. Asserted headlessly.
- **D2 — At the limit.** g-g mean utilization ≥ 0.60 for the v2 car over a
  clean 2-lap run (from today's 0.44), peak ≥ 0.95.
- **D3 — Dynamic rollouts.** Replan expansion at the root uses the car's own
  forward model rolled from the exact live state (incl. `yawRate`,
  `lateralVelocity`); verified by a unit test that a mid-corner replan's
  first primitive endpoint matches a live model rollout within ε, and by a
  closed-loop predErrorRms drop.
- **D4 — Model-in-the-loop executor.** MPPI (rolling each car's own model,
  wiring already landed) is the race-default tracker once it beats
  pure-pursuit lap times on both courses; both cars complete both courses
  under it, deterministically.
- **D5 — The course decides.** On the new circuit variant, the full v2 stack
  beats the full kinematic stack (`v2.avg < kin.avg`, ratio ratcheted through
  1.0) with v2 clean; the kinematic stack either strikes walls or laps slower.
  Identical controller + tuning both cars; only the forward model differs.
- **D6 — No regressions.** `pnpm verify` green; parking/carchase/ramp
  invariants hold; existing ratchets never loosened.

---

## 1. WS-0 — Measure the plant envelope (stop guessing what "maximum" is)

**Problem.** Every speed-limiting constant in the executor is hand-set below
the plant's real limits, and the derived capabilities themselves
underestimate the measured plant:

- Tracker longitudinal limits `maxAccel: 6`, `maxDecel: 8`
  (`race-scenario.ts:146-147`) vs derived traction accel **8.83 m/s²** and
  brake **13.89 m/s²** (`core/test/agent/capabilities.test.ts:26-34`) — and
  the *measured* brake is stronger still: brakeForce 1000/2000 stops a
  24 m/s chassis in ~0.8 s ≈ **26 m/s²**, grip-saturated (comment at
  `core/src/agent/vehicle-model.ts:301-307`).
- Lateral budget `TRACKER_MAX_LATERAL_ACCEL = 12` (`race-scenario.ts:91`)
  and preview at `0.8·µg ≈ 14.1` (`race-scenario.ts:164-165`) vs µg =
  **17.66 m/s²** — and nobody has measured the true steady-state cornering
  boundary of the raycast controller (suspension + `frictionSlip: 1.8` +
  `sideFrictionStiffness` make it non-trivially different from µg).
- Top speed "30 m/s" is documented folklore
  (`race-primitives-scenarios.ts:249-251`), pinned only as an upper bound.

**Work.**

1. Add a **plant-envelope characterization script** on the headless Rapier
   harness (`core/src/adapters/rapier/headless-trial.ts`):
   - top speed on flat ground (drive at full force until dv/dt < ε);
   - 0→vmax time and distance (traction-limited launch curve `a(v)`);
   - braking distance / decel from a grid of speeds (threshold-brake sweep
     over brakeForce to find shortest stop without lockup-induced yaw);
   - steady-state cornering: for a grid of fixed steer angles × entry
     speeds, measure sustained yaw rate / radius / lateral accel → the
     real g-g boundary `aLatMax(v)`.
2. Emit a `PlantEnvelope` record (JSON artifact + in-repo constant) and a
   regression test pinning it (tolerances, fixed seed) so plant-tuning
   drift is caught — the same pattern as `capability-drift.test.ts`.
3. Route executor and planner limits from the envelope with explicit
   margins, replacing the hand-set numbers above: pure-pursuit
   `maxAccel/maxDecel/maxLateralAccel/previewLateralAccel`, the speed
   profile's budgets (`race-scenario.ts:1253-1261`), and
   `RACE_AGENT.minTurnRadius` (resolve the deliberate 4.5 vs 4.68 inversion,
   `race-primitives-scenarios.ts:237-248`, flipping
   `capability-drift.test.ts:33` `it.fails` → `it` once WS-1's feedforward
   makes feasible-radius plans the fast ones).

**Acceptance.** Envelope artifact + test committed; a table in the PR body:
derived vs measured (vmax, a(0), a(20), brake decel, aLat @ R=10/20/40).
Executor limits cite the envelope, not literals.

*Note on "maximizing usage of the raycast controller": MPPI already emits
native `steer/driveForce/brakeForce` (`core/src/execute/mpc-tracker.ts:38-50`)
— the plant's own actuator space, no throttle-abstraction loss. WS-0 tells
us what the actuators can do; WS-1/WS-3 make the stack ask for all of it.*

---

## 2. WS-1 — Floor it: the speed-policy surgery (Track 2, made concrete)

The five model-agnostic brakes in `purePursuit`
(`core/src/execute/pure-pursuit.ts:245-253`) and their fixes, in order:

1. **Kill phantom horizon braking.** `vGoal = √(2·maxDecel·distToGoal)`
   (`pure-pursuit.ts:150-152`) must apply **only** when the plan genuinely
   stops (`stopsAtEnd`, already computed at line 57) or the gate is a true
   finish. Drive-through horizons get `vGoal = ∞`. Today the car is
   permanently decelerating toward the end of a 2-gate replanning window
   (`PLAN_LOOKAHEAD_COUNT = 2`, `race-scenario.ts:90`) that the next replan
   extends. This is the single biggest straightaway killer.
2. **Bang-bang throttle with braking-envelope timing.** Replace the
   proportional law `throttle = Δv/maxAccel` (`pure-pursuit.ts:260-272`)
   with: full throttle while `v < vBind − hysteresis`, where `vBind` is the
   binding cap propagated backward through the braking envelope; brake at
   the envelope-computed point, at envelope decel. A human floors it until
   the braking point; so should we. (Keep the P-law for parking — gear on
   `stopsAtEnd` or scenario tuning.)
3. **Separate feedback error from path curvature.** `vCurve` uses the
   pursuit-chord κ = 2y/Ld² (`pure-pursuit.ts:149`), which conflates
   cross-track error with real corners → phantom braking on straights. Use
   the rich `Plan.kappa` (`core/src/plan/build.ts:45-119`, built at
   `race-scenario.ts:1277` and currently produce-but-don't-consume) for the
   speed law and preview; keep chord κ only for steering feedback. This
   also supersedes the noisy Menger preview over replanned chords
   (`pure-pursuit.ts:183-204`).
4. **`respectPathSpeed: true` everywhere** (currently off on the open
   course, `DEFAULT_TUNING.respectPathSpeed: false`,
   `race-scenario.ts:470`) with plan speeds as a **cap** through the
   existing braking-envelope semantics (`pure-pursuit.ts:207-235`). With
   WS-2's honest plan speeds this is the cheap channel for v2 knowledge to
   reach the wheels under pure-pursuit.
5. **Raise the limits to the WS-0 envelope** (accel 6→~8.8, decel
   8→measured threshold-brake value, lateral 12→measured boundary with a
   margin). Conservative-limit "safety" is currently doing the model's job
   for both cars, which is exactly what equalizes them.

**Acceptance** (deterministic, `retry: 0`, open course): D1 holds
(≥ 95% throttle to the braking point, peak ≥ 28 m/s on the straight — new
assertion in `closed-loop-race-benchmark.test.ts`); both cars' mean speed
≥ +40%; laps beat today's 33.0/39.2 s; 0 off-track; determinism
bit-identical; parking untouched (all changes gated on drive-through
plans / race tuning).

### 2b. WS-1½ — Control feedforward: stop re-deriving actuators from geometry

**Problem.** The stack plans in actuator space, flattens to geometry, then
re-derives actuator commands from geometry — destroying the plan's control
knowledge twice:

- Every primitive stores the exact `[steer, driveForce, brakeForce]` that
  produced it (`characterize.ts:129`) and every plan edge remembers its
  primitive (`primId`, `vehicle-environment.ts:343`) — but plan extraction
  keeps only the state polyline; nothing downstream reads `primId` or
  `.controls` (verified: zero references in `race-scenario.ts`).
- The rich `Plan`'s `steerFf`/`accelFf` are *re-derived* from the
  already-smoothed geometry (`buildPlan`, `race-scenario.ts:1277`) — a
  reconstruction of what was discarded — and even that is unread.
- On the technical course, `smoothSpeedProfile` **overwrites** the plan's
  model-honest per-sample speeds with a generic curvature formula at the
  conservative hand-set limits (`race-scenario.ts:1253-1261`) — the v2
  planner's corner-speed knowledge is erased before the tracker sees it.

**Work.**

1. Thread `primId → MotionPrimitive.controls` through plan extraction so
   each plan sample carries the baked control vector alongside the pose
   (time-aligned; smoothing moves geometry, controls stay keyed by `t`).
2. Feedforward + feedback execution: the tracker emits the plan's controls
   as the feedforward term and adds a bounded geometric correction on top
   (pure open-loop replay would drift — the baked controls are only exact
   from the state they were baked at; FF+FB is the standard shape).
3. Speed profile becomes a **safety clamp, not a rewrite**: keep the plan's
   model-derived speeds; only reduce a sample's speed when it exceeds the
   envelope-backed friction/braking bound. (Today's pass replaces v2's
   honest speeds with model-agnostic math — an own-goal for the fidelity
   experiment.)
4. MPPI warm-start from the plan's control sequence (instead of only the
   shifted previous solution) — the plan's control knowledge becomes the
   sampling prior that MPPI refines, unifying this workstream with WS-3.

**Acceptance.** Steering/throttle traces on a corner-entry segment show the
feedforward term carrying most of the command (feedback correction small
and zero-mean); v2 lap times improve more than kinematic's (its baked
controls encode true-plant knowledge; kinematic's encode delusion — this
change is honestly asymmetric); tracking error (predErrorRms) does not
regress; parking unaffected (FF gated to drive-through plans).

---

## 3. WS-2 — Dynamic rollouts: plan from the true dynamic state (the new workstream)

**Problem, precisely.** The replanner starts from the live continuous state
— `startState = { ...c.car.readState(simTime), t: 0 }`
(`race-scenario.ts:1086-1088`), and `readState` supplies exact signed speed,
heading, `yawRate`, and `lateralVelocity`
(`core/src/adapters/rapier/raycast-vehicle.ts:278-297`). But the search then
throws that dynamic state away:

- Expansion is a **table lookup**: `this.lib.lookup(st.speed)` snaps to the
  nearest 4 m/s start-speed bucket (`vehicle-environment.ts:319`,
  `core/src/primitives/library.ts:20-37`, `RACE_START_SPEEDS`
  `race-primitives-scenarios.ts:327`).
- The baked primitives were characterized from **zero-slip canonical
  states** `{x:0, z:0, heading:0, speed, t:0}` — no `yawRate`, no
  `lateralVelocity` (`core/src/primitives/characterize.ts:119`).
- Successor states **drop the dynamic dims** entirely — `next` carries only
  `x/z/heading/speed/t` (`vehicle-environment.ts:324-330`).

So a car sweeping through a corner at 20 m/s with a large yaw rate and
sideslip replans as if it were rolling straight at 20 m/s with zero slip.
The v2 model is *stateful in exactly these dims* (`parametricForwardV2`
integrates `yawRate` through `yawRateTau` and `lateralVelocity` through
`lateralDamping`, `core/src/agent/vehicle-model.ts:286-394`; the residual
MLP takes both as inputs, `vehicle-model.ts:535-576`) — the planner
quantizes away the very state the learned model exists to exploit. The
kinematic model, memoryless by construction, loses nothing to this
quantization. **The current lattice is rigged in the kinematic model's
favor.** The characterize contract even anticipates the fix: *"characterize
at plan time from the actual state instead of caching"*
(`characterize.ts:46-52`).

**Work.**

1. **Root dynamic rollouts.** Extend `VehicleEnv` with an optional live
   expansion hook: for the **root node only**, roll the entry's own forward
   model (`RaceEntry.forwardModel` — same model MPPI rolls,
   `race-scenario.ts:1040`) from the exact live state across the race
   control sets, producing primitives (endpoint + sweep samples) on the
   fly. Cost per replan: |controlSets| ≈ 19 × 6 substeps ≈ 114 model steps
   (~200 ns each) — negligible. Collision sweeps come straight from the
   rollout samples, same `sweepClear` path. The committed first ~0.55 s of
   every plan becomes exactly model-consistent with the car's dynamic
   state; deeper nodes keep the baked lattice (their states are primitive
   endpoints, which are near-canonical by construction).
2. **Carry `yawRate`/`lateralVelocity` through successor states** where the
   generating model provides them (v2 primitive `end` states can bake
   terminal `yawRate`/`lateralVelocity` per bucket; kinematic stays zero —
   honest, since that's its worldview). Optional depth-1+ dynamic rollouts
   behind an expansion budget if root-only proves insufficient — measure
   first.
3. **Commit-window forward projection.** `commitWindowMs: 0` today because
   the linear plan-sampling predictor diverged
   (`race-scenario.ts:447-453`). At 30 m/s a 300 ms replan cadence
   (`REPLAN_INTERVAL_MS`, `race-scenario.ts:89`) means the car moves ~9 m
   during/after planning. Re-enable the commit window, but project the
   start state forward by rolling the car's **own model** over the
   currently-committed controls for the window — plan from where the car
   *will be*, not where it was. This is the second half of "current state,
   projected out."
4. **Keep determinism.** Model rollouts are pure; no RNG. The dedup hash
   (`speedQuant: 4`, `vehicle-environment.ts:245-251`) is unchanged — it
   only dedups, never mutates state.

**Acceptance.**

- Unit test: build a mid-corner state (v=18, yawRate=0.8, lateralVelocity=1.2),
  plan one expansion, assert the root primitive endpoints equal a direct
  model rollout within 1e-9 (and differ measurably from the baked lookup).
- Closed-loop: predErrorRms drops for the v2 car (it finally predicts from
  the state it's actually in); failed replans at speed don't increase.
- Benchmark laps: v2 improves ≥ kinematic (this change should be
  asymmetric by design — that's the point); ratchets tightened.

---

## 4. WS-3 — MPPI as the racing executor (Track 1, unchanged scope, gated rollout)

Scope as specified in `docs/racing-at-the-limit-plan.md` §Track 1 — kept
here as the dependency spine, since D2/D4/D5 all hang off it:

1. **Progress-reward cost**: replace position-tracking with
   `cost −= wProgress·(arc-length advanced)` + lateral-corridor penalty
   (half-width from course metadata; the Williams 2016 racing shape),
   fixing progress starvation in `buildReference`/`scoreRollout`
   (`mpc-tracker.ts:158-263`).
2. **Reference extension** past the plan end (extrapolate the last segment)
   so the horizon end is attractive, never a stop target.
3. **Substep the model**: 3 × 1/60 s model steps per 0.05 s MPPI step
   (64 samples × 10 steps × 3 ≈ 1920 model steps/tick, < 1 ms) to close
   the dt gap vs the 1/240 s plant.
4. **Per-car model** is already wired (`race-scenario.ts:1040`,
   `headless-race.ts` kinematic → `KINEMATIC_NATIVE_PARAMS`, v2 →
   `learnedForwardSimV2`); keep it — this is the honesty contract.
5. **Determinism test in MPC mode** (LCG seeding already in place,
   `createMPCTrackerState`, `mpc-tracker.ts:125-130`).
6. **Gated default flip**: `tracker: 'mpc'` becomes the race default per
   course only when its deterministic lap times beat pure-pursuit's there.
   Parking keeps pure-pursuit until separately tuned.

**Acceptance**: completes 2 laps on open + technical + circuit, both cars,
0 off-track; D2 (g-g ≥ 0.60 mean for v2); the headline ratchet
`v2.avg < kin.avg` on the technical/circuit courses.

---

## 5. WS-4 — A course that requires an advanced driving AI

**Problem.** The technical variant only guards *overshoot zones* (9 walls,
`race-primitives-scenarios.ts:199-219`); pure-pursuit's conservative
preview braking keeps the delusional car clean too, so walls rarely bind
and the kinematic car keeps its shorter-line edge (ratio 1.12). Meanwhile
the open course's 105 m straight (gate 7→8) can already host vmax — from
rest the plant needs ~51 m to reach 30 m/s at traction-limited accel — so
the course is not what caps speed today; the executor is (WS-1). Once WS-1
lands, the course must be the thing that separates the models.

**Design principles** (each feature targets a specific model-knowledge gap,
with the speed-differentiating numbers from WS-0's envelope):

1. **Heavy braking zone**: keep the long straight, then a hairpin
   (R ≈ 6–8 m) with an outside wall at corner exit. Entry from ~30 m/s;
   braking distance is the model question — the grip-saturating brake
   knowledge (v2) vs naive `F/m` (kinematic) decides the braking point.
   Late braking = wall.
2. **High-speed sweeper**: constant-radius R ≈ 35 m corner, walled on the
   outside. Takeable at `√(aLat·R)` ≈ 20 m/s at the measured boundary; the
   kinematic worldview says any speed. This is where 22.8 m of open-loop
   error at 28 m/s becomes a wall strike instead of a diagnostic.
3. **Decreasing-radius corner** (R 20 → 10 m): the classic
   fidelity-separator — requires trailing off entry speed *before* the
   tightening is visible in the pursuit chord; only plan-level model
   knowledge (or MPPI rollouts) gets it right.
4. **Chicane at speed**: keep the existing x=10/−12 staggered pinch
   (`race-primitives-scenarios.ts:217-218`).
5. **Full corridor**: a `'circuit'` variant walling **both** sides
   everywhere — wide corridors on straights (~14 m), tight in technical
   sections (~6 m vs the 4.8 × 2 m footprint). No free run-off anywhere;
   every line error costs. Built with the existing dual-layer pattern
   (2D inflated obstacle polygons for the planner + Rapier box colliders +
   rendered walls, `RACE_WALL_INFLATE`, `race-primitives-scenarios.ts:101`).
6. **Consequences dialed up**: wall strikes already count
   (`wallStrikes`); add a benchmark mode where a strike voids the lap
   (assert winner has 0), so "graze the wall, keep the time" cannot pay.
7. **Gate-heading prior fallback** for V-shaped plans stays in reserve as
   per the previous doc: finite `goalHeadingTol` aligned to the
   gate-to-gate chord, only if V-shapes persist after WS-1/WS-2 raise plan
   speeds (at high plan speed a pivot prices itself out organically).

**Exposure**: `buildRaceCourse('circuit')`; `/raceprimitives?course=circuit`
selector; `pnpm run race -- --course=circuit` (`demos/scripts/race.ts:56,92`);
a geometry/solvability test mirroring `technical-course.test.ts` (every
gate reachable by the v2 library with honest speeds — the course must be
hard, not impossible).

**Acceptance**: D5 — on the circuit, full v2 stack beats full kinematic
stack with v2 clean; kinematic strikes walls or laps slower at matched
aggression; demo renders the circuit; video walkthrough artifact.

---

## 6. WS-5 — Measurement, ratchets, and honesty rails

- **New assertions** (all deterministic, expansion-capped, `retry: 0`):
  - straightaway: peak speed ≥ 28 m/s AND sustained ≥ 95% drive command on
    the straight (D1);
  - g-g mean ≥ 0.60 / peak ≥ 0.95 for the v2 car (D2), via the existing
    `DrivingQuality` plumbing (`race-scenario.ts:552-568, 1790-1797`);
  - dynamic-rollout consistency unit test (D3);
  - MPC determinism (D4);
  - circuit ratchet `v2.avg < kin.avg` (D5) — tighten
    `V2_VS_KIN_AVG_RATIO` (open, currently 1.25/measured 1.188) and
    `V2_VS_KIN_TECH_RATIO` (currently 1.25/measured 1.12) monotonically as
    each workstream lands; never loosen.
- **Honesty rails** (unchanged): identical controller + tuning both cars;
  the only per-car difference is the forward model (planner primitives,
  root rollouts, MPPI rollouts, commit-window projection — all roll the
  car's own model). Any tuning change applies to both.
- **Per-workstream benchmark runs**: `pnpm run race -- --seed=42 --laps=3`
  on open + technical + circuit after every landing; table goes in the PR.
- **Full gate**: `pnpm verify` (typecheck + ~840 tests + build + size);
  parking/carchase/ramp invariants; GUI QA on `/raceprimitives` per course
  (walls render, HUD updates, no console errors), with the video artifact
  from the circuit as the hero.

---

## 7. Sequencing (each step independently verifiable, benchmark after each)

1. **WS-0** plant envelope (unblocks honest limits everywhere; small,
   self-contained).
2. **WS-1** speed-policy surgery (both cars get fast; D1 lands; fidelity
   starts being priced).
3. **WS-1½** control feedforward (plan controls reach the actuators;
   speed profile demoted to a clamp; sets up the MPPI warm-start prior).
4. **WS-2** dynamic rollouts (root rollouts → successor dyn-state →
   commit-window projection; D3 lands; v2's statefulness finally counts).
5. **WS-3** MPPI cost redesign + substepping + gated default flip (D2/D4).
6. **WS-4** circuit course (D5's arena; walls bind now that speeds are
   real).
7. **WS-5** ratchet tightening through 1.0 + final artifacts.

Rationale for the order: speed first (fidelity is priced at zero at
9 m/s), then model-in-the-loop planning/execution (converts the priced
fidelity into commands), then the course (measures it), then the ratchet
(locks it). WS-0 precedes everything because WS-1's bang-bang and WS-4's
braking-zone geometry are both parameterized by the measured envelope.

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Faster driving destabilizes pure-pursuit before MPPI is ready | WS-1 raises caps only through envelope-backed braking math; preview keeps binding on genuine corners; wall-strike/off-track ratchets guard every landing |
| Root dynamic rollouts introduce planner nondeterminism or perf cost | rollouts are pure + seedless; ~114 model steps/replan measured budget; root-only scope until data says otherwise |
| Commit-window projection re-creates the old divergence bug | projection now uses the model (the old failure was linear sampling, `race-scenario.ts:447-453`); adaptive lateral-drift replan trigger stays as the safety net |
| MPPI progress-reward cost cuts corners through walls | corridor penalty is part of the same cost; circuit solvability test keeps a feasible corridor; wall strikes assert 0 for the winner |
| Circuit too hard → both cars DNF, benchmark says nothing | course solvability test with the v2 library at honest speeds is a landing precondition; corridor widths start generous and tighten with measured clearance |
| Kinematic car improves too (WS-1 helps both) | expected and desired — the claim is *relative*: v2 wins because its extra speed is placed where the plant can actually deliver it; if kinematic still wins on the circuit, the course (or the model) needs another iteration, and the benchmark will say which (wall strikes vs lap time) |

## 9. Non-goals

- No model retraining or architecture changes in this phase (body-frame
  residuals, dynamic-bicycle backbone, residual-dt alignment remain queued
  behind it — current fidelity is sufficient for the pipeline to reward).
- No new dependencies (no JAX; everything stays in-repo and deterministic).
- No changes to parking/carchase behavior (tuning stays per-scenario).
