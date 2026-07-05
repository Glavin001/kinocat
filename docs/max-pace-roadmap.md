# Max-pace roadmap: plant envelope, dynamic rollouts, racing MPPI, and a course that demands them

*Successor to `docs/racing-at-the-limit-plan.md`. That document diagnosed WHY
both cars drive at ~30% of capability and sketched the executor fixes.
This document is the implementation plan for the full next phase. Every
workstream follows the same shape — **What to build → How to build it →
Acceptance checklist** — so each one is unambiguous about the deliverable
and mechanically verifiable when done.*

*Status: PLAN. Every file:line reference verified against this branch
(post-PR #50: grip-saturating brake model, technical course, per-car MPPI
model wiring, driving-quality metrics).*

---

## 0. Pain points → workstreams (traceability)

Each pain point raised in review, the workstream that resolves it, and the
checklist items that prove it:

| # | Pain point (as raised) | Resolved by | Proven by |
|---|---|---|---|
| P1 | Cars nowhere near max speed; on a straight a human floors it | WS-1 (speed policy), WS-0 (envelope limits) | A1.3, A1.4, A0.3 |
| P2 | Course not challenging enough to require an advanced driving AI; kinematic's sharp turns pay no overshoot cost | WS-4 (circuit course) | A4.5, A4.6, A4.7 |
| P3 | Planning must do dynamic rollouts: current state projected out through the model, leveraging v2's statefulness | WS-2 (dynamic rollouts) | A2.1, A2.2, A2.3 |
| P4 | Better controller/executor — MPPI | WS-3 (racing MPPI) | A3.3, A3.5, A3.6 |
| P5 | Maximize usage of the Rapier `DynamicRayCastVehicleController` — its true envelope, its native actuators | WS-0 (measured envelope), WS-3 (native-control MPPI) | A0.1, A0.3, A3.4 |
| P6 | Coasting: zero gas + zero brake (optionally while turning) must be available — braking is far more severe than gliding | v2 library (already has coast primitives, `race-primitives-scenarios.ts:464,487,517,521-522`); WS-1 adds an executor coast band | A1.2 |
| P7 | Primitive segments may be too long / compose imprecisely (seams) | WS-2 (dynamic state through seams; root rollouts) | A2.1, A2.2, A2.5 |
| P8 | The plan's rich control knowledge (exact steer/drive/brake per primitive) must actually drive the actuators, not be re-derived from geometry | WS-1½ (control feedforward) | A1½.1–A1½.5 |
| P9 | 0.55 s chunks may be too coarse/noisy — maneuvers needing in-between switch times (e.g. a flick at t = 0.25 s into a chunk) aren't representable in-plan | §0c (architecture: chunked deliberation + continuous execution); WS-3 (50 ms control resolution); WS-2 optional interruptible primitives | A3.3, A3.6, A2.7 |

**Definition of done for the phase** (the headline claims, all deterministic,
all ratcheted):

- **D1 — Floor it.** ≥ 95% drive command until the braking point and
  ≥ 28 m/s peak on the 105 m straight (gate 7→8).
- **D2 — At the limit.** v2 g-g mean utilization ≥ 0.60 (today 0.44),
  peak ≥ 0.95, on a clean run.
- **D3 — Dynamic rollouts.** Root expansions roll the car's own model from
  the exact live dynamic state; seams carry `yawRate`/`lateralVelocity`.
- **D4 — Model-in-the-loop executor.** MPPI rolling each car's own model is
  the race default, having beaten pure-pursuit's lap times.
- **D5 — The course decides.** On the circuit variant, full v2 stack beats
  full kinematic stack (`v2.avg < kin.avg`), v2 clean; identical
  controller + tuning both cars, only the forward model differs.
- **D6 — No regressions.** `pnpm verify` green; parking/carchase/ramp
  invariants hold; no ratchet ever loosened.

## 0b. Where we are (measured baseline)

| Signal | Kinematic | v2-trained | Plant capability |
|---|---|---|---|
| Mean executed speed (open, 2 laps) | 9.3 m/s | 8.8 m/s | 30 m/s ceiling |
| g-g mean utilization | 0.445 | 0.435 | 1.0 |
| Open-loop endpoint error @0.8 s | 6.1 m mean, 22.8 m worst @28 m/s | 0.68 m mean, 1.5 m worst | — |
| Closed-loop avg lap (open) | 33.0 s | 39.2 s (ratio 1.188) | — |
| Technical course avg lap | 38.4 s | 43.1 s (ratio 1.12) | — |

The model war is won (9× open-loop advantage, ratcheted); the *pipeline*
war is not: an executor that never exceeds 0.44 of the friction circle and
a planner that quantizes away dynamic state price model fidelity at zero.

## 0c. Temporal resolution: where in-between timing comes from (P9)

Within one committed plan, control switches happen only at primitive
boundaries — multiples of 0.55 s from the replan instant (~15 m at top
speed). Three layers make the *effective* resolution much finer, and this
hierarchy (chunked deliberation + continuous execution) is the standard
state-lattice architecture (Pivtoraiko/Kelly; Apollo/Autoware), not a
shortcut:

1. **Replans re-anchor the phase** every 300 ms + adaptive triggers
   (`race-scenario.ts:89`) — chunk boundaries realign to the live state
   roughly twice per chunk.
2. **The executor actuates continuously**: steering/throttle recomputed at
   60 Hz; brake-onset timing is answered by the braking-envelope math
   every tick (`pure-pursuit.ts:172-235`) — the brake point is NOT
   quantized to chunk boundaries. Chunks quantize the planner's *intent*
   (line/maneuver topology), not when the pedals move.
3. **MPPI (WS-3) owns fine timing by design**: 0.05 s control resolution
   re-optimized every tick, with the chunked plan as a progress corridor.

Counter-pressure against simply shrinking chunks (measured): at 4 m/s a
0.55 s primitive's endpoint fan was already too small for the planner to
distinguish choices (hull 0.19 m² < the 0.5 m² usability floor,
`race-primitives-scenarios.ts:403-404`) — chunks must stay big enough that
choices *differ*. The remaining genuine gap — maneuver topologies needing
an in-plan switch mid-chunk — has a cheap fix: **interruptible
primitives** (WS-2 optional item, A2.7). Each primitive already records 6
substep samples ~0.09 s apart (`characterize.ts:63-66`); exposing early
termination at substep boundaries (root node first, branching measured)
gives ~0.09 s in-plan switch granularity at zero extra model-rollout cost.

---

## 1. WS-0 — Measure the plant envelope

### Problem

Every speed-limiting constant is hand-set below the plant's real limits,
and the derived capabilities underestimate the measured plant:

- Tracker limits `maxAccel: 6`, `maxDecel: 8` (`race-scenario.ts:146-147`)
  vs derived traction accel **8.83 m/s²** / brake **13.89 m/s²**
  (`core/test/agent/capabilities.test.ts:26-34`) — and the *measured*
  brake is ~**26 m/s²**, grip-saturated (`vehicle-model.ts:301-307`).
- Lateral budget `TRACKER_MAX_LATERAL_ACCEL = 12` (`race-scenario.ts:91`)
  and preview at `0.8·µg ≈ 14.1` vs µg = **17.66 m/s²**; the raycast
  controller's true cornering boundary has never been measured.
- Top speed "30 m/s" is documented folklore
  (`race-primitives-scenarios.ts:249-251`), pinned only as an upper bound.

### What to build

1. `demos/scripts/plant-envelope.ts` — a characterization script on the
   headless Rapier harness (`core/src/adapters/rapier/headless-trial.ts`).
2. A `PlantEnvelope` record: `{ vMax, launchCurve: a(v) grid, brakeDecel(v)
   grid, latAccelBoundary: aLat(v) grid, minTurnRadiusExecuted }` — emitted
   as `demos/public/models/plant-envelope.json` + an exported constant.
3. `demos/test/plant-envelope.test.ts` — regression test pinning the
   envelope (the `capability-drift.test.ts` pattern).
4. Rewiring: executor + speed-profile + agent limits route from the
   envelope with explicit named margins.

### How to build it

- **vMax**: full `driveForce` on flat ground until `dv/dt < 0.01` for 1 s;
  record terminal speed.
- **Launch curve**: same run; sample `a(v)` at v = 0, 4, …, 28.
- **Brake decel**: from each v ∈ {8, 16, 24, 28}, sweep brakeForce over
  {0.25, 0.5, 0.75, 1.0}·max; record best stopping decel without
  lockup-yaw (heading deviation < 5°).
- **Cornering boundary**: grid of fixed steer ∈ {0.15, 0.3, 0.45, 0.6} ×
  entry v ∈ {8, 12, …, 28}; hold 3 s; record sustained yaw rate, radius,
  aLat at steady state → the boundary is the max sustained aLat per speed.
- All runs deterministic (fixed spawn, no RNG); JSON written with sorted
  keys so re-runs diff clean.
- Rewire: replace literals in `PURE_PURSUIT_CONFIG`
  (`race-scenario.ts:141-171`), the speed-profile budgets
  (`race-scenario.ts:1253-1261`), and `RACE_AGENT.minTurnRadius`
  (`race-primitives-scenarios.ts:248`) with envelope-derived constants ×
  named margin factors.

### Acceptance checklist

- [ ] **A0.1** `pnpm tsx demos/scripts/plant-envelope.ts` writes
  `plant-envelope.json`; running twice produces byte-identical output.
- [ ] **A0.2** `plant-envelope.test.ts` pins: vMax within ±0.5 m/s of the
  recorded value; brake decel and aLat boundary each within ±5% at every
  grid point; fails if plant tuning drifts.
- [ ] **A0.3** No hand-set speed/accel literals remain in the executor
  path: `PURE_PURSUIT_CONFIG.{maxAccel,maxDecel,maxLateralAccel,
  previewLateralAccel}` and the speed-profile budgets reference the
  envelope (code review + a grep-style test asserting the constants'
  provenance comments).
- [ ] **A0.4** PR table: derived vs measured (vMax, a(0), a(20), brake
  decel @24, aLat @ R=10/20/40).
- [ ] **A0.5** `capability-drift.test.ts:33` `it.fails` (minTurnRadius
  inversion) either flips to `it` or its deferral is re-justified against
  the measured `minTurnRadiusExecuted`.
- [ ] **A0.6** `pnpm verify` green.

---

## 2. WS-1 — Floor it: speed-policy surgery on pure-pursuit

### Problem

`purePursuit` commands the minimum of five caps, all model-agnostic
(`core/src/execute/pure-pursuit.ts:245-253`): phantom horizon braking
(`vGoal` treats the 2-gate replanning horizon end as a stop target,
lines 150-152), phantom corner braking (pursuit-chord κ conflates tracking
error with curvature, line 149), a proportional throttle that never floors
it (lines 260-272), and the one model-carrying cap (`respectPathSpeed`)
disabled on the open course (`race-scenario.ts:470`).

### What to build

1. `vGoal` gating: applies **only** when `stopsAtEnd` (already computed,
   `pure-pursuit.ts:57`) or the gate is a true finish; drive-through
   horizons get `vGoal = ∞`.
2. Plan-curvature speed law: `vCurve`/preview use the rich `Plan.kappa`
   (built at `race-scenario.ts:1277`, currently unread) instead of chord
   κ / Menger-over-chords; chord κ remains for steering feedback only.
3. Bang-bang throttle with an explicit **coast band** (P6): full throttle
   while `v < vBind − h`; **zero throttle, zero brake** while
   `|v − vBind| ≤ h` (glide); brake only when the braking envelope to the
   next binding cap requires it. Hysteresis `h` ≈ 0.5 m/s. The P-law stays
   for stop-terminated (parking) plans.
4. `respectPathSpeed: true` as the open-course default (envelope-cap
   semantics already implemented, `pure-pursuit.ts:207-235`).
5. Limits raised to WS-0 envelope values (with named margins).

### How to build it

- All changes inside `purePursuit` + `PurePursuitConfig`, gated so
  stop-terminated plans keep today's behavior exactly (parking
  untouched). New config fields: `goalBrakeOnlyAtStops: boolean`,
  `planCurvature?: (i: number) => number` (or pass the `Plan` alongside
  the polyline), `coastBandMs: number`.
- Wire `richPlan` through `race-scenario.ts` tick → tracker call
  (`race-scenario.ts:1570` region) so the tracker can read per-sample
  kappa. This makes the rich Plan consumed for the first time.
- Update `DEFAULT_TUNING` (`race-scenario.ts:465-479`):
  `respectPathSpeed: true`; keep every change per-scenario-gated.

### Acceptance checklist

- [ ] **A1.1** Unit tests in `core/test/execute/pure-pursuit.test.ts`:
  (a) drive-through plan (terminal speed > 0.05) → `vGoal` does not bind
  at any distance; (b) stop-terminated plan → braking behavior identical
  to today (snapshot); (c) straight plan with 1 m cross-track error →
  no `vCurve` slowdown (plan-kappa is zero even though chord κ isn't).
- [ ] **A1.2** Coast-band unit test (P6): current speed within hysteresis
  of the binding cap → command is exactly `throttle = 0, brake = 0`;
  above the band → brake per envelope; below → throttle = 1.
- [ ] **A1.3** (D1, P1) New assertion in
  `closed-loop-race-benchmark.test.ts`: on the gate 7→8 straight, drive
  command ≥ 0.95 for ≥ 60% of the straight's arc length AND peak speed
  ≥ 28 m/s, for BOTH cars.
- [ ] **A1.4** Open-course benchmark: both cars' mean speed ≥ 12.5 m/s
  (≥ +40% over 8.8/9.3); avg laps < 33.0 s (kin) and < 39.2 s (v2);
  0 off-track; failed replans ≤ 2; determinism test bit-identical.
- [ ] **A1.5** Parking invariants, `race-invariants.test.ts`, carchase and
  ramp suites unchanged/green (`pnpm verify`).
- [ ] **A1.6** Ratchets tightened to the new measured values (never
  loosened): lap-time bounds and `V2_VS_KIN_AVG_RATIO` re-pinned.

---

## 3. WS-1½ — Control feedforward: the plan's controls reach the actuators

### Problem (P8)

The stack plans in actuator space, flattens to geometry, then re-derives
actuator commands from geometry — destroying its own control knowledge
twice:

- Every primitive stores the exact `[steer, driveForce, brakeForce]` that
  produced it (`characterize.ts:129`); every plan edge remembers its
  primitive (`primId`, `vehicle-environment.ts:343`) — but plan extraction
  keeps only the state polyline. Zero downstream references to `primId` or
  `.controls` exist (verified in `race-scenario.ts`).
- The rich `Plan.steerFf/accelFf` are re-*derived* from already-smoothed
  geometry (`buildPlan`, `core/src/plan/build.ts:45-119`) — and unread.
- On the technical course, `smoothSpeedProfile` **overwrites** the plan's
  model-honest speeds with a generic curvature formula at conservative
  hand-set limits (`race-scenario.ts:1253-1261`).

### What to build

1. Plan extraction preserves per-edge controls: each plan sample carries
   `{ steer, driveForce, brakeForce }` from its generating primitive,
   time-aligned by `t` (smoothing moves geometry; controls stay keyed to
   time).
2. FF+FB tracking mode in pure-pursuit: command = plan feedforward controls
   + bounded geometric feedback correction (pure open-loop replay drifts —
   the baked controls are exact only from the state they were baked at).
3. Speed profile demoted from rewrite to **clamp**: never raise a sample's
   speed; lower it only where it exceeds the envelope-backed bound.
4. MPPI warm-start option: seed the prior from the plan's control sequence
   (instead of only the shifted previous solution), unifying with WS-3.

### How to build it

- Extend the multi-goal planner's result (`plan-vehicle-multi.ts`) to
  surface per-edge `primId`; resolve to `MotionPrimitive.controls` at plan
  assembly in `replanCar` (`race-scenario.ts:1209-1282`); store as a
  parallel `controls[]` on the committed plan (or extend `Plan`).
- FF term: interpolate controls by plan time; FB term: today's pursuit
  correction, clamped to a fraction of actuator range (start: ±30% steer,
  ±40% force).
- Clamp-only speed profile: new option in `smoothSpeedProfile`
  (`core/src/execute/speed-profile.ts:152-223`): `mode: 'clamp'` skips the
  forward-accel pass's speed raises and applies only cap/brake passes.
- MPPI warm start: map plan controls at the horizon's time offsets into
  the prior buffer (`mpc-tracker.ts:334-343`) behind
  `warmStartFromPlan: boolean`.

### Acceptance checklist

- [ ] **A1½.1** Unit test: for a 3-primitive plan, the extracted plan's
  `controls[]` at each sample time equals the generating primitive's
  control vector (exact match; no interpolation across primitive
  boundaries).
- [ ] **A1½.2** FF-dominance evidence: on the deterministic benchmark's
  90° corner entry, log |FF| vs |FB| per tick — feedforward carries
  ≥ 70% of the steering command magnitude; FB correction is zero-mean
  (|mean| < 10% of RMS). Logged by the benchmark, asserted loosely.
- [ ] **A1½.3** Clamp-only speed profile test: given a v2 plan whose
  model-honest speeds are lower than the curvature formula's, output
  speeds are UNCHANGED; given speeds exceeding the envelope bound, output
  is reduced to the bound; speeds are never raised.
- [ ] **A1½.4** Honest asymmetry (the point of the exercise): with FF+FB
  on for both cars, v2's lap improves ≥ kinematic's improvement
  (kinematic's baked controls encode delusion; measure and record both).
- [ ] **A1½.5** MPPI warm-start behind a flag; MPC determinism test still
  bit-identical with it on.
- [ ] **A1½.6** `predErrorRms` does not regress for either car; parking
  suite green (FF gated to drive-through plans).

---

## 4. WS-2 — Dynamic rollouts: plan from the true dynamic state

### Problem (P3, P7)

The replanner reads the live continuous state — `startState =
{ ...c.car.readState(simTime), t: 0 }` (`race-scenario.ts:1086-1088`),
including exact `yawRate` and `lateralVelocity`
(`raycast-vehicle.ts:278-297`) — then throws the dynamics away:

- Expansion is a table lookup snapped to the nearest 4 m/s bucket
  (`vehicle-environment.ts:319`, `library.ts:20-37`).
- Primitives were baked from zero-slip canonical states
  (`characterize.ts:119`).
- Successors drop `yawRate`/`lateralVelocity` entirely
  (`vehicle-environment.ts:324-330`) — so every seam between primitives
  resets slip to zero (P7: imprecise composition).

The v2 model is stateful in exactly these dims
(`vehicle-model.ts:286-394`); the kinematic model is memoryless. The
current lattice is rigged in the kinematic model's favor. The characterize
contract anticipates the fix: *"characterize at plan time from the actual
state instead of caching"* (`characterize.ts:46-52`).

### What to build

1. **Root dynamic rollouts**: for the root node, roll the entry's own
   forward model (`RaceEntry.forwardModel`, same model MPPI rolls,
   `race-scenario.ts:1040`) live from the exact current state across the
   control sets → primitives (endpoint + sweep) on the fly.
2. **Dynamic state through seams**: primitive `end` states bake terminal
   `yawRate`/`lateralVelocity` (v2; kinematic stays 0 — honest); successors
   carry them forward.
3. **Commit-window forward projection**: re-enable `commitWindowMs > 0`
   with the start state projected by rolling the car's own model over the
   committed controls for the window (the old failure was *linear* plan
   sampling, `race-scenario.ts:447-453`). At 30 m/s and 300 ms cadence the
   car moves ~9 m per replan — planning from where the car *will be* is
   mandatory at speed.
4. **Optional — interruptible primitives** (P9, see §0c): record
   per-substep speed alongside the existing per-substep pose samples in
   `characterize()`, and let the search terminate a primitive early at a
   substep boundary (~0.09 s granularity). Scope to the **root node
   first** (where dynamic rollouts are already generating fresh
   trajectories) to bound the branching growth (×6 if applied
   everywhere); expand deeper only if the measured benefit justifies the
   node count.

### How to build it

- Add an optional `rootExpand?: (state) => MotionPrimitive[]` to
  `VehicleEnvOptions`; `succ()` uses it when `node.parent == null`.
  Implementation: run `characterize()` inline with `runs =
  [{ startState: liveState, controls }]` per control set — ~19 sets × 6
  substeps ≈ 114 model steps (~200 ns each) per replan; collision sweeps
  come from the rollout samples through the existing `sweepClear` path.
- Extend `MotionPrimitive.end` with optional `yawRate`/`lateralVelocity`;
  populate in `characterizeVehicle` (`characterize.ts:126-140`); copy in
  `succ()` (`vehicle-environment.ts:324-330`). Dedup hash unchanged
  (`speedQuant` stays a dedup-only key).
- Commit-window projection: in `replanCar`, replace `samplePlanAt` linear
  sampling with an N-step rollout of the committed plan's controls
  (WS-1½'s `controls[]`) through `entry.forwardModel`.

### Acceptance checklist

- [ ] **A2.1** (D3) Unit test: from a mid-corner state (v = 18,
  yawRate = 0.8, lateralVelocity = 1.2), root expansion endpoints equal a
  direct model rollout within 1e-9 — and differ from the baked
  zero-slip lookup by a measurable margin (assert > 0.3 m for at least
  one control).
- [ ] **A2.2** (P7) Seam test: chaining two hard-corner v2 primitives via
  successor states reproduces a continuous 2-primitive model rollout
  within a tolerance that the old zero-slip seam violates (record both
  errors in the test).
- [ ] **A2.3** Commit-window test: with `commitWindowMs: 200`, the
  planning start state equals the model rollout of the committed controls
  at +200 ms (not the linear sample); closed-loop off-track events do not
  increase vs `commitWindowMs: 0`.
- [ ] **A2.4** Closed-loop: v2 `predErrorRms` drops (record before/after);
  failed replans do not increase; benchmark lap ratchets re-pinned (v2
  expected to gain more than kinematic — record both).
- [ ] **A2.5** Determinism test still bit-identical (rollouts are pure);
  replan CPU budget: expansion count per replan within 5% of before
  (root rollouts add ~114 model steps, no search-space change).
- [ ] **A2.6** `pnpm verify` green; parking unaffected.
- [ ] **A2.7** (optional, P9) Interruptible primitives at the root:
  unit test that a plan can commit a partial first primitive ending at a
  substep boundary (endpoint = the recorded substep state); benchmark
  A/B with the feature on vs off — land it only if lap time or
  failed-replan count improves with expansion count growth ≤ 2×; record
  both outcomes in the PR either way.

---

## 5. WS-3 — MPPI as the racing executor

### Problem (P4, P5)

MPPI exists, rolls each car's own model (wiring landed,
`race-scenario.ts:1040`), and emits the plant's native actuator space
(`mpc-tracker.ts:38-50`) — but both cars DNF under `tracker: 'mpc'`:
position-tracking cost starves progress (`buildReference` advances a
reference the samples must chase, `mpc-tracker.ts:158-216`), the horizon
end acts as a stop target, and the 0.05 s model step mismatches the
1/240 s plant.

### What to build

1. **Progress-reward cost**: `cost −= wProgress · (arc length advanced
   along the plan)` + lateral **corridor** penalty (quadratic outside
   half-width; from course metadata, default 3 m) replacing exact position
   tracking. The Williams 2016 racing shape.
2. **Reference extension**: extrapolate the plan's last segment when the
   plan is shorter than the horizon (the horizon end becomes attractive,
   never a stop).
3. **Model substepping**: 3 × 1/60 s model steps per 0.05 s MPPI step
   (64 × 10 × 3 ≈ 1920 steps/tick, < 1 ms).
4. **Completion + determinism tests**, then a **gated default flip** to
   `tracker: 'mpc'` per course, only where its deterministic lap times
   beat pure-pursuit's. Parking keeps pure-pursuit.

### How to build it

- New cost path in `scoreRollout` behind `costMode: 'progress' | 'track'`
  (default `'track'` so parking is untouched); progress measured by
  arc-length projection of each rollout state onto the plan polyline.
- Substepping inside the rollout loop (`mpc-tracker.ts:371-375`): step the
  model 3× at dt/3 per horizon step.
- Keep λ, sampling stds, and the seeded LCG as-is (determinism preserved,
  `createMPCTrackerState`, `mpc-tracker.ts:125-130`).
- New `demos/test/mppi-race-completion.test.ts`; extend
  `determinism.test.ts` with an MPC-mode case.

### Acceptance checklist

- [ ] **A3.1** Cost unit test: of two synthetic rollouts on a straight
  plan, the one advancing farther scores strictly lower cost; a rollout
  outside the corridor scores higher than one inside at equal progress.
- [ ] **A3.2** Reference-extension unit test: plan shorter than horizon →
  reference extends beyond the last sample along its tangent; no terminal
  speed drop for drive-through plans.
- [ ] **A3.3** (D4) `mppi-race-completion.test.ts`: 2 laps on open AND
  technical AND circuit, both cars, `tracker: 'mpc'`, `retry: 0`,
  0 off-track, 0 DNF.
- [ ] **A3.4** (P5) Per-tick MPPI compute < 1 ms measured in the benchmark
  (log + assert); commands are native `steer/driveForce/brakeForce`
  reaching `setWheelBrake`/engine-force paths unchanged.
- [ ] **A3.5** (D2) v2 g-g mean utilization ≥ 0.60, peak ≥ 0.95 on a clean
  2-lap technical run under MPPI.
- [ ] **A3.6** MPC determinism: two runs bit-identical (ε = 1e-9,
  matching `determinism.test.ts:35`).
- [ ] **A3.7** Default flip evidence: recorded lap-time table
  pure-pursuit vs MPPI per course; `tracker: 'mpc'` becomes default only
  for courses where MPPI wins; tuning identical for both cars.
- [ ] **A3.8** Headline ratchet: `v2.avg < kin.avg` on the technical
  course under MPPI (`V2_VS_KIN_TECH_RATIO` through 1.0).

---

## 6. WS-4 — A course that requires an advanced driving AI

### Problem (P2)

The technical variant only guards overshoot zones (9 walls,
`race-primitives-scenarios.ts:199-219`); conservative preview braking
keeps the delusional car clean too, so walls rarely bind (ratio 1.12).
The 105 m straight already supports vMax (~51 m to reach 30 m/s) — the
executor was the cap (WS-1), so post-WS-1 the course must be what
separates the models.

### What to build

`buildRaceCourse('circuit')` — same dual-layer pattern as `technical`
(planner-inflated 2D obstacles + Rapier box colliders + rendered walls,
`RACE_WALL_INFLATE`, `race-primitives-scenarios.ts:101`), with features
that each target a specific model-knowledge gap:

1. **Heavy braking zone**: hairpin (R ≈ 6–8 m) at the end of the long
   straight, outside wall at corner exit. Entry ~30 m/s; the braking point
   is decided by brake-model knowledge (v2's grip-saturating brake vs
   kinematic's instant-speed delusion). Late braking = wall.
2. **High-speed sweeper**: constant R ≈ 35 m, walled outside. Takeable at
   `√(aLat·R)` ≈ 20 m/s at the measured boundary; the kinematic worldview
   says any speed — 22.8 m of open-loop error at 28 m/s becomes a strike.
3. **Decreasing-radius corner** (R 20 → 10 m): requires slowing before the
   tightening is visible to a chord-based tracker; only model-level
   knowledge (plan speeds or MPPI rollouts) gets it right.
4. **Chicane at speed**: keep the existing x = 10/−12 staggered pinch.
5. **Full corridor**: both sides walled everywhere — ~14 m wide on
   straights, ~6 m in technical sections (footprint 4.8 × 2 m). No free
   run-off; every line error costs.
6. **Strict mode**: benchmark option where a wall strike voids the lap
   (winner must have 0 strikes).

### How to build it

- Extend `RaceCourseVariant` union + `buildRaceCourse`
  (`race-primitives-scenarios.ts:160-231`); wall list per feature above;
  waypoints re-authored for the circuit (the open-course gates don't fit a
  corridor circuit).
- Expose: `/raceprimitives?course=circuit` (course selector already parses
  `course=technical`); `pnpm run race -- --course=circuit`
  (`demos/scripts/race.ts:56,92`).
- New `demos/test/circuit-course.test.ts` mirroring
  `technical-course.test.ts`: geometry, planner-solvability per feature
  (each corner segment plannable by the v2 library at honest speeds —
  hard, not impossible), wall/obstacle dual-layer consistency.
- Benchmark: extend `technical-course-benchmark.test.ts` pattern into
  `circuit-benchmark.test.ts` with strict mode on.

### Acceptance checklist

- [ ] **A4.1** `buildRaceCourse('circuit')` returns all six features;
  dual-layer consistency test (every wall has matching inflated obstacle
  polygon; inflation = `RACE_WALL_INFLATE`).
- [ ] **A4.2** Solvability: every gate-to-gate segment plannable with the
  v2 library under the expansion cap (deterministic test) — the course is
  hard, not impossible.
- [ ] **A4.3** Demo: `/raceprimitives?course=circuit` renders walls, both
  cars drive, HUD updates, no console errors (manual GUI QA; screenshot
  artifact).
- [ ] **A4.4** CLI: `pnpm run race -- --course=circuit --seed=42 --laps=3`
  completes and prints the standard table incl. wall strikes and
  driving-quality columns.
- [ ] **A4.5** (D5, P2) `circuit-benchmark.test.ts`, deterministic,
  `retry: 0`, strict mode: v2 completes clean (0 strikes); AND
  (`v2.avg < kin.avg` OR kinematic strikes > 0) — the course separates
  the models. Target ratchet: `v2.avg < kin.avg` outright.
- [ ] **A4.6** The kinematic car's failure mode is *visible*: benchmark
  records per-car wall strikes and off-track; PR includes the table.
- [ ] **A4.7** Video walkthrough artifact of the circuit (both cars,
  split viewport) as the PR hero.
- [ ] **A4.8** Open + technical course benchmarks unchanged/green.

---

## 7. WS-5 — Ratchets, honesty rails, and final verification

### What to build

The measurement layer that locks every claim in place.

### Acceptance checklist

- [ ] **A5.1** All new assertions live in deterministic, expansion-capped,
  `retry: 0` benchmarks; no lap-time claim exists without one.
- [ ] **A5.2** Ratchet ledger in the PR: every ratchet constant
  (`V2_VS_KIN_AVG_RATIO`, `V2_VS_KIN_TECH_RATIO`, circuit ratio, lap-time
  bounds, mean-speed floors, g-g floors, fidelity budgets) with
  before → after values; each only ever tightened.
- [ ] **A5.3** Honesty rail asserted in code review: identical controller
  + tuning for both cars in every benchmark; the only per-car difference
  is the forward model (planner primitives, root rollouts, MPPI rollouts,
  commit-window projection, FF controls).
- [ ] **A5.4** `pnpm verify` green (typecheck + full test suite + build +
  size); parking/carchase/ramp invariant suites untouched.
- [ ] **A5.5** Per-workstream benchmark runs recorded:
  `pnpm run race -- --seed=42 --laps=3` on open + technical + circuit
  after each workstream lands; tables in each PR.

---

## 8. Sequencing

1. **WS-0** plant envelope (small, self-contained; parameterizes
   everything after it).
2. **WS-1** speed-policy surgery (both cars get fast; D1; fidelity starts
   being priced).
3. **WS-1½** control feedforward (plan controls reach the actuators;
   speed profile demoted to clamp; MPPI warm-start prior ready).
4. **WS-2** dynamic rollouts (root rollouts → seams → commit-window
   projection; D3; v2's statefulness finally counts).
5. **WS-3** MPPI cost redesign + substepping + gated default flip (D2/D4).
6. **WS-4** circuit course (D5's arena; walls bind now that speeds are
   real).
7. **WS-5** ratchet tightening through 1.0 + final artifacts.

Rationale: speed first (fidelity is priced at zero at 9 m/s), then
model-in-the-loop planning/execution (converts priced fidelity into
commands), then the course (measures it), then the ratchet (locks it).
WS-0 precedes everything because WS-1's bang-bang and WS-4's braking-zone
geometry are both parameterized by the measured envelope.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Faster driving destabilizes pure-pursuit before MPPI is ready | WS-1 raises caps only through envelope-backed braking math; preview keeps binding on genuine corners; wall-strike/off-track ratchets guard every landing |
| Root dynamic rollouts introduce planner nondeterminism or perf cost | rollouts are pure + seedless; ~114 model steps/replan measured budget (A2.5); root-only scope until data says otherwise |
| Commit-window projection re-creates the old divergence bug | projection now uses the model (the old failure was linear sampling, `race-scenario.ts:447-453`); adaptive lateral-drift replan trigger stays as the safety net; A2.3 asserts no off-track increase |
| FF replay drifts (controls exact only from their baked start state) | FF is paired with bounded FB correction (A1½.2 asserts FB stays small AND zero-mean — drift would show as bias); replan-from-live-state every 300 ms bounds accumulation |
| MPPI progress-reward cost cuts corners through walls | corridor penalty in the same cost; circuit solvability test keeps a feasible corridor; A4.5 asserts 0 strikes for the winner |
| Circuit too hard → both cars DNF, benchmark says nothing | A4.2 solvability precondition; corridor widths start generous and tighten with measured clearance |
| Kinematic car improves too (WS-1 helps both) | expected and desired — the claim is relative: v2 wins because its extra speed is placed where the plant can deliver it; if kinematic still wins on the circuit, the benchmark's strike/lap split (A4.6) says whether the course or the model needs iteration |

## 10. Non-goals

- No model retraining or architecture changes in this phase (body-frame
  residuals, dynamic-bicycle backbone, residual-dt alignment remain queued
  behind it — current fidelity is sufficient for the pipeline to reward).
- No new dependencies (no JAX; everything stays in-repo and deterministic).
- No changes to parking/carchase behavior (tuning stays per-scenario;
  every executor change is gated on drive-through plans or race tuning).
