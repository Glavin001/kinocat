# Racing at the limit: making model fidelity reach the wheels

*Plan for the follow-up PR. Goal: both cars drive as fast as they are
physically capable — full throttle on straights, braking only as late as
their model says is safe — so that the v2 learned model's superior knowledge
of the plant converts into lap time, and the kinematic model's delusions
finally cost it.*

*Status: PLAN — reviewed against the code as of this branch. Every file:line
reference verified.*

---

## 1. The two problems, precisely

### Problem A — both cars drive far below the vehicle's capability

Measured on the deterministic 2-lap benchmark (open course): mean executed
speed is **8.8–9.3 m/s** against a plant ceiling of **30 m/s**; g-g
(friction-circle) utilization is **~0.44 mean** for both cars. The chassis
reaches ~27 m/s only on the single long straight. There is roughly **2× pace**
left on the table, for both libraries.

This is not the planner's choice — it is the executor. `purePursuit`
(`core/src/execute/pure-pursuit.ts:245-253`) sets the commanded speed to the
**minimum of five caps**, all of them model-agnostic:

| Cap | Formula / source | Why it keeps the car slow |
|---|---|---|
| `cruiseSpeed` | 30 (agent max) | fine |
| `vCurve` | `√(maxLateralAccel/κ)` of the *pursuit chord* (line 149) | κ = 2y/Ld² conflates cross-track error with path curvature: any tracking offset reads as a corner → phantom braking on straights |
| `vPreview` | Menger curvature of upcoming plan geometry at **0.8·µg** budget (line 163-205, `race-scenario.ts:162-164`) | correct idea, conservative budget; replanned chord paths carry curvature noise → spurious slowdowns |
| `vPath` | plan per-sample speeds through the braking envelope (line 207-235) | **off on the open course** (`respectPathSpeed: false`) — the one cap that carries model knowledge is disabled |
| `vGoal` | `√(2·maxDecel·(distToGoal−tol))` (line 150-152) | **brakes toward the end of every plan** — but the "end" is just the 2-gate replanning horizon (`PLAN_LOOKAHEAD_COUNT = 2`, `race-scenario.ts:88`), not a real stop. The car is permanently decelerating toward a phantom finish ~20-40 m ahead that the next replan will extend |
| throttle law | `throttle = Δv / maxAccel` (line 260-272) | proportional controller: at 2 m/s deficit it commands 33% throttle — asymptotic, never-quite-cruise approach instead of full-throttle-until-there |

The binding constraints in practice are `vGoal` (phantom horizon braking),
`vCurve` (phantom corner braking from tracking error), and the throttle
P-law. None of them consult a dynamics model, so **the same speed policy
executes both cars' plans** — which is why measured g-g utilization is
identical (0.445 vs 0.435) and the lap-time difference reduces to line
length.

### Problem B — v2's cornering knowledge never reaches the wheels

The v2 planner's plans already contain honest, model-derived speeds (the
primitive endpoints carry the speed the model predicts after each
brake-into-corner/accelerate action). The rich `Plan`
(`kinocat/plan`, built at `race-scenario.ts` `buildPlan(smoothed, …)`)
additionally carries per-point curvature `kappa`, feedforward steer
`steerFf`, and target accel — and is **produce-but-don't-consume**: the
tracker re-derives steering from chord geometry and speed from the caps
above. The model that planned is not the model that executes; the executor
is the same crude proxy for both cars.

Consequence, measured: v2's open-loop prediction advantage vs the plant is
**~9×** (0.68 m vs 6.1 m mean endpoint error at 0.8 s, and the brake-in-turn
channel is now 10× better after the grip-saturating brake fix), yet its
closed-loop lap is 19% *slower*. Faster driving makes this worse for the
kinematic car — at 25+ m/s its model error explodes (22.8 m worst-case at
28 m/s vs v2's 1.5 m) — **but only if the executor actually drives at those
speeds and trusts the model to pick them.**

---

## 2. Design: model-in-the-loop racing executor (MPPI), with a feedforward fallback

Two tracks, both landing against the deterministic benchmark. MPPI is the
destination (the user's instinct is right); feedforward pure-pursuit is the
cheap insurance that also fixes Problem A for both cars.

### Track 1 — MPPI as the racing tracker (the real fix)

MPPI (`core/src/execute/mpc-tracker.ts`) rolls K=64 sampled control
sequences through a forward model every tick and emits the
importance-weighted average. **The per-car model wiring already landed in
this PR** (`RaceEntry.forwardModel`, `race-scenario.ts`): the v2 car rolls
its trained `learnedForwardSimV2`, the kinematic car rolls the naive
idealized-bicycle `KINEMATIC_NATIVE_PARAMS`. With MPPI, "how fast can this
corner be taken" is answered by the model itself at 20 Hz — the delusional
model over-drives and pays; the accurate model rides the true limit. That is
the honest head-to-head the project wants.

What's broken today (measured: both cars DNF under `tracker: 'mpc'`):

1. **Progress starvation.** The cost tracks a reference advanced along the
   plan at `cruiseSpeed` (`buildReference`, mpc-tracker.ts:156-216), but
   `wSpeed`-vs-`wLateral` balance plus the phantom-horizon issue (same as
   Problem A) means samples that push pace don't win. Fix by replacing the
   position-tracking cost with an explicit **progress reward**: cost −=
   `wProgress · (arc-length advanced along the plan)`, plus a lateral
   corridor penalty (stay within corridor half-width, e.g. 3 m) instead of
   exact position tracking. This is the standard racing-MPPI cost (Williams
   et al. 2016 drive aggressive laps with exactly this shape).
2. **Phantom horizon braking** — same `vGoal` disease via the reference
   speed. With a progress reward the horizon end is *attractive* rather than
   a stop target; additionally extend the reference by extrapolating the
   plan's last segment when the plan is shorter than the MPPI horizon.
3. **Launch dither** — brake-noise sampling against the near-binary brake
   stalls launches. Already fixed on this branch (`brakeStd` default
   0.10→0.03 of max force after measuring wheel lockup at 5% force). Keep
   the race config's explicit `brakeStd` aligned.
4. **Model-plant dt mismatch** — MPPI steps 0.05 s; the plant integrates at
   1/240 s (4 substeps of 1/60). Integrate each MPPI step as 3 × 1/60 model
   substeps (the parametric model is ~200 ns/step; 64 × 10 × 3 = 1920
   steps/tick is still <1 ms).
5. **Determinism** — LCG-seeded sampling is already deterministic
   (`createMPCTrackerState(horizon, seed)`, reset on `reset()` since this
   branch). Keep `retry: 0` and expansion-capped planning in the benchmark;
   add a determinism test in MPC mode (two runs, identical trajectories).

Acceptance (all deterministic, `retry: 0`):

- `mpc` tracker completes 2 laps on open AND technical courses, both cars,
  0 off-track. (New: `mppi-race-completion.test.ts`.)
- Mean executed speed ≥ 1.5× the pure-pursuit baseline on the open course;
  g-g mean utilization ≥ 0.6 for the v2 car.
- **The headline ratchet: v2+own-model MPPI beats kinematic+naive-model
  MPPI on lap time on the technical course** (`v2.avg < kin.avg`, ratio
  < 1.0) — and the kinematic car's wall strikes / off-track exceed v2's at
  matched speed. Tighten `V2_VS_KIN_TECH_RATIO` accordingly.
- Parking invariants untouched (MPPI stays opt-in for parking until tuned
  there; `tracker` remains per-scenario tuning).

### Track 2 — fix the pure-pursuit speed policy (cheap, benefits both cars, keeps the ablation honest)

Even with MPPI landed, pure-pursuit remains the default for other scenarios
and the "simple executor" baseline. Three surgical fixes to Problem A:

1. **Kill phantom horizon braking**: only apply `vGoal` when the plan's
   terminal sample actually stops (`stopsAtEnd`, already computed at
   line 57) or the goal is the course's true finish. Drive-through horizons
   get `vGoal = ∞`. This alone should raise straightaway speed
   substantially.
2. **Separate feedback error from path curvature**: use the plan's own
   curvature (`Plan.kappa`, already built) for `vCurve` instead of the
   pursuit chord κ; keep chord κ only for the steering feedback term. This
   removes phantom corner braking on straights (and is the first real
   consumer of the rich Plan).
3. **Full-throttle-until-there**: replace the P-law with a
   braking-envelope bang-bang: full throttle while
   `v < min(allCaps) − margin`, brake only when the braking envelope to the
   next binding cap requires it. (This is what a driver does.)
4. **`respectPathSpeed: true` with plan speeds as a CAP not a target**, so
   the v2 plan's honest corner speeds bind where the geometry caps don't —
   this is the cheap path for v2's knowledge to reach the wheels under
   pure-pursuit. (The braking-envelope semantics already exist,
   line 207-235; the blocker was the open course's tuning default.)

Acceptance: open-course benchmark — both cars' mean speed up ≥ 40%, laps
faster than today's 33.0/39.2 s, 0 off-track, determinism holds, parking
invariants green (changes gated to drive-through plans, `stopsAtEnd`
untouched).

### Sequencing

1. Track 2 items 1–3 (fast, de-risks everything; immediately raises pace so
   fidelity starts mattering — measure the fidelity-vs-pace claim directly).
2. Track 1 MPPI cost redesign + substepping + completion tests.
3. Flip the technical-course default tracker to `mpc` once its benchmark
   beats pure-pursuit's lap times there; keep pure-pursuit for parking.
4. Ratchets: tighten `V2_VS_KIN_AVG_RATIO` (open) and
   `V2_VS_KIN_TECH_RATIO` (technical) to the measured ratios; flip the
   technical ratio through 1.0 when v2 wins (the "v2 strictly faster"
   contract).

### Non-goals of the follow-up PR

- No retraining/model-architecture changes (body-frame residuals, dynamic
  bicycle backbone, residual-dt alignment stay separate — the current model
  is already accurate enough for its advantage to show once consumed).
- No new course geometry (the technical course exists; wall density can be
  tuned after MPPI lands).
- No planner changes beyond what the executor needs (the lattice and
  multi-goal search stay as-is).

---

## 3. Why this is the right order (the user's own observation, formalized)

> "the faster you go, that justification grows" — exactly. Model error grows
> superlinearly with speed (kinematic worst-case endpoint error: 0.4 m at
> 4 m/s → 22.8 m at 28 m/s; v2-trained: 0.1 → 1.5 m). Today's executor pins
> both cars at ~9 m/s mean, where even the kinematic model's error (~1 m per
> primitive) is smaller than a car length — fidelity is priced at zero.
> Raising executed speed toward 20+ m/s multiplies the error gap ~15×, and
> model-in-the-loop execution (MPPI) is the mechanism that converts that gap
> into lap time and wall strikes. Speed first, then model-in-the-loop, then
> ratchet.

## 4. Risk register

| Risk | Mitigation |
|---|---|
| MPPI cost redesign destabilizes parking | tracker stays per-scenario; parking keeps pure-pursuit until its own bench passes |
| Faster pure-pursuit breaks the technical course (more wall strikes) | technical tuning already runs the speed profile; recovery maneuver (landed) bounds the damage; wall-strike ratchet in the benchmark |
| Chaotic sensitivity of lap times to tuning | all claims measured on the deterministic expansion-capped benchmark, `retry: 0`; quality metrics (g-g, dist/lap) are the stable secondary signals |
| MPPI per-tick cost (64×10×3 model steps) | measured headroom (~1 ms); reduce K or horizon if the 60 Hz frame budget complains |
