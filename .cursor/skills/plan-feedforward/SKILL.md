---
name: plan-feedforward
description: Decide whether a plan is faithful enough to execute its OWN controls as feedforward, then wire and measure control feedforward through the MPPI tracker. Use when a car plans a good line but executes it poorly (overshoots corners, wedges, re-derives worse controls than the plan already found), or when you want a model's fidelity advantage to actually show up in the driven line.
---

# Plan control feedforward

## The idea in one sentence

If the plan's own controls (the constant `[steer, driveForce, brakeForce]` of
each motion primitive) roll open-loop close to the real plant, then the
executor should *drive those controls* as its feedforward baseline instead of
re-deriving them from geometry every tick — feedback only corrects
disturbances. This is the mechanism by which a more faithful model (v3) turns
into a better line.

## Step 1 — is the plan even worth feedforwarding? (open-loop fidelity)

Do NOT wire feedforward off a hunch. First prove the plan is faithful with
`demos/scripts/plan-vs-plant.mts <v2|v3>`: it takes the planner's exact
primitive controls and rolls them open-loop through (a) the planning model and
(b) the real Rapier plant, and reports position/speed divergence.

- **Small divergence** (v3: ~2.3 m over a 42 m maneuver, plant slightly faster
  than predicted) ⇒ the plan is feasible; a wedge is a CONTROLLER problem →
  feedforward is justified.
- **Large divergence** (v2: ~47 m; model under-predicts accel by ~11 m/s) ⇒ the
  model rounded up its capability; feedforward would command controls the plant
  doesn't obey. Fix the model/library first (see `model-vs-plant-fidelity`).

This gate matters: feedforward from an inaccurate model can *hurt*. The A/B in
step 3 confirms it — feedforward helped v3 far more than v2.

## Step 2 — how it is wired (already in the tree, default OFF)

Three pieces, all behind `tuning.controlFeedforward` (default false, so pinned
benchmarks are untouched):

1. **Capture** — `attachPlanFeedforward(smoothed, res, lib)` in
   `demos/app/lib/race-scenario.ts`. Each planner drive edge (`kind` ∈
   `{'drive','drive-reverse'}`) carries a `primId`; that primitive's `controls`
   are the feedforward command. They're resampled by arc-length (piecewise
   HOLD, not lerp — you can't interpolate a throttle primitive into a brake
   one) onto the dense smoothed samples and stored on each sample's `ff` field.
   Reeds-Shepp analytic edges contribute none.
2. **Carry** — `ff` lives on the `CarKinematicState` sample objects, so
   `splitAtGearCusps`'s `slice()` preserves it into each single-gear segment for
   free.
3. **Consume** — `MPCTrackerConfig.useFeedforward` (progress mode). After the
   progress anchor is found, the warm-start prior is seeded from `plan[idx].ff`
   walked along the horizon arc at the plan's projected speed; steps whose arc
   has no `ff` keep the shifted warm-start value. See `mpcTrack` in
   `core/src/execute/mpc-tracker.ts`.

Gotcha that WILL bite: reverse legs use the **`'drive-reverse'`** edge kind, not
`'drive'`. Capture must accept both or reverse shunts silently lose feedforward.
`demos/test/skills/feedforward.test.ts` guards this contract on a real plan.

## Step 3 — measure it (closed-loop A/B)

`demos/scripts/feedforward-compare.mts <kin|v2|v3> [secs] [open|technical] [budgetMs]`
runs the SAME segment twice (OFF vs ON) and prints per-metric deltas with ✓/✗.
Run with `KINOCAT_GEN_CONTROLS=1 KINOCAT_ANALYTIC_DT=1` (correctness branch).

The load-bearing metric is **`predErrRms`** (planned-vs-actual pose at primitive
boundaries — an always-on closed-loop fidelity signal). Feedforward should drop
it; if it doesn't, the plan wasn't faithful (revisit step 1) or the prior is
being washed out (raise the softmax concentration / lower λ, or check the arc
walk speed).

Measured (v3, open course, generous budget): `predErrRms` −39% (0.66→0.40 m),
best lap −26% (56→41 s), `timeStopped` −40%, `planChurnMean` −26%. v2 (the
inaccurate model) improved far less — exactly the step-1 prediction. Report the
caveats too: v3's `recoveryCount` rose (1→3) even as total stopped time fell.

## Rules

- Feedforward is a *baseline*, not a *command*: it seeds the MPPI prior, and the
  progress/overspeed/corridor costs still temper it in the softmax average. A
  brake-hold `ff` must NOT launch (guarded in `mpc-tracker.test.ts`).
- Keep it behind the flag and re-pin benchmarks; never let a feedforward change
  move the default (flag-off) numbers.
- Prove fidelity (step 1) before wiring (step 2) before measuring (step 3). The
  order is the whole point — it stops you feedforwarding a lie.
