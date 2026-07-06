---
name: model-vs-plant-fidelity
description: Decide whether a failure is caused by an inaccurate forward model / motion primitives, by planning, or by execution — using deterministic model-vs-plant rollout comparisons on the headless Rapier harness. Use when you suspect "the model is lying" or need to attribute a behavior failure to the right layer.
---

# Model-vs-plant fidelity checks

## Why this exists

The stack has three layers that can each cause the same visible failure
(overshoot, stall, wedge): the PLANNER (searches motion primitives), the
EXECUTOR (pure-pursuit, or MPPI rolling a forward model), and the PLANT
(Rapier raycast vehicle — ground truth). Fixing the wrong layer wastes days.
A 30-line deterministic experiment attributes the failure in minutes.

## The experiment shape

Compare each forward model against the plant on the SAME control trace from
the SAME initial state, integrated the same way the consumer integrates it
(e.g. MPPI: H×0.05 s steps, 3 substeps).

```ts
import { createHeadlessTrialHarness } from 'kinocat/adapters/rapier';
import { DEFAULT_VEHICLE_OPTS } from '../app/lib/training-driver'; // matches race tuning

const harness = await createHeadlessTrialHarness({
  vehicleOptions: DEFAULT_VEHICLE_OPTS, groundFriction: 1.5,
  groundBounds: { x0: -2000, x1: 2000, z0: -2000, z1: 2000 }, offArenaThreshold: 5000,
});
const r = harness.runTrial({
  pose: { x: 0, z: 0, heading: 0 },
  kin: { forwardSpeed: v0 },        // exact for v0=0 (wedge states are at rest!)
  controlsTrace,                     // [{steer, driveForce, brakeForce}, ...] at 1/60 s
  sampleEveryNTicks: 1, id: 'name',
});
// truth = r.trial.samples.at(-1); then roll each ForwardSim over the same
// trace and compare end pose, heading, speed.
```

Models to compare (from `kinocat/agent` / `demos/app/lib`):
`parametricForwardV2(KINEMATIC_NATIVE_PARAMS, ...)` (kinematic),
`learnedForwardSimV2(model)` (v2), `forwardSimV3Rollout(model)` (v3 as MPPI
uses it — single ensemble member).

Pick maneuvers from the FAILURE REGIME, not generic ones: e.g. wedge escapes
= from rest, full steer × {full, half, quarter} throttle, reverse; corner
cases = at speed, full steer, coast/throttle/brake.

## Reading the result (measured example, 1.5 s rollouts)

| Regime | kin err | v2 err | v3 err |
|---|---|---|---|
| rest, full steer + throttle | 5.9 m / 29° | 2.3 m / 19° | 2.6 m / 9° |
| 8 m/s, full steer + half throttle | 7.0 m / 72° | 2.5 m / 18° | 1.0 m / 7° |
| rest, straight, full throttle | 7.3 m (10 vs 19 m/s) | 0.3 m | 0.5 m |

Interpretation rules:
- Learned model errors ≲ 1–3 m over 1.5 s in the failure regime → the model
  is NOT the problem; look at cost shape or planning (see the MPPI skill's
  4-way separation).
- One model grossly wrong in one regime → that regime is missing from the
  training distribution; fix data collection, not the controller.
- If the DELUSIONAL model (kinematic) survives where the honest one fails,
  suspect a cost that rewards delusion (e.g. penalizes honest transient
  dynamics) — not a model problem at all.

## Attribution decision tree

1. **Plant capability**: can ANY control sequence do the required maneuver?
   Check `demos/public/models/plant-envelope.json` (measured launch/brake/
   cornering limits; e.g. full-steer turn radius ≈ 4.7 m ⇒ ~7.5 m/s ceiling
   on that arc). If not physically possible → planning asks too much.
2. **Model honesty** (this skill): model vs plant on the failure-regime traces.
3. **Planner intent**: log raw `res.path` node states. Red flags: 2-node
   straight chords whose direction disagrees with the stored heading
   (collapsed Reeds-Shepp shots hiding curved/reverse geometry — lift with
   `liftAnalyticPath`), interior samples with near-zero speed, kinks at gates.
4. **Executor**: MPPI `onDebug` / pure-pursuit lookahead diagnostics.

Also available: `predErrorRms` in race diagnostics (planned vs actual pose at
primitive boundaries) is a free, always-on closed-loop fidelity signal —
compare entries on it before instrumenting anything.

## Rules

- Wedge/stall states have v≈0 → `kin: { forwardSpeed: 0 }` initializes the
  plant EXACTLY; results are trustworthy. At speed, the harness pre-rolls to
  the entry speed — still deterministic, slightly less exact slip state.
- Integrate the model exactly as its consumer does (same step/substep split);
  integration-scheme mismatch masquerades as model error.
- Keep the whole thing deterministic: fixed traces, no RNG.
