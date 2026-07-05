# V3: the purely-learned dynamics model

## Why v2 had to be replaced (first-principles audit)

The v2 "learned" model is not purely learned. It is a hand-written
parametric bicycle-style formula (~14 fitted scalar knobs: `engineScale`,
`gripScale`, `yawRateTau`, …) clamped to hand-coded "physical-plausibility"
bounds, plus a residual MLP that learns whatever error is left. That
architecture has a failure class: **any hand-written internal structure is a
place for hidden falsehoods to live.** The audit found two.

### Bug 1 — the engineScale clamp (2× longitudinal error)

The parametric backbone models longitudinal acceleration as
`engineScale × driveForce / mass`, with a hand-set bound
`engineScale ≤ 1.05` justified as *"> 100% means the model amplifies
commanded force — unphysical."* That justification is factually wrong about
the plant: the Rapier adapter applies `engineForce` to **each** driven wheel
(`setWheelEngineForce` per wheel), so the real propulsion force is
`2 × 4000 N` and the true launch acceleration is **13.9 m/s²** (measured by
the WS-0 plant envelope) — while the clamped formula can only express
**7.3 m/s²**. The fitter slammed into the ceiling (the shipped artifact has
`engineScale = 1.05`, exactly at the bound) and the residual MLP silently
absorbed a 2× error on the model's primary longitudinal channel:

| full throttle from rest | 1 s | 2 s | 3 s |
|---|---|---|---|
| plant (measured) | 13.3 | 25.3 | 36.4 m/s |
| v2 parametric backbone | 7.2 | 14.2 | 21.2 m/s |
| v2 backbone + residual MLP | 13.1 | 24.6 | 34.7 m/s |

The "residual" is not a small correction — it is doing half the
longitudinal physics. Worse, v2's OOD safety net **falls back to the
parametric backbone** whenever the residual ensemble disagrees, so exactly
in unfamiliar states the model reverts to believing the car is half as
powerful as it is. This is the concrete mechanism behind "v2 never plans to
its top speed".

(The bound is corrected to 2.2 in `PARAMS_V2_LO/HI` for any future v2
refit, but the architecture lesson stands.)

### Bug 2 — spurious heading input (broken symmetry)

The v2 residual MLP receives `sin(heading), cos(heading)` as inputs. On a
flat uniform plane the plant's dynamics are exactly rotation-invariant —
there is nothing direction-dependent to learn, only direction-dependent
noise to overfit. The symmetry should be guaranteed by construction, not
spent model capacity on.

## V3 design principles

1. **No hand-written dynamics.** The transition function is a neural
   network trained directly on recorded plant transitions. There is no
   parametric backbone, hence no wrong backbone to fall back to.
2. **No hand-set constants.** Every number the model needs — input and
   output normalization — is a statistic computed from the training data.
   There are no parameter bounds because there are no parameters with
   physical names.
3. **Exact symmetries by construction, not assumption.** The model operates
   in the chassis body frame: inputs are `[speed, yawRate, lateralVelocity,
   steer, driveForce, brakeForce]` (no position, no heading); outputs are
   body-frame deltas `[dFwd, dRight, dHeading, dSpeed, dYawRate,
   dLateralVelocity]` rotated into the world frame at integration time.
   Translation/rotation equivariance — a property of the plant, verified
   rather than assumed — is therefore exact.
4. **Learn at the plant's native resolution.** Trials are recorded at the
   physics tick (1/60 s) with `sampleEveryNTicks = 1`, so every training
   pair is one exact plant transition under one constant control vector.
   Nothing is averaged, smoothed, or re-derived. Arbitrary-`dt` queries
   decompose into whole reference steps plus one fractional step.
5. **Data-level regularization only.**
   - *Mirror augmentation:* the chassis is geometrically left/right
     symmetric, so every transition implies its mirror image (negate steer,
     yawRate, lateralVelocity, dRight, dHeading, dYawRate, dLateralVelocity).
   - *Noise injection* (à la graph-network simulators): perturb dynamic-state
     inputs with noise sized to the one-step delta std, teaching the network
     to pull drifting rollouts back to the data manifold.
6. **Uncertainty without a fallback.** A 3-member ensemble provides a
   disagreement signal (`predictWithUncertaintyV3`) for diagnostics or
   planner cost shaping — but inference always uses the ensemble mean.
   There is deliberately no "safe" hand-written model to revert to.

## Implementation map

| Concern | Path |
|---|---|
| Model + inference + persistence | `core/src/agent/vehicle-model-v3.ts` |
| Training (pair extraction, normalization, augmentation, Adam) | `core/src/learning/dynamics-v3-fit.ts` |
| Training CLI (collect → fit → evaluate → artifact) | `demos/scripts/train-v3.ts` |
| Shipped artifact + provenance manifest | `demos/public/models/v3-default.json` (+ `.manifest.json`) |
| Race library builder | `buildLearnedRaceLibraryV3` in `demos/app/lib/race-primitives-scenarios.ts` |
| Race entry | `v3Entry` in `demos/app/lib/headless-race.ts` |
| Race CLI (auto-routes by payload kind) | `demos/scripts/race.ts --models=...,v3-default.json` |
| Machinery tests | `core/test/agent/vehicle-model-v3.test.ts` |
| Plant-fidelity tests | `demos/test/model-vs-plant-fidelity.test.ts` |

Training data: 400 maneuver trials (OU random-walk, transition probes,
saturation/panic, identification, constant holds) × 150 ticks at start
speeds `[0…36] m/s` — 60 000 transitions, doubled by mirroring. Network:
3 × MLP(6 → 64 → 64 → 6), Adam, 60 epochs, deterministic given `--seed`.

## Measured results (2026-07, seed 42)

Endpoint position error after one 0.8 s primitive vs the real Rapier
chassis, across the full probe grid (14 speeds × 4 control probes):

| model | mean | worst |
|---|---|---|
| kinematic | 6.13 m | 22.8 m |
| v2 untrained prior | 2.87 m | 6.58 m |
| v2 trained (backbone + residual) | 1.07 m | 3.33 m |
| **v3 purely learned** | **0.63 m** | **2.27 m** |

Full-throttle from rest (the channel the v2 backbone got 2× wrong):
plant 13.2 / 25.2 / 36.0 m/s at 1/2/3 s → v3 predicts 12.8 / 25.0 / 36.4.
