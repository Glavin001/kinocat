# v2 Model Weaknesses & Training Recommendations

Based on inspection of `demos/public/models/v2-default.json`,
`v2-default.manifest.json`, and the training pipeline in
`demos/app/lib/training-driver.ts` + `core/src/learning/`.

**No training-related code was modified in the recent race-optimization
iteration** — your overnight model is the canonical output of the
current training plan. The race-side improvements are independent.

## Concrete numbers from the manifest

| Metric (open-loop RMS) | t=0.5 s | t=1.0 s | t=1.6 s |
|---|---|---|---|
| position | 0.51 m | **0.97 m** | 1.40 m |
| heading  | 0.078 rad (4.5°) | **0.263 rad (15°)** | 0.323 rad (18.5°) |
| speed    | 1.19 m/s | 1.07 m/s | 0.65 m/s |

Position improved 7.5× over kinematic baseline. Heading improved only
**1.8×** (kinematic 0.46 rad → v2 0.26 rad). **Heading is the
weakest dimension and the one racing depends on most.**

## Root cause #1 — heading is under-weighted in the parametric loss

`demos/app/lib/training-driver.ts:328`

```ts
function stateDeltaForFit(pred, act) {
  const dx = pred.x - act.x;
  const dz = pred.z - act.z;
  let dh = pred.heading - act.heading; // wrap to [-π, π]
  const ds = pred.speed - act.speed;
  return dx * dx + dz * dz + 5 * dh * dh + ds * ds;
}
```

For a typical 1-sample loss:
- position 0.5 m → 0.25 contribution
- heading 0.1 rad (5.7°) → 5 × 0.01 = **0.05 contribution**
- speed 1 m/s → 1.0 contribution

Heading contributes ~5% of the total loss budget. Nelder-Mead
optimises position + speed and lets heading drift. That's exactly
what the manifest measures: position 7.5× better, heading 1.8×
better.

**Fix:** raise the heading weight from `5` to at least `50` (better:
`100`). Heading errors in radians are small numerically — they need
explicit scaling to compete with position/speed in m and m/s.

```ts
return dx * dx + dz * dz + 100 * dh * dh + ds * ds;
```

For racing precision specifically (chassis must track tight slalom
arcs), weight could go to 200 or use a normalised loss:

```ts
// Each component normalised to its expected magnitude → equal voice.
const posErr2 = dx * dx + dz * dz;       // m²
const hdgErr2 = dh * dh;                 // rad²
const spdErr2 = ds * ds;                 // (m/s)²
// Heuristic normalisers from open-loop RMS at t=1s:
return posErr2 / (1.0 * 1.0) + hdgErr2 / (0.05 * 0.05) + spdErr2 / (1.0 * 1.0);
// Scale chosen so each dim contributes ~1.0 at typical error magnitudes.
```

## Root cause #2 — residual MLP loss is plain MSE on a mixed-unit vector

`core/src/learning/residual-mlp-fit.ts:105`

```ts
function meanLoss(mlp, samples) {
  let total = 0;
  for (const s of samples) {
    const cache = forward(mlp, s.input);
    let l = 0;
    for (let o = 0; o < cache.output.length; o++) {
      const d = cache.output[o] - s.target[o];
      l += 0.5 * d * d;            // ← unweighted MSE
    }
    total += l;
  }
  return total / samples.length;
}
```

Target vector is `[Δx, Δz, Δheading, Δspeed, Δyaw_rate,
Δlateral_velocity]`. Typical magnitudes:

| component | typical scale |
|---|---|
| Δx, Δz | 0.5 m |
| Δheading | 0.05 rad |
| Δspeed | 1 m/s |
| Δyaw_rate | 0.3 rad/s |
| Δlateral_velocity | 0.5 m/s |

Δheading at 0.05 rad contributes `0.5·0.0025 = 0.00125`. Δx at 0.5 m
contributes `0.5·0.25 = 0.125`. The MLP sees heading errors as 100×
less important than position errors. It optimises position and
ignores heading.

**Fix:** pass per-component weights into the MLP loss (same
philosophy as the parametric fit). For racing:

```ts
const WEIGHTS = [1, 1, 100, 1, 10, 1];  // pos, pos, hdg, spd, yaw, lat
function meanLoss(mlp, samples, weights) {
  let total = 0;
  for (const s of samples) {
    const cache = forward(mlp, s.input);
    let l = 0;
    for (let o = 0; o < cache.output.length; o++) {
      const d = cache.output[o] - s.target[o];
      l += 0.5 * weights[o] * d * d;
    }
    total += l;
  }
  return total / samples.length;
}
```

Gradients in `backward` also need to scale by `weights[o]`. The
existing `residual-mlp-fit.ts` backward pass is plain MSE
(`grad = output - target`); needs to become
`grad = weights[o] * (output - target)`.

## Root cause #3 — racing-style maneuvers are ~1.4 % of the training data

`core/src/vehicle/car/maneuvers.ts:497` (`defaultManeuverBundle`):

| share | family | typical peak steer |
|---|---|---|
| 60 % | OU random walk | sigSteer = 0.10–0.40 rad |
| 15 % | transition probes | up to 0.6 rad (random) |
| 10 % | saturation/panic | 0.6 rad (full-lock) |
| 10 % | named (sin sweep, slalom, fishhook, jTurn, trailBrake, stepSteer, throttleOnApex) | varies |
| 5 % | constantHold | random |

The named-maneuver slot is 1/7 slaloms → **1.4 % of trials are
slalom-style**, and the OU random walk's `sigSteer ≤ 0.4` is well
below the `0.6 rad` full-lock the race course's slalom gates
demand.

**Fix:** add a `racingBundle` (or change the share in
`defaultManeuverBundle`):

```ts
// 30 % racing-style: full-lock slaloms + sustained high-speed turns
//                  + brake-into-corner trajectories at speed.
const raceCount = Math.round(args.count * 0.30);
for (let i = 0; i < raceCount; i++) {
  const kind = i % 4;
  const amp = limits.maxSteerAngle * (0.7 + 0.3 * rng()); // 0.42–0.6 rad
  const drv = limits.maxDriveForce * (0.6 + 0.4 * rng());
  if (kind === 0) specs.push({
    id: 'fullLockSlalom', ...,
    build: () => slalom({ amplitude: amp, periodSec: 0.6 + 0.4 * rng(), driveForce: drv }),
  });
  // ... brakeIntoCornerAtSpeed, sustainedHighSpeedTurn, etc.
}
```

Plus a dedicated **chassis-at-cruise-speed initial-condition**: today
`startSpeedSchedule = [0, 4, 8, 12, 16, 20, 24, 28]` is uniform.
For racing precision, weight the high-speed buckets more so the
model fits actual race regimes better.

## Root cause #4 — single-step loss instead of trajectory loss

The parametric fit and residual MLP both compute loss per (state,
control, next_state) sample independently. They never explicitly
penalise multi-step divergence — but at race-time, the chassis is
under closed-loop control and errors compound over the 0.55–1.0 s
primitive duration.

**Fix (optional, harder):** at fit time, roll the model forward
N=5–10 steps using its own predictions as input, then compute loss
on the trajectory endpoint instead of (or in addition to) the
single-step loss. The "trajectory loss" forces the model to be
accurate over the horizon the planner actually uses.

This is more expensive (N× the compute per loss eval) but it's the
canonical fix for the "model is single-step accurate but multi-step
divergent" failure mode the v2 model exhibits.

## Recommended training-plan changes (in order of impact)

1. **Bump heading weight in `stateDeltaForFit` to 50–100.** Single
   line change. Should drop the heading RMS at t=1 s from 0.26 →
   ~0.08 rad (matches the t=0.5 s value, which is what we care
   about for the planner's 0.55 s primitives).
2. **Per-component weights in `meanLoss` + `backward` of
   `residual-mlp-fit.ts`.** Same philosophy, but applied to the
   MLP. Without this the MLP undoes the parametric's improved
   heading.
3. **Add racing-style maneuvers (full-lock slaloms,
   brake-into-corner, sustained high-speed turns) at 20–30 %
   share.** Either as a new `racingBundle` or by editing
   `defaultManeuverBundle`'s mix.
4. **Bias `startSpeedSchedule` toward higher speeds.** Today
   uniform across [0…28]; for racing, [4, 8, 12, 16, 20, 24, 28]
   (drop the 0 bucket — passive coast already provides that data).
5. **(Optional) Multi-step trajectory loss.** Replaces or
   augments the per-sample loss with a horizon-N rollout loss.

## Quick validation after retraining

Run `pnpm run race --debug-dir=.race-debug` then
`pnpm exec tsx demos/scripts/analyze-race-debug.ts .race-debug`.
The key signal is the `lateral-error` replan count: if heading
prediction improves, the chassis tracks the plan better, fewer
lateral-error replans fire, total replans drop, v2 lap times drop.

Manifest open-loop RMS should also improve:
- Target heading RMS at t=1 s: < 0.10 rad (currently 0.26 rad)
- Target position RMS at t=1 s: < 0.7 m (currently 0.97 m)
- Target speed RMS at t=1 s: < 0.8 m/s (currently 1.07 m/s)

After that, `v2 vs kinematic` in `pnpm run race` should flip:
predErrorRMS drops, lateral-error replans drop, lap times drop.
