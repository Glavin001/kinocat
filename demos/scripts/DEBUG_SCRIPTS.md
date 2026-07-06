# Racing / MPPI debug & diagnostic scripts

Throwaway-but-kept scripts used to develop and debug the WS-3 MPPI racing
work. They are **not** part of CI and are deliberately named `tmp-*.mts` so
they never get picked up by test globs. They all run headlessly (no dev
server, no GPU) via `npx tsx`.

Run everything from the `demos/` directory. Models are read from
`demos/public/models/{v2-default,v3-default}.json`.

See also the reusable Cursor skills that explain the *methodology* behind
these scripts:
- `.cursor/skills/headless-race-debugging/SKILL.md`
- `.cursor/skills/mppi-diagnosis-and-tuning/SKILL.md`
- `.cursor/skills/model-vs-plant-fidelity/SKILL.md`

## The current, load-bearing kit

### `tmp-sweep.mts` — one-knob-per-run tuning + trajectory plot
```
npx tsx scripts/tmp-sweep.mts <kin|v2|v3> '<json mpcOverrides>' [maxSec] [open|technical] [plotPath.png]
```
Runs a single car for 2 laps under MPPI, printing `laps / lap times / recov /
stopped / meanSpd / offTrack`, and (if `plotPath` given) writes a top-down
speed-coloured trajectory PNG via `scripts/lib/trajectory-plot.ts`. The JSON
arg is spread into the MPC config (`tuning.mpcOverrides`), so you can A/B any
MPPI knob without editing code, e.g.:
```
npx tsx scripts/tmp-sweep.mts v3 '{"corridorHalfWidth":1.2,"lambda":1.5}' 120 open /tmp/v3.png
```

### `tmp-solve-probe.mts` — sample-level MPPI dissection at wedges
```
npx tsx scripts/tmp-solve-probe.mts <kin|v2|v3> [maxSec]
```
Attaches the tracker's `onDebug` hook (see `mpc-tracker.ts` `MPCDebugInfo`).
When the car wedges (v≈0 for >1.2 s) it dumps: cost quantiles across the K
samples, the top-3 samples' first control + horizon-mean drive/brake, the
emitted command, `bestWeightShare` (softmax concentration), the plan geometry,
and — crucially — `scoreSequence(...)` of hand-built maneuvers scored under the
*exact* solve cost. This is the tool that separates cost-wrong vs
sampler-can't-find vs softmax-dilution vs model-wrong. It also flips on the
`__replanLog` trace (see below) around each wedge.

### `tmp-model-vs-plant.mts` — is the model lying?
```
npx tsx scripts/tmp-model-vs-plant.mts
```
From a set of failure-regime maneuvers (from rest: full-steer × throttle
levels, reverse; at 8 m/s: coast / throttle / brake), rolls each MPPI forward
model (kin / v2 / v3) for 1.5 s the exact way the controller integrates it and
compares the endpoint against the **real Rapier plant** (`createHeadlessTrialHarness`).
Prints per-model position/heading error. Answers "is the failure the model's
fidelity, or the planner/executor?" — measured result: v3 ≲ 1 m over 1.5 s in
every regime; the kinematic model is the least faithful.

### `tmp-mppi-single.mts` — per-second closed-loop trace (+ optional plot)
```
npx tsx scripts/tmp-mppi-single.mts [open|technical] [kin|v2|v3]
```
One line per simulated second: pos, speed, waypoint, live controls, plan
length, the allowed-speed the progress cost would build here + min over the
next 30 m, single-gear segment structure (`16F/8R/78F`), and MPPI solve ms.
Good first look at "where does the lap time go".

### `tmp-wedge3.mts` — pre-recovery ring-buffer timeline
```
npx tsx scripts/tmp-wedge3.mts <kin|v2|v3> [maxSec] [maxDumps]
```
Keeps a rolling ~6 s window of state+controls+segment-structure and dumps it
whenever a stuck-recovery fires. Shows the causal sequence into each wedge
(arrive hot → brake to 0 misaligned → reverse shunt → …).

### `tmp-plan-plot.mts` — planning-vs-execution separation + replan churn
```
npx tsx scripts/tmp-plan-plot.mts <kin|v2|v3> [maxSec] [open|technical] <outPrefix>
```
Runs a race under MPPI and emits two artifacts plus a churn report, to answer
"planning error or execution error?" without eyeballing:
- `<out>-plans.png` — every committed plan overlaid, coloured by replan order
  (blue = early → red = late), waypoint-advance replans drawn bright. Plans
  stacked on one line = the planner commits/agrees; a fan of colours = thrash.
- `<out>-exec.png` — executed trajectory (speed-coloured) + committed plans in
  grey; the gap between grey plan and coloured drive is execution error.
- console: per-replan churn (mean/max lateral gap vs the previous plan over the
  near-chassis overlap), split by same-waypoint vs waypoint-advance replans,
  and the 8 largest-churn replans. Measured finding: churn is present in BOTH
  the clean kinematic car and the wedging learned cars (≈1 m mean, ~1.5× higher
  on waypoint advances) — so churn is NOT the differentiator; the honest models
  wedge because they correctly brake at infeasible slalom kinks. The same
  churn metric is surfaced in `DrivingQuality.planChurnMean/Max` for asserting
  plan stability in benchmarks.

### `tmp-wedge2.mts` — single-wedge dissection with raw planner path
```
npx tsx scripts/tmp-wedge2.mts <v2|v3>
```
Captures a wedge and prints the committed plan, the raw planner node path
(reveals collapsed Reeds-Shepp analytic chords), segment gears, and scores a
handful of escape maneuvers under `scoreRolloutProgress`.

## Earlier exploratory scripts (still runnable, mostly superseded)

- `tmp-mppi-baseline.mts` — first headless MPPI baseline harness (tracker/course selectable).
- `tmp-mppi-probe.mts` — probe `mpcTrack` in isolation on a straight plan (launch behaviour).
- `tmp-cost-debug.mts` — `scoreRolloutProgress` on full-throttle vs zero-throttle rollouts.
- `tmp-sample-debug.mts` — replicate the sampling loop to inspect the per-sample cost distribution.
- `tmp-wedge-debug.mts` — the kinematic-model precursor to `tmp-wedge2.mts`.

## Shared hooks these rely on (committed, in the library)

- `mpc-tracker.ts` → `MPCTrackerConfig.onDebug` + `MPCDebugInfo.scoreSequence` — the permanent MPPI introspection API.
- `race-scenario.ts` → `globalThis.__replanLog = true` — opt-in, inert-by-default replan/replan-reason trace.
- `race-scenario.ts` → `tuning.mpcOverrides` — raw MPC-config override bag for sweeps.
- `scripts/lib/trajectory-plot.ts` — the top-down speed-coloured PNG plotter (`TrajectoryRecorder` + `plotTrajectory`).
