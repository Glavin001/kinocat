---
name: headless-race-debugging
description: Debug racing/driving behavior (slow laps, wedges, recoveries, oscillation) using headless scenario runs, speed-coloured trajectory plots, and pre-failure ring-buffer dumps. Use when a car behaves badly in a race/parking scenario and you need to see WHAT it did before theorizing WHY.
---

# Headless race debugging

## Core principle

Never tune knobs against a single aggregate number (lap time, recovery count).
First make the failure *visible*, then localize it, then explain it with a
targeted probe. Aggregate → picture → timeline → solver internals.

## Step 1 — Run headless, get aggregates

`createRaceScenario` (from `demos/app/lib/race-scenario.ts`) is the single
source of truth for both the web page and CLI. A minimal single-car harness:

```ts
const scenario = await createRaceScenario({
  entries: [entry],            // kinematicEntry() / v2Entry() / v3Entry() from headless-race.ts
  targetLaps: 2,
  course: buildRaceCourse('open'),
  tuning: { plannerBudgetMs: 10_000, tracker: 'mpc', mpcOverrides: { /* knob overrides */ } },
});
while (scenario.simTime() < maxSec) { if (scenario.tick().allFinished) break; }
```

- `plannerBudgetMs: 10_000` removes planner wall-clock pressure so you measure
  the controller, not the CPU (sim time is decoupled from wall time headlessly).
- `tuning.mpcOverrides` (spread last into the MPC config) is the sweep escape
  hatch — A/B one knob per run.
- Key aggregates: `laps`, `quality.recoveryCount`, `quality.timeStopped`,
  `quality.meanSpeed`, `offTrackEvents`, `diagnostics.mpcSolveMsAvg`.

## Step 2 — Plot the trajectory (ALWAYS do this before tuning)

`demos/scripts/lib/trajectory-plot.ts` renders a top-down PNG: speed-coloured
executed path (blue→red), gates with arrive disks, walls, spawn, timestamps
every 5 s, red ✕ per recovery, purple ○ per stop episode, grey committed-plan
overlays. Wire it with `TrajectoryRecorder`:

```ts
const rec = new TrajectoryRecorder();
// per tick: rec.record(r.simTime, r.cars[0]!);
rec.save('/tmp/plots/run.png', { bounds: course.bounds, waypoints: course.waypoints,
  walls: course.walls, spawn: course.spawn, arriveRadius: 2.5 }, 'title', 30);
```

Read the PNG directly. What to look for:
- **Recovery markers clustered at specific gates** → the problem is that gate's
  geometry/approach, not a global tuning issue.
- **Loops after a gate** → overshoot + turn-around replans.
- **Color: line stays blue/cyan on straights** → speed is being capped
  (envelope / plan-speed caps), not a controller failure.
- **Executed line cutting far inside the grey plan overlays** → corridor too
  wide, car legally missing arrive disks.

## Step 3 — Ring-buffer timeline before each failure event

Don't log everything; keep a rolling window (~6 s at 4 Hz) of
`t, pos, heading, v, steer, thr, brk, plan segment structure, planAge`, and
dump it when the failure trigger fires (e.g. `quality.recoveryCount`
increments). See the pattern in a `tmp-wedge*.mts` script. This shows the
causal sequence (e.g. "arrived hot → braked to 0 misaligned → replan produced
reverse shunt → shunt never executed → blind recovery").

Useful one-line plan summary: split with `splitAtGearCusps(plan)` and print
`"16F/8R/78F"` (samples + gear per segment) — instantly shows reverse legs.

## Step 4 — Explain with a targeted probe

Once localized, use the matching deep probe:
- Controller decision quality → MPPI `onDebug` hook (see the
  `mppi-diagnosis-and-tuning` skill).
- "Can the chassis even do this?" → model-vs-plant harness (see the
  `model-vs-plant-fidelity` skill).
- Planner output quality → log `res.path` raw node states + `res.stats`
  at replan time; check for analytic-shot chords (2-node straight jumps whose
  heading disagrees with the chord direction — these hide curved/reverse
  Reeds-Shepp geometry).

## Known failure signatures (measured in this repo)

| Symptom | Likely cause |
|---|---|
| Stops + recoveries at gates, plan has `NR` reverse segments | Cusp shunt not executing (see MPPI skill: softmax dilution) |
| Whole field slow, riding a constant speed | vAllow envelope cap (check `buildProgressGeometry` output along the plan) |
| Anchor/progress jumps, car "parks" while cost says fine | Projection window grabbed a later plan leg (anchor teleport) |
| Only the LEARNED cars fail, kinematic sails | Cost taxes honest transient dynamics (e.g. per-step heading cost vs real yaw lag) |
| planAge grows while stopped | Replan cadence gate thinks the plan is still serving (`dLat < 0.25` vs the active segment) |

## Rules of thumb

- One knob per run; keep a `tmp-sweep.mts`-style CLI (`entry`, JSON overrides,
  budget, plot path) so every observation is reproducible.
- 90–120 s sim budget is enough to see 1–2 laps; don't run 240 s sweeps while
  iterating.
- The stuck-recovery (reverse blindly ~1 s, wipe plan) is itself a chaos
  source: any fix that makes the car dwell near 0 m/s for >1.5 s will trigger
  it and confound your A/B.
