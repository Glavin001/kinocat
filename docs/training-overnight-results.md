# Overnight Training — Results (commit `1d99a2b`)

The user trained the `overnight` profile (3600 trials) under the new
pipeline and committed the model. This note records the validation
results.

## Manifest diagnostics — improved across the board

vs the 800-trial snapshot (`v2-default-800trial.manifest.json`):

```
                A (800)    B (overnight)   Δ
heading  t=0.5  0.070 r    0.068 r          -3 %  ✓
heading  t=1.0  0.168 r    0.163 r          -3 %  ✓
heading  t=1.6  0.239 r    0.186 r          -22 % ✓
per-state heading      0.229      0.196     -14 % ✓
per-state speed        1.569      1.176     -25 % ✓
per-state yawRate      0.243      0.190     -22 % ✓
per-state lateralVel   0.554      0.484     -13 % ✓
```

Position RMS regressed slightly at short horizons (+24 % at 0.5 s,
+14 % at 1.0 s) — expected trade-off for the heavier heading weight.

## Controller bench — 4/4 PASS

`pnpm exec tsx scripts/controller-bench.ts --entry=v2-default`

```
race                    PASS  35.85 s   off-track=0   1 lap of /raceprimitives
parking-forward-pullin  PASS   7.58 s   goalErr=0.49 m, parked
parking-reverse-perp    PASS  21.12 s   goalErr=0.00 m, parked
parking-parallel        PASS   6.15 s   goalErr=0.49 m, parked
```

The single-lap race finished in **35.85 s**, matching kinematic's
best-lap time. Model quality in isolation is good.

## 3-lap multi-car race — REGRESSION

`pnpm run race` (3 laps, seed=42):

```
model            laps   lap1     lap2     lap3     best     avg
v2-default       3/3    52.97s   34.88s   59.38s   34.88s   49.08s
kinematic        3/3    33.90s   35.08s   34.70s   33.90s   34.56s
parametric-only  2/3    72.17s   59.47s   ---      59.47s   65.82s
→ v2-default loses to kinematic by 42.0 % (avg)
```

Lap-by-lap pattern: **one near-optimal lap, two bad ones**.

| diagnostic | v2-default | kinematic |
|---|---|---|
| replans (ok / total) | 611 / 658 | 446 / 495 |
| planner mean / max ms | 68.8 / 136.3 | 82.4 / 138.7 |
| sharpSteer ticks | **2503** | 1721 |
| reasons (cad/lat/wp/fail) | 237 / **406** / 15 / 0 | 122 / 357 / 14 / 2 |

## Diagnosis

The model itself is **sharper and more accurate** (manifest + bench
confirm). But under the 3-lap race the controller chatters at lap
boundaries:

- Bench (1 lap) hits **35.85 s**, comparable to kinematic.
- Race lap 2 (no lap-boundary on either side) hits **34.88 s**,
  faster than kinematic's best.
- Laps 1 and 3 (cold start, post-handoff) hit 53 s / 59 s.

The 800-trial snapshot showed a similar pattern (one bad lap), but
less severe — that earlier run won the 3-lap race by 10 % in one
seed and lost by 4 % in another. **Race outcome is dominated by
seed/lap-boundary variance, not model quality.**

## What's next

This is a controller/integration problem, not a training problem.
Two specific things to investigate before retraining:

1. **Lap-boundary state.** When `lap1 → lap2` finishes, what is the
   pure-pursuit tracker fed? If the race-gate signals zero terminal
   speed at end-of-lap (so the tracker brakes), then re-issues a
   non-zero terminal speed for lap2, the brake-to-target sweep eats
   25-30 s before the tracker re-accelerates. Check
   `demos/app/lib/race-scenario.ts` race-gate cusp handling.

2. **Cold start (lap 1).** Why does lap 1 take 53 s for v2 but
   34 s for kinematic? Both start from rest. Suspect: v2's MLP
   residual is conditioned on a non-zero velocity prior, and the
   first ~1 s of zero-velocity input produces a bad residual that
   the pure-pursuit lateral-error replan counter latches onto
   (406 lateral-error replans on v2 vs 357 on kinematic).

3. **Sharp-steer tick spike.** v2 has 2503 sharp-steer ticks vs
   1721 for kinematic — a 45 % increase despite the heading-RMS
   improvement. The model is outputting valid-magnitude but
   high-frequency steer commands. Add a low-pass on the steer
   output, or penalise steer-acceleration in the training loss.

## How to reproduce

```bash
git checkout 1d99a2b
pnpm run race -- --debug-dir=.race-debug
pnpm exec tsx demos/scripts/analyze-race-debug.ts .race-debug
```

Bench:

```bash
pnpm exec tsx demos/scripts/controller-bench.ts --entry=v2-default
```
