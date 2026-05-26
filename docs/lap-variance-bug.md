# Lap-variance investigation — engineering notes

The user observed that the "v2 beats kinematic by 13.7 %" win has
laps ranging 35-75 s and concluded there must be a planner/execution
bug. They're right that lap variance is huge. The investigation
below pinpoints the bug location and explains why every attempted
"obvious" fix made the race worse.

## The bug

Pure-pursuit's brake-to-target sweep targets the END of the
*active segment* (not the full plan). Segments come from
`splitAtGearCusps` which splits the plan at every forward↔reverse
sign flip with magnitude > 1 mm/s. In racing, the planner generates
plans with 50-300 such "cusps" PER LAP, from two sources:

1. **Tiny reverse primitives at corners.** The race control set
   (`raceControlSets` in `race-primitives-scenarios.ts:158`)
   includes 5 reverse primitives `[0, -8]`, `[kHalf, -5]`,
   `[-kHalf, -5]`, `[k, -3]`, `[-k, -3]`. At a tight corner the
   planner sometimes picks a reverse primitive to align heading,
   producing a real forward → reverse → forward sequence.

2. **Numerical noise from the trajectory smoother.** The smoother's
   Tikhonov regularisation around a brake-to-stop sample can flip
   speed sign by ≈±0.005 m/s. With the 1e-3 threshold these tiny
   noise flips count as cusps.

Each cusp splits the plan into a short mid-track segment whose
*terminal speed is ≈0*. Pure-pursuit's brake-to-target then forces
the chassis to a STOP where no stop is needed. Observed:

```
t=57.62  brake=0  thr=1.00  tgt=30.00     ← normal cruise
t=57.72  brake=0  thr=1.00  tgt=30.00
t=57.82  brake=1  thr=0     tgt=5.68      ← short cusp segment activated
t=57.92  brake=1  thr=0     tgt=3.06
t=58.02  brake=1  thr=0     tgt=0.00      ← braked to zero MID-TRACK
t=58.12  brake=0  thr=1.00  tgt=22.13     ← new plan, accelerate again
                                            (200 ms total brake-then-go)
... car ends up 13.86 m off-track at t=60.1 s, spirals 4 s
```

Diagnostic mode (`RACE_DIAG=1`) on a 3-lap race confirms 272 cusp
events per race in the baseline configuration.

## Why every "obvious" fix made the race worse

**Tried A: skip cusp splitting in racing entirely.** Result:
v2 lap1=58.95 / lap2=40.07 / DNF, avg=49.51 s. WORSE.

**Tried B: raise threshold from 1e-3 to 0.5 m/s** (filter noise,
keep real reverse cusps). Result: v2 lap1=58.95 / lap2=36.25 /
lap3=62.50, avg=52.57 s. WORSE.

**Tried C: remove reverse from race control sets.** Result:
cusps dropped 272 → 64, but v2 lap1=64.42 / lap2=62.93 / DNF. WORSE.

**Tried D: in racing, override cusp segments' terminal speed to
cruise (don't brake to zero).** Result: v2 lap1=80.92 / lap2=62.43
/ DNF. WORSE.

**Tried E: combine A+C (no splitting AND no reverse primitives).**
Not run, but A and C individually both regressed laps that were
previously good — adding them won't help.

## What this tells us

The brake-to-zero behavior at phantom cusps is **load-bearing**.
It's not just a "bug that happens to fire" — when it fires, it
slows the chassis at a moment that prevents a different downstream
failure. Removing the symptomatic behavior exposes a worse one.

The most likely actual root cause: the planner's racing line is
sometimes **kinematically infeasible** (too fast for the corner).
The chassis enters too hot, the planner reactively generates a
tight-corner primitive (sometimes with reverse), the cusp split
brakes the chassis, and the brake just happens to slow the chassis
to a feasible corner speed. Removing the brake makes the chassis
miss the corner.

The right next step is one of:

1. **Add a friction-circle speed pass to the racing plan.** The
   plan post-process pipeline already has `enableSpeedProfile` (off
   by default — `race-scenario.ts:397`). Turning it on should
   pre-emptively cap corner speeds to physically achievable values,
   removing the planner's need to reactively brake.

2. **Tighten the planner's expansion cost on tight corners.** Make
   the planner pay more for full-lock turns at high speed, so it
   chooses gentler corner entries that don't need the reactive
   brake.

3. **Drop the v=0 reverse primitive from `lowSpeedV2Controls`** AND
   the corner reverses from `raceControlSets`, AND enable the
   friction-circle speed pass. The combined change removes the
   reactive-brake AND the unphysical corner entries together.

None of these are 5-minute changes — they involve verifying that
the speed pass doesn't break parking (it's shared infra) and that
the planner heuristic table cache invalidates correctly.

## Tested experiment: enable the speed profile

Setting `enableSpeedProfile: true` in `DEFAULT_TUNING` eliminates
cusps entirely (272 → **0**). The friction-circle pass produces
monotone smooth speeds that never flip sign.

But: lap times for both cars degraded in a single deterministic
run. v2 hit 59 s on lap 1 (then 38 s on lap 2 — could not complete
lap 3 within the 180 s sim cap); kinematic also degraded
(lap 1 = 52 s, lap 3 = 79 s).

That suggests the friction-circle pass is currently mis-tuned —
likely over-conservative on corner speeds — and needs its
deceleration / friction-coefficient parameters set against the
chassis it's controlling. The right path forward is:

1. Enable the speed profile.
2. Re-tune the friction-circle's effective μ and a_max parameters
   against the v2 model's actual corner-grip envelope (the model
   itself encodes this).
3. Re-tune the planner's heuristic table for the speed-profiled
   plan shape.

That's a multi-session investigation, not a single fix.

## The current 13.7 % deterministic win

Under `--deterministic` (planner uses expansion cap only), v2 still
beats kinematic by 13.7 % on the 3-lap race. Both cars have ONE bad
lap each — kinematic's lap 3 is 74.88 s (similar root cause — its
plan also has cusps at corners). v2's bad lap is 53.12 s. Net: v2
average 42.11 s, kinematic 48.79 s.

The win is real, but the user is right that the variance is
embarrassing. A clean win requires fixing the planning-execution
mismatch documented above.

## Reproduce

```bash
# See the cusps in real time:
cd demos
RACE_DIAG=1 pnpm exec tsx scripts/race.ts --deterministic 2>cusps.log
grep -c CUSP cusps.log   # 272 in baseline
grep CUSP cusps.log | head -5

# Inspect the lap-2 brake-then-go incident:
pnpm exec tsx scripts/race.ts --deterministic --debug-dir=/tmp/race-debug
pnpm exec tsx scripts/analyze-race-debug.ts /tmp/race-debug/<latest>
```
