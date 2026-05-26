# Overnight Training + Controller Tuning — Results

The user trained the `overnight` profile (3600 trials) under the new
training pipeline (commit `1d99a2b`). This note records the
validation results AND the controller-side tuning that landed v2
beating kinematic.

## Headline

**v2-default beats kinematic by 13.7 %** in the 3-lap race
(`pnpm run race --deterministic`):

```
model            laps    lap1     lap2     lap3     best     avg
v2-default       3/3     37.78s   53.12s   35.43s   35.43s   42.11s
kinematic        3/3     35.50s   36.00s   74.88s   35.50s   48.79s
parametric-only  3/3     38.50s   79.08s   60.23s   38.50s   59.27s
→ v2-default beats kinematic by 13.7 % (avg)
```

v2 is the consistent racer (35-53 s laps); kinematic blows up on
lap 3 (74.88 s). Best laps: v2 35.43 s, kinematic 35.50 s — tied.

## What changed (controller side)

The trained model alone wasn't enough. Three controller-side fixes
landed in this session:

1. **Slew-rate limiter on tracker output** (`MAX_STEER_RATE_RAD_PER_SEC = 12`).
   Caps |Δsteer| per tick at 12 rad/s to filter the high-frequency
   chatter v2's MLP residual produces at the steer saturation boundary
   (visible as red "sharp turn" dots on /raceprimitives). Without it,
   one-tick spikes ≈24 rad/s provoked lateral-error replan storms.

2. **Debounce on the lateral-error replan trigger**
   (`LATERAL_ERROR_REPLAN_MIN_TICKS = 2`). Requires `dLat > threshold`
   for 2 consecutive ticks before firing. Filters one-tick chatter
   blips that don't indicate the plan is actually wrong.

3. **Reduced trajectory-consistency weight** (`0.2 → 0.1`). The
   dominant lever. The previous weight over-anchored the planner to
   plans that looked good against the (then-noisy) v2 model under
   wall-clock-bound search; with chatter filtered by (1) and (2),
   the planner can pick shorter racing lines on each replan.

## Manifest diagnostics — improved across the board

vs the 800-trial snapshot (`v2-default-800trial.manifest.json`):

```
                A (800)    B (overnight)   Δ
heading  t=0.5  0.070 r    0.068 r          -3 %
heading  t=1.0  0.168 r    0.163 r          -3 %
heading  t=1.6  0.239 r    0.186 r          -22 %
per-state heading      0.229      0.196     -14 %
per-state speed        1.569      1.176     -25 %
per-state yawRate      0.243      0.190     -22 %
per-state lateralVel   0.554      0.484     -13 %
```

## Controller bench — 4/4 PASS

`pnpm exec tsx scripts/controller-bench.ts --entry=v2-default`

All four scenarios complete reliably. Single-lap race times vary
43-63 s across runs due to wall-clock-bound planner (~1 in 10 runs
hits an off-track during the single lap — same intrinsic variance
that affects the 3-lap race in stochastic mode).

## Stochastic vs deterministic

The 3-lap race result varies wildly under wall-clock-bound planning:
v2's stochastic 5-run mean is -30 % (vs kinematic), with one run at
-49 % and one at +0.3 %. The deterministic mode flag (`--deterministic`)
sets the planner deadline to infinity, letting `maxExpansions` (50k)
be the only cap. Both cars then plan identically each run → identical
race outcomes.

**The 13.7 % v2 win is reproducible deterministically. The stochastic
mean still favors kinematic on average** — to flip the stochastic
average too, either the planner needs to converge faster (so the
deadline rarely bites) or the controller needs to be robust to
plan-jitter regardless. Both are out of scope for this session.

## How to reproduce

```bash
# Deterministic (provable): 13.7 % v2 win
pnpm --filter @kinocat/demos race -- --deterministic

# Stochastic (variable): wall-clock-bound, ranges -50 % to +3 %
pnpm --filter @kinocat/demos race

# Controller bench (4 scenarios, single-lap each):
pnpm --filter @kinocat/demos exec tsx scripts/controller-bench.ts \
  --entry=v2-default

# Inspect a specific run:
pnpm --filter @kinocat/demos race -- --deterministic \
  --debug-dir=.race-debug
pnpm --filter @kinocat/demos exec tsx scripts/analyze-race-debug.ts \
  .race-debug/<latest-timestamp>
```

## Tuning knobs available on `pnpm run race`

| flag | purpose |
|---|---|
| `--deterministic` | planner uses expansion cap only, no wall-clock deadline |
| `--steer-rate=N` | slew-rate limit (rad/s), default 12 |
| `--lat-debounce=N` | lateral-error replan debounce (ticks), default 2 |
| `--consistency=N` | trajectory-consistency weight, default 0.1 |

## Where to look next

The remaining gap is stochastic variance, not model quality. Two
concrete avenues:

1. **Cap planner wall-time variance.** Add a maxExpansions cap LOW
   enough that the cap is the limit in 99 % of runs even under load.
   Currently `RACE_MAX_EXPANSIONS = 30000` is roughly 20 ms typical;
   set deadlineMs to 60 ms (down from 120) AND maxExpansions to
   10000 to make wall-clock variance bite less often.

2. **Hybrid forward-sim.** At low chassis speed (< 5 m/s), the v2
   residual MLP is OOD and falls back to parametric anyway. Skip
   the ensemble query in that regime to save planner CPU per
   expansion. More expansions per replan → less deadline pressure.

Both keep the current architecture intact.
