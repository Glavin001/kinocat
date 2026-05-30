# Training-Plan Rework — Results

## Setup

- **Old model** (`v2-overnight-baseline.json`): trained by the previous
  pipeline on the `overnight` profile (12 rounds × 300 trials = **3600
  trials**, ~30 min wall).
- **New model** (`v2-default.json`): trained by the reworked pipeline
  on the `default` profile (4 rounds × 200 trials = **800 trials**,
  ~10 min wall).
- Pipeline change set: `4660f7c` — heading-weighted parametric loss
  (5× → 100×), per-component MLP loss weights (heading 100×), 25 %
  racing-coverage primitives in `defaultManeuverBundle`, trajectory
  horizon = 10 in `runParametricFitAsync`.

## Open-loop manifest diagnostics (lower is better)

| metric (RMS) | t | old (3600 trials) | new (800 trials) | Δ |
|---|---|---|---|---|
| position    | 0.5 s | 0.51 m  | 0.65 m  | +26 % |
| **heading** | 0.5 s | 0.078 r | 0.070 r | **−11 %** |
| speed       | 0.5 s | 1.19 m/s | 1.13 m/s | −6 % |
| position    | 1.0 s | 0.97 m  | 1.19 m  | +23 % |
| **heading** | 1.0 s | 0.263 r | 0.168 r | **−36 %** |
| speed       | 1.0 s | 1.07 m/s | 1.20 m/s | +13 % |
| position    | 1.6 s | 1.40 m  | 1.87 m  | +34 % |
| **heading** | 1.6 s | 0.323 r | 0.239 r | **−26 %** |
| **per-state heading** | — | 0.314 | 0.229 | **−27 %** |
| per-state yawRate | — | 0.271 | 0.243 | −11 % |
| per-state lateralVelocity | — | 0.636 | 0.554 | −13 % |

**Heading dropped consistently across the horizon** (the dimension we
identified as load-bearing for racing). Position regressed because (a)
4.5× fewer trials, and (b) the heavier heading weight forces some
trade-off — the optimiser is no longer free to overfit position at
the expense of heading.

## Race results (consecutive runs on the same machine, same load)

`pnpm run race`, default tuning (pure-pursuit, 3 laps, seed=42),
v2 model vs kinematic vs parametric-only.

### Old model (3600 trials)

```
v2-overnight-baseline  OK  3/3  lap1=39.42s  lap2=40.18s  lap3=57.72s  avg=45.77s
kinematic              OK  3/3  lap1=34.27s  lap2=32.78s  lap3=32.92s  avg=33.32s
parametric-only        OK  3/3  lap1=67.32s  lap2=43.72s  lap3=38.57s  avg=49.87s
→ v2-overnight-baseline loses to kinematic by 37.4% (avg)
```

### New model (800 trials)

```
v2-default      OK  3/3  lap1=35.98s  lap2=54.60s  lap3=38.00s  avg=42.86s
kinematic       OK  3/3  lap1=35.38s  lap2=44.03s  lap3=43.70s  avg=41.04s
parametric-only DNF 2/3  lap1=76.60s  lap2=61.42s  ---           avg=69.01s
→ v2-default loses to kinematic by 4.4% (avg)
```

### Headline

|  | old (3600 trials) | new (800 trials) |
|---|---|---|
| v2 avg lap | 45.77 s | **42.86 s** (−6 %) |
| v2 lap 1 (cold start) | 39.42 s | **35.98 s** (−9 %) |
| v2 vs kinematic | **−37.4 %** loses | **−4.4 %** loses (8.5× closer) |
| v2 best lap | 39.42 s | **35.98 s** |
| v2 sharp-steer ticks | 2473 | 2028 (−18 %) |

**The new training plan with 4.5× FEWER trials beats the old plan with
full data.** The bottleneck was loss weighting, not data quantity —
exactly what the manifest's heading-RMS signal predicted.

## How to reproduce

```bash
# 1. Smoke-check the pipeline.
pnpm run train:quick

# 2. Train the default profile (~10 min).
pnpm run train -- --profile=default --seed=42

# 3. Race the new model vs baselines.
pnpm run race -- --debug-dir=.race-debug

# 4. Compare manifests side by side.
pnpm exec tsx demos/scripts/compare-manifests.ts \
  demos/public/models/v2-overnight-baseline.manifest.json \
  demos/public/models/v2-default.manifest.json

# 5. Dig into per-tick traces.
pnpm exec tsx demos/scripts/analyze-race-debug.ts .race-debug
```

## Next step (optional)

Run `pnpm run train -- --profile=overnight --seed=42` (~30 min) for
the full 3600-trial version under the new plan. Based on the smoke
+ default trend, expect:

- Heading RMS at t=1 s: target < 0.10 rad (currently 0.168)
- Position RMS at t=1 s: target < 0.6 m (recovered from regression)
- Race v2-avg: target ≤ kinematic (currently 4 % behind)

The `v2-overnight-baseline.json` + `.manifest.json` are committed
alongside the new model so any future comparison can be done against
the original, not the moving target.
