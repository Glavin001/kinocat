# Overnight Training Attempt — Notes

## What was tried

After the default-profile (800-trial) retrain showed the new training
plan beats the old overnight (3600-trial) plan by 6 % on race lap
time (see `docs/training-rework-results.md`), the next natural step
was to retrain at full overnight scale (12 rounds × 300 trials × 180
ticks = 3600 trials) under the new pipeline so the comparison was
apples-to-apples on data volume.

```bash
pnpm --filter @kinocat/demos train:overnight
```

The run was started and aborted after >65 min wall time without
producing the final model file. The process was alive and progressing
(verified via CPU usage) — the bottleneck appears to be the
parametric fit's cumulative-trial loss eval: each subsequent round's
fit pass iterates over the full accumulated trial store, so round 12
re-runs the loss over 3600 trials × Nelder-Mead simplex moves, which
is substantially slower than the per-round time the default profile
exhibits.

## Why I aborted

For a single session with limited wall-clock budget, the overnight
profile's training-time cost outweighed the marginal validation it
would provide. We already had:

- the default-profile (800-trial) model checked in
  (`v2-default.json`) with measured race performance (lap avg 42.86
  s, loses to kinematic by 4.4 %);
- the original overnight-trained baseline preserved as
  `v2-overnight-baseline.json` (lap avg 45.77 s, loses by 37.4 %);
- the same-hardware A/B confirming the new pipeline produces a
  better model with 4.5× fewer trials.

A successful overnight run under the new pipeline is the right
production checkpoint, but it isn't the experiment that validates
the training-plan change — the 800-trial run already did that.

## What you should do

```bash
# When you have a quiet ~60-90 min, run:
pnpm --filter @kinocat/demos train:overnight

# It will overwrite v2-default.json. Verify the new manifest is
# better than the 800-trial snapshot we have:
pnpm exec tsx demos/scripts/compare-manifests.ts \
  demos/public/models/v2-default-800trial.manifest.json \
  demos/public/models/v2-default.manifest.json

# Then race it:
pnpm run race -- --debug-dir=.race-debug

# Expect heading RMS @ t=1 s to drop further (target < 0.10 rad —
# the smoke run at 30 trials hit 0.07 rad, the 800-trial run hit
# 0.17 rad, so 3600 trials should land somewhere between if the
# parametric fit converges fully). Race lap time should match or
# slightly beat kinematic.
```

## Optimisation worth considering before that

The training driver currently fits over the cumulative trial store
every round (Nelder-Mead's loss eval is O(N_trials) per simplex
move, and there are O(120) simplex moves per round). Round 12 of
the overnight profile pays the cost of running the fit over the
full 3600-trial dataset.

Two cheap wins on the training-loop side:

1. **Cap Nelder-Mead iterations on early rounds.** Currently
   `maxIter: ctx.round === 0 ? 200 : 120`. Bring later rounds down
   further (60 or 80) — the parametric usually converges fast once
   the model is close, and the marginal residual on rounds 6–12 of
   overnight is small.

2. **Use only the most recent K trials in the fit** (instead of the
   full store) for intermediate rounds. The final round still gets
   the full store. Tradeoff: less data per round = less precise
   parameter estimates per round, but the cumulative effect should
   wash out.

Both keep the architecture intact and reduce wall time substantially
for the overnight profile without changing the loss / coverage
properties of the trained model.
