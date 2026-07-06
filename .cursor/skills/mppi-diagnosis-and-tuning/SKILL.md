---
name: mppi-diagnosis-and-tuning
description: Diagnose and tune the MPPI tracker (mpc-tracker.ts) — cost design, temperature/λ scale matching, sampling shape, warm-start traps, and the onDebug/scoreSequence introspection workflow. Use when MPPI emits wrong/weak controls, stalls, oscillates, or a cost change behaves unexpectedly.
---

# MPPI diagnosis and tuning

Implementation: `core/src/execute/mpc-tracker.ts` (`mpcTrack`). Two cost
modes: `'track'` (reference tracking + terminal pose — parking) and
`'progress'` (arc-length reward + corridor + braking-envelope overspeed —
racing). Scenario wiring + weights: `MPC_CONFIG` in
`demos/app/lib/race-scenario.ts`.

## The one diagnostic that answers most questions

Pass `onDebug` via `tuning.mpcOverrides` — it exposes per-solve internals with
zero cost when unset:

```ts
mpcOverrides: { onDebug: (info) => { lastInfo = info; } }
// info.costs (all K sample costs), info.samples (K×H×3 controls),
// info.emitted (first command), info.bestWeightShare (softmax concentration),
// info.anchor (progress projection), info.gear,
// info.scoreSequence(controls) — score ANY hand-built control sequence under
//   THIS solve's exact geometry/weights/model.
```

Standard dissection at a bad moment: print cost quantiles, the top-3 samples'
first controls + horizon-mean drive/brake, the emitted command, and
`scoreSequence` of hand-built maneuvers (hold-brake, full/half/quarter
throttle × left/straight/right). This separates four distinct failures:

1. **Cost is wrong** — hand-built "obviously right" maneuver scores WORSE than
   hold-still. Fix the cost, not the sampler.
2. **Sampler can't find it** — good maneuver scores well but no sample is near
   it. Fix noise shape / stds / prior.
3. **Softmax dilutes it** — best sample IS the right maneuver but
   `bestWeightShare` is small and `emitted` ≈ brake/zero. Fix λ vs cost scale.
4. **Model can't represent it** — samples look right, emitted right, plant does
   something else. Go to the `model-vs-plant-fidelity` skill.

## λ (temperature) must match the cost scale AT STAKE

Weights are `exp(-(cost − min) / λ)`. What matters is the cost SPREAD between
meaningfully different maneuvers, which varies by situation:

- Open-road racing horizon: spreads of hundreds → λ ≈ 3 blends smoothly.
- A 1.5 m shunt / precision leg: total progress at stake ≈ wProgress × leg
  length ≈ 9 → at λ=3 everything gets similar weight, the average is a
  brake-hold, the car freezes (measured: `bestWeightShare` 0.07). Needs λ ≈ 0.5.

If the same controller must handle both, switch config by segment kind (the
race runner uses `MPC_CUSP_CONFIG` for non-final cusp legs) — or normalize.
Symptom → check `bestWeightShare`: ≈1/K means over-diluted; ≈1.0 means
argmax-chatter (λ too small for the regime).

## Sampling shape (progress mode) — the load-bearing details

- **Single pedal channel** `a ∈ [−1,1]` (a≥0 throttle, a<0 brake), sampled in
  the segment's gear. Independent drive/brake noise fights itself — the
  near-binary raycast brake out-muscles mean drive noise and nothing launches.
- **AR(1) correlated noise** (ρ≈0.8) across the horizon. White noise averages
  itself out over 30 steps and cannot represent "hold throttle down the
  straight".
- **Average in the channel you sampled in**: the softmax average must be
  re-projected onto the pedal channel (else it emits throttle+brake together).
- **Warm-start trap on regime change**: the prior is last solve's sequence
  shifted by one. After a gear change or long brake-hold the prior is deep in
  the wrong channel and correlated noise explores around it — expect several
  solves of lag, or re-seed the prior on gear change.

## Progress-cost pitfalls (all hit in practice)

- **Anchor teleport**: nearest-distance projection over the whole plan lets the
  anchor jump to a later leg passing nearby (free progress for standing
  still). Projection must be arc-bounded around a persistent cursor
  (`MPCTrackerState.lastAnchorIdx`), with a SMALL window (~2 m) for fresh
  plans — they start at the chassis by construction.
- **Curvature noise**: Menger curvature on adjacent samples of a dense smoothed
  plan produces phantom hairpins (κ≈5 from sub-cm jitter) that wreck the
  allowed-speed profile. Use a fixed arc baseline (~1.2 m) around each sample.
- **Envelope from raw kink radius**: rollouts may legally round a plan kink
  inside the corridor, so cap speed at radius (R + corridor slack), not R.
- **Per-step heading cost taxes honesty**: an honest model has real yaw-rate
  lag; charging heading error every step prices the truthful model out while
  the kinematic delusion (instant yaw) sails through. Terminal-only heading
  alignment keeps the anti-wrong-way purpose without the honesty tax.
- **Plan speeds as caps**: only when the plan went through the speed-profile
  pass. Raw plan speeds contain junk (analytic-shot samples at maxSpeed,
  near-cusp samples at ~1 m/s) — one bad interior sample pins the whole
  approach through the backward braking envelope.
- **Plan end is not a wall**: extend drive-through plans past their terminal
  (`referenceExtension`) or the horizon treats the plan end as a stop target
  and lifts early everywhere.

## Execution details that must stay consistent

- MPPI plans piecewise-constant controls at `stepDt` (0.05 s): execute by
  HOLDING each command for `stepDt` (3 physics ticks), not re-solving at 60 Hz.
- `substeps: 3` makes the model integrate at its native 1/60 s resolution
  inside each control step.
- Determinism: the sampler RNG lives in `MPCTrackerState` (LCG, seeded);
  reset state on scenario reset or reruns diverge.
- Multi-gear plans arrive as single-gear SEGMENTS (`splitAtGearCusps`); a
  non-final segment ends at a cusp = genuine stop, never a drive-through.
