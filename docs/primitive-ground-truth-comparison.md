# Motion primitives: learned model vs ground-truth Rapier

Model: `demos/public/models/v2-default.json` — ensemble size 3.

"Stage 2 from ground truth" = roll the SAME control × start-speed grid
straight through the Rapier raycast-vehicle (no learned model). Endpoint
position error is the gap between each model's predicted end pose and
where the real chassis actually ends up, in the start-local frame.

**Matched initial conditions**: the learned/parametric rollouts start from
Rapier's exact post-settle state (speed, yaw rate, lateral velocity), so the
measured gap is pure dynamics-model error — not a start-state mismatch.
Mean settle-phase speed decay across all primitives: 0.132 m/s (handed to both models, so it cancels).

## Tier: coarse (20 primitives, 0.5s each)

| metric | value |
|---|---|
| parametric-only endpoint RMS vs Rapier | **1.332 m** |
| full learned endpoint RMS vs Rapier | **0.427 m** |
| residual helps (full < para) | yes (68.0%) |
| primitives where OOD gate fires | 0 / 20 |
| primitives where residual is active | 20 / 20 |

Per start-speed bucket (full learned endpoint err vs Rapier):

| start speed | full err RMS | para err RMS | mean ensemble σ (pos) | gate fires |
|---|---|---|---|---|
| 0 m/s | 0.262 m | 0.463 m | 0.013 | 0/5 |
| 4 m/s | 0.220 m | 0.684 m | 0.034 | 0/5 |
| 8 m/s | 0.450 m | 1.370 m | 0.041 | 0/5 |
| 12 m/s | 0.640 m | 2.130 m | 0.044 | 0/5 |

Worst 8 primitives (largest full-model endpoint error):

```
startSpd  action            fullErr  paraErr  headErr  spdErr  ensσ(pos)  gate  residual
  12.0  reverse-straight   0.836   0.180   0.028   2.900   0.013  off  active
   8.0  reverse-straight   0.812   0.040   0.027   2.668   0.013  off  active
  12.0  drive-left         0.680   1.204   0.085   0.337   0.008  off  active
  12.0  brake-straight     0.678   4.292   0.021   0.413   0.095  off  active
  12.0  drive-right        0.642   1.204   0.073   0.370   0.014  off  active
   0.0  reverse-straight   0.558   0.239   0.023   2.193   0.010  off  active
   4.0  reverse-straight   0.449   0.099   0.027   0.925   0.019  off  active
   8.0  drive-left         0.358   0.965   0.035   0.460   0.008  off  active
```

## Tier: fine (114 primitives, 0.15s each)

| metric | value |
|---|---|
| parametric-only endpoint RMS vs Rapier | **0.542 m** |
| full learned endpoint RMS vs Rapier | **0.361 m** |
| residual helps (full < para) | yes (33.4%) |
| primitives where OOD gate fires | 22 / 114 |
| primitives where residual is active | 113 / 114 |

Per start-speed bucket (full learned endpoint err vs Rapier):

| start speed | full err RMS | para err RMS | mean ensemble σ (pos) | gate fires |
|---|---|---|---|---|
| 0 m/s | 0.013 m | 0.038 m | 0.012 | 0/19 |
| 3 m/s | 0.102 m | 0.187 m | 0.081 | 4/19 |
| 6 m/s | 0.218 m | 0.391 m | 0.103 | 4/19 |
| 9 m/s | 0.390 m | 0.582 m | 0.116 | 5/19 |
| 12 m/s | 0.528 m | 0.724 m | 0.127 | 5/19 |
| 15 m/s | 0.541 m | 0.842 m | 0.127 | 4/19 |

Worst 8 primitives (largest full-model endpoint error):

```
startSpd  action            fullErr  paraErr  headErr  spdErr  ensσ(pos)  gate  residual
  15.0  brake-right        1.149   1.274   0.049   7.523   0.301   ON  active
  12.0  brake-right        1.145   1.145   0.027   8.771   0.161   ON  —
  15.0  brake-left         1.103   1.279   0.034   7.166   0.184   ON  active
  15.0  brake-left         1.100   1.295   0.012   7.287   0.151   ON  active
  15.0  brake-right        1.081   1.181   0.060   6.969   0.359   ON  active
  12.0  brake-left         0.986   1.153   0.013   7.155   0.183   ON  active
  12.0  brake-right        0.978   1.120   0.042   6.968   0.276   ON  active
  12.0  brake-left         0.943   1.133   0.031   6.656   0.172   ON  active
```

## How to read this

- **parametric RMS** is the safety floor: the clean analytical model the
  residual is allowed to correct. **full RMS** is what the planner actually
  expands. If full < para, the overnight residual is net-helping; if full >
  para in a bucket, the residual is *confidently wrong* there (shared bias).
- **ensemble σ** is the OOD signal. High σ → the 3 MLPs disagree → the regime
  is under-trained. The gate falls back to parametric when any channel σ
  exceeds its threshold, so high-σ rows *should* show full ≈ para.
- The Rapier column is the alternative Stage 2: zero model error by
  construction, at the cost of needing the physics engine in the loop.

Caveats: `spdErr` is the error in *forward-axis-projected* speed (lin·forward,
measured identically for model and Rapier); under hard steer the chassis slides
and its velocity rotates off the heading axis, so high-steer rows show large
`spdErr` even when the path is close. Both models are rolled from Rapier's
exact post-settle start state, so suspension settle and start-speed decay
cancel — the residual gap is dynamics-model error alone.
