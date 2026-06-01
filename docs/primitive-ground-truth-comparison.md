# Motion primitives: learned model vs ground-truth Rapier

Model: `demos/public/models/v2-default.json` — ensemble size 3.

"Stage 2 from ground truth" = roll the SAME control × start-speed grid
straight through the Rapier raycast-vehicle (no learned model). Endpoint
position error is the gap between each primitive's predicted end pose and
where the real chassis actually ends up, in the start-local frame.

## Tier: coarse (20 primitives, 0.5s each)

| metric | value |
|---|---|
| parametric-only endpoint RMS vs Rapier | **1.339 m** |
| full learned endpoint RMS vs Rapier | **0.422 m** |
| residual helps (full < para) | yes (68.4%) |
| primitives where OOD gate fires | 0 / 20 |
| primitives where residual is active | 20 / 20 |

Per start-speed bucket (full learned endpoint err vs Rapier):

| start speed | full err RMS | para err RMS | mean ensemble σ (pos) | gate fires |
|---|---|---|---|---|
| 0 m/s | 0.262 m | 0.463 m | 0.013 | 0/5 |
| 4 m/s | 0.214 m | 0.675 m | 0.034 | 0/5 |
| 8 m/s | 0.437 m | 1.375 m | 0.041 | 0/5 |
| 12 m/s | 0.639 m | 2.147 m | 0.043 | 0/5 |

Worst 8 primitives (largest full-model endpoint error):

```
startSpd  action            fullErr  paraErr  headErr  spdErr  ensσ(pos)  gate  residual
   8.0  reverse-straight   0.757   0.013   0.027   2.576   0.013  off  active
  12.0  reverse-straight   0.749   0.099   0.028   2.736   0.013  off  active
  12.0  drive-left         0.701   1.174   0.086   0.177   0.008  off  active
  12.0  brake-straight     0.692   4.373   0.021   0.414   0.095  off  active
  12.0  drive-right        0.688   1.174   0.074   0.535   0.014  off  active
   0.0  reverse-straight   0.558   0.239   0.023   2.193   0.010  off  active
   4.0  reverse-straight   0.429   0.126   0.028   0.909   0.019  off  active
   8.0  drive-left         0.355   0.936   0.034   0.356   0.008  off  active
```

## Tier: fine (114 primitives, 0.15s each)

| metric | value |
|---|---|
| parametric-only endpoint RMS vs Rapier | **0.553 m** |
| full learned endpoint RMS vs Rapier | **0.372 m** |
| residual helps (full < para) | yes (32.8%) |
| primitives where OOD gate fires | 22 / 114 |
| primitives where residual is active | 113 / 114 |

Per start-speed bucket (full learned endpoint err vs Rapier):

| start speed | full err RMS | para err RMS | mean ensemble σ (pos) | gate fires |
|---|---|---|---|---|
| 0 m/s | 0.013 m | 0.038 m | 0.012 | 0/19 |
| 3 m/s | 0.104 m | 0.190 m | 0.082 | 4/19 |
| 6 m/s | 0.224 m | 0.398 m | 0.103 | 4/19 |
| 9 m/s | 0.401 m | 0.594 m | 0.116 | 5/19 |
| 12 m/s | 0.544 m | 0.739 m | 0.127 | 5/19 |
| 15 m/s | 0.559 m | 0.861 m | 0.127 | 4/19 |

Worst 8 primitives (largest full-model endpoint error):

```
startSpd  action            fullErr  paraErr  headErr  spdErr  ensσ(pos)  gate  residual
  15.0  brake-right        1.182   1.306   0.049   7.737   0.300   ON  active
  12.0  brake-right        1.171   1.171   0.027   8.939   0.161   ON  —
  15.0  brake-left         1.137   1.311   0.033   7.382   0.184   ON  active
  15.0  brake-left         1.127   1.328   0.012   7.466   0.148   ON  active
  15.0  brake-right        1.114   1.213   0.060   7.187   0.359   ON  active
  12.0  brake-left         1.012   1.179   0.013   7.322   0.182   ON  active
  12.0  brake-right        1.006   1.146   0.042   7.149   0.278   ON  active
  12.0  brake-left         0.970   1.159   0.031   6.830   0.174   ON  active
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

Caveats: (1) `spdErr` is the error in *forward-axis-projected* speed; under
hard steer the chassis slides and its velocity rotates off the heading axis,
so high-steer rows show large `spdErr` even when the path is close. (2) The
ground-truth harness settles the suspension (~0.15s coast) before recording,
matching training conditions, so the GT primitive starts a hair below its
bucket speed — a small handicap charged against the learned model. Endpoint
position error (re-zeroed to the post-settle start) is the robust headline.
