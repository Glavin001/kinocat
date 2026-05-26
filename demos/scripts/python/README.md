# JAX-based vehicle sysid trainer

Python side of the hybrid trainer described in
`plans/jax-parametric-trainer.md`. The Node driver (`demos/scripts/train.ts`)
collects Rapier trials in workers, serialises them to npz, then spawns
`train_fit.py` to do the actual fit and writes the params + residual MLP
back. Both sides share a versioned schema (see `trial_io.py`).

## Setup

Python 3.11+ is required (JAX wheels).

```sh
python3 -m venv demos/scripts/python/.venv
source demos/scripts/python/.venv/bin/activate
pip install -r demos/scripts/python/requirements.txt
```

GPU is optional — `pip install -U "jax[cuda12]"` (Linux/NVIDIA) or
`pip install jax-metal` (Apple Silicon) for autodetected acceleration.
The Node driver does not care which backend JAX picks.

## Entry points

| Script                | Purpose |
| --------------------- | ------- |
| `train_fit.py`        | LM parametric + Adam residual fit (the trainer). |
| `parametric.py`       | JAX port of `parametricForwardV2Smooth` (also importable). |
| `residual.py`         | Flax MLP ensemble training, vmap'd across members. |
| `mlp_sweep.py`        | One-off grid sweep over MLP shapes / ensemble sizes. |
| `trial_io.py`         | Versioned npz reader for trials emitted by Node. |
| `test_equivalence.py` | pytest that pins TS↔JAX numerical equivalence (golden file). |

## CLI: `train_fit.py`

```
python -m train_fit \
  --trials   /tmp/round-0.npz \
  --init-params /tmp/params-in.json \
  --out-params  /tmp/params-out.json \
  [--out-residual /tmp/mlp.npz] \
  [--max-iter 50] [--tol 1e-7] \
  [--mlp-shape 64,64] [--ensemble-size 3] [--epochs 200] \
  [--no-residual]
```

Reads trials.npz + initial params JSON, runs Levenberg-Marquardt on the
parametric backbone using `jaxopt.LevenbergMarquardt` with bounds &
regularisation toward `DEFAULT_LEARNED_PARAMS_V2`, optionally trains a
residual MLP ensemble with Adam, and writes the results back.

## Equivalence guard

`python -m pytest test_equivalence.py` regenerates the golden file
(`parametric-forward-v2-golden.json`, committed at the repo root under
`core/test/agent/`) and fails if either the TS or JAX implementation
moves. The matching TS-side check lives in
`core/test/agent/parametric-forward-v2-jax-equivalence.test.ts`.
