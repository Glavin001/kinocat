# JAX-based vehicle sysid trainer

Python side of the hybrid trainer described in
`plans/jax-parametric-trainer.md`. The Node driver (`demos/scripts/train.ts`)
collects Rapier trials in workers, serialises them to npz, then spawns
`train_fit.py` to do the actual fit and writes the params + residual MLP
back. Both sides share a versioned schema (see `trial_io.py`).

## Setup

Two options. **Docker is recommended** — no host Python install, fully
isolated, reproducible.

### Option A: Docker (recommended)

You need only Docker installed on the host (Docker Desktop on Mac
works). The image is auto-built on first use; the trainer runs inside
the container.

```sh
# One-shot — auto-builds the image on first invocation.
pnpm run train:docker -- --profile=quick

# Or pre-build explicitly:
pnpm run train:docker-build
pnpm run train:docker -- --profile=overnight
```

What happens under the hood:
1. Node collects Rapier trials on the host (uses your existing Node).
2. Trial data is dumped as npz into a host tmp dir.
3. `docker run` mounts that tmp dir + this directory into a container
   and invokes `python -m train_fit` inside.
4. The container writes the fitted params (and optional residual MLP
   ensemble) back to the same mounted dir.
5. Node reads the result and writes `v2-default.json`.

Image: `kinocat-jax-trainer:latest` (~600 MB, JAX + jaxopt + optax + numpy).
Image rebuild is needed only if `requirements.txt` changes — Python
source is mounted at runtime, so edits don't bust the image.

GPU on Docker: not accessible on Mac (Docker Desktop can't pass through
Metal). On a Linux+NVIDIA host, add `--gpus all` to the `docker run`
invocation in `train.ts` and use a `jax[cuda12]` base image.

### Option B: Local Python (for development)

Python 3.11+ is required (JAX wheels).

```sh
python3 -m venv demos/scripts/python/.venv
source demos/scripts/python/.venv/bin/activate
pip install -r demos/scripts/python/requirements.txt

pnpm run train -- --profile=quick --trainer=python
```

For Mac native GPU acceleration: `pip install jax-metal` after the
above. The Node driver does not care which backend JAX picks.

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
