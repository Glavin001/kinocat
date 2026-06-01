"""End-to-end JAX trainer entry point.

Reads trials from npz, runs Levenberg-Marquardt on the parametric
backbone, then (optionally) trains a residual MLP ensemble with Adam.
Writes params JSON + residual ensemble npz.

Invocation (see Node-side glue in demos/scripts/train.ts):

    python -m train_fit \
        --trials   /tmp/round-N.npz \
        --init-params /tmp/params-in.json \
        --out-params  /tmp/params-out.json \
        [--out-residual /tmp/mlp.npz] \
        [--max-iter 50] [--tol 1e-7] \
        [--mlp-shape 64,64] [--ensemble-size 3] [--epochs 200] \
        [--no-residual] [--reg-strength 0.05] [--trajectory-horizon 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from parametric import (  # noqa: E402
    PARAM_NAMES, parametric_forward_v2, rollout_trial,
)
from residual import (  # noqa: E402
    RESIDUAL_LOSS_WEIGHTS, ensemble_to_layer_arrays, train_ensemble,
)
from trial_io import load_trials, save_mlp_ensemble  # noqa: E402

# Loss weights — must match POS_LOSS_WEIGHT, HEADING_LOSS_WEIGHT,
# SPEED_LOSS_WEIGHT in demos/app/lib/training-driver.ts.
POS_W = 1.0
HEADING_W = 100.0
SPEED_W = 1.0

# Per-state component weights for the loss residual (matches stateDeltaForFit
# behaviour: position x/z, heading wrapped, speed, yawRate, lateralVelocity).
STATE_COMPONENT_WEIGHTS = jnp.array([
    POS_W, POS_W, HEADING_W, SPEED_W, 1.0, 1.0,
])


# 16-param bounds — must mirror PARAMS_V2_LO / PARAMS_V2_HI.
PARAM_LO = jnp.array([
    0.7, 0.7, 0.8, 0.02, 0.7, 0.95, 0.7, 0.0, -0.01, 0.05,
    0.5, 0.1, 0.0, 0.0, 0.0, 0.02,
])
PARAM_HI = jnp.array([
    1.3, 1.5, 4.0, 0.6, 1.3, 1.5, 1.6, 0.05, 0.04, 0.4,
    9.0, 1.5, 2.5, 0.1, 250.0, 0.6,
])


def params_to_vec(d: dict) -> np.ndarray:
    return np.array([d[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_params(v: np.ndarray) -> dict:
    return {k: float(v[i]) for i, k in enumerate(PARAM_NAMES)}


def wrap_residual(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a + jnp.pi, 2 * jnp.pi) - jnp.pi


def build_loss(trials, trajectory_horizon: int, reg_strength: float, prior_vec: jnp.ndarray):
    """Build the residual function the LM optimizer minimizes.

    Returns a function `residual(params) -> jnp.ndarray (flat residual vector)`.
    """
    dt = trials.dt
    sample_every = trials.sample_every

    init_states = jnp.asarray(trials.init_states)
    controls = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)
    configs = jnp.asarray(trials.config)

    @jax.jit
    def residual_fn(params_vec):
        # vmap rollout across all trials. Trials are stored as
        # [initial, sample1, sample2, ...] (S+1 entries); the rollout
        # produces S predictions aligned with samples[1:].
        def per_trial(init_s, ctrls, cfg, gt_samples_full):
            gt_after_init = gt_samples_full[1:]   # (S, 7)
            pred = rollout_trial(
                params=params_vec,
                config=cfg,
                init_state=init_s,
                controls_trace=ctrls,
                dt=dt,
                sample_every=sample_every,
                reseat_horizon=trajectory_horizon,
                ground_truth_samples=gt_after_init,
            )
            diff = pred - gt_after_init
            heading_diff = wrap_residual(pred[:, 2] - gt_after_init[:, 2])
            diff = diff.at[:, 2].set(heading_diff)
            diff = diff[:, :6]
            return diff * jnp.sqrt(STATE_COMPONENT_WEIGHTS)

        all_res = jax.vmap(per_trial)(init_states, controls, configs, samples)
        flat = all_res.reshape(-1)
        # L2 regularization toward prior (encoded as residual too).
        reg = jnp.sqrt(reg_strength) * (params_vec - prior_vec) / jnp.maximum(jnp.abs(prior_vec), 1e-3)
        return jnp.concatenate([flat, reg])

    return residual_fn


def fit_parametric_lm(
    *,
    trials,
    init_params: np.ndarray,
    max_iter: int,
    tol: float,
    reg_strength: float,
    trajectory_horizon: int,
    verbose: bool,
) -> np.ndarray:
    """Run Levenberg-Marquardt on the parametric backbone.

    Returns the fitted 16-vector (clamped to bounds).
    """
    prior_vec = jnp.asarray(init_params)
    residual_fn = build_loss(trials, trajectory_horizon, reg_strength, prior_vec)

    # LM in jaxopt doesn't accept hard bounds — apply a soft sigmoid mapping.
    lo = PARAM_LO
    hi = PARAM_HI

    def unconstrained_to_bounded(z):
        return lo + (hi - lo) * jax.nn.sigmoid(z)

    def bounded_to_unconstrained(p):
        p_clamped = jnp.clip(p, lo + 1e-6, hi - 1e-6)
        u = (p_clamped - lo) / (hi - lo)
        return jnp.log(u / (1 - u))

    def wrapped_residual(z):
        return residual_fn(unconstrained_to_bounded(z))

    z0 = bounded_to_unconstrained(prior_vec)

    # Pre-JIT pass so the first reported time isn't compile-dominated.
    print(f"[lm] jit-compiling forward + residual (this is the cold-start cost)…", file=sys.stderr, flush=True)
    t_compile_start = time.time()
    _ = wrapped_residual(z0).block_until_ready()
    print(f"[lm] jit-compile done in {time.time() - t_compile_start:.1f}s", file=sys.stderr, flush=True)

    lm = jaxopt.LevenbergMarquardt(
        residual_fun=wrapped_residual,
        maxiter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    t0 = time.time()
    sol = lm.run(z0)
    fitted = unconstrained_to_bounded(sol.params)
    fitted.block_until_ready()
    iters = int(sol.state.iter_num)
    elapsed = time.time() - t0
    per_iter = elapsed / max(1, iters)
    print(f"[lm] done in {elapsed:.1f}s ({iters} iter, {per_iter*1000:.0f} ms/iter, gradNorm={float(sol.state.gradient.dot(sol.state.gradient))**0.5:.2e})", file=sys.stderr, flush=True)
    return np.asarray(fitted)


def build_mlp_dataset(trials, parametric_params: np.ndarray):
    """Build (inputs, targets) pairs for residual training.

    Input layout matches `buildMLPInput` (21-dim: 5 state-as-sin/cos + 3 controls +
    13 config one-hot). For now we approximate with a simple layout — the
    Node-side glue will rebuild the exact 21-dim featurisation from the
    canonical `buildMLPInput` at residual-load time.

    Residual targets: (actual_next - parametric_predicted_next) for the 6
    output dims `[x, z, heading, speed, yawRate, vy]`.
    """
    # Roll one parametric step per sample boundary and compute the residual.
    # This is the *open-loop residual at the sample horizon* — same target
    # the JS `runResidualMLPFitAsync` uses.
    dt = trials.dt
    sample_every = trials.sample_every
    params_j = jnp.asarray(parametric_params)
    init_states = jnp.asarray(trials.init_states)
    controls = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)
    configs = jnp.asarray(trials.config)

    def per_trial(init_s, ctrls, cfg, gt_samples):
        # Roll out without reseating to get parametric predictions.
        pred = rollout_trial(
            params=params_j, config=cfg, init_state=init_s,
            controls_trace=ctrls, dt=dt, sample_every=sample_every,
            reseat_horizon=0, ground_truth_samples=None,
        )
        return pred

    pred_samples = jax.vmap(per_trial)(init_states, controls, configs, samples)

    # Inputs: rebuild a simple 16-dim featurisation — heading sin/cos,
    # speed, yawRate, vy, controls, config. The Node side rebuilds the
    # full 21-dim layout; for the Python trainer we use this reduced form
    # because the residual MLP layout is fully defined by the saved
    # weights anyway.
    # Note: keeping featurisation explicit here lets us re-train with the
    # exact training-driver buildMLPInput later if needed.
    states = samples[:, :-1, :]   # (N, S-1, 7) — input is state BEFORE each transition
    next_states = samples[:, 1:, :]  # (N, S-1, 7)
    # Controls midpoint of each sample window:
    # use control at the first physics tick of each window for simplicity.
    ctrl_by_sample = controls[:, ::sample_every, :][:, :states.shape[1], :]
    cfg_bcast = jnp.broadcast_to(configs[:, None, :], (configs.shape[0], states.shape[1], 3))

    cosH = jnp.cos(states[:, :, 2:3])
    sinH = jnp.sin(states[:, :, 2:3])
    # 16-dim: [cosH, sinH, speed, yawRate, vy, steer, drive/4000, brake/2000, mass/1500, wheelBase/1.5, friction]
    feats = jnp.concatenate([
        cosH, sinH,
        states[:, :, 3:4], states[:, :, 4:5], states[:, :, 5:6],
        ctrl_by_sample[:, :, 0:1], ctrl_by_sample[:, :, 1:2] / 4000.0, ctrl_by_sample[:, :, 2:3] / 2000.0,
        cfg_bcast[:, :, 0:1] / 1500.0, cfg_bcast[:, :, 1:2] / 1.5, cfg_bcast[:, :, 2:3],
    ], axis=-1)

    # `pred_samples[k]` predicts `samples[k+1]`, so it already aligns with
    # `next_states = samples[:, 1:, :]` — no extra slicing needed.
    tgt = (next_states - pred_samples)[:, :, :6]
    heading_diff2 = wrap_residual(next_states[:, :, 2] - pred_samples[:, :, 2])
    tgt = tgt.at[:, :, 2].set(heading_diff2)

    # Flatten across trial × sample.
    F = feats.shape[-1]
    inputs = feats.reshape(-1, F)
    targets_flat = tgt.reshape(-1, 6)

    # Split by trial-level split mask (broadcast).
    split = jnp.asarray(trials.split)
    split_b = jnp.broadcast_to(split[:, None], (split.shape[0], feats.shape[1])).reshape(-1)
    return np.asarray(inputs), np.asarray(targets_flat), np.asarray(split_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--init-params", required=True)
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--out-residual", default=None)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--reg-strength", type=float, default=0.05)
    ap.add_argument("--trajectory-horizon", type=int, default=10)
    ap.add_argument("--mlp-shape", default="64,64",
                    help="Comma-separated hidden dims, e.g. '64,64' or '128,128,128'.")
    ap.add_argument("--ensemble-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"[jax-trainer] loading trials from {args.trials}", file=sys.stderr)
    trials = load_trials(args.trials)
    print(f"[jax-trainer] N={trials.n} S={trials.n_samples} dt={trials.dt}", file=sys.stderr)

    with open(args.init_params) as f:
        init_dict = json.load(f)
    init_vec = params_to_vec(init_dict)
    print(f"[jax-trainer] init params: {init_vec}", file=sys.stderr)

    fitted_vec = fit_parametric_lm(
        trials=trials,
        init_params=init_vec,
        max_iter=args.max_iter,
        tol=args.tol,
        reg_strength=args.reg_strength,
        trajectory_horizon=args.trajectory_horizon,
        verbose=args.verbose,
    )
    print(f"[jax-trainer] fitted params: {fitted_vec}", file=sys.stderr)

    out = vec_to_params(fitted_vec)
    Path(args.out_params).write_text(json.dumps(out, indent=2))
    print(f"[jax-trainer] wrote params to {args.out_params}", file=sys.stderr)

    if not args.no_residual and args.out_residual:
        print(f"[jax-trainer] training residual MLP", file=sys.stderr)
        inputs, targets, split = build_mlp_dataset(trials, fitted_vec)
        train_mask = split == 0
        val_mask = split == 1
        # If no explicit val split, take last 20% as val.
        if val_mask.sum() == 0:
            n = inputs.shape[0]
            n_val = max(1, int(n * 0.2))
            train_inputs = inputs[:-n_val]
            train_targets = targets[:-n_val]
            val_inputs = inputs[-n_val:]
            val_targets = targets[-n_val:]
        else:
            train_inputs = inputs[train_mask]
            train_targets = targets[train_mask]
            val_inputs = inputs[val_mask]
            val_targets = targets[val_mask]

        hidden_dims = [int(x) for x in args.mlp_shape.split(",") if x]
        ensemble, history = train_ensemble(
            inputs=train_inputs,
            targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            input_dim=inputs.shape[1],
            output_dim=6,
            hidden_dims=hidden_dims,
            ensemble_size=args.ensemble_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            verbose=args.verbose,
        )
        save_mlp_ensemble(args.out_residual, ensemble_to_layer_arrays(ensemble))
        print(f"[jax-trainer] wrote MLP ensemble to {args.out_residual} (final val={history['val_loss'][-1]:.5f})", file=sys.stderr)


if __name__ == "__main__":
    main()
