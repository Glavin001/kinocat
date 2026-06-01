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
from trial_io import (  # noqa: E402
    CONFIG_SCALES, PARAMETRIC_CONFIG_INDICES,
    load_trials, save_mlp_ensemble,
)

# Normalisation scales used by buildMLPInput in vehicle-model.ts —
# must stay in sync.
STATE_SCALES_MLP = jnp.array([1.0, 1.0, 20.0, 4.0, 6.0])
CONTROL_SCALES_MLP = jnp.array([1.0, 4000.0, 2000.0])

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

    Returns `(residual_fn, data_residual_fn)`:
      - `residual_fn(params)`   : data residual + sqrt-reg pull toward prior,
                                  concatenated. This is what LM optimizes.
      - `data_residual_fn(params)`: data residual only, used by the safety
                                  gate to compare init vs fitted on the
                                  underlying trial loss (no reg bias).
    """
    dt = trials.dt
    sample_every = trials.sample_every

    init_states = jnp.asarray(trials.init_states)
    controls = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)
    configs = jnp.asarray(trials.config)[:, jnp.asarray(PARAMETRIC_CONFIG_INDICES)]

    @jax.jit
    def data_residual_fn(params_vec):
        def per_trial(init_s, ctrls, cfg, gt_samples_full):
            gt_after_init = gt_samples_full[1:]
            pred = rollout_trial(
                params=params_vec, config=cfg, init_state=init_s,
                controls_trace=ctrls, dt=dt, sample_every=sample_every,
                reseat_horizon=trajectory_horizon,
                ground_truth_samples=gt_after_init,
            )
            diff = pred - gt_after_init
            heading_diff = wrap_residual(pred[:, 2] - gt_after_init[:, 2])
            diff = diff.at[:, 2].set(heading_diff)
            diff = diff[:, :6]
            return diff * jnp.sqrt(STATE_COMPONENT_WEIGHTS)

        all_res = jax.vmap(per_trial)(init_states, controls, configs, samples)
        return all_res.reshape(-1)

    @jax.jit
    def residual_fn(params_vec):
        data = data_residual_fn(params_vec)
        reg = jnp.sqrt(reg_strength) * (params_vec - prior_vec) / jnp.maximum(jnp.abs(prior_vec), 1e-3)
        return jnp.concatenate([data, reg])

    return residual_fn, data_residual_fn


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
    residual_fn, data_residual_fn = build_loss(trials, trajectory_horizon, reg_strength, prior_vec)

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

    # Safety gate: reject the LM fit if it doesn't improve data loss
    # vs the init. This is the "no worse than init" guarantee — if
    # the LM pinned params to bounds and ended up worse than where it
    # started, we discard the result. Combined with initializing from
    # the previously-shipped model in train.ts, this means the worst
    # case is shipping the same model we started from.
    init_data = data_residual_fn(prior_vec)
    fit_data = data_residual_fn(fitted)
    init_loss = float(jnp.sum(init_data * init_data))
    fit_loss = float(jnp.sum(fit_data * fit_data))
    rel = (fit_loss - init_loss) / max(abs(init_loss), 1e-9)
    if fit_loss < init_loss:
        print(f"[lm] accepted — data loss {init_loss:.3f} -> {fit_loss:.3f} ({rel*100:+.2f}%)", file=sys.stderr, flush=True)
        return np.asarray(fitted)
    print(
        f"[lm] REJECTED — fitted data loss {fit_loss:.3f} >= init {init_loss:.3f}; "
        f"keeping init params (no-regress guard)", file=sys.stderr, flush=True,
    )
    return np.asarray(init_params)


def build_mlp_dataset(trials, parametric_params: np.ndarray):
    """Build (inputs, targets) pairs for residual training.

    Input layout MATCHES `buildMLPInput` EXACTLY (21-dim):
      [0..4]   state: [sin(h), cos(h), speed, yawRate, lateralVelocity]
               divided by STATE_SCALES_MLP = [1, 1, 20, 4, 6]
      [5..7]   controls: [steer, drive, brake]
               divided by CONTROL_SCALES_MLP = [1, 4000, 2000]
      [8..20]  config: full 13-dim encodeConfigOneHot vector
               divided by CONFIG_SCALES

    Residual targets: (actual_next - parametric_predicted_next) for the 6
    output dims [x, z, heading, speed, yawRate, vy]. `pred_samples[k]`
    predicts `samples[k+1]`, so they align 1:1 with samples[:, 1:, :].
    """
    dt = trials.dt
    sample_every = trials.sample_every
    params_j = jnp.asarray(parametric_params)
    init_states = jnp.asarray(trials.init_states)
    controls = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)
    configs_full = jnp.asarray(trials.config)                  # (N, 13)
    configs_param = configs_full[:, jnp.asarray(PARAMETRIC_CONFIG_INDICES)]  # (N, 3)

    def per_trial(init_s, ctrls, cfg, gt_samples):
        return rollout_trial(
            params=params_j, config=cfg, init_state=init_s,
            controls_trace=ctrls, dt=dt, sample_every=sample_every,
            reseat_horizon=0, ground_truth_samples=None,
        )

    pred_samples = jax.vmap(per_trial)(init_states, controls, configs_param, samples)

    # Input state is the sample BEFORE each transition (including the
    # initial state). Target is the residual at the NEXT sample.
    states = samples[:, :-1, :]                # (N, S, 7) where S = n_samples - 1
    next_states = samples[:, 1:, :]            # (N, S, 7), aligns with pred_samples
    S_eff = states.shape[1]

    # Controls at the start of each sample window.
    ctrl_by_sample = controls[:, ::sample_every, :][:, :S_eff, :]

    # Broadcast the 13-dim config across (N, S, 13).
    cfg_bcast = jnp.broadcast_to(configs_full[:, None, :], (configs_full.shape[0], S_eff, 13))

    # Build inputs in the canonical buildMLPInput order.
    sinH = jnp.sin(states[:, :, 2:3])
    cosH = jnp.cos(states[:, :, 2:3])
    state_feat = jnp.concatenate([
        sinH, cosH,
        states[:, :, 3:4],   # speed
        states[:, :, 4:5],   # yawRate
        states[:, :, 5:6],   # lateralVelocity
    ], axis=-1) / STATE_SCALES_MLP

    ctrl_feat = ctrl_by_sample / CONTROL_SCALES_MLP
    cfg_feat = cfg_bcast / jnp.asarray(CONFIG_SCALES)

    feats = jnp.concatenate([state_feat, ctrl_feat, cfg_feat], axis=-1)  # (N, S, 21)
    assert feats.shape[-1] == 21, f"expected 21-dim MLP input, got {feats.shape[-1]}"

    # Residual targets: actual next - predicted next, heading wrapped.
    tgt = (next_states - pred_samples)[:, :, :6]
    heading_diff = wrap_residual(next_states[:, :, 2] - pred_samples[:, :, 2])
    tgt = tgt.at[:, :, 2].set(heading_diff)

    inputs = feats.reshape(-1, feats.shape[-1])
    targets_flat = tgt.reshape(-1, 6)
    split = jnp.asarray(trials.split)
    split_b = jnp.broadcast_to(split[:, None], (split.shape[0], S_eff)).reshape(-1)
    return np.asarray(inputs), np.asarray(targets_flat), np.asarray(split_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--init-params", required=True)
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--out-residual", default=None)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-7)
    # Matches REG_SCALES strength in training-driver.ts fitParametric.
    ap.add_argument("--reg-strength", type=float, default=0.05)
    # Shorter horizon (10 -> 5) so the loss focuses on short-term
    # prediction quality. Long-horizon reseating with the smooth proxy
    # accumulates per-tick smooth-vs-piecewise drift that can push LM
    # toward bound-pinning unrelated to the underlying physics fit.
    ap.add_argument("--trajectory-horizon", type=int, default=5)
    ap.add_argument("--mlp-shape", default="64,64",
                    help="Comma-separated hidden dims, e.g. '64,64' or '128,128,128'.")
    ap.add_argument("--ensemble-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    # Stronger weight decay (1e-4 -> 1e-3) — with [64,64]×3 = ~17k
    # params the ensemble overfits aggressively on small trial sets
    # and predicts large residuals OOD. Higher decay keeps it close
    # to zero on inputs it hasn't seen.
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    # Skip residual MLP training when the trial set is too small to
    # constrain a [64,64]×3 ensemble — the MLP overfits and predicts
    # huge residuals at eval time. Tune via --min-residual-trials.
    ap.add_argument("--min-residual-trials", type=int, default=200)
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
        if trials.n < args.min_residual_trials:
            print(
                f"[jax-trainer] skipping residual MLP — {trials.n} trials < "
                f"--min-residual-trials={args.min_residual_trials} (would overfit)",
                file=sys.stderr,
            )
            return
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
        ensemble, history, best_val = train_ensemble(
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
        # Safety gate: only ship the residual MLP if it actually improves
        # val loss vs parametric-only (zero residual). Otherwise the MLP
        # adds noise at inference and the runtime model is strictly
        # worse than the parametric backbone. The previous overnight run
        # (commit a46b8a2) emitted a residual that doubled posRms; this
        # check makes that case auto-degrade to parametric-only.
        zero_baseline = float(np.mean(np.sum(
            np.asarray(RESIDUAL_LOSS_WEIGHTS) * val_targets * val_targets, axis=-1
        )))
        # Use the best val loss seen during training (early-stop can
        # leave the last-epoch value above the minimum).
        mlp_val = float(best_val)
        improved = mlp_val < zero_baseline * 0.95   # require >5% improvement
        print(
            f"[jax-trainer] residual val loss={mlp_val:.4f}  zero-residual baseline={zero_baseline:.4f}  "
            f"({'KEEP — MLP helps' if improved else 'DROP — MLP does not help, falling back to parametric-only'})",
            file=sys.stderr,
        )
        if improved:
            save_mlp_ensemble(args.out_residual, ensemble_to_layer_arrays(ensemble))
            print(f"[jax-trainer] wrote MLP ensemble to {args.out_residual} (final val={mlp_val:.5f})", file=sys.stderr)
        else:
            # Deliberately don't write the npz; Node side falls back to
            # parametric-only when the file is missing.
            print(f"[jax-trainer] residual MLP not shipped — parametric-only at inference", file=sys.stderr)


if __name__ == "__main__":
    main()
