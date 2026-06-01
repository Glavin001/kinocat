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
from scipy.optimize import minimize as scipy_minimize

sys.path.insert(0, str(Path(__file__).parent))
from parametric import (  # noqa: E402
    PARAM_NAMES, parametric_forward_v2, parametric_forward_v2_piecewise,
    rollout_trial, rollout_trial_piecewise,
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

# Per-state component weights for the loss residual. MUST mirror
# `stateDeltaForFit` in demos/app/lib/training-driver.ts:346 exactly —
# that function uses ONLY pos, heading, and speed (it deliberately
# excludes yawRate and lateralVelocity from the parametric fit because
# Rapier's measured values for those are noisy and don't constrain the
# 16-param model usefully). Including them with non-zero weight pulls
# the optimizer away from the position/heading optimum.
STATE_COMPONENT_WEIGHTS = jnp.array([
    POS_W, POS_W, HEADING_W, SPEED_W, 0.0, 0.0,
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

# Per-parameter regularization scales — must mirror REG_SCALES in
# demos/app/lib/training-driver.ts. Hand-tuned per-coefficient so the
# reg pulls weakly-identified params (high scale) less than tightly-
# bounded ones (low scale).
REG_SCALES = jnp.array([
    0.10, 0.15, 0.30, 0.10, 0.15, 0.10, 0.15,
    0.005, 0.005, 0.08, 2.5, 0.30, 0.30, 0.02, 80.0, 0.05,
])

# The reg prior is ALWAYS the factory defaults, not whatever was passed
# in as --init-params. This matches JS behaviour and is what keeps
# round N from drifting away from physical sense even when round N-1
# produced a slightly biased fit.
DEFAULT_PARAMS_VEC = jnp.array([
    0.85, 0.9, 1.6, 0.2, 1.0, 1.2, 1.0, 0.006, 0.002, 0.18,
    4.5, 0.6, 0.4, 0.02, 50.0, 0.05,
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


def build_piecewise_loss(trials, trajectory_horizon: int, reg_strength: float, prior_vec: jnp.ndarray):
    """JIT-compiled scalar loss against the PIECEWISE runtime forward.

    This is the same physics the planner uses at inference (parametricForwardV2,
    not the Smooth variant), so the optimum here is the optimum the runtime
    actually sees — no proxy-vs-runtime drift, no bound-pinning from smooth
    surrogates.

    The regularization formulation MIRRORS the JS path
    (core/src/learning/parametric-fit.ts):

      loss = sum_over_trials sum_over_samples stateDeltaForFit(pred, actual)
           + (reg_strength * total_count) * sum_i ((params[i] - DEFAULT[i]) / REG_SCALE[i])²

    Note the reg term scales by `total_count` so its weight stays balanced
    against the data term as the trial set grows. The prior is ALWAYS
    DEFAULT_PARAMS_VEC (not the per-round init), which is what gives JS
    the stability across rounds we previously lacked.

    Returns `(loss_fn, data_loss_fn)`.
    """
    dt = trials.dt
    sample_every = trials.sample_every
    init_states = jnp.asarray(trials.init_states)
    controls = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)
    configs = jnp.asarray(trials.config)[:, jnp.asarray(PARAMETRIC_CONFIG_INDICES)]
    # JS reg uses `totalCount = total sample comparisons`.
    total_count = int(init_states.shape[0]) * (int(samples.shape[1]) - 1)

    @jax.jit
    def data_loss_fn(params_vec):
        def per_trial(init_s, ctrls, cfg, gt_samples_full):
            gt_after_init = gt_samples_full[1:]
            pred = rollout_trial_piecewise(
                params=params_vec, config=cfg, init_state=init_s,
                controls_trace=ctrls, dt=dt, sample_every=sample_every,
                reseat_horizon=trajectory_horizon,
                ground_truth_samples=gt_after_init,
            )
            diff = pred - gt_after_init
            heading_diff = wrap_residual(pred[:, 2] - gt_after_init[:, 2])
            diff = diff.at[:, 2].set(heading_diff)
            diff = diff[:, :6]
            weighted = diff * jnp.sqrt(STATE_COMPONENT_WEIGHTS)
            return jnp.sum(weighted * weighted)

        per_trial_losses = jax.vmap(per_trial)(init_states, controls, configs, samples)
        return jnp.sum(per_trial_losses)

    reg_scale_factor = reg_strength * total_count

    @jax.jit
    def loss_fn(params_vec):
        data = data_loss_fn(params_vec)
        # Per-parameter scaled deviation from the FIXED factory default.
        d = (params_vec - DEFAULT_PARAMS_VEC) / REG_SCALES
        reg = reg_scale_factor * jnp.sum(d * d)
        return data + reg

    return loss_fn, data_loss_fn


def fit_parametric_nelder_mead(
    *,
    trials,
    init_params: np.ndarray,
    max_iter: int,
    tol: float,
    reg_strength: float,
    trajectory_horizon: int,
    verbose: bool,
) -> np.ndarray:
    """Scipy Nelder-Mead on the PIECEWISE loss, with native bounds.

    Same optimizer family as the JS path (NM is gradient-free), so quality
    parity is expected by construction. The speedup over JS comes from the
    JIT-compiled, vmap'd forward inside the loss — JAX/XLA does ~10-30×
    more forward-passes per second than the JS event loop.

    No-regress: caller compares fitted vs init data loss and discards on
    regression (gate already in main).
    """
    loss_fn, data_loss_fn = build_piecewise_loss(
        trials, trajectory_horizon, reg_strength, jnp.asarray(init_params),
    )

    # Pre-JIT pass so the first reported time isn't compile-dominated.
    print(f"[nm] jit-compiling piecewise loss…", file=sys.stderr, flush=True)
    t_compile_start = time.time()
    init_loss = float(loss_fn(jnp.asarray(init_params)))
    print(f"[nm] jit-compile done in {time.time() - t_compile_start:.1f}s  init loss={init_loss:.4f}", file=sys.stderr, flush=True)

    # scipy callback receiving a numpy 16-vector and returning a python float.
    iter_counter = {"n": 0, "best": init_loss}
    def py_loss(p):
        v = float(loss_fn(jnp.asarray(p)))
        iter_counter["n"] += 1
        if v < iter_counter["best"]:
            iter_counter["best"] = v
        return v

    bounds = list(zip(np.asarray(PARAM_LO).tolist(), np.asarray(PARAM_HI).tolist()))

    # Match the JS NM initial simplex: 10% proportional perturbation per
    # coord (`v[i] = x0[i] * 1.1` when x0[i] != 0, else 0.1). This is what
    # JS gets to 0.8 m posRms@1s with, so we mirror it exactly. When
    # init params are already near-optimal (rounds 1+ when round 0
    # already produced a good fit), this small simplex refines without
    # jiggling past the optimum.
    init_p = np.asarray(init_params)
    lo_np = np.asarray(PARAM_LO)
    hi_np = np.asarray(PARAM_HI)
    step = 0.1
    initial_simplex = np.zeros((17, 16))
    initial_simplex[0] = init_p
    for i in range(16):
        v = init_p.copy()
        if abs(v[i]) > 1e-12:
            v[i] = v[i] * (1.0 + step)
        else:
            v[i] = step
        # Clip to bounds — for params at the upper bound, fall back to
        # subtraction so the simplex stays inside the feasible region.
        if v[i] > hi_np[i]:
            v[i] = init_p[i] * (1.0 - step) if abs(init_p[i]) > 1e-12 else -step
            v[i] = max(v[i], lo_np[i])
        if v[i] < lo_np[i]:
            v[i] = lo_np[i]
        initial_simplex[i + 1] = v

    t0 = time.time()
    result = scipy_minimize(
        py_loss,
        init_p,
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'xatol': tol, 'fatol': tol,
            'maxiter': max_iter,
            'maxfev': max_iter * 16,    # match adaptive NM eval budget
            'adaptive': True,
            'initial_simplex': initial_simplex,
            'disp': verbose,
        },
    )
    elapsed = time.time() - t0
    fit_loss = float(loss_fn(jnp.asarray(result.x)))
    n_evals = iter_counter["n"]
    per_eval = elapsed / max(1, n_evals) * 1000
    print(
        f"[nm] done in {elapsed:.1f}s ({n_evals} evals, {per_eval:.0f} ms/eval)  "
        f"loss {init_loss:.4f} -> {fit_loss:.4f} ({(fit_loss/init_loss - 1)*100:+.2f}%)  "
        f"nit={result.nit}  status={result.status}  msg={result.message}",
        file=sys.stderr, flush=True,
    )

    # No-regress safety gate (against piecewise data loss, no reg bias).
    init_data = float(data_loss_fn(jnp.asarray(init_params)))
    fit_data = float(data_loss_fn(jnp.asarray(result.x)))
    if fit_data < init_data:
        print(f"[nm] accepted — piecewise data loss {init_data:.3f} -> {fit_data:.3f} ({(fit_data/init_data-1)*100:+.2f}%)", file=sys.stderr, flush=True)
        return np.asarray(result.x)
    print(
        f"[nm] REJECTED — fitted data loss {fit_data:.3f} >= init {init_data:.3f}; "
        f"keeping init params (no-regress guard)", file=sys.stderr, flush=True,
    )
    return np.asarray(init_params)


def build_mlp_dataset(trials, parametric_params: np.ndarray):
    """Build (inputs, targets) pairs for residual MLP training.

    MIRRORS the JS `buildSamples` in core/src/learning/residual-mlp-fit.ts:

      for each sample window (a, b) in each trial:
        windowDt = (b.t - a.t) / fitSubstepsPerSample   (== dt at our sampling)
        ctrl    = controls at the MIDPOINT of the window
        input   = buildMLPInput(state_at_a, ctrl, config)
        s = a.state
        for j in range(fitSubstepsPerSample):           (== sample_every here)
            s = sim_piecewise(s, ctrl, windowDt)
        target  = (actual_at_b - s)                     (heading wrapped)
        # IMPORTANT: implicit reseat — the next window starts from b.state,
        # NOT from the predicted s. This was the bug in the previous
        # implementation: an uninterrupted rollout meant residual targets
        # at sample 10 contained 9 windows of compounding parametric drift,
        # which the MLP couldn't learn reliably and which were noise on
        # the test split. With per-window reseating, targets are small
        # consistent residuals.

    Also fixed: use the PIECEWISE forward (runtime-identical), not the
    Smooth proxy, so the residual targets and inference baseline agree.

    Input layout matches `buildMLPInput` (21-dim).
    """
    dt = trials.dt
    sample_every = trials.sample_every
    params_j = jnp.asarray(parametric_params)
    controls_all = jnp.asarray(trials.controls_trace)
    samples = jnp.asarray(trials.samples)             # (N, S+1, 7)
    configs_full = jnp.asarray(trials.config)         # (N, 13)
    configs_param = configs_full[:, jnp.asarray(PARAMETRIC_CONFIG_INDICES)]  # (N, 3)

    S_eff = int(samples.shape[1]) - 1                 # sample windows per trial

    def per_trial(ctrls, cfg_param, cfg_full, sams):
        # ctrls: (T, 3); cfg_param: (3,); cfg_full: (13,); sams: (S+1, 7)
        def per_window(k):
            a = sams[k]
            b = sams[k + 1]
            ctrl_idx = jnp.minimum(
                k * sample_every + sample_every // 2,
                ctrls.shape[0] - 1,
            )
            ctrl = ctrls[ctrl_idx]

            # Roll piecewise baseline for sample_every ticks from a.
            def physics_step(s, _):
                return parametric_forward_v2_piecewise(params_j, cfg_param, s, ctrl, dt), None

            s_pred, _ = jax.lax.scan(physics_step, a, jnp.arange(sample_every))

            # Residual target: actual - predicted, heading wrapped.
            target = (b - s_pred)[:6]
            heading_diff = wrap_residual(b[2] - s_pred[2])
            target = target.at[2].set(heading_diff)

            # 21-dim input: featurise from state_at_a (BEFORE the window).
            sinH = jnp.sin(a[2])
            cosH = jnp.cos(a[2])
            state_feat = jnp.array([sinH, cosH, a[3], a[4], a[5]]) / STATE_SCALES_MLP
            ctrl_feat = ctrl / CONTROL_SCALES_MLP
            cfg_feat = cfg_full / jnp.asarray(CONFIG_SCALES)
            feat = jnp.concatenate([state_feat, ctrl_feat, cfg_feat])
            return feat, target

        feats, tgts = jax.vmap(per_window)(jnp.arange(S_eff))
        return feats, tgts

    feats_per_trial, tgts_per_trial = jax.vmap(per_trial)(
        controls_all, configs_param, configs_full, samples,
    )
    assert feats_per_trial.shape[-1] == 21, f"expected 21-dim MLP input, got {feats_per_trial.shape[-1]}"

    inputs = feats_per_trial.reshape(-1, feats_per_trial.shape[-1])
    targets_flat = tgts_per_trial.reshape(-1, 6)
    split = jnp.asarray(trials.split)
    split_b = jnp.broadcast_to(split[:, None], (split.shape[0], S_eff)).reshape(-1)
    return np.asarray(inputs), np.asarray(targets_flat), np.asarray(split_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--init-params", required=True)
    ap.add_argument("--out-params", required=True)
    ap.add_argument("--out-residual", default=None)
    ap.add_argument("--optimizer", choices=["nm", "lm"], default="nm",
                    help="nm: Nelder-Mead on the piecewise (runtime-identical) "
                         "loss. lm: legacy Levenberg-Marquardt on the smooth "
                         "proxy (kept as fallback).")
    # NM iter cap. JS uses 120-200 iters per round and gets the bulk of
    # the improvement in the first ~100 iters. Empirically (sandbox
    # overnight bench), running scipy NM to "convergence" (~1500-2000
    # iters) buys <1% additional data-loss improvement per round in
    # exchange for 5-10× wall time. 250 matches JS's effective iter
    # budget per round.
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--tol", type=float, default=1e-6)
    # Early-stop NM when relative improvement over a window is below
    # this threshold (in addition to scipy's xatol/fatol). Default
    # matches "JS-like" behavior of accepting modest progress.
    ap.add_argument("--early-stop-rel-impr", type=float, default=1e-4)
    ap.add_argument("--early-stop-window", type=int, default=60)
    # Matches REG_SCALES strength in training-driver.ts fitParametric.
    ap.add_argument("--reg-strength", type=float, default=0.05)
    # Matches JS training-driver.ts:781 (trajectoryHorizon=10) so the
    # NM landscape mirrors the JS one.
    ap.add_argument("--trajectory-horizon", type=int, default=10)
    ap.add_argument("--mlp-shape", default="64,64",
                    help="Comma-separated hidden dims, e.g. '64,64' or '128,128,128'.")
    ap.add_argument("--ensemble-size", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    # JS uses no weight decay; we keep a small amount (1e-4) to soften
    # OOD residual magnitudes without distorting in-distribution fit.
    ap.add_argument("--weight-decay", type=float, default=1e-4)
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

    if args.optimizer == "nm":
        # Default. JIT'd scipy Nelder-Mead on the piecewise (runtime-identical)
        # loss. No proxy drift, no bound-pinning, gradient-free.
        fitted_vec = fit_parametric_nelder_mead(
            trials=trials,
            init_params=init_vec,
            max_iter=args.max_iter,
            tol=args.tol,
            reg_strength=args.reg_strength,
            trajectory_horizon=args.trajectory_horizon,
            verbose=args.verbose,
        )
    else:
        # Legacy LM-on-smooth-proxy. Kept for comparison / fallback. LM caps
        # at maxiter=50 by convention; if the user left --max-iter at the
        # NM-tuned default of 2000, drop it back to 50 to avoid LM thrashing.
        lm_max_iter = min(args.max_iter, 50)
        fitted_vec = fit_parametric_lm(
            trials=trials,
            init_params=init_vec,
            max_iter=lm_max_iter,
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
