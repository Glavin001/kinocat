"""Residual MLP ensemble trainer (JAX + Flax + Optax).

Replaces the hand-coded MLP / SGD loop in
`core/src/internal/mlp.ts` for the training-time fit only. Inference still
runs through the TS MLP (which is fed the trained weights via npz, see
`v2-model-file.ts::deserializeMLPFromArrays`).

Default shape: `[64, 64]` (bumped from `[32, 32]` per the plan — manifest
data shows the ensemble is under-parameterized by ~20× at the old size).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Per-component residual loss weights — must match RESIDUAL_LOSS_WEIGHTS
# in demos/app/lib/training-driver.ts.
RESIDUAL_LOSS_WEIGHTS = jnp.array([1.0, 1.0, 100.0, 1.0, 10.0, 1.0])


@dataclass(frozen=True)
class MLPParams:
    """A single MLP's weights, layer-flat. Pytree-compatible by `jax.tree_util`."""
    Ws: list[jnp.ndarray]  # each (out, in)
    bs: list[jnp.ndarray]  # each (out,)


def init_mlp(key, input_dim: int, hidden_dims: list[int], output_dim: int) -> MLPParams:
    dims = [input_dim, *hidden_dims, output_dim]
    Ws, bs = [], []
    for li in range(len(dims) - 1):
        in_d, out_d = dims[li], dims[li + 1]
        key, sub = jax.random.split(key)
        is_last = li == len(dims) - 2
        std = 0.01 if is_last else jnp.sqrt(2.0 / in_d)
        W = jax.random.normal(sub, (out_d, in_d)) * std
        bs.append(jnp.zeros(out_d))
        Ws.append(W)
    return MLPParams(Ws=Ws, bs=bs)


def mlp_forward(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: ReLU hidden, linear output."""
    h = x
    n_layers = len(params.Ws)
    for li in range(n_layers):
        h = params.Ws[li] @ h + params.bs[li]
        if li < n_layers - 1:
            h = jax.nn.relu(h)
    return h


# Batched forward over a (B, in) batch
mlp_forward_batched = jax.vmap(mlp_forward, in_axes=(None, 0))
# Batched forward over (members, params) × batched inputs
ensemble_forward = jax.vmap(mlp_forward_batched, in_axes=(0, None))


def _stack_ensemble(ensemble: list[MLPParams]) -> MLPParams:
    """Stack a list of `MLPParams` into one `MLPParams` of shape (E, ...)."""
    return MLPParams(
        Ws=[jnp.stack([m.Ws[li] for m in ensemble]) for li in range(len(ensemble[0].Ws))],
        bs=[jnp.stack([m.bs[li] for m in ensemble]) for li in range(len(ensemble[0].bs))],
    )


def _unstack_ensemble(stacked: MLPParams, n: int) -> list[MLPParams]:
    return [MLPParams(Ws=[w[i] for w in stacked.Ws], bs=[b[i] for b in stacked.bs])
            for i in range(n)]


def train_ensemble(
    *,
    inputs: np.ndarray,           # (N, in)
    targets: np.ndarray,          # (N, out)
    val_inputs: np.ndarray | None,
    val_targets: np.ndarray | None,
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] = (64, 64),
    ensemble_size: int = 3,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_weights: jnp.ndarray | None = None,
    seed: int = 42,
    early_stop_patience: int = 20,
    verbose: bool = False,
) -> tuple[list[MLPParams], dict]:
    """Train an ensemble of `ensemble_size` MLPs, vmap'd across members.

    Returns `(ensemble, history)`.
    """
    hidden_dims = list(hidden_dims)
    loss_weights = loss_weights if loss_weights is not None else RESIDUAL_LOSS_WEIGHTS
    rng = jax.random.PRNGKey(seed)

    # Init ensemble with distinct seeds so the members diverge.
    members = []
    for i in range(ensemble_size):
        rng, sub = jax.random.split(rng)
        members.append(init_mlp(sub, input_dim, hidden_dims, output_dim))
    stacked = _stack_ensemble(members)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = tx.init(stacked)

    inputs_j = jnp.asarray(inputs, dtype=jnp.float32)
    targets_j = jnp.asarray(targets, dtype=jnp.float32)
    N = inputs_j.shape[0]

    def weighted_mse(pred, target):
        err = pred - target
        return jnp.mean(jnp.sum(loss_weights * err * err, axis=-1))

    def loss_fn(stacked_params, xs, ys):
        # ensemble_forward: (E, B, out)
        preds = ensemble_forward(stacked_params, xs)
        # Average ensemble loss (each member fits independently, no cross-coupling).
        per_member = jax.vmap(weighted_mse, in_axes=(0, None))(preds, ys)
        return jnp.mean(per_member)

    @jax.jit
    def train_step(params, opt_state, xs, ys):
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_loss(params, xs, ys):
        return loss_fn(params, xs, ys)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    bad_epochs = 0

    import sys as _sys
    import time as _time
    t_start = _time.time()
    log_every = max(1, epochs // 20)  # ~20 progress lines per run

    for epoch in range(epochs):
        rng, sub = jax.random.split(rng)
        perm = jax.random.permutation(sub, N)
        epoch_loss = 0.0
        n_batches = (N + batch_size - 1) // batch_size
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xs = inputs_j[idx]
            ys = targets_j[idx]
            stacked, opt_state, loss = train_step(stacked, opt_state, xs, ys)
            epoch_loss += float(loss) / n_batches
        history["train_loss"].append(epoch_loss)
        if val_inputs is not None and val_targets is not None:
            val_loss = float(eval_loss(stacked, jnp.asarray(val_inputs), jnp.asarray(val_targets)))
        else:
            val_loss = epoch_loss
        history["val_loss"].append(val_loss)
        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = _time.time() - t_start
            per_ep = elapsed / max(1, epoch + 1)
            eta = per_ep * (epochs - epoch - 1)
            print(
                f"[mlp] ep {epoch + 1:4d}/{epochs}  train={epoch_loss:.5f}  val={val_loss:.5f}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                file=_sys.stderr, flush=True,
            )
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"[mlp] early-stop at epoch {epoch + 1}, best val={best_val:.5f}", file=_sys.stderr, flush=True)
                break

    return _unstack_ensemble(stacked, ensemble_size), history


def ensemble_to_layer_arrays(ensemble: list[MLPParams]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Convert `MLPParams` to plain numpy arrays for npz dumping."""
    out: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for m in ensemble:
        layers = [(np.asarray(W), np.asarray(b)) for W, b in zip(m.Ws, m.bs)]
        out.append(layers)
    return out
