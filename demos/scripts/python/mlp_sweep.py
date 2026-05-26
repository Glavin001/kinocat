"""One-off sweep over residual MLP shapes / ensemble sizes.

Usage:

    python -m mlp_sweep \
        --trials   /tmp/round-N.npz \
        --init-params /tmp/params-in.json \
        --epochs 60

Reports val RMS + a rough forward-sim cost (params × ensemble_size) per
shape, so the elbow choice considers both quality and runtime. Per the
plan, "pick the smallest shape that's within 5% of the best val RMS".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from residual import train_ensemble  # noqa: E402
from train_fit import build_mlp_dataset, params_to_vec  # noqa: E402
from trial_io import load_trials  # noqa: E402

SHAPES = [
    [32, 32],
    [64, 64],
    [128, 128],
    [64, 64, 64],
    [128, 128, 128],
]
ENSEMBLES = [1, 3, 5]


def count_params(input_dim: int, hidden_dims: list[int], output_dim: int) -> int:
    n = 0
    dims = [input_dim, *hidden_dims, output_dim]
    for a, b in zip(dims[:-1], dims[1:]):
        n += a * b + b
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--init-params", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="mlp_sweep_results.json")
    args = ap.parse_args()

    trials = load_trials(args.trials)
    with open(args.init_params) as f:
        init_vec = params_to_vec(json.load(f))
    inputs, targets, split = build_mlp_dataset(trials, init_vec)
    n = inputs.shape[0]
    n_val = max(1, int(n * 0.2))
    train_inputs = inputs[:-n_val]
    train_targets = targets[:-n_val]
    val_inputs = inputs[-n_val:]
    val_targets = targets[-n_val:]

    results = []
    for shape in SHAPES:
        for ens in ENSEMBLES:
            t0 = time.time()
            _, history = train_ensemble(
                inputs=train_inputs, targets=train_targets,
                val_inputs=val_inputs, val_targets=val_targets,
                input_dim=inputs.shape[1], output_dim=6,
                hidden_dims=shape, ensemble_size=ens,
                epochs=args.epochs, verbose=False,
            )
            dt = time.time() - t0
            val_rms = float(np.sqrt(history["val_loss"][-1]))
            params_per_mlp = count_params(inputs.shape[1], shape, 6)
            sim_cost = params_per_mlp * ens
            results.append({
                "shape": shape, "ensemble": ens,
                "val_rms": val_rms,
                "params_per_mlp": params_per_mlp,
                "fwd_sim_cost": sim_cost,
                "train_time_s": dt,
            })
            print(f"shape={shape} ens={ens}  valRms={val_rms:.4f}  cost={sim_cost}  {dt:.1f}s")

    # Elbow pick: smallest fwd-sim cost within 5% of the best val_rms.
    best_rms = min(r["val_rms"] for r in results)
    candidates = [r for r in results if r["val_rms"] <= best_rms * 1.05]
    elbow = min(candidates, key=lambda r: r["fwd_sim_cost"])
    print(f"\nElbow pick: shape={elbow['shape']} ens={elbow['ensemble']}  (val={elbow['val_rms']:.4f}, cost={elbow['fwd_sim_cost']})")

    Path(args.out).write_text(json.dumps({"sweep": results, "elbow": elbow}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
