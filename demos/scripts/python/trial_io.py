"""Reader for the trial-bundle npz format emitted by Node.

Schema (versioned; bump TRIAL_NPZ_VERSION on incompatible changes):

    init_states      : float64 (N, 7)        — [x, z, heading, speed, yawRate, vy, t]
    controls_trace   : float64 (N, T, 3)     — [steer, drive, brake] per physics tick
    samples          : float64 (N, S, 7)     — observed states at sample boundaries
    sample_times     : float64 (N, S)        — t at each sample
    config           : float64 (N, 13)       — full encodeConfigOneHot vector
    split            : int32   (N,)          — 0=train, 1=val, 2=test
    dt               : float64 ()            — physics tick (e.g. 1/60)
    sample_every     : int32 ()              — ticks per sample (== T / S)
    version          : int32 ()              — TRIAL_NPZ_VERSION

config layout (13 dims, matches encodeConfigOneHot in
vehicle-config.ts):
    [0]  chassisMass
    [1]  wheelBase
    [2]  wheelTrack
    [3]  wheelRadius
    [4]  suspensionStiffness
    [5]  frictionSlip
    [6]  sideFrictionStiffness
    [7]  maxDriveForce
    [8]  maxBrakeForce
    [9]  maxSteerAngle
    [10] drivenWheels == 'rwd'
    [11] drivenWheels == 'fwd'
    [12] drivenWheels == 'awd'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRIAL_NPZ_VERSION = 2

# Indices into the 13-dim config vector that the parametric forward uses
# (chassisMass, wheelBase, frictionSlip).
PARAMETRIC_CONFIG_INDICES = (0, 1, 5)

# Per-dim normalisation scales for the config (matches
# CONFIG_SCALES_ORDINAL in vehicle-config.ts; one-hot dims share the
# trailing scale=2).
CONFIG_SCALES = np.array([
    1000.0, 2.0, 1.2, 0.5, 150.0,
    3.0, 2.0, 8000.0, 4000.0, 1.2,
    2.0, 2.0, 2.0,
])


@dataclass(frozen=True)
class Trials:
    init_states: np.ndarray      # (N, 7)
    controls_trace: np.ndarray   # (N, T, 3)
    samples: np.ndarray          # (N, S, 7)
    sample_times: np.ndarray     # (N, S)
    config: np.ndarray           # (N, 3)
    split: np.ndarray            # (N,)
    dt: float
    sample_every: int

    @property
    def n(self) -> int:
        return int(self.init_states.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1])


def load_trials(path: str) -> Trials:
    z = np.load(path, allow_pickle=False)
    version = int(z["version"])
    if version != TRIAL_NPZ_VERSION:
        raise ValueError(
            f"Trial npz version mismatch: got {version}, expected {TRIAL_NPZ_VERSION}. "
            f"Update Node-side writer (demos/scripts/lib/trial-npz.ts) or "
            f"this reader together."
        )
    return Trials(
        init_states=z["init_states"].astype(np.float64),
        controls_trace=z["controls_trace"].astype(np.float64),
        samples=z["samples"].astype(np.float64),
        sample_times=z["sample_times"].astype(np.float64),
        config=z["config"].astype(np.float64),
        split=z["split"].astype(np.int32),
        dt=float(z["dt"]),
        sample_every=int(z["sample_every"]),
    )


def save_mlp_ensemble(path: str, ensemble: list[list[tuple[np.ndarray, np.ndarray]]]) -> None:
    """Save an MLP ensemble as npz.

    `ensemble` is a list (one entry per ensemble member) of layers,
    each `(W, b)`. The on-disk layout writes per-member per-layer:

        m{i}_l{j}_W : (out, in) float64
        m{i}_l{j}_b : (out,)    float64
        n_members   : int32
        n_layers    : int32
        version     : int32
    """
    out: dict[str, np.ndarray] = {
        "n_members": np.int32(len(ensemble)),
        "n_layers": np.int32(len(ensemble[0])) if ensemble else np.int32(0),
        "version": np.int32(1),
    }
    for i, member in enumerate(ensemble):
        for j, (W, b) in enumerate(member):
            out[f"m{i}_l{j}_W"] = W.astype(np.float64)
            out[f"m{i}_l{j}_b"] = b.astype(np.float64)
    np.savez(path, **out)
