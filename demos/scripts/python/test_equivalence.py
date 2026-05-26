"""Pytest equivalence guard.

Pins the JAX `parametric_forward_v2` to a committed golden file. If
either the TS smooth variant (`parametricForwardV2Smooth`) or the JAX
implementation moves, the corresponding test on the other side fails
and the golden has to be regenerated deliberately.

Generate / refresh the golden:

    python -m test_equivalence --write-golden

Run the check (this is what pytest runs):

    pytest demos/scripts/python/test_equivalence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from parametric import parametric_forward_v2  # noqa: E402

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "core" / "test" / "agent" / "parametric-forward-v2-golden.json"
)


# 16-param defaults — must match DEFAULT_LEARNED_PARAMS_V2 in vehicle-model.ts.
DEFAULT_PARAMS = np.array([
    0.85, 0.9, 1.6, 0.2, 1.0, 1.2, 1.0,
    0.006, 0.002, 0.18,
    4.5, 0.6, 0.4, 0.02, 50.0, 0.05,
], dtype=np.float64)

# DEFAULT_LEARNABLE_CONFIG: chassisMass, wheelBase, frictionSlip (the prefix the model uses).
DEFAULT_CONFIG = np.array([1200.0, 1.25, 1.0], dtype=np.float64)


def golden_cases():
    """Fixed (state, controls, dt) tuples spanning the input distribution."""
    cases = []
    rng = np.random.default_rng(0)
    for _ in range(40):
        state = np.array([
            rng.uniform(-10, 10),     # x
            rng.uniform(-10, 10),     # z
            rng.uniform(-3.0, 3.0),   # heading
            rng.uniform(-15, 25),     # speed
            rng.uniform(-1.0, 1.0),   # yawRate
            rng.uniform(-2, 2),       # lateralVelocity
            rng.uniform(0, 10),       # t
        ], dtype=np.float64)
        controls = np.array([
            rng.uniform(-0.6, 0.6),
            rng.uniform(-3000, 4000),
            rng.uniform(0, 2000),
        ], dtype=np.float64)
        cases.append({"state": state.tolist(), "controls": controls.tolist(), "dt": 1 / 60})
    return cases


def compute_expected(cases):
    out = []
    for c in cases:
        s = jnp.asarray(c["state"])
        u = jnp.asarray(c["controls"])
        nxt = parametric_forward_v2(jnp.asarray(DEFAULT_PARAMS), jnp.asarray(DEFAULT_CONFIG), s, u, c["dt"])
        out.append([float(x) for x in nxt.tolist()])
    return out


def write_golden():
    cases = golden_cases()
    expected = compute_expected(cases)
    payload = {
        "schema": 1,
        "params": DEFAULT_PARAMS.tolist(),
        "param_names": [
            "engineScale", "reverseEffScale", "brakeScale", "accelTau",
            "gripScale", "frictionCircleSlack", "steerRatio",
            "understeerOffThrottle", "understeerPowerOn", "yawRateTau",
            "lateralDamping", "lateralFromSteer", "slipDrag",
            "loadTransferCoeff", "driveDeadzone", "rollingResistance",
        ],
        "config": DEFAULT_CONFIG.tolist(),
        "config_names": ["chassisMass", "wheelBase", "frictionSlip"],
        "cases": [
            {"state": c["state"], "controls": c["controls"], "dt": c["dt"], "expected": e}
            for c, e in zip(cases, expected)
        ],
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {GOLDEN_PATH}")


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="golden missing; run --write-golden")
def test_parametric_matches_golden():
    payload = json.loads(GOLDEN_PATH.read_text())
    params = jnp.asarray(payload["params"])
    config = jnp.asarray(payload["config"])
    for c in payload["cases"]:
        got = parametric_forward_v2(
            params, config,
            jnp.asarray(c["state"]), jnp.asarray(c["controls"]),
            c["dt"],
        )
        exp = np.asarray(c["expected"])
        np.testing.assert_allclose(np.asarray(got), exp, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-golden", action="store_true")
    args = ap.parse_args()
    if args.write_golden:
        write_golden()
    else:
        test_parametric_matches_golden()
        print("ok")
