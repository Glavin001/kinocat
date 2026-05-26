"""JAX port of `parametricForwardV2Smooth` (core/src/agent/vehicle-model.ts).

Op-for-op equivalent: the smoothing constants below MUST match the TS
side (V_EPS, F_EPS, ABS_EPS, DEADZONE_K) or the golden equivalence test
will fail. The TS implementation is the canonical reference; this is the
training target.

State layout (jnp arrays, shape `(7,)`):
    [x, z, heading, speed, yawRate, lateralVelocity, t]

Controls (shape `(3,)`):
    [steer, driveForce, brakeForce]

Config (shape `(2,)` at minimum):
    [chassisMass, wheelBase, frictionSlip]  — `LearnableVehicleConfig` ordinal.

Params (shape `(16,)`): order matches `PARAMS_V2_ORDER` in vehicle-model.ts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Smoothing constants — keep in sync with vehicle-model.ts.
V_EPS = 0.1
F_EPS = 5.0
ABS_EPS = 1e-3
DEADZONE_K = 0.1
BRAKE_GATE_K = 5.0
G = 9.81

# 16-param order. Matches PARAMS_V2_ORDER in core/src/agent/vehicle-model.ts.
PARAM_NAMES = (
    "engineScale", "reverseEffScale", "brakeScale", "accelTau",
    "gripScale", "frictionCircleSlack", "steerRatio",
    "understeerOffThrottle", "understeerPowerOn", "yawRateTau",
    "lateralDamping", "lateralFromSteer", "slipDrag",
    "loadTransferCoeff", "driveDeadzone", "rollingResistance",
)

# Config order. Matches the (chassisMass, wheelBase, frictionSlip) prefix
# that the parametric forward actually reads. Other config fields are
# carried in `config` but unused here.
CONFIG_NAMES = ("chassisMass", "wheelBase", "frictionSlip")


def soft_abs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(x * x + ABS_EPS * ABS_EPS)


def soft_sign(x: jnp.ndarray, eps: float) -> jnp.ndarray:
    return jnp.tanh(x / eps)


def softplus_thresh(x: jnp.ndarray, thr: jnp.ndarray, k: float) -> jnp.ndarray:
    z = k * (x - thr)
    # numerically stable softplus
    return jnp.where(z > 30.0, z, jnp.log1p(jnp.exp(jnp.minimum(z, 30.0)))) / k


def wrap_angle(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(a + jnp.pi, 2 * jnp.pi) - jnp.pi


def parametric_forward_v2(
    params: jnp.ndarray,    # (16,)
    config: jnp.ndarray,    # (3,) [mass, wheelBase, frictionSlip]
    state: jnp.ndarray,     # (7,) [x, z, heading, speed, yawRate, vy, t]
    controls: jnp.ndarray,  # (3,) [steer, driveForce, brakeForce]
    dt: float,
) -> jnp.ndarray:
    """Single-step smooth differentiable forward sim. Returns next state (7,)."""
    (engineScale, reverseEffScale, brakeScale, accelTau,
     gripScale, frictionCircleSlack, steerRatio,
     understeerOffThrottle, understeerPowerOn, yawRateTau,
     lateralDamping, lateralFromSteer, slipDrag,
     _loadTransferCoeff, driveDeadzone, rollingResistance) = (
        params[i] for i in range(16)
    )
    mass, wheelBase, frictionSlip = config[0], config[1], config[2]
    m = jnp.maximum(50.0, mass)

    x, z, heading, v, yaw_rate, vy, t = (state[i] for i in range(7))
    steer, drive_force, brake_force = controls[0], controls[1], controls[2]

    # --- Longitudinal command ---
    f_abs = soft_abs(drive_force)
    f_eff_mag = softplus_thresh(f_abs, driveDeadzone, DEADZONE_K)
    f_eff_sign = soft_sign(drive_force, F_EPS)
    f_eff = f_eff_sign * f_eff_mag
    fwd_mix = 0.5 * (1 + soft_sign(drive_force, F_EPS))
    dir_ = engineScale * (fwd_mix + (1 - fwd_mix) * reverseEffScale)
    drive_accel = (dir_ * f_eff) / m

    brake_accel = (brakeScale * brake_force) / m
    v_sign = soft_sign(v, V_EPS)
    brake_signed = -v_sign * brake_accel
    rolling = -v_sign * rollingResistance * soft_abs(v)
    a_long = drive_accel + brake_signed + rolling

    # --- Steer → bicycle yaw rate ---
    eff_steer = steer * steerRatio
    L = jnp.maximum(0.5, 2.0 * wheelBase)
    yaw_rate_cmd_raw = (v * jnp.sin(eff_steer)) / L
    power_on_gate = 0.5 * (1 + soft_sign(drive_force, F_EPS)) * 0.5 * (1 + soft_sign(v, V_EPS))
    ku = power_on_gate * understeerPowerOn + (1 - power_on_gate) * understeerOffThrottle
    yaw_rate_cmd = yaw_rate_cmd_raw / (1 + ku * v * v)

    # --- Friction-circle smooth saturation ---
    aMax = gripScale * frictionSlip * G * frictionCircleSlack
    a_lat_est = v * yaw_rate_cmd
    mag = jnp.sqrt(a_long * a_long + a_lat_est * a_lat_est + ABS_EPS * ABS_EPS)
    u = mag / jnp.maximum(aMax, 1e-6)
    scale = jnp.tanh(u) / jnp.maximum(u, 1e-9)
    a_long = a_long * scale
    yaw_rate_allowed = yaw_rate_cmd * scale

    # --- Speed dynamics ---
    tau = jnp.maximum(0.02, accelTau)
    a_eff = a_long * (1 - jnp.exp(-dt / tau)) + a_long * jnp.exp(-dt / tau)
    slip_loss = -slipDrag * soft_abs(vy) * v_sign * dt
    speed_pre = v + a_eff * dt + slip_loss
    brake_on = 0.5 * (1 + jnp.tanh(brake_force * 0.01))
    near_zero = 0.5 * (1 - jnp.tanh((soft_abs(v) - 0.05) * BRAKE_GATE_K))
    flip_gate = brake_on * near_zero
    speed = (1 - flip_gate) * speed_pre + flip_gate * 0.0

    # --- Yaw-rate inertia ---
    yTau = jnp.maximum(0.02, yawRateTau)
    yaw_rate_next = yaw_rate + ((yaw_rate_allowed - yaw_rate) * dt) / yTau

    # --- Lateral velocity ---
    vy_drive = lateralFromSteer * eff_steer * v
    vy_next = vy + (vy_drive - vy * lateralDamping) * dt

    # --- Integrate pose ---
    speed_avg = 0.5 * (v + speed)
    yaw_avg = 0.5 * (yaw_rate + yaw_rate_next)
    heading_next = wrap_angle(heading + yaw_avg * dt)
    cosH = jnp.cos(heading)
    sinH = jnp.sin(heading)
    vy_avg = 0.5 * (vy + vy_next)
    dx = (speed_avg * cosH + vy_avg * jnp.sin(heading)) * dt
    dz = (speed_avg * sinH - vy_avg * jnp.cos(heading)) * dt

    return jnp.stack([x + dx, z + dz, heading_next, speed, yaw_rate_next, vy_next, t + dt])


def rollout_trial(
    params: jnp.ndarray,
    config: jnp.ndarray,
    init_state: jnp.ndarray,       # (7,)
    controls_trace: jnp.ndarray,   # (T, 3) — one per physics tick
    dt: float,
    sample_every: int,
    reseat_horizon: int,           # >0 = re-inject ground truth every N samples
    ground_truth_samples: jnp.ndarray | None = None,  # (S, 7) or None
) -> jnp.ndarray:
    """Roll out a trial, returning predicted samples at the sample boundaries.

    Returns array shape `(S, 7)` where `S = T // sample_every`.

    If `reseat_horizon > 0` and `ground_truth_samples` is provided, the
    state is re-injected from ground truth every `reseat_horizon` sample
    boundaries — same trajectory horizon reseating that the JS trainer
    uses (see training-driver.ts, trajectoryHorizon=10).
    """
    T = controls_trace.shape[0]
    S = T // sample_every

    def physics_step(s, c):
        return parametric_forward_v2(params, config, s, c, dt), None

    def sample_step(carry, idx):
        state, _ = carry, None
        # Roll `sample_every` physics ticks from current state.
        ctrl_slice = jax.lax.dynamic_slice_in_dim(controls_trace, idx * sample_every, sample_every)
        state_after, _ = jax.lax.scan(physics_step, state, ctrl_slice)
        # Optional reseating.
        if ground_truth_samples is not None and reseat_horizon > 0:
            should_reseat = ((idx + 1) % reseat_horizon == 0) & (idx + 1 < S)
            gt = ground_truth_samples[idx + 1]
            state_next = jnp.where(should_reseat, gt, state_after)
        else:
            state_next = state_after
        return state_next, state_after

    _, samples = jax.lax.scan(sample_step, init_state, jnp.arange(S))
    return samples
