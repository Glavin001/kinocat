// Momentum-humanoid dynamics: an inertial point-mass with human movement
// envelopes. Velocity is world-frame and separate from facing; the envelope
// couples them — full sprint only along the facing, a strafe cap sideways/
// backward, launch weaker than braking, and a turn rate that degrades with
// speed. Pure, translation- and yaw-equivariant (see the characterize()
// contract), so primitives cached from a canonical start transform rigidly.

import type { ForwardSim } from '../primitives/types';
import type { MomentumHumanoidAgent, MomentumHumanoidState } from './types';
import { wrapAngle } from '../internal/math';

export function defaultMomentumHumanoidAgent(
  overrides: Partial<Omit<MomentumHumanoidAgent, 'kind'>> = {},
): MomentumHumanoidAgent {
  return {
    kind: 'momentum-humanoid',
    radius: 0.35,
    maxSpeed: 5,
    strafeSpeed: 2,
    maxAccel: 3,
    maxDecel: 6,
    maxTurnRate: Math.PI * 1.5,
    ...overrides,
  };
}

/** At sprint the body can only carve, not pivot: linear falloff from the
 *  at-rest rate down to 30% of it at maxSpeed. */
export function turnRateAt(agent: MomentumHumanoidAgent, speed: number): number {
  const s = Math.min(1, Math.abs(speed) / agent.maxSpeed);
  return agent.maxTurnRate * (1 - 0.7 * s);
}

/**
 * Controls (all optional, default 0):
 *   [0] accelFrac — commanded acceleration magnitude as a fraction (0..1).
 *   [1] accelDir  — acceleration direction RELATIVE to the facing (rad);
 *                   0 = forward, π = brake/backpedal, ±π/2 = strafe.
 *   [2] turnFrac  — turn-rate command (-1..1) of the speed-degraded rate.
 *
 * The magnitude limit is direction-aware: components opposing the current
 * velocity may use maxDecel, others maxAccel. After integration the velocity
 * is clamped to the human envelope: |v| ≤ maxSpeed overall, and the
 * non-facing part (lateral or backward) to strafeSpeed.
 */
export function momentumHumanoidForwardSim(
  agent: MomentumHumanoidAgent,
): ForwardSim<MomentumHumanoidState> {
  return (state, controls, dt) => {
    const aFrac = Math.max(0, Math.min(1, controls[0] ?? 0));
    const aDir = controls[1] ?? 0;
    const turnFrac = Math.max(-1, Math.min(1, controls[2] ?? 0));

    const speed0 = Math.hypot(state.vx, state.vz);
    const heading = wrapAngle(
      state.heading + turnFrac * turnRateAt(agent, speed0) * dt,
    );

    // Acceleration in world frame; braking limit when opposing the motion.
    const aWorldDir = heading + aDir;
    const ax = Math.cos(aWorldDir);
    const az = Math.sin(aWorldDir);
    const opposing = speed0 > 1e-9 && ax * state.vx + az * state.vz < 0;
    const aMag = aFrac * (opposing ? agent.maxDecel : agent.maxAccel);

    let vx = state.vx + ax * aMag * dt;
    let vz = state.vz + az * aMag * dt;
    // Braking must stop, not reverse: if the commanded accel opposed the
    // motion and the velocity flipped sign along it, clamp to rest.
    if (opposing && vx * state.vx + vz * state.vz < 0) {
      vx = 0;
      vz = 0;
    }

    // Envelope clamp. Decompose onto the (new) facing.
    const fx = Math.cos(heading);
    const fz = Math.sin(heading);
    let vFwd = vx * fx + vz * fz;
    let vLat = -vx * fz + vz * fx;
    vFwd = Math.max(-agent.strafeSpeed, Math.min(agent.maxSpeed, vFwd));
    vLat = Math.max(-agent.strafeSpeed, Math.min(agent.strafeSpeed, vLat));
    vx = vFwd * fx - vLat * fz;
    vz = vFwd * fz + vLat * fx;
    const speed = Math.hypot(vx, vz);
    if (speed > agent.maxSpeed) {
      const k = agent.maxSpeed / speed;
      vx *= k;
      vz *= k;
    }

    return {
      x: state.x + vx * dt,
      z: state.z + vz * dt,
      heading,
      vx,
      vz,
      t: state.t + dt,
    };
  };
}
