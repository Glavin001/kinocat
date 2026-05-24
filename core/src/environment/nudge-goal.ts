// Generic "walk a goal until it fits" helper. The AI tactical layer in games
// often proposes a goal pose by geometry (e.g. "15 m behind the target along
// its heading"); on a cluttered map those goals can land inside a building's
// inflated footprint and the planner rejects them outright (0 expansions,
// empty plan). `nudgeGoalClear` walks the goal back along the ray from
// `near` (typically the agent's current position) in fixed steps until the
// agent's rotated footprint at that pose is collision-free under the same
// `NavWorld.footprintClear` predicate the planner will use, so a goal
// accepted by this helper is guaranteed to pass `Environment.checkValidity`.
//
// Bounded by `maxSteps` so a degenerate layout can never spin forever — falls
// back to `near` (always accepted: the agent is currently there, more or
// less). The returned goal is otherwise unchanged (heading, speed, t, …).

import type { VehicleAgent } from '../agent/types';
import { placeFootprint } from '../internal/geom';
import type { NavWorld } from './nav-world';

export interface NudgeGoalOptions {
  /** Step size (world units) along the (goal → near) ray. Default 4. */
  step?: number;
  /** Hard cap on iterations (otherwise: ⌈distance / step⌉). */
  maxSteps?: number;
}

interface XZHeading {
  x: number;
  z: number;
  heading: number;
}

/** Walk `goal` toward `near` in fixed `step` increments until the agent's
 *  rotated footprint at `(goal.x, goal.z, goal.heading)` is accepted by
 *  `world.footprintClear`. Generic over any state shape extending
 *  `{ x, z, heading }` so it works for `CarKinematicState` and other planner
 *  states; non-position fields are preserved.
 *
 *  Returns the original `goal` immediately if it is already clear. Falls
 *  back to a goal at `near` (with the original heading and other fields) if
 *  `maxSteps` is exhausted without finding a clear pose. */
export function nudgeGoalClear<S extends XZHeading>(
  goal: S,
  near: { x: number; z: number },
  world: NavWorld,
  agent: VehicleAgent,
  opts: NudgeGoalOptions = {},
): S {
  const step = opts.step ?? 4;
  if (clearAt(goal.x, goal.z, goal.heading, world, agent)) return goal;

  const dx = near.x - goal.x;
  const dz = near.z - goal.z;
  const total = Math.hypot(dx, dz);
  if (total < 1e-6) return goal;
  const ux = dx / total;
  const uz = dz / total;
  const maxSteps = opts.maxSteps ?? Math.max(4, Math.ceil(total / step));
  for (let i = 1; i <= maxSteps; i++) {
    const nx = goal.x + ux * step * i;
    const nz = goal.z + uz * step * i;
    if (clearAt(nx, nz, goal.heading, world, agent)) {
      return { ...goal, x: nx, z: nz };
    }
  }
  return { ...goal, x: near.x, z: near.z };
}

function clearAt(
  x: number,
  z: number,
  heading: number,
  world: NavWorld,
  agent: VehicleAgent,
): boolean {
  return world.footprintClear(placeFootprint(agent.footprint, x, z, heading));
}
