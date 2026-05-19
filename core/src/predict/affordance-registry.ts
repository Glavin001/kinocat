// Affordances produce transitions the drive primitives can't (ramps/ballistic
// jumps, boost pads, elevators, moving platforms). Collisions are hard
// constraints; affordances are *extra edges*. They are generated lazily at
// Environment.succ time only when nearby and usable at the corresponding time.

import type { Predict, AffordanceState } from './types';
import type { VehicleState } from '../agent/types';

export enum AffordanceType {
  BallisticJump = 'ballistic_jump',
  BoostPad = 'boost_pad',
  Elevator = 'elevator',
  MovingPlatform = 'moving_platform',
  Teleporter = 'teleporter',
  /** A decoy: spatially looks like a shortcut but its honest cost makes the
   *  honest route cheaper. The planner rejects it on its own (no special
   *  code) — emergent intelligence. Distinct type so demos can render it. */
  Decoy = 'decoy',
}

export interface AffordanceUseResult {
  resultState: VehicleState;
  /** Sampled (x, y, z, t) along the use, for tracking/visualization. */
  trajectory: Array<{ x: number; y: number; z: number; t: number }>;
  duration: number;
  /** Additional g-cost beyond the agent's own motion. */
  cost: number;
}

export interface Affordance {
  id: string;
  type: AffordanceType;
  predict: Predict<AffordanceState>;
  validFrom: number;
  validTo: number;
  /** Proximity bound (world XZ + radius) for `queryNearby`. */
  spatialBound: { x: number; z: number; radius: number };
  /** Try to use this affordance from `agentState` at `useTime`. */
  tryUse(agentState: VehicleState, useTime: number): AffordanceUseResult | null;
}

export class AffordanceRegistry {
  private readonly items = new Map<string, Affordance>();

  add(a: Affordance): void {
    this.items.set(a.id, a);
  }

  remove(id: string): void {
    this.items.delete(id);
  }

  all(): Affordance[] {
    return [...this.items.values()];
  }

  /** Affordances usable near (x,z) at time `t` (within `radius`). */
  queryNearby(x: number, z: number, t: number, radius = 15): Affordance[] {
    const out: Affordance[] = [];
    for (const a of this.items.values()) {
      if (t < a.validFrom || t > a.validTo) continue;
      const b = a.spatialBound;
      const dx = x - b.x;
      const dz = z - b.z;
      const rr = radius + b.radius;
      if (dx * dx + dz * dz <= rr * rr) out.push(a);
    }
    return out;
  }
}

/**
 * A fixed launch→land jump affordance — the runtime shape of a Mononen-baked
 * static off-mesh jump (M8) and a usable standalone gap-crossing edge. Usable
 * when the agent is within `entryRadius` of the launch point during the
 * validity window; lands the agent at `land` after `duration`.
 */
export function createJumpAffordance(opts: {
  id: string;
  launch: { x: number; z: number };
  entryRadius: number;
  land: VehicleState;
  apexY?: number;
  duration: number;
  cost: number;
  validFrom?: number;
  validTo?: number;
}): Affordance {
  const apexY = opts.apexY ?? 2;
  const validFrom = opts.validFrom ?? -Infinity;
  const validTo = opts.validTo ?? Infinity;
  return {
    id: opts.id,
    type: AffordanceType.BallisticJump,
    validFrom,
    validTo,
    spatialBound: { x: opts.launch.x, z: opts.launch.z, radius: opts.entryRadius },
    predict: (t) =>
      t < validFrom || t > validTo
        ? null
        : { position: { x: opts.launch.x, y: 0, z: opts.launch.z } },
    tryUse(agentState, useTime) {
      if (useTime < validFrom || useTime > validTo) return null;
      const dx = agentState.x - opts.launch.x;
      const dz = agentState.z - opts.launch.z;
      if (dx * dx + dz * dz > opts.entryRadius * opts.entryRadius) return null;
      const land: VehicleState = { ...opts.land, t: useTime + opts.duration };
      return {
        resultState: land,
        duration: opts.duration,
        cost: opts.cost,
        trajectory: [
          { x: agentState.x, y: 0, z: agentState.z, t: useTime },
          {
            x: (agentState.x + land.x) / 2,
            y: apexY,
            z: (agentState.z + land.z) / 2,
            t: useTime + opts.duration / 2,
          },
          { x: land.x, y: 0, z: land.z, t: land.t },
        ],
      };
    },
  };
}

/**
 * A genuine shortcut: a boost pad that flings the agent to `exit` (typically
 * further along, at higher `exit.speed`) for a deliberately LOW `cost`, so
 * the true route cost through it beats driving — IGHA* adopts it.
 */
export function createBoostAffordance(opts: {
  id: string;
  pad: { x: number; z: number };
  entryRadius: number;
  exit: VehicleState;
  duration: number;
  cost: number;
  validFrom?: number;
  validTo?: number;
}): Affordance {
  const validFrom = opts.validFrom ?? -Infinity;
  const validTo = opts.validTo ?? Infinity;
  return {
    id: opts.id,
    type: AffordanceType.BoostPad,
    validFrom,
    validTo,
    spatialBound: { x: opts.pad.x, z: opts.pad.z, radius: opts.entryRadius },
    predict: (t) =>
      t < validFrom || t > validTo
        ? null
        : { position: { x: opts.pad.x, y: 0, z: opts.pad.z } },
    tryUse(agentState, useTime) {
      if (useTime < validFrom || useTime > validTo) return null;
      const dx = agentState.x - opts.pad.x;
      const dz = agentState.z - opts.pad.z;
      if (dx * dx + dz * dz > opts.entryRadius * opts.entryRadius) return null;
      const out: VehicleState = { ...opts.exit, t: useTime + opts.duration };
      return {
        resultState: out,
        duration: opts.duration,
        cost: opts.cost,
        trajectory: [
          { x: agentState.x, y: 0, z: agentState.z, t: useTime },
          { x: out.x, y: 0, z: out.z, t: out.t },
        ],
      };
    },
  };
}

/**
 * A *misdirect*: spatially it looks like a tempting shortcut (place `launch`
 * near the optimal route and `land` toward the goal), but its honest `cost`
 * and/or dead-end `land` make the no-affordance route cheaper. The factory
 * reports the real cost — so with an admissible heuristic IGHA*'s
 * branch-and-bound prunes the branch with NO special-case logic. This is the
 * emergent-intelligence demonstrator; do not special-case it elsewhere.
 */
export function createMisdirectAffordance(opts: {
  id: string;
  launch: { x: number; z: number };
  entryRadius: number;
  /** Where it actually drops the agent (a trap / dead-end / detour). */
  land: VehicleState;
  duration: number;
  /** The true (high) cost — reported honestly so the planner rejects it. */
  cost: number;
  apexY?: number;
  validFrom?: number;
  validTo?: number;
}): Affordance {
  const apexY = opts.apexY ?? 2;
  const validFrom = opts.validFrom ?? -Infinity;
  const validTo = opts.validTo ?? Infinity;
  return {
    id: opts.id,
    type: AffordanceType.Decoy,
    validFrom,
    validTo,
    spatialBound: { x: opts.launch.x, z: opts.launch.z, radius: opts.entryRadius },
    predict: (t) =>
      t < validFrom || t > validTo
        ? null
        : { position: { x: opts.launch.x, y: 0, z: opts.launch.z } },
    tryUse(agentState, useTime) {
      if (useTime < validFrom || useTime > validTo) return null;
      const dx = agentState.x - opts.launch.x;
      const dz = agentState.z - opts.launch.z;
      if (dx * dx + dz * dz > opts.entryRadius * opts.entryRadius) return null;
      const land: VehicleState = { ...opts.land, t: useTime + opts.duration };
      return {
        resultState: land,
        duration: opts.duration,
        cost: opts.cost,
        trajectory: [
          { x: agentState.x, y: 0, z: agentState.z, t: useTime },
          {
            x: (agentState.x + land.x) / 2,
            y: apexY,
            z: (agentState.z + land.z) / 2,
            t: useTime + opts.duration / 2,
          },
          { x: land.x, y: 0, z: land.z, t: land.t },
        ],
      };
    },
  };
}
