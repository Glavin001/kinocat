import type {
  WorkerPlanRequest,
  WorkerPlanResponse,
  WorkerWorldUpdateMsg,
  WorkerWorldUpdateAck,
} from './protocol';
import type { OffMeshLink } from '../environment/nav-world';
import type { VehicleAgent } from '../agent/types';
import type { AffordanceRegistry } from '../predict/affordance-registry';
import type { NavWorld } from '../environment/nav-world';
import type { MotionPrimitiveLibrary } from '../primitives/library';
import { planVehicleOnce } from '../planner/plan-vehicle';
import { rehydrateObstacles } from './rehydrate';

export interface WorkerContext {
  world: NavWorld;
  agent: VehicleAgent;
  lib: MotionPrimitiveLibrary;
  affordances: AffordanceRegistry;
}

let ctx: WorkerContext | null = null;

export function initWorkerContext(c: WorkerContext): void {
  ctx = c;
}

/** Structural view of the mutators a data-in world delta needs. */
interface MutableWorld {
  setObstacles?(obstacles: Array<[number, number][]>): void;
  addOffMeshLink?(link: OffMeshLink): void;
  bumpRevision?(): void;
}

/** Apply a live world delta to the worker's long-lived world — no re-init.
 *  Throws when a data delta targets a world without the matching mutator:
 *  silently skipping would leave the worker planning against stale geometry,
 *  which is exactly the failure this message exists to prevent. */
export function handleWorldUpdateMessage(
  msg: WorkerWorldUpdateMsg,
  postResponse: (r: WorkerWorldUpdateAck) => void,
): void {
  if (!ctx) throw new Error('Worker not initialized');
  const w = ctx.world as MutableWorld & { revision: number };
  if (msg.obstacles) {
    if (!w.setObstacles) {
      throw new Error('world-update: world does not support setObstacles');
    }
    w.setObstacles(msg.obstacles);
  }
  if (msg.addOffMeshLinks) {
    if (!w.addOffMeshLink) {
      throw new Error('world-update: world does not support addOffMeshLink');
    }
    for (const link of msg.addOffMeshLinks) w.addOffMeshLink(link);
  }
  if (msg.bumpRevisionOnly && !msg.obstacles && !msg.addOffMeshLinks) {
    if (!w.bumpRevision) {
      throw new Error('world-update: world does not support bumpRevision');
    }
    w.bumpRevision();
  }
  postResponse({ type: 'world-update-ack', seq: msg.seq, revision: w.revision });
}

export function handlePlanMessage(
  msg: WorkerPlanRequest,
  postResponse: (r: WorkerPlanResponse) => void,
): void {
  if (!ctx) throw new Error('Worker not initialized');
  const obstacles = rehydrateObstacles(msg.obstacles);
  const result = planVehicleOnce({
    start: msg.start,
    goal: msg.goal,
    world: ctx.world,
    agent: ctx.agent,
    lib: ctx.lib,
    movingObstacles: obstacles,
    affordances: ctx.affordances,
    timeOptions: { affordanceRadius: 10 },
    deadlineMs: msg.deadlineMs ?? 120,
    maxExpansions: msg.maxExpansions ?? 25000,
  });
  postResponse({
    type: 'plan-result',
    reqId: msg.reqId,
    npcId: msg.npcId,
    found: result.found,
    cost: result.cost,
    path: result.path,
    stats: result.stats,
  });
}
