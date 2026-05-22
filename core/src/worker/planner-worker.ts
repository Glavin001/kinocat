import type { WorkerPlanRequest, WorkerPlanResponse } from './protocol';
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
