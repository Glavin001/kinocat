import type { VehicleState, VehicleAgent } from '../agent/types';
import type { NavPolygon } from '../environment/nav-world';
import type { PlanStats } from '../planner/types';

/** Serializable descriptor for a MovingObstacle. The worker reconstructs
 *  the predict closure from this data. */
export type ObstacleDescriptor =
  | { kind: 'plan'; path: VehicleState[]; radius: number }
  | { kind: 'cv'; state: VehicleState; horizon: number; radius: number };

export interface WorkerInitMsg {
  type: 'init';
  polygons: NavPolygon[];
  obstacles: Array<[number, number][]>;
  agent: VehicleAgent;
  libJSON: string;
  /** JSON.stringify of the demo-specific course object (all plain data). */
  courseJSON: string;
}

export interface WorkerInitAck {
  type: 'init-ack';
}

export interface WorkerPlanRequest {
  type: 'plan';
  reqId: number;
  npcId: string;
  start: VehicleState;
  goal: VehicleState;
  obstacles: ObstacleDescriptor[];
  deadlineMs?: number;
  maxExpansions?: number;
}

export interface WorkerPlanResponse {
  type: 'plan-result';
  reqId: number;
  npcId: string;
  found: boolean;
  cost: number;
  path: VehicleState[];
  stats: PlanStats;
}

export type MainToWorker = WorkerInitMsg | WorkerPlanRequest;
export type WorkerToMain = WorkerInitAck | WorkerPlanResponse;
