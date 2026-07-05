import type { CarKinematicState, VehicleAgent } from '../agent/types';
import type { NavPolygon, OffMeshLink } from '../environment/nav-world';
import type { PlanStats } from '../planner/types';

/** Serializable descriptor for a MovingObstacle. The worker reconstructs
 *  the predict closure from this data. */
export type ObstacleDescriptor =
  | { kind: 'plan'; path: CarKinematicState[]; radius: number }
  | { kind: 'cv'; state: CarKinematicState; horizon: number; radius: number };

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
  start: CarKinematicState;
  goal: CarKinematicState;
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
  path: CarKinematicState[];
  stats: PlanStats;
}

/** Live world delta — lets a long-lived worker consume new tiles / obstacle
 *  sets / off-mesh links WITHOUT a re-init. Data-in deltas target worlds with
 *  mutators (`InMemoryNavWorld.setObstacles` / `addOffMeshLink`); worlds whose
 *  geometry is swapped externally (NavcatWorld) use `bumpRevisionOnly`. */
export interface WorkerWorldUpdateMsg {
  type: 'world-update';
  /** Host-monotonic sequence number, echoed in the ack. */
  seq: number;
  /** Full replacement obstacle set (the world rebuilds its indices anyway). */
  obstacles?: Array<[number, number][]>;
  /** Off-mesh links to append (plain data). */
  addOffMeshLinks?: OffMeshLink[];
  /** Invalidate caches without a data delta. */
  bumpRevisionOnly?: boolean;
}

export interface WorkerWorldUpdateAck {
  type: 'world-update-ack';
  seq: number;
  revision: number;
}

export type MainToWorker = WorkerInitMsg | WorkerPlanRequest | WorkerWorldUpdateMsg;
export type WorkerToMain = WorkerInitAck | WorkerPlanResponse | WorkerWorldUpdateAck;
