import {
  PlannerPool,
  type PlanRequestBody,
  type WorkerLike,
  type WorkerInitMsg,
  type WorkerPlanResponse,
  type WorkerWorldUpdateMsg,
} from 'kinocat/worker';
import type { VehicleAgent } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import type { CarChaseCourse } from '../lib/carchase-scenarios';

/** Thin browser binding of the core `PlannerPool` (one worker per agent —
 *  see the pool's docs for the scheduling/backpressure contract). All this
 *  file owns is what core cannot: constructing the bundler-resolved Worker
 *  and packing the carchase init message. */
export class CarChasePlannerHost {
  private readonly pool = new PlannerPool(
    () =>
      new Worker(new URL('./carchase.worker.ts', import.meta.url), {
        type: 'module',
      }) as unknown as WorkerLike,
  );

  init(
    course: CarChaseCourse,
    agent: VehicleAgent,
    lib: MotionPrimitiveLibrary,
    npcIds: ReadonlyArray<string>,
  ): Promise<void> {
    const initMsg: WorkerInitMsg = {
      type: 'init',
      polygons: course.polygons,
      obstacles: course.obstacles,
      agent,
      libJSON: lib.toJSON(),
      courseJSON: JSON.stringify(course),
    };
    return this.pool.init(initMsg, npcIds);
  }

  onResult(cb: (r: WorkerPlanResponse, elapsedMs: number) => void): void {
    this.pool.onResult(cb);
  }

  requestPlan(agentId: string, body: PlanRequestBody): boolean {
    return this.pool.requestPlan(agentId, body);
  }

  hasInflight(agentId: string): boolean {
    return this.pool.hasInflight(agentId);
  }

  /** Push a live world change to every planner worker (no re-init). */
  broadcast(update: Omit<WorkerWorldUpdateMsg, 'type' | 'seq'>): Promise<void> {
    return this.pool.broadcast(update);
  }

  dispose(): void {
    this.pool.dispose();
  }
}
