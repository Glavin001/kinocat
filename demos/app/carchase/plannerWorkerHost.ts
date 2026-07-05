import type {
  WorkerPlanRequest,
  WorkerPlanResponse,
  MainToWorker,
  WorkerToMain,
} from 'kinocat/worker';
import type { VehicleAgent } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import type { CarChaseCourse } from '../lib/carchase-scenarios';

/** Pool of planner workers, one per agent id. With N agents sharing a single
 *  worker the round-robin replan scheduler can only land one plan per slot —
 *  net per-agent cadence ≈ N × slot interval. A pool keyed by `npcId` lets all
 *  N plans of a round run in parallel, so each agent's replan rate matches the
 *  raw slot interval. The per-agent in-flight gate in the scheduler stays the
 *  authority on backpressure; the pool just removes the worker-side
 *  serialization. */
export class CarChasePlannerHost {
  private workers = new Map<string, Worker>();
  private resultCb: ((r: WorkerPlanResponse) => void) | null = null;

  async init(
    course: CarChaseCourse,
    agent: VehicleAgent,
    lib: MotionPrimitiveLibrary,
    npcIds: ReadonlyArray<string>,
  ): Promise<void> {
    const initMsg: MainToWorker = {
      type: 'init',
      polygons: course.polygons,
      obstacles: course.obstacles,
      agent,
      libJSON: lib.toJSON(),
      courseJSON: JSON.stringify(course),
    };

    const promises = npcIds.map(
      (id) =>
        new Promise<void>((resolve, reject) => {
          const w = new Worker(
            new URL('./carchase.worker.ts', import.meta.url),
            { type: 'module' },
          );
          this.workers.set(id, w);
          w.onerror = (err) => reject(err);
          w.onmessage = (e: MessageEvent<WorkerToMain>) => {
            if (e.data.type === 'init-ack') {
              w.onerror = null;
              w.onmessage = (ev: MessageEvent<WorkerToMain>) => {
                if (ev.data.type === 'plan-result') {
                  this.resultCb?.(ev.data);
                }
              };
              resolve();
            }
          };
          w.postMessage(initMsg);
        }),
    );

    await Promise.all(promises);
  }

  onResult(cb: (r: WorkerPlanResponse) => void): void {
    this.resultCb = cb;
  }

  requestPlan(req: WorkerPlanRequest): void {
    this.workers.get(req.npcId)?.postMessage(req);
  }

  dispose(): void {
    for (const w of this.workers.values()) w.terminate();
    this.workers.clear();
  }
}
