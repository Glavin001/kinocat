import type {
  WorkerPlanRequest,
  WorkerPlanResponse,
  MainToWorker,
  WorkerToMain,
} from 'kinocat/worker';
import type { VehicleAgent } from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import type { CarChaseCourse } from '../lib/carchase-scenarios';

export class CarChasePlannerHost {
  private worker: Worker | null = null;
  private resultCb: ((r: WorkerPlanResponse) => void) | null = null;

  async init(
    course: CarChaseCourse,
    agent: VehicleAgent,
    lib: MotionPrimitiveLibrary,
  ): Promise<void> {
    this.worker = new Worker(
      new URL('./carchase.worker.ts', import.meta.url),
      { type: 'module' },
    );

    return new Promise<void>((resolve, reject) => {
      this.worker!.onerror = (err) => reject(err);
      this.worker!.onmessage = (e: MessageEvent<WorkerToMain>) => {
        if (e.data.type === 'init-ack') {
          this.worker!.onerror = null;
          this.worker!.onmessage = (ev: MessageEvent<WorkerToMain>) => {
            if (ev.data.type === 'plan-result') {
              this.resultCb?.(ev.data);
            }
          };
          resolve();
        }
      };

      const initMsg: MainToWorker = {
        type: 'init',
        polygons: course.polygons,
        obstacles: course.obstacles,
        agent,
        libJSON: lib.toJSON(),
        courseJSON: JSON.stringify(course),
      };
      this.worker!.postMessage(initMsg);
    });
  }

  onResult(cb: (r: WorkerPlanResponse) => void): void {
    this.resultCb = cb;
  }

  requestPlan(req: WorkerPlanRequest): void {
    this.worker?.postMessage(req);
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
