// Tiny worker-pool wrapper for parallel Rapier trial collection.
//
// Used by train.ts when `--workers=N` is set. Falls back to the existing
// single-threaded `collectManeuverBatch` when not enabled.
//
// API:
//
//     const pool = await TrialWorkerPool.create({ size: 4 });
//     const trials = await pool.runShards(allSpecs);
//     await pool.dispose();
//
// Note: ManeuverSpec.build() (which closes over RNGs) MUST be expanded
// into static `TrialSpec` objects in the main thread before shipping —
// the worker only sees plain data.

import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

export interface CollectedTrialRaw {
  id: string;
  dt: number;
  samples: unknown[];
  config: unknown;
}

interface WorkerSlot {
  worker: Worker;
  ready: Promise<void>;
  busy: boolean;
}

export interface TrialWorkerPoolOptions {
  size?: number;
  workerScript?: string;
  vehicleOptions?: unknown;
}

export class TrialWorkerPool {
  private slots: WorkerSlot[] = [];

  static async create(opts: TrialWorkerPoolOptions = {}): Promise<TrialWorkerPool> {
    const size = opts.size ?? Math.max(1, availableParallelism());
    const here = fileURLToPath(new URL('.', import.meta.url));
    const script = opts.workerScript ?? resolve(here, 'rapier-trial-worker.ts');
    const pool = new TrialWorkerPool();
    for (let i = 0; i < size; i++) {
      const worker = new Worker(script, {
        // Use tsx to load TS workers — workerData must be JSON-serializable.
        execArgv: ['--import', 'tsx'],
        workerData: { vehicleOptions: opts.vehicleOptions },
      });
      const ready = new Promise<void>((resolveReady, rejectReady) => {
        const onMsg = (m: { type?: string }) => {
          if (m?.type === 'ready') { worker.off('message', onMsg); resolveReady(); }
        };
        worker.on('message', onMsg);
        worker.once('error', rejectReady);
      });
      pool.slots.push({ worker, ready, busy: false });
    }
    await Promise.all(pool.slots.map((s) => s.ready));
    return pool;
  }

  /** Distribute `specs` across workers in equal-sized shards.
   *  Returns the union of returned trials in main-thread order. */
  async runShards(specs: unknown[]): Promise<{ trials: CollectedTrialRaw[]; discarded: number }> {
    if (specs.length === 0) return { trials: [], discarded: 0 };
    const n = this.slots.length;
    const shardSize = Math.ceil(specs.length / n);
    const shards: unknown[][] = [];
    for (let i = 0; i < n; i++) shards.push(specs.slice(i * shardSize, (i + 1) * shardSize));

    const promises = shards.map((shard, idx) => new Promise<{ trials: CollectedTrialRaw[]; discarded: number }>((resolveShard, rejectShard) => {
      const slot = this.slots[idx]!;
      slot.busy = true;
      const onMsg = (m: { type?: string; shardId?: number; results?: CollectedTrialRaw[]; discarded?: number }) => {
        if (m?.type === 'done' && m.shardId === idx) {
          slot.worker.off('message', onMsg);
          slot.busy = false;
          resolveShard({ trials: m.results ?? [], discarded: m.discarded ?? 0 });
        }
      };
      slot.worker.on('message', onMsg);
      slot.worker.once('error', rejectShard);
      slot.worker.postMessage({ type: 'run', shardId: idx, specs: shard });
    }));

    const results = await Promise.all(promises);
    const trials: CollectedTrialRaw[] = [];
    let discarded = 0;
    for (const r of results) {
      trials.push(...r.trials);
      discarded += r.discarded;
    }
    return { trials, discarded };
  }

  async dispose(): Promise<void> {
    await Promise.all(this.slots.map((s) => new Promise<void>((res) => {
      s.worker.once('exit', () => res());
      s.worker.postMessage({ type: 'shutdown' });
      // Safety: terminate after 2s in case the worker hangs.
      setTimeout(() => { void s.worker.terminate().then(() => res()); }, 2000);
    })));
    this.slots = [];
  }
}
