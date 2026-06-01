// Worker-thread for Rapier trial collection.
//
// One worker owns a `HeadlessTrialHarness` (so the Rapier WASM module is
// initialized once per worker, not once per trial). The main thread
// ships `ManeuverSpec`-derived `TrialSpec`s in shards via the message
// channel; the worker returns `{trials, discarded}`.
//
// We can't transfer `ManeuverSpec` objects across the worker boundary
// directly because they carry functions (`spec.build`). Instead the
// main thread expands each spec into a static `TrialSpec` (with a
// pre-rolled controlsTrace) and ships only that.

import { parentPort, workerData } from 'node:worker_threads';
import {
  createHeadlessTrialHarness,
  type HeadlessTrialHarness,
  type TrialSpec,
} from 'kinocat/adapters/rapier';
import { DEFAULT_VEHICLE_OPTS } from '../../app/lib/training-driver';

interface WorkerInitMessage {
  type: 'init';
  vehicleOptions?: typeof DEFAULT_VEHICLE_OPTS;
}

interface WorkerRunMessage {
  type: 'run';
  shardId: number;
  specs: TrialSpec[];
}

interface WorkerShutdownMessage {
  type: 'shutdown';
}

type IncomingMessage = WorkerInitMessage | WorkerRunMessage | WorkerShutdownMessage;

let harness: HeadlessTrialHarness | null = null;

async function ensureHarness(vehicleOptions?: typeof DEFAULT_VEHICLE_OPTS): Promise<HeadlessTrialHarness> {
  if (harness) return harness;
  harness = await createHeadlessTrialHarness({
    vehicleOptions: vehicleOptions ?? DEFAULT_VEHICLE_OPTS,
  });
  return harness;
}

if (parentPort) {
  // Pre-warm the harness using the workerData payload.
  void ensureHarness(workerData?.vehicleOptions).then(() => {
    parentPort!.postMessage({ type: 'ready' });
  });

  parentPort.on('message', async (msg: IncomingMessage) => {
    if (msg.type === 'shutdown') {
      harness?.dispose();
      harness = null;
      process.exit(0);
    }
    if (msg.type === 'run') {
      const h = await ensureHarness();
      const results: unknown[] = [];
      let discarded = 0;
      for (const spec of msg.specs) {
        const r = h.runTrial(spec);
        if (!r.ok) {
          discarded++;
          continue;
        }
        const t = r.trial;
        results.push({
          id: t.id,
          dt: t.dt,
          samples: t.samples,
          config: t.config,
        });
      }
      parentPort!.postMessage({ type: 'done', shardId: msg.shardId, results, discarded });
    }
  });
}
