// Node worker_threads entry for the four-agents-frame-budget gate — the
// node twin of demos/app/carchase/carchase.worker.ts, reusing the exact same
// core handlers over parentPort instead of DedicatedWorkerGlobalScope.
// Loaded with `--import tsx` so it runs straight from source; the `kinocat/*`
// imports resolve like every other tsx-run demo script.

import { parentPort } from 'node:worker_threads';
import type { MainToWorker } from 'kinocat/worker';
import {
  initWorkerContext,
  handlePlanMessage,
  handleWorldUpdateMessage,
} from 'kinocat/worker';
import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { AffordanceRegistry } from 'kinocat/predict';

const port = parentPort;
if (!port) throw new Error('carchase-node-worker must run as a worker thread');

port.on('message', (msg: MainToWorker) => {
  if (msg.type === 'init') {
    const world = new InMemoryNavWorld(msg.polygons, msg.obstacles);
    const lib = MotionPrimitiveLibrary.fromJSON(msg.libJSON);
    // Affordances (course jump/boost pads) don't participate in this gate —
    // an empty registry keeps the worker free of demo-app imports so tsx
    // resolves everything through the `kinocat/*` package exports.
    initWorkerContext({ world, agent: msg.agent, lib, affordances: new AffordanceRegistry() });
    port.postMessage({ type: 'init-ack' });
    return;
  }
  if (msg.type === 'plan') {
    handlePlanMessage(msg, (resp) => port.postMessage(resp));
    return;
  }
  if (msg.type === 'world-update') {
    handleWorldUpdateMessage(msg, (resp) => port.postMessage(resp));
  }
});
