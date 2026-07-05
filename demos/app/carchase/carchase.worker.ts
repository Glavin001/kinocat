/// <reference lib="webworker" />

import type { MainToWorker } from 'kinocat/worker';
import {
  initWorkerContext,
  handlePlanMessage,
  handleWorldUpdateMessage,
} from 'kinocat/worker';
import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { carChaseAffordances, type CarChaseCourse } from '../lib/carchase-scenarios';

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = (e: MessageEvent<MainToWorker>) => {
  const msg = e.data;

  if (msg.type === 'init') {
    const course: CarChaseCourse = JSON.parse(msg.courseJSON);
    const world = new InMemoryNavWorld(msg.polygons, msg.obstacles);
    const lib = MotionPrimitiveLibrary.fromJSON(msg.libJSON);
    const affordances = carChaseAffordances(course);

    initWorkerContext({ world, agent: msg.agent, lib, affordances });
    self.postMessage({ type: 'init-ack' });
    return;
  }

  if (msg.type === 'plan') {
    handlePlanMessage(msg, (resp) => self.postMessage(resp));
    return;
  }

  if (msg.type === 'world-update') {
    handleWorldUpdateMessage(msg, (resp) => self.postMessage(resp));
  }
};
