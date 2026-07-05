import { describe, it, expect } from 'vitest';
import {
  initWorkerContext,
  handleWorldUpdateMessage,
  type WorkerContext,
} from '../../src/worker/planner-worker';
import type {
  WorkerWorldUpdateAck,
  WorkerWorldUpdateMsg,
} from '../../src/worker/protocol';
import { InMemoryNavWorld, type NavPolygon, type NavWorld } from '../../src/environment/nav-world';
import { AffordanceRegistry } from '../../src/predict/affordance-registry';
import { defaultVehicleAgent } from '../../src/agent/vehicle';
import type { MotionPrimitiveLibrary } from '../../src/primitives/library';

const floor: NavPolygon = {
  id: 1,
  y: 0,
  ring: [
    [0, 0],
    [20, 0],
    [20, 20],
    [0, 20],
  ],
};

function initWith(world: NavWorld): void {
  // Only `world` is exercised by world-update; the rest are inert stand-ins.
  initWorkerContext({
    world,
    agent: defaultVehicleAgent(),
    lib: {} as MotionPrimitiveLibrary,
    affordances: new AffordanceRegistry(),
  } satisfies WorkerContext);
}

function send(msg: Omit<WorkerWorldUpdateMsg, 'type'>): WorkerWorldUpdateAck {
  let ack: WorkerWorldUpdateAck | null = null;
  handleWorldUpdateMessage({ type: 'world-update', ...msg }, (r) => {
    ack = r;
  });
  expect(ack).not.toBeNull();
  return ack!;
}

describe('handleWorldUpdateMessage', () => {
  it('applies an obstacle delta: previously-clear segment becomes blocked', () => {
    const world = new InMemoryNavWorld([floor], []);
    initWith(world);
    expect(world.segmentClear(2, 10, 18, 10)).toBe(true);
    const before = world.revision;

    const ack = send({
      seq: 7,
      obstacles: [
        [
          [8, 8],
          [12, 8],
          [12, 12],
          [8, 12],
        ],
      ],
    });

    expect(world.segmentClear(2, 10, 18, 10)).toBe(false);
    expect(ack.seq).toBe(7);
    expect(ack.revision).toBe(world.revision);
    expect(world.revision).toBeGreaterThan(before);
  });

  it('appends off-mesh links', () => {
    const world = new InMemoryNavWorld([floor], []);
    initWith(world);
    const from = world.polygonAt(2, 2)!;
    const to = world.polygonAt(18, 18)!;
    send({
      seq: 1,
      addOffMeshLinks: [
        { from, to, start: [2, 0, 2], end: [18, 0, 18], kind: 'jump', cost: 3 },
      ],
    });
    expect(world.offMeshFrom(from).length).toBe(1);
  });

  it('bumpRevisionOnly works on worlds exposing bumpRevision', () => {
    let bumped = 0;
    const world = {
      revision: 0,
      bumpRevision() {
        bumped++;
        (this as { revision: number }).revision++;
      },
    } as unknown as NavWorld;
    initWith(world);
    const ack = send({ seq: 2, bumpRevisionOnly: true });
    expect(bumped).toBe(1);
    expect(ack.revision).toBe(1);
  });

  it('throws loudly when a data delta hits a world without the mutator', () => {
    // A NavWorld with no setObstacles (e.g. a NavcatWorld-like adapter).
    const world = { revision: 0 } as unknown as NavWorld;
    initWith(world);
    expect(() =>
      handleWorldUpdateMessage(
        { type: 'world-update', seq: 3, obstacles: [[[0, 0], [1, 0], [1, 1]]] },
        () => {},
      ),
    ).toThrow(/setObstacles/);
    expect(() =>
      handleWorldUpdateMessage(
        {
          type: 'world-update',
          seq: 4,
          addOffMeshLinks: [
            {
              from: { id: 1, cx: 0, cz: 0, y: 0 },
              to: { id: 2, cx: 1, cz: 1, y: 0 },
              start: [0, 0, 0],
              end: [1, 0, 1],
              kind: 'jump',
              cost: 1,
            },
          ],
        },
        () => {},
      ),
    ).toThrow(/addOffMeshLink/);
  });
});
