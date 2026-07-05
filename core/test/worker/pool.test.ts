import { describe, it, expect, vi } from 'vitest';
import {
  PlannerPool,
  ReplanScheduler,
  FrameBudget,
  type WorkerLike,
  type AgentPlanSource,
  type PlanRequestBody,
} from '../../src/worker/pool';
import type {
  MainToWorker,
  WorkerInitMsg,
  WorkerPlanRequest,
  WorkerPlanResponse,
  WorkerWorldUpdateMsg,
} from '../../src/worker/protocol';
import type { CarKinematicState } from '../../src/agent/types';
import type { PlanStats } from '../../src/planner/types';

const ST: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
const STATS = { expansions: 0 } as unknown as PlanStats;
const BODY: PlanRequestBody = { start: ST, goal: ST, obstacles: [] };
const INIT: WorkerInitMsg = {
  type: 'init',
  polygons: [],
  obstacles: [],
  agent: { maxSpeed: 1 } as WorkerInitMsg['agent'],
  libJSON: '{}',
  courseJSON: '{}',
};

/** Scripted fake worker: acks init immediately; records plan requests and
 *  lets the test answer them explicitly (out of order, stale, …). */
class FakeWorker implements WorkerLike {
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: ((err: unknown) => void) | null = null;
  received: MainToWorker[] = [];
  terminated = false;

  postMessage(msg: unknown): void {
    const m = msg as MainToWorker;
    this.received.push(m);
    if (m.type === 'init') {
      queueMicrotask(() => this.onmessage?.({ data: { type: 'init-ack' } }));
    }
  }

  answerPlan(reqId: number, npcId: string, found = true): void {
    const resp: WorkerPlanResponse = {
      type: 'plan-result',
      reqId,
      npcId,
      found,
      cost: 1,
      path: [ST],
      stats: STATS,
    };
    this.onmessage?.({ data: resp });
  }

  ackWorldUpdate(seq: number): void {
    this.onmessage?.({ data: { type: 'world-update-ack', seq, revision: 1 } });
  }

  lastPlanReq(): WorkerPlanRequest | undefined {
    return this.received.filter((m): m is WorkerPlanRequest => m.type === 'plan').at(-1);
  }

  terminate(): void {
    this.terminated = true;
  }
}

async function makePool(ids: string[]): Promise<{ pool: PlannerPool; fakes: Map<string, FakeWorker> }> {
  const fakes = new Map<string, FakeWorker>();
  const pool = new PlannerPool((id) => {
    const w = new FakeWorker();
    fakes.set(id, w);
    return w;
  });
  await pool.init(INIT, ids);
  return { pool, fakes };
}

describe('PlannerPool', () => {
  it('spawns one worker per agent and inits all with the same message', async () => {
    const { fakes } = await makePool(['a', 'b', 'c']);
    expect(fakes.size).toBe(3);
    for (const w of fakes.values()) {
      expect(w.received[0]).toEqual(INIT);
    }
  });

  it('routes plan requests by agent id and dedups in-flight', async () => {
    const { pool, fakes } = await makePool(['a', 'b']);
    expect(pool.requestPlan('a', BODY)).toBe(true);
    expect(pool.hasInflight('a')).toBe(true);
    expect(pool.requestPlan('a', BODY)).toBe(false); // deduped
    expect(pool.requestPlan('b', BODY)).toBe(true); // independent
    expect(fakes.get('a')!.lastPlanReq()).toBeDefined();
    expect(fakes.get('b')!.lastPlanReq()).toBeDefined();
    expect(pool.requestPlan('nope', BODY)).toBe(false); // unknown agent
  });

  it('delivers results and clears the in-flight gate', async () => {
    const { pool, fakes } = await makePool(['a']);
    const results: WorkerPlanResponse[] = [];
    pool.onResult((r) => results.push(r));
    pool.requestPlan('a', BODY);
    const req = fakes.get('a')!.lastPlanReq()!;
    fakes.get('a')!.answerPlan(req.reqId, 'a');
    expect(results.length).toBe(1);
    expect(pool.hasInflight('a')).toBe(false);
    expect(pool.requestPlan('a', BODY)).toBe(true); // gate reopened
  });

  it('drops stale results (superseded reqId)', async () => {
    const { pool, fakes } = await makePool(['a']);
    const results: WorkerPlanResponse[] = [];
    pool.onResult((r) => results.push(r));
    pool.requestPlan('a', BODY);
    const staleReq = fakes.get('a')!.lastPlanReq()!;
    // A stale answer with a reqId that is NOT the current one must be dropped.
    fakes.get('a')!.answerPlan(staleReq.reqId + 999, 'a');
    expect(results.length).toBe(0);
    expect(pool.hasInflight('a')).toBe(true); // still waiting on the real one
    fakes.get('a')!.answerPlan(staleReq.reqId, 'a');
    expect(results.length).toBe(1);
  });

  it('broadcast resolves only after every worker acks', async () => {
    const { pool, fakes } = await makePool(['a', 'b']);
    const resolved = vi.fn();
    const p = pool.broadcast({ obstacles: [] }).then(resolved);
    const seq = (fakes.get('a')!.received.at(-1) as WorkerWorldUpdateMsg).seq;
    fakes.get('a')!.ackWorldUpdate(seq);
    await Promise.resolve();
    expect(resolved).not.toHaveBeenCalled(); // b hasn't acked
    fakes.get('b')!.ackWorldUpdate(seq);
    await p;
    expect(resolved).toHaveBeenCalledOnce();
  });

  it('dispose terminates every worker', async () => {
    const { pool, fakes } = await makePool(['a', 'b']);
    pool.dispose();
    for (const w of fakes.values()) expect(w.terminated).toBe(true);
    expect(pool.requestPlan('a', BODY)).toBe(false);
  });
});

describe('ReplanScheduler', () => {
  function source(
    id: string,
    over: Partial<AgentPlanSource> = {},
  ): AgentPlanSource & { prepared: number } {
    const src = {
      id,
      prepared: 0,
      prepare(): PlanRequestBody | null {
        src.prepared++;
        return BODY;
      },
      ...over,
    };
    return src as AgentPlanSource & { prepared: number };
  }

  it('dispatches one agent per tick, round-robin', async () => {
    const { pool } = await makePool(['a', 'b', 'c']);
    const agents = [source('a'), source('b'), source('c')];
    const sched = new ReplanScheduler(pool, agents);
    expect(sched.tick(0)).toBe('a');
    expect(sched.tick(25)).toBe('b');
    expect(sched.tick(50)).toBe('c');
    // 'a' still in flight → its slot is skipped, not re-dispatched.
    expect(sched.tick(75)).toBe(null);
  });

  it('skips agents whose prepare declines', async () => {
    const { pool } = await makePool(['a', 'b']);
    const agents = [source('a', { prepare: () => null }), source('b')];
    const sched = new ReplanScheduler(pool, agents);
    expect(sched.tick(0)).toBe(null); // a declined
    expect(sched.tick(25)).toBe('b');
  });

  it('emergency slot-steal preempts the round-robin for the priority agent', async () => {
    const { pool, fakes } = await makePool(['a', 'b']);
    let remaining = 5;
    const agents = [
      source('a', { planRemainingSec: () => remaining }),
      source('b'),
    ];
    const sched = new ReplanScheduler(pool, agents, { priorityAgentId: 'a' });
    expect(sched.tick(0)).toBe('a'); // normal slot
    const req = fakes.get('a')!.lastPlanReq()!;
    fakes.get('a')!.answerPlan(req.reqId, 'a');
    // Next slot would be b's — but a's plan is nearly exhausted, so a
    // steals it (consuming the slot, matching the carchase semantics).
    remaining = 0.3;
    expect(sched.tick(25)).toBe('a');
    // Cursor advanced past b's slot; a's own slot comes up next but a is
    // in flight → skipped; then b finally goes.
    expect(sched.tick(50)).toBe(null);
    expect(sched.tick(75)).toBe('b');
  });
});

describe('FrameBudget', () => {
  it('caps work per frame and resets on startFrame', () => {
    const budget = new FrameBudget(5);
    expect(budget.allow()).toBe(true);
    const r = budget.run(() => {
      const t0 = performance.now();
      // Burn ≥ 6ms of wall time.
      while (performance.now() - t0 < 6) { /* spin */ }
      return 42;
    });
    expect(r).toBe(42);
    expect(budget.allow()).toBe(false);
    expect(budget.run(() => 1)).toBeUndefined(); // skip-not-queue
    budget.startFrame();
    expect(budget.allow()).toBe(true);
    expect(budget.run(() => 1)).toBe(1);
  });
});
