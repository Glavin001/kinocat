// Multi-agent planner pool + replan scheduler + frame budget — the core
// extraction of the patterns proven in the carchase demo (one worker per
// agent, staggered dispatch, in-flight dedup, stale-result rejection,
// emergency slot-steal). Core never constructs Workers — the environment
// hands in a spawn factory over the minimal `WorkerLike` seam (a browser
// `Worker` and a node `worker_threads` adapter both satisfy it), so this
// stays framework- and bundler-free.

import type {
  WorkerInitMsg,
  WorkerPlanRequest,
  WorkerPlanResponse,
  WorkerToMain,
  WorkerWorldUpdateMsg,
} from './protocol';

/** Structural worker seam. A DOM `Worker` satisfies this directly; wrap a
 *  node `worker_threads.Worker` with ~10 lines (postMessage/on('message')). */
export interface WorkerLike {
  postMessage(msg: unknown): void;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror?: ((err: unknown) => void) | null;
  terminate(): void;
}

export type SpawnWorker = (agentId: string) => WorkerLike;

/** Everything a plan request needs except the routing fields the pool owns. */
export type PlanRequestBody = Omit<WorkerPlanRequest, 'type' | 'reqId' | 'npcId'>;

/** One planner worker per agent id. N agents' plans run genuinely in
 *  parallel; the per-agent in-flight gate is the backpressure authority —
 *  `requestPlan` refuses (returns false) while a request is outstanding, and
 *  a result whose reqId was superseded is dropped, never delivered. */
export class PlannerPool {
  private readonly workers = new Map<string, WorkerLike>();
  private readonly inflightReqIds = new Map<string, number>();
  private readonly inflightSendTimes = new Map<number, number>();
  private nextReqId = 0;
  private nextSeq = 0;
  private resultCb: ((r: WorkerPlanResponse, elapsedMs: number) => void) | null = null;
  private pendingAcks = new Map<number, { remaining: number; resolve: () => void }>();

  constructor(private readonly spawn: SpawnWorker) {}

  /** Spawn one worker per agent and initialize them all in parallel with the
   *  same init message. Resolves when every worker has acked. */
  init(initMsg: WorkerInitMsg, agentIds: ReadonlyArray<string>): Promise<void> {
    const promises = agentIds.map(
      (id) =>
        new Promise<void>((resolve, reject) => {
          const w = this.spawn(id);
          this.workers.set(id, w);
          if ('onerror' in w) w.onerror = (err) => reject(err);
          w.onmessage = (e) => {
            const msg = e.data as WorkerToMain;
            if (msg.type === 'init-ack') {
              if ('onerror' in w) w.onerror = null;
              w.onmessage = (ev) => this.route(ev.data as WorkerToMain);
              resolve();
            }
          };
          w.postMessage(initMsg);
        }),
    );
    return Promise.all(promises).then(() => undefined);
  }

  private route(msg: WorkerToMain): void {
    if (msg.type === 'plan-result') {
      if (this.inflightReqIds.get(msg.npcId) !== msg.reqId) return; // stale
      this.inflightReqIds.delete(msg.npcId);
      const sendTime = this.inflightSendTimes.get(msg.reqId);
      this.inflightSendTimes.delete(msg.reqId);
      const elapsed = sendTime === undefined ? 0 : performance.now() - sendTime;
      this.resultCb?.(msg, elapsed);
      return;
    }
    if (msg.type === 'world-update-ack') {
      const pending = this.pendingAcks.get(msg.seq);
      if (!pending) return;
      pending.remaining -= 1;
      if (pending.remaining <= 0) {
        this.pendingAcks.delete(msg.seq);
        pending.resolve();
      }
    }
  }

  onResult(cb: (r: WorkerPlanResponse, elapsedMs: number) => void): void {
    this.resultCb = cb;
  }

  hasInflight(agentId: string): boolean {
    return this.inflightReqIds.has(agentId);
  }

  /** Dispatch a plan request for `agentId`. Returns false (and sends
   *  nothing) when the agent is unknown or already has a request in flight. */
  requestPlan(agentId: string, body: PlanRequestBody): boolean {
    const w = this.workers.get(agentId);
    if (!w || this.inflightReqIds.has(agentId)) return false;
    const reqId = this.nextReqId++;
    this.inflightReqIds.set(agentId, reqId);
    this.inflightSendTimes.set(reqId, performance.now());
    const req: WorkerPlanRequest = { type: 'plan', reqId, npcId: agentId, ...body };
    w.postMessage(req);
    return true;
  }

  /** Send a world update to EVERY worker; resolves when all have acked, i.e.
   *  no worker will plan against pre-update geometry after this settles. */
  broadcast(update: Omit<WorkerWorldUpdateMsg, 'type' | 'seq'>): Promise<void> {
    if (this.workers.size === 0) return Promise.resolve();
    const seq = this.nextSeq++;
    const msg: WorkerWorldUpdateMsg = { type: 'world-update', seq, ...update };
    return new Promise<void>((resolve) => {
      this.pendingAcks.set(seq, { remaining: this.workers.size, resolve });
      for (const w of this.workers.values()) w.postMessage(msg);
    });
  }

  dispose(): void {
    for (const w of this.workers.values()) w.terminate();
    this.workers.clear();
    this.inflightReqIds.clear();
    this.inflightSendTimes.clear();
    this.pendingAcks.clear();
  }
}

/** The slice of `PlannerPool` the scheduler needs — structural, so wrappers
 *  (e.g. a demo host that owns Worker construction) can stand in. */
export interface PlanDispatcher {
  hasInflight(agentId: string): boolean;
  requestPlan(agentId: string, body: PlanRequestBody): boolean;
}

/** How the scheduler asks the game for one agent's replan payload. */
export interface AgentPlanSource {
  id: string;
  /** Build the request body for a replan now, or null to skip this slot
   *  (agent paused, player-driven, cooling down, …). */
  prepare(nowMs: number): PlanRequestBody | null;
  /** Seconds of committed plan left, or null when unknown. Feeds the
   *  emergency slot-steal for the priority agent. */
  planRemainingSec?(nowMs: number): number | null;
}

export interface ReplanSchedulerOptions {
  /** Steal the slot for the priority agent when its plan has less than this
   *  many seconds left (default 0.6). */
  emergencyRemainingSec?: number;
  /** Agent whose plan exhaustion justifies stealing a slot (e.g. the one
   *  the camera follows). */
  priorityAgentId?: string;
}

/** Staggered round-robin dispatcher over a `PlannerPool` — at most ONE new
 *  request per `tick`, so planning load is spread across ticks while each
 *  agent's plans still run in parallel on its own worker. Drive it from the
 *  game loop or a `setInterval` (the carchase demo uses 25 ms). */
export class ReplanScheduler {
  private cursor = 0;
  private readonly emergencySec: number;

  constructor(
    private readonly pool: PlanDispatcher,
    private readonly agents: ReadonlyArray<AgentPlanSource>,
    opts: ReplanSchedulerOptions = {},
  ) {
    this.emergencySec = opts.emergencyRemainingSec ?? 0.6;
    this.priorityId = opts.priorityAgentId ?? null;
  }
  private readonly priorityId: string | null;

  /** Dispatch at most one replan. Returns the agent id dispatched, or null
   *  when the slot was skipped (in-flight, prepare() declined, …). */
  tick(nowMs: number): string | null {
    // Emergency: the priority agent is about to run out of committed plan —
    // steal this slot so it never coasts planless.
    if (this.priorityId !== null && !this.pool.hasInflight(this.priorityId)) {
      const prio = this.agents.find((a) => a.id === this.priorityId);
      const left = prio?.planRemainingSec?.(nowMs);
      if (prio && left !== null && left !== undefined && left < this.emergencySec) {
        const body = prio.prepare(nowMs);
        if (body && this.pool.requestPlan(prio.id, body)) {
          this.cursor += 1;
          return prio.id;
        }
      }
    }
    const agent = this.agents[this.cursor % this.agents.length];
    this.cursor += 1;
    if (!agent || this.pool.hasInflight(agent.id)) return null;
    const body = agent.prepare(nowMs);
    if (!body) return null;
    return this.pool.requestPlan(agent.id, body) ? agent.id : null;
  }
}

/** Per-frame cap on main-thread planning-adjacent work (plan adoption, sync
 *  fallbacks, debug redraws). Skip-not-queue: work refused this frame is NOT
 *  deferred — the caller retries next frame from live state, which is always
 *  fresher than a queued closure. */
export class FrameBudget {
  private used = 0;

  constructor(private readonly capMs: number) {}

  startFrame(): void {
    this.used = 0;
  }

  get remainingMs(): number {
    return Math.max(0, this.capMs - this.used);
  }

  allow(): boolean {
    return this.used < this.capMs;
  }

  /** Run `fn` if budget remains this frame (charging its wall time),
   *  else return undefined. */
  run<T>(fn: () => T): T | undefined {
    if (!this.allow()) return undefined;
    const t0 = performance.now();
    try {
      return fn();
    } finally {
      this.used += performance.now() - t0;
    }
  }
}
