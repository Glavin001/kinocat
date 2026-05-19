// Incremental Generalized Hybrid A* — anytime, multi-resolution, branch-and-
// bound. Coarse passes find a first solution fast; successively finer passes
// tighten the incumbent, sharing only the bound Ω. The finest pass uses exact
// state dedup (no coarse dominance), so with an admissible/consistent
// heuristic the returned plan is optimal given enough budget. Time-extension
// (M4) needs no change here — it lives entirely in the Environment.

import type { Node } from '../environment/types';
import { reconstructNodes, reconstructStates } from './node';
import { DominanceTable } from './resolution';
import { BinaryHeap } from '../internal/heap';
import { DEFAULT_HYSTERESIS, decideLevel } from './hysteresis';
import type { PlanRequest, PlanResult, PlanStats } from './types';

const EPS = 1e-9;

interface QItem<State> {
  node: Node<State>;
  seq: number;
}

function emptyResult<State>(): PlanResult<State> {
  const stats: PlanStats = {
    expansions: 0,
    generated: 0,
    deadlineHit: false,
    budgetHit: false,
    passesRun: 0,
    improvements: 0,
  };
  return { found: false, cost: Infinity, path: [], nodes: [], stats, solutionHistory: [] };
}

/**
 * Run IGHA* until a solution-improving deadline, returning the best plan so
 * far. `deadlineMs` is wall-clock; `options.maxExpansions` is a deterministic
 * budget (preferred for reproducible anytime behaviour).
 */
export function plan<State>(
  req: PlanRequest<State>,
  deadlineMs: number,
): PlanResult<State> {
  const { start, goal, environment: env } = req;
  const opts = req.options ?? {};
  const result = emptyResult<State>();
  const stats = result.stats;

  const [startValid, goalValid] = env.checkValidity(start, goal);
  if (!startValid || !goalValid) return result;

  const levels = Math.max(1, opts.levels ?? env.levels);
  const maxExpansions = opts.maxExpansions ?? Infinity;
  const checkEvery = opts.deadlineCheckEvery ?? 64;
  const hyst = opts.hysteresis ?? DEFAULT_HYSTERESIS;
  const useClock = Number.isFinite(deadlineMs);
  const t0 = useClock ? performance.now() : 0;
  const deadAt = t0 + deadlineMs;

  const goalNode = env.createNode(goal, null, null);

  let omega = Infinity;
  let incumbentNodes: Node<State>[] | null = null;
  let seq = 0;
  let expansionsSinceImprovement = 0;

  const cmp = (a: QItem<State>, b: QItem<State>): number => {
    if (a.node.f !== b.node.f) return a.node.f - b.node.f;
    if (a.node.h !== b.node.h) return a.node.h - b.node.h;
    return a.seq - b.seq;
  };

  let stop = false;

  for (let level = 0; level < levels && !stop; level++) {
    stats.passesRun++;
    const finest = level === levels - 1;
    const open = new BinaryHeap<QItem<State>>(cmp);
    const gExact = new Map<string, number>();
    const dom = new DominanceTable(env.levels);

    const startNode = env.createNode(start, null, null);
    startNode.g = 0;
    startNode.h = env.heuristic(start, goal);
    startNode.f = startNode.h;
    gExact.set(startNode.hash, 0);
    open.push({ node: startNode, seq: seq++ });

    while (!open.isEmpty()) {
      if (stats.expansions >= maxExpansions) {
        stats.budgetHit = true;
        stop = true;
        break;
      }
      if (
        useClock &&
        stats.expansions % checkEvery === 0 &&
        performance.now() >= deadAt
      ) {
        stats.deadlineHit = true;
        stop = true;
        break;
      }

      const v = open.pop()!.node;

      if (v.f > omega + EPS) continue; // branch-and-bound
      const gv = gExact.get(v.hash);
      if (gv !== undefined && v.g > gv + EPS) continue; // stale duplicate

      if (env.reachedGoalRegion(v, goalNode)) {
        if (v.g < omega - EPS) {
          omega = v.g;
          incumbentNodes = reconstructNodes(v);
          result.solutionHistory.push(reconstructStates(v));
          stats.improvements++;
          expansionsSinceImprovement = 0;
        }
        continue; // keep searching for a better solution (anytime)
      }

      stats.expansions++;
      expansionsSinceImprovement++;

      for (const n of env.succ(v, goalNode)) {
        if (n.f > omega + EPS) continue;
        const known = gExact.get(n.hash);
        if (known !== undefined && n.g >= known - EPS) continue; // exact dedup
        if (!finest) {
          const key = n.index[level];
          if (key !== undefined && n.g >= dom.best(level, key) - EPS) continue;
          if (key !== undefined) dom.relax(level, key, n.g);
        }
        gExact.set(n.hash, n.g);
        n.active = true;
        open.push({ node: n, seq: seq++ });
        stats.generated++;
      }

      // Hysteresis: abandon a stagnating coarse pass early to refine sooner.
      if (
        !finest &&
        decideLevel(level, levels - 1, expansionsSinceImprovement, hyst) !== level
      ) {
        break;
      }
    }
  }

  if (incumbentNodes && incumbentNodes.length > 0) {
    result.found = true;
    result.cost = omega;
    result.nodes = incumbentNodes;
    result.path = incumbentNodes.map((n) => n.state);
  }
  return result;
}
