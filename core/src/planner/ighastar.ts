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
import {
  NULL_RECORDER,
  makeRecorder,
  type PassStats,
  type PerfRecorder,
} from './perf';
import type { PlanRequest, PlanResult, PlanStats } from './types';

const EPS = 1e-9;

function emptyResult<State>(rec: PerfRecorder): PlanResult<State> {
  const stats: PlanStats = {
    expansions: 0,
    generated: 0,
    deadlineHit: false,
    budgetHit: false,
    passesRun: 0,
    improvements: 0,
    counters: rec.counters,
    perPass: [],
  };
  if (rec.timingsOn) stats.timings = rec.timings;
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
  const profile = opts.profile ?? 'counts';
  const rec = makeRecorder(profile);
  env.attachRecorder?.(rec);

  const tStart = rec.timingsOn ? performance.now() : 0;
  const result = emptyResult<State>(rec);
  const stats = result.stats;
  const counters = rec.counters;
  const timings = rec.timings;
  const timingsOn = rec.timingsOn;

  const [startValid, goalValid] = env.checkValidity(start, goal);
  if (!startValid || !goalValid) {
    if (timingsOn) timings.total = performance.now() - tStart;
    env.attachRecorder?.(NULL_RECORDER);
    return result;
  }

  const levels = Math.max(1, opts.levels ?? env.levels);
  const maxExpansions = opts.maxExpansions ?? Infinity;
  const checkEvery = opts.deadlineCheckEvery ?? 64;
  const hyst = opts.hysteresis ?? DEFAULT_HYSTERESIS;
  const weight = opts.weight ?? 1;
  const useClock = Number.isFinite(deadlineMs);
  const t0 = useClock ? performance.now() : 0;
  const deadAt = t0 + deadlineMs;

  const goalNode = env.createNode(goal, null, null);

  let omega = Infinity;
  let incumbentNodes: Node<State>[] | null = null;
  let seq = 0;
  let expansionsSinceImprovement = 0;

  // Heap pushes Node refs directly — node.seq breaks f/h ties (FIFO within
  // equal cost). Avoids per-push QItem object allocation (~200k saved on
  // canyon).
  const cmp = (a: Node<State>, b: Node<State>): number => {
    if (a.f !== b.f) return a.f - b.f;
    if (a.h !== b.h) return a.h - b.h;
    return a.seq - b.seq;
  };

  let stop = false;

  for (let level = 0; level < levels && !stop; level++) {
    const passStart = timingsOn ? performance.now() : 0;
    const passExpansionsBefore = stats.expansions;
    const passGeneratedBefore = stats.generated;
    const passImprovementsBefore = stats.improvements;
    stats.passesRun++;
    const finest = level === levels - 1;
    const open = new BinaryHeap<Node<State>>(cmp);
    const gExact = new Map<string, number>();
    const dom = new DominanceTable(env.levels);

    const startNode = env.createNode(start, null, null);
    startNode.g = 0;
    const th = timingsOn ? performance.now() : 0;
    startNode.h = env.heuristic(start, goal);
    if (timingsOn) timings.heuristic += performance.now() - th;
    counters.heuristicCalls++;
    startNode.f = weight * startNode.h;
    startNode.seq = seq++;
    gExact.set(startNode.hash, 0);
    open.push(startNode);

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

      const v = open.pop()!;

      if (v.f > omega + EPS) continue; // branch-and-bound
      const gv = gExact.get(v.hash);
      if (gv !== undefined && v.g > gv + EPS) continue; // stale duplicate

      counters.goalChecks++;
      if (env.reachedGoalRegion(v, goalNode)) {
        if (v.g < omega - EPS) {
          omega = v.g;
          const tr = timingsOn ? performance.now() : 0;
          incumbentNodes = reconstructNodes(v);
          result.solutionHistory.push(reconstructStates(v));
          if (timingsOn) timings.pathReconstruct += performance.now() - tr;
          stats.improvements++;
          counters.improvementWallMs.push(
            useClock ? performance.now() - t0 : 0,
          );
          expansionsSinceImprovement = 0;
        }
        continue; // keep searching for a better solution (anytime)
      }

      stats.expansions++;
      expansionsSinceImprovement++;

      counters.succCalls++;
      const ts = timingsOn ? performance.now() : 0;
      const succs = env.succ(v, goalNode, level);
      if (timingsOn) timings.succ += performance.now() - ts;
      counters.successorsTotal += succs.length;

      for (let i = 0; i < succs.length; i++) {
        const n = succs[i]!;
        // Apply weighted-A* f-inflation. Env returns n.h admissible; the
        // planner controls the f-ordering so weight is honored uniformly
        // regardless of env implementation.
        if (weight !== 1) n.f = n.g + weight * n.h;
        if (n.f > omega + EPS) {
          counters.rejectedByOmega++;
          continue;
        }
        const known = gExact.get(n.hash);
        if (known !== undefined && n.g >= known - EPS) {
          counters.rejectedByExact++;
          continue;
        }
        if (!finest) {
          const key = n.index[level];
          if (key !== undefined) {
            counters.domLookups++;
            if (n.g >= dom.best(level, key) - EPS) {
              counters.rejectedByDominance++;
              continue;
            }
            if (dom.relax(level, key, n.g)) counters.domRelaxes++;
          }
        }
        gExact.set(n.hash, n.g);
        n.seq = seq++;
        open.push(n);
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

    const pass: PassStats = {
      level,
      expansions: stats.expansions - passExpansionsBefore,
      generated: stats.generated - passGeneratedBefore,
      improvements: stats.improvements - passImprovementsBefore,
    };
    if (timingsOn) pass.ms = performance.now() - passStart;
    stats.perPass.push(pass);
  }

  if (incumbentNodes && incumbentNodes.length > 0) {
    result.found = true;
    result.cost = omega;
    result.nodes = incumbentNodes;
    result.path = incumbentNodes.map((n) => n.state);
  }
  if (timingsOn) timings.total = performance.now() - tStart;
  env.attachRecorder?.(NULL_RECORDER);
  return result;
}
