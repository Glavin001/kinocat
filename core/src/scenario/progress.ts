// Deterministic progress evaluator. Given a CompiledAutomaton and a trajectory
// (a state sequence — typically a committed plan or the live execution log),
// replay the SAME greedy guard advance the planner bridge uses to report which
// automaton state we are in, how many phases are satisfied, and whether the
// objective is met. This is the single source of truth the visualizer renders,
// so the on-screen automaton always matches the planner's internal state.

import type { CompiledAutomaton } from './automaton';
import type { ScenarioState } from './types';
import { guardSatisfied } from './guard';

export interface ProgressSnapshot {
  /** Current automaton state id after replaying the trajectory. */
  q: number;
  /** Phase-progress rank of `q` (best-progress depth). */
  depth: number;
  /** Max depth across all states (the "total" for a k/total readout). */
  maxDepth: number;
  /** True if `q` is an accepting state (objective satisfied). */
  done: boolean;
  /** Laps completed (number of times the run re-entered `start` after leaving
   *  it) — meaningful for `repeat` / progress automata. */
  laps: number;
  /** The automaton state id after each trajectory sample (length == states). */
  trace: number[];
}

/** Advance `q` as far as the single edge from->to allows, evaluating outgoing
 *  guards in phase order (greedy multi-advance), returning the new state. */
export function stepAutomaton(
  automaton: CompiledAutomaton,
  q: number,
  from: ScenarioState,
  to: ScenarioState,
): number {
  let cur = q;
  // Bound the greedy loop by the state count to avoid any pathological cycle.
  for (let guardSteps = 0; guardSteps < automaton.states.length + 1; guardSteps++) {
    const st = automaton.states[cur];
    if (!st || st.transitions.length === 0) break;
    let advanced = false;
    for (const tr of st.transitions) {
      if (guardSatisfied(tr.guard, from, to)) {
        cur = tr.target;
        advanced = true;
        break; // phase order: take the first satisfied guard, then re-evaluate
      }
    }
    if (!advanced) break;
  }
  return cur;
}

/** Replay a whole trajectory through the automaton from `start`. */
export function evaluateProgress(
  automaton: CompiledAutomaton,
  trajectory: ReadonlyArray<ScenarioState>,
): ProgressSnapshot {
  const maxDepth = automaton.states.reduce((m, s) => Math.max(m, s.depth), 0);
  const acceptSet = new Set(automaton.accepting);
  let q = automaton.start;
  let laps = 0;
  let prevDepth = automaton.states[q]?.depth ?? 0;
  const trace: number[] = [];
  if (trajectory.length > 0) trace.push(q);
  for (let i = 0; i + 1 < trajectory.length; i++) {
    q = stepAutomaton(automaton, q, trajectory[i]!, trajectory[i + 1]!);
    const d = automaton.states[q]?.depth ?? 0;
    // A depth DECREASE means we traversed a `repeat` back-edge — one lap done.
    if (d < prevDepth) laps++;
    prevDepth = d;
    trace.push(q);
  }
  return {
    q,
    depth: automaton.states[q]?.depth ?? 0,
    maxDepth,
    done: acceptSet.has(q),
    laps,
    trace,
  };
}
