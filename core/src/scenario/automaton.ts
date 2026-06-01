// Module A — compile(goal) -> CompiledAutomaton.
//
// The objective plane compiles to a small phase/progress automaton (NOT full
// LTL — that is a documented extension). Compilation is a recursive build over
// a shared mutable state list; composition wires sub-automata by EAGER
// epsilon-closure (`spliceEntry` copies a child entry's transitions onto each
// parent accept), since the product search has no epsilon steps — every edge is
// a motion primitive. After building, `depth` (best-progress rank) and
// `remainingChain` (admissible LB to F) are precomputed.

import type { Goal, Region, Acceptance, ScenarioState } from './types';
import { normalize } from './normalize';

/** A guarded transition: spatial membership (or crossing) of `region` plus the
 *  acceptance conjuncts must all hold for the edge to fire. */
export interface GuardPredicate {
  region: Region;
  accept?: Acceptance;
}
export interface Transition {
  guard: GuardPredicate;
  target: number;
}
export interface AutomatonState {
  id: number;
  transitions: Transition[];
  /** Phase-progress rank: monotone non-decreasing toward F along any run.
   *  Used by best-progress incumbent ranking. */
  depth: number;
}
export interface CompiledAutomaton {
  states: AutomatonState[];
  start: number;
  /** F. Empty => progress automaton (repeat): maximize depth, no terminal. */
  accepting: number[];
  progress: boolean;
  /** remainingChain[q] = admissible LB on additional g from AFTER q's next
   *  guard through the rest of the automaton to F (0 in F; all-0 for progress
   *  automata). */
  remainingChain: number[];
}

interface Fragment {
  start: number;
  accepts: number[];
  progress: boolean;
}

class Builder {
  readonly states: AutomatonState[] = [];
  newState(): AutomatonState {
    const s: AutomatonState = { id: this.states.length, transitions: [], depth: 0 };
    this.states.push(s);
    return s;
  }
  /** Copy `entry`'s outgoing transitions onto every parent-accept state, and
   *  return the resulting accept set: the child's accepts, plus (when the child
   *  entry is itself accepting, i.e. an empty leg) the parent accepts unchanged. */
  spliceOnto(parentAccepts: number[], childFrag: Fragment): number[] {
    const entry = this.states[childFrag.start]!;
    const childEntryIsAccept = childFrag.accepts.includes(childFrag.start);
    for (const pa of parentAccepts) {
      if (pa === childFrag.start) continue;
      for (const tr of entry.transitions) {
        this.states[pa]!.transitions.push({ guard: tr.guard, target: tr.target });
      }
    }
    const out: number[] = [];
    if (childEntryIsAccept) out.push(...parentAccepts);
    for (const a of childFrag.accepts) if (a !== childFrag.start) out.push(a);
    return dedupeNums(out);
  }
}

function dedupeNums(xs: number[]): number[] {
  return Array.from(new Set(xs));
}

function buildFragment(b: Builder, goal: Goal): Fragment {
  switch (goal.kind) {
    case 'reach': {
      const s0 = b.newState();
      const s1 = b.newState();
      s0.transitions.push({
        guard: { region: goal.region, ...(goal.accept ? { accept: goal.accept } : {}) },
        target: s1.id,
      });
      return { start: s0.id, accepts: [s1.id], progress: false };
    }

    case 'seq': {
      if (goal.goals.length === 0) {
        const s = b.newState(); // TOP: trivially accepting
        return { start: s.id, accepts: [s.id], progress: false };
      }
      let frag = buildFragment(b, goal.goals[0]!);
      for (let i = 1; i < goal.goals.length; i++) {
        const cf = buildFragment(b, goal.goals[i]!);
        const newAccepts = b.spliceOnto(frag.accepts, cf);
        frag = { start: frag.start, accepts: newAccepts, progress: frag.progress || cf.progress };
      }
      return frag;
    }

    case 'any': {
      const entry = b.newState();
      const accepts: number[] = [];
      let progress = false;
      for (const child of goal.goals) {
        const cf = buildFragment(b, child);
        const cStart = b.states[cf.start]!;
        for (const tr of cStart.transitions) {
          entry.transitions.push({ guard: tr.guard, target: tr.target });
        }
        if (cf.accepts.includes(cf.start)) accepts.push(entry.id);
        for (const a of cf.accepts) if (a !== cf.start) accepts.push(a);
        progress ||= cf.progress;
      }
      // any() with zero branches => dead entry (no transitions, not accepting):
      // normalize already rewrites it to BOTTOM, but this stays well-formed.
      return { start: entry.id, accepts: dedupeNums(accepts), progress };
    }

    case 'all': {
      const N = goal.goals.length;
      if (N > 12) {
        throw new Error(`compile: all() with ${N} children exceeds the 2^N lattice cap (12)`);
      }
      const guards: GuardPredicate[] = goal.goals.map((g) => {
        if (g.kind !== 'reach') {
          throw new Error(
            "compile: all() children must be single-phase reach goals in v1 " +
              `(got '${g.kind}'); nest with seq/any outside the all, or use a product automaton (follow-up)`,
          );
        }
        return { region: g.region, ...(g.accept ? { accept: g.accept } : {}) };
      });
      const masks: AutomatonState[] = [];
      for (let m = 0; m < 1 << N; m++) masks.push(b.newState());
      for (let m = 0; m < 1 << N; m++) {
        for (let i = 0; i < N; i++) {
          if ((m & (1 << i)) === 0) {
            masks[m]!.transitions.push({ guard: guards[i]!, target: masks[m | (1 << i)]!.id });
          }
        }
      }
      return { start: masks[0]!.id, accepts: [masks[(1 << N) - 1]!.id], progress: false };
    }

    case 'repeat': {
      const cf = buildFragment(b, goal.goal);
      const entry = b.states[cf.start]!;
      for (const a of cf.accepts) {
        if (a === cf.start) continue;
        for (const tr of entry.transitions) {
          b.states[a]!.transitions.push({ guard: tr.guard, target: tr.target });
        }
      }
      return { start: cf.start, accepts: [], progress: true };
    }
  }
}

/** Longest-path depth from `start`, treating back-edges (to a state already on
 *  the DFS stack, i.e. repeat loop-backs) as non-advancing. */
function computeDepth(states: AutomatonState[], start: number): void {
  const onStack = new Array<boolean>(states.length).fill(false);
  for (const s of states) s.depth = 0;
  const visit = (u: number): void => {
    onStack[u] = true;
    for (const tr of states[u]!.transitions) {
      if (onStack[tr.target]) continue; // back-edge (repeat loop)
      const cand = states[u]!.depth + 1;
      if (cand > states[tr.target]!.depth) {
        states[tr.target]!.depth = cand;
        visit(tr.target); // propagate the improvement
      }
    }
    onStack[u] = false;
  };
  visit(start);
}

/** Admissible LB from AFTER each state's next guard to F, over the DAG. */
function computeRemainingChain(
  states: AutomatonState[],
  accepting: number[],
  progress: boolean,
): number[] {
  const chain = new Array<number>(states.length).fill(0);
  if (progress || accepting.length === 0) return chain; // no terminal F

  const acceptSet = new Set(accepting);
  // entryReps[q] = representative poses of ALL guards that enter q (the candidate
  // "from" poses for estimating the next leg). Using the MIN leg over these is
  // deterministic (order-independent) and a valid lower bound — so the chain
  // stays admissible even for states with multiple incoming edges (`all`/`any`).
  const entryReps: ScenarioState[][] = states.map(() => []);
  for (const s of states) {
    for (const tr of s.transitions) {
      entryReps[tr.target]!.push(tr.guard.region.representative());
    }
  }
  const memo = new Array<number | undefined>(states.length).fill(undefined);
  const visiting = new Array<boolean>(states.length).fill(false);
  const D = (q: number): number => {
    if (acceptSet.has(q)) return 0;
    if (memo[q] !== undefined) return memo[q]!;
    if (visiting[q]) return 0; // cycle guard (shouldn't happen on a DAG)
    visiting[q] = true;
    const froms = entryReps[q]!;
    let best = Infinity;
    for (const tr of states[q]!.transitions) {
      // Admissible leg: cheapest cost-to-reach this guard over all entry poses
      // (0 at the start, which has no incoming guard).
      let leg = 0;
      if (froms.length > 0) {
        leg = Infinity;
        for (const from of froms) leg = Math.min(leg, tr.guard.region.costToGo(from));
      }
      best = Math.min(best, leg + D(tr.target));
    }
    visiting[q] = false;
    const val = Number.isFinite(best) ? best : 0; // dead non-accepting state
    memo[q] = val;
    return val;
  };
  for (let q = 0; q < states.length; q++) chain[q] = D(q);
  return chain;
}

/** Compile a goal AST into the canonical automaton. The goal is normalized
 *  first, so equivalent specs compile to equivalent automata. */
export function compile(goal: Goal): CompiledAutomaton {
  const normalized = normalize(goal);
  const b = new Builder();
  const frag = buildFragment(b, normalized);
  computeDepth(b.states, frag.start);
  const remainingChain = computeRemainingChain(b.states, frag.accepts, frag.progress);
  return {
    states: b.states,
    start: frag.start,
    accepting: frag.accepts,
    progress: frag.progress,
    remainingChain,
  };
}

/** The representative pose of `q`'s cheapest outgoing guard — the pose the
 *  planner bridge aims the base environment at. Returns null for dead/accepting
 *  states with no outgoing guard. */
export function nextGuardPose(automaton: CompiledAutomaton, q: number): ScenarioState | null {
  const st = automaton.states[q];
  if (!st || st.transitions.length === 0) return null;
  // Cheapest is data-dependent; the representative of the first guard is a
  // stable, deterministic choice (transitions are emitted in phase order).
  return st.transitions[0]!.guard.region.representative();
}
